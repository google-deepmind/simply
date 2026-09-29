# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""LiveCodeBench sampling task: FIXED eval framework + a minimal baseline.

This module provides the *locked* evaluation framework for the `sampling_lcb`
task and ONE trivial baseline. The research surface (what the agent designs) is
the DECODING PROCESS that produces a single final code string per problem; the
SCORING of that string is fixed and must not be reimplemented.

## What is FIXED (do not modify -- reuse verbatim)

`LcbEval.evaluate_async` runs the pipeline for one problem:
  1. call `self.decode(example, model_fn, ctx)` -> ONE final code string
     (this is the ONLY method an agent overrides; see below);
  2. score that string with the UNCHANGED LiveCodeBench executor over the FULL
     (public + private) test suite, marking `correct` iff every test returns
     RESULT_PASS.

`_LcbGrader.score_final_answer` is the single source of truth for correctness.
The reported `accuracy` in final_result.json is the mean of `correct` over the
fixed problem set. Keeping it fixed is what makes submissions comparable.

## What is FREE (the research surface -- design your own)

`decode(example, model_fn, ctx)` -- produce the final code by ANY test-time
sampling / decoding process: draw as many samples as you like at any
temperature / top_p / top_k, prompt however you like. `model_fn(messages)`
returns a decoded response dict. The baseline below draws a single sample;
override `decode` in your own registered `Evaluation` subclass to do better.
There is no explicit sample/compute cap -- the task's wall-clock limit is the
governor (scale test-time compute too far and the run times out).

## Contamination rule

You MAY use the PUBLIC tests as a decoding signal: `ctx.run_public_tests(code)`
executes a candidate against the PUBLIC tests ONLY and returns per-test
pass/fail (safe to use for filtering / repair / selection). You MUST NOT use the
private tests (or the full-suite grader) as a decoding signal -- the private
tests exist only for the fixed final grade, and the example passed to `decode`
does not carry them (`_without_private_tests`, enforced in code).

## Port note

The internal original delegated grading to an internal LiveCodeBench executor
(`_test_livecodebench_problem`) running in an internal sandbox. That code is
internal, so the executor is reproduced here line-for-line in behaviour --
preamble, per-test-type driver, exit-code taxonomy, float-tolerant output
comparison, first-failure short circuit -- on top of `code_exec_lib`, which
provides the subprocess sandbox. See EVAL_TASKS_NOTES.md for the isolation
difference (it is weaker than gVisor) and for what stays bit-identical.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import enum
import json
import math
import re
import time
from typing import Any, Callable, ClassVar, Mapping, Sequence

from absl import logging
from simply.utils import evaluation_lib
from tasks.research_bench import code_exec_lib


# The code-execution sandbox can fail for reasons unrelated to the candidate
# program (host resource contention, fork failure). Such a failure used to
# propagate out of the eval and end the job, and the restarted job replays the
# same example, so a single flake can cost a whole 167-problem run. Retry a
# bounded number of times, then report the example as not passing and continue.
_SANDBOX_MAX_ATTEMPTS = 3
_SANDBOX_RETRY_SLEEP_S = 5.0

# Per-test limits of the reference executor: 10s of solution time and 16GiB.
_SOLUTION_TIMEOUT_S = 10
_MEMORY_BYTES = 16 * 1024**3
_TEST_LIMITS = code_exec_lib.Limits(
    timeout_s=_SOLUTION_TIMEOUT_S, memory_bytes=_MEMORY_BYTES
)

# Exit status the functional-test driver uses for "ran fine, wrong answer".
_WRONG_OUTPUT_ERROR_NUM = 24

# https://github.com/LiveCodeBench/LiveCodeBench/blob/45015dd2a9fa4bf445613e2f29da505dc0ca5c03/lcb_runner/evaluation/testing_util.py#L114
_LIVECODEBENCH_PREAMBLE = """\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""


class ResultType(enum.Enum):
  """Per-test verdict taxonomy of the reference executor."""

  RESULT_UNKNOWN = 'unknown'  # The sandbox failed; the program was not judged.
  RESULT_PASS = 'pass'
  RESULT_FAIL_TIMEOUT = 'timeout'
  RESULT_FAIL_SYNTAX_ERROR = 'syntax_error'
  RESULT_FAIL_WRONG_OUTPUT = 'wrong_output'
  RESULT_FAIL_EXECUTION_ERROR = 'execution_error'


@dataclasses.dataclass(frozen=True)
class LiveCodeBenchSampleExecutionResult:
  """Outcome of running one candidate program over a set of tests."""

  result: ResultType
  stdout: str
  stderr: str
  num_tests_passed: int = 0
  num_tests: int = 0
  failing_test: dict[str, Any] | None = None


class TestsUsed(enum.Enum):
  """Visibility of the tests."""

  PUBLIC = 'public'
  ALL = 'all'


# --------------------------------------------------------------------------
# Response -> code (FIXED normalisation).
# --------------------------------------------------------------------------
_THOUGHT_START_END_PAIRS = (
    ('<ctrl94>thought', '<ctrl95>'),
    ('<ctrl3347>', '<ctrl3348>'),
    ('<thought>', '</thought>'),
    ('<think>', '</think>'),
)

_CODE_BLOCK_RE = re.compile(r'```[^\n]*\n([\s\S]*?)[\n|\s]*```')


def remove_thoughts(text: str) -> str:
  """Strips thought spans, and anything after a dangling thought opener."""
  for start, end in _THOUGHT_START_END_PAIRS:
    text = re.sub(f'{re.escape(start)}.*?{re.escape(end)}', '', text,
                  flags=re.DOTALL)
    text = text.split(start)[0]
  return text


def extract_last_code_block(text: str) -> str:
  """Returns the last ```-fenced block, or the whole text if there is none."""
  blocks = _CODE_BLOCK_RE.findall(text)
  return blocks[-1] if blocks else text


def add_function_call_if_missing(sample: str) -> str:
  """Calls the solution function if the program only defines it.

  Some models emit a no-argument `solve()`/`main()` and never call it, which
  fails every stdin test. The reference normaliser appends the call; kept
  verbatim because it changes graded outcomes.

  Args:
    sample: Candidate program.

  Returns:
    The program, with a trailing call appended when one is unambiguously
    missing.
  """
  try:
    tree = ast.parse(sample)
  except SyntaxError:
    return sample

  no_arg_functions = set()
  for node in tree.body:
    if isinstance(node, ast.FunctionDef):
      args = node.args
      if not (args.args or args.posonlyargs or args.kwonlyargs or args.vararg
              or args.kwarg):
        no_arg_functions.add(node.name)

  if not no_arg_functions:
    return sample
  elif len(no_arg_functions) > 1:
    if 'solve' in no_arg_functions:
      solution_fn_name = 'solve'
    else:
      return sample
  else:
    solution_fn_name = list(no_arg_functions)[0]

  function_def_node = None
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == solution_fn_name:
      function_def_node = node
      break
  if not function_def_node:
    return sample

  for node in tree.body:
    if node is function_def_node:
      continue
    for sub_node in ast.walk(node):
      is_reference = (
          isinstance(sub_node, ast.Name) and sub_node.id == solution_fn_name
      )
      is_call = (
          isinstance(sub_node, ast.Call)
          and isinstance(sub_node.func, ast.Name)
          and sub_node.func.id == solution_fn_name
      )
      if is_reference or is_call:
        return sample
  return f'{sample}\n{solution_fn_name}()\n'


def remove_thoughts_and_extract_last_code_block(sample: str) -> str:
  """FIXED response -> program normalisation used for every submission."""
  return add_function_call_if_missing(
      extract_last_code_block(remove_thoughts(sample))
  )


# --------------------------------------------------------------------------
# Output comparison (FIXED).
# --------------------------------------------------------------------------
def _values_equal(actual: str, expected: str, tol: float | None) -> bool:
  """Line-level comparison: exact, or float-close when either side is a float."""
  if tol is None:
    return actual == expected
  try:
    if isinstance(ast.literal_eval(expected), float) or isinstance(
        ast.literal_eval(actual), float
    ):
      return math.isclose(
          float(actual), float(expected), abs_tol=tol, rel_tol=tol
      )
  except (SyntaxError, ValueError, RecursionError, MemoryError):
    pass
  return actual == expected


def outputs_match(
    solution_output: str, test_output: str, tol: float | None = 1e-6
) -> bool:
  """Whitespace-insensitive, optionally float-tolerant stdout comparison."""
  actual_lines = [l.strip() for l in solution_output.strip().splitlines()]
  expected_lines = [l.strip() for l in test_output.strip().splitlines()]
  if len(actual_lines) != len(expected_lines):
    return False
  return all(
      _values_equal(a, e, tol) for a, e in zip(actual_lines, expected_lines)
  )


# --------------------------------------------------------------------------
# The executor (FIXED).
# --------------------------------------------------------------------------
def _functional_driver(
    snippet: str, input_to_test: str, expected_output: str, func_name: str,
    approx_float_matching: bool,
) -> str:
  """Appends the `Solution().<func>(...)` call + assertion for a `functional` test."""
  args = ', '.join(input_to_test.splitlines())
  expected = json.loads(expected_output)
  # json.loads('"abc"') is `abc`, which would be a bare name in the driver; put
  # the quotes back.
  if expected_output.startswith('"'):
    expected = repr(expected)
  return snippet + f"""
actual = Solution().{func_name}({args})
expected = {expected}
import math
import sys
def _eq(actual, expected, approx_float_matching):
  if isinstance(expected, float) and approx_float_matching:
    return math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-6)
  return actual == expected
if not _eq(actual, expected, {approx_float_matching}):
  sys.stderr.write(f"wrong answer, expected:{{expected}}, actual:{{actual}}")
  sys.exit({_WRONG_OUTPUT_ERROR_NUM})
"""


def _extract_generated_output(stderr: str, test: Mapping[str, Any],
                              stdout: str) -> str:
  """Recovers what the program produced, for the `failing_test` report."""
  if test['testtype'] == 'stdin':
    return stdout
  if test['testtype'] == 'functional':
    lines = stderr.splitlines()
    match = re.search(r'actual:\s*(.*)', lines[-1]) if lines else None
    return match.group(1).strip() if match else ''
  raise ValueError(f'Unsupported test type: {test["testtype"]}')


def _test_problem(
    snippet: str,
    test: Mapping[str, Any],
    func_name: str,
    approx_float_matching: bool,
    limits: code_exec_lib.Limits = _TEST_LIMITS,
) -> LiveCodeBenchSampleExecutionResult:
  """Runs one candidate program against ONE test case.

  Args:
    snippet: The program (preamble already prepended).
    test: `{input, output, testtype}` from the dataset.
    func_name: Entry point for `functional` tests.
    approx_float_matching: Compare floats with 1e-6 tolerance.
    limits: Sandbox resource limits.

  Returns:
    The per-test verdict; sandbox failures come back as RESULT_UNKNOWN rather
    than raising.

  Raises:
    ValueError: for an unknown `testtype` (a dataset bug, not a program bug).
  """
  testtype = test['testtype']
  if testtype == 'stdin':
    res = code_exec_lib.run_python(snippet, stdin=test['input'], limits=limits)
    if res.status is code_exec_lib.Status.OK and (
        not res.stdout
        or not outputs_match(
            res.stdout, test['output'],
            tol=1e-6 if approx_float_matching else None,
        )
    ):
      returncode = _WRONG_OUTPUT_ERROR_NUM
    else:
      returncode = res.returncode
  elif testtype == 'functional':
    driver = _functional_driver(
        snippet, test['input'], test['output'], func_name,
        approx_float_matching,
    )
    res = code_exec_lib.run_python(driver, limits=limits)
    returncode = res.returncode
  else:
    raise ValueError(f'Unsupported test type: {testtype}')

  generated_output = ''
  if res.status is code_exec_lib.Status.SANDBOX_ERROR:
    result = ResultType.RESULT_UNKNOWN
  elif res.status is code_exec_lib.Status.TIMEOUT:
    result = ResultType.RESULT_FAIL_TIMEOUT
  elif returncode:
    if 'SyntaxError' in res.stderr:
      result = ResultType.RESULT_FAIL_SYNTAX_ERROR
    elif returncode == _WRONG_OUTPUT_ERROR_NUM:
      result = ResultType.RESULT_FAIL_WRONG_OUTPUT
      generated_output = _extract_generated_output(res.stderr, test, res.stdout)
    else:
      result = ResultType.RESULT_FAIL_EXECUTION_ERROR
  else:
    result = ResultType.RESULT_PASS

  return LiveCodeBenchSampleExecutionResult(
      result=result,
      stdout=res.stdout,
      stderr=res.stderr,
      num_tests=1,
      num_tests_passed=int(result is ResultType.RESULT_PASS),
      failing_test={
          'testtype': testtype,
          'input': test['input'],
          'expected_output': test['output'],
          'actual_output': generated_output,
      },
  )


def _test_livecodebench_problem(
    problem: Mapping[str, Any],
    sample: str,
    approx_float_matching: bool,
    tests_used: TestsUsed = TestsUsed.ALL,
) -> LiveCodeBenchSampleExecutionResult:
  """Runs a candidate program over a problem's tests, stopping at the first failure.

  Args:
    problem: `example['auxiliary']` (public/private test JSON + func_name).
    sample: The candidate program (already extracted from the response).
    approx_float_matching: Compare floats with 1e-6 tolerance.
    tests_used: PUBLIC (decoding signal) or ALL (the final grade).

  Returns:
    The aggregate result; `result is RESULT_PASS` iff every test passed.

  Raises:
    ValueError: for an unsupported `tests_used`.
  """
  public_tests = json.loads(problem['public_test_cases'])
  if tests_used is TestsUsed.PUBLIC:
    visible_tests = public_tests
  elif tests_used is TestsUsed.ALL:
    visible_tests = public_tests + json.loads(problem['private_test_cases'])
  else:
    raise ValueError(f'Unsupported test types: {tests_used}')

  snippet = _LIVECODEBENCH_PREAMBLE + sample
  num_tests_passed = 0
  final_result = ResultType.RESULT_UNKNOWN
  final_stdout = final_stderr = ''
  failing_test = None
  # Stops at the first failing test: the verdict is unchanged and a broken
  # program does not burn 40 sandbox launches.
  for test in visible_tests:
    res = _test_problem(
        snippet, test, func_name=problem.get('func_name', ''),
        approx_float_matching=approx_float_matching,
    )
    final_result, final_stdout, final_stderr = res.result, res.stdout, res.stderr
    if res.result is ResultType.RESULT_PASS:
      num_tests_passed += 1
    else:
      failing_test = res.failing_test
      break

  return LiveCodeBenchSampleExecutionResult(
      result=final_result,
      stdout=final_stdout,
      stderr=final_stderr,
      num_tests=len(visible_tests),
      num_tests_passed=num_tests_passed,
      failing_test=failing_test,
  )


class _LcbGrader:
  """The UNCHANGED LiveCodeBench executor + extractor, with flake retries.

  All grading goes through `_test_livecodebench_problem`. This class is FIXED:
  it defines the correctness contract for the task.
  """

  def extract_code(self, response_text: str) -> str:
    """Extracts the last fenced code block from a response (thoughts removed)."""
    return remove_thoughts_and_extract_last_code_block(response_text)

  def _retry(self, fn, what: str):
    """Runs `fn`, retrying sandbox failures a bounded number of times.

    Args:
      fn: zero-arg callable returning a `LiveCodeBenchSampleExecutionResult`.
      what: short description used in the log lines.

    Returns:
      `fn()`'s result, or None if every attempt hit a sandbox failure.
    """
    for attempt in range(1, _SANDBOX_MAX_ATTEMPTS + 1):
      try:
        res = fn()
        if res.result is not ResultType.RESULT_UNKNOWN:
          return res
        reason = 'sandbox reported RESULT_UNKNOWN'
      except Exception as e:  # pylint: disable=broad-except
        reason = repr(e)
      logging.warning(
          '%s failed on attempt %d/%d: %s', what, attempt,
          _SANDBOX_MAX_ATTEMPTS, reason,
      )
      if attempt < _SANDBOX_MAX_ATTEMPTS:
        time.sleep(_SANDBOX_RETRY_SLEEP_S * attempt)
    logging.error(
        '%s failed %d times; scoring this example as not passing.',
        what, _SANDBOX_MAX_ATTEMPTS,
    )
    return None

  def score_final_answer(self, example: Mapping[str, Any], code: str) -> bool:
    """FIXED final grade: True iff `code` passes ALL (public+private) tests.

    This is the ONLY thing that determines `correct`. It is identical for every
    submission and must not be reimplemented by the agent's decoding process.

    Args:
      example: the raw example dict (reads example['auxiliary']).
      code: the agent's single FINAL code string for this problem.

    Returns:
      Whether the code passes the full LiveCodeBench test suite.
    """
    if not code.strip():
      return False
    res = self._retry(
        lambda: _test_livecodebench_problem(
            problem=example['auxiliary'],
            sample=code,
            approx_float_matching=True,
            tests_used=TestsUsed.ALL,
        ),
        'livecodebench grading',
    )
    return res is not None and res.result is ResultType.RESULT_PASS

  def run_public_tests(
      self, example: Mapping[str, Any], code: str
  ) -> dict[str, Any]:
    """Convenience DECODING SIGNAL: run `code` against the PUBLIC tests only.

    Safe to call during decoding (uses PUBLIC tests only, never private).
    Every public test is run (no early exit) so the caller sees per-test
    results.

    Args:
      example: the raw example dict.
      code: a candidate code string.

    Returns:
      dict(passes_public, num_public, num_public_passed, results) where
      `results` is the per-test `ResultType` value as a string.
    """
    if not code.strip():
      return dict(
          passes_public=False, num_public=0, num_public_passed=0, results=[]
      )
    aux = example['auxiliary']
    public_tests = json.loads(aux['public_test_cases'])
    snippet = _LIVECODEBENCH_PREAMBLE + code
    func_name = aux.get('func_name', '')
    results = []
    for test in public_tests:
      res = self._retry(
          lambda t=test: _test_problem(
              snippet=snippet,
              test=t,
              func_name=func_name,
              approx_float_matching=True,
          ),
          'public-test execution',
      )
      results.append(
          (res.result if res is not None else ResultType.RESULT_UNKNOWN).value
      )
    n_pass = sum(r == ResultType.RESULT_PASS.value for r in results)
    return dict(
        passes_public=(len(public_tests) > 0 and n_pass == len(public_tests)),
        num_public=len(public_tests),
        num_public_passed=n_pass,
        results=results,
    )


_LCB_GRADER = _LcbGrader()


# Keys under `example['auxiliary']` that hold the PRIVATE tests. They are
# removed from the example handed to `decode` (the full example is still used
# for the final grade), so a decoding process reads only what it is allowed to
# use.
_PRIVATE_TEST_KEYS = ('private_test_cases',)


def _without_private_tests(example: Mapping[str, Any]) -> Mapping[str, Any]:
  """Returns a copy of `example` with the private test cases removed.

  Args:
    example: the raw LiveCodeBench example.

  Returns:
    The example unchanged if it carries no private tests, else a shallow copy
    whose `auxiliary` omits them (public tests and `func_name` are kept).
  """
  aux = example.get('auxiliary')
  if not isinstance(aux, Mapping):
    return example
  if not any(k in aux for k in _PRIVATE_TEST_KEYS):
    return example
  redacted = dict(example)
  redacted['auxiliary'] = {
      k: v for k, v in aux.items() if k not in _PRIVATE_TEST_KEYS
  }
  return redacted


@dataclasses.dataclass
class DecodeContext:
  """The FIXED helpers a `decode` implementation may use.

  * `run_public_tests(code)` -- execute a candidate on the PUBLIC tests only
    (a legitimate decoding signal). Do NOT use private tests.
  * `score_final_answer` is intentionally NOT exposed here: the framework
    applies it to your returned answer. Calling the full grader yourself during
    decoding would be contamination -- and it cannot be done by accident, since
    `example` here has no private tests to grade against.
  """

  example: Mapping[str, Any]

  def run_public_tests(self, code: str) -> dict[str, Any]:
    return _LCB_GRADER.run_public_tests(self.example, code)

  async def run_public_tests_async(self, code: str) -> dict[str, Any]:
    """`run_public_tests` off the event loop, so decoding can overlap it."""
    return await asyncio.to_thread(self.run_public_tests, code)

  async def run_public_tests_many(
      self, codes: Sequence[str]
  ) -> list[dict[str, Any]]:
    """Public-tests several candidates concurrently, in input order.

    Concurrency is bounded by this event loop's default thread pool, and every
    candidate costs host CPU that the decode loop also needs -- keep the fan-out
    modest.

    Args:
      codes: candidate programs.

    Returns:
      One `run_public_tests` dict per candidate.
    """
    return list(
        await asyncio.gather(*(self.run_public_tests_async(c) for c in codes))
    )

  def extract_code(self, response_text: str) -> str:
    return _LCB_GRADER.extract_code(response_text)


@dataclasses.dataclass(frozen=True)
class LcbEval(evaluation_lib.Evaluation):
  r"""FIXED LiveCodeBench sampling-eval framework + minimal baseline.

  Subclass and override ONLY `decode` to implement your test-time sampling /
  decoding strategy. Everything else (final ALL-tests grading, accuracy
  accounting) is fixed and shared across submissions for a controllable
  comparison. Do NOT override `evaluate_async` or the grading path.
  """

  # Tells page_decode_eval to drive this eval through `evaluate_async` (the
  # multi-turn / agentic path) instead of one prompt -> one response.
  in_sandbox_loop: ClassVar[bool] = True

  def get_messages(
      self, example: Mapping[str, Any]
  ) -> Sequence[Mapping[str, Any]]:
    """FIXED default prompt (the baseline's); a decoder may build its own."""
    return [
        dict(role='system', content='SPECIAL INSTRUCTION: think silently.'),
        dict(role='user', content=example['prompt']),
    ]

  def evaluate(
      self, example: Mapping[str, Any], response: Any
  ) -> Mapping[str, Any]:
    """Single-turn grading path (full test suite), for non-agentic callers."""
    code = _LCB_GRADER.extract_code(response or '')
    correct = _LCB_GRADER.score_final_answer(example, code)
    return dict(correct=int(correct), reward=float(correct))

  async def decode(
      self,
      example: Mapping[str, Any],
      model_fn: Callable[..., Any],
      ctx: DecodeContext,
  ) -> str:
    """Produce the FINAL code string for `example` (OVERRIDE THIS).

    The baseline draws a single sample at the configured sampling params and
    returns its extracted code. Override with any decoding process you like,
    using `model_fn` to decode and `ctx.run_public_tests` as an optional signal.

    Args:
      example: the raw example dict, WITHOUT the private tests.
      model_fn: async callable; `await model_fn(messages)` returns a response
        dict with `output_text` (and `input_len`, `tokens`, `truncated`).
      ctx: fixed decoding helpers (public-test signal, code extraction).

    Returns:
      The single final code string to be graded.
    """
    resp = await model_fn(self.get_messages(example))
    return ctx.extract_code((resp or {}).get('output_text', '') or '')

  async def evaluate_async(self, example, model_fn):
    """FIXED pipeline: agent decodes ONE answer; framework grades ALL tests.

    Do NOT override. This method defines the graded metric.

    Args:
      example: the LiveCodeBench problem record.
      model_fn: the decode callable used to sample one candidate program.

    Returns:
      A result dict with the graded `correct`/`reward` and decode metadata.
    """
    # The decoding process sees the example WITHOUT the private tests; the
    # final grade below uses the full example.
    decode_example = _without_private_tests(example)
    ctx = DecodeContext(example=decode_example)
    final_code = await self.decode(decode_example, model_fn, ctx)
    final_code = final_code or ''
    # The one and only correctness verdict: full (public+private) test suite.
    correct = await asyncio.to_thread(
        _LCB_GRADER.score_final_answer, example, final_code
    )
    return dict(
        auxiliary={},
        prompt='',
        lm_request=self.get_messages(example),
        lm_response=dict(output_text=final_code, tokens=[], input_len=0),
        correct=int(correct),
        reward=float(correct),
    )


@evaluation_lib.EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class LcbBaseline(LcbEval):
  """Minimal baseline: a single sample, graded on the full test suite (pass@1).

  This is the score-0 anchor. It is deliberately trivial -- improving on it by
  designing a better test-time sampling/decoding process is the task.
  """

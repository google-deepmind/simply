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

"""Tests for the FIXED LiveCodeBench grading harness and its baseline.

The properties that make submissions comparable are the ones asserted here:
the grader accepts a correct program and rejects a wrong one on the FULL test
suite, a program that hangs or crashes is a failure (not an exception), the
public tests are usable as a decoding signal, and the private tests are
unreachable from `decode`.
"""

import asyncio
import json

from absl.testing import absltest
from simply.utils import evaluation_lib
from tasks.research_bench import lcb_sampling_lib as lcb


# Add 1 to the number on stdin. Public test: 1 -> 2. Private tests: 41 -> 42
# (and one the wrong program below also gets wrong).
_ADD_ONE = dict(
    example_id='stdin_problem',
    prompt='### Question:\nprint n+1\n',
    auxiliary=dict(
        public_test_cases=json.dumps(
            [dict(input='1\n', output='2\n', testtype='stdin')]
        ),
        private_test_cases=json.dumps([
            dict(input='41\n', output='42\n', testtype='stdin'),
            dict(input='7\n', output='8\n', testtype='stdin'),
        ]),
        func_name='',
        question='print n+1',
        starter_code='',
    ),
)

_FUNCTIONAL = dict(
    example_id='functional_problem',
    prompt='### Question:\ndouble the list\n',
    auxiliary=dict(
        public_test_cases=json.dumps(
            [dict(input='[1, 2]', output='[2, 4]', testtype='functional')]
        ),
        private_test_cases=json.dumps(
            [dict(input='[3]', output='[6]', testtype='functional')]
        ),
        func_name='double',
        question='double the list',
        starter_code='class Solution:\n    def double(self, xs): ...',
    ),
)

_CORRECT = 'print(int(input()) + 1)'
# Passes the public test (1 -> 2) but fails a private one (41 -> 43): exactly
# the case public-test filtering cannot catch.
_PUBLIC_ONLY = 'n = int(input())\nprint(2 if n == 1 else n + 2)'
_WRONG = 'print(0)'
_CRASH = 'raise RuntimeError("boom")'
_HANG = 'while True: pass'
_CORRECT_FUNCTIONAL = (
    'class Solution:\n    def double(self, xs):\n        return [x * 2 for x in xs]'
)


class GraderTest(absltest.TestCase):

  def test_accepts_a_correct_program(self):
    self.assertTrue(lcb._LCB_GRADER.score_final_answer(_ADD_ONE, _CORRECT))

  def test_rejects_a_wrong_program(self):
    self.assertFalse(lcb._LCB_GRADER.score_final_answer(_ADD_ONE, _WRONG))

  def test_grades_on_private_tests_too(self):
    self.assertFalse(
        lcb._LCB_GRADER.score_final_answer(_ADD_ONE, _PUBLIC_ONLY)
    )

  def test_crash_is_a_failure_not_an_exception(self):
    self.assertFalse(lcb._LCB_GRADER.score_final_answer(_ADD_ONE, _CRASH))

  def test_timeout_is_a_failure_not_an_exception(self):
    res = lcb._test_problem(
        lcb._LIVECODEBENCH_PREAMBLE + _HANG,
        json.loads(_ADD_ONE['auxiliary']['public_test_cases'])[0],
        func_name='',
        approx_float_matching=True,
        limits=lcb.code_exec_lib.Limits(timeout_s=1.0),
    )
    self.assertIs(res.result, lcb.ResultType.RESULT_FAIL_TIMEOUT)

  def test_empty_answer_is_a_failure(self):
    self.assertFalse(lcb._LCB_GRADER.score_final_answer(_ADD_ONE, '   '))

  def test_functional_problem(self):
    self.assertTrue(
        lcb._LCB_GRADER.score_final_answer(_FUNCTIONAL, _CORRECT_FUNCTIONAL)
    )
    self.assertFalse(
        lcb._LCB_GRADER.score_final_answer(
            _FUNCTIONAL,
            'class Solution:\n    def double(self, xs):\n        return xs',
        )
    )

  def test_float_outputs_are_compared_with_tolerance(self):
    self.assertTrue(lcb.outputs_match('0.3333333\n', '0.33333334\n'))
    self.assertFalse(lcb.outputs_match('0.34\n', '0.33333334\n'))
    self.assertTrue(lcb.outputs_match(' 42 \n', '42\n'))
    self.assertFalse(lcb.outputs_match('42\n7\n', '42\n'))

  def test_first_failure_short_circuits(self):
    res = lcb._test_livecodebench_problem(
        _ADD_ONE['auxiliary'], _WRONG, approx_float_matching=True
    )
    self.assertEqual(res.num_tests, 3)
    self.assertEqual(res.num_tests_passed, 0)
    self.assertIsNotNone(res.failing_test)


class PublicTestSignalTest(absltest.TestCase):

  def test_reports_per_test_pass_fail(self):
    res = lcb._LCB_GRADER.run_public_tests(_ADD_ONE, _CORRECT)
    self.assertEqual(
        res,
        dict(
            passes_public=True,
            num_public=1,
            num_public_passed=1,
            results=['pass'],
        ),
    )

  def test_failing_candidate(self):
    res = lcb._LCB_GRADER.run_public_tests(_ADD_ONE, _WRONG)
    self.assertFalse(res['passes_public'])
    self.assertEqual(res['results'], ['wrong_output'])

  def test_many_candidates_in_order(self):
    ctx = lcb.DecodeContext(example=lcb._without_private_tests(_ADD_ONE))
    results = asyncio.run(ctx.run_public_tests_many([_CORRECT, _WRONG]))
    self.assertEqual([r['passes_public'] for r in results], [True, False])

  def test_uses_public_tests_only(self):
    # The program is right on the public test and wrong on a private one.
    self.assertTrue(
        lcb._LCB_GRADER.run_public_tests(_ADD_ONE, _PUBLIC_ONLY)[
            'passes_public'
        ]
    )


class CodeExtractionTest(absltest.TestCase):

  def test_takes_the_last_fenced_block_without_thoughts(self):
    response = (
        '<think>maybe print(0)</think>\n'
        'First attempt:\n```python\nprint(0)\n```\n'
        'Better:\n```python\nprint(int(input()) + 1)\n```\n'
    )
    self.assertEqual(lcb._LCB_GRADER.extract_code(response), _CORRECT)

  def test_unfenced_response_is_used_as_is(self):
    self.assertEqual(lcb._LCB_GRADER.extract_code(_CORRECT), _CORRECT)

  def test_uncalled_solve_function_gets_called(self):
    code = lcb._LCB_GRADER.extract_code(
        '```python\ndef solve():\n    print(1)\n```'
    )
    self.assertEndsWith(code, '\nsolve()\n')

  def test_called_function_is_left_alone(self):
    source = 'def solve():\n    print(1)\nsolve()'
    self.assertEqual(lcb._LCB_GRADER.extract_code(source), source)


class PrivateTestIsolationTest(absltest.TestCase):

  def test_decode_never_sees_private_tests(self):
    seen = {}

    class Spy(lcb.LcbEval):

      async def decode(self, example, model_fn, ctx):
        seen['example'] = example
        seen['ctx_example'] = ctx.example
        return _CORRECT

    result = asyncio.run(Spy().evaluate_async(dict(_ADD_ONE), _unused_model_fn))

    for key in ('example', 'ctx_example'):
      self.assertNotIn('private_test_cases', seen[key]['auxiliary'])
      self.assertIn('public_test_cases', seen[key]['auxiliary'])
    # ... and the grade still used them.
    self.assertEqual(result['correct'], 1)

  def test_private_tests_survive_in_the_original_example(self):
    example = dict(_ADD_ONE)
    lcb._without_private_tests(example)
    self.assertIn('private_test_cases', example['auxiliary'])

  def test_decode_context_exposes_no_full_grader(self):
    ctx = lcb.DecodeContext(example=lcb._without_private_tests(_ADD_ONE))
    self.assertFalse(
        [name for name in dir(ctx) if 'score' in name or 'private' in name]
    )


async def _unused_model_fn(messages):
  del messages
  raise AssertionError('the decoder under test should not call the model')


class BaselineTest(absltest.TestCase):

  def test_registered_for_the_evaluation_flag(self):
    self.assertIn('LcbBaseline', evaluation_lib.EvaluationRegistry.keys())
    self.assertIsInstance(
        evaluation_lib.EvaluationRegistry.get_instance('LcbBaseline'),
        lcb.LcbEval,
    )

  def test_one_sample_graded_on_the_full_suite(self):
    calls = []

    async def model_fn(messages):
      calls.append(messages)
      return dict(output_text=f'```python\n{_CORRECT}\n```')

    result = asyncio.run(
        lcb.LcbBaseline().evaluate_async(dict(_ADD_ONE), model_fn)
    )
    self.assertLen(calls, 1)
    self.assertEqual(result['correct'], 1)
    self.assertEqual(result['reward'], 1.0)
    self.assertEqual(result['lm_response']['output_text'], _CORRECT)

  def test_wrong_sample_scores_zero(self):
    async def model_fn(messages):
      del messages
      return dict(output_text=f'```python\n{_PUBLIC_ONLY}\n```')

    result = asyncio.run(
        lcb.LcbBaseline().evaluate_async(dict(_ADD_ONE), model_fn)
    )
    self.assertEqual(result['correct'], 0)

  def test_prompt_is_the_fixed_two_turn_message(self):
    messages = lcb.LcbBaseline().get_messages(_ADD_ONE)
    self.assertEqual([m['role'] for m in messages], ['system', 'user'])
    self.assertEqual(messages[1]['content'], _ADD_ONE['prompt'])


class CustomDecoderTest(absltest.TestCase):
  """The research surface: a subclass that only overrides `decode`."""

  def test_public_test_filtering_picks_the_passing_candidate(self):
    class BestOfN(lcb.LcbEval):

      async def decode(self, example, model_fn, ctx):
        best = ''
        for _ in range(3):
          resp = await model_fn(self.get_messages(example))
          code = ctx.extract_code(resp['output_text'])
          best = best or code
          if ctx.run_public_tests(code)['passes_public']:
            return code
        return best

    samples = iter([_WRONG, _CRASH, _CORRECT])

    async def model_fn(messages):
      del messages
      return dict(output_text=f'```python\n{next(samples)}\n```')

    result = asyncio.run(BestOfN().evaluate_async(dict(_ADD_ONE), model_fn))
    self.assertEqual(result['correct'], 1)


if __name__ == '__main__':
  absltest.main()

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

"""LiveCodeBench v5 data source for the sampling_lcb task.

Registers `simply_json:livecodebench_v5` into the simply `DataSourceRegistry`
used by the page decode-eval harness. Each example carries its unit tests in
`auxiliary`; `lcb_sampling_lib` grades them (see that module for the fixed
grader and the public/private split) and `eval_main.py` is the entry point.

Data: the 167 problems of LiveCodeBench v5 (contest dates 2024-09-22 ..
2025-01-04). The internal original read them from an internal loader;
this reads the same rows from disk, staged by
`setup/build_datasets.py::build_livecodebench()` from HF
`livecodebench/code_generation_lite` `test5.jsonl`. That upstream file is
byte-identical (sha256 7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214)
to the internal copy the reference runs used, so the problem set is the same,
in the same order.

On-disk schema (`$SIMPLY_DATASETS/livecodebench/livecodebench_v5.jsonl`, one
JSON object per line, upstream order, every value a string)::

    question_id question_title question_content platform contest_id
    contest_date difficulty starter_code func_name
    public_test_cases   # JSON string: [{input, output, testtype}, ...]
    private_test_cases  # JSON string, already decoded from the upstream
                        # base64+zlib+pickle blob by the staging script

The decoded private tests are ~1.1 GiB (6099 tests over 167 problems), hence
JSONL + a line-offset index: a problem is parsed only when it is read.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import mmap
import os
from typing import Any

from etils import epath
from simply.data_lib import DataSourceRegistry  # pylint: disable=g-importing-member


# Asserted by the loader: a silent upstream reshuffle should fail the eval, not
# quietly change the denominator of `accuracy`. Mirrors the build-time checks in
# setup/build_datasets.py (LCB_V5_COUNT / LCB_V5_PUBLIC_TESTS / ...).
LCB_V5_COUNT = 167

DEFAULT_LCB_V5_PATH = 'livecodebench/livecodebench_v5.jsonl'

# Prompt wording of the LiveCodeBench harness, kept verbatim from the internal
# path (cms `data/factories/livecodebench.py`), which in turn mirrors the
# upstream LiveCodeBench repo. FIXED: it is part of the eval protocol.
SYSTEM_MESSAGE = (
    'You are an expert Python programmer. You will be given a question'
    ' (problem specification) and will generate a correct Python program that'
    ' matches the specification and passes all tests.'
)
_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    'You will use the following starter code to write the solution to the'
    ' problem and enclose your code within delimiters.'
)
_FORMATTING_WITHOUT_STARTER_CODE = (
    'Read the inputs from stdin solve the problem and write the answer to'
    ' stdout (do not directly test on the sample inputs). Enclose your code'
    ' within delimiters as follows.'
)


def datasets_dir() -> str:
  """Root of the staged datasets; read late so tests can point it elsewhere."""
  return os.getenv('SIMPLY_DATASETS', os.path.expanduser('~/.cache/simply/datasets/'))


def make_prompt(question_content: str, starter_code: str) -> str:
  """Builds the LiveCodeBench prompt (FIXED; do not reword).

  Args:
    question_content: The problem statement.
    starter_code: The class/function stub for `functional` problems, or ''.

  Returns:
    The user-turn prompt text.
  """
  prompt = f'### Question:\n{question_content}\n\n'
  if starter_code:
    prompt += f'### Format: {_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'
    prompt += f'```python\n{starter_code}\n```\n\n'
  else:
    prompt += f'### Format: {_FORMATTING_WITHOUT_STARTER_CODE}\n'
    prompt += '```python\n# YOUR CODE HERE\n```\n\n'
  prompt += '### Answer: (use the provided format with backticks)\n\n'
  return f'{SYSTEM_MESSAGE}\n' + prompt


def to_example(row: dict[str, Any]) -> dict[str, Any]:
  """Converts one staged row into the example dict the eval consumes.

  The shape (`example_id` / `prompt` / `auxiliary`) is the one the internal
  loader produced, because `lcb_sampling_lib` and the grader read it.

  Args:
    row: One staged JSONL record.

  Returns:
    The example dict.
  """
  return dict(
      example_id=row['question_id'],
      prompt=make_prompt(row['question_content'], row.get('starter_code', '')),
      auxiliary=dict(
          public_test_cases=row['public_test_cases'],
          private_test_cases=row['private_test_cases'],
          func_name=row.get('func_name', ''),
          question=row['question_content'],
          starter_code=row.get('starter_code', ''),
      ),
  )


class _JsonlIndex:
  """Byte offsets of every line, so one problem can be parsed on demand.

  A local file is mmap'd (the 1.1 GiB of test data stays in the page cache and
  out of the heap); a remote one (`gs://...`) is read once into memory.
  """

  def __init__(self, path: epath.PathLike):
    path = os.fspath(path)
    if '://' in path:
      self._data = epath.Path(path).read_bytes()
    else:
      with open(path, 'rb') as f:
        self._data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    data = self._data
    offsets, start = [], 0
    while start < len(data):
      end = data.find(b'\n', start)
      if end == -1:
        end = len(data)
      if data[start:end].strip():
        offsets.append((start, end))
      start = end + 1
    self._offsets = offsets

  def __len__(self) -> int:
    return len(self._offsets)

  def __getitem__(self, index: int) -> dict[str, Any]:
    start, end = self._offsets[index]
    return json.loads(self._data[start:end])


@functools.partial(
    DataSourceRegistry.register, name='simply_json:livecodebench_v5'
)
@dataclasses.dataclass(frozen=True)
class LiveCodeBenchV5:
  """LiveCodeBench v5 code-generation dataset (tests carried per example).

  Ported from the internal `lcb_data_lib.py`: same registered name, same
  example shape, same 167 problems -- only the source changes (staged JSONL
  instead of the internal leapfrog loader).

  Attributes:
    path: Override for the staged JSONL file.
    start_index: Optional slice start (for smoke runs).
    end_index: Optional slice end.
    expected_count: Row count the file must have before slicing; 0 disables the
      check.
  """

  path: str = ''
  start_index: int | None = None
  end_index: int | None = None
  expected_count: int = LCB_V5_COUNT

  def resolved_path(self) -> str:
    return self.path or os.path.join(datasets_dir(), DEFAULT_LCB_V5_PATH)

  @functools.cached_property
  def _index(self) -> _JsonlIndex:
    path = self.resolved_path()
    if not epath.Path(path).exists():
      raise FileNotFoundError(
          f'LiveCodeBench v5 not staged at {path}. Run: python -m'
          ' tasks.research_bench.setup.prepare_assets --datasets'
          ' (or set SIMPLY_DATASETS to the directory holding'
          f' {DEFAULT_LCB_V5_PATH}).'
      )
    index = _JsonlIndex(path)
    if self.expected_count and len(index) != self.expected_count:
      raise ValueError(
          f'{path}: {len(index)} problems, expected {self.expected_count}.'
          ' The graded metric is a mean over a FIXED problem set; refusing to'
          ' score a different one.'
      )
    return index

  @functools.cached_property
  def _range(self) -> range:
    return range(*slice(self.start_index, self.end_index).indices(
        len(self._index)
    ))

  def __len__(self) -> int:
    return len(self._range)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return to_example(self._index[self._range[index]])

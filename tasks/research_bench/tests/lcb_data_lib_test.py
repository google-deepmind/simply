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

"""Tests for the LiveCodeBench v5 data source.

The synthetic fixture covers the contract (shape, prompt, slicing, count
check); the real-data test covers the staged 167-problem file and skips
cleanly when it has not been built.
"""

import json
import os

from absl.testing import absltest
from simply import data_lib as core_data_lib
from tasks.research_bench import lcb_data_lib


def _row(question_id: str, starter_code: str = '') -> dict[str, str]:
  """An LCB-shaped staged row (as `setup/build_datasets.py` writes them)."""
  return dict(
      question_id=question_id,
      question_title=f'title {question_id}',
      question_content=f'solve {question_id}',
      platform='leetcode' if starter_code else 'atcoder',
      contest_id='c1',
      contest_date='2024-10-01T00:00:00',
      difficulty='easy',
      starter_code=starter_code,
      func_name='solveIt' if starter_code else '',
      public_test_cases=json.dumps(
          [dict(input='1\n', output='2\n', testtype='stdin')]
      ),
      private_test_cases=json.dumps(
          [dict(input='3\n', output='4\n', testtype='stdin')]
      ),
  )


class LiveCodeBenchV5Test(absltest.TestCase):

  def _write(self, rows) -> str:
    path = os.path.join(self.create_tempdir().full_path, 'lcb.jsonl')
    with open(path, 'w') as f:
      for row in rows:
        f.write(json.dumps(row) + '\n')
    return path

  def test_example_shape(self):
    path = self._write([_row('q0'), _row('q1', 'class Solution:\n    pass')])
    source = lcb_data_lib.LiveCodeBenchV5(path=path, expected_count=2)
    self.assertLen(source, 2)

    example = source[0]
    self.assertEqual(example['example_id'], 'q0')
    self.assertEqual(
        sorted(example['auxiliary']),
        ['func_name', 'private_test_cases', 'public_test_cases', 'question',
         'starter_code'],
    )
    self.assertLen(json.loads(example['auxiliary']['public_test_cases']), 1)
    self.assertLen(json.loads(example['auxiliary']['private_test_cases']), 1)

  def test_prompt_matches_the_fixed_livecodebench_wording(self):
    path = self._write([_row('q0'), _row('q1', 'class Solution:\n    pass')])
    source = lcb_data_lib.LiveCodeBenchV5(path=path, expected_count=2)

    stdin_prompt = source[0]['prompt']
    self.assertStartsWith(stdin_prompt, lcb_data_lib.SYSTEM_MESSAGE)
    self.assertIn('### Question:\nsolve q0', stdin_prompt)
    self.assertIn('Read the inputs from stdin', stdin_prompt)
    self.assertIn('# YOUR CODE HERE', stdin_prompt)

    starter_prompt = source[1]['prompt']
    self.assertIn('following starter code', starter_prompt)
    self.assertIn('```python\nclass Solution:\n    pass\n```', starter_prompt)

  def test_slicing(self):
    path = self._write([_row(f'q{i}') for i in range(5)])
    source = lcb_data_lib.LiveCodeBenchV5(
        path=path, start_index=1, end_index=3, expected_count=5
    )
    self.assertLen(source, 2)
    self.assertEqual(
        [source[i]['example_id'] for i in range(2)], ['q1', 'q2']
    )

  def test_wrong_problem_count_is_fatal(self):
    path = self._write([_row('q0')])
    source = lcb_data_lib.LiveCodeBenchV5(path=path)  # expects 167
    with self.assertRaisesRegex(ValueError, '1 problems, expected 167'):
      len(source)

  def test_missing_file_names_the_setup_command(self):
    source = lcb_data_lib.LiveCodeBenchV5(path='/nonexistent/lcb.jsonl')
    with self.assertRaisesRegex(FileNotFoundError, 'prepare_assets'):
      len(source)

  def test_registered_under_the_internal_name(self):
    self.assertIn(
        'simply_json:livecodebench_v5', core_data_lib.DataSourceRegistry.keys()
    )


class StagedLiveCodeBenchV5Test(absltest.TestCase):
  """Runs against the real staged asset when `prepare_assets` has been run."""

  def setUp(self):
    super().setUp()
    self.source = lcb_data_lib.LiveCodeBenchV5()
    if not os.path.exists(self.source.resolved_path()):
      self.skipTest(f'{self.source.resolved_path()} not staged')

  def test_counts_and_decoded_tests(self):
    self.assertLen(self.source, lcb_data_lib.LCB_V5_COUNT)
    n_public = n_private = 0
    testtypes = set()
    for i in range(len(self.source)):
      aux = self.source[i]['auxiliary']
      public = json.loads(aux['public_test_cases'])
      private = json.loads(aux['private_test_cases'])
      self.assertNotEmpty(public)
      self.assertNotEmpty(private)
      n_public += len(public)
      n_private += len(private)
      testtypes.update(t['testtype'] for t in public + private)
    # Pinned by setup/build_datasets.py; a change here means a different
    # benchmark, not a different loader.
    self.assertEqual(n_public, 441)
    self.assertEqual(n_private, 6099)
    self.assertEqual(testtypes, {'stdin', 'functional'})

  def test_first_problem_is_upstream_order(self):
    self.assertEqual(self.source[0]['example_id'], 'abc374_c')


if __name__ == '__main__':
  absltest.main()

# Copyright 2026 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
from unittest import mock

from absl.testing import absltest

from simply.agent import env
from simply.agent import tools


class BashToolTest(absltest.TestCase):

  def test_execute_without_timeout(self):
    tool = tools.BashTool(executor=env.execute_bash_locally)
    with mock.patch.object(subprocess, 'run') as mock_run:
      mock_run.return_value.stdout = ''
      mock_run.return_value.stderr = ''
      mock_run.return_value.returncode = 0

      action, observation = tool.execute(
          args_json='{"command": "sleep 10", "timeout": null}'
      )

    self.assertIsNotNone(action)
    self.assertEqual(action.to_llm(), 'COMMAND:\nsleep 10\nTIMEOUT: None')
    self.assertIn('RETURN CODE: 0', observation)
    self.assertIsNone(action.timeout)
    mock_run.assert_called_once_with(
        'sleep 10',
        shell=True,
        check=False,
        text=True,
        encoding='utf-8',
        errors='backslashreplace',
        capture_output=True,
        timeout=None,
        cwd=None,
    )

  def test_execute_success(self):
    tool = tools.BashTool(executor=env.execute_bash_locally)
    with mock.patch.object(subprocess, 'run') as mock_run:
      mock_run.return_value.stdout = 'hello'
      mock_run.return_value.stderr = ''
      mock_run.return_value.returncode = 0

      action, observation = tool.execute(args_json='{"command": "echo hello"}')

      self.assertIsNotNone(action)
      self.assertIn('STDOUT:\nhello', observation)
      self.assertIn('RETURN CODE: 0', observation)
      mock_run.assert_called_once()

  def test_execute_timeout(self):
    tool = tools.BashTool(executor=env.execute_bash_locally)
    with mock.patch.object(subprocess, 'run') as mock_run:
      mock_run.side_effect = subprocess.TimeoutExpired(
          cmd='sleep 10', timeout=1
      )

      action, observation = tool.execute(
          args_json='{"command": "sleep 10", "timeout": 1}'
      )

      self.assertIsNotNone(action)
      self.assertIn('Command timed out', observation)

  def test_input_validation_error(self):
    tool = tools.BashTool(executor=env.execute_bash_locally)
    # Missing 'command' argument
    action, observation = tool.execute(args_json='{}')
    self.assertIsNone(action)
    self.assertIn('Error: Invalid arguments', observation)

  def test_output_truncation(self):
    tool = tools.BashTool(executor=env.execute_bash_locally)
    stdout_head = '8293ifejwifjwewijr'
    stdout_tail = '23ijfwojir32jijfjwsljf'
    stderr_head = '3298ufjiwjfjowsj'
    stderr_tail = 'ijfjwoijfowu9u32rjjfwoj'
    dummy_text = ' word ' * 1000
    max_output_length = 1000
    args_json = (
        '{"command": "echo hello", "max_output_length": %d}' % max_output_length
    )
    # only stdout is too big
    with mock.patch.object(subprocess, 'run') as mock_run:
      mock_run.return_value.stdout = (
          f'{stdout_head}{dummy_text}{stdout_tail}'
      )
      entire_stderr = f'{stderr_head}{stderr_tail}'
      mock_run.return_value.stderr = entire_stderr
      mock_run.return_value.returncode = 0
      action, observation = tool.execute(args_json=args_json)
      self.assertIsNotNone(action)
      self.assertIn(stdout_head, observation)
      self.assertIn(stdout_tail, observation)
      self.assertIn(entire_stderr, observation)
      self.assertIn('chars omitted', observation)
      self.assertIn('RETURN CODE: 0\n\nTIME ELAPSED: ', observation)
      self.assertLessEqual(len(observation), max_output_length)

    # both stdout and stderr are too big, stderr is prioritized.
    with mock.patch.object(subprocess, 'run') as mock_run:
      mock_run.return_value.stdout = (
          f'{stdout_head}{dummy_text}{stdout_tail}'
      )
      mock_run.return_value.stderr = (
          f'{stderr_head}{dummy_text}{stderr_tail}'
      )
      mock_run.return_value.returncode = 0
      action, observation = tool.execute(args_json=args_json)
      self.assertIsNotNone(action)
      self.assertNotIn(stdout_head, observation)
      self.assertNotIn(stdout_tail, observation)
      self.assertIn(stderr_head, observation)
      self.assertIn(stderr_tail, observation)
      self.assertIn('chars omitted', observation)
      self.assertIn('RETURN CODE: 0\n\nTIME ELAPSED: ', observation)
      self.assertLessEqual(len(observation), max_output_length)

    # normal case, no truncation
    with mock.patch.object(subprocess, 'run') as mock_run:
      mock_run.return_value.stdout = f'{stdout_head}{stdout_tail}'
      mock_run.return_value.stderr = f'{stderr_head}{stderr_tail}'
      mock_run.return_value.returncode = 0
      action, observation = tool.execute(args_json=args_json)
      self.assertIsNotNone(action)
      self.assertIn(stdout_head, observation)
      self.assertIn(stdout_tail, observation)
      self.assertIn(stderr_head, observation)
      self.assertIn(stderr_tail, observation)
      self.assertNotIn('chars omitted', observation)
      self.assertIn('RETURN CODE: 0\n\nTIME ELAPSED: ', observation)
      self.assertLessEqual(len(observation), max_output_length)


if __name__ == '__main__':
  absltest.main()

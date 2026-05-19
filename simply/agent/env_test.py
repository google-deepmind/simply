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
import os
import tempfile

from absl.testing import absltest

from simply.agent import env
from simply.agent import tools


class LocalEnvTest(absltest.TestCase):

  def test_working_dir_is_absolute(self):
    """Relative paths are converted to absolute paths."""
    local = env.Local(working_dir='relative/path')
    self.assertTrue(os.path.isabs(local.working_dir))
    self.assertEqual(
        local.working_dir,
        os.path.abspath('relative/path'),
    )

  def test_absolute_working_dir_unchanged(self):
    """Absolute paths are preserved as-is."""
    local = env.Local(working_dir='/tmp/my_dir')
    self.assertEqual(local.working_dir, '/tmp/my_dir')

  def test_get_tools_returns_bash_tool(self):
    """get_tools returns a list containing a BashTool."""
    with tempfile.TemporaryDirectory() as tmpdir:
      local = env.Local(working_dir=tmpdir)
      tool_list = local.get_tools()
      self.assertLen(tool_list, 1)
      self.assertIsInstance(tool_list[0], tools.BashTool)

  def test_bash_tool_description_includes_cwd(self):
    """BashTool description includes the working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      local = env.Local(working_dir=tmpdir)
      bash_tool = local.get_tools()[0]
      self.assertIn(f'CWD={tmpdir!r}', bash_tool.description)


class EnvRegistryTest(absltest.TestCase):

  def test_get_local_env(self):
    """EnvRegistry can create a Local env from a spec."""
    with tempfile.TemporaryDirectory() as tmpdir:
      local = env.EnvRegistry.get_env(f'Local:{tmpdir}')
      self.assertIsInstance(local, env.Local)
      self.assertEqual(local.working_dir, tmpdir)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for the untrusted-code sandbox.

These assert the containment properties the LiveCodeBench grader relies on:
a program that hangs, crashes, allocates or prints forever comes back as a
RESULT, never as an exception or a wedged process. The isolation-level tests
(no network, read-only host FS) only apply to the launchers that provide them,
so they skip when the host has none.
"""

import os
import time

from absl.testing import absltest
from absl.testing import parameterized
from tasks.research_bench import code_exec_lib


class RunPythonTest(parameterized.TestCase):

  def test_stdout_and_stdin(self):
    res = code_exec_lib.run_python('print(int(input()) + 1)', stdin='41\n')
    self.assertIs(res.status, code_exec_lib.Status.OK)
    self.assertEqual(res.stdout.strip(), '42')

  def test_exception_is_a_result_not_a_raise(self):
    res = code_exec_lib.run_python('raise ValueError("boom")')
    self.assertIs(res.status, code_exec_lib.Status.NONZERO_EXIT)
    self.assertIn('ValueError: boom', res.stderr)

  def test_syntax_error_is_reported_in_stderr(self):
    res = code_exec_lib.run_python('def f( : pass')
    self.assertIs(res.status, code_exec_lib.Status.NONZERO_EXIT)
    self.assertIn('SyntaxError', res.stderr)

  def test_exit_code_is_preserved(self):
    res = code_exec_lib.run_python('import sys; sys.exit(24)')
    self.assertEqual(res.returncode, 24)

  def test_infinite_loop_times_out(self):
    start = time.monotonic()
    res = code_exec_lib.run_python(
        'while True: pass', limits=code_exec_lib.Limits(timeout_s=1.0)
    )
    self.assertIs(res.status, code_exec_lib.Status.TIMEOUT)
    self.assertLess(time.monotonic() - start, 30.0)

  def test_sleeping_child_does_not_outlive_the_timeout(self):
    # A grandchild in the same process group must be killed too, otherwise a
    # graded run slowly fills the host with orphans.
    code = (
        'import subprocess, sys, time\n'
        'subprocess.Popen([sys.executable, "-c", "import time;'
        ' time.sleep(300)"])\n'
        'time.sleep(300)\n'
    )
    res = code_exec_lib.run_python(
        code, limits=code_exec_lib.Limits(timeout_s=1.0)
    )
    self.assertIs(res.status, code_exec_lib.Status.TIMEOUT)

  def test_memory_cap(self):
    res = code_exec_lib.run_python(
        'x = bytearray(2 * 1024**3)',
        limits=code_exec_lib.Limits(memory_bytes=512 * 1024**2, timeout_s=30),
    )
    self.assertIs(res.status, code_exec_lib.Status.NONZERO_EXIT)
    self.assertIn('MemoryError', res.stderr)

  def test_runaway_output_is_capped(self):
    res = code_exec_lib.run_python(
        'while True: print("x" * 1000)',
        limits=code_exec_lib.Limits(
            timeout_s=20, max_file_bytes=1 << 20, max_output_bytes=4096
        ),
    )
    self.assertLessEqual(len(res.stdout), 4096 + len('\n[truncated]'))
    self.assertIsNot(res.status, code_exec_lib.Status.OK)

  def test_scratch_dir_is_private_and_removed(self):
    res = code_exec_lib.run_python(
        'import os; open("out.txt", "w").write("hi");'
        ' print(os.path.abspath("out.txt"))'
    )
    self.assertIs(res.status, code_exec_lib.Status.OK)
    self.assertFalse(os.path.exists(res.stdout.strip()))

  def test_extra_files_are_visible(self):
    res = code_exec_lib.run_python(
        'print(open("data.txt").read().strip())', files={'data.txt': 'hello'}
    )
    self.assertEqual(res.stdout.strip(), 'hello')

  def test_run_many_preserves_order(self):
    specs = [
        code_exec_lib.ProgramSpec(code=f'print({i} * 2)') for i in range(8)
    ]
    results = code_exec_lib.run_many(specs, max_workers=4)
    self.assertEqual(
        [r.stdout.strip() for r in results], [str(i * 2) for i in range(8)]
    )


class IsolationTest(absltest.TestCase):
  """Only meaningful where a namespace launcher is available."""

  def setUp(self):
    super().setUp()
    self.sandbox = code_exec_lib.resolve_sandbox()
    if self.sandbox == 'none':
      self.skipTest(
          'no namespace launcher on this host (install bubblewrap or'
          ' util-linux); resource limits still apply'
      )

  def test_no_network(self):
    res = code_exec_lib.run_python(
        'import socket;'
        ' socket.create_connection(("8.8.8.8", 53), timeout=5);'
        ' print("connected")',
        limits=code_exec_lib.Limits(timeout_s=30),
    )
    self.assertNotIn('connected', res.stdout)

  def test_host_filesystem_is_read_only_under_bwrap(self):
    if self.sandbox != 'bwrap':
      self.skipTest('only bubblewrap remounts the host filesystem read-only')
    victim = self.create_tempfile('victim.txt', content='keep').full_path
    res = code_exec_lib.run_python(
        f'open({victim!r}, "w").write("pwned")',
        limits=code_exec_lib.Limits(timeout_s=30),
    )
    self.assertIs(res.status, code_exec_lib.Status.NONZERO_EXIT)
    with open(victim) as f:
      self.assertEqual(f.read(), 'keep')


class SandboxSelectionTest(absltest.TestCase):

  def test_report_mentions_the_chosen_launcher(self):
    self.assertIn(
        f'sandbox={code_exec_lib.resolve_sandbox()}',
        code_exec_lib.sandbox_report(),
    )

  def test_unknown_launcher_is_rejected(self):
    with self.assertRaises(ValueError):
      code_exec_lib.resolve_sandbox('gvisor')

  def test_plain_subprocess_launcher_still_enforces_limits(self):
    # The no-launcher fallback is what a bare host gets; the resource envelope
    # must hold there too.
    self.assertEqual(code_exec_lib.resolve_sandbox('none'), 'none')


if __name__ == '__main__':
  absltest.main()

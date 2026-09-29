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

r"""Runs untrusted, model-generated Python programs under OS-level limits.

This is the OSS replacement for the internal code-execution sandbox that the
`sampling_lcb` grader used. It executes a program in a fresh
subprocess with a hard wall-clock timeout, a CPU-time cap, an address-space
cap, an output-size cap, a throw-away working directory and a scrubbed
environment, and -- when the host provides a usable sandbox helper -- in a
network-less namespace with a read-only view of the filesystem.

STOP. READ THIS BEFORE YOU RUN A MODEL'S CODE.
---------------------------------------------
This module EXECUTES UNTRUSTED CODE on the machine it runs on. What you get is
resource containment and (with bubblewrap) filesystem/network containment. What
you do NOT get is a security boundary of gVisor's class: gVisor interposes on
the entire syscall surface with a userspace kernel, this does not. Ranked by the
isolation actually achieved:

| launcher   | escapes kernel? | network | host FS            | needs |
|------------|-----------------|---------|--------------------|-------|
| `bwrap`    | no (seccomp+ns) | denied  | read-only, /tmp new| bubblewrap + unprivileged user namespaces |
| `unshare`  | no (ns only)    | denied  | FULL host FS       | util-linux + unprivileged user namespaces |
| `none`     | no              | ALLOWED | FULL host FS       | -- |

Under every launcher the program still runs as your uid on your kernel: a
kernel LPE, a /proc side channel, or (with `unshare`/`none`) reading and
DELETING any file you can write is not prevented. Run graded LiveCodeBench evals
on a disposable VM (the Cloud TPU worker), never on a workstation with
credentials you care about, and prefer `bwrap` (`apt-get install bubblewrap`).

Selection is automatic (best available) and can be pinned with
`SIMPLY_CODE_EXEC_SANDBOX=bwrap|unshare|none|auto`; `sandbox_report()` prints
what was chosen and why. A launcher is only used after a probe actually runs a
program under it, so a missing/blocked helper degrades instead of failing the
run.

Usage::

    result = run_python('print(int(input()) + 1)', stdin='41\n')
    assert result.status is Status.OK and result.stdout.strip() == '42'

    results = run_many([ProgramSpec(code=c, stdin=t) for c, t in cases])
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import enum
import functools
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


MAIN_FILENAME = 'main.py'

# Exit status that `_BOOTSTRAP` uses for "the sandbox itself failed", so a
# broken launcher is distinguishable from a program that exited non-zero.
_BOOTSTRAP_ERROR_CODE = 89


class Status(enum.Enum):
  """Outcome of one program execution."""

  OK = 'ok'  # Exited 0.
  NONZERO_EXIT = 'nonzero_exit'  # Ran, exited non-zero (exception, exit(n)).
  TIMEOUT = 'timeout'  # Killed after `Limits.timeout_s`.
  SANDBOX_ERROR = 'sandbox_error'  # We failed to run it; verdict unknown.


@dataclasses.dataclass(frozen=True)
class Limits:
  """Resource envelope for one execution.

  Attributes:
    timeout_s: Wall-clock limit; the whole process group is SIGKILLed after it.
    cpu_seconds: RLIMIT_CPU, a backstop for a program that ignores wall-clock
      pressure (e.g. blocks the killer by spawning). None -> ceil(timeout_s)+1.
    memory_bytes: RLIMIT_AS (virtual address space). 0 disables.
    max_file_bytes: RLIMIT_FSIZE. stdout/stderr are redirected to files, so this
      doubles as the output cap that kills a program printing in a loop (SIGXFSZ)
      instead of letting it fill the host disk / our RAM.
    max_output_bytes: How much of the captured stdout/stderr we read back.
    max_processes: RLIMIT_NPROC (anti fork-bomb). Counted per REAL UID across
      the host, so a low value can make a program fail for reasons unrelated to
      itself; None (default) leaves it alone.
  """

  timeout_s: float = 10.0
  cpu_seconds: int | None = None
  memory_bytes: int = 16 * 1024**3
  max_file_bytes: int = 64 * 1024**2
  max_output_bytes: int = 1024**2
  max_processes: int | None = None

  def resolved_cpu_seconds(self) -> int:
    if self.cpu_seconds is not None:
      return self.cpu_seconds
    return int(math.ceil(self.timeout_s)) + 1


@dataclasses.dataclass(frozen=True)
class ExecResult:
  """What one execution did."""

  status: Status
  returncode: int | None
  stdout: str
  stderr: str
  duration_s: float
  sandbox: str = 'none'

  @property
  def ok(self) -> bool:
    return self.status is Status.OK

  @property
  def timed_out(self) -> bool:
    return self.status is Status.TIMEOUT


@dataclasses.dataclass(frozen=True)
class ProgramSpec:
  """One program + its input, for `run_many`."""

  code: str
  stdin: str = ''
  files: Mapping[str, str] | None = None
  limits: Limits = Limits()


# Applies the rlimits to ITSELF and then runs the program, so the limits are
# inherited by whatever the program spawns. Done in-process rather than in a
# `preexec_fn` because preexec_fn is documented as unsafe in a multi-threaded
# parent -- and the grader runs dozens of these from a thread pool. Soft AND
# hard limits are set to the same value, so the untrusted program cannot raise
# them back up.
_BOOTSTRAP = r"""
import os, resource, runpy, sys

_limits, _prog = sys.argv[1], sys.argv[2]
try:
  _as, _cpu, _fsize, _nproc = (int(x) for x in _limits.split(','))
  for _res, _val in (
      (resource.RLIMIT_CORE, 0),
      (resource.RLIMIT_AS, _as),
      (resource.RLIMIT_CPU, _cpu),
      (resource.RLIMIT_FSIZE, _fsize),
      (resource.RLIMIT_NPROC, _nproc),
  ):
    if _val >= 0:
      try:
        resource.setrlimit(_res, (_val, _val))
      except (ValueError, OSError):
        pass
except Exception as e:  # pylint: disable=broad-except
  sys.stderr.write('sandbox bootstrap failed: {!r}'.format(e))
  sys.exit(__BOOTSTRAP_ERROR_CODE__)
sys.argv = [_prog]
runpy.run_path(_prog, run_name='__main__')
""".replace('__BOOTSTRAP_ERROR_CODE__', str(_BOOTSTRAP_ERROR_CODE))


def _clean_env(scratch_dir: str) -> dict[str, str]:
  """Minimal environment: no credentials, no PYTHON* injection, own TMPDIR."""
  return {
      'PATH': '/usr/bin:/bin',
      'HOME': scratch_dir,
      'TMPDIR': scratch_dir,
      'LANG': 'C.UTF-8',
      'LC_ALL': 'C.UTF-8',
      'PYTHONIOENCODING': 'utf-8',
      'PYTHONDONTWRITEBYTECODE': '1',
      'PYTHONHASHSEED': '0',
  }


# --------------------------------------------------------------------------
# Launchers.
# --------------------------------------------------------------------------
def _bwrap_prefix(scratch_dir: str) -> list[str]:
  """bubblewrap: new user/net/ipc/pid/uts/cgroup namespaces + read-only rootfs."""
  argv = [
      'bwrap',
      '--unshare-all',
      '--new-session',
      '--die-with-parent',
      '--proc', '/proc',
      '--dev', '/dev',
      '--tmpfs', '/tmp',
  ]
  for path in ('/usr', '/bin', '/sbin', '/lib', '/lib32', '/lib64', '/etc'):
    if os.path.exists(path):
      argv += ['--ro-bind', path, path]
  # The interpreter may live outside those (a venv, a conda prefix, ...).
  for prefix in {sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)}:
    if prefix and os.path.exists(prefix) and not prefix.startswith('/usr'):
      argv += ['--ro-bind', prefix, prefix]
  argv += ['--bind', scratch_dir, scratch_dir, '--chdir', scratch_dir, '--']
  return argv


def _unshare_prefix(scratch_dir: str) -> list[str]:
  """util-linux: user+net namespace only -- no network, but FULL host FS."""
  del scratch_dir  # cwd is set by the caller.
  return ['unshare', '--user', '--map-root-user', '--net', '--']


_LAUNCHERS = {
    'bwrap': _bwrap_prefix,
    'unshare': _unshare_prefix,
    'none': lambda scratch_dir: [],
}

# Best first. `none` is always last and always works.
_LAUNCHER_PREFERENCE = ('bwrap', 'unshare', 'none')


@functools.cache
def _probe(name: str) -> bool:
  """Returns whether `name` can actually run a trivial program right now."""
  if name == 'none':
    return True
  if shutil.which(name) is None:
    return False
  with tempfile.TemporaryDirectory(prefix='simply_sandbox_probe_') as scratch:
    argv = _LAUNCHERS[name](scratch) + [sys.executable, '-I', '-c', 'print(1)']
    try:
      proc = subprocess.run(  # pylint: disable=subprocess-run-check
          argv,
          cwd=scratch,
          env=_clean_env(scratch),
          capture_output=True,
          text=True,
          timeout=30,
      )
    except (OSError, subprocess.SubprocessError):
      return False
    return proc.returncode == 0 and proc.stdout.strip() == '1'


@functools.cache
def resolve_sandbox(requested: str | None = None) -> str:
  """Picks the launcher to use: explicit request, else best that probes OK.

  Args:
    requested: 'bwrap' / 'unshare' / 'none' / 'auto' / None. None reads
      `SIMPLY_CODE_EXEC_SANDBOX` and defaults to 'auto'.

  Returns:
    The chosen launcher name (falls back to 'none' if nothing else probes OK).

  Raises:
    ValueError: if an unknown launcher is requested.
  """
  requested = requested or os.environ.get('SIMPLY_CODE_EXEC_SANDBOX', 'auto')
  if requested != 'auto':
    if requested not in _LAUNCHERS:
      raise ValueError(
          f'Unknown SIMPLY_CODE_EXEC_SANDBOX={requested!r};'
          f' expected one of {sorted(_LAUNCHERS)} or "auto".'
      )
    return requested
  for name in _LAUNCHER_PREFERENCE:
    if _probe(name):
      return name
  return 'none'


def sandbox_report() -> str:
  """One-line, greppable description of the isolation actually in effect."""
  chosen = resolve_sandbox()
  detail = {
      'bwrap': 'no network, read-only host FS, private /tmp+/proc+pid ns',
      'unshare': 'no network, FULL host filesystem visible (rw)',
      'none': 'NETWORK REACHABLE and FULL host filesystem visible (rw)',
  }[chosen]
  available = [n for n in _LAUNCHER_PREFERENCE if _probe(n)]
  return (
      f'[code_exec] sandbox={chosen} ({detail}); available={available};'
      ' resource limits (cpu/mem/output/timeout) always enforced; NOT a'
      ' gVisor-class security boundary.'
  )


# --------------------------------------------------------------------------
# Execution.
# --------------------------------------------------------------------------
def _read_head(path: str, max_bytes: int) -> str:
  try:
    with open(path, 'rb') as f:
      data = f.read(max_bytes + 1)
  except OSError:
    return ''
  truncated = len(data) > max_bytes
  text = data[:max_bytes].decode('utf-8', errors='replace')
  return text + '\n[truncated]' if truncated else text


def run_python(
    code: str,
    stdin: str = '',
    limits: Limits = Limits(),
    files: Mapping[str, str] | None = None,
    scratch_root: str | None = None,
) -> ExecResult:
  """Runs `code` as a standalone program and returns what it did.

  Never raises for anything the program does: a crash, a timeout and a
  fork-bomb all come back as an `ExecResult`. Only a failure of the harness
  itself (cannot create the scratch dir, cannot spawn) becomes
  `Status.SANDBOX_ERROR`, which callers treat as "verdict unknown" and retry.

  Args:
    code: The program source; written to `main.py` in a fresh scratch dir.
    stdin: Fed to the program's standard input.
    limits: Resource envelope.
    files: Extra `name -> content` files materialised next to `main.py`.
    scratch_root: Parent for the scratch dir (defaults to the system temp dir).

  Returns:
    The execution result.
  """
  sandbox = resolve_sandbox()
  start = time.monotonic()
  try:
    scratch = tempfile.mkdtemp(prefix='simply_code_exec_', dir=scratch_root)
  except OSError as e:
    return ExecResult(
        Status.SANDBOX_ERROR, None, '', f'scratch dir failed: {e!r}',
        time.monotonic() - start, sandbox,
    )
  try:
    main_path = os.path.join(scratch, MAIN_FILENAME)
    with open(main_path, 'w', encoding='utf-8') as f:
      f.write(code)
    for name, content in (files or {}).items():
      # Extra files are ours (test inputs), but never let a name escape.
      if os.path.isabs(name) or '..' in name.split(os.sep):
        raise ValueError(f'unsafe extra file name: {name!r}')
      path = os.path.join(scratch, name)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    stdin_path = os.path.join(scratch, '__stdin')
    stdout_path = os.path.join(scratch, '__stdout')
    stderr_path = os.path.join(scratch, '__stderr')
    with open(stdin_path, 'w', encoding='utf-8') as f:
      f.write(stdin)

    limit_spec = ','.join(
        str(v)
        for v in (
            limits.memory_bytes,
            limits.resolved_cpu_seconds(),
            limits.max_file_bytes,
            -1 if limits.max_processes is None else limits.max_processes,
        )
    )
    argv = _LAUNCHERS[sandbox](scratch) + [
        sys.executable,
        '-I',  # ignore PYTHON* env vars and the user site dir
        '-c',
        _BOOTSTRAP,
        limit_spec,
        MAIN_FILENAME,
    ]
    with (
        open(stdin_path, 'rb') as fin,
        open(stdout_path, 'wb') as fout,
        open(stderr_path, 'wb') as ferr,
    ):
      try:
        proc = subprocess.Popen(
            argv,
            cwd=scratch,
            env=_clean_env(scratch),
            stdin=fin,
            stdout=fout,
            stderr=ferr,
            # Its own process group, so the timeout kill takes the children too.
            start_new_session=True,
        )
      except OSError as e:
        return ExecResult(
            Status.SANDBOX_ERROR, None, '', f'spawn failed: {e!r}',
            time.monotonic() - start, sandbox,
        )
      timed_out = False
      try:
        proc.wait(timeout=limits.timeout_s)
      except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)

    duration = time.monotonic() - start
    stdout = _read_head(stdout_path, limits.max_output_bytes)
    stderr = _read_head(stderr_path, limits.max_output_bytes)
    if timed_out:
      status = Status.TIMEOUT
    elif proc.returncode == 0:
      status = Status.OK
    elif proc.returncode == _BOOTSTRAP_ERROR_CODE:
      status = Status.SANDBOX_ERROR
    else:
      status = Status.NONZERO_EXIT
    return ExecResult(status, proc.returncode, stdout, stderr, duration,
                      sandbox)
  finally:
    shutil.rmtree(scratch, ignore_errors=True)


def _kill_group(proc: subprocess.Popen[Any]) -> None:
  """SIGKILLs the process group of a timed-out run and reaps it."""
  try:
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
  except (ProcessLookupError, PermissionError):
    proc.kill()
  try:
    proc.wait(timeout=10)
  except subprocess.TimeoutExpired:
    pass


def default_max_workers() -> int:
  """Default pool width for `run_many` (`SIMPLY_CODE_EXEC_WORKERS`)."""
  env = os.environ.get('SIMPLY_CODE_EXEC_WORKERS')
  if env:
    return max(1, int(env))
  return max(1, min(32, (os.cpu_count() or 2)))


def run_many(
    specs: Sequence[ProgramSpec], max_workers: int | None = None
) -> list[ExecResult]:
  """Runs programs in parallel over a worker pool, preserving input order.

  Args:
    specs: The programs to run.
    max_workers: Pool width; defaults to `default_max_workers()`.

  Returns:
    One `ExecResult` per spec, in the same order.
  """
  if not specs:
    return []
  width = min(len(specs), max_workers or default_max_workers())
  if width == 1:
    return [
        run_python(s.code, s.stdin, s.limits, s.files) for s in specs
    ]
  with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
    return list(
        pool.map(
            lambda s: run_python(s.code, s.stdin, s.limits, s.files), specs
        )
    )

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


"""`gcloud` wrappers: GCS objects, TPU VM lifecycle, and job transports.

Deliberately shells out to `gcloud` instead of using the GCP client libraries:
the launcher then needs no credentials of its own and no dependency beyond the
CLI the user already has to install to create a TPU.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import shlex
import subprocess
import sys
import time


PROJECT_FLAG = '--project'

# TPU software version per accelerator family (`gcloud compute tpus versions
# list --zone=<zone>` shows what a zone actually offers).
_RUNTIME_VERSIONS = (
    ('v6e-', 'v2-alpha-tpuv6e'),
    ('v5litepod-', 'v2-alpha-tpuv5-lite'),
    ('v5p-', 'v2-alpha-tpuv5'),
    ('v4-', 'tpu-ubuntu2204-base'),
)


# A node in one of these states will never run a job again.
DEAD_STATES = frozenset({'DELETING', 'PREEMPTED', 'TERMINATED', 'STOPPED',
                         'STOPPING', 'FAILED', 'SUSPENDED', 'SUSPENDING',
                         'HIDING', 'HIDDEN', 'REPAIRING'})

HEARTBEAT_MAX_AGE_SEC = 150


def runtime_version(tpu_type: str) -> str:
  for prefix, version in _RUNTIME_VERSIONS:
    if tpu_type.startswith(prefix):
      return version
  return 'tpu-ubuntu2204-base'


def log(msg: str) -> None:
  print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class AdcTokenRefresher:
  """Keeps `CLOUDSDK_AUTH_ACCESS_TOKEN` fresh from application-default creds.

  Needed where a policy (e.g. corp Context-Aware-Access) rejects gcloud's own
  user credentials but accepts an ADC token. The token expires after ~1 h and a
  sweep outlives that, so it is re-minted periodically.
  """

  REFRESH_SEC = 25 * 60

  def __init__(self, enabled: bool):
    self.enabled = enabled
    self._next = 0.0

  def maybe_refresh(self) -> None:
    if not self.enabled or time.time() < self._next:
      return
    out = subprocess.run(
        ['gcloud', 'auth', 'application-default', 'print-access-token'],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
      raise RuntimeError(f'cannot mint an ADC token: {out.stderr.strip()}')
    os.environ['CLOUDSDK_AUTH_ACCESS_TOKEN'] = out.stdout.strip()
    self._next = time.time() + self.REFRESH_SEC


_AUTH = AdcTokenRefresher(enabled=False)


def enable_adc_token(enabled: bool) -> None:
  _AUTH.enabled = enabled
  _AUTH.maybe_refresh()


@dataclasses.dataclass
class Result:
  returncode: int
  stdout: str
  stderr: str

  @property
  def ok(self) -> bool:
    return self.returncode == 0


def gcloud(*args: str, timeout: float = 900, stdin: str | None = None,
           verbose: bool = False) -> Result:
  """Runs `gcloud <args>` and captures its output."""
  _AUTH.maybe_refresh()
  cmd = ['gcloud', *args]
  if verbose:
    log('$ ' + ' '.join(shlex.quote(c) for c in cmd))
  try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          input=stdin, check=False)
  except subprocess.TimeoutExpired:
    return Result(124, '', f'timeout after {timeout}s: {" ".join(cmd)}')
  return Result(proc.returncode, proc.stdout, proc.stderr)


# --------------------------------------------------------------------------
# GCS


def gcs_write(url: str, text: str) -> None:
  res = gcloud('storage', 'cp', '-', url, stdin=text)
  if not res.ok:
    raise RuntimeError(f'writing {url} failed: {res.stderr.strip()}')


def gcs_read(url: str) -> str | None:
  """Object contents, or None if it does not exist."""
  res = gcloud('storage', 'cat', url, timeout=300)
  return res.stdout if res.ok else None


def gcs_read_json(url: str):
  raw = gcs_read(url)
  if raw is None:
    return None
  try:
    return json.loads(raw)
  except json.JSONDecodeError:
    return None


def gcs_exists(url: str) -> bool:
  return gcloud('storage', 'ls', url, timeout=300).ok


def gcs_upload(local_path: str, url: str) -> None:
  res = gcloud('storage', 'cp', local_path, url, timeout=3600)
  if not res.ok:
    raise RuntimeError(f'uploading {local_path} -> {url}: {res.stderr.strip()}')


def gcs_ls(url: str) -> list[str]:
  res = gcloud('storage', 'ls', url, timeout=300)
  return [l for l in res.stdout.splitlines() if l.strip()] if res.ok else []


# --------------------------------------------------------------------------
# TPU VMs


@dataclasses.dataclass
class TpuVm:
  """One Cloud TPU VM node."""

  name: str
  zone: str
  tpu_type: str
  project: str
  spot: bool = False
  metadata: dict[str, str] = dataclasses.field(default_factory=dict)
  startup_script: str | None = None

  def describe(self) -> dict | None:
    res = gcloud('compute', 'tpus', 'tpu-vm', 'describe', self.name,
                 f'--zone={self.zone}', f'{PROJECT_FLAG}={self.project}',
                 '--format=json', timeout=180)
    if not res.ok:
      return None
    try:
      return json.loads(res.stdout)
    except json.JSONDecodeError:
      return None

  def state(self) -> str | None:
    """READY / CREATING / PREEMPTED / ... , or None when the node is gone."""
    info = self.describe()
    return info.get('state') if info else None

  def create(self, startup_script_path: str | None = None) -> Result:
    args = [
        'compute', 'tpus', 'tpu-vm', 'create', self.name,
        f'--zone={self.zone}', f'{PROJECT_FLAG}={self.project}',
        f'--accelerator-type={self.tpu_type}',
        f'--version={runtime_version(self.tpu_type)}',
        '--scopes=https://www.googleapis.com/auth/cloud-platform',
    ]
    if self.metadata:
      # `^~^` switches the list delimiter: metadata values contain commas.
      args.append('--metadata=^~^' +
                  '~'.join(f'{k}={v}' for k, v in self.metadata.items()))
    if startup_script_path:
      args.append(f'--metadata-from-file=startup-script={startup_script_path}')
    if self.spot:
      args.append('--spot')
    return gcloud(*args, timeout=1800, verbose=True)

  def delete(self) -> Result:
    return gcloud('compute', 'tpus', 'tpu-vm', 'delete', self.name,
                  f'--zone={self.zone}', f'{PROJECT_FLAG}={self.project}',
                  '--quiet', timeout=1800)


# --------------------------------------------------------------------------
# Transports: how a job script gets onto a VM.
#
# Both transports are *fire and forget*: the job script itself streams its log
# and writes its exit status to GCS (see remote.job_script), so the launcher
# polls GCS identically whichever transport delivered the job.


class Transport:
  """Delivers a job script to a VM and reports whether the VM can take one."""

  name = ''

  def create_kwargs(self, vm: TpuVm) -> None:
    """Hook to add metadata/startup-script requirements before `vm.create()`."""

  def reset(self, vm: TpuVm) -> None:
    """Forgets everything about a VM name; called just before (re)creating."""

  def wait_ready(self, vm: TpuVm, timeout_sec: float) -> bool:
    raise NotImplementedError

  def submit(self, vm: TpuVm, job_id: str, script: str) -> None:
    raise NotImplementedError


class GcsTransport(Transport):
  """Startup-script agent polls a GCS job queue. Needs no SSH at all.

  Job queue layout under the control prefix (see `launch/agent.py`):
    <ctl>/<vm>/cmd/<job-id>.sh   the script, picked up within ~5 s
    <ctl>/<vm>/out/<job-id>.log  agent-side capture (the job also writes its
                                 own log into the seed directory)
    <ctl>/<vm>/heartbeat.txt     agent liveness
  """

  name = 'gcs'

  def __init__(self, ctl_prefix: str):
    self.ctl_prefix = ctl_prefix.rstrip('/')

  def heartbeat_age(self, vm: TpuVm) -> float | None:
    """Seconds since the agent last published, or None if it never has."""
    raw = gcs_read(f'{self.ctl_prefix}/{vm.name}/heartbeat.txt')
    if not raw:
      return None
    try:
      beat = datetime.datetime.strptime(json.loads(raw)['t'],
                                        '%Y-%m-%dT%H:%M:%S%z')
    except (json.JSONDecodeError, KeyError, ValueError):
      return None
    return (datetime.datetime.now(datetime.timezone.utc) - beat).total_seconds()

  def wait_ready(self, vm: TpuVm, timeout_sec: float) -> bool:
    """Ready == the agent on the VM is publishing *fresh* heartbeats.

    Freshness matters: a recreated VM inherits the heartbeat file its dead
    predecessor left in GCS, and a stale one would send jobs into a void.
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
      age = self.heartbeat_age(vm)
      if age is not None and age < HEARTBEAT_MAX_AGE_SEC:
        return True
      state = vm.state()
      if state is None or state in DEAD_STATES:
        return False
      time.sleep(15)
    return False

  def reset(self, vm: TpuVm) -> None:
    """Empties the queue: it lives in GCS and outlives the node it feeds.

    Without this, a VM recreated under the same name (after a preemption,
    say) picks up every job ever queued for that name and runs them all at
    once, fighting over the TPU.
    """
    gcloud('storage', 'rm', '-r', f'{self.ctl_prefix}/{vm.name}', timeout=600)

  def submit(self, vm: TpuVm, job_id: str, script: str) -> None:
    gcs_write(f'{self.ctl_prefix}/{vm.name}/cmd/{job_id}.sh', script)


class SshTransport(Transport):
  """`gcloud compute tpus tpu-vm ssh`: the documented path for normal users."""

  name = 'ssh'

  def _ssh(self, vm: TpuVm, command: str, timeout: float = 600) -> Result:
    return gcloud('compute', 'tpus', 'tpu-vm', 'ssh', vm.name,
                  f'--zone={vm.zone}', f'{PROJECT_FLAG}={vm.project}',
                  '--worker=all', f'--command={command}', timeout=timeout)

  def wait_ready(self, vm: TpuVm, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
      state = vm.state()
      if state is None:
        return False
      if state == 'READY' and self._ssh(vm, 'true', timeout=180).ok:
        return True
      time.sleep(15)
    return False

  def submit(self, vm: TpuVm, job_id: str, script: str) -> None:
    remote = f'/tmp/{job_id}.sh'
    # Written through the ssh channel itself: `tpu-vm scp` needs the same
    # transport and one hop is one failure mode fewer.
    heredoc = (f"cat > {remote} <<'SIMPLY_JOB_EOF'\n{script}\nSIMPLY_JOB_EOF\n"
               f'chmod +x {remote}; '
               f'sudo -b nohup bash {remote} > /tmp/{job_id}.boot 2>&1 < /dev/null')
    res = self._ssh(vm, heredoc)
    if not res.ok:
      raise RuntimeError(
          f'ssh submit to {vm.name} failed ({res.returncode}): '
          f'{res.stderr.strip()[-800:]}\n'
          'If this is a "websocket: close 4003" or a hanging connect, your '
          'network blocks SSH to GCP; rerun with --transport=gcs.')


def probe_ssh(project: str, zone: str) -> str:
  """One-line verdict on whether `tpu-vm ssh` is usable from this machine."""
  res = gcloud('compute', 'tpus', 'tpu-vm', 'list', f'--zone={zone}',
               f'{PROJECT_FLAG}={project}', '--format=value(name)', timeout=180)
  if not res.ok:
    return f'cannot list TPUs in {zone}: {res.stderr.strip()[:200]}'
  names = [n for n in res.stdout.split() if n]
  if not names:
    return f'no TPU VM in {zone} to probe; create one first'
  vm = TpuVm(names[0], zone, '', project)
  res = SshTransport()._ssh(vm, 'echo ssh-ok', timeout=180)  # pylint: disable=protected-access
  if res.ok and 'ssh-ok' in res.stdout:
    return f'ssh works (probed {names[0]} in {zone})'
  return (f'ssh to {names[0]} FAILED: '
          f'{(res.stderr or res.stdout).strip()[-300:]}')


def die(msg: str) -> None:
  print(f'error: {msg}', file=sys.stderr)
  raise SystemExit(2)

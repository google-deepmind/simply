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


r"""Runs a research-bench task's multi-seed sweep on Cloud TPU VMs.

This is the OSS replacement for the internal sweep-launcher flow. One command creates the TPUs, runs one seed per VM, and leaves a
submission-shaped directory in GCS:

    gs://<bucket>/<experiment_name>/launch_manifest.json
    gs://<bucket>/<experiment_name>/seed_<seed>/final_result.json
    gs://<bucket>/<experiment_name>/seed_<seed>/log.txt
    gs://<bucket>/<experiment_name>/seed_<seed>/tb_log/...

Example -- the 3-seed submission sweep for a pretraining task:

    python -m tasks.research_bench.launch.launch_gcp \
        --task=pretrain_bpb_byte --experiment_name=bpb_byte_v1 \
        --bucket=gs://my-bucket --seeds=42,43,44 \
        --zone=us-central1-b --tpu-type=v6e-4 --spot --num-workers=3

It is restartable: re-running the same command skips seeds that already have a
`final_result.json`, reuses VMs that are still up, and re-runs seeds whose VM
was preempted. See GCLOUD.md for setup, costs and the SSH-less fallback.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import getpass
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time

from tasks.research_bench.launch import cloud
from tasks.research_bench.launch import remote
from tasks.research_bench.launch import task_defaults

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))

_TAR_EXCLUDES = ('.git', '__pycache__', '*.pyc', '.pytest_cache', '.venv',
                 'venv', 'build', 'dist', '*.egg-info', '.mypy_cache')


def now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


# --------------------------------------------------------------------------
# Experiment layout


@dataclasses.dataclass(frozen=True)
class Experiment:
  """The GCS layout a submission is read back from."""

  bucket: str
  name: str

  @property
  def root(self) -> str:
    return f'{self.bucket.rstrip("/")}/{self.name}'

  @property
  def manifest_url(self) -> str:
    return f'{self.root}/launch_manifest.json'

  @property
  def state_url(self) -> str:
    return f'{self.root}/_ctl/launch_state.json'

  def seed_dir(self, seed: int) -> str:
    return f'{self.root}/seed_{seed}'

  def result_url(self, seed: int) -> str:
    return f'{self.seed_dir(seed)}/final_result.json'

  def log_url(self, seed: int) -> str:
    return f'{self.seed_dir(seed)}/log.txt'

  def status_url(self, job_id: str) -> str:
    return f'{self.root}/_ctl/{job_id}.status'


# --------------------------------------------------------------------------
# Command construction


def _flag_name(flag: str) -> str:
  return flag.lstrip('-').split('=', 1)[0]


def build_command(args, defaults: task_defaults.TaskDefaults, experiment_dir: str,
                  seed: int) -> list[str]:
  """The `python -m <entry>` command line for one seed."""
  config = args.experiment_config or defaults.experiment_config or args.task
  flags = [f'--experiment_config={config}', f'--experiment_dir={experiment_dir}']

  overlay = {**defaults.config_overlay, **json.loads(args.config_overlay)}
  if defaults.seed_flag == 'config_overlay':
    # Core simply has no seed flags: the seeds are config fields, so the sweep
    # varies them through the config overlay.
    overlay.update({'model_seed': seed, 'dataset_seed': seed})
  else:
    flags.append(f'--{defaults.seed_flag}={seed}')
  if overlay:
    flags.append('--config_overlay=' + json.dumps(overlay, sort_keys=True))

  extra = ['--' + f.lstrip('-') for f in args.extra_flag]
  overridden = {_flag_name(f) for f in extra}
  flags += [f'--{f}' for f in defaults.flags if _flag_name(f) not in overridden]
  flags += extra
  if not any(_flag_name(f) == 'alsologtostderr' for f in flags):
    flags.append('--alsologtostderr')
  return ['python', '-u', '-m', args.entry_module, *flags]


def resolve_defaults(args) -> task_defaults.TaskDefaults:
  """Task defaults, with every field overridable by a flag."""
  base = (task_defaults.get(args.task) if args.task
          else task_defaults.TaskDefaults(tpu_type='v6e-4'))
  if not args.task and not args.experiment_config:
    cloud.die('pass --task=<task_id> (or --experiment_config for a raw config)')
  args.apt = args.apt or list(base.apt_packages)
  args.pip_extras = args.pip_extras or base.pip_extras
  args.entry_module = args.entry_module or base.entry_module
  args.tpu_type = args.tpu_type or base.tpu_type
  if args.timeout_min is None:
    # 4x the reference wall-clock: enough slack for a cold VM and a slower
    # accelerator, short enough that a wedged run is not billed all night.
    args.timeout_min = max(60, 4 * base.runtime_min)
  return base


# --------------------------------------------------------------------------
# Repo staging + provenance


def git_info(root: str) -> dict:
  def git(*a):
    res = subprocess.run(['git', '-C', root, *a], capture_output=True,
                         text=True, check=False)
    return res.stdout.strip() if res.returncode == 0 else ''

  return {
      'sha': git('rev-parse', 'HEAD'),
      'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
      'dirty': bool(git('status', '--porcelain')),
      'describe': git('describe', '--always', '--dirty'),
  }


def _tracked_files(root: str) -> bytes | None:
  """NUL-separated working-tree files per git, i.e. .gitignore applied."""
  res = subprocess.run(
      ['git', '-C', root, 'ls-files', '-z', '-co', '--exclude-standard'],
      capture_output=True, check=False)
  return res.stdout if res.returncode == 0 and res.stdout else None


def stage_code(exp: Experiment, root: str) -> tuple[str, str, int]:
  """Tars the working tree and uploads it; returns (url, sha256, bytes)."""
  fd, path = tempfile.mkstemp(suffix='.tar.gz', prefix='simply-code-')
  os.close(fd)
  cmd = ['tar', '--sort=name', '--numeric-owner']
  cmd += [f'--exclude={p}' for p in _TAR_EXCLUDES]
  cmd += ['-czf', path, '-C', root]
  # In a git checkout the ignore rules already describe "the working tree
  # minus junk" (a local .venv/ is 60x the size of the code); outside one,
  # fall back to the whole directory with the exclude list above.
  file_list = _tracked_files(root)
  if file_list:
    subprocess.run(cmd + ['--null', '-T', '-'], input=file_list, check=True)
  else:
    subprocess.run(cmd + ['.'], check=True)
  digest = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      digest.update(chunk)
  sha = digest.hexdigest()
  size = os.path.getsize(path)
  url = f'{exp.root}/_code/code-{sha[:16]}.tar.gz'
  if cloud.gcs_exists(url):
    cloud.log(f'code already staged: {url} ({size / 1e6:.1f} MB)')
  else:
    cloud.log(f'staging {size / 1e6:.1f} MB of working tree -> {url}')
    cloud.gcs_upload(path, url)
  os.unlink(path)
  return url, sha, size


# --------------------------------------------------------------------------
# Scheduler


@dataclasses.dataclass
class SeedState:
  seed: int
  status: str = 'pending'  # pending | running | done | failed
  attempts: int = 0
  preemptions: int = 0
  job_id: str | None = None
  vm: str | None = None
  started: str | None = None
  finished: str | None = None
  note: str = ''


def vm_prefix(name: str) -> str:
  """A TPU node name: lowercase letters, digits and hyphens, <= 63 chars."""
  slug = re.sub(r'[^a-z0-9-]+', '-', name.lower()).strip('-') or 'simply'
  if not slug[0].isalpha():
    slug = 'x' + slug
  return slug[:50]


class Sweep:
  """Runs one seed per TPU VM until every seed has a final_result.json."""

  def __init__(self, args, exp: Experiment, defaults, code_url: str,
               commands: dict[int, list[str]]):
    self.args = args
    self.exp = exp
    self.defaults = defaults
    self.code_url = code_url
    self.commands = commands
    self.lock = threading.Lock()
    self.states = {s: SeedState(s) for s in commands}
    self.queue = list(commands)
    # Which VMs this experiment owns has to survive a restart of the
    # launcher, or a resumed sweep would leak the VMs it created earlier.
    previous = cloud.gcs_read_json(exp.state_url) or {}
    self.vms: dict[str, dict] = {
        name: info for name, info in (previous.get('vms') or {}).items()
        if 'deleted' not in info}
    # The agent job queue is bucket-scoped, not experiment-scoped: a VM
    # outlives one experiment and may serve several.
    self.ctl_prefix = f'{args.bucket.rstrip("/")}/{args.ctl_prefix.strip("/")}'
    self.transport = (cloud.GcsTransport(self.ctl_prefix)
                      if args.transport == 'gcs' else cloud.SshTransport())

  # -- state -------------------------------------------------------------
  def publish(self) -> None:
    with self.lock:
      payload = {
          'updated': now(),
          'experiment': self.exp.name,
          'transport': self.transport.name,
          'seeds': {str(s): dataclasses.asdict(st)
                    for s, st in sorted(self.states.items())},
          'vms': self.vms,
      }
    try:
      cloud.gcs_write(self.exp.state_url, json.dumps(payload, indent=2) + '\n')
    except RuntimeError as e:
      cloud.log(f'warning: could not publish launch state: {e}')

  def take_seed(self) -> int | None:
    with self.lock:
      return self.queue.pop(0) if self.queue else None

  def requeue(self, seed: int) -> None:
    with self.lock:
      self.queue.append(seed)

  # -- VM ----------------------------------------------------------------
  def make_vm(self, index: int) -> cloud.TpuVm:
    # `--vms` adopts VMs the user already has; anything else is ours to
    # create, and ours to delete when the sweep ends.
    name = (self.args.vms[index] if index < len(self.args.vms)
            else f'{vm_prefix(self.exp.name)}-w{index}')
    metadata = {'ctl-node': name}
    if self.args.transport == 'gcs':
      without_scheme = self.ctl_prefix.replace('gs://', '')
      metadata['ctl-bucket'], _, prefix = without_scheme.partition('/')
      metadata['ctl-prefix'] = prefix
    return cloud.TpuVm(name=name, zone=self.args.zone[0],
                       tpu_type=self.args.tpu_type, project=self.args.project,
                       spot=self.args.spot, metadata=metadata)

  def ensure_vm(self, vm: cloud.TpuVm) -> bool:
    """Creates the VM if needed and waits until it can accept a job."""
    state = self.reap_dead(vm)
    if state is None:
      with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as f:
        f.write(remote.startup_script())
        script_path = f.name
      self.transport.reset(vm)
      try:
        zones = self.args.zone
        for attempt in range(1, self.args.create_retries + 1):
          # Capacity, not quota, is what fails: rotate through the zones the
          # user offered and, with --spot-fallback, through spot as well.
          vm.zone = zones[(attempt - 1) % len(zones)]
          vm.spot = self.args.spot or (self.args.spot_fallback
                                       and attempt > len(zones))
          cloud.log(f'{vm.name}: creating {vm.tpu_type} in {vm.zone}'
                    f'{" (spot)" if vm.spot else ""} [attempt {attempt}]')
          res = vm.create(script_path if self.args.transport == 'gcs' else None)
          if res.ok or vm.state() is not None:
            break
          cloud.log(f'{vm.name}: create failed in {vm.zone}: '
                    f'{(res.stderr or res.stdout).strip()[-300:]}')
          time.sleep(self.args.create_retry_sec)
        else:
          return False
      finally:
        os.unlink(script_path)
      with self.lock:
        self.vms[vm.name] = {'zone': vm.zone, 'type': vm.tpu_type,
                             'spot': vm.spot, 'created': now(),
                             'created_by_launcher': True}
    else:
      cloud.log(f'{vm.name}: reusing existing VM (state={state})')
      with self.lock:
        self.vms.setdefault(vm.name, {'zone': vm.zone, 'type': vm.tpu_type,
                                      'spot': vm.spot, 'created': now(),
                                      'created_by_launcher': False})
    if not self.transport.wait_ready(vm, self.args.ready_timeout_min * 60):
      cloud.log(f'{vm.name}: never became ready')
      return False
    cloud.log(f'{vm.name}: ready')
    return True

  def reap_dead(self, vm: cloud.TpuVm) -> str | None:
    """Clears a node that is dying or dead; returns the usable state, if any.

    A name is reusable only once the old node is really gone: `create` on a
    name still in DELETING fails, and a DELETING node still answers
    `describe`, so "it exists" is not "it can run a job".
    """
    state = vm.state()
    if state is None or state not in cloud.DEAD_STATES:
      return state
    cloud.log(f'{vm.name}: state={state}; deleting and recreating')
    if state != 'DELETING':
      vm.delete()
    deadline = time.time() + 15 * 60
    while time.time() < deadline:
      if vm.state() is None:
        return None
      time.sleep(20)
    cloud.log(f'{vm.name}: still {vm.state()} after 15 min')
    return vm.state()

  # -- one seed ----------------------------------------------------------
  def run_seed(self, vm: cloud.TpuVm, seed: int) -> str:
    """Submits one seed and waits for it. Returns the new seed status."""
    st = self.states[seed]
    st.attempts += 1
    st.job_id = f'seed{seed}-{int(time.time())}'
    st.vm, st.status, st.started = vm.name, 'running', now()
    self.publish()
    spec = remote.JobSpec(
        job_id=st.job_id,
        seed=seed,
        run_dir=self.exp.seed_dir(seed),
        status_url=self.exp.status_url(st.job_id),
        code_url=self.code_url,
        command=tuple(self.commands[seed]),
        assets_url=self.args.assets,
        assets_mode=self.args.assets_mode,
        assets_include=tuple(self.args.assets_include),
        verify_assets=self.args.verify_assets,
        apt_packages=tuple(self.args.apt),
        env=tuple(kv.split('=', 1) for kv in self.args.env),
        pip_extras=self.args.pip_extras,
    )
    cloud.log(f'seed {seed}: submitting {st.job_id} to {vm.name}')
    self.transport.submit(vm, st.job_id, remote.job_script(spec))

    deadline = time.time() + self.args.timeout_min * 60
    next_vm_check = 0.0
    while time.time() < deadline:
      raw = cloud.gcs_read(self.exp.status_url(st.job_id))
      if raw and raw.strip().startswith('exit='):
        rc = int(raw.strip().split('=', 1)[1])
        st.finished = now()
        if rc == 0 and cloud.gcs_exists(self.exp.result_url(seed)):
          st.note = ''
          return 'done'
        st.note = (f'exit={rc}' if rc else 'exit=0 but no final_result.json')
        return 'failed'
      if time.time() >= next_vm_check:
        next_vm_check = time.time() + 120
        state = vm.state()
        if state is None or state in cloud.DEAD_STATES:
          st.preemptions += 1
          st.note = f'VM {vm.name} went {state or "away"} mid-run'
          cloud.log(f'seed {seed}: {st.note}')
          return 'preempted'
      time.sleep(self.args.poll_sec)
    st.note = f'timed out after {self.args.timeout_min} min'
    return 'failed'

  # -- worker ------------------------------------------------------------
  def owns(self, vm: cloud.TpuVm) -> bool:
    return self.vms.get(vm.name, {}).get('created_by_launcher', False)

  def worker(self, index: int) -> None:
    vm = self.make_vm(index)
    try:
      while True:
        # A worker claims a seed only once its VM can run it: a worker stuck
        # hunting for capacity must not sit on a seed another worker's idle
        # VM could already be training.
        with self.lock:
          if not self.queue:
            return
        if self.reap_dead(vm) is None:
          if not self.ensure_vm(vm):
            cloud.log(f'{vm.name}: giving up, no {vm.tpu_type} capacity')
            return
        elif not self.transport.wait_ready(vm, self.args.ready_timeout_min * 60):
          cloud.log(f'{vm.name}: not ready; retrying')
          time.sleep(30)
          continue
        seed = self.take_seed()
        if seed is None:
          return
        result = self.run_seed(vm, seed)
        st = self.states[seed]
        if result == 'done':
          st.status = 'done'
          cloud.log(f'seed {seed}: DONE -> {self.exp.result_url(seed)}')
        elif result == 'preempted':
          cloud.log(f'seed {seed}: preempted, recreating {vm.name}')
          vm.delete()
          st.status = 'pending'
          self.requeue_or_fail(seed)
          if st.status == 'failed':
            write_seed_status(self.exp, st, 'FAILED')
        else:
          cloud.log(f'seed {seed}: FAILED ({st.note}); log: {self.exp.log_url(seed)}')
          st.status = 'failed'
          self.requeue_or_fail(seed)
          if st.status == 'failed':
            write_seed_status(self.exp, st, 'FAILED')
        self.publish()
    finally:
      if self.owns(vm) and not self.args.keep_vms:
        cloud.log(f'{vm.name}: deleting')
        vm.delete()
        with self.lock:
          self.vms.get(vm.name, {})['deleted'] = now()
        self.publish()

  def requeue_or_fail(self, seed: int) -> None:
    st = self.states[seed]
    budget = self.args.max_preemptions if st.preemptions else self.args.max_retries
    if st.attempts <= budget:
      st.status = 'pending'
      self.requeue(seed)
    else:
      st.status = 'failed'

  # -- entry -------------------------------------------------------------
  def run(self) -> int:
    n = max(1, min(self.args.num_workers, len(self.commands)))
    cloud.log(f'{len(self.commands)} seed(s) over {n} worker VM(s), '
              f'transport={self.transport.name}')
    self.publish()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
      futures = [pool.submit(self.worker, i) for i in range(n)]
      for f in futures:
        f.result()
    self.publish()
    return print_summary(self.exp, self.states)


def write_seed_status(exp: Experiment, st: SeedState, state: str) -> None:
  """A VM that vanishes mid-run never writes its own status.json."""
  payload = {'state': state, 'exit_code': None, 'seed': st.seed,
             'job_id': st.job_id, 'started_at': st.started,
             'finished_at': now(), 'note': st.note,
             'written_by': 'launch_gcp'}
  try:
    cloud.gcs_write(f'{exp.seed_dir(st.seed)}/status.json',
                    json.dumps(payload, indent=2) + '\n')
  except RuntimeError as e:
    cloud.log(f'warning: could not write status.json for seed {st.seed}: {e}')


def print_summary(exp: Experiment, states: dict[int, SeedState]) -> int:
  print()
  print(f'experiment: {exp.root}')
  for seed, st in sorted(states.items()):
    mark = {'done': 'OK  ', 'failed': 'FAIL',
            'pending': 'SKIP'}.get(st.status, '??  ')
    extra = f'  {st.note}' if st.note else ''
    print(f'  {mark} seed_{seed}  attempts={st.attempts} '
          f'preemptions={st.preemptions} vm={st.vm}{extra}')
  failed = [s for s, st in states.items() if st.status != 'done']
  if failed:
    print(f'\n{len(failed)} seed(s) did not finish. Re-run the same command to '
          'retry only those; inspect a log with:\n'
          f'  python -m tasks.research_bench.launch.launch_gcp logs '
          f'--bucket={exp.bucket} --experiment_name={exp.name} --seed={failed[0]}')
  return 1 if failed else 0


# --------------------------------------------------------------------------
# Subcommands


def cmd_run(args) -> int:
  defaults = resolve_defaults(args)
  exp = Experiment(args.bucket, args.experiment_name)
  seeds = ([int(s) for s in args.seeds.split(',') if s.strip()]
           or list(defaults.seeds))
  commands = {s: build_command(args, defaults, exp.seed_dir(s), s) for s in seeds}

  git = git_info(REPO_ROOT)
  manifest = {
      'schema': 'simply.research_bench.launch_manifest/1',
      # Fields the validator reads (validator/submission.py).
      'task': args.task,
      'experiment_config': args.experiment_config or defaults.experiment_config
                           or args.task,
      'experiment_dir': exp.root,
      'seeds': seeds,
      'git_commit': git['sha'],
      'git_dirty': git['dirty'],
      'launched_at': now(),
      'tpu_type': args.tpu_type,
      # Launcher provenance.
      'experiment_name': exp.name,
      'entry_module': args.entry_module,
      'bucket': args.bucket,
      'git_branch': git['branch'],
      'git_describe': git['describe'],
      'tpu_zone': args.zone[0],
      'tpu_zones': args.zone,
      'tpu_spot': args.spot,
      'tpu_runtime_version': cloud.runtime_version(args.tpu_type),
      'num_workers': args.num_workers,
      'assets': {'url': args.assets, 'mode': args.assets_mode,
                 'include': args.assets_include,
                 'verified': args.verify_assets},
      'vm_setup': {'pip_extras': args.pip_extras, 'apt': args.apt},
      'per_seed_command': {str(s): ' '.join(shlex.quote(c) for c in cmd)
                           for s, cmd in commands.items()},
      'command_line': ' '.join(shlex.quote(a) for a in sys.argv),
      'launched_by': f'{getpass.getuser()}@{socket.gethostname()}',
      'transport': args.transport,
  }
  port_dir = args.port_experiment_dir or (
      f'{exp.root}/port_run' if (args.task or '').startswith('port_') else '')
  if port_dir:
    manifest['port_experiment_dir'] = port_dir

  if args.dry_run:
    print(json.dumps(manifest, indent=2))
    print('\n# per-seed remote command lines')
    for s in seeds:
      print(f'# seed {s}  (experiment_dir={exp.seed_dir(s)})')
      print('  ' + manifest['per_seed_command'][str(s)])
    print('\n# dry run: nothing was created and nothing was written to GCS')
    return 0

  cloud.enable_adc_token(args.adc_token)
  args.project = args.project or default_project()
  manifest['project'] = args.project

  pending = [s for s in seeds
             if args.force or not cloud.gcs_exists(exp.result_url(s))]
  for s in seeds:
    if s not in pending:
      cloud.log(f'seed {s}: already complete ({exp.result_url(s)}), skipping')
  if not pending:
    cloud.log('nothing to do; all seeds already have a final_result.json')
    return 0

  code_url, code_sha, code_size = stage_code(exp, REPO_ROOT)
  manifest['code'] = {'archive': code_url, 'sha256': code_sha,
                      'bytes': code_size}
  merge_manifest(exp, manifest)
  sweep = Sweep(args, exp, defaults, code_url,
                {s: commands[s] for s in pending})
  try:
    return sweep.run()
  except KeyboardInterrupt:
    cloud.log('interrupted; VMs are left running. Tear them down with:\n'
              f'  python -m tasks.research_bench.launch.launch_gcp teardown '
              f'--bucket={exp.bucket} --experiment_name={exp.name}')
    return 130


def merge_manifest(exp: Experiment, manifest: dict) -> None:
  """Writes the manifest, keeping earlier launches of the same experiment."""
  previous = cloud.gcs_read_json(exp.manifest_url) or {}
  history = list(previous.get('launches', []))
  if previous:
    history.append({k: previous[k] for k in
                    ('launched_at', 'relaunched_at', 'command_line',
                     'git_commit', 'git_dirty', 'code', 'tpu_type')
                    if k in previous})
    # The validator rejects a final_result.json older than `launched_at`
    # (anti-replay). A restart must not invalidate the seeds that already
    # finished, so the first launch's timestamp is the one that sticks.
    manifest['relaunched_at'] = manifest['launched_at']
    manifest['launched_at'] = previous.get('launched_at',
                                           manifest['launched_at'])
  manifest['launches'] = history
  cloud.gcs_write(exp.manifest_url, json.dumps(manifest, indent=2) + '\n')
  cloud.log(f'manifest -> {exp.manifest_url}')


def default_project() -> str:
  res = cloud.gcloud('config', 'get-value', 'project', timeout=60)
  project = res.stdout.strip()
  if not project or project == '(unset)':
    cloud.die('no GCP project: pass --project or run `gcloud config set project`')
  return project


def cmd_status(args) -> int:
  cloud.enable_adc_token(args.adc_token)
  exp = Experiment(args.bucket, args.experiment_name)
  manifest = cloud.gcs_read_json(exp.manifest_url)
  if manifest is None:
    cloud.die(f'no manifest at {exp.manifest_url}')
  state = cloud.gcs_read_json(exp.state_url) or {}
  print(f'experiment : {exp.root}')
  print(f'task       : {manifest.get("task")} '
        f'(config={manifest.get("experiment_config")})')
  print(f'entry      : {manifest.get("entry_module")}')
  print(f'tpu        : {manifest.get("tpu_type")} in '
        f'{manifest.get("tpu_zone")}'
        f'{" (spot)" if manifest.get("tpu_spot") else ""}')
  print(f'git        : {(manifest.get("git_commit") or "?")[:12]} '
        f'{"(dirty)" if manifest.get("git_dirty") else "(clean)"} '
        f'{manifest.get("git_branch", "")}')
  print(f'launched   : {manifest.get("launched_at")} by '
        f'{manifest.get("launched_by")}')
  print('seeds      :')
  for seed in manifest.get('seeds', []):
    st = (state.get('seeds', {}) or {}).get(str(seed), {})
    done = cloud.gcs_exists(exp.result_url(seed))
    print(f'  seed_{seed}: {"complete" if done else st.get("status", "unknown")}'
          f'  attempts={st.get("attempts", 0)}'
          f' preemptions={st.get("preemptions", 0)}'
          f' vm={st.get("vm")}' + (f'  {st["note"]}' if st.get('note') else ''))
  vms = {n: v for n, v in (state.get('vms') or {}).items() if 'deleted' not in v}
  if vms:
    print(f'live VMs   : {", ".join(vms)}')
  return 0


def cmd_collect(args) -> int:
  cloud.enable_adc_token(args.adc_token)
  exp = Experiment(args.bucket, args.experiment_name)
  manifest = cloud.gcs_read_json(exp.manifest_url) or {}
  seeds = ([int(s) for s in args.seeds.split(',') if s.strip()] if args.seeds
           else manifest.get('seeds', []))
  results = {}
  for seed in seeds:
    results[seed] = cloud.gcs_read_json(exp.result_url(seed))
  if args.json:
    print(json.dumps({'experiment': exp.root, 'results': results}, indent=2))
    return 0 if all(results.values()) else 1
  keys = sorted({k for r in results.values() if r for k in r
                 if isinstance(r[k], (int, float, str, bool))})
  print(f'{exp.root}')
  for seed, res in results.items():
    if res is None:
      print(f'  seed_{seed}: MISSING ({exp.result_url(seed)})')
      continue
    fields = ' '.join(f'{k}={res[k]!r}' for k in keys if k in res)
    print(f'  seed_{seed}: {fields}')
  numeric = [k for k in keys
             if all(isinstance((r or {}).get(k), (int, float))
                    and not isinstance(r[k], bool) for r in results.values())]
  for k in numeric:
    vals = [results[s][k] for s in results]
    mean = sum(vals) / len(vals)
    print(f'  mean {k}: {mean:.6g}  (n={len(vals)}, min={min(vals):.6g}, '
          f'max={max(vals):.6g})')
  return 0 if all(results.values()) else 1


def cmd_logs(args) -> int:
  cloud.enable_adc_token(args.adc_token)
  exp = Experiment(args.bucket, args.experiment_name)
  url = exp.log_url(args.seed)
  shown = 0
  while True:
    text = cloud.gcs_read(url) or ''
    if len(text) > shown:
      sys.stdout.write(text[shown:])
      sys.stdout.flush()
      shown = len(text)
    if not args.follow:
      return 0 if text else 1
    if cloud.gcs_exists(exp.result_url(args.seed)):
      return 0
    time.sleep(15)


def cmd_teardown(args) -> int:
  cloud.enable_adc_token(args.adc_token)
  exp = Experiment(args.bucket, args.experiment_name)
  args.project = args.project or default_project()
  state = cloud.gcs_read_json(exp.state_url) or {}
  vms = {n: v for n, v in (state.get('vms') or {}).items() if 'deleted' not in v}
  if not vms:
    print('no VMs recorded for this experiment')
    return 0
  for name, info in vms.items():
    vm = cloud.TpuVm(name, info['zone'], info.get('type', ''), args.project)
    cloud.log(f'deleting {name} in {info["zone"]}')
    res = vm.delete()
    info['deleted'] = now()
    if not res.ok:
      cloud.log(f'  delete failed: {res.stderr.strip()[-200:]}')
  cloud.gcs_write(exp.state_url, json.dumps(state, indent=2) + '\n')
  return 0


def cmd_probe_ssh(args) -> int:
  cloud.enable_adc_token(args.adc_token)
  args.project = args.project or default_project()
  print(cloud.probe_ssh(args.project, args.zone))
  return 0


# --------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog='launch_gcp',
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest='command')

  def common(p, need_experiment=True):
    p.add_argument('--bucket', required=True,
                   help='gs://<bucket>[/<prefix>] holding experiment dirs')
    if need_experiment:
      p.add_argument('--experiment_name', '--experiment-name', required=True,
                     dest='experiment_name',
                     help='GCS subdirectory holding this submission')
    p.add_argument('--project', default=os.environ.get('CLOUDSDK_CORE_PROJECT'))
    p.add_argument('--adc-token', '--adc_token', dest='adc_token',
                   action='store_true',
                   help='mint gcloud auth from application-default credentials '
                        '(needed where policy blocks gcloud user creds)')

  run = sub.add_parser('run', help='launch the multi-seed sweep (default)')
  common(run)
  run.add_argument('--task', help=f'one of {sorted(task_defaults.TASKS)}')
  run.add_argument('--experiment_config', '--experiment-config',
                   dest='experiment_config',
                   help='registered config name (default: --task)')
  run.add_argument('--seeds', default='',
                   help='comma-separated seeds (default: the task\'s required '
                        'sweep, usually 42,43,44)')
  run.add_argument('--entry-module', '--entry_module', dest='entry_module',
                   help='module run as `python -m <module>` on the VM')
  run.add_argument('--extra-flag', '--extra_flag', dest='extra_flag',
                   action='append', default=[], metavar='k=v',
                   help='extra flag for the remote binary; repeatable')
  run.add_argument('--config-overlay', '--config_overlay',
                   dest='config_overlay', default='{}',
                   help='JSON merged into --config_overlay (seeds are added)')
  run.add_argument('--env', action='append', default=[], metavar='K=V',
                   help='environment variable for the remote run; repeatable')
  run.add_argument('--tpu-type', '--tpu_type', dest='tpu_type',
                   help='accelerator type, e.g. v6e-4 (default: per task)')
  run.add_argument('--zone', default='us-central1-b',
                   type=lambda v: [z for z in v.split(',') if z],
                   help='zone, or a comma-separated list tried in turn when '
                        'a zone is out of capacity')
  run.add_argument('--spot-fallback', '--spot_fallback', dest='spot_fallback',
                   action='store_true',
                   help='after one pass over --zone on demand, retry on spot')
  run.add_argument('--spot', action='store_true',
                   help='request spot capacity (cheaper, preemptible)')
  run.add_argument('--num-workers', '--num_workers', dest='num_workers',
                   type=int, default=1,
                   help='TPU VMs to use; seeds are spread over them '
                        '(1 = all seeds sequentially on one VM)')
  run.add_argument('--keep-vms', '--keep_vms', dest='keep_vms',
                   action='store_true', help='do not delete VMs afterwards')
  run.add_argument('--port-experiment-dir', '--port_experiment_dir',
                   dest='port_experiment_dir', default='',
                   help='porting tasks: where the reference port run lives '
                        '(default <experiment_dir>/port_run)')
  run.add_argument('--vms', default='', type=lambda v: [x for x in v.split(',') if x],
                   help='comma-separated names of existing TPU VMs to run on '
                        'instead of creating new ones (never deleted)')
  run.add_argument('--ctl-prefix', '--ctl_prefix', dest='ctl_prefix',
                   default='_simply_ctl',
                   help='--transport=gcs: object prefix under --bucket '
                        'holding the per-VM job queues')
  run.add_argument('--transport', choices=('ssh', 'gcs'), default='ssh',
                   help='how jobs reach the VM: ssh (default) or a GCS job '
                        'queue polled by a startup-script agent (no SSH)')
  run.add_argument('--assets', default='',
                   help='gs://.../assets with models/ datasets/ vocabs/ '
                        '(sets SIMPLY_MODELS/SIMPLY_DATASETS/SIMPLY_VOCABS)')
  run.add_argument('--assets-mode', '--assets_mode', dest='assets_mode',
                   choices=('mirror', 'direct'), default='mirror',
                   help='mirror the asset cache to local disk, or point the '
                        'env vars straight at gs://')
  run.add_argument('--apt', default='',
                   type=lambda v: [x for x in v.split(',') if x],
                   help='extra apt packages for the VM (default: per task)')
  run.add_argument('--assets-include', '--assets_include',
                   dest='assets_include', default='',
                   type=lambda v: [x for x in v.split(',') if x],
                   help='mirror only these subpaths of the asset cache, e.g. '
                        'datasets/c4_bin,vocabs (default: all of it)')
  run.add_argument('--verify-assets', '--verify_assets',
                   dest='verify_assets', action='store_true',
                   help='crc32c every mirrored asset against the staged '
                        'manifest before running (~1 min per 80 GB); catches '
                        'a download that is the right size and the wrong bytes')
  run.add_argument('--pip-extras', '--pip_extras', dest='pip_extras',
                   default='',
                   help='extras installed on the VM: pip install -e .[<this>]')
  run.add_argument('--dry-run', '--dry_run', dest='dry_run',
                   action='store_true',
                   help='print the manifest and per-seed command lines only')
  run.add_argument('--force', action='store_true',
                   help='re-run seeds that already have a final_result.json')
  run.add_argument('--timeout-min', dest='timeout_min', type=float,
                   default=None,
                   help='give up on a seed after this long (default: 4x the '
                        'task\'s reference wall-clock)')
  run.add_argument('--ready-timeout-min', dest='ready_timeout_min', type=float,
                   default=25)
  run.add_argument('--poll-sec', dest='poll_sec', type=float, default=20)
  run.add_argument('--create-retries', dest='create_retries', type=int,
                   default=20, help='capacity is the usual reason a create '
                                    'fails; retry this many times')
  run.add_argument('--create-retry-sec', dest='create_retry_sec', type=float,
                   default=60)
  run.add_argument('--max-retries', dest='max_retries', type=int, default=1,
                   help='re-runs of a seed that failed on its own')
  run.add_argument('--max-preemptions', dest='max_preemptions', type=int,
                   default=5, help='re-runs of a seed whose VM was preempted')
  run.set_defaults(func=cmd_run)

  status = sub.add_parser('status', help='per-seed state of an experiment')
  common(status)
  status.set_defaults(func=cmd_status)

  collect = sub.add_parser('collect', help='per-seed final_result.json metrics')
  common(collect)
  collect.add_argument('--seeds', default='')
  collect.add_argument('--json', action='store_true')
  collect.set_defaults(func=cmd_collect)

  logs = sub.add_parser('logs', help='stream one seed\'s log.txt')
  common(logs)
  logs.add_argument('--seed', type=int, required=True)
  logs.add_argument('--follow', '-f', action='store_true')
  logs.set_defaults(func=cmd_logs)

  teardown = sub.add_parser('teardown', help='delete the VMs of an experiment')
  common(teardown)
  teardown.set_defaults(func=cmd_teardown)

  probe = sub.add_parser('probe-ssh', help='can this machine ssh to a TPU VM?')
  probe.add_argument('--project', default=os.environ.get('CLOUDSDK_CORE_PROJECT'))
  probe.add_argument('--zone', default='us-central1-b')
  probe.add_argument('--adc-token', '--adc_token', dest='adc_token',
                     action='store_true')
  probe.set_defaults(func=cmd_probe_ssh)
  return parser


def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  known = {'run', 'status', 'collect', 'logs', 'teardown', 'probe-ssh'}
  if not argv or argv[0] not in known:
    argv.insert(0, 'run')  # `run` is the default subcommand
  args = build_parser().parse_args(argv)
  return args.func(args)


if __name__ == '__main__':
  raise SystemExit(main())

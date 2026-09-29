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

"""Fetches a submission's artifacts, so callers do not have to.

A submission is an EXPERIMENT DIRECTORY -- a `gs://` bucket path or a local
directory, read through `etils.epath` either way -- with this layout, which is
what `tasks/research_bench/launch/` produces:

    <experiment_dir>/launch_manifest.json      what was launched, and from where
    <experiment_dir>/seed_<seed>/final_result.json        the scored artifact
    <experiment_dir>/seed_<seed>/experiment_config.json   optional
    <experiment_dir>/seed_<seed>/status.json              optional
    <experiment_dir>/seed_<seed>/log.txt                  optional (see below)
    <experiment_dir>/port_run/seed_<seed>/final_result.json   porting tasks

`launch_manifest.json` (all fields optional; a missing manifest downgrades the
provenance checks to a warning rather than failing the submission):

    {"task": "pretrain_bpb_byte",        # must match the validated task
     "experiment_config": "pretrain_bpb_byte",
     "experiment_dir": "gs://bucket/exp",  # self-identity of the manifest
     "seeds": [42, 43, 44],
     "git_commit": "<40-hex sha>",         # ownership: see `claimed_commit`
     "git_dirty": false,
     "launched_at": "2026-09-29T08:00:00Z",  # ISO-8601 (or epoch seconds)
     "tpu_type": "v6e-4",
     "per_seed_command": {"42": "python -m ... --lm_format=QwenV2Chat ..."},
     "port_experiment_dir": "gs://bucket/exp_port"}   # porting tasks

For the two eval-only tasks the fixed protocol is a set of command-line flags
rather than a registered config, so `TaskSpec.launch_fields` is matched against
the manifest's own fields AND the flags of the `per_seed_command` it records.

`status.json`, when the launcher writes one, is `{"state": "COMPLETED", ...}`;
any other state fails the run. `final_result.json` is the authoritative
completion marker -- the training loop writes it last.

`log.txt` is read only for the tasks that load a checkpoint, and only for one
line: `simply.model_lib` re-initialises the parameter branches a checkpoint
does not cover and carries on at INFO, so a half-mapped checkpoint yields a
plausible bad metric rather than an error. See `TaskSpec.checkpoint_policy`;
an absent log is a warning, never a failure.

Library entry points:

    verdict = score_submission('rl_qwen2p5_math_1p5b', 'gs://bucket/exp')

    # or, if you already have the artifacts (offline scoring, replay, tests):
    verdict = validator.validate(validator.Submission(...))

The I/O seams are injectable -- `read_text_fn`, `list_seeds_fn`, `mtime_fn` --
so a caller with its own storage client, or a test, can substitute without
patching globals.
"""

import datetime
import json
import re
import shlex
from typing import Any, Callable, Mapping, Sequence

from etils import epath

from tasks.research_bench.validator import task_specs
from tasks.research_bench.validator import validator


# Contents of an artifact path, or None when it does not exist.
ReadTextFn = Callable[[str], 'str | None']
# The seeds that have a `seed_<seed>/` directory under an experiment dir.
ListSeedsFn = Callable[[str], Sequence[int]]
# Last-modified time of a path in epoch seconds, or None when unknown.
MtimeFn = Callable[[str], 'float | None']

MANIFEST = 'launch_manifest.json'
RESULT = 'final_result.json'
CONFIG = 'experiment_config.json'
STATUS = 'status.json'
LOG = 'log.txt'
PORT_RUN_DIR = 'port_run'
COMPLETED = 'COMPLETED'

# A result written before the launch was not produced by it. The slack absorbs
# clock skew between the launcher and the storage system.
CLOCK_SKEW_S: float = 300.0

_SEED_DIR = re.compile(r'^seed_(\d+)$')

# `simply.model_lib` re-initialises the branches a checkpoint does not cover
# and keeps going, logging this at INFO -- the run then trains a partly random
# model and reports a plausible bad metric (see TaskSpec.checkpoint_policy).
_CKPT_MISMATCH = re.compile(
    r'Checkpoint\s+structural\s+mismatch\s+detected\s+\((\d+)\s+valid\s+arrays'
    r'\s+vs\s+(\d+)\s+leaves\)')


def read_text(path: str) -> str | None:
  """Reads an artifact, None when it is absent or unreadable."""
  try:
    file = epath.Path(path)
    return file.read_text() if file.exists() else None
  except (OSError, ValueError):
    return None


def list_seeds(experiment_dir: str) -> list[int]:
  """Seeds with a `seed_<seed>/` directory, empty when the dir is unreadable."""
  try:
    children = list(epath.Path(experiment_dir).glob('seed_*'))
  except (OSError, ValueError):
    return []
  found = [_SEED_DIR.match(child.name) for child in children]
  return sorted(int(m.group(1)) for m in found if m)


def mtime(path: str) -> float | None:
  """Last-modified time in epoch seconds, None when unknown."""
  try:
    return float(epath.Path(path).stat().mtime)
  except (OSError, ValueError, AttributeError, TypeError):
    return None


def _read_json(path: str, read_text_fn: ReadTextFn) -> dict[str, Any] | None:
  """Reads one JSON artifact, None when absent or unparseable."""
  raw = read_text_fn(path)
  if not raw or not raw.strip():
    return None
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError:
    return None
  return parsed if isinstance(parsed, dict) else None


def _epoch(value: Any) -> float | None:
  """Epoch seconds from an ISO-8601 string (naive == UTC) or a number."""
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return float(value)
  if not isinstance(value, str) or not value.strip():
    return None
  try:
    stamp = datetime.datetime.fromisoformat(
        value.strip().replace('Z', '+00:00'))
  except ValueError:
    return None
  if stamp.tzinfo is None:
    stamp = stamp.replace(tzinfo=datetime.timezone.utc)
  return stamp.timestamp()


def _observed_seed(result: Mapping[str, Any] | None,
                   config: Mapping[str, Any] | None) -> int | None:
  """The seed a run actually ran, from its own artifacts.

  Read rather than assumed from the directory name, so that a sweep which
  repeated one seed, or used the wrong ones, is visible.

  Args:
    result: the run's final_result.json, if any. Checked for a top-level
      `seed` / `model_seed` and for `run_provenance.model_seed`.
    config: the run's experiment_config.json, if any.

  Returns:
    The seed, or None when neither artifact records one.
  """
  provenance = (result or {}).get('run_provenance') or None
  for source, key in ((result, 'seed'), (result, 'model_seed'),
                      (provenance, 'model_seed'), (config, 'model_seed')):
    if not source:
      continue
    value = source.get(key)
    if value is None and key in ('model_seed',):
      value = (source.get('config') or {}).get(key)
    if value is not None:
      try:
        return int(value)
      except (TypeError, ValueError):
        return None
  return None


def _checkpoint_load(run_dir: str, label: str, policy: str,
                     read_text_fn: ReadTextFn) -> tuple[list[str], list[str]]:
  """Did this run's checkpoint map onto the model, per `log.txt`.

  Args:
    run_dir: The run's `seed_<seed>/` directory.
    label: How to name the run in a message.
    policy: `TaskSpec.checkpoint_policy()`; '' skips the check entirely.
    read_text_fn: Reads an artifact path.

  Returns:
    (errors, warnings). A missing log is never fatal: an agent may have
    results without the launcher's log capture.
  """
  if not policy:
    return [], []
  log = read_text_fn(f'{run_dir}/{LOG}')
  if not log:
    return [], [f'{label}: no {LOG}, so the checkpoint load cannot be checked '
                '(a partly mapped checkpoint is silently re-initialised)']
  found = _CKPT_MISMATCH.search(log)
  if not found:
    return [], []
  loaded, total = int(found.group(1)), int(found.group(2))
  detail = (f'{total - loaded} of {total} parameter branches were '
            f're-initialised from scratch ({LOG}: "Checkpoint structural '
            'mismatch detected")')
  if policy == 'fail':
    return [f'{label}: the pinned base checkpoint did not load completely -- '
            f'{detail}; the run did not start from the model the task '
            'fixes'], []
  return [], [f'{label}: your checkpoint format mapped only part of the model '
              f'-- {detail}; the score stands, but the port is partial and '
              'the metric is measuring a partly random model']


def read_runs(experiment_dir: str, spec_seeds: Sequence[int],
              launched_at: float | None = None,
              read_text_fn: ReadTextFn = read_text,
              list_seeds_fn: ListSeedsFn = list_seeds,
              mtime_fn: MtimeFn = mtime,
              ckpt_policy: str = '',
              ) -> tuple[list[validator.WorkUnit], list[str], list[str]]:
  """Reads every `seed_<seed>/` run of one experiment dir.

  Args:
    experiment_dir: The experiment directory (gs:// or local).
    spec_seeds: The seeds the task fixes; read even when no directory exists,
      so a missing one shows up as an incomplete sweep.
    launched_at: Epoch seconds the sweep was launched, None to skip the
      timestamp check.
    read_text_fn: Reads an artifact path.
    list_seeds_fn: Lists the seed directories that exist.
    mtime_fn: Last-modified time of an artifact path.
    ckpt_policy: `TaskSpec.checkpoint_policy()`, '' to skip the log check.

  Returns:
    (runs, errors, warnings). Directories present but not in `spec_seeds` are
    included, so a sweep over the wrong seeds is reported as such rather than
    as a missing one.
  """
  errors: list[str] = []
  warnings: list[str] = []
  runs: list[validator.WorkUnit] = []
  seeds = sorted(set(list_seeds_fn(experiment_dir)) | set(spec_seeds))
  for index, seed in enumerate(seeds, start=1):
    run_dir = f'{experiment_dir}/seed_{seed}'
    result = _read_json(f'{run_dir}/{RESULT}', read_text_fn)
    if result is None:
      runs.append(validator.WorkUnit(index=index, final_result=None))
      continue
    config = _read_json(f'{run_dir}/{CONFIG}', read_text_fn)
    recorded = _observed_seed(result, config)
    if recorded is None:
      warnings.append(f'seed_{seed}: the artifacts do not record a seed, so '
                      'the directory name is taken as the seed')
    elif recorded != seed:
      errors.append(f'seed_{seed}: the artifacts record seed {recorded}, not '
                    f'{seed} -- the run in this directory is not the one the '
                    'directory claims')
    status = _read_json(f'{run_dir}/{STATUS}', read_text_fn)
    if status is not None and status.get('state') != COMPLETED:
      errors.append(f'seed_{seed}: status.json reports state='
                    f'{status.get("state")!r}, not {COMPLETED!r}')
    ckpt_errors, ckpt_warnings = _checkpoint_load(
        run_dir, f'seed_{seed}', ckpt_policy, read_text_fn)
    errors += ckpt_errors
    warnings += ckpt_warnings
    if launched_at is not None:
      written = mtime_fn(f'{run_dir}/{RESULT}')
      if written is not None and written < launched_at - CLOCK_SKEW_S:
        errors.append(
            f'seed_{seed}: {RESULT} was written before the launch recorded in '
            f'{MANIFEST} ({_iso(written)} < {_iso(launched_at)}): it was not '
            'produced by this launch')
    runs.append(validator.WorkUnit(
        index=index, final_result=result,
        seed=recorded if recorded is not None else seed))
  return runs, errors, warnings


def _iso(epoch: float) -> str:
  return datetime.datetime.fromtimestamp(
      epoch, datetime.timezone.utc).isoformat()


def _command_flags(command: str) -> dict[str, str]:
  """`--flag=value` / `--flag value` pairs of one recorded command line."""
  try:
    tokens = shlex.split(command)
  except ValueError:
    return {}
  flags: dict[str, str] = {}
  for index, token in enumerate(tokens):
    if not token.startswith('--'):
      continue
    name, sep, value = token[2:].partition('=')
    if not sep:
      following = tokens[index + 1] if index + 1 < len(tokens) else ''
      value = following if following and not following.startswith('-') else (
          'false' if name.startswith('no') else 'true')
      name = name.removeprefix('no') if value == 'false' else name
    flags[name] = value
  return flags


def _launched_values(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
  """What the launch recorded, as {key: every value seen for it}.

  Merges the manifest's own scalar fields with the flags of every per-seed
  command it recorded, so that a key the two disagree on -- or that one seed
  was launched differently on -- is visible rather than silently resolved.

  Args:
    manifest: the parsed launch_manifest.json.

  Returns:
    Mapping of key to the set of string values recorded for it.
  """
  values: dict[str, set[str]] = {}
  for key, value in manifest.items():
    if isinstance(value, (str, int, float, bool)):
      values.setdefault(key, set()).add(str(value))
  commands = manifest.get('per_seed_command') or {}
  if isinstance(commands, str):
    commands = {'': commands}
  for command in (commands.values() if isinstance(commands, dict) else []):
    for key, value in _command_flags(str(command)).items():
      values.setdefault(key, set()).add(value)
  return values


def _check_launch_fields(manifest: Mapping[str, Any],
                         spec: task_specs.TaskSpec,
                         ) -> tuple[list[str], list[str]]:
  """The launch must be the task's fixed one, where the task pins it."""
  if not spec.launch_fields:
    return [], []
  recorded = _launched_values(manifest)
  errors, unverifiable = [], []
  for key, want in spec.launch_fields.items():
    seen = recorded.get(key)
    if not seen:
      unverifiable.append(key)
    elif seen != {str(want)}:
      errors.append(f'{MANIFEST}: {key}={sorted(seen)} (expected '
                    f'{str(want)!r})')
  warnings = []
  if unverifiable:
    warnings.append(f'{MANIFEST} records nothing for {sorted(unverifiable)}, '
                    "so the task's fixed eval setup cannot be verified")
  return errors, warnings


def _check_manifest(manifest: Mapping[str, Any] | None, task: str,
                    spec: task_specs.TaskSpec, experiment_dir: str,
                    ) -> tuple[list[str], list[str]]:
  """Checks what the launch manifest says against the task being scored."""
  if manifest is None:
    return [], [f'no {MANIFEST} in {experiment_dir}: the sweep\'s provenance '
                '(task, seeds, commit, launch time) cannot be checked']
  errors: list[str] = []
  warnings: list[str] = []
  declared_task = str(manifest.get('task') or '')
  if declared_task and declared_task != task:
    errors.append(f'{MANIFEST} was launched for task {declared_task!r}, not '
                  f'{task!r}')
  seeds = manifest.get('seeds')
  if seeds is not None:
    try:
      declared = sorted({int(s) for s in seeds})
    except (TypeError, ValueError):
      declared = []
    if declared != sorted(spec.seeds):
      errors.append(f'{MANIFEST} declares seeds {declared}, the task fixes '
                    f'{sorted(spec.seeds)}')
  recorded_dir = str(manifest.get('experiment_dir') or '').rstrip('/')
  if recorded_dir and recorded_dir != experiment_dir.rstrip('/'):
    warnings.append(f'{MANIFEST} records experiment_dir {recorded_dir!r} but '
                    f'was read from {experiment_dir!r}: the results were '
                    'moved, or the manifest was copied from another run')
  if not str(manifest.get('git_commit') or ''):
    warnings.append(f'{MANIFEST} records no git_commit: the code that '
                    'produced this run cannot be identified')
  if manifest.get('git_dirty'):
    warnings.append(f'{MANIFEST} records git_dirty=True: the run used '
                    'uncommitted changes and cannot be reproduced from the '
                    'recorded commit alone')
  if _epoch(manifest.get('launched_at')) is None:
    warnings.append(f'{MANIFEST} records no usable launched_at: the results '
                    'cannot be checked against the launch time')
  launch_errors, launch_warnings = _check_launch_fields(manifest, spec)
  return errors + launch_errors, warnings + launch_warnings


def load_submission(task: str, experiment_dir: str,
                    port_experiment_dir: str = '',
                    claimed_commit: str = '',
                    read_text_fn: ReadTextFn = read_text,
                    list_seeds_fn: ListSeedsFn = list_seeds,
                    mtime_fn: MtimeFn = mtime,
                    ) -> validator.Submission | None:
  """Builds a Submission by reading an experiment dir's artifacts.

  Args:
    task: Task id (a key of task_specs.TASKS).
    experiment_dir: The submitted experiment directory (gs:// or local).
    port_experiment_dir: Stage-1 port experiment dir, for the porting tasks.
      Defaults to the manifest's `port_experiment_dir`, then to
      `<experiment_dir>/port_run`.
    claimed_commit: The commit the submitting agent claims it ran, '' to skip
      the ownership check.
    read_text_fn: Reads an artifact path.
    list_seeds_fn: Lists the seed directories of an experiment dir.
    mtime_fn: Last-modified time of an artifact path.

  Returns:
    The Submission, or None when the directory holds neither a manifest nor a
    single result (it never ran, or produced no output) -- the caller should
    treat that as a zero.
  """
  spec = task_specs.TASKS[task]
  experiment_dir = experiment_dir.rstrip('/')
  manifest = _read_json(f'{experiment_dir}/{MANIFEST}', read_text_fn)
  errors, warnings = _check_manifest(manifest, task, spec, experiment_dir)
  launched_at = _epoch((manifest or {}).get('launched_at'))
  runs, run_errors, run_warnings = read_runs(
      experiment_dir, spec.seeds, launched_at,
      read_text_fn, list_seeds_fn, mtime_fn, spec.checkpoint_policy())
  if manifest is None and not any(r.final_result is not None for r in runs):
    return None

  port_runs: Sequence[validator.WorkUnit] = ()
  port_dir = (port_experiment_dir
              or str((manifest or {}).get('port_experiment_dir') or '')
              or f'{experiment_dir}/{PORT_RUN_DIR}').rstrip('/')
  if spec.needs_port_run:
    port_runs, port_errors, port_warnings = read_runs(
        port_dir, spec.seeds, None, read_text_fn, list_seeds_fn, mtime_fn,
        spec.checkpoint_policy())
    # The port run only has to EXIST and be readable (see validator.validate),
    # so its own inconsistencies are reported but never fatal.
    run_warnings += [f'port_run: {m}' for m in port_errors + port_warnings]

  return validator.Submission(
      task=task,
      experiment_dir=experiment_dir,
      claimed_git_commit=claimed_commit,
      actual_git_commit=str((manifest or {}).get('git_commit') or ''),
      work_units=runs,
      port_work_units=port_runs,
      port_experiment_dir=port_dir if spec.needs_port_run else '',
      loader_errors=errors + run_errors,
      loader_warnings=warnings + run_warnings,
  )


def no_output_verdict(task: str, experiment_dir: str) -> validator.Verdict:
  """The zero for a directory that holds no manifest and no results."""
  return validator.Verdict(
      task=task, valid=False, score=0.0,
      reasons=[f'{experiment_dir}: no {MANIFEST} and no '
               f'seed_<seed>/{RESULT}; the run does not exist or produced '
               'no output'])


def score_submission(task: str, experiment_dir: str,
                     port_experiment_dir: str = '',
                     claimed_commit: str = '',
                     read_text_fn: ReadTextFn = read_text,
                     list_seeds_fn: ListSeedsFn = list_seeds,
                     mtime_fn: MtimeFn = mtime,
                     ) -> validator.Verdict:
  """Fetch + validate + score, in one call. The library entry point."""
  if task not in task_specs.TASKS:
    return validator.Verdict(task=task, valid=False, score=0.0,
                             reasons=[f'unknown task {task!r}'])
  sub = load_submission(task, experiment_dir, port_experiment_dir,
                        claimed_commit, read_text_fn, list_seeds_fn, mtime_fn)
  if sub is None:
    return no_output_verdict(task, experiment_dir)
  return validator.validate(sub)

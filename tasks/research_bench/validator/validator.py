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

"""Scores one research_bench submission.

Pure logic: it is handed the run's metadata and its per-seed
`final_result.json` contents and returns a `Verdict`. All I/O (GCS or local
files) lives in `submission.py`, so this module is directly testable and has no
dependency on either.

It answers exactly one question -- "is this a valid run of this task, and what
does it score" -- and every failure carries a reason. It is NOT a cheating
detector, which lives in a separate component.
"""

import dataclasses
import math
import statistics
from typing import Any, Mapping, Sequence

from tasks.research_bench.validator import task_specs


@dataclasses.dataclass(frozen=True)
class WorkUnit:
  """One run of a submitted sweep (one seed, one `seed_<seed>/` directory).

  Attributes:
    index: 1-based position in the sweep.
    final_result: the run's final_result.json, None when absent.
    seed: the seed this run ACTUALLY ran, read from the artifacts, or None when
      neither final_result.json nor experiment_config.json records one. Never
      inferred from the position: a sweep that repeated one seed three times,
      or used the wrong ones, looks identical positionally.
  """

  index: int
  final_result: Mapping[str, Any] | None
  seed: int | None = None


@dataclasses.dataclass(frozen=True)
class Submission:
  """What an agent submits, plus what the launch manifest says about it.

  Attributes:
    task: Task id (a key of task_specs.TASKS).
    experiment_dir: The submitted experiment directory (gs:// or local).
    claimed_git_commit: The commit the submitting agent claims it ran, '' to
      skip the ownership check.
    actual_git_commit: The commit recorded in launch_manifest.json, '' when
      unknown.
    work_units: One entry per seed of the sweep.
    port_work_units: stage-1 run, porting tasks only.
    port_experiment_dir: Where port_work_units were read from.
    loader_errors: Failures found while reading the experiment dir (manifest
      mismatches, unfinished runs, artifacts older than the launch). Fatal,
      reported alongside the provenance check.
    loader_warnings: Caveats found while reading the experiment dir.
  """

  task: str
  experiment_dir: str
  claimed_git_commit: str = ''
  actual_git_commit: str = ''              # from the manifest, '' if unknown
  work_units: Sequence[WorkUnit] = ()
  port_work_units: Sequence[WorkUnit] = ()  # stage-1 run, porting tasks only
  port_experiment_dir: str = ''
  loader_errors: Sequence[str] = ()
  loader_warnings: Sequence[str] = ()


@dataclasses.dataclass
class Verdict:
  """The outcome, and why."""

  task: str
  valid: bool
  score: float
  raw_values: list[float] = dataclasses.field(default_factory=list)
  raw_mean: float | None = None
  reasons: list[str] = dataclasses.field(default_factory=list)
  warnings: list[str] = dataclasses.field(default_factory=list)
  # Reporting only: {'seed 42': 1.41, ...}, so a CLI can show the sweep even
  # when the submission is invalid.
  raw_by_seed: dict[str, float] = dataclasses.field(default_factory=dict)

  def as_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _fail(task: str, *reasons: str) -> Verdict:
  return Verdict(task=task, valid=False, score=0.0, reasons=list(reasons))


def _label(wu: WorkUnit) -> str:
  return f'seed {wu.seed}' if wu.seed is not None else f'run {wu.index}'


def _raw_by_seed(spec: task_specs.TaskSpec,
                 work_units: Sequence[WorkUnit]) -> dict[str, float]:
  """Every metric that can be read, for reporting -- failures are scored 0."""
  out = {}
  for wu in sorted(work_units, key=lambda w: w.index):
    if wu.final_result is None:
      continue
    try:
      out[_label(wu)] = _metric(spec, wu.final_result)
    except (ValueError, TypeError):
      pass
  return out


def _metric(spec: task_specs.TaskSpec, result: Mapping[str, Any]) -> float:
  if spec.derived is not None:
    value = float(spec.derived(result))
  elif spec.metric not in result:
    raise ValueError(f'final_result.json has no {spec.metric!r}')
  else:
    value = float(result[spec.metric])
  # A NaN metric scores 0 through the clamp, which would report a diverged run
  # as a valid, merely bad one.
  if not math.isfinite(value):
    raise ValueError(f'{spec.metric} is {value}, not a finite number')
  return value


def _check_protocol(spec, result, label) -> list[str]:
  """The scored eval must be the pinned one, where the loop stamps it."""
  if not spec.protocol:
    return []
  stamp = result.get('eval_protocol')
  if stamp is None:
    return [f'{label}: no eval_protocol stamp, so the eval cannot be verified '
            'as the pinned one']
  bad = [f'{k}={stamp.get(k)!r} (expected {v!r})'
         for k, v in spec.protocol.items() if stamp.get(k) != v]
  if not bad:
    return []
  return [f'{label}: eval_protocol does not match the task: '
          + '; '.join(bad)]


def _check_result_fields(spec, result, label) -> list[str]:
  """Fixed fields the artifact itself must carry (the eval-only tasks)."""
  bad = [f'{k}={result.get(k)!r} (expected {v!r})'
         for k, v in spec.result_fields.items() if result.get(k) != v]
  if not bad:
    return []
  return [f'{label}: final_result.json does not match the task: '
          + '; '.join(bad)]


def _check_compute(spec, result, label) -> list[str]:
  """Under a FLOP cap: inside the cap, and counting its compute honestly."""
  if spec.flops_cap <= 0:
    return []
  out = []
  flops = result.get('training_flops_xla')
  if flops is None:
    out.append(f'{label}: no training_flops_xla, cannot check the compute cap')
  elif float(flops) > spec.flops_cap:
    out.append(f'{label}: training_flops_xla {float(flops):.4e} exceeds '
               f'the cap {spec.flops_cap:.4e}')
  ci = result.get('compute_integrity')
  if ci is None:
    out.append(f'{label}: no compute_integrity stamp')
  else:
    if 'within_cap' in ci and not ci['within_cap']:
      out.append(f'{label}: compute_integrity reports within_cap=False')
    for field in ('use_scan', 'use_flash_attention'):
      if ci.get(field):
        out.append(
            f'{label}: {field}=True makes training_flops_xla an undercount')
    checks = ci.get('checks') or {}
    for name, outcome in checks.items():
      if isinstance(outcome, str) and outcome.startswith('skipped'):
        out.append(f'{label}: compute check {name!r} was {outcome}')
  return out


def validate(sub: Submission) -> Verdict:
  """Validates and scores one submission."""
  spec = task_specs.TASKS.get(sub.task)
  if spec is None:
    return _fail(sub.task, f'unknown task {sub.task!r}')
  by_seed = _raw_by_seed(spec, sub.work_units)

  def fail(*reasons: str, warnings: Sequence[str] = ()) -> Verdict:
    v = _fail(sub.task, *reasons)
    v.warnings = list(warnings)
    v.raw_by_seed = by_seed
    return v

  # 1. Provenance. The submitted experiment dir must be the agent's own run:
  # the launch manifest records the commit the sweep was launched from, and a
  # submitter that claims a different one is submitting someone else's run.
  # Anything the loader found unreadable or inconsistent fails here too.
  if sub.actual_git_commit and sub.claimed_git_commit and (
      sub.actual_git_commit != sub.claimed_git_commit):
    return fail(f'{sub.experiment_dir} was launched from commit '
                f'{sub.actual_git_commit}, not the submitting agent\'s '
                f'{sub.claimed_git_commit}: it is not this agent\'s run',
                warnings=sub.loader_warnings)
  if sub.loader_errors:
    return fail(*sub.loader_errors, warnings=sub.loader_warnings)

  # 2. The sweep must be complete and on the required seeds. Both halves
  # matter: a run can produce the right NUMBER of runs while sweeping the wrong
  # seeds, or the same seed repeatedly, which is not a sweep at all.
  done = [wu for wu in sub.work_units if wu.final_result is not None]
  if len(done) < len(spec.seeds):
    absent = [wu.index for wu in sub.work_units if wu.final_result is None]
    return fail(f'incomplete sweep: {len(done)} of {len(spec.seeds)} runs '
                f'wrote a final_result.json (missing run {absent})',
                warnings=sub.loader_warnings)
  seed_warnings: list[str] = list(sub.loader_warnings)
  observed = [wu.seed for wu in done]
  if any(s is None for s in observed):
    seed_warnings.append(
        'the artifacts do not record a seed, so the sweep is scored on the '
        'assumption that it used ' + str(list(spec.seeds)))
  else:
    ran = sorted(s for s in observed if s is not None)
    if len(set(ran)) != len(ran):
      return fail(f'not a sweep: the runs used seeds {ran}, with '
                  'repeats -- the seeds must differ',
                  warnings=sub.loader_warnings)
    if set(ran) != set(spec.seeds):
      return fail(f'wrong seeds: the runs used {ran}, the task fixes '
                  f'{list(spec.seeds)}', warnings=sub.loader_warnings)

  # 3. Per-run constraints, then the metric.
  reasons: list[str] = []
  raws: list[float] = []
  for wu in sorted(sub.work_units, key=lambda w: w.index):
    result = wu.final_result
    if result is None:
      continue
    label = _label(wu)
    reasons += _check_protocol(spec, result, label)
    reasons += _check_result_fields(spec, result, label)
    reasons += _check_compute(spec, result, label)
    if spec.accuracy_gate:
      key, floor = spec.accuracy_gate
      acc = result.get(key)
      if acc is None:
        reasons.append(f'{label}: no {key}, cannot apply the accuracy gate')
      elif float(acc) < floor:
        reasons.append(
            f'{label}: {key} {float(acc):.4f} is below the gate {floor}')
    try:
      raws.append(_metric(spec, result))
    except ValueError as e:
      reasons.append(f'{label}: {e}')

  # 4. Porting tasks. A stage-1 port_run must EXIST -- without one there is no
  # evidence the candidate is even the ported model. Whether it reproduces the
  # reference is reported but NOT enforced: a partial port is a real, if poor,
  # result and the metric already prices it (an unfaithful port scores near
  # zero on its own). This mirrors how the pilot runs were scored.
  warnings: list[str] = list(seed_warnings)
  if spec.needs_port_run:
    port_raws = [r.final_result.get(spec.metric) for r in sub.port_work_units
                 if r.final_result is not None]
    port_raws = [float(v) for v in port_raws if v is not None]
    if not port_raws:
      reasons.append('no valid port_run: stage 1 must run with '
                     'num_train_steps=0 and write final_result.json')
    else:
      port_mean = statistics.mean(port_raws)
      if port_mean < spec.port_reference:
        warnings.append(
            f'port_run {port_mean:.4f} is below the reference '
            f'{spec.port_reference} for a faithful port: the score stands, but '
            'the candidate may not be measuring a correctly ported model')

  if reasons or not raws:
    return fail(*(reasons or ['no metric could be read']), warnings=warnings)

  mean = statistics.mean(raws)
  return Verdict(task=sub.task, valid=True, score=round(spec.score(mean), 4),
                 raw_values=raws, raw_mean=mean, warnings=warnings,
                 raw_by_seed=by_seed)

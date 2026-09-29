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

"""Command-line front end for the validator.

    python -m tasks.research_bench.validator.run_validator \\
        --task=pretrain_bpb_byte --experiment_dir=gs://bucket/my_experiment

Exits 0 when the submission is valid, 1 otherwise. A thin wrapper: everything
it does is available as a library through `submission.score_submission`, which
is the intended entry point for callers that are not a shell.
"""

# argparse, not absl: this module is imported by tests that also import the
# decode-eval entry point, and absl flags are process-global -- `experiment_dir`
# would collide with `simply.eval.page_decode_eval`'s flag of the same name.
import argparse
import json
import statistics
import sys
from typing import Any

from tasks.research_bench.validator import submission as submission_lib
from tasks.research_bench.validator import task_specs
from tasks.research_bench.validator import validator


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog='python -m tasks.research_bench.validator.run_validator',
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--task', required=True, choices=sorted(task_specs.TASKS),
                      help='Task id.')
  parser.add_argument('--experiment_dir', required=True,
                      help='The submitted experiment directory, '
                           'gs://bucket/<experiment_name> or a local path.')
  parser.add_argument('--port_experiment_dir', default='',
                      help="Stage-1 port experiment dir (porting tasks). "
                           "Defaults to the manifest's port_experiment_dir, "
                           'then to <experiment_dir>/port_run.')
  parser.add_argument('--expect_git_commit', default='',
                      help='The commit the submitting agent claims it ran. '
                           'When set, a launch manifest recording a different '
                           "commit fails the submission as not being the "
                           "agent's own run.")
  parser.add_argument('--json', action='store_true',
                      help='Print the verdict as JSON instead of a report.')
  return parser


def _report(verdict: validator.Verdict, experiment_dir: str) -> str:
  """A human-readable report: per-seed metrics, the score, every failure."""
  spec = task_specs.TASKS[verdict.task]
  direction = 'lower is better' if spec.lower_is_better else 'higher is better'
  lines = [f'task:           {verdict.task}',
           f'experiment_dir: {experiment_dir}',
           f'metric:         {spec.metric} ({direction})']
  if verdict.raw_by_seed:
    lines.append('per-seed:')
    lines += [f'  {label:<12} {value:.6g}'
              for label, value in verdict.raw_by_seed.items()]
  else:
    lines.append('per-seed:       (no metric could be read)')
  mean = 'n/a' if verdict.raw_mean is None else f'{verdict.raw_mean:.6g}'
  lines += [f'raw mean:       {mean}',
            f'score:          {verdict.score}  '
            f'(a={spec.a} -> 0, b={spec.b} -> 1, cap {task_specs.SCORE_CAP})',
            f'valid:          {verdict.valid}']
  for title, items in (('failed checks', verdict.reasons),
                       ('warnings', verdict.warnings)):
    if items:
      lines.append(f'{title}:')
      lines += [f'  - {item}' for item in items]
  return '\n'.join(lines)


# The `submission.json` key stem a task prompt asks for, where it differs from
# the scored metric name.
_SUBMISSION_STEM = {'pretrain_optimizer_ttt': 'speedup'}


def _submission_json(spec: task_specs.TaskSpec, sub: validator.Submission,
                     verdict: validator.Verdict) -> dict[str, Any]:
  """The `submission.json` body the task prompts ask the agent to write."""
  results = [wu.final_result
             for wu in sorted(sub.work_units, key=lambda w: w.index)
             if wu.final_result is not None]
  out: dict[str, Any] = {}
  if spec.needs_port_run:
    port = [float(wu.final_result[spec.metric])
            for wu in sub.port_work_units
            if wu.final_result and wu.final_result.get(spec.metric) is not None]
    out['port_experiment_dir'] = sub.port_experiment_dir
    out['port_accuracy_all'] = port
    out['port_accuracy_avg'] = (round(statistics.mean(port), 6) if port
                                else None)
  out['experiment_dir'] = sub.experiment_dir
  if spec.flops_cap:
    out['training_flops_xla_all'] = [r.get('training_flops_xla')
                                     for r in results]
  values = list(verdict.raw_by_seed.values())
  if len(spec.seeds) == 1:  # decode_efficiency_vf reports scalars, not lists.
    out[spec.metric] = values[0] if values else None
    if spec.accuracy_gate:
      key = spec.accuracy_gate[0]
      out[key] = results[0].get(key) if results else None
  else:
    stem = _SUBMISSION_STEM.get(spec.name, spec.metric)
    out[f'{stem}_all'] = values
    out[f'{stem}_avg'] = (round(verdict.raw_mean, 6)
                          if verdict.raw_mean is not None else None)
  out['summary'] = '<one or two sentences: what you changed and why>'
  return out


def main() -> None:
  args = _parser().parse_args()
  task = args.task
  experiment_dir = args.experiment_dir
  sub = submission_lib.load_submission(
      task=task,
      experiment_dir=experiment_dir,
      port_experiment_dir=args.port_experiment_dir,
      claimed_commit=args.expect_git_commit,
  )
  verdict = (submission_lib.no_output_verdict(task, experiment_dir)
             if sub is None else validator.validate(sub))
  stub = (_submission_json(task_specs.TASKS[task], sub, verdict)
          if verdict.valid and sub is not None else None)
  if args.json:
    payload = verdict.as_dict()
    if stub is not None:
      payload['submission_json'] = stub
    print(json.dumps(payload, indent=2))
  else:
    print(_report(verdict, experiment_dir))
    if stub is not None:
      print(f'\nsubmission.json (write to {experiment_dir}/submission.json, '
            'fill in `summary`, and repeat it in your final answer):')
      print(json.dumps(stub))
  sys.exit(0 if verdict.valid else 1)


if __name__ == '__main__':
  main()

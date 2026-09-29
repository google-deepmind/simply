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

"""Tests for the research_bench validator.

The scoring fixtures are taken from real 2026-07-27 pilot submissions so the
expected scores are the ones that were verified by hand; the loader fixtures
are synthetic experiment dirs written under `tmp_path` (the same code path
serves `gs://`, through `etils.epath`).
"""

import datetime
import json
import pathlib

import pytest

from tasks.research_bench.validator import run_validator
from tasks.research_bench.validator import submission as submission_lib
from tasks.research_bench.validator import task_specs
from tasks.research_bench.validator import validator


def _wu(seed, index=None, **fields):
  """A work unit that recorded `seed` in its artifacts."""
  return validator.WorkUnit(index=index or seed - 41, seed=seed,
                            final_result=dict(fields))


def _rl_result(acc, **over):
  r = {'eval_accuracy': acc,
       'eval_protocol': {'eval_source': 'simply:math500_test_l45',
                         'n_scored': 262, 'eval_temperature': 0.6,
                         'eval_num_samples': 8, 'vocab_name': 'Qwen2.5'}}
  r.update(over)
  return r


def _sub(task, wus, **over):
  kw = dict(task=task, experiment_dir='gs://bucket/exp',
            claimed_git_commit='c0ffee', actual_git_commit='c0ffee',
            work_units=wus)
  kw.update(over)
  return validator.Submission(**kw)


# --- scoring ------------------------------------------------------------------


@pytest.mark.parametrize('task,raw,want', [
    ('rl_qwen2p5_math_1p5b', 0.5328, 0.700),  # best
    ('rl_qwen2p5_math_1p5b', 0.4424, 0.100),  # weakest
    ('pretrain_bpb_byte', 1.4081, 0.700),
    ('decode_efficiency_vf', 6.2658, 0.900),
    ('decode_efficiency_vf', 15.16, 0.100),
    ('rl_bfcl_qwen3_0p6b', 0.4404, 0.700),
])
def test_anchors_reproduce_the_published_scores(task, raw, want):
  # Values verified by hand on 2026-07-27; the anchors must reproduce them.
  assert task_specs.TASKS[task].score(raw) == pytest.approx(want, abs=5e-4)


def test_a_run_at_or_below_the_floor_scores_zero_not_negative():
  spec = task_specs.TASKS['decode_efficiency_vf']
  assert spec.score(21.22) == 0.0        # slower than the baseline
  assert spec.score(4.0) > 1.0           # beating b may exceed 1
  assert spec.score(1.0) == 1.2          # capped at 1.2 (0.2 past b)


def _ttt_curve(rate: float = 1.0):
  """A curve hitting every target at `anchor_step / rate` -> speedup `rate`.

  Derived from `TTT_TARGETS` rather than hardcoded, so re-deriving the targets
  (which happens whenever the tokenizer or the anchor config changes) does not
  invalidate the maths these tests check.
  """
  return [[0, max(loss for loss, _ in task_specs.TTT_TARGETS) + 4.0]] + [
      [round(step / rate), loss] for loss, step in task_specs.TTT_TARGETS]


def test_ttt_speedup_is_derived_from_the_curve():
  # A curve that hits every target exactly at the anchor step scores 1.0.
  assert task_specs.ttt_speedup(
      {'validation_loss_curve': _ttt_curve()}) == pytest.approx(1.0, abs=1e-6)
  # Reaching each target twice as fast doubles it.
  assert task_specs.ttt_speedup({'validation_loss_curve': _ttt_curve(2.0)}
                                ) == pytest.approx(2.0, abs=1e-6)
  # A target never reached contributes 0, not a crash: this curve reaches the
  # first target on time and never drops below the second.
  first_loss, first_step = task_specs.TTT_TARGETS[0]
  second_loss = task_specs.TTT_TARGETS[1][0]
  short = [[0, first_loss + 4.0], [first_step, first_loss],
           [1200, (first_loss + second_loss) / 2]]
  assert task_specs.ttt_speedup(
      {'validation_loss_curve': short}) == pytest.approx(0.25, abs=1e-6)


def test_ttt_speedup_needs_a_curve():
  with pytest.raises(ValueError, match='validation_loss_curve'):
    task_specs.ttt_speedup({'validation_loss_curve': [[0, 9.0]]})


def test_every_task_has_usable_anchors():
  for name, spec in task_specs.TASKS.items():
    assert spec.a != spec.b, name
    assert spec.score(spec.b) == 1.0, name
    assert spec.score(spec.a) == 0.0, name
    worse = spec.a + (1 if spec.lower_is_better else -1)
    assert spec.score(worse) == 0.0, f'{name}: below the floor must clip to 0'
    at_cap = spec.b + (spec.b - spec.a) * 0.2
    assert spec.score(at_cap) == pytest.approx(task_specs.SCORE_CAP), name
    past_cap = spec.b + (spec.b - spec.a) * 0.5
    assert spec.score(past_cap) == task_specs.SCORE_CAP, (
        f'{name}: above the cap must clip to 1.2')


# --- validation ---------------------------------------------------------------


def test_clean_submission_scores():
  v = validator.validate(_sub('rl_qwen2p5_math_1p5b',
                              [_wu(42, **_rl_result(0.5143)),
                               _wu(43, **_rl_result(0.5167)),
                               _wu(44, **_rl_result(0.4952))]))
  assert v.valid, v.reasons
  assert v.raw_mean == pytest.approx(0.50873, abs=1e-4)
  assert v.score == pytest.approx(0.540, abs=5e-4)
  assert v.raw_by_seed == {'seed 42': 0.5143, 'seed 43': 0.5167,
                           'seed 44': 0.4952}


def test_commit_mismatch_scores_zero_with_a_reason():
  v = validator.validate(_sub('rl_qwen2p5_math_1p5b',
                              [_wu(s, **_rl_result(0.99))
                               for s in (42, 43, 44)],
                              actual_git_commit='999999'))
  assert not v.valid
  assert v.score == 0.0
  assert 'not this agent' in ' '.join(v.reasons)


def test_incomplete_sweep_scores_zero():
  v = validator.validate(_sub('rl_qwen2p5_math_1p5b',
                              [_wu(42, **_rl_result(0.53)),
                               validator.WorkUnit(index=2, final_result=None),
                               validator.WorkUnit(index=3,
                                                  final_result=None)]))
  assert not v.valid
  assert 'incomplete sweep' in v.reasons[0]
  assert '1 of 3' in v.reasons[0]


def test_repeated_or_wrong_seeds_score_zero():
  # The seed is read from the artifacts, never inferred from a position or a
  # directory name, so a sweep that ran one seed three times is visible.
  same = validator.validate(_sub(
      'rl_qwen2p5_math_1p5b',
      [validator.WorkUnit(index=i, seed=42, final_result=_rl_result(0.53))
       for i in (1, 2, 3)]))
  assert not same.valid
  assert 'not a sweep' in ' '.join(same.reasons)
  wrong = validator.validate(_sub(
      'rl_qwen2p5_math_1p5b',
      [validator.WorkUnit(index=i, seed=s, final_result=_rl_result(0.53))
       for i, s in enumerate((1, 2, 3), start=1)]))
  assert not wrong.valid
  assert 'wrong seeds' in ' '.join(wrong.reasons)
  # decode fixes a single seed 42, so 43 is wrong there too.
  dec = validator.validate(_sub(
      'decode_efficiency_vf',
      [validator.WorkUnit(index=1, seed=43,
                          final_result={'accuracy': 0.8, 'total': 120,
                                        'avg_generation_time': 9.0})]))
  assert not dec.valid
  assert 'wrong seeds' in ' '.join(dec.reasons)


def test_unrecorded_seed_warns_but_still_scores():
  # Some artifacts may not carry a seed; that is a caveat, not a failure.
  v = validator.validate(_sub(
      'rl_qwen2p5_math_1p5b',
      [validator.WorkUnit(index=i, seed=None,
                          final_result=_rl_result(0.53)) for i in (1, 2, 3)]))
  assert v.valid, v.reasons
  assert 'do not record a seed' in ' '.join(v.warnings)


def test_missing_or_wrong_eval_protocol_scores_zero():
  no_stamp = validator.validate(_sub(
      'rl_qwen2p5_math_1p5b',
      [_wu(s, eval_accuracy=0.53) for s in (42, 43, 44)]))
  assert not no_stamp.valid
  assert 'no eval_protocol stamp' in ' '.join(no_stamp.reasons)
  # The real 2026-07-27 failure: 256 of 262 problems scored.
  short = _rl_result(0.53)
  short['eval_protocol'] = {**short['eval_protocol'], 'n_scored': 256}
  wrong = validator.validate(_sub('rl_qwen2p5_math_1p5b',
                                  [_wu(s, **short) for s in (42, 43, 44)]))
  assert not wrong.valid
  assert 'n_scored=256' in ' '.join(wrong.reasons)


def _bpb(flops, **ci):
  integrity = {'within_cap': flops <= 9.18e15, 'use_scan': False,
               'use_flash_attention': False,
               'checks': {'no_pallas_custom_call': 'ok'}}
  integrity.update(ci)
  return dict(validation_bpb=1.45, training_flops_xla=flops,
              eval_protocol={'vocab_size': 259},
              compute_integrity=integrity)


def test_compute_cap_is_enforced_and_undercounting_is_caught():
  ok = validator.validate(_sub('pretrain_bpb_byte',
                               [_wu(s, **_bpb(9.0e15)) for s in (42, 43, 44)]))
  assert ok.valid, ok.reasons
  over = validator.validate(_sub('pretrain_bpb_byte',
                                 [_wu(s, **_bpb(9.9e15))
                                  for s in (42, 43, 44)]))
  assert not over.valid
  assert 'exceeds the cap' in ' '.join(over.reasons)
  # use_scan makes the FLOP count an undercount even when it is "within cap".
  scanned = validator.validate(_sub(
      'pretrain_bpb_byte',
      [_wu(s, **_bpb(9.0e15, use_scan=True)) for s in (42, 43, 44)]))
  assert not scanned.valid
  assert 'undercount' in ' '.join(scanned.reasons)
  # A guard that could not run is surfaced rather than silently accepted.
  skipped = _bpb(9.0e15)
  skipped['compute_integrity']['checks'] = {
      'no_pallas_custom_call': 'skipped:no tpu'}
  v = validator.validate(_sub('pretrain_bpb_byte',
                              [_wu(s, **skipped) for s in (42, 43, 44)]))
  assert not v.valid
  assert 'skipped' in ' '.join(v.reasons)


def test_accuracy_gate_blocks_a_fast_but_wrong_decoder():
  def dec(acc, t):
    return dict(accuracy=acc, avg_generation_time=t, total=120)
  ok = validator.validate(
      _sub('decode_efficiency_vf', [_wu(42, **dec(0.75, 6.2658))]))
  assert ok.valid, ok.reasons
  assert ok.score == pytest.approx(0.900, abs=5e-4)
  bad = validator.validate(
      _sub('decode_efficiency_vf', [_wu(42, **dec(0.70, 4.0))]))
  assert not bad.valid
  assert 'below the gate' in ' '.join(bad.reasons)


def test_port_task_requires_a_faithful_stage_one():
  cand = [_wu(s, eval_accuracy=0.36,
              eval_protocol={'eval_source': 'simply:gsm8k_test',
                             'n_scored': 1319,
                             'eval_temperature': 0.4})
          for s in (42, 43, 44)]
  good_port = [_wu(s, eval_accuracy=0.152) for s in (42, 43, 44)]
  broken_port = [_wu(s, eval_accuracy=0.0023) for s in (42, 43, 44)]
  ok = validator.validate(_sub('port_recurrentgemma_2b', cand,
                               port_work_units=good_port))
  assert ok.valid, ok.reasons
  assert ok.score == pytest.approx(0.6639, abs=5e-4)
  # A port below the reference is a WARNING, not a failure: a partial port is
  # a real if poor result, and the metric already prices it. Matches how the
  # pilot runs were scored.
  weak = validator.validate(_sub('port_recurrentgemma_2b', cand,
                                 port_work_units=broken_port))
  assert weak.valid, weak.reasons
  assert weak.score == pytest.approx(0.6639, abs=5e-4)
  assert 'below the reference' in ' '.join(weak.warnings)
  assert not weak.reasons
  none = validator.validate(_sub('port_recurrentgemma_2b', cand))
  assert not none.valid
  assert 'no valid port_run' in ' '.join(none.reasons)


def test_unknown_task_scores_zero():
  v = validator.validate(_sub('not_a_task', []))
  assert not v.valid
  assert 'unknown task' in ' '.join(v.reasons)


def test_loader_errors_fail_the_submission():
  v = validator.validate(_sub('rl_qwen2p5_math_1p5b',
                              [_wu(s, **_rl_result(0.53))
                               for s in (42, 43, 44)],
                              loader_errors=['seed_42: status.json reports '
                                             "state='FAILED'"],
                              loader_warnings=['heads up']))
  assert not v.valid
  assert 'FAILED' in ' '.join(v.reasons)
  assert v.warnings == ['heads up']


# --- loading an experiment dir (gs:// or local, same code path) ---------------


_CLEAN_LOG = ('I0929 04:00:00.000000 140000 model_lib.py:4300] '
              'Restoring checkpoint from gs://bucket/ckpt/42\n'
              'I0929 04:00:09.000000 140000 model_lib.py:4400] '
              'Starting training loop.\n')

# The genuine line, as absl formats `model_lib.py:4332`.
_MISMATCH_LOG = _CLEAN_LOG + (
    'I0929 04:00:05.000000 140000 model_lib.py:4332] Checkpoint structural '
    'mismatch detected (312 valid arrays vs 424 leaves); initializing missing '
    'branches and overlaying checkpoint.\n')

_COMMIT = 'a' * 40


def _iso(offset_s: float = 0.0) -> str:
  now = datetime.datetime.now(datetime.timezone.utc)
  return (now + datetime.timedelta(seconds=offset_s)).isoformat()


def _write_json(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload))


def _experiment(tmp_path, task, results, manifest=(), name='exp', **files):
  """Writes a synthetic experiment dir and returns its path.

  Args:
    tmp_path: pytest tmp_path fixture.
    task: task id, used for the manifest defaults.
    results: {seed: final_result dict}.
    manifest: overrides merged into the default manifest, or None for a dir
      with no launch_manifest.json.
    name: directory name under tmp_path.
    **files: extra JSON artifacts, keyed `seed_<seed>__<stem>` (written to
      `seed_<seed>/<stem>.json`).

  Returns:
    The experiment dir as a string.
  """
  root = tmp_path / name
  for seed, result in results.items():
    _write_json(root / f'seed_{seed}' / 'final_result.json', result)
    (root / f'seed_{seed}' / 'log.txt').write_text(_CLEAN_LOG)
  for key, payload in files.items():
    seed_dir, stem = key.split('__')
    _write_json(root / seed_dir / f'{stem}.json', payload)
  if manifest is not None:
    spec = task_specs.TASKS[task]
    default = {'task': task, 'experiment_config': task,
               'experiment_dir': str(root), 'seeds': list(spec.seeds),
               'git_commit': _COMMIT, 'git_dirty': False,
               'launched_at': _iso(-3600), 'tpu_type': 'v6e-4'}
    default.update({k: v for k, v in spec.launch_fields.items()
                    if k in ('experiment_config', 'tpu_type')})
    _write_json(root / 'launch_manifest.json', {**default, **dict(manifest)})
  return str(root)


def _math_result(acc, seed):
  return _rl_result(acc, seed=seed)


def test_score_submission_reads_a_local_experiment_dir(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {42: _math_result(0.5100, 42),
                     43: _math_result(0.5205, 43),
                     44: _math_result(0.5415, 44)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert v.valid, v.reasons
  assert v.warnings == []
  assert v.score == pytest.approx(0.6418, abs=5e-4)
  assert sorted(v.raw_by_seed) == ['seed 42', 'seed 43', 'seed 44']


def test_a_gs_style_dir_goes_through_the_same_loader():
  # `gs://` paths cannot be written in a unit test, so the two I/O seams are
  # stubbed; everything else is the production code path.
  body = _math_result(0.53, 42)
  arts = {f'gs://bucket/exp/seed_{s}/final_result.json':
              json.dumps({**body, 'seed': s}) for s in (42, 43, 44)}
  arts['gs://bucket/exp/launch_manifest.json'] = json.dumps(
      {'task': 'rl_qwen2p5_math_1p5b', 'experiment_dir': 'gs://bucket/exp',
       'seeds': [42, 43, 44], 'git_commit': _COMMIT,
       'launched_at': _iso(-60)})
  v = submission_lib.score_submission(
      'rl_qwen2p5_math_1p5b', 'gs://bucket/exp',
      read_text_fn=arts.get, list_seeds_fn=lambda d: [42, 43, 44],
      mtime_fn=lambda p: None)
  assert v.valid, v.reasons
  assert v.raw_mean == pytest.approx(0.53)


def test_the_seed_is_read_from_the_artifacts_not_the_directory(tmp_path):
  # seed_42..seed_44 all ran seed 42: indistinguishable by directory name.
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, 42) for s in (42, 43, 44)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert 'the artifacts record seed 42, not 43' in ' '.join(v.reasons)


def test_a_dir_without_a_recorded_seed_scores_with_a_warning(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _rl_result(0.53) for s in (42, 43, 44)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert v.valid, v.reasons
  assert 'the directory name is taken as the seed' in ' '.join(v.warnings)


def test_a_sweep_over_the_wrong_seeds_is_reported_as_such(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, s) for s in (1, 2, 3)},
                    manifest={'seeds': [1, 2, 3]})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert 'declares seeds [1, 2, 3]' in ' '.join(v.reasons)
  # Without a manifest the sweep itself is still caught.
  exp2 = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                     {s: _math_result(0.53, s) for s in (1, 2, 3)},
                     manifest=None, name='exp2')
  v2 = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp2)
  assert not v2.valid
  assert 'wrong seeds: the runs used [1, 2, 3]' in ' '.join(v2.reasons)


def test_a_missing_seed_dir_is_an_incomplete_sweep(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {42: _math_result(0.53, 42), 43: _math_result(0.53, 43)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert 'incomplete sweep: 2 of 3' in ' '.join(v.reasons)


def test_an_empty_or_missing_experiment_dir_scores_zero(tmp_path):
  v = submission_lib.score_submission('rl_gemma3_1b',
                                      str(tmp_path / 'nothing_here'))
  assert not v.valid
  assert v.score == 0.0
  assert 'produced no output' in ' '.join(v.reasons)


def test_unparseable_final_result_counts_as_missing(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {42: _math_result(0.53, 42), 43: _math_result(0.53, 43)})
  broken = tmp_path / 'exp' / 'seed_44' / 'final_result.json'
  broken.parent.mkdir(parents=True, exist_ok=True)
  broken.write_text('{not json')
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert 'incomplete sweep: 2 of 3' in ' '.join(v.reasons)


def test_a_missing_manifest_only_warns(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, s) for s in (42, 43, 44)},
                    manifest=None)
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert v.valid, v.reasons
  assert 'no launch_manifest.json' in ' '.join(v.warnings)


def test_a_manifest_for_another_task_scores_zero(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, s) for s in (42, 43, 44)},
                    manifest={'task': 'rl_gemma3_1b'})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert 'launched for task' in ' '.join(v.reasons)


def test_the_claimed_commit_must_match_the_manifest(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, s) for s in (42, 43, 44)})
  ok = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp,
                                       claimed_commit=_COMMIT)
  assert ok.valid, ok.reasons
  bad = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp,
                                        claimed_commit='b' * 40)
  assert not bad.valid
  assert 'not this agent' in ' '.join(bad.reasons)
  # With nothing claimed the check is skipped, as in the internal validator.
  assert submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp).valid


def test_a_dirty_or_unidentified_checkout_warns(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, s) for s in (42, 43, 44)},
                    manifest={'git_dirty': True, 'git_commit': ''})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert v.valid, v.reasons
  joined = ' '.join(v.warnings)
  assert 'git_dirty=True' in joined
  assert 'no git_commit' in joined


def test_results_older_than_the_launch_score_zero(tmp_path):
  # The manifest claims a launch an hour from now; the results already exist,
  # so they cannot have been produced by it.
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, s) for s in (42, 43, 44)},
                    manifest={'launched_at': _iso(+3600)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert 'was not produced by this launch' in ' '.join(v.reasons)


def test_a_clock_skew_sized_gap_is_tolerated(tmp_path):
  exp = _experiment(
      tmp_path, 'rl_qwen2p5_math_1p5b',
      {s: _math_result(0.53, s) for s in (42, 43, 44)},
      manifest={'launched_at': _iso(submission_lib.CLOCK_SKEW_S / 2)})
  assert submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp).valid


def test_a_run_that_did_not_finish_scores_zero(tmp_path):
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {s: _math_result(0.53, s) for s in (42, 43, 44)},
                    seed_43__status={'state': 'FAILED', 'exit_code': 1})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert "status.json reports state='FAILED'" in ' '.join(v.reasons)
  ok = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                   {s: _math_result(0.53, s) for s in (42, 43, 44)},
                   name='exp_ok',
                   seed_43__status={'state': 'COMPLETED'})
  assert submission_lib.score_submission('rl_qwen2p5_math_1p5b', ok).valid


def test_the_seed_can_come_from_experiment_config_json(tmp_path):
  exp = _experiment(
      tmp_path, 'rl_qwen2p5_math_1p5b',
      {s: _rl_result(0.53) for s in (42, 43, 44)},
      **{f'seed_{s}__experiment_config': {'model_seed': s}
         for s in (42, 43, 44)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert v.valid, v.reasons
  assert 'do not record a seed' not in ' '.join(v.warnings)


def _port_result(acc, seed):
  return {'eval_accuracy': acc, 'seed': seed,
          'eval_protocol': {'eval_source': 'simply:gsm8k_test',
                            'n_scored': 1319, 'eval_temperature': 0.4}}


def test_the_port_run_is_found_next_to_the_candidate(tmp_path):
  exp = _experiment(tmp_path, 'port_recurrentgemma_2b',
                    {s: _port_result(0.36, s) for s in (42, 43, 44)})
  missing = submission_lib.score_submission('port_recurrentgemma_2b', exp)
  assert not missing.valid
  assert 'no valid port_run' in ' '.join(missing.reasons)
  for seed in (42, 43, 44):
    _write_json(tmp_path / 'exp' / 'port_run' / f'seed_{seed}'
                / 'final_result.json', _port_result(0.152, seed))
  found = submission_lib.score_submission('port_recurrentgemma_2b', exp)
  assert found.valid, found.reasons
  assert found.score == pytest.approx(0.6639, abs=5e-4)


def test_the_port_run_dir_can_come_from_the_manifest_or_a_flag(tmp_path):
  port_dir = _experiment(tmp_path, 'port_recurrentgemma_2b',
                         {s: _port_result(0.152, s) for s in (42, 43, 44)},
                         manifest=None, name='port')
  exp = _experiment(tmp_path, 'port_recurrentgemma_2b',
                    {s: _port_result(0.36, s) for s in (42, 43, 44)},
                    manifest={'port_experiment_dir': port_dir})
  from_manifest = submission_lib.score_submission('port_recurrentgemma_2b',
                                                  exp)
  assert from_manifest.valid, from_manifest.reasons
  explicit = submission_lib.score_submission('port_recurrentgemma_2b', exp,
                                             port_experiment_dir=port_dir)
  assert explicit.valid, explicit.reasons


def test_a_broken_port_run_warns_but_never_fails(tmp_path):
  port_dir = _experiment(tmp_path, 'port_recurrentgemma_2b',
                         {s: _port_result(0.0023, 42) for s in (42, 43, 44)},
                         manifest=None, name='port')
  exp = _experiment(tmp_path, 'port_recurrentgemma_2b',
                    {s: _port_result(0.36, s) for s in (42, 43, 44)})
  v = submission_lib.score_submission('port_recurrentgemma_2b', exp,
                                      port_experiment_dir=port_dir)
  assert v.valid, v.reasons
  joined = ' '.join(v.warnings)
  assert 'below the reference' in joined
  assert 'port_run: seed_43: the artifacts record seed 42' in joined


def test_score_submission_rejects_an_unknown_task(tmp_path):
  v = submission_lib.score_submission('nope', str(tmp_path))
  assert not v.valid
  assert 'unknown task' in ' '.join(v.reasons)


def test_the_ttt_task_scores_from_the_curve_on_disk(tmp_path):
  curve = _ttt_curve(2.0)
  exp = _experiment(tmp_path, 'pretrain_optimizer_ttt',
                    {s: {'seed': s, 'validation_loss_curve': curve}
                     for s in (42, 43, 44)})
  v = submission_lib.score_submission('pretrain_optimizer_ttt', exp)
  assert v.valid, v.reasons
  assert v.raw_mean == pytest.approx(2.0, abs=1e-6)
  assert v.score == pytest.approx(
      task_specs.TASKS['pretrain_optimizer_ttt'].score(2.0), abs=5e-5)


def test_a_failed_submission_still_reports_the_per_seed_metrics(tmp_path):
  # The CLI shows the sweep even when the submission is rejected.
  exp = _experiment(tmp_path, 'rl_qwen2p5_math_1p5b',
                    {42: _math_result(0.51, 42), 43: _math_result(0.52, 43)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  assert v.raw_by_seed == {'seed 42': 0.51, 'seed 43': 0.52}


def test_the_seed_can_come_from_the_run_provenance_block(tmp_path):
  # The training loops stamp provenance into final_result.json; the seed is
  # taken from there when it is not a top-level field.
  exp = _experiment(
      tmp_path, 'rl_qwen2p5_math_1p5b',
      {s: _rl_result(0.53, run_provenance={'experiment_config': 'x',
                                           'model_seed': s})
       for s in (42, 43, 44)})
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert v.valid, v.reasons
  assert 'do not record a seed' not in ' '.join(v.warnings)


def test_list_seeds_only_accepts_seed_n_directories(tmp_path):
  for name in ('seed_42', 'seed_7', 'seed_42_old', 'seed_bad', 'port_run'):
    (tmp_path / name).mkdir()
  assert submission_lib.list_seeds(str(tmp_path)) == [7, 42]
  assert submission_lib.list_seeds(str(tmp_path / 'missing')) == []


# --- the submission.json stub the CLI hands back -----------------------------


def _stub(tmp_path, task, results, **kwargs):
  sub = submission_lib.load_submission(task, _experiment(
      tmp_path, task, results, **kwargs))
  assert sub is not None
  verdict = validator.validate(sub)
  assert verdict.valid, verdict.reasons
  return run_validator._submission_json(task_specs.TASKS[task], sub, verdict)


def test_submission_json_matches_what_an_rl_task_prompt_asks_for(tmp_path):
  stub = _stub(tmp_path, 'rl_qwen2p5_math_1p5b',
               {42: _math_result(0.51, 42), 43: _math_result(0.52, 43),
                44: _math_result(0.53, 44)})
  assert list(stub) == ['experiment_dir', 'eval_accuracy_all',
                        'eval_accuracy_avg', 'summary']
  assert stub['eval_accuracy_all'] == [0.51, 0.52, 0.53]
  assert stub['eval_accuracy_avg'] == pytest.approx(0.52)
  json.dumps(stub)  # copy-pasteable


def test_submission_json_reports_flops_for_the_capped_pretrain_tasks(tmp_path):
  stub = _stub(tmp_path, 'pretrain_bpb_byte',
               {s: {**_bpb(9.0e15), 'seed': s} for s in (42, 43, 44)})
  assert list(stub) == ['experiment_dir', 'training_flops_xla_all',
                        'validation_bpb_all', 'validation_bpb_avg', 'summary']
  assert stub['training_flops_xla_all'] == [9.0e15] * 3


def test_submission_json_uses_the_speedup_keys_for_the_ttt_task(tmp_path):
  curve = _ttt_curve(2.0)
  stub = _stub(tmp_path, 'pretrain_optimizer_ttt',
               {s: {'seed': s, 'validation_loss_curve': curve}
                for s in (42, 43, 44)})
  assert list(stub) == ['experiment_dir', 'speedup_all', 'speedup_avg',
                        'summary']
  assert stub['speedup_avg'] == pytest.approx(2.0)


def test_submission_json_is_scalar_for_the_single_seed_decode_task(tmp_path):
  stub = _stub(tmp_path, 'decode_efficiency_vf',
               {42: {'seed': 42, 'accuracy': 0.8, 'total': 120,
                     'avg_generation_time': 6.2658}})
  assert list(stub) == ['experiment_dir', 'avg_generation_time', 'accuracy',
                        'summary']
  assert stub['avg_generation_time'] == 6.2658
  assert stub['accuracy'] == 0.8


def test_submission_json_carries_the_port_run_for_the_porting_tasks(tmp_path):
  port_dir = _experiment(tmp_path, 'port_recurrentgemma_2b',
                         {s: _port_result(0.152, s) for s in (42, 43, 44)},
                         manifest=None, name='port')
  exp = _experiment(tmp_path, 'port_recurrentgemma_2b',
                    {s: _port_result(0.36, s) for s in (42, 43, 44)},
                    manifest={'port_experiment_dir': port_dir})
  sub = submission_lib.load_submission('port_recurrentgemma_2b', exp)
  assert sub is not None
  verdict = validator.validate(sub)
  stub = run_validator._submission_json(
      task_specs.TASKS['port_recurrentgemma_2b'], sub, verdict)
  assert list(stub) == ['port_experiment_dir', 'port_accuracy_all',
                        'port_accuracy_avg', 'experiment_dir',
                        'eval_accuracy_all', 'eval_accuracy_avg', 'summary']
  assert stub['port_experiment_dir'] == port_dir
  assert stub['port_accuracy_avg'] == pytest.approx(0.152)


# --- golden artifacts from real runs (testdata/) ------------------------------


_TESTDATA = pathlib.Path(__file__).parent / 'testdata'


def _golden(task):
  """A real `final_result.json`; see testdata/README.md."""
  return json.loads((_TESTDATA / f'{task}.final_result.json').read_text())


# Every task whose scoring depends on what the loop stamps; the other two
# (sampling_lcb, decode_efficiency_vf) pin no protocol and derive nothing.
_GOLDEN_TASKS = ['pretrain_bpb_byte', 'pretrain_bpb_v32k',
                 'pretrain_optimizer_ttt', 'rl_gemma3_1b',
                 'rl_qwen2p5_math_1p5b', 'rl_bfcl_qwen3_0p6b',
                 'rl_bfcl_gemma3_1b', 'port_falcon_h1_0p5b',
                 'port_recurrentgemma_2b']


def _golden_sweep(tmp_path, task, mutate=lambda result, seed: result):
  """The golden result over the task's seeds, as an experiment dir."""
  spec = task_specs.TASKS[task]
  results = {}
  for seed in spec.seeds:
    result = _golden(task)
    result['run_provenance'] = {**result['run_provenance'], 'model_seed': seed}
    results[seed] = mutate(result, seed)
  experiment_dir = _experiment(tmp_path, task, results)
  if spec.needs_port_run:  # a stage-1 port at exactly the advisory reference
    for seed in spec.seeds:
      port = _golden(task)
      port['run_provenance'] = {**port['run_provenance'], 'model_seed': seed}
      port['eval_accuracy'] = spec.port_reference
      port_dir = pathlib.Path(experiment_dir) / 'port_run' / f'seed_{seed}'
      _write_json(port_dir / 'final_result.json', port)
      (port_dir / 'log.txt').write_text(_CLEAN_LOG)
  return experiment_dir


@pytest.mark.parametrize('task', _GOLDEN_TASKS)
def test_a_real_run_satisfies_the_pinned_protocol_and_compute_checks(
    tmp_path, task):
  # The stamps in testdata/ come from the real loops, so this is the contract
  # between them and the specs, not a fixture I wrote.
  v = submission_lib.score_submission(task, _golden_sweep(tmp_path, task))
  assert v.valid, v.reasons
  assert v.warnings == []
  assert len(v.raw_values) == 3


@pytest.mark.parametrize('task', _GOLDEN_TASKS)
def test_every_pinned_protocol_key_exists_in_the_real_stamp(task):
  # A spec pinning a key the loop never emits would fail every submission with
  # a confusing `key=None (expected ...)`.
  stamp = _golden(task).get('eval_protocol', {})
  missing = [k for k in task_specs.TASKS[task].protocol if k not in stamp]
  assert not missing, f'{task}: {missing} not emitted by the loop'


def test_every_task_whose_scoring_reads_a_stamp_has_a_golden_fixture():
  needs_fixture = {name for name, spec in task_specs.TASKS.items()
                   if spec.protocol or spec.derived}
  assert needs_fixture == set(_GOLDEN_TASKS)


def test_the_real_ttt_curve_is_scored_without_crashing(tmp_path):
  # The smoke run's losses (~8.6) never reach the frozen targets (~4.9), so
  # every target contributes 0 -- a real, if terrible, result.
  exp = _golden_sweep(tmp_path, 'pretrain_optimizer_ttt')
  v = submission_lib.score_submission('pretrain_optimizer_ttt', exp)
  assert v.valid, v.reasons
  assert v.raw_mean == 0.0
  assert v.score == 0.0


def test_the_seed_is_recovered_from_a_real_experiment_config():
  # A crashed run has no run_provenance, but experiment_helper wrote the
  # config before training started.
  config = json.loads(
      (_TESTDATA / 'pretrain_bpb_byte.experiment_config.json').read_text())
  assert config['model_seed'] == 42
  result = _golden('pretrain_bpb_byte')
  del result['run_provenance']
  assert submission_lib._observed_seed(result, config) == 42
  assert submission_lib._observed_seed(result, None) is None


def test_a_diverged_run_fails_instead_of_scoring_zero(tmp_path):
  nan = float('nan')  # json round-trips NaN verbatim, as a real run would.
  bpb = _golden_sweep(tmp_path, 'pretrain_bpb_byte',
                      mutate=lambda r, s: {**r, 'validation_bpb': nan})
  v = submission_lib.score_submission('pretrain_bpb_byte', bpb)
  assert not v.valid
  assert 'not a finite number' in ' '.join(v.reasons)
  # The ttt curve is the other way a run can diverge: an all-NaN curve would
  # otherwise look like a run that merely never reached any target.
  curve = [[0, 11.9], [5, nan], [10, nan]]
  with pytest.raises(ValueError, match='non-finite'):
    task_specs.ttt_speedup({'validation_loss_curve': curve})


# --- the two eval-only tasks: pinned on the artifact + the launch ------------


_LCB_CMD = ('python -m tasks.research_bench.eval_main '
            '--experiment_config=qwen3_4b --lm_format=QwenV2Chat '
            '--evaluation=LcbBaseline '
            '--datasource_name=simply_json:livecodebench_v5 '
            '--temperature=0.6 --top_p=0.95 --top_k=20 --batch_size=48 '
            '--n_repeats=1 --max_seq_len=12000 --mesh_shape=1,1,4 '
            '--num_eval_threads=96 --seed={seed} '
            '--experiment_dir={run_dir}')

_DECODE_CMD = ('python -m tasks.research_bench.eval_main '
               '--experiment_config=qwen3_30b_a3b_thinking_2507 '
               '--lm_format=QwQChat '
               '--evaluation=ZeroShotDeepSeekQwenR1CoTBoxed '
               '--datasource_name=simply:aime25 --n_repeats=4 '
               '--mesh_shape=1,1,8 --seed={seed} '
               '--experiment_dir={run_dir}')


def _eval_only(tmp_path, task, results, command, tpu_type, **manifest):
  """An eval-only task's experiment dir, with an A3-shaped launch manifest."""
  seeds = task_specs.TASKS[task].seeds
  root = tmp_path / 'exp'
  return _experiment(
      tmp_path, task, results,
      manifest={'experiment_config':
                    task_specs.TASKS[task].launch_fields['experiment_config'],
                'tpu_type': tpu_type,
                'per_seed_command': {
                    str(s): command.format(seed=s, run_dir=f'{root}/seed_{s}')
                    for s in seeds},
                **manifest})


def _lcb_result(accuracy, seed, total=167):
  return {'accuracy': accuracy, 'correct': round(accuracy * total),
          'total': total, 'avg_generation_time': 12.3, 'seed': seed}


def test_a_clean_sampling_lcb_submission_scores(tmp_path):
  exp = _eval_only(tmp_path, 'sampling_lcb',
                   {s: _lcb_result(0.55, s) for s in (42, 43, 44)},
                   _LCB_CMD, 'v6e-4')
  v = submission_lib.score_submission('sampling_lcb', exp)
  assert v.valid, v.reasons
  assert v.warnings == []
  assert v.raw_mean == pytest.approx(0.55)


def test_a_sharded_lcb_run_is_rejected_by_the_total(tmp_path):
  # --data_shard_count is legitimate for smoke runs and shows up as a short
  # problem set; the scored submission must be the full 167.
  exp = _eval_only(tmp_path, 'sampling_lcb',
                   {s: _lcb_result(0.9, s, total=9) for s in (42, 43, 44)},
                   _LCB_CMD, 'v6e-4')
  v = submission_lib.score_submission('sampling_lcb', exp)
  assert not v.valid
  assert 'total=9 (expected 167)' in ' '.join(v.reasons)


def test_a_relaunched_lcb_model_or_data_is_rejected(tmp_path):
  for flag, replacement in (('--lm_format=QwenV2Chat', '--lm_format=QwQChat'),
                            ('--experiment_config=qwen3_4b',
                             '--experiment_config=qwen3_30b_a3b_thinking_2507'),
                            ('--n_repeats=1', '--n_repeats=4')):
    exp = _eval_only(tmp_path, 'sampling_lcb',
                     {s: _lcb_result(0.9, s) for s in (42, 43, 44)},
                     _LCB_CMD.replace(flag, replacement), 'v6e-4',
                     name=f'exp_{flag[2:6]}')
    v = submission_lib.score_submission('sampling_lcb', exp)
    assert not v.valid, flag
    assert 'launch_manifest.json: ' in ' '.join(v.reasons)


def test_the_lcb_decoder_itself_is_not_pinned(tmp_path):
  # `--evaluation` IS the research surface for sampling_lcb.
  exp = _eval_only(tmp_path, 'sampling_lcb',
                   {s: _lcb_result(0.62, s) for s in (42, 43, 44)},
                   _LCB_CMD.replace('--evaluation=LcbBaseline',
                                    '--evaluation=MyCleverLcbEval'), 'v6e-4')
  assert submission_lib.score_submission('sampling_lcb', exp).valid


def _decode_result(seconds, accuracy=0.80, total=120):
  return {'accuracy': accuracy, 'correct': round(accuracy * total),
          'total': total, 'avg_generation_time': seconds, 'seed': 42}


def test_a_clean_decode_efficiency_submission_scores(tmp_path):
  exp = _eval_only(tmp_path, 'decode_efficiency_vf',
                   {42: _decode_result(6.2658)}, _DECODE_CMD, 'v6e-8')
  v = submission_lib.score_submission('decode_efficiency_vf', exp)
  assert v.valid, v.reasons
  assert v.score == pytest.approx(0.900, abs=5e-4)


def test_a_decode_run_on_another_accelerator_is_rejected(tmp_path):
  # avg_generation_time is a throughput number: another slice is not a score.
  exp = _eval_only(tmp_path, 'decode_efficiency_vf', {42: _decode_result(3.0)},
                   _DECODE_CMD.replace('--mesh_shape=1,1,8',
                                       '--mesh_shape=1,1,4'), 'v6e-4')
  v = submission_lib.score_submission('decode_efficiency_vf', exp)
  assert not v.valid
  assert "tpu_type=['v6e-4'] (expected 'v6e-8')" in ' '.join(v.reasons)


def test_a_decode_run_with_fewer_repeats_is_rejected(tmp_path):
  exp = _eval_only(tmp_path, 'decode_efficiency_vf',
                   {42: _decode_result(4.0, total=30)},
                   _DECODE_CMD.replace('--n_repeats=4', '--n_repeats=1'),
                   'v6e-8')
  v = submission_lib.score_submission('decode_efficiency_vf', exp)
  assert not v.valid
  # The launch is checked first and short-circuits, so the mismatched `total`
  # (30 instead of 30x4) is not reported here -- see the sharded-lcb test.
  assert "n_repeats=['1'] (expected '4')" in ' '.join(v.reasons)


def test_an_unverifiable_launch_warns_rather_than_failing(tmp_path):
  # A hand-launched run (no per_seed_command) still scores, with a caveat.
  exp = _experiment(tmp_path, 'sampling_lcb',
                    {s: _lcb_result(0.55, s) for s in (42, 43, 44)})
  v = submission_lib.score_submission('sampling_lcb', exp)
  assert v.valid, v.reasons
  assert 'records nothing for' in ' '.join(v.warnings)


def test_command_flag_parsing_handles_the_shapes_a_launcher_emits():
  flags = submission_lib._command_flags(
      "python -m x --a=1 --b 2 --mesh_shape=1,1,8 --flag --noother "
      "--quoted='a b' -v")
  assert flags == {'a': '1', 'b': '2', 'mesh_shape': '1,1,8',
                   'flag': 'true', 'other': 'false', 'quoted': 'a b'}


# --- a checkpoint that only partly mapped onto the model ---------------------


def _with_log(experiment_dir, text, seeds=(42, 43, 44)):
  for seed in seeds:
    (pathlib.Path(experiment_dir) / f'seed_{seed}' / 'log.txt').write_text(text)
  return experiment_dir


def test_a_partly_loaded_pinned_checkpoint_fails_the_rl_tasks(tmp_path):
  exp = _with_log(_golden_sweep(tmp_path, 'rl_qwen2p5_math_1p5b'),
                  _MISMATCH_LOG)
  v = submission_lib.score_submission('rl_qwen2p5_math_1p5b', exp)
  assert not v.valid
  joined = ' '.join(v.reasons)
  assert 'the pinned base checkpoint did not load completely' in joined
  assert '112 of 424 parameter branches were re-initialised' in joined
  assert 'did not start from the model the task fixes' in joined


def test_a_partly_loaded_checkpoint_only_warns_on_the_porting_tasks(tmp_path):
  # The model class is the agent's own, so this is a defect in their port: a
  # real, if poor, result -- the same reasoning as the port_run reference.
  exp = _with_log(_golden_sweep(tmp_path, 'port_recurrentgemma_2b'),
                  _MISMATCH_LOG)
  v = submission_lib.score_submission('port_recurrentgemma_2b', exp)
  assert v.valid, v.reasons
  joined = ' '.join(v.warnings)
  assert 'your checkpoint format mapped only part of the model' in joined
  assert '112 of 424 parameter branches' in joined


def test_a_clean_log_says_nothing(tmp_path):
  exp = _with_log(_golden_sweep(tmp_path, 'rl_bfcl_qwen3_0p6b'), _CLEAN_LOG)
  v = submission_lib.score_submission('rl_bfcl_qwen3_0p6b', exp)
  assert v.valid, v.reasons
  assert v.warnings == []


def test_a_missing_log_warns_but_never_fails(tmp_path):
  exp = _golden_sweep(tmp_path, 'rl_gemma3_1b')
  for seed in (42, 43, 44):
    (pathlib.Path(exp) / f'seed_{seed}' / 'log.txt').unlink()
  v = submission_lib.score_submission('rl_gemma3_1b', exp)
  assert v.valid, v.reasons
  assert 'the checkpoint load cannot be checked' in ' '.join(v.warnings)


def test_tasks_that_load_no_checkpoint_do_not_read_the_log(tmp_path):
  # Nothing in the pretrain tasks loads a checkpoint, so the same line in the
  # log is none of the validator's business.
  exp = _with_log(_golden_sweep(tmp_path, 'pretrain_bpb_byte'), _MISMATCH_LOG)
  v = submission_lib.score_submission('pretrain_bpb_byte', exp)
  assert v.valid, v.reasons
  assert v.warnings == []


def test_the_checkpoint_policy_is_stated_once_per_task():
  policy = {name: spec.checkpoint_policy()
            for name, spec in task_specs.TASKS.items()}
  assert {n for n, p in policy.items() if p == 'fail'} == {
      'rl_gemma3_1b', 'rl_qwen2p5_math_1p5b', 'rl_bfcl_qwen3_0p6b',
      'rl_bfcl_gemma3_1b'}
  assert {n for n, p in policy.items() if p == 'warn'} == {
      'port_falcon_h1_0p5b', 'port_recurrentgemma_2b'}

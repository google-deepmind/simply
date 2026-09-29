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

"""What each research_bench task measures, and what makes a run valid.

The validators are lightweight checker and parser. LLM-agent based cheating
detection lives elsewhere.

Scoring is affine in the raw metric, pinned by two per-task constants: `a` is
the raw value scoring 0, `b` the raw value scoring 1, and scores are capped at
`1.2` (0.2 past `b`).

    score = clamp((a - raw) / (a - b), 0.0, 1.2)    for LOWER-is-better tasks
    score = clamp((raw - a) / (b - a), 0.0, 1.2)    for HIGHER-is-better tasks

A run better than `b` can score up to `1.2`, preventing extreme outliers on a
single task from dominating cross-task score averages.

The anchors, seeds, protocol stamps and caps below are copied verbatim from the
internal suite: they were measured on internal baselines and must not be
re-derived here. Where the open-source port substitutes an asset that would
move a pinned number, the number is left untouched and flagged with a `TODO`
constant (see `PRETRAIN_V32K_VOCAB_SIZE` and `TTT_TARGETS_VOCAB`).
"""

import dataclasses
import math
from typing import Any, Callable, Mapping, Sequence

SCORE_CAP: float = 1.2


@dataclasses.dataclass(frozen=True)
class TaskSpec:
  """Everything the validator needs for one task.

  Attributes:
    name: The task id, matching the experiment config and the task md.
    metric: Key in final_result.json holding the scored value.
    lower_is_better: Direction of the metric.
    a: Raw value scoring 0.
    b: Raw value scoring 1.
    seeds: Sweep seeds the submission must contain, one run each.
    protocol: eval_protocol fields that must match exactly, when the task's
      loop emits a stamp. Empty for the two eval-only tasks, which have none.
    result_fields: final_result.json fields that must match exactly. The
      eval-only tasks emit no eval_protocol stamp, so their fixed eval extent
      is checked on the artifact itself (`total`), which also catches a
      sharded or sliced run.
    launch_fields: launch_manifest.json fields that must match exactly,
      including flags read out of the per-seed command it records. The
      eval-only tasks' protocol is set by command-line flags rather than by a
      registered config, so this is where it is pinned.
    flops_cap: Training-compute cap, or 0.0 when the task has none.
    accuracy_gate: (key, minimum) the run must satisfy, or None.
    fixed_init_ckpt: True when the task pins the base checkpoint the run
      starts from. `simply.model_lib` re-initialises missing parameter
      branches on a structural mismatch and carries on, so a half-mapped
      checkpoint trains a partly random model and reports a plausible bad
      metric; where the checkpoint is pinned that is a failed run, not a weak
      recipe. See `TaskSpec.checkpoint_policy`.
    needs_port_run: True for the porting tasks, where a separate 0-step
      `port_run` must reproduce the reference before the candidate is scored.
    port_reference: Accuracy a faithful stage-1 port is expected to reach.
      ADVISORY: falling short is reported as a warning, not a failure -- a
      partial port is a real if poor result, and the metric already prices it.
    derived: Optional callable computing the metric from the artifact, for
      tasks whose scored number is not written out directly.
  """

  name: str
  metric: str
  lower_is_better: bool
  a: float
  b: float
  seeds: tuple[int, ...] = (42, 43, 44)
  protocol: Mapping[str, object] = dataclasses.field(default_factory=dict)
  result_fields: Mapping[str, object] = dataclasses.field(default_factory=dict)
  launch_fields: Mapping[str, object] = dataclasses.field(default_factory=dict)
  flops_cap: float = 0.0
  accuracy_gate: tuple[str, float] | None = None
  fixed_init_ckpt: bool = False
  needs_port_run: bool = False
  port_reference: float = 0.0
  derived: Callable[[Mapping[str, Any]], float] | None = None

  def score(self, raw: float) -> float:
    """Normalized score for a raw value, clamped to [0.0, SCORE_CAP]."""
    num = (self.a - raw) if self.lower_is_better else (raw - self.a)
    den = (self.a - self.b) if self.lower_is_better else (self.b - self.a)
    return min(SCORE_CAP, max(0.0, num / den))

  def checkpoint_policy(self) -> str:
    """What a partly loaded checkpoint means for this task.

    Returns:
      'fail' where the base checkpoint is pinned (the run did not start from
      the model the task fixes), 'warn' for the porting tasks (the model class
      is the agent's own, so a mismatch is a defect in their port and a partial
      port is an admissible, poor result), '' where nothing is loaded.
    """
    if self.fixed_init_ckpt:
      return 'fail'
    return 'warn' if self.needs_port_run else ''


# --- pretrain_optimizer_ttt: the one task whose metric is derived -------------
# Four loss targets frozen from a weak-Adam anchor curve, at 40/60/80/100% of
# the 1200-step budget. Per target the speedup is anchor_step / first crossing
# of that loss (linear interpolation on the run's validation_loss_curve); a
# target never reached contributes 0. The metric is the mean over the four.
#
# HOW TO RE-DERIVE (needed whenever the vocab or the anchor config changes --
# see TTT_TARGETS_VOCAB; keep TTT_TARGETS, `a` and `b` in one edit):
#   1. Run `config_lib.pretrain_optimizer_ttt_anchor` (the weak-Adam anchor:
#      the fixed 41M/C4/1200-step setup on the untuned optimizer that core's
#      41M recipe inherits) on seeds 42/43/44.
#   2. Take each run's `validation_loss_curve` ([[step, loss], ...]) and average
#      the three seeds' held-out C4 validation loss per step.
#   3. TTT_TARGETS = that mean curve's loss at 40/60/80/100% of 1200 steps,
#      i.e. at steps 480/720/960/1200, paired with those steps: a candidate
#      that reaches target i at step s scores anchor_step_i / s on it. The
#      anchor itself then scores ~1.0 by construction.
#   4. Re-anchor the score. Run the TUNED baseline `pretrain_optimizer_ttt` on
#      the same seeds and read its mean `ttt_speedup` (`base`). Absent a new
#      pilot round, preserve the internal calibration's ratios:
#      a = 1.070 * base, b = 2.028 * a  (internally base=1.277, a=1.3662,
#      b=2.7709). See ANCHORS.md.
TTT_TARGETS: Sequence[tuple[float, int]] = (
    (5.4272, 480), (4.9969, 720), (4.7037, 960), (4.5901, 1200),
)

# The targets above are raw validation LOSSES, so they only mean anything under
# the vocabulary they were measured with. They were re-derived for this port on
# 2026-09-29 by the recipe above: `pretrain_optimizer_ttt_anchor` on seeds
# 42/43/44, v6e-4, with `config_lib.TTT_VOCAB_NAME` = the 100864-piece C4 SPM
# (`gs://.../baseline_ttt_anchor`, see ../BASELINES.md). The tuned baseline
# then measured base=1.5746, giving a = 1.070 * base = 1.6848 and
# b = 2.028 * a = 3.4168. Re-derive all six numbers together if the tokenizer,
# the anchor config or the C4 build changes.
TTT_TARGETS_VOCAB: str = 'c4_spm100864'

# TODO(port): `decode_efficiency_vf`'s metric is wall-clock seconds, so its
# anchors belong to the accelerator they were measured on -- the internal
# bundle DROPPED the task when it homogenized hardware, for exactly that
# reason. This port pins the task to `v6e-8`, not the topology below, so
# a=16.2718 / b=5.1540 are STALE: re-measure the unchanged decode pipeline on
# `v6e-8` (seed 42) for a new reference `base` and set a = 0.884 * base,
# b = 0.280 * base (the internal calibration's ratios, base=18.4s). The
# accuracy gate is hardware-independent and stands. See ANCHORS.md.
DECODE_ANCHOR_HARDWARE: str = 'internal 8-chip reference slice'

# The internal 32k vocab `nanodo_c4` (cc_all.32000.100extra.bos.model) is the
# public gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model plus one
# `<s>` control piece at index 2 (bos_id=2), which `setup/prepare_assets.py`
# reproduces bit-exactly -- hence 32101 pieces here and in the config.
PRETRAIN_V32K_VOCAB_SIZE: int = 32101


def ttt_speedup(result: Mapping[str, Any]) -> float:
  """Time-to-target speedup from a run's validation_loss_curve."""
  curve: Sequence[Sequence[float]] = result.get('validation_loss_curve') or []
  if len(curve) < 2:
    raise ValueError('no usable validation_loss_curve in final_result.json')
  # A diverged run (NaN losses) would otherwise silently "never reach" every
  # target and score 0 as if it had merely been slow.
  if any(not math.isfinite(value) for point in curve for value in point):
    raise ValueError('validation_loss_curve contains non-finite values: the '
                     'run diverged')
  speedups = []
  for target, anchor_step in TTT_TARGETS:
    crossing = None
    for (s0, l0), (s1, l1) in zip(curve, curve[1:]):
      if l1 <= target:
        crossing = (s1 if l0 <= target
                    else s0 + (s1 - s0) * (l0 - target) / (l0 - l1))
        break
    speedups.append(anchor_step / crossing if crossing else 0.0)
  return sum(speedups) / len(speedups)


_GSM8K = {'eval_source': 'simply:gsm8k_test', 'n_scored': 1319}

TASKS: Mapping[str, TaskSpec] = {t.name: t for t in (
    TaskSpec(
        name='pretrain_bpb_v32k', metric='validation_bpb', lower_is_better=True,
        a=1.3790, b=1.2296, flops_cap=1.811e16,
        protocol={'vocab_size': PRETRAIN_V32K_VOCAB_SIZE},
    ),
    TaskSpec(
        name='pretrain_bpb_byte', metric='validation_bpb', lower_is_better=True,
        a=1.7519, b=1.2607, flops_cap=9.180e15,
        protocol={'vocab_size': 259},
    ),
    TaskSpec(
        name='pretrain_optimizer_ttt', metric='time_to_target_speedup',
        lower_is_better=False, a=1.6848, b=3.4168, derived=ttt_speedup,
    ),
    TaskSpec(
        name='rl_gemma3_1b', metric='eval_accuracy', lower_is_better=False,
        a=0.0572, b=0.3015, fixed_init_ckpt=True,
        protocol={**_GSM8K, 'eval_temperature': 0.0,
                  'vocab_name': 'vb262144_gemma3'},
    ),
    TaskSpec(
        name='rl_qwen2p5_math_1p5b', metric='eval_accuracy',
        lower_is_better=False,
        a=0.4273, b=0.5780, fixed_init_ckpt=True,
        protocol={'eval_source': 'simply:math500_test_l45', 'n_scored': 262,
                  'eval_temperature': 0.6, 'eval_num_samples': 8,
                  'vocab_name': 'Qwen2.5'},
    ),
    TaskSpec(
        name='rl_bfcl_qwen3_0p6b', metric='eval_accuracy',
        lower_is_better=False,
        a=0.2607, b=0.5174, fixed_init_ckpt=True,
        protocol={'eval_source': 'simply:bfcl_live_eval', 'n_scored': 1351,
                  'evaluation': 'BFCLFunctionCallEvaluation',
                  'vocab_name': 'Qwen3'},
    ),
    TaskSpec(
        name='rl_bfcl_gemma3_1b', metric='eval_accuracy', lower_is_better=False,
        a=0.0759, b=0.4839, fixed_init_ckpt=True,
        protocol={'eval_source': 'simply:bfcl_live_eval', 'n_scored': 1351,
                  'evaluation': 'BFCLFunctionCallEvaluation'},
    ),
    # The two eval-only tasks: a different entry point writes these artifacts
    # (eval_main.py -> the unchanged page_decode_eval) and it emits no
    # eval_protocol stamp, so their fixed setup is pinned on the artifact
    # (`total`) and on the launch manifest instead.
    TaskSpec(
        name='sampling_lcb', metric='accuracy', lower_is_better=False,
        a=0.3896, b=0.7056,
        # `evaluation` is deliberately absent: the decoder IS the research
        # surface here. A sharded run (--data_shard_count) shows up as a
        # `total` below 167.
        result_fields={'total': 167},
        launch_fields={'experiment_config': 'qwen3_4b',
                       'lm_format': 'QwenV2Chat',
                       'datasource_name': 'simply_json:livecodebench_v5',
                       'n_repeats': 1},
    ),
    TaskSpec(
        name='decode_efficiency_vf', metric='avg_generation_time',
        lower_is_better=True, a=16.2718, b=5.1540,
        seeds=(42,), accuracy_gate=('accuracy', 0.75),
        # 30 AIME-2025 problems x n_repeats=4; the whole decode pipeline is
        # the research surface, so the eval, the model and the SLICE are
        # pinned -- a throughput number from another accelerator is not
        # comparable (see DECODE_ANCHOR_HARDWARE).
        result_fields={'total': 120},
        launch_fields={'experiment_config': 'qwen3_30b_a3b_thinking_2507',
                       'lm_format': 'QwQChat',
                       'evaluation': 'ZeroShotDeepSeekQwenR1CoTBoxed',
                       'datasource_name': 'simply:aime25',
                       'n_repeats': 4,
                       'tpu_type': 'v6e-8'},
    ),
    TaskSpec(
        name='port_falcon_h1_0p5b', metric='eval_accuracy',
        lower_is_better=False,
        a=0.4317, b=0.9584, needs_port_run=True, port_reference=0.60,
        protocol={**_GSM8K, 'eval_temperature': 0.0},
    ),
    TaskSpec(
        name='port_recurrentgemma_2b', metric='eval_accuracy',
        lower_is_better=False,
        a=0.00, b=0.5423, needs_port_run=True, port_reference=0.12,
        protocol={**_GSM8K, 'eval_temperature': 0.4},
    ),
)}

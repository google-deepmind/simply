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

"""research-bench task baseline configs."""

import dataclasses
import os
from typing import ClassVar

from simply import config_lib as core
from simply.utils import evaluation_lib
from simply.utils import optimizers as opt_lib
from tasks.research_bench import data_lib
from tasks.research_bench import port_eval_lib
from tasks.research_bench import tool_use_eval


class ExperimentConfigRegistry(core.ExperimentConfigRegistry):
  """Benchmark configs live in their OWN namespace, not core simply's.

  Core's registry is a single global keyspace, so a config of the same name here
  and in core `config_lib.py` would compete for one key and the core definition
  would win -- taking the run onto core's default `train_loop_name='rl'`, which
  has no full-coverage assert on the held-out eval and writes no `eval_protocol`
  stamp. A private namespace makes the collision impossible: `main.py` resolves
  `--experiment_config` here, so it can only get a benchmark config, and a name
  defined only in core is not found and fails immediately.

  Scope: ONLY experiment configs are namespaced. Train loops, RL algorithms,
  model modules, checkpoint formats and sharding configs stay on the shared core
  registries by design -- e.g. core `create_model` has to be able to find the
  PORT model classes registered from this package.
  """

  namespace: ClassVar[str] = 'ExperimentV0p3'


ShardingConfigRegistry = core.ShardingConfigRegistry


@dataclasses.dataclass(frozen=True)
class FixedComputeConfig(core.BaseExperimentConfig):
  """Pretraining config that declares a FIXED training-compute (FLOP) cap.

  `fixed_compute_cap_flops` is the marker `main.py` keys the compute-integrity
  guards on (see `main._resolve_train_loop`): any run whose config carries a
  positive cap gets the scan/custom-kernel checks AND a `compute_integrity`
  block in `final_result.json`, no matter which train loop the recipe selects.
  The value is the cap from the task spec (1.5x the baseline's measured
  `training_flops_xla`), recorded in the run's own result so the number the run
  reports can be checked against the budget it was run under.
  """

  # <= 0 means "no fixed-compute cap" (the default for every non-bpb task).
  fixed_compute_cap_flops: float = 0.0


# Fixed training-compute caps (= 1.5x the measured baseline `training_flops_xla`
# of each variant); these are the numbers quoted in the task descriptions.
_BPB_V32K_FLOPS_CAP = 1.811e16
_BPB_BYTE_FLOPS_CAP = 9.180e15

# The C4 (en) pretraining stream. Core's `TFDSSource(name='c4:3.1.0')` has no
# public mirror, so every pretraining config reads the repacked shards instead
# (`data_lib.C4FileSource`, built by setup/build_c4.py; `SIMPLY_C4_DIR` points
# at a local directory or a `gs://` prefix). `validation` is the FIXED eval
# split.
C4_TRAIN = data_lib.DatasetConfig(
    source=data_lib.C4FileSource(split='train'),
    lm_format_name='Pretrain',
)
C4_VALIDATION = data_lib.DatasetConfig(
    source=data_lib.C4FileSource(split='validation'),
    lm_format_name='Pretrain',
)

# Tokenizer for `pretrain_optimizer_ttt`. The internal baseline used
# `vb100864_openmix_v1`, which has no public twin; the port trains a
# 100864-piece SPM on the public C4 instead (setup/prepare_assets.py), so the
# model shape -- and therefore the FLOPs profile the task measures -- is
# unchanged and no embedding row is unreachable. The loss scale still moves
# with the tokenizer, which is why the task's loss targets are re-derived from
# `pretrain_optimizer_ttt_anchor`. See PORTING_NOTES.md and ASSETS.md.
TTT_VOCAB_NAME = 'c4_spm100864'


def _research_bench_base():
  """Shared pretraining base: 15M/C4/2e16 config w/ checkpointing disabled.

  Checkpoints are disabled (`should_save_ckpt=False`) because the task reads its
  metric (val_bpb) only at the end of a short (~1717-step) run and never
  reloads. Disabling them cuts checkpoint I/O overhead.

  `use_scan=False` unrolls the transformer layer stack (instead of a
  `jax.lax.scan` over layers), which the task spec requires: XLA's HLO
  `cost_analysis` counts a `scan` body only once, so a scanned stack would
  report the FLOPs of a single layer regardless of depth. Unrolled, every layer
  is counted in `training_flops_xla`. At this scale (<=~40 layers,
  model_dim<=~256) the unrolled graph compiles and fits without issue.

  Returns:
    The base config with checkpointing disabled and layers unrolled.
  """
  return dataclasses.replace(
      FixedComputeConfig().override_from(core.flops2e16_tfm15m_c4_l2048()),
      should_save_ckpt=False,
      use_scan=False,
      # The `research_bench_pretrain` loop is the core train loop plus the fixed-compute
      # checks (see research_bench/model_lib.py); those checks follow
      # `fixed_compute_cap_flops` rather than this loop name.
      train_loop_name='research_bench_pretrain',
      dataset=C4_TRAIN,
      validation_datasets=(C4_VALIDATION,),
  )


@ExperimentConfigRegistry.register
def pretrain_bpb_v32k():
  """pretraining baseline: 15M / C4 / seq2048 with the 32k nanodo_c4 vocab.

  nanodo_c4 = cc_all.32000.100extra.bos.model; GetPieceSize()==32101.  Note the
  actual total params is ~6.0M with the 32k vocab. This port regenerates that
  model bit-exactly from the public T5 `cc_all.32000.100extra` SPM (see
  data_lib.NANODO_C4_VOCAB), so the tokenization is the internal one.
  """
  return dataclasses.replace(
      _research_bench_base(),
      vocab_name='nanodo_c4',
      vocab_size=32_101,
      fixed_compute_cap_flops=_BPB_V32K_FLOPS_CAP,
  )


@ExperimentConfigRegistry.register
def pretrain_bpb_byte():
  """pretraining baseline: 15M / C4 / seq2048 with a raw utf-8 byte vocab.

  byte256 = tokenization.ByteVocab; vocab_size==259. Note the actual total
  params is ~1.9M with the byte vocab.
  """
  return dataclasses.replace(
      _research_bench_base(),
      vocab_name='byte256',
      vocab_size=259,
      fixed_compute_cap_flops=_BPB_BYTE_FLOPS_CAP,
  )


def _ttt_base():
  """The FIXED half of `pretrain_optimizer_ttt`: model, data, H=1200, vocab.

  Everything a submission may NOT change (41M model, C4 train/validation, 1200
  steps, batch 80, seq 2048, vocab) lives here, so the tuned baseline and the
  weak-Adam anchor below differ only in their optimizer/schedule.

  Returns:
    The shared `pretrain_optimizer_ttt` base config on the `research_bench_ttt` loop.
  """
  return dataclasses.replace(
      core.flops2e17_tfm41m_c4_l2048(),
      should_save_ckpt=False,
      num_train_steps=1200,
      validation_eval_interval=120,
      train_loop_name='research_bench_ttt',
      vocab_name=TTT_VOCAB_NAME,
      dataset=C4_TRAIN,
      validation_datasets=(C4_VALIDATION,),
  )


@ExperimentConfigRegistry.register
def pretrain_optimizer_ttt():
  """pretraining-optimizer baseline: 41M / C4 / 100k vocab, H=1200, WSD."""
  base = dataclasses.replace(
      _ttt_base(),
      lr=opt_lib.LinearWarmupCosineDecay(
          value=0.013,
          warmup_fraction=0.1,
          decay_start_fraction=0.8,
          end_decay=0.0,
      ),
      optimizer=opt_lib.Adam(beta1=0.9, beta2=0.95, epsilon=1e-8),
      weight_decay=0.1,
  )
  return base


@ExperimentConfigRegistry.register
def pretrain_optimizer_ttt_anchor():
  """The weak-Adam ANCHOR whose val-loss curve defines the TTT loss targets.

  `time_to_target_speedup` is `anchor_step / first_crossing_step` averaged over
  four loss targets taken from this curve at 40/60/80/100% of the 1200-step
  budget, so the targets are only meaningful together with the anchor run that
  produced them. The internal anchor's hyperparameters were not shipped with
  the scaffolding, so the port defines the anchor as the UNTUNED optimizer the
  41M recipe inherits from core (`flops2e17_tfm41m_c4_l2048`: plain Adam, a
  ~7x smaller LR fitted for a 4140-step horizon, cosine decay to 10%) run over
  the task's fixed 1200-step budget -- weak precisely because it is untuned for
  this horizon. Registering it by name keeps the targets reproducible: re-run
  this config, read `validation_loss_curve` at steps 480/720/960/1200, and
  those four losses ARE the targets (validator `TTT_TARGETS`).

  Returns:
    The anchor config (identical to the baseline except for the optimizer).
  """
  return _ttt_base()


# ------------------------------------------------------------------------------
# Research-bench RL scaffold (task-agnostic).
#
# A single explicit, modular RL loop (`research_bench_rl`, rl_loop.py) + an
# RLAlgorithm registry (rl_algorithms.py) drives EVERY research-bench RL task
# (function-calling, math, ...). `RLConfig` adds fields the loop reads on top
# of the standard RLExperimentConfig; `_apply_rl` wires the loop + generic
# RL/eval defaults. Each task supplies its base model, data, training reward +
# FIXED held-out eval, and sizing on top (see the per-task configs below).
# ------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RLConfig(core.RLExperimentConfig):
  """Config for research-bench RL tasks (read by the `research_bench_rl` loop).

  Adds `rl_algorithm` (the registered RLAlgorithm name) so the algorithm is
  selected without any core simply change, plus FIXED held-out eval decoding
  fields that are decoupled from the training sampler. Everything else is the
  standard RLExperimentConfig.
  """

  rl_algorithm: str = 'simple_grpo'
  # Held-out eval input budget (decoupled from training, identical across all
  # configs incl. `_pf`) so long tool-schema prompts aren't front-truncated and
  # the scored accuracy is comparable. 0 => fall back to sampling_max_input_len.
  eval_max_input_len: int = 0
  eval_prefill_size: int = 0
  # Held-out eval decoding is FIXED and separate from the training sampler (its
  # own temperature / #samples / decode budget), so the scored metric does not
  # move when the training sampler is retuned. Defaults: greedy (temp 0), pass@1
  eval_temperature: float = 0.0  # 0 => greedy (argmax); NOT the training temp
  eval_num_samples: int = 1  # samples/example averaged (avg@k; 1 => pass@1)
  eval_max_decode_steps: int = 160  # fixed eval decode budget (tokens)
  # Stop tokens for the held-out eval, separate from the training-side
  # `extra_eos_tokens` (which the recipe may retune along with the training
  # prompt format; `validation_lm_format_name` is the format counterpart).
  # Empty => fall back to `extra_eos_tokens`.
  eval_extra_eos_tokens: tuple[str, ...] = ()
  # Optional entropy-bonus coefficient read by entropy-regularized algorithms
  # (0 => off; the shipped baseline algorithms ignore it).
  entropy_coeff: float = 0.0
  # Decode buffer padding multiple (`SamplingParams.decode_buffer_multiple`)
  # for held-out evaluation. Avoids recompiling `decode_fn` for batches with
  # differing prompt lengths by padding buffers to a fixed grid.
  # Defaults to 0 (disabled).
  eval_decode_buffer_multiple: int = 0
  # Decode buffer padding multiple for rollout generation during training.
  # Defaults to 0 (disabled) to avoid subtle numerical differences from
  # padding changes affecting stochastic rollout trajectories.
  sampling_decode_buffer_multiple: int = 0
  # How the policy is moved to the decoding mesh every step -- the single
  # largest host-side item in an RL step. 'jit' | 'device_put' | 'per_array'
  # (the original); see `rl_loop._make_decode_resharder`.
  decode_reshard: str = 'jit'


def _apply_rl(config):
  """Task-agnostic research-bench RL setup: wire the explicit modular RL loop.

  Selects the self-contained `research_bench_rl` loop (rl_loop.py) + the
  RLAlgorithm registry (rl_algorithms.py) and sets the generic RL/eval defaults
  shared by every RL task. The task supplies its base model, data, training
  reward + FIXED held-out eval, sampling/eval lengths, batch sizes, optimizer
  on top of this.

  Args:
    config: the base experiment config to wire.

  Returns:
    The config with the research-bench RL loop + generic RL/eval defaults.
  """
  config = core.apply_simple_rl(config)
  return dataclasses.replace(
      config,
      # The self-contained explicit RL loop (NOT core's `rl` loop).
      train_loop_name='research_bench_rl',
      lm_format_name='Pretrain',
      # Held-out eval structure: full eval (-1 => iterate to exhaustion),
      # one pass. The loop reads every eval example directly (no dropped
      # remainder), so the scored metric covers the entire eval set.
      validation_num_eval_steps=-1,
      validation_eval_epochs=1,
      # On-policy: one optimizer update per sampled batch.
      num_train_steps_per_batch=1,
      max_num_samples_per_train_batch=None,
      init_ckpt_opt_state=False,
      # The research-bench baselines train without gradient/update clipping
      clip_grad_norm=-1.0,
      clip_update_norm=-1.0,
      clip_local_update_rms=-1.0,
  )


def _apply_bfcl_rl(config):
  """BFCL function-calling RL setup (data + verifiable reward) on top of _apply_rl."""
  config = _apply_rl(config)
  return dataclasses.replace(
      config,
      # Data + reward (training) and the FIXED held-out eval. The eval source
      # excludes the abstention category, so ~40% of the generation is skipped
      # and every scored example requires an actual call.
      dataset=data_lib.DatasetConfig(
          source='simply:bfcl_nonlive_train',
          packing=data_lib.PACKING_NONE,
          lm_format_name=None,
      ),
      evaluation=tool_use_eval.BFCLFunctionCallEvaluation(),
      validation_datasets=(
          data_lib.DatasetConfig(
              source='simply:bfcl_live_eval',
              packing=data_lib.PACKING_NONE,
              lm_format_name=None,
          ),
      ),
      validation_evaluation=tool_use_eval.BFCLFunctionCallEvaluation(),
      validation_eval_interval=20,
      validation_eval_batch_size=256,
      # Sampling: short outputs (a call list), moderate context (schemas).
      sampling_temperature=1.0,
      sampling_max_decode_steps=160,
      train_max_seq_len=2048,
      sampling_prefill_size=1536,
      sampling_max_input_len=1536,
      # Larger, config-independent eval input budget so long tool-schema prompts
      # are not front-truncated. The rendered eval prompts are 361 tokens
      # (median) / 1569 (max) over the 1351 scored examples, so nothing is
      # truncated at 3072.
      eval_max_input_len=3072,
      eval_prefill_size=3072,
      # Held-out eval decoding: FIXED (greedy pass@1, 160-token decode budget),
      # separate from the training sampler.
      eval_temperature=0.0,
      eval_num_samples=1,
      eval_max_decode_steps=160,
      sampling_intermediate_decode_steps=160,
      extra_eos_tokens=core.newlines_from_counts(range(2, 5)),
      # FIXED eval prompt format + stop tokens (the training-side
      # `lm_format_name` / `extra_eos_tokens` above are part of the recipe).
      validation_lm_format_name='Pretrain',
      eval_extra_eos_tokens=core.newlines_from_counts(range(2, 5)),
      # Batch / RL algorithm. Sized to fit a single 4-chip slice for 1.7-2.6B
      # models: grad-accum keeps the per-microbatch activation memory small, and
      # flash attention avoids the O(L^2) attention buffer.
      train_batch_size=16 * 8,
      batch_size=16,
      num_samples_per_example=8,
      grad_accum_steps=8,
      use_flash_attention=True,
      flash_attention_block_size=512,
      lr=opt_lib.LinearWarmupConstant(value=2e-6, warmup_steps=4),
      num_train_steps=200,
      ckpt_max_to_keep=1,
      tb_log_interval=2,
      ckpt_interval=50,
  )


def _bfcl_task(config):
  """Task-level settings shared by the tool_use baselines (100 steps).

  Checkpointing is ON (interval 20, keep 1) so a preempted run RESUMES from the
  latest ckpt instead of restarting from step 0 -- essential for the 3-seed
  sweep to finish on preemptible (spot) TPU VMs. The research_bench_rl loop is
  resume-aware (starts from the restored step count).

  Args:
    config: the RL config to specialize for the tool_use task.

  Returns:
    The config with tool_use task-level training/eval settings.
  """
  return dataclasses.replace(
      config,
      num_train_steps=100,
      should_save_ckpt=True,
      ckpt_interval=20,
      ckpt_max_to_keep=1,
      validation_eval_interval=10,
      # Full held-out eval: the 1351 call-required BFCL-live examples (the 882
      # `live_irrelevance` abstention examples are excluded by the eval data
      # source, so an "always abstain" policy scores 0). NOTE the loop iterates
      # the eval source directly and always covers it in full; this field is
      # kept only for parity with core's config surface.
      validation_num_eval_steps=-1,
  )


# Qwen3 -Base (true pretrained) checkpoints for the BFCL tool-use RL tasks. The
# bare Qwen3-0.6B/ORBAX is POST-TRAINED (instruct/chat) and truncates under the
# Pretrain format; the `-Base` variant is the genuine pretrained checkpoint
# (converted from `Qwen/Qwen3-0.6B-Base` by setup/setup_assets.py).


@ExperimentConfigRegistry.register
def rl_bfcl_qwen3_0p6b():
  """Naive-GRPO baseline (Qwen3-0.6B-Base PT) on the BFCL tool-use task.

  The scored reference (normalized score 0). Same self-contained
  `research_bench_rl` loop + FIXED call-required BFCL-live AST eval as the other
  tool-use tasks; only the base model differs.
  """
  base = dataclasses.replace(
      core.qwen3_0p6b(),
      init_ckpt_dir=os.path.join(core.MODELS_DIR, 'Qwen3-0.6B-Base/ORBAX'),
  )
  config = RLConfig().override_from(base)
  config = _apply_bfcl_rl(config)
  config = _bfcl_task(config)
  return dataclasses.replace(config, rl_algorithm='simple_grpo')


@ExperimentConfigRegistry.register
def rl_bfcl_gemma3_1b():
  """Naive-GRPO baseline (Gemma-3-1B PT) on the BFCL tool-use task (score 0)."""
  config = RLConfig().override_from(core.gemma3_1b())
  config = _apply_bfcl_rl(config)
  config = _bfcl_task(config)
  return dataclasses.replace(config, rl_algorithm='simple_grpo')


# ------------------------------------------------------------------------------
# Math-reasoning RL tasks (rl_gemma3_1b, rl_qwen2p5_math_1p5b).
#
# 0-shot \boxed{} answer RL: the SAME ZeroShotBoxedInQuestionEvaluation is both
# the training reward (on the train split) and the FIXED held-out eval (on the
# test split). The baseline is on-policy, R1-Zero-style naive GRPO (no KL / no
# reference policy).
# ------------------------------------------------------------------------------
def _apply_boxed_answer_rl(
    config,
    *,
    train_source,
    eval_source,
    train_max_seq_len,
    decode_budget,
    eval_temperature,
    eval_num_samples,
    eval_batch_size,
    use_flash_attention,
):
  """0-shot boxed-answer RL setup (on-policy naive GRPO) on top of _apply_rl.

  The boxed-answer Evaluation is used BOTH as the training reward (on
  `train_source`) and the FIXED held-out eval (on `eval_source`). On-policy:
  batch_size(16) x num_samples(8) = 128 rollouts/round -> one full-batch update
  (grad-accum keeps per-microbatch memory small). Each task passes its data,
  sequence/decode lengths, eval sampling (greedy pass@1 vs avg@k), eval batch,
  and flash-attention setting; the model/vocab/dtype come from the base config.

  Args:
    config: the base experiment config (supplies model/vocab/dtype).
    train_source: training data source name.
    eval_source: held-out eval data source name.
    train_max_seq_len: max training sequence length.
    decode_budget: max decode steps for sampling and eval.
    eval_temperature: eval decoding temperature (0 => greedy).
    eval_num_samples: eval samples per example (avg@k; 1 => pass@1).
    eval_batch_size: held-out eval batch size.
    use_flash_attention: whether to enable flash attention.

  Returns:
    The config wired for 0-shot boxed-answer RL.
  """
  config = _apply_rl(config)
  input_len = 1024
  boxed = evaluation_lib.ZeroShotBoxedInQuestionEvaluation()
  return dataclasses.replace(
      config,
      dataset=data_lib.DatasetConfig(
          source=train_source,
          packing=data_lib.PACKING_NONE,
          lm_format_name=None,
      ),
      evaluation=boxed,
      validation_datasets=(
          data_lib.DatasetConfig(
              source=eval_source,
              packing=data_lib.PACKING_NONE,
              lm_format_name=None,
          ),
      ),
      validation_evaluation=boxed,
      validation_eval_interval=50,
      validation_eval_batch_size=eval_batch_size,
      # On-policy naive GRPO: 16 prompts x 8 samples = 128 rollouts/round, one
      # full-batch update (grad-accum microbatch = 128 / 8 = 16).
      batch_size=16,
      num_samples_per_example=8,
      train_batch_size=16 * 8,
      grad_accum_steps=8,
      # R1-Zero-style: no KL, no reference policy (a sane naive-GRPO floor).
      rl_algorithm='simple_grpo',
      use_ref_params=False,
      kl_coeff=0.0,
      # Training sampling (reasoning-length decode; the model stops on EOS).
      sampling_temperature=1.0,
      sampling_max_input_len=input_len,
      sampling_prefill_size=input_len,
      sampling_max_decode_steps=decode_budget,
      sampling_intermediate_decode_steps=min(decode_budget, 1024),
      train_max_seq_len=train_max_seq_len,
      # FIXED held-out eval decoding (decoupled from the training sampler).
      eval_max_input_len=input_len,
      eval_prefill_size=input_len,
      eval_max_decode_steps=decode_budget,
      eval_temperature=eval_temperature,
      eval_num_samples=eval_num_samples,
      use_flash_attention=use_flash_attention,
      flash_attention_block_size=512,
      extra_eos_tokens=core.newlines_from_counts(range(3, 6)),
      # FIXED eval prompt format + stop tokens (the training-side
      # `lm_format_name` / `extra_eos_tokens` above are part of the recipe).
      validation_lm_format_name='Pretrain',
      eval_extra_eos_tokens=core.newlines_from_counts(range(3, 6)),
      # Untuned baseline optimizer (participants tune / replace).
      lr=opt_lib.LinearWarmupConstant(value=1e-6, warmup_steps=4),
      weight_decay=0.0,
      num_train_steps=100,
      # Checkpoint on so a preempted 3-seed sweep resumes (eval interval divides
      # ckpt interval, as the loop asserts).
      should_save_ckpt=True,
      ckpt_interval=50,
      ckpt_max_to_keep=1,
      tb_log_interval=2,
  )


@ExperimentConfigRegistry.register
def rl_gemma3_1b():
  """Baseline: Gemma-3-1B (PT) naive GRPO on GSM8K, 0-shot boxed full-test eval.

  From the PRETRAINED Gemma-3-1B checkpoint. The PT model rarely boxes answers,
  so naive RL gets ~no reward signal and the held-out eval stays ~0. bf16;
  greedy pass@1 eval on the full GSM8K test.
  """
  config = RLConfig().override_from(core.gemma3_1b())
  return _apply_boxed_answer_rl(
      config,
      train_source='simply:gsm8k_train',
      eval_source='simply:gsm8k_test',
      train_max_seq_len=2048,
      decode_budget=1024,
      eval_temperature=0.0,  # greedy
      eval_num_samples=1,  # pass@1
      eval_batch_size=128,
      use_flash_attention=False,
  )


@ExperimentConfigRegistry.register
def rl_qwen2p5_math_1p5b():
  """Baseline: Qwen2.5-Math-1.5B naive GRPO on DeepScaleR, MATH500 L4-5 eval.

  From the math-pretrained Qwen2.5-Math-1.5B checkpoint. Scored on MATH500
  levels 4-5 with avg@8 at temp 0.6. Flash attention on; longer decode budget
  for competition-math reasoning.
  """
  config = RLConfig().override_from(core.qwen_math_1p5b_v2p5())
  config = _apply_boxed_answer_rl(
      config,
      train_source='simply:dsr40k_train',
      eval_source='simply:math500_test_l45',
      train_max_seq_len=4096,
      decode_budget=3072,
      eval_temperature=0.6,
      eval_num_samples=8,  # avg@8
      eval_batch_size=16,
      use_flash_attention=True,
  )
  return dataclasses.replace(
      config,
      activation_dtype_name='float32',
      ref_params_dtype='float32',
      decoding_quant_scheme='float32',
  )


# ==============================================================================
# Model-PORTING tasks: port_falcon_h1_0p5b + port_recurrentgemma_2b.
#
# Both are UNIFIED train+eval submissions on the `research_bench_rl` loop: the
# DEFAULT `num_train_steps=0` just loads the ported checkpoint and runs the
# FIXED inline held-out eval (writing `eval_accuracy` to final_result.json);
# stage-2 is to raise `num_train_steps` and design a (GRPO) post-training
# recipe. The eval and its wiring are FIXED. The model classes (FalconH1LM /
# RecurrentGemmaLM) and checkpoint formats (FalconH1Format /
# RecurrentGemmaFormat) ship as STUBS -- the port IS the task. NO core simply
# change is required.
# ==============================================================================

# Both porting-task checkpoints are staged under the research-bench directory
# (see data_lib for the matching tokenizers).
FALCON_H1_0P5B_CKPT_DIR = os.path.join(
    core.MODELS_DIR, 'research_bench/Falcon-H1-0.5B-Base/ORBAX'
)
RECURRENTGEMMA_2B_CKPT_DIR = os.path.join(
    core.MODELS_DIR, 'research_bench/RecurrentGemma-2B/ORBAX'
)


@dataclasses.dataclass(frozen=True)
class FalconH1RLConfig(RLConfig):
  """`research_bench_rl` config carrying Falcon-H1 hyperparameters + muP mults.

  Inherits the research-bench RL surface (research_bench_rl loop + decoupled
  held-out eval fields) and adds the Falcon-H1-specific architecture fields that
  the self-contained `FalconH1LM` reads via `config.<field>`. Source of truth:
  the published Falcon-H1-0.5B-Base config + checkpoint tensors.
  """

  model_name: str = 'FalconH1LM'
  lm_format_name: str = 'FalconH1GSM8K'
  # RoPE base (config.rope_theta).
  falcon_rope_theta: float = 1e11
  # muP multipliers (properties of the model; provided so the port need not
  # derive them).
  falcon_embedding_multiplier: float = 5.656854249492381
  falcon_lm_head_multiplier: float = 0.0390625
  falcon_attention_in_multiplier: float = 1.0
  falcon_attention_out_multiplier: float = 0.9375
  falcon_key_multiplier: float = 0.39062499999999994
  falcon_ssm_in_multiplier: float = 1.25
  falcon_ssm_out_multiplier: float = 0.23570226039551587
  falcon_mlp_multipliers: tuple[float, ...] = (0.8838834764831844, 0.5859375)
  falcon_ssm_multipliers: tuple[float, ...] = (
      0.3535533905932738,
      0.25,
      0.3535533905932738,
      0.5,
      0.3535533905932738,
  )
  # Mamba-2 SSM sizes. -1 sentinels are recovered by the participant from the
  # checkpoint ('?' in the spec table); d_state + n_groups are kept as the
  # anchors that pin the [x, B, C] split of the conv block (task-designs.md).
  mamba_d_ssm: int = -1  # recover: mamba out_proj weight shape
  mamba_n_heads: int = -1  # recover: A_log / D / dt_bias vector length
  mamba_d_head: int = -1  # recover: mamba_d_ssm / mamba_n_heads
  mamba_d_state: int = 128
  mamba_n_groups: int = 1
  mamba_d_conv: int = -1  # recover: mamba conv1d weight shape


@dataclasses.dataclass(frozen=True)
class RecurrentGemmaRLConfig(RLConfig):
  """`research_bench_rl` config carrying RecurrentGemma-2B (Griffin) arch fields.

  Inherits the research-bench RL surface and adds the RecurrentGemma-specific
  fields the self-contained `RecurrentGemmaLM` reads via `getattr(config, ...)`.
  Source of truth: the published RecurrentGemma-2B config + checkpoint tensors.
  """

  model_name: str = 'RecurrentGemmaLM'
  lm_format_name: str = 'RecurrentGemmaGSM8K'
  # Griffin per-layer block pattern: (recurrent, recurrent, attention) repeated.
  block_type: tuple[str, ...] = ('recurrent', 'recurrent', 'attention')
  # RG-LRU recurrent-mixer knobs. -1 sentinels are recovered by the participant
  # from the checkpoint ('?' in the spec table): lru_width from linear_x/rg_lru
  # shapes, conv1d_width from the conv weight, rglru_num_heads from the
  # block-diagonal gate weight [G, bw, bw]. See internal-notes/task-designs.md.
  lru_width: int = -1
  conv1d_width: int = -1
  rglru_num_heads: int = -1
  # Attention: MQA local attention with an o_proj bias.
  o_use_bias: bool = True


def _falcon_h1_0p5b_arch(config):
  """Applies the Falcon-H1-0.5B-Base architecture + checkpoint/vocab wiring."""
  # Some size hyperparameters are set to the sentinel -1 (a `?` in the task
  # spec table): the participant must RECOVER them from the provided checkpoint
  # (tensor shapes) / vocab. All -1 values are unambiguously recoverable given
  # anchors (model_dim, per_head_dim, mamba_d_state, mamba_n_groups) with NO
  # circular dependency; real values in internal-notes/task-designs.md.
  return dataclasses.replace(
      config,
      seq_len=4096,
      vocab_size=-1,  # recover: embedding tensor row count
      model_dim=1024,
      n_layers=-1,  # recover: count the per-layer tensor groups
      n_heads=8,
      n_kv_heads=2,
      per_head_dim=64,
      ffn_expand_dim=-1,  # recover: MLP gate/up/down weight shape
      expand_factor=0,
      ffn_activation='silu',
      rms_norm_epsilon=1e-5,
      use_tied_embedding=False,
      output_layer_use_bias=False,
      qkv_use_bias=False,
      ffn_use_bias=False,
      attn_soft_cap=-1.0,
      output_logits_soft_cap=-1.0,
      norm_scale_plus_one=False,
      # RoPE handled internally by FalconH1LM; keep NoPE at framework level.
      position_encoding=None,
      vocab_name='FalconH1',
      lm_format_name='FalconH1GSM8K',
      init_ckpt_dir=FALCON_H1_0P5B_CKPT_DIR,
      init_ckpt_step=-1,
      init_ckpt_format='FalconH1Format',
      reset_steps=True,
  )


def _recurrentgemma_2b_arch(config):
  """Applies the RecurrentGemma-2B-Base architecture + ckpt/vocab wiring."""
  # Some size hyperparameters are set to the sentinel -1 (a `?` in the task
  # spec table): the participant RECOVERS them from the provided checkpoint
  # tensor shapes. All are unambiguously recoverable given the kept anchors
  # (model_dim labels the 2-D weight axes; n_kv_heads=1 makes k_proj = head_dim;
  # block_type + the per-block tensors give n_layers) with NO circular
  # dependency -- real values in internal-notes/task-designs.md.
  return dataclasses.replace(
      config,
      seq_len=2048,
      vocab_size=-1,  # recover: embedding tensor row count
      model_dim=2560,
      n_layers=-1,  # recover: count the per-layer tensor groups
      n_heads=-1,  # recover: q_proj out-dim / per_head_dim
      n_kv_heads=1,
      per_head_dim=-1,  # recover: k_proj out-dim (MQA: n_kv_heads=1)
      # FFN: gated GeGLU (gate/up/down).
      ffn_expand_dim=-1,  # recover: MLP gate/up/down weight shape
      expand_factor=0,
      ffn_activation='gelu',
      use_tied_embedding=True,
      output_layer_use_bias=False,
      qkv_use_bias=False,
      rms_norm_epsilon=1e-6,
      # RMSNorm with (1+scale); output logits soft-capped at 30; no attn cap.
      norm_scale_plus_one=True,
      attn_soft_cap=-1.0,
      output_logits_soft_cap=30.0,
      # Local sliding-window attention on the attention layers (window 2048).
      window_size=2048,
      # PartialRoPE (rotate the first half of each head's dims, theta 10000) is
      # applied INTERNALLY by RecurrentGemmaLM; keep NoPE at framework level.
      position_encoding=None,
      vocab_name='RecurrentGemma',
      lm_format_name='RecurrentGemmaGSM8K',
      init_ckpt_dir=RECURRENTGEMMA_2B_CKPT_DIR,
      init_ckpt_step=-1,
      init_ckpt_format='RecurrentGemmaFormat',
      reset_steps=True,
  )


def _port_falcon_h1_0p5b_base():
  """Shared base for `port_falcon_h1_0p5b` on the research_bench_rl loop.

  Falcon-H1-0.5B ported checkpoint; GRPO on gsm8k train is the (stage-2)
  training surface; scored inline on the FULL gsm8k test (1319) under the FIXED
  5-shot-strict no-trailing-space harness, greedy (temp 0, pass@1). The eval and
  its wiring below are FIXED.

  Returns:
    The base `port_falcon_h1_0p5b` config (the registered task wrapper then sets
    the final training-recipe knobs + num_train_steps).
  """
  config = FalconH1RLConfig()
  config = _apply_rl(config)  # research_bench_rl loop + generic defaults
  config = _falcon_h1_0p5b_arch(config)
  return dataclasses.replace(
      config,
      # Data: gsm8k train -> gsm8k FULL test, FIXED 5-shot-strict no-trailspace.
      dataset=data_lib.DatasetConfig(
          source='simply:gsm8k_train',
          packing=data_lib.PACKING_NONE,
          lm_format_name=None,
      ),
      evaluation=port_eval_lib.FewShotGSM8K5ShotStrictNoTrailSpaceEvaluation(),
      validation_datasets=(
          data_lib.DatasetConfig(
              source='simply:gsm8k_test',
              packing=data_lib.PACKING_NONE,
              lm_format_name=None,
          ),
      ),
      # FIXED scored held-out eval (read by research_bench_rl).
      validation_evaluation=(
          port_eval_lib.FewShotGSM8K5ShotStrictNoTrailSpaceEvaluation()
      ),
      validation_eval_interval=50,
      validation_eval_batch_size=16,
      lm_format_name='FalconH1GSM8K',
      # PT base, R1-Zero-like: no chat wrapper, no reference params.
      use_ref_params=False,
      use_validation_set=False,
      use_flash_attention=False,
      # FIXED held-out eval decoding (decoupled from the training sampler):
      # greedy, pass@1, matching the verified port harness.
      eval_temperature=0.0,
      eval_num_samples=1,
      eval_max_decode_steps=512,
      eval_max_input_len=1536,
      eval_prefill_size=1536,
      # Training sampler (used only on the stage-2 num_train_steps>0 path).
      sampling_temperature=1.0,
      sampling_max_decode_steps=512,
      train_max_seq_len=2560,
      sampling_prefill_size=1536,
      sampling_max_input_len=1536,
      sampling_intermediate_decode_steps=512,
      batch_size=16,
      num_samples_per_example=8,
      # '\n\n' few-shot separator is the stop token (also in FalconH1GSM8K).
      extra_eos_tokens=core.newlines_from_counts(range(2, 6)),
      # FIXED eval prompt format + stop tokens.
      validation_lm_format_name='FalconH1GSM8K',
      eval_extra_eos_tokens=core.newlines_from_counts(range(2, 6)),
      activation_dtype_name='bfloat16',
      should_save_ckpt=False,
      ckpt_max_to_keep=1,
      tb_log_interval=20,
      ckpt_interval=100,
  )


@ExperimentConfigRegistry.register
def port_falcon_h1_0p5b():
  """Falcon-H1-0.5B PORT task: 0-step default = load ckpt + inline eval.

  DEFAULT `num_train_steps=0`: loads the ported Falcon-H1-0.5B checkpoint and
  evaluates it inline on the FULL gsm8k test under the FIXED
  5-shot-strict-no-trailspace harness (greedy, pass@1), writing `eval_accuracy`
  to final_result.json. Submit as a 3-seed sweep
  (42/43/44); the mean `eval_accuracy` is the metric. Stage-2 = raise
  `num_train_steps` and design the (GRPO) training recipe; the eval is FIXED.
  """
  return dataclasses.replace(
      _port_falcon_h1_0p5b_base(),
      train_batch_size=32,
      lr=opt_lib.LinearWarmupConstant(value=1e-7, warmup_steps=1),
      num_train_steps=0,
  )


def _port_recurrentgemma_2b_base():
  """Shared base for `port_recurrentgemma_2b` on the research_bench_rl loop.

  RecurrentGemma-2B ported checkpoint; GRPO on gsm8k train is the (stage-2)
  training surface; scored inline on the FULL gsm8k test (1319) under the
  FIXED, paper-faithful 5-shot-strict harness with a SINGLE sample at
  temperature 0.4. The 2.6B model + a 2.6GB fp32 tied embedding require a
  MODEL-SHARDED mesh; a replicated (model=1) embedding OOMs even a 4-chip slice,
  so shard 4-way on a single-host 4-chip slice (research_bench_rl is
  single-host).
  The eval and its wiring below are FIXED.

  Returns:
    The base `port_recurrentgemma_2b` config (the registered task wrapper then
    sets the final training-recipe knobs + num_train_steps).
  """
  config = RecurrentGemmaRLConfig()
  config = _apply_rl(config)  # research_bench_rl loop + generic defaults
  config = _recurrentgemma_2b_arch(config)
  return dataclasses.replace(
      config,
      # Shard the 2.6B model (and the fp32 tied embedding/logits) across the
      # slice's chips on the 'model' axis: a 4-chip slice => 4-way model mesh. The
      # research_bench_rl loop is single-host, so use a single-host 4-chip
      # slice, NOT an 8-chip 2-host slice.
      mesh_shape={'model': 4},
      decoding_mesh_shape={'model': 4},
      dataset=data_lib.DatasetConfig(
          source='simply:gsm8k_train',
          packing=data_lib.PACKING_NONE,
          lm_format_name=None,
      ),
      evaluation=port_eval_lib.FewShot5StrictGSM8KEvaluation(),
      validation_datasets=(
          data_lib.DatasetConfig(
              source='simply:gsm8k_test',
              packing=data_lib.PACKING_NONE,
              lm_format_name=None,
          ),
      ),
      # FIXED scored held-out eval (read by research_bench_rl).
      validation_evaluation=port_eval_lib.FewShot5StrictGSM8KEvaluation(),
      validation_eval_interval=50,
      # Small eval batch so the 2.6B model + decode activations fit in HBM; the
      # held-out eval iterates the full 1319 test regardless of batch size.
      validation_eval_batch_size=8,
      lm_format_name='RecurrentGemmaGSM8K',
      # PT base, R1-Zero-like: no chat wrapper, no reference params.
      use_ref_params=False,
      use_validation_set=False,
      use_flash_attention=False,
      # FIXED held-out eval decoding: SINGLE sample at temperature 0.4 (the
      # paper's maj@1 draw), matching the verified port harness.
      eval_temperature=0.4,
      eval_num_samples=1,
      eval_max_decode_steps=512,
      eval_max_input_len=1024,
      eval_prefill_size=256,
      # Training sampler (used only on the stage-2 num_train_steps>0 path).
      sampling_temperature=1.0,
      sampling_max_decode_steps=512,
      train_max_seq_len=1536,
      sampling_prefill_size=256,
      sampling_max_input_len=1024,
      sampling_intermediate_decode_steps=512,
      batch_size=16,
      num_samples_per_example=8,
      # '\n\n' few-shot separator is the stop token (also in the lm_format).
      extra_eos_tokens=core.newlines_from_counts(range(2, 6)),
      # FIXED eval prompt format + stop tokens.
      validation_lm_format_name='RecurrentGemmaGSM8K',
      eval_extra_eos_tokens=core.newlines_from_counts(range(2, 6)),
      activation_dtype_name='bfloat16',
      should_save_ckpt=False,
      ckpt_max_to_keep=1,
      tb_log_interval=20,
      ckpt_interval=100,
  )


@ExperimentConfigRegistry.register
def port_recurrentgemma_2b():
  """RecurrentGemma-2B PORT task: 0-step default = load ckpt + inline eval.

  DEFAULT `num_train_steps=0`: loads the ported RecurrentGemma-2B (Griffin)
  checkpoint and evaluates it inline on the FULL gsm8k test under the FIXED,
  paper-faithful 5-shot-strict harness with a SINGLE sample at temperature 0.4,
  writing `eval_accuracy` to final_result.json. Submit as a 3-seed sweep
  (42/43/44); the mean `eval_accuracy` is the metric (temp>0 ->
  ~0.8pt/seed variance). Stage-2 =
  raise `num_train_steps` and design the (GRPO) training recipe; eval is FIXED.
  """
  return dataclasses.replace(
      _port_recurrentgemma_2b_base(),
      train_batch_size=32,
      lr=opt_lib.LinearWarmupConstant(value=1e-7, warmup_steps=1),
      num_train_steps=0,
  )

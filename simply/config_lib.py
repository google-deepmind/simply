# Copyright 2024 The Simply Authors
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
"""Experiments and sharding configs."""

from collections.abc import Iterable
import dataclasses
import functools
import math
import os
from typing import Any, ClassVar

import jax

from simply.utils import evaluation_lib
from simply.utils import optimizers as opt_lib
from simply.utils import registry


SimplyConfig = Any

################################################################################
# Checkpoint directories.
MODELS_DIR = os.getenv('SIMPLY_MODELS', os.path.expanduser('~/.cache/simply/models/'))

GEMMA2_2B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-2.0-2B-PT-ORBAX')
GEMMA2_9B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-2.0-9B-PT-ORBAX')
GEMMA2_27B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-2.0-27B-PT-ORBAX')
GEMMA2_2B_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-2.0-2B-IT-ORBAX')
GEMMA2_9B_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-2.0-9B-IT-ORBAX')
GEMMA2_27B_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-2.0-27B-IT-ORBAX')

GEMMA3_270M_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-270M-PT-ORBAX')
GEMMA3_1B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-1B-PT-ORBAX')
GEMMA3_4B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-4B-PT-ORBAX')
GEMMA3_12B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-12B-PT-ORBAX')
GEMMA3_27B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-27B-PT-ORBAX')
GEMMA3_270M_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-270M-IT-ORBAX')
GEMMA3_1B_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-1B-IT-ORBAX')
GEMMA3_4B_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-4B-IT-ORBAX')
GEMMA3_12B_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-12B-IT-ORBAX')
GEMMA3_27B_IT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-27B-IT-ORBAX')

DEEPSEEK_QWEN_1P5B_CKPT_DIR = os.path.join(MODELS_DIR, 'DeepSeek-R1-Distill-Qwen-1.5B/ORBAX')
DEEPSEEK_QWEN_7B_CKPT_DIR = os.path.join(MODELS_DIR, 'DeepSeek-R1-Distill-Qwen-7B/ORBAX')
DEEPSEEK_QWEN_14B_CKPT_DIR = os.path.join(MODELS_DIR, 'DeepSeek-R1-Distill-Qwen-14B/ORBAX')
DEEPSEEK_QWEN_32B_CKPT_DIR = os.path.join(MODELS_DIR, 'DeepSeek-R1-Distill-Qwen-32B/ORBAX')

QWEN2p5_MATH_1p5B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen2.5-Math-1.5B/ORBAX')
QWEN2p5_MATH_7B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen2.5-Math-7B/ORBAX')
QWEN2p5_MATH_14B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen2.5-Math-14B/ORBAX')
QWEN2p5_MATH_32B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen2.5-Math-32B/ORBAX')
QWQ_32B_CKPT_DIR = os.path.join(MODELS_DIR, 'QwQ-32B/ORBAX')

QWEN3_0P6B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen3-0.6B/ORBAX')
QWEN3_1P7B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen3-1.7B/ORBAX')
QWEN3_4B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen3-4B/ORBAX')
QWEN3_8B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen3-8B/ORBAX')
QWEN3_14B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen3-14B/ORBAX')
QWEN3_32B_CKPT_DIR = os.path.join(MODELS_DIR, 'Qwen3-32B/ORBAX')


################################################################################
# Config registries.


class ExperimentConfigRegistry(registry.RootRegistry):
  namespace: ClassVar[str] = 'Experiment'

  @classmethod
  def get_config(cls, name: str):
    return cls.get(name)()


class ShardingConfigRegistry(registry.RootRegistry):
  namespace: ClassVar[str] = 'Sharding'

  @classmethod
  def get_config(cls, name: str):
    return cls.get(name)()


################################################################################
# Utilities.


def newlines_from_counts(counts: Iterable[int]) -> tuple[str, ...]:
  return tuple('\n' * i for i in counts)


################################################################################
# Sharding Configs.


@ShardingConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class GSPMDSharding:
  # Shape (model_dim, model_dim * expansion_factor)
  ffn0_partition: Any = ('data', 'model')

  # Shape (model_dim * expansion_factor, model_dim)
  ffn1_partition: Any = ('model', 'data')

  # Shape (model_dim, num_heads, per_head_size)
  attn_qkv_partition: Any = ('data', 'model', None)

  # Shape (model_dim, num_heads, per_head_size)
  attn_o_partition: Any = ('data', 'model', None)

  # Shape (vocab_size, model_dim)
  embed_partition: Any = ('model', 'data')

  # Shape (batch_size, seq_len, num_heads, per_head_size)
  attn_activation_partition: Any = (('replica', 'data'), None, 'model', None)

  # Shape (batch_size, seq_len, model_dim)
  activation_partition: Any = (('replica', 'data'), None, 'model')

  # Shape (batch_size, seq_len, model_dim * expansion_factor)
  ffn0_activation_partition: Any = (('replica', 'data'), None, 'model')

  # Shape (batch_size, seq_len, vocab_size)
  logits_partition: Any = (('replica', 'data'), None, 'model')

  # Shape (batch_size, seq_len)
  data_partition: Any = (('replica', 'data'), None)


@ShardingConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DecodeGSPMDSharding(GSPMDSharding):
  # Shape (batch_size, seq_len, model_dim)
  activation_partition: Any = (('replica', 'data'), None, None)


@ShardingConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DataParallelSharding:
  # Shape (model_dim, model_dim * expansion_factor)
  ffn0_partition: Any = (None, None)

  # Shape (model_dim * expansion_factor, model_dim)
  ffn1_partition: Any = (None, None)

  # Shape (model_dim, num_heads, per_head_size)
  attn_qkv_partition: Any = (None, None, None)

  # Shape (model_dim, num_heads, per_head_size)
  attn_o_partition: Any = (None, None, None)

  # Shape (vocab_size, model_dim)
  embed_partition: Any = (None, None)

  # Shape (b, l, num_heads, per_head_size)
  attn_activation_partition: Any = (
      ('replica', 'data', 'model'),
      None,
      None,
      None,
  )

  # Shape (batch_size, seq_len, model_dim)
  activation_partition: Any = (('replica', 'data', 'model'), None, None)

  # Shape (batch_size, seq_len, model_dim * expansion_factor)
  ffn0_activation_partition: Any = (('replica', 'data', 'model'), None, None)

  # Shape (batch_size, seq_len, vocab_size)
  logits_partition: Any = (('replica', 'data', 'model'), None, None)

  # Shape (batch_size, seq_len)
  data_partition: Any = (('replica', 'data', 'model'), None)


################################################################################
# Experiment Configs.


################################################################################
## Base experiment for others to inherit.


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
  """Base experiment config for others to inherit."""


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class BaseExperimentConfig(ExperimentConfig):
  # number of parameters: ~1700.790328 M
  # `_variant` is used when you need to create a series of configs based on
  # a base config, e.g., for different hparams search. Default to empty string.
  _variant: str = ''
  seq_len: int = 1024
  vocab_size: int = 32_000
  model_dim: int = 2048
  per_head_dim: int = 128
  n_heads: int = 16
  n_layers: int = 14
  expand_factor: int = 8
  use_scan: bool = True
  use_remat: bool = True
  # use dots_with_no_batch_dims_saveable for faster speed and more memory cost.
  remat_policy: str = 'nothing_saveable'
  model_seed: int = 42
  use_rmsnorm: bool = True
  use_pre_ln: bool = True
  use_post_ln: bool = True
  use_post_skip_ln: bool = False
  use_qk_norm: bool = False
  use_per_dim_scale: bool = True
  use_gated_activation_in_ffn: bool = True
  activation_dtype_name: str = 'bfloat16'
  use_flash_attention: bool = False
  # The block size should be smaller than the sequence length and it should be
  # tuned for different config for best performance, 512 and 1024 are
  # good starting points.
  flash_attention_block_size: int = 512
  window_size: int = 0
  use_window_chunk: bool = False
  use_combined_qkv: bool = True
  qkv_use_bias: bool = False
  n_kv_heads: int = 0
  block_attn_pattern: tuple[str, ...] = ('global',)
  output_layer_use_bias: bool = True
  use_tied_embedding: bool = True
  ffn_use_bias: bool = True
  # NOTE: When this is set to None, the expand_dim will be set to
  # expand_factor * model_dim. Otherwise, expand_dim will be set to this value.
  ffn_expand_dim: int | None = None
  ffn_activation: str = 'gelu'
  embedding_lookup_scale: float | None = 1.0
  norm_scale_plus_one: bool = True
  attn_soft_cap: float = 50.0  # If negative, no softcap.
  output_logits_soft_cap: float = 30.0  # If negative, no softcap.
  rms_norm_epsilon: float = 1e-6
  local_rope_max_timescale: int = 10_000
  global_rope_max_timescale: int = 10_000
  local_rope_scale_factor: float = 1.0
  global_rope_scale_factor: float = 1.0
  query_scale: float = -1.0
  # MoE related.
  use_moe: bool = False
  num_experts: int = 32
  num_experts_per_token: int = 2
  expert_capacity_factor: float | None = None
  lbl_loss_weight: float = 0.01
  router_z_loss_weight: float = 0.0
  tile_batch_seq: int = 1024
  tile_model_dim: int = 1024
  tile_expand_dim: int = 1024
  gmm_impl: str = 'ragged_dot'

  # Data config
  batch_size: int = 64 * 16
  dataset_name: str = 'lm1b'
  dataset_seed: int = 42
  use_packing: bool = True
  use_validation_set: bool = False
  # How many steps / validation examples to evaluate on,
  # set to -1 to use whole set
  validation_num_eval_steps: int = -1
  # How often to run evaluation on validation set.
  validation_eval_interval: int = 1000
  # Batch size for evaluation on validation set,
  # set to -1 to use the same as `batch_size`.
  validation_eval_batch_size: int = -1
  # Number of epochs to run evaluation on validation set.
  validation_eval_epochs: int = 1
  # NOTE: This is only used when `use_validation_set` is True. If not set,
  # `dataset_name` will be used.
  validation_dataset_name: str | None = None
  feature_converter_name: str = 'LMFeatureConverter'
  # Number of prefetch workers for the data pipeline. Since we use
  # pygrain's multi-processing prefetching, this is the number of processes.
  # Set to 0 to disable multi-processing. Note that changing
  # prefetch_num_workers will change the order of the data loading.
  prefetch_num_workers: int = 8
  prefetch_per_worker_buffer_size: int = 2

  # Training config
  train_loop_name: str = 'default'
  optimizer: opt_lib.Optimizer = opt_lib.Adam(
      beta1=0.9, beta2=0.95, epsilon=1e-6
  )
  weight_decay: float = 1e-3
  num_train_steps: int = 100_000
  lr: opt_lib.Schedule = opt_lib.LinearWarmupCosineDecay(
      value=1e-3,
      warmup_steps=1_000,
      steps_after_decay=0,
      end_decay=0.1,
  )
  # The following two fields are used for backward compatibility and will
  # be deprecated.
  lr_schedule_name: str = ''
  lr_schedule_config: tuple[tuple[str, Any], ...] = ()
  clip_grad_norm: float = 1.0
  clip_update_norm: float = -1.0
  clip_local_update_rms: float = 1.0
  grad_accum_steps: int = -1

  # Checkpoint and tensorboard config
  # should_save_ckpt refers to whether to save any checkpoint. If False,
  # overrides all the other checkpoint configs. If True, the other checkpoint
  # configs come into effect.
  should_save_ckpt: bool = True
  ckpt_interval: int = 1000
  ckpt_max_to_keep: int = 3
  ckpt_keep_period: int | None = None
  tb_log_interval: int = 100
  log_additional_info: bool = True

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = ''
  init_ckpt_step: int = -1
  init_ckpt_opt_state: bool = False
  init_ckpt_format: str = ''
  reset_steps: bool = False

  # Add masks to only calculate loss on assistant responses.
  add_chat_loss_mask: bool = False
  mask_start_token: str = ''
  mask_end_token: str = ''
  vocab_path: str = ''
  vocab_name: str = ''

  # Name for the model, i.e., the main module.
  model_name: str = 'TransformerLM'

  # TODO: The type should be model_lib.InputEncoderInterface, but we
  # need to resolve some cyclic dependency issue.
  input_encoders: list[Any] = dataclasses.field(default_factory=list)

  # Utilities for patching code snippets before running an experiment.
  code_patch: tuple[tuple[str, str], ...] = ()

  # Early stopping threshold.
  early_stop: opt_lib.EarlyStop | None = None


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class RLExperimentConfig(BaseExperimentConfig):
  train_loop_name: str = 'rl'
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotBoxedInQuestionEvaluation()
  )
  dataset_name: str = 'simply_json:gsm8k_train'
  num_train_steps: int = 1_000_000
  use_validation_set: bool = False
  validation_num_eval_steps: int = 8
  validation_eval_interval: int = 100
  validation_dataset_name: str | None = 'simply_json:gsm8k_test'
  validation_eval_batch_size: int = -1
  validation_eval_epochs: int = 1

  # We need to define the format for sampling.
  lm_format_name: str = 'Pretrain'
  # We draw `batch_size` examples from the dataset, generate
  # `num_samples_per_example` samples per example, wait for their rewards
  # asynchronously, and finally start training when there are
  # `train_batch_size` total samples.
  batch_size: int = 16
  train_batch_size: int = 16 * 8
  num_samples_per_example: int = 8
  sampling_temperature: float = 1.0

  # Use train_max_seq_len to control the max decode steps.
  sampling_max_decode_steps: int = 32768
  train_max_seq_len: int = 2048
  sampling_prefill_size: int = 1024
  sampling_max_input_len: int = 1024
  sampling_intermediate_decode_steps: int = 1024
  # How many steps to run given one batch of samples for training.
  num_train_steps_per_batch: int = 4
  max_num_samples_per_train_batch: int | None = None

  extra_eos_tokens: tuple[str, ...] = ()

  # RL algorithm configs.
  gamma: float = 1.0
  kl_coeff: float = 0.001
  use_grpo: bool = True
  ppo_clip_eps: float = 0.2
  ppo_clip_eps_high: float | None = None
  ppo_clip_eps_low: float | None = None
  policy_ratio_cap: float | None = 10.0
  normalize_reward_method: str = 'ByGroup'
  normalize_advantage: bool = False
  max_abs_advantage: float | None = 10.0
  filter_truncated: bool = False
  use_policy_logp_as_sampler_logp: bool = False


################################################################################
## C4 experiments.


@ExperimentConfigRegistry.register
def flops6e20_tfm2b_c4_l2048():
  config = BaseExperimentConfig()
  config = dataclasses.replace(
      config,
      model_dim=2048,
      per_head_dim=256,
      n_heads=8,
      n_layers=18,
      expand_factor=8,
      seq_len=2048,
      vocab_size=100_864,
      dataset_name='c4.vb100864_openmix_v1',
      batch_size=1024,
      clip_grad_norm=1.0,
      num_train_steps=21_297,
      weight_decay=0.3,
      lr=opt_lib.LinearWarmupCosineDecay(
          value=1e-3,
          warmup_steps=1_000,
          steps_after_decay=0,
          end_decay=0.1,
      ),
      ckpt_max_to_keep=1,
      use_validation_set=True,
      validation_num_eval_steps=2,
      validation_eval_interval=1000,
      validation_eval_batch_size=1024,
      ffn_use_bias=False,
  )
  return config


@ExperimentConfigRegistry.register
def flops1e20_tfm986m_c4_l2048():
  # num_params: 985.890304 M
  # num_non_embedding_params: 830.9632 M
  # num_embedding_params: 154.927104 M
  # embedding_params_ratio: 0.15714436319276348
  # num_tokens: 18405.654528 M
  # num_tokens / num_params: 18.669069422149423
  # num_tokens / num_non_embedding_params: 22.14978296030438
  # num_flops: 1.0887573802757338e+20
  # Fitted optimal ratio for 1.1e20: 18.67
  config = flops6e20_tfm2b_c4_l2048()
  config = dataclasses.replace(
      config,
      model_dim=1536,
      n_layers=12,
      # 985890304 * 18.67 / 2048 / 256 = 35106 steps
      batch_size=256,
      num_train_steps=35_106,
      validation_num_eval_steps=4,
      validation_eval_batch_size=512,
  )
  return config


@ExperimentConfigRegistry.register
def flops1e19_tfm338m_c4_l2048():
  # num_params: 338.440192 M
  # num_non_embedding_params: 235.155456 M
  # num_embedding_params: 103.284736 M
  # embedding_params_ratio: 0.3051786946155615
  # num_tokens: 6149.89824 M
  # num_tokens / num_params: 18.171299938276835
  # num_tokens / num_non_embedding_params: 26.15247948999321
  # num_flops: 1.2488236446756372e+19
  # Fitted optimal ratio for 1.2e19: 17.97
  config = flops6e20_tfm2b_c4_l2048()
  config = dataclasses.replace(
      config,
      model_dim=1024,
      per_head_dim=128,
      n_layers=8,
      # 338440192 * 17.97 / 2048 / 192 = 15466 steps
      batch_size=192,
      num_train_steps=15_466,
      weight_decay=0.276,
      lr=opt_lib.LinearWarmupCosineDecay(
          value=0.0013656918867398535,
          warmup_steps=1_000,
          steps_after_decay=0,
          end_decay=0.1,
      ),
      validation_num_eval_steps=4,
      validation_eval_interval=500,
      validation_eval_batch_size=512,
  )
  return config


@ExperimentConfigRegistry.register
def flops1e18_tfm111m_c4_l2048():
  # num_params: 110.550528 M
  # num_non_embedding_params: 58.90816 M
  # num_embedding_params: 51.642368 M
  # embedding_params_ratio: 0.467138139765375
  # num_tokens: 1901.068288 M
  # num_tokens / num_params: 17.19637456638832
  # num_tokens / num_non_embedding_params: 32.27173091130329
  # num_flops: 1.2609846180147364e+18
  # Predicted optimal ratio for 1.3e18: 17.2
  config = flops6e20_tfm2b_c4_l2048()
  config = dataclasses.replace(
      config,
      model_dim=512,
      per_head_dim=64,
      n_layers=8,
      # 110550528 * 17.2 / 2048 / 128 = 7252 steps
      batch_size=128,
      num_train_steps=7252,
      weight_decay=0.261,
      lr=opt_lib.LinearWarmupCosineDecay(
          value=0.0016486710944803309,
          warmup_steps=1_000,
          steps_after_decay=0,
          end_decay=0.1,
      ),
      validation_num_eval_steps=8,
      validation_eval_interval=500,
      validation_eval_batch_size=256,
  )
  return config


@ExperimentConfigRegistry.register
def flops2e17_tfm41m_c4_l2048():
  # num_params: 40.645632 M
  # num_non_embedding_params: 14.824448 M
  # num_embedding_params: 25.821184 M
  # embedding_params_ratio: 0.6352757413145895
  # num_tokens: 678.428672 M
  # num_tokens / num_params: 16.691305771798554
  # num_tokens / num_non_embedding_params: 45.764177661117635
  # num_flops: 1.6545097284216422e+17
  # Fitted optimal ratio for 1.7e17: 16.69
  config = flops6e20_tfm2b_c4_l2048()
  config = dataclasses.replace(
      config,
      model_dim=256,  # 2048 // 8
      per_head_dim=32,  # 256 // 8
      n_layers=8,  # 18 // 2 - 1
      # 40645632 * 16.69 / 2048 / 80 = 4140 steps
      batch_size=80,
      num_train_steps=4140,
      weight_decay=0.248,
      lr=opt_lib.LinearWarmupCosineDecay(
          value=0.0019495171900601506,
          warmup_steps=1_000,
          steps_after_decay=0,
          end_decay=0.1,
      ),
      validation_num_eval_steps=16,
      validation_eval_interval=500,
      validation_eval_batch_size=128,
  )
  return config


@ExperimentConfigRegistry.register
def flops2e16_tfm15m_c4_l2048():
  # num_params: 14.857408 M
  # num_non_embedding_params: 1.946816 M
  # num_embedding_params: 12.910592 M
  # embedding_params_ratio: 0.86896664
  # num_tokens: 225.050624 M
  # num_tokens / num_params: 15.1473678316
  # num_tokens / num_non_embedding_params: 116.00547628866
  # num_flops: 2.006201364854e+16
  # Data / model ratio for 2.0e16: 15.15
  config = flops6e20_tfm2b_c4_l2048()
  config = dataclasses.replace(
      config,
      model_dim=128,  # 2048 // 16
      per_head_dim=16,  # 256 // 16
      n_layers=4,  # 18 => 4
      # 14.857408 * 15.15 / 2048 / 64 = 1717 steps
      batch_size=64,
      num_train_steps=1717,
      weight_decay=0.1,
      lr=opt_lib.LinearWarmupCosineDecay(
          value=0.01,
          warmup_steps=1_000,
          steps_after_decay=0,
          end_decay=0.1,
      ),
      validation_num_eval_steps=16,
      validation_eval_interval=500,
      validation_eval_batch_size=128,
  )
  return config

##########################
## Gemma models.
## https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_gemma.py


@ExperimentConfigRegistry.register
def gemma2_2b():
  config = BaseExperimentConfig()
  return dataclasses.replace(
      config,
      # number of parameters: ~2.61B
      seq_len=4096,
      vocab_size=256128,
      model_dim=2304,
      per_head_dim=256,
      n_heads=8,
      n_layers=26,
      expand_factor=4,
      use_scan=True,
      use_remat=True,
      model_seed=42,
      use_rmsnorm=True,
      use_pre_ln=True,
      use_post_ln=True,
      use_post_skip_ln=False,
      use_per_dim_scale=False,
      use_gated_activation_in_ffn=True,
      use_flash_attention=False,
      window_size=4096 - 1,
      use_window_chunk=False,
      n_kv_heads=4,
      block_attn_pattern=(
          'local',
          'global',
      ),
      output_layer_use_bias=False,
      ffn_use_bias=False,
      # NOTE: Data config is vocab dependent. We currently do not have dataset
      # prepared with Gemma2 vocab.
      vocab_name='vb256128_gemma2',
      # Config for init from existing checkpoint.
      init_ckpt_dir=GEMMA2_2B_PT_CKPT_DIR,
      init_ckpt_step=-1,
      init_ckpt_opt_state=False,
      init_ckpt_format='Gemma3pLegacyFormat',
      reset_steps=True,
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
  )


@ExperimentConfigRegistry.register
def gemma2_2b_c4_vocab100864_l2048_bs1024():
  """Gemma 2B model with C4 vocab 100864 and seq_len 2048."""
  config = gemma2_2b()
  return dataclasses.replace(
      config,
      dataset_name='c4.vb100864_openmix_v1',
      seq_len=2048,  # 4096 // 2
      vocab_size=100_864,
      init_ckpt_dir='',
      use_validation_set=False,
      num_train_steps=45_000,
  )


@ExperimentConfigRegistry.register
def gemma2_9b():
  # number of parameters: ~9.24B
  config = gemma2_2b()
  return dataclasses.replace(
      config,
      model_dim=3584,
      per_head_dim=256,
      n_heads=16,
      n_layers=42,
      expand_factor=4,
      n_kv_heads=8,
      # Config for init from existing checkpoint.
      init_ckpt_dir=GEMMA2_9B_PT_CKPT_DIR,
      init_ckpt_format='Gemma3pFormat',
  )


@ExperimentConfigRegistry.register
def gemma2_27b():
  # number of parameters: ~27.23B
  config = gemma2_2b()
  return dataclasses.replace(
      config,
      model_dim=4608,
      per_head_dim=128,
      n_heads=32,
      n_layers=46,
      expand_factor=8,
      n_kv_heads=16,
      batch_size=256,
      query_scale=math.sqrt(4608 / 32),
      # Config for init from existing checkpoint.
      init_ckpt_dir=GEMMA2_27B_PT_CKPT_DIR,
      init_ckpt_format='Gemma3pFormat',
  )


@ExperimentConfigRegistry.register
def gemma2_2b_it():
  config = gemma2_2b()
  return dataclasses.replace(
      config,
      init_ckpt_dir=GEMMA2_2B_IT_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def gemma3_1b():
  config = BaseExperimentConfig()
  return dataclasses.replace(
      config,
      seq_len=4096,
      vocab_size=262144,
      model_dim=1152,
      per_head_dim=256,
      n_heads=4,
      n_layers=26,
      expand_factor=6,
      use_scan=True,
      use_remat=True,
      model_seed=42,
      use_rmsnorm=True,
      use_pre_ln=True,
      use_post_ln=True,
      use_post_skip_ln=False,
      use_qk_norm=True,
      use_per_dim_scale=False,
      use_gated_activation_in_ffn=True,
      activation_dtype_name='bfloat16',
      use_flash_attention=False,
      window_size=512 - 1,
      use_window_chunk=False,
      n_kv_heads=1,
      block_attn_pattern=(
          'local',
          'local',
          'local',
          'local',
          'local',
          'global',
      ),
      output_layer_use_bias=False,
      ffn_use_bias=False,
      local_rope_max_timescale=10_000,
      global_rope_max_timescale=1_000_000,
      attn_soft_cap=-1.0,
      output_logits_soft_cap=-1.0,
      # NOTE: Data config is vocab dependent. We currently do not have dataset
      # prepared with Gemma3 vocab.
      vocab_name='vb262144_gemma3',
      # Config for init from existing checkpoint.
      init_ckpt_dir=GEMMA3_1B_PT_CKPT_DIR,
      init_ckpt_step=-1,
      init_ckpt_opt_state=False,
      init_ckpt_format='Gemma3pFormat',
      reset_steps=True,
  )


@ExperimentConfigRegistry.register
def gemma3_270m():
  config = gemma3_1b()
  return dataclasses.replace(
      config,
      model_dim=640,
      ffn_expand_dim=2048,
      n_layers=18,
      init_ckpt_dir=GEMMA3_270M_PT_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def gemma3_4b():
  config = gemma3_1b()
  return dataclasses.replace(
      config,
      model_dim=2560,
      expand_factor=4,
      n_layers=34,
      n_heads=8,
      n_kv_heads=4,
      window_size=1024 - 1,
      global_rope_scale_factor=8.0,
      init_ckpt_dir=GEMMA3_4B_PT_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def gemma3_12b():
  config = gemma3_1b()
  return dataclasses.replace(
      config,
      model_dim=30 * 128,
      expand_factor=4,
      n_layers=48,
      n_heads=16,
      n_kv_heads=8,
      window_size=1024 - 1,
      global_rope_scale_factor=8.0,
      init_ckpt_dir=GEMMA3_12B_PT_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def gemma3_27b():
  config = gemma3_1b()
  return dataclasses.replace(
      config,
      model_dim=5376,
      expand_factor=4,
      n_layers=62,
      per_head_dim=128,
      n_heads=32,
      n_kv_heads=16,
      window_size=1024 - 1,
      global_rope_scale_factor=8.0,
      query_scale=math.sqrt(5376 / 32),
      init_ckpt_dir=GEMMA3_27B_PT_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def gemma3_12b_it_dsr40k_b2k_l10k_rl():  # PF_4x4x8
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_bf16_v2()
  base_config = deepseek_qwen2_1p5b()
  new_base_config = gemma3_12b()
  config = apply_config_diff(config, base_config, new_base_config)
  train_batch_size = 2048
  config = dataclasses.replace(
      config,
      train_batch_size=train_batch_size,
      train_max_seq_len=1024 * 10,
      sampling_prefill_size=1024 * 2,
      sampling_intermediate_decode_steps=1024 * 2,
      batch_size=32,  # number of prompts
      num_samples_per_example=16,
      grad_accum_steps=8,
      use_flash_attention=True,
      flash_attention_block_size=512,
      use_window_chunk=False,
      lm_format_name='GeminiChat',
      vocab_name='gemini3_v6',
      init_ckpt_dir=GEMMA3_12B_IT_CKPT_DIR,
      tb_log_interval=1,
      ckpt_interval=5,
      use_validation_set=False,
  )
  return config


@ExperimentConfigRegistry.register
def gemma3_27b_it_dsr40k_b2k_l10k_rl():  # VF_4x4x8
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v3()
  base_config = deepseek_qwen2_1p5b()
  new_base_config = gemma3_27b()
  config = apply_config_diff(config, base_config, new_base_config)
  config = dataclasses.replace(
      config,
      train_max_seq_len=1024 * 10,
      sampling_prefill_size=1024 * 2,
      sampling_intermediate_decode_steps=1024 * 2,
      train_batch_size=2048,
      batch_size=64,  # number of prompts
      num_samples_per_example=16,
      grad_accum_steps=4,
      use_flash_attention=True,
      flash_attention_block_size=512,
      use_window_chunk=False,
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
      lm_format_name='GemmaV2Chat',
      init_ckpt_dir=GEMMA3_27B_IT_CKPT_DIR,
      tb_log_interval=1,
      ckpt_interval=5,
  )
  return config


@ExperimentConfigRegistry.register
def gemma2_2b_gsm8k_0shot_rl():
  config = gemma2_2b()
  return dataclasses.replace(
      config,
      dataset_name='simply_json:gsm8k_train',
      num_train_steps=1_000_000,
      train_loop_name='rl',
      evaluation=evaluation_lib.ZeroShotBoxedInQuestionEvaluation(),
      use_validation_set=True,
      validation_num_eval_steps=8,
      validation_eval_interval=100,
      validation_dataset_name='simply_json:gsm8k_test',
      validation_eval_batch_size=-1,
      validation_eval_epochs=1,
      lm_format_name='Pretrain',
      train_batch_size=16 * 8,
      batch_size=16,
      num_samples_per_example=8,
      sampling_temperature=1.0,
      # Use train_max_seq_len to control the max decode steps.
      sampling_max_decode_steps=32768,
      train_max_seq_len=2048,
      sampling_prefill_size=1024,
      sampling_max_input_len=1024,
      sampling_intermediate_decode_steps=1024,
      num_train_steps_per_batch=4,
      max_num_samples_per_train_batch=None,
      # TODO: Change the extra_eos_tokens when the prompt is improved.
      extra_eos_tokens=newlines_from_counts(range(3, 6)),
      # RL algorithm configs.
      gamma=1.0,
      kl_coeff=0.001,
      use_grpo=True,
      ppo_clip_eps=0.2,
      ppo_clip_eps_high=None,
      ppo_clip_eps_low=None,
      policy_ratio_cap=10.0,
      normalize_reward_method='ByGroup',
      normalize_advantage=False,
      max_abs_advantage=10.0,
      filter_truncated=False,
      use_policy_logp_as_sampler_logp=False,
      # Optimizer configs.
      optimizer=opt_lib.Adam(beta1=0.9, beta2=0.95, epsilon=1e-8),
      weight_decay=0.0,
      lr=opt_lib.LinearWarmupCosineDecay(
          value=1e-7,
          warmup_steps=1,
      ),
      # Checkpoint and tensorboard configs.
      init_ckpt_opt_state=False,
      ckpt_max_to_keep=1,
      tb_log_interval=20,
      ckpt_interval=100,
      # Sharding config.
      decoding_sharding_config=DecodeGSPMDSharding(),
  )


@ExperimentConfigRegistry.register
def gemma2_2b_gsm8k_cot_0shot_rl():
  config = gemma2_2b_gsm8k_0shot_rl()
  return dataclasses.replace(
      config,
      evaluation=evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation(),
  )


@ExperimentConfigRegistry.register
def gemma2_2b_dsr40k_0shot_rl():
  config = gemma2_2b_gsm8k_0shot_rl()
  return dataclasses.replace(
      config,
      dataset_name='simply_json:dsr40k_train',
  )


@ExperimentConfigRegistry.register
def gemma2_2b_dsr40k_cot_0shot_rl():
  config = gemma2_2b_dsr40k_0shot_rl()
  return dataclasses.replace(
      config,
      evaluation=evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation(),
  )


@ExperimentConfigRegistry.register
def gemma2_2b_gsm8k_seqlen2k_rl():
  config = gemma2_2b_gsm8k_0shot_rl()
  return dataclasses.replace(
      config,
      # Use train_max_seq_len to control the max decode steps.
      sampling_max_decode_steps=32768,
      train_max_seq_len=2048,
      sampling_prefill_size=1024,
      sampling_max_input_len=1024,
      num_train_steps_per_batch=4,
  )


@ExperimentConfigRegistry.register
def gemma2_2b_gsm8k_seqlen2k_bs16x16_rl():
  config = gemma2_2b_gsm8k_seqlen2k_rl()
  return dataclasses.replace(
      config,
      batch_size=16,
      num_samples_per_example=16,
      tb_log_interval=8,
      ckpt_interval=40,
  )


@ExperimentConfigRegistry.register
def gemma2_2b_gsm8k_seqlen2k_bs16x8_rl():
  config = gemma2_2b_gsm8k_seqlen2k_rl()
  return dataclasses.replace(
      config,
      batch_size=16,
      num_samples_per_example=8,
      tb_log_interval=20,
      ckpt_interval=100,
  )


@ExperimentConfigRegistry.register
def gemma2_2b_gsm8k_seqlen2k_bs32x16_rl():
  config = gemma2_2b_gsm8k_seqlen2k_rl()
  return dataclasses.replace(
      config,
      # Feasible setup: glp_2x4
      batch_size=32,
      num_samples_per_example=16,
      tb_log_interval=4,
      ckpt_interval=20,
      use_flash_attention=True,
      grad_accum_steps=2,
  )


@ExperimentConfigRegistry.register
def gemma2_2b_gsm8k_32examples_rl():
  config = gemma2_2b()
  return dataclasses.replace(
      config,
      dataset_name='simply_json:gsm8k_train32',
  )


@ExperimentConfigRegistry.register
def gemma2_2b_it_gsm8k_0shot_rl():
  """Gemma 2B IT model for GSM8K RL."""
  config = gemma2_2b_gsm8k_0shot_rl()
  return dataclasses.replace(
      config,
      lm_format_name='GemmaV2Chat',
      extra_eos_tokens=(),
      evaluation=evaluation_lib.ZeroShotBoxedInQuestionEvaluation(),
      init_ckpt_dir=GEMMA2_2B_IT_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def gemma2_2b_it_gsm8k_cot_0shot_rl():
  config = gemma2_2b_it_gsm8k_0shot_rl()
  return dataclasses.replace(
      config,
      evaluation=evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation(),
  )


@ExperimentConfigRegistry.register
def gemma2_2b_it_dsr40k_0shot_rl():
  config = gemma2_2b_it_gsm8k_0shot_rl()
  return dataclasses.replace(
      config,
      dataset_name='simply_json:dsr40k_train',
  )


@ExperimentConfigRegistry.register
def gemma2_2b_it_dsr40k_cot_0shot_rl():
  config = gemma2_2b_it_dsr40k_0shot_rl()
  return dataclasses.replace(
      config,
      evaluation=evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation(),
  )


@ExperimentConfigRegistry.register
def gemma3_4b_it_simple_qa_number_only_tool_use_rl():
  config = gemma3_4b()
  return dataclasses.replace(
      config,
      # Model config.
      init_ckpt_dir=GEMMA3_4B_IT_CKPT_DIR,
      lm_format_name='GemmaV2Chat',
      # Dataset & evaluation config.
      dataset_name='simply_json:simple_qa_num',
      evaluation=evaluation_lib.QAToolUseEvaluation(),
      # Tool config.
      tool_manager_name='GoogleSearchToolExecutor',
      max_turns=3,
      filter_throttled=True,
      # Sampling configs.
      sampling_max_decode_steps=512,
      train_max_seq_len=2048,
      sampling_prefill_size=512,
      sampling_max_input_len=512,
      sampling_temperature=1.0,
      sampling_intermediate_decode_steps=512,
      sampling_max_tool_response_len=1024,
      # TODO: Change the extra_eos_tokens when the prompt is improved.
      extra_eos_tokens=newlines_from_counts(range(3, 6)),
      # RL algorithm configs.
      train_loop_name='rl',
      num_train_steps=500,
      num_train_steps_per_batch=4,
      max_num_samples_per_train_batch=None,
      use_validation_set=False,
      train_batch_size=16 * 32,
      batch_size=32,
      grad_accum_steps=8,
      num_samples_per_example=4,
      gamma=1.0,
      kl_coeff=0.001,
      use_grpo=True,
      ppo_clip_eps=0.2,
      ppo_clip_eps_high=None,
      ppo_clip_eps_low=None,
      policy_ratio_cap=10.0,
      normalize_reward_method='ByGroup',
      normalize_advantage=False,
      max_abs_advantage=10.0,
      use_kl_correction=True,
      filter_zero_variance=True,
      filter_truncated=False,
      use_policy_logp_as_sampler_logp=False,
      soft_ppo_forward_kl_coeff=0.0,
      soft_ppo_reverse_kl_coeff=0.0,
      # Optimizer configs.
      optimizer=opt_lib.Adam(beta1=0.9, beta2=0.95, epsilon=1e-8),
      weight_decay=0.0,
      lr_schedule_name='constant',
      lr_schedule_config=(
          ('lr', 1e-6),
          ('warmup_steps', 100),
      ),
      # bf16 config from Gemma2BV2.
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
      # Checkpoint and tensorboard configs.
      init_ckpt_opt_state=False,
      ckpt_max_to_keep=1,
      tb_log_interval=10,
      ckpt_interval=100,
      # Sharding config.
      decoding_sharding_config=DecodeGSPMDSharding(),
  )


#################################################################################
## Qwen2 models.
## https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b():
  config = BaseExperimentConfig()
  return dataclasses.replace(
      config,
      # number of parameters: ~1.5B
      seq_len=4096,
      vocab_size=151936,
      model_dim=1536,
      expand_factor=0,
      ffn_expand_dim=8960,
      per_head_dim=128,
      n_heads=12,
      n_layers=28,
      n_kv_heads=2,
      ffn_activation='silu',
      use_post_ln=False,
      use_per_dim_scale=False,
      ffn_use_bias=False,
      qkv_use_bias=True,
      output_layer_use_bias=False,
      use_tied_embedding=False,
      use_combined_qkv=False,
      embedding_lookup_scale=None,
      norm_scale_plus_one=False,
      attn_soft_cap=-1.0,
      output_logits_soft_cap=-1.0,
      # NOTE: Data config is vocab dependent. We currently do not have dataset
      # prepared with qwen vocab.
      vocab_name='DeepSeek-R1-Distill-Qwen',
      # Config for init from existing checkpoint.
      init_ckpt_dir=DEEPSEEK_QWEN_1P5B_CKPT_DIR,
      init_ckpt_step=-1,
      init_ckpt_opt_state=False,
      reset_steps=True,
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl():
  config = deepseek_qwen2_1p5b()
  return dataclasses.replace(
      config,
      dataset_name='simply_json:dsr40k_train',
      num_train_steps=1_000_000,
      train_loop_name='rl',
      train_batch_size=16 * 8,
      batch_size=16,
      num_samples_per_example=8,
      sampling_temperature=1.0,
      num_train_steps_per_batch=4,
      # RL algorithm configs.
      gamma=1.0,
      kl_coeff=0.001,
      use_grpo=True,
      ppo_clip_eps=0.2,
      ppo_clip_eps_high=None,
      ppo_clip_eps_low=None,
      policy_ratio_cap=10.0,
      normalize_reward_method='ByGroup',
      normalize_advantage=False,
      max_abs_advantage=10.0,
      use_policy_logp_as_sampler_logp=False,
      filter_truncated=False,
      max_num_samples_per_train_batch=None,
      # Optimizer configs.
      optimizer=opt_lib.Adam(beta1=0.9, beta2=0.95, epsilon=1e-8),
      weight_decay=0.0,
      lr_schedule_name='constant',
      lr_schedule_config=(
          ('lr', 1e-6),
          ('warmup_steps', 1),
      ),
      # Checkpoint and tensorboard configs.
      init_ckpt_opt_state=False,
      ckpt_max_to_keep=1,
      tb_log_interval=4,
      ckpt_interval=4,
      # Sharding config.
      decoding_sharding_config=DecodeGSPMDSharding(),
      # Use train_max_seq_len to control the max decode steps.
      sampling_max_decode_steps=32768,
      train_max_seq_len=9 * 1024,
      sampling_prefill_size=1024,
      sampling_max_input_len=1024,
      sampling_intermediate_decode_steps=1024,
      lm_format_name='DeepSeekQwenR1DistillChat',
      evaluation=evaluation_lib.ZeroShotDeepSeekQwenR1CoTBoxed(),
      extra_eos_tokens=(),
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
      use_flash_attention=True,
      flash_attention_block_size=512,
      use_validation_set=True,
      validation_eval_interval=50,
      validation_dataset_name='simply_json:aime24',
      validation_eval_batch_size=64,
      validation_eval_epochs=5,
      validation_evaluation=None,
      validation_lm_format_name='',
      validation_max_decode_steps=None,
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32():
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl()
  return dataclasses.replace(
      config,
      activation_dtype_name='float32',
      decoding_quant_scheme='float32',
      ref_params_dtype='float32',
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2():
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32()
  return dataclasses.replace(
      config,
      # Batch size and gradient accumulation configs.
      train_batch_size=64 * 8,
      batch_size=64,
      num_samples_per_example=8,
      grad_accum_steps=4,
      # Checkpoint and tensorboard logging configs.
      tb_log_interval=1,
      ckpt_interval=20,
      ckpt_max_to_keep=1,
      # Flash attention configs.
      use_flash_attention=True,
      flash_attention_block_size=512,
      # RL algorithm configs.
      num_train_steps_per_batch=1,
      normalize_reward_method='ByGroup',
      policy_ratio_cap=None,
      max_abs_advantage=None,
      lr_schedule_name='constant',
      lr_schedule_config=(
          ('lr', 1e-6),
          ('warmup_steps', 1),
      ),
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_v2():
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2()
  return dataclasses.replace(
      config,
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v3():
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2()
  return dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 3e-6),
          ('warmup_steps', 1),
      ),
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_bf16_v2():
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2()
  return dataclasses.replace(
      config,
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
      use_validation_set=True,
      validation_eval_batch_size=64,
      validation_eval_interval=50,
      use_policy_logp_as_sampler_logp=True,
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v4():
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2()
  return dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 1e-5),
          ('warmup_steps', 1),
      ),
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_t0p6():
  config = deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl()
  return dataclasses.replace(
      config,
      activation_dtype_name='float32',
      decoding_quant_scheme='float32',
      ref_params_dtype='float32',
      sampling_temperature=0.6,
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_7b():
  config = deepseek_qwen2_1p5b()
  return dataclasses.replace(
      config,
      vocab_size=152064,
      model_dim=3584,
      ffn_expand_dim=18944,
      n_layers=28,
      n_heads=28,
      n_kv_heads=4,
      init_ckpt_dir=DEEPSEEK_QWEN_7B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_14b():
  config = deepseek_qwen2_1p5b()
  return dataclasses.replace(
      config,
      vocab_size=152064,
      model_dim=5120,
      ffn_expand_dim=13824,
      n_layers=48,
      n_heads=40,
      n_kv_heads=8,
      global_rope_max_timescale=1_000_000,
      rms_norm_epsilon=1e-5,
      init_ckpt_dir=DEEPSEEK_QWEN_14B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def deepseek_qwen2_32b():
  config = deepseek_qwen2_1p5b()
  return dataclasses.replace(
      config,
      vocab_size=152064,
      model_dim=5120,
      ffn_expand_dim=27648,
      n_layers=64,
      n_heads=40,
      n_kv_heads=8,
      global_rope_max_timescale=1_000_000,
      rms_norm_epsilon=1e-5,
      init_ckpt_dir=DEEPSEEK_QWEN_32B_CKPT_DIR,
  )


# TODO: The ideal way should be first define Qwen native configs and use
# them to define DeepSeek Qwen configs.
@ExperimentConfigRegistry.register
def qwen_math_1p5b_v2p5():
  config = deepseek_qwen2_1p5b()
  return dataclasses.replace(
      config,
      use_tied_embedding=True,
      vocab_name='Qwen2.5',
      init_ckpt_dir=QWEN2p5_MATH_1p5B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwen_math_7b_v2p5():
  config = deepseek_qwen2_7b()
  return dataclasses.replace(
      config,
      init_ckpt_dir=QWEN2p5_MATH_7B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwen_math_14b_v2p5():
  config = deepseek_qwen2_14b()
  return dataclasses.replace(
      config,
      init_ckpt_dir=QWEN2p5_MATH_14B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwen_math_32b_v2p5():
  config = deepseek_qwen2_32b()
  return dataclasses.replace(
      config,
      init_ckpt_dir=QWEN2p5_MATH_32B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwq_32b():
  config = deepseek_qwen2_32b()
  return dataclasses.replace(
      config,
      vocab_name='QwQ',
      init_ckpt_dir=QWQ_32B_CKPT_DIR,
  )


def apply_config_diff(config, diff_base, diff_new):
  updates = {}
  for field in dataclasses.fields(diff_base):
    name = field.name
    base_val = getattr(diff_base, name)
    new_val = getattr(diff_new, name)
    if base_val != new_val:
      updates[name] = new_val
  return dataclasses.replace(config, **updates)


@ExperimentConfigRegistry.register
def deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl():
  return apply_config_diff(
      config=deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl(),
      diff_base=deepseek_qwen2_1p5b(),
      diff_new=deepseek_qwen2_7b())


@ExperimentConfigRegistry.register
def deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2():
  config = apply_config_diff(
      config=deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2(),
      diff_base=deepseek_qwen2_1p5b(),
      diff_new=deepseek_qwen2_7b())
  config = dataclasses.replace(config, batch_size=32)
  return config


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2,
    name='dsqwen_v2-7b-dsr40k-R1_distill_cot0shot-rl-f32-v2')


@ExperimentConfigRegistry.register
def deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v3():
  config = deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2()
  config = dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 3e-6),
          ('warmup_steps', 1),
      ),
  )
  return config


@ExperimentConfigRegistry.register
def deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_bf16_v2():
  config = dataclasses.replace(
      deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2(),
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
      use_validation_set=True,
      validation_eval_interval=50,
      validation_eval_batch_size=32,
      use_policy_logp_as_sampler_logp=True,
  )
  return config


@ExperimentConfigRegistry.register
def deepseek_qwen2_14b_it_dsr40k_r1_distill_cot_0shot_rl():
  return apply_config_diff(
      config=deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl(),
      diff_base=deepseek_qwen2_1p5b(),
      diff_new=deepseek_qwen2_14b())


@ExperimentConfigRegistry.register
def deepseek_qwen2_14b_it_dsr40k_r1_distill_cot_0shot_rl_v2():
  config = apply_config_diff(
      config=deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_v2(),
      diff_base=deepseek_qwen2_1p5b(),
      diff_new=deepseek_qwen2_14b())
  config = dataclasses.replace(config, batch_size=16)
  return config


@ExperimentConfigRegistry.register
def deepseek_qwen2_14b_it_dsr40k_r1_distill_cot_0shot_rl_v3():
  config = deepseek_qwen2_14b_it_dsr40k_r1_distill_cot_0shot_rl_v2()
  config = dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 3e-6),
          ('warmup_steps', 1),
      ),
  )
  return config


@ExperimentConfigRegistry.register
def deepseek_qwen2_32b_it_dsr40k_r1_distill_cot_0shot_rl():
  return apply_config_diff(
      config=deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl(),
      diff_base=deepseek_qwen2_1p5b(),
      diff_new=deepseek_qwen2_32b())


@ExperimentConfigRegistry.register
def deepseek_qwen2_32b_it_dsr40k_r1_distill_cot_0shot_rl_v2():
  config = apply_config_diff(
      config=deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_v2(),
      diff_base=deepseek_qwen2_1p5b(),
      diff_new=deepseek_qwen2_32b())
  config = dataclasses.replace(config, batch_size=16)
  return config


@ExperimentConfigRegistry.register
def deepseek_qwen2_32b_it_dsr40k_r1_distill_cot_0shot_rl_v3():
  config = deepseek_qwen2_32b_it_dsr40k_r1_distill_cot_0shot_rl_v2()
  config = dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 3e-6),
          ('warmup_steps', 1),
      ),
  )
  return config


#################################################################################
## Qwen3 models.
## https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3


@ExperimentConfigRegistry.register
def qwen3_0p6b():
  config = BaseExperimentConfig()
  return dataclasses.replace(
      config,
      seq_len=4096,
      vocab_size=151936,
      model_dim=1024,
      expand_factor=0,
      ffn_expand_dim=3072,
      per_head_dim=128,
      n_heads=16,
      n_layers=28,
      n_kv_heads=8,
      ffn_activation='silu',
      use_post_ln=False,
      use_qk_norm=True,
      use_per_dim_scale=False,
      ffn_use_bias=False,
      qkv_use_bias=False,
      output_layer_use_bias=False,
      use_tied_embedding=True,
      use_combined_qkv=False,
      embedding_lookup_scale=None,
      norm_scale_plus_one=False,
      attn_soft_cap=-1.0,
      output_logits_soft_cap=-1.0,
      global_rope_max_timescale=1_000_000,
      # NOTE: Data config is vocab dependent. We currently do not have dataset
      # prepared with qwen vocab.
      vocab_name='Qwen3',
      # Config for init from existing checkpoint.
      init_ckpt_dir=QWEN3_0P6B_CKPT_DIR,
      init_ckpt_step=-1,
      init_ckpt_opt_state=False,
      reset_steps=True,
  )


@ExperimentConfigRegistry.register
def qwen3_1p7b():
  config = qwen3_0p6b()
  return dataclasses.replace(
      config,
      model_dim=2048,
      ffn_expand_dim=6144,
      init_ckpt_dir=QWEN3_1P7B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwen3_4b():
  config = qwen3_0p6b()
  return dataclasses.replace(
      config,
      model_dim=2560,
      ffn_expand_dim=9728,
      n_heads=32,
      n_layers=36,
      init_ckpt_dir=QWEN3_4B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwen3_8b():
  config = qwen3_0p6b()
  return dataclasses.replace(
      config,
      model_dim=4096,
      ffn_expand_dim=12288,
      n_heads=32,
      n_layers=36,
      use_tied_embedding=False,
      init_ckpt_dir=QWEN3_8B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwen3_14b():
  config = qwen3_0p6b()
  return dataclasses.replace(
      config,
      model_dim=5120,
      ffn_expand_dim=17408,
      n_heads=40,
      n_layers=40,
      use_tied_embedding=False,
      init_ckpt_dir=QWEN3_14B_CKPT_DIR,
  )


@ExperimentConfigRegistry.register
def qwen3_32b():
  config = qwen3_0p6b()
  return dataclasses.replace(
      config,
      model_dim=5120,
      ffn_expand_dim=25600,
      n_heads=64,
      n_layers=64,
      use_tied_embedding=False,
      init_ckpt_dir=QWEN3_32B_CKPT_DIR,
  )


################################################################################
## Tiny experiments for tests.


@ExperimentConfigRegistry.register
def lm_test():
  config = BaseExperimentConfig()
  return dataclasses.replace(
      config,
      # Model config
      model_dim=8,
      per_head_dim=4,
      n_heads=2,
      n_layers=2,
      expand_factor=2,
      use_scan=True,
      use_flash_attention=False,
      activation_dtype_name='bfloat16',
      # Data config
      num_train_steps=50,
      batch_size=4,
      vocab_size=32_000,
      seq_len=64,
      prefetch_num_workers=0,
      prefetch_per_worker_buffer_size=2,
      dataset_name='imdb_reviews.vb32000_t5_cc',
      lr=opt_lib.LinearWarmupCosineDecay(
          value=1e-3,
          warmup_steps=10,
          steps_after_decay=10,
          end_decay=0.1,
      ),
      clip_grad_norm=-1.0,
      clip_update_norm=-1.0,
      use_validation_set=True,
      validation_num_eval_steps=2,
      validation_eval_interval=5,
      validation_eval_batch_size=-1,
      # Checkpoint and tensorboard config
      ckpt_interval=10,
      ckpt_max_to_keep=3,
      tb_log_interval=2,
  )


@ExperimentConfigRegistry.register
def lm_c4_test():
  config = lm_test()
  return dataclasses.replace(
      config,
      seq_len=4096,
      dataset_name='c4.vb100864_openmix_v1',
  )


@ExperimentConfigRegistry.register
def lm_no_scan_test():
  config = lm_test()
  return dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
  )


@ExperimentConfigRegistry.register
def lm_sft_test():
  config = lm_test()
  return dataclasses.replace(
      config,
      num_train_steps=2000,
      lr_schedule_config=(
          ('lr', 1e-4),
          ('warmup_steps', 100),
          ('steps_after_decay', 10),
          ('end_decay', 0.1),
      ),
      dataset_name='tulu_v2_sft.vb100864_openmix_v1',
      # Config for init from existing checkpoint.
      init_ckpt_dir='/tmp/simply_test_pt_1/checkpoints',
      init_ckpt_step=-1,
      use_validation_set=False,
      # Add masks to only calculate loss on assistant responses.
      add_chat_loss_mask=True,
      mask_start_token='<reserved_2>',
      mask_end_token='<reserved_4>',
  )


@ExperimentConfigRegistry.register
def lm_rl_test():
  config = lm_test()
  return dataclasses.replace(
      config,
      dataset_name='simply_json:dsr40k_train',
      num_train_steps=30,
      train_loop_name='rl',
      evaluation=evaluation_lib.ZeroShotBoxedInQuestionEvaluation(),
      use_validation_set=False,
      vocab_name='vb32768_openmix_v1',
      lm_format_name='SimplyV1Chat',
      batch_size=4,
      train_batch_size=4,
      num_samples_per_example=1,
      sampling_temperature=1.0,
      filter_truncated=False,
      max_num_samples_per_train_batch=None,
      # Use train_max_seq_len to control the max decode steps.
      sampling_max_decode_steps=32768,
      train_max_seq_len=8,
      sampling_prefill_size=16,
      sampling_max_input_len=8,
      sampling_intermediate_decode_steps=-1,
      sampling_microbatch_size=2,
      num_train_steps_per_batch=4,
      lr_schedule_name='constant',
      lr_schedule_config=(
          ('lr', 1e-7),
          ('warmup_steps', 100),
      ),
      extra_eos_tokens=newlines_from_counts(range(1, 2)),
      decoding_sharding_config=DecodeGSPMDSharding(),
      # RL algorithm configs.
      gamma=1.0,
      kl_coeff=0.01,
      use_grpo=True,
      ppo_clip_eps=0.2,
      ppo_clip_eps_low=None,
      ppo_clip_eps_high=None,
      policy_ratio_cap=10.0,
      normalize_reward_method='ByGroup',
      normalize_advantage=False,
      max_abs_advantage=10.0,
      use_policy_logp_as_sampler_logp=False,
      activation_dtype_name='float32',
      decoding_quant_scheme='float32',
      ref_params_dtype='float32',
      grad_accum_steps=2,
      # Early stopping threshold.
      early_stop=opt_lib.SimpleEarlyStop(
          thresholds=(
              (20, ('<', 'accuracy', 0.5)),
          )
      ),
  )


def get_default_mesh_shape(
    config: BaseExperimentConfig, mode: str = 'train',
    dcn_mesh_shape=None) -> list[int]:
  """Returns the default mesh shape."""
  if dcn_mesh_shape is None:
    num_slices = 1
  else:
    num_slices = math.prod(dcn_mesh_shape)
    # DCN mesh should only be used for replica parallelism.
    assert dcn_mesh_shape == [num_slices, 1, 1]
  device_count = jax.device_count()
  device_count //= num_slices
  if mode == 'train':
    # Do fully sharded data and replicaparallel for training.
    data_parallel = math.gcd(config.model_dim, device_count)
    replica_parallel = device_count // data_parallel
    # RL uses train_batch_size, while pretraining uses batch_size.
    train_batch_size = (
        getattr(config, 'train_batch_size', None) or config.batch_size
    )
    if config.grad_accum_steps > 0:
      if train_batch_size % config.grad_accum_steps != 0:
        raise ValueError(
            f'Training requires {train_batch_size=} to be divisible by '
            f'{config.grad_accum_steps=}.'
        )
      train_batch_size //= config.grad_accum_steps
    if train_batch_size % (data_parallel * replica_parallel) != 0:
      raise ValueError(
          f'Training requires {train_batch_size=} to be divisible by '
          f'{data_parallel=} * {replica_parallel=}.'
      )
    return [replica_parallel, data_parallel, 1]
  elif mode == 'decode':
    # Do model parallelism as much as possible and replica parallelism for the
    # rest.
    model_parallel = functools.reduce(
        math.gcd,
        [config.model_dim, config.n_kv_heads, config.n_heads, device_count])
    replica_parallel = device_count // model_parallel
    decode_batch_size = config.batch_size * getattr(
        config, 'num_samples_per_example', 1)
    if decode_batch_size % replica_parallel != 0:
      raise ValueError(
          f'Decoding requires {decode_batch_size=} to be divisible by '
          f'{replica_parallel=}.'
      )
    return [replica_parallel, 1, model_parallel]
  else:
    raise ValueError(f'Unsupported mode: {mode}')

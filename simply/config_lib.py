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
from simply.utils import optimizer as opt_lib
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

GEMMA3_1B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-1B-PT-ORBAX')
GEMMA3_4B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-4B-PT-ORBAX')
GEMMA3_12B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-12B-PT-ORBAX')
GEMMA3_27B_PT_CKPT_DIR = os.path.join(MODELS_DIR, 'GEMMA-3.0-27B-PT-ORBAX')
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
class TFM1p7BLM1B(ExperimentConfig):
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

  # Name for the model, i.e., the main module.
  model_name: str = 'TransformerLM'

  # TODO: The type should be model_lib.InputEncoderInterface, but we
  # need to resolve some cyclic dependency issue.
  input_encoders: list[Any] = dataclasses.field(default_factory=list)

  # Utilities for patching code snippets before running an experiment.
  code_patch: tuple[tuple[str, str], ...] = ()

  # Early stopping threshold.
  early_stop: opt_lib.EarlyStop | None = None

# TODO: consider renaming BaseExperimentConfig since ExperimentConfig
# is the real base class.
BaseExperimentConfig = TFM1p7BLM1B


################################################################################
## C4 experiments.


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops6e20TFM2BC4L2048(TFM1p7BLM1B):
  # num_params: 2321.380352 M
  # num_non_embedding_params: 2114.81088 M
  # num_embedding_params: 206.569472 M
  # embedding_params_ratio: 0.08898562091387943
  # num_tokens: 44755.320832 M
  # num_tokens / num_params: 19.27961559312793
  # num_tokens / num_non_embedding_params: 21.162800539403314
  # num_flops: 6.233647345611666e+20
  # Fitted optimal ratio for 6.2e20: 19.24
  model_dim: int = 2048
  per_head_dim: int = 256
  n_heads: int = 8
  n_layers: int = 18
  expand_factor: int = 8
  seq_len: int = 2048
  vocab_size: int = 100_864

  # 2321380352 * 19.24 / 2048 / 1024 = 21297 steps
  dataset_name: str = 'c4.vb100864_openmix_v1'
  batch_size: int = 1024
  clip_grad_norm: float = 1.0
  num_train_steps: int = 21_297
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 0.3
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-3),
      ('warmup_steps', 1_000),
      ('steps_after_decay', 0),
      ('end_decay', 0.1),
  )

  ckpt_max_to_keep: int = 1

  use_validation_set: bool = True
  validation_num_eval_steps: int = 2
  validation_eval_interval: int = 1000
  validation_eval_batch_size: int = 1024


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops1e20TFM986MC4L2048(Flops6e20TFM2BC4L2048):
  # num_params: 985.890304 M
  # num_non_embedding_params: 830.9632 M
  # num_embedding_params: 154.927104 M
  # embedding_params_ratio: 0.15714436319276348
  # num_tokens: 18405.654528 M
  # num_tokens / num_params: 18.669069422149423
  # num_tokens / num_non_embedding_params: 22.14978296030438
  # num_flops: 1.0887573802757338e+20
  # Fitted optimal ratio for 1.1e20: 18.67
  model_dim: int = 1536
  per_head_dim: int = 256
  n_heads: int = 8
  n_layers: int = 12
  expand_factor: int = 8
  vocab_size: int = 100_864

  # 985890304 * 18.67 / 2048 / 256 = 35106 steps
  batch_size: int = 256
  num_train_steps: int = 35_106
  weight_decay: float = 0.3
  lr_schedule_name: str = 'cosine_decay'
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-3),
      ('warmup_steps', 1_000),
      ('steps_after_decay', 0),
      ('end_decay', 0.1),
  )

  use_validation_set: bool = True
  validation_num_eval_steps: int = 4
  validation_eval_interval: int = 1000
  validation_eval_batch_size: int = 512


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops1e19TFM338MC4L2048(Flops6e20TFM2BC4L2048):
  # num_params: 338.440192 M
  # num_non_embedding_params: 235.155456 M
  # num_embedding_params: 103.284736 M
  # embedding_params_ratio: 0.3051786946155615
  # num_tokens: 6149.89824 M
  # num_tokens / num_params: 18.171299938276835
  # num_tokens / num_non_embedding_params: 26.15247948999321
  # num_flops: 1.2488236446756372e+19
  # Fitted optimal ratio for 1.2e19: 17.97
  model_dim: int = 1024
  per_head_dim: int = 128
  n_heads: int = 8
  n_layers: int = 8
  expand_factor: int = 8
  vocab_size: int = 100_864

  # 338440192 * 17.97 / 2048 / 192 = 15466 steps
  batch_size: int = 192
  num_train_steps: int = 15_466
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 0.276
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 0.0013656918867398535),
      ('steps_after_decay', 0),
      ('warmup_steps', 1_000),
      ('end_decay', 0.1),
  )

  use_validation_set: bool = True
  validation_num_eval_steps: int = 4
  validation_eval_interval: int = 500
  validation_eval_batch_size: int = 512


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops1e18TFM111MC4L2048(Flops6e20TFM2BC4L2048):
  # num_params: 110.550528 M
  # num_non_embedding_params: 58.90816 M
  # num_embedding_params: 51.642368 M
  # embedding_params_ratio: 0.467138139765375
  # num_tokens: 1901.068288 M
  # num_tokens / num_params: 17.19637456638832
  # num_tokens / num_non_embedding_params: 32.27173091130329
  # num_flops: 1.2609846180147364e+18
  # Predicted optimal ratio for 1.3e18: 17.2
  model_dim: int = 512
  per_head_dim: int = 64
  n_heads: int = 8
  n_layers: int = 8
  expand_factor: int = 8
  vocab_size: int = 100_864

  # 110550528 * 17.2 / 2048 / 128 = 7252 steps
  batch_size: int = 128
  num_train_steps: int = 7252
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 0.261
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 0.0016486710944803309),
      ('steps_after_decay', 0),
      ('end_decay', 0.1),
  )

  use_validation_set: bool = True
  validation_num_eval_steps: int = 8
  validation_eval_interval: int = 500
  validation_eval_batch_size: int = 256


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops2e17TFM41MC4L2048(Flops6e20TFM2BC4L2048):
  # num_params: 40.645632 M
  # num_non_embedding_params: 14.824448 M
  # num_embedding_params: 25.821184 M
  # embedding_params_ratio: 0.6352757413145895
  # num_tokens: 678.428672 M
  # num_tokens / num_params: 16.691305771798554
  # num_tokens / num_non_embedding_params: 45.764177661117635
  # num_flops: 1.6545097284216422e+17
  # Fitted optimal ratio for 1.7e17: 16.69
  model_dim: int = 256  # 2048 // 8
  per_head_dim: int = 32  # 256 // 8
  n_heads: int = 8
  n_layers: int = 8  # 18 // 2 - 1
  expand_factor: int = 8
  vocab_size: int = 100_864

  # 40645632 * 16.69 / 2048 / 80 = 4140 steps
  batch_size: int = 80
  num_train_steps: int = 4140
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 0.248
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 0.0019495171900601506),
      ('steps_after_decay', 0),
      ('end_decay', 0.1),
  )

  use_validation_set: bool = True
  validation_num_eval_steps: int = 16
  validation_eval_interval: int = 500
  validation_eval_batch_size: int = 128


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops2e16TFM15MC4L2048(Flops6e20TFM2BC4L2048):
  # num_params: 14.857408 M
  # num_non_embedding_params: 1.946816 M
  # num_embedding_params: 12.910592 M
  # embedding_params_ratio: 0.86896664
  # num_tokens: 225.050624 M
  # num_tokens / num_params: 15.1473678316
  # num_tokens / num_non_embedding_params: 116.00547628866
  # num_flops: 2.006201364854e+16
  # Data / model ratio for 2.0e16: 15.15
  model_dim: int = 128  # 2048 // 16
  per_head_dim: int = 16  # 256 // 16
  n_heads: int = 8
  n_layers: int = 4  # 18 => 4
  expand_factor: int = 8
  vocab_size: int = 100_864

  # 14.066880 * 16 / 2048 / 64 = 1717 steps
  batch_size: int = 64
  num_train_steps: int = 1717  # TODO: update the number of steps.
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 0.1
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 0.01),
      ('steps_after_decay', 0),
      ('end_decay', 0.1),
  )

  use_validation_set: bool = True
  validation_num_eval_steps: int = 16
  validation_eval_interval: int = 500
  validation_eval_batch_size: int = 128

##########################
## Gemma models.
## https://github.com/google-deepmind/gemma/blob/main/gemma/transformer.py


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2(TFM1p7BLM1B):
  # number of parameters: ~2.61B
  seq_len: int = 4096
  vocab_size: int = 256128
  model_dim: int = 2304
  per_head_dim: int = 256
  n_heads: int = 8
  n_layers: int = 26
  expand_factor: int = 4
  use_scan: bool = True
  use_remat: bool = True
  model_seed: int = 42
  use_rmsnorm: bool = True
  use_pre_ln: bool = True
  use_post_ln: bool = True
  use_post_skip_ln: bool = False
  use_per_dim_scale: bool = False
  use_gated_activation_in_ffn: bool = True
  activation_dtype_name: str = 'bfloat16'
  use_flash_attention: bool = False
  window_size: int = 4096 - 1
  use_window_chunk: bool = False
  n_kv_heads: int = 4
  block_attn_pattern: tuple[str, ...] = (
      'local',
      'global',
  )
  output_layer_use_bias: bool = False
  ffn_use_bias: bool = False

  # NOTE: Data config is vocab dependent. We currently do not have dataset
  # prepared with Gemma2 vocab.
  vocab_name: str = 'vb256128_gemma2'

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = GEMMA2_2B_PT_CKPT_DIR
  init_ckpt_step: int = -1
  init_ckpt_opt_state: bool = False
  init_ckpt_format: str = 'Gemma3pLegacyFormat'
  reset_steps: bool = True

  activation_dtype_name: str = 'bfloat16'
  decoding_quant_scheme: str = 'bfloat16'
  ref_params_dtype: str = 'bfloat16'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BC4Vocab100864L2048BS1024(Gemma2BV2):
  """Gemma 2B model with C4 vocab 100864 and seq_len 2048."""
  dataset_name: str = 'c4.vb100864_openmix_v1'
  seq_len: int = 2048  # 4096 // 2
  vocab_size: int = 100_864
  init_ckpt_dir: str = ''
  use_validation_set: bool = False
  num_train_steps: int = 45_000


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma9BV2(Gemma2BV2):
  # number of parameters: ~9.24B
  model_dim: int = 3584
  per_head_dim: int = 256
  n_heads: int = 16
  n_layers: int = 42
  expand_factor: int = 4
  n_kv_heads: int = 8

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = GEMMA2_9B_PT_CKPT_DIR
  init_ckpt_format: str = 'Gemma3pFormat'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma27BV2(Gemma2BV2):
  # number of parameters: ~27.23B
  model_dim: int = 4608
  per_head_dim: int = 128
  n_heads: int = 32
  n_layers: int = 46
  expand_factor: int = 8
  n_kv_heads: int = 16
  batch_size: int = 256
  query_scale: float = math.sqrt(model_dim / n_heads)

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = GEMMA2_27B_PT_CKPT_DIR
  init_ckpt_format: str = 'Gemma3pFormat'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2IT(Gemma2BV2):
  init_ckpt_dir: str = GEMMA2_2B_IT_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma1BV3(TFM1p7BLM1B):
  seq_len: int = 4096
  vocab_size: int = 262144
  model_dim: int = 1152
  per_head_dim: int = 256
  n_heads: int = 4
  n_layers: int = 26
  expand_factor: int = 6
  use_scan: bool = True
  use_remat: bool = True
  model_seed: int = 42
  use_rmsnorm: bool = True
  use_pre_ln: bool = True
  use_post_ln: bool = True
  use_post_skip_ln: bool = False
  use_qk_norm: bool = True
  use_per_dim_scale: bool = False
  use_gated_activation_in_ffn: bool = True
  activation_dtype_name: str = 'bfloat16'
  use_flash_attention: bool = False
  window_size: int = 512 - 1
  use_window_chunk: bool = False
  n_kv_heads: int = 1
  block_attn_pattern: tuple[str, ...] = (
      'local',
      'local',
      'local',
      'local',
      'local',
      'global',
  )
  output_layer_use_bias: bool = False
  ffn_use_bias: bool = False
  local_rope_max_timescale: int = 10_000
  global_rope_max_timescale: int = 1_000_000
  attn_soft_cap: float = -1.0
  output_logits_soft_cap: float = -1.0

  # NOTE: Data config is vocab dependent. We currently do not have dataset
  # prepared with Gemma3 vocab.
  vocab_name: str = 'vb262144_gemma3'

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = GEMMA3_1B_PT_CKPT_DIR
  init_ckpt_step: int = -1
  init_ckpt_opt_state: bool = False
  init_ckpt_format: str = 'Gemma3pFormat'
  reset_steps: bool = True


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma4BV3(Gemma1BV3):
  model_dim: int = 2560
  expand_factor: int = 4
  n_layers: int = 34
  n_heads: int = 8
  n_kv_heads: int = 4
  window_size: int = 1024 - 1
  global_rope_scale_factor: float = 8.0
  init_ckpt_dir: str = GEMMA3_4B_PT_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma12BV3(Gemma1BV3):
  model_dim: int = 30 * 128
  expand_factor: int = 4
  n_layers: int = 48
  n_heads: int = 16
  n_kv_heads: int = 8
  window_size: int = 1024 - 1
  global_rope_scale_factor: float = 8.0
  init_ckpt_dir: str = GEMMA3_12B_PT_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma27BV3(Gemma1BV3):
  model_dim: int = 5376
  expand_factor: int = 4
  n_layers: int = 62
  per_head_dim: int = 128
  n_heads: int = 32
  n_kv_heads: int = 16
  window_size: int = 1024 - 1
  global_rope_scale_factor: float = 8.0
  query_scale: float = math.sqrt(model_dim / n_heads)
  init_ckpt_dir: str = GEMMA3_27B_PT_CKPT_DIR


@ExperimentConfigRegistry.register
def Gemma12BV3_IT_DSR40K_B2K_L10K_RL():  # PF_4x4x8
  config = DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLBF16V2()
  base_config = DeepSeekQwen1p5BV2()
  new_base_config = Gemma12BV3()
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
def Gemma27BV3_IT_DSR40K_B2K_L10K_RL():  # VF_4x4x8
  config = DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V3()
  base_config = DeepSeekQwen1p5BV2()
  new_base_config = Gemma27BV3()
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
@dataclasses.dataclass(frozen=True)
class Gemma2BV2GSM8K0ShotRL(Gemma2BV2):
  dataset_name: str = 'simply_json:gsm8k_train'
  num_train_steps: int = 1_000_000
  train_loop_name: str = 'rl'
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotBoxedInQuestionEvaluation()
  )
  use_validation_set: bool = True
  validation_num_eval_steps: int = 8
  validation_eval_interval: int = 100
  validation_dataset_name: str | None = 'simply_json:gsm8k_test'
  validation_eval_batch_size: int = -1
  validation_eval_epochs: int = 1

  lm_format_name: str = 'Pretrain'
  train_batch_size: int = 16 * 8
  batch_size: int = 16
  num_samples_per_example: int = 8
  sampling_temperature: float = 1.0
  # Use train_max_seq_len to control the max decode steps.
  sampling_max_decode_steps: int = 32768
  train_max_seq_len: int = 2048
  sampling_prefill_size: int = 1024
  sampling_max_input_len: int = 1024
  sampling_intermediate_decode_steps: int = 1024
  num_train_steps_per_batch: int = 4
  max_num_samples_per_train_batch: int | None = None

  # TODO: Change the extra_eos_tokens when the prompt is improved.
  extra_eos_tokens: tuple[str, ...] = newlines_from_counts(range(3, 6))

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

  # Optimizer configs.
  optimizer: opt_lib.Optimizer = opt_lib.Adam(
      beta1=0.9, beta2=0.95, epsilon=1e-8
  )
  weight_decay: float = 0.0
  lr_schedule_name: str = 'constant'
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-7),
      ('warmup_steps', 1),
  )

  # Checkpoint and tensorboard configs.
  init_ckpt_opt_state: bool = False
  ckpt_max_to_keep: int = 1
  tb_log_interval: int = 20
  ckpt_interval: int = 100

  # Sharding config.
  decoding_sharding_config: SimplyConfig = DecodeGSPMDSharding()


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2GSM8KCoT0ShotRL(Gemma2BV2GSM8K0ShotRL):
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation()
  )


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2DSR40K0ShotRL(Gemma2BV2GSM8K0ShotRL):
  dataset_name: str = 'simply_json:dsr40k_train'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2DSR40KCoT0ShotRL(Gemma2BV2DSR40K0ShotRL):
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation()
  )


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2GSM8KSeqLen2kRL(Gemma2BV2GSM8K0ShotRL):
  # Use train_max_seq_len to control the max decode steps.
  sampling_max_decode_steps: int = 32768
  train_max_seq_len: int = 2048
  sampling_prefill_size: int = 1024
  sampling_max_input_len: int = 1024
  num_train_steps_per_batch: int = 4


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2GSM8KSeqLen2kBS16x16RL(Gemma2BV2GSM8KSeqLen2kRL):
  batch_size: int = 16
  num_samples_per_example: int = 16
  tb_log_interval: int = 8
  ckpt_interval: int = 40


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2GSM8KSeqLen2kBS16x8RL(Gemma2BV2GSM8KSeqLen2kRL):
  batch_size: int = 16
  num_samples_per_example: int = 8
  tb_log_interval: int = 20
  ckpt_interval: int = 100


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2GSM8KSeqLen2kBS32x16RL(Gemma2BV2GSM8KSeqLen2kRL):
  # Feasible setup: glp_2x4
  batch_size: int = 32
  num_samples_per_example: int = 16
  tb_log_interval: int = 4
  ckpt_interval: int = 20
  use_flash_attention: bool = True
  grad_accum_steps: int = 2


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2GSM8K32ExamplesRL(Gemma2BV2):
  dataset_name: str = 'simply_json:gsm8k_train32'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2ITGSM8K0ShotRL(Gemma2BV2GSM8K0ShotRL):
  """Gemma 2B IT model for GSM8K RL."""

  lm_format_name: str = 'GemmaV2Chat'
  extra_eos_tokens: tuple[str, ...] = ()
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotBoxedInQuestionEvaluation()
  )
  init_ckpt_dir: str = GEMMA2_2B_IT_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2ITGSM8KCoT0ShotRL(Gemma2BV2ITGSM8K0ShotRL):
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation()
  )


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2ITDSR40K0ShotRL(Gemma2BV2ITGSM8K0ShotRL):
  dataset_name: str = 'simply_json:dsr40k_train'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2BV2ITDSR40KCoT0ShotRL(Gemma2BV2ITDSR40K0ShotRL):
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation()
  )


#################################################################################
## Qwen2 models.
## https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2(TFM1p7BLM1B):
  # number of parameters: ~1.5B
  seq_len: int = 4096
  vocab_size: int = 151936
  model_dim: int = 1536
  expand_factor: int = 0
  ffn_expand_dim: int = 8960
  per_head_dim: int = 128
  n_heads: int = 12
  n_layers: int = 28
  n_kv_heads: int = 2
  ffn_activation: str = 'silu'
  use_post_ln: bool = False
  use_per_dim_scale: bool = False
  ffn_use_bias: bool = False
  qkv_use_bias: bool = True
  output_layer_use_bias: bool = False
  use_tied_embedding: bool = False
  use_combined_qkv: bool = False
  embedding_lookup_scale: float | None = None
  norm_scale_plus_one: bool = False
  attn_soft_cap: float = -1.0
  output_logits_soft_cap: float = -1.0

  # NOTE: Data config is vocab dependent. We currently do not have dataset
  # prepared with qwen vocab.
  vocab_name: str = 'DeepSeek-R1-Distill-Qwen'

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = DEEPSEEK_QWEN_1P5B_CKPT_DIR
  init_ckpt_step: int = -1
  init_ckpt_opt_state: bool = False
  reset_steps: bool = True


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRL(DeepSeekQwen1p5BV2):
  dataset_name: str = 'simply_json:dsr40k_train'
  num_train_steps: int = 1_000_000
  train_loop_name: str = 'rl'

  train_batch_size: int = 16 * 8
  batch_size: int = 16
  num_samples_per_example: int = 8
  sampling_temperature: float = 1.0
  num_train_steps_per_batch: int = 4

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
  use_policy_logp_as_sampler_logp: bool = False
  filter_truncated: bool = False
  max_num_samples_per_train_batch: int | None = None

  # Optimizer configs.
  optimizer: opt_lib.Optimizer = opt_lib.Adam(
      beta1=0.9, beta2=0.95, epsilon=1e-8
  )
  weight_decay: float = 0.0
  lr_schedule_name: str = 'constant'
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-6),
      ('warmup_steps', 1),
  )

  # Checkpoint and tensorboard configs.
  init_ckpt_opt_state: bool = False
  ckpt_max_to_keep: int = 1
  tb_log_interval: int = 4
  ckpt_interval: int = 4

  # Sharding config.
  decoding_sharding_config: SimplyConfig = DecodeGSPMDSharding()

  # Use train_max_seq_len to control the max decode steps.
  sampling_max_decode_steps: int = 32768
  train_max_seq_len: int = 9 * 1024
  sampling_prefill_size: int = 1024
  sampling_max_input_len: int = 1024
  sampling_intermediate_decode_steps: int = 1024

  lm_format_name: str = 'DeepSeekQwenR1DistillChat'
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotDeepSeekQwenR1CoTBoxed()
  )
  extra_eos_tokens: tuple[str, ...] = ()

  activation_dtype_name: str = 'bfloat16'
  decoding_quant_scheme: str = 'bfloat16'
  ref_params_dtype: str = 'bfloat16'
  use_flash_attention: bool = True
  flash_attention_block_size: int = 512

  use_validation_set: bool = True
  validation_eval_interval: int = 50
  validation_dataset_name: str | None = 'simply_json:aime24'
  validation_eval_batch_size: int = 64
  validation_eval_epochs: int = 5
  validation_evaluation: evaluation_lib.Evaluation | None = None
  validation_lm_format_name: str = ''
  validation_max_decode_steps: int | None = None


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32(DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRL):
  activation_dtype_name: str = 'float32'
  decoding_quant_scheme: str = 'float32'
  ref_params_dtype: str = 'float32'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V2(DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32):
  # Batch size and gradient accumulation configs.
  train_batch_size: int = 64 * 8
  batch_size: int = 64
  num_samples_per_example: int = 8
  grad_accum_steps: int = 4

  # Checkpoint and tensorboard logging configs.
  tb_log_interval: int = 1
  ckpt_interval: int = 20
  ckpt_max_to_keep: int = 1

  # Flash attention configs.
  use_flash_attention: bool = True
  flash_attention_block_size: int = 512

  # RL algorithm configs.
  num_train_steps_per_batch: int = 1
  normalize_reward_method: str = 'ByGroup'
  policy_ratio_cap: float | None = None
  max_abs_advantage: float | None = None

  lr_schedule_name: str = 'constant'
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-6),
      ('warmup_steps', 1),
  )


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLV2(DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V2):
  activation_dtype_name: str = 'bfloat16'
  decoding_quant_scheme: str = 'bfloat16'
  ref_params_dtype: str = 'bfloat16'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V3(DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V2):
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 3e-6),
      ('warmup_steps', 1),
  )


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLBF16V2(
    DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V2
):
  activation_dtype_name: str = 'bfloat16'
  decoding_quant_scheme: str = 'bfloat16'
  ref_params_dtype: str = 'bfloat16'
  use_validation_set: bool = True
  validation_eval_batch_size: int = 64
  validation_eval_interval: int = 50
  use_policy_logp_as_sampler_logp: bool = True


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V4(DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V2):
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-5),
      ('warmup_steps', 1),
  )


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32T0p6(DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRL):
  activation_dtype_name: str = 'float32'
  decoding_quant_scheme: str = 'float32'
  ref_params_dtype: str = 'float32'
  sampling_temperature: float = 0.6


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen7BV2(DeepSeekQwen1p5BV2):
  vocab_size: int = 152064
  model_dim: int = 3584
  ffn_expand_dim: int = 18944
  n_layers: int = 28
  n_heads: int = 28
  n_kv_heads: int = 4
  init_ckpt_dir: str = DEEPSEEK_QWEN_7B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen14BV2(DeepSeekQwen1p5BV2):
  vocab_size: int = 152064
  model_dim: int = 5120
  ffn_expand_dim: int = 13824
  n_layers: int = 48
  n_heads: int = 40
  n_kv_heads: int = 8
  global_rope_max_timescale: int = 1_000_000
  rms_norm_epsilon: float = 1e-5

  init_ckpt_dir: str = DEEPSEEK_QWEN_14B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DeepSeekQwen32BV2(DeepSeekQwen1p5BV2):
  vocab_size: int = 152064
  model_dim: int = 5120
  ffn_expand_dim: int = 27648
  n_layers: int = 64
  n_heads: int = 40
  n_kv_heads: int = 8
  global_rope_max_timescale: int = 1_000_000
  rms_norm_epsilon: float = 1e-5
  init_ckpt_dir: str = DEEPSEEK_QWEN_32B_CKPT_DIR


# TODO: The ideal way should be first define Qwen native configs and use
# them to define DeepSeek Qwen configs.
@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class QwenMath1p5BV2p5(DeepSeekQwen1p5BV2):
  use_tied_embedding = True

  vocab_name: str = 'Qwen2.5'
  init_ckpt_dir: str = QWEN2p5_MATH_1p5B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class QwenMath7BV2p5(DeepSeekQwen7BV2):
  init_ckpt_dir: str = QWEN2p5_MATH_7B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class QwenMath14BV2p5(DeepSeekQwen14BV2):
  init_ckpt_dir: str = QWEN2p5_MATH_14B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class QwenMath32BV2p5(DeepSeekQwen32BV2):
  init_ckpt_dir: str = QWEN2p5_MATH_32B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class QwQ32B(DeepSeekQwen32BV2):
  vocab_name: str = 'QwQ'
  init_ckpt_dir: str = QWQ_32B_CKPT_DIR


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
def DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRL():
  return apply_config_diff(
      config=DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRL(),
      diff_base=DeepSeekQwen1p5BV2(),
      diff_new=DeepSeekQwen7BV2())


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRL,
    name='dsqwen_v2-7b-dsr40k-R1_distill_cot0shot-rl')


@ExperimentConfigRegistry.register
def DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRLF32V2():
  config = apply_config_diff(
      config=DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLF32V2(),
      diff_base=DeepSeekQwen1p5BV2(),
      diff_new=DeepSeekQwen7BV2())
  config = dataclasses.replace(config, batch_size=32)
  return config


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRLF32V2,
    name='dsqwen_v2-7b-dsr40k-R1_distill_cot0shot-rl-f32-v2')


@ExperimentConfigRegistry.register
def DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRLF32V3():
  config = DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRLF32V2()
  config = dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 3e-6),
          ('warmup_steps', 1),
      ),
  )
  return config


@ExperimentConfigRegistry.register
def DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRLBF16V2():
  config = dataclasses.replace(
      DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRLF32V2(),
      activation_dtype_name='bfloat16',
      decoding_quant_scheme='bfloat16',
      ref_params_dtype='bfloat16',
      use_validation_set=True,
      validation_eval_interval=50,
      validation_eval_batch_size=32,
      use_policy_logp_as_sampler_logp=True,
  )
  return config


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen7BV2ITDSR40KR1DistillCoT0ShotRLF32V3,
    name='dsqwen_v2-7b-dsr40k-R1_distill_cot0shot-rl-f32-v3')


@ExperimentConfigRegistry.register
def DeepSeekQwen14BV2ITDSR40KR1DistillCoT0ShotRL():
  return apply_config_diff(
      config=DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRL(),
      diff_base=DeepSeekQwen1p5BV2(),
      diff_new=DeepSeekQwen14BV2())

# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen14BV2ITDSR40KR1DistillCoT0ShotRL,
    name='dsqwen_v2-14b-dsr40k-R1_distill_cot0shot-rl')


@ExperimentConfigRegistry.register
def DeepSeekQwen14BV2ITDSR40KR1DistillCoT0ShotRLV2():
  config = apply_config_diff(
      config=DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLV2(),
      diff_base=DeepSeekQwen1p5BV2(),
      diff_new=DeepSeekQwen14BV2())
  config = dataclasses.replace(config, batch_size=16)
  return config


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen14BV2ITDSR40KR1DistillCoT0ShotRLV2,
    name='dsqwen_v2-14b-dsr40k-R1_distill_cot0shot-rl-v2')


@ExperimentConfigRegistry.register
def DeepSeekQwen14BV2ITDSR40KR1DistillCoT0ShotRLV3():
  config = DeepSeekQwen14BV2ITDSR40KR1DistillCoT0ShotRLV2()
  config = dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 3e-6),
          ('warmup_steps', 1),
      ),
  )
  return config


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen14BV2ITDSR40KR1DistillCoT0ShotRLV3,
    name='dsqwen_v2-14b-dsr40k-R1_distill_cot0shot-rl-v3')


@ExperimentConfigRegistry.register
def DeepSeekQwen32BV2ITDSR40KR1DistillCoT0ShotRL():
  return apply_config_diff(
      config=DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRL(),
      diff_base=DeepSeekQwen1p5BV2(),
      diff_new=DeepSeekQwen32BV2())

# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen32BV2ITDSR40KR1DistillCoT0ShotRL,
    name='dsqwen_v2-32b-dsr40k-R1_distill_cot0shot-rl')


@ExperimentConfigRegistry.register
def DeepSeekQwen32BV2ITDSR40KR1DistillCoT0ShotRLV2():
  config = apply_config_diff(
      config=DeepSeekQwen1p5BV2ITDSR40KR1DistillCoT0ShotRLV2(),
      diff_base=DeepSeekQwen1p5BV2(),
      diff_new=DeepSeekQwen32BV2())
  config = dataclasses.replace(config, batch_size=16)
  return config


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen32BV2ITDSR40KR1DistillCoT0ShotRLV2,
    name='dsqwen_v2-32b-dsr40k-R1_distill_cot0shot-rl-v2')


@ExperimentConfigRegistry.register
def DeepSeekQwen32BV2ITDSR40KR1DistillCoT0ShotRLV3():
  config = DeepSeekQwen32BV2ITDSR40KR1DistillCoT0ShotRLV2()
  config = dataclasses.replace(
      config,
      lr_schedule_config=(
          ('lr', 3e-6),
          ('warmup_steps', 1),
      ),
  )
  return config


# Register the config with a more readable name.
ExperimentConfigRegistry.register(
    DeepSeekQwen32BV2ITDSR40KR1DistillCoT0ShotRLV3,
    name='dsqwen_v2-32b-dsr40k-R1_distill_cot0shot-rl-v3')


#################################################################################
## Qwen3 models.
## https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen0p6BV3(TFM1p7BLM1B):
  seq_len: int = 4096
  vocab_size: int = 151936
  model_dim: int = 1024
  expand_factor: int = 0
  ffn_expand_dim: int = 3072
  per_head_dim: int = 128
  n_heads: int = 16
  n_layers: int = 28
  n_kv_heads: int = 8
  ffn_activation: str = 'silu'
  use_post_ln: bool = False
  use_qk_norm: bool = True
  use_per_dim_scale: bool = False
  ffn_use_bias: bool = False
  qkv_use_bias: bool = False
  output_layer_use_bias: bool = False
  use_tied_embedding: bool = True
  use_combined_qkv: bool = False
  embedding_lookup_scale: float | None = None
  norm_scale_plus_one: bool = False
  attn_soft_cap: float = -1.0
  output_logits_soft_cap: float = -1.0
  global_rope_max_timescale: int = 1_000_000

  # NOTE: Data config is vocab dependent. We currently do not have dataset
  # prepared with qwen vocab.
  vocab_name: str = 'Qwen3'

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = QWEN3_0P6B_CKPT_DIR
  init_ckpt_step: int = -1
  init_ckpt_opt_state: bool = False
  reset_steps: bool = True


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen1p7BV3(Qwen0p6BV3):
  model_dim: int = 2048
  ffn_expand_dim: int = 6144

  init_ckpt_dir: str = QWEN3_1P7B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen4BV3(Qwen0p6BV3):
  model_dim: int = 2560
  ffn_expand_dim: int = 9728
  n_heads: int = 32
  n_layers: int = 36

  init_ckpt_dir: str = QWEN3_4B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen8BV3(Qwen0p6BV3):
  model_dim: int = 4096
  ffn_expand_dim: int = 12288
  n_heads: int = 32
  n_layers: int = 36
  use_tied_embedding: bool = False

  init_ckpt_dir: str = QWEN3_8B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen14BV3(Qwen0p6BV3):
  model_dim: int = 5120
  ffn_expand_dim: int = 17408
  n_heads: int = 40
  n_layers: int = 40
  use_tied_embedding: bool = False

  init_ckpt_dir: str = QWEN3_14B_CKPT_DIR


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen32BV3(Qwen0p6BV3):
  model_dim: int = 5120
  ffn_expand_dim: int = 25600
  n_heads: int = 64
  n_layers: int = 64
  use_tied_embedding: bool = False

  init_ckpt_dir: str = QWEN3_32B_CKPT_DIR


################################################################################
## Tiny experiments for tests.


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TransformerLMTest(TFM1p7BLM1B):
  # Model config
  model_dim: int = 8
  per_head_dim: int = 4
  n_heads: int = 2
  n_layers: int = 2
  expand_factor: int = 2
  use_scan: bool = True
  use_flash_attention: bool = False
  activation_dtype_name: str = 'bfloat16'

  # Data config
  num_train_steps: int = 50
  batch_size: int = 4

  vocab_size: int = 32_000
  seq_len: int = 64
  dataset_name: str = 'imdb_reviews.vb32000_t5_cc'
  lr: opt_lib.Schedule = opt_lib.LinearWarmupCosineDecay(
      value=1e-3,
      warmup_steps=10,
      steps_after_decay=10,
      end_decay=0.1,
  )
  clip_grad_norm: float = -1.0
  clip_update_norm: float = -1.0
  use_validation_set: bool = True
  validation_num_eval_steps: int = 2
  validation_eval_interval: int = 5
  validation_eval_batch_size: int = -1

  # Checkpoint and tensorboard config
  ckpt_interval: int = 10
  ckpt_max_to_keep: int = 3
  tb_log_interval: int = 2


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TransformerLMTestC4(TransformerLMTest):
  seq_len: int = 4096
  dataset_name: str = 'c4.vb100864_openmix_v1'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TransformerLMTestNoScan(TransformerLMTest):
  use_scan: bool = False
  use_remat: bool = False


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TransformerLMTestSFT(TransformerLMTest):
  num_train_steps: int = 2000
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-4),
      ('warmup_steps', 100),
      ('steps_after_decay', 10),
      ('end_decay', 0.1),
  )
  dataset_name: str = 'tulu_v2_sft.vb100864_openmix_v1'
  # Config for init from existing checkpoint.
  init_ckpt_dir: str = '/tmp/simply_test_pt_1/checkpoints'
  init_ckpt_step: int = -1

  use_validation_set: bool = False

  # Add masks to only calculate loss on assistant responses.
  add_chat_loss_mask: bool = True
  mask_start_token: str = '<reserved_2>'
  mask_end_token: str = '<reserved_4>'


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TransformerLMTestRL(TransformerLMTest):
  dataset_name: str = 'simply_json:dsr40k_train'
  num_train_steps: int = 30
  train_loop_name: str = 'rl'
  evaluation: evaluation_lib.Evaluation = (
      evaluation_lib.ZeroShotBoxedInQuestionEvaluation()
  )
  use_validation_set: bool = False

  vocab_name: str = 'vb32768_openmix_v1'
  lm_format_name: str = 'SimplyV1Chat'
  batch_size: int = 4
  train_batch_size: int = 4
  num_samples_per_example: int = 1
  normalize_reward_method: str = 'Global'
  sampling_temperature: float = 1.0
  filter_truncated: bool = False
  max_num_samples_per_train_batch: int | None = None

  # Use train_max_seq_len to control the max decode steps.
  sampling_max_decode_steps: int = 32768
  train_max_seq_len: int = 8
  sampling_prefill_size: int = 16
  sampling_max_input_len: int = 8
  sampling_intermediate_decode_steps: int = -1
  sampling_microbatch_size: int | None = 2

  num_train_steps_per_batch: int = 4
  lr_schedule_name: str = 'constant'
  lr_schedule_config: tuple[tuple[str, Any], ...] = (
      ('lr', 1e-7),
      ('warmup_steps', 100),
  )
  extra_eos_tokens: tuple[str, ...] = newlines_from_counts(range(1, 2))
  decoding_sharding_config: SimplyConfig = DecodeGSPMDSharding()

  # RL algorithm configs.
  gamma: float = 1.0
  kl_coeff: float = 0.01
  use_grpo: bool = True
  ppo_clip_eps: float = 0.2
  ppo_clip_eps_low: float | None = None
  ppo_clip_eps_high: float | None = None
  policy_ratio_cap: float | None = 10.0
  normalize_reward_method: str = 'ByGroup'
  normalize_advantage: bool = False
  max_abs_advantage: float | None = 10.0
  use_policy_logp_as_sampler_logp: bool = False

  activation_dtype_name: str = 'float32'
  decoding_quant_scheme: str = 'float32'
  ref_params_dtype: str = 'float32'
  grad_accum_steps: int = 2

  # Early stopping threshold.
  early_stop: opt_lib.EarlyStop | None = opt_lib.SimpleEarlyStop(
      thresholds=(
          (20, ('<', 'accuracy', 0.5)),
      )
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

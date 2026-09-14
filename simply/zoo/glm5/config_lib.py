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
"""GLM-5-series `glm_moe_dsa` experiment configs, kept out of core.

`Glm5Config` subclasses the core `BaseExperimentConfig` to carry the MLA and
GLM-MoE fields that `model_lib.GlmTransformerLM` reads; each version is a
factory function here (e.g. `glm5p2()`) that registers an experiment. Because
the model is dispatched by `model_name` and the config is a plain dataclass
subclass, no GLM field lives in core `config_lib`.

Convention (architecture vs version): this `zoo/glm5` package is scoped to the
GLM-5-series `glm_moe_dsa` architecture. A new GLM version that SHARES this
architecture (e.g. a 5.3 post-trained on the same base) is added here as another
config function (`glm5pN()`), reusing the model code. A genuinely DIFFERENT
architecture (e.g. GLM-5.3-Flash / `glm5_next`: KDA + mHC residual + multimodal)
should be a SEPARATE sibling plugin (`zoo/glm5_next/`) that imports the shared
primitives (`MLAAttention`, etc.) from here — not a version-suffixed folder.
"""

import dataclasses
import math
import os
from typing import Literal

from simply import config_lib
from simply.zoo.glm5.utils import mla as mla_lib

BaseExperimentConfig = config_lib.BaseExperimentConfig
ExperimentConfigRegistry = config_lib.ExperimentConfigRegistry
moe_sharding = config_lib.moe_sharding

# The converted ORBAX stores already-transformed Simply params; loaded as a
# pass-through. Use 'GlmMoeDsaFormat' (see utils/ckpt_format.py) to load a raw
# HF-keyed checkpoint instead.
GLM5P2_CKPT_DIR = os.path.join(config_lib.MODELS_DIR, 'glm5p2/ORBAX')


@dataclasses.dataclass(frozen=True)
class Glm5Config(BaseExperimentConfig):
  """Config for GLM-5.2: MLA attention + sigmoid/noaux_tc MoE + shared expert.

  Only the extra GLM/DeepSeek-V3 fields live here; everything else is inherited
  from `BaseExperimentConfig`. `model_lib.GlmTransformerLM` reads these fields.
  """

  # Multi-head Latent Attention (see utils.mla.MLAAttention).
  mla_q_lora_rank: int = 0
  mla_kv_lora_rank: int = 0
  mla_qk_nope_head_dim: int = 0
  mla_qk_rope_head_dim: int = 0
  mla_v_head_dim: int = 0
  # Compact/latent MLA KV cache for decode: cache only the kv-LoRA latent +
  # shared RoPE key (~57x smaller/token than materialized 64-head K/V) and
  # up-project with kv_b via absorption at attention time. Training/prefill is
  # unaffected (materialized einsum MLA math).
  mla_use_latent_kv_cache: bool = False
  # Flash-style (KV-tiled, online-softmax) latent absorption decode attention.
  # Default on; bit-exact (within fp tol) with the dense path.
  mla_latent_flash_decode: bool = True
  mla_latent_flash_block_k: int = 512
  # GLM / DeepSeek-V3 MoE routing (see utils.moe.GlmMoeFeedForward).
  router_score_func: Literal['softmax', 'sigmoid'] = 'softmax'
  router_use_correction_bias: bool = False
  norm_topk_prob: bool = True
  routed_scaling_factor: float = 1.0
  num_shared_experts: int = 0
  # The first `first_k_dense_replace` layers use a dense FFN even when the model
  # is MoE (DeepSeek-V3 / GLM convention). 0 = all MoE layers are MoE.
  first_k_dense_replace: int = 0
  # Intermediate size for the dense (non-MoE) FFN layers; falls back to
  # `ffn_expand_dim` when None. GLM dense layers are wider than the per-expert
  # MoE intermediate size.
  dense_ffn_expand_dim: int | None = None


@ExperimentConfigRegistry.register
def glm5p2() -> Glm5Config:
  """GLM-5.2 (744B-A40B, glm_moe_dsa) config for Simply.

  MLA attention + sigmoid/noaux_tc MoE with a shared expert. Decode uses the
  memory-efficient compact-latent KV cache with flash-tiled absorption attention
  by default (see MLAAttention), so per-token decode cost stays ~flat with
  context; prefill/training use the materialized MLA math (bit-exact).
  """
  config = Glm5Config()
  return dataclasses.replace(
      config,
      model_name='GlmTransformerLM',
      # Model config.
      vocab_size=154880,
      model_dim=6144,
      n_heads=64,
      n_kv_heads=64,
      per_head_dim=256,  # qk_head_dim (= qk_nope 192 + qk_rope 64); v_head=256.
      n_layers=78,
      use_post_ln=False,
      use_per_dim_scale=False,
      ffn_activation='silu',
      ffn_use_bias=False,
      # Heterogeneous dense + MoE blocks can't be stacked by jax.lax.scan.
      use_scan=False,
      output_layer_use_bias=False,
      use_tied_embedding=False,
      embedding_lookup_scale=None,
      norm_scale_plus_one=False,
      attn_soft_cap=-1.0,
      output_logits_soft_cap=-1.0,
      rms_norm_epsilon=1e-5,
      query_scale=float(math.sqrt(256)),  # 1/sqrt(qk_head_dim).
      position_encoding=mla_lib.InterleavedRoPE(max_timescale=8_000_000),
      # MLA.
      mla_q_lora_rank=2048,
      mla_kv_lora_rank=512,
      mla_qk_nope_head_dim=192,
      mla_qk_rope_head_dim=64,
      mla_v_head_dim=256,
      # Efficient decode: compact-latent KV cache + flash-tiled absorption.
      mla_use_latent_kv_cache=True,
      mla_latent_flash_decode=True,
      # MoE.
      use_moe=True,
      num_experts=256,
      num_experts_per_token=8,
      ffn_expand_dim=2048,  # moe_intermediate_size (per-expert).
      dense_ffn_expand_dim=12288,  # intermediate_size (dense layers 0..2).
      first_k_dense_replace=3,
      router_score_func='sigmoid',
      router_use_correction_bias=True,
      routed_scaling_factor=2.5,
      num_shared_experts=1,
      lbl_loss_weight=0.0,
      gmm_impl='megablox',
      vocab_name='GLM-5.2',
      init_ckpt_dir=GLM5P2_CKPT_DIR,
      init_ckpt_format='V2Format',
      sharding_config=moe_sharding(),
  )

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
"""GLM-5.2 (`glm_moe_dsa`, 744B-A40B) transformer, assembled as a plugin.

Fully self-contained: nothing GLM-specific lives in core `model_lib`. Both
classes override `setup`, call `super().setup()` to reuse all the standard
construction (LayerNorms, embedding, final norm), then **reassign** just the
submodules that differ:
  * `GlmTransformerBlock` swaps `self.attn` -> `MLAAttention` and `self.ffn` ->
    `GlmMoeFeedForward` (or a dense `FeedForward` for the first
    `first_k_dense_replace` layers).
  * `GlmTransformerLM` swaps `self.blocks` -> a list of `GlmTransformerBlock`.
The base submodules built by `super().setup()` are inert dataclass objects
(they allocate no params until `init`, which reads the final `self.*`), so the
reassign leaves no orphan params. Registered as `GlmTransformerBlock` /
`GlmTransformerLM`.
"""

import dataclasses
from typing import Literal

from simply import model_lib
from simply.utils import module
from simply.zoo.glm5.utils import mla as mla_lib
from simply.zoo.glm5.utils import moe as moe_lib

TransformerBlock = model_lib.TransformerBlock
TransformerLM = model_lib.TransformerLM
FeedForward = model_lib.FeedForward
MLAAttention = mla_lib.MLAAttention
GlmMoeFeedForward = moe_lib.GlmMoeFeedForward


@module.ModuleRegistry.register
@dataclasses.dataclass
class GlmTransformerBlock(TransformerBlock):
  """Transformer block using MLA attention and GLM sigmoid/noaux_tc MoE."""

  # MLA (see utils.mla.MLAAttention).
  mla_q_lora_rank: int = 0
  mla_kv_lora_rank: int = 0
  mla_qk_nope_head_dim: int = 0
  mla_qk_rope_head_dim: int = 0
  mla_v_head_dim: int = 0
  mla_use_latent_kv_cache: bool = False
  mla_latent_flash_decode: bool = True
  mla_latent_flash_block_k: int = 512
  # GLM MoE routing (see utils.moe.GlmMoeFeedForward).
  router_score_func: Literal['softmax', 'sigmoid'] = 'softmax'
  router_use_correction_bias: bool = False
  norm_topk_prob: bool = True
  routed_scaling_factor: float = 1.0
  num_shared_experts: int = 0

  def setup(self) -> None:
    super().setup()
    # Swap the attention + FFN built by the base for GLM's variants. The base
    # objects are inert (no params until `init`), so this leaves no orphans.
    self.attn = self._make_mla_attention()  # pyrefly: ignore[bad-assignment]
    if self.use_moe:
      self.ffn = self._make_glm_moe()
    else:
      self.ffn = self._make_dense_ffn()

  def _make_mla_attention(self) -> MLAAttention:
    return MLAAttention(
        model_dim=self.model_dim,
        n_heads=self.n_heads,
        q_lora_rank=self.mla_q_lora_rank,
        kv_lora_rank=self.mla_kv_lora_rank,
        qk_nope_head_dim=self.mla_qk_nope_head_dim,
        qk_rope_head_dim=self.mla_qk_rope_head_dim,
        v_head_dim=self.mla_v_head_dim,
        rms_norm_epsilon=self.rms_norm_epsilon,
        norm_scale_plus_one=self.norm_scale_plus_one,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        qkv_partition=self.sharding_config.attn_qkv_partition,
        o_partition=self.sharding_config.attn_o_partition,
        attn_activation_partition=(
            self.sharding_config.attn_activation_partition
        ),
        output_partition=self.sharding_config.activation_partition,
        # Others.
        window_size=self.window_size,
        attn_soft_cap=self.attn_soft_cap,
        position_encoding=self.position_encoding,
        query_scale=self.query_scale,
        weight_init=self.attn_weight_init,
        # Compact/latent KV cache (efficient DeepSeek-MLA decode).
        use_latent_kv_cache=self.mla_use_latent_kv_cache,
        latent_flash_decode=self.mla_latent_flash_decode,
        latent_flash_block_k=self.mla_latent_flash_block_k,
    )

  def _make_glm_moe(self) -> GlmMoeFeedForward:
    return GlmMoeFeedForward(
        num_experts=self.num_experts,
        num_experts_per_token=self.num_experts_per_token,
        ep_capacity_factor=self.ep_capacity_factor,
        router_z_loss_weight=self.router_z_loss_weight,
        lbl_loss_weight=self.lbl_loss_weight,
        router_score_func=self.router_score_func,
        router_use_correction_bias=self.router_use_correction_bias,
        norm_topk_prob=self.norm_topk_prob,
        routed_scaling_factor=self.routed_scaling_factor,
        num_shared_experts=self.num_shared_experts,
        model_dim=self.model_dim,
        expand_factor=self.expand_factor,
        use_gated_activation_in_ffn=self.use_gated_activation_in_ffn,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        sharding_config=self.sharding_config,
        # Below are for experimental usage.
        ffn_expand_dim=self.ffn_expand_dim,
        ffn_use_bias=self.ffn_use_bias,
        ffn_activation=self.ffn_activation,
        # tile sizes for gmm.
        tile_batch_seq=self.tile_batch_seq,
        tile_model_dim=self.tile_model_dim,
        tile_expand_dim=self.tile_expand_dim,
        # Implementation of gmm.
        gmm_impl=self.gmm_impl,
        weight_quant=self.ffn_weight_quant,
    )

  def _make_dense_ffn(self) -> FeedForward:
    # Dense (first_k_dense_replace) layer under a MoE sharding config: squeeze
    # the 3D expert partitions to 2D so the dense weight shards correctly.
    sharding_config = moe_lib.dense_ffn_sharding(self.sharding_config)
    return FeedForward(
        model_dim=self.model_dim,
        expand_factor=self.expand_factor,
        use_gated_activation_in_ffn=self.use_gated_activation_in_ffn,
        activation_dtype=self.activation_dtype,
        sharding_config=sharding_config,
        ffn_expand_dim=self.ffn_expand_dim,
        ffn_use_bias=self.ffn_use_bias,
        ffn_activation=self.ffn_activation,
        ffn_weight_init=self.ffn_weight_init,
        weight_quant=self.ffn_weight_quant,
    )


@module.ModuleRegistry.register
@dataclasses.dataclass
class GlmTransformerLM(TransformerLM):
  """GLM-5.2 decoder-only Transformer (MLA + sigmoid/noaux_tc MoE)."""

  def setup(self) -> None:
    super().setup()
    # Replace the base blocks with GLM blocks. The base blocks are inert
    # (no params until `init`), so no orphan params are left behind.
    config = self.config
    self.blocks = [
        self._make_glm_block(
            config.block_attn_pattern[i % len(config.block_attn_pattern)],
            layer_idx=i,
        )
        for i in range(config.n_layers)
    ]

  def _make_glm_block(
      self, pattern: str, layer_idx: int
  ) -> GlmTransformerBlock:
    config = self.config
    if isinstance(config.position_encoding, dict):
      pe = config.position_encoding.get(pattern)
    else:
      pe = config.position_encoding
    total_num_pages = config.global_total_num_pages
    if pattern == 'local':
      total_num_pages = config.local_total_num_pages
    # Per-layer dense-vs-MoE: the first `first_k_dense_replace` layers use a
    # dense FFN even when the model is MoE (DeepSeek-V3 / GLM convention).
    first_k_dense = getattr(config, 'first_k_dense_replace', 0)
    block_use_moe = config.use_moe and layer_idx >= first_k_dense
    # MoE expand dim is the per-expert intermediate size; dense layers use
    # `dense_ffn_expand_dim` when provided (GLM dense layers are wider).
    block_ffn_expand_dim = getattr(config, 'ffn_expand_dim', None)
    if not block_use_moe:
      block_ffn_expand_dim = (
          getattr(config, 'dense_ffn_expand_dim', None) or block_ffn_expand_dim
      )
    return GlmTransformerBlock(
        config.model_dim,
        config.n_heads,
        config.per_head_dim,
        config.expand_factor,
        use_rmsnorm=config.use_rmsnorm,
        use_pre_ln=config.use_pre_ln,
        use_post_ln=config.use_post_ln,
        use_post_skip_ln=config.use_post_skip_ln,
        use_qk_norm=config.use_qk_norm,
        use_per_dim_scale=config.use_per_dim_scale,
        use_gated_activation_in_ffn=config.use_gated_activation_in_ffn,
        ffn_use_bias=config.ffn_use_bias,
        ffn_expand_dim=block_ffn_expand_dim,
        # MLA.
        mla_q_lora_rank=config.mla_q_lora_rank,
        mla_kv_lora_rank=config.mla_kv_lora_rank,
        mla_qk_nope_head_dim=config.mla_qk_nope_head_dim,
        mla_qk_rope_head_dim=config.mla_qk_rope_head_dim,
        mla_v_head_dim=config.mla_v_head_dim,
        mla_use_latent_kv_cache=config.mla_use_latent_kv_cache,
        mla_latent_flash_decode=config.mla_latent_flash_decode,
        mla_latent_flash_block_k=config.mla_latent_flash_block_k,
        # MoE.
        use_moe=block_use_moe,
        num_experts=config.num_experts,
        ep_capacity_factor=config.ep_capacity_factor,
        num_experts_per_token=config.num_experts_per_token,
        router_score_func=config.router_score_func,
        router_use_correction_bias=config.router_use_correction_bias,
        norm_topk_prob=config.norm_topk_prob,
        routed_scaling_factor=config.routed_scaling_factor,
        num_shared_experts=config.num_shared_experts,
        lbl_loss_weight=config.lbl_loss_weight,
        router_z_loss_weight=config.router_z_loss_weight,
        tile_batch_seq=config.tile_batch_seq,
        tile_model_dim=config.tile_model_dim,
        tile_expand_dim=config.tile_expand_dim,
        gmm_impl=config.gmm_impl,
        ffn_weight_quant=config.ffn_weight_quant,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        sharding_config=self.sharding_config,
        # Others.
        use_flash_attention=config.use_flash_attention,
        flash_attention_block_size=config.flash_attention_block_size,
        window_size=config.window_size if pattern == 'local' else 0,
        use_window_chunk=config.use_window_chunk,
        n_kv_heads=config.n_kv_heads,
        qkv_use_bias=config.qkv_use_bias,
        ffn_activation=config.ffn_activation,
        norm_scale_plus_one=config.norm_scale_plus_one,
        attn_soft_cap=config.attn_soft_cap,
        attn_mask_value=config.attn_mask_value,
        rms_norm_epsilon=config.rms_norm_epsilon,
        position_encoding=pe,
        query_scale=config.query_scale,
        total_num_pages=total_num_pages,
        page_size=config.page_size,
        rpa_block_q=config.rpa_block_q,
        kv_cache_quant=config.kv_cache_quant,
        ffn_weight_init=config.ffn_weight_init,
        attn_weight_init=config.attn_weight_init,
    )

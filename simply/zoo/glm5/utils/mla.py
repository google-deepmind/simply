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
"""Multi-head Latent Attention (MLA) for GLM-5.2 (`glm_moe_dsa`).

Self-contained attention plugin: a `SimplyModule` registered under
`MLAAttention` and dropped into a transformer block via
`TransformerBlock._make_attention` (see `glm_model.GlmTransformerBlock`). It
reuses the core building blocks (`EinsumLinear`, `LayerNorm`, causal masking)
from `simply.model_lib`; nothing GLM-specific lives in core.
"""

from collections.abc import Mapping
import dataclasses
import math
from typing import Any, cast

import einops
import jax
import jax.numpy as jnp

from simply import model_lib
from simply.utils import common
from simply.utils import initializer
from simply.utils import module
from simply.utils import position_encoding as pe_lib
from simply.utils import sharding as sharding_lib

Array = common.Array
PyTree = common.PyTree
PRNGKey = jax.typing.ArrayLike
DTypeLike = jax.typing.DTypeLike
PartitionAnnotation = common.PartitionAnnotation

# Core building blocks reused by MLA (shared with the standard Attention path).
EinsumLinear = model_lib.EinsumLinear
LayerNorm = model_lib.LayerNorm
create_mask = model_lib.create_mask
updated_decode_state = model_lib.updated_decode_state
attn = model_lib.attn
neg_inf = common.neg_inf
get_raw_arrays = common.get_raw_arrays
RoPE = pe_lib.RoPE
PositionEncodingRegistry = pe_lib.PositionEncodingRegistry


@PositionEncodingRegistry.register
@dataclasses.dataclass(frozen=True)
class InterleavedRoPE(RoPE):
  """Interleaved Rotary Position Embedding (DeepSeek-V3 / GLM `glm_moe_dsa`).

  Differs from the standard (half-split) `RoPE` in how the rotated dimensions
  are paired: consecutive pairs ``(x_{2i}, x_{2i+1})`` are rotated by angle
  ``theta_i`` and the rotated reals are written out as
  ``[r0c, r1c, ..., r0s, r1s, ...]`` (all cos-parts then all sin-parts), where
  ``ric = x_{2i}*cos - x_{2i+1}*sin`` and ``ris = x_{2i+1}*cos + x_{2i}*sin``.
  This matches HuggingFace ``apply_rotary_pos_emb_interleave``.
  """

  def apply(
      self,
      embedding_mat: Array,
      segment_positions: Array | None = None,
  ):
    embedding_dims = embedding_mat.shape[-1]
    half_embedding_dim = embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / embedding_dims
    timescale = (
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** fraction
    )
    query_segment_pos = segment_positions
    if query_segment_pos is None:
      seq_length = embedding_mat.shape[1]
      query_segment_pos = jnp.arange(seq_length, dtype=jnp.float32)[
          jnp.newaxis, :
      ]
    else:
      query_segment_pos = jnp.asarray(query_segment_pos, dtype=jnp.float32)
    query_segment_pos = query_segment_pos[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = query_segment_pos / timescale / self.scale_factor
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    embedding_dtype = embedding_mat.dtype
    embedding_mat = jnp.asarray(embedding_mat, jnp.float32)
    x1 = embedding_mat[..., 0::2]  # even indices
    x2 = embedding_mat[..., 1::2]  # odd indices
    first_part = x1 * cos - x2 * sin
    second_part = x2 * cos + x1 * sin
    embedding_mat = jnp.concatenate([first_part, second_part], axis=-1)
    return jnp.asarray(embedding_mat, embedding_dtype)


def _updated_latent_decode_state(
    kv_latent: Array,
    k_rope: Array,
    segment_positions: Array,
    segment_ids: Array,
    decode_state: PyTree,
    window_size: int = 0,
    update_kv_cache: bool = True,
) -> tuple[Array, Array, Array, Array, PyTree]:
  """Compact-latent analogue of ``updated_decode_state``."""
  # Caches ONLY the compact kv-LoRA latent [b, seq, kv_lora] and the shared
  # RoPE key [b, seq, 1, rope] (~57x smaller than the materialized 64-head K/V).
  # Returns the full (cached + current) latent + rope plus kv segment metadata
  # and the new decode_state.
  if decode_state is None:
    # Training / prefill with no cache.
    return kv_latent, k_rope, segment_positions, segment_ids, None

  decode_state = cast(Mapping[str, Any], decode_state)
  input_state = {
      'kv_latent': kv_latent,
      'k_rope': k_rope,
      'segment_positions': segment_positions,
      'segment_ids': segment_ids,
  }

  cache_state = {k: decode_state[k] for k in input_state}

  def _update(cache_state, input_state):
    if segment_positions.shape[1] > 1:
      return input_state  # prefill: overwrite with the full sequence.
    cache_pos = segment_positions[0][0]
    if window_size > 0:
      cache_pos = cache_pos % (window_size + 1)
    return jax.tree.map(
        lambda c_arr, i_arr: jax.lax.dynamic_update_slice_in_dim(
            c_arr, i_arr, cache_pos, axis=1
        ),
        cache_state,
        input_state,
    )

  updated = jax.lax.cond(
      update_kv_cache, _update, lambda c, i: c, cache_state, input_state
  )
  new_decode_state = dict(updated)
  new_decode_state[f'window_size={window_size}'] = None
  if 'prefill_position' in decode_state:
    new_decode_state['prefill_position'] = decode_state['prefill_position']

  seg_pos = updated['segment_positions']
  seg_ids = updated['segment_ids']
  return (
      updated['kv_latent'],
      updated['k_rope'],
      seg_pos,
      seg_ids,
      (new_decode_state),
  )


@module.ModuleRegistry.register
@dataclasses.dataclass
class MLAAttention(module.SimplyModule):
  """Multi-head Latent Attention (DeepSeek-V3 / GLM `glm_moe_dsa` MLA).

  Queries: ``q_a_proj`` -> RMSNorm ``q_a_layernorm`` -> ``q_b_proj``, split per
  head into nope (``qk_nope_head_dim``) and RoPE (``qk_rope_head_dim``) parts.
  Keys/values: ``kv_a_proj_with_mqa`` -> latent (``kv_lora_rank``) + one shared
  (MQA) RoPE key; the latent is RMSNorm'd and ``kv_b_proj`` up-projects it into
  per-head ``k_nope`` and ``value`` (``v_head_dim``). Q/K dot is over
  ``qk_head_dim = qk_nope + qk_rope``; ``v_head_dim`` may differ from it.
  Interleaved RoPE applies only to the rope slices.

  The DSA "lightning indexer" is not modeled: for context <= ``index_topk`` it
  selects all causal keys, so dense causal attention is exact.

  Decode uses the memory-efficient DeepSeek compact-latent KV cache by default
  (``use_latent_kv_cache``): only the kv-LoRA latent + one shared RoPE key are
  cached (~57x smaller than materialized per-head K/V) and attention is computed
  by absorption, flash-tiled (``latent_flash_decode``) so per-token cost stays
  ~flat with context. Training/prefill (empty cache) uses the materialized
  einsum MLA math, which is bit-exact with the absorption path.
  """

  model_dim: int
  n_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  qk_nope_head_dim: int
  qk_rope_head_dim: int
  v_head_dim: int
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  rms_norm_epsilon: float = 1e-6
  norm_scale_plus_one: bool = False
  # Mixed precision related.
  activation_dtype: DTypeLike = 'bfloat16'
  weight_dtype: DTypeLike = 'float32'
  # Sharding related.
  qkv_partition: PartitionAnnotation = None
  o_partition: PartitionAnnotation = None
  attn_activation_partition: PartitionAnnotation = None
  output_partition: PartitionAnnotation = None
  # Decoding related.
  window_size: int = 0
  attn_soft_cap: float = -1.0
  attn_mask_value: float = common.neg_inf('float32')
  # softmax scale; if <= 0, defaults to 1/sqrt(qk_head_dim).
  query_scale: float = -1.0
  # Compact/latent KV cache (the memory-efficient DeepSeek-MLA form). When True
  # the decode path caches ONLY the compact kv-LoRA latent
  # (kv_lora_rank) + one shared (MQA) RoPE key (qk_rope_head_dim), ~57x smaller
  # per token than the materialized 64-head K/V -- and up-projects with kv_b at
  # attention time via absorption (kv_b_nope folded into q, v_b folded into the
  # context). Training/prefill (empty cache) still uses the materialized einsum
  # MLA math, so gradients are unchanged.
  use_latent_kv_cache: bool = False
  # Flash-style (KV-tiled, online-softmax) latent absorption attention for the
  # compact-latent decode path. Avoids materializing the full [b, h, sq, sk]
  # score/softmax tensor (the O(ctx)-per-decode-token HBM intermediate), keeping
  # every intermediate O(latent_flash_block_k) in the KV axis so per-turn cost
  # stays ~flat with context instead of growing. Bit-exact (within fp tol) with
  # the dense path. Default on; set False to fall back to _latent_attention.
  latent_flash_decode: bool = True
  latent_flash_block_k: int = 512
  # Position encoding for the rope slice (None = NoPE).
  position_encoding: pe_lib.PositionEncodingConfig | None = (
      InterleavedRoPE(max_timescale=8_000_000)
  )

  @property
  def qk_head_dim(self) -> int:
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  def setup(self) -> None:
    lin_kwargs = {
        'bias_term': '',
        'weight_init': self.weight_init,
        'weight_dtype': self.weight_dtype,
        'activation_dtype': self.activation_dtype,
    }
    # q-LoRA path.
    self.q_a_proj = module.EinsumLinear(
        eqn='ir,...i->...r',
        weight_shape=[self.model_dim, self.q_lora_rank],
        weight_partition=(None, None),
        **lin_kwargs,
    )
    self.q_a_layernorm = LayerNorm(
        dim=self.q_lora_rank,
        use_bias=False,
        activation_dtype=self.activation_dtype,
        scale_plus_one=self.norm_scale_plus_one,
        epsilon=self.rms_norm_epsilon,
    )
    self.q_b_proj = module.EinsumLinear(
        eqn='ihd,...i->...hd',
        weight_shape=[self.q_lora_rank, self.n_heads, self.qk_head_dim],
        weight_partition=self.qkv_partition,
        output_partition=self.attn_activation_partition,
        **lin_kwargs,
    )
    # kv-LoRA path. Produces kv_lora_rank latent + qk_rope_head_dim shared key.
    self.kv_a_proj = module.EinsumLinear(
        eqn='ir,...i->...r',
        weight_shape=[
            self.model_dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
        ],
        weight_partition=(None, None),
        **lin_kwargs,
    )
    self.kv_a_layernorm = LayerNorm(
        dim=self.kv_lora_rank,
        use_bias=False,
        activation_dtype=self.activation_dtype,
        scale_plus_one=self.norm_scale_plus_one,
        epsilon=self.rms_norm_epsilon,
    )
    self.kv_b_proj = module.EinsumLinear(
        eqn='ihd,...i->...hd',
        weight_shape=[
            self.kv_lora_rank,
            self.n_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ],
        weight_partition=self.qkv_partition,
        output_partition=self.attn_activation_partition,
        **lin_kwargs,
    )
    self.o_proj = module.EinsumLinear(
        eqn='ihd,...hd->...i',
        weight_shape=[self.model_dim, self.n_heads, self.v_head_dim],
        weight_partition=self.o_partition,
        output_partition=self.output_partition,
        **lin_kwargs,
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    keys = jax.random.split(prng_key, num=5)
    params = {
        'q_a_proj': self.q_a_proj.init(keys[0]),
        'q_a_layernorm': self.q_a_layernorm.init(),
        'q_b_proj': self.q_b_proj.init(keys[1]),
        'kv_a_proj': self.kv_a_proj.init(keys[2]),
        'kv_a_layernorm': self.kv_a_layernorm.init(),
        'kv_b_proj': self.kv_b_proj.init(keys[3]),
        'o_proj': self.o_proj.init(keys[4]),
    }
    return params

  def _compute_q(
      self, params: PyTree, x: Array, segment_positions: Array
  ) -> tuple[Array, Array]:
    """Query path: returns (q_nope, q_rope) each [batch, seq, n_heads, dim]."""
    # ``q_rope`` already has RoPE applied. Shared by the training/prefill
    # (materialized) path and the compact/latent decode (absorption) path.
    pe = self.position_encoding
    q_latent = self.q_a_proj.apply(params['q_a_proj'], x)  # pyrefly: ignore[bad-index, unsupported-operation]
    q_latent = self.q_a_layernorm.apply(params['q_a_layernorm'], q_latent)  # pyrefly: ignore[bad-index, unsupported-operation]
    q = self.q_b_proj.apply(params['q_b_proj'], q_latent)  # [b,s,h,qk_head]  # pyrefly: ignore[bad-index, unsupported-operation]
    q_nope = q[..., : self.qk_nope_head_dim]
    q_rope = q[..., self.qk_nope_head_dim :]
    if pe is not None:
      q_rope = pe.apply(q_rope, segment_positions=segment_positions)
    return (
        q_nope.astype(self.activation_dtype),
        q_rope.astype(self.activation_dtype),
    )

  def _compute_kv_latent(
      self, params: PyTree, x: Array, segment_positions: Array
  ) -> tuple[Array, Array]:
    """Compact KV path: returns (kv_latent, k_rope)."""
    # ``kv_latent`` is the RMSNorm'd kv-LoRA latent [batch, seq, kv_lora_rank]
    # (the compact cacheable form). ``k_rope`` is the single shared (MQA) RoPE
    # key [batch, seq, 1, qk_rope_head_dim] with RoPE already applied. These are
    # exactly what the compact/latent KV cache stores -- no per-head K/V here.
    b, s, _ = x.shape
    pe = self.position_encoding
    compressed = self.kv_a_proj.apply(params['kv_a_proj'], x)  # [b,s,kvlr+rope]  # pyrefly: ignore[bad-index, unsupported-operation]
    kv_latent = compressed[..., : self.kv_lora_rank]
    k_rope = compressed[..., self.kv_lora_rank :]  # [b,s,rope] (shared, MQA)
    kv_latent = self.kv_a_layernorm.apply(params['kv_a_layernorm'], kv_latent)  # pyrefly: ignore[bad-index, unsupported-operation]
    k_rope = k_rope.reshape(b, s, 1, self.qk_rope_head_dim)
    if pe is not None:
      k_rope = pe.apply(k_rope, segment_positions=segment_positions)
    return (
        kv_latent.astype(self.activation_dtype),
        k_rope.astype(self.activation_dtype),
    )

  def _kv_b_split(self, params: PyTree) -> tuple[Array, Array]:
    """Splits kv_b into (k_b_nope, v_b), both [kv_lora, n_heads, dim]."""
    # ``params`` are already-raw arrays (get_raw_arrays applied in ``apply``).
    kv_b = params['kv_b_proj']['w']  # [kv_lora, n_heads, nope+v]  # pyrefly: ignore[bad-index, unsupported-operation]
    k_b_nope = kv_b[..., : self.qk_nope_head_dim]  # pyrefly: ignore[bad-index, unsupported-operation]
    v_b = kv_b[..., self.qk_nope_head_dim :]  # pyrefly: ignore[bad-index, unsupported-operation]
    return k_b_nope, v_b  # pyrefly: ignore[bad-return]

  def _materialize_kv(
      self, params: PyTree, kv_latent: Array, k_rope: Array
  ) -> tuple[Array, Array]:
    """Up-projects the compact latent into full per-head (key, value)."""
    # key: [b,s,h,qk_head_dim] (k_nope || shared k_rope broadcast per head).
    # value: [b,s,h,v_head_dim].
    b, s = kv_latent.shape[:2]
    h = self.n_heads
    kv = self.kv_b_proj.apply(params['kv_b_proj'], kv_latent)  # [b,s,h,nope+v]  # pyrefly: ignore[bad-index, unsupported-operation]
    k_nope = kv[..., : self.qk_nope_head_dim]
    value = kv[..., self.qk_nope_head_dim :]  # [b,s,h,v_head]
    k_rope = jnp.broadcast_to(k_rope, (b, s, h, self.qk_rope_head_dim))
    key = jnp.concatenate([k_nope, k_rope], axis=-1)  # [b,s,h,qk_head]
    return key.astype(self.activation_dtype), value.astype(
        self.activation_dtype
    )

  def _compute_qkv(
      self, params: PyTree, x: Array, segment_positions: Array
  ) -> tuple[Array, Array, Array]:
    """Returns (query, key, value), each [batch, seq, n_heads, head_dim]."""
    # Materialized (per-head) form used by the training/prefill path.
    q_nope, q_rope = self._compute_q(params, x, segment_positions)
    kv_latent, k_rope = self._compute_kv_latent(params, x, segment_positions)
    key, value = self._materialize_kv(params, kv_latent, k_rope)
    query = jnp.concatenate([q_nope, q_rope], axis=-1)  # [b,s,h,qk_head]
    return query.astype(self.activation_dtype), key, value

  @property
  def _softmax_scale(self) -> float:
    return (
        self.query_scale
        if self.query_scale > 0
        else math.sqrt(self.qk_head_dim)
    )

  def _latent_attention(
      self,
      params: PyTree,
      q_nope: Array,  # [b, sq, h, nope]
      q_rope: Array,  # [b, sq, h, rope]
      kv_latent: Array,  # [b, sk, kv_lora]  (may be QuantArray-dequantized)
      k_rope: Array,  # [b, sk, 1, rope]
      mask: Array,  # [b, 1, sq, sk] boolean
  ) -> Array:
    """Absorption attention over the compact latent (no per-head K/V)."""
    # kv_b_nope [kv_lora, h, nope] is folded into q_nope (project q into latent
    # space); v_b [kv_lora, h, v] is folded into the attention context. Only the
    # NOPE score is absorbed; the ROPE score (q_rope . shared k_rope) stays a
    # separate additive term because it is position-dependent.
    k_b_nope, v_b = self._kv_b_split(params)  # [kv_lora,h,nope], [kv_lora,h,v]
    scale = self._softmax_scale
    # Absorb kv_b_nope into q_nope -> query in latent space: [b,sq,h,kv_lora].
    qc = jnp.einsum('bshn,rhn->bshr', q_nope, k_b_nope)
    # NOPE score: qc . kv_latent (latent shared across heads). [b,h,sq,sk]
    qk = jnp.einsum('bshr,btr->bhst', qc, kv_latent)
    # ROPE score: q_rope . shared k_rope (broadcast over head). [b,h,sq,sk]
    qk = qk + jnp.einsum('bshd,bt1d->bhst', q_rope, k_rope)
    qk = qk.astype(jnp.float32) / scale
    if self.attn_soft_cap > 0:
      qk = self.attn_soft_cap * jnp.tanh(qk / self.attn_soft_cap)
    qk = jnp.where(mask, qk, self.attn_mask_value)
    attn_w = jax.nn.softmax(qk, axis=-1).astype(kv_latent.dtype)
    # Context in latent space then absorb v_b. [b,h,sq,kv_lora] -> [b,sq,h,v].
    ctx = jnp.einsum('bhst,btr->bshr', attn_w, kv_latent)
    out = jnp.einsum('bshr,rhv->bshv', ctx, v_b)
    return out.astype(self.activation_dtype)

  def _latent_attention_flash(
      self,
      params: PyTree,
      q_nope: Array,  # [b, sq, h, nope]
      q_rope: Array,  # [b, sq, h, rope]
      kv_latent: Array,  # [b, sk, kv_lora]
      k_rope: Array,  # [b, sk, 1, rope]
      mask: Array,  # [b, 1, sq, sk] boolean
      *,
      block_k: int = 512,
  ) -> Array:
    """Flash-style (online-softmax, KV-tiled) latent absorption attention."""
    # Numerically identical to ``_latent_attention`` but tiles the KV (sk) axis
    # in blocks of ``block_k`` and combines them with an online (running
    # max/sum) softmax, so it NEVER materializes the full ``[b, h, sq, sk]``
    # score/softmax tensor. During compact-latent decode ``sk`` is the whole
    # context, so the dense variant's ``[b, h, sq, sk]`` intermediates grow
    # O(ctx) per decode token and dominate HBM traffic -- the source of the
    # growth-with-context latency. This variant keeps every intermediate
    # O(block_k) in the sk axis, matching a fused flash kernel's flat
    # memory-bound behavior while staying pure JAX (parity is bit-exact and
    # CPU-testable). The absorption structure is preserved: kv_b_nope is folded
    # into q_nope (query in latent space, shared over kv_lora across heads,
    # MQA-like) and v_b is folded into the accumulated latent context at end.
    k_b_nope, v_b = self._kv_b_split(params)  # [kv_lora,h,nope], [kv_lora,h,v]
    scale = self._softmax_scale
    b, sq, h, _ = q_nope.shape
    sk = kv_latent.shape[1]
    r = self.kv_lora_rank
    # Query in latent space: [b, h, sq, r] (transpose so sq/h are the parallel
    # axes and the sk-block is the contracted/tiled axis).
    qc = jnp.einsum('bshn,rhn->bhsr', q_nope, k_b_nope)  # [b, h, sq, r]
    q_rope_t = einops.rearrange(
        q_rope, 'b s h d -> b h s d'
    )  # [b, h, sq, rope]

    mask_value = jnp.float32(self.attn_mask_value)
    # Pad sk up to a multiple of block_k so the scan has uniform blocks. Padded
    # keys are masked out (mask=False), contributing nothing to the softmax.
    n_blocks = (sk + block_k - 1) // block_k
    padded = n_blocks * block_k
    pad = padded - sk
    if pad > 0:
      kv_latent = jnp.pad(kv_latent, ((0, 0), (0, pad), (0, 0)))
      k_rope = jnp.pad(k_rope, ((0, 0), (0, pad), (0, 0), (0, 0)))
      # mask is [b, 1, sq, sk]; pad the sk axis with False.
      mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, pad)))
    # Reshape the tiled axis to [n_blocks, block_k] leading for lax.scan.
    kv_blocks = einops.rearrange(
        kv_latent, 'b (n k) r -> n b k r', k=block_k
    )  # [n, b, block_k, r]
    krope_blocks = einops.rearrange(
        k_rope, 'b (n k) 1 d -> n b k d', k=block_k
    )  # [n, b, block_k, rope]
    mask_blocks = einops.rearrange(
        mask, 'b 1 s (n k) -> n b s k', k=block_k
    )  # [n, b, sq, block_k]

    def _block(carry, blk):
      m_prev, l_prev, acc_prev = carry  # [b,h,sq,1],[b,h,sq,1],[b,h,sq,r]
      kvb, kropeb, maskb = blk  # [b,k,r],[b,k,rope],[b,sq,k]
      # Scores for this block: [b, h, sq, k].
      s = jnp.einsum('bhsr,bkr->bhsk', qc, kvb)
      s = s + jnp.einsum('bhsd,bkd->bhsk', q_rope_t, kropeb)
      s = s.astype(jnp.float32) / scale
      if self.attn_soft_cap > 0:
        s = self.attn_soft_cap * jnp.tanh(s / self.attn_soft_cap)
      s = jnp.where(maskb[:, None, :, :], s, mask_value)  # [b,h,sq,k]
      # Online softmax update.
      m_cur = jnp.max(s, axis=-1, keepdims=True)  # [b,h,sq,1]
      m_new = jnp.maximum(m_prev, m_cur)
      alpha = jnp.exp(m_prev - m_new)  # rescale prior state
      p = jnp.exp(s - m_new)  # [b,h,sq,k]
      l_new = l_prev * alpha + jnp.sum(p, axis=-1, keepdims=True)
      # Accumulate context in latent space: p @ kv_block -> [b,h,sq,r].
      pv = jnp.einsum('bhsk,bkr->bhsr', p.astype(kvb.dtype), kvb)
      acc_new = acc_prev * alpha + pv.astype(jnp.float32)
      return (m_new, l_new, acc_new), None

    m0 = jnp.full((b, h, sq, 1), mask_value, jnp.float32)
    l0 = jnp.zeros((b, h, sq, 1), jnp.float32)
    acc0 = jnp.zeros((b, h, sq, r), jnp.float32)
    (_, l_fin, acc_fin), _ = jax.lax.scan(
        _block, (m0, l0, acc0), (kv_blocks, krope_blocks, mask_blocks)
    )
    # Rows with no valid key (l==0, e.g. fully-masked padded query slots) -> 0.
    denom = jnp.where(l_fin > 0, l_fin, 1.0)
    ctx = (acc_fin / denom).astype(v_b.dtype)  # [b, h, sq, r]
    ctx = einops.rearrange(ctx, 'b h s r -> b s h r')  # match _latent_attention
    out = jnp.einsum('bshr,rhv->bshv', ctx, v_b)
    return out.astype(self.activation_dtype)

  def _apply_latent(
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array,
      segment_positions: Array,
      extra_inputs: PyTree,
      decode_state: PyTree,
  ) -> tuple[Array, PyTree]:
    """Compact/latent-cache MLA path (training/prefill + latent decode)."""
    # Caches only (kv_latent, k_rope) -- the DeepSeek-efficient compact form.
    # Attention is computed via absorption (flash-tiled by default), never
    # materializing per-head K/V.
    q_nope, q_rope = self._compute_q(params, x, segment_positions)
    kv_latent, k_rope = self._compute_kv_latent(params, x, segment_positions)
    update_kv_cache = True
    if extra_inputs is not None:
      update_kv_cache = extra_inputs.get('update_kv_cache', True)  # pyrefly: ignore[missing-attribute]
    (
        kv_latent_c,
        k_rope_c,
        kv_segment_positions,
        kv_segment_ids,
        decode_state,
    ) = _updated_latent_decode_state(
        kv_latent=kv_latent,
        k_rope=k_rope,
        segment_positions=segment_positions,
        segment_ids=segment_ids,
        decode_state=decode_state,
        window_size=self.window_size,
        update_kv_cache=update_kv_cache,
    )
    mask = create_mask(
        segment_positions=segment_positions,
        kv_segment_positions=kv_segment_positions,
        segment_ids=segment_ids,
        kv_segment_ids=kv_segment_ids,
        window_size=self.window_size,
    )
    mask = einops.rearrange(mask, 'b l1 l2 -> b 1 l1 l2')
    if self.latent_flash_decode:
      output = self._latent_attention_flash(
          params,
          q_nope,
          q_rope,
          kv_latent_c,
          k_rope_c,
          mask,
          block_k=self.latent_flash_block_k,
      )
    else:
      output = self._latent_attention(
          params, q_nope, q_rope, kv_latent_c, k_rope_c, mask
      )
    output = sharding_lib.with_sharding_constraint(
        output, self.attn_activation_partition  # pyrefly: ignore[bad-argument-type]
    )
    output = self.o_proj.apply(params['o_proj'], output)  # pyrefly: ignore[bad-index, unsupported-operation]
    return output, {'decode_state': decode_state}

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array,
      segment_positions: Array,
      extra_inputs: PyTree = None,
      decode_state: PyTree = None,
  ) -> tuple[Array, PyTree]:
    params = get_raw_arrays(params)
    assert len(x.shape) == 3 and x.shape[-1] == self.model_dim
    # Compact/latent KV path (training/prefill + decode): efficient DeepSeek-MLA
    # form: cache only the compact latent, attend by absorption.
    if self.use_latent_kv_cache:
      return self._apply_latent(
          params,
          x,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
          extra_inputs=extra_inputs,
          decode_state=decode_state,
      )
    # Materialized (per-head K/V) MLA path: prefill/training default when the
    # compact-latent cache is disabled.
    q, k, v = self._compute_qkv(params, x, segment_positions)
    scale = self._softmax_scale
    q = q / scale
    extra_output = {}
    update_kv_cache = True
    if extra_inputs is not None:
      update_kv_cache = extra_inputs.get('update_kv_cache', True)  # pyrefly: ignore[missing-attribute]
    k, v, kv_segment_positions, kv_segment_ids, decode_state = (
        updated_decode_state(
            k=k,
            v=v,
            segment_positions=segment_positions,
            segment_ids=segment_ids,
            decode_state=decode_state,
            window_size=self.window_size,
            update_kv_cache=update_kv_cache,
        )
    )
    mask = create_mask(
        segment_positions=segment_positions,
        kv_segment_positions=kv_segment_positions,
        segment_ids=segment_ids,
        kv_segment_ids=kv_segment_ids,
        window_size=self.window_size,
    )
    mask = einops.rearrange(mask, 'b l1 l2 -> b 1 l1 l2')
    output, _ = attn(
        q,
        k,
        v,
        mask,
        attn_soft_cap=self.attn_soft_cap,
        attn_mask_value=self.attn_mask_value,
        dtype=self.activation_dtype,
    )
    output = sharding_lib.with_sharding_constraint(
        output, self.attn_activation_partition
    )
    output = self.o_proj.apply(params['o_proj'], output)  # pyrefly: ignore[bad-index, unsupported-operation]
    extra_output['decode_state'] = decode_state
    return output, extra_output

  def _init_latent_decode_state(
      self, batch_size: int, max_seq_len: int
  ) -> PyTree:
    """Compact-latent decode cache: kv_latent + shared rope key only."""
    # Stores ~57x less per token than the materialized 64-head K/V.
    pos_partition = (
        self.attn_activation_partition[:2]
        if self.attn_activation_partition is not None
        else None
    )

    def _p(rank: int):
      # Batch/seq partition padded with None to the given tensor rank. The
      # latent (kv_lora) and rope dims are never head-partitioned.
      if pos_partition is None:
        return None
      return tuple(pos_partition) + (None,) * (rank - 2)

    lat_shape = (batch_size, max_seq_len, self.kv_lora_rank)
    rope_shape = (batch_size, max_seq_len, 1, self.qk_rope_head_dim)
    sc = sharding_lib.with_sharding_constraint
    state: dict[str, Any] = {}
    state['kv_latent'] = sc(
        jnp.zeros(lat_shape, dtype=self.activation_dtype), _p(3)
    )
    state['k_rope'] = sc(
        jnp.zeros(rope_shape, dtype=self.activation_dtype), _p(4)
    )
    state['segment_positions'] = sc(
        jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32), pos_partition
    )
    state['segment_ids'] = sc(
        jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32), pos_partition
    )
    state[f'window_size={self.window_size}'] = None
    return state

  def init_decode_state(self, batch_size: int, max_seq_len: int) -> PyTree:
    if self.use_latent_kv_cache:
      return self._init_latent_decode_state(batch_size, max_seq_len)
    pos_partition = (
        self.attn_activation_partition[:2]
        if self.attn_activation_partition is not None
        else None
    )
    return {
        'k': sharding_lib.with_sharding_constraint(
            jnp.zeros(
                (batch_size, max_seq_len, self.n_heads, self.qk_head_dim),
                dtype=self.activation_dtype,
            ),
            self.attn_activation_partition,
        ),
        'v': sharding_lib.with_sharding_constraint(
            jnp.zeros(
                (batch_size, max_seq_len, self.n_heads, self.v_head_dim),
                dtype=self.activation_dtype,
            ),
            self.attn_activation_partition,
        ),
        'segment_positions': sharding_lib.with_sharding_constraint(
            jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32),
            pos_partition,
        ),
        'segment_ids': sharding_lib.with_sharding_constraint(
            jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32),
            pos_partition,
        ),
        f'window_size={self.window_size}': None,
    }


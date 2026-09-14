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
"""Qwen3.8's gated softmax attention (16 of the 64 layers).

The math is HF `Qwen3_5Attention.forward` + `eager_attention_forward`
(`transformers/models/qwen3_5/modeling_qwen3_5.py`, release
`Qwen/Qwen3.8-27B`), i.e. grouped-query attention -- 24 query heads over 4
key/value heads of 256 -- with three Qwen twists, in this order:

    q, gate = split(q_proj(x), 2, axis=-1)      # per head, 2 * 256 wide
    q       = rms_norm(q, q_norm) ; k = rms_norm(k_proj(x), k_norm)
    q, k    = mrope(q), mrope(k)                # `utils/rope.Qwen38RoPE`
    o       = softmax(q k^T / sqrt(256) + mask) v
    y       = o_proj(o * sigmoid(gate))

  * **q/k head norms** -- an RMSNorm over the 256 channels of each head,
    applied BEFORE the rotary, so the norm sees unrotated channels. Both are
    HF `Qwen3_5RMSNorm`: `x_normed * (1 + w)`, which is core
    `model_lib.LayerNorm(scale_plus_one=True)`.
  * **partial interleaved mRoPE** -- only the leading quarter of each head is
    rotated; see `utils/rope.py`.
  * **a sigmoid output gate** -- `q_proj` emits `2 * per_head_dim` per head and
    the second half gates the attention output through `sigmoid`, elementwise
    per (head, channel), before `o_proj`. It is NOT the silu ("swish") gate:
    `config.json:output_gate_type = "swish"` names the *GatedDeltaNet*
    output-norm gate (`Qwen3_5RMSNormGated`), a different layer. HF
    hardcodes `torch.sigmoid` here with no config knob; `attn_multi_device_test.py` A/Bs the
    two activations so a silu regression cannot pass.

`cos`/`sin` are rebuilt per layer, once for q and once for k, as core's
`model_lib.Attention` also does; HF builds them once in `Qwen3_5Model` and
passes them down. At 16 attention layers that is 32 reconstructions of a
`[3, B, T, 32]` table -- redundant transcendental work, no extra live memory
(XLA fuses it), and the sharing would have to go through `Qwen38Block`.

There is no soft cap (core's `attn` defaults to 50, which HF does not have),
no sliding window (`layer_types` only ever says `full_attention`), and no bias
on any projection.

Decoding uses core's mapping KV cache verbatim -- `model_lib.Attention(
...).init_decode_state` allocates it and `model_lib.updated_decode_state`
maintains it (a multi-token call replaces the buffers, a single-token call
writes one slot) -- so `model_lib.pad_decode_state_to` grows this layer with no
registration, unlike the GatedDeltaNet layers next door. The cache is dense
`[B, max_seq_len, n_kv_heads, per_head_dim]` and the attention costs the
allocated length, not the filled one; a paged variant is out of scope for this
package (see README).
"""

import dataclasses
from typing import Any

import einops
import jax
import jax.numpy as jnp
from simply import model_lib
from simply.utils import common
from simply.utils import initializer
from simply.utils import module
from simply.utils import position_encoding as pe_lib
from simply.utils import sharding as sharding_lib
from simply.zoo.qwen3p8.utils import rope as rope_lib

Array = common.Array
DTypeLike = jax.typing.DTypeLike
PRNGKey = jax.typing.ArrayLike
PartitionAnnotation = common.PartitionAnnotation
PyTree = Any
SimplyConfig = Any

get_raw_arrays = common.get_raw_arrays

# Finite, so a fully masked row (an all-pad query) yields a uniform
# distribution instead of NaN -- the Simply convention.
_MASK_VALUE = common.neg_inf('float32')
# HF caps nothing; core's `attn` caps at 50.0 unless this is <= 0.
_NO_SOFT_CAP = -1.0


# --- Attention core ---------------------------------------------------------


def grouped_query_attention(
    q: Array,
    k: Array,
    v: Array,
    mask: Array,
    *,
    dtype: DTypeLike,
) -> Array:
  """Masked softmax attention with query heads grouped over the KV heads.

  Query head `i` reads KV head `i // (H / H_kv)`, which is HF `repeat_kv`'s
  grouping (`expand` then `reshape`, so each KV head serves a contiguous run of
  query heads). Core `model_lib.attn` implements exactly that once the group
  axis is split out, and the softmax runs in float32 inside it.

  Args:
    q: `[B, T, H, D]` queries, already scaled by `1 / sqrt(D)`.
    k: `[B, S, H_kv, D]` keys.
    v: `[B, S, H_kv, D]` values.
    mask: `[B, T, S]` bool; True is attendable.
    dtype: Dtype of the probability-times-value matmul.

  Returns:
    `[B, T, H, D]` per-head attention outputs.
  """
  grouped = einops.rearrange(
      q, '... (n_kv g) d -> ... n_kv g d', n_kv=k.shape[-2]
  )
  output, _ = model_lib.attn(
      q=grouped,
      k=k,
      v=v,
      mask=einops.rearrange(mask, 'b t s -> b 1 1 t s'),
      attn_soft_cap=_NO_SOFT_CAP,
      attn_mask_value=_MASK_VALUE,
      dtype=dtype,
  )
  return jnp.asarray(
      einops.rearrange(output, '... n_kv g d -> ... (n_kv g) d')
  )


# --- Layer ------------------------------------------------------------------


@module.ModuleRegistry.register
@dataclasses.dataclass
class Qwen38Attention(module.SimplyModule):
  """Qwen3.8 gated grouped-query attention.

  Attributes:
    model_dim: Residual width `D`.
    n_heads: Query heads.
    n_kv_heads: Key/value heads; `n_heads` must be a multiple of it.
    per_head_dim: Channels per head, of query, key and value alike.
    attn_output_gate: Released as True and not ported otherwise; see `setup`.
    use_qk_norm: Released as True and not ported otherwise; see `setup`.
    rms_norm_epsilon: Epsilon of the q/k head norms (HF `rms_norm_eps`).
    position_encoding: Applied to q and k after the head norms. None is NoPE,
      which no released Qwen3.8 layer uses; it exists because core's attention
      has it and a NoPE ablation is one field away.
    activation_dtype: Matmul dtype; also the KV cache dtype.
    weight_dtype: Dtype of freshly initialized weights.
    weight_init: Initializer of the four projections.
    sharding_config: Source of the weight and activation partitions; None
      leaves every tensor unannotated.
  """

  model_dim: int
  n_heads: int = 24
  n_kv_heads: int = 4
  per_head_dim: int = 256
  attn_output_gate: bool = True
  use_qk_norm: bool = True
  rms_norm_epsilon: float = 1e-6
  position_encoding: pe_lib.PositionEncodingConfig | None = (
      rope_lib.Qwen38RoPE()
  )
  activation_dtype: DTypeLike = 'bfloat16'
  weight_dtype: DTypeLike = 'float32'
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  sharding_config: SimplyConfig | None = None

  def _activation_partition(self, name: str) -> PartitionAnnotation:
    """Partition of one activation, or `NOT_ANNOTATED` without a config.

    Args:
      name: 'q' for a `[B, T, H, D]` query-shaped tensor, 'kv' for a
        `[B, S, H_kv, D]` one, 'out' for a `[B, T, D]` one.

    Returns:
      The annotation to constrain that tensor with.
    """
    if self.sharding_config is None:
      return sharding_lib.NOT_ANNOTATED
    attn_partition = self.sharding_config.attn_activation_partition
    if name == 'out':
      return self.sharding_config.activation_partition
    if name == 'q' or attn_partition is None:
      return attn_partition
    # K/V shard on `per_head_dim` instead of on the 4 KV heads. Both are legal
    # under `eval/decode_eval._check_mesh`, which refuses `model > n_kv_heads`,
    # and at `model = 4` they compile to the same collectives; `per_head_dim`
    # (256) is preferred because it is the axis that stays divisible if a
    # deployment ever wants more model shards than there are KV heads. Entry 3
    # of the query annotation is dropped: it is None in every Simply sharding
    # config (`config_lib.BaseSharding.attn_activation_partition`).
    return (attn_partition[0], attn_partition[1], None, attn_partition[2])

  def setup(self) -> None:
    if not self.attn_output_gate or not self.use_qk_norm:
      raise ValueError(
          'Qwen3.8 attention always gates its output and always normalizes q'
          f' and k per head; {self.attn_output_gate=} {self.use_qk_norm=} asks'
          ' for a variant that is not ported.'
      )
    if self.n_heads % self.n_kv_heads:
      raise ValueError(
          f'{self.n_heads=} must be a multiple of {self.n_kv_heads=}.'
      )
    qkv_partition = None
    kv_partition = None
    o_partition = None
    if self.sharding_config is not None:
      qkv_partition = self.sharding_config.attn_qkv_partition
      if qkv_partition is not None:
        # Same choice as `_activation_partition`, and it has to be the same one
        # or every K/V projection would end in a reshard.
        kv_partition = (qkv_partition[0], None, qkv_partition[1])
      o_partition = self.sharding_config.attn_o_partition

    # `output_partition` is left at its `NOT_ANNOTATED` default throughout:
    # passing None would constrain every projection output to *replicated*
    # (`sharding.partition_spec(None) == PartitionSpec()`), forcing an
    # all-gather that the explicit constraints in `apply` immediately undo.
    def projection(
        heads: int, head_dim: int, partition: PartitionAnnotation
    ) -> module.EinsumLinear:
      return module.EinsumLinear(
          eqn='ihd,...i->...hd',
          weight_shape=[self.model_dim, heads, head_dim],
          bias_term='',
          weight_dtype=self.weight_dtype,
          activation_dtype=self.activation_dtype,
          weight_partition=partition,
          weight_init=self.weight_init,
      )

    def head_norm() -> model_lib.LayerNorm:
      # HF `Qwen3_5RMSNorm` is `x_normed * (1 + w)` over the head dim only,
      # with `w` zero-initialized, hence `scale_plus_one=True`. It scales in
      # float32 and rounds once at the end ("Llama does `x.to(float16) * w`
      # whilst Qwen3_5 is `(x * w).to(float16)`", modeling_qwen3_5.py:735),
      # while core's LayerNorm rounds to its `activation_dtype` *before* the
      # scale -- so the norms run in float32 and hand back the caller's dtype
      # (`LayerNorm.apply` restores `inputs_dtype`). In bfloat16 the
      # difference is a systematic per-channel error of up to 4e-3, because
      # `1 + w` itself is quantized.
      return model_lib.LayerNorm(
          dim=self.per_head_dim,
          use_bias=False,
          scale_plus_one=True,
          epsilon=self.rms_norm_epsilon,
          weight_dtype=self.weight_dtype,
          activation_dtype='float32',
      )

    # The gate rides in the second half of every query head's channels:
    # `w[i, h, j] = q_proj_hf[h * 2 * per_head_dim + j, i]`, i.e. the released
    # `[H * 2D, D]` matrix read as per-head `[query | gate]` blocks
    # (HF `q_proj(x).view(*input_shape, -1, head_dim * 2).chunk(2, -1)`).
    self.q_proj = projection(self.n_heads, 2 * self.per_head_dim, qkv_partition)
    self.k_proj = projection(self.n_kv_heads, self.per_head_dim, kv_partition)
    self.v_proj = projection(self.n_kv_heads, self.per_head_dim, kv_partition)
    self.o_proj = module.EinsumLinear(
        eqn='ihd,...hd->...i',
        weight_shape=[self.model_dim, self.n_heads, self.per_head_dim],
        bias_term='',
        weight_dtype=self.weight_dtype,
        activation_dtype=self.activation_dtype,
        weight_partition=o_partition,
        weight_init=self.weight_init,
    )
    self.q_norm = head_norm()
    self.k_norm = head_norm()

  def init(self, prng_key: PRNGKey) -> PyTree:
    keys = jax.random.split(prng_key, num=4)
    return {
        'q_proj': self.q_proj.init(keys[0]),
        'k_proj': self.k_proj.init(keys[1]),
        'v_proj': self.v_proj.init(keys[2]),
        'o_proj': self.o_proj.init(keys[3]),
        'q_norm': self.q_norm.init(),
        'k_norm': self.k_norm.init(),
    }

  def init_decode_state(self, batch_size: int, max_seq_len: int) -> PyTree:
    """Core's `{k, v, segment_positions, segment_ids}` mapping cache.

    Delegating to `model_lib.Attention` is what keeps
    `model_lib.pad_decode_state_to` working on this layer without a
    `pad_block_decode_state` registration: it grows any `block_*` mapping whose
    leaves have a sequence axis.

    Args:
      batch_size: Sequences in flight.
      max_seq_len: Positions to allocate. Core's prefill *replaces* the buffers
        with the prefill's own length, so a driver allocates the prefill length
        and grows the state afterwards.

    Returns:
      The cache mapping, annotated like this layer's own activations.
    """
    return model_lib.Attention(
        self.model_dim,
        self.n_heads,
        self.per_head_dim,
        n_kv_heads=self.n_kv_heads,
        activation_dtype=self.activation_dtype,
        attn_activation_partition=self._kv_cache_partition(),
    ).init_decode_state(batch_size, max_seq_len)

  def _kv_cache_partition(self) -> PartitionAnnotation:
    """`[B, S, H_kv, D]` cache annotation; None when there is no config."""
    partition = self._activation_partition('kv')
    return None if partition is sharding_lib.NOT_ANNOTATED else partition

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array,
      segment_positions: Array,
      extra_inputs: PyTree | None = None,
      decode_state: PyTree | None = None,
  ) -> tuple[Array, PyTree]:
    """Runs the layer.

    Args:
      params: This module's param subtree.
      x: `[B, T, D]` normed block input.
      segment_ids: `[B, T]` int32; 0 marks padding, which no valid query
        attends to (`model_lib.create_mask` matches ids exactly).
      segment_positions: `[B, T]` int32 absolute positions, used both for the
        rotary and for the causal mask. Required. For a decode step (`T == 1`)
        it is also the cache slot written -- core takes that slot from row 0
        for the whole batch, so a batch decodes in lockstep.
      extra_inputs: Side channel; `prefill_position` (core's `Sampler` sets it)
        is forwarded into the decode state, and `update_kv_cache` (default
        True) gates the cache write, as in `model_lib.updated_decode_state`.
      decode_state: Cache to attend over and extend; None runs stateless.

    Returns:
      `([B, T, D] output, {'decode_state': state or None})`.
    """
    raw: PyTree = get_raw_arrays(params)
    extra_inputs = extra_inputs or {}

    q_and_gate = jnp.asarray(self.q_proj.apply(raw['q_proj'], x))
    q_and_gate = sharding_lib.with_sharding_constraint(
        q_and_gate, self._activation_partition('q')
    )
    q, gate = jnp.split(q_and_gate, 2, axis=-1)
    gate = jnp.reshape(gate, (*x.shape[:-1], self.n_heads * self.per_head_dim))

    k = sharding_lib.with_sharding_constraint(
        jnp.asarray(self.k_proj.apply(raw['k_proj'], x)),
        self._activation_partition('kv'),
    )
    v = sharding_lib.with_sharding_constraint(
        jnp.asarray(self.v_proj.apply(raw['v_proj'], x)),
        self._activation_partition('kv'),
    )

    q = self.q_norm.apply(raw['q_norm'], q)
    k = self.k_norm.apply(raw['k_norm'], k)
    if self.position_encoding is not None:
      q = self.position_encoding.apply(q, segment_positions=segment_positions)
      k = self.position_encoding.apply(k, segment_positions=segment_positions)
    q = q * jnp.asarray(self.per_head_dim**-0.5, q.dtype)

    if (prefill_position := extra_inputs.get('prefill_position')) is not None:
      # Core's windowed-cache bookkeeping. Copied rather than mutated in place
      # so a caller's state is never rewritten under it.
      decode_state = {
          **(decode_state or {}),
          'prefill_position': prefill_position,
      }
    k, v, kv_segment_positions, kv_segment_ids, decode_state = (
        model_lib.updated_decode_state(
            k=k,
            v=v,
            segment_positions=segment_positions,
            segment_ids=segment_ids,
            decode_state=decode_state,
            window_size=0,
            update_kv_cache=extra_inputs.get('update_kv_cache', True),
        )
    )
    mask = model_lib.create_mask(
        segment_positions=segment_positions,
        kv_segment_positions=kv_segment_positions,
        segment_ids=segment_ids,
        kv_segment_ids=kv_segment_ids,
    )
    output = grouped_query_attention(
        q, k, v, mask, dtype=self.activation_dtype
    )
    output = sharding_lib.with_sharding_constraint(
        jnp.asarray(output), self._activation_partition('q')
    )

    output = jnp.asarray(einops.rearrange(output, '... n d -> ... (n d)'))
    output = output * jax.nn.sigmoid(gate)
    output = einops.rearrange(
        output, '... (n d) -> ... n d', n=self.n_heads, d=self.per_head_dim
    )
    output = sharding_lib.with_sharding_constraint(
        jnp.asarray(self.o_proj.apply(raw['o_proj'], output)),
        self._activation_partition('out'),
    )
    return output, {'decode_state': decode_state}

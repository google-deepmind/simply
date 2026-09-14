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
"""Kimi K3's gated NoPE multi-head latent attention (24 of the 93 layers).

The math is HF `KimiMLAAttention` (`modeling_kimi_linear.py`, release
`moonshotai/Kimi-K3`), i.e. DeepSeek-V3 MLA with two K3 twists:

  * **NoPE** -- no rotary embedding is applied anywhere. The `qk_rope_head_dim`
    channels exist (extra query dims plus one MQA-style key block shared by all
    heads) but are never rotated, so position reaches this layer only through
    the causal mask and nothing in the KV cache is position dependent.
  * **Output gate** -- `attn_out * sigmoid(g_proj(x))` before `o_proj`, with a
    full-rank per-(head, dim) gate.

The cached path is the *absorbed* form of the same math: `kv_b_proj` splits
into `W_UK | W_UV`, `W_UK` folds into the query (`q_nope @ W_UK`, linear, so
the softmax scale carries through) and `W_UV` is applied to the attention
context in the latent basis. Only the 576-unit compressed row
(`kv_lora_rank` + `qk_rope_head_dim`) is cached per token per layer, which is
what makes a 1M-token context affordable; the decompressed form
(`use_absorbed_decode=False`) is kept as an equivalence oracle.

The decompressing prefill core is a plain masked einsum; it is isolated in
`_decompressed_attention` / `_absorbed_attention` so a flash/splash kernel can
replace it without touching the projections, the gate or the cache. Two things
that core will have to fix, both out of scope here. (1) Absorption spends
`H*R*(dn+dv)` per QUERY token where decompression spends it per CACHED token,
but then costs 1088 instead of 320 MACs per (query, key, head): against a long
cache absorption stops paying at roughly 170 queries. Swapping cores is not the
answer, though, because decompressing a long cache materializes
`[B, S, H, dn+dv]` (49 KB per cached token) -- a many-query chunk needs a kernel
that decompresses per KV block. (2) Both cores cost the ALLOCATED `max_seq_len`,
not `lengths`, so a driver that wants `O(length)` decode must bucket its cache.
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
from simply.utils import sharding as sharding_lib

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KimiK3MLADecodeState:
  """Compressed KV cache of one MLA layer.

  Cache ORDER, not position, is what this layer attends on: NoPE makes a cached
  row valid wherever it lands. The invariants, shared with `kda.KDADecodeState`
  so a hybrid block can drive both mixers with one rule set:

    * a padded token (`segment_ids == 0`) is never written and never counted;
    * a non-pad token at `segment_positions == 0` opens a new sequence, and
      when it is the first valid token of the call the row's cache restarts at
      slot 0 (KDA resets its recurrent state on the same condition);
    * an append past `max_seq_len` saturates: the extra rows are dropped.

  It is a frozen registered dataclass, not the dict `model_lib.Sampler`
  expects, so K3 drives decoding through its own loop (`model_lib.py`): allocate
  with `init_decode_state`, then call `apply(..., decode_state=state)` for
  prefill and for every step.

  Attributes:
    kv_cache: `[B, max_seq_len, kv_lora_rank + qk_rope_head_dim]`. One row per
      token: the RMSNorm'd latent followed by the head-shared `k_rope` block.
        Both halves are read together by the absorbed core, so they share a
        buffer.
    lengths: `int32[B]`, number of valid rows per sequence.
  """

  kv_cache: Array
  lengths: Array

  @property
  def max_seq_len(self) -> int:
    """Row capacity of the cache, i.e. the longest sequence it can decode."""
    return self.kv_cache.shape[1]


@module.ModuleRegistry.register
@dataclasses.dataclass
class KimiK3MLA(module.SimplyModule):
  """Kimi K3 gated NoPE multi-head latent attention.

  Attributes:
    model_dim: Residual width `D`.
    n_heads: Query head count `H` (MLA has no separate KV heads).
    q_lora_rank: Rank of the query down-projection.
    kv_lora_rank: Rank of the KV latent, the cached payload.
    qk_nope_head_dim: Per-head query/key dims that come from the latent.
    qk_rope_head_dim: Per-head query dims paired with the head-shared key block.
      Never rotated (NoPE); the name follows the checkpoint.
    v_head_dim: Per-head value dims.
    use_output_gate: Applies `sigmoid(g_proj(x))` to the attention output.
    use_absorbed_decode: Cached path folds `W_UK`/`W_UV` around the latent
      instead of decompressing the cache. Same math; False is the oracle.
    rms_norm_epsilon: Epsilon of the two LoRA norms (HF `rms_norm_eps`).
    activation_dtype: Matmul dtype; also the KV cache dtype.
    weight_dtype: Dtype of freshly initialized weights.
    weight_init: Initializer of the projection weights.
    sharding_config: Source of the default partitions and of the activation
      constraints; None disables both.
    q_a_partition: Partition of `q_a_proj/w` `[D, Rq]`, and so on for the other
      `*_partition` fields. None means "derive from `sharding_config`"; pass an
      all-None tuple to force replication.
  """

  model_dim: int
  n_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  qk_nope_head_dim: int
  qk_rope_head_dim: int
  v_head_dim: int
  use_output_gate: bool = True
  use_absorbed_decode: bool = True
  rms_norm_epsilon: float = 1e-5
  activation_dtype: DTypeLike = 'bfloat16'
  weight_dtype: DTypeLike = 'float32'
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  sharding_config: SimplyConfig | None = None
  q_a_partition: PartitionAnnotation = None
  q_b_partition: PartitionAnnotation = None
  kv_a_partition: PartitionAnnotation = None
  kv_b_partition: PartitionAnnotation = None
  g_partition: PartitionAnnotation = None
  o_partition: PartitionAnnotation = None

  @property
  def q_head_dim(self) -> int:
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  @property
  def compressed_kv_dim(self) -> int:
    """Width of one cached row."""
    return self.kv_lora_rank + self.qk_rope_head_dim

  def _default_partitions(self) -> dict[str, PartitionAnnotation]:
    """Per-weight partitions derived from the standard Simply annotations.

    The LoRA bottlenecks (`q_a_proj`, `kv_a_proj`) keep their rank axis
    replicated -- 512/1536-wide bottlenecks would force a collective for no
    parallelism -- heads are column-sharded on the up-projections and the gate,
    and `o_proj` is row-sharded so the heads contract inside the shard.

    Returns:
      Annotation per weight prefix (`q_a`, `kv_a`, `q_b`, `kv_b`, `g`, `o`),
      empty when this module has no sharding config; a prefix is absent when
      the config leaves the corresponding projection unannotated.
    """
    if self.sharding_config is None:
      return {}
    qkv = self.sharding_config.attn_qkv_partition
    o = self.sharding_config.attn_o_partition
    partitions = {}
    if qkv is not None:
      partitions['q_a'] = (qkv[0], None)
      partitions['kv_a'] = (qkv[0], None)
      partitions['q_b'] = (None, qkv[1], None)
      partitions['kv_b'] = (None, qkv[1], None)
      partitions['g'] = (qkv[0], qkv[1], None)
    if o is not None:
      partitions['o'] = (o[1], None, o[0])
    return partitions

  def _activation_partition(self, rank: int) -> PartitionAnnotation:
    """Activation annotation of a rank-3 `[B, T, D]` or rank-4 head tensor."""
    if self.sharding_config is None:
      return sharding_lib.NOT_ANNOTATED
    if rank == 4:
      return self.sharding_config.attn_activation_partition
    return self.sharding_config.activation_partition

  def setup(self) -> None:
    # The cache carries no head axis, so it shards on batch only and stays
    # replicated over the model axes. Re-pinned on every append too, or a
    # jitted decode loop may reshard the whole buffer once per step.
    activation_partition = self._activation_partition(3)
    self._cache_partition: PartitionAnnotation = sharding_lib.NOT_ANNOTATED
    if (
        activation_partition is not sharding_lib.NOT_ANNOTATED
        and activation_partition is not None
    ):
      self._cache_partition = (activation_partition[0], None, None)
    defaults = self._default_partitions()

    def partition(name: str) -> PartitionAnnotation:
      explicit = getattr(self, f'{name}_partition')
      return defaults.get(name) if explicit is None else explicit

    def linear(
        eqn: str, weight_shape: list[int], name: str
    ) -> module.EinsumLinear:
      return module.EinsumLinear(
          eqn=eqn,
          weight_shape=weight_shape,
          bias_term='',
          weight_dtype=self.weight_dtype,
          activation_dtype=self.activation_dtype,
          weight_partition=partition(name),
          output_partition=None,
          weight_init=self.weight_init,
      )

    def norm(dim: int) -> model_lib.LayerNorm:
      return model_lib.LayerNorm(
          dim=dim,
          use_bias=False,
          scale_plus_one=False,
          epsilon=self.rms_norm_epsilon,
          weight_dtype=self.weight_dtype,
          activation_dtype=self.activation_dtype,
      )

    self.q_a_proj = linear(
        'ir,...i->...r', [self.model_dim, self.q_lora_rank], 'q_a'
    )
    self.q_a_norm = norm(self.q_lora_rank)
    self.q_b_proj = linear(
        'rhd,...r->...hd',
        [self.q_lora_rank, self.n_heads, self.q_head_dim],
        'q_b',
    )
    self.kv_a_proj = linear(
        'ir,...i->...r', [self.model_dim, self.compressed_kv_dim], 'kv_a'
    )
    self.kv_a_norm = norm(self.kv_lora_rank)
    # Also consumed as a raw array (split into W_UK | W_UV) by the absorbed
    # path, which is why it is not applied through the module there.
    self.kv_b_proj = linear(
        'rhd,...r->...hd',
        [
            self.kv_lora_rank,
            self.n_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ],
        'kv_b',
    )
    if self.use_output_gate:
      self.g_proj = linear(
          'ihd,...i->...hd',
          [self.model_dim, self.n_heads, self.v_head_dim],
          'g',
      )
    self.o_proj = linear(
        'hdi,...hd->...i',
        [self.n_heads, self.v_head_dim, self.model_dim],
        'o',
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    keys = jax.random.split(prng_key, num=6)
    params = {
        'q_a_proj': self.q_a_proj.init(keys[0]),
        'q_a_norm': self.q_a_norm.init(),
        'q_b_proj': self.q_b_proj.init(keys[1]),
        'kv_a_proj': self.kv_a_proj.init(keys[2]),
        'kv_a_norm': self.kv_a_norm.init(),
        'kv_b_proj': self.kv_b_proj.init(keys[3]),
    }
    if self.use_output_gate:
      params['g_proj'] = self.g_proj.init(keys[4])
    params['o_proj'] = self.o_proj.init(keys[5])
    return params

  def init_decode_state(
      self, batch_size: int, max_seq_len: int
  ) -> KimiK3MLADecodeState:
    kv_cache = jnp.zeros(
        (batch_size, max_seq_len, self.compressed_kv_dim),
        dtype=self.activation_dtype,
    )
    return KimiK3MLADecodeState(
        kv_cache=sharding_lib.with_sharding_constraint(
            kv_cache, self._cache_partition
        ),
        lengths=jnp.zeros((batch_size,), jnp.int32),
    )

  # --- Projections -----------------------------------------------------------

  def _queries(self, params: PyTree, x: Array) -> tuple[Array, Array]:
    """Returns the scaled `(q_nope, q_rope)` pair, both `[B, T, H, *]`.

    The softmax scale is folded into the query -- exactly once, and before
    absorption, which is linear and therefore carries it through.

    Args:
      params: This module's raw params.
      x: `[B, T, D]` residual-stream activations.
    """
    q = self.q_a_proj.apply(params['q_a_proj'], x)
    q = self.q_a_norm.apply(params['q_a_norm'], q)
    q = self.q_b_proj.apply(params['q_b_proj'], q)
    q = sharding_lib.with_sharding_constraint(
        jnp.asarray(q), self._activation_partition(4)
    )
    q = q * jnp.asarray(self.q_head_dim**-0.5, q.dtype)
    return q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]

  def _compressed_kv(self, params: PyTree, x: Array) -> Array:
    """Returns one cache row per token, `[B, T, kv_lora_rank + rope]`."""
    kv = self.kv_a_proj.apply(params['kv_a_proj'], x)
    latent = self.kv_a_norm.apply(
        params['kv_a_norm'], kv[..., : self.kv_lora_rank]
    )
    return jnp.concatenate([latent, kv[..., self.kv_lora_rank :]], axis=-1)

  def _kv_b_weight(self, params: PyTree) -> tuple[Array, Array]:
    """Returns `(W_UK, W_UV)`, the `[R, H, *]` halves of `kv_b_proj`."""
    w = common.convert_or_dequantize(
        params['kv_b_proj']['w'], dtype=self.activation_dtype
    )
    return w[..., : self.qk_nope_head_dim], w[..., self.qk_nope_head_dim :]

  # --- Attention cores -------------------------------------------------------

  def _decompressed_attention(
      self,
      params: PyTree,
      q_nope: Array,
      q_rope: Array,
      kv_rows: Array,
      mask: Array,
  ) -> Array:
    """Attention over decompressed per-head keys/values. `[B, T, H, Dv]`.

    Args:
      params: This module's raw params.
      q_nope: `[B, T, H, Dn]` scaled queries.
      q_rope: `[B, T, H, Dr]` scaled queries.
      kv_rows: `[B, S, R + Dr]` compressed rows attended over.
      mask: `[B, T, S]` bool; True is attendable.

    Returns:
      `[B, T, H, Dv]` per-head attention outputs.
    """
    latent, k_rope = (
        kv_rows[..., : self.kv_lora_rank],
        kv_rows[..., self.kv_lora_rank :],
    )
    k_nope_v = self.kv_b_proj.apply(params['kv_b_proj'], latent)
    k_nope = k_nope_v[..., : self.qk_nope_head_dim]
    v = k_nope_v[..., self.qk_nope_head_dim :]
    logits = jnp.einsum('bthd,bshd->bhts', q_nope, k_nope) + jnp.einsum(
        'bthd,bsd->bhts', q_rope, k_rope
    )
    probs = _masked_softmax(logits, mask, self.activation_dtype)
    return jnp.einsum('bhts,bshd->bthd', probs, v)

  def _absorbed_attention(
      self,
      params: PyTree,
      q_nope: Array,
      q_rope: Array,
      kv_rows: Array,
      mask: Array,
  ) -> Array:
    """Same value as `_decompressed_attention`, computed on the latent.

    `W_UK` folds into the query and `W_UV` into the context, so neither the
    `[B, S, H, Dn]` keys nor the `[B, S, H, Dv]` values are ever materialized:
    per step the cache is read once, at 576 units per token.

    Args:
      params: This module's raw params.
      q_nope: `[B, T, H, Dn]` scaled queries.
      q_rope: `[B, T, H, Dr]` scaled queries.
      kv_rows: `[B, S, R + Dr]` compressed rows attended over.
      mask: `[B, T, S]` bool; True is attendable.

    Returns:
      `[B, T, H, Dv]` per-head attention outputs.
    """
    w_uk, w_uv = self._kv_b_weight(params)
    # Absorbed MLA is plain MQA, so core `attn` fits in its decode shaping: one
    # KV head carrying `n_heads` query groups, the cache row itself as the key,
    # and the latent half alone as the value -- `attn` rebinds the contraction
    # index between its two einsums, so `Dv != Dqk` is legal.
    q = jnp.concatenate(
        [jnp.einsum('bthd,rhd->bthr', q_nope, w_uk), q_rope], axis=-1
    )
    context, _ = model_lib.attn(
        q[:, :, None],
        kv_rows[:, :, None],  # pyrefly: ignore[bad-argument-type]
        kv_rows[..., : self.kv_lora_rank][:, :, None],  # pyrefly: ignore[bad-argument-type]
        einops.rearrange(mask, 'b t s -> b 1 1 t s'),  # pyrefly: ignore[bad-argument-type]
        attn_soft_cap=0.0,  # K3 never caps logits; core defaults to 50.
        attn_mask_value=_MASK_VALUE,
        dtype=self.activation_dtype,
    )
    return jnp.einsum('bthr,rhd->bthd', context[:, :, 0], w_uv)

  # --- Forward ---------------------------------------------------------------

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array,
      segment_positions: Array,
      extra_inputs: PyTree | None = None,
      decode_state: KimiK3MLADecodeState | None = None,
  ) -> tuple[Array, PyTree]:
    """Runs the layer.

    Args:
      params: This module's param subtree.
      x: `[B, T, D]` normed block input.
      segment_ids: `[B, T]` int32; 0 marks padding, which is never cached.
      segment_positions: `[B, T]` int32 positions, used for the causal mask of
        the stateless path (the cached path masks on cache order instead).
      extra_inputs: Side channel; `update_kv_cache` (default True) gates the
        cache write, as in `model_lib.updated_decode_state`.
      decode_state: Compressed KV cache to attend over and extend. None runs the
        stateless prefill path.

    Returns:
      `([B, T, D] output, {'decode_state': state or None})`.
    """
    raw: PyTree = get_raw_arrays(params)
    extra_inputs = extra_inputs or {}
    valid = None if segment_ids is None else jnp.asarray(segment_ids) != 0

    q_nope, q_rope = self._queries(raw, x)
    kv_rows = self._compressed_kv(raw, x)

    if decode_state is None:
      mask = model_lib.create_mask(
          segment_positions=segment_positions,
          kv_segment_positions=segment_positions,
          segment_ids=segment_ids,
          kv_segment_ids=segment_ids,
      )
      context = self._decompressed_attention(raw, q_nope, q_rope, kv_rows, mask)
    else:
      decode_state, mask = _append_to_cache(
          decode_state,
          kv_rows,
          valid=valid,
          segment_positions=segment_positions,
          update_kv_cache=extra_inputs.get('update_kv_cache', True),
      )
      decode_state = dataclasses.replace(
          decode_state,
          kv_cache=sharding_lib.with_sharding_constraint(
              jnp.asarray(decode_state.kv_cache), self._cache_partition
          ),
      )
      core = (
          self._absorbed_attention
          if self.use_absorbed_decode
          else self._decompressed_attention
      )
      context = core(raw, q_nope, q_rope, decode_state.kv_cache, mask)

    context = sharding_lib.with_sharding_constraint(
        jnp.asarray(context), self._activation_partition(4)
    )
    if self.use_output_gate:
      gate = self.g_proj.apply(raw['g_proj'], x)
      context = context * jnp.asarray(
          jax.nn.sigmoid(jnp.asarray(gate, jnp.float32)), context.dtype
      )
    out = self.o_proj.apply(raw['o_proj'], context)
    if valid is not None:
      # Pad positions carry garbage into the residual stream otherwise; KDA
      # zeroes them too, so the whole stack agrees on what a pad row holds.
      out = jnp.where(valid[:, :, None], jnp.asarray(out), 0.0)
    out = sharding_lib.with_sharding_constraint(
        jnp.asarray(out), self._activation_partition(3)
    )
    return out, {'decode_state': decode_state}


def _masked_softmax(logits: Array, mask: Array, dtype: DTypeLike) -> Array:
  """Softmax in f32 over `[B, H, T, S]` logits under a `[B, T, S]` mask."""
  logits = jnp.asarray(logits, jnp.float32)
  logits = jnp.where(
      einops.rearrange(mask, 'b t s -> b 1 t s'), logits, _MASK_VALUE
  )
  return jnp.asarray(jax.nn.softmax(logits, axis=-1), dtype)


def _append_to_cache(
    state: KimiK3MLADecodeState,
    kv_rows: Array,
    *,
    valid: Array | None,
    segment_positions: Array | None,
    update_kv_cache: Array | bool = True,
) -> tuple[KimiK3MLADecodeState, Array]:
  """Appends this call's valid rows and returns `(state, [B, T, S] mask)`.

  Pad rows are dropped rather than written, so a right-padded chunk leaves the
  cache identical to an exact-length one and a sequence stays contiguous. The
  mask lets query `t` see the slot it was written to and everything before it
  back to its sequence's first slot -- causality in cache order (see
  `KimiK3MLADecodeState` for the shared invariants).

  A restart is honoured only when it falls on the call's FIRST valid token: two
  sequences inside one cached call would have to occupy the same slots. Pack
  segments on the stateless path instead, which masks them exactly.

  Args:
    state: Cache to extend.
    kv_rows: `[B, T, R + Dr]` compressed rows.
    valid: `[B, T]` bool; None means every row is valid.
    segment_positions: `[B, T]` int32; a 0 on the first valid row restarts that
      sequence. None disables restarts.
    update_kv_cache: False keeps the cache (and therefore the mask) as it was,
      matching `model_lib.updated_decode_state`: the current tokens are then
      invisible even to themselves.

  Returns:
    The extended state and the `[B, T, max_seq_len]` attendance mask.
  """
  batch, chunk_len, _ = kv_rows.shape
  max_seq_len = state.max_seq_len
  if chunk_len > max_seq_len:
    raise ValueError(f'Chunk of {chunk_len} exceeds the cache ({max_seq_len}).')
  if valid is None:
    valid = jnp.ones((batch, chunk_len), bool)
  valid_i32 = valid.astype(jnp.int32)
  ranks = jnp.cumsum(valid_i32, axis=1) - 1
  counts = ranks[:, -1] + 1

  base = state.lengths
  if segment_positions is not None:
    restarts = jnp.any(
        valid & (jnp.asarray(segment_positions) == 0) & (ranks == 0), axis=1
    )
    base = jnp.where(restarts, 0, base)
  # Out of range, so the scatter drops pad rows instead of writing them.
  rows = base[:, None] + jnp.where(valid, ranks, max_seq_len)

  # Batch as a vmapped (batching) scatter dim rather than an index column: the
  # partitioner then keeps each write on its own batch shard instead of
  # replicating the updates. Same bytes as `cache.at[batch_iota, rows].set`.
  scatter = jax.vmap(lambda buf, upd, r: buf.at[r].set(upd, mode='drop'))
  written = KimiK3MLADecodeState(
      kv_cache=scatter(
          state.kv_cache, jnp.asarray(kv_rows, state.kv_cache.dtype), rows
      ),
      lengths=jnp.minimum(base + counts, max_seq_len),
  )
  if not (isinstance(update_kv_cache, bool) and update_kv_cache):
    written = jax.lax.cond(update_kv_cache, lambda: written, lambda: state)

  # Pad rows keep the whole live prefix visible; their output is discarded.
  last_visible = base[:, None] + jnp.where(valid, ranks, max_seq_len - 1)
  slots = jnp.arange(max_seq_len, dtype=jnp.int32)
  mask = (slots[None, None] <= last_visible[:, :, None]) & (
      slots[None, None] < written.lengths[:, None, None]
  )
  return written, mask

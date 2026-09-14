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
"""Kimi Delta Attention (KDA): the K3 linear-attention token mixer.

Reference semantics are fla-core 0.5.2's `chunk_kda` / `fused_recurrent_kda`
with the flag set HF `modeling_kimi_linear.py` passes for Kimi K3
(`use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel,
safe_gate, lower_bound=-5.0`). Per head, with state `S in R^{K x V}`:

    g_t     = lower_bound * sigmoid(exp(a_log_h) * (z_t + dt_bias))  # per K
    qn_t    = l2norm(q_t) * K**-0.5 ;  kn_t = l2norm(k_t)
    beta_t  = sigmoid(b_proj(x)_t)
    S      <- diag(exp(g_t)) S                     # decay BEFORE the delta
    u_t     = beta_t * (v_t - S^T kn_t)            # error vs the DECAYED state
    S      <- S + kn_t u_t^T
    o_t     = S^T qn_t                             # read AFTER the update

`q, k, v` are each preceded by a depthwise causal short convolution (width 4,
no bias, SiLU after the conv), and the layer output is
`o_proj(rms_norm(o, o_norm) * sigmoid(g_proj(x)))` per head. The gate is the
K3 *sigmoid* form, NOT the softplus gate of Qwen3.5's gated delta net.

Two interchangeable cores, both fp32 inside:
  * `chunk_kda`: UT-transform chunked parallel form (fla `chunk_kda`), used
    for prefill. A Pallas kernel can replace it behind the same signature.
  * `recurrent_kda_step` / `recurrent_kda`: the token recurrence (fla
    `fused_recurrent_kda`), used for decode (and by tests as the ground
    truth for the chunked form).

Packing and padding are load-bearing: a recurrence cannot be masked after the
fact. Padded positions (`segment_ids == 0`) get `g = 0, beta = 0`, i.e. an
exact identity update, and a new segment (`segment_positions == 0`) resets the
state, the short-conv window and the intra-chunk interactions.
"""

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.scipy import linalg as jax_linalg
import jax.typing
from simply import model_lib
from simply.utils import common
from simply.utils import initializer
from simply.utils import module
from simply.utils import sharding as sharding_lib

Array = common.Array
PyTree = common.PyTree
PRNGKey = jax.typing.ArrayLike
DTypeLike = jax.typing.DTypeLike
PartitionAnnotation = common.PartitionAnnotation
SimplyConfig = Any

# fla/modules/l2norm.py: rstd = 1 / sqrt(sum(x * x) + eps), eps INSIDE the
# sqrt and on the sum of squares (not F.normalize's eps on the norm).
L2NORM_EPS = 1e-6

# q/k/v short-conv branches, in the order they are stacked in `conv_state`.
_CONV_BRANCHES = ('q', 'k', 'v')


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KDADecodeState:
  """Constant-size decode state of one KDA layer (the KV-cache analogue).

  `apply` returns it under `extra_output['decode_state']`.

  Attributes:
    conv_state: `[B, 3, H, D, W]` float32 pre-conv inputs of the q/k/v branches
      (axis 1 indexes `_CONV_BRANCHES`), newest at slot `W-1`. The W-slot layout
      is fla's `ShortConvolution` cache layout, so it can be compared with an HF
      `KimiDynamicCache` directly; slot 0 is never read by a width-W kernel, it
      only exists so a decode step is a shift-and-append.
    recurrent_state: `[B, H, K, V]` float32 delta-rule state, K-major (HF stores
      the transpose, `transpose_state_layout=True`).
    lengths: `[B]` int32 count of non-padding tokens absorbed since the start of
      the sequence currently in the slot (a new segment restarts it).
  """

  conv_state: jax.Array
  recurrent_state: jax.Array
  lengths: jax.Array


def l2norm(x: Array, eps: float = L2NORM_EPS) -> jax.Array:
  """fla's l2 normalization over the last axis, in f32."""
  x32 = jnp.asarray(x, jnp.float32)
  return x32 * jax.lax.rsqrt(
      jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + eps
  )


def kda_log_decay(
    z: Array,
    a_log: Array,
    dt_bias: Array,
    lower_bound: float | None = -5.0,
) -> jax.Array:
  """Per-(head, channel) log decay `g <= 0`, in f32.

  With a lower bound (K3's safe gate, -5.0):
  `lower_bound * sigmoid(exp(a_log_h) * (z + dt_bias))`, bounded in
  (lower_bound, 0). Without one (fla's fallback, used by K3 variants that ship
  `gate_lower_bound=None`): `-exp(a_log_h) * softplus(z + dt_bias)`.
  `dt_bias` is added BEFORE the multiplication by `exp(a_log)`.

  Args:
    z: `[..., H, K]` low-rank gate projection `f_b_proj(f_a_proj(x))`.
    a_log: `[H]` per-head log decay rate.
    dt_bias: `[H, K]` per-channel bias.
    lower_bound: negative decay bound, or None for the softplus form.

  Returns:
    `[..., H, K]` float32 log decay.
  """
  z32 = jnp.asarray(z, jnp.float32) + jnp.asarray(dt_bias, jnp.float32)
  a = jnp.exp(jnp.asarray(a_log, jnp.float32))[:, None]
  if lower_bound is None:
    # fla's thresholded softplus (fla/ops/utils/softplus.py).
    softplus = jnp.where(
        z32 > 20.0, z32, jnp.log1p(jnp.exp(jnp.minimum(z32, 20.0)))
    )
    return -a * softplus
  return lower_bound * jax.nn.sigmoid(a * z32)


def _segment_index(segment_start: jax.Array) -> jax.Array:
  """Per-token index of the segment it belongs to, 0 for the carried one."""
  return jnp.cumsum(segment_start.astype(jnp.int32), axis=1)


def short_conv(
    x: Array,
    w: Array,
    state: Array | None,
    *,
    valid: jax.Array | None = None,
    seg_index: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Causal depthwise conv (width W, no bias) + SiLU, fla semantics.

  `y[t] = silu(sum_i w[..., i] * x[t - (W - 1) + i])`, i.e. `w[..., W-1]` taps
  the current token and missing history comes from `state` (zeros for a fresh
  sequence). Taps that reach into a different segment are dropped, and the
  returned state holds the last W inputs of the row's LAST segment among its
  valid positions, so neither padding nor a packed neighbour leaks into the
  next call.

  Args:
    x: `[B, T, H, D]` pre-conv projections.
    w: `[H, D, W]` depthwise filters.
    state: `[B, H, D, W]` previous window, or None for a fresh sequence.
    valid: `[B, T]` bool; False positions are padding.
    seg_index: `[B, T]` int32 segment counter (`_segment_index`); the carried
      segment is 0, so a row with no new segment reads all of `state`.

  Returns:
    `(y, new_state)`: `[B, T, H, D]` float32 activations (zero at padding) and
    the `[B, H, D, W]` float32 updated window.

  Precondition: the valid positions of a segment are contiguous (padding sits
  before or after a segment, never inside one). A hole inside a segment would
  shift the conv taps by one slot relative to the recurrence, which skips
  padding entirely.
  """
  b, t, h, d = x.shape
  kw = w.shape[-1]
  xf = jnp.asarray(x, jnp.float32)
  wf = jnp.asarray(w, jnp.float32)
  if state is None:
    prev = jnp.zeros((b, kw, h, d), jnp.float32)
  else:
    prev = jnp.moveaxis(jnp.asarray(state, jnp.float32), -1, 1)
  ext = jnp.concatenate([prev, xf], axis=1)  # [B, W + T, H, D]

  if valid is None:
    valid = jnp.ones((b, t), bool)
  if seg_index is None:
    seg_index = jnp.zeros((b, t), jnp.int32)
  # The W history slots belong to the carried segment (index 0), so a row that
  # opens a new segment at t=0 masks all of them out.
  ext_seg = jnp.concatenate([jnp.zeros((b, kw), jnp.int32), seg_index], axis=1)

  y = jnp.zeros((b, t, h, d), jnp.float32)
  for i in range(kw):
    # Tap i reads ext[1 + i + t]; slot 0 of the cache is never read.
    src = jax.lax.dynamic_slice_in_dim(ext, 1 + i, t, axis=1)
    same = jax.lax.dynamic_slice_in_dim(ext_seg, 1 + i, t, axis=1) == seg_index
    y = y + wf[:, :, i] * src * same[:, :, None, None]
  y = jax.nn.silu(y) * valid[:, :, None, None]

  # New window = the last W valid positions of the extended timeline. Ranking
  # by the running count of valid tokens (the cache slots count as valid) is
  # exact for arbitrary pad placement, not just right padding.
  ext_valid = jnp.concatenate(
      [jnp.ones((b, kw), jnp.int32), valid.astype(jnp.int32)], axis=1
  )
  rank = jnp.cumsum(ext_valid, axis=1)  # [B, W + T], 1-based
  lengths = rank[:, -1] - kw  # valid tokens in x
  target = lengths[:, None] + 1 + jnp.arange(kw, dtype=jnp.int32)[None, :]
  # rank is non-decreasing and increments exactly at valid positions, so the
  # FIRST hit is the valid token itself.
  idx = jnp.argmax(rank[:, :, None] == target[:, None, :], axis=1)  # [B, W]
  window = jnp.take_along_axis(ext, idx[:, :, None, None], axis=1)
  window_seg = jnp.take_along_axis(ext_seg, idx, axis=1)
  keep = window_seg == window_seg[:, -1:]
  window = window * keep[:, :, None, None]
  return y, jnp.moveaxis(window, 1, -1)  # [B, H, D, W]


def recurrent_kda_step(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
) -> tuple[jax.Array, jax.Array]:
  """One KDA token step (fla `fused_recurrent_kda`), all f32.

  Args:
    q: `[B, H, K]` l2-normalized and `K**-0.5`-scaled query.
    k: `[B, H, K]` l2-normalized key.
    v: `[B, H, V]` value.
    g: `[B, H, K]` log decay (<= 0); 0 makes the step an identity update.
    beta: `[B, H]` delta-rule step size in (0, 1); 0 makes the step an identity
      update.
    state: `[B, H, K, V]` recurrent state.

  Returns:
    `(o, new_state)`: `[B, H, V]` output read from the POST-update state, and
    the updated `[B, H, K, V]` state.
  """
  s = (
      jnp.asarray(state, jnp.float32)
      * jnp.exp(jnp.asarray(g, jnp.float32))[..., None]
  )
  u = jnp.asarray(beta, jnp.float32)[..., None] * (
      jnp.asarray(v, jnp.float32) - jnp.einsum('bhkv,bhk->bhv', s, k)
  )
  s = s + jnp.asarray(k, jnp.float32)[..., None] * u[..., None, :]
  return jnp.einsum('bhkv,bhk->bhv', s, q), s


def recurrent_kda(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    initial_state: Array | None = None,
    *,
    segment_start: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Sequential KDA over `[B, T, H, *]` inputs; the chunked form's reference.

  One `recurrent_kda_step` per token, scanned over time. Kept because it is the
  definition of the op: the prefill path uses `chunk_kda`, decode uses
  `recurrent_kda_step`.

  Args:
    q: `[B, T, H, K]` l2-normalized and `K**-0.5`-scaled queries.
    k: `[B, T, H, K]` l2-normalized keys.
    v: `[B, T, H, V]` values.
    g: `[B, T, H, K]` log decay (<= 0); 0 at padding.
    beta: `[B, T, H]` step sizes in (0, 1); 0 at padding.
    initial_state: `[B, H, K, V]` carried state, or None for zeros.
    segment_start: `[B, T]` bool, True at the first token of a new segment,
      which resets the state before that token is consumed.

  Returns:
    `(o, final_state)`: `[B, T, H, V]` float32 outputs and the `[B, H, K, V]`
    float32 state after the last token, as `chunk_kda` returns them.
  """
  b, t, h, kdim = q.shape
  vdim = v.shape[-1]
  s0 = (
      jnp.zeros((b, h, kdim, vdim), jnp.float32)
      if initial_state is None
      else jnp.asarray(initial_state, jnp.float32)
  )
  reset = jnp.zeros((b, t), bool) if segment_start is None else segment_start

  def step(
      s: jax.Array, xs: tuple[jax.Array, ...]
  ) -> tuple[jax.Array, jax.Array]:
    q_t, k_t, v_t, g_t, beta_t, reset_t = xs
    s = jnp.where(reset_t[:, None, None, None], 0.0, s)
    o_t, s = recurrent_kda_step(q_t, k_t, v_t, g_t, beta_t, s)
    return s, o_t

  with jax.default_matmul_precision('float32'):
    xs = jax.tree.map(
        lambda a: jnp.swapaxes(jnp.asarray(a), 0, 1),
        (q, k, v, g, beta, reset),
    )
    s_final, o = jax.lax.scan(step, s0, xs)
  return jnp.swapaxes(o, 0, 1), s_final


def chunk_kda(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    initial_state: Array | None = None,
    chunk_size: int = 64,
    *,
    segment_start: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Chunkwise-parallel KDA (fla `chunk_kda`), f32; equals `recurrent_kda`.

  Per chunk of C tokens, with `gc` the inclusive within-chunk cumulative log
  decay (per channel) and all pairs restricted to one segment:

    Aqk[i,j] = <q_i e^{gc_i - gc_j}, k_j>                     j <= i
    Akk[i,j] = beta_i <k_i e^{gc_i - gc_j}, k_j>              j <  i
    Tm       = (I + Akk)^{-1} diag(beta)
    w        = Tm (k e^{gc}) ;  u = Tm v
    v_new    = u - w S ;  o = Aqk v_new + (q e^{gc}) S
    S       <- diag(e^{gc_last}) S + (k e^{gc_last - gc})^T v_new

  Every exponent argument is <= 0 (`g <= 0` makes `gc` non-increasing) because
  the pairwise differences are materialized instead of being factored into
  `e^{gc_i} * e^{-gc_j}` matmul operands, whose second factor is unbounded.
  The price is the `[B, H, C, C, K]` `decay` tensor -- XLA does materialize it
  (~200 MB per scan step at K3's H=96, C=64, K=128, unsharded), and its two
  contractions run on the vector unit rather than the MXU. That is the reason
  this function is meant to be replaced, and it also bounds the cost linearly
  in `chunk_size`. The replacement (fla's `chunk_intra`, Pallas or pure JAX)
  splits the chunk into BC=16 sub-blocks: for a pair of DIFFERENT sub-blocks,
  `e^{gc_i - gc_j} = e^{gc_i - gc_n} * e^{gc_n - gc_j}` with `n` the last index
  of j's sub-block keeps BOTH factors in (0, 1] and turns the Gram matrices
  into matmuls; only the diagonal sub-blocks then need explicit differences
  (fla factors those too, around a mid-sub-block reference, which is safe only
  under `gate_lower_bound = -5`).

  Args:
    q: `[B, T, H, K]` l2-normalized and `K**-0.5`-scaled queries.
    k: `[B, T, H, K]` l2-normalized keys.
    v: `[B, T, H, V]` values.
    g: `[B, T, H, K]` log decay (<= 0); 0 at padding.
    beta: `[B, T, H]` step sizes in (0, 1); 0 at padding.
    initial_state: `[B, H, K, V]` carried state, or None for zeros.
    chunk_size: tokens per chunk; T is padded up to a multiple of it with
      identity updates.
    segment_start: `[B, T]` bool, True at the first token of a new segment,
      which resets the state and cuts intra-chunk interactions.

  Returns:
    `(o, final_state)`: `[B, T, H, V]` float32 outputs and the `[B, H, K, V]`
    float32 state after the last token.
  """
  b, t, h, kdim = q.shape
  vdim = v.shape[-1]
  c = chunk_size
  s0 = (
      jnp.zeros((b, h, kdim, vdim), jnp.float32)
      if initial_state is None
      else jnp.asarray(initial_state, jnp.float32)
  )
  reset = jnp.zeros((b, t), bool) if segment_start is None else segment_start

  pad = (-t) % c
  if pad:
    pad_time = lambda a: jnp.pad(  # pylint: disable=g-long-lambda
        jnp.asarray(a), ((0, 0), (0, pad)) + ((0, 0),) * (jnp.ndim(a) - 2)
    )
    q, k, v, g, beta, reset = (pad_time(a) for a in (q, k, v, g, beta, reset))
  nc = (t + pad) // c

  def to_chunks(a: Array) -> jax.Array:
    # [B, T, ...] -> [NC, B, H, C, ...]; the chunk axis leads for the scan.
    a = jnp.reshape(jnp.asarray(a), (b, nc, c) + jnp.shape(a)[2:])
    if a.ndim == 5:  # q/k/v/g: [B, NC, C, H, D]
      return jnp.transpose(a, (1, 0, 3, 2, 4))
    return jnp.transpose(a, (1, 0, 3, 2))  # beta: [B, NC, C, H]

  qc, kc, vc, gc_in = (
      to_chunks(jnp.asarray(a, jnp.float32)) for a in (q, k, v, g)
  )
  bc = to_chunks(jnp.asarray(beta, jnp.float32))
  # Segment counter, restarted at every chunk: 0 = the segment carried in.
  seg = jnp.cumsum(
      jnp.transpose(jnp.reshape(reset, (b, nc, c)), (1, 0, 2)).astype(
          jnp.int32
      ),
      axis=-1,
  )  # [NC, B, C]

  tri_causal = jnp.tril(jnp.ones((c, c), jnp.float32))
  tri_strict = jnp.tril(jnp.ones((c, c), jnp.float32), k=-1)
  eye = jnp.eye(c, dtype=jnp.float32)

  def chunk_step(
      s: jax.Array, xs: tuple[jax.Array, ...]
  ) -> tuple[jax.Array, jax.Array]:
    q_i, k_i, v_i, g_i, beta_i, seg_i = xs
    same = (seg_i[:, :, None] == seg_i[:, None, :])[:, None]  # [B,1,C,C]
    carry = (seg_i == 0).astype(jnp.float32)[:, None, :, None]  # [B,1,C,1]
    gc = jnp.cumsum(g_i, axis=2)  # [B, H, C, K]
    decay = jnp.exp(
        jnp.minimum(gc[:, :, :, None, :] - gc[:, :, None, :, :], 0.0)
    )
    aqk = (
        jnp.einsum('bhtk,bhjk,bhtjk->bhtj', q_i, k_i, decay) * tri_causal * same
    )
    akk = (
        jnp.einsum('bhtk,bhjk,bhtjk->bhtj', k_i, k_i, decay)
        * beta_i[..., None]
        * tri_strict
        * same
    )
    # (I + Akk)^{-1}, block diagonal per segment since Akk is masked to
    # same-segment pairs. `unit_diagonal` supplies the I.
    t_inv = jax_linalg.solve_triangular(
        akk, jnp.broadcast_to(eye, akk.shape), lower=True, unit_diagonal=True
    )
    # Rows whose segment started inside this chunk see no carried state, and
    # for the rest the within-chunk cumsum IS the within-segment cumsum.
    w = carry * jnp.einsum(
        'bhtj,bhjk->bhtk', t_inv, beta_i[..., None] * k_i * jnp.exp(gc)
    )
    u = jnp.einsum('bhtj,bhjv->bhtv', t_inv, beta_i[..., None] * v_i)
    v_new = u - jnp.einsum('bhtk,bhkv->bhtv', w, s)
    o_i = jnp.einsum('bhtj,bhjv->bhtv', aqk, v_new) + carry * jnp.einsum(
        'bhtk,bhkv->bhtv', q_i * jnp.exp(gc), s
    )
    g_last = gc[:, :, -1:, :]  # [B, H, 1, K]
    last_same = (seg_i == seg_i[:, -1:]).astype(jnp.float32)  # [B, C]
    k_rest = (
        k_i
        * jnp.exp(jnp.minimum(g_last - gc, 0.0))
        * last_same[:, None, :, None]
    )
    s = s * (carry[:, :, -1] * jnp.exp(g_last[:, :, 0]))[
        ..., None
    ] + jnp.einsum('bhtk,bhtv->bhkv', k_rest, v_new)
    return s, o_i

  with jax.default_matmul_precision('float32'):
    s_final, o_chunks = jax.lax.scan(
        chunk_step, s0, (qc, kc, vc, gc_in, bc, seg)
    )
  # [NC, B, H, C, V] -> [B, T, H, V]
  o = jnp.transpose(o_chunks, (1, 0, 3, 2, 4)).reshape(b, nc * c, h, vdim)
  return o[:, :t], s_final


@module.ModuleRegistry.register
@dataclasses.dataclass
class KimiK3DeltaAttention(module.SimplyModule):
  """Kimi Delta Attention, the linear-attention token mixer of Kimi K3.

  Parameters (K = V = `head_dim`, W = `conv_kernel_dim`, R = `gate_lora_rank`):

    q_proj/w [D,H,K]  k_proj/w [D,H,K]  v_proj/w [D,H,V]
    q_conv/w [H,K,W]  k_conv/w [H,K,W]  v_conv/w [H,V,W]     (f32, depthwise)
    f_a_proj/w [D,R]  f_b_proj/w [R,H,K]  dt_bias [H,K]  a_log [H]   (f32)
    b_proj/w [D,H]    g_proj/w [D,H,V]    o_norm/scale [V]  (f32)
    o_proj/w [H,V,D]

  Attributes:
    model_dim: residual stream width D.
    num_heads: H; K3 uses 96 (no GVA: value heads == key heads).
    head_dim: K = V; 128 for K3.
    conv_kernel_dim: W of the causal depthwise short convolution.
    gate_lora_rank: R of the decay-gate bottleneck `f_b_proj(f_a_proj(x))`.
    gate_lower_bound: `g_min` of the safe gate, or None for fla's softplus gate.
    chunk_size: chunk length of the prefill core.
    rms_norm_epsilon: epsilon of the gated output RMSNorm.
    activation_dtype: dtype of the projections; the recurrence stays f32.
    weight_dtype: dtype of the projection weights.
    sharding_config: `config_lib.ShardingConfig` or None for no annotations.
    weight_init: initializer of the projection and conv weights.
    qkv_partition: weight annotation of q/k/v/g_proj `[D,H,*]`; defaults to
      `sharding_config.attn_qkv_partition`. Every `*_partition` field below
      behaves the same way: None means "derive from `sharding_config`", and
      `sharding_config=None` means "annotate nothing".
    o_partition: weight annotation of `o_proj [H,V,D]`.
    conv_partition: weight annotation of the conv filters `[H,D,W]`.
    gate_a_partition: weight annotation of `f_a_proj [D,R]`.
    gate_b_partition: weight annotation of `f_b_proj [R,H,K]`.
    beta_partition: weight annotation of `b_proj [D,H]`.
    state_partition: annotation of the recurrent state `[B,H,K,V]`.
    conv_state_partition: annotation of the conv state `[B,3,H,D,W]`.
  """

  model_dim: int
  num_heads: int
  head_dim: int = 128
  conv_kernel_dim: int = 4
  gate_lora_rank: int = 128
  gate_lower_bound: float | None = -5.0
  chunk_size: int = 64
  rms_norm_epsilon: float = 1e-5
  activation_dtype: DTypeLike = 'bfloat16'
  weight_dtype: DTypeLike = 'float32'
  sharding_config: SimplyConfig | None = None
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  qkv_partition: PartitionAnnotation = None
  o_partition: PartitionAnnotation = None
  conv_partition: PartitionAnnotation = None
  gate_a_partition: PartitionAnnotation = None
  gate_b_partition: PartitionAnnotation = None
  beta_partition: PartitionAnnotation = None
  state_partition: PartitionAnnotation = None
  conv_state_partition: PartitionAnnotation = None

  @property
  def inner_dim(self) -> int:
    return self.num_heads * self.head_dim

  def setup(self) -> None:
    lb = self.gate_lower_bound
    # fla refuses anything else (chunk.py's safe-gate guard): a positive bound
    # would make g > 0 and the chunked form's exponents overflow.
    if lb is not None and not -5.0 <= lb < 0.0:
      raise ValueError(f'gate_lower_bound must be in [-5, 0) or None, got {lb}')
    self._resolve_partitions()
    proj = lambda shape, partition, out_partition: module.EinsumLinear(  # pylint: disable=g-long-lambda
        eqn='ihd,...i->...hd',
        weight_shape=shape,
        bias_term='',
        activation_dtype=self.activation_dtype,
        weight_dtype=self.weight_dtype,
        weight_partition=partition,
        output_partition=out_partition,
        weight_init=self.weight_init,
    )
    head_shape = [self.model_dim, self.num_heads, self.head_dim]
    self.q_proj = proj(head_shape, self.qkv_partition, self._attn_activation)
    self.k_proj = proj(head_shape, self.qkv_partition, self._attn_activation)
    self.v_proj = proj(head_shape, self.qkv_partition, self._attn_activation)
    self.g_proj = proj(head_shape, self.qkv_partition, self._attn_activation)
    self.f_a_proj = module.EinsumLinear(
        eqn='io,...i->...o',
        weight_shape=[self.model_dim, self.gate_lora_rank],
        bias_term='',
        activation_dtype=self.activation_dtype,
        weight_dtype=self.weight_dtype,
        weight_partition=self.gate_a_partition,
        output_partition=sharding_lib.NOT_ANNOTATED,
        weight_init=self.weight_init,
    )
    self.f_b_proj = proj(
        [self.gate_lora_rank, self.num_heads, self.head_dim],
        self.gate_b_partition,
        sharding_lib.NOT_ANNOTATED,
    )
    self.b_proj = module.EinsumLinear(
        eqn='io,...i->...o',
        weight_shape=[self.model_dim, self.num_heads],
        bias_term='',
        activation_dtype=self.activation_dtype,
        weight_dtype=self.weight_dtype,
        weight_partition=self.beta_partition,
        output_partition=sharding_lib.NOT_ANNOTATED,
        weight_init=self.weight_init,
    )
    # K3 has two RMS-norm flavours and `LayerNorm` spells both, told apart by
    # `activation_dtype`: HF `KimiRMSNorm` casts before applying the gain (see
    # `moe.py`'s `latent_norm`), while fla's `FusedRMSNormGated` -- this one --
    # stays in f32 throughout, so the gated output only rounds at `o_proj`.
    self.o_norm = model_lib.LayerNorm(
        dim=self.head_dim,
        use_bias=False,
        use_scale=True,
        scale_plus_one=False,
        weight_dtype='float32',
        activation_dtype='float32',
        # `LayerNorm.init` annotates unconditionally (`EinsumLinear.init` does
        # not), so the gain needs the sentinel to stay unannotated.
        scale_partition=sharding_lib.NOT_ANNOTATED,
        epsilon=self.rms_norm_epsilon,
    )
    self.o_proj = module.EinsumLinear(
        eqn='hdo,...hd->...o',
        weight_shape=[self.num_heads, self.head_dim, self.model_dim],
        bias_term='',
        activation_dtype=self.activation_dtype,
        weight_dtype=self.weight_dtype,
        weight_partition=self.o_partition,
        output_partition=self._activation,
        weight_init=self.weight_init,
    )

  def _resolve_partitions(self) -> None:
    """Fills the unset `*_partition` fields from `sharding_config`."""
    self._activation = sharding_lib.NOT_ANNOTATED
    self._attn_activation = sharding_lib.NOT_ANNOTATED
    if self.sharding_config is None:
      # The sentinel, not None: `with_sharding_constraint` passes
      # `NOT_ANNOTATED` through, while None constrains to fully replicated.
      if self.conv_partition is None:
        self.conv_partition = sharding_lib.NOT_ANNOTATED
      if self.state_partition is None:
        self.state_partition = sharding_lib.NOT_ANNOTATED
      if self.conv_state_partition is None:
        self.conv_state_partition = sharding_lib.NOT_ANNOTATED
      return
    self._activation = self.sharding_config.activation_partition
    self._attn_activation = self.sharding_config.attn_activation_partition
    # The recurrent state is 6.3 MB per sequence per layer, so it follows the
    # activations' batch axis instead of being replicated over it.
    batch = self._activation[0] if self._activation else None
    if self.state_partition is None:
      self.state_partition = (batch, 'model', None, None)
    if self.conv_state_partition is None:
      self.conv_state_partition = (batch, None, 'model', None, None)
    qkv = self.sharding_config.attn_qkv_partition  # (data, model, None)
    # Under an expert-parallel sharding `ffn0_partition` is the rank-3
    # expert-stack annotation; the dense projections here take its last two
    # entries, as `moe.py` does for its own dense projections.
    ffn0 = tuple(self.sharding_config.ffn0_partition)[-2:]
    if self.qkv_partition is None:
      self.qkv_partition = qkv
    if self.o_partition is None:
      # o_proj is the transpose of the usual [D, H, K] output projection.
      self.o_partition = (qkv[1], qkv[2], qkv[0])
    if self.conv_partition is None:
      self.conv_partition = ('model', None, None)
    if self.gate_a_partition is None:
      self.gate_a_partition = (ffn0[0], None)
    if self.gate_b_partition is None:
      self.gate_b_partition = (None, 'model', None)
    if self.beta_partition is None:
      self.beta_partition = ffn0

  def init(self, prng_key: PRNGKey) -> PyTree:
    keys = jax.random.split(prng_key, 11)
    conv_shape = (self.num_heads, self.head_dim, self.conv_kernel_dim)
    conv = lambda key: {  # pylint: disable=g-long-lambda
        'w': common.AnnotatedArray.create(
            sharding_lib.with_sharding_constraint(
                self.weight_init(
                    key,
                    shape=conv_shape,
                    dtype='float32',
                    dim_annotation='h..',
                ),
                self.conv_partition,
            ),
            dim_annotation='h..',
        )
    }
    return {
        'q_proj': self.q_proj.init(keys[0]),
        'k_proj': self.k_proj.init(keys[1]),
        'v_proj': self.v_proj.init(keys[2]),
        'q_conv': conv(keys[3]),
        'k_conv': conv(keys[4]),
        'v_conv': conv(keys[5]),
        'f_a_proj': self.f_a_proj.init(keys[6]),
        'f_b_proj': self.f_b_proj.init(keys[7]),
        # HF initializes dt_bias from the Mamba-style inverse softplus schedule
        # and a_log to zeros; both are overwritten by the checkpoint.
        'dt_bias': common.AnnotatedArray.create(
            jnp.zeros((self.num_heads, self.head_dim), jnp.float32),
            dim_annotation='h.',
        ),
        'a_log': common.AnnotatedArray.create(
            jnp.zeros((self.num_heads,), jnp.float32), dim_annotation='h'
        ),
        'b_proj': self.b_proj.init(keys[8]),
        'g_proj': self.g_proj.init(keys[9]),
        'o_norm': self.o_norm.init(),
        'o_proj': self.o_proj.init(keys[10]),
    }

  def init_decode_state(
      self, batch_size: int, max_seq_len: int
  ) -> KDADecodeState:
    """Zeroed `KDADecodeState`; KDA's state is constant in the sequence length.

    Args:
      batch_size: number of sequences.
      max_seq_len: unused; the state does not grow with the sequence.

    Returns:
      A zero `KDADecodeState`.
    """
    del max_seq_len
    conv_state = jnp.zeros(
        (
            batch_size,
            len(_CONV_BRANCHES),
            self.num_heads,
            self.head_dim,
            self.conv_kernel_dim,
        ),
        jnp.float32,
    )
    recurrent_state = jnp.zeros(
        (batch_size, self.num_heads, self.head_dim, self.head_dim), jnp.float32
    )
    return KDADecodeState(
        conv_state=sharding_lib.with_sharding_constraint(
            conv_state, self.conv_state_partition
        ),
        recurrent_state=sharding_lib.with_sharding_constraint(
            recurrent_state, self.state_partition
        ),
        lengths=jnp.zeros((batch_size,), jnp.int32),
    )

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array | None = None,
      segment_positions: Array | None = None,
      extra_inputs: PyTree | None = None,
      inputs_mask: Array | None = None,
      decode_state: KDADecodeState | None = None,
  ) -> tuple[jax.Array, dict[str, KDADecodeState | None]]:
    """Runs one KDA layer.

    Args:
      params: this module's parameter subtree.
      x: `[B, T, D]` normed block input.
      segment_ids: `[B, T]` int32; 0 marks padding, which must leave the
        recurrent state untouched.
      segment_positions: `[B, T]` int32 position within the segment; a 0 at a
        non-padding token starts a new segment and resets the state.
      extra_inputs: unused side channel.
      inputs_mask: `[B, T]` bool override of `segment_ids != 0`.
      decode_state: carried `KDADecodeState`, or None for a stateless prefill.

    Returns:
      `(out, extra_output)` with `out` of shape `[B, T, D]` and
      `extra_output['decode_state']` the updated state (None iff
      `decode_state` was None).
    """
    del extra_inputs
    # `cast`: the leaves are AnnotatedArray / quantized dicts until
    # `convert_or_dequantize`, and PyTree is too wide to index.
    raw = cast(dict[str, Any], common.get_raw_arrays(params))
    b, t, _ = x.shape
    h, dim = self.num_heads, self.head_dim

    if inputs_mask is None:
      valid = (
          jnp.ones((b, t), bool)
          if segment_ids is None
          else jnp.asarray(segment_ids) != 0
      )
    else:
      valid = jnp.asarray(inputs_mask) != 0
    if segment_positions is None:
      segment_start = jnp.zeros((b, t), bool)
    else:
      segment_start = valid & (jnp.asarray(segment_positions) == 0)
    seg_index = _segment_index(segment_start)

    x = jnp.where(valid[:, :, None], jnp.asarray(x), 0.0)
    conv_state = None
    state = None
    if decode_state is not None:
      conv_state = sharding_lib.with_sharding_constraint(
          decode_state.conv_state, self.conv_state_partition
      )
      state = sharding_lib.with_sharding_constraint(
          decode_state.recurrent_state, self.state_partition
      )

    conv_out = {}
    new_conv = []
    for i, branch in enumerate(_CONV_BRANCHES):
      proj = getattr(self, f'{branch}_proj')
      pre = proj.apply(raw[f'{branch}_proj'], x)
      w = common.convert_or_dequantize(
          raw[f'{branch}_conv']['w'], dtype=jnp.float32
      )
      post, new_state = short_conv(
          pre,
          w,
          None if conv_state is None else conv_state[:, i],
          valid=valid,
          seg_index=seg_index,
      )
      conv_out[branch] = post
      new_conv.append(new_state)
    q, k, v = conv_out['q'], conv_out['k'], conv_out['v']
    new_conv_state = jnp.stack(new_conv, axis=1)

    z = self.f_b_proj.apply(
        raw['f_b_proj'], self.f_a_proj.apply(raw['f_a_proj'], x)
    )
    a_log = common.convert_or_dequantize(raw['a_log'], dtype=jnp.float32)
    dt_bias = common.convert_or_dequantize(raw['dt_bias'], dtype=jnp.float32)
    g = kda_log_decay(z, a_log, dt_bias, self.gate_lower_bound)
    beta = jax.nn.sigmoid(
        jnp.asarray(self.b_proj.apply(raw['b_proj'], x), jnp.float32)
    )
    # Padding must be an exact identity update of the recurrence.
    g = jnp.where(valid[:, :, None, None], g, 0.0)
    beta = jnp.where(valid[:, :, None], beta, 0.0)

    qn = l2norm(q) * jnp.float32(dim**-0.5)
    kn = l2norm(k)
    vf = jnp.asarray(v, jnp.float32)
    if state is None:
      state = jnp.zeros((b, h, dim, dim), jnp.float32)
    if t == 1:  # Decode (or a one-token prefill): no chunking to do.
      state = jnp.where(segment_start[:, 0, None, None, None], 0.0, state)
      o, new_recurrent_state = recurrent_kda_step(
          qn[:, 0], kn[:, 0], vf[:, 0], g[:, 0], beta[:, 0], state
      )
      o = o[:, None]
    else:
      o, new_recurrent_state = chunk_kda(
          qn,
          kn,
          vf,
          g,
          beta,
          state,
          self.chunk_size,
          segment_start=segment_start,
      )

    gate = self.g_proj.apply(raw['g_proj'], x)
    # fla FusedRMSNormGated(activation='sigmoid'): per head, everything in f32,
    # the gain applied before the (unnormalized) gate.
    o = self.o_norm.apply(raw['o_norm'], o) * jax.nn.sigmoid(
        jnp.asarray(gate, jnp.float32)
    )
    out = self.o_proj.apply(
        raw['o_proj'], jnp.asarray(o, self.activation_dtype)
    )
    out = jnp.where(valid[:, :, None], out, 0.0)
    absorbed = jnp.sum(
        (valid & (seg_index == seg_index[:, -1:])).astype(jnp.int32), axis=1
    )

    new_decode_state = None
    if decode_state is not None:
      new_decode_state = KDADecodeState(
          conv_state=sharding_lib.with_sharding_constraint(
              new_conv_state, self.conv_state_partition
          ),
          recurrent_state=sharding_lib.with_sharding_constraint(
              new_recurrent_state, self.state_partition
          ),
          # Tokens of the row's LAST segment only, so a restart resets the
          # count (matching the MLA cache's `lengths`).
          lengths=jnp.where(
              seg_index[:, -1] == 0, decode_state.lengths + absorbed, absorbed
          ),
      )
    return out, {'decode_state': new_decode_state}

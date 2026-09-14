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
"""GatedDeltaNet: the linear-attention token mixer of Qwen3.8.

Equivalent to `Qwen3_5GatedDeltaNet` of HuggingFace transformers v5
`models/qwen3_5/modeling_qwen3_5.py` -- `apply_mask_to_padding_states`,
`causal_conv1d_fn` / `causal_conv1d_update`, `torch_chunk_gated_delta_rule`,
`torch_recurrent_gated_delta_rule` and `Qwen3_5RMSNormGated` -- run with
`use_qk_l2norm_in_kernel=True`, the only setting Qwen3.8 ships.

Per value head, with state `S in R^{K x V}` (K = `key_head_dim`,
V = `value_head_dim`) and `x` the normed block input:

    qkv_t     = silu(conv(x W_qkv)_t)              # depthwise, width W, causal
    q_t, k_t  = l2norm(q_t) K^{-1/2}, l2norm(k_t)
    beta_t    = sigmoid(x_t W_b)                   # sigmoid applied ONCE, here
    g_t       = -exp(A_log) * softplus(x_t W_a + dt_bias)      # <= 0, float32
    S        <- exp(g_t) S                         # decay BEFORE the delta
    u_t       = beta_t (v_t - S^T k_t)             # error vs the DECAYED state
    S        <- S + k_t u_t^T
    o_t       = S^T q_t                            # read AFTER the update
    y_t       = out_proj(rms_norm(o_t) * silu(z_t))

`beta` and `g` are per value head; the 16 key heads are shared by the 48 value
heads (q/k are repeated `num_value_heads // num_key_heads` times). The output
gate is **silu**, which is what the release's `output_gate_type: "swish"` names,
and
the gain of `rms_norm` is applied with no plus-one (HF `Qwen3_5RMSNormGated`,
unlike the `1 + w` of `Qwen3_5RMSNorm` used by the block norms).

`log_decay` and `l2_normalize` are the two elementwise pieces the cores share;
both stay in float32 for the reason HF casts with `.float()`.

Two interchangeable cores, both f32 inside:
  * `chunk_gated_delta_rule`: the UT-transform chunked parallel form
    (`torch_chunk_gated_delta_rule`), used for prefill.
  * `recurrent_gated_delta_rule_step` / `recurrent_gated_delta_rule`: the token
    recurrence (`torch_recurrent_gated_delta_rule`), used for decode and, by
    the tests, as the ground truth of the chunked form.

`gdn_compute_dtype` (the release's `mamba_ssm_dtype`, float32) is the dtype the
delta-rule inputs are rounded to and the dtype of the two O(C^2) Gram matmuls;
the UT transform and the state recurrence are float32 whatever it says. At
`activation_dtype='float32'` that reproduces HF's kernels exactly; at bfloat16
it reproduces HF's rounding *schedule* but not its every rounding (`l2norm`
accumulates in float32 here, and the conv accumulates in
`conv_accumulation_dtype`). The result does not depend on `chunk_size`: the
chunked and recurrent cores agree to 1e-16 in float64 for every chunk length.

Packing and padding are load-bearing: a recurrence cannot be masked after the
fact. Padded positions (`segment_ids == 0`) get `g = 0, beta = 0`, i.e. an
exact identity update, are dropped from the conv window, and never reach the
state carried into decode. A new segment (`segment_positions == 0` at a valid
token) resets the state, the conv window and the intra-chunk interactions.
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

# HF `l2norm`: rsqrt(sum(x * x) + eps), the epsilon INSIDE the sqrt and on the
# sum of squares (modeling_qwen3_5.py:242-247).
L2NORM_EPS = 1e-6


# --- decode state -----------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class GatedDeltaNetDecodeState:
  """Constant-size decode state of one GatedDeltaNet layer.

  `apply` returns it under `extra_output['decode_state']`.

  Attributes:
    conv_state: `[B, conv_dim, W - 1]` float32 window of pre-conv inputs,
      newest at slot `W - 2`. HF's `conv_states` keeps W slots of which the
      oldest is never read by a width-W kernel; the slot is dropped here, so
      slot `i` of this state is HF's slot `i + 1`.
    recurrent_state: `[B, num_value_heads, key_head_dim, value_head_dim]`
      float32 delta-rule state, key-major as HF stores it.
  """

  conv_state: jax.Array
  recurrent_state: jax.Array


# --- functional core --------------------------------------------------------


def l2_normalize(x: Array, eps: float = L2NORM_EPS) -> jax.Array:
  """HF `l2norm` over the last axis, accumulated in float32.

  Args:
    x: array to normalize along its last axis.
    eps: added to the sum of squares before the rsqrt.

  Returns:
    The normalized array, in the dtype of `x`: HF normalizes before its kernels
    cast to float32, so a bfloat16 activation is normalized and rounded back to
    bfloat16.
  """
  x32 = jnp.asarray(x, jnp.float32)
  y = x32 * jax.lax.rsqrt(
      jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + eps
  )
  return jnp.asarray(y, jnp.result_type(x))


def log_decay(a: Array, a_log: Array, dt_bias: Array) -> jax.Array:
  """Per-head log decay `g <= 0`, in float32.

  `-exp(A_log) * softplus(a + dt_bias)`, HF's
  `-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`: float32
  whatever the activation dtype is, because `g` is cumulated over a whole chunk
  before it is exponentiated, so its rounding compounds.

  Args:
    a: `[..., H]` the `in_proj_a` projection.
    a_log: `[H]` log decay rate.
    dt_bias: `[H]` per-head bias, added BEFORE the softplus.

  Returns:
    `[..., H]` float32 log decay.
  """
  a32 = jnp.asarray(a, jnp.float32) + jnp.asarray(dt_bias, jnp.float32)
  return -jnp.exp(jnp.asarray(a_log, jnp.float32)) * jax.nn.softplus(a32)


def segment_index(segment_start: jax.Array) -> jax.Array:
  """Per-token index of the segment it belongs to, 0 for the carried one."""
  return jnp.cumsum(segment_start.astype(jnp.int32), axis=1)


def causal_depthwise_conv(
    x: Array,
    w: Array,
    state: Array | None = None,
    *,
    valid: jax.Array | None = None,
    seg_index: jax.Array | None = None,
    accum_dtype: DTypeLike = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
  """Causal depthwise conv (width W, no bias) + SiLU, HF `causal_conv1d_fn`.

  `y[t] = silu(sum_i w[:, i] * x[t - (W - 1) + i])`, i.e. `w[:, W - 1]` taps the
  current token and missing history comes from `state` (zeros for a fresh
  sequence). Taps that reach into a different segment are dropped, and the
  returned window holds the last `W - 1` valid inputs of the row's LAST
  segment, so neither padding nor a packed neighbour leaks into the next call.

  Args:
    x: `[B, T, C]` pre-conv projections.
    w: `[C, W]` depthwise filters.
    state: `[B, C, W - 1]` previous window, or None for a fresh sequence.
    valid: `[B, T]` bool; False positions are padding.
    seg_index: `[B, T]` int32 segment counter (`segment_index`); the carried
      segment is 0, so a row with no new segment reads all of `state`.
    accum_dtype: dtype of the multiply-accumulate; float32 in the release.

  Returns:
    `(y, new_state)`: `[B, T, C]` float32 activations (zero at padding) and the
    `[B, C, W - 1]` float32 updated window.

  Precondition: padding only FOLLOWS a segment's valid positions, unless the
  taps it occupies would have read zeros anyway -- i.e. unless the row's conv
  history is empty, because it is a fresh sequence or because its first valid
  token opens a new segment. Padding that precedes a CONTINUATION displaces the
  carried window, and a hole inside a segment shifts the taps by one slot
  relative to the recurrence, which skips padding entirely. The returned window
  is correct either way -- it is extracted by valid-rank -- so only the current
  call's output is affected.
  """
  b, t, c = jnp.shape(x)
  kw = jnp.shape(w)[-1]
  xf = jnp.asarray(x, accum_dtype)
  wf = jnp.asarray(w, accum_dtype)
  if state is None:
    prev = jnp.zeros((b, kw - 1, c), accum_dtype)
  else:
    prev = jnp.asarray(jnp.swapaxes(jnp.asarray(state), 1, 2), accum_dtype)
  ext = jnp.concatenate([prev, xf], axis=1)  # [B, W - 1 + T, C]

  if valid is None:
    valid = jnp.ones((b, t), bool)
  if seg_index is None:
    seg_index = jnp.zeros((b, t), jnp.int32)
  # The W - 1 history slots belong to the carried segment (index 0), so a row
  # that opens a new segment at t = 0 masks all of them out.
  ext_seg = jnp.concatenate(
      [jnp.zeros((b, kw - 1), jnp.int32), seg_index], axis=1
  )

  y = jnp.zeros((b, t, c), accum_dtype)
  for i in range(kw):
    src = jax.lax.dynamic_slice_in_dim(ext, i, t, axis=1)
    same = jax.lax.dynamic_slice_in_dim(ext_seg, i, t, axis=1) == seg_index
    y = y + wf[:, i] * src * same[:, :, None]
  y = jax.nn.silu(jnp.asarray(y, jnp.float32)) * valid[:, :, None]

  # New window = the last W - 1 valid positions of the extended timeline.
  # Ranking by the running count of valid tokens (the carried slots count as
  # valid) is exact for arbitrary pad placement, not just right padding.
  ext_valid = jnp.concatenate(
      [jnp.ones((b, kw - 1), jnp.int32), valid.astype(jnp.int32)], axis=1
  )
  rank = jnp.cumsum(ext_valid, axis=1)  # [B, W - 1 + T], 1-based
  lengths = rank[:, -1] - (kw - 1)  # valid tokens in x
  target = lengths[:, None] + 1 + jnp.arange(kw - 1, dtype=jnp.int32)[None, :]
  # `rank` is non-decreasing and increments exactly at valid positions, so the
  # FIRST hit is the valid token itself.
  idx = jnp.argmax(rank[:, :, None] == target[:, None, :], axis=1)  # [B, W - 1]
  window = jnp.take_along_axis(ext, idx[:, :, None], axis=1)
  window_seg = jnp.take_along_axis(ext_seg, idx, axis=1)
  if kw > 1:
    window = window * (window_seg == window_seg[:, -1:])[:, :, None]
  new_state = jnp.asarray(jnp.swapaxes(window, 1, 2), jnp.float32)
  return y, new_state


def recurrent_gated_delta_rule_step(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
    *,
    compute_dtype: DTypeLike = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
  """One token of the gated delta rule, HF `torch_recurrent_gated_delta_rule`.

  Args:
    q: `[B, H, K]` query, neither normalized nor scaled yet.
    k: `[B, H, K]` key, not normalized yet.
    v: `[B, H, V]` value.
    g: `[B, H]` log decay (<= 0); 0 makes the step an identity update.
    beta: `[B, H]` delta-rule step size in (0, 1); 0 makes the step an identity
      update.
    state: `[B, H, K, V]` recurrent state.
    compute_dtype: dtype the inputs are rounded to; the state math is float32.

  Returns:
    `(o, new_state)`: `[B, H, V]` float32 output read from the POST-update
    state, and the updated `[B, H, K, V]` float32 state.
  """
  cdt = jnp.dtype(compute_dtype)
  kdim = jnp.shape(q)[-1]
  qn = jnp.asarray(l2_normalize(q), cdt) * jnp.asarray(kdim**-0.5, cdt)
  kn = jnp.asarray(l2_normalize(k), cdt)
  qf = jnp.asarray(qn, jnp.float32)
  kf = jnp.asarray(kn, jnp.float32)
  vf = jnp.asarray(jnp.asarray(v, cdt), jnp.float32)
  betaf = jnp.asarray(jnp.asarray(beta, cdt), jnp.float32)
  # The default precision would contract these in bfloat16 passes on TPU, i.e.
  # decode would carry the state at a lower precision than prefill.
  with jax.default_matmul_precision('float32'):
    s = jnp.asarray(state, jnp.float32) * jnp.exp(
        jnp.asarray(g, jnp.float32)
    )[..., None, None]
    u = betaf[..., None] * (vf - jnp.einsum('bhkv,bhk->bhv', s, kf))
    s = s + kf[..., None] * u[..., None, :]
    o = jnp.einsum('bhkv,bhk->bhv', s, qf)
  return o, s


def recurrent_gated_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    initial_state: Array | None = None,
    *,
    segment_start: jax.Array | None = None,
    compute_dtype: DTypeLike = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
  """Sequential gated delta rule over `[B, T, H, *]`; the chunked form's twin.

  One `recurrent_gated_delta_rule_step` per token, scanned over time. Kept
  because it is the definition of the op: prefill uses
  `chunk_gated_delta_rule`, decode uses the single step.

  Args:
    q: `[B, T, H, K]` queries, neither normalized nor scaled yet.
    k: `[B, T, H, K]` keys, not normalized yet.
    v: `[B, T, H, V]` values.
    g: `[B, T, H]` log decay (<= 0); 0 at padding.
    beta: `[B, T, H]` step sizes in (0, 1); 0 at padding.
    initial_state: `[B, H, K, V]` carried state, or None for zeros.
    segment_start: `[B, T]` bool, True at the first token of a new segment,
      which resets the state before that token is consumed.
    compute_dtype: dtype the inputs are rounded to; the state math is float32.

  Returns:
    `(o, final_state)`: `[B, T, H, V]` float32 outputs and the `[B, H, K, V]`
    float32 state after the last token, as `chunk_gated_delta_rule` returns
    them.
  """
  b, t, h, kdim = jnp.shape(q)
  vdim = jnp.shape(v)[-1]
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
    o_t, s = recurrent_gated_delta_rule_step(
        q_t, k_t, v_t, g_t, beta_t, s, compute_dtype=compute_dtype
    )
    return s, o_t

  # No precision context here: the step owns the precision of its own matmuls.
  xs = jax.tree.map(
      lambda a: jnp.swapaxes(jnp.asarray(a), 0, 1),
      (q, k, v, g, beta, reset),
  )
  s_final, o = jax.lax.scan(step, s0, xs)
  return jnp.swapaxes(o, 0, 1), s_final


def chunk_gated_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    initial_state: Array | None = None,
    chunk_size: int = 32,
    *,
    segment_start: jax.Array | None = None,
    compute_dtype: DTypeLike = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
  """Chunkwise-parallel gated delta rule, HF `torch_chunk_gated_delta_rule`.

  Per chunk of C tokens, with `gc` the inclusive within-chunk cumulative log
  decay and all pairs restricted to one segment:

    Aqk[i,j] = <q_i, k_j> e^{gc_i - gc_j}                      j <= i
    Akk[i,j] = beta_i <k_i, k_j> e^{gc_i - gc_j}               j <  i
    Tm       = (I + Akk)^{-1}
    w        = Tm (beta k e^{gc}) ;  u = Tm (beta v)
    v_new    = u - w S ;  o = Aqk v_new + (q e^{gc}) S
    S       <- e^{gc_last} S + (k e^{gc_last - gc})^T v_new

  The exponent arguments are all <= 0 because `g <= 0` makes `gc`
  non-increasing and the pairwise differences are materialized instead of being
  factored into `e^{gc_i} * e^{-gc_j}`, whose second factor is unbounded. HF
  runs the UT transform as a C-step forward substitution; the equivalent
  triangular solve here is one `solve_triangular` per chunk.

  Args:
    q: `[B, T, H, K]` queries, neither normalized nor scaled yet.
    k: `[B, T, H, K]` keys, not normalized yet.
    v: `[B, T, H, V]` values.
    g: `[B, T, H]` log decay (<= 0); 0 at padding.
    beta: `[B, T, H]` step sizes in (0, 1); 0 at padding.
    initial_state: `[B, H, K, V]` carried state, or None for zeros.
    chunk_size: tokens per chunk; T is padded up to a multiple of it with
      identity updates.
    segment_start: `[B, T]` bool, True at the first token of a new segment,
      which resets the state and cuts the intra-chunk interactions.
    compute_dtype: dtype the inputs and the two Gram matmuls are rounded to;
      the UT transform and the state recurrence are float32.

  Returns:
    `(o, final_state)`: `[B, T, H, V]` float32 outputs and the `[B, H, K, V]`
    float32 state after the last token.
  """
  b, t, h, kdim = jnp.shape(q)
  vdim = jnp.shape(v)[-1]
  c = chunk_size
  cdt = jnp.dtype(compute_dtype)
  qi = jnp.asarray(l2_normalize(q), cdt) * jnp.asarray(kdim**-0.5, cdt)
  ki = jnp.asarray(l2_normalize(k), cdt)
  vi = jnp.asarray(v, cdt)
  betai = jnp.asarray(beta, cdt)
  gi = jnp.asarray(g, jnp.float32)
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
    qi, ki, vi, gi, betai, reset = (
        pad_time(a) for a in (qi, ki, vi, gi, betai, reset)
    )
  nc = (t + pad) // c

  def to_chunks(a: jax.Array) -> jax.Array:
    # [B, T, H, ...] -> [NC, B, H, C, ...]; the chunk axis leads for the scan.
    a = jnp.reshape(a, (b, nc, c) + jnp.shape(a)[2:])
    if a.ndim == 5:  # q/k/v: [B, NC, C, H, D]
      return jnp.transpose(a, (1, 0, 3, 2, 4))
    return jnp.transpose(a, (1, 0, 3, 2))  # g/beta: [B, NC, C, H]

  qs, ks, vs, gs, betas = (to_chunks(a) for a in (qi, ki, vi, gi, betai))
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
    same = (seg_i[:, :, None] == seg_i[:, None, :])[:, None]  # [B, 1, C, C]
    carry = (seg_i == 0).astype(jnp.float32)[:, None, :, None]  # [B, 1, C, 1]
    gc = jnp.cumsum(g_i, axis=-1)  # [B, H, C]
    decay = jnp.exp(jnp.minimum(gc[..., :, None] - gc[..., None, :], 0.0))
    k_beta = k_i * beta_i[..., None]
    akk = jnp.asarray(
        jnp.matmul(k_beta, jnp.swapaxes(k_i, -1, -2)), jnp.float32
    )
    akk = akk * decay * tri_strict * same
    # (I + Akk)^{-1}, block diagonal per segment since Akk is masked to
    # same-segment pairs. `unit_diagonal` supplies the I.
    t_inv = jax_linalg.solve_triangular(
        akk, jnp.broadcast_to(eye, akk.shape), lower=True, unit_diagonal=True
    )
    u = jnp.matmul(
        t_inv, jnp.asarray(v_i * beta_i[..., None], jnp.float32)
    )
    # Rows whose segment started inside this chunk see no carried state, and
    # for the rest the within-chunk cumsum IS the within-segment cumsum.
    w = carry * jnp.matmul(
        t_inv, jnp.asarray(k_beta, jnp.float32) * jnp.exp(gc)[..., None]
    )
    v_new = u - jnp.matmul(w, s)
    aqk = jnp.asarray(
        jnp.matmul(q_i, jnp.swapaxes(k_i, -1, -2)), jnp.float32
    )
    aqk = aqk * decay * tri_causal * same
    o_i = jnp.matmul(aqk, v_new) + carry * jnp.matmul(
        jnp.asarray(q_i, jnp.float32) * jnp.exp(gc)[..., None], s
    )
    g_last = gc[..., -1:]  # [B, H, 1]
    last_same = (seg_i == seg_i[:, -1:]).astype(jnp.float32)[:, None, :, None]
    k_rest = (
        jnp.asarray(k_i, jnp.float32)
        * jnp.exp(jnp.minimum(g_last - gc, 0.0))[..., None]
        * last_same
    )
    s = s * (carry[:, :, -1] * jnp.exp(g_last))[..., None] + jnp.matmul(
        jnp.swapaxes(k_rest, -1, -2), v_new
    )
    return s, o_i

  with jax.default_matmul_precision('float32'):
    s_final, o_chunks = jax.lax.scan(
        chunk_step, s0, (qs, ks, vs, gs, betas, seg)
    )
  # [NC, B, H, C, V] -> [B, T, H, V]
  o = jnp.transpose(o_chunks, (1, 0, 3, 2, 4)).reshape(b, nc * c, h, vdim)
  return o[:, :t], s_final


# --- the module -------------------------------------------------------------


@module.ModuleRegistry.register
@dataclasses.dataclass
class Qwen38GatedDeltaNet(module.SimplyModule):
  """Qwen3.8's GatedDeltaNet linear-attention token mixer.

  Parameters (`conv_dim = 2 * num_key_heads * key_head_dim + value_dim`,
  `value_dim = num_value_heads * value_head_dim`), named as the released
  checkpoint's `model.language_model.layers.N.linear_attn.*` are mapped by
  `utils/ckpt_format.py`:

    in_proj_qkv/w [D, conv_dim]   in_proj_z/w [D, value_dim]
    in_proj_b/w   [D, num_v]      in_proj_a/w [D, num_v]
    conv1d/w [conv_dim, W] (f32)  A_log [num_v] (f32)  dt_bias [num_v] (f32)
    norm/scale [value_head_dim] (f32)   out_proj/w [value_dim, D]

  Attributes:
    model_dim: residual stream width D.
    num_key_heads: key/query heads; 16 for the 27B.
    num_value_heads: value heads; 48, i.e. three value heads per key head.
    key_head_dim: K, the query/key width per head.
    value_head_dim: V, the value width per head.
    conv_kernel_dim: W of the causal depthwise short convolution.
    chunk_size: chunk length C of the prefill core.
    activation_dtype: dtype of the projections and of the layer output.
    gdn_compute_dtype: dtype the delta-rule inputs are rounded to (the
      release's `mamba_ssm_dtype`); None means `activation_dtype`.
    conv_accumulation_dtype: dtype of the conv multiply-accumulate; None means
      `activation_dtype`.
    rms_norm_epsilon: epsilon of the gated output RMSNorm.
    weight_init: initializer of the projection and conv weights.
    sharding_config: `config_lib.ShardingConfig` or None for no annotations.
    in_proj_partition: weight annotation of the four input projections `[D, *]`;
      defaults to the last two axes of `sharding_config.ffn0_partition`. Every
      `*_partition` field below behaves the same way: None means "derive from
      `sharding_config`", and `sharding_config=None` means "annotate nothing".
    out_proj_partition: weight annotation of `out_proj [value_dim, D]`.
    conv_partition: weight annotation of the conv filters `[conv_dim, W]`.
    state_partition: annotation of the recurrent state `[B, H, K, V]`.
    conv_state_partition: annotation of the conv state `[B, conv_dim, W - 1]`.
  """

  model_dim: int
  num_key_heads: int = 16
  num_value_heads: int = 48
  key_head_dim: int = 128
  value_head_dim: int = 128
  conv_kernel_dim: int = 4
  chunk_size: int = 32
  activation_dtype: DTypeLike = 'bfloat16'
  gdn_compute_dtype: DTypeLike | None = 'float32'
  conv_accumulation_dtype: DTypeLike | None = 'float32'
  rms_norm_epsilon: float = 1e-6
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  sharding_config: SimplyConfig | None = None
  in_proj_partition: PartitionAnnotation = None
  out_proj_partition: PartitionAnnotation = None
  conv_partition: PartitionAnnotation = None
  state_partition: PartitionAnnotation = None
  conv_state_partition: PartitionAnnotation = None

  @property
  def key_dim(self) -> int:
    return self.num_key_heads * self.key_head_dim

  @property
  def value_dim(self) -> int:
    return self.num_value_heads * self.value_head_dim

  @property
  def conv_dim(self) -> int:
    return 2 * self.key_dim + self.value_dim

  @property
  def compute_dtype(self) -> jnp.dtype:
    if self.gdn_compute_dtype is None:
      return jnp.dtype(self.activation_dtype)
    return jnp.dtype(self.gdn_compute_dtype)

  def setup(self) -> None:
    if self.num_value_heads % self.num_key_heads != 0:
      raise ValueError(
          f'{self.num_value_heads=} must be divisible by {self.num_key_heads=}'
      )
    self._resolve_partitions()
    # pylint: disable-next=g-long-lambda
    proj = lambda out_dim, out_partition: module.EinsumLinear(
        eqn='io,...i->...o',
        weight_shape=[self.model_dim, out_dim],
        bias_term='',
        activation_dtype=self.activation_dtype,
        weight_partition=self.in_proj_partition,
        output_partition=out_partition,
        weight_init=self.weight_init,
    )
    self.in_proj_qkv = proj(self.conv_dim, self._activation)
    self.in_proj_z = proj(self.value_dim, self._activation)
    self.in_proj_b = proj(self.num_value_heads, sharding_lib.NOT_ANNOTATED)
    self.in_proj_a = proj(self.num_value_heads, sharding_lib.NOT_ANNOTATED)
    # HF `Qwen3_5RMSNormGated`: the gain multiplies the normalized value with
    # NO plus-one (unlike `Qwen3_5RMSNorm`, the block norm), and the gate is
    # applied after it, in float32, by `apply` below.
    self.norm = model_lib.LayerNorm(
        dim=self.value_head_dim,
        use_bias=False,
        use_scale=True,
        scale_plus_one=False,
        activation_dtype=self.activation_dtype,
        # `LayerNorm.init` annotates unconditionally (`EinsumLinear.init` does
        # not), so the gain needs the sentinel to stay unannotated.
        scale_partition=sharding_lib.NOT_ANNOTATED,
        epsilon=self.rms_norm_epsilon,
    )
    self.out_proj = module.EinsumLinear(
        eqn='io,...i->...o',
        weight_shape=[self.value_dim, self.model_dim],
        bias_term='',
        activation_dtype=self.activation_dtype,
        weight_partition=self.out_proj_partition,
        output_partition=self._activation,
        weight_init=self.weight_init,
    )

  def _resolve_partitions(self) -> None:
    """Fills the unset `*_partition` fields from `sharding_config`."""
    self._activation = sharding_lib.NOT_ANNOTATED
    if self.sharding_config is None:
      # The sentinel, not None: `with_sharding_constraint` passes
      # `NOT_ANNOTATED` through, while None constrains to fully replicated.
      for name in ('conv_partition', 'state_partition', 'conv_state_partition'):
        if getattr(self, name) is None:
          setattr(self, name, sharding_lib.NOT_ANNOTATED)
      return
    self._activation = self.sharding_config.activation_partition
    # Under an expert-parallel sharding `ffn0_partition` is the rank-3
    # expert-stack annotation; the dense projections here take its last two
    # entries, as the input projections of a dense FFN do.
    if self.in_proj_partition is None:
      self.in_proj_partition = tuple(self.sharding_config.ffn0_partition)[-2:]
    if self.out_proj_partition is None:
      self.out_proj_partition = tuple(self.sharding_config.ffn1_partition)[-2:]
    if self.conv_partition is None:
      self.conv_partition = ('model', None)
    # The states are per sequence, so they follow the activations' batch axis
    # instead of being replicated over it.
    batch = self._activation[0] if self._activation else None
    if self.state_partition is None:
      self.state_partition = (batch, 'model', None, None)
    if self.conv_state_partition is None:
      self.conv_state_partition = (batch, 'model', None)

  def init(self, prng_key: PRNGKey) -> PyTree:
    keys = jax.random.split(prng_key, 7)
    return {
        'in_proj_qkv': self.in_proj_qkv.init(keys[0]),
        'in_proj_z': self.in_proj_z.init(keys[1]),
        'in_proj_b': self.in_proj_b.init(keys[2]),
        'in_proj_a': self.in_proj_a.init(keys[3]),
        'conv1d': {
            'w': common.AnnotatedArray.create(
                sharding_lib.with_sharding_constraint(
                    self.weight_init(
                        keys[4],
                        shape=(self.conv_dim, self.conv_kernel_dim),
                        dtype='float32',
                        dim_annotation='h.',
                    ),
                    self.conv_partition,
                ),
                dim_annotation='h.',
            )
        },
        # HF initializes `dt_bias` to ones and `A_log` to log U(0.01, 16); both
        # are overwritten by the checkpoint.
        'dt_bias': common.AnnotatedArray.create(
            jnp.ones((self.num_value_heads,), jnp.float32), dim_annotation='h'
        ),
        'A_log': common.AnnotatedArray.create(
            jnp.log(
                jax.random.uniform(
                    keys[5],
                    (self.num_value_heads,),
                    minval=0.01,
                    maxval=16.0,
                    dtype=jnp.float32,
                )
            ),
            dim_annotation='h',
        ),
        'norm': self.norm.init(),
        'out_proj': self.out_proj.init(keys[6]),
    }

  def init_decode_state(
      self, batch_size: int, max_seq_len: int
  ) -> GatedDeltaNetDecodeState:
    """Zeroed state; a GatedDeltaNet's is constant in the sequence length.

    Args:
      batch_size: number of sequences.
      max_seq_len: unused; the state does not grow with the sequence.

    Returns:
      A zero `GatedDeltaNetDecodeState`.
    """
    del max_seq_len
    conv_state = jnp.zeros(
        (batch_size, self.conv_dim, self.conv_kernel_dim - 1), jnp.float32
    )
    recurrent_state = jnp.zeros(
        (
            batch_size,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        ),
        jnp.float32,
    )
    return GatedDeltaNetDecodeState(
        conv_state=sharding_lib.with_sharding_constraint(
            conv_state, self.conv_state_partition
        ),
        recurrent_state=sharding_lib.with_sharding_constraint(
            recurrent_state, self.state_partition
        ),
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
      decode_state: GatedDeltaNetDecodeState | None = None,
      **kwargs: Any,
  ) -> tuple[jax.Array, dict[str, GatedDeltaNetDecodeState | None]]:
    """Runs one GatedDeltaNet layer.

    Args:
      params: this module's parameter subtree.
      x: `[B, T, D]` normed block input.
      segment_ids: `[B, T]` int32; 0 marks padding, which must leave the
        recurrent state and the conv window untouched.
      segment_positions: `[B, T]` int32 position within the segment; a 0 at a
        non-padding token starts a new segment and resets the state. This is
        the only signal of a segment boundary (as in `kimi_k3/utils/kda.py`):
        `segment_ids` says what is padding, not where a segment begins.
      extra_inputs: unused side channel.
      inputs_mask: `[B, T]` bool override of `segment_ids != 0`.
      decode_state: carried state, or None for a stateless prefill.
      **kwargs: unused; keeps the mixer interface open.

    Returns:
      `(out, extra_output)` with `out` of shape `[B, T, D]` and
      `extra_output['decode_state']` the updated state (None iff
      `decode_state` was None).
    """
    del extra_inputs, kwargs
    # `cast`: the leaves are AnnotatedArray / quantized dicts until
    # `convert_or_dequantize`, and PyTree is too wide to index.
    raw = cast(dict[str, Any], common.get_raw_arrays(params))
    b, t, _ = x.shape

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
    seg_index = segment_index(segment_start)

    # HF `apply_mask_to_padding_states`.
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

    mixed_qkv = self.in_proj_qkv.apply(raw['in_proj_qkv'], x)
    conv_out, new_conv_state = causal_depthwise_conv(
        mixed_qkv,
        common.convert_or_dequantize(raw['conv1d']['w'], dtype=jnp.float32),
        conv_state,
        valid=valid,
        seg_index=seg_index,
        accum_dtype=self.conv_accumulation_dtype or self.activation_dtype,
    )
    conv_out = jnp.asarray(conv_out, self.activation_dtype)
    query, key, value = jnp.split(
        conv_out, [self.key_dim, 2 * self.key_dim], axis=-1
    )
    query = jnp.reshape(query, (b, t, self.num_key_heads, self.key_head_dim))
    key = jnp.reshape(key, (b, t, self.num_key_heads, self.key_head_dim))
    value = jnp.reshape(
        value, (b, t, self.num_value_heads, self.value_head_dim)
    )
    if self.num_value_heads != self.num_key_heads:
      repeats = self.num_value_heads // self.num_key_heads
      query = jnp.repeat(query, repeats, axis=2)
      key = jnp.repeat(key, repeats, axis=2)

    z = self.in_proj_z.apply(raw['in_proj_z'], x)
    z = jnp.reshape(z, (b, t, self.num_value_heads, self.value_head_dim))
    # Exactly one sigmoid: HF `beta = b.sigmoid()` in the layer, and neither
    # core applies another one.
    beta = jax.nn.sigmoid(self.in_proj_b.apply(raw['in_proj_b'], x))
    gate = log_decay(
        self.in_proj_a.apply(raw['in_proj_a'], x),
        common.convert_or_dequantize(raw['A_log'], dtype=jnp.float32),
        common.convert_or_dequantize(raw['dt_bias'], dtype=jnp.float32),
    )

    # Padding must be an exact identity update of the recurrence: `g = 0` is no
    # decay and `beta = 0` is no delta. That pair is the load-bearing mask --
    # `g` and `beta` are the only quantities a pad position would otherwise
    # feed into the state. Zeroing q/k/v is belt-and-braces: the conv already
    # returns 0 there (`causal_depthwise_conv` multiplies by `valid`).
    query = jnp.where(valid[:, :, None, None], query, 0.0)
    key = jnp.where(valid[:, :, None, None], key, 0.0)
    value = jnp.where(valid[:, :, None, None], value, 0.0)
    gate = jnp.where(valid[:, :, None], gate, 0.0)
    beta = jnp.where(valid[:, :, None], beta, 0.0)

    if state is None:
      state = jnp.zeros(
          (b, self.num_value_heads, self.key_head_dim, self.value_head_dim),
          jnp.float32,
      )
    # HF additionally requires a non-empty cache (`use_precomputed_states`); a
    # one-token prefill from a zero state is the same update either way.
    if t == 1 and decode_state is not None:  # Cached single-token decode.
      state = jnp.where(segment_start[:, 0, None, None, None], 0.0, state)
      out, new_recurrent_state = recurrent_gated_delta_rule_step(
          query[:, 0],
          key[:, 0],
          value[:, 0],
          gate[:, 0],
          beta[:, 0],
          state,
          compute_dtype=self.compute_dtype,
      )
      out = out[:, None]
    else:
      out, new_recurrent_state = chunk_gated_delta_rule(
          query,
          key,
          value,
          gate,
          beta,
          state,
          self.chunk_size,
          segment_start=segment_start,
          compute_dtype=self.compute_dtype,
      )

    flat_out = jnp.reshape(
        jnp.asarray(out, self.activation_dtype), (-1, self.value_head_dim)
    )
    flat_z = jnp.reshape(z, (-1, self.value_head_dim))
    flat_out = self.norm.apply(raw['norm'], flat_out)
    # The gate is silu -- the release's `output_gate_type: "swish"` -- and is
    # applied in float32 on top of the normalized value.
    flat_out = jnp.asarray(
        flat_out * jax.nn.silu(jnp.asarray(flat_z, jnp.float32)),
        self.activation_dtype,
    )
    out = self.out_proj.apply(
        raw['out_proj'], jnp.reshape(flat_out, (b, t, self.value_dim))
    )
    # Belt-and-braces as well: `q = 0` at a pad already makes the core output,
    # the norm and `out_proj(0)` exactly zero.
    out = jnp.where(valid[:, :, None], out, 0.0)
    out = sharding_lib.with_sharding_constraint(out, self._activation)

    new_decode_state = None
    if decode_state is not None:
      new_decode_state = GatedDeltaNetDecodeState(
          conv_state=sharding_lib.with_sharding_constraint(
              new_conv_state, self.conv_state_partition
          ),
          recurrent_state=sharding_lib.with_sharding_constraint(
              new_recurrent_state, self.state_partition
          ),
      )
    return out, {'decode_state': new_decode_state}

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
"""Tests for Kimi Delta Attention.

The numpy reference below is transcribed from a standalone NumPy reference
that was validated against fla-core 0.5.2 (the version HF
`modeling_kimi_linear.py` requires) to 1.6e-16 in f64; every function keeps the
fla file it mirrors in its docstring. It is deliberately an independent
re-derivation of the math rather than a call into `kda.py`.
"""

import functools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.zoo.kimi_k3.utils import kda

# Small but non-degenerate geometry (K3 is D=7168, H=96, K=V=128, W=4, R=128).
B, T, D, H, K, R, W = 2, 12, 16, 2, 8, 4, 4
RMS_EPS = 1e-5
LOWER_BOUND = -5.0
ATOL = 2e-5


# --------------------------------------------------------------------------
# numpy reference (f64)
# --------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))


def _silu(x: np.ndarray) -> np.ndarray:
  return x * _sigmoid(x)


def _l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  """fla/modules/l2norm.py: eps is added to the SUM OF SQUARES."""
  return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def _kda_gate(
    z: np.ndarray,  # [..., H, K]
    a_log: np.ndarray,  # [H]
    dt_bias: np.ndarray,  # [H, K]
    lower_bound: float | None = LOWER_BOUND,
) -> np.ndarray:
  """fla/ops/kda/gate.py::naive_kda_lowerbound_gate."""
  z = np.asarray(z, np.float64) + dt_bias
  a = np.exp(np.asarray(a_log, np.float64))[:, None]
  if lower_bound is None:
    return -a * np.logaddexp(0.0, z)
  return lower_bound * _sigmoid(a * z)


def _rms_norm_gated(
    x: np.ndarray, gate: np.ndarray, weight: np.ndarray, eps: float = RMS_EPS
) -> np.ndarray:
  """fla/modules/fused_norm_gate.py: gain before the (unnormalized) gate."""
  x = np.asarray(x, np.float64)
  rstd = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
  return x * rstd * weight * _sigmoid(gate)


def _short_conv(
    x: np.ndarray,  # [B, T, H, D]
    weight: np.ndarray,  # [H, D, W]
    state: np.ndarray | None = None,  # [B, H, D, W], newest at slot W-1
) -> tuple[np.ndarray, np.ndarray]:
  """Causal depthwise conv + silu; fla ShortConvolution (bias-free)."""
  x = np.asarray(x, np.float64)
  b, t = x.shape[:2]
  kw = weight.shape[-1]
  buf = (
      np.zeros((b, kw) + x.shape[2:])
      if state is None
      else np.moveaxis(np.asarray(state, np.float64), -1, 1)
  )
  ext = np.concatenate([buf, x], axis=1)  # [B, W + T, H, D]
  y = np.zeros_like(x)
  for i in range(kw):
    y += ext[:, 1 + i : 1 + i + t] * weight[:, :, i]
  return _silu(y), np.moveaxis(ext[:, -kw:], 1, -1)


def _kda_recurrent(
    q: np.ndarray,  # [B, T, H, K] post-l2norm
    k: np.ndarray,  # [B, T, H, K] post-l2norm
    v: np.ndarray,  # [B, T, H, V]
    g: np.ndarray,  # [B, T, H, K] log decay
    beta: np.ndarray,  # [B, T, H] post-sigmoid
    scale: float,
    initial_state: np.ndarray | None = None,  # [B, H, K, V]
) -> tuple[np.ndarray, np.ndarray]:
  """fla/ops/kda/naive.py::naive_recurrent_kda.

  S <- diag(exp(g)) S ;  u = beta (v - S^T k) ;  S <- S + k u^T ;  o = S^T q

  Args:
    q: `[B, T, H, K]` l2-normalized queries, still unscaled.
    k: `[B, T, H, K]` l2-normalized keys.
    v: `[B, T, H, V]` values.
    g: `[B, T, H, K]` log decay (<= 0).
    beta: `[B, T, H]` delta-rule step sizes in (0, 1).
    scale: query scale, applied here rather than by the caller.
    initial_state: `[B, H, K, V]` carried state, or None for zeros.

  Returns:
    `(o, state)`: `[B, T, H, V]` float64 outputs and the `[B, H, K, V]` state
    after the last token.
  """
  q, k, v, g, beta = (np.asarray(a, np.float64) for a in (q, k, v, g, beta))
  q = q * scale
  b, t, h, vdim = v.shape
  s = np.zeros((b, h, q.shape[-1], vdim))
  if initial_state is not None:
    s = s + np.asarray(initial_state, np.float64)
  o = np.zeros_like(v)
  for i in range(t):
    s = s * np.exp(g[:, i])[..., None]
    err = v[:, i] - np.einsum('bhk,bhkv->bhv', k[:, i], s)
    s = s + np.einsum('bhk,bhv->bhkv', beta[:, i][..., None] * k[:, i], err)
    o[:, i] = np.einsum('bhk,bhkv->bhv', q[:, i], s)
  return o, s


def _reference_layer(
    x: np.ndarray,  # [B, T, D]
    w: dict[str, np.ndarray],
    state: tuple[np.ndarray, np.ndarray] | None = None,
    lower_bound: float | None = LOWER_BOUND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """One KDA layer; mirrors HF `KimiDeltaAttention.forward`.

  Args:
    x: `[B, T, D]` input.
    w: weights in the Simply layout (see `_random_weights`).
    state: optional `(conv_state [B, 3, H, D, W], recurrent_state [B,H,K,V])`.
    lower_bound: gate bound, or None for fla's softplus gate.

  Returns:
    `(y, conv_state, recurrent_state)`.
  """
  x = np.asarray(x, np.float64)
  conv_state = None if state is None else state[0]
  conv_out = {}
  new_conv = []
  for i, name in enumerate(('q', 'k', 'v')):
    pre = np.einsum('bti,ihd->bthd', x, w[f'{name}_proj'])
    post, conv_i = _short_conv(
        pre, w[f'{name}_conv'], None if conv_state is None else conv_state[:, i]
    )
    conv_out[name] = post
    new_conv.append(conv_i)
  q, k, v = conv_out['q'], conv_out['k'], conv_out['v']

  z = np.einsum('btr,rhd->bthd', x @ w['f_a_proj'], w['f_b_proj'])
  g = _kda_gate(z, w['a_log'], w['dt_bias'], lower_bound)
  beta = _sigmoid(np.einsum('bti,ih->bth', x, w['b_proj']))
  o, s = _kda_recurrent(
      _l2norm(q),
      _l2norm(k),
      v,
      g,
      beta,
      scale=K**-0.5,
      initial_state=None if state is None else state[1],
  )
  gate = np.einsum('bti,ihd->bthd', x, w['g_proj'])
  o = _rms_norm_gated(o, gate, w['o_norm'])
  y = np.einsum('bthd,hdo->bto', o, w['o_proj'])
  return y, np.stack(new_conv, axis=1), s


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _random_weights(seed: int = 0) -> dict[str, np.ndarray]:
  """Weights in the layout of the `KimiK3DeltaAttention` param tree."""
  rng = np.random.default_rng(seed)
  normal = lambda *shape: rng.normal(size=shape) / np.sqrt(shape[0])
  return {
      'q_proj': normal(D, H, K),
      'k_proj': normal(D, H, K),
      'v_proj': normal(D, H, K),
      'q_conv': rng.normal(size=(H, K, W)),
      'k_conv': rng.normal(size=(H, K, W)),
      'v_conv': rng.normal(size=(H, K, W)),
      'f_a_proj': normal(D, R),
      'f_b_proj': normal(R, H, K),
      'dt_bias': rng.normal(size=(H, K)),
      'a_log': np.log(rng.uniform(1.0, 16.0, size=(H,))),
      'b_proj': normal(D, H),
      'g_proj': normal(D, H, K),
      'o_norm': rng.normal(1.0, 0.1, size=(K,)),
      'o_proj': normal(H, K, D),
  }


def _params(w: dict[str, np.ndarray]) -> dict[str, Any]:
  f32 = lambda name: jnp.asarray(w[name], jnp.float32)
  return {
      'q_proj': {'w': f32('q_proj')},
      'k_proj': {'w': f32('k_proj')},
      'v_proj': {'w': f32('v_proj')},
      'q_conv': {'w': f32('q_conv')},
      'k_conv': {'w': f32('k_conv')},
      'v_conv': {'w': f32('v_conv')},
      'f_a_proj': {'w': f32('f_a_proj')},
      'f_b_proj': {'w': f32('f_b_proj')},
      'dt_bias': f32('dt_bias'),
      'a_log': f32('a_log'),
      'b_proj': {'w': f32('b_proj')},
      'g_proj': {'w': f32('g_proj')},
      'o_norm': {'scale': f32('o_norm')},
      'o_proj': {'w': f32('o_proj')},
  }


@functools.lru_cache(maxsize=None)
def _layer(
    chunk_size: int = 8,
    activation_dtype: str = 'float32',
    sharded: bool = False,
    lower_bound: float | None = LOWER_BOUND,
) -> kda.KimiK3DeltaAttention:
  """A tiny KDA layer; cached so `_apply_fn` compiles each variant once."""
  return kda.KimiK3DeltaAttention(
      model_dim=D,
      num_heads=H,
      head_dim=K,
      conv_kernel_dim=W,
      gate_lora_rank=R,
      gate_lower_bound=lower_bound,
      chunk_size=chunk_size,
      rms_norm_epsilon=RMS_EPS,
      activation_dtype=activation_dtype,
      sharding_config=_StubSharding() if sharded else None,
  )


@functools.lru_cache(maxsize=None)
def _apply_fn(
    chunk_size: int = 8,
    activation_dtype: str = 'float32',
    sharded: bool = False,
    lower_bound: float | None = LOWER_BOUND,
):
  """`layer.apply` under jit; eager dispatch dominates the test's runtime."""
  layer = _layer(chunk_size, activation_dtype, sharded, lower_bound)

  @jax.jit
  def fn(params, x, segment_ids, segment_positions, decode_state):
    return layer.apply(
        params,
        x,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        decode_state=decode_state,
    )

  return fn


_chunk_kda = jax.jit(kda.chunk_kda, static_argnames=('chunk_size',))
_recurrent_kda = jax.jit(kda.recurrent_kda)
_recurrent_kda_step = jax.jit(kda.recurrent_kda_step)
_log_decay = jax.jit(kda.kda_log_decay, static_argnames=('lower_bound',))


class _StubSharding:
  """The `config_lib.BaseSharding` fields KDA reads, without the dependency."""

  activation_partition = (('replica', 'data'), None, 'model')
  attn_activation_partition = (('replica', 'data'), None, 'model', None)
  attn_qkv_partition = ('data', 'model', None)
  ffn0_partition = ('data', 'model')


def _state_of(
    extra: dict[str, kda.KDADecodeState | None],
) -> kda.KDADecodeState:
  """The updated decode state of an `apply` call that carried one."""
  state = extra['decode_state']
  assert state is not None
  return state


def _max_diff(a, b) -> float:
  return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b))))


class KdaCoreTest(parameterized.TestCase):
  """The chunkwise core against the token recurrence."""

  def _core_inputs(self, t: int = T, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = _l2norm(rng.normal(size=(B, t, H, K))) * K**-0.5
    k = _l2norm(rng.normal(size=(B, t, H, K)))
    v = rng.normal(size=(B, t, H, K))
    g = _kda_gate(
        rng.normal(size=(B, t, H, K)),
        np.log(rng.uniform(1.0, 16.0, size=(H,))),
        rng.normal(size=(H, K)),
    )
    beta = _sigmoid(rng.normal(size=(B, t, H)))
    s0 = rng.normal(size=(B, H, K, K))
    return q, k, v, g, beta, s0

  @parameterized.parameters(4, 5, 8, 16)
  def test_chunkwise_matches_recurrent(self, chunk_size: int):
    q, k, v, g, beta, s0 = self._core_inputs()
    o_ref, s_ref = _kda_recurrent(q, k, v, g, beta, scale=1.0, initial_state=s0)
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    o, s = _chunk_kda(
        f32(q),
        f32(k),
        f32(v),
        f32(g),
        f32(beta),
        f32(s0),
        chunk_size=chunk_size,
    )
    o_seq, s_seq = _recurrent_kda(
        f32(q), f32(k), f32(v), f32(g), f32(beta), f32(s0)
    )
    with self.subTest('vs_numpy'):
      self.assertLess(_max_diff(o, o_ref), 1e-5)
      self.assertLess(_max_diff(s, s_ref), 1e-5)
    with self.subTest('vs_jax_recurrent'):
      self.assertLess(_max_diff(o, o_seq), 1e-5)
      self.assertLess(_max_diff(s, s_seq), 1e-5)
    with self.subTest('jax_recurrent_vs_numpy'):
      self.assertLess(_max_diff(o_seq, o_ref), 1e-5)
      self.assertLess(_max_diff(s_seq, s_ref), 1e-5)

  @parameterized.parameters(LOWER_BOUND, None)
  def test_log_decay_matches_reference(self, lower_bound: float | None):
    rng = np.random.default_rng(21)
    z = rng.normal(size=(B, T, H, K)) * 3.0
    a_log = np.log(rng.uniform(1.0, 16.0, size=(H,)))
    dt_bias = rng.normal(size=(H, K))
    g_ref = _kda_gate(z, a_log, dt_bias, lower_bound)
    g = _log_decay(
        jnp.asarray(z, jnp.float32),
        jnp.asarray(a_log, jnp.float32),
        jnp.asarray(dt_bias, jnp.float32),
        lower_bound=lower_bound,
    )
    self.assertLess(_max_diff(g, g_ref), 1e-5)
    self.assertTrue(bool(jnp.all(g <= 0.0)))
    if lower_bound is not None:
      # Saturating sigmoid() makes the bound attainable in f32, not just a
      # limit.
      self.assertTrue(bool(jnp.all(g >= lower_bound)))

  def test_chunkwise_without_initial_state(self):
    q, k, v, g, beta, _ = self._core_inputs(t=17, seed=1)
    o_ref, s_ref = _kda_recurrent(q, k, v, g, beta, scale=1.0)
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    o, s = _chunk_kda(f32(q), f32(k), f32(v), f32(g), f32(beta), chunk_size=8)
    self.assertLess(_max_diff(o, o_ref), 1e-5)
    self.assertLess(_max_diff(s, s_ref), 1e-5)

  def test_step_matches_recurrent(self):
    q, k, v, g, beta, s0 = self._core_inputs(t=3, seed=2)
    o_ref, s_ref = _kda_recurrent(q, k, v, g, beta, scale=1.0, initial_state=s0)
    s = jnp.asarray(s0, jnp.float32)
    f32 = lambda a, i: jnp.asarray(a[:, i], jnp.float32)
    outs = []
    for i in range(3):
      o_i, s = _recurrent_kda_step(
          f32(q, i), f32(k, i), f32(v, i), f32(g, i), f32(beta, i), s
      )
      outs.append(o_i)
    self.assertLess(_max_diff(jnp.stack(outs, axis=1), o_ref), 1e-5)
    self.assertLess(_max_diff(s, s_ref), 1e-5)

  def test_padding_is_an_identity_update(self):
    q, k, v, g, beta, s0 = self._core_inputs(t=9, seed=3)
    # Positions 5.. are padding: g = 0, beta = 0.
    g_pad, beta_pad = g.copy(), beta.copy()
    g_pad[:, 5:] = 0.0
    beta_pad[:, 5:] = 0.0
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    _, s = _chunk_kda(
        f32(q),
        f32(k),
        f32(v),
        f32(g_pad),
        f32(beta_pad),
        f32(s0),
        chunk_size=4,
    )
    _, s_short = _chunk_kda(
        f32(q[:, :5]),
        f32(k[:, :5]),
        f32(v[:, :5]),
        f32(g[:, :5]),
        f32(beta[:, :5]),
        f32(s0),
        chunk_size=4,
    )
    self.assertLess(_max_diff(s, s_short), 1e-5)

  @parameterized.parameters(
      ((6,),),  # mid-chunk
      ((4,),),  # on a chunk boundary
      ((0,),),  # the whole call is a fresh sequence
      ((3, 4, 9),),  # three segments, one boundary-aligned
  )
  def test_segment_starts_reset_state(self, starts: tuple[int, ...]):
    """Packed segments == the segments run one by one from a zero state."""
    t = 12
    q, k, v, g, beta, s0 = self._core_inputs(t=t, seed=4)
    start = np.zeros((B, t), bool)
    start[:, list(starts)] = True
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    o, s = _chunk_kda(
        f32(q),
        f32(k),
        f32(v),
        f32(g),
        f32(beta),
        f32(s0),
        chunk_size=4,
        segment_start=jnp.asarray(start),
    )
    edges = sorted({0, t, *starts})
    carried = None if 0 in starts else s0
    s_ref = s0  # the loop always runs; this keeps the type checker happy
    for lo, hi in zip(edges[:-1], edges[1:]):
      cut = lambda a: a[:, lo:hi]  # pylint: disable=cell-var-from-loop
      o_ref, s_ref = _kda_recurrent(
          cut(q), cut(k), cut(v), cut(g), cut(beta), 1.0, carried
      )
      self.assertLess(_max_diff(o[:, lo:hi], o_ref), 1e-5)
      carried = None  # every later edge is a reset
    self.assertLess(_max_diff(s, s_ref), 1e-5)


class KdaModuleTest(parameterized.TestCase):
  """The `KimiK3DeltaAttention` module."""

  def test_param_tree_shapes(self):
    layer = _layer()
    params = layer.init(jax.random.PRNGKey(0))
    # Leaves may be `AnnotatedArray`; stop the traversal at anything shaped.
    has_shape = lambda a: hasattr(a, 'shape')
    shapes = jax.tree.map(lambda a: tuple(a.shape), params, is_leaf=has_shape)
    self.assertEqual(
        shapes,
        {
            'q_proj': {'w': (D, H, K)},
            'k_proj': {'w': (D, H, K)},
            'v_proj': {'w': (D, H, K)},
            'q_conv': {'w': (H, K, W)},
            'k_conv': {'w': (H, K, W)},
            'v_conv': {'w': (H, K, W)},
            'f_a_proj': {'w': (D, R)},
            'f_b_proj': {'w': (R, H, K)},
            'dt_bias': (H, K),
            'a_log': (H,),
            'b_proj': {'w': (D, H)},
            'g_proj': {'w': (D, H, K)},
            'o_norm': {'scale': (K,)},
            'o_proj': {'w': (H, K, D)},
        },
    )
    dtypes = jax.tree.map(lambda a: a.dtype, params, is_leaf=has_shape)
    for path in ('q_conv', 'k_conv', 'v_conv'):
      self.assertEqual(dtypes[path]['w'], jnp.float32)
    for path in ('dt_bias', 'a_log'):
      self.assertEqual(dtypes[path], jnp.float32)
    self.assertEqual(dtypes['o_norm']['scale'], jnp.float32)

  @parameterized.parameters(4, 8, 64)
  def test_apply_matches_reference_layer(self, chunk_size: int):
    w = _random_weights()
    x = np.random.default_rng(7).normal(size=(B, T, D))
    y_ref, _, s_ref = _reference_layer(x, w)
    layer = _layer(chunk_size)
    y, extra = _apply_fn(chunk_size)(
        _params(w),
        jnp.asarray(x, jnp.float32),
        jnp.ones((B, T), jnp.int32),
        jnp.tile(jnp.arange(T), (B, 1)),
        layer.init_decode_state(B, T),
    )
    self.assertEqual(y.shape, (B, T, D))
    self.assertLess(_max_diff(y, y_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)

  def test_apply_without_decode_state(self):
    w = _random_weights()
    x = np.random.default_rng(7).normal(size=(B, T, D))
    y_ref, _, _ = _reference_layer(x, w)
    y, extra = _apply_fn()(
        _params(w),
        jnp.asarray(x, jnp.float32),
        jnp.ones((B, T), jnp.int32),
        jnp.tile(jnp.arange(T), (B, 1)),
        None,
    )
    self.assertIsNone(extra['decode_state'])
    self.assertLess(_max_diff(y, y_ref), ATOL)

  def test_prefill_then_decode_matches_full_prefill(self):
    w = _random_weights(seed=1)
    params = _params(w)
    x = np.random.default_rng(11).normal(size=(B, T, D))
    layer = _layer()
    apply_fn = _apply_fn()
    positions = jnp.tile(jnp.arange(T), (B, 1))
    y_full, extra_full = apply_fn(
        params,
        jnp.asarray(x, jnp.float32),
        jnp.ones((B, T), jnp.int32),
        positions,
        layer.init_decode_state(B, T),
    )

    prefill = 7
    y_pre, extra = apply_fn(
        params,
        jnp.asarray(x[:, :prefill], jnp.float32),
        jnp.ones((B, prefill), jnp.int32),
        positions[:, :prefill],
        layer.init_decode_state(B, T),
    )
    steps = [y_pre]
    for i in range(prefill, T):
      y_i, extra = apply_fn(
          params,
          jnp.asarray(x[:, i : i + 1], jnp.float32),
          jnp.ones((B, 1), jnp.int32),
          positions[:, i : i + 1],
          _state_of(extra),
      )
      steps.append(y_i)
    y_inc = jnp.concatenate(steps, axis=1)
    self.assertLess(_max_diff(y_inc, y_full), ATOL)
    self.assertLess(
        _max_diff(
            _state_of(extra).recurrent_state,
            _state_of(extra_full).recurrent_state,
        ),
        ATOL,
    )
    self.assertLess(
        _max_diff(
            _state_of(extra).conv_state,
            _state_of(extra_full).conv_state,
        ),
        ATOL,
    )
    np.testing.assert_array_equal(_state_of(extra).lengths, np.full((B,), T))

  def test_padding_does_not_corrupt_state(self):
    """A right-padded row equals the same row without the padding.

    The padded positions carry GARBAGE (a real batch embeds a pad token id, not
    zeros), so this exercises the input masking, the beta/gate masking and the
    conv taps rather than relying on zeros to be neutral.
    """
    w = _random_weights(seed=2)
    params = _params(w)
    n_valid = 9
    x = np.random.default_rng(13).normal(size=(B, T, D))
    x[:, n_valid:] = 7.0 * np.random.default_rng(14).normal(
        size=(B, T - n_valid, D)
    )
    segment_ids = np.ones((B, T), np.int32)
    segment_ids[:, n_valid:] = 0
    y, extra = _apply_fn(4)(
        params,
        jnp.asarray(x, jnp.float32),
        jnp.asarray(segment_ids),
        jnp.tile(jnp.arange(T), (B, 1)),
        _layer(4).init_decode_state(B, T),
    )
    y_ref, conv_ref, s_ref = _reference_layer(x[:, :n_valid], w)  # no pads
    self.assertLess(_max_diff(y[:, :n_valid], y_ref), ATOL)
    np.testing.assert_array_equal(np.asarray(y[:, n_valid:]), 0.0)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_ref), ATOL)
    np.testing.assert_array_equal(
        _state_of(extra).lengths, np.full((B,), n_valid)
    )

  def test_packed_segments_match_separate_sequences(self):
    """Two segments packed in one row == the two rows run on their own."""
    w = _random_weights(seed=3)
    params = _params(w)
    cut = 5
    x = np.random.default_rng(17).normal(size=(1, T, D))
    segment_ids = np.ones((1, T), np.int32)
    segment_ids[:, cut:] = 2
    positions = np.concatenate([np.arange(cut), np.arange(T - cut)], axis=0)[
        None
    ]
    y, extra = _apply_fn(4)(
        params,
        jnp.asarray(x, jnp.float32),
        jnp.asarray(segment_ids),
        jnp.asarray(positions),
        _layer(4).init_decode_state(1, T),
    )
    y_first, _, _ = _reference_layer(x[:, :cut], w)
    y_second, conv_second, s_second = _reference_layer(x[:, cut:], w)
    self.assertLess(_max_diff(y[:, :cut], y_first), ATOL)
    self.assertLess(_max_diff(y[:, cut:], y_second), ATOL)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_second), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_second), ATOL)
    np.testing.assert_array_equal(
        _state_of(extra).lengths, np.full((1,), T - cut)
    )

  def test_padding_between_packed_segments(self):
    """Interior padding is neutral for both the conv window and the state."""
    w = _random_weights(seed=6)
    first, gap, second = 4, 2, 4
    x = np.random.default_rng(29).normal(size=(1, T, D))
    segment_ids = np.zeros((1, T), np.int32)
    segment_ids[:, :first] = 1
    segment_ids[:, first + gap : first + gap + second] = 2
    positions = np.zeros((1, T), np.int32)
    positions[:, :first] = np.arange(first)
    positions[:, first + gap : first + gap + second] = np.arange(second)
    y, extra = _apply_fn(4)(
        _params(w),
        jnp.asarray(x, jnp.float32),
        jnp.asarray(segment_ids),
        jnp.asarray(positions),
        _layer(4).init_decode_state(1, T),
    )
    lo, hi = first + gap, first + gap + second
    y_first, _, _ = _reference_layer(x[:, :first], w)
    y_second, conv_second, s_second = _reference_layer(x[:, lo:hi], w)
    self.assertLess(_max_diff(y[:, :first], y_first), ATOL)
    self.assertLess(_max_diff(y[:, lo:hi], y_second), ATOL)
    np.testing.assert_array_equal(np.asarray(y[:, first:lo]), 0.0)
    np.testing.assert_array_equal(np.asarray(y[:, hi:]), 0.0)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_second), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_second), ATOL)

  def test_decode_step_starts_a_new_segment(self):
    """A decode token at position 0 restarts the state and the conv window."""
    w = _random_weights(seed=7)
    params = _params(w)
    apply_fn = _apply_fn()
    x = np.random.default_rng(31).normal(size=(B, T, D))
    _, extra = apply_fn(
        params,
        jnp.asarray(x[:, :-1], jnp.float32),
        jnp.ones((B, T - 1), jnp.int32),
        jnp.tile(jnp.arange(T - 1), (B, 1)),
        _layer().init_decode_state(B, T),
    )
    y, extra = apply_fn(
        params,
        jnp.asarray(x[:, -1:], jnp.float32),
        jnp.full((B, 1), 2, jnp.int32),
        jnp.zeros((B, 1), jnp.int32),
        _state_of(extra),
    )
    y_ref, conv_ref, s_ref = _reference_layer(x[:, -1:], w)
    self.assertLess(_max_diff(y, y_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_ref), ATOL)
    np.testing.assert_array_equal(_state_of(extra).lengths, np.full((B,), 1))

  def test_softplus_gate(self):
    """`gate_lower_bound=None` selects fla's unbounded softplus gate."""
    w = _random_weights(seed=8)
    x = np.random.default_rng(37).normal(size=(B, T, D))
    y_ref, _, _ = _reference_layer(x, w, lower_bound=None)
    y, _ = _apply_fn(lower_bound=None)(
        _params(w),
        jnp.asarray(x, jnp.float32),
        jnp.ones((B, T), jnp.int32),
        jnp.tile(jnp.arange(T), (B, 1)),
        _layer(lower_bound=None).init_decode_state(B, T),
    )
    self.assertLess(_max_diff(y, y_ref), ATOL)

  def test_rejects_an_invalid_gate_bound(self):
    with self.assertRaises(ValueError):
      _layer(lower_bound=1.0)

  def test_decode_state_shapes(self):
    layer = _layer()
    state = layer.init_decode_state(B, 128)
    self.assertEqual(state.conv_state.shape, (B, 3, H, K, W))
    self.assertEqual(state.recurrent_state.shape, (B, H, K, K))
    self.assertEqual(state.lengths.shape, (B,))
    self.assertEqual(state.recurrent_state.dtype, jnp.float32)
    leaves = jax.tree.leaves(state)
    self.assertLen(leaves, 3)

  def test_sharding_annotations_match_ranks(self):
    """`with_sharding_constraint` raises when an annotation has the wrong rank."""
    w = _random_weights(seed=4)
    layer = _layer(sharded=True)
    x = np.random.default_rng(23).normal(size=(B, T, D))
    y_ref, _, _ = _reference_layer(x, w)
    layer.init(jax.random.PRNGKey(0))
    y, extra = _apply_fn(sharded=True)(
        _params(w),
        jnp.asarray(x, jnp.float32),
        jnp.ones((B, T), jnp.int32),
        jnp.tile(jnp.arange(T), (B, 1)),
        layer.init_decode_state(B, T),
    )
    self.assertEqual(_state_of(extra).conv_state.shape, (B, 3, H, K, W))
    self.assertLess(_max_diff(y, y_ref), ATOL)

  def test_bfloat16_activations(self):
    w = _random_weights(seed=5)
    layer = _layer(activation_dtype='bfloat16')
    params = _params(w)
    x = np.random.default_rng(19).normal(size=(B, T, D))
    y, extra = _apply_fn(activation_dtype='bfloat16')(
        params,
        jnp.asarray(x, jnp.bfloat16),
        jnp.ones((B, T), jnp.int32),
        jnp.tile(jnp.arange(T), (B, 1)),
        layer.init_decode_state(B, T),
    )
    self.assertEqual(y.dtype, jnp.bfloat16)
    self.assertEqual(_state_of(extra).recurrent_state.dtype, jnp.float32)
    y_ref, _, _ = _reference_layer(x, w)
    # bf16 projections: only a sanity bound, the numerics are pinned in f32.
    self.assertLess(_max_diff(y, y_ref), 0.2 * float(np.abs(y_ref).max()))

  @parameterized.parameters('float32', 'bfloat16')
  def test_o_norm_applies_the_gain_in_f32(self, activation_dtype: str):
    """fla `FusedRMSNormGated`, not `KimiRMSNorm`: no rounding before the gate.

    Fails if `o_norm` is ever built with the module's `activation_dtype`, which
    would round the normalized output to bf16 before the sigmoid gate.

    Args:
      activation_dtype: dtype of the surrounding layer's projections.
    """
    rng = np.random.default_rng(31)
    o = rng.normal(size=(B, T, H, K))
    gate = rng.normal(size=(B, T, H, K))
    scale = rng.normal(1.0, 0.1, size=(K,))
    o_norm = _layer(activation_dtype=activation_dtype).o_norm
    normed = o_norm.apply(
        {'scale': jnp.asarray(scale, jnp.float32)}, jnp.asarray(o, jnp.float32)
    )
    got = normed * jax.nn.sigmoid(jnp.asarray(gate, jnp.float32))
    self.assertLess(_max_diff(got, _rms_norm_gated(o, gate, scale)), ATOL)

  def test_o_norm_matches_the_plain_f32_rms_norm(self):
    """`o_norm` is exactly `x * rsqrt(mean(x^2) + eps) * scale`, all in f32."""
    rng = np.random.default_rng(37)
    x = jnp.asarray(rng.normal(size=(B, T, H, K)), jnp.float32)
    scale = jnp.asarray(rng.normal(1.0, 0.1, size=(K,)), jnp.float32)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True) + RMS_EPS
    )
    got = _layer().o_norm.apply({'scale': scale}, x)
    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(x * inv_rms * scale)
    )


if __name__ == '__main__':
  absltest.main()

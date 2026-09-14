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
"""Tests for Qwen3.8's gated grouped-query attention.

The oracle is the `# --- HF reference (float64) ---` section: one NumPy
function per HF function of
`transformers/v5/models/qwen3_5/modeling_qwen3_5.py` (`Qwen3_5RMSNorm`,
`Qwen3_5TextRotaryEmbedding`, `rotate_half`, `apply_rotary_pos_emb`,
`repeat_kv`, `eager_attention_forward`, `Qwen3_5Attention.forward`),
transcribed from the release and evaluated in float64. The rotary part is
deliberately re-transcribed here rather than shared with `rope_test.py`, so
this file is an oracle for the whole layer and not just for the parts
`rope.py` does not own.

HF stores `nn.Linear` weights transposed relative to this module
(`[out, in]` vs `[in, heads, dim]`) and keeps heads on axis 1; the reference
contracts the Simply layout directly and keeps heads on axis 2.

`_reference_attention` takes the three switches the A/B tests need
(`gate_activation`, `use_qk_norm`, `soft_cap`), each of which turns the
reference into a model Qwen3.8 is *not*: a silu gate is the GatedDeltaNet's,
dropping the head norms is Qwen2, and a 50.0 soft cap is core's default.
"""

from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply import model_lib
from simply.utils import common
from simply.utils import sharding as sharding_lib
from simply.zoo.qwen3p8.utils import attn as attn_lib
from simply.zoo.qwen3p8.utils import rope as rope_lib

# `replica, data, model` over 4 CPU devices. On one device
# `with_sharding_constraint` only checks the rank (`utils/sharding.py:253`), so
# a wrong axis is invisible; on this mesh every annotated axis is really split,
# and every dimension the layer shards (batch 2, model_dim 32, 4 q heads, 2 KV
# heads, 16 channels per head) divides by 2, so a wrong axis fails on the
# placement rather than on divisibility.
_MESH_DEVICE_COUNT = 4
_MESH_SHAPE = (1, 2, 2)
_P = jax.sharding.PartitionSpec


def setUpModule():
  # Must run before the backend initializes, i.e. not inside a test.
  jax.config.update('jax_num_cpu_devices', _MESH_DEVICE_COUNT)


# Small but non-degenerate: 2 query heads per KV head, and a rotary fraction
# that leaves half of every head unrotated with all three mRoPE rows in use.
_BATCH, _SEQ_LEN = 2, 6
_MODEL_DIM, _HEADS, _KV_HEADS, _HEAD_DIM = 32, 4, 2, 16
_ROTARY_FRACTION = 0.5
_MROPE_SECTION = (2, 1, 1)
_MAX_TIMESCALE = 10_000_000
_RMS_EPS = 1e-6

# float32 layer vs the float64 oracle. The chain is projection, RMSNorm,
# rotary, softmax and two more matmuls over 32 channels; measured max
# deviation for these inputs is 4.6e-6, and 1e-4 keeps a 20x margin.
_ATOL = 1e-4
# bfloat16 keeps 8 mantissa bits, so every intermediate carries ~4e-3 relative
# error, which the 64-term `o_proj` contraction accumulates. Bounded relative
# to the peak output (~27 here): measured 1.2e-1, i.e. a ratio of 4.6e-3, so
# this leaves a 6x margin.
_RTOL_BF16 = 3e-2

# --- HF reference (float64) -------------------------------------------------


def _rms_norm(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
  """HF `Qwen3_5RMSNorm.forward`: normalize, then scale by `1 + w`."""
  variance = np.mean(np.square(x), axis=-1, keepdims=True)
  return x / np.sqrt(variance + _RMS_EPS) * (1.0 + scale)


def _rotary_embedding(
    position_ids: np.ndarray, head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
  """HF `Qwen3_5TextRotaryEmbedding.forward` (interleaved mRoPE)."""
  dim = int(head_dim * _ROTARY_FRACTION)
  inv_freq = 1.0 / (
      _MAX_TIMESCALE ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
  )
  freqs = position_ids[..., None].astype(np.float64) * inv_freq
  freqs_t = np.array(freqs[0])
  for row, offset in enumerate((1, 2), start=1):
    idx = slice(offset, _MROPE_SECTION[row] * 3, 3)
    freqs_t[..., idx] = freqs[row, ..., idx]
  emb = np.concatenate([freqs_t, freqs_t], axis=-1)
  return np.cos(emb), np.sin(emb)


def _apply_rotary_pos_emb(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> np.ndarray:
  """HF `apply_rotary_pos_emb` + `rotate_half`, heads on axis 2."""
  cos, sin = cos[:, :, None, :], sin[:, :, None, :]
  rotary_dim = cos.shape[-1]
  x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
  half = rotary_dim // 2
  rotated = np.concatenate([-x_rot[..., half:], x_rot[..., :half]], axis=-1)
  return np.concatenate([x_rot * cos + rotated * sin, x_pass], axis=-1)


def _repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
  """HF `repeat_kv`: KV head `j` serves query heads `j*n_rep .. j*n_rep+n-1`."""
  return np.repeat(x, n_rep, axis=2)


def _reference_attention(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    position_ids: np.ndarray,
    mask: np.ndarray,
    *,
    gate_activation: str = 'sigmoid',
    use_qk_norm: bool = True,
    soft_cap: float = 0.0,
    interleaved_groups: bool = False,
) -> np.ndarray:
  """NumPy `Qwen3_5Attention.forward`; `mask` is bool `[B, T, S]`.

  Args:
    x: `[B, T, D]` float64 inputs.
    weights: The Simply param leaves as float64 arrays.
    position_ids: `[3, B, T]` mRoPE position rows.
    mask: `[B, T, S]` bool; True is attendable.
    gate_activation: 'sigmoid' is Qwen3.8; 'silu' is the wrong gate.
    use_qk_norm: False drops the per-head q/k RMSNorms.
    soft_cap: > 0 applies core `model_lib.attn`'s default cap, which HF has
      not got.
    interleaved_groups: True binds query head `i` to KV head `i % H_kv`, the
      wrong GQA grouping.

  Returns:
    `[B, T, D]` float64 layer output.
  """
  q_and_gate = np.einsum('btd,dhe->bthe', x, weights['q_proj'])
  q = q_and_gate[..., :_HEAD_DIM]
  gate = q_and_gate[..., _HEAD_DIM:].reshape(x.shape[0], x.shape[1], -1)
  k = np.einsum('btd,dhe->bthe', x, weights['k_proj'])
  v = np.einsum('btd,dhe->bthe', x, weights['v_proj'])

  if use_qk_norm:
    q = _rms_norm(q, weights['q_norm'])
    k = _rms_norm(k, weights['k_norm'])
  cos, sin = _rotary_embedding(position_ids, _HEAD_DIM)
  q = _apply_rotary_pos_emb(q, cos, sin)
  k = _apply_rotary_pos_emb(k, cos, sin)

  n_rep = _HEADS // _KV_HEADS
  if interleaved_groups:
    k, v = np.tile(k, (1, 1, n_rep, 1)), np.tile(v, (1, 1, n_rep, 1))
  else:
    k, v = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
  scores = np.einsum('bthd,bshd->bhts', q, k) * _HEAD_DIM**-0.5
  if soft_cap > 0:
    scores = soft_cap * np.tanh(scores / soft_cap)
  scores = np.where(mask[:, None], scores, -np.inf)
  scores = scores - np.max(scores, axis=-1, keepdims=True)
  probs = np.exp(scores)
  probs /= np.sum(probs, axis=-1, keepdims=True)
  context = np.einsum('bhts,bshd->bthd', probs, v)

  context = context.reshape(x.shape[0], x.shape[1], -1)
  if gate_activation == 'silu':
    context = context * gate / (1.0 + np.exp(-gate))
  else:
    context = context / (1.0 + np.exp(-gate))
  context = context.reshape(x.shape[0], x.shape[1], _HEADS, _HEAD_DIM)
  return np.einsum('dhe,bthe->btd', weights['o_proj'], context)


# --- Helpers ----------------------------------------------------------------


class _StubSharding:
  """The `config_lib.BaseSharding` fields this layer reads, without the dep."""

  activation_partition = (('replica', 'data'), None, 'model')
  attn_activation_partition = (('replica', 'data'), None, 'model', None)
  attn_qkv_partition = ('data', 'model', None)
  attn_o_partition = ('data', 'model', None)


def _layer(**overrides) -> attn_lib.Qwen38Attention:
  kwargs: dict[str, Any] = dict(
      model_dim=_MODEL_DIM,
      n_heads=_HEADS,
      n_kv_heads=_KV_HEADS,
      per_head_dim=_HEAD_DIM,
      rms_norm_epsilon=_RMS_EPS,
      activation_dtype='float32',
      position_encoding=rope_lib.Qwen38RoPE(
          rotary_fraction=_ROTARY_FRACTION,
          mrope_section=_MROPE_SECTION,
          max_timescale=_MAX_TIMESCALE,
      ),
  )
  kwargs.update(overrides)
  return attn_lib.Qwen38Attention(**kwargs)


def _random_params(
    layer: attn_lib.Qwen38Attention, seed: int = 0
) -> tuple[Any, dict[str, np.ndarray]]:
  """Returns `(params, the same leaves as float64 numpy)`."""
  params = layer.init(jax.random.PRNGKey(seed))
  leaves, treedef = jax.tree.flatten(params)
  keys = jax.random.split(jax.random.PRNGKey(seed + 1), len(leaves))
  params = jax.tree.unflatten(
      treedef,
      [
          0.5 * jax.random.normal(key, leaf.shape, jnp.float32)
          for key, leaf in zip(keys, leaves)
      ],
  )
  raw: Any = common.get_raw_arrays(params)
  weights = {
      name: np.asarray(
          subtree['w'] if 'w' in subtree else subtree['scale'], np.float64
      )
      for name, subtree in raw.items()
  }
  return params, weights


def _inputs(seq_len: int = _SEQ_LEN, seed: int = 7) -> np.ndarray:
  return np.asarray(
      jax.random.normal(
          jax.random.PRNGKey(seed), (_BATCH, seq_len, _MODEL_DIM)
      ),
      np.float64,
  )


def _positions(batch: int, seq_len: int) -> jax.Array:
  return jnp.broadcast_to(
      jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len)
  )


def _reference_positions(positions: jax.Array | np.ndarray) -> np.ndarray:
  """`[B, T]` Simply positions -> the `[3, B, T]` HF wants (pads clamp to 0)."""
  positions = np.maximum(np.asarray(positions), 0)
  return np.broadcast_to(positions[None], (3,) + positions.shape)


def _causal_mask(
    segment_ids: np.ndarray, positions: np.ndarray | None = None
) -> np.ndarray:
  """`[B, T, S]` causal mask restricted to matching segments, as core does."""
  if positions is None:
    positions = np.cumsum(np.ones_like(segment_ids), axis=-1) - 1
  causal = positions[:, :, None] >= positions[:, None, :]
  return causal & (segment_ids[:, :, None] == segment_ids[:, None, :])


def _spec(x) -> jax.sharding.PartitionSpec:
  """The `PartitionSpec` of a concrete array; `.sharding` is typed too wide."""
  sharding = x.sharding
  assert isinstance(sharding, jax.sharding.NamedSharding), sharding
  return sharding.spec


def _max_diff(a, b) -> float:
  delta = float(
      np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
  )
  logging.info('max |delta|: %.3e', delta)
  return delta


def _state_of(extra: dict[str, Any]) -> dict[str, Any]:
  state = extra['decode_state']
  assert state is not None
  return state


class AttentionTest(parameterized.TestCase):
  """`Qwen38Attention` against the float64 HF reference."""

  def _prefill(self, layer, params, x, segment_ids=None, positions=None):
    """Stateless full-sequence call with all-valid, 0..T-1 defaults."""
    seq_len = x.shape[1]
    if segment_ids is None:
      segment_ids = jnp.ones((_BATCH, seq_len), jnp.int32)
    if positions is None:
      positions = _positions(_BATCH, seq_len)
    return layer.apply(
        params,
        jnp.asarray(x, jnp.float32),
        segment_ids=segment_ids,
        segment_positions=positions,
    )

  def test_param_tree_matches_contract(self):
    params = common.get_raw_arrays(_layer().init(jax.random.PRNGKey(0)))

    self.assertEqual(
        jax.tree.map(lambda a: tuple(a.shape), params),
        {
            'q_proj': {'w': (32, 4, 32)},
            'k_proj': {'w': (32, 2, 16)},
            'v_proj': {'w': (32, 2, 16)},
            'o_proj': {'w': (32, 4, 16)},
            'q_norm': {'scale': (16,)},
            'k_norm': {'scale': (16,)},
        },
    )

  def test_released_defaults_are_the_released_values(self):
    """`Qwen38Block` builds this with the config's dims and nothing else."""
    layer = attn_lib.Qwen38Attention(model_dim=5120)

    self.assertEqual(
        (
            layer.n_heads,
            layer.n_kv_heads,
            layer.per_head_dim,
            layer.attn_output_gate,
            layer.use_qk_norm,
            layer.rms_norm_epsilon,
            layer.activation_dtype,
        ),
        (24, 4, 256, True, True, 1e-6, 'bfloat16'),
    )
    self.assertIsInstance(layer.position_encoding, rope_lib.Qwen38RoPE)

  def test_released_geometry_param_shapes(self):
    """The 27B's own dims: 24 q / 4 kv heads of 256 over a 5120 residual."""
    layer = attn_lib.Qwen38Attention(model_dim=5120)

    shapes = jax.tree.map(
        lambda a: tuple(a.shape),
        common.get_raw_arrays(
            jax.eval_shape(layer.init, jax.random.PRNGKey(0))
        ),
    )
    self.assertEqual(
        shapes,
        {
            'q_proj': {'w': (5120, 24, 512)},
            'k_proj': {'w': (5120, 4, 256)},
            'v_proj': {'w': (5120, 4, 256)},
            'o_proj': {'w': (5120, 24, 256)},
            'q_norm': {'scale': (256,)},
            'k_norm': {'scale': (256,)},
        },
    )

  def test_prefill_matches_hf_reference(self):
    layer = _layer()
    params, weights = _random_params(layer)
    x = _inputs()

    out, extra = self._prefill(layer, params, x)

    self.assertIsNone(extra['decode_state'])
    self.assertEqual(out.shape, (_BATCH, _SEQ_LEN, _MODEL_DIM))
    expected = _reference_attention(
        x,
        weights,
        _reference_positions(_positions(_BATCH, _SEQ_LEN)),
        _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32)),
    )
    self.assertLess(_max_diff(out, expected), _ATOL)

  def test_output_gate_is_sigmoid_not_silu(self):
    """`output_gate_type: "swish"` names the GDN gate, not this one."""
    layer = _layer()
    params, weights = _random_params(layer)
    x = _inputs()
    reference = lambda **kw: _reference_attention(
        x,
        weights,
        _reference_positions(_positions(_BATCH, _SEQ_LEN)),
        _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        **kw,
    )

    out, _ = self._prefill(layer, params, x)

    silu = reference(gate_activation='silu')
    with self.subTest('the_two_gates_are_far_apart'):
      self.assertGreater(_max_diff(silu, reference()), 100 * _ATOL)
    with self.subTest('sigmoid'):
      self.assertLess(_max_diff(out, reference()), _ATOL)
    with self.subTest('not_silu'):
      self.assertGreater(_max_diff(out, silu), 100 * _ATOL)

  def test_q_and_k_are_head_normed(self):
    layer = _layer()
    params, weights = _random_params(layer)
    x = _inputs()
    reference = lambda **kw: _reference_attention(
        x,
        weights,
        _reference_positions(_positions(_BATCH, _SEQ_LEN)),
        _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        **kw,
    )

    out, _ = self._prefill(layer, params, x)

    unnormed = reference(use_qk_norm=False)
    with self.subTest('the_norm_changes_the_answer'):
      self.assertGreater(_max_diff(unnormed, reference()), 100 * _ATOL)
    with self.subTest('normed'):
      self.assertLess(_max_diff(out, reference()), _ATOL)
    with self.subTest('not_unnormed'):
      self.assertGreater(_max_diff(out, unnormed), 100 * _ATOL)

  def test_logits_are_not_soft_capped(self):
    """Core `attn` caps at 50.0 unless told not to; HF never caps."""
    layer = _layer()
    params, weights = _random_params(layer)
    x = 3.0 * _inputs()
    reference = lambda **kw: _reference_attention(
        x,
        weights,
        _reference_positions(_positions(_BATCH, _SEQ_LEN)),
        _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        **kw,
    )

    out, _ = self._prefill(layer, params, x)

    capped = reference(soft_cap=50.0)
    with self.subTest('the_cap_would_be_visible'):
      self.assertGreater(_max_diff(capped, reference()), 10 * _ATOL)
    with self.subTest('uncapped'):
      self.assertLess(_max_diff(out, reference()), _ATOL)

  def test_kv_heads_serve_contiguous_query_head_groups(self):
    """GQA grouping is HF `repeat_kv`, not an interleave."""
    layer = _layer()
    params, weights = _random_params(layer)
    x = _inputs()
    reference = lambda **kw: _reference_attention(
        x,
        weights,
        _reference_positions(_positions(_BATCH, _SEQ_LEN)),
        _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        **kw,
    )

    out, _ = self._prefill(layer, params, x)

    interleaved = reference(interleaved_groups=True)
    with self.subTest('the_groupings_differ'):
      self.assertGreater(_max_diff(interleaved, reference()), 100 * _ATOL)
    with self.subTest('contiguous'):
      self.assertLess(_max_diff(out, reference()), _ATOL)
    with self.subTest('not_interleaved'):
      self.assertGreater(_max_diff(out, interleaved), 100 * _ATOL)

  def test_packed_segments_do_not_attend_across(self):
    layer = _layer()
    params, weights = _random_params(layer)
    x = _inputs()
    segment_ids = np.asarray([[1, 1, 1, 2, 2, 2]] * _BATCH, np.int32)
    positions = np.asarray([[0, 1, 2, 0, 1, 2]] * _BATCH, np.int32)

    out, _ = self._prefill(
        layer,
        params,
        x,
        segment_ids=jnp.asarray(segment_ids),
        positions=jnp.asarray(positions),
    )

    expected = _reference_attention(
        x,
        weights,
        _reference_positions(positions),
        _causal_mask(segment_ids, positions),
    )
    self.assertLess(_max_diff(out, expected), _ATOL)

  def test_padding_does_not_reach_valid_queries(self):
    """`segment_ids == 0` rows are masked out however loud they are."""
    layer = _layer()
    params, _ = _random_params(layer)
    valid_len = 4
    x = _inputs()
    padded = x.copy()
    padded[:, valid_len:] = 50.0  # Garbage the pad rows must not leak.
    segment_ids = np.zeros((_BATCH, _SEQ_LEN), np.int32)
    segment_ids[:, :valid_len] = 1

    padded_out, _ = self._prefill(
        layer, params, padded, segment_ids=jnp.asarray(segment_ids)
    )
    short_out, _ = self._prefill(layer, params, x[:, :valid_len])

    self.assertLess(_max_diff(padded_out[:, :valid_len], short_out), _ATOL)

  def test_cached_decode_matches_prefill(self):
    """Token-by-token decode reproduces the full-sequence prefill."""
    layer = _layer()
    params, weights = _random_params(layer)
    x = jnp.asarray(_inputs(), jnp.float32)
    prefill_out, _ = self._prefill(layer, params, x)

    state = layer.init_decode_state(_BATCH, _SEQ_LEN)
    steps = []
    for t in range(_SEQ_LEN):
      step_out, extra = layer.apply(
          params,
          x[:, t : t + 1],
          segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
          segment_positions=jnp.full((_BATCH, 1), t, jnp.int32),
          decode_state=state,
      )
      state = _state_of(extra)
      steps.append(step_out)
    decoded = jnp.concatenate(steps, axis=1)

    with self.subTest('vs_prefill'):
      self.assertLess(_max_diff(decoded, prefill_out), _ATOL)
    with self.subTest('vs_reference'):
      expected = _reference_attention(
          _inputs(),
          weights,
          _reference_positions(_positions(_BATCH, _SEQ_LEN)),
          _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32)),
      )
      self.assertLess(_max_diff(decoded, expected), _ATOL)

  def test_prefill_then_padded_decode_matches_prefill(self):
    """The real driver path: prefill, `pad_decode_state_to`, then a step."""
    layer = _layer()
    params, _ = _random_params(layer)
    x = jnp.asarray(_inputs(), jnp.float32)
    prefill_out, _ = self._prefill(layer, params, x)
    chunk = 4

    _, extra = layer.apply(
        params,
        x[:, :chunk],
        segment_ids=jnp.ones((_BATCH, chunk), jnp.int32),
        segment_positions=_positions(_BATCH, chunk),
        decode_state=layer.init_decode_state(_BATCH, chunk),
        extra_inputs={'prefill_position': chunk},
    )
    grown: Any = model_lib.pad_decode_state_to(
        {'block_0': _state_of(extra)}, _SEQ_LEN
    )
    step_out, _ = layer.apply(
        params,
        x[:, chunk : chunk + 1],
        segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
        segment_positions=jnp.full((_BATCH, 1), chunk, jnp.int32),
        decode_state=grown['block_0'],
    )

    with self.subTest('grown_cache_shapes'):
      self.assertEqual(
          grown['block_0']['k'].shape,
          (_BATCH, _SEQ_LEN, _KV_HEADS, _HEAD_DIM),
      )
      self.assertEqual(
          grown['block_0']['segment_ids'].shape, (_BATCH, _SEQ_LEN)
      )
    with self.subTest('step_matches_prefill'):
      self.assertLess(_max_diff(step_out[:, 0], prefill_out[:, chunk]), _ATOL)

  def test_decode_state_is_cores_mapping_cache(self):
    layer = _layer()
    state = layer.init_decode_state(_BATCH, _SEQ_LEN)

    with self.subTest('keys'):
      self.assertEqual(
          set(state),
          {'k', 'v', 'segment_positions', 'segment_ids', 'window_size=0'},
      )
    with self.subTest('shapes'):
      self.assertEqual(
          state['k'].shape, (_BATCH, _SEQ_LEN, _KV_HEADS, _HEAD_DIM)
      )
      self.assertEqual(state['v'].shape, state['k'].shape)
      self.assertEqual(state['segment_ids'].shape, (_BATCH, _SEQ_LEN))
    with self.subTest('pad_decode_state_to_grows_every_buffer'):
      grown: Any = model_lib.pad_decode_state_to(
          {'block_0': state}, 3 * _SEQ_LEN
      )
      self.assertEqual(
          grown['block_0']['k'].shape,
          (_BATCH, 3 * _SEQ_LEN, _KV_HEADS, _HEAD_DIM),
      )
      self.assertEqual(
          grown['block_0']['segment_positions'].shape,
          (_BATCH, 3 * _SEQ_LEN),
      )

  def test_update_kv_cache_false_leaves_the_cache(self):
    layer = _layer()
    params, _ = _random_params(layer)
    x = jnp.asarray(_inputs(), jnp.float32)
    state = layer.init_decode_state(_BATCH, _SEQ_LEN)

    _, extra = layer.apply(
        params,
        x[:, :1],
        segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
        segment_positions=jnp.zeros((_BATCH, 1), jnp.int32),
        extra_inputs={'update_kv_cache': False},
        decode_state=state,
    )

    np.testing.assert_array_equal(
        np.asarray(_state_of(extra)['k']), np.asarray(state['k'])
    )

  def test_a_fully_masked_query_is_finite(self):
    """`_MASK_VALUE` is finite, so an all-masked row is uniform, not NaN."""
    layer = _layer()
    params, _ = _random_params(layer)
    x = jnp.asarray(_inputs(), jnp.float32)
    # A cache the query cannot match (`segment_ids` 2 vs 1) and cannot extend,
    # holding non-zero values: with `-inf` the softmax is NaN and `NaN @ v`
    # poisons the row.
    state = layer.init_decode_state(_BATCH, _SEQ_LEN)
    state = dict(state)
    state['v'] = jnp.ones_like(state['v'])
    state['segment_ids'] = jnp.full((_BATCH, _SEQ_LEN), 2, jnp.int32)

    out, _ = layer.apply(
        params,
        x[:, :1],
        segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
        segment_positions=jnp.zeros((_BATCH, 1), jnp.int32),
        extra_inputs={'update_kv_cache': False},
        decode_state=state,
    )

    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_apply_does_not_mutate_the_caller_state(self):
    layer = _layer()
    params, _ = _random_params(layer)
    x = jnp.asarray(_inputs(), jnp.float32)
    state = layer.init_decode_state(_BATCH, _SEQ_LEN)

    layer.apply(
        params,
        x,
        segment_ids=jnp.ones((_BATCH, _SEQ_LEN), jnp.int32),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
        extra_inputs={'prefill_position': _SEQ_LEN},
        decode_state=state,
    )

    self.assertNotIn('prefill_position', state)

  def test_head_norms_do_not_round_before_the_scale(self):
    """HF `Qwen3_5RMSNorm` scales in float32 and rounds once, at the end."""
    layer = _layer(activation_dtype='bfloat16')
    params, weights = _random_params(layer)
    x = jnp.asarray(
        jax.random.normal(jax.random.PRNGKey(5), (1, 1, 1, _HEAD_DIM)),
        jnp.float32,
    )

    out = layer.q_norm.apply(params['q_norm'], x)

    expected = _rms_norm(np.asarray(x, np.float64), weights['q_norm'])
    # Rounding `x_normed` or `1 + w` to bfloat16 first would cost ~4e-3.
    self.assertEqual(out.dtype, jnp.float32)
    self.assertLess(_max_diff(out, expected), 1e-6)
    # ... and the float32 must not leak: `LayerNorm.apply` hands back the
    # caller's dtype, which is what keeps q/k and the whole KV cache bfloat16
    # (a multi-token call *replaces* the cache buffers with these tensors).
    self.assertEqual(
        layer.q_norm.apply(
            params['q_norm'], jnp.asarray(x, jnp.bfloat16)
        ).dtype,
        jnp.bfloat16,
    )

  def test_bfloat16_tracks_the_float32_layer(self):
    params, _ = _random_params(_layer())
    x = jnp.asarray(_inputs(), jnp.float32)
    expected, _ = self._prefill(_layer(), params, x)
    layer = _layer(activation_dtype='bfloat16')

    out, _ = self._prefill(layer, params, x)
    step, extra = layer.apply(
        params,
        x[:, :1],
        segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
        segment_positions=jnp.zeros((_BATCH, 1), jnp.int32),
        decode_state=layer.init_decode_state(_BATCH, _SEQ_LEN),
    )

    with self.subTest('dtypes'):
      self.assertEqual(out.dtype, jnp.bfloat16)
      self.assertEqual(_state_of(extra)['k'].dtype, jnp.bfloat16)
    peak = float(jnp.max(jnp.abs(expected)))
    logging.info('bfloat16 peak output %.3e', peak)
    with self.subTest('prefill'):
      self.assertLess(_max_diff(out, expected), _RTOL_BF16 * peak)
    with self.subTest('decode_step'):
      self.assertLess(_max_diff(step, expected[:, :1]), _RTOL_BF16 * peak)

  def test_sharded_execution_matches_the_unsharded_layer(self):
    """The annotations must fit their tensors and not change the answer."""
    params, _ = _random_params(_layer())
    x = jnp.asarray(_inputs(), jnp.float32)
    expected, _ = self._prefill(_layer(), params, x)
    sharded = _layer(sharding_config=_StubSharding())

    with sharding_lib.set_mesh(_MESH_SHAPE):
      out, _ = self._prefill(sharded, params, x)
      step, _ = sharded.apply(
          params,
          x[:, :1],
          segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
          segment_positions=jnp.zeros((_BATCH, 1), jnp.int32),
          decode_state=sharded.init_decode_state(_BATCH, _SEQ_LEN),
      )

    with self.subTest('prefill'):
      self.assertLess(_max_diff(out, expected), _ATOL)
    with self.subTest('decode_step'):
      self.assertLess(_max_diff(step, expected[:, :1]), _ATOL)

  def test_nope_variant_drops_the_rotary(self):
    """`position_encoding=None` is the one branch a config may switch off."""
    layer = _layer(position_encoding=None)
    params, _ = _random_params(layer)
    x = _inputs()

    out, _ = self._prefill(layer, params, x)
    shifted, _ = self._prefill(
        layer,
        params,
        x,
        positions=_positions(_BATCH, _SEQ_LEN) + 100,
    )

    self.assertLess(_max_diff(out, shifted), _ATOL)

  @parameterized.named_parameters(
      ('ungated', dict(attn_output_gate=False)),
      ('unnormed', dict(use_qk_norm=False)),
      ('heads_not_a_multiple_of_kv_heads', dict(n_heads=3, n_kv_heads=2)),
  )
  def test_rejects_an_unported_variant(self, overrides):
    with self.assertRaises(ValueError):
      _layer(**overrides)

  def test_decode_step_without_positions_is_an_error(self):
    """The rotary refuses to guess a cached step's absolute position."""
    layer = _layer()
    params, _ = _random_params(layer)
    x = jnp.asarray(_inputs(seq_len=1), jnp.float32)
    # `apply` declares positions required; this checks the runtime guard
    # behind that annotation, so the None has to be smuggled past pytype.
    no_positions: Any = None

    with self.assertRaisesRegex(ValueError, 'seq_len == 1'):
      layer.apply(
          params,
          x,
          segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
          segment_positions=no_positions,
      )


class ShardingTest(parameterized.TestCase):
  """Where each axis actually lands, on a mesh that can tell them apart.

  `test_sharded_execution_matches_the_unsharded_layer` only proves the
  annotations fit their tensors: a wrong axis gives the same numbers. These
  pin the layout itself -- the two behavioural cases through real device
  placement, the third through the annotations that never reach an array.
  """

  def test_cache_and_output_land_on_the_documented_axes(self):
    layer = _layer(sharding_config=_StubSharding())
    x = jnp.asarray(_inputs(), jnp.float32)

    # Eagerly, not under `jit`: XLA normalizes a jitted output's sharding
    # (it drops the size-1 `replica` axis and collapses trailing entries), so
    # a jitted read encodes XLA's rewrite instead of the module's request.
    with sharding_lib.set_mesh(_MESH_SHAPE):
      params = layer.init(jax.random.PRNGKey(0))
      state = layer.init_decode_state(_BATCH, _SEQ_LEN)
      out, _ = layer.apply(
          params,
          x,
          segment_ids=jnp.ones((_BATCH, _SEQ_LEN), jnp.int32),
          segment_positions=_positions(_BATCH, _SEQ_LEN),
      )

    batch = ('replica', 'data')
    with self.subTest('kv_cache_shards_per_head_dim_not_the_kv_heads'):
      # `[B, S, H_kv, D]`: `model` on D, so a mesh wider than `n_kv_heads`
      # still splits the cache. Undoing the entry shift puts it on H_kv.
      self.assertEqual(_spec(state['k']), _P(batch, None, None, 'model'))
      self.assertEqual(_spec(state['v']), _spec(state['k']))
    with self.subTest('output_shards_the_residual_stream'):
      self.assertEqual(_spec(out), _P(batch, None, 'model'))
    with self.subTest('kv_weights_shard_per_head_dim_too'):
      # Or every K/V projection would end in a reshard.
      raw: Any = common.get_raw_arrays(params)
      self.assertEqual(_spec(raw['k_proj']['w']), _P('data', None, 'model'))
      self.assertEqual(_spec(raw['q_proj']['w']), _P('data', 'model', None))

  def test_annotations_that_never_reach_an_array(self):
    layer = _layer(sharding_config=_StubSharding())

    with self.subTest('projection_outputs_are_left_alone'):
      # `None` here is a *replicate* constraint, not "unannotated": it would
      # all-gather every projection output for `apply` to reshard it back.
      for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        self.assertIs(
            getattr(layer, name).output_partition, sharding_lib.NOT_ANNOTATED
        )
    # pylint: disable=protected-access
    with self.subTest('activation_annotations'):
      self.assertEqual(
          layer._activation_partition('q'),
          (('replica', 'data'), None, 'model', None),
      )
      self.assertEqual(
          layer._activation_partition('kv'),
          (('replica', 'data'), None, None, 'model'),
      )
      self.assertEqual(
          layer._activation_partition('out'),
          (('replica', 'data'), None, 'model'),
      )
    with self.subTest('unsharded_layers_annotate_nothing'):
      plain = _layer()
      self.assertIs(
          plain._activation_partition('q'), sharding_lib.NOT_ANNOTATED
      )
      self.assertIsNone(plain._kv_cache_partition())
    # pylint: enable=protected-access


if __name__ == '__main__':
  absltest.main()

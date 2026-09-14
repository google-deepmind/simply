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
"""Tests for the Kimi K3 gated NoPE MLA layer.

The oracle is `_reference_mla`, a NumPy transcription of HF
`KimiMLAAttention.forward` + `eager_attention_forward`
(`modeling_kimi_linear.py`) with `mla_use_nope=True` and
`mla_use_output_gate=True`. HF stores `nn.Linear` weights transposed relative
to this module ([out, in] vs [in, ...]), so the reference contracts the Simply
layout directly.
"""

from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import common
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3.utils import mla as mla_lib

_EPSILON = 1e-5
_ATOL = 2e-5

_DIMS = dict(
    model_dim=32,
    n_heads=2,
    q_lora_rank=16,
    kv_lora_rank=8,
    qk_nope_head_dim=8,
    qk_rope_head_dim=4,
    v_head_dim=8,
)
_BATCH = 2
_SEQ_LEN = 6


def _rms_norm(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
  """HF `KimiRMSNorm.forward`."""
  variance = np.mean(np.square(x), axis=-1, keepdims=True)
  return x / np.sqrt(variance + _EPSILON) * scale


def _reference_mla(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    mask: np.ndarray,
    use_output_gate: bool = True,
    logit_soft_cap: float = 0.0,
) -> np.ndarray:
  """NumPy `KimiMLAAttention.forward`; `mask` is bool `[B, T, S]`."""
  dims = _DIMS
  q_head_dim = dims['qk_nope_head_dim'] + dims['qk_rope_head_dim']

  q = _rms_norm(x @ weights['q_a_proj'], weights['q_a_norm'])
  q = np.einsum('btr,rhd->bthd', q, weights['q_b_proj'])
  q_nope = q[..., : dims['qk_nope_head_dim']]
  q_rope = q[..., dims['qk_nope_head_dim'] :]

  kv = x @ weights['kv_a_proj']
  latent = _rms_norm(kv[..., : dims['kv_lora_rank']], weights['kv_a_norm'])
  k_rope = kv[..., dims['kv_lora_rank'] :]
  k_nope_v = np.einsum('bsr,rhd->bshd', latent, weights['kv_b_proj'])
  k_nope = k_nope_v[..., : dims['qk_nope_head_dim']]
  v = k_nope_v[..., dims['qk_nope_head_dim'] :]
  # The k_rope block is shared by all heads (MQA-style broadcast) and, being
  # NoPE, is never rotated.
  k = np.concatenate(
      [
          k_nope,
          np.broadcast_to(
              k_rope[:, :, None],
              k_nope.shape[:-1] + (dims['qk_rope_head_dim'],),
          ),
      ],
      axis=-1,
  )

  scores = np.einsum('bthd,bshd->bhts', np.concatenate([q_nope, q_rope], -1), k)
  scores = scores * q_head_dim**-0.5
  # Not K3 math: the `model_lib.attn` default the absorbed core switches off,
  # here only so a test can measure what leaving it on would cost.
  if logit_soft_cap > 0:
    scores = logit_soft_cap * np.tanh(scores / logit_soft_cap)
  scores = np.where(mask[:, None], scores, -np.inf)
  scores = scores - np.max(scores, axis=-1, keepdims=True)
  probs = np.exp(scores)
  probs = probs / np.sum(probs, axis=-1, keepdims=True)
  context = np.einsum('bhts,bshd->bthd', probs, v)

  if use_output_gate:
    gate = np.einsum('bti,ihd->bthd', x, weights['g_proj'])
    context = context * (1.0 / (1.0 + np.exp(-gate)))
  return np.einsum('bthd,hdi->bti', context, weights['o_proj'])


def _causal_mask(segment_ids: np.ndarray) -> np.ndarray:
  """`[B, T, S]` causal mask restricted to matching non-pad segments."""
  positions = np.cumsum(np.ones_like(segment_ids), axis=-1) - 1
  causal = positions[:, :, None] >= positions[:, None, :]
  same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
  return causal & same_segment


def _build(**overrides) -> mla_lib.KimiK3MLA:
  kwargs = dict(
      **_DIMS,
      rms_norm_epsilon=_EPSILON,
      activation_dtype='float32',
      weight_dtype='float32',
  )
  kwargs.update(overrides)
  return mla_lib.KimiK3MLA(**kwargs)


def _random_params(mla: mla_lib.KimiK3MLA, seed: int = 0):
  """Returns `(params, numpy weights keyed by param name)`."""
  params = mla.init(jax.random.PRNGKey(seed))
  leaves, treedef = jax.tree.flatten(params)
  keys = jax.random.split(jax.random.PRNGKey(seed + 1), len(leaves))
  params = jax.tree.unflatten(
      treedef,
      [
          0.5 * jax.random.normal(k, leaf.shape, jnp.float32)
          for k, leaf in zip(keys, leaves)
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


def _assert_close(
    actual, expected, atol: float = _ATOL, rtol: float = 0.0, name: str = ''
) -> None:
  """`assert_allclose` that reports the achieved deviation in the test log."""
  delta = float(
      np.max(
          np.abs(
              np.asarray(actual, np.float64) - np.asarray(expected, np.float64)
          )
      )
  )
  logging.info('max |delta| %s: %.3e (atol %.1e)', name, delta, atol)
  np.testing.assert_allclose(
      np.asarray(actual, np.float64),
      np.asarray(expected, np.float64),
      atol=atol,
      rtol=rtol,
  )


def _segment_ids(mask: np.ndarray) -> jax.Array:
  return jnp.asarray(mask.astype(np.int32))


def _positions(batch: int, seq_len: int) -> jax.Array:
  return jnp.broadcast_to(
      jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len)
  )


class MlaTest(parameterized.TestCase):

  def _inputs(self, seq_len: int = _SEQ_LEN, seed: int = 7):
    x = np.asarray(
        jax.random.normal(jax.random.PRNGKey(seed), (_BATCH, seq_len, 32)),
        np.float64,
    )
    return x

  def test_param_tree_matches_contract(self):
    mla = _build()
    params = common.get_raw_arrays(mla.init(jax.random.PRNGKey(0)))
    shapes = jax.tree.map(lambda a: tuple(a.shape), params)
    self.assertEqual(
        shapes,
        {
            'q_a_proj': {'w': (32, 16)},
            'q_a_norm': {'scale': (16,)},
            'q_b_proj': {'w': (16, 2, 12)},
            'kv_a_proj': {'w': (32, 12)},
            'kv_a_norm': {'scale': (8,)},
            'kv_b_proj': {'w': (8, 2, 16)},
            'g_proj': {'w': (32, 2, 8)},
            'o_proj': {'w': (2, 8, 32)},
        },
    )

  def test_released_config_param_shapes(self):
    """Shapes at the real K3 dims, straight from the experiment config."""
    config = k3_config_lib.KimiK3ExperimentConfig()
    mla = mla_lib.KimiK3MLA(
        model_dim=config.model_dim,
        n_heads=config.n_heads,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        use_output_gate=config.mla_use_output_gate,
        rms_norm_epsilon=config.rms_norm_epsilon,
    )
    shapes = jax.tree.map(
        lambda a: tuple(a.shape),
        common.get_raw_arrays(jax.eval_shape(mla.init, jax.random.PRNGKey(0))),
    )
    self.assertEqual(
        shapes,
        {
            'q_a_proj': {'w': (7168, 1536)},
            'q_a_norm': {'scale': (1536,)},
            'q_b_proj': {'w': (1536, 96, 192)},
            'kv_a_proj': {'w': (7168, 576)},
            'kv_a_norm': {'scale': (512,)},
            'kv_b_proj': {'w': (512, 96, 256)},
            'g_proj': {'w': (7168, 96, 128)},
            'o_proj': {'w': (96, 128, 7168)},
        },
    )

  @parameterized.named_parameters(('gated', True), ('ungated', False))
  def test_matches_numpy_reference(self, use_output_gate: bool):
    mla = _build(use_output_gate=use_output_gate)
    params, weights = _random_params(mla)
    x = self._inputs()
    valid = np.ones((_BATCH, _SEQ_LEN), np.int32)

    out, extra = mla.apply(
        params,
        jnp.asarray(x, jnp.float32),
        segment_ids=_segment_ids(valid),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )
    expected = _reference_mla(
        x, weights, _causal_mask(valid), use_output_gate=use_output_gate
    )

    self.assertIsNone(extra['decode_state'])
    self.assertEqual(out.shape, (_BATCH, _SEQ_LEN, 32))
    _assert_close(out, expected, name=f'reference (gate={use_output_gate})')

  def test_ungated_module_has_no_gate_params(self):
    params = _build(use_output_gate=False).init(jax.random.PRNGKey(0))
    self.assertNotIn('g_proj', params)

  def test_output_is_independent_of_future_tokens(self):
    mla = _build()
    params, _ = _random_params(mla)
    x = self._inputs()
    perturbed = x.copy()
    perturbed[:, 4:] += 3.0
    kwargs = dict(
        segment_ids=_segment_ids(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )

    out, _ = mla.apply(params, jnp.asarray(x, jnp.float32), **kwargs)
    out_perturbed, _ = mla.apply(
        params, jnp.asarray(perturbed, jnp.float32), **kwargs
    )

    _assert_close(out[:, :4], out_perturbed[:, :4], atol=1e-6, name='causal')
    self.assertGreater(
        float(jnp.max(jnp.abs(out[:, 4:] - out_perturbed[:, 4:]))), 1e-3
    )

  def test_padding_does_not_change_valid_outputs(self):
    mla = _build()
    params, _ = _random_params(mla)
    short_len = 4
    x = self._inputs()
    x_padded = x.copy()
    x_padded[:, short_len:] = 17.0  # Garbage the pad rows must not leak.
    valid = np.zeros((_BATCH, _SEQ_LEN), np.int32)
    valid[:, :short_len] = 1

    padded_out, _ = mla.apply(
        params,
        jnp.asarray(x_padded, jnp.float32),
        segment_ids=_segment_ids(valid),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )
    short_out, _ = mla.apply(
        params,
        jnp.asarray(x[:, :short_len], jnp.float32),
        segment_ids=_segment_ids(np.ones((_BATCH, short_len), np.int32)),
        segment_positions=_positions(_BATCH, short_len),
    )

    _assert_close(padded_out[:, :short_len], short_out, name='padding')

  def test_packed_segments_do_not_attend_across(self):
    mla = _build()
    params, weights = _random_params(mla)
    x = self._inputs()
    segment_ids = np.asarray([[1, 1, 1, 2, 2, 2]] * _BATCH, np.int32)
    positions = np.asarray([[0, 1, 2, 0, 1, 2]] * _BATCH, np.int32)

    out, _ = mla.apply(
        params,
        jnp.asarray(x, jnp.float32),
        segment_ids=jnp.asarray(segment_ids),
        segment_positions=jnp.asarray(positions),
    )
    expected = _reference_mla(x, weights, _causal_mask(segment_ids))

    _assert_close(out, expected, name='packed segments')

  @parameterized.named_parameters(('absorbed', True), ('decompressed', False))
  def test_cached_decode_matches_prefill(self, use_absorbed_decode: bool):
    """Token-by-token decode reproduces the full-sequence prefill."""
    mla = _build(use_absorbed_decode=use_absorbed_decode)
    params, weights = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    prefill_out, _ = mla.apply(
        params,
        x,
        segment_ids=_segment_ids(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )

    state = mla.init_decode_state(_BATCH, _SEQ_LEN)
    steps = []
    for t in range(_SEQ_LEN):
      step_out, extra = mla.apply(
          params,
          x[:, t : t + 1],
          segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
          segment_positions=jnp.full((_BATCH, 1), t, jnp.int32),
          decode_state=state,
      )
      state = extra['decode_state']
      steps.append(step_out)
      np.testing.assert_array_equal(state.lengths, np.full((_BATCH,), t + 1))

    decoded = jnp.concatenate(steps, axis=1)
    _assert_close(
        decoded, prefill_out, name=f'decode (absorbed={use_absorbed_decode})'
    )
    # Against the HF oracle directly, not just against our own prefill.
    _assert_close(
        decoded,
        _reference_mla(
            self._inputs(),
            weights,
            _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        ),
        name=f'decode vs reference (absorbed={use_absorbed_decode})',
    )

  def test_absorbed_core_does_not_soft_cap_logits(self):
    """`model_lib.attn` caps logits at 50.0 unless told not to; K3 never caps."""
    mla = _build()
    params, weights = _random_params(mla)
    x = self._inputs()
    mask = _causal_mask(np.ones((_BATCH, _SEQ_LEN), np.int32))

    # A full-length chunk into an empty cache runs the absorbed core.
    out, _ = mla.apply(
        params,
        jnp.asarray(x, jnp.float32),
        segment_ids=jnp.ones((_BATCH, _SEQ_LEN), jnp.int32),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
        decode_state=mla.init_decode_state(_BATCH, _SEQ_LEN),
    )

    uncapped = _reference_mla(x, weights, mask)
    capped = _reference_mla(x, weights, mask, logit_soft_cap=50.0)
    # Without this the check below would pass with the cap left on; at these
    # logits the cap is worth ~80x the tolerance.
    self.assertGreater(float(np.max(np.abs(capped - uncapped))), 10 * _ATOL)
    _assert_close(out, uncapped, name='absorbed core, uncapped')

  def test_chunked_prefill_then_decode_matches_prefill(self):
    """A cached chunk followed by decode steps reproduces one long prefill."""
    mla = _build()
    params, _ = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    prefill_out, _ = mla.apply(
        params,
        x,
        segment_ids=_segment_ids(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )

    chunk = 4
    state = mla.init_decode_state(_BATCH, _SEQ_LEN)
    chunk_out, extra = mla.apply(
        params,
        x[:, :chunk],
        segment_ids=jnp.ones((_BATCH, chunk), jnp.int32),
        segment_positions=_positions(_BATCH, chunk),
        decode_state=state,
    )
    state = extra['decode_state']
    outs = [chunk_out]
    for t in range(chunk, _SEQ_LEN):
      step_out, extra = mla.apply(
          params,
          x[:, t : t + 1],
          segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
          segment_positions=jnp.full((_BATCH, 1), t, jnp.int32),
          decode_state=state,
      )
      state = extra['decode_state']
      outs.append(step_out)

    _assert_close(
        jnp.concatenate(outs, axis=1),
        prefill_out,
        name='chunked prefill + decode',
    )

  def test_segment_restart_reuses_the_cache_slot(self):
    """`segment_positions == 0` on the first valid token starts a new sequence."""
    mla = _build()
    params, _ = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    first, second = x[:, :3], x[:, 3:]
    ones = jnp.ones((_BATCH, 3), jnp.int32)

    fresh, _ = mla.apply(
        params,
        second,
        segment_ids=ones,
        segment_positions=_positions(_BATCH, 3),
    )
    _, extra = mla.apply(
        params,
        first,
        segment_ids=ones,
        segment_positions=_positions(_BATCH, 3),
        decode_state=mla.init_decode_state(_BATCH, _SEQ_LEN),
    )
    reused, extra = mla.apply(
        params,
        second,
        segment_ids=ones,
        segment_positions=_positions(_BATCH, 3),
        decode_state=extra['decode_state'],
    )

    np.testing.assert_array_equal(extra['decode_state'].lengths, [3, 3])
    _assert_close(reused, fresh, name='segment restart')

  def test_left_padded_chunk_lands_compactly(self):
    """Pads anywhere in the chunk are skipped, not just right-pads."""
    mla = _build()
    params, _ = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    padded = jnp.concatenate([jnp.full((_BATCH, 2, 32), 9.0), x[:, :2]], axis=1)
    valid = np.asarray([[0, 0, 1, 1]] * _BATCH, np.int32)

    out, extra = mla.apply(
        params,
        padded,
        segment_ids=_segment_ids(valid),
        segment_positions=jnp.asarray([[0, 0, 0, 1]] * _BATCH, jnp.int32),
        decode_state=mla.init_decode_state(_BATCH, _SEQ_LEN),
    )
    exact, _ = mla.apply(
        params,
        x[:, :2],
        segment_ids=jnp.ones((_BATCH, 2), jnp.int32),
        segment_positions=_positions(_BATCH, 2),
    )

    np.testing.assert_array_equal(extra['decode_state'].lengths, [2, 2])
    np.testing.assert_array_equal(
        np.asarray(out[:, :2]), np.zeros((_BATCH, 2, 32))
    )
    _assert_close(out[:, 2:], exact, name='left padding')

  def test_update_kv_cache_false_leaves_the_cache(self):
    mla = _build()
    params, _ = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    state = mla.init_decode_state(_BATCH, _SEQ_LEN)

    _, extra = mla.apply(
        params,
        x[:, :1],
        segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
        segment_positions=jnp.ones((_BATCH, 1), jnp.int32),
        extra_inputs={'update_kv_cache': False},
        decode_state=state,
    )

    np.testing.assert_array_equal(extra['decode_state'].lengths, state.lengths)
    np.testing.assert_array_equal(
        np.asarray(extra['decode_state'].kv_cache), np.asarray(state.kv_cache)
    )

  def test_cache_overflow_saturates(self):
    mla = _build()
    params, _ = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    state = mla.init_decode_state(_BATCH, 4)
    kwargs = dict(
        segment_ids=jnp.ones((_BATCH, 3), jnp.int32),
        segment_positions=jnp.full((_BATCH, 3), 1, jnp.int32),
    )

    _, extra = mla.apply(params, x[:, :3], decode_state=state, **kwargs)
    out, extra = mla.apply(
        params, x[:, 3:], decode_state=extra['decode_state'], **kwargs
    )

    np.testing.assert_array_equal(extra['decode_state'].lengths, [4, 4])
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    with self.assertRaises(ValueError):
      mla.apply(
          params,
          x[:, :5],
          segment_ids=jnp.ones((_BATCH, 5), jnp.int32),
          segment_positions=_positions(_BATCH, 5),
          decode_state=mla.init_decode_state(_BATCH, 4),
      )

  def test_decode_scan_under_jit_matches_prefill(self):
    """The decode state is a fixed point of a jitted `lax.scan` loop."""
    mla = _build()
    params, _ = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    prefill_out, _ = mla.apply(
        params,
        x,
        segment_ids=_segment_ids(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )

    @jax.jit
    def decode(params, x, state):
      def step(state, x_t):
        out, extra = mla.apply(
            params,
            x_t[:, None],
            segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
            segment_positions=state.lengths[:, None],
            decode_state=state,
        )
        return extra['decode_state'], out[:, 0]

      return jax.lax.scan(step, state, jnp.swapaxes(x, 0, 1))

    state, outs = decode(params, x, mla.init_decode_state(_BATCH, _SEQ_LEN))

    np.testing.assert_array_equal(state.lengths, [_SEQ_LEN] * _BATCH)
    _assert_close(jnp.swapaxes(outs, 0, 1), prefill_out, name='scanned decode')

  def test_bfloat16_tracks_the_float32_module(self):
    params, _ = _random_params(_build())
    x = jnp.asarray(self._inputs(), jnp.float32)
    kwargs = dict(
        segment_ids=_segment_ids(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )

    expected, _ = _build().apply(params, x, **kwargs)
    mla_bf16 = _build(activation_dtype='bfloat16')
    out, _ = mla_bf16.apply(params, x, **kwargs)
    step, extra = mla_bf16.apply(
        params,
        x[:, :1],
        segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
        segment_positions=jnp.zeros((_BATCH, 1), jnp.int32),
        decode_state=mla_bf16.init_decode_state(_BATCH, _SEQ_LEN),
    )

    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(extra['decode_state'].kv_cache.dtype, jnp.bfloat16)
    _assert_close(out, expected, atol=2e-2, rtol=2e-2, name='bfloat16 prefill')
    _assert_close(
        step, expected[:, :1], atol=2e-2, rtol=2e-2, name='bfloat16 decode'
    )

  def test_padded_chunk_is_cached_compactly(self):
    """Pad rows are neither written nor counted, so decode stays contiguous."""
    mla = _build()
    params, _ = _random_params(mla)
    x = jnp.asarray(self._inputs(), jnp.float32)
    valid = np.ones((_BATCH, 4), np.int32)
    valid[1, 2:] = 0  # Row 1 carries two right-pads.

    _, extra = mla.apply(
        params,
        x[:, :4],
        segment_ids=_segment_ids(valid),
        segment_positions=_positions(_BATCH, 4),
        decode_state=mla.init_decode_state(_BATCH, _SEQ_LEN),
    )
    state = extra['decode_state']

    np.testing.assert_array_equal(state.lengths, np.asarray([4, 2]))
    # Untouched slots stay zero: rows 2..5 of the padded sequence.
    np.testing.assert_array_equal(
        np.asarray(state.kv_cache[1, 2:]), np.zeros((4, 12), np.float32)
    )

  def test_sharding_annotations_have_matching_ranks(self):
    """Every `with_sharding_constraint` must match its tensor's rank."""
    sharding_config = k3_config_lib.KimiK3ExperimentConfig().sharding_config
    sharded = _build(sharding_config=sharding_config)
    plain = _build()
    params, _ = _random_params(plain)
    x = jnp.asarray(self._inputs(), jnp.float32)
    kwargs = dict(
        segment_ids=_segment_ids(np.ones((_BATCH, _SEQ_LEN), np.int32)),
        segment_positions=_positions(_BATCH, _SEQ_LEN),
    )

    expected, _ = plain.apply(params, x, **kwargs)
    out, _ = sharded.apply(params, x, **kwargs)
    step, _ = sharded.apply(
        params,
        x[:, :1],
        segment_ids=jnp.ones((_BATCH, 1), jnp.int32),
        segment_positions=jnp.zeros((_BATCH, 1), jnp.int32),
        decode_state=sharded.init_decode_state(_BATCH, _SEQ_LEN),
    )

    _assert_close(out, expected, name='sharded prefill')
    _assert_close(step, expected[:, :1], name='sharded decode step')

  def test_decode_state_is_a_pytree(self):
    state = _build().init_decode_state(_BATCH, _SEQ_LEN)
    leaves = jax.tree.leaves(state)
    self.assertLen(leaves, 2)
    self.assertEqual(leaves[0].shape, (_BATCH, _SEQ_LEN, 12))
    doubled = jax.tree.map(lambda a: a * 2, state)
    self.assertIsInstance(doubled, mla_lib.KimiK3MLADecodeState)


if __name__ == '__main__':
  absltest.main()

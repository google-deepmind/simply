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
"""Tests for Qwen3.8's interleaved partial mRoPE.

The oracle is the `# --- HF reference (float64) ---` section: one NumPy
function per HF function of
`transformers/v5/models/qwen3_5/modeling_qwen3_5.py`, transcribed from the
release rather than derived from `rope.py`, and evaluated in float64. HF keeps
heads on axis 1 (`[B, H, T, D]`) and unsqueezes `cos` at axis 1; Simply keeps
them on axis 2, so the reference unsqueezes at axis 2 instead. That is the only
edit.
"""

import dataclasses

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import position_encoding as pe_lib
from simply.zoo.qwen3p8.utils import rope as rope_lib

# Small but non-degenerate: 3 sections over 8 rotated channels of a 16-wide
# head, so half of every head passes through unrotated and each of the three
# position rows owns at least one channel.
_BATCH, _SEQ_LEN, _HEADS, _HEAD_DIM = 2, 5, 3, 16
_ROTARY_FRACTION = 0.5
_MROPE_SECTION = (2, 1, 1)
_MAX_TIMESCALE = 10_000_000

# The released geometry, used where a test pins the layout of the real model.
_RELEASED_HEAD_DIM = 256
_RELEASED_FRACTION = 0.25
_RELEASED_SECTION = (11, 11, 10)

# float32 rotation vs the float64 oracle. The error is dominated by rounding
# the angle, so it grows with the position: at the positions these cases use
# (<= 100 rad) the measured max is 2.4e-7 and 2e-6 keeps an 8x margin. The
# long-context case has its own, larger bound; see
# `test_positions_beyond_the_window_match_hf`.
_ATOL_F32 = 2e-6
# bfloat16 carries 8 mantissa bits, so a unit-scale output is quantized at
# 2^-8 = 3.9e-3 and the rotation mixes two such terms. Measured max deviation
# for these inputs (|x| <= 4.5) is 7.7e-3.
_ATOL_BF16 = 2e-2
# `test_cos_sin_stay_in_float32`: how much of HF's rounding error the float32
# cos/sin must remove. Measured 3.80e-3 against 5.78e-3, a factor of 0.66; HF's
# order scores exactly `theirs` here, so the bound only has to sit below 1.
_COS_SIN_ADVANTAGE = 0.85
# `test_released_geometry_at_the_last_trained_position`: at position 262143 the
# second-fastest channel's angle is 9.6e4 rad, one float32 ulp of which is
# 7.8e-3 -- HF rounds identically (it also builds `freqs` in float32), so this
# bounds the shared floor rather than a divergence. Measured 9.2e-3.
_ATOL_LAST_POSITION = 2e-2

# --- HF reference (float64) -------------------------------------------------


def _inv_freq(
    head_dim: int, rotary_fraction: float, max_timescale: float
) -> np.ndarray:
  """HF `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`."""
  dim = int(head_dim * rotary_fraction)
  return 1.0 / (
      max_timescale ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
  )


def _apply_interleaved_mrope(
    freqs: np.ndarray, mrope_section: tuple[int, ...]
) -> np.ndarray:
  """HF `Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope`."""
  freqs_t = np.array(freqs[0])
  for dim, offset in enumerate((1, 2), start=1):
    length = mrope_section[dim] * 3
    idx = slice(offset, length, 3)
    freqs_t[..., idx] = freqs[dim, ..., idx]
  return freqs_t


def _rotary_embedding(
    position_ids: np.ndarray,
    head_dim: int,
    rotary_fraction: float,
    mrope_section: tuple[int, ...],
    max_timescale: float,
) -> tuple[np.ndarray, np.ndarray]:
  """HF `Qwen3_5TextRotaryEmbedding.forward`; `position_ids` is `[3, B, T]`."""
  inv_freq = _inv_freq(head_dim, rotary_fraction, max_timescale)
  inv_freq_expanded = inv_freq[None, None, :, None]
  position_ids_expanded = position_ids[:, :, None, :].astype(np.float64)
  freqs = np.swapaxes(inv_freq_expanded @ position_ids_expanded, 2, 3)
  freqs = _apply_interleaved_mrope(freqs, mrope_section)
  emb = np.concatenate([freqs, freqs], axis=-1)
  return np.cos(emb), np.sin(emb)


def _rotate_half(x: np.ndarray) -> np.ndarray:
  """HF `rotate_half`."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return np.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> np.ndarray:
  """HF `apply_rotary_pos_emb` for one tensor, with `unsqueeze_dim=2`."""
  cos = cos[:, :, None, :]
  sin = sin[:, :, None, :]
  rotary_dim = cos.shape[-1]
  x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
  return np.concatenate([x_rot * cos + _rotate_half(x_rot) * sin, x_pass], -1)


def _reference_rope(
    x: np.ndarray,
    position_ids: np.ndarray,
    *,
    rotary_fraction: float = _ROTARY_FRACTION,
    mrope_section: tuple[int, ...] = _MROPE_SECTION,
    max_timescale: float = _MAX_TIMESCALE,
) -> np.ndarray:
  """The two HF calls a Qwen3.5 attention layer makes, in float64.

  Args:
    x: `[B, T, H, D]` queries or keys.
    position_ids: `[3, B, T]` mRoPE position rows.
    rotary_fraction: HF `partial_rotary_factor`.
    mrope_section: HF `mrope_section`.
    max_timescale: HF `rope_theta`.

  Returns:
    `[B, T, H, D]` float64.
  """
  cos, sin = _rotary_embedding(
      position_ids, x.shape[-1], rotary_fraction, mrope_section, max_timescale
  )
  return _apply_rotary_pos_emb(np.asarray(x, np.float64), cos, sin)


# --- Helpers ----------------------------------------------------------------


def _rope(**overrides) -> rope_lib.Qwen38RoPE:
  kwargs = dict(
      rotary_fraction=_ROTARY_FRACTION,
      mrope_section=_MROPE_SECTION,
      max_timescale=_MAX_TIMESCALE,
  )
  kwargs.update(overrides)
  return rope_lib.Qwen38RoPE(**kwargs)


def _inputs(seq_len: int = _SEQ_LEN, seed: int = 0) -> np.ndarray:
  return np.asarray(
      jax.random.normal(
          jax.random.PRNGKey(seed), (_BATCH, seq_len, _HEADS, _HEAD_DIM)
      ),
      np.float64,
  )


def _broadcast_rows(position_ids: np.ndarray) -> np.ndarray:
  """`[B, T]` -> the `[3, B, T]` the HF reference wants."""
  return np.broadcast_to(position_ids[None], (3,) + position_ids.shape)


def _max_diff(a, b) -> float:
  """Deviation between two arrays, reported in the test log."""
  delta = float(
      np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
  )
  logging.info('max |delta|: %.3e', delta)
  return delta


class RopeCoreTest(parameterized.TestCase):
  """The free functions, against the HF layout they transcribe."""

  def test_released_geometry_rotates_a_quarter_of_the_head(self):
    width = rope_lib.rotary_width(
        _RELEASED_HEAD_DIM, _RELEASED_FRACTION
    )
    inv_freq = rope_lib.inverse_frequencies(
        width, min_timescale=1, max_timescale=_MAX_TIMESCALE
    )

    self.assertEqual(width, 64)
    self.assertEqual(inv_freq.shape, (32,))
    # (11, 11, 10) tiles the module's own frequency count exactly once.
    self.assertEqual(sum(_RELEASED_SECTION), inv_freq.shape[0])

  def test_inverse_frequencies_match_hf(self):
    width = rope_lib.rotary_width(
        _RELEASED_HEAD_DIM, _RELEASED_FRACTION
    )
    inv_freq = rope_lib.inverse_frequencies(
        width, min_timescale=1, max_timescale=_MAX_TIMESCALE
    )

    expected = _inv_freq(
        _RELEASED_HEAD_DIM, _RELEASED_FRACTION, _MAX_TIMESCALE
    )
    self.assertLess(_max_diff(inv_freq, expected), 1e-7)

  @parameterized.named_parameters(
      ('tiny', _MROPE_SECTION, 8),
      ('released', _RELEASED_SECTION, 32),
      ('section_wider_than_the_rotary_width', (11, 11, 10), 4),
  )
  def test_sections_own_the_channels_hf_gives_them(self, section, channels):
    """Each channel reads the position row the interleaved layout assigns."""
    # One position row per value 0, 1, 2 and a unit frequency, so the angle of
    # channel j *is* the index of the row it came from.
    positions = jnp.asarray(
        np.broadcast_to(np.arange(3.0)[:, None, None], (3, 1, 1))
    )
    angles = rope_lib.interleaved_mrope_freqs(
        positions, jnp.ones((channels,), jnp.float32), section
    )

    rows = np.asarray(angles).reshape(-1).round().astype(int)
    expected = np.zeros((channels,), int)
    for row, offset in ((1, 1), (2, 2)):
      expected[offset : section[row] * 3 : 3] = row
    np.testing.assert_array_equal(rows, expected)

  def test_rotate_half_matches_hf(self):
    x = np.arange(12.0).reshape(1, 1, 1, 12)
    np.testing.assert_array_equal(
        np.asarray(rope_lib.rotate_half(jnp.asarray(x))), _rotate_half(x)
    )

  @parameterized.named_parameters(
      ('zero', 0.0), ('too_small', 1 / 32), ('above_one', 1.5)
  )
  def test_rotary_width_rejects_an_unusable_fraction(self, fraction):
    with self.assertRaises(ValueError):
      rope_lib.rotary_width(_HEAD_DIM, fraction)

  def test_positions_are_required_for_a_single_token(self):
    with self.assertRaisesRegex(ValueError, 'seq_len == 1'):
      rope_lib.position_rows(None, 1)

  def test_position_rows_clamp_and_broadcast(self):
    rows = rope_lib.position_rows(jnp.asarray([[-1, 0, 7]]), 3)

    self.assertEqual(rows.shape, (3, 1, 3))
    np.testing.assert_array_equal(
        np.asarray(rows), np.broadcast_to([[0.0, 0.0, 7.0]], (3, 1, 3))
    )

  def test_position_rows_rejects_a_bad_rank(self):
    with self.assertRaisesRegex(ValueError, '1D, 2D, or 3D'):
      rope_lib.position_rows(jnp.zeros((2, 2, 3, 4)), 3)


class Qwen38RopeTest(parameterized.TestCase):
  """`Qwen38RoPE.apply`, against the float64 HF reference."""

  @parameterized.named_parameters(
      ('float32', jnp.float32, _ATOL_F32),
      ('bfloat16', jnp.bfloat16, _ATOL_BF16),
  )
  def test_matches_hf_reference(self, dtype, atol):
    x = _inputs()
    positions = np.tile(np.arange(_SEQ_LEN), (_BATCH, 1))

    out = _rope().apply(
        jnp.asarray(x, dtype), segment_positions=jnp.asarray(positions)
    )

    self.assertEqual(out.dtype, dtype)
    expected = _reference_rope(x, _broadcast_rows(positions))
    self.assertLess(_max_diff(out, expected), atol)

  def test_three_position_rows_interleave(self):
    """Distinct t/h/w rows: the sections must pick them apart, not average."""
    x = _inputs()
    rows = np.stack([
        np.tile(np.arange(_SEQ_LEN), (_BATCH, 1)),
        np.tile(np.arange(_SEQ_LEN) + 100, (_BATCH, 1)),
        np.tile(np.arange(_SEQ_LEN) * 3, (_BATCH, 1)),
    ])

    out = _rope().apply(
        jnp.asarray(x, jnp.float32), segment_positions=jnp.asarray(rows)
    )

    expected = _reference_rope(x, rows)
    with self.subTest('vs_reference'):
      self.assertLess(_max_diff(out, expected), _ATOL_F32)
    with self.subTest('differs_from_the_time_row_alone'):
      time_only = _reference_rope(x, _broadcast_rows(rows[0]))
      self.assertGreater(_max_diff(expected, time_only), 0.1)

  def test_one_row_equals_three_identical_rows(self):
    x = jnp.asarray(_inputs(), jnp.float32)
    positions = np.tile(np.arange(_SEQ_LEN) + 4, (_BATCH, 1))

    broadcast = _rope().apply(x, segment_positions=jnp.asarray(positions))
    explicit = _rope().apply(
        x, segment_positions=jnp.asarray(_broadcast_rows(positions))
    )

    np.testing.assert_array_equal(np.asarray(broadcast), np.asarray(explicit))

  def test_partial_rotary_leaves_the_tail_untouched(self):
    x = jnp.asarray(_inputs(), jnp.float32)
    positions = jnp.asarray(np.tile(np.arange(_SEQ_LEN) + 1, (_BATCH, 1)))
    width = rope_lib.rotary_width(_HEAD_DIM, _ROTARY_FRACTION)

    out = _rope().apply(x, segment_positions=positions)

    with self.subTest('tail_is_bit_identical'):
      np.testing.assert_array_equal(
          np.asarray(out[..., width:]), np.asarray(x[..., width:])
      )
    with self.subTest('head_actually_rotated'):
      self.assertGreater(_max_diff(out[..., :width], x[..., :width]), 0.1)

  def test_default_positions_are_zero_to_t_minus_one(self):
    x = jnp.asarray(_inputs(), jnp.float32)

    out = _rope().apply(x)

    expected = _rope().apply(
        x,
        segment_positions=jnp.asarray(
            np.tile(np.arange(_SEQ_LEN), (_BATCH, 1))
        ),
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

  def test_decode_step_requires_positions(self):
    x = jnp.asarray(_inputs(seq_len=1), jnp.float32)

    with self.assertRaisesRegex(ValueError, 'seq_len == 1'):
      _rope().apply(x)

  def test_decode_step_matches_its_row_of_the_prefill(self):
    """A cached step at position t reproduces row t of the whole prefill."""
    x = jnp.asarray(_inputs(), jnp.float32)
    prefill = _rope().apply(x)

    for t in range(_SEQ_LEN):
      with self.subTest(position=t):
        step = _rope().apply(
            x[:, t : t + 1],
            segment_positions=jnp.full((_BATCH, 1), t, jnp.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(step[:, 0]), np.asarray(prefill[:, t])
        )

  def test_positions_beyond_the_window_match_hf(self):
    """The decode step of a long context: positions are not 0 .. T-1."""
    x = _inputs(seq_len=1)
    positions = np.asarray([[4000], [4001]])

    out = _rope().apply(
        jnp.asarray(x, jnp.float32), segment_positions=jnp.asarray(positions)
    )

    expected = _reference_rope(x, _broadcast_rows(positions))
    # Looser than `_ATOL_F32`: the fastest channel's angle is ~4000 rad, where
    # one float32 ulp is 2.4e-4 rad. XLA's range reduction keeps the measured
    # deviation at 7.6e-6; 1e-4 bounds it without pinning that implementation.
    self.assertLess(_max_diff(out, expected), 1e-4)

  def test_ragged_positions_match_hf(self):
    """Packed segments restart positions mid-row; -1 pads clamp to 0."""
    x = _inputs()
    positions = np.asarray([[0, 1, 2, 0, 1], [0, 1, -1, -1, -1]])

    out = _rope().apply(
        jnp.asarray(x, jnp.float32), segment_positions=jnp.asarray(positions)
    )

    expected = _reference_rope(
        x, _broadcast_rows(np.maximum(positions, 0).astype(np.float64))
    )
    self.assertLess(_max_diff(out, expected), _ATOL_F32)

  def test_odd_rotary_width_follows_hf(self):
    """`int(D * f)` odd: HF divides by it but rotates one channel more."""
    x = _inputs()[..., :10]
    positions = np.tile(np.arange(_SEQ_LEN), (_BATCH, 1))
    rope = _rope(rotary_fraction=0.5, mrope_section=(1, 1, 1))

    out = rope.apply(
        jnp.asarray(x, jnp.float32), segment_positions=jnp.asarray(positions)
    )

    expected = _reference_rope(
        x, _broadcast_rows(positions), mrope_section=(1, 1, 1)
    )
    with self.subTest('vs_reference'):
      self.assertLess(_max_diff(out, expected), _ATOL_F32)
    with self.subTest('rotates_six_of_ten_channels'):
      np.testing.assert_array_equal(
          np.asarray(out[..., 6:]), np.asarray(x[..., 6:], np.float32)
      )

  def test_cos_sin_stay_in_float32(self):
    """The module's one deliberate deviation from HF, measured both ways.

    HF rounds `cos`/`sin` to the activation dtype before the multiply
    (`cos.to(dtype=x.dtype)`, modeling_qwen3_5.py:147); this module keeps them
    in float32. Same rotation, same inputs, only the angle dtype differs --
    `rope_lib.apply_rotary` is the shared kernel both sides run.
    """
    x = jnp.asarray(_inputs(), jnp.bfloat16)
    rounded = np.asarray(x, np.float64)  # So the input rounding cancels.
    positions = np.tile(np.arange(_SEQ_LEN), (_BATCH, 1))
    rows = _broadcast_rows(positions)
    cos, sin = _rotary_embedding(
        rows, _HEAD_DIM, _ROTARY_FRACTION, _MROPE_SECTION, _MAX_TIMESCALE
    )

    out = _rope().apply(x, segment_positions=jnp.asarray(positions))
    hf_style = rope_lib.apply_rotary(
        x,
        jnp.asarray(cos[:, :, None, :], jnp.bfloat16),
        jnp.asarray(sin[:, :, None, :], jnp.bfloat16),
    )

    exact = _reference_rope(rounded, rows)
    mine, theirs = _max_diff(out, exact), _max_diff(hf_style, exact)
    with self.subTest('float32_cos_sin_is_closer_to_the_exact_rotation'):
      self.assertLess(mine, _COS_SIN_ADVANTAGE * theirs)
    with self.subTest('and_the_two_are_far_enough_apart_to_see'):
      self.assertGreater(_max_diff(out, hf_style), 1e-3)

  def test_released_geometry_at_the_last_trained_position(self):
    """Position 262143 with the released theta: the worst rounding there is."""
    x = np.asarray(
        jax.random.normal(jax.random.PRNGKey(9), (1, 2, 2, 256)), np.float64
    )
    positions = np.asarray([[262143, 262142]])

    out = rope_lib.Qwen38RoPE().apply(
        jnp.asarray(x, jnp.float32),
        segment_positions=jnp.asarray(positions),
    )

    expected = _reference_rope(
        x,
        _broadcast_rows(positions),
        rotary_fraction=_RELEASED_FRACTION,
        mrope_section=_RELEASED_SECTION,
    )
    self.assertLess(_max_diff(out, expected), _ATOL_LAST_POSITION)

  def test_released_defaults_are_the_released_values(self):
    rope = rope_lib.Qwen38RoPE()

    self.assertEqual(
        (
            rope.min_timescale,
            rope.max_timescale,
            rope.rotary_fraction,
            rope.mrope_section,
        ),
        (1, 10_000_000, 0.25, (11, 11, 10)),
    )

  def test_released_geometry_matches_hf(self):
    """The real 256-wide head with the real (11, 11, 10) sections."""
    x = np.asarray(
        jax.random.normal(jax.random.PRNGKey(3), (1, 4, 2, 256)), np.float64
    )
    rows = np.stack([
        np.asarray([[0, 1, 2, 3]]),
        np.asarray([[0, 1, 1, 2]]),
        np.asarray([[0, 0, 1, 1]]),
    ])

    out = rope_lib.Qwen38RoPE().apply(
        jnp.asarray(x, jnp.float32), segment_positions=jnp.asarray(rows)
    )

    expected = _reference_rope(
        x,
        rows,
        rotary_fraction=_RELEASED_FRACTION,
        mrope_section=_RELEASED_SECTION,
    )
    self.assertLess(_max_diff(out, expected), _ATOL_F32)

  @parameterized.named_parameters(
      ('non_interleaved', dict(mrope_interleaved=False)),
      ('empty_section', dict(mrope_section=())),
      ('two_sections', dict(mrope_section=(16, 16))),
      ('empty_row', dict(mrope_section=(11, 11, 0))),
  )
  def test_rejects_an_unported_layout(self, overrides):
    with self.assertRaises(ValueError):
      rope_lib.Qwen38RoPE(**overrides)

  def test_is_a_frozen_position_encoding_config(self):
    rope = rope_lib.Qwen38RoPE()

    with self.subTest('registered'):
      self.assertIs(
          pe_lib.PositionEncodingRegistry.get('Qwen38RoPE'),
          rope_lib.Qwen38RoPE,
      )
    with self.subTest('replace_revalidates'):
      with self.assertRaises(ValueError):
        dataclasses.replace(rope, mrope_interleaved=False)

  def test_apply_is_jittable(self):
    x = jnp.asarray(_inputs(), jnp.float32)
    positions = jnp.asarray(np.tile(np.arange(_SEQ_LEN), (_BATCH, 1)))
    rope = _rope()

    out = jax.jit(rope.apply)(x, positions)

    # Not bit-identical to the eager call: XLA fuses the multiply-add.
    self.assertLess(
        _max_diff(out, rope.apply(x, segment_positions=positions)), _ATOL_F32
    )


if __name__ == '__main__':
  absltest.main()

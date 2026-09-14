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
"""Qwen3.8's position encoding: interleaved mRoPE over a quarter of each head.

The math is HF `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb`
(`transformers/models/qwen3_5/modeling_qwen3_5.py`, release `Qwen/Qwen3.8-27B`,
`rope_parameters = {rope_theta: 1e7, partial_rotary_factor: 0.25,
mrope_section: [11, 11, 10], mrope_interleaved: true}`). Per head, writing
`W = int(per_head_dim * rotary_fraction)` for the rotated width (64 of 256),
`inv_freq` for its `W/2 = 32` frequencies and `p = (p_t, p_h, p_w)` for the
three position rows:

    inv_freq[j]  = 1 / (max_timescale ** (2j / W))            j = 0 .. W/2 - 1
    freqs[a, j]  = p_a * inv_freq[j]                          a in {t, h, w}
    theta[j]     = freqs[row(j), j]                           the interleave
    x[:W]       <- x[:W] * cos(theta ++ theta)
                   + rot_half(x[:W]) * sin(theta ++ theta)
    x[W:]       <- x[W:]                                      never touched

`row(j)` is the *interleaved* mRoPE layout of `apply_interleaved_mrope`, which
reads the frequency axis as `t h w t h w ... t t`: `j % 3 == 1` takes the
height row for the first `mrope_section[1]` such `j`, `j % 3 == 2` takes the
width row for the first `mrope_section[2]`, and everything else -- including
every `j` past `3 * mrope_section[a]` -- stays on the time row. Text-only
inference passes a single position row, which makes the three rows equal and
the interleave an identity; the layout still has to be exact, because the
released weights were trained with it.

Two deliberate deviations from HF, both documented in `rope_test.py`:

  * `cos`/`sin` stay float32 through the multiply instead of being rounded to
    the activation dtype first (HF `cos.to(dtype=x.dtype)`), as core
    `pe_lib.RoPE` also does. Strictly more accurate; worth ~1 bf16 ulp.
  * `min_timescale` generalizes HF's fixed `1.0`; it is a Simply knob that the
    released config never moves, and `min_timescale = 1` is HF exactly.

Only the interleaved-mRoPE branch of the Qwen3.5 port exists here: Qwen3.8-27B
has `mrope_interleaved: true` and a non-empty `mrope_section`, so the port's
plain partial-RoPE fallback is unreachable. `__post_init__` rejects the
configurations that would have taken it rather than quietly encoding positions
some other way.
"""

import dataclasses

import jax.numpy as jnp
from simply.utils import common
from simply.utils import position_encoding as pe_lib

Array = common.Array

# --- Frequencies ------------------------------------------------------------


def rotary_width(head_dim: int, rotary_fraction: float) -> int:
  """HF `dim = int(head_dim * partial_rotary_factor)`.

  Args:
    head_dim: Channels per head.
    rotary_fraction: HF `partial_rotary_factor`.

  Returns:
    The leading channel count the rotation is derived from. It is also the
    exponent's divisor, and it is *not* rounded up to even: an odd `W` yields
    `ceil(W / 2)` frequencies and therefore rotates `W + 1` channels, exactly
    as HF's `arange(0, W, 2)` does.

  Raises:
    ValueError: if the fraction selects no channels, or more than there are.
  """
  width = int(head_dim * rotary_fraction)
  if width <= 0 or width > head_dim:
    raise ValueError(
        f'{rotary_fraction=} selects {width} of {head_dim=} channels; it must'
        ' select at least one and at most all of them.'
    )
  return width


def inverse_frequencies(
    width: int, *, min_timescale: float, max_timescale: float
) -> Array:
  """HF `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`.

  Args:
    width: `rotary_width(...)`; HF's `dim`.
    min_timescale: Period of the fastest channel (HF fixes it at 1).
    max_timescale: HF `rope_theta`.

  Returns:
    `[ceil(width / 2)]` float32 inverse frequencies.
  """
  fraction = jnp.arange(0, width, 2, dtype=jnp.float32) / width
  return 1.0 / (min_timescale * (max_timescale / min_timescale) ** fraction)


def interleaved_mrope_freqs(
    positions: Array, inv_freq: Array, mrope_section: tuple[int, ...]
) -> Array:
  """HF `Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope`.

  Args:
    positions: `[3, B, T]` float32 time / height / width position rows.
    inv_freq: `[W/2]` float32 from `inverse_frequencies`.
    mrope_section: Channels per row. Only entries 1 and 2 are read, by HF too
      -- the time row is the default that the other two overwrite -- and a
      section wider than the rotary width clips, leaving the tail on the time
      row. That is what the tiny test config does (a 64-wide head leaves 8
      frequencies for sections summing to 32).

  Returns:
    `[B, T, W/2]` float32 rotation angles.
  """
  freqs = positions[..., jnp.newaxis] * inv_freq
  angles = freqs[0]
  for row, offset in enumerate((1, 2), start=1):
    channels = slice(offset, mrope_section[row] * 3, 3)
    angles = angles.at[..., channels].set(freqs[row, ..., channels])
  return angles


def position_rows(segment_positions: Array | None, seq_len: int) -> Array:
  """The `[3, B, T]` float32 position rows a call encodes with.

  Args:
    segment_positions: `[T]`, `[B, T]` or `[3, B, T]` positions; the first two
      broadcast to all three rows, which is what text-only inference passes.
      Negative positions clamp to 0. HF does not clamp, and no Simply driver
      emits a negative position today (core's prefill defaults to `arange`, and
      `continue_decode` passes a non-negative scalar); the clamp is defensive,
      and any row it touches is a pad row that no valid query attends to.
      None means `0 .. seq_len - 1`.
    seq_len: `T`, used only when `segment_positions` is None.

  Returns:
    `[3, B, T]` float32, non-negative.

  Raises:
    ValueError: if positions are omitted for a single-token call, or their rank
      is not 1, 2, or 3-with-a-leading-3.
  """
  if segment_positions is None:
    if seq_len == 1:
      raise ValueError(
          'segment_positions must be explicitly provided in Qwen38RoPE during'
          ' decoding (seq_len == 1) to maintain absolute position invariant.'
      )
    positions = jnp.arange(seq_len, dtype=jnp.float32)[jnp.newaxis, :]
  else:
    positions = jnp.maximum(
        0.0, jnp.asarray(segment_positions, dtype=jnp.float32)
    )
  if positions.ndim == 1:
    positions = positions[jnp.newaxis, :]
  if positions.ndim == 2:
    return jnp.broadcast_to(positions[jnp.newaxis], (3,) + positions.shape)
  if positions.ndim != 3 or positions.shape[0] != 3:
    raise ValueError(
        'Position IDs must be 1D, 2D, or 3D (with shape[0]==3), got'
        f' {positions.shape}'
    )
  return positions


# --- Rotation ---------------------------------------------------------------


def rotate_half(x: Array) -> Array:
  """HF `rotate_half`: `[x1, x2] -> [-x2, x1]`, halves of the last axis."""
  first, second = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([-second, first], axis=-1)


def apply_rotary(x: Array, cos: Array, sin: Array) -> Array:
  """HF `apply_rotary_pos_emb`, for one of query / key.

  Args:
    x: `[B, T, H, D]` per-head activations.
    cos: `[B, T, 1, R]` cosines, `R <= D` the rotated width.
    sin: `[B, T, 1, R]` sines.

  Returns:
    `[B, T, H, D]` in `x`'s dtype: the first `R` channels rotated, the rest
    passed through bit-for-bit.
  """
  width = cos.shape[-1]
  rotary, passthrough = x[..., :width], x[..., width:]
  rotated = rotary * cos + rotate_half(rotary) * sin
  return jnp.concatenate([rotated, passthrough], axis=-1).astype(x.dtype)


# --- Config -----------------------------------------------------------------


@pe_lib.PositionEncodingRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38RoPE(pe_lib.PositionEncodingConfig):
  """Interleaved multimodal RoPE with a partial rotary fraction.

  Defaults are the released `Qwen/Qwen3.8-27B` values; `config_lib` builds one
  from `config.json:text_config.rope_parameters` rather than relying on them.

  Attributes:
    min_timescale: Period of the fastest rotated channel. HF hardcodes 1.
      HF's `attention_scaling` is not modelled: it is 1.0 for `rope_type:
      "default"`, which `config_lib.config_from_hf` is the guard for.
    max_timescale: HF `rope_theta`.
    rotary_fraction: HF `partial_rotary_factor`; 0.25 rotates 64 of 256
      channels and leaves the other 192 untouched.
    mrope_section: HF `mrope_section`, channels per position row. HF defaults
      it to `[11, 11, 10]` when the key is absent, as this does.
    mrope_interleaved: Present so that a config asking for the layout this
      module does not implement fails loudly; see `__post_init__`.
  """

  min_timescale: int = 1
  max_timescale: int = 10_000_000
  rotary_fraction: float = 0.25
  mrope_section: tuple[int, ...] = (11, 11, 10)
  mrope_interleaved: bool = True

  def __post_init__(self):
    if not self.mrope_interleaved:
      raise ValueError(
          'Qwen38RoPE implements only the interleaved mRoPE layout. HF'
          ' ignores this flag -- `Qwen3_5TextRotaryEmbedding.forward` calls'
          ' `apply_interleaved_mrope` unconditionally and the key is in'
          ' `ignore_keys_at_rope_validation` -- so False can only mean the'
          ' caller wants the chunked (qwen3_vl) layout, which is a different'
          ' model, or that this config was not read by `config_from_hf`.'
      )
    if len(self.mrope_section) != 3 or any(s <= 0 for s in self.mrope_section):
      raise ValueError(
          'mrope_section must hold one positive channel count per position row'
          f' (time, height, width); got {self.mrope_section}.'
      )

  def apply(
      self,
      embedding_mat: Array,
      segment_positions: Array | None = None,
  ) -> Array:
    """Rotates the leading `rotary_fraction` of each head.

    Args:
      embedding_mat: `[B, T, H, D]` queries or keys.
      segment_positions: `[T]`, `[B, T]` or `[3, B, T]` positions; see
        `position_rows`. Required when `T == 1`.

    Returns:
      `[B, T, H, D]` in `embedding_mat`'s dtype.
    """
    width = rotary_width(embedding_mat.shape[-1], self.rotary_fraction)
    inv_freq = inverse_frequencies(
        width,
        min_timescale=self.min_timescale,
        max_timescale=self.max_timescale,
    )
    positions = position_rows(segment_positions, embedding_mat.shape[1])
    angles = interleaved_mrope_freqs(positions, inv_freq, self.mrope_section)
    # `[angles, angles]`, so channel j and j + R/2 share an angle: that is the
    # pairing `rotate_half` undoes.
    emb = jnp.concatenate([angles, angles], axis=-1)[:, :, jnp.newaxis, :]
    return apply_rotary(embedding_mat, jnp.cos(emb), jnp.sin(emb))

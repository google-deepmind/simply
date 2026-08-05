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
"""Weight-only integer quantization for MoE expert grouped-matmul at decode.

Decode of a fine-grained MoE (many experts, ~1 token/expert/step) is bound by
the HBM bandwidth needed to stream the expert weights. Storing the expert
weights as int8 (instead of bf16) halves that traffic; int4 quarters it.

This module exposes the offline (load-time) quantizer used to produce the
weights that the `gmm_v2` grouped-matmul kernel then consumes at decode:

  * `parse_weight_quant(spec)` parses a fused `'<dtype>[:<block_size>]'`
    string -- e.g. `'int8'`, `'int4:128'` -- into `(quant_dtype, block_size)`.
  * `quantize_moe_weight(w, block_size, quant_dtype)` does per-expert,
    symmetric, block-wise-along-K quantization of an `[E, k, n]` expert weight
    into `(quant [E,k,n], scale f32 [E,n_blocks,n])`. Supports `jnp.int8`
    (W8A16) and `jnp.int4` (W4A16, sub-byte; the gmm_v2 kernel packs/unpacks
    int4 via `pltpu.bitcast`). `block_size == 0` means per-channel (one scale
    over the whole k axis); a positive `block_size` must divide k.

At decode, the integer weights and their per-expert scales are fed straight to
`gmm_v2` (no dequant in HBM); activations stay bf16.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np


# Map the dtype part of a `weight_quant` spec to the quantized weight dtype.
# `int8` is W8A16; `int4` is W4A16 (sub-byte: the gmm_v2 kernel packs/
# unpacks int4 via pltpu.bitcast).
_QUANT_DTYPES = {
    'int8': jnp.int8,
    'int4': jnp.int4,
}


def parse_weight_quant(weight_quant: str) -> tuple[jnp.dtype, int]:
  """Parses a fused `weight_quant` spec into `(quant_dtype, block_size)`.

  The spec is `<dtype>[:<block_size>]`: the dtype part selects the integer width
  ('int8' -> W8A16, 'int4' -> W4A16) and the optional
  colon-suffixed integer is the block-wise quantization group size along the
  contraction (k) axis -- i.e. how many contiguous k elements share one scale.
  Omitting the suffix returns `block_size == 0` meaning per-channel (one scale
  over the whole k axis). A positive `block_size` must divide k (enforced by
  `quantize_moe_weight`, which raises `ValueError` otherwise -- there is no
  silent fallback to per-channel).
  Examples: 'int8' (int8 per-channel), 'int4:128' (int4 with 128-element
  K-blocks).

  Args:
    weight_quant: The fused spec string. Must be non-empty.

  Returns:
    (quant_dtype, block_size); block_size == 0 means per-channel (whole k).
  """
  dtype_part, _, block_size_part = weight_quant.partition(':')
  if dtype_part not in _QUANT_DTYPES:
    raise ValueError(
        f'Unsupported weight_quant dtype {dtype_part!r} in {weight_quant!r};'
        f' expected one of {sorted(_QUANT_DTYPES)} (optionally followed by'
        ' ":<block_size>").'
    )
  if block_size_part:
    if not block_size_part.isdigit() or int(block_size_part) < 1:
      raise ValueError(
          f'Invalid block_size {block_size_part!r} in'
          f' weight_quant={weight_quant!r}; expected a positive integer.'
      )
    block_size = int(block_size_part)
  else:
    block_size = 0  # per-channel (whole k)
  return _QUANT_DTYPES[dtype_part], block_size


# Supported low-precision floating dtypes for the paged KV cache, keyed by the
# `<dtype>` part of a `kv_cache_quant` spec.
_KV_CACHE_FLOAT_DTYPES = {
    'fp8': jnp.float8_e4m3fn,
    'fp8_e4m3': jnp.float8_e4m3fn,
    'fp8_e5m2': jnp.float8_e5m2,
}

# Default per-tensor dequant scales for the fp8 KV cache, by dtype. The cache
# stores `clip(kv / scale, +-dtype_max).astype(fp8)`; the RPA kernel multiplies
# the dequantized QK^T / PV contributions back by k_scale / v_scale.
_KV_CACHE_DEFAULT_KSCALE = {
    'float8_e4m3fn': 1.0,
    'float8_e5m2': None,
}
_KV_CACHE_DEFAULT_VSCALE = {
    'float8_e4m3fn': 2.0,
    'float8_e5m2': None,
}


def parse_kv_cache_quant(
    kv_cache_quant: str,
) -> tuple[jnp.dtype, float | None, float | None]:
  """Parses a `kv_cache_quant` spec into `(cache_dtype, k_scale, v_scale)`.

  Spec forms:
    '<dtype>'                       -> dtype + default k/v scales
    '<dtype>:<scale>'               -> dtype + that scale for BOTH k and v
    '<dtype>:<k_scale>,<v_scale>'   -> dtype + separate k and v scales
  where <dtype> is 'fp8'/'fp8_e4m3' (float8_e4m3fn) or 'fp8_e5m2'. The cache
  stores `clip(kv / scale, +-dtype_max).astype(dtype)` (clip is a mandatory NaN
  backstop -- a raw cast NaNs above the dtype max), and the RPA kernel
  dequantizes via k_scale / v_scale. Defaults give K small headroom and V a
  large range (the V tail reaches ~1248; see module comment).

  Args:
    kv_cache_quant: The spec string. Must be non-empty.

  Returns:
    `(cache_dtype, k_scale, v_scale)` with positive python-float scales.
  """
  dtype_part, _, scale_part = kv_cache_quant.partition(':')
  if dtype_part not in _KV_CACHE_FLOAT_DTYPES:
    raise ValueError(
        f'Unsupported kv_cache_quant dtype {dtype_part!r} in'
        f' {kv_cache_quant!r}; expected one of'
        f' {sorted(_KV_CACHE_FLOAT_DTYPES)} (optionally followed by'
        ' ":<scale>" or ":<k_scale>,<v_scale>").'
    )
  cache_dtype = jnp.dtype(_KV_CACHE_FLOAT_DTYPES[dtype_part])

  def _parse_pos_float(s: str) -> float:
    try:
      val = float(s)
    except ValueError as e:
      raise ValueError(
          f'Invalid scale {s!r} in kv_cache_quant={kv_cache_quant!r};'
          ' expected a positive float.'
      ) from e
    if not val > 0.0:
      raise ValueError(
          f'Invalid scale {s!r} in kv_cache_quant={kv_cache_quant!r};'
          ' expected a positive float.'
      )
    return val

  if not scale_part:
    k_scale = _KV_CACHE_DEFAULT_KSCALE[cache_dtype.name]
    v_scale = _KV_CACHE_DEFAULT_VSCALE[cache_dtype.name]
  elif ',' in scale_part:
    ks, vs = scale_part.split(',', 1)
    k_scale, v_scale = _parse_pos_float(ks), _parse_pos_float(vs)
  else:
    k_scale = v_scale = _parse_pos_float(scale_part)
  return cache_dtype, k_scale, v_scale


def cast_to_low_precision_float(
    x: jax.Array, dtype: jax.typing.DTypeLike, scale: float | None = None
) -> jax.Array:
  """Clips and casts `x / scale` to a low-precision float `dtype`."""
  dtype = jnp.dtype(dtype)
  assert jnp.issubdtype(
      dtype, jnp.floating
  ), f'cast_to_low_precision_float expects a float dtype, got {dtype}'
  if scale is not None:
    assert scale > 0 and math.isfinite(
        scale
    ), f'scale must be positive finite, got {scale}'
  finfo = jnp.finfo(dtype)
  scaled = x.astype(jnp.float32)
  if scale is not None:
    scaled = scaled / scale
  scaled = jnp.clip(scaled, float(finfo.min), float(finfo.max))
  return scaled.astype(dtype)


def _symmetric_qmax(quant_dtype: jnp.dtype) -> int:
  """Returns the symmetric signed-int max for `quant_dtype`.

  Args:
    quant_dtype: A signed integer dtype (e.g. `jnp.int8`, `jnp.int4`).

  Returns:
    The symmetric positive limit, e.g. int8 -> 127, int4 -> 7.
  """
  bits = jax.dtypes.itemsize_bits(jnp.dtype(quant_dtype))
  return (1 << (bits - 1)) - 1


# Clip-ratio grid for MSE-optimal weight calibration. Each ratio scales the
# per-group abs-max BEFORE deriving the symmetric scale; the ratio that
# minimizes per-group reconstruction MSE is kept. The grid spans [0.5, 1.0] and
# ALWAYS includes 1.0 (== plain RTN), so the calibrated scale can never be worse
# (in per-group MSE) than RTN at the same group size / dtype / layout.
_CLIP_CALIB_MIN_RATIO = 0.5
_CLIP_CALIB_NUM_RATIOS = 20


def _mse_optimal_clip_scale(
    w_blocked: jax.Array,
    abs_max: jax.Array,
    qmax: int,
) -> jax.Array:
  """Per-group MSE-optimal symmetric clip scale (cheapest AWQ-family calib).

  For each quantization group, searches a small grid of clip ratios
  ``alpha in [0.5, 1.0]`` (``1.0`` == plain RTN, always included) and keeps the
  ``alpha`` whose clipped scale ``(alpha * abs_max) / qmax`` minimizes the
  per-group reconstruction MSE ``mean((w - dequant(quant(w)))**2)``. Because the
  grid contains ``1.0``, the returned scale is never worse (in per-group MSE)
  than the RTN scale -- it only ever tightens reconstruction at a fixed group
  size / dtype / layout.

  Args:
    w_blocked: Weight viewed as ``[E, n_blocks, block_size, n]`` (the
      per-channel case uses ``n_blocks == 1``, ``block_size == k``).
    abs_max: Per-group abs-max ``[E, n_blocks, n]`` (f32).
    qmax: Symmetric positive integer limit (e.g. 7 for int4, 127 for int8).

  Returns:
    The MSE-optimal per-group scale ``[E, n_blocks, n]`` (f32), zero-guarded
    exactly like the RTN path (dead groups map to scale 1.0).
  """
  num = _CLIP_CALIB_NUM_RATIOS
  # f32 clip-ratio grid in [_CLIP_CALIB_MIN_RATIO, 1.0] (last == 1.0 exactly ==
  # RTN), built as a single vectorized numpy expression. The arithmetic runs in
  # float64 (numpy default) and is then cast to f32 -- byte-for-byte the same
  # IEEE ops, in the same order, as the prior python-float list comprehension --
  # so the grid (and hence the scan and the served scale) is bitwise-identical
  # to the original running-argmin loop (asserted by the lax.scan golden test).
  # A pure-f32 `jnp.arange` form is deliberately NOT used: it double-rounds the
  # divisions and shifts several ratios (e.g. 4 of 20) by 1 ULP.
  alphas = jnp.asarray(
      _CLIP_CALIB_MIN_RATIO
      + (1.0 - _CLIP_CALIB_MIN_RATIO) * np.arange(num) / (num - 1),
      dtype=jnp.float32,
  )

  def _candidate(alpha: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Candidate per-group scale; guard dead groups to 1.0 (matches RTN).
    cand = (alpha * abs_max) / qmax  # [E, n_blocks, n]
    cand = jnp.where(cand == 0.0, 1.0, cand)
    inv = (1.0 / cand)[:, :, None, :].astype(w_blocked.dtype)
    q = jnp.clip(jnp.round(w_blocked * inv), -qmax, qmax)
    # Fused cast->sub->square->reduce: XLA keeps only the small [E,n_blocks,n]
    # reduction live, so no persistent full-size f32 copy of the weight is
    # materialized (the load-time HBM discipline of the RTN path is preserved;
    # the grid is walked one ratio at a time).
    mse = jnp.mean(
        jnp.square(
            w_blocked.astype(jnp.float32)
            - q.astype(jnp.float32) * cand[:, :, None, :]
        ),
        axis=2,
    )  # [E, n_blocks, n]
    return cand, mse

  def _scan_step(
      carry: tuple[jax.Array, jax.Array], alpha: jax.Array
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    best_scale, best_mse = carry
    cand, mse = _candidate(alpha)
    # Strict '<' keeps the EARLIER (smaller-ratio) alpha on ties, exactly
    # matching the prior python loop's running-argmin.
    better = mse < best_mse
    best_scale = jnp.where(better, cand, best_scale)
    best_mse = jnp.where(better, mse, best_mse)
    return (best_scale, best_mse), None

  # Seed the carry MSE with +inf so the first (smallest) alpha always wins the
  # opening comparison, reproducing the loop's `best = _candidate(alphas[0])`
  # init for all finite inputs (every group's MSE is finite here). The seed
  # scale is arbitrary -- it is overwritten on the first scan step.
  init_scale = jnp.ones_like(abs_max)
  init_mse = jnp.full_like(abs_max, jnp.inf)
  (best_scale, _), _ = jax.lax.scan(_scan_step, (init_scale, init_mse), alphas)
  return best_scale


def quantize_moe_weight(
    w: jax.Array,
    block_size: int = 0,
    quant_dtype: jnp.dtype = jnp.int8,
    calibrate: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Per-expert, block-wise-along-K symmetric integer weight quantization.

  Supports any symmetric signed integer width via `quant_dtype` (`jnp.int8` for
  W8A16, `jnp.int4` for W4A16). The quant range is derived from the dtype's bit
  width, so int4 simply clips to [-7, 7] and stores `jnp.int4` (the gmm_v2
  kernel handles the sub-byte packing).

  `block_size` controls the quantization granularity along the contraction (k)
  axis: `k` is split into contiguous blocks of `block_size` elements and each
  block gets its own per-output-channel scale. `block_size == 0` (the default,
  and the per-channel case) uses a single block over the whole k axis,
  reproducing the original `[E, 1, n]` scale exactly. A smaller `block_size`
  (finer granularity) is the usual way to recover the accuracy that plain
  per-channel int4 loses. Must divide k.

  Args:
    w: Expert weight of shape [num_experts, k, n] (bf16/f32).
    block_size: K-elements per quantization block. 0 = per-channel (whole k).
      Otherwise must divide k.
    quant_dtype: Quantized weight dtype (`jnp.int8` or `jnp.int4`).
    calibrate: If True, derive each group's scale by MSE-optimal clipping
      (search clip ratios in [0.5, 1.0], keep the one minimizing per-group
      reconstruction MSE) instead of plain absmax RTN. The output dtype / shape
      / layout are IDENTICAL to the RTN path (only the stored scale, and hence
      the rounded integers, differ); the grid includes 1.0 so calibration never
      regresses reconstruction vs RTN. Defaults to False, which is byte-for-byte
      identical to the original RTN quantizer.

  Returns:
    (quant_array quant_dtype [E, k, n], scale f32 [E, n_blocks, n]) where
    n_blocks = k // block_size (1 in the per-channel case). For gmm_v2's
    `rhs_scale` (shape [E, n_blocks, 1, n]) a singleton lane axis is inserted
    at position 2 by the gmm_v2 call site in model_lib.
  """
  e, k, n = w.shape
  if block_size and k % block_size != 0:
    raise ValueError(f'k={k} must be divisible by block_size={block_size}.')
  n_blocks = k // block_size if block_size else 1
  qmax = _symmetric_qmax(quant_dtype)
  # Compute abs-max in f32 for numerical safety, but keep it as a SMALL
  # reduction ([E, n_blocks, n]) so we never materialize a full-size f32 copy of
  # the weight (which, summed over all experts at load, would blow the HBM
  # ceiling -- the per-chip bf16 model already nearly fills HBM). The
  # element-wise divide+round is done against the bf16 weight with an f32
  # reciprocal scale, so the only full-size temporary is the quantized output.
  # abs() and max() are exact in the weight's own dtype, so reduce first and
  # only upcast the small [E, n_blocks, n] result -- this lets XLA fuse the
  # reduction without ever materializing a full-size f32 copy of the weight.
  # Sub-byte (int4) outputs are clipped to the symmetric range before cast,
  # since round() can land on +/-(qmax+1) which would wrap in the narrow dtype.
  if n_blocks == 1:
    # Fast path identical to the original per-channel reduction (no reshape).
    abs_max = jnp.max(jnp.abs(w), axis=1, keepdims=True).astype(
        jnp.float32
    )  # [E, 1, n]
    if calibrate:
      # View k as a single block ([E, 1, k, n]) for the per-group MSE search.
      scale = _mse_optimal_clip_scale(w[:, None, :, :], abs_max, qmax)
    else:
      scale = abs_max / qmax
    scale = jnp.where(scale == 0.0, 1.0, scale)
    inv_scale = (1.0 / scale).astype(w.dtype)  # bf16 reciprocal, [E, 1, n]
    q = jnp.clip(jnp.round(w * inv_scale), -qmax, qmax)
    quant = q.astype(quant_dtype)
    return quant, scale  # quant [E, k, n], scale [E, 1, n]
  # Block-wise: view k as (n_blocks, block_size) and reduce within each block.
  w_blocked = w.reshape(e, n_blocks, block_size, n)
  abs_max = jnp.max(jnp.abs(w_blocked), axis=2).astype(
      jnp.float32
  )  # [E, n_blocks, n]
  if calibrate:
    scale = _mse_optimal_clip_scale(w_blocked, abs_max, qmax)
  else:
    scale = abs_max / qmax
  scale = jnp.where(scale == 0.0, 1.0, scale)
  inv_scale = (1.0 / scale)[:, :, None, :].astype(w.dtype)  # [E,n_blocks,1,n]
  q = jnp.clip(jnp.round(w_blocked * inv_scale), -qmax, qmax)
  quant = q.astype(quant_dtype).reshape(e, k, n)
  return quant, scale  # quant [E, k, n], scale [E, n_blocks, n]

"""CPU tests for simply.utils.quant (int8/int4 MoE weight quantization)."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import quant as quant_lib


class QuantParamsTest(absltest.TestCase):

  def test_dequant_roundtrip_broadcasts(self):
    # Mimic common.convert_or_dequantize: quant_w[E,k,n] * scale[E,1,n].
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (4, 8, 16), jnp.float32)
    quant, scale = quant_lib.quantize_moe_weight(w)
    self.assertEqual(scale.shape, (4, 1, 16))
    dequant = quant.astype(jnp.float32) * scale  # [4,8,16]*[4,1,16]
    self.assertEqual(dequant.shape, (4, 8, 16))
    rel = float(jnp.linalg.norm(dequant - w) / jnp.linalg.norm(w))
    self.assertLess(rel, 0.02)  # per-channel int8 ~<2% L2

  def test_blockwise_scale_shape_and_roundtrip(self):
    # block_size>0: scale is [E, n_blocks, n] (n_blocks = k // block_size) and
    # dequant broadcasts per block.
    key = jax.random.PRNGKey(1)
    e, k, n, block_size = 4, 8, 16, 2  # n_blocks = k // block_size = 4
    n_blocks = k // block_size
    w = jax.random.normal(key, (e, k, n), jnp.float32)
    quant, scale = quant_lib.quantize_moe_weight(w, block_size=block_size)
    self.assertEqual(quant.shape, (e, k, n))
    self.assertEqual(scale.shape, (e, n_blocks, n))
    # Block-wise dequant: expand scale over each block of `block_size` rows.
    deq = (
        quant.reshape(e, n_blocks, block_size, n).astype(jnp.float32)
        * scale[:, :, None, :]
    ).reshape(e, k, n)
    rel = float(jnp.linalg.norm(deq - w) / jnp.linalg.norm(w))
    # Finer granularity should be at least as accurate as per-channel (~<2%).
    self.assertLess(rel, 0.02)

  def test_per_channel_default_matches_block_size_zero(self):
    # block_size=0 (default) is the per-channel path: a single [E, 1, n] scale.
    key = jax.random.PRNGKey(2)
    w = jax.random.normal(key, (4, 8, 16), jnp.float32)
    q0, s0 = quant_lib.quantize_moe_weight(w)
    q1, s1 = quant_lib.quantize_moe_weight(w, block_size=0)
    self.assertEqual(s0.shape, (4, 1, 16))
    self.assertEqual(s1.shape, (4, 1, 16))
    self.assertTrue(bool(jnp.array_equal(q0, q1)))
    self.assertTrue(bool(jnp.array_equal(s0, s1)))

  def test_k_not_divisible_raises(self):
    w = jnp.ones((4, 8, 16), jnp.bfloat16)
    with self.assertRaises(ValueError):
      quant_lib.quantize_moe_weight(w, block_size=3)  # 8 % 3 != 0

  def test_int4_w4a16_dtype_range_and_roundtrip(self):
    # W4A16: quant_array is jnp.int4, values clip to [-7, 7], scale = absmax/7.
    key = jax.random.PRNGKey(4)
    e, k, n = 4, 8, 16
    w = jax.random.normal(key, (e, k, n), jnp.float32)
    quant, scale = quant_lib.quantize_moe_weight(
        w, quant_dtype=jnp.int4
    )
    self.assertEqual(quant.dtype, jnp.int4)
    self.assertEqual(quant.shape, (e, k, n))
    self.assertEqual(scale.shape, (e, 1, n))
    qv = quant.astype(jnp.int32)
    self.assertGreaterEqual(int(jnp.min(qv)), -7)
    self.assertLessEqual(int(jnp.max(qv)), 7)
    dequant = qv.astype(jnp.float32) * scale  # [E,k,n]*[E,1,n]
    rel = float(jnp.linalg.norm(dequant - w) / jnp.linalg.norm(w))
    # Per-channel int4 is coarse; just sanity-check it's a real approximation.
    self.assertLess(rel, 0.2)

  def test_int4_blockwise_more_accurate_than_per_channel(self):
    # Block-wise int4 should reduce error vs per-channel int4 (the whole point).
    key = jax.random.PRNGKey(5)
    e, k, n = 4, 64, 16
    w = jax.random.normal(key, (e, k, n), jnp.float32)

    def _rel(block_size):
      q, s = quant_lib.quantize_moe_weight(
          w, block_size=block_size, quant_dtype=jnp.int4
      )
      n_blocks = k // block_size if block_size else 1
      bs = k // n_blocks
      deq = (
          q.astype(jnp.int32)
          .reshape(e, n_blocks, bs, n)
          .astype(jnp.float32)
          * s[:, :, None, :]
      ).reshape(e, k, n)
      return float(jnp.linalg.norm(deq - w) / jnp.linalg.norm(w))

    # block_size=4 (16 blocks over k=64) should beat per-channel (block_size=0).
    self.assertLess(_rel(4), _rel(0))

  def test_parse_weight_quant_fused_spec(self):
    # Fused '<dtype>[:<block_size>]' spec parses to (quant_dtype, block_size);
    # block_size 0 means per-channel (omitted suffix).
    self.assertEqual(quant_lib.parse_weight_quant('int8'), (jnp.int8, 0))
    self.assertEqual(quant_lib.parse_weight_quant('int4'), (jnp.int4, 0))
    self.assertEqual(quant_lib.parse_weight_quant('int4:128'), (jnp.int4, 128))
    self.assertEqual(quant_lib.parse_weight_quant('int8:64'), (jnp.int8, 64))

  def test_int4_via_spec(self):
    # int4:2 spec -> int4 dtype + block_size 2; quantize emits int4 + a scale
    # with k // 2 = 4 blocks for k=8.
    quant_dtype, block_size = quant_lib.parse_weight_quant('int4:2')
    self.assertEqual(quant_dtype, jnp.int4)
    self.assertEqual(block_size, 2)
    w = jax.random.normal(jax.random.PRNGKey(6), (4, 8, 16), jnp.float32)
    q, s = quant_lib.quantize_moe_weight(
        w, block_size=block_size, quant_dtype=quant_dtype
    )
    self.assertEqual(q.dtype, jnp.int4)
    self.assertEqual(s.shape, (4, 4, 16))  # n_blocks = k // block_size = 4

  def test_bad_quant_spec_raises(self):
    with self.assertRaises(ValueError):
      quant_lib.parse_weight_quant('int3')  # bad dtype
    with self.assertRaises(ValueError):
      quant_lib.parse_weight_quant('int4:0')  # bad block_size
    with self.assertRaises(ValueError):
      quant_lib.parse_weight_quant('int4:x')  # non-numeric block_size


def _dequant(quant, scale, block_size):
  """Reconstruct the f32 weight from a (quant, scale) pair."""
  e, k, n = quant.shape
  n_blocks = k // block_size if block_size else 1
  bs = k // n_blocks
  return (
      quant.astype(jnp.int32).reshape(e, n_blocks, bs, n).astype(jnp.float32)
      * scale[:, :, None, :]
  ).reshape(e, k, n)


def _rel_w(w, quant, scale, block_size):
  """Relative L2 reconstruction error ||dequant - w|| / ||w||."""
  deq = _dequant(quant, scale, block_size)
  return float(jnp.linalg.norm(deq - w) / jnp.linalg.norm(w))


def _rtn_reference(w, block_size, quant_dtype):
  """Independent plain-RTN quantizer (the pre-calibration behavior)."""
  e, k, n = w.shape
  n_blocks = k // block_size if block_size else 1
  bs = k // n_blocks
  qmax = 7 if quant_dtype == jnp.int4 else 127  # int4 / int8 symmetric limits
  wb = w.reshape(e, n_blocks, bs, n)
  abs_max = jnp.max(jnp.abs(wb), axis=2).astype(jnp.float32)  # [E, n_blocks, n]
  scale = abs_max / qmax
  scale = jnp.where(scale == 0.0, 1.0, scale)
  inv = (1.0 / scale)[:, :, None, :].astype(w.dtype)
  q = jnp.clip(jnp.round(wb * inv), -qmax, qmax).astype(quant_dtype)
  return q.reshape(e, k, n), scale


class ClipCalibrationTest(absltest.TestCase):
  """MSE-optimal clip calibration: additive, default-RTN-invariant accuracy."""

  def test_calibration_reduces_relative_error_int4_group128(self):
    # DECISION GATE: at the SAME int4 group size (128), MSE-clip
    # calibration must reduce reconstruction relative error below plain RTN.
    # Representative MoE FFN expert-weight tensor (Gaussian, k a multiple
    # of 128).
    key = jax.random.PRNGKey(0)
    e, k, n = 8, 1024, 512  # 8 groups of 128 along k
    w = jax.random.normal(key, (e, k, n), jnp.float32)
    block_size = 128
    q_rtn, s_rtn = quant_lib.quantize_moe_weight(
        w, block_size=block_size, quant_dtype=jnp.int4
    )
    q_cal, s_cal = quant_lib.quantize_moe_weight(
        w, block_size=block_size, quant_dtype=jnp.int4, calibrate=True
    )
    rel_rtn = _rel_w(w, q_rtn, s_rtn, block_size)
    rel_cal = _rel_w(w, q_cal, s_cal, block_size)
    print(
        f'\n[int4:128] rel_w RTN={rel_rtn:.6f}  calib={rel_cal:.6f}  '
        f'reduction={100 * (rel_rtn - rel_cal) / rel_rtn:.2f}%'
    )
    # (quant, scale) must stay kernel-transparent: identical dtype/shape/layout.
    self.assertEqual(q_cal.dtype, q_rtn.dtype)
    self.assertEqual(q_cal.shape, q_rtn.shape)
    self.assertEqual(s_cal.shape, s_rtn.shape)
    # Calibration strictly tightens reconstruction at the same group size.
    self.assertLess(rel_cal, rel_rtn)

  def test_calibrate_off_is_byte_identical_to_rtn(self):
    # With calibrate=False (the default) the quantizer must be byte-for-byte
    # identical to the original RTN quantizer for every dtype / granularity.
    w = jax.random.normal(jax.random.PRNGKey(1), (4, 256, 64), jnp.float32)
    for quant_dtype in (jnp.int8, jnp.int4):
      for block_size in (0, 128):
        with self.subTest(dtype=str(quant_dtype), block_size=block_size):
          q, s = quant_lib.quantize_moe_weight(
              w, block_size=block_size, quant_dtype=quant_dtype
          )
          q_ref, s_ref = _rtn_reference(w, block_size, quant_dtype)
          np.testing.assert_array_equal(
              q.astype(jnp.int32), q_ref.astype(jnp.int32)
          )
          np.testing.assert_array_equal(np.asarray(s), np.asarray(s_ref))

  def test_calibration_never_worse_than_rtn(self):
    # The clip grid includes 1.0 (== RTN), so calibration can never increase the
    # reconstruction error for any dtype / granularity.
    w = jax.random.normal(jax.random.PRNGKey(2), (4, 512, 128), jnp.float32)
    for quant_dtype in (jnp.int8, jnp.int4):
      for block_size in (0, 128):
        with self.subTest(dtype=str(quant_dtype), block_size=block_size):
          q_rtn, s_rtn = quant_lib.quantize_moe_weight(
              w, block_size=block_size, quant_dtype=quant_dtype
          )
          q_cal, s_cal = quant_lib.quantize_moe_weight(
              w, block_size=block_size, quant_dtype=quant_dtype, calibrate=True
          )
          rel_rtn = _rel_w(w, q_rtn, s_rtn, block_size)
          rel_cal = _rel_w(w, q_cal, s_cal, block_size)
          self.assertLessEqual(rel_cal, rel_rtn + 1e-6)

  def test_calibration_preserves_dtype_shape_layout(self):
    # Calibrated outputs must match RTN outputs in dtype/shape so the gmm_v2
    # kernel stays unchanged.
    w = jax.random.normal(jax.random.PRNGKey(7), (4, 256, 32), jnp.float32)
    for quant_dtype in (jnp.int8, jnp.int4):
      for block_size in (0, 128):
        with self.subTest(dtype=str(quant_dtype), block_size=block_size):
          q_rtn, s_rtn = quant_lib.quantize_moe_weight(
              w, block_size=block_size, quant_dtype=quant_dtype
          )
          q_cal, s_cal = quant_lib.quantize_moe_weight(
              w, block_size=block_size, quant_dtype=quant_dtype, calibrate=True
          )
          self.assertEqual(q_cal.dtype, q_rtn.dtype)
          self.assertEqual(q_cal.shape, q_rtn.shape)
          self.assertEqual(s_cal.dtype, s_rtn.dtype)
          self.assertEqual(s_cal.shape, s_rtn.shape)

  def test_calibration_dead_group_matches_rtn(self):
    # An all-zero expert must map to all-zero ints and unit scale under
    # calibration (the RTN zero-guard is preserved).
    w = jnp.zeros((2, 256, 8), jnp.float32)
    w = w.at[1].set(jax.random.normal(jax.random.PRNGKey(3), (256, 8)))
    for block_size in (0, 128):
      with self.subTest(block_size=block_size):
        q_cal, s_cal = quant_lib.quantize_moe_weight(
            w, block_size=block_size, quant_dtype=jnp.int4, calibrate=True
        )
        self.assertTrue(bool(jnp.all(q_cal[0].astype(jnp.int32) == 0)))
        np.testing.assert_array_equal(
            np.asarray(s_cal[0]), np.ones_like(np.asarray(s_cal[0]))
        )


def _loop_clip_scale_reference(w_blocked, abs_max, qmax):
  """Faithful copy of the ORIGINAL running-argmin python loop (pre-lax.scan).

  Kept here as the golden reference so the lax.scan rewrite of
  `quant._mse_optimal_clip_scale` can be asserted byte-for-byte identical.

  Args:
    w_blocked: Per-group blocked weights, shaped `[experts, n_blocks,
      block_size, n]`, over which the MSE-optimal clip scale is searched.
    abs_max: Per-group absolute maximum of `w_blocked` (the unclipped reference
      amplitude), shaped `[experts, n_blocks, n]`.
    qmax: Maximum quantized magnitude for the target int dtype (e.g. 7 for
      int4).

  Returns:
    The per-group symmetric scale (same shape as `abs_max`) that minimizes
    reconstruction MSE across the clip-ratio grid.
  """
  num = quant_lib._CLIP_CALIB_NUM_RATIOS
  min_ratio = quant_lib._CLIP_CALIB_MIN_RATIO
  alphas = [min_ratio + (1.0 - min_ratio) * i / (num - 1) for i in range(num)]

  def _candidate(alpha):
    cand = (alpha * abs_max) / qmax
    cand = jnp.where(cand == 0.0, 1.0, cand)
    inv = (1.0 / cand)[:, :, None, :].astype(w_blocked.dtype)
    q = jnp.clip(jnp.round(w_blocked * inv), -qmax, qmax)
    mse = jnp.mean(
        jnp.square(
            w_blocked.astype(jnp.float32)
            - q.astype(jnp.float32) * cand[:, :, None, :]
        ),
        axis=2,
    )
    return cand, mse

  best_scale, best_mse = _candidate(alphas[0])
  for alpha in alphas[1:]:
    cand, mse = _candidate(alpha)
    better = mse < best_mse
    best_scale = jnp.where(better, cand, best_scale)
    best_mse = jnp.where(better, mse, best_mse)
  return best_scale


class ClipScaleScanGoldenTest(absltest.TestCase):
  """The lax.scan grid search must be bitwise-identical to the old python loop."""

  def _blocked(self, w, block_size):
    e, k, n = w.shape
    n_blocks = k // block_size if block_size else 1
    bs = k // n_blocks
    wb = w.reshape(e, n_blocks, bs, n)
    abs_max = jnp.max(jnp.abs(wb), axis=2).astype(jnp.float32)
    return wb, abs_max

  def test_scan_matches_loop_reference_bitwise(self):
    # Sweep shapes, granularities (per-channel + group 128) and both symmetric
    # limits (int4 qmax=7, int8 qmax=127); require exact equality with the loop.
    shapes = ((4, 256, 64), (8, 1024, 512), (2, 128, 16))
    for seed, shape in enumerate(shapes):
      e, k, n = shape
      w = jax.random.normal(jax.random.PRNGKey(seed), (e, k, n), jnp.float32)
      for block_size in (0, 128):
        if block_size and k % block_size:
          continue
        for qmax in (7, 127):
          with self.subTest(shape=shape, block_size=block_size, qmax=qmax):
            wb, abs_max = self._blocked(w, block_size)
            got = quant_lib._mse_optimal_clip_scale(wb, abs_max, qmax)
            ref = _loop_clip_scale_reference(wb, abs_max, qmax)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_scan_matches_loop_reference_dead_group(self):
    # All-zero (dead) groups: abs-max is 0 for every alpha, so the guarded
    # scale is 1.0 and every candidate ties -- the scan must keep 1.0 exactly
    # like the loop.
    w = jnp.zeros((2, 256, 8), jnp.float32)
    w = w.at[1].set(jax.random.normal(jax.random.PRNGKey(9), (256, 8)))
    for block_size in (0, 128):
      with self.subTest(block_size=block_size):
        wb, abs_max = self._blocked(w, block_size)
        got = quant_lib._mse_optimal_clip_scale(wb, abs_max, 7)
        ref = _loop_clip_scale_reference(wb, abs_max, 7)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_scan_matches_loop_reference_on_ties(self):
    # Constant-magnitude weights make several clip ratios tie on MSE; both the
    # loop and the scan use strict '<', so the earliest (smallest) alpha must
    # win in both.
    w = jnp.full((2, 256, 4), 0.5, jnp.float32)
    for qmax in (7, 127):
      with self.subTest(qmax=qmax):
        wb, abs_max = self._blocked(w, 128)
        got = quant_lib._mse_optimal_clip_scale(wb, abs_max, qmax)
        ref = _loop_clip_scale_reference(wb, abs_max, qmax)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


class KvCacheQuantParseTest(absltest.TestCase):

  def test_dtype_and_default_scales(self):
    # The spec returns (dtype, k_scale, v_scale). Defaults: K small headroom,
    # V a large range (the V tail reaches ~1248, far above e4m3's 448).
    dt, ks, vs = quant_lib.parse_kv_cache_quant('fp8')
    self.assertEqual(dt, jnp.dtype(jnp.float8_e4m3fn))
    self.assertGreater(ks, 0.0)  # pyrefly: ignore[no-matching-overload]
    self.assertGreater(vs, 0.0)  # pyrefly: ignore[no-matching-overload]
    self.assertEqual(
        quant_lib.parse_kv_cache_quant('fp8_e4m3')[0],
        jnp.dtype(jnp.float8_e4m3fn),
    )
    self.assertEqual(
        quant_lib.parse_kv_cache_quant('fp8_e5m2')[0],
        jnp.dtype(jnp.float8_e5m2),
    )

  def test_explicit_scales(self):
    # Single scale -> both k and v; comma -> separate k,v.
    dt, ks, vs = quant_lib.parse_kv_cache_quant('fp8:0.02')
    self.assertEqual(dt, jnp.dtype(jnp.float8_e4m3fn))
    self.assertAlmostEqual(ks, 0.02)  # pyrefly: ignore[no-matching-overload]
    self.assertAlmostEqual(vs, 0.02)  # pyrefly: ignore[no-matching-overload]
    _, ks2, vs2 = quant_lib.parse_kv_cache_quant('fp8:0.03,4.0')
    self.assertAlmostEqual(ks2, 0.03)  # pyrefly: ignore[no-matching-overload]
    self.assertAlmostEqual(vs2, 4.0)  # pyrefly: ignore[no-matching-overload]

  def test_bad_spec_raises(self):
    with self.assertRaises(ValueError):
      quant_lib.parse_kv_cache_quant('int8')  # not a kv cache float dtype
    with self.assertRaises(ValueError):
      quant_lib.parse_kv_cache_quant('fp8:0')  # non-positive scale
    with self.assertRaises(ValueError):
      quant_lib.parse_kv_cache_quant('fp8:x')  # non-numeric scale
    with self.assertRaises(ValueError):
      quant_lib.parse_kv_cache_quant('')  # empty is not a valid dtype


class Fp8LargeVQuantizeTest(absltest.TestCase):
  """Regression for the fp8 0%/NaN root cause: V activations exceed e4m3's range.

  A raw `astype(float8_e4m3fn)` NaNs above ~456 (e4m3fn has no inf); real V
  reaches ~1248. The scale+clip quantize chokepoint
  (quant.cast_to_low_precision_float) must
  stay finite and track the reference. This is the LARGE-V case every prior
  harness missed (they used N(0,1) inputs < 448). CPU/any-hardware numeric
  property.
  """

  def _quantize(self, x, scale, dtype):
    # Exercise the production chokepoint directly (lives in quant.py now).
    return quant_lib.cast_to_low_precision_float(x, dtype, scale)

  def test_raw_astype_nans_but_scaled_clip_is_finite(self):
    rng = jax.random.PRNGKey(0)
    # V with a heavy tail well above e4m3's 448 (peaks ~1500, like real decode).
    v = jax.random.normal(rng, (256, 4, 128), jnp.bfloat16) * 300.0
    v = v.at[0, 0, 0].set(jnp.bfloat16(1500.0))
    vmax = float(jnp.max(jnp.abs(v.astype(jnp.float32))))
    self.assertGreater(vmax, 448.0)  # exceeds e4m3 range -> raw cast would NaN

    # (a) raw astype overflows to NaN (documents the hazard / root cause).
    raw = v.astype(jnp.float8_e4m3fn).astype(jnp.float32)
    self.assertTrue(bool(jnp.any(jnp.isnan(raw))), 'raw fp8 cast should NaN')

    # (b) The quantize path stays FINITE for ANY scale (the clip is the NaN
    # backstop). This is the core safety property and must hold regardless of
    # the configured scale -- including the production default.
    _, _, default_v_scale = quant_lib.parse_kv_cache_quant('fp8')
    for v_scale in (default_v_scale, 1.0, vmax / 448.0):
      deq = (
          self._quantize(v, v_scale, jnp.float8_e4m3fn).astype(jnp.float32)
          * v_scale
      )
      self.assertTrue(
          bool(jnp.all(jnp.isfinite(deq))),
          f'scaled+clip must be finite for v_scale={v_scale}',
      )

    # (c) Accuracy: with a scale sized to COVER the range (no clipping of the
    # tail), the dequant tracks the reference closely. (A smaller scale that
    # clips the tail -- e.g. the default when V exceeds its range -- is finite
    # but lossier on the clipped entries; that accuracy/headroom tradeoff is a
    # deployment choice, not a correctness property, so it is not asserted.)
    cover_scale = vmax / 448.0
    deq = (
        self._quantize(v, cover_scale, jnp.float8_e4m3fn).astype(jnp.float32)
        * cover_scale
    )
    a = np.asarray(v, np.float64).reshape(-1)
    b = np.asarray(deq, np.float64).reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    self.assertGreater(cos, 0.99)


if __name__ == '__main__':
  absltest.main()

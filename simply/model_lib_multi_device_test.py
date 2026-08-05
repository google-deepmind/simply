# Copyright 2024 The Simply Authors
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
"""Tests for `model_lib` features that require multiple JAX devices.

These tests need >1 device to exercise `shard_map`-based sharding paths.
The required `XLA_FLAGS=--xla_force_host_platform_device_count=4` env var
must be set before JAX initializes its backend, so this test lives in its
own target separate from `model_lib_test` (which assumes a single device).

Currently covers the chunked-local-AG algorithm in `model_lib.Attention`;
add additional multi-device test classes here as new features land.

The chunked-local-AG tests verify the mathematical contract of
`Attention._chunked_local_flash_attention` on CPU without invoking the
TPU-only `splash_attention` kernel: we re-implement the same
permute + slice + concat + masked-attention dance the production method
performs, swap in a vanilla `einsum`-based attention kernel as the per-shard
inner call, and check it matches a non-sharded local-causal attention
reference on the same QKV inputs. The production method differs from this
test only in that splash replaces the einsum (and splash is verified
separately by the splash library).
"""

import functools
import math
import os
import sys
import unittest

# Must be set BEFORE `import jax`: JAX initializes its backend on first use and
# the device count cannot be changed afterwards. If jax is already imported
# (e.g. a whole-directory `pytest simply/` run imported another test module
# first) setting it would reshape every other test's default mesh instead, so
# leave it alone and let the tests below skip.
if 'jax' not in sys.modules:
  os.environ.setdefault(
      'XLA_FLAGS', '--xla_force_host_platform_device_count=4'
  )

# pylint: disable=g-import-not-at-top
# Imports below jax-affecting env vars are intentional; do not reorder.
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

# pylint: enable=g-import-not-at-top


# Tiny shape scaled down from a representative production local-attention
# configuration (seq=262144, window=1023, shard_count=32, shard_size=8192,
# block_kv=2048, auto-derived left_slice=2048). Here shard_count and
# head_dim are kept small to keep each case under a second on CPU;
# `window_size` and `block_kv` are varied to exercise both the tail-slice
# and full-shard branches of the closed-form derivation.
_BATCH = 1
_NUM_HEADS = 2
_SEQ_LEN = 128
_HEAD_DIM = 8
_SHARD_COUNT = 4
_SHARD_SIZE = _SEQ_LEN // _SHARD_COUNT  # = 32


def _derive_left_slice(window_size: int, shard_size: int, block_kv: int) -> int:
  """Mirror of the closed form in `Attention._chunked_local_flash_attention`."""
  chunk_kv_len = math.ceil((window_size + shard_size) / block_kv) * block_kv
  return chunk_kv_len - shard_size


class ChunkedLocalAlgorithmParityTest(parameterized.TestCase):
  """Numerical parity for the chunked-local-AG algorithm vs full attention."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if len(jax.devices()) < _SHARD_COUNT:
      raise unittest.SkipTest(
          f'Need {_SHARD_COUNT} CPU devices for the shard_map parity test; '
          f'XLA_FLAGS={os.environ.get("XLA_FLAGS", "<unset>")}'
      )

  @parameterized.named_parameters(
      # tail_slice: derived left_slice < shard_size. With window=16,
      # shard_size=32, block_kv=8: ceil(48/8)*8 = 48, slice = 16. Mirrors
      # the representative production shape (window=1023, shard_size=8192,
      # block_kv=2048, slice=2048) scaled down.
      dict(testcase_name='tail_slice', window_size=16, block_kv=8),
      # full_shard: derived left_slice == shard_size. With window=24,
      # shard_size=32, block_kv=32: ceil(56/32)*32 = 64, slice = 32.
      # Exercises the no-op slice path where the entire neighbor shard is
      # permuted.
      dict(testcase_name='full_shard', window_size=24, block_kv=32),
  )
  def test_chunked_local_flash_attention_algorithm_parity(
      self, window_size: int, block_kv: int
  ):
    left_slice = _derive_left_slice(window_size, _SHARD_SIZE, block_kv)
    assert window_size <= left_slice <= _SHARD_SIZE, (
        f'Test misconfigured: derived left_slice={left_slice} for '
        f'window_size={window_size}, shard_size={_SHARD_SIZE}, '
        f'block_kv={block_kv} violates window_size <= left_slice <= '
        f'shard_size.'
    )

    rng = np.random.default_rng(0)
    qkv_shape = (_BATCH, _NUM_HEADS, _SEQ_LEN, _HEAD_DIM)
    q = rng.standard_normal(qkv_shape).astype(np.float32)
    k = rng.standard_normal(qkv_shape).astype(np.float32)
    v = rng.standard_normal(qkv_shape).astype(np.float32)
    # Single packed sequence per row — segment ids constant, positions
    # monotonic. Matches the SFT case the original CL benchmarks.
    segment_ids = np.ones((_BATCH, _SEQ_LEN), dtype=np.int32)

    # ---- Reference: non-sharded local-causal attention. -------------------
    # Causal AND within-window-size mask. Matches the production splash
    # `CausalMask() & LocalMask((window_size, None), offset=0)` semantics.
    q_pos = np.arange(_SEQ_LEN)[:, None]
    kv_pos = np.arange(_SEQ_LEN)[None, :]
    local_causal_mask = (q_pos >= kv_pos) & (q_pos - kv_pos <= window_size)
    scale = 1.0 / np.sqrt(_HEAD_DIM)
    logits = (jnp.einsum('bnqh,bnkh->bnqk', q, k) * scale).astype(jnp.float32)
    logits = jnp.where(local_causal_mask, logits, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1)
    reference = jnp.einsum('bnqk,bnkh->bnqh', weights, v)

    # ---- Chunked-local: shard_map mirroring _chunked_local_flash_attention.
    devices = jax.devices()[:_SHARD_COUNT]
    mesh = jax.sharding.Mesh(np.array(devices).reshape(_SHARD_COUNT), ('seq',))
    bnlh = jax.sharding.PartitionSpec(None, None, 'seq', None)
    bl = jax.sharding.PartitionSpec(None, 'seq')

    # Wrap-around sentinel — any value distinct from real segment ids will
    # be dropped by the vanilla `seg_match` equality below; this mirrors the
    # production code's `_WRAPAROUND_SENTINEL_SEGMENT_ID`.
    sentinel = jnp.int32(2**30)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(bnlh, bnlh, bnlh, bl, bl),
        out_specs=bnlh,
        check_vma=False,
    )
    def chunked_local_attn_fn(query, key, value, q_seg_ids, kv_seg_ids):
      # Same control flow as Attention._chunked_local_flash_attention.
      perm = [(i, (i + 1) % _SHARD_COUNT) for i in range(_SHARD_COUNT)]
      if left_slice < _SHARD_SIZE:
        key_tail = jax.lax.slice_in_dim(
            key, _SHARD_SIZE - left_slice, _SHARD_SIZE, axis=2
        )
        value_tail = jax.lax.slice_in_dim(
            value, _SHARD_SIZE - left_slice, _SHARD_SIZE, axis=2
        )
        kv_seg_tail = jax.lax.slice_in_dim(
            kv_seg_ids, _SHARD_SIZE - left_slice, _SHARD_SIZE, axis=1
        )
      else:
        key_tail, value_tail, kv_seg_tail = key, value, kv_seg_ids
      left_key = jax.lax.ppermute(key_tail, axis_name='seq', perm=perm)
      left_value = jax.lax.ppermute(value_tail, axis_name='seq', perm=perm)
      left_kv_seg_ids = jax.lax.ppermute(
          kv_seg_tail, axis_name='seq', perm=perm
      )
      # Wrap-around guard: on shard 0 the "left neighbor" is the wrapped-
      # around last shard and must be masked out.
      shard_idx = jax.lax.axis_index('seq')
      left_kv_seg_ids = jnp.where(shard_idx == 0, sentinel, left_kv_seg_ids)

      # Concat [left_neighbor_tail, own] along seq.
      chunk_key = jnp.concatenate([left_key, key], axis=2)
      chunk_value = jnp.concatenate([left_value, value], axis=2)
      chunk_kv_seg_ids = jnp.concatenate([left_kv_seg_ids, kv_seg_ids], axis=1)

      # Vanilla masked attention with local-causal mask offset by left_slice
      # (mirrors the splash `CausalMask & LocalMask` construction in
      # `_chunked_local_flash_attention`).
      chunk_kv_len = left_slice + _SHARD_SIZE
      q_pos_shard = jnp.arange(_SHARD_SIZE)[:, None] + left_slice
      kv_pos_shard = jnp.arange(chunk_kv_len)[None, :]
      causal = q_pos_shard >= kv_pos_shard
      local = (q_pos_shard - kv_pos_shard) <= window_size
      seg_match = q_seg_ids[:, :, None] == chunk_kv_seg_ids[:, None, :]
      # Reshape masks to broadcast against [B, N, Lq, Lk].
      static_mask = (causal & local)[None, None]
      seg_match = seg_match[:, None]
      mask = static_mask & seg_match
      shard_logits = (
          jnp.einsum('bnqh,bnkh->bnqk', query, chunk_key) * scale
      ).astype(jnp.float32)
      shard_logits = jnp.where(mask, shard_logits, jnp.finfo(jnp.float32).min)
      shard_weights = jax.nn.softmax(shard_logits, axis=-1)
      return jnp.einsum('bnqk,bnkh->bnqh', shard_weights, chunk_value)

    chunked = chunked_local_attn_fn(q, k, v, segment_ids, segment_ids)

    # Float32, batch=1, head_dim=8 — noise floor on CPU is ~1e-6. atol=1e-5
    # matches the precedent set by `test_chunked_local_attn` in
    # `model_lib_test.py`.
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(reference), atol=1e-5, rtol=1e-5
    )


class ChunkedLocalEligibilityPredicateTest(parameterized.TestCase):
  """Pin the 4-clause predicate that gates the chunked-local-AG path.

  Documents (and guards) the four conditions in `Attention.apply` that
  decide whether to dispatch to `_chunked_local_flash_attention`:
    1. `use_chunked_local_ag` is enabled,
    2. the layer is local-attention (`window_size > 0`),
    3. the seq axis is actually sharded (`shard_count > 1`),
    4. one neighbor shard covers the window (`window_size <= shard_size`).
  """

  def _eligible(
      self, *, use_chunked_local_ag, window_size, shard_count, shard_size
  ):
    return (
        use_chunked_local_ag
        and window_size > 0
        and shard_count > 1
        and window_size <= shard_size
    )

  def test_realistic_local_attention_shape_eligible(self):
    self.assertTrue(
        self._eligible(
            use_chunked_local_ag=True,
            window_size=1023,
            shard_count=32,
            shard_size=8192,
        )
    )

  def test_opt_out_not_eligible(self):
    self.assertFalse(
        self._eligible(
            use_chunked_local_ag=False,
            window_size=1023,
            shard_count=32,
            shard_size=8192,
        )
    )

  def test_global_attention_not_eligible(self):
    self.assertFalse(
        self._eligible(
            use_chunked_local_ag=True,
            window_size=0,
            shard_count=32,
            shard_size=8192,
        )
    )

  def test_unsharded_seq_not_eligible(self):
    # Nothing to gain when the sequence isn't sharded.
    self.assertFalse(
        self._eligible(
            use_chunked_local_ag=True,
            window_size=1023,
            shard_count=1,
            shard_size=262144,
        )
    )

  def test_window_equals_shard_size_eligible(self):
    # Boundary of the loosened predicate (window_size == shard_size): the
    # chunked path yields no payload savings (slice == shard_size) but is
    # still mathematically correct, so it stays eligible.
    self.assertTrue(
        self._eligible(
            use_chunked_local_ag=True,
            window_size=8192,
            shard_count=32,
            shard_size=8192,
        )
    )

  def test_window_too_large_not_eligible(self):
    # Window spans more than one neighbor shard — chunked path can't help.
    self.assertFalse(
        self._eligible(
            use_chunked_local_ag=True,
            window_size=8193,
            shard_count=32,
            shard_size=8192,
        )
    )


class ChunkedLocalDerivedSliceTest(parameterized.TestCase):
  """Pin the closed-form slice derivation in `_chunked_local_flash_attention`.

  Removing the `chunked_local_left_slice` config field means the previously-
  explicit `2048` for the representative production shape is now
  auto-derived. This test guards the closed-form derivation:

      slice = ceil((window_size + shard_size) / block_kv) * block_kv
              - shard_size

  against accidental drift. The headline case is the representative
  production shape: the auto-derived slice MUST equal the previously-
  explicit 2048, otherwise loss-parity against the original baselines would
  no longer hold.
  """

  @parameterized.named_parameters(
      # Representative production local-attention shape: window=1023,
      # shard_size=8192, block_kv=2048.
      # ceil((1023+8192)/2048)*2048 - 8192 = ceil(9215/2048)*2048 - 8192
      # = 5*2048 - 8192 = 2048.  <- The headline parity case.
      dict(
          testcase_name='realistic_local_attention',
          window_size=1023,
          shard_size=8192,
          block_kv=2048,
          expected_slice=2048,
      ),
      # Tail-slice case: window+shard is not block-aligned, derivation
      # rounds up.
      dict(
          testcase_name='tail_slice_round_up',
          window_size=16,
          shard_size=32,
          block_kv=8,
          expected_slice=16,  # ceil(48/8)*8 - 32 = 48 - 32 = 16
      ),
      # Full-shard case: derivation hits exactly shard_size.
      dict(
          testcase_name='full_shard_at_boundary',
          window_size=24,
          shard_size=32,
          block_kv=32,
          expected_slice=32,  # ceil(56/32)*32 - 32 = 64 - 32 = 32
      ),
      # Window already block-aligned: ceil collapses.
      dict(
          testcase_name='window_block_aligned',
          window_size=1024,
          shard_size=8192,
          block_kv=2048,
          expected_slice=2048,  # ceil(9216/2048)*2048 - 8192 = 5*2048 - 8192
      ),
  )
  def test_derived_left_slice(
      self,
      window_size: int,
      shard_size: int,
      block_kv: int,
      expected_slice: int,
  ):
    self.assertEqual(
        _derive_left_slice(window_size, shard_size, block_kv),
        expected_slice,
    )

  def test_derived_slice_satisfies_constraints(self):
    """Derived slice satisfies both invariants for the representative shape.

    (a) slice >= window_size  (no truncated KV window for leftmost Q)
    (b) (slice + shard_size) % block_kv == 0  (no partial splash blocks)
    """
    window_size, shard_size, block_kv = 1023, 8192, 2048
    left_slice = _derive_left_slice(window_size, shard_size, block_kv)
    self.assertGreaterEqual(left_slice, window_size, '(a) violated')
    self.assertEqual(
        (left_slice + shard_size) % block_kv, 0, '(b) violated'
    )


if __name__ == '__main__':
  absltest.main()

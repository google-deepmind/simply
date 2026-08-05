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

"""Repro: seq-sharded paged attention + sliding window."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.kernels import ragged_paged_attention as rpa_kernel
from simply.utils import common
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sharding

RaggedArray = common.RaggedArray

_PAGE_SIZE = 2
_N_KV_HEADS = 1
_N_Q_HEADS = 2
_PER_HEAD_DIM = 2
_MAX_SEQ_LEN = 65
_TOTAL_NUM_PAGES = 32


def _make_config(
    *, window_size: int | None, num_seq_shards: int, batch_size: int = 1
) -> rpa.DecodeStateConfig:
  del num_seq_shards  # implied by the mesh
  return rpa.DecodeStateConfig(
      total_num_pages=_TOTAL_NUM_PAGES,
      page_size=_PAGE_SIZE,
      n_kv_heads=_N_KV_HEADS,
      per_head_dim=_PER_HEAD_DIM,
      batch_size=batch_size,
      dtype='float32',
      max_seq_len=_MAX_SEQ_LEN,
      window_size=window_size,
      head_partition=None,
      seq_partition='seq',
  )


def _interleave_kv(k: np.ndarray, v: np.ndarray) -> np.ndarray:
  """[T, H, D] x2 -> [T, 2H, D] with rows (k_h, v_h) interleaved."""
  out = np.zeros((k.shape[0], k.shape[1] * 2, k.shape[2]), dtype=k.dtype)
  out[:, 0::2] = k
  out[:, 1::2] = v
  return out


def _windowed_ref_attn(
    q: np.ndarray,  # [q_len, n_q_heads, d]
    k: np.ndarray,  # [kv_len, n_kv_heads, d]
    v: np.ndarray,  # [kv_len, n_kv_heads, d]
    window_size: int | None,
) -> np.ndarray:
  """Plain causal (+ sliding window) attention reference, float64."""
  q_len, n_q_heads, d = q.shape
  kv_len, n_kv_heads, _ = k.shape
  g = n_q_heads // n_kv_heads
  q64 = q.astype(np.float64).reshape(q_len, n_kv_heads, g, d)
  k64 = k.astype(np.float64)
  v64 = v.astype(np.float64)
  logits = np.einsum('tmgh,smh->tmgs', q64, k64)
  q_pos = np.arange(kv_len - q_len, kv_len)[:, None]
  kv_pos = np.arange(kv_len)[None, :]
  mask = q_pos >= kv_pos
  if window_size is not None:
    # Kernel semantics: sliding_window = window_size + 1, mask keeps
    # q_pos < kv_pos + sliding_window.
    mask = np.logical_and(mask, q_pos < kv_pos + window_size + 1)
  logits = np.where(mask[:, None, None, :], logits, -1e30)
  probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
  probs /= probs.sum(axis=-1, keepdims=True)
  out = np.einsum('tmgs,smh->tmgh', probs, v64)
  return out.reshape(q_len, n_q_heads, d)


class ShardedSlidingWindowTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    if jax.device_count() < 4:
      self.skipTest('Requires at least 4 devices.')

  def _set_mesh(self, num_seq_shards: int):
    n = jax.device_count()
    assert n % num_seq_shards == 0
    # Leftover devices go on 'data' (heads are too few to shard here).
    sharding.set_mesh(
        [1, n // num_seq_shards, num_seq_shards, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )

  # ---------------------------------------------------------------
  # 1. Every issued token's KV must be persisted in the paged cache.
  # ---------------------------------------------------------------
  @parameterized.named_parameters(
      dict(testcase_name='shards2_window1', num_seq_shards=2, window_size=1),
      dict(testcase_name='shards4_window1', num_seq_shards=4, window_size=1),
      dict(testcase_name='shards4_window7', num_seq_shards=4, window_size=7),
      dict(testcase_name='shards4_global', num_seq_shards=4, window_size=None),
  )
  def test_kv_writeback_is_complete(
      self, num_seq_shards: int, window_size: int | None
  ):
    """All `q_len` newly issued tokens must be written into `pages`.

    `q_len = 8` tokens are issued in ONE pass with a query block size of 1
    (`num_queries_per_block=1`), exactly like a long chunked-prefill pass
    where `bq_sz << q_len`.  Afterwards the paged cache must contain the
    K/V of all 8 tokens.
    """
    self._set_mesh(num_seq_shards)
    q_len = 8
    cfg = _make_config(window_size=window_size, num_seq_shards=num_seq_shards)
    ds = cfg.init()

    rng = np.random.default_rng(0)
    q = rng.normal(size=(q_len, _N_Q_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    k = rng.normal(size=(q_len, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    v = rng.normal(size=(q_len, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)

    ds_out, _ = ds.update_decode_state_and_compute_attn(
        q=RaggedArray(jnp.asarray(q), lens=jnp.array([q_len])),
        k=jnp.asarray(k),
        v=jnp.asarray(v),
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
    )
    got = ds_out.kv_np(0, per_head_dim=_PER_HEAD_DIM)
    want = _interleave_kv(k, v)
    self.assertEqual(got.shape, want.shape)
    bad = np.flatnonzero(~np.isclose(got, want, atol=1e-5).all(axis=(1, 2)))
    self.assertEmpty(
        bad.tolist(),
        f'KV cache is missing/garbled for token positions {bad.tolist()} '
        f'(num_seq_shards={num_seq_shards}, window_size={window_size}).\n'
        f'got=\n{got}\nwant=\n{want}',
    )

  # ---------------------------------------------------------------
  # 2. Sharded windowed attention output == unsharded reference.
  # ---------------------------------------------------------------
  @parameterized.named_parameters(
      dict(testcase_name='shards2_window1', num_seq_shards=2, window_size=1),
      dict(testcase_name='shards4_window1', num_seq_shards=4, window_size=1),
      dict(testcase_name='shards4_window7', num_seq_shards=4, window_size=7),
      dict(testcase_name='shards4_global', num_seq_shards=4, window_size=None),
  )
  def test_attention_matches_reference(
      self, num_seq_shards: int, window_size: int | None
  ):
    """Two-pass windowed prefill: attention must match a dense reference."""
    self._set_mesh(num_seq_shards)
    cfg = _make_config(window_size=window_size, num_seq_shards=num_seq_shards)
    ds = cfg.init()

    rng = np.random.default_rng(1)
    total = 16
    q = rng.normal(size=(total, _N_Q_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    k = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    v = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)

    outs = []
    for start in (0, 8):
      sl = slice(start, start + 8)
      ds, out = ds.update_decode_state_and_compute_attn(
          q=RaggedArray(jnp.asarray(q[sl]), lens=jnp.array([8])),
          k=jnp.asarray(k[sl]),
          v=jnp.asarray(v[sl]),
          num_kv_pages_per_block=1,
          num_queries_per_block=1,
      )
      outs.append(np.asarray(out))

    want = _windowed_ref_attn(q, k, v, window_size)
    got = np.concatenate(outs, axis=0)
    self.assertFalse(
        np.isnan(got).any(), f'NaN in sharded attention output:\n{got}'
    )
    np.testing.assert_allclose(got, want, atol=1e-2, rtol=1e-2)

  # ---------------------------------------------------------------
  # 3. Resident-but-never-written pages poison the output.
  # ---------------------------------------------------------------
  @parameterized.named_parameters(
      dict(
          testcase_name='shards4_window1_bkv2',
          num_seq_shards=4,
          window_size=1,
          bkv_pages=2,
      ),
      dict(
          testcase_name='shards4_window3_bkv2',
          num_seq_shards=4,
          window_size=3,
          bkv_pages=2,
      ),
      dict(
          testcase_name='shards4_window7_bkv2',
          num_seq_shards=4,
          window_size=7,
          bkv_pages=2,
      ),
      dict(
          testcase_name='shards4_global_bkv2',
          num_seq_shards=4,
          window_size=None,
          bkv_pages=2,
      ),
      dict(
          testcase_name='shards2_window1_bkv2',
          num_seq_shards=2,
          window_size=1,
          bkv_pages=2,
      ),
      dict(
          testcase_name='shards4_window1_bkv1',
          num_seq_shards=4,
          window_size=1,
          bkv_pages=1,
      ),
  )
  def test_stale_pages_do_not_poison_output(
      self, num_seq_shards: int, window_size: int | None, bkv_pages: int = 1
  ):
    """Uninitialised page memory must not leak into the attention output.

    `pages` starts as `jax.lax.empty` (uninitialised HBM), and released
    pages are recycled without being zeroed, so a page can hold anything
    -- including fp8/f32 NaN/Inf bit patterns.  That is harmless as long
    as every token inside `kv_lens` was actually written.  This test
    fills `pages` with NaN up front, runs two chunk-sized prefill passes,
    and requires the outputs to stay finite and correct.
    """
    self._set_mesh(num_seq_shards)
    cfg = _make_config(window_size=window_size, num_seq_shards=num_seq_shards)
    ds = cfg.init()
    # Simulate uninitialised / recycled HBM.
    ds = dataclasses.replace(ds, pages=jnp.full_like(ds.pages, jnp.nan))

    rng = np.random.default_rng(2)
    total = 16
    q = rng.normal(size=(total, _N_Q_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    k = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    v = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)

    outs = []
    for start in (0, 8):
      sl = slice(start, start + 8)
      ds, out = ds.update_decode_state_and_compute_attn(
          q=RaggedArray(jnp.asarray(q[sl]), lens=jnp.array([8])),
          k=jnp.asarray(k[sl]),
          v=jnp.asarray(v[sl]),
          num_kv_pages_per_block=bkv_pages,
          num_queries_per_block=1,
      )
      outs.append(np.asarray(out))

    got = np.concatenate(outs, axis=0)
    n_nan = int(np.isnan(got).sum())
    self.assertEqual(
        n_nan,
        0,
        f'{n_nan} NaNs leaked into the attention output '
        f'(num_seq_shards={num_seq_shards}, window_size={window_size}):\n'
        f'{got}',
    )
    want = _windowed_ref_attn(q, k, v, window_size)
    np.testing.assert_allclose(got, want, atol=1e-2, rtol=1e-2)


class BuggyLikeLoopTest(parameterized.TestCase):
  """Mini version of the production disaggregated prefill loop."""

  def setUp(self):
    super().setUp()
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    if jax.device_count() < 4:
      self.skipTest('Requires at least 4 devices.')

  def _run(
      self,
      *,
      num_seq_shards: int,
      window_size: int | None,
      total: int,
      bkv_pages: int,
      poison: float | None,
  ):
    n = jax.device_count()
    sharding.set_mesh(
        [1, n // num_seq_shards, num_seq_shards, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    cfg = _make_config(window_size=window_size, num_seq_shards=num_seq_shards)
    ds = cfg.init()
    if poison is not None:
      ds = dataclasses.replace(ds, pages=jnp.full_like(ds.pages, poison))

    rng = np.random.default_rng(3)
    q = rng.normal(size=(total, _N_Q_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    k = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    v = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)

    outs = []
    pass_lens = []
    pos = 0
    while pos < total:
      avail = int(np.asarray(ds.max_available_kv_lens)[0])
      npass = min(avail, total - pos)
      assert npass > 0, 'no capacity'
      sl = slice(pos, pos + npass)
      ds, out = ds.update_decode_state_and_compute_attn(
          q=RaggedArray(jnp.asarray(q[sl]), lens=jnp.array([npass])),
          k=jnp.asarray(k[sl]),
          v=jnp.asarray(v[sl]),
          num_kv_pages_per_block=bkv_pages,
          num_queries_per_block=1,
      )
      outs.append(np.asarray(out))
      pass_lens.append(npass)
      pos += npass
    return np.concatenate(outs, axis=0), q, k, v, pass_lens

  @parameterized.named_parameters(
      # Chunk (num_shards * page_size = 8) > sliding window (10 tokens is
      # only 5 pages) -> production-like regime (mesh 1,1,16,4).
      dict(testcase_name='shards4_window9', num_seq_shards=4, window_size=9),
      # Window >= everything issued -> healthy regime (mesh 1,1,8,4).
      dict(testcase_name='shards4_window31', num_seq_shards=4, window_size=31),
      dict(testcase_name='shards4_global', num_seq_shards=4, window_size=None),
      dict(testcase_name='shards1_window9', num_seq_shards=1, window_size=9),
      # Same broken geometry, but the recycled pages hold FINITE junk
      # (a previous request's KV) instead of uninitialised NaN: tells us
      # whether the leak is the softmax mask or `0 * NaN` in the PV matmul.
      dict(
          testcase_name='shards4_window9_finite_junk',
          num_seq_shards=4,
          window_size=9,
          poison=1e4,
      ),
      dict(
          testcase_name='shards4_window9_zero_pages',
          num_seq_shards=4,
          window_size=9,
          poison=0.0,
      ),
  )
  def test_multi_pass_prefill_with_recycled_pages(
      self,
      num_seq_shards: int,
      window_size: int | None,
      poison: float = float('nan'),
  ):
    got, q, k, v, pass_lens = self._run(
        num_seq_shards=num_seq_shards,
        window_size=window_size,
        total=32,
        bkv_pages=2,
        poison=poison,
    )
    n_nan = int(np.isnan(got).sum())
    self.assertEqual(
        n_nan,
        0,
        f'{n_nan}/{got.size} NaNs leaked into the attention output '
        f'(num_seq_shards={num_seq_shards}, window_size={window_size}, '
        f'pass_lens={pass_lens}).',
    )
    want = _windowed_ref_attn(q, k, v, window_size)
    np.testing.assert_allclose(got, want, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(
      dict(testcase_name='shards4_window9', num_seq_shards=4, window_size=9),
      dict(testcase_name='shards4_window31', num_seq_shards=4, window_size=31),
      dict(testcase_name='shards4_global', num_seq_shards=4, window_size=None),
  )
  def test_multi_pass_prefill_kv_is_persisted(
      self, num_seq_shards: int, window_size: int | None
  ):
    """After the loop, the in-window KV must all be present in `pages`."""
    n = jax.device_count()
    sharding.set_mesh(
        [1, n // num_seq_shards, num_seq_shards, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    cfg = _make_config(window_size=window_size, num_seq_shards=num_seq_shards)
    ds = cfg.init()
    ds = dataclasses.replace(ds, pages=jnp.full_like(ds.pages, jnp.nan))
    rng = np.random.default_rng(4)
    total = 24
    q = rng.normal(size=(total, _N_Q_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    k = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    v = rng.normal(size=(total, _N_KV_HEADS, _PER_HEAD_DIM)).astype(np.float32)
    ds, _ = ds.update_decode_state_and_compute_attn(
        q=RaggedArray(jnp.asarray(q), lens=jnp.array([total])),
        k=jnp.asarray(k),
        v=jnp.asarray(v),
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
    )
    got = ds.kv_np(0, per_head_dim=_PER_HEAD_DIM)
    want = _interleave_kv(k, v)[-got.shape[0] :]
    bad = np.flatnonzero(~np.isclose(got, want, atol=1e-5).all(axis=(1, 2)))
    nan_rows = np.flatnonzero(np.isnan(got).any(axis=(1, 2)))
    self.assertEmpty(
        bad.tolist(),
        f'resident KV rows {bad.tolist()} are wrong (of {got.shape[0]} '
        'resident tokens); rows still holding uninitialised NaN: '
        f'{nan_rows.tolist()} '
        f'(num_seq_shards={num_seq_shards}, window_size={window_size})',
    )


if __name__ == '__main__':
  absltest.main()

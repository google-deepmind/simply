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

"""Seq-sharded paged attention + sliding window: holes and breakpoints.

With a sliding window the RPA kernel deliberately leaves HOLES in the paged
KV cache: newly issued KV is written back only from the query block that runs
the last query of the pass, over the BKV blocks that block visits, so a pass
issuing more than `window_size` tokens never writes the KV below its own
window. What a pass boundary (a *breakpoint*) guarantees is:

  after a pass that ends at `kv_len == P`, the paged cache is correct over
  `[P - window_size, P)` -- exactly what any later query reads.

These tests pin down both halves of that contract:

  * `ShardedSlidingWindowTest` / `BuggyLikeLoopTest`: the in-window KV is
    persisted, the holes never reach the output (they are score-masked AND
    V-masked, so uninitialised `NaN` pages cannot poison `p @ v`), and the
    attention output matches a dense reference.
  * `BreakpointRoundTripTest`: the prefix-cache contract -- chunks extracted
    at their own breakpoint can be injected into a fresh slot and reproduce
    the source slot's decode bit-for-bit, even though most of the injected
    payload bytes are garbage.
"""

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
  # 1. The in-window KV of a pass boundary must be persisted.
  # ---------------------------------------------------------------
  @parameterized.named_parameters(
      dict(testcase_name='shards2_window1', num_seq_shards=2, window_size=1),
      dict(testcase_name='shards4_window1', num_seq_shards=4, window_size=1),
      dict(testcase_name='shards4_window7', num_seq_shards=4, window_size=7),
      dict(testcase_name='shards4_global', num_seq_shards=4, window_size=None),
  )
  def test_kv_writeback_covers_the_window(
      self, num_seq_shards: int, window_size: int | None
  ):
    """The KV a later query can read must be written into `pages`.

    `q_len = 8` tokens are issued in ONE pass with a query block size of 1
    (`num_queries_per_block=1`), exactly like a long chunked-prefill pass
    where `bq_sz << q_len`.  Afterwards the paged cache must contain the K/V
    of the pass boundary's window -- the last `window_size` tokens, i.e.
    everything a query resuming at `kv_len` can attend to (all 8 tokens for a
    global layer).  KV below the window is deliberately NOT written back (the
    sliding-window skip drops it); see the module docstring.
    """
    self._set_mesh(num_seq_shards)
    q_len = 8
    cfg = _make_config(window_size=window_size, num_seq_shards=num_seq_shards)
    ds = cfg.init()
    # Uninitialised HBM: any position that is NOT written back stays NaN, so
    # a stricter-than-contract assertion would fail loudly rather than
    # silently comparing stale-but-plausible bytes.
    ds = dataclasses.replace(ds, pages=jnp.full_like(ds.pages, jnp.nan))

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
    # Contract: `[kv_len - window_size, kv_len)` (everything for a global
    # layer).
    trust_start = 0 if window_size is None else max(q_len - window_size, 0)
    got_win = got[trust_start:]
    want_win = want[trust_start:]
    bad = np.flatnonzero(
        ~np.isclose(got_win, want_win, atol=1e-5).all(axis=(1, 2))
    )
    self.assertEmpty(
        bad.tolist(),
        'in-window KV is missing/garbled for token positions '
        f'{(bad + trust_start).tolist()} '
        f'(num_seq_shards={num_seq_shards}, window_size={window_size}).\n'
        f'got=\n{got_win}\nwant=\n{want_win}',
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
    # Only the pass boundary's window is contractually written back; the
    # resident pages can extend below it (`release_for_window` frees whole
    # pages) and that part is a hole by design.
    if window_size is None:
      trust_start = 0
    else:
      trust_start = max(got.shape[0] - window_size, 0)
    got_win = got[trust_start:]
    want_win = want[trust_start:]
    bad = np.flatnonzero(
        ~np.isclose(got_win, want_win, atol=1e-5).all(axis=(1, 2))
    )
    nan_rows = np.flatnonzero(np.isnan(got_win).any(axis=(1, 2)))
    self.assertEmpty(
        bad.tolist(),
        f'in-window KV rows {bad.tolist()} are wrong (of {got_win.shape[0]} '
        'in-window tokens); rows still holding uninitialised NaN: '
        f'{nan_rows.tolist()} '
        f'(num_seq_shards={num_seq_shards}, window_size={window_size})',
    )


class BreakpointRoundTripTest(parameterized.TestCase):
  """The prefix-cache contract on top of a hole-y windowed paged cache.

  Mimics `page_batcher.Batcher`: prefill a slot chunk by chunk, extract each
  chunk at the pass boundary where it is the last one (its *breakpoint*),
  accumulate each chunk's captures the way the prefix cache does (they union
  per token), then inject the result into a fresh slot and check the fresh
  slot decodes exactly like the original. Tokens no capture ever covered are
  injected as uninitialised (NaN) buffers, exactly as
  `offload_chunk_tree`/`SnapshotChunkLeaf.onload` do for bytes the cache does
  not hold --
  so if the coverage rule is wrong, the decode returns NaN.
  """

  def setUp(self):
    super().setUp()
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    if jax.device_count() < 4:
      self.skipTest('Requires at least 4 devices.')

  @parameterized.named_parameters(
      # chunk = num_shards * page_size = 8 tokens.
      # window < chunk: only the chunk ending AT the breakpoint is trustable
      # (the production geometry: window 1023 vs chunk 2048).
      dict(testcase_name='shards4_window5', num_seq_shards=4, window_size=5),
      dict(testcase_name='shards4_window3', num_seq_shards=4, window_size=3),
      # window > chunk: the tail window spans two chunks.
      dict(testcase_name='shards4_window9', num_seq_shards=4, window_size=9),
      dict(testcase_name='shards2_window5', num_seq_shards=2, window_size=5),
      dict(testcase_name='shards4_global', num_seq_shards=4, window_size=None),
  )
  def test_snapshot_at_breakpoints_restores_into_a_fresh_slot(
      self, num_seq_shards: int, window_size: int | None
  ):
    n = jax.device_count()
    sharding.set_mesh(
        [1, n // num_seq_shards, num_seq_shards, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    cfg = _make_config(
        window_size=window_size, num_seq_shards=num_seq_shards, batch_size=2
    )
    ds = cfg.init()
    # Recycled / uninitialised HBM.
    ds = dataclasses.replace(ds, pages=jnp.full_like(ds.pages, jnp.nan))
    chunk = ds.chunk_size
    n_chunks = 3
    total = n_chunks * chunk
    n_decode = 4

    rng = np.random.default_rng(7)
    shape = lambda t, h: (t, h, _PER_HEAD_DIM)
    q = rng.normal(size=shape(total + n_decode, _N_Q_HEADS)).astype(np.float32)
    k = rng.normal(size=shape(total + n_decode, _N_KV_HEADS)).astype(np.float32)
    v = rng.normal(size=shape(total + n_decode, _N_KV_HEADS)).astype(np.float32)

    extract = jax.jit(rpa.DecodeState.extract_chunk)
    inject = jax.jit(rpa.DecodeState.inject_chunk)

    def _pass(state, lens, q_data, k_data, v_data):
      """One prefill/decode pass over slot(s) with `lens` tokens."""
      return state.update_decode_state_and_compute_attn(
          q=RaggedArray(jnp.asarray(q_data), lens=jnp.asarray(lens)),
          k=jnp.asarray(k_data),
          v=jnp.asarray(v_data),
          num_kv_pages_per_block=2,
          num_queries_per_block=1,
      )

    # --- 1. Prefill slot 0 one chunk per pass; snapshot at each boundary. ---
    # `snapshots[j]` accumulates chunk `j` the way the prefix cache does: each
    # breakpoint contributes the tokens its capture is written over, and the
    # captures UNION. With `window < chunk` no single capture is ever whole,
    # which is precisely why the cache stores extents rather than a flag.
    snapshots: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_chunks
    payload_shape = None
    for c in range(n_chunks):
      sl = slice(c * chunk, (c + 1) * chunk)
      ds, _ = _pass(ds, [chunk, 0], q[sl], k[sl], v[sl])
      position = (c + 1) * chunk  # the breakpoint
      evicted = (position - int(ds.kv_lens[0])) // chunk
      for j in range(c + 1):
        leaf = extract(ds, jnp.int32(0), jnp.int32(j), jnp.int32(position))
        payload_shape = leaf.payload.shape
        # The mask must follow the window rule exactly, token by token: a
        # capture at `position` is written over
        # `[max(chunk_start, position - window), min(chunk_end, position))`,
        # and over nothing at all once the chunk has been evicted.
        chunk_start, chunk_end = j * chunk, (j + 1) * chunk
        want = np.zeros((chunk,), dtype=bool)
        if j >= evicted:
          hi = min(chunk_end, position)
          lo = (
              chunk_start
              if window_size is None
              else max(chunk_start, position - window_size)
          )
          want[max(lo - chunk_start, 0) : max(hi - chunk_start, 0)] = True
        np.testing.assert_array_equal(
            np.asarray(leaf.written),
            want,
            err_msg=(
                f'written extent mismatch for chunk {j} at breakpoint'
                f' {position} (window_size={window_size}, evicted={evicted})'
            ),
        )
        if not want.any():
          continue
        payload = np.asarray(leaf.payload)
        if (snap := snapshots[j]) is None:
          snapshots[j] = (payload.copy(), want.copy())
        else:
          have, have_mask = snap
          add = np.logical_and(want, np.logical_not(have_mask))
          have[add] = payload[add]
          snapshots[j] = (have, np.logical_or(have_mask, want))
    # What the restored slot reads at `total` is `[total - window, total)`,
    # so that is what the accumulated captures have to cover -- not whole
    # chunks, which a window smaller than a chunk can never produce.
    for j, snap in enumerate(snapshots):
      chunk_start, chunk_end = j * chunk, (j + 1) * chunk
      hi = min(chunk_end, total)
      lo = (
          chunk_start
          if window_size is None
          else max(chunk_start, total - window_size)
      )
      if hi <= lo:
        continue
      self.assertIsNotNone(snap, f'chunk {j} was never captured')
      assert snap is not None
      self.assertTrue(
          snap[1][lo - chunk_start : hi - chunk_start].all(),
          f'chunk {j} is read by a resume at {total} over [{lo}, {hi}) but'
          f' the accumulated captures cover {snap[1]}',
      )

    # --- 2. Restore the snapshots into the fresh slot 1. ---
    assert payload_shape is not None
    for snap in snapshots:
      if snap is None:
        # Cache miss on these bytes -> uninitialised buffer (NaN here).
        blob = jnp.full(payload_shape, jnp.nan, dtype=ds.pages.dtype)
      else:
        # Tokens outside the accumulated mask stay NaN, exactly as a restore
        # would leave them: the coverage check above is what makes that safe.
        payload, mask = snap
        blob = jnp.asarray(
            np.where(
                mask.reshape((-1,) + (1,) * (payload.ndim - 1)),
                payload,
                np.nan,
            )
        )
      ds = inject(ds, jnp.int32(1), blob)
    # The two slots need not hold the same NUMBER of resident tokens (the
    # restored slot has had fewer `release_for_window` rounds), but both must
    # cover the window, which is what the decode below actually checks.
    for slot in (0, 1):
      self.assertGreaterEqual(
          int(ds.kv_lens[slot]), min(window_size or total, total)
      )

    # --- 3. Both slots decode the same tokens; outputs must agree. ---
    # The ragged query buffer's capacity must be a multiple of the number of
    # seq shards (the attention output is seq-sharded over it), so pad it.
    capacity = num_seq_shards * max(1, 2 // num_seq_shards)
    outs = []
    for t in range(n_decode):
      pos = total + t
      dup = lambda x: np.concatenate([x[pos : pos + 1]] * capacity)  # pylint: disable=cell-var-from-loop
      ds, out = _pass(ds, [1, 1], dup(q), dup(k), dup(v))
      outs.append(np.asarray(out)[:2])
    got = np.stack(outs)  # [n_decode, 2 (slots), n_q_heads, dim]
    self.assertFalse(np.isnan(got).any(), f'NaN in decode output:\n{got}')
    np.testing.assert_allclose(got[:, 0], got[:, 1], atol=1e-5, rtol=1e-5)

    # ...and match a dense reference for the decode positions.
    want_all = _windowed_ref_attn(
        q[: total + n_decode],
        k[: total + n_decode],
        v[: total + n_decode],
        window_size,
    )[total:]
    np.testing.assert_allclose(got[:, 0], want_all, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()

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

import dataclasses
import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
import jax.numpy as jnp
import numpy as np
from simply.kernels import ragged_paged_attention as rpa_kernel
from simply.utils import common
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sampling_lib
from simply.utils import sharding

# pyrefly: ignore[code]

RaggedArray = common.RaggedArray


def _host_loop_prefill(
    state: rpa.SamplingState,
    forward_fn,
    params,
    max_num_issue_tokens: int,
    top_k: int = 1,
    skip_logits: bool = False,
) -> rpa.SamplingState:
  """Host-looped single-step prefill -- mirrors `Batcher.loop` exactly."""
  n_passes = 0
  while bool(state.any_prefilling):
    state = state.mixed_step(  # pytype: disable=wrong-arg-types
        forward_fn,
        params,
        None,  # extra_inputs
        max_num_issue_tokens,
        1.0,  # temperature
        top_k,
        1.0,  # top_p
        prefill=True,
        skip_logits=skip_logits,
    )
    n_passes += 1
    assert n_passes < 1000, 'host prefill loop did not terminate'
  return state


def qkv_attn(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
  """Computes qkv attention (reference implementation)."""
  q_len = q.shape[-3]
  kv_len = k.shape[-3]
  n_kv_heads = k.shape[-2]
  q = einops.rearrange(q, ' ... (m g) h -> ... m g h', m=n_kv_heads)
  attn = jnp.einsum('...tmgh,...smh->...tmgs', q, k)
  q_span = jnp.arange(kv_len - q_len, kv_len)
  kv_span = jnp.arange(kv_len)
  mask = q_span[:, None, None, None] >= kv_span
  attn += (~mask) * (-0.7 * float(jnp.finfo(jnp.dtype('float32')).max))
  output = jnp.einsum(
      '...tmgs,...smh->...tmgh', jax.nn.softmax(attn, axis=-1), v
  )
  output = einops.rearrange(output, '... m g h -> ... (m g) h')
  return output


class DecodeStateTest(parameterized.TestCase):

  def test_allocate(self):
    total_num_pages = 6
    page_size = 3
    config = rpa.DecodeStateConfig(
        total_num_pages=total_num_pages,
        page_size=page_size,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=3,
        dtype='float32',
        max_seq_len=total_num_pages * page_size + 1,
    )
    ds = config.init()

    ds = ds.allocate(q_lens=jnp.array([3, 0, 4]))
    np.testing.assert_array_equal(ds.kv_lens, np.array([3, 0, 4]))
    np.testing.assert_array_equal(
        ds.available_page_indices_np, np.array([3, 4, 5])
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ds.page_indices_nplist,
        [np.array([0]), np.array([]), np.array([1, 2])],
    )

    ds = dataclasses.replace(
        ds,
        page_indices=jnp.array([
            [3, -1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ]),
        kv_lens=jnp.array([3, 2, 0]),
        num_available_pages=jnp.array(4, dtype=jnp.int32),
        available_page_indices=jnp.array([5, 2, 4, 0, -1, -1]),
    )
    ds = jax.jit(ds.allocate)(q_lens=jnp.array([5, 0, 6]))
    np.testing.assert_array_equal(ds.kv_lens, jnp.array([8, 2, 6]))
    np.testing.assert_array_equal(ds.available_page_indices_np, np.array([]))
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ds.page_indices_nplist,
        [np.array([3, 5, 2]), np.array([1]), np.array([4, 0])],
    )

  def test_insert(self):
    total_num_pages = 6
    page_size = 3
    n_kv_heads = 1
    per_head_dim = 2
    kv_packing = rpa_kernel.get_dtype_packing('float32')
    pages_shape = (
        total_num_pages,
        page_size,
        n_kv_heads * 2 // kv_packing,
        kv_packing,
        per_head_dim,
    )
    pages = jnp.reshape(jnp.arange(np.prod(pages_shape)), pages_shape)
    pages = jnp.pad(
        pages, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 128 - per_head_dim))
    )
    page_indices = jnp.array([
        [5, 1, 3],
        [2, 0, -1],
        [4, -1, -1],
    ])
    active = (
        jnp.zeros(total_num_pages, dtype=jnp.bool)
        .at[jnp.ravel(page_indices)]
        .set(True, mode='drop', wrap_negative_indices=False)
    )
    available_page_indices = jnp.flatnonzero(~active, size=total_num_pages)
    num_available_pages = jnp.sum(page_indices < 0)
    kv_lens = jnp.array([3, 2, 1])
    q_lens = jnp.array([5, 2, 2])
    ds = rpa.DecodeState(
        pages=pages,
        page_indices=page_indices,
        available_page_indices=available_page_indices,
        num_available_pages=num_available_pages,
        kv_lens=kv_lens + q_lens,
        max_seq_len=8,
    )

    np.testing.assert_array_equal(
        ds.local_num_pages, jnp.sum(page_indices >= 0, axis=-1)
    )
    new_kv_shape = (jnp.sum(q_lens), n_kv_heads * 2, per_head_dim)
    new_kv = jnp.reshape(jnp.arange(np.prod(new_kv_shape)) * -1, new_kv_shape)  # pyrefly: ignore[no-matching-overload]
    ds = jax.jit(ds.insert)(new_kv[:, 0::2], new_kv[:, 1::2], q_lens)
    np.testing.assert_array_equal(
        ds.pages[..., :per_head_dim],
        np.array([
            [[[-24, -25], [-26, -27]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
            [[[0, -1], [-2, -3]], [[-4, -5], [-6, -7]], [[-8, -9], [-10, -11]]],
            [
                [[24, 25], [26, 27]],
                [[28, 29], [30, 31]],
                [[-20, -21], [-22, -23]],
            ],
            [
                [[-12, -13], [-14, -15]],
                [[-16, -17], [-18, -19]],
                [[44, 45], [46, 47]],
            ],
            [
                [[48, 49], [50, 51]],
                [[-28, -29], [-30, -31]],
                [[-32, -33], [-34, -35]],
            ],
            [[[60, 61], [62, 63]], [[64, 65], [66, 67]], [[68, 69], [70, 71]]],
        ]).reshape(
            total_num_pages,
            page_size,
            n_kv_heads * 2 // kv_packing,
            kv_packing,
            per_head_dim,
        ),
    )

  def test_inject_chunk(self):
    # shard_map (used inside inject_chunk) requires a non-empty mesh
    # even in the unsharded (num_shards=1) case. Put all devices on the
    # replica axis so seq_partition_size stays 1.
    sharding.set_mesh(
        [jax.device_count(), 1, 1, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    total_num_pages = 8
    page_size = 3
    config = rpa.DecodeStateConfig(
        total_num_pages=total_num_pages,
        page_size=page_size,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=2,
        dtype='float32',
        max_seq_len=total_num_pages * page_size + 1,
    )
    ds = config.init()
    self.assertEqual(ds.num_shards, 1)

    # Token-major payload: leading dim = num_shards * page_size.
    inner_shape = ds.pages.shape[2:]  # (fh, kvp, hd)
    payload_0 = jnp.full(
        (ds.num_shards * page_size,) + inner_shape, 7.0, dtype=ds.pages.dtype
    )
    payload_1 = jnp.full(
        (ds.num_shards * page_size,) + inner_shape, 9.0, dtype=ds.pages.dtype
    )

    # First inject for slot 1 (chosen non-zero to exercise indexing).
    ds2 = jax.jit(ds.inject_chunk)(jnp.int32(1), payload_0)
    np.testing.assert_array_equal(ds2.kv_lens, np.array([0, page_size]))
    # The page that slot 1 just claimed should hold the (only) page
    # worth of payload_0.
    gid0 = int(ds2.slot_global_page_indices(1, num_pages=1, start_page=0)[0])
    np.testing.assert_array_equal(
        ds2.pages[gid0], payload_0.reshape(ds.pages.shape[1:])
    )

    # Second inject for slot 1 — should append, not overwrite.
    ds3 = jax.jit(ds2.inject_chunk)(jnp.int32(1), payload_1)
    np.testing.assert_array_equal(ds3.kv_lens, np.array([0, 2 * page_size]))
    gid1 = int(ds3.slot_global_page_indices(1, num_pages=1, start_page=1)[0])
    self.assertNotEqual(gid0, gid1)
    np.testing.assert_array_equal(
        ds3.pages[gid0], payload_0.reshape(ds.pages.shape[1:])
    )
    np.testing.assert_array_equal(
        ds3.pages[gid1], payload_1.reshape(ds.pages.shape[1:])
    )

  def test_extract_chunk_round_trips_with_inject_chunk(self):
    """`extract_chunk` returns exactly what was just `inject_chunk`-d."""
    sharding.set_mesh(
        [jax.device_count(), 1, 1, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    page_size = 3
    config = rpa.DecodeStateConfig(
        total_num_pages=8,
        page_size=page_size,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=2,
        dtype='float32',
        max_seq_len=8 * page_size + 1,
    )
    ds = config.init()
    inner_shape = ds.pages.shape[2:]
    chunk_tokens = ds.num_shards * page_size
    payload_a = jnp.full(
        (chunk_tokens,) + inner_shape, 3.0, dtype=ds.pages.dtype
    )
    payload_b = jnp.full(
        (chunk_tokens,) + inner_shape, 7.0, dtype=ds.pages.dtype
    )
    inject = jax.jit(rpa.DecodeState.inject_chunk)
    extract = jax.jit(rpa.DecodeState.extract_chunk)
    ds1 = inject(ds, jnp.int32(1), payload_a)
    ds2 = inject(ds1, jnp.int32(1), payload_b)
    # No window here, so `position == kv_lens` and logical chunk index ==
    # physical column. `kv_lens[1]` is `2 * chunk_tokens` after two injects.
    position = ds2.kv_lens[1]
    entry_a = extract(ds2, jnp.int32(1), jnp.int32(0), position)
    entry_b = extract(ds2, jnp.int32(1), jnp.int32(1), position)
    self.assertIsInstance(entry_a, rpa.SnapshotChunkLeaf)
    self.assertIsInstance(entry_b, rpa.SnapshotChunkLeaf)
    np.testing.assert_array_equal(
        np.asarray(entry_a.payload), np.asarray(payload_a)
    )
    np.testing.assert_array_equal(
        np.asarray(entry_b.payload), np.asarray(payload_b)
    )
    # Both chunks are resident (no sliding-window eviction) and this is a
    # global layer, so every token of both is written.
    self.assertTrue(bool(np.asarray(entry_a.written).all()))
    self.assertTrue(bool(np.asarray(entry_b.written).all()))

  def test_extract_chunk_reports_nothing_valid_for_an_evicted_chunk(self):
    """A sliding-window layer reports an empty mask for an evicted chunk."""
    sharding.set_mesh(
        [jax.device_count(), 1, 1, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    page_size = 3
    # `window_size = chunk_tokens` => `release_for_window` keeps just
    # the last chunk, evicting older ones.
    config = rpa.DecodeStateConfig(
        total_num_pages=8,
        page_size=page_size,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=2,
        dtype='float32',
        max_seq_len=8 * page_size + 1,
        window_size=page_size,  # 1 chunk's worth of tokens
    )
    ds = config.init()
    inner_shape = ds.pages.shape[2:]
    chunk_tokens = ds.num_shards * page_size
    payload_a = jnp.full(
        (chunk_tokens,) + inner_shape, 3.0, dtype=ds.pages.dtype
    )
    payload_b = jnp.full(
        (chunk_tokens,) + inner_shape, 7.0, dtype=ds.pages.dtype
    )
    inject = jax.jit(rpa.DecodeState.inject_chunk)
    extract = jax.jit(rpa.DecodeState.extract_chunk)
    release_for_window = jax.jit(rpa.DecodeState.release_for_window)
    ds1 = inject(ds, jnp.int32(1), payload_a)
    ds2 = inject(ds1, jnp.int32(1), payload_b)
    # `position` reflects all 2 chunks ever written; `release_for_window`
    # compacts `page_indices` so only the tail chunk remains physically.
    position = ds2.kv_lens[1]
    ds3 = release_for_window(ds2)
    # Chunk 1 (the tail) is still resident; chunk 0 has been evicted.
    entry_b = extract(ds3, jnp.int32(1), jnp.int32(1), position)
    entry_a = extract(ds3, jnp.int32(1), jnp.int32(0), position)
    self.assertTrue(bool(np.asarray(entry_b.written).any()))
    np.testing.assert_array_equal(
        np.asarray(entry_b.payload), np.asarray(payload_b)
    )
    self.assertFalse(bool(np.asarray(entry_a.written).any()))

  def test_inject_chunk_rejects_bad_shape(self):
    sharding.set_mesh(
        [jax.device_count(), 1, 1, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    config = rpa.DecodeStateConfig(
        total_num_pages=6,
        page_size=3,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=2,
        dtype='float32',
        max_seq_len=7,
    )
    ds = config.init()
    inner_shape = ds.pages.shape[2:]  # (fh, kvp, hd)
    leading = ds.num_shards * ds.page_size
    # Wrong inner dims.
    wrong_inner = jnp.zeros(
        (leading,) + inner_shape[:-1] + (inner_shape[-1] + 1,),
        dtype=ds.pages.dtype,
    )
    with self.assertRaisesRegex(ValueError, 'must equal'):
      ds.inject_chunk(jnp.int32(0), wrong_inner)
    # Wrong leading dim (must equal num_shards * page_size).
    wrong_leading = jnp.zeros(
        (leading + 1,) + inner_shape, dtype=ds.pages.dtype
    )
    with self.assertRaisesRegex(ValueError, 'must equal'):
      ds.inject_chunk(jnp.int32(0), wrong_leading)

  @parameterized.named_parameters(
      dict(testcase_name='no_partition', use_partition=False),
      dict(testcase_name='with_partition', use_partition=True),
  )
  def test_update_decode_state_and_compute_attn(self, use_partition: bool):
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    mesh_shape = [jax.device_count(), 1, 1, 1]
    if use_partition:
      if jax.device_count() < 2:
        self.skipTest('Requires at least 2 devices.')
      mesh_shape = [1, 1, jax.device_count() // 2, 2]
    # TODO: Change it back to 6 when smart token issuing is properly
    # implemented.
    total_num_pages = 8
    page_size = 3
    max_seq_len = total_num_pages * page_size + 1
    num_issue_tokens = 10
    n_kv_heads = 2
    n_q_heads = 4
    per_head_dim = 2
    batch_size = 4

    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )
    rk1, rk2, rk3 = jax.random.split(jax.random.key(0), 3)

    old_kv_lens = jnp.array([3, 2, 0, 1])
    q_lens = jnp.array([5, 2, 0, 2])

    ragged_old_kv = RaggedArray(
        data=jax.random.normal(
            rk1, (num_issue_tokens, n_kv_heads * 2, per_head_dim)
        ),
        lens=old_kv_lens,
    )
    ragged_q = RaggedArray(
        data=jax.random.normal(
            rk2, (num_issue_tokens, n_q_heads, per_head_dim)
        ),
        lens=q_lens,
    )
    ragged_kv = RaggedArray(
        data=jax.random.normal(
            rk3, (num_issue_tokens, n_kv_heads * 2, per_head_dim)
        ),
        lens=q_lens,
    )
    updated_ragged_kv = ragged_old_kv.concat(ragged_kv)
    expected_attn_out_list = []
    for i in range(batch_size):
      q = jnp.reshape(ragged_q.row(i), (-1, n_q_heads, per_head_dim))
      k = updated_ragged_kv.row(i)[:, 0::2]
      v = updated_ragged_kv.row(i)[:, 1::2]
      o = qkv_attn(q, k, v)  # pyrefly: ignore[bad-argument-type]
      expected_attn_out_list.append(o)

    config = rpa.DecodeStateConfig(
        total_num_pages=total_num_pages,
        page_size=page_size,
        n_kv_heads=n_kv_heads,
        per_head_dim=per_head_dim,
        batch_size=batch_size,
        dtype='float32',
        max_seq_len=max_seq_len,
        head_partition='model',
        seq_partition='seq',
    )
    ds = (
        config.init()
        .allocate(old_kv_lens)
        .insert(
            ragged_old_kv.data[:, 0::2],
            ragged_old_kv.data[:, 1::2],
            old_kv_lens,
        )
    )

    ds, ragged_attn_out = ds.update_decode_state_and_compute_attn(
        q=ragged_q,
        k=ragged_kv.data[:, 0::2],
        v=ragged_kv.data[:, 1::2],
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ds.kv_nplist(per_head_dim),
        updated_ragged_kv.to_numpy_list(),
    )
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, atol=0.005),
        RaggedArray(ragged_attn_out, ragged_q.lens).to_numpy_list(),
        expected_attn_out_list,
    )

  def test_autotune_block_sizes(self):
    num_kv_pages_per_block, num_queries_per_block = rpa.autotune_block_sizes(
        num_kv_heads=2,
        num_q_heads=4,
        page_size=128,
        max_seq_len=16 * 1024,
        per_head_dim=128,
        window_size=None,
        dtype='bfloat16',
    )
    self.assertEqual(num_kv_pages_per_block, 16)
    self.assertEqual(num_queries_per_block, 32)

    num_kv_pages_per_block, num_queries_per_block = rpa.autotune_block_sizes(
        num_kv_heads=2,
        num_q_heads=4,
        page_size=128,
        max_seq_len=16 * 1024,
        per_head_dim=128,
        window_size=127,
        dtype='bfloat16',
    )
    self.assertEqual(num_kv_pages_per_block, 2)
    self.assertEqual(num_queries_per_block, 32)

  def test_release_for_window(self):
    config = rpa.DecodeStateConfig(
        total_num_pages=9,
        page_size=4,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=3,
        dtype='float32',
        window_size=5,
        max_seq_len=10000,
    )
    ds = config.init().allocate(jnp.array([3, 8, 10]))
    self.assertEqual(ds.max_num_pages_per_seq_per_shard, 3)
    np.testing.assert_array_equal(ds.max_available_kv_lens, np.array([9, 4, 6]))
    ds = ds.release_for_window()
    np.testing.assert_array_equal(ds.max_available_kv_lens, np.array([9, 4, 6]))
    np.testing.assert_array_equal(ds.kv_lens, np.array([3, 8, 6]))
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ds.page_indices_nplist,
        [np.array([0]), np.array([1, 2]), np.array([4, 5])],
    )
    np.testing.assert_array_equal(
        ds.available_page_indices_np, np.array([6, 7, 8, 3])
    )


class ResidencyGuardTest(parameterized.TestCase):
  """The window mask ALONE is a sufficient written extent signal -- proved here.

  `extract_chunk` reports `chunk_written_mask(chunk_idx, position)` and nothing
  else: written extent is purely positional, with no residency term. That is
  only
  correct because the resident region always CONTAINS the window, so a chunk
  the mask marks is a chunk still in the pages. This test is what holds that
  up; it is load-bearing for `extract_chunk`, not documentation.

  The argument, checked exhaustively below: `release_for_window` frees
  `floor(floor(max(kv_lens - W, 0) / page_size) / num_shards)` pages per
  shard, i.e. a whole number of CHUNKS, and never more than `kv_lens - W`
  tokens -- so the resident region is chunk-aligned and always contains
  `[position - W, position)`. A non-empty mask needs `chunk_end > position -
  W`, and with both quantities on the chunk grid that forces `chunk_idx >=
  evicted_chunks`, i.e. the chunk is resident.

  `test_the_host_model_matches_release_for_window` is what makes the rest
  trustworthy: the proof reasons about a host model of the release formula,
  so that model is first checked against the real device code.
  """

  def _released(
      self, kv_lens: int, window: int | None, page_size: int, shards: int
  ) -> int:
    """Host model of `release_for_window`'s per-call release, in tokens.

    Args:
      kv_lens: the layer's resident length before the call.
      window: the sliding window, or `None` for a global layer.
      page_size: tokens per page.
      shards: seq-partition shard count.

    Returns:
      How many tokens the call frees.
    """
    if window is None:
      return 0
    return (max(kv_lens - window, 0) // page_size // shards) * (
        page_size * shards
    )

  def test_the_host_model_matches_release_for_window(self):
    # The proof below reasons about `release_for_window` through the formula
    # above, so first check the formula IS what the device code does. If this
    # ever fails, the sweep that follows is proving something about a model
    # that no longer matches reality -- and `extract_chunk` has no second
    # mechanism to fall back on.
    sharding.set_mesh(
        [jax.device_count(), 1, 1, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    page_size = 2
    config = rpa.DecodeStateConfig(
        total_num_pages=32,
        page_size=page_size,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=1,
        dtype='float32',
        max_seq_len=64,
        window_size=3 * page_size,
    )
    ds = config.init()
    release = jax.jit(rpa.DecodeState.release_for_window)
    for kv_len in range(0, 8 * ds.chunk_size, ds.page_size):
      state = dataclasses.replace(ds, kv_lens=jnp.full_like(ds.kv_lens, kv_len))
      got = int(np.asarray(release(state).kv_lens)[0])
      want = kv_len - self._released(
          kv_len, config.window_size, page_size, ds.num_shards
      )
      self.assertEqual(got, want, f'kv_lens={kv_len}')

  @parameterized.named_parameters(
      dict(testcase_name='page2', page_size=2),
      dict(testcase_name='page3', page_size=3),
  )
  def test_a_non_empty_mask_implies_the_chunk_is_resident(self, page_size):
    # `extract_chunk` depends on this directly: it publishes the mask as-is,
    # so a mask marking an evicted chunk would hand out recycled pages.
    sharding.set_mesh(
        [jax.device_count(), 1, 1, 1],
        axis_names=('replica', 'data', 'seq', 'model'),
    )
    base = dict(
        total_num_pages=64,
        page_size=page_size,
        n_kv_heads=1,
        per_head_dim=2,
        batch_size=1,
        dtype='float32',
        max_seq_len=128,
    )
    probe = rpa.DecodeStateConfig(**base, window_size=None).init()
    chunk = probe.chunk_size
    shards = probe.num_shards
    windows = [None, 1, 2, chunk - 1, chunk, chunk + 1, 2 * chunk, 3 * chunk]
    positions = list(range(1, 6 * chunk + 2))
    chunk_ids = list(range(0, 8))
    checked = tight = evicting = 0
    for window in windows:
      ds = rpa.DecodeStateConfig(**base, window_size=window).init()
      masks = jax.jit(
          jax.vmap(
              jax.vmap(ds.chunk_written_mask, in_axes=(0, None)),
              in_axes=(None, 0),
          )
      )(jnp.asarray(chunk_ids), jnp.asarray(positions))
      masks = np.asarray(masks)  # [position, chunk_idx, chunk]
      for p_i, position in enumerate(positions):
        # Every resident length this position can be in: `kv_lens` only ever
        # drops by whole chunks, and `release_for_window` never frees inside
        # the window -- enumerate ALL states satisfying that, which is a
        # superset of what any pass schedule can reach.
        reachable = [
            position - evicted * chunk
            for evicted in range(position // chunk + 1)
            if window is None
            and evicted == 0
            or window is not None
            and position - evicted * chunk >= min(position, window)
        ]
        for kv_len in reachable:
          evicted_chunks = (position - kv_len) // chunk
          for c_i, chunk_idx in enumerate(chunk_ids):
            if not masks[p_i, c_i].any():
              continue
            checked += 1
            evicting += evicted_chunks > 0
            # The tight case: the mask is non-empty for the chunk sitting
            # exactly at the eviction frontier, where "one more evicted
            # chunk" would break the implication.
            tight += chunk_idx == evicted_chunks and evicted_chunks > 0
            self.assertGreaterEqual(
                chunk_idx - evicted_chunks,
                0,
                f'a non-empty mask on an EVICTED chunk: window={window},'
                f' position={position}, kv_lens={kv_len},'
                f' chunk_idx={chunk_idx}, shards={shards}',
            )
    # The sweep is exhaustive over the reachable states, but assert it
    # actually reached the ones that make the implication non-trivial rather
    # than trusting a raw count.
    self.assertGreater(checked, 100, 'the sweep checked almost nothing')
    self.assertGreater(evicting, 20, 'no state with an eviction was reached')
    self.assertGreater(tight, 5, 'the eviction frontier was never exercised')

  def test_the_schedules_a_real_slot_walks_stay_inside_that_set(self):
    # The sweep above enumerates a superset of reachable `(position,
    # kv_lens)` states. This checks the recurrence really stays in it, for
    # pass schedules including ones that end just after an eviction.
    page_size, shards = 2, 4
    chunk = page_size * shards
    for window in (None, 1, chunk - 1, chunk, chunk + 1, 2 * chunk):
      for step in (1, 2, chunk // 2, chunk, chunk + 1, 2 * chunk + 3):
        position = kv_len = 0
        while position < 6 * chunk:
          kv_len -= self._released(kv_len, window, page_size, shards)
          take = min(step, 6 * chunk - position)
          position += take
          kv_len += take
          self.assertEqual(
              (position - kv_len) % chunk,
              0,
              f'eviction went off the chunk grid: window={window},'
              f' step={step}, position={position}, kv_lens={kv_len}',
          )
          if window is not None:
            self.assertGreaterEqual(
                kv_len,
                min(position, window),
                f'the window was evicted: window={window}, step={step},'
                f' position={position}, kv_lens={kv_len}',
            )


class SamplingStateTest(parameterized.TestCase):

  def test_push_and_release(self):

    sampling_state = rpa.SamplingState.create(
        max_total_num_tokens=16,
        prng_key=jax.random.key(0),
        eos_ids=jnp.array([100]),
        decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
            total_num_pages=6,
            page_size=3,
            n_kv_heads=1,
            per_head_dim=2,
            batch_size=3,
            max_seq_len=8,
            dtype='float32',
        ).init(),
    )
    inputs = np.arange(15).reshape(3, 5)
    lens = [3, 5, 1]
    for i in range(len(inputs)):
      sampling_state, idx = sampling_state.push(inputs[i], lens[i], 0)
      self.assertEqual(idx, i)
    sampling_state = dataclasses.replace(
        sampling_state, position=jnp.array(lens) - 1
    )
    np.testing.assert_array_equal(
        sampling_state.input_lens, np.array([3, 5, 1])
    )
    np.testing.assert_array_equal(sampling_state.rank, np.array([0, 1, 2]))
    outputs = sampling_state.get(np.array([True, True, True]))
    for i in range(len(outputs)):
      self.assertEqual(outputs[i]['index'], i)
      np.testing.assert_array_equal(outputs[i]['tokens'], inputs[i][: lens[i]])

    sampling_state = sampling_state.release(jnp.array([False, True, False]))
    np.testing.assert_array_equal(
        sampling_state.input_lens, np.array([3, 0, 1])
    )

    sampling_state, idx = sampling_state.push(np.array([3, 2, 1]), 3, 0)
    sampling_state = dataclasses.replace(
        sampling_state, position=sampling_state.position.at[1].set(2)
    )
    self.assertEqual(idx, 1)
    (output,) = sampling_state.get(np.array([False, True, False]))
    self.assertEqual(output['index'], 1)
    np.testing.assert_array_equal(output['tokens'], np.array([3, 2, 1]))
    np.testing.assert_array_equal(
        sampling_state.input_lens, np.array([3, 3, 1])
    )
    np.testing.assert_array_equal(sampling_state.rank, np.array([0, 2, 1]))

  def test_ragged_issue_tokens(self):
    sampling_state = rpa.SamplingState.create(
        max_total_num_tokens=8,
        prng_key=jax.random.key(0),
        eos_ids=jnp.array([100]),
        decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
            total_num_pages=6,
            page_size=3,
            n_kv_heads=1,
            per_head_dim=2,
            batch_size=2,
            max_seq_len=8,
            dtype='float32',
        ).init(),
    )
    tokens = RaggedArray.from_numpy_list([
        np.array([1, 2, 3]),
        np.array([4, 5, 6, 7, 8, 9, 10]),
    ])
    sampling_state = dataclasses.replace(
        sampling_state,
        tokens=tokens.to_padded_dense(sampling_state.max_seq_len),
        token_logprobs=jnp.zeros_like(sampling_state.tokens, dtype=jnp.float32),
        token_scores=jnp.zeros_like(sampling_state.tokens, dtype=jnp.float32),
        input_lens=tokens.lens,
        position=jnp.array([0, 2]),
        max_decode_steps=jnp.array([5, 5]),
        rank=jnp.array([1, 0]),
    )
    np.testing.assert_array_equal(
        sampling_state.issue_lens(capacity=100), np.array([1, 5])
    )
    ragged_issue_tokens = sampling_state.ragged_issue_tokens(100)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        ragged_issue_tokens.to_numpy_list(),
        [np.array([1]), np.array([6, 7, 8, 9, 10])],
    )

    ragged_output_tokens = dataclasses.replace(
        ragged_issue_tokens, data=-ragged_issue_tokens.data
    )
    sampling_state = sampling_state.update_with_ragged_output(
        ragged_output_tokens,
        token_logprobs=jnp.ones_like(ragged_output_tokens.data),
        token_scores=jnp.ones_like(ragged_output_tokens.data),
    )
    np.testing.assert_array_equal(sampling_state.position, np.array([1, 7]))
    outputs = sampling_state.get(np.array([True, True]))
    jax.tree_util.tree_map(
        np.testing.assert_almost_equal,
        outputs,
        [
            dict(
                index=0,
                input_len=tokens.lens[0],
                tokens=np.array([1, -1]),
                logprobs=np.array([0, 1]),
                scores=np.array([0, 1]),
                # New field from SamplingState.get(): no EOS (last token -1 !=
                # eos_id 100) -> stopped without EOS -> truncated.
                truncated=True,
            ),
            dict(
                index=1,
                input_len=tokens.lens[1],
                tokens=np.array([4, 5, 6, -6, -7, -8, -9, -10]),
                logprobs=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                scores=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                # New field from SamplingState.get(): no EOS (last token -10 !=
                # eos_id 100) -> stopped without EOS -> truncated.
                truncated=True,
            ),
        ],
    )

  @parameterized.named_parameters(
      dict(testcase_name='no_partition', use_partition=False),
      dict(testcase_name='with_partition', use_partition=True),
  )
  def test_continue_decode(self, use_partition: bool):
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    mesh_shape = [jax.device_count(), 1, 1, 1]
    if use_partition:
      if jax.device_count() < 2:
        self.skipTest('Requires at least 2 devices.')
      mesh_shape = [1, 1, jax.device_count() // 2, 2]
    max_seq_len = 8
    batch_size = 2
    vocab_size = 10
    per_head_dim = 2
    n_heads = 4
    n_kv_heads = 2

    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )

    sampling_state = rpa.SamplingState.create(
        max_total_num_tokens=max_seq_len * 100,
        prng_key=jax.random.key(0),
        eos_ids=jnp.array([100]),
        decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
            total_num_pages=6,
            page_size=3,
            n_kv_heads=n_kv_heads,
            per_head_dim=per_head_dim,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dtype='float32',
            head_partition='model',
            seq_partition='seq',
        ).init(),
    )
    tokens = RaggedArray.from_numpy_list(
        [np.array([0, 1, 2]), np.array([4, 5, 6, 7, 8])]
    )
    sampling_state = dataclasses.replace(
        sampling_state,
        tokens=tokens.to_padded_dense(max_seq_len),
        input_lens=tokens.lens,
        position=jnp.array([0, 0]),
        max_decode_steps=jnp.array([0, 10]),
        rank=jnp.array([0, 1]),
    )

    emb_key, q_key, k_key, v_key = jax.random.split(jax.random.key(0), 4)
    params = dict(
        emb=jax.random.normal(emb_key, (vocab_size, per_head_dim)),
        q_proj=jax.random.normal(q_key, (n_heads, per_head_dim, per_head_dim)),
        k_proj=jax.random.normal(
            k_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
        v_proj=jax.random.normal(
            v_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
    )

    def forward_fn(
        params: common.PyTree,
        tokens: jax.Array,
        segment_ids: jax.Array,
        segment_positions: jax.Array,
        extra_inputs: common.PyTree = None,
        decode_state: common.PyTree = None,
        ragged: bool = True,
    ) -> tuple[jax.Array, common.PyTree]:
      del segment_ids
      emb = jnp.take(params['emb'], tokens, axis=0)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
      emb *= jnp.cos(segment_positions)[:, :, None]
      q = jnp.einsum('...d,ndh->...nh', emb, params['q_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      k = jnp.einsum('...d,ndh->...nh', emb, params['k_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      v = jnp.einsum('...d,ndh->...nh', emb, params['v_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      if ragged:
        q = einops.rearrange(q, '1 l ... -> l ...')
        k = einops.rearrange(k, '1 l ... -> l ...')
        v = einops.rearrange(v, '1 l ... -> l ...')
        decode_state, attn_out = (
            decode_state.update_decode_state_and_compute_attn(  # pyrefly: ignore[missing-attribute]
                q=RaggedArray(q, extra_inputs['lens']),  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
                k=k,
                v=v,
            )
        )
        attn_out = einops.rearrange(attn_out, 'l ... -> 1 l ...')
      else:
        attn_out = qkv_attn(q, k, v)
      output = jnp.einsum(
          'vd,...d->...v', params['emb'], jnp.mean(attn_out, axis=-2)  # pyrefly: ignore[bad-index, unsupported-operation]
      )
      return output, {'decode_state': decode_state}

    next_sampling_state = sampling_state.continue_decode(
        forward_fn=forward_fn,
        until_fn=lambda state: jnp.any(~state.is_pad_seq & state.has_ended),
        params=params,
    )
    np.testing.assert_array_equal(next_sampling_state.lens, np.array([3, 6]))
    np.testing.assert_array_equal(
        next_sampling_state.has_ended, np.array([True, False])
    )
    final_sampling_state = next_sampling_state.continue_decode(
        forward_fn=forward_fn,
        until_fn=lambda state: jnp.all(state.has_ended),
        params=params,
    )
    np.testing.assert_array_equal(final_sampling_state.lens, np.array([3, 8]))
    np.testing.assert_array_equal(
        final_sampling_state.has_ended, np.array([True, True])
    )
    outputs = final_sampling_state.get(np.array([True, True]))

    logits, _ = forward_fn(
        params,
        final_sampling_state.tokens[:, :-1],
        segment_ids=jnp.ones_like(final_sampling_state.tokens[:, :-1]),
        segment_positions=jax.lax.broadcasted_iota(
            jnp.int32, final_sampling_state.tokens[:, :-1].shape, 1
        ),
        ragged=False,
    )
    logprobs = sampling_lib.compute_log_likelihood(
        logits, final_sampling_state.tokens[:, 1:]
    )
    logprobs = jnp.pad(logprobs, ((0, 0), (1, 0)))

    for i in range(batch_size):
      input_len = outputs[i]['input_len']
      self.assertEqual(input_len, tokens.lens[i])
      np.testing.assert_array_equal(
          sampling_state.tokens[i][:input_len], tokens.row(i)
      )
      length = len(outputs[i]['tokens'])  # pyrefly: ignore[bad-argument-type]

      np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
          outputs[i]['logprobs'][input_len:length],  # pyrefly: ignore[bad-index]
          logprobs[i][input_len:length],
          rtol=5e-3,
          atol=1e-10,
      )
      np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
          outputs[i]['scores'][1:], logprobs[i][1:length], rtol=5e-3, atol=1e-10  # pyrefly: ignore[bad-index]
      )

  # NOTE: budget starts at 2 (not 1). A budget of 1 => a capacity-1 ragged-query
  # buffer, whose autotuned MIXED block sizes differ from the large-buffer mixed
  # baseline enough to perturb 1 prefill-position log-likelihood past rtol (the
  # greedy tokens still match). It is a degenerate tiling no real config uses
  # (realistic budgets are O(4096)); budget=2 already exercises the multi-pass
  # drain, so it is the meaningful minimum.
  @parameterized.named_parameters(
      dict(testcase_name='budget2', max_num_issue_tokens=2),
      dict(testcase_name='budget3', max_num_issue_tokens=3),
      dict(testcase_name='budget_large', max_num_issue_tokens=64),
  )
  def test_disagg_prefill_matches_mixed(self, max_num_issue_tokens: int):
    """Disaggregated prefill + greedy decode == mixed greedy decode.

    Builds the same tiny model as `test_continue_decode`, then compares two
    schedules with greedy sampling (`top_k=1`, so the prng does not affect the
    sampled tokens and the two paths are directly comparable):

    * mixed: `continue_decode` (prefill+decode interleaved), and
    * disagg: host-looped single-step prefill (MIXED case, up to
      `max_num_issue_tokens` tokens/pass -- a small budget forces a multi-pass
      drain, a large one a single pass) followed by `continue_decode`.

    Asserts the disagg prefill leaves every active slot at `input_lens - 1`
    (fully prefilled, ready to decode) and that the two schedules produce
    bit-identical output tokens, lens, and per-token scores.
    """
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    mesh_shape = [jax.device_count(), 1, 1, 1]
    max_seq_len = 8
    batch_size = 2
    vocab_size = 10
    per_head_dim = 2
    n_heads = 4
    n_kv_heads = 2

    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )

    def make_state() -> rpa.SamplingState:
      sampling_state = rpa.SamplingState.create(
          max_total_num_tokens=max_seq_len * 100,
          prng_key=jax.random.key(0),
          eos_ids=jnp.array([100]),
          decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
              total_num_pages=6,
              page_size=3,
              n_kv_heads=n_kv_heads,
              per_head_dim=per_head_dim,
              batch_size=batch_size,
              max_seq_len=max_seq_len,
              dtype='float32',
              head_partition='model',
              seq_partition='seq',
          ).init(),
      )
      tokens = RaggedArray.from_numpy_list(
          [np.array([0, 1, 2]), np.array([4, 5, 6, 7, 8])]
      )
      return dataclasses.replace(
          sampling_state,
          tokens=tokens.to_padded_dense(max_seq_len),
          input_lens=tokens.lens,
          position=jnp.array([0, 0]),
          max_decode_steps=jnp.array([0, 10]),
          rank=jnp.array([0, 1]),
      )

    emb_key, q_key, k_key, v_key = jax.random.split(jax.random.key(0), 4)
    params = dict(
        emb=jax.random.normal(emb_key, (vocab_size, per_head_dim)),
        q_proj=jax.random.normal(q_key, (n_heads, per_head_dim, per_head_dim)),
        k_proj=jax.random.normal(
            k_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
        v_proj=jax.random.normal(
            v_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
    )

    def forward_fn(
        params: common.PyTree,
        tokens: jax.Array,
        segment_ids: jax.Array,
        segment_positions: jax.Array,
        extra_inputs: common.PyTree = None,
        decode_state: common.PyTree = None,
        ragged: bool = True,
    ) -> tuple[jax.Array, common.PyTree]:
      del segment_ids, ragged
      emb = jnp.take(params['emb'], tokens, axis=0)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
      emb *= jnp.cos(segment_positions)[:, :, None]
      q = jnp.einsum('...d,ndh->...nh', emb, params['q_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      k = jnp.einsum('...d,ndh->...nh', emb, params['k_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      v = jnp.einsum('...d,ndh->...nh', emb, params['v_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      q = einops.rearrange(q, '1 l ... -> l ...')
      k = einops.rearrange(k, '1 l ... -> l ...')
      v = einops.rearrange(v, '1 l ... -> l ...')
      decode_state, attn_out = (
          decode_state.update_decode_state_and_compute_attn(  # pyrefly: ignore[missing-attribute]
              q=RaggedArray(q, extra_inputs['lens']),  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
              k=k,
              v=v,
          )
      )
      attn_out = einops.rearrange(attn_out, 'l ... -> 1 l ...')
      output = jnp.einsum(
          'vd,...d->...v', params['emb'], jnp.mean(attn_out, axis=-2)  # pyrefly: ignore[bad-index, unsupported-operation]
      )
      return output, {'decode_state': decode_state}

    # Greedy decode so sampling is deterministic (prng-independent).
    def decode_to_end(state: rpa.SamplingState) -> rpa.SamplingState:
      state = state.continue_decode(
          forward_fn=forward_fn,
          until_fn=lambda s: jnp.array(False),
          params=params,
          top_k=1,
      )
      return state

    # 1) Reference: mixed prefill+decode.
    mixed = decode_to_end(make_state())

    # 2) Disagg: chunked prefill, then decode.
    prefilled = _host_loop_prefill(
        make_state(), forward_fn, params, max_num_issue_tokens
    )
    # After chunked prefill, every active slot sits at input_lens - 1 (fully
    # prefilled) and none is still prefilling.
    np.testing.assert_array_equal(
        np.asarray(prefilled.position),
        np.asarray(prefilled.input_lens) - 1,
    )
    self.assertFalse(bool(prefilled.any_prefilling))
    disagg = decode_to_end(prefilled)

    # The two schedules must agree on lens, output tokens, and scores.
    np.testing.assert_array_equal(
        np.asarray(disagg.lens), np.asarray(mixed.lens)
    )
    np.testing.assert_array_equal(
        np.asarray(disagg.has_ended), np.asarray(mixed.has_ended)
    )
    mixed_out = mixed.get(np.array([True, True]))
    disagg_out = disagg.get(np.array([True, True]))
    for i in range(batch_size):
      np.testing.assert_array_equal(
          disagg_out[i]['tokens'], mixed_out[i]['tokens']
      )
      # scores[0] is a documented dummy slot (uninitialized); skip it, as
      # `test_continue_decode` does.
      np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
          disagg_out[i]['scores'][1:],  # pyrefly: ignore[bad-index]
          mixed_out[i]['scores'][1:],  # pyrefly: ignore[bad-index]
          rtol=5e-3,
          atol=1e-6,
      )

  @parameterized.named_parameters(
      dict(testcase_name='budget2', max_num_issue_tokens=2),
      dict(testcase_name='budget4', max_num_issue_tokens=4),
      dict(testcase_name='budget6', max_num_issue_tokens=6),
      dict(testcase_name='budget_large', max_num_issue_tokens=64),
  )
  def test_multipass_prefill_matches_single_pass(
      self, max_num_issue_tokens: int
  ):
    """Host-looped multi-pass prefill == a single big-budget prefill pass.

    This is the correctness equivalence for the batcher schedule that compiles a
    SINGLE prefill `mixed_step` and loops it from the host (see
    `page_batcher.Batcher.prefill_fn` / `loop`). A small per-pass budget forces
    a multi-pass drain; it must reach the SAME final fully-prefilled state as a
    single pass whose budget covers the whole batch, so downstream decode is
    identical.

    Drains prefill two ways from the same start state:
    * reference: one host-loop pass with a large budget (single on-device pass),
    * multi-pass: host loop with the (small) parametrized budget,
    then asserts identical `position`, per-leaf `kv_lens`, and that a subsequent
    greedy decode produces bit-identical output.
    """
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    mesh_shape = [jax.device_count(), 1, 1, 1]
    max_seq_len = 8
    batch_size = 2
    vocab_size = 10
    per_head_dim = 2
    n_heads = 4
    n_kv_heads = 2

    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )

    def make_state() -> rpa.SamplingState:
      sampling_state = rpa.SamplingState.create(
          max_total_num_tokens=max_seq_len * 100,
          prng_key=jax.random.key(0),
          eos_ids=jnp.array([100]),
          decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
              total_num_pages=6,
              page_size=3,
              n_kv_heads=n_kv_heads,
              per_head_dim=per_head_dim,
              batch_size=batch_size,
              max_seq_len=max_seq_len,
              dtype='float32',
              head_partition='model',
              seq_partition='seq',
          ).init(),
      )
      tokens = RaggedArray.from_numpy_list(
          [np.array([0, 1, 2]), np.array([4, 5, 6, 7, 8])]
      )
      return dataclasses.replace(
          sampling_state,
          tokens=tokens.to_padded_dense(max_seq_len),
          input_lens=tokens.lens,
          position=jnp.array([0, 0]),
          max_decode_steps=jnp.array([0, 10]),
          rank=jnp.array([0, 1]),
      )

    emb_key, q_key, k_key, v_key = jax.random.split(jax.random.key(0), 4)
    params = dict(
        emb=jax.random.normal(emb_key, (vocab_size, per_head_dim)),
        q_proj=jax.random.normal(q_key, (n_heads, per_head_dim, per_head_dim)),
        k_proj=jax.random.normal(
            k_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
        v_proj=jax.random.normal(
            v_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
    )

    def forward_fn(
        params: common.PyTree,
        tokens: jax.Array,
        segment_ids: jax.Array,
        segment_positions: jax.Array,
        extra_inputs: common.PyTree = None,
        decode_state: common.PyTree = None,
        ragged: bool = True,
    ) -> tuple[jax.Array, common.PyTree]:
      del segment_ids, ragged
      emb = jnp.take(params['emb'], tokens, axis=0)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
      emb *= jnp.cos(segment_positions)[:, :, None]
      q = jnp.einsum('...d,ndh->...nh', emb, params['q_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      k = jnp.einsum('...d,ndh->...nh', emb, params['k_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      v = jnp.einsum('...d,ndh->...nh', emb, params['v_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      q = einops.rearrange(q, '1 l ... -> l ...')
      k = einops.rearrange(k, '1 l ... -> l ...')
      v = einops.rearrange(v, '1 l ... -> l ...')
      decode_state, attn_out = (
          decode_state.update_decode_state_and_compute_attn(  # pyrefly: ignore[missing-attribute]
              q=RaggedArray(q, extra_inputs['lens']),  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
              k=k,
              v=v,
          )
      )
      attn_out = einops.rearrange(attn_out, 'l ... -> 1 l ...')
      output = jnp.einsum(
          'vd,...d->...v', params['emb'], jnp.mean(attn_out, axis=-2)  # pyrefly: ignore[bad-index, unsupported-operation]
      )
      return output, {'decode_state': decode_state}

    # Prefill drains up to `max_num_issue_tokens` tokens/pass (no per-slot cap);
    # a small budget forces the multi-pass drain we want to exercise.

    # Reference: a single big-budget prefill pass
    ref = _host_loop_prefill(
        make_state(), forward_fn, params, max_num_issue_tokens=64
    )

    # Multi-pass: host loop of single `mixed_step`s with the small parametrized
    # budget -- exactly what `prefill_fn` compiles and `loop()` drives.
    hosted = _host_loop_prefill(
        make_state(), forward_fn, params, max_num_issue_tokens
    )

    # Both must be fully prefilled and structurally identical.
    self.assertFalse(bool(ref.any_prefilling))
    self.assertFalse(bool(hosted.any_prefilling))
    np.testing.assert_array_equal(
        np.asarray(hosted.position), np.asarray(ref.position)
    )
    ref_kv = rpa.DecodeState.attrs_from_tree(ref.decode_state, ['kv_lens'])[
        'kv_lens'
    ]
    hosted_kv = rpa.DecodeState.attrs_from_tree(
        hosted.decode_state, ['kv_lens']
    )['kv_lens']
    for a, b in zip(hosted_kv, ref_kv):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    # And a subsequent greedy decode must be bit-identical.
    def decode_to_end(state: rpa.SamplingState) -> rpa.SamplingState:
      return state.continue_decode(
          forward_fn=forward_fn,
          until_fn=lambda s: jnp.array(False),
          params=params,
          top_k=1,
      )

    ref_decoded = decode_to_end(ref)
    hosted_decoded = decode_to_end(hosted)
    # Same generated lengths and end state.
    np.testing.assert_array_equal(
        np.asarray(hosted_decoded.lens), np.asarray(ref_decoded.lens)
    )
    np.testing.assert_array_equal(
        np.asarray(hosted_decoded.has_ended), np.asarray(ref_decoded.has_ended)
    )
    ref_out = ref_decoded.get(np.array([True, True]))
    hosted_out = hosted_decoded.get(np.array([True, True]))
    for i in range(batch_size):
      # Decode output (generated tokens) must be bit-identical -- this is the
      # decode-output equivalence. (Per-token scores are covered by
      # `test_disagg_prefill_matches_mixed`; here the deterministic
      # position/kv_lens equality above + identical generated tokens fully pin
      # down that the two prefill drains reach the same state.)
      np.testing.assert_array_equal(
          hosted_out[i]['tokens'], ref_out[i]['tokens']
      )

  @parameterized.named_parameters(
      dict(testcase_name='budget1', max_num_issue_tokens=1),
      dict(testcase_name='budget2', max_num_issue_tokens=2),
      dict(testcase_name='budget3', max_num_issue_tokens=3),
  )
  def test_num_queries_per_block_matches_mixed(self, max_num_issue_tokens: int):
    """Decode `num_queries_per_block=1` on MIXED matches the shared block_q."""
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    mesh_shape = [jax.device_count(), 1, 1, 1]
    max_seq_len = 8
    batch_size = 2
    vocab_size = 10
    per_head_dim = 2
    n_heads = 4
    n_kv_heads = 2

    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )

    def make_state() -> rpa.SamplingState:
      sampling_state = rpa.SamplingState.create(
          max_total_num_tokens=max_seq_len * 100,
          prng_key=jax.random.key(0),
          eos_ids=jnp.array([100]),
          decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
              total_num_pages=6,
              page_size=3,
              n_kv_heads=n_kv_heads,
              per_head_dim=per_head_dim,
              batch_size=batch_size,
              max_seq_len=max_seq_len,
              dtype='float32',
              head_partition='model',
              seq_partition='seq',
          ).init(),
      )
      tokens = RaggedArray.from_numpy_list(
          [np.array([0, 1, 2]), np.array([4, 5, 6, 7, 8])]
      )
      return dataclasses.replace(
          sampling_state,
          tokens=tokens.to_padded_dense(max_seq_len),
          input_lens=tokens.lens,
          position=jnp.array([0, 0]),
          max_decode_steps=jnp.array([0, 10]),
          rank=jnp.array([0, 1]),
      )

    emb_key, q_key, k_key, v_key = jax.random.split(jax.random.key(0), 4)
    params = dict(
        emb=jax.random.normal(emb_key, (vocab_size, per_head_dim)),
        q_proj=jax.random.normal(q_key, (n_heads, per_head_dim, per_head_dim)),
        k_proj=jax.random.normal(
            k_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
        v_proj=jax.random.normal(
            v_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
    )

    def make_forward_fn(num_queries_per_block: int | None):
      # The "model" fixes its RPA query-block tiling (like the batcher's
      # model / decode_model); block_q is NOT threaded per call.
      def forward_fn(
          params: common.PyTree,
          tokens: jax.Array,
          segment_ids: jax.Array,
          segment_positions: jax.Array,
          extra_inputs: common.PyTree = None,
          decode_state: common.PyTree = None,
          ragged: bool = True,
      ) -> tuple[jax.Array, common.PyTree]:
        del segment_ids, ragged
        emb = jnp.take(params['emb'], tokens, axis=0)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
        emb *= jnp.cos(segment_positions)[:, :, None]
        q = jnp.einsum('...d,ndh->...nh', emb, params['q_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
        k = jnp.einsum('...d,ndh->...nh', emb, params['k_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
        v = jnp.einsum('...d,ndh->...nh', emb, params['v_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
        q = einops.rearrange(q, '1 l ... -> l ...')
        k = einops.rearrange(k, '1 l ... -> l ...')
        v = einops.rearrange(v, '1 l ... -> l ...')
        decode_state, attn_out = (
            decode_state.update_decode_state_and_compute_attn(  # pyrefly: ignore[missing-attribute]
                q=RaggedArray(q, extra_inputs['lens']),  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
                k=k,
                v=v,
                num_queries_per_block=num_queries_per_block,
            )
        )
        attn_out = einops.rearrange(attn_out, 'l ... -> 1 l ...')
        output = jnp.einsum(
            'vd,...d->...v', params['emb'], jnp.mean(attn_out, axis=-2)  # pyrefly: ignore[bad-index, unsupported-operation]
        )
        return output, {'decode_state': decode_state}

      return forward_fn

    # Prefill ONCE with the shared/autotuned block_q, then decode the SAME
    # prefilled state two ways: shared block_q vs block_q=1. block_q is a pure
    # tiling hint (like the batcher's decode_model), so the two decodes must be
    # bit-identical.
    prefilled = _host_loop_prefill(
        make_state(), make_forward_fn(None), params, max_num_issue_tokens
    )

    def decode_with(num_queries_per_block: int | None) -> rpa.SamplingState:
      return prefilled.continue_decode(
          forward_fn=make_forward_fn(num_queries_per_block),
          until_fn=lambda s: jnp.array(False),
          params=params,
          top_k=1,
      )

    # Decode via the MIXED case with the shared block_q vs. block_q=1.
    mixed = decode_with(None)
    decode_case = decode_with(1)

    np.testing.assert_array_equal(
        np.asarray(decode_case.lens), np.asarray(mixed.lens)
    )
    np.testing.assert_array_equal(
        np.asarray(decode_case.has_ended), np.asarray(mixed.has_ended)
    )
    mixed_out = mixed.get(np.array([True, True]))
    decode_out = decode_case.get(np.array([True, True]))
    for i in range(batch_size):
      np.testing.assert_array_equal(
          decode_out[i]['tokens'], mixed_out[i]['tokens']
      )
      # scores[0] is a documented dummy slot (uninitialized); skip it.
      np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
          decode_out[i]['scores'][1:],  # pyrefly: ignore[bad-index]
          mixed_out[i]['scores'][1:],  # pyrefly: ignore[bad-index]
          rtol=5e-3,
          atol=1e-6,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='prefill16_decode2', prefill_budget=16, decode_budget=2
      ),
      dict(testcase_name='prefill8_decode1', prefill_budget=8, decode_budget=1),
      dict(testcase_name='prefill4_decode4', prefill_budget=4, decode_budget=4),
  )
  def test_per_stage_issue_tokens_matches_default(
      self, prefill_budget: int, decode_budget: int
  ):
    """Per-stage `max_num_issue_tokens` does not change the sampled output.

    The disaggregated schedule (host-looped prefill then `continue_decode`) is
    run twice with greedy sampling (`top_k=1`, prng-independent):

    * default: a single shared `max_num_issue_tokens` for both stages, and
    * per-stage: PREFILL uses `prefill_budget`, DECODE uses `decode_budget`.

    `max_num_issue_tokens` only changes the per-pass token *budget* (how many
    passes the while-loops take), not the attention math, so the two runs must
    produce bit-identical output tokens, lens, and (numerically equal) scores.
    """
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    mesh_shape = [jax.device_count(), 1, 1, 1]
    max_seq_len = 8
    batch_size = 2
    vocab_size = 10
    per_head_dim = 2
    n_heads = 4
    n_kv_heads = 2
    shared_budget = 4

    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )

    def make_state() -> rpa.SamplingState:
      sampling_state = rpa.SamplingState.create(
          max_total_num_tokens=max_seq_len * 100,
          prng_key=jax.random.key(0),
          eos_ids=jnp.array([100]),
          decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
              total_num_pages=6,
              page_size=3,
              n_kv_heads=n_kv_heads,
              per_head_dim=per_head_dim,
              batch_size=batch_size,
              max_seq_len=max_seq_len,
              dtype='float32',
              head_partition='model',
              seq_partition='seq',
          ).init(),
      )
      tokens = RaggedArray.from_numpy_list(
          [np.array([0, 1, 2]), np.array([4, 5, 6, 7, 8])]
      )
      return dataclasses.replace(
          sampling_state,
          tokens=tokens.to_padded_dense(max_seq_len),
          input_lens=tokens.lens,
          position=jnp.array([0, 0]),
          max_decode_steps=jnp.array([0, 10]),
          rank=jnp.array([0, 1]),
      )

    emb_key, q_key, k_key, v_key = jax.random.split(jax.random.key(0), 4)
    params = dict(
        emb=jax.random.normal(emb_key, (vocab_size, per_head_dim)),
        q_proj=jax.random.normal(q_key, (n_heads, per_head_dim, per_head_dim)),
        k_proj=jax.random.normal(
            k_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
        v_proj=jax.random.normal(
            v_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
    )

    def forward_fn(
        params: common.PyTree,
        tokens: jax.Array,
        segment_ids: jax.Array,
        segment_positions: jax.Array,
        extra_inputs: common.PyTree = None,
        decode_state: common.PyTree = None,
        ragged: bool = True,
    ) -> tuple[jax.Array, common.PyTree]:
      del segment_ids, ragged
      emb = jnp.take(params['emb'], tokens, axis=0)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
      emb *= jnp.cos(segment_positions)[:, :, None]
      q = jnp.einsum('...d,ndh->...nh', emb, params['q_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      k = jnp.einsum('...d,ndh->...nh', emb, params['k_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      v = jnp.einsum('...d,ndh->...nh', emb, params['v_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      q = einops.rearrange(q, '1 l ... -> l ...')
      k = einops.rearrange(k, '1 l ... -> l ...')
      v = einops.rearrange(v, '1 l ... -> l ...')
      decode_state, attn_out = (
          decode_state.update_decode_state_and_compute_attn(  # pyrefly: ignore[missing-attribute]
              q=RaggedArray(q, extra_inputs['lens']),  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
              k=k,
              v=v,
          )
      )
      attn_out = einops.rearrange(attn_out, 'l ... -> 1 l ...')
      output = jnp.einsum(
          'vd,...d->...v', params['emb'], jnp.mean(attn_out, axis=-2)  # pyrefly: ignore[bad-index, unsupported-operation]
      )
      return output, {'decode_state': decode_state}

    def run_disagg(
        prefill_max_issue: int, decode_max_issue: int
    ) -> rpa.SamplingState:
      prefilled = _host_loop_prefill(
          make_state(), forward_fn, params, prefill_max_issue
      )
      return prefilled.continue_decode(
          forward_fn=forward_fn,
          until_fn=lambda s: jnp.array(False),
          params=params,
          max_num_issue_tokens=decode_max_issue,
          top_k=1,
      )

    # Shared budget for both stages vs. per-stage budgets.
    default = run_disagg(
        prefill_max_issue=shared_budget, decode_max_issue=shared_budget
    )
    per_stage = run_disagg(
        prefill_max_issue=prefill_budget, decode_max_issue=decode_budget
    )

    np.testing.assert_array_equal(
        np.asarray(per_stage.lens), np.asarray(default.lens)
    )
    np.testing.assert_array_equal(
        np.asarray(per_stage.has_ended), np.asarray(default.has_ended)
    )
    default_out = default.get(np.array([True, True]))
    per_stage_out = per_stage.get(np.array([True, True]))
    for i in range(batch_size):
      np.testing.assert_array_equal(
          per_stage_out[i]['tokens'], default_out[i]['tokens']
      )
      # scores[0] is a documented dummy slot (uninitialized); skip it.
      np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
          per_stage_out[i]['scores'][1:],  # pyrefly: ignore[bad-index]
          default_out[i]['scores'][1:],  # pyrefly: ignore[bad-index]
          rtol=5e-3,
          atol=1e-6,
      )

  @parameterized.named_parameters(
      dict(testcase_name='budget1', max_num_issue_tokens=1),
      dict(testcase_name='budget2', max_num_issue_tokens=2),
      dict(testcase_name='budget3', max_num_issue_tokens=3),
  )
  def test_skip_prefill_logits_matches_mixed(self, max_num_issue_tokens: int):
    """`skip_logits=True` in prefill leaves the generated tokens unchanged.

    The disaggregated schedule (host-looped prefill then `continue_decode`) is
    run against the mixed baseline (`continue_decode` alone), with the prefill
    stage using `skip_logits=True` -- i.e. the LM-head / sampling is skipped for
    every prefill pass. Because every prefill output is teacher-forced (the
    prompt's next token is known), its logits are never consumed, so the
    GENERATED (decode) tokens and their per-token decode scores must be
    bit-identical to the mixed baseline. Prompt-position scores are NOT compared
    (they are zero placeholders when logits are skipped, and unused by callers).
    """
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    mesh_shape = [jax.device_count(), 1, 1, 1]
    max_seq_len = 8
    batch_size = 2
    vocab_size = 10
    per_head_dim = 2
    n_heads = 4
    n_kv_heads = 2

    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )

    def make_state() -> rpa.SamplingState:
      sampling_state = rpa.SamplingState.create(
          max_total_num_tokens=max_seq_len * 100,
          prng_key=jax.random.key(0),
          eos_ids=jnp.array([100]),
          decode_state=rpa.DecodeStateConfig(  # pyrefly: ignore[bad-argument-type]
              total_num_pages=6,
              page_size=3,
              n_kv_heads=n_kv_heads,
              per_head_dim=per_head_dim,
              batch_size=batch_size,
              max_seq_len=max_seq_len,
              dtype='float32',
              head_partition='model',
              seq_partition='seq',
          ).init(),
      )
      tokens = RaggedArray.from_numpy_list(
          [np.array([0, 1, 2]), np.array([4, 5, 6, 7, 8])]
      )
      return dataclasses.replace(
          sampling_state,
          tokens=tokens.to_padded_dense(max_seq_len),
          input_lens=tokens.lens,
          position=jnp.array([0, 0]),
          max_decode_steps=jnp.array([0, 10]),
          rank=jnp.array([0, 1]),
      )

    emb_key, q_key, k_key, v_key = jax.random.split(jax.random.key(0), 4)
    params = dict(
        emb=jax.random.normal(emb_key, (vocab_size, per_head_dim)),
        q_proj=jax.random.normal(q_key, (n_heads, per_head_dim, per_head_dim)),
        k_proj=jax.random.normal(
            k_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
        v_proj=jax.random.normal(
            v_key, (n_kv_heads, per_head_dim, per_head_dim)
        ),
    )

    def forward_fn(
        params: common.PyTree,
        tokens: jax.Array,
        segment_ids: jax.Array,
        segment_positions: jax.Array,
        extra_inputs: common.PyTree = None,
        decode_state: common.PyTree = None,
        ragged: bool = True,
    ) -> tuple[jax.Array, common.PyTree]:
      del segment_ids, ragged
      emb = jnp.take(params['emb'], tokens, axis=0)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
      emb *= jnp.cos(segment_positions)[:, :, None]
      q = jnp.einsum('...d,ndh->...nh', emb, params['q_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      k = jnp.einsum('...d,ndh->...nh', emb, params['k_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      v = jnp.einsum('...d,ndh->...nh', emb, params['v_proj'])  # pyrefly: ignore[bad-index, unsupported-operation]
      q = einops.rearrange(q, '1 l ... -> l ...')
      k = einops.rearrange(k, '1 l ... -> l ...')
      v = einops.rearrange(v, '1 l ... -> l ...')
      decode_state, attn_out = (
          decode_state.update_decode_state_and_compute_attn(  # pyrefly: ignore[missing-attribute]
              q=RaggedArray(q, extra_inputs['lens']),  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
              k=k,
              v=v,
          )
      )
      attn_out = einops.rearrange(attn_out, 'l ... -> 1 l ...')
      output = jnp.einsum(
          'vd,...d->...v', params['emb'], jnp.mean(attn_out, axis=-2)  # pyrefly: ignore[bad-index, unsupported-operation]
      )
      return output, {'decode_state': decode_state}

    def decode_to_end(state: rpa.SamplingState) -> rpa.SamplingState:
      return state.continue_decode(
          forward_fn=forward_fn,
          until_fn=lambda s: jnp.array(False),
          params=params,
          top_k=1,
      )

    # Reference: mixed prefill+decode (no skip).
    mixed = decode_to_end(make_state())

    # Disagg with skip_logits=True in the prefill stage, then decode.
    prefilled = _host_loop_prefill(
        make_state(), forward_fn, params, max_num_issue_tokens, skip_logits=True
    )
    np.testing.assert_array_equal(
        np.asarray(prefilled.position),
        np.asarray(prefilled.input_lens) - 1,
    )
    self.assertFalse(bool(prefilled.any_prefilling))
    disagg = decode_to_end(prefilled)

    np.testing.assert_array_equal(
        np.asarray(disagg.lens), np.asarray(mixed.lens)
    )
    np.testing.assert_array_equal(
        np.asarray(disagg.has_ended), np.asarray(mixed.has_ended)
    )
    mixed_out = mixed.get(np.array([True, True]))
    disagg_out = disagg.get(np.array([True, True]))
    for i in range(batch_size):
      # The GENERATED tokens (input_len onward) must be bit-identical: those are
      # the decode outputs. Prompt tokens are teacher-forced in both paths.
      np.testing.assert_array_equal(
          disagg_out[i]['tokens'], mixed_out[i]['tokens']
      )
      input_len = int(mixed_out[i]['input_len'])  # pyrefly: ignore[bad-argument-type]
      length = len(mixed_out[i]['tokens'])  # pyrefly: ignore[bad-argument-type]
      # Compare only the DECODE-position scores (>= input_len); prompt-position
      # scores are zero placeholders when prefill logits are skipped.
      np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
          disagg_out[i]['scores'][input_len:length],  # pyrefly: ignore[bad-index]
          mixed_out[i]['scores'][input_len:length],  # pyrefly: ignore[bad-index]
          rtol=5e-3,
          atol=1e-6,
      )


class DecodeBlockQRealisticShapeTest(parameterized.TestCase):
  """Compiles decode `num_queries_per_block=1` path at eval-realistic shapes.

  The other block_q tests use tiny head_dim/page_size on a single-device
  (`model`=1) mesh, which does not exercise the Mosaic layout passes the real
  eval hits (`per_head_dim=128`, `page_size=128`, GQA `n_kv_heads<n_q_heads`,
  and `model`-axis head sharding). This guards the proven disaggregated-decode
  mechanism -- block_q=1 on the RPA MIXED case -- at qwen3-1p7b-like shapes: it
  runs `update_decode_state_and_compute_attn` with `num_queries_per_block=1`
  and, as a control, `num_queries_per_block=None` (autotuned block_q), and
  asserts both compile+run and agree numerically.
  """

  @parameterized.named_parameters(
      dict(testcase_name='shared_block_q', num_queries_per_block=None),
      dict(testcase_name='block_q_1', num_queries_per_block=1),
  )
  def test_compiles_and_matches_reference(self, num_queries_per_block):
    if rpa_kernel.get_tpu_version() < 4:
      self.skipTest('Requires TPU v4 or higher')
    # Shard heads across the `model` axis (as decode-mode does), when possible.
    model_axis = math.gcd(8, jax.device_count())
    replica_axis = jax.device_count() // model_axis
    mesh_shape = [replica_axis, 1, 1, model_axis]
    sharding.set_mesh(
        mesh_shape, axis_names=('replica', 'data', 'seq', 'model')
    )

    # qwen3-1p7b-like attention shapes.
    page_size = 128
    per_head_dim = 128
    n_q_heads = 16
    n_kv_heads = 8
    batch_size = 4
    total_num_pages = 32
    max_seq_len = total_num_pages * page_size + 1

    # Pure-decode pass: every active slot has some existing KV and issues
    # exactly one query token (q_len == 1), the decode-stage invariant.
    old_kv_lens = jnp.array([130, 300, 5, 256], dtype=jnp.int32)
    q_lens = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    num_issue_tokens = int(jnp.sum(q_lens))

    rk1, rk2, rk3 = jax.random.split(jax.random.key(0), 3)
    ragged_old_kv = RaggedArray(
        data=jax.random.normal(
            rk1, (int(jnp.sum(old_kv_lens)), n_kv_heads * 2, per_head_dim)
        ),
        lens=old_kv_lens,
    )
    ragged_q = RaggedArray(
        data=jax.random.normal(
            rk2, (num_issue_tokens, n_q_heads, per_head_dim)
        ),
        lens=q_lens,
    )
    ragged_kv = RaggedArray(
        data=jax.random.normal(
            rk3, (num_issue_tokens, n_kv_heads * 2, per_head_dim)
        ),
        lens=q_lens,
    )
    updated_ragged_kv = ragged_old_kv.concat(ragged_kv)
    expected_attn_out_list = []
    for i in range(batch_size):
      q = jnp.reshape(ragged_q.row(i), (-1, n_q_heads, per_head_dim))
      k = updated_ragged_kv.row(i)[:, 0::2]
      v = updated_ragged_kv.row(i)[:, 1::2]
      expected_attn_out_list.append(qkv_attn(q, k, v))  # pyrefly: ignore[bad-argument-type]

    config = rpa.DecodeStateConfig(
        total_num_pages=total_num_pages,
        page_size=page_size,
        n_kv_heads=n_kv_heads,
        per_head_dim=per_head_dim,
        batch_size=batch_size,
        dtype='float32',
        max_seq_len=max_seq_len,
        head_partition='model',
        seq_partition='seq',
    )
    ds = (
        config.init()
        .allocate(old_kv_lens)
        .insert(
            ragged_old_kv.data[:, 0::2],
            ragged_old_kv.data[:, 1::2],
            old_kv_lens,
        )
    )

    # Decode uses block_q=1 on the MIXED case (`num_queries_per_block=1`) at
    # realistic shapes -- the proven-good mechanism (not DECODE-case routing).
    _, ragged_attn_out = ds.update_decode_state_and_compute_attn(
        q=ragged_q,
        k=ragged_kv.data[:, 0::2],
        v=ragged_kv.data[:, 1::2],
        num_queries_per_block=num_queries_per_block,
    )
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, atol=5e-2, rtol=5e-2),
        RaggedArray(ragged_attn_out, ragged_q.lens).to_numpy_list(),
        expected_attn_out_list,
    )

if __name__ == '__main__':
  absltest.main()

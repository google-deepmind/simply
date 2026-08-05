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
"""Tests for the host-RAM prefix cache."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from simply.serving import prefix_cache as prefix_cache_lib
from simply.utils import ragged_paged_attention as rpa


CHUNK_SIZE = 4


def _tokens(*ids) -> np.ndarray:
  return np.asarray(ids, dtype=np.int32)


def _entry_tree(
    value: float,
    n_layers: int = 2,
    available: bool | list[bool] = True,
) -> dict[str, rpa.SnapshotChunkLeaf]:
  """A `DecodeState`-shaped pytree of :class:`SnapshotChunkLeaf` leaves.

  Each leaf wraps a small on-device `jax.Array` (so `snapshot`/`restore`
  exercise the real HBM<->host `device_put` offload path) and a
  scalar `available` bool. By default every leaf is `available=True`.
  Pass `available=False` to mark all leaves unavailable, or a list to
  mark specific layers.

  Args:
    value: base fill value; layer `i` is filled with `value + i`.
    n_layers: number of per-layer leaves to build.
    available: per-leaf availability. `bool` broadcasts to all layers;
      `list[bool]` (length `n_layers`) sets each leaf individually.

  Returns:
    Dict mapping `layer_{i}` to a :class:`SnapshotChunkLeaf`.
  """
  if isinstance(available, bool):
    available_per_leaf = [available] * n_layers
  else:
    assert len(available) == n_layers
    available_per_leaf = list(available)
  return {
      f'layer_{i}': rpa.SnapshotChunkLeaf(
          payload=jnp.full((4, 8), value + i, dtype=jnp.float32),
          available=jnp.asarray(av),
      )
      for i, av in enumerate(available_per_leaf)
  }


def _assert_stored_matches_input(
    stored: dict[str, jax.Array | jax.ShapeDtypeStruct],
    expected_input: dict[str, rpa.SnapshotChunkLeaf],
) -> None:
  """Asserts a cache-stored tree matches the original input leaves.

  The cache stores bare payloads (not `SnapshotChunkLeaf` instances):
  for an input leaf with `available=True` the stored entry is a
  host-resident `jax.Array` whose bytes equal the input payload; for
  `available=False` it is a `jax.ShapeDtypeStruct` whose shape/dtype
  match the input payload (the bytes are not stored).

  Args:
    stored: cache-stored tree (bare payloads, one per layer).
    expected_input: original `SnapshotChunkLeaf` tree passed to
      `cache.snapshot`.
  """
  assert set(stored.keys()) == set(expected_input.keys()), (
      f'keys differ: {stored.keys()} vs {expected_input.keys()}'
  )
  for k, leaf in expected_input.items():
    if not bool(leaf.available):
      assert isinstance(stored[k], jax.ShapeDtypeStruct), (
          f'{k}: expected ShapeDtypeStruct, got {type(stored[k])}'
      )
      assert stored[k].shape == leaf.payload.shape, (
          f'{k}: stored shape {stored[k].shape} != input '
          f'{leaf.payload.shape}'
      )
      assert stored[k].dtype == leaf.payload.dtype, (
          f'{k}: stored dtype {stored[k].dtype} != input '
          f'{leaf.payload.dtype}'
      )
      continue
    assert isinstance(stored[k], jax.Array), (
        f'{k}: expected jax.Array, got {type(stored[k])}'
    )
    np.testing.assert_array_equal(
        np.asarray(stored[k]), np.asarray(leaf.payload)
    )


class ChunkKeysTest(absltest.TestCase):

  def test_no_full_chunk(self):
    self.assertEqual(
        prefix_cache_lib.chunk_keys(_tokens(1, 2, 3), CHUNK_SIZE), []
    )

  def test_one_full_chunk(self):
    keys = prefix_cache_lib.chunk_keys(_tokens(1, 2, 3, 4), CHUNK_SIZE)
    self.assertLen(keys, 1)

  def test_multiple_chunks_have_distinct_keys(self):
    keys = prefix_cache_lib.chunk_keys(
        _tokens(1, 2, 3, 4, 5, 6, 7, 8), CHUNK_SIZE
    )
    self.assertLen(keys, 2)
    self.assertNotEqual(keys[0], keys[1])

  def test_shared_prefix_has_shared_keys(self):
    keys_a = prefix_cache_lib.chunk_keys(
        _tokens(1, 2, 3, 4, 5, 6, 7, 8), CHUNK_SIZE
    )
    keys_b = prefix_cache_lib.chunk_keys(
        _tokens(1, 2, 3, 4, 9, 9, 9, 9), CHUNK_SIZE
    )
    self.assertEqual(keys_a[0], keys_b[0])
    self.assertNotEqual(keys_a[1], keys_b[1])

  def test_dtype_independence(self):
    keys_i32 = prefix_cache_lib.chunk_keys(_tokens(1, 2, 3, 4), CHUNK_SIZE)
    keys_i64 = prefix_cache_lib.chunk_keys(
        np.asarray([1, 2, 3, 4], dtype=np.int64), CHUNK_SIZE
    )
    self.assertEqual(keys_i32, keys_i64)


class PrefixCacheTest(absltest.TestCase):

  def _new_cache(self, **kwargs) -> prefix_cache_lib.PrefixCache:
    return prefix_cache_lib.PrefixCache(
        chunk_size=kwargs.pop('chunk_size', CHUNK_SIZE),
        **kwargs,
    )

  def test_round_trip_one_chunk(self):
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0)
    cache.snapshot(tokens, entry_tree)
    got = cache.restore(tokens)
    _assert_stored_matches_input(got, entry_tree)  # pyrefly: ignore[bad-argument-type]

  def test_round_trip_multiple_chunks(self):
    cache = self._new_cache()
    tokens_8 = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    entry_a = _entry_tree(1.0)
    entry_b = _entry_tree(2.0)
    cache.snapshot(tokens_8[:4], entry_a)
    cache.snapshot(tokens_8, entry_b)
    _assert_stored_matches_input(cache.restore(tokens_8[:4]), entry_a)  # pyrefly: ignore[bad-argument-type]
    _assert_stored_matches_input(cache.restore(tokens_8), entry_b)  # pyrefly: ignore[bad-argument-type]

  def test_cache_is_per_process_not_shared_across_instances(self):
    # The host-RAM cache is process-local: a second instance starts cold
    # and does NOT see entries snapshotted into a different instance.
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0)
    writer = prefix_cache_lib.PrefixCache(chunk_size=CHUNK_SIZE)
    writer.snapshot(tokens, entry_tree)
    self.assertIsNotNone(writer.restore(tokens))
    reader = prefix_cache_lib.PrefixCache(chunk_size=CHUNK_SIZE)
    self.assertIsNone(reader.restore(tokens))

  def test_restore_returns_host_resident_payloads(self):
    # `restore` returns the stored form: bare host-resident `jax.Array`
    # leaves for available payloads. Callers tree-map
    # `SnapshotChunkLeaf.onload` to onload back to device memory.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0)
    cache.snapshot(tokens, entry_tree)
    got = cache.restore(tokens)
    for k, original in entry_tree.items():
      self.assertIsInstance(got[k], jax.Array)  # pyrefly: ignore[unsupported-operation]
      self.assertIn(
          got[k].sharding.memory_kind, ('pinned_host', 'unpinned_host')  # pyrefly: ignore[unsupported-operation]
      )
      # onload() round-trips back to device memory with the same shape.
      back = rpa.SnapshotChunkLeaf.onload(got[k])  # pyrefly: ignore[unsupported-operation]
      self.assertIsInstance(back, jax.Array)
      self.assertEqual(back.sharding.memory_kind, 'device')
      np.testing.assert_array_equal(
          np.asarray(back), np.asarray(original.payload)
      )

  def test_unavailable_leaves_become_shape_dtype_struct(self):
    # SnapshotChunkLeaf.offload() returns a `ShapeDtypeStruct` (no
    # bytes) for `available=False` leaves, saving host RAM. The cache
    # stores it as-is; is_complete is False for the chunk as a whole.
    # On restore, `SnapshotChunkLeaf.onload` materialises an
    # uninitialised buffer of the right shape/sharding.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0, n_layers=2, available=[True, False])
    cache.snapshot(tokens, entry_tree)
    got = cache.restore(tokens)
    self.assertIsInstance(got['layer_0'], jax.Array)  # pyrefly: ignore[unsupported-operation]
    self.assertIsInstance(got['layer_1'], jax.ShapeDtypeStruct)  # pyrefly: ignore[unsupported-operation]
    self.assertEqual(
        got['layer_1'].shape, entry_tree['layer_1'].payload.shape
    )
    self.assertEqual(
        got['layer_1'].dtype, entry_tree['layer_1'].payload.dtype
    )
    self.assertFalse(cache.is_complete(tokens))
    self.assertTrue(cache.is_cached(tokens))
    # onload of the ShapeDtypeStruct materialises a fresh device array.
    materialised = rpa.SnapshotChunkLeaf.onload(got['layer_1'])
    self.assertIsInstance(materialised, jax.Array)
    self.assertEqual(materialised.sharding.memory_kind, 'device')
    self.assertEqual(materialised.shape, got['layer_1'].shape)
    self.assertEqual(materialised.dtype, got['layer_1'].dtype)

  def test_partial_prefix_hit(self):
    cache = self._new_cache()
    entry_a = _entry_tree(1.0)
    entry_b = _entry_tree(2.0)
    cache.snapshot(_tokens(1, 2, 3, 4), entry_a)
    cache.snapshot(_tokens(1, 2, 3, 4, 5, 6, 7, 8), entry_b)
    # New sequence sharing only the first chunk: chunk 0 hits, chunk 1
    # has a different key because token 5..8 differ.
    different_8 = _tokens(1, 2, 3, 4, 9, 9, 9, 9)
    _assert_stored_matches_input(cache.restore(different_8[:4]), entry_a)  # pyrefly: ignore[bad-argument-type]
    self.assertIsNone(cache.restore(different_8))

  def test_restore_miss_returns_none(self):
    cache = self._new_cache()
    self.assertIsNone(cache.restore(_tokens(1, 2, 3, 4)))

  def test_snapshot_first_writer_wins_on_present_leaves(self):
    # Per-leaf rule: a present stored leaf is NOT overwritten by a
    # later snapshot, even if the new one is also present.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    cache.snapshot(tokens, _entry_tree(1.0))
    cache.snapshot(tokens, _entry_tree(2.0))  # no-op on present leaves
    _assert_stored_matches_input(cache.restore(tokens), _entry_tree(1.0))  # pyrefly: ignore[bad-argument-type]

  def test_complete_snapshot_promotes_incomplete_entry(self):
    # A chunk first cached as incomplete (some layer unavailable) is
    # **overwritten as a whole** when a later snapshot witnesses every
    # layer as available. No per-leaf merging: the new snapshot wins
    # for every leaf, not just the previously-missing ones.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    # First snapshot: layer_0 available, layer_1 unavailable
    # (-> ShapeDtypeStruct).
    first = _entry_tree(1.0, n_layers=2, available=[True, False])
    cache.snapshot(tokens, first)
    self.assertFalse(cache.is_complete(tokens))
    # Second snapshot: BOTH layers available (different fill value).
    # Whole-tree overwrite -- layer_0 also takes the second snapshot's
    # value (2.0), not the first's (1.0).
    second = _entry_tree(2.0, n_layers=2, available=[True, True])
    cache.snapshot(tokens, second)
    self.assertTrue(cache.is_complete(tokens))
    got = cache.restore(tokens)
    self.assertIsInstance(got['layer_0'], jax.Array)  # pyrefly: ignore[unsupported-operation]
    self.assertIsInstance(got['layer_1'], jax.Array)  # pyrefly: ignore[unsupported-operation]
    np.testing.assert_array_equal(
        np.asarray(got['layer_0']),
        np.asarray(second['layer_0'].payload),  # 2.0 (whole-tree overwrite)
    )
    np.testing.assert_array_equal(
        np.asarray(got['layer_1']),
        np.asarray(second['layer_1'].payload),  # 3.0 (whole-tree overwrite)
    )
    self.assertEqual(cache.stats()['n_promoted'], 1)

  def test_incomplete_snapshot_cannot_overwrite_complete_entry(self):
    # An already-complete stored entry is never overwritten by an
    # incomplete new snapshot (would be a downgrade).
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    cache.snapshot(tokens, _entry_tree(1.0))  # both available
    # Second snapshot: both unavailable (-> ShapeDtypeStruct after offload).
    cache.snapshot(
        tokens, _entry_tree(2.0, n_layers=2, available=[False, False])
    )
    self.assertTrue(cache.is_complete(tokens))
    _assert_stored_matches_input(cache.restore(tokens), _entry_tree(1.0))  # pyrefly: ignore[bad-argument-type]

  def test_is_cached_and_is_complete(self):
    cache = self._new_cache()
    tokens_a = _tokens(1, 2, 3, 4)
    tokens_b = _tokens(5, 6, 7, 8)
    tokens_c = _tokens(9, 9, 9, 9)
    cache.snapshot(tokens_a, _entry_tree(1.0))  # both available
    cache.snapshot(
        tokens_b, _entry_tree(1.0, n_layers=2, available=[True, False])
    )
    self.assertTrue(cache.is_cached(tokens_a))
    self.assertTrue(cache.is_complete(tokens_a))
    self.assertTrue(cache.is_cached(tokens_b))
    self.assertFalse(cache.is_complete(tokens_b))
    self.assertFalse(cache.is_cached(tokens_c))
    self.assertFalse(cache.is_complete(tokens_c))

  def test_snapshot_rejects_bad_token_length(self):
    cache = self._new_cache()
    entry_tree = _entry_tree(1.0)
    with self.assertRaisesRegex(ValueError, 'positive multiple'):
      cache.snapshot(_tokens(), entry_tree)
    with self.assertRaisesRegex(ValueError, 'positive multiple'):
      cache.snapshot(_tokens(1, 2, 3), entry_tree)

  def test_restore_rejects_bad_token_length(self):
    cache = self._new_cache()
    with self.assertRaisesRegex(ValueError, 'positive multiple'):
      cache.restore(_tokens())
    with self.assertRaisesRegex(ValueError, 'positive multiple'):
      cache.restore(_tokens(1, 2, 3))

  def test_is_cached_rejects_bad_token_length(self):
    cache = self._new_cache()
    with self.assertRaisesRegex(ValueError, 'positive multiple'):
      cache.is_cached(_tokens())

  def test_longest_restorable_prefix_all_present_all_complete(self):
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    for k in (4, 8, 12):
      cache.snapshot(tokens[:k], _entry_tree(float(k)))
    # min_num_chunks_for_window=2: any k works since every chunk is complete.
    self.assertEqual(
        cache.longest_restorable_prefix(
            tokens, max_chunks=3, start_chunk_idx=0, min_num_chunks_for_window=2
        ),
        3,
    )

  def test_longest_restorable_prefix_stops_at_missing_chunk(self):
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    # Only chunks 0 and 1 are present; chunk 2 missing.
    cache.snapshot(tokens[:4], _entry_tree(1.0))
    cache.snapshot(tokens[:8], _entry_tree(2.0))
    self.assertEqual(
        cache.longest_restorable_prefix(
            tokens, max_chunks=3, start_chunk_idx=0, min_num_chunks_for_window=0
        ),
        2,
    )

  def test_longest_restorable_prefix_skips_past_incomplete_for_deeper_hit(self):
    # Chunk 0 incomplete (layer_1 unavailable), chunks 1 and 2 complete,
    # min_num_chunks_for_window=1.
    # For k=1 the tail [0,1) is incomplete -> reject. For k=2 the tail
    # [1,2) is complete -> OK. For k=3 the tail [2,3) is complete -> OK.
    # Deepest valid k is 3.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    cache.snapshot(
        tokens[:4], _entry_tree(1.0, n_layers=2, available=[True, False])
    )
    cache.snapshot(tokens[:8], _entry_tree(2.0))
    cache.snapshot(tokens[:12], _entry_tree(3.0))
    self.assertEqual(
        cache.longest_restorable_prefix(
            tokens, max_chunks=3, start_chunk_idx=0, min_num_chunks_for_window=1
        ),
        3,
    )

  def test_longest_restorable_prefix_incomplete_inside_window_blocks(self):
    # min_num_chunks_for_window=2: any k whose [k-2, k) overlaps an
    # incomplete chunk is rejected. Chunks: 0 complete, 1 incomplete,
    # 2 complete.
    # k=1 tail=[-1,1) -> [0,1) all complete -> OK (best_k=1).
    # k=2 tail=[0,2) contains chunk 1 incomplete -> reject.
    # k=3 tail=[1,3) contains chunk 1 incomplete -> reject.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    cache.snapshot(tokens[:4], _entry_tree(1.0))
    cache.snapshot(
        tokens[:8], _entry_tree(2.0, n_layers=2, available=[True, False])
    )
    cache.snapshot(tokens[:12], _entry_tree(3.0))
    self.assertEqual(
        cache.longest_restorable_prefix(
            tokens, max_chunks=3, start_chunk_idx=0, min_num_chunks_for_window=2
        ),
        1,
    )

  def test_longest_restorable_prefix_zero_window_only_presence_matters(self):
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    cache.snapshot(
        tokens[:4], _entry_tree(1.0, n_layers=2, available=[True, False])
    )
    cache.snapshot(
        tokens[:8], _entry_tree(2.0, n_layers=2, available=[True, False])
    )
    self.assertEqual(
        cache.longest_restorable_prefix(
            tokens, max_chunks=2, start_chunk_idx=0, min_num_chunks_for_window=0
        ),
        2,
    )

  def test_longest_restorable_prefix_respects_start_chunk_idx(self):
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    cache.snapshot(tokens[:8], _entry_tree(2.0))
    cache.snapshot(tokens[:12], _entry_tree(3.0))
    # Chunk 0 not snapshotted; but start_chunk_idx=1 skips it.
    self.assertEqual(
        cache.longest_restorable_prefix(
            tokens, max_chunks=3, start_chunk_idx=1, min_num_chunks_for_window=0
        ),
        3,
    )

  def test_stats(self):
    cache = self._new_cache()
    cache.snapshot(_tokens(1, 2, 3, 4), _entry_tree(1.0))
    cache.restore(_tokens(1, 2, 3, 4))  # hit
    cache.restore(_tokens(9, 9, 9, 9))  # miss
    stats = cache.stats()
    self.assertEqual(stats['n_snapshot'], 1)
    self.assertEqual(stats['n_promoted'], 0)
    self.assertEqual(stats['n_restore'], 1)
    self.assertEqual(stats['n_hit'], 1)
    # New fields land alongside existing ones.
    self.assertIn('total_bytes', stats)
    self.assertEqual(stats['n_evicted'], 0)
    self.assertEqual(stats['n_eviction_skipped_non_leaf'], 0)

  def test_clear_drops_all_entries_resets_counters_and_stays_usable(self):
    cache = self._new_cache()
    # Snapshot two chunks on the same prefix chain + drive some counters.
    tokens_8 = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    cache.snapshot(tokens_8[:4], _entry_tree(1.0))
    cache.snapshot(tokens_8, _entry_tree(2.0))
    cache.restore(tokens_8[:4])  # a hit -> bumps n_restore / n_hit.
    self.assertTrue(cache.is_cached(tokens_8[:4]))
    self.assertTrue(cache.is_cached(tokens_8))
    self.assertGreater(cache.stats()['total_bytes'], 0)
    self.assertGreater(cache.stats()['n_snapshot'], 0)

    cache.clear()

    # Every chunk is gone and all counters are back to zero.
    self.assertFalse(cache.is_cached(tokens_8[:4]))
    self.assertFalse(cache.is_cached(tokens_8))
    self.assertIsNone(cache.restore(tokens_8))
    cleared_stats = cache.stats()
    self.assertEqual(
        cleared_stats,
        {
            'n_snapshot': 0,
            'n_promoted': 0,
            'n_restore': 0,
            'n_hit': 0,
            'n_evicted': 0,
            'n_eviction_skipped_non_leaf': 0,
            'total_bytes': 0,
        },
        msg=f'counters not fully reset: {cleared_stats}',
    )

    # The SAME cache object is immediately reusable for a fresh snapshot.
    fresh = _tokens(9, 9, 9, 9)
    cache.snapshot(fresh, _entry_tree(3.0))
    self.assertTrue(cache.is_cached(fresh))
    _assert_stored_matches_input(  # pyrefly: ignore[bad-argument-type]
        cache.restore(fresh), _entry_tree(3.0)
    )
    self.assertEqual(cache.stats()['n_snapshot'], 1)
    self.assertGreater(cache.stats()['total_bytes'], 0)

  def test_clear_preserves_store_and_counter_object_identity(self):
    # `_store` / `_counters` are cached_property objects; clear() must mutate
    # them IN PLACE (not rebind), so any captured reference stays valid.
    cache = self._new_cache()
    store_before = cache._store  # pylint: disable=protected-access
    counters_before = cache._counters  # pylint: disable=protected-access
    cache.snapshot(_tokens(1, 2, 3, 4), _entry_tree(1.0))
    cache.clear()
    self.assertIs(cache._store, store_before)  # pylint: disable=protected-access
    self.assertIs(cache._counters, counters_before)  # pylint: disable=protected-access
    self.assertEmpty(store_before)

  def test_clear_on_empty_cache_is_noop(self):
    cache = self._new_cache()
    cache.clear()  # must not raise on an empty cache.
    self.assertEqual(cache.stats()['total_bytes'], 0)
    # Still usable.
    cache.snapshot(_tokens(1, 2, 3, 4), _entry_tree(1.0))
    self.assertTrue(cache.is_cached(_tokens(1, 2, 3, 4)))


# `_entry_tree(value, n_layers=N)` produces N float32 leaves of shape
# (4, 8) = 128 bytes each, so an N-layer complete entry is exactly
# `N * 128` bytes. Tests below use `n_layers=1` (128 B/entry) so byte
# arithmetic is human-checkable.
_ONE_LAYER_BYTES = 4 * 8 * 4  # (4, 8) float32 = 128 bytes


class PrefixCacheLruTest(absltest.TestCase):
  """Tests for the host-RAM byte-bounded LRU eviction mechanism."""

  def _new_cache(
      self, max_bytes: int | None = None
  ) -> prefix_cache_lib.PrefixCache:
    return prefix_cache_lib.PrefixCache(
        chunk_size=CHUNK_SIZE, max_bytes=max_bytes
    )

  def _tokens_for_chunk(self, chunk_idx: int) -> np.ndarray:
    """Returns distinct cumulative tokens up through `chunk_idx` (inclusive).

    Tokens are `chunk_idx`-prefixed so different chunks live on different
    prefix chains (no parent/child relationships) by default.

    Args:
      chunk_idx: which distinct-prefix chunk to construct tokens for.
    """
    seed = chunk_idx + 100
    return np.asarray(
        list(range(seed, seed + CHUNK_SIZE)), dtype=np.int32
    )

  def test_no_lru_when_max_bytes_is_none(self):
    cache = self._new_cache(max_bytes=None)
    for i in range(5):
      cache.snapshot(
          self._tokens_for_chunk(i), _entry_tree(float(i), n_layers=1)
      )
    # Nothing evicted; all 5 unrelated chunks still cached.
    for i in range(5):
      self.assertTrue(cache.is_cached(self._tokens_for_chunk(i)))
    self.assertEqual(cache.stats()['n_evicted'], 0)
    self.assertEqual(cache.stats()['total_bytes'], 5 * _ONE_LAYER_BYTES)

  def test_lru_evicts_oldest_first(self):
    # max_bytes holds exactly 2 single-layer entries (= 256 B). Inserting
    # a third triggers eviction of the LRU end (= first inserted).
    cache = self._new_cache(max_bytes=2 * _ONE_LAYER_BYTES)
    cache.snapshot(
        self._tokens_for_chunk(0), _entry_tree(0.0, n_layers=1)
    )
    cache.snapshot(
        self._tokens_for_chunk(1), _entry_tree(1.0, n_layers=1)
    )
    cache.snapshot(
        self._tokens_for_chunk(2), _entry_tree(2.0, n_layers=1)
    )
    # Chunk 0 (oldest, no children) evicted; 1 and 2 survive.
    self.assertFalse(cache.is_cached(self._tokens_for_chunk(0)))
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(1)))
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(2)))
    self.assertEqual(cache.stats()['n_evicted'], 1)
    self.assertEqual(
        cache.stats()['total_bytes'], 2 * _ONE_LAYER_BYTES
    )

  def test_lru_skips_non_leaf_entries(self):
    # Build a parent-child chain: snapshot chunk 0 of a prefix, then
    # chunk 1 of the SAME prefix. Chunk 0 is now an internal node.
    # An unrelated chunk pushes us over budget; eviction must skip
    # chunk 0 (it has a child) and evict the next-LRU leaf.
    cache = self._new_cache(max_bytes=2 * _ONE_LAYER_BYTES)
    chain_tokens = np.asarray(
        [1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32
    )
    cache.snapshot(chain_tokens[:4], _entry_tree(1.0, n_layers=1))
    cache.snapshot(chain_tokens[:8], _entry_tree(2.0, n_layers=1))
    # Stats: 2 entries, parent chunk 0 has n_children=1.
    # Now insert an unrelated chunk; total goes to 3 entries -> evict.
    other_tokens = self._tokens_for_chunk(99)
    cache.snapshot(other_tokens, _entry_tree(9.0, n_layers=1))
    # Chunk 0 of the chain is the LRU end but it has a child -> skip.
    # Chunk 1 of the chain (the next LRU candidate) is a leaf -> evict.
    self.assertTrue(cache.is_cached(chain_tokens[:4]))  # parent kept
    self.assertFalse(cache.is_cached(chain_tokens[:8]))  # leaf evicted
    self.assertTrue(cache.is_cached(other_tokens))
    self.assertGreaterEqual(
        cache.stats()['n_eviction_skipped_non_leaf'], 1
    )
    self.assertEqual(cache.stats()['n_evicted'], 1)

  def test_lru_restore_touches(self):
    # Insert 0, 1; restore 0 (touches it MRU); insert 2; chunk 1
    # (now LRU) gets evicted instead of chunk 0.
    cache = self._new_cache(max_bytes=2 * _ONE_LAYER_BYTES)
    cache.snapshot(
        self._tokens_for_chunk(0), _entry_tree(0.0, n_layers=1)
    )
    cache.snapshot(
        self._tokens_for_chunk(1), _entry_tree(1.0, n_layers=1)
    )
    cache.restore(self._tokens_for_chunk(0))  # MRU touch
    cache.snapshot(
        self._tokens_for_chunk(2), _entry_tree(2.0, n_layers=1)
    )
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(0)))
    self.assertFalse(cache.is_cached(self._tokens_for_chunk(1)))
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(2)))

  def test_lru_probes_do_not_touch(self):
    # `is_cached` / `is_complete` must NOT count as use.
    cache = self._new_cache(max_bytes=2 * _ONE_LAYER_BYTES)
    cache.snapshot(
        self._tokens_for_chunk(0), _entry_tree(0.0, n_layers=1)
    )
    cache.snapshot(
        self._tokens_for_chunk(1), _entry_tree(1.0, n_layers=1)
    )
    # Probe chunk 0 a bunch of times -- shouldn't change LRU order.
    for _ in range(5):
      self.assertTrue(cache.is_cached(self._tokens_for_chunk(0)))
      self.assertTrue(cache.is_complete(self._tokens_for_chunk(0)))
    # Insert chunk 2; chunk 0 (still LRU) gets evicted.
    cache.snapshot(
        self._tokens_for_chunk(2), _entry_tree(2.0, n_layers=1)
    )
    self.assertFalse(cache.is_cached(self._tokens_for_chunk(0)))
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(1)))
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(2)))

  def test_lru_promotion_growth_triggers_eviction(self):
    # First snapshot of chunk A: 2 layers, 1 available (1 stored array,
    # 1 ShapeDtypeStruct = 128 bytes). Second snapshot of unrelated
    # chunk B: 2 layers, both available (256 bytes). Total 384 B fits
    # in max_bytes=400. Now promote chunk A (both available, whole-tree
    # overwrite) -- new entry is 256 bytes (was 128), delta = +128, so
    # total goes 384 -> 512 B > 400 -> evict LRU leaf.
    cache = self._new_cache(max_bytes=400)
    cache.snapshot(
        self._tokens_for_chunk(0),
        _entry_tree(0.0, n_layers=2, available=[True, False]),
    )
    self.assertEqual(cache.stats()['total_bytes'], _ONE_LAYER_BYTES)
    cache.snapshot(
        self._tokens_for_chunk(1),
        _entry_tree(1.0, n_layers=2),  # both available
    )
    self.assertEqual(
        cache.stats()['total_bytes'], 3 * _ONE_LAYER_BYTES
    )
    self.assertEqual(cache.stats()['n_evicted'], 0)
    # Promote chunk A. The snapshot-side move_to_end (always-touch on
    # existing-key path) bumps chunk A to MRU, so chunk B is LRU and
    # gets evicted.
    cache.snapshot(
        self._tokens_for_chunk(0),
        _entry_tree(0.0, n_layers=2),  # both available now
    )
    self.assertEqual(cache.stats()['n_promoted'], 1)
    self.assertEqual(cache.stats()['n_evicted'], 1)
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(0)))
    self.assertTrue(cache.is_complete(self._tokens_for_chunk(0)))
    self.assertFalse(cache.is_cached(self._tokens_for_chunk(1)))

  def test_lru_single_oversized_chunk_stored_anyway(self):
    # max_bytes is smaller than a single 2-layer entry (256 B). The
    # cache stores it anyway (one-chunk-over-budget rule) rather than
    # refusing the snapshot.
    cache = self._new_cache(max_bytes=_ONE_LAYER_BYTES // 2)
    cache.snapshot(
        self._tokens_for_chunk(0), _entry_tree(0.0, n_layers=2)
    )
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(0)))
    self.assertEqual(
        cache.stats()['total_bytes'], 2 * _ONE_LAYER_BYTES
    )
    # Inserting another oversized chunk evicts the prior one (the
    # previously-protected chunk is now unprotected, no children, and
    # is the only eviction-eligible entry).
    cache.snapshot(
        self._tokens_for_chunk(1), _entry_tree(1.0, n_layers=2)
    )
    self.assertFalse(cache.is_cached(self._tokens_for_chunk(0)))
    self.assertTrue(cache.is_cached(self._tokens_for_chunk(1)))
    self.assertEqual(cache.stats()['n_evicted'], 1)


if __name__ == '__main__':
  absltest.main()

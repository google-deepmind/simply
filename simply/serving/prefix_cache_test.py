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

from collections.abc import MutableSequence, Sequence
import dataclasses
import math
import typing
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.serving import prefix_cache as prefix_cache_lib
from simply.utils import ragged_paged_attention as rpa


CHUNK_SIZE = 4


def _tokens(*ids) -> np.ndarray:
  return np.asarray(ids, dtype=np.int32)


def _mask(
    spec: bool | Sequence[int] | np.ndarray, chunk_size: int = CHUNK_SIZE
) -> np.ndarray:
  """Builds a `bool[chunk_size]` written mask.

  Args:
    spec: `True` / `False` for a full / empty mask, a `[lo, hi)` pair for a
      token range, or an explicit mask.
    chunk_size: the cache's chunk size, which the mask must match: the cache
      intersects it with what the storing node owns, and that is
      `bool[chunk_size]`.

  Returns:
    The mask.
  """
  if isinstance(spec, bool):
    return np.full((chunk_size,), spec, dtype=bool)
  if isinstance(spec, np.ndarray):
    return spec.astype(bool)
  lo, hi = spec
  out = np.zeros((chunk_size,), dtype=bool)
  out[lo:hi] = True
  return out


def _offloaded(leaf_tree: Any) -> Any:
  """A capture in the form the cache is handed it: host-resident.

  The driver offloads before it builds a tile (`page_batcher`), so a test
  playing the driver's part does the same.

  Args:
    leaf_tree: pytree of :class:`rpa.SnapshotChunkLeaf`, still on the device.

  Returns:
    The host-resident tree.
  """
  return rpa.offload_chunk_tree(leaf_tree)


def _entry_tree(
    value: float,
    n_layers: int = 2,
    written: Any = True,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, rpa.SnapshotChunkLeaf]:
  """A `DecodeState`-shaped pytree of :class:`SnapshotChunkLeaf` leaves.

  Each leaf wraps a small on-device `jax.Array` (so `snapshot`/`restore`
  exercise the real HBM<->host `device_put` offload path) and a PER-TOKEN
  written mask over the chunk's tokens.

  Args:
    value: base fill value; layer `i` is filled with `value + i`.
    n_layers: number of per-layer leaves to build.
    written: per-leaf written mask. A single spec broadcasts to all layers; a
      list of specs (length `n_layers`) sets each leaf. Each spec is `True`,
      `False`, a `[lo, hi)` range, or an explicit mask -- see :func:`_mask`. A
      leaf records WHICH TOKENS are real and nothing else; the windows that
      decide what a resume needs live on the cache.
    chunk_size: the chunk size of the cache this will be stored in. The two
      have to agree, because the cache intersects the capture's mask with what
      the storing node owns, which is `bool[chunk_size]`.

  Returns:
    Dict mapping `layer_{i}` to a :class:`SnapshotChunkLeaf`.
  """
  if isinstance(written, list) and len(written) == n_layers:
    per_leaf = list(written)
  else:
    per_leaf = [written] * n_layers
  return {
      f'layer_{i}': rpa.SnapshotChunkLeaf(
          payload=jnp.full((chunk_size, 8), value + i, dtype=jnp.float32),
          written=jnp.asarray(_mask(spec, chunk_size)),
      )
      for i, spec in enumerate(per_leaf)
  }


def _assert_stored_matches_input(
    stored: dict[str, rpa.StoredChunkLeaf],
    expected_input: dict[str, rpa.SnapshotChunkLeaf],
) -> None:
  """Asserts a cache-stored tree matches the original input leaves.

  A stored leaf holds host-resident bytes plus the written mask; a leaf with
  nothing written holds a `jax.ShapeDtypeStruct` instead of bytes (the RAM
  rule), and its shape/dtype still match the input payload.

  Args:
    stored: cache-stored tree, one :class:`rpa.StoredChunkLeaf` per layer.
    expected_input: original `SnapshotChunkLeaf` tree passed to
      `cache.snapshot`.
  """
  assert set(stored.keys()) == set(expected_input.keys()), (
      f'keys differ: {stored.keys()} vs {expected_input.keys()}'
  )
  for k, leaf in expected_input.items():
    want_written = np.asarray(leaf.written)
    np.testing.assert_array_equal(
        stored[k].written, want_written, err_msg=f'{k}: written mask differs'
    )
    payload = stored[k].payload
    if not want_written.any():
      assert isinstance(
          payload, jax.ShapeDtypeStruct
      ), f'{k}: expected ShapeDtypeStruct, got {type(payload)}'
      assert (
          payload.shape == leaf.payload.shape
      ), f'{k}: stored shape {payload.shape} != input {leaf.payload.shape}'
      assert (
          payload.dtype == leaf.payload.dtype
      ), f'{k}: stored dtype {payload.dtype} != input {leaf.payload.dtype}'
      continue
    assert isinstance(
        payload, jax.Array
    ), f'{k}: expected jax.Array, got {type(payload)}'
    np.testing.assert_array_equal(
        np.asarray(payload)[want_written],
        np.asarray(leaf.payload)[want_written],
    )


def _all_nodes(
    cache: prefix_cache_lib.PrefixCache,
) -> list[prefix_cache_lib.PrefixNode]:
  """Every node of `cache`'s trie, the root included.

  Args:
    cache: the cache to walk.

  Returns:
    The nodes, in no particular order.
  """
  out: list[prefix_cache_lib.PrefixNode] = []
  stack: list[prefix_cache_lib.PrefixNode] = [cache.root]
  while stack:
    node = stack.pop()
    out.append(node)
    stack.extend(node.children.values())
  return out


def _stored_stops(
    cache: prefix_cache_lib.PrefixCache,
) -> list[prefix_cache_lib.PrefixNode]:
  """Every node of `cache`'s trie that carries a payload.

  The cache no longer reports its own size, so the tests that used to read
  `len(cache)` count the stored stops themselves. Walking the trie is also a
  slightly stronger statement than a counter would be: what it finds is what
  is REACHABLE from the root.

  A node is now made only for a CHUNK that has none on its path, so this
  counts chunks-on-a-path, not pass boundaries: several passes ending inside
  one chunk set bits on one node's payload and add nothing here.

  Args:
    cache: the cache to walk.

  Returns:
    The stored stops, in no particular order. Branch points hold no payload
    and are skipped -- they are structure, not data.
  """
  return [node for node in _all_nodes(cache) if node.has_payload]


def _n_stops(cache: prefix_cache_lib.PrefixCache) -> int:
  """How many payload-carrying nodes the cache holds."""
  return len(_stored_stops(cache))


def _n_holders(cache: prefix_cache_lib.PrefixCache) -> int:
  """How many DISTINCT chunk buffers the cache holds.

  Counts :class:`ChunkHolder` OBJECTS, not references to them: co-owners of
  one chunk (the two sides of a split) share the box, so this counts what is
  really in host RAM. It is what the sharing tests assert on now that the
  byte total is gone -- every chunk stores the same shape, so the byte total
  was only ever a way of counting these.

  Args:
    cache: the cache to walk.

  Returns:
    The number of distinct chunk buffers in the trie.
  """
  seen = set()
  for node in _stored_stops(cache):
    seen.update(id(holder) for holder in node.holders)
  return len(seen)


def _chunk_indices(
    cache: prefix_cache_lib.PrefixCache, node: prefix_cache_lib.PrefixNode
) -> list[int]:
  """The chunk indices `node` owns KV for, in order.

  A node's holders are POSITIONAL -- a contiguous run starting at the chunk
  its first owned token falls in -- so which chunk each one is for is
  arithmetic rather than a key. This is that arithmetic, kept in one place so
  the tests read like the model rather than like the layout.

  Args:
    cache: the cache the node belongs to.
    node: the node to read.

  Returns:
    The chunk indices, ascending; empty when the node owns no KV.
  """
  first = node.start // cache.chunk_size
  return [first + i for i in range(len(node.holders))]


def _holder(
    cache: prefix_cache_lib.PrefixCache,
    node: prefix_cache_lib.PrefixNode,
    chunk_idx: int,
) -> Any:
  """`node`'s buffer for one chunk, or `None` when it owns none of it."""
  idx = chunk_idx - (node.start // cache.chunk_size)
  if 0 <= idx < len(node.holders):
    return node.holders[idx]
  return None


def _owned(
    cache: prefix_cache_lib.PrefixCache,
    node: prefix_cache_lib.PrefixNode,
    chunk_idx: int,
) -> np.ndarray:
  """Which tokens of one chunk `node`'s interval covers."""
  chunk_start = chunk_idx * cache.chunk_size
  positions = chunk_start + np.arange(1, cache.chunk_size + 1)
  return (positions > node.start) & (positions <= node.end)


def _tile_sizes(tiles: Sequence[prefix_cache_lib.ChunkTile]) -> list[int]:
  """The per-tile token counts of a plan; `[]` for a miss."""
  return [tile.num_tokens for tile in tiles]


def _assert_every_node_earns_its_place(
    test: absltest.TestCase, cache: prefix_cache_lib.PrefixCache
) -> None:
  """Asserts the trie's structural invariant over every node.

  A node is the root, or it OWNS KV, or it is a branch point with at least
  two children. NO EXCEPTIONS: a node that is none of those is one path
  compression should have swallowed, costing a comparison on every descent
  through it and standing for nothing. (`_consolidate` is what enforces it: a
  node with one child and nothing of its own is folded into that child, its
  interval, its holders and its boundaries going with it.)

  A branch point may own KV too, now that ownership is an interval rather
  than a stop: the node where two prompts diverge holds the part of the
  shared chunk that lies below the divergence, which is exactly what both of
  them may read.

  There is no longer any such thing as a node kept alive by a cursor. A
  cursor is a hint, not a claim: when the node it names is folded away it
  retreats and its slot re-captures (see
  `PrefixFrontierTest.test_a_cursor_whose_node_is_folded_away_retreats`).

  Args:
    test: the test case to assert on.
    cache: the cache to walk.
  """
  for node in _all_nodes(cache):
    if node.parent is None:
      test.assertIs(node, cache.root, 'a second root')
      continue
    test.assertTrue(
        node.has_payload or len(node.children) >= 2,
        f'the node ({node.start}, {node.end}] owns no KV and has'
        f' {len(node.children)} children: it stands for nothing',
    )


def _assert_cursor_promises_a_resume(
    test: absltest.TestCase,
    cache: Any,
    tokens: np.ndarray,
    cursor: int,
) -> None:
  """Asserts a slot's cursor is 0 or a position the slot may RESUME at.

  The promise the integer carries, in its strongest form:

      cursor == 0  or  cache.can_resume_at(slot_tokens, cursor)

  "The prefix is there" would be enough to make the descent below it safe,
  but not enough to make it USEFUL: a restore may only land on a recorded
  pass boundary, so a cursor sitting on a position that is merely present is
  one the slot can never resume from. Every way a cursor is set keeps the
  stronger form -- a push sets 0, a restore sets a boundary it just
  verified, and the snapshot walk advances it once, to the pass boundary,
  and only when every store up to it was accepted.

  Args:
    test: the test case to assert on.
    cache: the cache the cursor indexes.
    tokens: the slot's whole token row.
    cursor: the slot's cursor.
  """
  if not cursor:
    return
  test.assertTrue(
      cache.can_resume_at(tokens, cursor),
      f'a cursor at {cursor} names a position the slot cannot resume at',
  )


def _assert_ownership_invariant(
    test: absltest.TestCase, cache: Any
) -> None:
  """Asserts every token of every chunk has exactly one owner on its path.

  THE OWNERSHIP INVARIANT. A node owns the tokens its edge spells,
  `(start, end]`, because that is exactly what reaching it proves:
  everyone who gets there matched those tokens and nothing above. Serving a
  token to a reader who has not matched it is the wrong-answer shape this
  model exists to make impossible.

  Ownership is not STORED any more -- `(start, end]` already says it, so
  the per-chunk `owned` masks were a copy of two integers -- which means the
  invariant is now a statement about the GEOMETRY and about the layout that
  hangs off it:

  1. Every node's interval abuts its parent's and matches its edge, so the
     intervals on a path PARTITION the prompt. That is what makes "one owner
     per token" true by construction rather than by agreement between masks.
  2. A node's holders are a CONTIGUOUS run from the chunk its first owned
     token falls in. They are positional, so a hole would silently renumber
     every later chunk and serve the wrong chunk's KV with no error at all.
  3. Its resume points lie inside its interval, are sorted and unique (both
     `can_resume_at`'s binary search and a split's `searchsorted` need that),
     and each one is backed by the KV of the chunk it falls in.

  Args:
    test: the test case to assert on.
    cache: the cache to walk.
  """
  chunk_size = cache.chunk_size
  for node in _all_nodes(cache):
    parent = node.parent
    if parent is None:
      test.assertIs(node, cache.root, 'a second root')
      test.assertEqual((node.start, node.end), (0, 0), 'the root owns tokens')
    else:
      # 1. THE GEOMETRY. The interval starts where the parent's ends and is
      #    exactly as long as the edge, so no token is owned twice and none
      #    is owned by nobody.
      test.assertEqual(
          node.start,
          parent.end,
          f'the node ending at {node.end} does not abut its parent',
      )
      test.assertEqual(
          node.end,
          node.start + len(node.edge),
          f'the node ({node.start}, {node.end}] claims a different span'
          f' from the {len(node.edge)} tokens its edge spells',
      )
    # 2. CONTIGUITY: as many holders as chunks the interval touches, or none
    #    at all. Anything between would be read as a different chunk.
    spanned = (
        math.ceil(node.end / chunk_size) - (node.start // chunk_size)
        if node.end > node.start
        else 0
    )
    test.assertIn(
        len(node.holders),
        (0, spanned),
        f'the node ({node.start}, {node.end}] spans {spanned} chunks but'
        f' holds {len(node.holders)}: a positional run with a hole in it',
    )
    # 3. THE RESUME POINTS.
    boundaries = list(node.resumable_positions)
    test.assertEqual(
        boundaries,
        sorted(set(boundaries)),
        f'the node ({node.start}, {node.end}] has an unsorted or repeated'
        ' boundary list',
    )
    for position in boundaries:
      test.assertTrue(
          node.start < position <= node.end,
          f'the node ({node.start}, {node.end}] offers a resume at'
          f' {position}, which is not its to offer',
      )
      chunk_idx = (position - 1) // chunk_size
      test.assertIsNotNone(
          _holder(cache, node, chunk_idx),
          f'the node ({node.start}, {node.end}] offers a resume at'
          f' {position} but holds no KV for chunk {chunk_idx}',
      )
  # 4. THE LEDGER. Eviction is the one thing that reads a number the cache
  #    maintains rather than derives -- how many host bytes it holds -- so
  #    every check of the structure also checks that number against the
  #    structure. A ledger that has drifted would evict against a fiction.
  test.assertEqual(
      cache.nbytes,
      _walked_nbytes(cache),
      'the byte ledger has drifted from what the trie actually holds',
  )


def _walked_nbytes(cache: prefix_cache_lib.PrefixCache) -> int:
  """The total host RAM bytes, recomputed from a walk of the trie."""
  seen: dict[int, int] = {}
  stack = [cache.root]
  while stack:
    node = stack.pop()
    for holder in node.holders:
      seen[id(holder)] = holder.nbytes
    stack.extend(node.children.values())
  return sum(seen.values())


def _assert_stored_trees_agree(
    test: absltest.TestCase, got: Any, want: Any
) -> None:
  """Asserts two cache-stored chunk trees hold the same thing.

  Compares what a reader can actually observe: which tokens each leaf claims,
  and the bytes of the tokens it claims. Unwritten bytes are meaningless by
  construction (the RAM rule), so they are not compared.

  Args:
    test: the test case to assert on.
    got: one stored tree.
    want: the other.
  """
  test.assertIsNotNone(got)
  test.assertIsNotNone(want)
  test.assertEqual(set(got.keys()), set(want.keys()))
  for name, want_leaf in want.items():
    written = np.asarray(want_leaf.written)
    np.testing.assert_array_equal(
        np.asarray(got[name].written), written, err_msg=f'{name}: mask differs'
    )
    if not written.any():
      continue
    np.testing.assert_array_equal(
        np.asarray(got[name].payload)[written],
        np.asarray(want_leaf.payload)[written],
        err_msg=f'{name}: bytes differ',
    )


def _resumable(
    cache: Any, tokens: np.ndarray, position: int
) -> np.ndarray:
  """Which tokens of the chunk `position` falls in this reader may resume at.

  A chunk-shaped view of `can_resume_at`, kept because a chunk is the unit
  the tests reason in. It is DERIVED, not stored: the boundaries live on the
  nodes as absolute positions now, and which of them a given reader may see
  depends on what it has witnessed -- so this asks the public probe once per
  token rather than reading any node's list directly.

  Args:
    cache: the cache to read.
    tokens: the reader's whole token row.
    position: any position inside the chunk of interest.

  Returns:
    `bool[chunk_size]`; offset `i` is the position `chunk_start + i + 1`.
  """
  chunk_start = ((position - 1) // cache.chunk_size) * cache.chunk_size
  return np.array(
      [
          cache.can_resume_at(tokens, chunk_start + offset + 1)
          for offset in range(cache.chunk_size)
      ],
      dtype=bool,
  )


def _composed(
    cache: Any, tokens: np.ndarray, position: int
) -> tuple[Any, np.ndarray] | None:
  """The chunk containing `position`, as a FULL-ROW reader composes it.

  What a chunk holds is no longer one object to look up: it is composed from
  every slice on the path that owns part of it, masked by the part this
  caller has WITNESSED (:meth:`PrefixCache._slices_for_chunk`). That makes
  the answer depend on the reader -- two prompts that fork inside the chunk
  get different composites, which is the point -- so a test that wants to
  know what a SLOT would inject has to ask with the slot's whole token row,
  as `restore_chunk_tiles` does. `stored_at` takes a prefix and is the right
  accessor only when the prefix is really all the caller knows.

  Args:
    cache: the cache to read.
    tokens: the reader's whole token row.
    position: a token position.

  Returns:
    `(tree, written)`, or `None` when nothing there is readable.
  """
  chunk_end = min(
      len(tokens),
      ((position - 1) // cache.chunk_size + 1) * cache.chunk_size,
  )
  tree = cache.stored_at(tokens[:chunk_end])
  if tree is None:
    return None
  written = np.zeros((cache.chunk_size,), dtype=bool)
  is_leaf = lambda x: isinstance(x, rpa.StoredChunkLeaf)
  for leaf in jax.tree_util.tree_leaves(tree, is_leaf=is_leaf):
    written = np.logical_or(written, np.asarray(leaf.written))
  return tree, written


def _values(
    tree: Any, layer: str = 'layer_0'
) -> tuple[list[float], list[int]]:
  """The per-token tracer value and written mask of one leaf of a chunk.

  Args:
    tree: a composed (or stored) chunk tree.
    layer: which leaf to read.

  Returns:
    `(values, written)`, one entry per token of the chunk.
  """
  leaf = tree[layer]
  return (
      np.asarray(leaf.payload)[:, 0].tolist(),
      np.asarray(leaf.written).astype(int).tolist(),
  )


class PrefixCache(prefix_cache_lib.PrefixCache):
  """Test wrapper for PrefixCache providing snapshot test helpers."""

  def node_at(
      self, slot_tokens: np.ndarray, position: int | None = None
  ) -> prefix_cache_lib.PrefixNode | None:
    if position is None:
      position = len(slot_tokens)
    if position <= 0 or position > len(slot_tokens):
      return None
    node = self.root
    while node.end < position:
      child = node.children.get(slot_tokens[node.end])
      if child is None:
        return None
      matched_len = child.common_prefix_len(
          slot_tokens[child.start : min(position, child.end)]
      )
      if matched_len < min(position - child.start, child.length):
        return None
      node = child
    if node.start < position <= node.end:
      return node
    return None

  def stored_at(
      self, token_prefix: np.ndarray
  ) -> prefix_cache_lib.StoredTree | None:
    if len(token_prefix) == 0:
      return None
    position = len(token_prefix)
    chunk_idx = (position - 1) // self.chunk_size
    chunk_start = chunk_idx * self.chunk_size
    node = self.root
    path = []
    witnessed = 0
    while node.end < position:
      child = node.children.get(token_prefix[node.end])
      if child is None:
        break
      matched_len = child.common_prefix_len(
          token_prefix[child.start : min(position, child.end)]
      )
      if matched_len == 0:
        break
      path.append(child)
      witnessed = node.end + matched_len
      node = child
    if not path or witnessed <= chunk_start:
      return None
    slices = []
    for n in path:
      for idx, holder in enumerate(n.holders):
        holder_chunk = (n.start // self.chunk_size) + idx
        if holder_chunk == chunk_idx:
          tile_start = max(n.start, holder_chunk * self.chunk_size)
          tile_end = min(n.end, (holder_chunk + 1) * self.chunk_size)
          tile = prefix_cache_lib.ChunkTile(
              tree=holder.tree, start=tile_start, end=tile_end
          )
          slices.append(tile.clamp(tile_start, min(tile_end, witnessed)))
    if not slices:
      return None
    tree = slices[0]
    for s in slices[1:]:
      tree = rpa.merge_chunk_trees(tree, s)
    return tree

  def is_cached(
      self, slot_tokens: np.ndarray, position: int | None = None
  ) -> bool:
    if position is None:
      position = len(slot_tokens)
    if position <= 0 or position > len(slot_tokens):
      return False
    node = self.node_at(slot_tokens, position)
    if node is None:
      return False
    if position % self.chunk_size == 0:
      return True
    return self.can_resume_at(slot_tokens, position)

  def can_resume_at(
      self, slot_tokens: np.ndarray, position: int | None = None
  ) -> bool:
    if position is None:
      position = len(slot_tokens)
    if position <= 0 or position > len(slot_tokens):
      return False
    node = self.node_at(slot_tokens, position)
    if node is None:
      return False
    idx = position - node.start - 1
    del idx
    return node.deepest_resumable(position) == position

  def snapshot_at(
      self,
      slot_tokens: np.ndarray,
      position: int,
      leaf_tree: Any,
  ) -> bool:
    floor = ((position - 1) // self.chunk_size) * self.chunk_size
    if floor and not self.is_cached(slot_tokens, floor):
      return False
    tile = prefix_cache_lib.ChunkTile(
        rpa.offload_chunk_tree(leaf_tree), floor, position
    )
    reached = self.store_tiles(slot_tokens, [tile])
    return reached == position

  def snapshot(
      self,
      token_prefix: np.ndarray,
      leaf_tree: Any,
  ) -> bool:
    return self.snapshot_at(
        token_prefix,
        len(token_prefix),
        leaf_tree,
    )


class TrieStructureTest(parameterized.TestCase):
  """The shape of the radix token trie: shared edges, splits, stops."""

  def _new_cache(self) -> PrefixCache:
    return PrefixCache(chunk_size=CHUNK_SIZE)

  def test_shared_prefixes_share_nodes(self):
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = _tokens(1, 2, 3, 4, 9, 9, 9, 9)  # same chunk 0, different chunk 1
    cache.snapshot(a[:4], _entry_tree(0.0))
    cache.snapshot(a, _entry_tree(1.0))
    cache.snapshot(b, _entry_tree(2.0))
    # Three owners: the shared node over chunk 0, plus one per prompt over
    # chunk 1. (A and B's chunk-0 captures were folded into one node by
    # consolidation before B arrived; storing B split it apart again at the
    # divergence, which is what sharing a prefix MEANS here.)
    self.assertEqual(_n_stops(cache), 3)
    self.assertEqual(
        sorted((n.start, n.end) for n in _stored_stops(cache)),
        [(0, 4), (4, 8), (4, 8)],
    )
    self.assertTrue(cache.is_cached(b[:4]))  # reached through the shared node
    _assert_stored_matches_input(  # pyrefly: ignore[bad-argument-type]
        cache.stored_at(b), _entry_tree(2.0)  # pyrefly: ignore[bad-argument-type]
    )
    # The shared chunk really is ONE object, read by both.
    self.assertEqual(_n_holders(cache), 3)  # chunk 0 + one chunk 1 each
    _assert_ownership_invariant(self, cache)

  def test_a_diverging_prefix_reads_only_what_it_matched(self):
    # WAS: a diverging prefix is a flat miss. Under witnessed reads it is a
    # PARTIAL hit -- and that is the point of the model: the tokens below the
    # divergence really are this prompt's own, so it may have them, and the
    # token it disagrees about it may not. What it must never get is a resume
    # point it did not witness.
    cache = self._new_cache()
    stored = _tokens(1, 2, 3, 4)
    diverging = _tokens(1, 2, 3, 5)
    cache.snapshot(stored, _entry_tree(7.0, n_layers=1))
    got = cache.stored_at(diverging)
    assert got is not None
    values, written = _values(got)
    self.assertEqual(written, [1, 1, 1, 0])  # never token 4, where they differ
    self.assertEqual(values[:3], [7.0] * 3)  # ...and the three it matched
    for position in range(1, 5):
      self.assertFalse(cache.can_resume_at(diverging, position), position)
    self.assertFalse(cache.restore_chunk_tiles(diverging, 0, 4))

  def test_lookup_is_dtype_independent(self):
    cache = self._new_cache()
    cache.snapshot(np.array([1, 2, 3, 4], dtype=np.int32), _entry_tree(0.0))
    self.assertTrue(cache.is_cached(np.array([1, 2, 3, 4], dtype=np.int64)))

  def test_the_index_keeps_no_hash_or_key(self):
    # The radix trie indexes the TOKENS themselves: there is no digest to
    # collide, and no key to keep in sync with what is stored. What a node
    # holds is WHERE it is (edge / end / parent / children), the KV of the
    # interval its edge spells (one slice per chunk that interval touches),
    # and the POSITIONS a pass ended at inside it. A slice is bytes plus a
    # a run of chunk buffers and nothing else: WHICH tokens of them are its
    # own is not stored at all, because `(start, end]` already says it.
    self.assertEqual(
        {f.name for f in dataclasses.fields(prefix_cache_lib.PrefixNode)},
        {
            'edge',
            'end',
            'parent',
            'children',
            'holders',
            'resumables',
        },
    )
    # No `pins`: a cursor is a HINT, not a claim. It had one consumer -- a
    # node a live cursor sat on was not folded -- and that turned out to be
    # work-avoidance rather than correctness, since a cursor whose node goes
    # away simply retreats and its slot re-captures. Eviction keeps its
    # recency OUTSIDE the node -- one LRU order for the whole cache -- and on
    # the buffer only the BYTES it would free, never a count of who is
    # looking at it. `nbytes` is a field rather than a derived property
    # because the buffer is MUTABLE: a merge rebinds `tree` in place so every
    # node that co-owns it sees the union, and the byte count has to move
    # with it.
    self.assertEqual(
        {f.name for f in dataclasses.fields(prefix_cache_lib.ChunkHolder)},
        {'tree', 'nbytes'},
    )
    # A resume point is a fact about a POSITION, and positions are what a
    # node owns -- so it lives there, as absolute token positions, not as a
    # mask per chunk. A long interval keeps a handful of ints rather than a
    # `chunk_size` mask for every chunk it touches.
    #
    # And there is no per-chunk claim OBJECT at all any more: what a node
    # owns of a chunk is `(start, end]` intersected with that chunk, which
    # two integers already say. The class that used to store it is gone
    # rather than moved, which is the strongest form of "not duplicated".
    self.assertFalse(
        hasattr(prefix_cache_lib, '_ChunkSlice'),
        'ownership is derived from the interval; nothing should store it',
    )
    self.assertEmpty([
        name
        for name in dir(prefix_cache_lib)
        if 'hash' in name.lower() or 'digest' in name.lower()
    ])

  def test_every_node_is_the_root_a_stop_or_a_branch(self):
    # THE STRUCTURAL INVARIANT. A node exists for exactly two reasons: it
    # OWNS KV, or two cached prompts diverge at it. Anything else is a node
    # path compression should have swallowed -- and now does, since
    # `_consolidate` folds a node with one child into that child. What is
    # left is proportional to the trie's real shape rather than to how many
    # passes happened to end where.
    cache = self._new_cache()
    a = _tokens(*range(1, 21))
    b = np.concatenate([a[:6], _tokens(91, 92, 93, 94, 95, 96)])
    c = np.concatenate([a[:13], _tokens(81, 82, 83)])
    for tokens, positions in (
        (a, (4, 5, 7, 8, 12, 13, 16, 20)),  # boundaries inside chunks
        (b, (7, 8, 12)),  # diverges inside chunk 1: splits an edge
        (c, (14, 16)),  # diverges inside chunk 3
    ):
      for position in positions:
        cache.snapshot_at(
            tokens,
            position,
            _entry_tree(float(position), n_layers=1),
        )
    _assert_every_node_earns_its_place(self, cache)
    # ...and the trie really is the shape this is asserting about: it has
    # branch points, and far fewer nodes than there were pass boundaries.
    branches = [node for node in _all_nodes(cache) if len(node.children) >= 2]
    self.assertNotEmpty(branches)
    # The holders are a POSITIONAL run: entry `i` is the chunk `i` after the
    # one the node's first owned token falls in, with no key to check it
    # against, so the run has to line up with the interval exactly. (The
    # global check is in `_assert_ownership_invariant`; this spells out what
    # it means for a node that really owns several chunks.)
    for node in _stored_stops(cache):
      indices = _chunk_indices(cache, node)
      self.assertEqual(
          indices, list(range(indices[0], indices[0] + len(indices)))
      )
      for chunk_idx, holder in zip(indices, node.holders, strict=True):
        self.assertIs(_holder(cache, node, chunk_idx), holder)
        self.assertTrue(_owned(cache, node, chunk_idx).any())
    self.assertTrue(
        any(len(node.holders) > 1 for node in _stored_stops(cache)),
        'no node here owns more than one chunk, so the run is untested',
    )
    # A branch point owns KV too: the part of the shared chunk below the
    # divergence is exactly what both sides may read.
    self.assertTrue(any(node.has_payload for node in branches))
    self.assertLess(_n_stops(cache), 13)  # 13 positions were stored
    _assert_ownership_invariant(self, cache)

  def test_a_node_with_one_child_is_folded_into_it(self):
    # CONSOLIDATION. A node that merely marks where some pass stopped makes
    # every later descent through it one comparison longer, and makes the
    # chunk it half-owns something to compose on every read. So when it has a
    # single child and nobody is parked on it, the two intervals become one:
    # the child adopts its tokens AND its slices, and the node goes.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    cache.snapshot_at(tokens, 4, _entry_tree(4.0))
    self.assertEqual(
        sorted((n.start, n.end) for n in _stored_stops(cache)), [(0, 4)]
    )
    cache.snapshot_at(tokens, 6, _entry_tree(6.0))
    # One node again, owning BOTH chunks over the whole interval it now
    # spells -- not a node per stop.
    self.assertEqual(
        sorted((n.start, n.end) for n in _stored_stops(cache)), [(0, 6)]
    )
    merged = cache.node_at(tokens, 6)
    assert merged is not None
    self.assertEqual((merged.start, merged.end), (0, 6))
    self.assertEqual(_chunk_indices(cache, merged), [0, 1])
    np.testing.assert_array_equal(merged.edge, tokens[:6])  # edges merged
    # The boundaries are CONCATENATED: the parent's all lie below the
    # child's, so the union of two sorted lists is one sorted list.
    self.assertEqual(list(merged.resumable_positions), [4, 6])
    self.assertEqual(sorted(n.end for n in _all_nodes(cache)), [0, 6])
    # Nothing was lost in the fold: both stops still read, and both are
    # still resume points.
    for position in (4, 6):
      self.assertTrue(cache.can_resume_at(tokens, position), position)
    self.assertEqual(_values(cache.stored_at(tokens[:4]))[0], [4.0] * 4)
    cache.snapshot_at(tokens, 8, _entry_tree(8.0))
    self.assertEqual(sorted(n.end for n in _all_nodes(cache)), [0, 8])
    _assert_every_node_earns_its_place(self, cache)
    _assert_ownership_invariant(self, cache)

  def test_a_shared_first_token_is_confirmed_against_the_edge(self):
    # Children are bucketed by the first token of their edge, so a prompt
    # that merely STARTS the same lands on the same child. The edge
    # comparison decides what it may have: the one token it really shares,
    # and not a byte of the three it does not.
    cache = self._new_cache()
    theirs = _tokens(5, 6, 7, 8)
    mine = _tokens(5, 9, 9, 9)
    cache.snapshot(theirs, _entry_tree(7.0, n_layers=1))
    root = cache.root
    self.assertEqual(list(root.children), [theirs[0]])  # one bucket
    got = cache.stored_at(mine)
    assert got is not None
    self.assertEqual(_values(got)[1], [1, 0, 0, 0])  # token 5 and no more
    # Nothing about that one token is a place to resume, so the prompt gets
    # no plan at all -- a shared first token is not a shared prefix.
    self.assertFalse(cache.restore_chunk_tiles(mine, 0, 4))
    for position in range(1, 5):
      self.assertFalse(cache.can_resume_at(mine, position), position)

  def test_a_partial_stop_is_a_token_of_an_interval(self):
    # The index has had three shapes: a chunk node keyed by its hash, then a
    # node per stop, then a node per chunk. Now a node owns an INTERVAL and
    # a stop is just a token inside it -- one bit of the slice's `resumable`
    # mask. Nothing about the trie's shape records where a pass ended.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6)
    cache.snapshot(tokens[:4], _entry_tree(0.0))
    cache.snapshot(tokens, _entry_tree(1.0))  # a pass boundary inside chunk 1
    self.assertEqual(_n_stops(cache), 1)  # one interval, (0, 6]
    owner = cache.root.children[tokens[0]]
    self.assertEqual((owner.start, owner.end), (0, 6))
    np.testing.assert_array_equal(owner.edge, tokens)
    self.assertEmpty(owner.children)
    self.assertEqual(_chunk_indices(cache, owner), [0, 1])  # both it touches
    # It owns the WHOLE of chunk 0 and the first half of chunk 1 -- the part
    # of each chunk its interval covers, and no more.
    np.testing.assert_array_equal(_owned(cache, owner, 0), [True] * 4)
    np.testing.assert_array_equal(
        _owned(cache, owner, 1), [True, True, False, False]
    )
    # Every position of the prompt resolves to it, whichever chunk it is in.
    for position in range(1, 7):
      self.assertIs(cache.node_at(tokens, position), owner)
    # ...but only the position a pass really ended at is a stop.
    self.assertTrue(cache.is_cached(tokens, 6))
    self.assertFalse(cache.is_cached(tokens, 5))
    np.testing.assert_array_equal(
        _resumable(cache, tokens, 6), [False, True, False, False]
    )
    _assert_ownership_invariant(self, cache)

  def test_diverging_mid_edge_splits_it(self):
    # Two prompts that share the first two tokens of chunk 1 must SHARE
    # them: chunk-hash keying gave them unrelated chunk-1 entries, a token
    # trie splits the edge where they actually diverge -- and the OWNERSHIP
    # splits with it, the shared node keeping the tokens below the cut.
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = _tokens(1, 2, 3, 4, 5, 6, 9, 9)
    cache.snapshot(a[:4], _entry_tree(0.0))
    cache.snapshot(a, _entry_tree(1.0))
    cache.snapshot(b, _entry_tree(2.0))
    split = cache.root.children[a[0]]
    self.assertEqual((split.start, split.end), (0, 6))  # where they diverge
    self.assertLen(split.children, 2)
    # The split point is not "structure, not data" any more: it owns the part
    # of chunk 1 that lies below the divergence, which is precisely what both
    # prompts may read.
    np.testing.assert_array_equal(
        _owned(cache, split, 1), [True, True, False, False]
    )
    self.assertEqual(_n_stops(cache), 3)  # the split point plus one tail each
    # Each prompt's chunk 1 composes as: the shared tokens from the split
    # point, its own above. Sharing exactly as far as the tokens agree is the
    # whole reason to split the edge where they diverge.
    self.assertEqual(_values(cache.stored_at(a))[0], [1.0, 1.0, 1.0, 1.0])
    self.assertEqual(_values(cache.stored_at(b))[0], [1.0, 1.0, 2.0, 2.0])
    self.assertEqual(_values(cache.stored_at(a))[1], [1, 1, 1, 1])
    self.assertEqual(_values(cache.stored_at(b))[1], [1, 1, 1, 1])
    _assert_ownership_invariant(self, cache)

  def test_a_split_divides_the_ownership_and_copies_no_bytes(self):
    # A stop inside an existing interval splits it -- a node may only record
    # a boundary it OWNS, and ownership ends at a node's end. What the
    # split moves is the CLAIM, not the KV: both sides go on pointing at the
    # same payload box and simply own different tokens of it. (Before the
    # interval model this was a copy of the whole chunk per stop, which is
    # what multiplied the bytes.)
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7)
    cache.snapshot(tokens[:4], _entry_tree(0.0, n_layers=1))
    # Chunk 1's first three tokens, which is all a pass ending at 7 can hold.
    cache.snapshot(tokens[:7], _entry_tree(1.0, n_layers=1, written=(0, 3)))
    self.assertEqual(_n_stops(cache), 1)  # one interval, (0, 7]
    owner = _stored_stops(cache)[0]
    holders_before = {
        chunk_idx: _holder(cache, owner, chunk_idx)
        for chunk_idx in _chunk_indices(cache, owner)
    }
    # A pass that ended at 6 -- earlier than the one that made the node.
    # Nothing new reaches host RAM (the capture holds no token the chunk is
    # missing), but the prefix IS held, so the store reports True and the
    # position becomes a resume point -- and for that the interval has to be
    # cut at 6.
    self.assertTrue(
        cache.snapshot(tokens[:6], _entry_tree(2.0, n_layers=1, written=(0, 2)))
    )
    below = cache.root.children[tokens[0]]
    self.assertEqual((below.start, below.end), (0, 6))
    above = below.children[tokens[6]]
    self.assertEqual((above.start, above.end), (6, 7))
    self.assertEqual(_n_stops(cache), 2)
    # THE MASKS WERE SLICED WITH THE TOKENS...
    np.testing.assert_array_equal(_owned(cache, below, 0), [True] * 4)
    np.testing.assert_array_equal(
        _owned(cache, below, 1), [True, True, False, False]
    )
    np.testing.assert_array_equal(
        _owned(cache, above, 1), [False, False, True, False]
    )
    # ...AND SO WERE THE BOUNDARIES: each side keeps the ones its interval
    # contains, which is what makes them still findable from either half.
    self.assertEqual(list(below.resumable_positions), [4, 6])
    self.assertEqual(list(above.resumable_positions), [7])
    self.assertIs(_holder(cache, below, 1), _holder(cache, above, 1))
    _assert_stored_trees_agree(
        self, _holder(cache, below, 1).tree, holders_before[1].tree
    )
    self.assertIs(_holder(cache, below, 0), holders_before[0])
    self.assertEqual(_n_holders(cache), 2)  # two chunks, two buffers
    # The two halves compose back into exactly the chunk that was there.
    self.assertTrue(cache.can_resume_at(tokens, 6))
    self.assertTrue(cache.can_resume_at(tokens, 7))
    np.testing.assert_array_equal(
        _resumable(cache, tokens, 7), [False, True, True, False]
    )
    _assert_stored_matches_input(  # pyrefly: ignore[bad-argument-type]
        cache.stored_at(tokens[:7]),  # pyrefly: ignore[bad-argument-type]
        _entry_tree(1.0, n_layers=1, written=(0, 3)),
    )
    _assert_every_node_earns_its_place(self, cache)
    _assert_ownership_invariant(self, cache)

  def test_restore_chunk_tiles_never_moves_a_slot_backwards(self):
    # A slot that already sits at or past the deepest resume point must not
    # be handed a shallower one.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    cache.snapshot(tokens[:4], _entry_tree(0.0))
    cache.snapshot(tokens[:5], _entry_tree(1.0))  # shallow boundary, chunk 1
    cache.snapshot(tokens[:8], _entry_tree(2.0))  # deeper boundary, chunk 1
    # From a fresh slot: the deepest resume point wins, and the shallower one
    # is skipped -- the resume point's own payload covers those tokens.
    plan = cache.restore_chunk_tiles(tokens, 0, 12)
    self.assertEqual(_tile_sizes(plan), [4, 4])
    assert plan
    self.assertEqual(plan[-1].end, 8)
    # From a slot already AT the deepest resume point: nothing. (Chunk 2 is
    # stored, so the walk really does reach it and really does decline.)
    self.assertFalse(
        cache.restore_chunk_tiles(tokens, 8, 12)
    )
    # From the chunk boundary below it: only what is strictly deeper, and the
    # injection continues exactly where the slot is (4 -> 8).
    plan = cache.restore_chunk_tiles(tokens, 4, 12)
    self.assertEqual(_tile_sizes(plan), [4])
    assert plan
    self.assertEqual(plan[-1].end, 8)

  def test_a_slot_parked_mid_chunk_is_caught_up_from_its_chunk_floor(self):
    # WAS: refused. Every tile starts at a chunk floor, so a slot sitting at
    # 5 could not be advanced by the stop at 7 without re-injecting tokens 5
    # and 6 -- and since pass boundaries stopped being chunk-aligned a slot
    # almost never sits on a boundary, so "refused" meant "never caught up".
    #
    # Now the tile is the range `[start, position)` and the slot lands on its
    # `end`. The device half is `rpa.DecodeState.inject_chunk`, which writes
    # only that range into the page: the tokens below `start` are the ones
    # the slot computed itself and are left alone.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    cache.snapshot(tokens[:4], _entry_tree(0.0))
    cache.snapshot(tokens[:5], _entry_tree(1.0, written=(0, 1)))
    cache.snapshot(tokens[:7], _entry_tree(1.0, written=(0, 3)))
    plan = cache.restore_chunk_tiles(tokens, 5, 8)
    assert plan
    self.assertEqual(plan[-1].end, 7)
    # ONE tile, paying from where the slot SITS (5) rather than from the
    # chunk floor -- which is what the coverage check now compares against.
    self.assertEqual(_tile_sizes(plan), [2])
    self.assertEqual(sum(_tile_sizes(plan)), plan[-1].end - 5)
    # A slot on the boundary below gets the whole chunk, for the same reason.
    self.assertEqual(
        _tile_sizes(cache.restore_chunk_tiles(tokens, 4, 8)), [3]
    )

  def test_restore_chunk_tiles_never_passes_max_position(self):
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    for k in (4, 8, 11):
      cache.snapshot(tokens[:k], _entry_tree(float(k)))
    # `max_position` is the slot's prefill end: a restore may not overshoot.
    plan = cache.restore_chunk_tiles(tokens, 0, 9)
    self.assertEqual(_tile_sizes(plan), [4, 4])
    plan = cache.restore_chunk_tiles(tokens, 0, 11)
    self.assertEqual(_tile_sizes(plan), [4, 4, 3])

  def test_the_deepest_resume_point_in_a_chunk_wins(self):
    # Two passes ended inside chunk 1, at 5 and at 7. They share one node and
    # one payload, so which one a restore takes is decided by the payload's
    # `resumable` mask -- highest set bit at or below `max_position`.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    cache.snapshot_at(tokens, 4, _entry_tree(0.0))
    cache.snapshot_at(tokens, 5, _entry_tree(1.0))
    cache.snapshot_at(tokens, 7, _entry_tree(2.0))
    self.assertEqual(_n_stops(cache), 1)  # one interval owns both chunks
    np.testing.assert_array_equal(
        _resumable(cache, tokens, 7), [True, False, True, False]
    )
    # The deeper one, when the slot may go that far...
    plan = cache.restore_chunk_tiles(tokens, 0, 8)
    assert plan
    self.assertEqual(plan[-1].end, 7)
    self.assertEqual(_tile_sizes(plan), [4, 3])
    # ...and the shallower one when `max_position` falls between them: a
    # restore may never move a slot past its own prefill end.
    plan = cache.restore_chunk_tiles(tokens, 0, 6)
    assert plan
    self.assertEqual(plan[-1].end, 5)
    self.assertEqual(_tile_sizes(plan), [4, 1])
    # What the caller installs is the POSITION and nothing else: a plan has
    # no node any more, because a cursor has nowhere to put one.
    self.assertFalse(hasattr(plan, 'node'))
    self.assertEqual(_tile_sizes(plan)[-1], plan[-1].end - CHUNK_SIZE)
    _assert_ownership_invariant(self, cache)

  @parameterized.named_parameters(
      dict(testcase_name='_one_shared_token', shared=1),
      dict(testcase_name='_two_shared_tokens', shared=2),
  )
  def test_a_capture_on_someone_elses_sliver_is_refused(self, shared):
    # A PREFIX IS ONLY THERE IF IT IS PAVEABLE. Witnessed reads make "is the
    # preceding chunk on this path?" too weak a question: a prompt that
    # shares one token with a cached one has WITNESSED a sliver of its chunk
    # 0, which is real and readable but is not something a restore could
    # inject -- a restore takes whole chunks. Building a deeper capture on
    # that sliver would leave the node owning tokens of two chunks and KV
    # for only one, and a positional run with a hole in it serves one
    # chunk's KV as another's. So the store is refused, and refused
    # QUIETLY -- a miss, not a raise, since a serving loop is on the other
    # end of it.
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = np.concatenate([a[:shared], _tokens(*range(91, 95 - shared))])
    self.assertTrue(
        cache.snapshot_at(
            b, 4, _entry_tree(5.0, n_layers=1)
        )
    )
    # A's chunk 0 is NOT cached; all it has is the shared sliver of B's.
    self.assertFalse(
        cache.snapshot_at(
            a, 8, _entry_tree(1.0, n_layers=1)
        )
    )
    # ...and the one-shot API, which is the same call underneath.
    self.assertFalse(cache.snapshot(a, _entry_tree(1.0, n_layers=1)))
    self.assertFalse(cache.is_cached(a, 8))
    self.assertFalse(cache.restore_chunk_tiles(a, 0, 8))
    # B is untouched by the attempt.
    self.assertEqual(_values(cache.stored_at(b))[0], [5.0] * 4)
    _assert_ownership_invariant(self, cache)
    _assert_every_node_earns_its_place(self, cache)

  def test_the_in_order_walk_over_the_same_shape_is_unaffected(self):
    # The other half: the refusal must not cost the batcher anything. Its
    # snapshot walk stores chunk by chunk from the cursor, so A's chunk 0
    # lands first, gives A a path of its own, and A's chunk 1 builds on
    # THAT. Same two prompts, same shared first token, everything cached.
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = _tokens(1, 91, 92, 93)
    cache.snapshot_at(b, 4, _entry_tree(5.0, n_layers=1))
    cursor = 0
    for position in (4, 8):
      self.assertTrue(
          cache.snapshot_at(
              a,
              position,
              _entry_tree(1.0, n_layers=1),
          )
      )
      cursor = position
    self.assertEqual(cursor, 8)
    # Each prompt reads its own KV above the token they really do share.
    self.assertEqual(_values(cache.stored_at(a))[0], [1.0] * 4)
    self.assertEqual(_values(cache.stored_at(b))[0], [5.0, 5.0, 5.0, 5.0])
    self.assertEqual(_values(cache.stored_at(a[:4]))[0], [5.0, 1.0, 1.0, 1.0])
    plan = cache.restore_chunk_tiles(a, 0, 8)
    assert plan
    self.assertEqual(plan[-1].end, 8)
    # Chunk 0 is DIVIDED by the fork at token 1, so it comes back as two
    # tiles over the same page -- `[0, 1)` from the node both prompts reach
    # and `[1, 4)` from A's own -- which the inject writes as two disjoint
    # token ranges into one column rather than merging them on the host.
    self.assertEqual(_tile_sizes(plan), [1, 3, 4])
    _assert_ownership_invariant(self, cache)

  def test_a_stop_whose_chunk_floor_is_missing_is_not_stored(self):
    # A restore injects every chunk along the path, so a stop with no
    # payload at the chunk boundary below it could never be resumed.
    cache = self._new_cache()
    self.assertFalse(
        cache.snapshot(_tokens(1, 2, 3, 4, 5, 6, 7, 8), _entry_tree(0.0))
    )
    self.assertEmpty(_stored_stops(cache))


class PrefixCacheTest(absltest.TestCase):

  def _new_cache(self, **kwargs) -> PrefixCache:
    return PrefixCache(
        chunk_size=kwargs.pop('chunk_size', CHUNK_SIZE),
        **kwargs,
    )

  def test_round_trip_one_chunk(self):
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0)
    cache.snapshot(tokens, entry_tree)
    got = cache.stored_at(tokens)
    _assert_stored_matches_input(got, entry_tree)  # pyrefly: ignore[bad-argument-type]

  def test_round_trip_multiple_chunks(self):
    cache = self._new_cache()
    tokens_8 = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    entry_a = _entry_tree(1.0)
    entry_b = _entry_tree(2.0)
    cache.snapshot(tokens_8[:4], entry_a)
    cache.snapshot(tokens_8, entry_b)
    _assert_stored_matches_input(cache.stored_at(tokens_8[:4]), entry_a)  # pyrefly: ignore[bad-argument-type]
    _assert_stored_matches_input(cache.stored_at(tokens_8), entry_b)  # pyrefly: ignore[bad-argument-type]

  def test_cache_is_per_process_not_shared_across_instances(self):
    # The host-RAM cache is process-local: a second instance starts cold
    # and does NOT see entries snapshotted into a different instance.
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0)
    writer = PrefixCache(chunk_size=CHUNK_SIZE)
    writer.snapshot(tokens, entry_tree)
    self.assertIsNotNone(writer.stored_at(tokens))
    reader = PrefixCache(chunk_size=CHUNK_SIZE)
    self.assertIsNone(reader.stored_at(tokens))

  def test_restore_returns_host_resident_payloads(self):
    # The stored form keeps a host-resident `jax.Array` payload; callers
    # tree-map `SnapshotChunkLeaf.onload` to get back to device memory.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0)
    cache.snapshot(tokens, entry_tree)
    got = cache.stored_at(tokens)
    for k, original in entry_tree.items():
      self.assertIsInstance(got[k].payload, jax.Array)  # pyrefly: ignore[unsupported-operation]
      self.assertIn(
          got[k].payload.sharding.memory_kind,  # pyrefly: ignore[unsupported-operation]
          ('pinned_host', 'unpinned_host'),
      )
      # onload() round-trips back to device memory with the same shape.
      back = rpa.SnapshotChunkLeaf.onload(got[k].payload)  # pyrefly: ignore[unsupported-operation]
      self.assertIsInstance(back, jax.Array)
      self.assertEqual(back.sharding.memory_kind, 'device')
      np.testing.assert_array_equal(
          np.asarray(back), np.asarray(original.payload)
      )

  def test_a_leaf_with_nothing_written_becomes_shape_dtype_struct(self):
    # The RAM rule: a leaf whose mask is all-false keeps its shape but not
    # its bytes. On restore, `SnapshotChunkLeaf.onload` materialises an
    # uninitialised buffer of the right shape/sharding -- safe because a
    # resume only happens where the masks cover what is read.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    entry_tree = _entry_tree(1.0, n_layers=2, written=[True, False])
    cache.snapshot(tokens, entry_tree)
    got = cache.stored_at(tokens)
    self.assertIsInstance(got['layer_0'].payload, jax.Array)  # pyrefly: ignore[unsupported-operation]
    self.assertIsInstance(got['layer_1'].payload, jax.ShapeDtypeStruct)  # pyrefly: ignore[unsupported-operation]
    self.assertEqual(
        got['layer_1'].payload.shape, entry_tree['layer_1'].payload.shape  # pyrefly: ignore[unsupported-operation]
    )
    self.assertEqual(
        got['layer_1'].payload.dtype, entry_tree['layer_1'].payload.dtype  # pyrefly: ignore[unsupported-operation]
    )
    self.assertTrue(cache.is_cached(tokens))
    # onload of the ShapeDtypeStruct materialises a fresh device array.
    materialised = rpa.SnapshotChunkLeaf.onload(got['layer_1'].payload)  # pyrefly: ignore[unsupported-operation]
    self.assertIsInstance(materialised, jax.Array)
    self.assertEqual(materialised.sharding.memory_kind, 'device')
    self.assertEqual(materialised.shape, got['layer_1'].payload.shape)  # pyrefly: ignore[unsupported-operation]
    self.assertEqual(materialised.dtype, got['layer_1'].payload.dtype)  # pyrefly: ignore[unsupported-operation]

  def test_partial_prefix_hit(self):
    cache = self._new_cache()
    entry_a = _entry_tree(1.0)
    entry_b = _entry_tree(2.0)
    cache.snapshot(_tokens(1, 2, 3, 4), entry_a)
    cache.snapshot(_tokens(1, 2, 3, 4, 5, 6, 7, 8), entry_b)
    # New sequence sharing only the first chunk: chunk 0 hits, chunk 1
    # has a different key because token 5..8 differ.
    different_8 = _tokens(1, 2, 3, 4, 9, 9, 9, 9)
    _assert_stored_matches_input(cache.stored_at(different_8[:4]), entry_a)  # pyrefly: ignore[bad-argument-type]
    self.assertIsNone(cache.stored_at(different_8))

  def test_restore_miss_returns_none(self):
    cache = self._new_cache()
    self.assertIsNone(cache.stored_at(_tokens(1, 2, 3, 4)))

  def test_a_second_capture_of_the_same_tokens_changes_nothing(self):
    # Two captures covering the same tokens: the union adds nothing, so the
    # stored bytes stay as they are and the merge is skipped outright.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    self.assertTrue(cache.snapshot(tokens, _entry_tree(1.0)))
    # TRUE again, and it means "the cache holds this prefix", not "bytes
    # moved": the second capture adds no token, and the prefix is there
    # either way, so a caller is right to move its cursor here.
    self.assertTrue(cache.snapshot(tokens, _entry_tree(2.0)))
    # What did NOT happen is an overwrite: the first capture's bytes stand.
    _assert_stored_matches_input(cache.stored_at(tokens), _entry_tree(1.0))  # pyrefly: ignore[bad-argument-type]

  def test_captures_union_per_token(self):
    # The heart of the model: two captures written over different token ranges
    # UNION. Neither is "more complete" than the other -- they describe
    # different tokens -- and the merged entry holds each token's bytes from
    # whichever capture had it.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    first = _entry_tree(1.0, n_layers=1, written=[(0, 2)])
    second = _entry_tree(5.0, n_layers=1, written=[(2, 4)])
    self.assertTrue(cache.snapshot(tokens, first))
    self.assertTrue(cache.snapshot(tokens, second))
    stored = cache.stored_at(tokens)['layer_0']  # pyrefly: ignore[unsupported-operation]
    payload = np.asarray(stored.payload)
    np.testing.assert_array_equal(payload[:2], np.full((2, 8), 1.0))
    np.testing.assert_array_equal(payload[2:], np.full((2, 8), 5.0))

  def test_the_union_does_not_depend_on_the_order(self):
    # Same two captures, opposite order: same bytes, same mask. This is what
    # makes two prompts sharing a prefix safe whichever one runs first.
    def _run(order):
      cache = self._new_cache()
      tokens = _tokens(1, 2, 3, 4)
      for tree in order:
        cache.snapshot(tokens, tree)
      stored = cache.stored_at(tokens)['layer_0']  # pyrefly: ignore[unsupported-operation]
      return np.asarray(stored.payload), stored.written

    first = _entry_tree(1.0, n_layers=1, written=[(0, 2)])
    second = _entry_tree(5.0, n_layers=1, written=[(1, 4)])
    ab_payload, ab_written = _run([first, second])
    ba_payload, ba_written = _run([second, first])
    np.testing.assert_array_equal(ab_written, ba_written)
    np.testing.assert_array_equal(ab_written, np.ones((CHUNK_SIZE,), bool))
    # Token 1 is claimed by both; whoever got there first keeps it, and that
    # is the only difference an order can make -- a token, never a hole.
    np.testing.assert_array_equal(ab_payload[0], np.full((8,), 1.0))
    np.testing.assert_array_equal(ba_payload[0], np.full((8,), 1.0))
    np.testing.assert_array_equal(ab_payload[3], np.full((8,), 5.0))
    np.testing.assert_array_equal(ba_payload[3], np.full((8,), 5.0))

  def test_a_non_contiguous_union_is_kept_as_a_mask(self):
    # Captures at different pass boundaries can leave a GAP between them.
    # The stored mask has to represent that faithfully rather than
    # collapsing to an interval.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    cache.snapshot(tokens, _entry_tree(1.0, n_layers=1, written=[(0, 1)]))
    cache.snapshot(tokens, _entry_tree(2.0, n_layers=1, written=[(3, 4)]))
    stored = cache.stored_at(tokens)['layer_0']  # pyrefly: ignore[unsupported-operation]
    np.testing.assert_array_equal(
        stored.written, np.array([True, False, False, True])
    )

  def test_an_all_false_mask_stores_no_bytes(self):
    # The RAM rule: a capture with nothing written keeps its shape but not its
    # bytes, and a later capture that does hold tokens supplies them.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4)
    cache.snapshot(tokens, _entry_tree(1.0, n_layers=1, written=[False]))
    stored = cache.stored_at(tokens)['layer_0']  # pyrefly: ignore[unsupported-operation]
    self.assertIsInstance(stored.payload, jax.ShapeDtypeStruct)  # no bytes
    cache.snapshot(tokens, _entry_tree(2.0, n_layers=1, written=[True]))
    stored = cache.stored_at(tokens)['layer_0']  # pyrefly: ignore[unsupported-operation]
    self.assertIsInstance(stored.payload, jax.Array)  # ...and now there are

  def test_is_cached(self):
    cache = self._new_cache()
    tokens_a = _tokens(1, 2, 3, 4)
    tokens_b = _tokens(5, 6, 7, 8)
    tokens_c = _tokens(9, 9, 9, 9)
    cache.snapshot(tokens_a, _entry_tree(1.0))  # everything written
    cache.snapshot(
        tokens_b, _entry_tree(1.0, n_layers=2, written=[True, False])
    )
    self.assertTrue(cache.is_cached(tokens_a))
    self.assertTrue(cache.is_cached(tokens_b))
    self.assertFalse(cache.is_cached(tokens_c))

  def test_a_non_aligned_length_is_a_tail_not_an_error(self):
    # The API is unified: `position % chunk_size == 0` is just the aligned
    # case. A shorter length stores/reads the trailing partial stop.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6)
    cache.snapshot(tokens[:4], _entry_tree(1.0))
    # A capture taken at 6 can only hold chunk 1's first two tokens: the slot
    # has not computed the rest, and the cache clamps the claim either way
    # (`test_a_capture_is_clamped_to_the_position_it_was_taken_at`).
    self.assertTrue(cache.snapshot(tokens, _entry_tree(2.0, written=(0, 2))))
    self.assertTrue(cache.is_cached(tokens))
    _assert_stored_matches_input(  # pyrefly: ignore[bad-argument-type]
        cache.stored_at(tokens), _entry_tree(2.0, written=(0, 2))  # pyrefly: ignore[bad-argument-type]
    )
    # ...and it is a DIFFERENT stop from the chunk it lives in.
    _assert_stored_matches_input(  # pyrefly: ignore[bad-argument-type]
        cache.stored_at(tokens[:4]), _entry_tree(1.0)  # pyrefly: ignore[bad-argument-type]
    )

  def test_empty_prefix_is_a_noop(self):
    cache = self._new_cache()
    self.assertFalse(cache.snapshot(_tokens(), _entry_tree(1.0)))
    self.assertFalse(cache.is_cached(_tokens()))
    self.assertIsNone(cache.stored_at(_tokens()))
    self.assertEmpty(_stored_stops(cache))

  def test_a_capture_is_clamped_to_the_position_it_was_taken_at(self):
    # THE ANCHOR INVARIANT'S ENFORCEMENT POINT. A capture taken at `position`
    # can only hold KV for tokens below it -- the slot has not computed the
    # rest -- so an over-claiming mask is TRIMMED to `position - chunk_floor`
    # rather than believed. In production `chunk_written_mask` already bounds
    # it; the clamp is what stops a caller (or a future writer) from
    # anchoring a payload that claims tokens its node does not witness, which
    # is exactly how a chunk's bytes used to leak across a divergence.
    cache = self._new_cache()
    tokens = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    cache.snapshot_at(tokens, 4, _entry_tree(1.0))
    # A whole-chunk claim, made at a position only 3 tokens into the chunk.
    cache.snapshot_at(tokens, 7, _entry_tree(2.0))
    stored = cache.stored_at(tokens[:7])
    assert stored is not None
    np.testing.assert_array_equal(
        stored['layer_0'].written, [True, True, True, False]
    )
    node = cache.node_at(tokens, 7)
    assert node is not None
    self.assertEqual(node.end, 7)  # ...and the payload stops where it does
    _assert_ownership_invariant(self, cache)
    # A later capture that really has reached the chunk end supplies the
    # missing token, and only it: the union keeps what was there.
    cache.snapshot_at(tokens, 8, _entry_tree(3.0))
    stored = cache.stored_at(tokens[:8])
    assert stored is not None
    self.assertTrue(stored['layer_0'].is_full)
    payload = np.asarray(stored['layer_0'].payload)
    np.testing.assert_array_equal(payload[:3], np.full((3, 8), 2.0))
    np.testing.assert_array_equal(payload[3], np.full((8,), 3.0))
    _assert_ownership_invariant(self, cache)


class DescentShapeTest(absltest.TestCase):
  """What a descent lands on, and what asking the cache for coverage costs.

  The edge-comparison counter these tests used to read is gone, so what is
  left is the SHAPE of a descent (a shallower path is the truncation of a
  deeper one) and the one cost still countable from outside: how many
  paveability checks a walk's entry needs.
  """

  def test_a_shallower_path_is_the_truncation_of_a_deeper_one(self):
    # THE IDENTITY THAT REPLACES THE PROMISE. A descent is deterministic and
    # prefix-consistent: it appends children while their end stays within
    # the bound, so asking for a SHALLOWER position cannot land elsewhere --
    # it must stop earlier on the same nodes. That is what lets a caller who
    # has already verified a deep path slice it, and a walk extend from its
    # own tip, instead of re-descending on someone's word that the prefix is
    # there.
    #
    # Same awkward trie as the descent-cost tests: five prompts forking at 3,
    # 7, 11 and 15 over three pass schedules, so splits land inside chunks,
    # at chunk ends, and inside consolidated intervals.
    cache = PrefixCache(chunk_size=CHUNK_SIZE)
    base = _tokens(*range(1, 21))
    prompts = [base]
    for fork in (3, 7, 11, 15):
      prompts.append(np.concatenate([base[:fork], base[fork:] + 100 * fork]))
    schedules = ((4, 8, 12, 16), (3, 5, 9, 14), (5, 6, 7, 16))
    for index, tokens in enumerate(prompts):
      cursor = 0
      for position in schedules[index % len(schedules)]:
        cursor = _run_pass(cache, tokens, position, (None,), cursor)
    self.assertGreater(len(_all_nodes(cache)), 6, 'the trie is too simple')
    checked = 0
    for tokens in prompts:
      limit = len(tokens)
      deep, _ = cache.root.resumable_range(tokens, 0, limit)
      self.assertGreater(len(deep), 1, 'nothing was stored for this prompt')
      for position in range(0, limit + 1):
        shallow, last_resumable = (
            cache.root.resumable_range(tokens, 0, position)
            if position > 0
            else ([], 0)
        )
        self.assertEqual(
            [id(node) for node in shallow],
            [id(node) for node in deep if node.start < last_resumable],
            f'the path to {position} is not the truncation of the path to'
            f' {limit}',
        )
        checked += 1
    self.assertGreater(checked, 100, 'the sweep is too small to mean anything')


def _readback(
    cache: Any, tokens: np.ndarray
) -> list[Any]:
  """Everything a reader of `tokens` can observe, position by position.

  Used to say "the cache holds exactly what it held before" without naming a
  structure: two tries that answer every probe identically are the same
  cache as far as any caller is concerned, however their nodes are arranged.

  Args:
    cache: the cache to read.
    tokens: the reader's whole token row.

  Returns:
    One entry per position: whether it is a stop, whether it is a resume
    point, and the composed chunk's per-token values and mask.
  """
  out: list[Any] = []
  for position in range(1, len(tokens) + 1):
    composed = _composed(cache, tokens, position)
    out.append((
        position,
        cache.is_cached(tokens, position),
        cache.can_resume_at(tokens, position),
        None if composed is None else _values(composed[0]),
    ))
  return out


class SharedChunkHolderTest(absltest.TestCase):
  """One buffer per chunk: what the stops inside it share, and what they don't.

  A chunk's BYTES are one box, shared by every slice that owns part of it, so
  a capture taken at one boundary is a capture for all of them and a split
  costs nothing. What a reader gets back is not that box but a COMPOSITE of
  the parts it has witnessed, which is why these tests compare values rather
  than object identity: two readers of one chunk legitimately see different
  composites.
  """

  def _new_cache(self) -> PrefixCache:
    return PrefixCache(chunk_size=CHUNK_SIZE)

  def test_a_capture_at_one_stop_is_a_capture_for_all_of_them(self):
    # The stops inside a chunk describe the same pages, so what one of them
    # learns the others hold immediately -- that is the whole point of
    # sharing the payload rather than copying it.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    # A pass that ended at 6 could only write the first half of chunk 1.
    cache.snapshot(tokens[:4], _entry_tree(0.0, n_layers=1))
    cache.snapshot(tokens[:6], _entry_tree(1.0, n_layers=1, written=(0, 2)))
    # The pass that closed the chunk fills in the rest...
    cache.snapshot(tokens[:8], _entry_tree(1.0, n_layers=1, written=(2, 4)))
    composed_at_6 = _composed(cache, tokens, 6)
    composed_at_8 = _composed(cache, tokens, 8)
    assert composed_at_6 is not None and composed_at_8 is not None
    _assert_stored_trees_agree(self, composed_at_6[0], composed_at_8[0])
    self.assertTrue(composed_at_6[0]['layer_0'].is_full)
    # One box, not two: the capture at 6 and the one at 8 merged into it.
    self.assertEqual(_n_holders(cache), 2)  # chunk 0 and chunk 1

  def test_a_resume_point_only_ever_reads_its_own_first_tokens(self):
    # The chunk physically covers four tokens, but a resume at `position`
    # injects only the tokens below it -- and how many that is, is DERIVED
    # from the geometry (`min(chunk_end, position) - chunk_start`) rather
    # than recorded per stop. What sits above is a deeper resume point's
    # business.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    cache.snapshot(tokens[:4], _entry_tree(0.0, n_layers=1))
    cache.snapshot(tokens[:8], _entry_tree(1.0, n_layers=1))
    cache.snapshot(tokens[:6], _entry_tree(2.0, n_layers=1))
    plan = cache.restore_chunk_tiles(tokens, 0, 6)
    self.assertEqual(_tile_sizes(plan), [4, 2])  # chunk 0, then 2 tokens
    assert plan
    self.assertEqual(plan[-1].end, 6)
    # ...and the tile it reads 2 tokens out of is the WHOLE chunk, the same
    # bytes the stop at 8 injects 4 tokens of. What stops at the resume
    # point is the tile's RANGE, not its payload.
    composed_at_8 = _composed(cache, tokens, 8)
    assert composed_at_8 is not None
    _assert_stored_trees_agree(self, plan[-1].tree, composed_at_8[0])

  def test_prompts_that_diverge_inside_a_chunk_share_only_what_they_agree_on(
      self,
  ):
    # Sharing is per TOKEN. Two prompts that split inside a chunk hold the
    # part below the split once, between them, and their own KV above it.
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = _tokens(1, 2, 3, 4, 5, 60, 70, 80)
    cache.snapshot(a[:4], _entry_tree(0.0, n_layers=1))
    cache.snapshot(a[:8], _entry_tree(1.0, n_layers=1))
    cache.snapshot(b[:8], _entry_tree(2.0, n_layers=1))
    # Token 5 is shared (A wrote it first); tokens 6..8 are each prompt's own.
    self.assertEqual(_values(cache.stored_at(a[:8]))[0], [1.0, 1.0, 1.0, 1.0])
    self.assertEqual(_values(cache.stored_at(b[:8]))[0], [1.0, 2.0, 2.0, 2.0])
    # Three payloads: chunk 0, the shared-and-then-A part of chunk 1, and B's.
    self.assertEqual(_n_holders(cache), 3)
    _assert_ownership_invariant(self, cache)

  def test_a_later_boundary_inside_a_stored_chunk_joins_it(self):
    # The chunk can be closed before a pass boundary inside it is recorded
    # (another prompt got there first), so recording the boundary has to find
    # the chunk that is already there rather than start a second copy.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    cache.snapshot(tokens[:4], _entry_tree(0.0, n_layers=1))
    cache.snapshot(tokens[:8], _entry_tree(1.0, n_layers=1))
    self.assertEqual(_n_holders(cache), 2)
    # The slot's WHOLE token row, as the drivers pass it.
    cache.snapshot_at(
        tokens, 6, _entry_tree(2.0, n_layers=1)
    )
    self.assertEqual(_n_holders(cache), 2, 'a copy was made')
    # The interval was cut at 6 so the boundary has an owner, but the bytes
    # stayed put: both halves of chunk 1 point at the one payload.
    self.assertEqual(_n_stops(cache), 2)
    below = cache.node_at(tokens, 6)
    above = cache.node_at(tokens, 8)
    assert below is not None and above is not None
    self.assertIsNot(below, above)
    self.assertIs(_holder(cache, below, 1), _holder(cache, above, 1))
    # ...and the bit landed on the chunk that was already there, so the
    # boundary is a resume point for everybody reading it.
    self.assertTrue(cache.can_resume_at(tokens, 6))
    self.assertEqual(_values(cache.stored_at(tokens))[0], [1.0] * 4)

  def test_a_caller_holding_only_a_prefix_joins_the_chunk_too(self):
    # `snapshot` takes just the prefix, so the tokens of the edge below are
    # not there to confirm -- but the descent compares only the OVERLAP, and
    # the tokens agree everywhere both sides have them, which is enough to
    # prove the chunk is this caller's. `snapshot(prefix)` and
    # `snapshot_at(whole_row, position)` therefore agree.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    cache.snapshot(tokens[:4], _entry_tree(0.0, n_layers=1))
    cache.snapshot(tokens[:8], _entry_tree(1.0, n_layers=1))
    cache.snapshot(tokens[:6], _entry_tree(2.0, n_layers=1))  # prefix only
    self.assertEqual(_n_holders(cache), 2)  # chunk 1's bytes are held ONCE
    below = cache.node_at(tokens, 6)
    above = cache.node_at(tokens, 8)
    assert below is not None and above is not None
    self.assertIs(_holder(cache, below, 1), _holder(cache, above, 1))
    self.assertTrue(cache.can_resume_at(tokens, 6))
    self.assertTrue(cache.can_resume_at(tokens, 8))
    # ...and a prefix that DIVERGES inside the chunk gets its own bytes for
    # the tokens it does not share.
    other = np.concatenate([tokens[:4], _tokens(55, 66)])
    cache.snapshot(other, _entry_tree(3.0, n_layers=1))
    self.assertEqual(_values(cache.stored_at(other))[0][:2], [3.0, 3.0])
    self.assertEqual(_values(cache.stored_at(tokens))[0][:2], [1.0, 1.0])
    _assert_ownership_invariant(self, cache)

  def test_a_capture_through_one_co_owner_is_visible_through_the_other(self):
    # WHY THE HOLDER IS A BOX WITH ONE FIELD IN IT. Merging is FUNCTIONAL --
    # `rpa.merge_chunk_trees` returns a new pytree rather than mutating one --
    # so the tree object changes with every capture. A bare tree reference on
    # each node would leave the co-owners of a split holding the version they
    # parted on, and tokens added afterwards would silently stop propagating.
    # Rebinding one field inside the box reaches all of them.
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = _tokens(1, 2, 3, 4, 5, 6, 77, 88)
    # A stores chunk 0, then the TOP half of chunk 1 (tokens 7 and 8).
    cache.snapshot_at(a, 4, _entry_tree(1.0, n_layers=1))
    cache.snapshot_at(
        a, 8, _entry_tree(1.0, n_layers=1, written=(2, 4))
    )
    # B forks inside chunk 1. The split leaves the node over the shared
    # tokens and A's own tail CO-OWNING one buffer; B's tail gets its own.
    cache.snapshot_at(b, 4, _entry_tree(2.0, n_layers=1))
    cache.snapshot_at(
        b, 8, _entry_tree(2.0, n_layers=1, written=(2, 4))
    )
    below = cache.node_at(b, 6)
    above_a = cache.node_at(a, 8)
    above_b = cache.node_at(b, 8)
    assert below is not None and above_a is not None and above_b is not None
    shared = _holder(cache, below, 1)
    self.assertIs(shared, _holder(cache, above_a, 1), 'the split copied')
    self.assertIsNot(shared, _holder(cache, above_b, 1), 'the fork shared')
    # Nobody has written tokens 5 and 6 yet.
    self.assertEqual(_values(cache.stored_at(a))[1], [0, 0, 1, 1])
    self.assertEqual(_values(cache.stored_at(b))[1], [0, 0, 1, 1])

    tree_before = shared.tree
    # A pass ending at 6 captures them. It is stored through the node that
    # owns those tokens -- one of the two co-owners.
    cache.snapshot_at(
        a, 6, _entry_tree(5.0, n_layers=1, written=(0, 2))
    )
    # The buffer is updated across co-owners to hold the NEW tree.
    shared_now = _holder(cache, below, 1)
    self.assertIs(_holder(cache, above_a, 1), shared_now)
    self.assertIsNot(
        shared_now.tree, tree_before, 'the merge was not published'
    )
    # ...so the other co-owner sees the tokens too, and so does every reader
    # that goes through either of them -- including B, which never captured
    # them and shares only the tokens below the fork.
    self.assertEqual(_values(shared_now.tree)[1], [1, 1, 1, 1])
    self.assertEqual(_values(shared_now.tree)[0][:2], [5.0, 5.0])
    self.assertEqual(_values(cache.stored_at(a))[0], [5.0, 5.0, 1.0, 1.0])
    self.assertEqual(_values(cache.stored_at(b))[0], [5.0, 5.0, 2.0, 2.0])
    # One buffer per chunk per path, still: chunk 0, the shared chunk 1, B's.
    self.assertEqual(_n_holders(cache), 3)
    _assert_ownership_invariant(self, cache)

  def test_a_pass_boundary_inside_a_chunk_adds_a_bit_not_a_copy(self):
    # THE POINT OF ONE PAYLOAD PER CHUNK. A pass boundary inside a chunk
    # describes the same pages as the boundary that closes it, so it sets a
    # bit in a `resumable` mask and adds no bytes. Before, each of them held
    # its own full chunk-sized payload -- `extract_chunk` hands over whole
    # pages however few tokens are real -- which is what multiplied the RAM.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    cache.snapshot(tokens[:4], _entry_tree(0.0, n_layers=1))  # chunk 0
    self.assertEqual(_n_holders(cache), 1)
    cache.snapshot(tokens[:6], _entry_tree(1.0, n_layers=1))  # pass boundary
    cache.snapshot(tokens[:7], _entry_tree(2.0, n_layers=1))  # another one
    cache.snapshot(tokens[:8], _entry_tree(3.0, n_layers=1))  # chunk 1
    self.assertEqual(_n_holders(cache), 2)  # two chunks, two payloads
    # Every one of those boundaries is a resume point, and the mask they are
    # recorded in composes back into one.
    np.testing.assert_array_equal(
        _resumable(cache, tokens, 8), [False, True, True, True]
    )
    for position in (4, 6, 7, 8):
      self.assertTrue(cache.can_resume_at(tokens, position), position)
    # ...and whoever wrote a token first keeps it: the boundary at 6 wrote
    # chunk 1's first two tokens, and the later captures added the rest.
    self.assertEqual(_values(cache.stored_at(tokens))[0], [1.0, 1.0, 2.0, 3.0])
    _assert_ownership_invariant(self, cache)

  def test_many_passes_over_many_chunks_make_exactly_one_node(self):
    # The same statement at scale, and with the drivers' calling convention
    # (the whole token row plus a position). Nine pass boundaries over four
    # chunks cost ONE node -- nothing branches off this prompt, so nothing
    # divides it -- holding four chunk slices and nine positions.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 17))
    boundaries = (4, 5, 6, 7, 8, 9, 12, 13, 16)
    for position in boundaries:
      cache.snapshot_at(
          tokens, position, _entry_tree(0.0, n_layers=1)
      )
    self.assertEqual(_n_stops(cache), 1)
    self.assertEqual(_n_holders(cache), 4)  # one per chunk
    owner = _stored_stops(cache)[0]
    self.assertEqual((owner.start, owner.end), (0, 16))
    self.assertEqual(_chunk_indices(cache, owner), [0, 1, 2, 3])
    # The boundaries are kept as they are: absolute positions, sorted, one
    # list for the whole interval rather than a mask per chunk.
    self.assertEqual(list(owner.resumable_positions), list(boundaries))
    for position in boundaries:
      self.assertTrue(cache.can_resume_at(tokens, position), position)
    # A position no pass ended at is not a resume point, even though its
    # chunk is right there.
    for position in (1, 2, 3, 10, 11, 14, 15):
      self.assertFalse(cache.can_resume_at(tokens, position), position)
    _assert_ownership_invariant(self, cache)
    _assert_every_node_earns_its_place(self, cache)


class CrossPathIsolationTest(parameterized.TestCase):
  """What two prompts that diverge INSIDE a chunk share, and what they must not.

  THE WRONG-ANSWER BUG THIS GUARDS. A chunk used to be owned whole by one
  node -- whichever pass reached it first -- while being READ by everyone who
  walked through that node. Two prompts diverging above it therefore shared
  tokens they disagree about, and the second one attended to the first one's
  KV. Measured on that code: `can_resume_at(B, 8) == True` for a prompt B
  that had stored nothing, with A's bytes in the plan.

  Ownership by INTERVAL removes it without giving up the sharing. A node owns
  the tokens its edge spells, so the fork chunk is held as two slices -- the
  part below the divergence on the node both prompts reach, the part above on
  each prompt's own node -- and a read COMPOSES the parts the reader has
  witnessed. Both halves of the property are checked here: each prompt reads
  its OWN tokens above the fork, and they read the SAME bytes below it, from
  one copy.
  """

  # A and B share six tokens and diverge at token 7, INSIDE chunk 1 = (4, 8].
  A = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
  B = _tokens(1, 2, 3, 4, 5, 6, 77, 88)
  A_VALUE = 1.0
  B_VALUE = 2.0

  def _new_cache(self) -> PrefixCache:
    return PrefixCache(chunk_size=CHUNK_SIZE)

  def _write_a(self, cache: PrefixCache) -> None:
    """Prompt A: a pass ends at 5, mid-chunk-1, then one closes the chunk."""
    for position in (4, 5, 8):
      cache.snapshot_at(
          self.A,
          position,
          _entry_tree(self.A_VALUE, n_layers=1),
      )

  def _write_b(self, cache: PrefixCache) -> None:
    """Prompt B: the same chunk 0, then its OWN half of chunk 1."""
    for position in (4, 8):
      cache.snapshot_at(
          self.B,
          position,
          _entry_tree(self.B_VALUE, n_layers=1),
      )

  def _fork_slices(self, cache, tokens) -> list[Any]:
    """The slice objects of the fork chunk that `tokens` can read."""
    nodes, _ = cache.root.resumable_range(tokens, 0, 8)
    slices = []
    for node in nodes:
      for idx, holder in enumerate(node.holders):
        if (node.start // cache.chunk_size) + idx == 1:
          slices.append(holder)
    return slices

  @parameterized.named_parameters(
      dict(testcase_name='_a_then_b', b_first=False),
      dict(testcase_name='_b_then_a', b_first=True),
  )
  def test_the_fork_chunk_is_shared_below_the_fork_and_split_above(
      self, b_first
  ):
    # ORDER INDEPENDENCE IS THE PROPERTY THAT WAS BROKEN: the old union let
    # whichever prompt wrote tokens 7,8 first own them, and the other read
    # them back. Both orders were wrong for one of the two. What has to hold
    # in EITHER order is that each prompt reads its own tokens above the fork
    # and they agree below it.
    cache = self._new_cache()
    for write in (
        (self._write_b, self._write_a) if b_first else
        (self._write_a, self._write_b)
    ):
      write(cache)
    a_values, a_written = _values(cache.stored_at(self.A))
    b_values, b_written = _values(cache.stored_at(self.B))
    self.assertEqual(a_written, [1, 1, 1, 1])  # both read the whole chunk...
    self.assertEqual(b_written, [1, 1, 1, 1])
    # ...ABOVE THE FORK (tokens 7, 8 = offsets 2, 3) each reads its own.
    self.assertEqual(a_values[2:], [self.A_VALUE] * 2)
    self.assertEqual(b_values[2:], [self.B_VALUE] * 2)
    # ...and BELOW IT (tokens 5, 6) they read the very same bytes. Which
    # prompt captured them depends on the order and does not matter: the two
    # prefixes are identical there, so the KV is too.
    self.assertEqual(a_values[:2], b_values[:2])
    first = self.B_VALUE if b_first else self.A_VALUE
    self.assertEqual(a_values[:2], [first] * 2)
    # STORED ONCE: the shared tokens come from slice objects that are on BOTH
    # prompts' paths, not from a copy each.
    shared = {id(s) for s in self._fork_slices(cache, self.A)} & {
        id(s) for s in self._fork_slices(cache, self.B)
    }
    self.assertNotEmpty(shared, 'the shared half of the chunk was copied')
    self.assertEqual(_n_holders(cache), 3)  # chunk 0, the shared part, B's
    # Chunk 0, which they agree on for every token, is shared outright.
    self.assertEqual(
        _values(cache.stored_at(self.A[:4]))[0],
        _values(cache.stored_at(self.B[:4]))[0],
    )
    # Each may resume where its own pass ended, and nowhere the other's did.
    self.assertTrue(cache.can_resume_at(self.A, 8))
    self.assertTrue(cache.can_resume_at(self.B, 8))
    _assert_ownership_invariant(self, cache)
    _assert_every_node_earns_its_place(self, cache)

  def test_a_prompt_that_stored_nothing_reads_up_to_its_fork_and_no_further(
      self,
  ):
    # THE MEASURED REPRO, now with the right answer. B has stored nothing at
    # all; everything in the trie is A's. B may have what it has WITNESSED --
    # chunk 0, and the tokens of chunk 1 below the fork, including A's pass
    # boundary at 5 -- and nothing above the fork, where the two disagree.
    cache = self._new_cache()
    self._write_a(cache)
    self.assertEqual(
        [cache.can_resume_at(self.B, position) for position in (4, 5, 6, 7, 8)],
        [True, True, False, False, False],
    )
    plan = cache.restore_chunk_tiles(self.B, 0, 8)
    assert plan
    self.assertEqual(plan[-1].end, 5)  # A's boundary at 5, below the fork
    self.assertEqual(_tile_sizes(plan), [4, 1])
    self.assertEqual(sum(_tile_sizes(plan)), plan[-1].end)
    # WHAT PROTECTS B IS THE RANGE, not the payload. It is handed A's own box
    # for the fork chunk -- one copy, nothing composed -- and may take from it
    # exactly `[4, 5)`: the resume point below the fork. The tokens A holds
    # above it are in those bytes and are never written, because the inject
    # writes only the tile's range into the page.
    tile = plan[-1]
    self.assertEqual((tile.start, tile.end), (4, 5))
    values, _ = _values(tile.tree)
    self.assertEqual(values[0], self.A_VALUE)  # token 5, which they share
    # A, which wrote the whole chunk, still resumes at 8: the isolation costs
    # the writer nothing.
    plan = cache.restore_chunk_tiles(self.A, 0, 8)
    assert plan
    self.assertEqual(plan[-1].end, 8)
    self.assertEqual(_tile_sizes(plan), [4, 4])

  def test_the_ownership_check_would_catch_a_boundary_outside_an_interval(
      self,
  ):
    # The same for the resume points, which are absolute positions now: a
    # node may only offer a boundary its own interval contains, or a prompt
    # that merely walks through it would be handed a resume it never
    # witnessed.
    cache = self._new_cache()
    self._write_a(cache)
    self._write_b(cache)
    node = cache.node_at(self.B, 6)
    assert node is not None
    # A stop at a position the node's interval does not contain.
    node.resumables = node.resumables + [(node.end + 1, -1)]
    with self.assertRaises(self.failureException):
      _assert_ownership_invariant(self, cache)

  def test_the_contiguity_check_would_catch_a_hole_in_the_holders(self):
    # THE HAZARD A POSITIONAL LIST BRINGS. A node's holders are indexed by
    # position, not keyed by chunk, so a missing entry does not read as
    # "that chunk is absent" -- it renumbers every later one, and the cache
    # would serve chunk 2's KV as chunk 1's without an error anywhere.
    cache = self._new_cache()
    self._write_a(cache)
    owner = cache.node_at(self.A, 8)
    assert owner is not None
    self.assertEqual(_chunk_indices(cache, owner), [0, 1])
    del owner.holders[0]  # the hole
    with self.assertRaises(self.failureException):
      _assert_ownership_invariant(self, cache)

  def test_no_token_has_two_owners_on_a_path(self):
    # Single source of truth, asserted directly rather than through a
    # corruption: with ownership derived from `(start, end]` and each
    # node's interval starting where its parent's ends, two owners for one
    # token is not something the structure can represent. What the test can
    # still check -- and what the derivation rests on -- is that the
    # intervals really do PARTITION each path, token for token.
    cache = self._new_cache()
    self._write_a(cache)
    self._write_b(cache)
    for tokens in (self.A, self.B):
      owners: dict[int, prefix_cache_lib.PrefixNode] = {}
      for position in range(1, len(tokens) + 1):
        node = cache.node_at(tokens, position)
        assert node is not None, position
        self.assertTrue(node.start < position <= node.end)
        owners[position] = node
      # Every token has exactly one owner, and the owners tile the prompt in
      # order: each one picks up where the last left off.
      seen: list[prefix_cache_lib.PrefixNode] = []
      for position in range(1, len(tokens) + 1):
        node = owners[position]
        if not seen or seen[-1] is not node:
          self.assertNotIn(id(node), [id(x) for x in seen], 'an owner recurs')
          if seen:
            self.assertEqual(seen[-1].end, node.start)
          seen.append(node)
      self.assertEqual(seen[0].start, 0)
      self.assertEqual(seen[-1].end, tokens.size)

  def test_a_divergence_above_a_stored_boundary_keeps_both_prompts_whole(self):
    # When the divergence is ABOVE everything a prompt has stored, the second
    # prompt inherits all of it and the first LOSES NOTHING -- the node that
    # owns the shared tokens keeps owning them however deep the trie grows
    # past it. (Ownership by interval is what makes that true: under the
    # anchor model the chunk moved onto the deeper path and the first prompt
    # had to re-capture it.)
    cache = self._new_cache()
    for position in (4, 6):
      cache.snapshot_at(
          self.A,
          position,
          _entry_tree(self.A_VALUE, n_layers=1),
      )
    # B inherits: it witnessed every token A stored.
    self.assertTrue(cache.can_resume_at(self.B, 6))
    self.assertEqual(
        _values(cache.stored_at(self.B[:6]))[1], [1, 1, 0, 0]
    )
    plan = cache.restore_chunk_tiles(self.B, 0, 8)
    assert plan
    self.assertEqual(plan[-1].end, 6)
    self.assertEqual(_tile_sizes(plan), [4, 2])
    # B now stores its own tail, which forks the trie above the shared part.
    cache.snapshot_at(
        self.B, 8, _entry_tree(self.B_VALUE, n_layers=1)
    )
    # A still has everything it had: same resume points, same bytes.
    self.assertTrue(cache.can_resume_at(self.A, 4))
    self.assertTrue(cache.can_resume_at(self.A, 6))
    a_values, a_written = _values(cache.stored_at(self.A))
    self.assertEqual(a_written, [1, 1, 0, 0])  # only what A ever stored
    self.assertEqual(a_values[:2], [self.A_VALUE] * 2)
    # ...and B reads the shared tokens plus its own above them.
    b_values, b_written = _values(cache.stored_at(self.B))
    self.assertEqual(b_written, [1, 1, 1, 1])
    self.assertEqual(
        b_values, [self.A_VALUE, self.A_VALUE] + [self.B_VALUE] * 2
    )
    self.assertFalse(cache.can_resume_at(self.A, 8))  # B's boundary, not A's
    self.assertTrue(cache.can_resume_at(self.B, 8))
    _assert_ownership_invariant(self, cache)
    _assert_every_node_earns_its_place(self, cache)


class CursorPositionTest(absltest.TestCase):
  """The per-slot cursor: one integer, and what makes it safe to trust.

  A cursor used to be an object naming a trie node, and every structural
  change the cache made to itself -- an edge split, a fold, a payload
  dropped -- was a way for it to go stale. It is now a POSITION: how many of
  this slot's prompt tokens the cache holds. It names nothing, so nothing the
  trie does to itself can invalidate it, and there is no cursor state to
  validate, retreat or repair.

  What the integer PROMISES is that this position is a RESUME POINT for this
  slot's tokens: the trie holds the prefix up to it, every chunk below is
  paveable, and a pass ended exactly there. Every way one is set keeps that
  true -- a push sets 0 vacuously, a restore sets a point it just verified,
  a snapshot pass sets its own boundary and only if the whole walk was
  accepted -- and the promise is what lets the descent below it skip the edge
  comparisons (`PrefixCacheLookupCostTest`).

  WHY A POSITION CANNOT GO STALE: nothing removes KV. The cache only ever
  grows, so a prefix that was there when the integer was taken is there
  still. That is a property of the cache as it stands, not a law, and it is
  what an eviction policy would be the first thing to break -- it would have
  to be PREFIX-CLOSED with respect to live positions, never dropping a chunk
  some slot's cursor still spans. The cost of getting that wrong is not a
  miss: a position that outlives its prefix descends WITHOUT COMPARING into
  whatever now occupies the same first-token bucket, which is another
  prompt's KV read as this slot's own. (Measured, while the cache still had
  a `clear()`: prompt A cached, cleared, prompt B admitted with the same
  first token, and A's stale position grafted A's next capture onto B's path
  -- after which a third prompt spelling B's prefix and A's tail was served
  KV that no prefix of it ever produced.)
  """

  def _new_cache(self) -> PrefixCache:
    return PrefixCache(chunk_size=CHUNK_SIZE)

  def _snap(self, cache, tokens, position, cursor=0, value=0.0) -> int:
    """Stores one capture and returns the position the batcher would report."""
    accepted = cache.snapshot_at(
        tokens,
        position,
        _entry_tree(value, n_layers=1),
    )
    return position if accepted else cursor

  def _stored_depths(self, cache, tokens) -> list[int]:
    """Every position of `tokens` that is a stored stop."""
    return [
        depth
        for depth in range(1, len(tokens) + 1)
        if cache.is_cached(tokens, depth)
    ]

  def test_a_cursor_starts_at_the_empty_prefix(self):
    # A pushed slot knows nothing, and zero is the promise that says so.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 9))
    self.assertFalse(
        cache.restore_chunk_tiles(tokens, 0, 8)
    )
    self.assertEqual(self._snap(cache, tokens, 4, cursor=0), 4)

  def test_a_store_that_is_refused_leaves_the_cursor_where_it_was(self):
    # THE CONTRACT. `snapshot_at` returns whether the cache now HOLDS this
    # prefix -- which is exactly the condition for moving the cursor -- not
    # whether bytes moved. A capture that adds nothing still returns True,
    # because the prefix is there either way; only a refusal returns False,
    # and a cursor that advanced on a refusal would be a promise the trie
    # cannot keep.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    cursor = self._snap(cache, tokens, 4)
    self.assertEqual(cursor, 4)
    # Storing the same thing again adds no byte -- and still means "held".
    self.assertTrue(
        cache.snapshot_at(tokens, 4, _entry_tree(1.0, n_layers=1))
    )
    # A capture whose preceding chunk is missing is REFUSED: nothing could
    # ever inject it, so the cache does not pretend to hold the prefix.
    self.assertFalse(
        cache.snapshot_at(tokens, 12, _entry_tree(2.0, n_layers=1))
    )
    self.assertEqual(self._snap(cache, tokens, 12, cursor=cursor), 4)
    self.assertFalse(cache.is_cached(tokens, 12))
    # ...and the slot picks up from where it really is, one chunk at a time.
    cursor = self._snap(cache, tokens, 8, cursor=cursor)
    self.assertEqual(cursor, 8)
    self.assertEqual(self._snap(cache, tokens, 12, cursor=cursor), 12)
    self.assertEqual(self._stored_depths(cache, tokens), [4, 8, 12])

  def test_nothing_the_trie_does_to_itself_can_stale_a_position(self):
    # THE POINT OF THE INTEGER. Another prompt splits an edge this slot's
    # prefix runs through; consolidation folds nodes away under it; the node
    # that owned its tokens is replaced by two. None of it touches the
    # cursor, because the cursor names none of it -- the prefix it promises
    # is still there, token for token.
    cache = self._new_cache()
    mine = _tokens(*range(1, 21))
    cursor = self._snap(cache, mine, 4)
    cursor = self._snap(cache, mine, 8, cursor=cursor)
    self.assertEqual(cursor, 8)
    self.assertEqual(
        sorted((n.start, n.end) for n in _stored_stops(cache)), [(0, 8)]
    )
    before = _readback(cache, mine)
    # Another prompt diverges at token 3, splitting the interval underneath.
    theirs = _tokens(1, 2, 99, 98, 97, 96, 95, 94)
    self._snap(cache, theirs, 4)
    self.assertEqual(
        sorted((n.start, n.end) for n in _stored_stops(cache)),
        [(0, 2), (2, 4), (2, 8)],
    )
    # The cursor is untouched and still true: everything it promised reads
    # back exactly as it did.
    self.assertEqual(_readback(cache, mine), before)
    # ...and the slot goes on caching from it.
    cursor = self._snap(cache, mine, 12, cursor=cursor)
    self.assertEqual(cursor, 12)
    self.assertEqual(self._stored_depths(cache, mine), [4, 8, 12])
    self.assertTrue(cache.can_resume_at(theirs, 4))
    _assert_ownership_invariant(self, cache)
    _assert_every_node_earns_its_place(self, cache)

  def test_two_slots_on_one_prompt_keep_their_own_positions(self):
    # Cursors are per-slot and nothing else: two slots working through the
    # same tokens neither share nor disturb one another's, and the trie ends
    # up holding one copy of the prefix.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 17))
    slow, fast = 0, 0
    fast = self._snap(cache, tokens, 4, cursor=fast)
    fast = self._snap(cache, tokens, 8, cursor=fast)
    slow = self._snap(cache, tokens, 4, cursor=slow)
    self.assertEqual((slow, fast), (4, 8))
    fast = self._snap(cache, tokens, 12, cursor=fast)
    slow = self._snap(cache, tokens, 8, cursor=slow)
    self.assertEqual((slow, fast), (8, 12))
    self.assertEqual(self._stored_depths(cache, tokens), [4, 8, 12])
    # Three intervals: each stop cut the interval so the boundary has an owner.
    # The KV is still held once (three chunks across the nodes).
    self.assertEqual(_n_stops(cache), 3)
    self.assertEqual(_n_holders(cache), 3)
    _assert_ownership_invariant(self, cache)

  def test_the_cursor_always_names_a_resume_point(self):
    # THE PROMISE, IN ITS STRONGEST FORM: `cursor == 0 or
    # can_resume_at(tokens, cursor)`, checkable at any moment. "The prefix is
    # there" would make the descent below it safe but not USEFUL -- a restore
    # may only land on a recorded pass boundary, so a cursor on a position
    # that is merely present is one the slot could never resume from. The
    # walk therefore advances it ONCE, at the boundary, not at every chunk it
    # stores along the way (only the last store of a pass records one).
    windows = (None, 5)
    for boundaries in ((4, 8, 12, 16), (3, 5, 9, 14), (5, 6, 7, 16)):
      cache = self._new_cache()
      tokens = _tokens(*range(1, 17))
      cursor = 0
      for position in boundaries:
        cursor = _run_pass(cache, tokens, position, windows, cursor)
        self.assertEqual(cursor, position)
        _assert_cursor_promises_a_resume(self, cache, tokens, cursor)
        # ...and the mid-walk positions it passed through, which ARE stored,
        # are exactly the ones it did not stop on.
        for chunk_end in range(CHUNK_SIZE, position, CHUNK_SIZE):
          if chunk_end not in boundaries:
            self.assertTrue(cache.is_cached(tokens, chunk_end), chunk_end)
            self.assertFalse(cache.can_resume_at(tokens, chunk_end), chunk_end)

  def test_a_pass_extracts_each_chunk_once(self):
    # The cursor's advance moved to the pass boundary; the WALK did not. Each
    # pass still extracts the chunks from the cursor's own chunk up to the
    # boundary, once each. The obvious way to get the new rule wrong is to
    # re-walk from a cursor that never moved, which doubles the device work
    # every pass -- so the counts are pinned.
    windows = (None, 5)
    cache = self._new_cache()
    tokens = _tokens(*range(1, 17))
    dispatches = [0]
    cursor = 0
    # (boundary, extracts): a pass re-extracts the chunk its cursor sits
    # INSIDE (it has more tokens now) and every chunk above it.
    for position, expected in ((3, 1), (7, 2), (11, 2), (16, 2)):
      before = dispatches[0]
      cursor = _run_pass(
          cache, tokens, position, windows, cursor, dispatches
      )
      self.assertEqual(cursor, position)
      self.assertEqual(
          dispatches[0] - before,
          expected,
          f'the pass ending at {position} extracted the wrong number of'
          ' chunks',
      )
      _assert_cursor_promises_a_resume(self, cache, tokens, cursor)
    self.assertEqual(dispatches[0], 7)  # four passes over four chunks
    # A pass with nothing new to say costs nothing at all.
    before = dispatches[0]
    self.assertEqual(
        _run_pass(cache, tokens, 16, windows, cursor, dispatches), 16
    )
    self.assertEqual(dispatches[0], before)

  def test_a_position_from_another_prompt_can_no_longer_be_offered(self):
    # This used to be the cost of a WRONG claim: a caller could tell the
    # cache it already held four tokens of a prefix it had never seen, and
    # the best the cache could do was refuse the store on the paving guard.
    # There is no longer anywhere to say it -- a store descends and compares
    # -- so the case survives only as the thing it always should have been:
    # tokens nobody has cached are not storable above their first chunk.
    cache = self._new_cache()
    mine = _tokens(*range(1, 13))
    theirs = _tokens(*range(51, 63))
    self._snap(cache, mine, 4)
    # `theirs` shares no token with `mine`, so chunk 1 has nothing to build
    # on and the store is refused.
    self.assertFalse(
        cache.snapshot_at(theirs, 8, _entry_tree(3.0, n_layers=1))
    )
    self.assertFalse(cache.is_cached(theirs, 8))
    self.assertEqual(self._stored_depths(cache, mine), [4])


# ---------------------------------------------------------------------------
# A PREFILL SCHEDULE, played against the cache the way the batcher plays it.
#
# `page_batcher._maybe_snapshot_prefix_cache` is the cache's only writer in
# production, and the properties below are properties of what IT does: every
# chunk from the slot's cursor up to the pass boundary is extracted and
# stored, only the boundary carries the resume bit, and the cursor advances
# once, to that boundary, if the whole walk was accepted. Replaying that call
# sequence here is what makes the equivalence test a statement about the real
# induction rather than about a hand-built trie.


@dataclasses.dataclass(frozen=True)
class _WindowGeometry:
  """The only two attributes `rpa.DecodeState.chunk_written_mask` reads."""

  chunk_size: int
  window_size: int | None


def _written_mask(
    chunk_idx: int, position: int, window: int | None
) -> np.ndarray:
  """Which tokens of `chunk_idx` a pass ending at `position` writes back.

  Calls the PRODUCTION rule on a stub carrying the two attributes it reads,
  so what the tests capture cannot drift from what the device writes: the
  window arithmetic has exactly one statement in the codebase
  (:meth:`rpa.DecodeState.chunk_written_mask`) and this is it.

  Args:
    chunk_idx: which chunk is being extracted.
    position: the pass boundary the slot has reached.
    window: the layer's sliding-window size; `None` for a global layer.

  Returns:
    `bool[CHUNK_SIZE]`.
  """
  geometry = _WindowGeometry(chunk_size=CHUNK_SIZE, window_size=window)
  return np.asarray(
      rpa.DecodeState.chunk_written_mask(
          typing.cast(rpa.DecodeState, geometry), chunk_idx, position
      )
  )


def _kv_value(token: int, layer: int) -> float:
  """A distinct value per (token, layer), so a misplaced token is visible."""
  return float(1000 * (layer + 1) + token)


def _capture(
    chunk_idx: int, position: int, windows: Sequence[int | None]
) -> dict[str, rpa.SnapshotChunkLeaf]:
  """What `extract_chunk` hands the cache for one chunk at a pass boundary.

  Args:
    chunk_idx: which chunk was extracted.
    position: the pass boundary the slot has reached.
    windows: one window size per layer (`None` for a global layer).

  Returns:
    A `DecodeState`-shaped tree of :class:`rpa.SnapshotChunkLeaf`, each leaf
    holding the chunk's per-token KV and the mask the model wrote it under.
  """
  chunk_start = chunk_idx * CHUNK_SIZE
  return {
      f'layer_{layer}': rpa.SnapshotChunkLeaf(
          payload=jnp.asarray(
              np.array(
                  [
                      [_kv_value(chunk_start + j, layer)] * 8
                      for j in range(CHUNK_SIZE)
                  ],
                  dtype=np.float32,
              )
          ),
          written=jnp.asarray(_written_mask(chunk_idx, position, window)),
      )
      for layer, window in enumerate(windows)
  }


def _run_pass(
    cache: prefix_cache_lib.PrefixCache,
    tokens: np.ndarray,
    position: int,
    windows: Sequence[int | None],
    cursor: int = 0,
    dispatches: MutableSequence[int] | None = None,
) -> int:
  """Plays ONE prefill pass ending at `position`, as the batcher does.

  Mirrors `page_batcher._maybe_snapshot_prefix_cache`, including the rule
  that makes a pass a pass:

  * the position advances ONCE, to the pass boundary, and only if every
    store up to it was accepted, so it always lands on a position a pass
    ended at. A refusal breaks the walk: the chunk above one that could not
    be stored cannot be stored either, since its predecessor is not
    paveable.

  Args:
    cache: the cache to write into.
    tokens: the slot's whole token row.
    position: where this pass ended (need not be on the chunk grid).
    windows: one window size per layer.
    cursor: how much of this prefix the cache already holds for this slot.
    dispatches: if given, `dispatches[0]` is incremented once per extract --
      the count the batcher pays in device work.

  Returns:
    The slot's cursor after the pass: `position` if the whole walk was
    accepted, otherwise the cursor it came in with.
  """
  chunk_size = cache.chunk_size
  if cursor >= position:
    return cursor
  tiles = []
  for chunk_idx in range(
      cursor // chunk_size, math.ceil(position / chunk_size)
  ):
    chunk_start = chunk_idx * chunk_size
    if dispatches is not None:
      dispatches[0] += 1
    tiles.append(
        prefix_cache_lib.ChunkTile(
            _offloaded(_capture(chunk_idx, position, windows)),
            max(chunk_start, cursor),
            min(chunk_start + chunk_size, position),
        )
    )
  return position if cache.store_tiles(tokens, tiles) == position else cursor


def _run_passes(
    cache: prefix_cache_lib.PrefixCache,
    tokens: np.ndarray,
    boundaries: Sequence[int],
    windows: Sequence[int | None],
    cursor: int = 0,
) -> int:
  """Plays a whole prefill schedule for one slot.

  Args:
    cache: the cache to write into.
    tokens: the slot's whole token row.
    boundaries: the pass boundaries, in order.
    windows: one window size per layer.
    cursor: the slot's starting cursor.

  Returns:
    The slot's cursor after the last pass.
  """
  for position in boundaries:
    cursor = _run_pass(cache, tokens, position, windows, cursor)
  return cursor


def _window_is_covered(
    cache: prefix_cache_lib.PrefixCache,
    tokens: np.ndarray,
    position: int,
    window: int | None,
    layer: str,
) -> bool:
  """Whether the stored masks cover one layer's window at `position`.

  DERIVED FROM THE DATA, never from the bit: it walks the chunks that
  `[position - window, position)` falls in, COMPOSES each one exactly as a
  restore would (:func:`_composed`), and asks the composite
  (:meth:`StoredChunkLeaf.covers`) whether every token of the range is
  written. That is what a resume needs to be safe -- the resuming query reads
  exactly its window -- and it is computed here without consulting
  `can_resume_at` at all, which is what lets the two be compared.

  Composing is what makes it honest now that ownership is an interval: a
  chunk divided by a branch is held as several slices, and reading one node's
  slice would report a hole where the neighbour holds the tokens. It also
  has to be the SLOT's view -- the whole token row -- because what each
  reader may see is what it has witnessed.

  Args:
    cache: the cache to read.
    tokens: the slot's tokens.
    position: the candidate resume point.
    window: the layer's window; `None` (global) means the whole prefix.
    layer: which leaf of the chunk tree to read.

  Returns:
    Whether every token the layer would read back is really stored.
  """
  chunk_size = cache.chunk_size
  token = 0 if window is None else max(0, position - window)
  while token < position:
    chunk_start = (token // chunk_size) * chunk_size
    hi = min(chunk_start + chunk_size, position)
    composed = _composed(cache, tokens, hi)
    if composed is None:
      return False  # the chunk is not even stored
    leaf = composed[0][layer]
    if not leaf.covers(token - chunk_start, hi - chunk_start):
      return False
    token = hi
  return True


def _masks_cover_every_window(
    cache: prefix_cache_lib.PrefixCache,
    tokens: np.ndarray,
    position: int,
    windows: Sequence[int | None],
) -> bool:
  """Whether EVERY layer's window at `position` is covered by stored masks.

  Args:
    cache: the cache to read.
    tokens: the slot's tokens.
    position: the candidate resume point.
    windows: one window size per layer.

  Returns:
    Whether a resume there would find every token it reads back.
  """
  return all(
      _window_is_covered(cache, tokens, position, window, f'layer_{layer}')
      for layer, window in enumerate(windows)
  )


class ResumableBitEquivalenceTest(parameterized.TestCase):
  """The resume bit must MEAN window coverage -- not merely record a pass.

  `can_resume_at` is ONE STORED BIT of a chunk's `resumable` mask, read as-is
  with no window arithmetic anywhere near it. What makes that sound is an
  INDUCTION over the batcher's snapshot walk: a capture taken at a pass
  boundary is written over `[position - W_leaf, position)` of every chunk the
  pass touched, captures of a chunk UNION, and the walk starts at the chunk
  the slot's frontier sits in -- so at every boundary it records, the
  accumulated masks already cover every layer's window.

  This test is the other half of that argument: it replays real schedules,
  RE-DERIVES the coverage from the stored `written` masks alone
  (:func:`_masks_cover_every_window`), and asserts the derivation and the bit
  agree. The two are computed from disjoint state -- the bit from
  `PrefixNode.resumable`, the coverage from `StoredChunkLeaf.written` --
  so an induction that stops holding (a walk that skips the frontier's own
  chunk, a merge that loses a token, a window wider than a pass writes)
  shows up here as a bit that lies.

  It is also where the per-token refactor is load-bearing: several passes
  ending inside one chunk now set several bits on ONE payload, and every one
  of them has to be independently true.
  """

  N_TOKENS = 16

  def _new_cache(self) -> PrefixCache:
    return PrefixCache(chunk_size=CHUNK_SIZE)

  def _tokens(self) -> np.ndarray:
    return _tokens(*range(1, self.N_TOKENS + 1))

  def _assert_bit_agrees_with_masks(self, cache, tokens, windows, boundaries):
    """The two halves of the equivalence, over every position."""
    # 1. The bit is set at exactly the pass boundaries: nowhere else is a
    #    resume point, however much of the prefix happens to be stored.
    self.assertEqual(
        [
            position
            for position in range(1, len(tokens) + 1)
            if cache.can_resume_at(tokens, position)
        ],
        sorted(set(boundaries)),
    )
    # 2. Wherever it is set, the stored masks really do cover every layer's
    #    window -- which is the only reason resuming there is safe.
    for position in range(1, len(tokens) + 1):
      if not cache.can_resume_at(tokens, position):
        continue
      self.assertTrue(
          _masks_cover_every_window(cache, tokens, position, windows),
          f'the bit at {position} claims a resume the masks cannot support'
          f' (windows={windows}, boundaries={boundaries})',
      )
    # 3. ...and no chunk claims a token its anchor does not witness, which is
    #    what makes any of this safe to hand to a second prompt.
    _assert_ownership_invariant(self, cache)

  @parameterized.named_parameters(
      # A global-only model: every capture writes its whole chunk prefix.
      dict(testcase_name='_global', windows=(None,), boundaries=(4, 8, 12, 16)),
      # A window NARROWER than a chunk: a capture leaves holes below it, and
      # the resume point still has to be covered.
      dict(testcase_name='_window_below_chunk', windows=(2,),
           boundaries=(4, 8, 12, 16)),
      dict(testcase_name='_window_at_chunk', windows=(CHUNK_SIZE,),
           boundaries=(4, 8, 12, 16)),
      # A window WIDER than a chunk: one capture cannot hold it, so coverage
      # has to come from the union with what earlier passes stored.
      dict(testcase_name='_window_above_chunk', windows=(9,),
           boundaries=(4, 8, 12, 16)),
      # Per-leaf windows, the production shape (some global, some windowed).
      dict(testcase_name='_mixed_windows', windows=(None, 3, 6),
           boundaries=(3, 7, 10, 16)),
      # Boundaries that land mid-chunk, which is the normal case: the
      # scheduler's budget has nothing to do with the chunk grid.
      dict(testcase_name='_unaligned_strides', windows=(None, 5),
           boundaries=(5, 9, 11, 14, 16)),
      # Several boundaries INSIDE one chunk -- the case that used to make a
      # node each and now sets bits on one mask.
      dict(testcase_name='_many_boundaries_in_one_chunk', windows=(None, 3),
           boundaries=(4, 5, 6, 7, 8)),
      dict(testcase_name='_one_token_at_a_time', windows=(3,),
           boundaries=tuple(range(1, 17))),
      # One giant pass: chunk 0 is stored with NOTHING written (it is far
      # below the window), which is exactly the case the RAM rule covers.
      dict(testcase_name='_one_giant_pass', windows=(6,), boundaries=(16,)),
  )
  def test_the_bit_agrees_with_the_masks(self, windows, boundaries):
    cache = self._new_cache()
    tokens = self._tokens()
    cursor = _run_passes(cache, tokens, boundaries, windows)
    self._assert_bit_agrees_with_masks(cache, tokens, windows, boundaries)
    _assert_cursor_promises_a_resume(self, cache, tokens, cursor)

  def test_the_bit_agrees_when_two_slots_union_the_same_chunks(self):
    # The cross-slot case: two slots running the SAME tokens on different
    # pass boundaries write different tokens of the same chunks, and the
    # captures union. Every boundary either slot recorded must be a resume
    # point, and each one must be covered by the UNION -- which is the whole
    # reason two prompts sharing a prefix may share its KV.
    cache = self._new_cache()
    tokens = self._tokens()
    windows = (None, 3)
    cursors = [0, 0]
    # Interleaved, so neither slot ever sees a trie only it has written.
    for position, slot_id in ((3, 0), (5, 1), (6, 0), (9, 1), (14, 0)):
      cursors[slot_id] = _run_pass(
          cache, tokens, position, windows, cursors[slot_id]
      )
      _assert_cursor_promises_a_resume(
          self, cache, tokens, cursors[slot_id]
      )
    self._assert_bit_agrees_with_masks(
        cache, tokens, windows, (3, 5, 6, 9, 14)
    )
    # ...and they really did share: one trie, no forks (the two slots run the
    # same tokens), so every node on it is on both slots' paths.
    self.assertLen(_all_nodes(cache), _n_stops(cache) + 1)  # + the root
    for node in _stored_stops(cache):
      self.assertLessEqual(len(node.children), 1, 'a fork on one token row')
    _assert_ownership_invariant(self, cache)

  def test_a_second_prompt_inherits_the_first_ones_resume_points(self):
    # A resume point is a property of the CHUNK, so a slot that never ran a
    # pass there can still resume there -- and the coverage it will read is
    # the coverage the first slot's masks provide.
    cache = self._new_cache()
    tokens = self._tokens()
    windows = (None, 5)
    _run_passes(cache, tokens, (6, 11), windows)
    # The reader knows nothing yet: a fresh slot's cursor is 0.
    plan = cache.restore_chunk_tiles(tokens, 0, len(tokens))
    assert plan
    self.assertEqual(plan[-1].end, 11)
    self.assertTrue(
        _masks_cover_every_window(cache, tokens, plan[-1].end, windows)
    )

  def test_the_bit_stays_honest_across_a_restore(self):
    # The induction has a second step the writer alone never reaches: a slot
    # that RESTORED a prefix never captured it, so the coverage its own later
    # boundaries rely on is the coverage the FIRST slot left behind. Its walk
    # picks up in the chunk it landed in, and the bits it goes on to record
    # have to be as true as the ones it inherited.
    cache = self._new_cache()
    tokens = self._tokens()
    windows = (None, 3)
    _run_passes(cache, tokens, (5, 9), windows)
    # A second slot is admitted on the same tokens and restores.
    plan = cache.restore_chunk_tiles(tokens, 0, len(tokens))
    assert plan
    self.assertEqual(plan[-1].end, 9)
    cursor = plan[-1].end  # the restore is where its cursor comes from
    _assert_cursor_promises_a_resume(self, cache, tokens, cursor)
    cursor = _run_passes(cache, tokens, (13, 16), windows, cursor)
    self.assertEqual(cursor, 16)
    _assert_cursor_promises_a_resume(self, cache, tokens, cursor)
    self._assert_bit_agrees_with_masks(
        cache, tokens, windows, (5, 9, 13, 16)
    )

  def test_the_derivation_would_catch_a_lost_write(self):
    # A guard on the guard. The agreement above is only worth something if
    # the derivation can DISAGREE, so: take one token away from a resume
    # point's window, exactly as a merge that dropped it would, and check
    # that the masks then contradict the bit.
    cache = self._new_cache()
    tokens = self._tokens()
    windows = (None, 3)
    _run_passes(cache, tokens, (4, 8, 12, 16), windows)
    self.assertTrue(_masks_cover_every_window(cache, tokens, 16, windows))
    node = cache.node_at(tokens, 16)
    assert node is not None
    tree = _holder(cache, node, 3).tree  # the chunk position 16 falls in
    leaf = tree['layer_1']  # the window-3 layer
    self.assertTrue(leaf.written[-1])
    lost = np.array(leaf.written)
    lost[-1] = False  # the write that went missing
    tree['layer_1'] = dataclasses.replace(leaf, written=lost)
    self.assertTrue(cache.can_resume_at(tokens, 16))  # the bit still says yes
    self.assertFalse(  # ...and the masks now say no
        _masks_cover_every_window(cache, tokens, 16, windows)
    )

  def test_a_window_wider_than_the_prefix_is_covered_from_zero(self):
    # `window >= position` means the resuming query reads the WHOLE prefix,
    # so coverage is the strongest it ever gets. It is worth its own case
    # because it is the one where a hole anywhere below the boundary --
    # including in a chunk no recent pass touched -- would make the bit lie.
    cache = self._new_cache()
    tokens = self._tokens()
    windows = (None, 64)
    _run_passes(cache, tokens, (5, 10, 15), windows)
    self._assert_bit_agrees_with_masks(cache, tokens, windows, (5, 10, 15))
    for position in (5, 10, 15):
      for layer in ('layer_0', 'layer_1'):
        self.assertTrue(
            _window_is_covered(cache, tokens, position, None, layer),
            f'{layer} has a hole below {position}',
        )


class RestoreStartPositionTest(absltest.TestCase):
  """`start_position` does two jobs, and both rest on the same guarantee.

  It is where the restore starts -- the slot's device KV length, so a plan
  may not re-inject below it -- AND the licence for the free part of the
  descent: below it, edges are not compared. The two are the same number
  because a snapshot pass caches exactly what the prefill pass computed, so
  the cache is never behind a prefilling slot; the batcher is what
  guarantees that, and `Batcher._cache_behind` is what it does when the
  guarantee is broken by a failed store.

  THAT GUARANTEE IS THE THING TO PROTECT. A caller that passes a position
  the cache does not hold the prefix for gets ANOTHER PROMPT'S KV, not a
  miss -- `test_a_start_the_cache_does_not_hold_reads_someone_elses_kv`
  measures exactly that, so the precondition is written down as an
  executable consequence rather than a warning.

  (This class used to be `RestoreFromALaggingCursorTest`. There is no cursor
  to lag any more: the number the caller passes is the slot's own position,
  and the `ValueError` that used to catch a disagreement between two numbers
  went with the second number.)
  """

  def _new_cache(self) -> PrefixCache:
    return PrefixCache(chunk_size=CHUNK_SIZE)

  def _warm(self):
    """A cache holding five chunks of a 24-token prompt.

    Returns:
      `(cache, tokens)`.
    """
    cache = self._new_cache()
    tokens = _tokens(*range(1, 25))
    for position in (4, 8, 12, 16, 20):
      cache.snapshot_at(
          tokens, position, _entry_tree(float(position), n_layers=1)
      )
    return cache, tokens

  def test_a_slot_at_the_deepest_resume_point_is_offered_nothing(self):
    cache, tokens = self._warm()
    # The slot is already at 20 and nothing deeper is stored: a miss.
    self.assertFalse(cache.restore_chunk_tiles(tokens, 20, 23))

  def test_a_restore_starts_where_the_slot_is(self):
    # A plan covers exactly the gap between the slot and the resume point --
    # never re-injecting what the slot already holds, which would shift its
    # KV by that many tokens.
    cache, tokens = self._warm()
    for start, tiles in ((0, [4, 4, 4, 4, 4]), (8, [4, 4, 4]), (16, [4])):
      plan = cache.restore_chunk_tiles(tokens, start, 23)
      assert plan, start
      self.assertEqual(plan[-1].end, 20)
      self.assertEqual(_tile_sizes(plan), tiles)
      self.assertEqual(sum(_tile_sizes(plan)), plan[-1].end - start)

  def test_a_slot_parked_mid_chunk_is_caught_up(self):
    # The case that used to be refused, and after unaligned pass boundaries
    # it is the common one: a slot sitting at 18 is handed the chunk it is
    # IN and lands at 20.
    cache, tokens = self._warm()
    # A slot sits where a pass left it, so 18 is a boundary like any other:
    # `_warm`'s chunk-aligned ones are the exception, not the rule.
    cache.snapshot_at(
        tokens, 18, _entry_tree(18.0, n_layers=1)
    )
    plan = cache.restore_chunk_tiles(tokens, 18, 23)
    assert plan
    self.assertEqual(plan[-1].end, 20)
    # The tile is `[18, 20)` -- what the slot GAINS, not the page that gets
    # rewritten. The page still goes in whole (its column is `start //
    # chunk_size`), but the tokens below 18 are the slot's own and are left
    # alone, so the tiles pave exactly `[start, position)`, which is what the
    # plan's coverage check compares against.
    self.assertEqual(_tile_sizes(plan), [2])
    self.assertEqual(sum(_tile_sizes(plan)), plan[-1].end - 18)

  def test_the_rewritten_chunk_holds_the_slots_own_tokens(self):
    # THE SUBSTANCE OF THE OVERWRITE. The tile rewrites tokens the slot
    # already computed, which is only safe because they are the SAME tokens
    # -- same prefix, so the same KV up to reduction order. Made visible
    # here: the pass that ended at 18 captured tokens 17 and 18, the next
    # one captured 19 and 20, and the tile the mid-chunk slot is handed
    # holds the first pair for the tokens it already had and the second for
    # the tokens it is being given.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 25))
    for position in (4, 8, 12, 16):
      cache.snapshot_at(
          tokens, position, _entry_tree(float(position), n_layers=1)
      )
    cache.snapshot_at(
        tokens, 18, _entry_tree(18.0, n_layers=1, written=(0, 2))
    )
    cache.snapshot_at(
        tokens, 20, _entry_tree(20.0, n_layers=1, written=(2, 4))
    )
    plan = cache.restore_chunk_tiles(tokens, 18, 23)
    assert plan
    self.assertEqual(plan[-1].end, 20)
    values, written = _values(plan[0].tree)
    self.assertEqual(written, [1, 1, 1, 1])
    self.assertEqual(values, [18.0, 18.0, 20.0, 20.0])
    # ...and it is the cache's own composed chunk, not a copy that could
    # drift from it.
    composed = _composed(cache, tokens, 20)
    assert composed is not None
    _assert_stored_trees_agree(self, plan[0].tree, composed[0])

  def test_a_plan_is_a_query_and_changes_nothing(self):
    # Asking must leave the cache exactly as it was: the caller installs the
    # result by moving its slot, and nothing about the trie moves with it.
    cache, tokens = self._warm()
    before = _readback(cache, tokens)
    plan = cache.restore_chunk_tiles(tokens, 8, 23)
    assert plan
    self.assertEqual(_readback(cache, tokens), before)
    self.assertFalse(hasattr(plan, 'node'))

  def test_a_prompt_the_cache_does_not_hold_is_offered_nothing(
      self,
  ):
    # A LAGGING CACHE IS NO LONGER REPRESENTABLE. `start_position` used to be
    # a promise the caller INVENTED -- the slot's device position, which runs
    # ahead of the cache whenever a store fails -- and it licensed the
    # descent below it to skip comparing, so such a caller was handed another
    # prompt's KV.
    #
    # `start_position` is still a promise; what changed is who may make it.
    # The batcher passes back a position THIS CACHE reported (a walk's own,
    # or a restore it just served), which cannot go stale upward while there
    # is no eviction -- see
    # `page_batcher_test.StoreFailureRecoveryTest.
    #  test_a_failed_store_leaves_the_restore_a_shorter_start_not_a_wrong_one`
    # for the failed-store case that used to produce a false one. What is
    # tested here is the other half: a prompt the cache holds NOTHING of gets
    # nothing, at the only start such a caller can honestly give -- zero.
    #
    # The shape is the sharpest one available: B's tail spells the SAME
    # tokens as A's after a different prefix, so a descent that stopped
    # comparing anywhere above token 1 would hand A B's bytes and every
    # token it did compare would still match.
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = _tokens(1, 91, 92, 93, 5, 6, 7, 8)
    c = _tokens(1, 91, 92, 93, 55, 56, 57, 58)  # forks B at token 5
    for tokens, value in ((b, 9.0), (c, 7.0)):
      for position in (4, 8):
        cache.snapshot_at(
            tokens, position, _entry_tree(value, n_layers=1)
        )
    # A has nothing cached, so A is offered nothing.
    self.assertFalse(
        cache.restore_chunk_tiles(a, 0, 8),
        'A was offered something it never stored',
    )
    # B, which really does own that prefix, is offered its own bytes.
    plan = cache.restore_chunk_tiles(b, 0, 8)
    assert plan
    self.assertEqual(plan[-1].end, 8)
    for tile in plan:
      self.assertEqual(_values(tile.tree)[0], [9.0] * CHUNK_SIZE)


class PrefillScheduleTest(absltest.TestCase):
  """A whole prefill schedule, host-only: write, share, resume, extend.

  The stand-in for an end-to-end run that needs no model and no device: the
  cache is driven through the exact call sequence `page_batcher` makes, and
  what comes back out is checked TOKEN BY TOKEN against what went in. It is
  the level at which the pieces have to agree with each other -- the tiles a
  plan hands back must pave `[start, position)` exactly, and every token of
  every tile must hold the bytes the writer captured for that token.
  """

  WINDOWS = (None, 5)  # one global layer, one sliding-window layer

  def _new_cache(self) -> PrefixCache:
    return PrefixCache(chunk_size=CHUNK_SIZE)

  def _assert_plan_paves(
      self, plan: Sequence[prefix_cache_lib.ChunkTile], start: int
  ) -> None:
    """The tiles tile `[start, plan[-1].end)` exactly, in order."""
    self.assertEqual(sum(_tile_sizes(plan)), plan[-1].end - start)
    for i, size in enumerate(_tile_sizes(plan)[:-1]):
      self.assertEqual(size, CHUNK_SIZE, f'tile {i} is not a whole chunk')
    self.assertLessEqual(_tile_sizes(plan)[-1], CHUNK_SIZE)
    self.assertEqual(start % CHUNK_SIZE, 0)

  def _assert_tiles_hold_what_was_captured(
      self, plan: Sequence[prefix_cache_lib.ChunkTile]
  ) -> None:
    """Every WRITTEN token of every tile holds the writer's bytes.

    Unwritten tokens are uninitialised by design (the RAM rule) and the model
    evicts them before they can feed into attention, so what is checked is
    the written ones -- against the absolute token they belong to, which is
    what catches a tile injected at the wrong offset.


    Args:
      plan: the plan to check.
    """
    for tile in plan:
      chunk_start = (tile.start // CHUNK_SIZE) * CHUNK_SIZE
      lo = tile.start - chunk_start
      for layer in range(len(self.WINDOWS)):
        leaf = tile.tree[f'layer_{layer}']
        payload = np.asarray(leaf.payload)
        for offset in range(tile.num_tokens):
          pos = lo + offset
          token = tile.start + offset
          if not bool(leaf.written[pos]):
            continue
          np.testing.assert_allclose(
              payload[pos],
              _kv_value(token, layer),
              err_msg=(
                  f'token {token} of layer_{layer} came back'
                  ' with another token\'s KV'
              ),
          )

  def test_a_second_prompt_resumes_from_the_first_ones_breakpoint(self):
    cache = self._new_cache()
    first = _tokens(*range(1, 15))
    # The second prompt shares 11 tokens and then diverges.
    second = np.concatenate([first[:11], _tokens(71, 72, 73)])
    cursors = [0, 0]  # one integer per slot, as the batcher keeps them
    # SLOT 0 prefills the first prompt in three passes, none of them on the
    # chunk grid -- the scheduler's budget has nothing to do with chunks.
    cursors[0] = _run_passes(cache, first, (3, 7, 11), self.WINDOWS)
    self.assertEqual(cursors[0], 11)
    _assert_cursor_promises_a_resume(self, cache, first, cursors[0])
    # ONE node so far: three chunks and three boundaries, but nothing has
    # forked off this prompt, so nothing divides it.
    self.assertEqual(_n_stops(cache), 1)
    self.assertEqual(_n_holders(cache), 3)  # one per chunk

    # SLOT 1 admits the second prompt: a pushed slot's cursor is 0.
    cursors[1] = 0
    plan = cache.restore_chunk_tiles(
        second, 0, len(second) - 1
    )
    assert plan, 'the shared prefix should be a hit'
    # It resumes at the deepest breakpoint the first prompt left, which is
    # mid-chunk -- the whole point of storing partial stops.
    self.assertEqual(plan[-1].end, 11)
    self.assertEqual(_tile_sizes(plan), [4, 4, 3])
    self._assert_plan_paves(plan, start=0)
    self._assert_tiles_hold_what_was_captured(plan)
    # What it is about to inject really is enough for the windowed layer.
    self.assertTrue(
        _masks_cover_every_window(cache, second, plan[-1].end, self.WINDOWS)
    )
    # Installing the plan is one assignment: the slot's cursor becomes the
    # position the restore verified.
    cursors[1] = plan[-1].end
    self.assertEqual(cursors[1], 11)

    # SLOT 1 then prefills the rest of ITS prompt, which diverges. The
    # restored prefix is not re-captured (the walk starts at the cursor),
    # and the divergence gets its own node.
    cursors[1] = _run_pass(cache, second, 13, self.WINDOWS, cursors[1])
    self.assertEqual(cursors[1], 13)
    for tokens, cursor in ((first, cursors[0]), (second, cursors[1])):
      _assert_cursor_promises_a_resume(self, cache, tokens, cursor)
    # Still ONE node: the second prompt's tail was the only thing hanging
    # off the shared interval, so the two folded into one that spells the
    # SECOND prompt's tokens. The first prompt reads it as far as the two
    # agree -- 11 tokens -- which is exactly what it stored.
    self.assertEqual(_n_stops(cache), 1)
    self.assertEqual(
        sorted((n.start, n.end) for n in _stored_stops(cache)), [(0, 13)]
    )
    self.assertEqual(_n_holders(cache), 4)  # four chunks between them
    self.assertTrue(cache.can_resume_at(second, 13))
    self.assertFalse(cache.can_resume_at(first, 13))  # a different prompt
    _assert_every_node_earns_its_place(self, cache)
    _assert_ownership_invariant(self, cache)

    # ...and a third slot on the second prompt's tokens now resumes at 13.
    plan = cache.restore_chunk_tiles(second, 0, len(second))
    assert plan
    self.assertEqual(plan[-1].end, 13)
    self.assertEqual(_tile_sizes(plan), [4, 4, 4, 1])
    self._assert_plan_paves(plan, start=0)
    self._assert_tiles_hold_what_was_captured(plan)

  def test_a_resumed_slot_continues_the_schedule_from_where_it_landed(self):
    # A hit lands a slot mid-chunk, and its next pass has to pick up from
    # there: re-extract the chunk it is sitting in (to add the tokens the
    # first prompt's window left out) and go on. What must NOT happen is a
    # second node for that chunk, or a plan that re-injects it.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 17))
    cursors = [0, 0]
    cursors[0] = _run_passes(cache, tokens, (6,), self.WINDOWS)
    self.assertEqual(_n_stops(cache), 1)  # one interval over both chunks

    plan = cache.restore_chunk_tiles(tokens, 0, 15)
    assert plan
    self.assertEqual(plan[-1].end, 6)
    self._assert_plan_paves(plan, start=0)
    cursors[1] = plan[-1].end
    # The resumed slot runs on to 12: the walk restarts in the chunk it
    # landed in, so chunk 1 gains the tokens the first pass did not hold.
    cursors[1] = _run_pass(cache, tokens, 12, self.WINDOWS, cursors[1])
    self.assertEqual(cursors[1], 12)
    for cursor in cursors:
      _assert_cursor_promises_a_resume(self, cache, tokens, cursor)
    # The resumed slot's own capture extends the interval it landed in --
    # same tokens, nothing forking off them, so the two nodes fold into one.
    self.assertEqual(_n_stops(cache), 1)
    self.assertEqual(
        sorted((n.start, n.end) for n in _stored_stops(cache)), [(0, 12)]
    )
    self.assertEqual(list(_stored_stops(cache)[0].resumable_positions),
                     [6, 12])
    # Chunk 1 was captured twice, at 6 and at 12, and the two UNION: the
    # global layer ends up whole...
    stored = cache.stored_at(tokens[:8])
    assert stored is not None
    self.assertTrue(stored['layer_0'].is_full)
    # ...while the windowed layer keeps a hole at token 6, which no pass's
    # window ever covered (the pass at 6 wrote [1, 6), the pass at 12 wrote
    # [7, 12)). That is not a defect: nothing may resume where a hole would
    # be read, and the resume points there are covered.
    self.assertFalse(stored['layer_1'].is_full)
    for position in (6, 12):
      self.assertTrue(cache.can_resume_at(tokens, position))
      self.assertTrue(
          _masks_cover_every_window(cache, tokens, position, self.WINDOWS)
      )
    # A slot already at 12 has nothing left to gain from the cache.
    self.assertFalse(
        cache.restore_chunk_tiles(tokens, 12, 15)
    )
    _assert_ownership_invariant(self, cache)

  def test_a_tile_spanning_multiple_nodes_stores_to_every_node(self):
    # When multiple earlier prompts diverge within a chunk, the trie contains
    # multiple sub-chunk nodes. A subsequent store of a whole chunk tile must
    # store and clamp into EVERY node it spans, not just the last one.
    cache = self._new_cache()
    p1 = _tokens(1, 2, 91, 92, 93, 94, 95, 96)  # forks at 2
    p2 = _tokens(1, 2, 3, 92, 93, 94, 95, 96)   # forks at 3
    p3 = _tokens(1, 2, 3, 4, 5, 6, 7, 8)        # full run
    _run_passes(cache, p1, (4,), self.WINDOWS)
    _run_passes(cache, p2, (4,), self.WINDOWS)
    # p3 covers chunk 0 (0..4) and chunk 1 (4..8).
    # Storing tile [0, 4) spans across all nodes in chunk 0.
    _run_passes(cache, p3, (4, 8), self.WINDOWS)
    plan = cache.restore_chunk_tiles(p3, 0, 8)
    assert plan
    self.assertEqual(plan[-1].end, 8)
    self.assertEqual(sum(t.num_tokens for t in plan), 8)
    for i in range(len(plan) - 1):
      self.assertEqual(plan[i].end, plan[i + 1].start)
    self._assert_tiles_hold_what_was_captured(plan)
    _assert_ownership_invariant(self, cache)


class EvictionTest(absltest.TestCase):
  """Eviction is TRUNCATION AT A RESUME POINT, coldest tip first.

  The cache may only ever drop a SUFFIX, and only down to a position a pass
  ended at. That is what keeps the rest of it honest: every prefix it still
  holds is whole, so paveability stays monotone and a stored resume bit still
  stands for the coverage it was recorded with. What it costs is a tail; what
  it must never cost is a hole.
  """

  def _new_cache(self, budget: int | None = None) -> PrefixCache:
    return PrefixCache(
        chunk_size=CHUNK_SIZE, max_bytes=budget
    )

  def _fill(self, cache, tokens, boundaries) -> None:
    """One prompt, one pass per boundary -- the schedule the batcher runs."""
    cursor = 0
    for position in boundaries:
      tiles = []
      for chunk_idx in range(cursor // CHUNK_SIZE, math.ceil(position / 4)):
        chunk_start = chunk_idx * CHUNK_SIZE
        tiles.append(
            prefix_cache_lib.ChunkTile(
                _offloaded(_entry_tree(1.0, n_layers=1)),
                max(chunk_start, cursor),
                min(chunk_start + CHUNK_SIZE, position),
            )
        )
      cache.store_tiles(tokens, tiles)
      cursor = position

  def test_the_tail_above_the_deepest_resume_point_is_free(self):
    # Eviction pops the COLDEST resume point. When that is a tip's deepest,
    # the tokens above the next one down are what nothing can reach any more,
    # and they are what comes back.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    self._fill(cache, tokens, (4, 8, 12))
    before = cache.nbytes
    self.assertEqual(cache.restore_chunk_tiles(tokens, 0, 12)[-1].end, 12)
    # Reading at 4 and 8 warms them, leaving 12 the coldest thing in the cache.
    cache.restore_chunk_tiles(tokens, 0, 4)
    cache.restore_chunk_tiles(tokens, 0, 8)
    cache.max_bytes = before - 1
    self.assertGreater(cache.evict(), 0)
    self.assertLess(cache.nbytes, before)
    # ...and what it can serve now stops at 8.
    plan = cache.restore_chunk_tiles(tokens, 0, 12)
    assert plan
    self.assertEqual(plan[-1].end, 8)
    self.assertEqual(sum(tile.num_tokens for tile in plan), 8)
    _assert_ownership_invariant(self, cache)

  def test_a_cut_lands_on_a_resume_point_and_leaves_a_whole_prefix(self):
    # A prompt whose deepest stop keeps going cold is given back one resume
    # point at a time, and what survives is still restorable: 12 -> 8 -> 4 ->
    # nothing. Each cut lands ON a stop, so the chunk below it stays whole.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    self._fill(cache, tokens, (4, 8, 12))
    self.assertEqual(cache.restore_chunk_tiles(tokens, 0, 12)[-1].end, 12)
    served = []
    for depth in (12, 8, 4):
      # Read at every stop below the deepest, which is what leaves the deepest
      # coldest -- the only case where popping one gives tokens back.
      for warm in range(4, depth, 4):
        cache.restore_chunk_tiles(tokens, 0, warm)
      cache.max_bytes = cache.nbytes - 1
      cache.evict()
      plan = cache.restore_chunk_tiles(tokens, 0, 12)
      served.append(plan[-1].end if plan else 0)
      _assert_ownership_invariant(self, cache)
    self.assertEqual(served, [8, 4, 0])
    self.assertEqual(cache.nbytes, 0)

  def test_a_prompt_goes_whole_once_its_shallow_stops_are_spent(self):
    # The other order, and the price of ranking resume points rather than
    # prompts: the shallow stops are the cold ones, and popping them frees
    # nothing, because the tokens they name are still under the deepest. They
    # leave the order all the same, so when the deepest is finally reached
    # there is no stop under it to cut back to and the prompt goes whole.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    self._fill(cache, tokens, (4, 8, 12))
    # The last read is at 12, so 4 and 8 are colder than the deepest stop.
    self.assertEqual(cache.restore_chunk_tiles(tokens, 0, 12)[-1].end, 12)
    cache.max_bytes = cache.nbytes - 1
    self.assertGreater(cache.evict(), 0)
    self.assertEqual(cache.nbytes, 0)
    self.assertFalse(cache.restore_chunk_tiles(tokens, 0, 12))
    _assert_ownership_invariant(self, cache)

  def test_the_coldest_prompt_goes_first(self):
    # Two prompts sharing nothing; one of them is read again. The other is
    # what a byte-starved cache gives up.
    cache = self._new_cache()
    hot = _tokens(*range(1, 9))
    cold = _tokens(*range(101, 109))
    self._fill(cache, hot, (4, 8))
    self._fill(cache, cold, (4, 8))
    # Reading `hot` warms it.
    self.assertEqual(cache.restore_chunk_tiles(hot, 0, 8)[-1].end, 8)
    cache.max_bytes = cache.nbytes // 2
    cache.evict()
    self.assertEqual(cache.restore_chunk_tiles(hot, 0, 8)[-1].end, 8)
    self.assertFalse(cache.restore_chunk_tiles(cold, 0, 8))
    _assert_ownership_invariant(self, cache)

  def test_a_shared_prefix_survives_the_branch_that_is_dropped(self):
    # Eviction may only take TIPS, so the prefix two prompts share is not
    # something either of them can lose on its own.
    cache = self._new_cache()
    a = _tokens(1, 2, 3, 4, 5, 6, 7, 8)
    b = _tokens(1, 2, 3, 4, 55, 66, 77, 88)
    self._fill(cache, a, (4, 8))
    self._fill(cache, b, (4, 8))
    self.assertEqual(cache.restore_chunk_tiles(b, 0, 8)[-1].end, 8)  # warms `b`
    cache.max_bytes = cache.nbytes - 1
    cache.evict()  # one branch's worth is enough
    # A lost its own half; the chunk both of them proved is still there.
    self.assertEqual(cache.restore_chunk_tiles(a, 0, 8)[-1].end, 4)
    self.assertEqual(cache.restore_chunk_tiles(b, 0, 8)[-1].end, 8)
    _assert_ownership_invariant(self, cache)

  def test_what_survives_can_still_be_extended(self):
    # The point of cutting at a resume point: a slot can pick the prefix up
    # again and store on top of it, which needs the chunk below to be whole.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    self._fill(cache, tokens, (4, 8, 12))
    # Warm the stops below the deepest, so the cut is the one that frees bytes.
    cache.restore_chunk_tiles(tokens, 0, 4)
    cache.restore_chunk_tiles(tokens, 0, 8)
    cache.max_bytes = cache.nbytes - 1
    cache.evict()
    plan = cache.restore_chunk_tiles(tokens, 0, 12)
    assert plan
    position = plan[-1].end
    self.assertGreater(position, 0)
    self._fill(cache, tokens, (position, 12))
    self.assertEqual(cache.restore_chunk_tiles(tokens, 0, 12)[-1].end, 12)
    _assert_ownership_invariant(self, cache)

  def test_no_budget_means_nothing_is_ever_dropped(self):
    cache = self._new_cache()  # max_bytes=None
    tokens = _tokens(*range(1, 9))
    self._fill(cache, tokens, (4, 8))
    held = cache.nbytes
    self.assertEqual(cache.evict(), 0)
    self.assertEqual(cache.nbytes, held)

  def test_an_edge_cut_back_does_not_pin_the_whole_prompt(self):
    # `split_at` slices, which is right while both halves stay in the trie.
    # After a cut the upper half is dropped, and the survivor must not be left
    # holding the original -- a 4-token edge keeping a 114k-token buffer alive.
    cache = self._new_cache()
    tokens = _tokens(*range(1, 13))
    self._fill(cache, tokens, (4, 8, 12))  # one node, a 12-token edge
    cache.restore_chunk_tiles(tokens, 0, 4)
    cache.restore_chunk_tiles(tokens, 0, 8)
    cache.max_bytes = cache.nbytes - 1
    self.assertGreater(cache.evict(), 0)  # cuts the edge back to 8

    stack = [cache.root]
    while stack:
      node = stack.pop()
      owner = node.edge
      while owner.base is not None:
        owner = owner.base
      self.assertEqual(
          owner.nbytes,
          node.edge.nbytes,
          f'a {node.edge.nbytes}-byte edge is pinning {owner.nbytes} bytes',
      )
      stack.extend(node.children.values())
    _assert_ownership_invariant(self, cache)

  def test_nothing_kept_is_a_view_of_the_callers_token_buffer(self):
    # The batcher hands in ONE ROW of `i32[batch, global_max_seq_len]`, and a
    # numpy slice of it is a view: whatever the cache keeps -- a node's edge,
    # an entry in the order -- would hold the whole matrix resident for as long
    # as it lives, 64 MiB per prefill pass, none of it in `nbytes`.
    batch = np.zeros((4, 32), dtype=np.int32)  # owns its data: `base is None`
    batch[1, :12] = np.arange(1, 13)
    batch[2, :8] = [1, 2, 3, 4, 91, 92, 93, 94]  # branches off `tokens` at 4
    batch[3, :4] = [51, 52, 53, 54]  # its own root child
    cache = self._new_cache()
    tokens = batch[1]
    self._fill(cache, tokens, (4, 8, 12))
    self._fill(cache, batch[2], (4, 8))  # a branch: a fresh node under a split
    self._fill(cache, batch[3], (4,))  # a fresh node under the root
    cache.restore_chunk_tiles(tokens, 0, 12)  # and a warm, which re-marks

    def _owner(array: np.ndarray) -> np.ndarray:
      while array.base is not None:
        array = array.base
      return array

    # The row IS a view of the matrix -- that is the thing not to keep.
    self.assertIs(_owner(tokens), batch)

    kept = list(cache.lru.values())
    stack = [cache.root]
    while stack:
      node = stack.pop()
      kept.append(node.edge)
      stack.extend(node.children.values())
    self.assertEmpty(
        [a for a in kept if _owner(a) is batch],
        'the cache is holding a view into the buffer it was handed',
    )


if __name__ == '__main__':
  absltest.main()

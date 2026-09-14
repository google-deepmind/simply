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
"""KV prefix cache for the paged-attention batcher.

Stores captured KV at *chunk* granularity -- one chunk is `num_shards *
page_size` tokens, the unit `SamplingState.extract_chunk` and
`inject_chunk` deal in -- and indexes it at token granularity in a
path-compressed (radix) trie over token sequences (:class:`PrefixNode`), so
two prompts share exactly the tokens they agree on and nothing is hashed.
Storage is host RAM, in-process and host-local: each host holds only its own
addressable shard, with no cross-host IO or barrier on the path.

A slot may only be captured or resumed at a *pass boundary*, the one position
where a windowed layer's KV is trustworthy, so the cache records the
positions where a prefill pass ended (:attr:`PrefixNode.resumables`) and
eviction may only truncate at one of them. Captures of the same chunk taken
at different boundaries union per token, so nothing is ever overwritten.

See `serving/ragged_paged_attention.md` for the design.
"""

import bisect
import collections
from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any

import jax
import numpy as np
from simply.utils import ragged_paged_attention as rpa

# The cache's stored form per chunk: a pytree shaped like `decode_state` with
# each leaf replaced by an `rpa.StoredChunkLeaf` — host-resident bytes (or a
# `ShapeDtypeStruct` when nothing is written) plus its per-token written mask.
StoredTree = Any


@dataclasses.dataclass
class ChunkHolder:
  """One chunk's KV, boxed so co-owners of a split share one buffer.

  Merging is functional (`rpa.merge_chunk_trees` returns a new pytree), so
  rebinding `tree` on this box is what lets every node that co-owns the chunk
  see the union. Splitting an edge therefore costs no bytes: both sides go on
  sharing this object and own different parts of it.

  Fields
    tree: the host-resident chunk tree (see `StoredTree`).
    nbytes: host RAM behind `tree`, kept in step with it, per BUFFER, so
      co-owners of a split count it once.
  """

  tree: StoredTree
  nbytes: int = dataclasses.field(default=0, init=False)

  def __post_init__(self) -> None:
    self.nbytes = self._measure()

  def _measure(self) -> int:
    """Host RAM behind `tree`; a placeholder leaf holds none."""
    leaves = jax.tree_util.tree_leaves(
        self.tree, is_leaf=lambda x: isinstance(x, rpa.StoredChunkLeaf)
    )
    return sum(leaf.nbytes for leaf in leaves)

  def merge(self, stored_tree: StoredTree) -> None:
    """Unions a fresh capture into what this chunk already holds, IN PLACE."""
    self.tree = jax.tree_util.tree_map(
        lambda x, y: x.merge(y),
        self.tree,
        stored_tree,
        is_leaf=lambda x: isinstance(x, rpa.StoredChunkLeaf),
    )
    self.nbytes = self._measure()


@dataclasses.dataclass
class PrefixNode:
  """One node of the path-compressed (radix) trie over token sequences."""

  edge: np.ndarray
  end: int
  parent: 'PrefixNode | None'
  children: dict[int, 'PrefixNode'] = dataclasses.field(default_factory=dict)
  holders: list[ChunkHolder] = dataclasses.field(default_factory=list)
  # (position, key) per resume point, sorted; a handful per node. Absolute
  # positions, so a split partitions the list and a fold concatenates two,
  # neither with any arithmetic to do. PRESENCE IS THE BIT: eviction removes
  # the entry, so there is no disabled-but-present state to test for.
  resumables: list[tuple[int, Any]] = dataclasses.field(default_factory=list)

  @property
  def length(self) -> int:
    return len(self.edge)

  @property
  def start(self) -> int:
    return self.end - self.length

  def common_prefix_len(self, tokens: np.ndarray) -> int:
    """Returns how many leading tokens `self.edge` and `tokens` share."""
    n = min(len(self.edge), len(tokens))
    if n == 0:
      return 0
    same = np.equal(self.edge[:n], tokens[:n])
    if bool(same.all()):
      return n
    return int(np.argmin(same))

  def resumable_range(
      self, slot_tokens: np.ndarray, start: int, end: int
  ) -> tuple[Sequence['PrefixNode'], int]:
    """Returns nodes covering [start, last_resumable).

    last_resumable <= end.

    Args:
      slot_tokens: the slot's tokens.
      start: the beginning of the interval to cover, absolute.
      end: the end of the interval to cover, absolute.

    Returns:
      The nodes covering [start, last_resumable), and last_resumable.
    """
    assert start < end <= len(slot_tokens)
    nodes: list['PrefixNode'] = []
    node = self.navigate_to_known(slot_tokens, start)
    if node.end == start:
      node = node.children.get(slot_tokens[node.end])

    last_resumable = start
    last_resumable_node_idx = -1

    while node is not None:
      ours = slot_tokens[node.start : node.end]
      matched_len = node.common_prefix_len(ours)
      if not matched_len:
        break
      limit = min(end, node.start + matched_len)
      stop_pos = node.deepest_resumable(limit)
      if stop_pos is not None:
        if stop_pos > start:
          last_resumable = stop_pos
          last_resumable_node_idx = len(nodes)

      nodes.append(node)
      if matched_len < node.length or matched_len < len(ours):
        break
      if node.end >= end:
        break
      node = node.children.get(slot_tokens[node.end])

    nodes = nodes[: last_resumable_node_idx + 1]
    return nodes, last_resumable

  def navigate_to_known(
      self, slot_tokens: np.ndarray, known: int
  ) -> 'PrefixNode':
    """Descends to the deepest node ending at or below `known`."""
    node = self
    while node.end < known and slot_tokens[node.end] in node.children:
      node = node.children[slot_tokens[node.end]]
    if node.parent is not None and node.end >= known:
      assert node.deepest_resumable(known) == known
    return node

  def ensure_resumable_range(
      self, slot_tokens: np.ndarray, start: int, end: int, chunk_size: int
  ) -> Sequence['PrefixNode']:
    """Ensures nodes exist covering [start, end), and returns them.

    Args:
      slot_tokens: the slot's tokens.
      start: the beginning of the interval to cover, absolute.
      end: the end of the interval to cover, absolute.
      chunk_size: the cache's chunk size -- what a split cuts a holder run on.

    Returns:
      The nodes covering [start, end).
    """
    assert start < end <= len(slot_tokens)
    nodes: list['PrefixNode'] = []
    foot = self.navigate_to_known(slot_tokens, start)
    if foot.end == start:
      if foot.parent is not None and not foot.children:
        foot.edge = np.concatenate((foot.edge, slot_tokens[foot.end : end]))
        foot.end = end
        return [foot]
      node = foot.children.get(slot_tokens[foot.end])
      if node is None:
        fresh = PrefixNode(
            # COPY: a slice of the caller's buffer is a view, and the caller's
            # buffer is a whole batch's tokens. Holding one would keep all of
            # it alive for as long as this node lives.
            edge=slot_tokens[foot.end : end].copy(),
            end=end,
            parent=foot,
        )
        foot.children[fresh.edge[0]] = fresh
        return [fresh]
    else:
      node = foot

    while node is not None:
      shared_len = node.common_prefix_len(
          slot_tokens[node.start : min(end, node.end)]
      )
      if shared_len < len(node.edge):
        node = node.split_at(shared_len, chunk_size)

      nodes.append(node)
      if node.end >= end:
        break

      child = node.children.get(slot_tokens[node.end])
      if child is None:
        if not node.children:
          node.edge = np.concatenate((node.edge, slot_tokens[node.end : end]))
          node.end = end
        else:
          fresh = PrefixNode(
              edge=slot_tokens[node.end : end].copy(),
              end=end,
              parent=node,
          )
          node.children[fresh.edge[0]] = fresh
          nodes.append(fresh)
        break
      node = child

    return nodes

  def split_at(self, offset: int, chunk_size: int) -> 'PrefixNode':
    """Materialises a node `offset` tokens into this one's edge.

    Args:
      offset: how many of this node's leading edge tokens stay above the cut;
        must be in `(0, len(edge))`.
      chunk_size: the cache's chunk size, which is the grid the holder run is
        indexed on.

    Returns:
      The new internal node, at `end = self.end - len(self.edge) +
      offset`.
    """
    parent = self.parent
    if parent is None or not 0 < offset < len(self.edge):
      raise ValueError(
          f'cannot split a {len(self.edge)}-token edge at {offset}'
      )
    split = PrefixNode(
        edge=self.edge[:offset], end=parent.end + offset, parent=parent
    )
    self.edge = self.edge[offset:]
    self.parent = split
    split.children[self.edge[0]] = self
    parent.children[split.edge[0]] = split
    cut = split.end
    split.resumables = [e for e in self.resumables if e[0] <= cut]
    self.resumables = [e for e in self.resumables if e[0] > cut]
    # The run divides where the tokens do. The entry holding the chunk the
    # cut falls inside goes to BOTH sides -- the same buffer, so the split
    # copies no bytes -- and each side then owns a different part of it.
    # `split` begins where this node used to, so its first chunk IS the run's
    # old base -- `self.start` has already moved by this point.
    base = split.start // chunk_size
    spanned = math.ceil(split.end / chunk_size) - base
    split.holders = self.holders[:spanned]
    self.holders = self.holders[self.start // chunk_size - base :]
    return split

  @property
  def has_payload(self) -> bool:
    """Whether this node owns any KV (as opposed to being pure structure)."""
    return bool(self.holders)

  def deepest_resumable(self, limit: int | None = None) -> int | None:
    """The deepest live resume point at or below `limit`, or `None`.

    Args:
      limit: the bound, inclusive; `None` for this node's own end.

    Returns:
      An absolute token position, or `None` when the node holds no live stop
      that low.
    """
    bound = self.end if limit is None else limit
    idx = bisect.bisect_right(self.resumables, bound, key=lambda x: x[0])
    return self.resumables[idx - 1][0] if idx > 0 else None

  @property
  def resumable_positions(self) -> list[int]:
    """Returns absolute token positions where a pass ended."""
    return [position for position, _ in self.resumables]


@dataclasses.dataclass(frozen=True)
class ChunkTile:
  """The tokens `[start, end)` of one chunk, and the payload holding them."""

  tree: StoredTree
  start: int
  end: int

  @property
  def num_tokens(self) -> int:
    return self.end - self.start

  def clamp(self, start: int, end: int) -> StoredTree:
    """Bounds this capture's written mask to the interval `[start, end)`."""
    is_leaf = lambda x: isinstance(x, rpa.StoredChunkLeaf)

    def _clamp(leaf: rpa.StoredChunkLeaf) -> rpa.StoredChunkLeaf:
      written = np.asarray(leaf.written)
      chunk_size = len(written)
      chunk_start = (self.start // chunk_size) * chunk_size
      lo = max(0, start - chunk_start)
      hi = min(chunk_size, end - chunk_start)
      clamped = np.zeros(chunk_size, dtype=bool)
      if lo < hi:
        clamped[lo:hi] = True
        clamped = np.logical_and(written, clamped)
      if np.array_equal(clamped, written):
        return leaf
      return dataclasses.replace(leaf, written=clamped)

    return jax.tree_util.tree_map(_clamp, self.tree, is_leaf=is_leaf)


# Fraction of the process's memory budget the prefix cache is allowed to hold.
_MEMORY_FRACTION = 0.2

# Budget assumed when the memory ceiling cannot be determined. Deliberately
# small: over-estimating the budget is what causes out-of-memory kills.
_FALLBACK_MAX_BYTES = 16 * 1024**3

# cgroups report "no limit" either as the string "max" (unparseable) or as a
# sentinel near 2**63; anything at or above this is treated as unbounded.
_UNBOUNDED_LIMIT_BYTES = 2**62

# Ordered most- to least-specific: the limit enforced on the container, then
# cgroup v2, then cgroup v1.
_MEMORY_LIMIT_PATHS = (
    '/proc/container/memory/limit',
    '/sys/fs/cgroup/memory.max',
    '/sys/fs/cgroup/memory/memory.limit_in_bytes',
)


def _read_int_file(path: str) -> int | None:
  """Returns the positive integer stored in `path`, or None if unreadable."""
  try:
    with open(path) as f:
      value = int(f.read().strip())
  except (OSError, ValueError):
    return None
  return value if value > 0 else None


def _read_mem_total_bytes() -> int | None:
  """Returns total machine memory from /proc/meminfo, or None if unreadable."""
  try:
    with open('/proc/meminfo') as f:
      for line in f:
        if line.startswith('MemTotal:'):
          return int(line.split()[1]) * 1024
  except (OSError, ValueError, IndexError):
    return None
  return None


def memory_budget_bytes() -> int | None:
  """Returns the memory ceiling this process runs under, or None if unknown."""
  for path in _MEMORY_LIMIT_PATHS:
    value = _read_int_file(path)
    if value is not None and value < _UNBOUNDED_LIMIT_BYTES:
      return value
  return _read_mem_total_bytes()


def default_max_bytes() -> int:
  """Returns a conservative prefix-cache budget for the current process.

  The budget is a small fraction of the memory ceiling the process actually
  runs under, so it shrinks automatically on small containers. It is never
  scaled up to a floor: returning more than the container has is exactly the
  failure this guards against.
  """
  budget = memory_budget_bytes()
  if budget is None:
    return _FALLBACK_MAX_BYTES
  return int(budget * _MEMORY_FRACTION)


@dataclasses.dataclass(kw_only=True)
class PrefixCache:
  """Prefix cache backed by host RAM."""

  chunk_size: int
  # Eviction threshold. Defaults to a conservative fraction of the memory
  # ceiling this process runs under; pass `None` to disable eviction entirely.
  max_bytes: int | None = dataclasses.field(default_factory=default_max_bytes)
  _nbytes: int = dataclasses.field(default=0, init=False)

  def __post_init__(self) -> None:
    if self.chunk_size <= 0:
      raise ValueError(f'chunk_size must be positive; got {self.chunk_size}')

  @functools.cached_property
  def root(self) -> PrefixNode:
    """The empty prefix. Carries no payload and is never evicted."""
    return PrefixNode(edge=np.zeros((0,), dtype=np.int32), end=0, parent=None)

  @functools.cached_property
  def lru(
      self,
  ) -> collections.OrderedDict[Any, np.ndarray]:
    """The RESUME POINTS, coldest first -- what eviction chooses between."""
    return collections.OrderedDict()

  @property
  def nbytes(self) -> int:
    """Host RAM the cache holds."""
    return self._nbytes

  def mark_resumable(
      self, slot_tokens: np.ndarray, node: PrefixNode, position: int
  ) -> None:
    """Records a resume point at `position`, or warms one already there.

    Args:
      slot_tokens: the tokens this stop belongs to; what the order keeps, and
        what eviction re-derives the node from.
      node: the node whose interval contains `position`.
      position: the token position a pass ended at.
    """
    idx = bisect.bisect_left(node.resumables, position, key=lambda x: x[0])
    if idx < len(node.resumables) and node.resumables[idx][0] == position:
      key = node.resumables[idx][1]
      self.lru.move_to_end(key)
      return
    prefix = slot_tokens[:position].copy()
    key = id(prefix)
    node.resumables.insert(idx, (position, key))
    self.lru[key] = prefix

  def store_tiles(
      self, slot_tokens: np.ndarray, tiles: Sequence[ChunkTile]
  ) -> int:
    """Stores a run of captures, and returns how far the cache then holds.

    Args:
      slot_tokens: the slot's tokens.
      tiles: the captures, shallowest first, each covering one chunk and
        abutting the next (`tiles[i].end == tiles[i + 1].start`). The first
        must start at a position the cache already holds.

    Returns:
      The deepest position the cache holds for these tokens once the run is
      done -- the last tile's `end` -- or `0` for an empty run.

    Raises:
      ValueError: if a capture cannot be stored where the tiles say it goes.
    """
    if not tiles:
      return 0
    chunk_size = self.chunk_size
    start = tiles[0].start
    nodes = self.root.ensure_resumable_range(
        slot_tokens, tiles[0].start, tiles[-1].end, chunk_size
    )
    reached = start
    node_idx = 0
    for tile in tiles:
      chunk_idx = tile.start // chunk_size
      while node_idx < len(nodes) and nodes[node_idx].start < tile.end:
        node = nodes[node_idx]
        if node.end <= tile.start:
          node_idx += 1
          continue
        idx = chunk_idx - (node.start // chunk_size)
        if not 0 <= idx <= len(node.holders):
          raise ValueError(
              f'prefix_cache: refusing a capture at {tile.end} that would leave'
              f' a hole in the run of the node ({node.start}, {node.end}]'
          )
        stored = tile.clamp(node.start, node.end)
        if idx == len(node.holders):
          holder = ChunkHolder(tree=stored)
          node.holders.append(holder)
          self._nbytes += holder.nbytes
        else:
          # A capture never overwrites what is stored: the two UNION per token,
          # in the buffer itself, so a chunk two nodes straddle is updated for
          # both of them at once.
          before_nbytes = node.holders[idx].nbytes
          node.holders[idx].merge(stored)
          self._nbytes += node.holders[idx].nbytes - before_nbytes
        if node.end <= tile.end:
          node_idx += 1
        else:
          break
      reached = tile.end
    # The bit says only "a pass ended at this token". It is safe to read back
    # with no window arithmetic because the writer captures every chunk
    # boundary the slot passes and merging only ever adds tokens, so the
    # coverage it stands for cannot be lost; `prefix_cache_test`'s
    # equivalence test re-derives that coverage from the written masks and
    # asserts the two agree. The other half -- that nothing below it is ever
    # dropped -- is why eviction may only TRUNCATE AT A RESUME POINT
    # (:meth:`evict`).
    if reached <= start:
      raise ValueError(
          f'prefix_cache: empty or invalid capture interval [{start},'
          f' {reached})'
      )
    self.mark_resumable(slot_tokens, nodes[-1], reached)
    return reached

  def restore_chunk_tiles(
      self, slot_tokens: np.ndarray, start: int, end: int
  ) -> Sequence[ChunkTile]:
    """Returns what to inject to resume as deep as possible, or `[]`."""
    if end <= start:
      return []
    sub, last_resumable = self.root.resumable_range(slot_tokens, start, end)
    if sub:
      # Warm the stop this restore is about to use -- the one thing here the
      # order should rank on. `resumable_range` cuts `sub` at the node that
      # holds it, so it is the last one.
      self.mark_resumable(slot_tokens, sub[-1], last_resumable)
    # The range of interest is [start, position) -- everything up
    # to the deepest usable stop.
    tiles: list[ChunkTile] = []
    chunk_size = self.chunk_size
    for node in sub:
      base_idx = node.start // chunk_size
      # A node's holders run from the chunk its interval starts in, which for
      # a consolidated node can be a whole prompt below where this restore
      # begins. The first chunk that can contribute anything is the one
      # `start` falls in; everything under it is skipped rather than walked
      # and discarded.
      first = max(0, start // chunk_size - base_idx)
      for offset, holder in enumerate(node.holders[first:], start=first):
        chunk_idx = base_idx + offset
        chunk_start = chunk_idx * chunk_size
        tile_start = max(chunk_start, start, node.start)
        tile_end = min(chunk_start + chunk_size, last_resumable, node.end)
        if tile_start < tile_end:
          if (
              tiles
              and tiles[-1].tree is holder.tree
              and tiles[-1].end == tile_start
          ):
            tiles[-1] = ChunkTile(holder.tree, tiles[-1].start, tile_end)
          else:
            tiles.append(ChunkTile(holder.tree, tile_start, tile_end))
    covered = sum(t.num_tokens for t in tiles)
    if covered != last_resumable - start:
      raise ValueError(
          f'prefix_cache: incoherent restore plan for [{start},'
          f' {last_resumable}) -- {covered} tokens covered'
      )
    return tiles

  # ------------------------------------------------------------ eviction

  def evict(self) -> int:
    """Pops resume points, coldest first, until the cache fits `max_bytes`.

    One rule, applied to whatever the order hands over: the coldest resume
    point stops being one. What that gives back depends only on where it sat.
    Taking a stop with others above it frees nothing yet -- those tokens are
    still reachable -- but it is gone from the order, so the loop always makes
    progress and terminates when the order empties.

    Returns:
      Host bytes freed.
    """
    if self.max_bytes is None:
      return 0
    freed = 0
    while self.nbytes > self.max_bytes and self.lru:
      key, prefix = self.lru.popitem(last=False)  # THE EVICTION: coldest first
      position = len(prefix)
      node = self.root.navigate_to_known(prefix, position)
      idx = bisect.bisect_left(node.resumables, position, key=lambda x: x[0])
      if idx >= len(node.resumables) or node.resumables[idx] != (position, key):
        continue
      node.resumables.pop(idx)
      if node.children:
        continue  # what lies above belongs to child prompts: not ours to drop

      # Node has no children: track up through parent, and trim until another
      # resumable position becomes leaf.
      curr: PrefixNode = node
      while curr.parent is not None and not curr.children:
        deepest = curr.deepest_resumable()
        if deepest is not None:
          freed += self._truncate(curr, deepest)
          break
        parent = curr.parent
        freed += self._truncate(curr, curr.start)
        curr = parent

    return freed

  def _truncate(self, node: PrefixNode, cut: int) -> int:
    """Cuts `node`'s interval back to `cut`, freeing what was above it."""
    if cut >= node.end:
      return 0
    if cut > node.start:
      survivor = node.split_at(cut - node.start, self.chunk_size)
      survivor.children.clear()
      # `split_at` SLICES the edge -- cheap, and right when both halves stay in
      # the trie. Here the upper half is about to be dropped, so the survivor
      # would be the last reference to the whole prompt's token array: a
      # 4-token edge pinning 456 KiB. Copy it and let the original go.
      survivor.edge = survivor.edge.copy()
      dropped, keeper = node.holders, survivor
    else:
      parent = node.parent
      if parent is None:
        return 0
      parent.children.pop(node.edge[0], None)
      dropped, keeper, survivor = node.holders, parent, None
    freed = 0
    resharable = False
    for i, holder in enumerate(dropped):
      if i == 0 and keeper.holders and keeper.holders[-1] is holder:
        resharable = True
        continue
      freed += holder.nbytes
    node.holders = []
    if survivor is None and keeper.parent is not None and keeper.children:
      if resharable:
        child = next(iter(keeper.children.values()))
        stale = keeper.holders[-1]
        merged = child.holders[0]
        freed += merged.nbytes + stale.nbytes
        merged.merge(stale.tree)
        freed -= merged.nbytes
        # `stale` is only FREED once nothing points at it any more, and the
        # keeper is not the only one that can: a chunk straddling a node
        # boundary is ONE buffer, handed down by every split whose boundary
        # falls inside it. So walk straight up while the ancestors still end
        # in this chunk -- and take the branches hanging off that run with
        # them -- repointing each at the union. Then the count is true.
        anc: PrefixNode = keeper
        while anc.parent is not None and anc.holders[-1] is stale:
          anc.holders[-1] = merged
          anc = anc.parent
      if len(keeper.children) == 1:
        (child,) = keeper.children.values()
        parent = keeper.parent
        parent.children.pop(keeper.edge[0], None)
        child.edge = np.concatenate((keeper.edge, child.edge))
        child.parent = parent
        parent.children[child.edge[0]] = child
        child.resumables = keeper.resumables + child.resumables
        keeper_holders = keeper.holders
        if keeper.end % self.chunk_size != 0:
          assert keeper_holders[-1] is child.holders[0]
          keeper_holders = keeper_holders[:-1]
        child.holders = keeper_holders + child.holders
        keeper.holders = []
        keeper.resumables = []
    self._nbytes -= freed
    return freed

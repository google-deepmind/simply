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

The cache is content-addressable at *chunk* granularity, where one
chunk = `num_shards * page_size` tokens — the natural inject /
extract granularity used by `DecodeState.inject_chunk` and
`DecodeState.extract_chunk`. The key for the `i`-th chunk of a sequence
is the (incremental) hash of `tokens[0 : (i + 1) * chunk_size]`. Two
prompts sharing their first `M * chunk_size` tokens share their first
`M` keys, and a lookup on the second prompt returns the cached payloads
for those chunks.

Each cache entry stores a **pytree of** :class:`SnapshotChunkLeaf` (one
per `DecodeState` leaf), i.e. the same shape as `decode_state` with
each leaf carrying the chunk's KV bytes + an `available` bool. The
cache is **leaf-agnostic**: it just stores whatever the caller passes
and returns it on lookup. Per-leaf gating happens at offload time via
:meth:`SnapshotChunkLeaf.offload`, which drops payload bytes for leaves
with `available=False` to save host RAM (those bytes are stale
post-eviction garbage and would be replaced by an uninitialised
buffer at inject time anyway).

What the cache *exposes* per chunk for the lookup-time tail-window
rule is whether every leaf has a present payload: see
:meth:`PrefixCache.is_complete`. A chunk with even one `available=
False` leaf is considered incomplete, and the lookup rule then
requires every chunk in the candidate tail-window to be complete.
Older chunks may safely be incomplete: their windowed-layer garbage
will be evicted by `release_for_window` before it can feed into
attention (the tail of real chunks fills the resident window first).
See `PrefixCache.snapshot` / `PrefixCache.restore` /
`PrefixCache.longest_restorable_prefix`.

Per-chunk promotion on snapshot: when a chunk key already has an
entry, the cache uses these rules (no per-leaf merging):
  * complete  + any new       -> no-op (first writer wins)
  * incomplete + new incomplete -> no-op
  * incomplete + new complete   -> OVERWRITE whole tree
A chunk's stored entry can thus be completed in one shot when any
subsequent snapshot from any slot observes every layer as available.

Storage is **host (CPU) RAM**, in-process. `snapshot()` calls
:meth:`SnapshotChunkLeaf.offload` on every leaf (HBM->host
`jax.device_put` to the same sharding with `memory_kind='pinned_host'`)
and stashes the offloaded tree in a per-process dict keyed by
`chunk_key`. `restore()` returns the host-resident tree as-is; the
caller is expected to tree-map :meth:`SnapshotChunkLeaf.onload` before
feeding the payload back to `inject_chunk`.

This is entirely **host-local**: each host stores (and restores) only
its own addressable shard. There is NO file IO, NO Orbax, NO
collective op and NO `sync_global_devices` barrier anywhere in the
snapshot/restore path — the whole point of host offloading vs. the
old synchronous multi-host on-disk checkpointer (which globally
barrier-synced on every `save()` and was the source of both the
latency and the multi-host barrier-desync crashes).

Because the cache is per-process and never reads shared state, the
save/skip dedup decision still derives from a replicated in-memory key
set (the `_store` keys): every host runs `snapshot`/`restore` in
lockstep over identical state, so the set evolves identically on every
host. Note that the moves themselves (`device_put`) are host-local and
do NOT need lockstep for correctness; the lockstep only matters for
the surrounding `extract_chunk` / `inject_chunk` device programs.

The cache does not persist across processes (it is RAM-only), so a
fresh process starts cold.
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import hashlib
from typing import Any

from absl import logging
import jax
import numpy as np
from simply.utils import ragged_paged_attention as rpa

# The cache's stored form per chunk: a pytree with the same structure
# as `decode_state`, where each DecodeState leaf is replaced by the
# offloaded payload — a host-resident `jax.Array` (when the original
# `SnapshotChunkLeaf.available` was `True`) or a `jax.ShapeDtypeStruct`
# carrying just shape/dtype/sharding (when `available` was `False`;
# the bytes would be post-eviction garbage and aren't worth storing).
# See :meth:`rpa.SnapshotChunkLeaf.offload` / :meth:`onload`.
StoredTree = Any

# 16 hex chars (64 bits) — plenty for a per-job cache, keeps keys short.
_HASH_HEX_LEN = 16


def chunk_keys(tokens: np.ndarray, chunk_size: int) -> Sequence[str]:
  """Returns one hash per *full* chunk in `tokens`."""
  n_full = len(tokens) // chunk_size
  if n_full == 0:
    return []
  tokens_i32 = np.ascontiguousarray(tokens, dtype=np.int32)
  hasher = hashlib.blake2b(digest_size=_HASH_HEX_LEN // 2)
  keys: list[str] = []
  for i in range(n_full):
    hasher.update(tokens_i32[i * chunk_size : (i + 1) * chunk_size].tobytes())
    keys.append(hasher.copy().hexdigest())
  return keys


def _tree_is_complete(stored_tree: StoredTree) -> bool:
  """Whether every leaf is a real `jax.Array` (not a `ShapeDtypeStruct`).

  A stored tree is "complete" iff every leaf holds real KV bytes —
  equivalent to `available=True` for every leaf at offload time, since
  :meth:`rpa.SnapshotChunkLeaf.offload` returns a `ShapeDtypeStruct`
  exactly when `available=False`.

  Args:
    stored_tree: The offloaded payload pytree for a chunk.

  Returns:
    `True` if every leaf is a real `jax.Array`, `False` otherwise.
  """
  for leaf in jax.tree_util.tree_leaves(stored_tree):
    if not isinstance(leaf, jax.Array):
      return False
  return True


def _tree_nbytes(stored_tree: StoredTree) -> int:
  """Returns the host-RAM byte footprint of a stored tree.

  Sums `nbytes` over `jax.Array` leaves only — `ShapeDtypeStruct`
  leaves carry shape/dtype metadata but no bytes, so they contribute
  zero. Cached once per entry in :class:`_Record` so eviction and
  promotion don't re-walk the pytree.

  Args:
    stored_tree: The offloaded payload pytree for a chunk.

  Returns:
    The host-RAM byte footprint of the stored tree.
  """
  total = 0
  for leaf in jax.tree_util.tree_leaves(stored_tree):
    if isinstance(leaf, jax.Array):
      total += leaf.nbytes
  return total


@dataclasses.dataclass
class _Record:
  """One stored chunk's record: tree + accounting metadata.

  Fields
    stored_tree: the offloaded payload pytree for this chunk.
    nbytes: host-RAM byte footprint of `stored_tree`. Recomputed
      when an incomplete entry is promoted to complete (whole-tree
      overwrite).
    is_complete: whether every leaf of `stored_tree` is a real
      `jax.Array` (i.e. equivalent to `_tree_is_complete(stored_tree)`).
      Cached at snapshot/promotion time so the hot lookup paths
      (`is_complete` / `longest_restorable_prefix`) are O(1) per
      entry instead of re-walking the pytree. Completeness only
      changes monotonically `False -> True` via promotion.
    parent_key: the immediate-parent chunk key (the chunk one step
      shorter than this one in the same prefix), or `None` for chunk
      0. Used to keep parent-child counts in sync on eviction.
    n_children: number of cached chunks immediately *following* this
      one (i.e. chunks whose `parent_key` is this entry's key). The
      LRU rule only evicts entries with `n_children == 0` so that
      eviction never strands a longer cached chunk behind a missing
      prefix.
  """

  stored_tree: StoredTree
  nbytes: int
  is_complete: bool
  parent_key: str | None
  n_children: int = 0


@dataclasses.dataclass(frozen=True, kw_only=True)
class PrefixCache:
  """Chunk-granular content-addressable prefix cache, backed by host RAM.

  Each chunk is stored as a pytree of offloaded payloads (one per
  DecodeState leaf) — see :type:`StoredTree`. The cache treats the
  tree as opaque structure-preserving storage; the per-leaf
  availability decision lives entirely in
  :meth:`rpa.SnapshotChunkLeaf.offload`.

  The lookup-time "complete" predicate is just "every leaf is a real
  `jax.Array` (not a `ShapeDtypeStruct`)", which the caller drives
  via `longest_restorable_prefix`. See the module docstring.

  LRU eviction
    When `max_bytes` is set, after each `snapshot()` (and after any
    promotion that grew an existing entry) the cache evicts from its
    LRU end until the total host-RAM footprint is at or under the
    bound. The LRU order is touched by `restore()` and by every
    `snapshot()` of an existing key (any code path that observes a
    chunk on-device is a liveness signal). `is_cached`/`is_complete`
    probes are pure metadata reads and don't count as use.

    Eviction is **leaf-only**: a chunk key is evicted only if it has
    no longer cached chunk continuing it (i.e. no other cached entry
    whose `parent_key` is this key). This guarantees that every entry
    surviving in the cache is reachable from the prefix root —
    otherwise `longest_restorable_prefix` would stop at the gap and
    the orphaned tail would just hold bytes without ever being a hit.
    If the LRU end is an internal node (has children), we walk
    further into the LRU end skipping it until we find a leaf.
  """

  chunk_size: int
  max_bytes: int | None = None

  def __post_init__(self) -> None:
    if self.chunk_size <= 0:
      raise ValueError(f'chunk_size must be positive; got {self.chunk_size}')
    if self.max_bytes is not None and self.max_bytes < 0:
      raise ValueError(f'max_bytes must be non-negative; got {self.max_bytes}')

  @functools.cached_property
  def _counters(self) -> dict[str, int]:
    return {
        'n_snapshot': 0,
        'n_promoted': 0,
        'n_restore': 0,
        'n_hit': 0,
        'n_evicted': 0,
        'n_eviction_skipped_non_leaf': 0,
        'total_bytes': 0,
    }

  @functools.cached_property
  def _store(self) -> 'collections.OrderedDict[str, _Record]':
    """Per-process host-RAM store: `chunk_key -> _Record`.

    `OrderedDict` ordering = LRU: oldest-used at the front, most
    recently used at the back. `restore()` moves entries to the back;
    eviction pops from the front (skipping non-leaf entries).

    Each entry holds one chunk's offloaded payload pytree (per-leaf:
    either a host-resident `jax.Array` or a `jax.ShapeDtypeStruct`)
    plus byte accounting and parent-child bookkeeping. Membership
    doubles as the in-memory dedup set, so :meth:`is_cached` is just
    a dict membership test.

    The store is process-local (RAM, not disk): every host holds
    only its own addressable shards and never reads peers' state. The
    callers nonetheless drive snapshot/restore from replicated state in
    lockstep, so the key set evolves identically on every host — which
    keeps the surrounding `extract_chunk`/`inject_chunk` device programs
    (the only collective ops) in step.
    """
    return collections.OrderedDict()

  def _chunk_key(self, token_chunks: np.ndarray) -> str:
    """Returns the content key for the chunk ending at `len(token_chunks)`."""
    n = len(token_chunks)
    if n <= 0 or n % self.chunk_size != 0:
      raise ValueError(
          f'token_chunks length {n} must be a positive multiple of '
          f'chunk_size={self.chunk_size}.'
      )
    return chunk_keys(token_chunks, self.chunk_size)[-1]

  def is_cached(self, token_chunks: np.ndarray) -> bool:
    """Whether the chunk at `len(token_chunks)` has any entry in the cache.

    Uses the in-memory key set (identical across hosts), so the decision
    is collective — callers can use it to skip the collective
    `extract_chunk` read for chunks already cached, without desyncing.

    Args:
      token_chunks: Cumulative prefix tokens up through the chunk to check.

    Returns:
      `True` if the chunk is present in the cache, `False` otherwise.
    """
    return self._chunk_key(token_chunks) in self._store

  def is_complete(self, token_chunks: np.ndarray) -> bool:
    """Whether every leaf of the cached entry is a real `jax.Array`.

    Returns `False` for both "not in cache" and "in cache but at least
    one leaf is a `ShapeDtypeStruct` (the leaf was extracted with
    `available=False`)." Callers needing to distinguish should also
    check :meth:`is_cached`.

    Args:
      token_chunks: Cumulative prefix tokens up through the chunk to check.

    Returns:
      `True` if the cached entry is present and complete, `False` otherwise.
    """
    record = self._store.get(self._chunk_key(token_chunks))
    return record is not None and record.is_complete

  def longest_restorable_prefix(
      self,
      slot_tokens: np.ndarray,
      max_chunks: int,
      start_chunk_idx: int,
      min_num_chunks_for_window: int,
  ) -> int:
    """Returns the deepest restorable prefix length `k` (in chunks).

    Walks chunks `start_chunk_idx .. max_chunks - 1` and picks the
    deepest `k` such that:

      * every chunk in `[start_chunk_idx, k)` is **present** in the
        cache (so global-layer leaves are real for every chunk), AND
      * every chunk in `[max(start_chunk_idx, k - min_num_chunks_for_window),
        k)` is **complete** (every leaf's payload is present, so the
        widest windowed layer's tail is real; older chunks' windowed-
        layer payloads, if `None`, are filled with uninitialised
        buffers by the inject path and evicted by
        `release_for_window` before they can feed into attention).

    The walk doesn't stop at the first incomplete chunk — an
    incomplete chunk at index `i` simply forces any returned `k` to
    satisfy `k - min_num_chunks_for_window > i`, leaving room for deeper
    hits whose tail starts past `i`. It does stop at the first
    *missing* chunk: no `k` past a gap is reachable.

    Args:
      slot_tokens: Cumulative prompt tokens for the slot being restored.
      max_chunks: The upper bound on the number of chunks to check.
      start_chunk_idx: The starting chunk index to walk from.
      min_num_chunks_for_window: The caller's
        `SamplingState.min_num_chunks_for_window` (the max over windowed
        DecodeState leaves of `ceil(window_size / chunk_size)`); `0` means "no
        windowed layers, presence alone is enough."

    Returns:
      The deepest restorable prefix length `k` (in chunks), or `start_chunk_idx`
      if no deeper prefix is fully satisfiable.
    """
    # `last_incomplete_at` = highest chunk idx (inclusive) whose entry
    # is present but not complete. A prefix length `k` is OK iff
    # `last_incomplete_at < k - min_num_chunks_for_window` (i.e. no incomplete
    # chunk falls inside the candidate tail `[k - tail, k)`).
    #
    # Sentinel: `-(min_num_chunks_for_window + 1)` is strictly less than
    # `k - min_num_chunks_for_window` for every `k >= 1`, so before we see any
    # incomplete chunk the condition trivially holds.
    last_incomplete_at = -(min_num_chunks_for_window + 1)
    best_k = start_chunk_idx
    for chunk_idx in range(start_chunk_idx, max_chunks):
      key = self._chunk_key(
          slot_tokens[: (chunk_idx + 1) * self.chunk_size]
      )
      record = self._store.get(key)
      if record is None:
        return best_k
      if not record.is_complete:
        last_incomplete_at = chunk_idx
      candidate_k = chunk_idx + 1
      if last_incomplete_at < candidate_k - min_num_chunks_for_window:
        best_k = candidate_k
    return best_k

  def restore(self, token_chunks: np.ndarray) -> StoredTree | None:
    """Returns the cached stored tree, or `None` on miss.

    The tree is in the **stored** form: each leaf is either a
    host-resident `jax.Array` (real KV bytes) or a `jax.ShapeDtypeStruct`
    (the leaf was unavailable at extract time). The caller is expected
    to tree-map :meth:`rpa.SnapshotChunkLeaf.onload`, which moves real
    payloads host->HBM and materialises uninitialised buffers from
    `ShapeDtypeStruct` leaves, before passing the result to
    `inject_chunk`. Host-local: no collective op, no barrier.

    Hits also **touch** the LRU position (move to most-recently-used
    end). Misses don't. `is_cached`/`is_complete` probes don't touch
    either — only an actual restore (= "we injected this chunk")
    counts as use.

    Args:
      token_chunks: Cumulative prefix tokens up through the chunk to restore.

    Returns:
      The cached stored tree, or `None` on miss.
    """
    key = self._chunk_key(token_chunks)
    record = self._store.get(key)
    if record is None:
      return None
    self._store.move_to_end(key)
    self._counters['n_restore'] += 1
    self._counters['n_hit'] += 1
    return record.stored_tree

  def snapshot(self, token_chunks: np.ndarray, leaf_tree: Any) -> None:
    """Caches `leaf_tree` for the chunk at `len(token_chunks)`.

    Calls :meth:`rpa.SnapshotChunkLeaf.offload` on every leaf, yielding
    a pytree where each entry is either a host-resident `jax.Array`
    (real KV; `available=True`) or a `jax.ShapeDtypeStruct` (just
    shape/dtype/sharding; `available=False` — saves host RAM by not
    storing post-eviction garbage). Host-local: NO collective op and
    NO `sync_global_devices` barrier (unlike the old Orbax `save()`).

    Per-chunk promotion rule (no per-leaf merging):

      * If no entry exists for this chunk key, the offloaded tree is
        stored as-is and the new entry goes to the MRU end of the LRU
        order.
      * If an entry exists and is complete, the snapshot is a no-op
        on stored bytes (first writer wins), but the LRU position is
        touched to MRU end (the on-device observation is a liveness
        signal).
      * If an entry exists and is incomplete:
          - new is incomplete: no-op on stored bytes; touch LRU.
          - new is complete: OVERWRITE the whole stored tree; touch
            LRU. (Per-leaf carryover is intentionally not done; the
            new tree is strictly more useful.)

    LRU eviction (only when `max_bytes` is set): after the insert or
    promotion, if the running total exceeds `max_bytes`, evict from
    the LRU end skipping non-leaf entries (chunks with cached
    continuations) until under the bound. If the only thing in the
    cache is the just-stored chunk and it alone exceeds `max_bytes`,
    it is kept (one-chunk-over-budget rather than refusing the
    snapshot).

    Args:
      token_chunks: cumulative prefix tokens up through the chunk to cache.
      leaf_tree: a pytree of :class:`rpa.SnapshotChunkLeaf` (HBM-resident),
        typically the result of :meth:`SamplingState.extract_chunk`.
    """
    key = self._chunk_key(token_chunks)
    try:
      offloaded = jax.tree_util.tree_map(
          rpa.SnapshotChunkLeaf.offload,
          leaf_tree,
          is_leaf=lambda x: isinstance(x, rpa.SnapshotChunkLeaf),
      )
    except (RuntimeError, ValueError) as e:
      # Defensive: a device_put failure must not crash the decode loop
      # nor leave a half-written entry. Drop the chunk; it will simply
      # be recomputed on the next prompt that needs it.
      logging.warning(
          'prefix_cache: failed to offload chunk %s to host: %s', key, e
      )
      return
    existing = self._store.get(key)
    if existing is None:
      # Fresh entry. Compute parent key and bump its child count.
      # The plain `__setitem__` places the new key at the MRU end of
      # the LRU order, so freshly-snapshotted chunks are never the
      # first to be evicted.
      parent_key = self._parent_key(token_chunks)
      nbytes = _tree_nbytes(offloaded)
      record = _Record(
          stored_tree=offloaded,
          nbytes=nbytes,
          is_complete=_tree_is_complete(offloaded),
          parent_key=parent_key,
      )
      self._store[key] = record
      self._counters['total_bytes'] += nbytes
      if parent_key is not None and parent_key in self._store:
        self._store[parent_key].n_children += 1
      self._counters['n_snapshot'] += 1
      self._evict_until_under_bound(protected_key=key)
      return
    # TODO: The way it does LRU is pretty messy, and probably incorrect
    # when the capacity is very tight (may result in evicting something that is
    # not the true LRU).
    # Existing-key path: any snapshot that observes this chunk on-device
    # is evidence of liveness (a slot is currently reading or writing
    # its KV). Touch the LRU position so actively-decoded prefixes don't
    # age toward the eviction front while a slot is still working
    # through them. Snapshots walk chunks in ascending order, so the
    # parent-before-child LRU invariant is preserved.
    self._store.move_to_end(key)
    # Per-chunk promotion rule (no per-leaf merging):
    #   - existing complete    + any new       -> no-op (first writer wins)
    #   - existing incomplete  + new incomplete -> no-op
    #   - existing incomplete  + new complete   -> OVERWRITE whole tree
    # Completeness only flips False -> True (never the other way), so
    # an already-complete entry can never be improved.
    if existing.is_complete:
      return
    if not _tree_is_complete(offloaded):
      return
    new_nbytes = _tree_nbytes(offloaded)
    delta = new_nbytes - existing.nbytes
    existing.stored_tree = offloaded
    existing.nbytes = new_nbytes
    existing.is_complete = True
    self._counters['total_bytes'] += delta
    self._counters['n_promoted'] += 1
    self._evict_until_under_bound(protected_key=key)

  def _parent_key(self, token_chunks: np.ndarray) -> str | None:
    """Returns the chunk key of the immediate parent, or `None` for chunk 0."""
    n = len(token_chunks)
    if n <= self.chunk_size:
      return None
    parent_tokens = token_chunks[: n - self.chunk_size]
    return chunk_keys(parent_tokens, self.chunk_size)[-1]

  def _evict_until_under_bound(self, *, protected_key: str) -> None:
    """Evicts LRU leaf entries until `_total_bytes <= max_bytes`.

    Walks the LRU order from least-recently-used end, skipping:
      * the just-inserted entry (`protected_key`) — a freshly stored
        chunk shouldn't be the first to go;
      * any entry with `n_children > 0` (eviction would strand its
        cached continuations behind a gap).

    If no eligible entry exists, returns silently with the total
    still over budget. (The "single chunk > max_bytes" case lands
    here: the only entry is the protected one, nothing to evict, and
    we accept being over-budget.)

    Args:
      protected_key: The key of the just-inserted entry to protect from
        eviction.
    """
    if self.max_bytes is None:
      return
    if self._counters['total_bytes'] <= self.max_bytes:
      return
    # Walk a snapshot of the LRU order so we can skip without breaking
    # iteration. `OrderedDict` iterates oldest -> newest.
    candidates = list(self._store.keys())
    for victim_key in candidates:
      if self._counters['total_bytes'] <= self.max_bytes:
        return
      if victim_key == protected_key:
        continue
      record = self._store.get(victim_key)
      if record is None:
        continue  # Already evicted earlier in this pass.
      if record.n_children > 0:
        self._counters['n_eviction_skipped_non_leaf'] += 1
        continue
      # Evict.
      del self._store[victim_key]
      self._counters['total_bytes'] -= record.nbytes
      if (
          record.parent_key is not None
          and record.parent_key in self._store
      ):
        self._store[record.parent_key].n_children -= 1
      self._counters['n_evicted'] += 1

  def clear(self) -> None:
    """Drops every cached chunk and resets accounting."""
    n_entries = len(self._store)
    bytes_freed = self._counters['total_bytes']
    self._store.clear()
    for counter_key in self._counters:
      self._counters[counter_key] = 0
    logging.info(
        'prefix_cache: cleared %d entries, freed %d bytes (cache retired for'
        ' stale-weight invalidation at RL step boundary).',
        n_entries,
        bytes_freed,
    )

  def stats(self) -> Mapping[str, Any]:
    return dict(self._counters)

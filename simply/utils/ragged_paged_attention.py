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
"""Ragged paged attention API for Simply LM."""

import collections
from collections.abc import Callable, Hashable, Iterable, Mapping, MutableMapping, Sequence
import dataclasses
import functools
import math
from typing import Any, Self

from absl import logging
import einops
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply.kernels import ragged_paged_attention as rpa_kernel
from simply.utils import common
from simply.utils import quant
from simply.utils import sampling_lib
from simply.utils import sharding

RaggedArray = common.RaggedArray


def autotune_block_sizes(
    *,
    num_kv_heads: int,
    num_q_heads: int,
    page_size: int,
    max_seq_len: int,
    per_head_dim: int,
    window_size: int | None,
    dtype: jax.typing.DTypeLike,
    max_num_issue_tokens: int = np.iinfo(np.int32).max,
    num_shards: int = 1,
):
  """Autotunes block sizes for ragged paged attention."""
  del num_q_heads
  # TODO: More analysis on this value.
  # Increasing this value would shift the attention module from memory bandwidth
  # bound to compute bound, but in the meanwhile, it would cause more padding
  # overhead, given decoding does one-by-one token generation. 32 is a good
  # empirical trade-off so far.
  num_queries_per_block = min(32, max_num_issue_tokens)
  num_combined_kv_heads = rpa_kernel.align_to(
      num_kv_heads * 2, rpa_kernel.get_dtype_packing(dtype)
  )

  # This is an empirical estimation of the DMA issuing/waiting overhead
  # (non-data-transport cost). It indicates the multiplication of the overhead
  # latency and max HBM->VMEM bandwidth.
  # We assume at each new TPU generation, the overhead latency would be reduced
  # and in the meanwhile the HBM->VMEM bandwidth would be increased. Therefore,
  # the equivalent bytes of the overhead should remain at the similar level.
  dma_overhead_equivalent_bytes = 0.5 * 1024 * 1024  # 0.5MiB
  dma_overhead = dma_overhead_equivalent_bytes / (
      page_size
      * num_combined_kv_heads
      * per_head_dim
      * jnp.dtype(dtype).itemsize
  )
  avg_seq_len_per_shard = rpa_kernel.cdiv(max_seq_len, num_shards) / 2
  accum_seq_len_per_shard = avg_seq_len_per_shard / 2
  padding_overhead_per_kv_page_blk = (page_size / 2) / (
      min(rpa_kernel.cdiv(window_size, num_shards), accum_seq_len_per_shard)
      if window_size
      else accum_seq_len_per_shard
  )
  num_kv_pages_per_block = round(
      math.sqrt(dma_overhead / padding_overhead_per_kv_page_blk)
  )
  max_num_kv_pages_upper_bound = max_num_pages_per_seq_per_shard(
      max_seq_len, page_size, window_size, num_shards
  )
  return (
      min(num_kv_pages_per_block, max_num_kv_pages_upper_bound),
      num_queries_per_block,
  )


def max_num_pages_per_seq_per_shard(
    max_seq_len: int,
    page_size: int,
    window_size: int | None,  # self excluded
    num_shards: int = 1,
) -> int:
  """Returns the maximum number of pages per sequence per shard."""
  upper_bound = rpa_kernel.cdiv(
      rpa_kernel.cdiv(max_seq_len - 1, page_size), num_shards
  )
  if window_size is None:
    return upper_bound
  num_pages_for_window = rpa_kernel.cdiv(
      rpa_kernel.cdiv(window_size, page_size), num_shards
  )
  return min(upper_bound, num_pages_for_window + 1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeStateConfig:
  """Paged KV cache config."""

  total_num_pages: int
  page_size: int
  n_kv_heads: int
  per_head_dim: int
  batch_size: int
  dtype: jax.typing.DTypeLike
  max_seq_len: int
  window_size: int | None = None  # self excluded
  head_partition: str | Sequence[str] | None = None
  seq_partition: str | Sequence[str] | None = None
  # Per-tensor dequant scales for a low-precision (fp8) cache.
  k_scale: float | None = None
  v_scale: float | None = None

  @property
  def padded_per_head_dim(self) -> int:
    return (self.per_head_dim + 127) // 128 * 128

  @property
  def max_num_pages_per_seq_per_shard(self) -> int:
    num_shards = sharding.get_partition_size(self.seq_partition)
    return max_num_pages_per_seq_per_shard(
        self.max_seq_len, self.page_size, self.window_size, num_shards
    )

  def init(self) -> 'DecodeState':
    kv_packing = rpa_kernel.get_dtype_packing(self.dtype)
    num_shards = sharding.get_partition_size(self.seq_partition)
    num_pages_per_shard = self.total_num_pages // num_shards
    return DecodeState(
        pages=sharding.with_sharding_constraint(
            jax.lax.empty(
                (
                    self.total_num_pages,
                    self.page_size,
                    self.n_kv_heads * 2 // kv_packing,
                    kv_packing,
                    self.padded_per_head_dim,
                ),
                dtype=self.dtype,
            ),
            (self.seq_partition, None, self.head_partition, None, None),
        ),
        page_indices=jax.lax.empty(
            (self.batch_size, self.max_num_pages_per_seq_per_shard),
            dtype=jnp.int32,
        ),
        available_page_indices=jnp.arange(num_pages_per_shard, dtype=jnp.int32),
        num_available_pages=jnp.array(num_pages_per_shard, dtype=jnp.int32),
        kv_lens=jnp.zeros(self.batch_size, dtype=jnp.int32),
        max_seq_len=self.max_seq_len,
        window_size=self.window_size,
        head_partition=self.head_partition,
        seq_partition=self.seq_partition,
        k_scale=self.k_scale,
        v_scale=self.v_scale,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeState:
  """Paged KV cache."""

  # [total_num_pages, page_size, num_kv_heads * 2 // kv_packing,
  #   kv_packing, padded_per_head_dim]
  pages: jax.Array
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq_per_shard]
  available_page_indices: jax.Array  # i32[total_num_pages_per_shard]
  num_available_pages: jax.Array  # i32[]

  kv_lens: jax.Array  # i32[batch_size]
  max_seq_len: int = dataclasses.field(metadata=dict(static=True))
  window_size: int | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )
  head_partition: str | Sequence[str] | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )
  seq_partition: str | Sequence[str] | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )
  k_scale: float | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )
  v_scale: float | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )

  def __post_init__(self):
    if not isinstance(self.pages, jax.Array):
      # This is for jax compatibility check.
      return
    head_partition_size = sharding.get_partition_size(self.head_partition)
    kv_packing = rpa_kernel.get_dtype_packing(self.dtype)
    if self.num_kv_heads * 2 // kv_packing % head_partition_size != 0:
      raise ValueError(
          f'{self.num_kv_heads * 2=} // {kv_packing=} must be a multiple of'
          f' {head_partition_size=}'
      )
    if self.page_indices.shape != (
        self.batch_size,
        self.max_num_pages_per_seq_per_shard,
    ):
      raise ValueError(
          f'{self.page_indices.shape=} does not match'
          f' {self.batch_size=}, {self.max_num_pages_per_seq_per_shard=}'
      )
    if self.page_indices.dtype != jnp.int32:
      raise ValueError(f'{self.page_indices.dtype=} must be int32.')
    if self.available_page_indices.shape != (self.total_num_pages_per_shard,):
      raise ValueError(
          f'{self.available_page_indices.shape=} must be'
          f' {self.total_num_pages_per_shard=}'
      )
    if self.available_page_indices.dtype != jnp.int32:
      raise ValueError(f'{self.available_page_indices.dtype=} must be int32.')
    if (
        self.num_available_pages.shape
        or self.num_available_pages.dtype != jnp.int32
    ):
      raise ValueError(f'{self.num_available_pages=} must be i32()')
    if (
        self.kv_lens.shape != (self.batch_size,)
        or self.kv_lens.dtype != jnp.int32
    ):
      raise ValueError(
          f'KV lens must be i32[{self.batch_size=}].'
          f' {self.kv_lens.shape=}, {self.kv_lens.dtype=}'
      )
    if self.padded_per_head_dim % 128 != 0:
      raise ValueError(
          f'Pages {self.padded_per_head_dim=} must be a multiple of 128.'
      )
    if self.window_size is not None and self.window_size <= 0:
      logging.info(
          'Resetting window_size=%d to None, because it is <= 0.',
          self.window_size,
      )
      object.__setattr__(self, 'window_size', None)
    if self.max_num_pages_per_seq_per_shard > self.total_num_pages_per_shard:
      raise ValueError(
          f'{self.max_num_pages_per_seq_per_shard=} must be <='
          f' {self.total_num_pages_per_shard=}'
      )

  @classmethod
  def attrs_from_tree(
      cls, tree: common.PyTree, attr_names: Iterable[str]
  ) -> Mapping[str, Sequence[Any]]:
    """Returns attributes from the tree."""
    leaves = jax.tree_util.tree_leaves(
        tree, is_leaf=lambda x: isinstance(x, DecodeState)
    )
    attrs = collections.defaultdict(list)
    for leaf in leaves:
      if not isinstance(leaf, DecodeState):
        raise ValueError(f'{leaf=} is not a DecodeState.')
      for name in attr_names:
        attrs[name].append(getattr(leaf, name))
    return attrs

  @property
  def batch_size(self) -> int:
    return self.kv_lens.shape[0]

  @property
  def max_num_pages_per_seq_per_shard(self) -> int:
    return max_num_pages_per_seq_per_shard(
        self.max_seq_len, self.page_size, self.window_size, self.num_shards
    )

  @property
  def total_num_pages(self) -> int:
    return self.pages.shape[0]

  @property
  def num_shards(self) -> int:
    return sharding.get_partition_size(self.seq_partition)

  @property
  def total_num_pages_per_shard(self) -> int:
    return self.total_num_pages // self.num_shards

  @property
  def page_size(self) -> int:
    return self.pages.shape[1]

  @property
  def chunk_size(self) -> int:
    """Inject / extract granularity in tokens (`num_shards * page_size`)."""
    return self.num_shards * self.page_size

  @property
  def min_num_chunks_for_window(self) -> int:
    """Smallest #chunks that fully cover this layer's sliding window.

    For a windowed layer this is `ceil(window_size / chunk_size)` --
    that many consecutive chunks (when freshly injected) saturate the
    layer's resident window, so any older chunks injected behind them
    are evicted by `release_for_window` before they can feed into
    attention. For a global layer (`window_size is None`) this is `0`
    -- a global layer's window is effectively infinite, so no
    re-injection of older chunks is needed to displace stale tail KV.
    """
    if self.window_size is None:
      return 0
    return rpa_kernel.cdiv(self.window_size, self.chunk_size)

  @property
  def num_kv_heads(self) -> int:
    return self.pages.shape[2] * self.pages.shape[3] // 2

  @property
  def padded_per_head_dim(self) -> int:
    return self.pages.shape[-1]

  @property
  def dtype(self) -> jax.typing.DTypeLike:
    return self.pages.dtype

  @functools.cached_property
  def local_num_pages(self) -> jax.Array:
    return rpa_kernel.cdiv(
        rpa_kernel.cdiv(self.kv_lens, self.page_size), self.num_shards
    )

  def pad_per_head_dim(self, x: jax.Array) -> jax.Array:
    # Return in shape [batch_size, n_heads, padded_per_head_dim]
    if x.shape[-1] < self.padded_per_head_dim:
      return jnp.pad(
          x,
          (
              (0, 0),
              (0, 0),
              (0, self.padded_per_head_dim - x.shape[-1]),
          ),
      )
    return x

  @functools.cached_property
  def available_page_indices_np(self) -> np.ndarray:
    available_page_indices = np.asarray(self.available_page_indices)[
        : int(self.num_available_pages)
    ]
    available_page_indices = (
        available_page_indices[:, None]
        + np.arange(self.num_shards) * self.total_num_pages_per_shard
    )
    return np.reshape(available_page_indices, -1)

  def page_indices_np(self, idx: jax.typing.ArrayLike) -> np.ndarray:
    """Returns the page indices for the given idx."""
    page_indices = np.asarray(self.page_indices[idx])[
        : int(self.local_num_pages[idx])
    ]
    page_indices = (
        page_indices[:, None]
        + np.arange(self.num_shards) * self.total_num_pages_per_shard
    )
    return np.reshape(page_indices, -1)

  @functools.cached_property
  def page_indices_nplist(self) -> Sequence[np.ndarray]:
    """Returns the page indices for each sequence as a list of numpy arrays."""
    return [self.page_indices_np(i) for i in range(self.batch_size)]

  def kv_np(
      self, idx: jax.typing.ArrayLike, per_head_dim: int = 0
  ) -> np.ndarray:
    """Returns the kv for the given idx."""
    # Return shape in [kv_len, num_kv_heads * 2, per_head_dim]
    page_indices = self.page_indices_np(idx)
    context = []
    for pi in page_indices:
      page = multihost_utils.process_allgather(
          self.pages[pi][..., :per_head_dim], tiled=True
      )
      page = page.reshape(-1, self.num_kv_heads * 2, per_head_dim)
      context.append(page)
    kv_len = int(self.kv_lens[idx])
    if kv_len:
      return np.concatenate(context)[:kv_len]
    return np.empty((0, self.num_kv_heads * 2, per_head_dim), dtype=self.dtype)

  def kv_nplist(self, per_head_dim: int = 0) -> Sequence[np.ndarray]:
    return [
        self.kv_np(i, per_head_dim=per_head_dim) for i in range(self.batch_size)
    ]

  @functools.cached_property
  def max_available_kv_lens(self) -> jax.Array:
    """Returns the maximum available KV lens for each sequence."""
    kv_lens = self.kv_lens
    if self.window_size is not None:
      max_num_local_removable_pages = (
          jnp.maximum(kv_lens - self.window_size, 0)
          // self.page_size
          // self.num_shards
      )
      kv_lens -= (
          max_num_local_removable_pages * self.page_size * self.num_shards
      )
    return (
        self.num_shards * self.page_size * self.max_num_pages_per_seq_per_shard
        - kv_lens
    )

  @jax.named_call
  def release_for_window(self) -> Self:
    """Releases the decode state for local attention."""
    if self.window_size is None:
      return self
    num_pages_to_release_per_shard = (
        jnp.maximum(self.kv_lens - self.window_size, 0)
        // self.page_size
        // self.num_shards
    )
    page_indices_irows = jnp.arange(self.batch_size)[:, None]
    page_indices_icols = (
        jnp.arange(self.max_num_pages_per_seq_per_shard)
        + num_pages_to_release_per_shard[:, None]
    )
    updated_page_indices = self.page_indices[
        page_indices_irows, page_indices_icols
    ]
    release_helper = RaggedArray(
        data=jax.lax.empty((self.total_num_pages_per_shard,), dtype=jnp.int32),
        lens=num_pages_to_release_per_shard,
    )
    released_page_indices = self.page_indices[
        release_helper.row_ids, release_helper.intra_offset
    ]
    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages_per_shard) + self.num_available_pages
    ].set(released_page_indices, mode='drop')
    num_pages_to_release = num_pages_to_release_per_shard * self.num_shards
    return dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        available_page_indices=updated_available_page_indices,
        num_available_pages=self.num_available_pages
        + release_helper.total_length,
        kv_lens=self.kv_lens - num_pages_to_release * self.page_size,
    )

  @jax.named_call
  def allocate(self, q_lens: jax.Array) -> Self:
    """Allocates pages for new tokens.

    Pages are managed at num_shards granularity. Each shard gets
    ceil(num_pages / num_shards) pages, making allocations uniform across
    shards and eliminating the need for shard_map.

    Args:
      q_lens: number of new tokens to allocate pages for.

    Returns:
      Updated decode state with pages allocated for new tokens.
    """
    required_local_num_pages = rpa_kernel.cdiv(
        rpa_kernel.cdiv(self.kv_lens + q_lens, self.page_size),
        self.num_shards,
    )
    num_pages_to_allocate = required_local_num_pages - self.local_num_pages
    # User should guarantee:
    # total_num_pages_to_allocate <= num_available_pages
    page_indices_to_allocate = RaggedArray(
        data=self.available_page_indices, lens=num_pages_to_allocate
    )
    page_indices_irows = page_indices_to_allocate.row_ids
    page_indices_icols = (
        self.local_num_pages[page_indices_to_allocate.row_ids]
        + page_indices_to_allocate.intra_offset
    )
    updated_page_indices = self.page_indices.at[
        page_indices_irows, page_indices_icols
    ].set(page_indices_to_allocate.data)
    updated_num_available_pages = (
        self.num_available_pages - page_indices_to_allocate.total_length
    )
    updated_available_page_indices = jnp.roll(
        self.available_page_indices, -page_indices_to_allocate.total_length
    )
    return dataclasses.replace(
        self,
        kv_lens=self.kv_lens + q_lens,
        page_indices=updated_page_indices,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )

  @jax.named_call
  def release(self, should_release: jax.Array) -> Self:
    """Releases the decode state."""
    updated_kv_lens = jnp.where(should_release, 0, self.kv_lens)
    page_indices_to_release = RaggedArray(
        data=jax.lax.empty((self.total_num_pages_per_shard,), dtype=jnp.int32),
        lens=jnp.where(should_release, self.local_num_pages, 0),
    )
    page_indices_irows = page_indices_to_release.row_ids
    page_indices_icols = page_indices_to_release.intra_offset
    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages_per_shard) + self.num_available_pages
    ].set(self.page_indices[page_indices_irows, page_indices_icols])
    updated_num_available_pages = (
        self.num_available_pages + page_indices_to_release.total_length
    )
    return dataclasses.replace(
        self,
        kv_lens=updated_kv_lens,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )

  def slot_global_page_indices(
      self,
      slot_id: jax.typing.ArrayLike,
      num_pages: int,
      start_page: jax.typing.ArrayLike = 0,
  ) -> jax.Array:
    """Returns `num_pages` *global* page IDs for a slot, starting at `start_page`.

    `num_pages` must be static (used as `jnp.arange`'s size).
    `start_page` may be either a static int or a dynamic scalar
    `jax.Array` — the latter is useful when injecting at a runtime
    offset (e.g. `kv_lens // page_size`).
    The result has shape `[num_pages]` and is suitable for indexing into
    `self.pages`.

    Token position `p` of slot `S` lives on shard `(p // page_size) %
    num_shards` at the slot's shard-local column `(p // page_size) //
    num_shards` (see :meth:`insert`). Inverting: page `i` of slot `S`
    (i.e. covering tokens `[i * page_size, (i + 1) * page_size)`) is at
    `page_indices[S, i // num_shards] + (i % num_shards) *
    total_num_pages_per_shard`.

    Args:
      slot_id: scalar i32 — the batch slot to read page IDs for.
      num_pages: static number of pages to return (the result length).
      start_page: static int or dynamic scalar i32 — the first page index.

    Returns:
      An `[num_pages]` i32 array of global page IDs into `self.pages`.
    """
    iota = jnp.arange(num_pages, dtype=jnp.int32) + start_page
    shard_ids = iota % self.num_shards
    local_cols = iota // self.num_shards
    local_indices = self.page_indices[slot_id, local_cols]
    return local_indices + shard_ids * self.total_num_pages_per_shard

  @jax.named_call
  def insert(self, k: jax.Array, v: jax.Array, q_lens: jax.Array) -> Self:
    """Inserts new kv into kv_pages at [kv_lens - q_lens, kv_lens).

    For test only.

    Args:
      k: keys to insert.
      v: values to insert.
      q_lens: number of new tokens to insert.

    Returns:
      Updated decode state with new kv inserted.
    """
    k = self.pad_per_head_dim(k)
    v = self.pad_per_head_dim(v)
    kv_packing = self.pages.shape[3]
    new_ragged_kv = RaggedArray(
        jnp.stack([k, v], axis=-2).reshape(
            k.shape[0], -1, kv_packing, self.padded_per_head_dim
        ),
        lens=q_lens,
    )

    row_ids = new_ragged_kv.row_ids
    intra_offset = new_ragged_kv.intra_offset
    positions = (self.kv_lens - q_lens)[row_ids] + intra_offset

    # page_indices has shape [batch, max_pages_per_shard_per_seq] with
    # shard-local page IDs.  Global page position p lives on shard
    # (p % num_shards) at local column (p // num_shards).
    global_page_pos = positions // self.page_size
    shard_ids = global_page_pos % self.num_shards
    page_indices_icols = global_page_pos // self.num_shards

    page_indices_irows = row_ids
    page_indices = self.page_indices[page_indices_irows, page_indices_icols]
    # Values in page_indices are shard-local (0-based within each shard).
    # Offset to global page IDs for indexing into the full pages array.
    page_indices = page_indices + shard_ids * self.total_num_pages_per_shard

    # We must do a filter here to prevent unexpected page updates.
    safe_page_indices = jnp.where(
        jnp.arange(new_ragged_kv.capacity) < new_ragged_kv.total_length,
        page_indices,
        self.total_num_pages,
    )
    page_offsets = positions % self.page_size

    updated_pages = self.pages.at[safe_page_indices, page_offsets].set(
        new_ragged_kv.data, mode='drop'
    )
    return dataclasses.replace(self, pages=updated_pages)

  @jax.named_call
  def inject_chunk(
      self,
      slot_id: jax.typing.ArrayLike,
      pages_payload: jax.Array,
  ) -> Self:
    """Writes `num_shards * page_size` tokens of pre-formed KV for `slot_id`.

    The payload is token-major: shape `(num_shards * page_size,
    full_heads, kv_packing, head_dim)`. The leading dim is sharded
    along the seq axis, so each seq-shard sees its own `page_size`
    tokens (== one page) of KV.

    Implementation:
      1. `allocate(q_lens)` reserves `num_shards` pages for
         `slot_id` (one per shard; advances `kv_lens` and updates
         `page_indices`).
      2. A `shard_map` performs one `.at[].set()` (== DUS) per
         shard, writing that shard's page into its local `pages`
         buffer at the shard-local column the precondition below pins
         down. DUS has zero HLO temp scratch (vs. ~2x pages_bytes for
         the equivalent batched scatter), keeping the inject program's
         HBM footprint negligible.

    Pre-condition: `kv_lens[slot_id]` (pre-call) is a multiple of
    `num_shards * page_size` so every shard's freshly-allocated
    column lines up.

    Args:
      slot_id: scalar i32 — the batch slot to inject into.
      pages_payload: `[num_shards * page_size, full_heads, kv_packing,
        head_dim]` — the pre-paged KV bytes to insert, in token-major
        order. Sharded `(seq, head, None, None)` so each shard
        contributes only the `page_size` tokens (one page) it owns.

    Returns:
      Updated decode state.
    """
    page_shape = self.pages.shape[1:]  # (page_size, full_heads, kvp, hd)
    expected_shape = (self.num_shards * self.page_size,) + page_shape[1:]
    if pages_payload.shape != expected_shape:
      raise ValueError(
          f'pages_payload shape {pages_payload.shape} must equal '
          f'(num_shards * page_size, full_heads, kv_packing, head_dim) = '
          f'{expected_shape}.'
      )
    chunk_size = self.num_shards * self.page_size
    bs = self.kv_lens.shape[0]
    # Recycle any pages that have fallen outside the sliding window
    # *before* allocating this chunk, mirroring the decode path
    # (`self.release_for_window().allocate(q.lens)` in
    # `update_decode_state_and_compute_attn`). Doing it here keeps
    # `inject_chunk` self-contained: a cached prefix longer than the
    # window can be injected chunk-by-chunk without the page pool ever
    # overflowing, and callers don't have to remember to release. It
    # also compacts `page_indices`, so `kv_lens`, `base_col`, and
    # `allocate` below all operate on the same (post-release) layout.
    # No-op when `window_size is None` (all-global model).
    released = self.release_for_window()
    # Round `kv_lens` up to the next chunk boundary and append one chunk:
    # the chunk just injected ends at `(kv_lens // chunk_size + 1) *
    # chunk_size`. (When `kv_lens` is already chunk-aligned — the
    # precondition — this is exactly `kv_lens + chunk_size`.)
    after_len = (released.kv_lens[slot_id] // chunk_size + 1) * chunk_size
    num_inject_tokens = after_len - released.kv_lens[slot_id]
    q_lens = (
        jnp.zeros(bs, dtype=jnp.int32).at[slot_id].set(num_inject_tokens)
    )
    new_state = released.allocate(q_lens)
    # Exploit the precondition that the post-release `kv_lens[slot_id]`
    # is a multiple of `num_shards * page_size`: `allocate` therefore
    # wrote into exactly one local column on each shard — the SAME
    # column index on every shard, namely `base_col`. Each shard's
    # DUS target is the shard-local page id at that column, read
    # directly from the (post-release, compacted) `page_indices` with
    # no global-id math.
    base_col = released.kv_lens[slot_id] // chunk_size
    pages_spec = js.PartitionSpec(
        self.seq_partition, None, self.head_partition, None, None
    )
    # The payload's leading dim (num_shards * page_size) is sharded
    # along seq; each shard sees page_size tokens. No explicit page dim
    # in the user-facing shape; we acquire it after entering the shard
    # by reshape.
    payload_spec = js.PartitionSpec(
        self.seq_partition, self.head_partition, None, None
    )

    @jax.shard_map(
        mesh=js.get_abstract_mesh(),
        in_specs=(
            pages_spec,           # pages (full sharded layout)
            payload_spec,         # pages_payload (page_size tokens per shard)
            js.PartitionSpec(),   # page_indices (replicated)
            js.PartitionSpec(),   # base_col (replicated scalar)
        ),
        out_specs=pages_spec,
        check_vma=False,
    )
    def _scatter_local(local_pages, local_payload, page_indices, base_col):
      # `local_payload` has shape (page_size, full_heads, kvp, hd) —
      # exactly one page's worth, matching `local_pages`' per-page
      # shape (which is `local_pages.shape[1:]`).
      # `.at[].set()` with a single dynamic index lowers to DUS —
      # zero HLO temp scratch.
      local_pid = page_indices[slot_id, base_col]
      return local_pages.at[local_pid].set(local_payload)

    new_pages = _scatter_local(
        new_state.pages, pages_payload, new_state.page_indices, base_col
    )
    return dataclasses.replace(new_state, pages=new_pages)

  @jax.named_call
  def extract_chunk(
      self,
      slot_id: jax.typing.ArrayLike,
      chunk_idx: jax.typing.ArrayLike,
      position: jax.typing.ArrayLike,
  ) -> 'SnapshotChunkLeaf':
    """Reads the `chunk_idx`-th *logical* chunk for `slot_id`.

    The on-device read counterpart of :meth:`inject_chunk`; symmetric
    with it, this takes a *logical* (absolute) chunk index and maps it to
    the physical `page_indices` column internally, accounting for any
    sliding-window compaction.

    `release_for_window` evicts the oldest chunks and compacts
    `page_indices` left, reducing `kv_lens` to the windowed length. The
    currently-resident chunks occupy physical columns `[0, kv_lens //
    chunk_size)` and correspond to the *last* `kv_lens` logical tokens.
    Given `position` (the logical length, which this windowed state no
    longer carries), the number of evicted chunks is
    `(position - kv_lens) // chunk_size`, so:

        base_col = chunk_idx - (position - kv_lens) // chunk_size

    For an all-global layer `kv_lens == position`, so `base_col ==
    chunk_idx`. A request for an evicted chunk has `base_col < 0`; the
    returned entry has `available=False` in that case and the payload
    bytes are unspecified (the shard_map reads physical column
    `base_col` anyway, which is past the start of `page_indices` and
    yields stale page ids). Callers must check `available` before
    relying on the bytes, or call :meth:`SnapshotChunkLeaf.offload` which
    drops the bytes when `available=False`.

    Implementation: a `shard_map` performs one `page_indices` lookup
    + one `dynamic_slice` (= DUS-read) per shard, returning the page
    that shard owns in this chunk. The leading dim of the result
    (`num_shards * page_size`) is sharded along seq, so each shard
    contributes only its own `page_size` tokens.

    Args:
      slot_id: scalar i32 — the batch slot to read from.
      chunk_idx: scalar i32 — the *logical* (absolute) chunk index.
      position: scalar i32 — the slot's logical length (number of tokens
        seen so far), used to recover the evicted-chunk offset.

    Returns:
      A :class:`SnapshotChunkLeaf` with `payload` shape
      `[num_shards * page_size, full_heads, kv_packing, head_dim]`
      sharded `(seq, head, None, None)` — same shape and sharding as
      the `pages_payload` argument of :meth:`inject_chunk` — and
      `available` a scalar `bool` array that is `True` iff the
      requested chunk is still resident in this layer's window
      (`base_col >= 0`).
    """
    evicted_chunks = (
        jnp.asarray(position, dtype=jnp.int32) - self.kv_lens[slot_id]
    ) // self.chunk_size
    base_col = jnp.asarray(chunk_idx, dtype=jnp.int32) - evicted_chunks
    available = base_col >= 0
    pages_spec = js.PartitionSpec(
        self.seq_partition, None, self.head_partition, None, None
    )
    out_spec = js.PartitionSpec(
        self.seq_partition, self.head_partition, None, None
    )

    @jax.shard_map(
        mesh=js.get_abstract_mesh(),
        in_specs=(
            pages_spec,           # pages (full sharded layout)
            js.PartitionSpec(),   # page_indices (replicated)
            js.PartitionSpec(),   # base_col (replicated scalar)
        ),
        out_specs=out_spec,
        check_vma=False,
    )
    def _gather_local(local_pages, page_indices, base_col):
      # `local_pages[local_pid]` is the shard's single page for this
      # chunk, shape (page_size, full_heads, kvp, hd) — exactly the
      # local view of the (num_shards * page_size, fh, kvp, hd) output.
      local_pid = page_indices[slot_id, base_col]
      return local_pages[local_pid]

    payload = _gather_local(self.pages, self.page_indices, base_col)
    return SnapshotChunkLeaf(payload=payload, available=available)

  @property
  def page_manage_key(self) -> Hashable:
    return (self.total_num_pages, self.page_size, self.window_size)

  @jax.named_call
  def update_decode_state_and_compute_attn(
      self,
      q: RaggedArray,  # [max_num_tokens, num_q_heads, per_head_dim]
      k: jax.Array,  # [max_num_tokens, num_kv_heads, per_head_dim]
      v: jax.Array,  # [max_num_tokens, num_kv_heads, per_head_dim]
      soft_cap: float | None = None,
      mask_value: float | None = None,
      update_kv_cache: bool = True,
      page_manage_cache: MutableMapping[Hashable, Self] | None = None,
      num_kv_pages_per_block: int | None = None,
      num_queries_per_block: int | None = None,
  ) -> tuple[Self, jax.Array]:
    """Updates decode state.

    Args:
      q: ragged query buffer.
      k: key buffer.
      v: value buffer.
      soft_cap: attention logit soft cap.
      mask_value: causal-mask fill value.
      update_kv_cache: if True, write the new K/V into the paged cache.
      page_manage_cache: optional page-management cache shared across layers.
      num_kv_pages_per_block: OPTIONAL RPA-kernel KV-block size (in pages). This
        is a per-call kernel TILING hint, NOT cache state. ``None`` => autotune.
      num_queries_per_block: OPTIONAL RPA-kernel query block size
        (``bq_sz``/``bq_csz``) for the MIXED case. This is a per-call kernel
        TILING hint, NOT cache state. ``None`` => autotune. PREFILL passes the
        model's configured value (``config.rpa_block_q``); the DECODE stage
        passes ``1`` (one query token per slot). Changes only the query-block
        tiling, not the attention math.
    """
    if update_kv_cache:
      if (
          page_manage_cache is None
          or self.page_manage_key not in page_manage_cache
      ):
        decode_state = self.release_for_window().allocate(q.lens)
        if page_manage_cache is not None:
          page_manage_cache[self.page_manage_key] = decode_state
      else:
        manage_cache = page_manage_cache[self.page_manage_key]
        decode_state = dataclasses.replace(
            self,
            kv_lens=manage_cache.kv_lens,
            page_indices=manage_cache.page_indices,
            available_page_indices=manage_cache.available_page_indices,
            num_available_pages=manage_cache.num_available_pages,
        )
    else:
      decode_state = self

    rpa_kwargs = dict(
        sliding_window=self.window_size + 1 if self.window_size else None,
        update_kv_cache=update_kv_cache,
        soft_cap=soft_cap,
        k_scale=self.k_scale,
        v_scale=self.v_scale,
    )
    k = quant.cast_to_low_precision_float(k, self.dtype, self.k_scale)
    v = quant.cast_to_low_precision_float(v, self.dtype, self.v_scale)
    if mask_value is not None:
      rpa_kwargs['mask_value'] = mask_value

    if jax.devices()[0].platform == 'cpu':
      # Pallas RPA kernel is TPU-only; always use reference impl on CPU.
      num_seqs = jnp.sum(q.lens > 0)
      attn_output, updated_kv_cache, _ = rpa_kernel.ref_ragged_paged_attention(
          q.data,
          k,
          v,
          decode_state.pages,
          decode_state.kv_lens,
          decode_state.page_indices,
          jnp.cumulative_sum(q.lens, include_initial=True),
          jnp.array([0, 0, num_seqs], dtype=jnp.int32),
          **rpa_kwargs,
      )
      if update_kv_cache:
        decode_state = dataclasses.replace(decode_state, pages=updated_kv_cache)
      return decode_state, attn_output

    per_head_dim = q.data.shape[-1]

    distribution = jnp.array([0, 0, q.batch_size], dtype=jnp.int32)
    seq_partition_size = sharding.get_partition_size(self.seq_partition)
    logging.info('seq_partition_size: %d', seq_partition_size)
    if num_kv_pages_per_block is None or num_queries_per_block is None:
      head_partition_size = sharding.get_partition_size(self.head_partition)
      if q.capacity % seq_partition_size != 0:
        raise ValueError(
            f'{q.capacity=} must be divisible by {seq_partition_size=}'
        )
      num_kv_heads_per_shard = self.num_kv_heads // head_partition_size
      num_q_heads_per_shard = q.data.shape[1] // head_partition_size
      max_num_issue_tokens_per_shard = q.capacity // seq_partition_size
      autotuned_bkv, autotuned_bq = autotune_block_sizes(
          num_kv_heads=num_kv_heads_per_shard,
          num_q_heads=num_q_heads_per_shard,
          page_size=self.page_size,
          max_seq_len=self.max_seq_len,
          per_head_dim=per_head_dim,
          window_size=self.window_size,
          dtype=self.dtype,
          max_num_issue_tokens=max_num_issue_tokens_per_shard,
          num_shards=seq_partition_size,
      )
      if num_kv_pages_per_block is None:
        num_kv_pages_per_block = autotuned_bkv
        logging.info(
            'num_kv_pages_per_block=%d (autotuned)', num_kv_pages_per_block
        )
      else:
        logging.info(
            'num_kv_pages_per_block=%d (caller-supplied)',
            num_kv_pages_per_block,
        )
      if num_queries_per_block is None:
        num_queries_per_block = autotuned_bq
        logging.info(
            'num_queries_per_block=%d (autotuned)', num_queries_per_block
        )
      else:
        logging.info(
            'num_queries_per_block=%d (caller-supplied)',
            num_queries_per_block,
        )

    m_block_sizes = (
        num_queries_per_block,
        num_kv_pages_per_block * self.page_size,
        num_queries_per_block,
        num_kv_pages_per_block * self.page_size,
    )
    p_block_sizes = m_block_sizes
    d_block_sizes = (
        1,
        num_kv_pages_per_block * self.page_size,
        1,
        num_kv_pages_per_block * self.page_size,
    )
    logging.info('Autotuned m_block_sizes: %s', m_block_sizes)
    if seq_partition_size > 1:
      # --- Sharded page path: distribute KV cache pages across shards ---

      rpa_sharded_kwargs = rpa_kwargs.copy()
      rpa_sharded_kwargs['save_residuals'] = True

      _row_ids = q.row_ids

      @jax.shard_map(
          mesh=js.get_abstract_mesh(),
          in_specs=(
              js.PartitionSpec(None, self.head_partition, None),  # q
              js.PartitionSpec(None, self.head_partition, None),  # k
              js.PartitionSpec(None, self.head_partition, None),  # v
              js.PartitionSpec(
                  self.seq_partition, None, self.head_partition, None, None
              ),  # pages
              js.PartitionSpec(),  # kv_lens
              js.PartitionSpec(),  # page_indices (replicated, per-shard)
              js.PartitionSpec(),  # cu_q_lens
              js.PartitionSpec(),  # distribution
          ),
          out_specs=(
              js.PartitionSpec(
                  self.seq_partition, self.head_partition, None
              ),  # attn_output
              js.PartitionSpec(
                  self.seq_partition, None, self.head_partition, None, None
              ),  # pages
          ),
          check_vma=False,
      )
      def _sharded_rpa_fn(
          q_data,
          k_data,
          v_data,
          pages,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
      ):
        shard_id = 0
        if self.seq_partition is not None:
          shard_id = jax.lax.axis_index(self.seq_partition)

        # page_indices already contains only this shard's pages with
        # shard-local page IDs — no filtering or compaction needed.

        # Call kernel with shard-local metadata.
        attn_out, updated_pages, lse = rpa_kernel.ragged_paged_attention(
            q_data,
            k_data,
            v_data,
            pages,
            kv_lens,  # Global lengths; kernel computes local via shard_info
            page_indices,
            cu_q_lens,
            distribution,
            shard_info=jnp.array([shard_id, self.num_shards]),
            vmem_limit_bytes=50 * 1024 * 1024,
            d_block_sizes=d_block_sizes,
            p_block_sizes=p_block_sizes,
            m_block_sizes=m_block_sizes,
            **rpa_sharded_kwargs,
        )

        # Compute local KV lengths for masking tokens with no local KV.
        # Use the same formula as the kernel to determine per-shard tokens.
        page_size = pages.shape[1]
        num_full_pages = kv_lens // page_size
        tail_tokens = kv_lens % page_size
        full_pages_on_shard = jnp.maximum(
            0,
            (num_full_pages - shard_id + self.num_shards - 1)
            // self.num_shards,
        )
        local_kv_lens = full_pages_on_shard * page_size + jnp.where(
            (num_full_pages % self.num_shards) == shard_id, tail_tokens, 0
        )

        # Mask LSE for tokens whose sequences have no local KV.
        token_seq_ids = _row_ids
        token_has_local_kv = local_kv_lens[token_seq_ids] > 0
        lse_f32 = lse.astype(jnp.float32)
        lse_f32 = jnp.where(token_has_local_kv[:, None], lse_f32, -jnp.inf)
        attn_out = jnp.where(token_has_local_kv[:, None, None], attn_out, 0.0)

        # Numerically stable cross-shard attention accumulation.
        # lse_f32: (max_tokens, q_heads), attn_out: (max_tokens, q_heads, dim)
        max_lse = jax.lax.pmax(lse_f32, axis_name=self.seq_partition)
        w = jnp.exp(lse_f32 - max_lse)  # (max_tokens, q_heads)
        attn_f32 = attn_out.astype(jnp.float32)
        # Broadcast w to (max_tokens, q_heads, 1) for element-wise multiply.
        weighted_out = attn_f32 * w[:, :, None]
        sum_weighted = jax.lax.psum_scatter(
            weighted_out,
            axis_name=self.seq_partition,
            scatter_dimension=0,
            tiled=True,
        )
        sum_w = jax.lax.psum_scatter(
            w[:, :, None],
            axis_name=self.seq_partition,
            scatter_dimension=0,
            tiled=True,
        )
        # In theory, sum_w should always be greater than 0. Just in case, add a
        # small epsilon to avoid division by zero.
        sum_w = jnp.maximum(sum_w, 1e-9)
        final_out = (sum_weighted / sum_w).astype(attn_out.dtype)
        return final_out, updated_pages

      attn_output, updated_pages = _sharded_rpa_fn(
          q.data,
          k,
          v,
          decode_state.pages,
          decode_state.kv_lens,
          decode_state.page_indices,
          q.row_starts_with_end,
          distribution,
      )
    else:
      # --- Original non-sharded path ---
      rpa_fn = jax.shard_map(
          functools.partial(
              rpa_kernel.ragged_paged_attention,
              # vmem_limit_bytes=50 * 1024 * 1024,
              d_block_sizes=d_block_sizes,
              p_block_sizes=p_block_sizes,
              m_block_sizes=m_block_sizes,
              **rpa_kwargs,
          ),
          mesh=js.get_abstract_mesh(),
          in_specs=(
              js.PartitionSpec(None, self.head_partition, None),  # q
              js.PartitionSpec(None, self.head_partition, None),  # k
              js.PartitionSpec(None, self.head_partition, None),  # v
              js.PartitionSpec(
                  None, None, self.head_partition, None, None
              ),  # pages
              js.PartitionSpec(),  # kv_lens
              js.PartitionSpec(),  # page_indices
              js.PartitionSpec(),  # cu_q_lens
              js.PartitionSpec(),  # distribution
          ),
          out_specs=(
              js.PartitionSpec(None, self.head_partition, None),  # attn_out
              js.PartitionSpec(
                  None, None, self.head_partition, None, None
              ),  # pages
              None,
          ),
          check_vma=False,
      )
      attn_output, updated_pages, _ = rpa_fn(
          q.data,
          k,
          v,
          decode_state.pages,
          decode_state.kv_lens,
          decode_state.page_indices,
          q.row_starts_with_end,
          distribution,
      )
      # Return in shape [max_num_tokens, num_q_heads, per_head_dim]
      attn_output = sharding.with_sharding_constraint(
          attn_output, (self.seq_partition, self.head_partition, None)
      )
    if update_kv_cache:
      decode_state = dataclasses.replace(decode_state, pages=updated_pages)

    # Return in shape [max_num_tokens, num_q_heads, per_head_dim]
    return decode_state, attn_output


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SnapshotChunkLeaf:
  """One chunk's KV payload for one DecodeState leaf, with availability flag.

  Returned by :meth:`DecodeState.extract_chunk` (one per leaf) and
  :meth:`SamplingState.extract_chunk` (a pytree, one leaf per
  DecodeState leaf). The :meth:`offload` / :meth:`onload` pair converts
  between this carrier form and the bare payload representation the
  host-RAM prefix cache stores.

  Fields
    payload: the chunk's KV bytes — a `jax.Array` of shape
      `[num_shards * page_size, full_heads, kv_packing, head_dim]`
      sharded `(seq, head, None, None)`. Always concrete-on-device
      when produced by `extract_chunk` (even when `available=False`,
      the bytes are garbage but the shape/sharding are real, which is
      what `offload` uses to build the abstract form).
    available: a scalar `bool` array. `True` iff the requested chunk
      is still resident in this layer's sliding window
      (`base_col >= 0` in :meth:`DecodeState.extract_chunk`).
      Aliased to `False` for chunks evicted by `release_for_window`.
  """

  payload: jax.Array
  available: jax.Array

  def offload(self) -> jax.Array | jax.ShapeDtypeStruct:
    """Returns the offloaded payload only (no `available` bit).

    * `available=True`  — moves `payload` HBM->host via
      `jax.device_put` to the same sharding but with `memory_kind`
      switched to host memory (`pinned_host`, falling back to
      `unpinned_host` on platforms without pinned host memory). The
      shard layout is preserved exactly, so every host copies only
      its own addressable shard — host-local, no collective op.
    * `available=False` — returns a `jax.ShapeDtypeStruct` describing
      `payload`'s shape / dtype / sharding (the bytes would be
      garbage — stale post-eviction HBM — so we don't pay to copy or
      store them). :meth:`onload` materialises an uninitialised
      buffer of this shape on the inject path.

    Host-side branching on `available` forces a device->host transfer
    of the (scalar) bool, so this must NOT be called inside a
    jit-trace.
    """
    if not bool(self.available):
      return jax.ShapeDtypeStruct(
          self.payload.shape,
          self.payload.dtype,
          sharding=self.payload.sharding,
      )
    src_sharding = self.payload.sharding
    try:
      host_sharding = src_sharding.with_memory_kind('pinned_host')
    except ValueError:
      host_sharding = src_sharding.with_memory_kind('unpinned_host')
    return jax.device_put(self.payload, host_sharding)

  @staticmethod
  def onload(payload: jax.Array | jax.ShapeDtypeStruct) -> jax.Array:
    """Returns the on-device payload (inverse of :meth:`offload`).

    Args:
      payload: The payload to onload. Can be either:
        * A host-resident `jax.Array`: moves it host->HBM via `jax.device_put`
          to the same sharding with `memory_kind` switched to `device`.
        * A `jax.ShapeDtypeStruct` (offload-time `available=False`):
          materializes a fresh `jax.lax.empty` buffer of the described
          shape/dtype, placed on the described sharding. The bytes are
          uninitialized; safe ONLY when the caller's lookup rule guarantees a
          real tail of `min_num_chunks_for_window` complete chunks follows, so
          `release_for_window` evicts these pages before they can feed into
          attention.
    """
    if isinstance(payload, jax.ShapeDtypeStruct):
      return jax.device_put(
          jax.lax.empty(payload.shape, dtype=payload.dtype),
          payload.sharding,
      )
    return jax.device_put(
        payload, payload.sharding.with_memory_kind('device')
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _StepState:
  step: jax.Array  # i32
  state: 'SamplingState'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplingState:
  """Sampling state for ragged paged attention."""

  prng_key: jax.Array
  decode_state: common.PyTree
  tokens: jax.Array  # i32[batch, global_max_seq_len]
  token_logprobs: jax.Array  # f32[batch, global_max_seq_len], [:, 0] is dummy
  token_scores: jax.Array  # f32[batch, global_max_seq_len], [:, 0] is dummy
  position: jax.Array  # i32[batch], must be >= 0
  input_lens: jax.Array  # i32[batch], bos counted
  max_decode_steps: jax.Array  # i32[batch]
  rank: jax.Array  # i32[batch], smaller number be processed first
  # TODO: Support per sequence eos_ids
  eos_ids: jax.Array  # i32[n_eos]
  max_total_num_tokens: int = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self):
    if not isinstance(self.tokens, jax.Array):
      # This is for jax compatibility check.
      return
    if self.max_total_num_tokens < self.max_seq_len - 1:
      raise ValueError(
          f'{self.max_total_num_tokens=} must be >= {self.max_seq_len - 1=}'
      )
    # TODO: verify if max_total_num_tokens is feasible for each decode
    # state.

  @classmethod
  def create(
      cls,
      max_total_num_tokens: int,
      eos_ids: jax.typing.ArrayLike,
      prng_key: jax.typing.ArrayLike,
      decode_state: common.PyTree,
  ) -> Self:
    """Creates a sampling state."""
    attrs = DecodeState.attrs_from_tree(
        decode_state, ['batch_size', 'max_seq_len']
    )
    batch_size = common.reduce_same(attrs['batch_size'])
    max_seq_len = common.reduce_same(attrs['max_seq_len'])
    return cls(
        prng_key=prng_key,  # pyrefly: ignore[bad-argument-type]
        decode_state=decode_state,
        tokens=jax.lax.empty((batch_size, max_seq_len), dtype=jnp.int32),
        token_logprobs=jax.lax.empty(
            (batch_size, max_seq_len), dtype=jnp.float32
        ),
        token_scores=jax.lax.empty(
            (batch_size, max_seq_len), dtype=jnp.float32
        ),
        position=jax.lax.empty((batch_size,), dtype=jnp.int32),
        max_decode_steps=jax.lax.empty((batch_size,), dtype=jnp.int32),
        input_lens=jnp.zeros(batch_size, dtype=jnp.int32),
        rank=jnp.zeros(batch_size, dtype=jnp.int32),
        eos_ids=eos_ids,  # pyrefly: ignore[bad-argument-type]
        max_total_num_tokens=max_total_num_tokens,
    )

  @property
  def batch_size(self) -> int:
    return self.tokens.shape[0]

  @property
  def max_seq_len(self) -> int:
    return self.tokens.shape[1]

  @functools.cached_property
  def num_shards(self) -> int:
    """Seq-partition shard count, shared across all DecodeState leaves."""
    return common.reduce_same(
        DecodeState.attrs_from_tree(self.decode_state, ['num_shards'])[
            'num_shards'
        ]
    )

  @functools.cached_property
  def page_size(self) -> int:
    """KV page size in tokens, shared across all DecodeState leaves."""
    return common.reduce_same(
        DecodeState.attrs_from_tree(self.decode_state, ['page_size'])[
            'page_size'
        ]
    )

  @functools.cached_property
  def chunk_size(self) -> int:
    """Inject / extract granularity in tokens (`num_shards * page_size`).

    Shared across all DecodeState leaves: each leaf reduces the same
    `num_shards` and `page_size` via :func:`common.reduce_same`, so
    this product is well-defined for the whole sampling state.
    """
    return self.num_shards * self.page_size

  @functools.cached_property
  def min_num_chunks_for_window(self) -> int:
    """Largest `DecodeState.min_num_chunks_for_window` across all leaves.

    Reduces with `max` (not `reduce_same`): different layers may have
    different `window_size`s in a mixed model (e.g. global + windowed),
    and what matters for the prefix-cache's lookup rule is the
    *deepest* windowed tail any layer needs to be fully covered. A
    layer with no sliding window contributes `0` to the max.
    Returns `0` if every layer is global (no windowed coverage
    requirement).
    """
    per_leaf = DecodeState.attrs_from_tree(
        self.decode_state, ['min_num_chunks_for_window']
    )['min_num_chunks_for_window']
    return max(per_leaf) if per_leaf else 0

  @functools.cached_property
  def is_pad_seq(self) -> jax.Array:
    """This sequence is a padding sequence, in [batch, 1]."""
    return self.input_lens == 0

  def desired_issue_lens(self, prefill: bool = False) -> jax.Array:
    """Desired per-slot issue lens for the next pass."""
    attrs = DecodeState.attrs_from_tree(
        self.decode_state, ['max_available_kv_lens']
    )
    max_available_kv_lens = jnp.min(
        jnp.array(attrs['max_available_kv_lens']), axis=0
    )
    if prefill:
      remaining = jnp.maximum(self.input_lens - 1 - self.position, 0)
    else:
      remaining = jnp.maximum(
          self.input_lens
          - jnp.astype(self.max_decode_steps <= 0, jnp.int32)
          - self.position,
          1,
      )
    return jnp.where(
        self.has_ended, 0, jnp.minimum(max_available_kv_lens, remaining)
    )

  @functools.cached_property
  def is_prefilling(self) -> jax.Array:
    return (~self.has_ended) & (self.position < self.input_lens - 1)

  @functools.cached_property
  def any_prefilling(self) -> jax.Array:
    return jnp.any(self.is_prefilling)

  @functools.cached_property
  def rank_indices(self) -> jax.Array:
    inner_rank = jnp.where(
        self.is_pad_seq,
        2 * self.batch_size,
        jnp.where(self.is_prefilling, self.batch_size + self.rank, self.rank),
    )
    return jnp.argsort(inner_rank)

  @functools.cached_property
  def max_rank(self) -> jax.Array:
    return jnp.max(self.rank, where=~self.is_pad_seq, initial=-1)

  @functools.cached_property
  def rank_inv_indices(self) -> jax.Array:
    return jnp.argsort(self.rank_indices)

  @functools.cached_property
  def num_available_slots(self) -> jax.Array:
    """Returns the number of available slots."""
    return jnp.sum(self.is_pad_seq)

  def push(
      self, input_tokens: jax.typing.ArrayLike, n: int, max_decode_steps: int
  ) -> tuple[Self, jax.Array]:
    """Pushes new input tokens."""
    input_tokens = jnp.asarray(input_tokens)
    if len(input_tokens.shape) != 1 or input_tokens.dtype != jnp.int32:
      raise ValueError(
          f'tokens must be 1d int32: {input_tokens.shape=},'
          f' {input_tokens.dtype=}'
      )
    if input_tokens.shape[0] > self.max_seq_len:
      raise ValueError(
          f'{input_tokens.shape[0]=} must be less or equal to'
          f' {self.max_seq_len=}'
      )
    index = jnp.flatnonzero(self.is_pad_seq, size=1, fill_value=self.batch_size)
    index = jnp.reshape(index, ())
    updated_tokens = self.tokens.at[index, : input_tokens.shape[0]].set(
        input_tokens
    )
    updated_position = self.position.at[index].set(0)
    updated_input_lens = self.input_lens.at[index].set(n)
    updated_rank = self.rank.at[index].set(self.max_rank + 1)
    updated_max_decode_steps = self.max_decode_steps.at[index].set(
        max_decode_steps
    )
    return (
        dataclasses.replace(
            self,
            tokens=updated_tokens,
            position=updated_position,
            input_lens=updated_input_lens,
            rank=updated_rank,
            max_decode_steps=updated_max_decode_steps,
        ),
        index,
    )

  def get(
      self, mask: jax.typing.ArrayLike
  ) -> Sequence[Mapping[str, np.typing.ArrayLike]]:
    """Returns the tokens, logprobs, scores, and truncation flag per slot.

    `truncated` is True iff this slot stopped without sampling an EOS token
    (i.e. `has_ended` fired via `lens >= max_seq_len` or
    `lens - input_lens >= max_decode_steps` rather than `reached_eos`).
    Derived per-slot from `~self.reached_eos`, which is shape `[batch]`.

    Args:
      mask: Boolean per-slot mask selecting which slots to materialise. Shape
        `[batch]`; typically `completed_mask` from the batcher.
    """
    indices = np.flatnonzero(mask)
    lens = np.where(
        np.asarray(self.is_pad_seq), 0, np.asarray(self.position) + 1
    )
    input_lens = np.asarray(self.input_lens)
    reached_eos = np.asarray(self.reached_eos)
    results = []
    for index in indices:
      results.append(
          dict(
              index=int(index),
              input_len=int(input_lens[index]),
              tokens=np.asarray(self.tokens[index])[: lens[index]],
              logprobs=np.asarray(self.token_logprobs[index])[: lens[index]],
              scores=np.asarray(self.token_scores[index])[: lens[index]],
              truncated=not bool(reached_eos[index]),
          )
      )
    return results

  def release(self, should_release: jax.Array) -> Self:
    """Pops and releases the sampling state."""
    sampling_state = dataclasses.replace(
        self,
        decode_state=jax.tree_util.tree_map(
            lambda ds: ds.release(should_release),
            self.decode_state,
            is_leaf=lambda x: isinstance(x, DecodeState),
        ),
        input_lens=jnp.where(should_release, 0, self.input_lens),
    )
    return dataclasses.replace(
        sampling_state, rank=sampling_state.rank_inv_indices
    )

  def extract_chunk(self, slot_id: int, chunk_idx: int) -> common.PyTree:
    """Extracts the `chunk_idx`-th *logical* chunk of KV for `slot_id`.

    Companion read for :meth:`DecodeState.inject_chunk` / the
    page-batcher inject path. The chunk size is fixed at
    `num_shards * page_size` tokens — the natural inject granularity.

    `chunk_idx` is a *logical* (absolute) chunk index from the sequence
    start. Each DecodeState leaf maps it to its own physical
    `page_indices` column (accounting for sliding-window eviction) inside
    :meth:`DecodeState.extract_chunk`, using `self.position[slot_id]` (the
    logical length) which the windowed `DecodeState` no longer carries.
    A leaf reports `available=False` for chunks evicted out of its
    sliding window; callers MUST treat the corresponding payload bytes
    as unspecified (or call :meth:`SnapshotChunkLeaf.offload` which drops
    them).

    The caller is expected to only request chunks whose KV has been
    computed (`chunk_idx < position // chunk_size`); requesting a
    chunk past that frontier returns uninitialised bytes regardless of
    `available`.

    Args:
      slot_id: scalar i32 — the batch slot.
      chunk_idx: scalar i32 — the *logical* chunk index.

    Returns:
      A pytree with the same structure as `self.decode_state`, with
      each DecodeState leaf replaced by the leaf's
      :meth:`DecodeState.extract_chunk` result (a
      :class:`SnapshotChunkLeaf`). No top-level reduction is performed;
      callers can AND-reduce the per-leaf `available`s host-side to
      decide whether to cache the chunk (see
      :meth:`SnapshotChunkLeaf.offload`). The corresponding input tokens
      are not returned: callers already hold them host-side and use
      them as the cache key.

    Raises:
      TypeError: if any leaf in `decode_state` is not a `DecodeState`.
    """
    position = self.position[slot_id]
    is_leaf = lambda x: isinstance(x, DecodeState)

    def _extract(leaf):
      if not isinstance(leaf, DecodeState):
        raise TypeError(
            f'decode_state leaves must be DecodeState; got {type(leaf)}'
        )
      return leaf.extract_chunk(slot_id, chunk_idx, position)

    return jax.tree_util.tree_map(_extract, self.decode_state, is_leaf=is_leaf)

  @functools.cached_property
  def num_used_tokens(self) -> jax.Array:
    """Returns the number of used tokens."""
    return jnp.sum(self.position, where=~self.is_pad_seq, initial=0)

  @jax.jit(static_argnames=('capacity', 'prefill'))
  @jax.named_call
  def issue_lens(self, capacity: int, prefill: bool = False) -> jax.Array:
    """Clamps the `prefill`-selected desired issue lens to `capacity`."""
    desired_issue_lens = self.desired_issue_lens(prefill)
    sorted_desired_issue_lens = desired_issue_lens[self.rank_indices]

    cum_sorted_issue_lens = jnp.minimum(
        jnp.cumulative_sum(sorted_desired_issue_lens, include_initial=True),
        capacity,
    )

    # 2. Max total num tokens constraint, guarantee first seq can be complete.
    # TODO: This is not guaranteed anymore, need to fix.
    if self.batch_size > 1:
      seq0_len = self.position[self.rank_indices[0]]

      seq0_remaining_capacity = jnp.maximum(
          self.max_total_num_tokens - self.num_used_tokens, 0
      )
      other_remaining_capacity = jnp.maximum(
          self.max_total_num_tokens
          - (self.num_used_tokens - seq0_len + self.max_seq_len - 1),
          0,
      )
      cum_sorted_issue_lens = jnp.minimum(
          cum_sorted_issue_lens,
          jnp.minimum(cum_sorted_issue_lens[1], seq0_remaining_capacity)
          + other_remaining_capacity,
      )

    sorted_issue_lens = cum_sorted_issue_lens[1:] - cum_sorted_issue_lens[:-1]
    return sorted_issue_lens[self.rank_inv_indices]

  @jax.named_call
  def ragged_issue_tokens(
      self, capacity: int, prefill: bool = False
  ) -> common.RaggedArray:
    """Returns the ragged issue tokens."""
    # follows priority, and do not issue when oversubscriped.
    issue_lens = self.issue_lens(capacity, prefill=prefill)
    ragged_buffer = common.RaggedArray(
        data=jax.lax.empty((capacity,), dtype=self.tokens.dtype),
        lens=issue_lens,
    )
    irows = ragged_buffer.row_ids
    icols = self.position[ragged_buffer.row_ids] + ragged_buffer.intra_offset
    return dataclasses.replace(ragged_buffer, data=self.tokens[irows, icols])

  @jax.named_call
  def update_with_ragged_output(
      self, ragged_output_tokens: common.RaggedArray, **kwargs: jax.Array
  ) -> Self:
    """Updates the sampling state with the ragged output tokens."""
    assert self.batch_size == ragged_output_tokens.batch_size
    updated_position = self.position + ragged_output_tokens.lens

    safe_row_ids = jnp.where(
        jnp.arange(ragged_output_tokens.capacity)
        < ragged_output_tokens.total_length,
        ragged_output_tokens.row_ids,
        ragged_output_tokens.batch_size,
    )
    intra_offset = (
        self.position[ragged_output_tokens.row_ids]
        + ragged_output_tokens.intra_offset
        + 1
    )

    updated_tokens = self.tokens.at[safe_row_ids, intra_offset].set(
        ragged_output_tokens.data, mode='drop'
    )

    extra_replacements = {}
    if (token_logprobs := kwargs.get('token_logprobs')) is not None:
      extra_replacements['token_logprobs'] = self.token_logprobs.at[
          safe_row_ids, intra_offset
      ].set(token_logprobs, mode='drop')
    if (token_scores := kwargs.get('token_scores')) is not None:
      extra_replacements['token_scores'] = self.token_scores.at[
          safe_row_ids, intra_offset
      ].set(token_scores, mode='drop')

    return dataclasses.replace(
        self,
        position=updated_position,
        tokens=updated_tokens,
        **extra_replacements,
    )

  @functools.cached_property
  def current_tokens(self) -> jax.Array:
    return self.tokens[jnp.arange(self.batch_size), self.position]

  @functools.cached_property
  def reached_eos(self) -> jax.Array:
    """This position is output and eos, in [batch]."""
    # eos_ids: [n_eos]
    # current_tokens: [batch] -> [batch, 1]
    # output: [batch, n_eos] -> [batch]
    return (self.position >= self.input_lens) & jnp.any(
        jnp.expand_dims(self.current_tokens, axis=-1) == self.eos_ids,
        axis=-1,
    )

  @functools.cached_property
  def lens(self) -> jax.Array:
    return jnp.where(self.is_pad_seq, 0, self.position + 1)

  @functools.cached_property
  def has_ended(self) -> jax.Array:
    """Returns whether each sequence in the batch is done with generation."""
    return (
        self.is_pad_seq
        | (self.lens >= self.max_seq_len)
        | (self.lens - self.input_lens >= self.max_decode_steps)
        | self.reached_eos
    )

  @functools.cached_property
  def is_continuable(self) -> jax.Array:
    seq0_len = self.position[self.rank_indices[0]]
    return jnp.any(~self.has_ended) & (
        self.max_total_num_tokens - self.num_used_tokens + seq0_len
        >= self.max_seq_len - 1
    )

  def mixed_step(
      self,
      forward_fn: Callable[..., jax.Array],
      params: common.PyTree,
      extra_inputs: common.PyTree = None,
      max_num_issue_tokens: int = 128,
      temperature: float = 1.0,
      top_k: int = -1,
      top_p: float = 1.0,
      scoring_temperature: float = 1.0,
      scoring_top_k: int = -1,
      scoring_top_p: float = 1.0,
      prefill: bool = False,
      skip_logits: bool = False,
  ) -> Self:
    """Executes a mixed step (prefill+decode)."""
    if skip_logits and not prefill:
      raise ValueError('skip_logits is only valid for prefill')
    # User should guarantee self.is_continuable is True.
    # logits: [batch_size, 1, vocab_size]
    ragged_issue_tokens = self.ragged_issue_tokens(
        max_num_issue_tokens, prefill=prefill
    )

    # segment_ids == 0 means padding.
    segment_ids = jnp.where(
        jnp.arange(ragged_issue_tokens.capacity)
        < ragged_issue_tokens.total_length,
        ragged_issue_tokens.row_ids + 1,
        0,
    )
    segment_positions = (
        self.position[ragged_issue_tokens.row_ids]
        + ragged_issue_tokens.intra_offset
    )
    if extra_inputs is None:
      extra_inputs = {}
    extra_inputs['lens'] = ragged_issue_tokens.lens  # pyrefly: ignore[unsupported-operation]
    extra_inputs['page_manage_cache'] = {}  # pyrefly: ignore[unsupported-operation]

    logits, extra_output = forward_fn(
        params,
        einops.rearrange(ragged_issue_tokens.data, 'l -> 1 l'),
        segment_ids=einops.rearrange(segment_ids, 'l -> 1 l'),
        segment_positions=einops.rearrange(segment_positions, 'l -> 1 l'),
        extra_inputs=extra_inputs,
        decode_state=self.decode_state,
    )

    next_tokens = self.tokens[
        ragged_issue_tokens.row_ids, segment_positions + 1
    ]
    if skip_logits:
      prng_key = self.prng_key
      output_tokens = next_tokens  # [capacity]
      output_logprobs = jnp.zeros_like(next_tokens, dtype=jnp.float32)
      output_scores = jnp.zeros_like(next_tokens, dtype=jnp.float32)
    else:
      # Split the prng only on a sampling (decode/mixed) pass.
      prng_key, key = jax.random.split(self.prng_key, 2)
      # output_tokens: [1, capacity], output_logprobs: [1, capacity]
      output_tokens, output_logprobs = sampling_lib.sample_from_logits(
          key, logits, temperature=temperature, top_k=top_k, top_p=top_p
      )
      output_tokens = jnp.where(
          segment_positions + 1 >= self.input_lens[ragged_issue_tokens.row_ids],
          output_tokens,
          next_tokens,
      )
      output_scores = sampling_lib.compute_log_likelihood(
          logits,
          output_tokens,
          temperature=scoring_temperature,
          top_k=scoring_top_k,
          top_p=scoring_top_p,
      )
      output_tokens = einops.rearrange(output_tokens, '1 l -> l')
      output_logprobs = einops.rearrange(output_logprobs, '1 l -> l')
      output_scores = einops.rearrange(output_scores, '1 l -> l')

    sampling_state = self.update_with_ragged_output(
        RaggedArray(output_tokens, ragged_issue_tokens.lens),
        token_logprobs=output_logprobs,
        token_scores=output_scores,
    )

    return dataclasses.replace(
        sampling_state,
        prng_key=prng_key,
        decode_state=extra_output['decode_state'],
    )

  def continue_decode(
      self,
      forward_fn: Callable[..., tuple[jax.Array, common.PyTree]],
      until_fn: Callable[[Self], jax.Array],
      params: common.PyTree,
      extra_inputs: common.PyTree = None,
      max_num_issue_tokens: int = 1024,
      temperature: float = 1.0,
      top_k: int = -1,
      top_p: float = 1.0,
      scoring_temperature: float = 1.0,
      scoring_top_k: int = -1,
      scoring_top_p: float = 1.0,
      intermediate_steps: int = np.iinfo(np.int32).max // 2,
  ) -> Self:
    """Continues decoding."""

    final_sampling_state = jax.lax.while_loop(
        lambda step_state: step_state.state.is_continuable
        & ~until_fn(step_state.state)
        & (step_state.step < intermediate_steps),
        lambda step_state: _StepState(
            step_state.step + 1,
            step_state.state.mixed_step(
                forward_fn,
                params,
                extra_inputs,
                max_num_issue_tokens,
                temperature,
                top_k,
                top_p,
                scoring_temperature,
                scoring_top_k,
                scoring_top_p,
            ),
        ),
        _StepState(step=jnp.array(0), state=self),
    )
    return final_sampling_state.state

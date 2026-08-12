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

  def chunk_written_mask(
      self,
      chunk_idx: jax.typing.ArrayLike,
      position: jax.typing.ArrayLike,
  ) -> jax.Array:
    """Which tokens of chunk `chunk_idx` hold real KV at `position`.

    Args:
      chunk_idx: scalar i32 -- the *logical* (absolute) chunk index.
      position: scalar i32 -- the slot's logical length. MUST be a pass
        boundary; mid-pass the paged cache is not in a consistent state at all.

    Returns:
      `bool[chunk_size]`, token-major, aligned with the payload's axis 0.
    """
    chunk_size = self.chunk_size
    chunk_idx = jnp.asarray(chunk_idx, dtype=jnp.int32)
    position = jnp.asarray(position, dtype=jnp.int32)
    chunk_start = chunk_idx * chunk_size
    token_pos = chunk_start + jnp.arange(chunk_size, dtype=jnp.int32)
    hi = jnp.minimum(chunk_start + chunk_size, position)
    if self.window_size is None:
      lo = chunk_start
    else:
      lo = jnp.maximum(chunk_start, position - self.window_size)
    return jnp.logical_and(token_pos >= lo, token_pos < hi)

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
      start: jax.typing.ArrayLike | None = None,
      end: jax.typing.ArrayLike | None = None,
  ) -> Self:
    """Writes tokens `[start, end)` of one chunk of pre-formed KV for `slot_id`.

    THE RANGE IS THE WHOLE CONTRACT, and it is in ABSOLUTE token positions:
    `start` is where the slot stands, `end` is where it lands, and the
    payload is the chunk `start` falls in. Everything else is read off it:

      * the slot gains exactly `end - start` tokens -- no `kv_lens` remainder
        to reason about, and no difference between an append and a rewrite;
      * the page written is `start`'s chunk, translated from the logical grid
        to the physical one by whatever the window has evicted -- the same
        translation :meth:`extract_chunk` does in reverse;
      * only the tokens INSIDE the range are written. What lies below `start`
        in that page is what the slot computed ITSELF, and it stays: a
        mid-chunk restore no longer overwrites live KV with a cached copy of
        it, which also means a payload with holes down there (a windowed
        capture) can no longer damage anything.

    Args:
      slot_id: scalar i32 — the batch slot to inject into.
      pages_payload: `[num_shards * page_size, full_heads, kv_packing,
        head_dim]` — the pre-paged KV bytes to insert, in token-major order
        over the WHOLE chunk `start` falls in. Sharded `(seq, head, None,
        None)` so each shard contributes only the `page_size` tokens (one
        page) it owns.
      start: scalar i32 — first token position to write; MUST be the slot's
        logical length, since that is what makes it the boundary between what
        the slot has and what the payload brings. `None` (default) means the
        slot's current frontier, `kv_lens`.
      end: scalar i32 — one past the last token position to write, and the
        slot's new length. Must lie inside `start`'s chunk, since that is the
        chunk the payload holds. `None` (default) means that chunk's end.

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
    released = self.release_for_window()
    kv_len = released.kv_lens[slot_id]
    start_int = (
        kv_len if start is None else jnp.asarray(start, dtype=jnp.int32)
    )
    chunk_start = (start_int // chunk_size) * chunk_size
    end_int = (
        chunk_start + chunk_size
        if end is None
        else jnp.asarray(end, dtype=jnp.int32)
    )
    q_lens = jnp.zeros(bs, dtype=jnp.int32).at[slot_id].set(end_int - start_int)
    new_state = released.allocate(q_lens)
    # LOGICAL -> PHYSICAL. `start_int` counts every token the slot has ever
    # held; `kv_len` counts the ones still resident. The difference is what
    # `release_for_window` has dropped -- always whole chunks -- so taking it
    # off the logical chunk start gives the column this page lives in.
    # (:meth:`extract_chunk` maps a logical chunk index the same way.)
    base_col = (chunk_start - (start_int - kv_len)) // chunk_size
    # Where the range falls INSIDE the chunk: offsets `[lo, hi)` of the
    # payload's token axis, which is the chunk's own grid.
    lo = start_int - chunk_start
    hi = end_int - chunk_start
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
            js.PartitionSpec(),   # lo (replicated scalar)
            js.PartitionSpec(),   # hi (replicated scalar)
        ),
        out_specs=pages_spec,
        check_vma=False,
    )
    def _scatter_local(
        local_pages, local_payload, page_indices, base_col, lo, hi
    ):
      # `local_payload` has shape (page_size, full_heads, kvp, hd) —
      # exactly one page's worth, matching `local_pages`' per-page
      # shape (which is `local_pages.shape[1:]`).
      local_pid = page_indices[slot_id, base_col]
      # WHICH OF THIS SHARD'S TOKENS THE RANGE COVERS. The chunk is split
      # token-major, so shard `s` holds its tokens `[s * page_size, (s + 1) *
      # page_size)`; a shard entirely outside `[lo, hi)` keeps its page as it
      # stands. Selecting rather than slicing keeps the shapes static (both
      # bounds are traced), and costs one page read against the DUS.
      shard_id = (
          0
          if self.seq_partition is None
          else jax.lax.axis_index(self.seq_partition)
      )
      offsets = shard_id * self.page_size + jnp.arange(
          self.page_size, dtype=jnp.int32
      )
      inside = jnp.logical_and(offsets >= lo, offsets < hi)
      merged = jnp.where(
          inside.reshape((-1,) + (1,) * (local_payload.ndim - 1)),
          local_payload,
          local_pages[local_pid],
      )
      return local_pages.at[local_pid].set(merged)

    new_pages = _scatter_local(
        new_state.pages,
        pages_payload,
        new_state.page_indices,
        base_col,
        lo,
        hi,
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
    chunk_idx`. A request for an evicted chunk reads physical column
    `base_col < 0`, which is past the start of `page_indices` and yields
    stale page ids, so its payload bytes are unspecified -- but its mask is
    empty, so nothing points at them.

    **Validity (windowed layers).** Residency is necessary but not
    sufficient: the kernel only writes newly issued KV back for the BKV
    blocks the LAST query block visits, so a pass issuing more than
    `window_size` tokens leaves HOLES below `position - window_size`. What
    is real is therefore the per-token mask of :meth:`chunk_written_mask`,
    and `position` MUST be a pass boundary -- mid-pass the paged cache is
    not in a consistent state at all.

    Implementation: a `shard_map` performs one `page_indices` lookup
    + one `dynamic_slice` (= DUS-read) per shard, returning the page
    that shard owns in this chunk. The leading dim of the result
    (`num_shards * page_size`) is sharded along seq, so each shard
    contributes only its own `page_size` tokens.

    Args:
      slot_id: scalar i32 — the batch slot to read from.
      chunk_idx: scalar i32 — the *logical* (absolute) chunk index.
      position: scalar i32 — the slot's logical length (number of tokens seen so
        far), used to recover the evicted-chunk offset.

    Returns:
      A :class:`SnapshotChunkLeaf` with `payload` shape
      `[num_shards * page_size, full_heads, kv_packing, head_dim]`
      sharded `(seq, head, None, None)` — same shape and sharding as
      the `pages_payload` argument of :meth:`inject_chunk` — and `written` the
      per-token mask of :meth:`chunk_written_mask`.
    """
    evicted_chunks = (
        jnp.asarray(position, dtype=jnp.int32) - self.kv_lens[slot_id]
    ) // self.chunk_size
    base_col = jnp.asarray(chunk_idx, dtype=jnp.int32) - evicted_chunks
    # No residency term here (`base_col >= 0`): `written` is purely
    # positional, and that is sound because the resident region always
    # CONTAINS the window -- `release_for_window` frees whole chunk-aligned
    # columns and never one inside it, so a non-empty mask implies the chunk
    # is still there. `ResidencyGuardTest` is what holds that up.
    written = self.chunk_written_mask(chunk_idx, position)
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
    return SnapshotChunkLeaf(payload=payload, written=written)

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
  """One chunk's KV for one DecodeState leaf, with a PER-TOKEN written mask.

  Returned by :meth:`DecodeState.extract_chunk` (one per leaf) and
  :meth:`SamplingState.extract_chunk` (a pytree, one leaf per DecodeState
  leaf). :meth:`offload` converts one leaf to the host-resident
  :class:`StoredChunkLeaf` form the prefix cache keeps;
  :func:`offload_chunk_tree`
  converts a whole tree of them. :meth:`onload` converts a stored payload back.

  Fields
    payload: the chunk's KV bytes — a `jax.Array` of shape
      `[num_shards * page_size, full_heads, kv_packing, head_dim]` sharded
      `(seq, head, None, None)`. Token-major: axis 0 lines up with `written`.
      Always concrete on device, even where `written` is `False` (the bytes are
      garbage there, but the shape / sharding are what `offload` needs).
    written: `bool[num_shards * page_size]` -- which tokens of this chunk hold
      real KV in `payload`.
  """

  payload: jax.Array
  written: jax.Array

  def offload(self) -> 'StoredChunkLeaf':
    """Returns a host-resident copy of this leaf (inverse of :meth:`onload`).

    * any written token — moves `payload` HBM->host via `jax.device_put` to the
      same sharding with `memory_kind` switched to host memory (`pinned_host`,
      falling back to `unpinned_host`). The shard layout is preserved exactly,
      so every host copies only its own addressable shard — host-local, no
      collective op.
    * no written token — stores a `jax.ShapeDtypeStruct` instead of the bytes
      (they are all garbage, so we do not pay to copy or keep them).
      :meth:`onload` materialises an uninitialised buffer of that shape.
    """
    written = np.asarray(self.written, dtype=bool)
    assert written.shape == (self.payload.shape[0],), (
        f'mask {written.shape} does not match the payload token axis'
        f' {self.payload.shape[0]}'
    )
    if not written.any():
      return StoredChunkLeaf(
          payload=jax.ShapeDtypeStruct(
              self.payload.shape,
              self.payload.dtype,
              sharding=self.payload.sharding,
          ),
          written=written,
      )
    src_sharding = self.payload.sharding
    try:
      host_sharding = src_sharding.with_memory_kind('pinned_host')
    except ValueError:
      host_sharding = src_sharding.with_memory_kind('unpinned_host')
    return StoredChunkLeaf(
        payload=jax.device_put(self.payload, host_sharding), written=written
    )

  @staticmethod
  def onload(payload: jax.Array | jax.ShapeDtypeStruct) -> jax.Array:
    """Returns the on-device payload (inverse of :meth:`offload`).

    Args:
      payload: The payload to onload. Can be either: * A host-resident
        `jax.Array`: moves it host->HBM via `jax.device_put` to the same
        sharding with `memory_kind` switched to `device`. * A
        `jax.ShapeDtypeStruct` (nothing written at offload time): materializes a
        fresh `jax.lax.empty` buffer of the described shape/dtype, placed on the
        described sharding. The bytes are uninitialized; safe because a restore
        only ever resumes where the accumulated masks cover what the resuming
        query reads, and `release_for_window` evicts the rest before it can feed
        into attention.
    """
    if isinstance(payload, jax.ShapeDtypeStruct):
      return jax.device_put(
          jax.lax.empty(payload.shape, dtype=payload.dtype),
          payload.sharding,
      )
    return jax.device_put(
        payload, payload.sharding.with_memory_kind('device')
    )


def _merge_host_arrays(
    old: jax.Array, new: jax.Array, add: np.ndarray
) -> jax.Array:
  """Returns `where(add, new, old)` without leaving host memory.

  Both inputs are host-resident with the same (host `memory_kind`) sharding.
  Rebuilding shard by shard keeps every byte on the host: a shard with nothing
  to add is REUSED as-is, and a shard that does gain tokens is merged in numpy
  and put back into the same single-device host sharding. Measured on a TPU
  host at the production chunk shape, this costs ~1 ms per touched shard and
  ~0 for the rest; a device round-trip would be two collectives plus a
  compute.

  Args:
    old: the accumulated payload.
    new: the fresh capture.
    add: `bool[chunk_size]` -- tokens `new` contributes (already masked to what
      `old` was missing).

  Returns:
    A host-resident array with `old`'s shape, sharding and memory kind.
  """
  broadcast = (slice(None),) + (None,) * (old.ndim - 1)
  out_shards = []
  for shard_old, shard_new in zip(
      old.addressable_shards, new.addressable_shards
  ):
    token_slice = shard_old.index[0]
    lo = token_slice.start or 0
    hi = token_slice.stop if token_slice.stop is not None else old.shape[0]
    segment = add[lo:hi]
    if not segment.any():
      out_shards.append(shard_old.data)  # untouched: no copy at all
      continue
    merged = np.where(
        segment[broadcast],
        np.asarray(shard_new.data),
        np.asarray(shard_old.data),
    )
    out_shards.append(jax.device_put(merged, shard_old.data.sharding))
  return jax.make_array_from_single_device_arrays(
      old.shape, old.sharding, out_shards
  )


@dataclasses.dataclass(frozen=True)
class StoredChunkLeaf:
  """One chunk's KV for one leaf, host-resident, with its written mask.

  Deliberately carries NO window: the mask is an absolute per-token fact,
  while a window belongs to the model about to read it back
  (`test_a_stored_leaf_carries_no_window`).
  """

  payload: jax.Array | jax.ShapeDtypeStruct
  written: np.ndarray

  @property
  def nbytes(self) -> int:
    """Host RAM held by this leaf (a placeholder holds none)."""
    return self.payload.nbytes if isinstance(self.payload, jax.Array) else 0

  @property
  def is_full(self) -> bool:
    """Whether every token of the chunk is written."""
    return bool(self.written.all())

  def covers(self, lo: int, hi: int) -> bool:
    """Whether tokens `[lo, hi)` (chunk-relative) are all written.

    Args:
      lo: first token index into the chunk.
      hi: one past the last token index; `hi <= lo` is vacuously covered.

    Returns:
      Whether every token in the range is written.
    """
    if hi <= lo:
      return True
    return bool(self.written[lo:hi].all())

  def merge(self, new: Self) -> Self:
    """Unions two captures of the same chunk for the same leaf."""
    add = np.logical_and(new.written, np.logical_not(self.written))
    if not add.any():
      return self
    if not self.written.any():
      return new  # nothing to preserve: take the new capture whole
    merged_written = np.logical_or(self.written, new.written)
    if not isinstance(new.payload, jax.Array):
      raise TypeError(
          f'Expected new.payload to be a jax.Array when add.any() is True, got'
          f' {type(new.payload).__name__}'
      )
    if not isinstance(self.payload, jax.Array):
      raise TypeError(
          f'Expected self.payload to be a jax.Array when self.written.any() is'
          f' True, got {type(self.payload).__name__}'
      )
    return dataclasses.replace(
        self,
        payload=_merge_host_arrays(self.payload, new.payload, add),
        written=merged_written,
    )


def merge_chunk_trees(
    old_tree: common.PyTree, new_tree: common.PyTree
) -> common.PyTree:
  """Leaf-wise :meth:`StoredChunkLeaf.merge` over two stored chunk trees.

  Args:
    old_tree: the stored tree.
    new_tree: the freshly offloaded tree.

  Returns:
    The merged stored tree.
  """
  is_leaf = lambda x: isinstance(x, StoredChunkLeaf)
  old_leaves = jax.tree_util.tree_leaves(old_tree, is_leaf=is_leaf)
  new_leaves = jax.tree_util.tree_leaves(new_tree, is_leaf=is_leaf)
  merged_leaves = [
      old_leaf.merge(new_leaf)
      for old_leaf, new_leaf in zip(old_leaves, new_leaves, strict=True)
  ]
  structure = jax.tree_util.tree_structure(old_tree, is_leaf=is_leaf)
  return jax.tree_util.tree_unflatten(structure, merged_leaves)


def stored_tree_nbytes(stored_tree: common.PyTree) -> int:
  """Returns the host RAM a stored chunk tree holds."""
  is_leaf = lambda x: isinstance(x, StoredChunkLeaf)
  return sum(
      leaf.nbytes
      for leaf in jax.tree_util.tree_leaves(stored_tree, is_leaf=is_leaf)
  )


def onload_chunk_tree(stored_tree: common.PyTree) -> common.PyTree:
  """Returns the on-device payload tree for `inject_chunk`."""
  is_leaf = lambda x: isinstance(x, StoredChunkLeaf)
  return jax.tree_util.tree_map(
      lambda leaf: SnapshotChunkLeaf.onload(leaf.payload),
      stored_tree,
      is_leaf=is_leaf,
  )


def offload_chunk_tree(leaf_tree: common.PyTree) -> common.PyTree:
  """Offloads a whole `SnapshotChunkLeaf` tree to host RAM.

  Args:
    leaf_tree: a pytree of :class:`SnapshotChunkLeaf` (HBM-resident), e.g. the
      result of :meth:`SamplingState.extract_chunk`.

  Returns:
    The same tree shape with each leaf replaced by a :class:`StoredChunkLeaf`.
  """
  is_leaf = lambda x: isinstance(x, SnapshotChunkLeaf)
  return jax.tree_util.tree_map(
      lambda leaf: leaf.offload(),
      leaf_tree,
      is_leaf=is_leaf,
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

    PROMPT REGION OF `logprobs` / `scores`. Only the DECODED positions
    (`>= input_len`) are meaningful. Prefill runs with `skip_logits=True` and
    writes zeros below that; a prefix-cache restore
    (:meth:`inject_chunk`) writes nothing at all, so with caching on those
    positions hold whatever `jax.lax.empty` left -- possibly NaN. Callers
    that feed these arrays anywhere numeric (e.g. an RL trainer's per-token
    logprobs) must slice or mask the prompt region rather than assume zeros.

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

  @jax.named_call
  def inject_chunk(
      self,
      slot_id: jax.typing.ArrayLike,
      payload: common.PyTree,
      start: jax.typing.ArrayLike | None = None,
      end: jax.typing.ArrayLike | None = None,
  ) -> Self:
    """Writes one cached chunk of KV for `slot_id` and advances `position`.

    The write counterpart of :meth:`extract_chunk`, and the whole body of the
    page batcher's inject program. Each DecodeState leaf gets its own
    payload (`[num_shards * page_size, full_heads, kv_packing, head_dim]`,
    sharded `(seq, head, None, None)`) written by
    :meth:`DecodeState.inject_chunk`, which is self-contained: it releases
    sliding-window pages *before* allocating the chunk (mirroring the decode
    path), so a cached prefix longer than the window can be injected chunk by
    chunk without the page pool ever overflowing. Only `kv_lens` / `pages`
    are windowed; `position` tracks the true logical length and is advanced
    here.

    `position` is *set* to `end` -- the range says where the slot lands, so
    there is nothing to add up and nothing to round. `position` is also what
    `start` must be: the leaves are windowed to different lengths, and the
    logical position is the one thing they agree on, so it is the frame the
    range is expressed in and each leaf translates it to its own columns.

    Args:
      slot_id: scalar i32 -- the batch slot to inject into.
      payload: pytree with the same structure as `self.decode_state`, each
        DecodeState leaf replaced by that leaf's chunk payload (the `onload`-ed
        form of what :meth:`extract_chunk` produced).
      start: scalar i32 -- first token position to write; MUST be the slot's
        `position`, which is the default when `None`.
      end: scalar i32 -- one past the last token position to write, and the
        slot's new `position`. `None` (default) means the end of `start`'s
        chunk.

    Returns:
      The updated `SamplingState`. `token_logprobs` / `token_scores` are NOT
      written for the restored span: unlike the prefill pass this replaces
      (which zeroes them via `skip_logits=True`), the injected positions keep
      whatever uninitialised bytes the buffers held. Positions below
      `input_len` are therefore UNDEFINED once anything has been injected --
      see :meth:`get`.
    """
    chunk_size = self.chunk_size
    start_int = (
        self.position[slot_id]
        if start is None
        else jnp.asarray(start, dtype=jnp.int32)
    )
    end_int = (
        (start_int // chunk_size + 1) * chunk_size
        if end is None
        else jnp.asarray(end, dtype=jnp.int32)
    )
    is_leaf = lambda x: isinstance(x, DecodeState)
    new_decode_state = jax.tree_util.tree_map(
        lambda ds, p: ds.inject_chunk(slot_id, p, start_int, end_int),
        self.decode_state,
        payload,
        is_leaf=is_leaf,
    )
    new_position = self.position.at[slot_id].set(end_int)
    # DELIBERATELY not touching `token_logprobs` / `token_scores`: those slots
    # are meaningful for DECODED tokens only, and writing them here would cost
    # two `jnp.where`s over the whole `[batch, max_seq_len]` f32 buffers on
    # every inject dispatch. See :meth:`get` for the contract this hands to
    # callers.
    return dataclasses.replace(
        self,
        decode_state=new_decode_state,
        position=new_position,
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

    The caller is expected to only request chunks whose KV has been
    computed (`chunk_idx < position // chunk_size`); a chunk past that
    frontier returns uninitialised bytes with an all-false mask.

    Args:
      slot_id: scalar i32 — the batch slot.
      chunk_idx: scalar i32 — the *logical* chunk index.

    Returns:
      A pytree with the same structure as `self.decode_state`, with each
      DecodeState leaf replaced by that leaf's
      :meth:`DecodeState.extract_chunk` result (a
      :class:`SnapshotChunkLeaf`: payload plus its per-token `written` mask).
      The input tokens are not returned: callers already hold them host-side,
      and the cache indexes on them.

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
    """Clamps the `prefill`-selected desired issue lens to `capacity`.

    Args:
      capacity: total number of tokens the pass may issue across all slots.
      prefill: select the prefill-flavoured desired issue lens.

    Returns:
      Per-slot token counts for this pass, `i32[batch]`.
    """
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
    """Executes a mixed step (prefill+decode).

    Args:
      forward_fn: the model's forward function.
      params: the model parameters pytree.
      extra_inputs: extra inputs threaded into `forward_fn`.
      max_num_issue_tokens: upper bound on tokens issued by this pass.
      temperature: sampling temperature for the issued token.
      top_k: sampling top-k for the issued token (`-1` disables).
      top_p: sampling top-p for the issued token.
      scoring_temperature: temperature used for the returned token scores.
      scoring_top_k: top-k used for the returned token scores (`-1` disables).
      scoring_top_p: top-p used for the returned token scores.
      prefill: whether this is a prefill pass (as opposed to pure decode).
      skip_logits: skip the logits computation; only written with `prefill`.

    Returns:
      The updated `SamplingState`.
    """
    if skip_logits and not prefill:
      raise ValueError('skip_logits is only written for prefill')
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

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
from jax.experimental.pallas.ops.tpu import ragged_paged_attention
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply.utils import common
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
):
  """Autotunes block sizes for ragged paged attention."""
  # TODO: More analysis on this value.
  # Increasing this value would shift the attention module from memory bandwidth
  # bound to compute bound, but in the meanwhile, it would cause more padding
  # overhead, given decoding does one-by-one token generation. 32 is a good
  # emperical trade-off so far.
  num_queries_per_block = min(32, max_num_issue_tokens)
  _, num_combined_kv_heads = (
      ragged_paged_attention.kernel.get_min_heads_per_blk(
          num_q_heads,
          num_kv_heads * 2,
          dtype,
          dtype,
      )
  )

  # This is an emperical estimation of the DMA issuing/waiting overhead
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
  padding_overhead_per_kv_page_blk = (page_size / 2) / min(
      window_size + 1 if window_size else max_seq_len, max_seq_len / 4
  )
  num_kv_pages_per_block = round(
      math.sqrt(dma_overhead / padding_overhead_per_kv_page_blk)
  )
  max_num_kv_pages_upper_bound = max_num_pages_per_seq(
      max_seq_len, page_size, window_size
  )
  return (
      min(num_kv_pages_per_block, max_num_kv_pages_upper_bound),
      num_queries_per_block,
  )


def max_num_pages_per_seq(
    max_seq_len: int,
    page_size: int,
    window_size: int | None,  # self excluded
) -> int:
  """Returns the maximum number of pages per sequence."""
  upper_bound = (max_seq_len - 1 + page_size - 1) // page_size
  if window_size is None:
    return upper_bound
  num_pages_for_window = (window_size + page_size - 1) // page_size
  return min(upper_bound, num_pages_for_window + 1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeStateConfig:
  """Paged KV cache config."""

  total_num_pages: int
  page_size: int
  n_kv_heads: int
  per_head_dim: int
  batch_size: int
  dtype: str
  max_seq_len: int
  window_size: int | None = None  # self excluded
  head_partition: str | Sequence[str] | None = None
  seq_partition: str | Sequence[str] | None = None
  num_kv_pages_per_block: int | None = None
  num_queries_per_block: int | None = None

  @property
  def padded_per_head_dim(self) -> int:
    return (self.per_head_dim + 127) // 128 * 128

  @property
  def max_num_pages_per_seq(self) -> int:
    return max_num_pages_per_seq(
        self.max_seq_len, self.page_size, self.window_size
    )

  def init(self) -> 'DecodeState':
    # TODO: Support data sharding
    return DecodeState(
        pages=sharding.with_sharding_constraint(
            jax.lax.empty(
                (
                    self.total_num_pages,
                    self.page_size,
                    self.n_kv_heads * 2,
                    self.padded_per_head_dim,
                ),
                dtype=self.dtype,
            ),
            (None, None, self.head_partition, None),
        ),
        page_indices=jax.lax.empty(
            (self.batch_size, self.max_num_pages_per_seq), dtype=jnp.int32
        ),
        available_page_indices=jnp.arange(
            self.total_num_pages, dtype=jnp.int32
        ),
        num_available_pages=jnp.array(self.total_num_pages, dtype=jnp.int32),
        kv_lens=jnp.zeros(self.batch_size, dtype=jnp.int32),
        max_seq_len=self.max_seq_len,
        window_size=self.window_size,
        head_partition=self.head_partition,
        seq_partition=self.seq_partition,
        num_kv_pages_per_block=self.num_kv_pages_per_block,
        num_queries_per_block=self.num_queries_per_block,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeState:
  """Paged KV cache."""

  # [total_num_pages, page_size, num_kv_heads * 2, padded_per_head_dim]
  pages: jax.Array
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq]
  available_page_indices: jax.Array  # i32[total_num_pages]
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
  num_kv_pages_per_block: int | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )
  num_queries_per_block: int | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )

  def __post_init__(self):
    head_partition_size = sharding.get_partition_size(self.head_partition)
    if self.num_kv_heads % head_partition_size != 0:
      raise ValueError(
          f'{self.num_kv_heads=} must be a multiple of {head_partition_size=}'
      )
    if self.page_indices.shape != (self.batch_size, self.max_num_pages_per_seq):
      raise ValueError(
          f'{self.page_indices.shape=} does not match'
          f' {self.batch_size=}, {self.max_num_pages_per_seq=}'
      )
    if self.page_indices.dtype != jnp.int32:
      raise ValueError(f'{self.page_indices.dtype=} must be int32.')
    if self.available_page_indices.shape != (self.total_num_pages,):
      raise ValueError(
          f'{self.available_page_indices.shape=} must be'
          f' {self.total_num_pages=}'
      )
    if self.available_page_indices.dtype != jnp.int32:
      raise ValueError(f'{self.available_page_indices.dtype=} must be int32.')
    if (
        self.num_available_pages.shape
        or self.num_available_pages.dtype != jnp.int32
    ):
      raise ValueError(f'{self.num_available_pages=} must be int32()')
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
    if self.max_num_pages_per_seq > self.total_num_pages:
      raise ValueError(
          f'{self.max_num_pages_per_seq=} must be <= {self.total_num_pages=}'
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
  def max_num_pages_per_seq(self) -> int:
    return max_num_pages_per_seq(
        self.max_seq_len, self.page_size, self.window_size
    )

  @property
  def total_num_pages(self) -> int:
    return self.pages.shape[0]

  @property
  def page_size(self) -> int:
    return self.pages.shape[1]

  @property
  def num_kv_heads(self) -> int:
    return self.pages.shape[-2] // 2

  @property
  def padded_per_head_dim(self) -> int:
    return self.pages.shape[-1]

  @property
  def dtype(self) -> jax.typing.DTypeLike:
    return self.pages.dtype

  @functools.cached_property
  def num_pages(self) -> jax.Array:
    return (self.kv_lens + (self.page_size - 1)) // self.page_size

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
    return np.asarray(self.available_page_indices)[
        : int(self.num_available_pages)
    ]

  @functools.cached_property
  def page_indices_nplist(self) -> Sequence[np.ndarray]:
    page_indices = np.asarray(self.page_indices)
    return [
        page_indices[i, : int(self.num_pages[i])]
        for i in range(self.batch_size)
    ]

  def kv_np(
      self, idx: jax.typing.ArrayLike, per_head_dim: int = 0
  ) -> np.ndarray:
    """Returns the kv for the given idx."""
    # Return shape in [kv_len, num_kv_heads * 2, per_head_dim]
    pages = self.pages[self.page_indices[idx]]
    context = einops.rearrange(pages, 'n p ... -> (n p) ...')
    if per_head_dim > 0:
      context = context[:, :, :per_head_dim]
    return np.asarray(context)[: self.kv_lens[idx]]

  def kv_nplist(self, per_head_dim: int = 0) -> Sequence[np.ndarray]:
    return [
        self.kv_np(i, per_head_dim=per_head_dim) for i in range(self.batch_size)
    ]

  @functools.cached_property
  def max_available_kv_lens(self) -> jax.Array:
    kv_lens = self.kv_lens
    if self.window_size is not None:
      kv_lens -= (
          jnp.maximum(kv_lens - self.window_size, 0)
          // self.page_size
          * self.page_size
      )
    return self.page_size * self.max_num_pages_per_seq - kv_lens

  @jax.named_call
  def release_for_window(self) -> Self:
    """Releases the decode state for local attention."""
    if self.window_size is None:
      return self
    num_pages_to_release = (
        jnp.maximum(self.kv_lens - self.window_size, 0) // self.page_size
    )
    page_indices_irows = jnp.arange(self.batch_size)[:, None]
    page_indices_icols = (
        jnp.arange(self.max_num_pages_per_seq) + num_pages_to_release[:, None]
    )
    updated_page_indices = self.page_indices[
        page_indices_irows, page_indices_icols
    ]
    release_helper = RaggedArray(
        data=jax.lax.empty((self.total_num_pages,), dtype=jnp.int32),
        lens=num_pages_to_release,
    )
    released_page_indices = self.page_indices[
        release_helper.row_ids, release_helper.intra_offset
    ]
    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages) + self.num_available_pages
    ].set(released_page_indices, mode='drop')
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
    """Allocates pages for new tokens."""
    required_num_pages = (
        self.kv_lens + q_lens + (self.page_size - 1)
    ) // self.page_size
    num_pages_to_allocate = required_num_pages - self.num_pages
    # User should guarantee:
    # total_num_pages_to_allocate <= num_available_pages
    page_indices_to_allocate = RaggedArray(
        data=self.available_page_indices, lens=num_pages_to_allocate
    )
    page_indices_irows = page_indices_to_allocate.row_ids
    page_indices_icols = (
        self.num_pages[page_indices_to_allocate.row_ids]
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
        data=jax.lax.empty((self.total_num_pages,), dtype=jnp.int32),
        lens=jnp.where(should_release, self.num_pages, 0),
    )
    page_indices_irows = page_indices_to_release.row_ids
    page_indices_icols = page_indices_to_release.intra_offset
    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages) + self.num_available_pages
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

  @jax.named_call
  def insert(self, k: jax.Array, v: jax.Array, q_lens: jax.Array) -> Self:
    """Inserts new kv into kv_pages at [kv_lens - q_lens, kv_lens)."""
    k = self.pad_per_head_dim(k)
    v = self.pad_per_head_dim(v)
    new_ragged_kv = RaggedArray(
        data=einops.rearrange(
            jnp.stack([k, v], axis=-2), '... n kv h -> ... (n kv) h'
        ),
        lens=q_lens,
    )

    row_ids = new_ragged_kv.row_ids
    intra_offset = new_ragged_kv.intra_offset
    positions = (self.kv_lens - q_lens)[row_ids] + intra_offset

    page_indices_irows = row_ids
    page_indices_icols = positions // self.page_size

    page_indices = self.page_indices[page_indices_irows, page_indices_icols]

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
  ) -> tuple[Self, jax.Array]:
    """Updates decode state."""
    if update_kv_cache:
      k = sharding.with_sharding_constraint(
          k, (None, self.head_partition, None)
      )
      v = sharding.with_sharding_constraint(
          v, (None, self.head_partition, None)
      )
      k = self.pad_per_head_dim(k)
      v = self.pad_per_head_dim(v)
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

      decode_state = decode_state.insert(k, v, q.lens)
    else:
      decode_state = self

    if jax.devices()[0].platform == 'cpu':
      # Pallas RPA kernel is TPU-only; always use reference impl on CPU.
      attn_output = ragged_paged_attention.ref_ragged_paged_attention(
          self.pad_per_head_dim(q.data),
          decode_state.pages,
          decode_state.kv_lens,
          decode_state.page_indices,
          jnp.cumulative_sum(q.lens, include_initial=True),
          jnp.reshape(jnp.sum(q.lens > 0), 1),
          sliding_window=self.window_size,
      )
      return decode_state, attn_output

    seq_partition_size = sharding.get_partition_size(self.seq_partition)
    num_kv_pages_per_block = self.num_kv_pages_per_block
    num_queries_per_block = self.num_queries_per_block
    if num_kv_pages_per_block is None or num_queries_per_block is None:
      head_partition_size = sharding.get_partition_size(self.head_partition)
      if q.capacity % seq_partition_size != 0:
        raise ValueError(
            f'{q.capacity=} must be divisible by {seq_partition_size=}'
        )
      num_kv_heads_per_shard = self.num_kv_heads // head_partition_size
      num_q_heads_per_shard = q.data.shape[1] // head_partition_size
      max_num_issue_tokens_per_shard = q.capacity // seq_partition_size
      num_kv_pages_per_block, num_queries_per_block = autotune_block_sizes(
          num_kv_heads=num_kv_heads_per_shard,
          num_q_heads=num_q_heads_per_shard,
          page_size=self.page_size,
          max_seq_len=self.max_seq_len,
          per_head_dim=self.padded_per_head_dim,
          window_size=self.window_size,
          dtype=self.dtype,
          max_num_issue_tokens=max_num_issue_tokens_per_shard,
      )
      logging.info(
          'Autotuned num_kv_pages_per_block: %d, num_queries_per_block: %d',
          num_kv_pages_per_block,
          num_queries_per_block,
      )
    batch_size = q.batch_size

    def _rpa_kernel_wrapper(q_shard, kv_pages, kv_lens, page_indices, q_lens):
      """Wrapper that compacts sequences and adjusts for seq partition."""
      shard_size = q_shard.shape[0]
      if seq_partition_size > 1:
        shard_idx = jax.lax.axis_index(self.seq_partition)
      else:
        shard_idx = 0
      shard_start = shard_idx * shard_size
      # Compute global cumulative q_lens and shift to shard-local coordinates.
      global_cu_q_lens = jnp.cumulative_sum(q_lens, include_initial=True)
      shard_cu_q_lens = jnp.clip(
          global_cu_q_lens - shard_start, 0, shard_size
      )

      shard_q_ends = shard_cu_q_lens[1:]
      shard_q_starts = shard_cu_q_lens[:-1]
      shard_q_lens = shard_q_ends - shard_q_starts

      # Compact to only sequences with tokens in this shard.
      shard_compact = jnp.flatnonzero(shard_q_lens, size=batch_size)
      num_seqs = jnp.sum(shard_q_lens > 0)
      compact_page_indices = page_indices[shard_compact]
      # Compute shard-local q_lens by clamping to non-negative.
      compact_shard_q_lens = shard_q_lens[shard_compact]
      cu_q_lens = jnp.cumulative_sum(
          compact_shard_q_lens, include_initial=True
      )

      # Adjust kv_lens for correct causal masking when a sequence
      # spans multiple shards.  The kernel computes position as
      #   kv_len - q_len + q_idx
      # so for the first shard of a cross-boundary sequence, we must
      # subtract the number of query tokens on *later* shards
      # (tokens_after) to shift the positions back correctly.
      #
      # Example: global_q_len=5 split [3 | 2] across 2 shards.
      #   Shard 0: tokens_after=2, kv_adj=K-2, q_len=3
      #            pos = (K-2)-3+idx -> K-5, K-4, K-3
      #   Shard 1: tokens_after=0, kv_adj=K,   q_len=2
      #            pos = K-2+idx     -> K-2, K-1
      tokens_after = jnp.maximum(
          global_cu_q_lens[1:] - shard_start - shard_size, 0
      )
      compact_kv_lens = (kv_lens - tokens_after)[shard_compact]

      return jax.lax.cond(
          num_seqs > 0,
          lambda: ragged_paged_attention.ragged_paged_attention(
              q_shard,
              kv_pages,
              compact_kv_lens,
              compact_page_indices,
              cu_q_lens,
              jnp.reshape(num_seqs, 1),
              # RPA kernel's window size includes self
              sliding_window=(
                  self.window_size + 1 if self.window_size else None
              ),
              soft_cap=soft_cap,
              mask_value=mask_value,
              num_kv_pages_per_block=num_kv_pages_per_block,
              num_queries_per_block=num_queries_per_block,
              vmem_limit_bytes=50 * 1024 * 1024,
          ),
          lambda: jnp.zeros_like(q_shard),
      )

    attn_output = jax.shard_map(
        _rpa_kernel_wrapper,
        mesh=js.get_abstract_mesh(),
        in_specs=(
            js.PartitionSpec(self.seq_partition, self.head_partition, None),
            js.PartitionSpec(None, None, self.head_partition, None),
            js.PartitionSpec(),
            js.PartitionSpec(),
            js.PartitionSpec(),
        ),
        out_specs=js.PartitionSpec(
            self.seq_partition, self.head_partition, None
        ),
        check_vma=False,
    )(
        self.pad_per_head_dim(q.data),
        decode_state.pages,
        decode_state.kv_lens,
        decode_state.page_indices,
        q.lens,
    )

    if attn_output.shape[0] < q.capacity:
      attn_output = jnp.pad(
          attn_output,
          (
              (0, q.capacity - attn_output.shape[0]),
              (0, 0),
              (0, 0),
          ),
      )
    per_head_dim = q.data.shape[-1]
    if attn_output.shape[-1] > per_head_dim:
      attn_output = attn_output[:, :, :per_head_dim]
    # Return in shape [max_num_tokens, num_q_heads, per_head_dim]
    return decode_state, attn_output


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
        prng_key=prng_key,
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
        eos_ids=eos_ids,
        max_total_num_tokens=max_total_num_tokens,
    )

  @property
  def batch_size(self) -> int:
    return self.tokens.shape[0]

  @property
  def max_seq_len(self) -> int:
    return self.tokens.shape[1]

  @functools.cached_property
  def is_pad_seq(self) -> jax.Array:
    """This sequence is a padding sequence, in [batch, 1]."""
    return self.input_lens == 0

  @functools.cached_property
  def desired_issue_lens(self) -> jax.Array:
    attrs = DecodeState.attrs_from_tree(
        self.decode_state, ['max_available_kv_lens']
    )
    max_available_kv_lens = jnp.min(
        jnp.array(attrs['max_available_kv_lens']), axis=0
    )
    return jnp.where(
        self.has_ended,
        0,
        jnp.minimum(
            max_available_kv_lens,
            jnp.maximum(
                self.input_lens
                - jnp.astype(self.max_decode_steps <= 0, jnp.int32)
                - self.position,
                1,
            ),
        ),
    )

  @functools.cached_property
  def rank_indices(self) -> jax.Array:
    inner_rank = jnp.where(
        self.is_pad_seq,
        2 * self.batch_size,
        jnp.where(
            self.desired_issue_lens == 1,
            self.rank,
            self.batch_size + self.rank,
        ),
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
    """Returns the tokens, logprobs, and scores for the given mask."""
    indices = np.flatnonzero(mask)
    lens = np.where(
        np.asarray(self.is_pad_seq), 0, np.asarray(self.position) + 1
    )
    input_lens = np.asarray(self.input_lens)
    results = []
    for index in indices:
      results.append(
          dict(
              index=int(index),
              input_len=int(input_lens[index]),
              tokens=np.asarray(self.tokens[index])[: lens[index]],
              logprobs=np.asarray(self.token_logprobs[index])[: lens[index]],
              scores=np.asarray(self.token_scores[index])[: lens[index]],
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

  @functools.cached_property
  def num_used_tokens(self) -> jax.Array:
    """Returns the number of used tokens."""
    return jnp.sum(self.position, where=~self.is_pad_seq, initial=0)

  @jax.jit(static_argnames=('capacity',))
  @jax.named_call
  def issue_lens(self, capacity: int) -> jax.Array:
    """Returns the issue lens."""
    sorted_desired_issue_lens = self.desired_issue_lens[self.rank_indices]

    # 1. Input length constraint.
    cum_sorted_issue_lens = jnp.minimum(
        jnp.cumulative_sum(sorted_desired_issue_lens, include_initial=True),
        capacity,
    )

    # 2. Max total num tokens constraint, guarantee first seq can be complete.
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
  def ragged_issue_tokens(self, capacity: int) -> common.RaggedArray:
    """Returns the ragged issue tokens."""
    # follows priority, and do not issue when oversubscriped.
    issue_lens = self.issue_lens(capacity)
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
  ) -> Self:
    """Executes a mixed step (prefill+decode)."""
    # User should guarantee self.is_continuable is True.
    # logits: [batch_size, 1, vocab_size]
    ragged_issue_tokens = self.ragged_issue_tokens(max_num_issue_tokens)

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
    extra_inputs['lens'] = ragged_issue_tokens.lens
    extra_inputs['page_manage_cache'] = {}

    logits, extra_output = forward_fn(
        params,
        einops.rearrange(ragged_issue_tokens.data, 'l -> 1 l'),
        segment_ids=einops.rearrange(segment_ids, 'l -> 1 l'),
        segment_positions=einops.rearrange(segment_positions, 'l -> 1 l'),
        extra_inputs=extra_inputs,
        decode_state=self.decode_state,
    )

    prng_key, key = jax.random.split(self.prng_key, 2)
    # output_tokens: [batch_size, 1], output_logprobs: [batch_size, 1]
    output_tokens, output_logprobs = sampling_lib.sample_from_logits(
        key, logits, temperature=temperature, top_k=top_k, top_p=top_p
    )

    next_tokens = self.tokens[
        ragged_issue_tokens.row_ids, segment_positions + 1
    ]
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

    sampling_state = self.update_with_ragged_output(
        RaggedArray(
            einops.rearrange(output_tokens, '1 l -> l'),
            ragged_issue_tokens.lens,
        ),
        token_logprobs=einops.rearrange(output_logprobs, '1 l -> l'),
        token_scores=einops.rearrange(output_scores, '1 l -> l'),
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

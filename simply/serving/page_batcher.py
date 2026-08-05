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
"""Batcher for the Simply gRPC server."""

# pyrefly: ignore-all-errors

import asyncio
from collections.abc import Callable, MutableSequence
import dataclasses
import functools
import queue
import threading
import time
from typing import Any

from absl import logging
import grpc
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply import model_lib
from simply.serving import common as serving_common
from simply.serving import prefix_cache as prefix_cache_lib
from simply.utils import checkpoint_lib
from simply.utils import common as core_common
from simply.utils import experiment_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import pytree
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sampling_lib
from simply.utils import sharding
from simply.utils import tokenization


PyTree = core_common.PyTree


SimplyServiceResponse = serving_common.SimplyServiceResponse


@dataclasses.dataclass(frozen=True)
class Batcher:
  """The batcher."""

  config: config_lib.BaseExperimentConfig
  lm_format: lm_format_lib.LMFormat
  state: PyTree = dataclasses.field(default_factory=dict)

  max_queue_size: int = 4096
  max_queue_timeout: float = 1.0  # seconds

  max_seq_len: int = 65537
  max_decode_steps: int = np.iinfo(np.int32).max // 2
  temperature: float = 1.0
  top_k: int = -1
  top_p: float = 1.0
  intermediate_steps: int = 1024
  response_asap: bool = False

  # Seed for the decode sampling PRNG. None (default) preserves the legacy
  # behavior of seeding from wall-clock time (`int(time.time() * 1000)`), i.e.
  # every run draws differently. Set to a fixed int for fixed-seed sampling.
  # Note this won't be fully deterministic due to async queueing in batch
  # decode.
  decode_seed: int | None = None

  # Maximum number of newly-issued (prefill) tokens dispatched per decode pass.
  # 0 (default) means "use batch_size", preserving the legacy bit-identical
  # behavior. Setting it > 0 DECOUPLES the issue-token budget from batch_size,
  # e.g. batch_size=4 with max_num_issue_tokens=64. The real per-pass advance
  # is `intermediate_steps * effective_max_num_issue_tokens`.
  max_num_issue_tokens: int = 0

  # ----- Prefix cache configuration -----
  # When True, the batcher consults a KV prefix cache on each push and
  # writes back snapshots of in-flight sequences after every device
  # round trip. The cache is held in **host (CPU) RAM** in-process
  # (offloaded from HBM) -- NOT persisted to disk. The cache key for
  # the k-th chunk is a deterministic hash of the first
  # `(k + 1) * chunk_size` input tokens, so different prompts that share
  # a prefix share their first KV chunks. False (default) disables the
  # prefix cache entirely.
  enable_prefix_caching: bool = False

  @functools.cached_property
  def model(self) -> model_lib.TransformerLM:
    return model_lib.TransformerLM(self.config)

  @functools.cached_property
  def decode_model(self) -> model_lib.TransformerLM:
    return model_lib.TransformerLM(
        dataclasses.replace(self.config, rpa_block_q=1)
    )

  @functools.cached_property
  def input_processor(self) -> sampling_lib.InputProcessorInterface:
    vocab = tokenization.TokenizerRegistry.get_instance(self.config.vocab_name)
    return sampling_lib.create_input_processor(
        self.config,
        vocab=vocab,
        bos_id_override=self.lm_format.bos_id,
        pad_id_override=self.lm_format.pad_id,
        extra_eos_tokens=self.lm_format.extra_eos_tokens,
    )

  @functools.cached_property
  def abstract_model_state(self) -> PyTree:
    """Abstract model state with full-precision (bf16) params.

    Used as the load target for the checkpoint (which is stored bf16).
    """

    def _init_fn():
      params = self.model.init(jax.random.key(0))
      if self.config.activation_dtype_name == 'bfloat16':
        params = jax.tree_util.tree_map(
            lambda x: jnp.astype(x, jnp.bfloat16)
            if x.dtype == jnp.float32
            else x,
            params,
        )
      params = self.model.quantize(params)
      return {'params': params}

    return core_common.eval_abstract_output(_init_fn)

  @property
  def sampling_state(self) -> rpa.SamplingState:
    sampling_state = self.state.get('sampling_state')
    if sampling_state is None:
      raise ValueError('Sampling state is not initialized.')
    return sampling_state

  def update_params(self, params: PyTree):
    self.state['params'] = params

  def clear_prefix_cache(self) -> None:
    """Retires the KV prefix cache; no-op when caching is disabled."""
    cache = self.prefix_cache
    if cache is not None:
      cache.clear()

  def update_params_from_checkpoint_path(self, ckpt_path: str):
    """Updates the model params from a checkpoint path."""
    if ckpt_format := self.config.init_ckpt_format:
      if isinstance(ckpt_format, str):
        ckpt_format_cls = checkpoint_lib.CheckpointFormatRegistry.get(
            ckpt_format
        )
        ckpt_format = ckpt_format_cls(
            restore_dtype=self.config.activation_dtype_name
        )
      elif isinstance(ckpt_format, checkpoint_lib.CheckpointFormat):
        ckpt_format = dataclasses.replace(
            ckpt_format, restore_dtype=self.config.activation_dtype_name
        )
      elif isinstance(ckpt_format, type) and issubclass(
          ckpt_format, checkpoint_lib.CheckpointFormat
      ):
        ckpt_format = ckpt_format(
            restore_dtype=self.config.activation_dtype_name
        )
    logging.info('ckpt_format: %s', ckpt_format)

    with self.set_mesh():
      model_state = checkpoint_lib.load_checkpoint_from_path(
          ckpt_path, self.abstract_model_state, ckpt_format=ckpt_format
      )
    self.update_params(core_common.get_raw_arrays(model_state['params']))

  @functools.cached_property
  def request_queue(
      self,
  ) -> queue.Queue[tuple[Any, asyncio.Future[SimplyServiceResponse]]]:
    return queue.Queue[tuple[Any, asyncio.Future[SimplyServiceResponse]]](
        maxsize=self.max_queue_size
    )

  def enqueue(
      self,
      request: Any,
      future: asyncio.Future[SimplyServiceResponse],
  ):
    self.request_queue.put((request, future), timeout=self.max_queue_timeout)

  def _try_get_request(
      self, max_seq_len: int, timeout: float
  ) -> tuple[Any, asyncio.Future[SimplyServiceResponse], np.ndarray] | None:
    """Tries to get a request from the queue."""
    deadline = time.time() + timeout
    while True:
      try:
        request, future = self.request_queue.get(
            timeout=max(deadline - time.time(), 0)
        )
      except queue.Empty:
        return None

      logging.info('request: %s', request)
      if future.cancelled():
        logging.info('Future is already cancelled.')
        continue

      try:
        inp = request
        if pytree.tree_is_sequence(inp):
          inp = self.lm_format.format(inp)
        input_chunks = sampling_lib.input_as_chunks(inp)
        logging.info('input_chunks: %s', input_chunks)
        processed_input = self.input_processor.encode(input_chunks, max_seq_len)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception('Failed to process input: %s', e)
        future.get_loop().call_soon_threadsafe(
            future.set_result,
            SimplyServiceResponse(
                code=grpc.StatusCode.INVALID_ARGUMENT,
                details=str(e),
            ),
        )
        continue

      return request, future, np.asarray(processed_input.tokens)

  def decode_fn(
      self, sampling_state: rpa.SamplingState, params: PyTree
  ) -> rpa.SamplingState:
    """Runs one pure-decode pass: 1 token per active slot, RPA block_q=1.

    Args:
      sampling_state: The current decode ``SamplingState``.
      params: The model parameters pytree.

    Returns:
      The advanced ``SamplingState`` after one decode pass.
    """
    if self.response_asap:
      until_fn = lambda state: jnp.any(~state.is_pad_seq & state.has_ended)
    else:
      until_fn = lambda state: jnp.array(False)

    return sampling_state.continue_decode(
        forward_fn=self.decode_model.apply,
        until_fn=until_fn,
        params=params,
        max_num_issue_tokens=self.config.batch_size,
        temperature=self.temperature,
        top_k=self.top_k,
        top_p=self.top_p,
        intermediate_steps=self.intermediate_steps,
    )

  def inject_chunk_fn(
      self,
      sampling_state: rpa.SamplingState,
      slot_id: int,
      payload: PyTree,
  ) -> rpa.SamplingState:
    """Pure inject function for one num_shards-page chunk.

    Writes exactly `num_shards` cached KV pages — one per
    seq-shard — into the live `decode_state.pages` tensor for
    `slot_id` by delegating to :meth:`DecodeState.inject_chunk` per
    layer. Has no decode body; the caller is expected to issue a
    subsequent decode dispatch.

    Multi-page cache hits are drained as multiple consecutive calls;
    each call appends `num_shards * page_size` tokens worth of KV and
    advances `kv_lens` accordingly. `position` is *set* to the end of
    the chunk just injected — the next chunk boundary at or after the
    current position, `(position // chunk + 1) * chunk` — so it is
    robust to the caller's starting position and to chunk granularity
    (no partial-add drift).

    Args:
      sampling_state: the current sampling state (donated).
      slot_id: i32[] scalar (replicated) — the batch slot to inject into.
      payload: pytree with the same structure as
        `sampling_state.decode_state` but each DecodeState leaf is
        replaced by a bf16/f32 `[num_shards * page_size, full_heads,
        kv_packing, head_dim]` array sharded `(seq, head, None, None)`
        — the token-major KV bytes for the page that DecodeState owns.

    Returns:
      The updated `SamplingState` with the chunk's KV injected for
      `slot_id` and its `position` advanced to the covered-prefix length.
    """
    assert (cache := self.prefix_cache) is not None
    is_leaf = lambda x: isinstance(x, rpa.DecodeState)
    # `inject_chunk` is self-contained: it releases sliding-window pages
    # *before* allocating each chunk (mirroring the decode path), so a
    # cached prefix longer than the window can be injected chunk-by-chunk
    # without the page pool ever overflowing. `position` tracks the true
    # logical length and is advanced separately below; only `kv_lens` /
    # `pages` are windowed — exactly the decode-loop invariant
    # (`update_decode_state_and_compute_attn`).
    new_decode_state = jax.tree_util.tree_map(
        lambda ds, p: ds.inject_chunk(slot_id, p),
        sampling_state.decode_state,
        payload,
        is_leaf=is_leaf,
    )
    chunk_size = cache.chunk_size
    # Set position to the end of the chunk just injected rather than
    # adding, so it can't drift if the starting position is not chunk-
    # aligned. The chunk index is inferred from the current position
    # (`position // num_inject_tokens`); the covered prefix after this
    # inject is the next chunk boundary, `(chunk_idx + 1) *
    # num_inject_tokens`.
    chunk_idx = sampling_state.position[slot_id] // chunk_size
    new_position = sampling_state.position.at[slot_id].set(
        (chunk_idx + 1) * chunk_size
    )
    return dataclasses.replace(
        sampling_state,
        decode_state=new_decode_state,
        position=new_position,
    )

  def init_sampling_state(self, prng_key: jax.Array) -> rpa.SamplingState:
    """Initializes sampling state."""
    if jax.sharding.get_abstract_mesh() is None:
      raise ValueError('Mesh is not set.')
    page_size = self.config.page_size
    total_num_pages = self.config.global_total_num_pages
    logging.info('page_size=%s', page_size)
    logging.info('total_num_pages=%s', total_num_pages)
    # TODO: Deprecate max_total_num_tokens and use better way to do the
    # constraint.
    max_total_num_tokens = page_size * total_num_pages + self.config.batch_size
    logging.info('max_total_num_tokens=%s', max_total_num_tokens)
    return rpa.SamplingState.create(
        max_total_num_tokens=max_total_num_tokens,
        eos_ids=jnp.asarray(self.input_processor.eos_ids),
        prng_key=prng_key,
        decode_state=jax.jit(self.model.init_decode_state, static_argnums=0)(
            self.max_seq_len
        ),
    )

  @functools.cached_property
  def abstract_sampling_state(self) -> PyTree:
    return core_common.eval_abstract_output(
        self.init_sampling_state, jax.random.key(0)
    )

  def set_mesh(self) -> jax.sharding.set_mesh:
    """Sets the mesh for the current process."""
    return sharding.set_mesh(
        self.config.mesh_shape,
        axis_names=self.config.sharding_config.mesh_axis_names,
    )

  @functools.cached_property
  def compiled_decode_fn(
      self,
  ) -> Callable[[rpa.SamplingState, PyTree], rpa.SamplingState]:
    """Compiled decode function."""
    if jax.config.jax_disable_jit:
      return self.decode_fn
    logging.info('Compiling decode function...')
    time_start = time.time()
    with self.set_mesh():
      compiled = (
          jax.jit(self.decode_fn, donate_argnames='sampling_state')
          .lower(
              self.abstract_sampling_state,
              core_common.get_raw_arrays(self.abstract_model_state['params']),
          )
          .compile()
      )
    logging.info(
        'Compiled decode function. Took %s seconds.', time.time() - time_start
    )
    return compiled

  def prefill_fn(
      self, sampling_state: rpa.SamplingState, params: PyTree
  ) -> rpa.SamplingState:
    """Runs ONE disaggregated prefill pass over the current batch.

    Args:
      sampling_state: the current decode ``SamplingState``.
      params: model parameters pytree.

    Returns:
      The ``SamplingState`` advanced by one prefill pass.
    """
    max_num_issue_tokens = self.max_num_issue_tokens or self.config.batch_size
    return sampling_state.mixed_step(
        forward_fn=self.model.apply,
        params=params,
        extra_inputs=None,
        max_num_issue_tokens=max_num_issue_tokens,
        scoring_temperature=self.temperature,
        scoring_top_k=self.top_k,
        scoring_top_p=self.top_p,
        prefill=True,
        skip_logits=True,
    )

  @functools.cached_property
  def compiled_prefill_fn(
      self,
  ) -> Callable[[rpa.SamplingState, PyTree], rpa.SamplingState]:
    """Compiled disaggregated chunked-prefill function."""
    if jax.config.jax_disable_jit:
      return self.prefill_fn
    logging.info('Compiling prefill function...')
    time_start = time.time()
    with self.set_mesh():
      compiled = (
          jax.jit(self.prefill_fn, donate_argnames='sampling_state')
          .lower(
              self.abstract_sampling_state,
              core_common.get_raw_arrays(self.abstract_model_state['params']),
          )
          .compile()
      )
    logging.info(
        'Compiled prefill function. Took %s seconds.', time.time() - time_start
    )
    return compiled

  @functools.cached_property
  def abstract_inject_payload(self) -> PyTree:
    """Abstract `payload` pytree for `inject_chunk_fn` AOT lowering."""
    assert (cache := self.prefix_cache) is not None
    abstract_mesh = jax.sharding.get_abstract_mesh()

    def _abs_for(ds: rpa.DecodeState) -> jax.ShapeDtypeStruct:
      payload_spec = jax.sharding.PartitionSpec(
          ds.seq_partition, ds.head_partition, None, None
      )
      payload_sharding = jax.sharding.NamedSharding(
          abstract_mesh, payload_spec
      )
      inner_shape = ds.pages.shape[2:]  # (full_heads, kvp, hd)
      return jax.ShapeDtypeStruct(
          (cache.chunk_size,) + inner_shape,
          ds.pages.dtype,
          sharding=payload_sharding,
      )

    return jax.tree_util.tree_map(
        _abs_for,
        self.abstract_sampling_state.decode_state,
        is_leaf=lambda x: isinstance(x, rpa.DecodeState),
    )

  @functools.cached_property
  def compiled_inject_chunk_fn(
      self,
  ) -> Callable[[rpa.SamplingState, int, PyTree], rpa.SamplingState]:
    """Returns the compiled inject-only program.

    The returned callable has signature `(sampling_state, pending) ->
    sampling_state`; `sampling_state` is donated. Has no decode
    body and always injects exactly `num_shards` pages (one per
    seq-shard) — multi-page cache hits are drained as multiple calls.

    HBM-wise, the inject program is tiny: pages aliases input -> output
    (donation fires); the prelude allocate + DUS + position update has
    zero scratch (see `test_dus_merged_inject_then_loop_no_extra_temp`
    — the DUS temp is zero, the allocate is i32 ops on small replicated
    tensors). So its program-loader output slot is essentially the same
    pages buffer that `compiled_decode_fn` reuses, with no extra HBM.
    """
    if jax.config.jax_disable_jit:
      return self.inject_chunk_fn
    logging.info('Compiling inject-only function...')
    time_start = time.time()
    with self.set_mesh():
      compiled = (
          jax.jit(self.inject_chunk_fn, donate_argnames='sampling_state')
          .lower(
              self.abstract_sampling_state,
              0,  # slot_id
              self.abstract_inject_payload,
          )
          .compile()
      )
    logging.info(
        'Compiled inject-only function. Took %s seconds.',
        time.time() - time_start,
    )
    return compiled

  @functools.cached_property
  def compiled_extract_chunk_fn(
      self,
  ) -> Callable[[rpa.SamplingState, int, int], PyTree]:
    """Returns the compiled :meth:`SamplingState.extract_chunk` program.

    Takes `(sampling_state, slot_id, chunk_idx)` and returns a pytree
    of :class:`rpa.SnapshotChunkLeaf` (one per DecodeState leaf), each with
    `payload` shape/sharding matching the inject side (see
    :meth:`abstract_inject_payload`) and a scalar `available` bool.
    The chunk size is fixed at `num_shards * page_size` and inferred
    internally by `extract_chunk` from the first DecodeState leaf.
    """
    if jax.config.jax_disable_jit:
      return rpa.SamplingState.extract_chunk
    logging.info('Compiling extract-chunk function...')
    time_start = time.time()
    with self.set_mesh():
      compiled = (
          jax.jit(rpa.SamplingState.extract_chunk)
          .lower(
              self.abstract_sampling_state,
              0,  # slot_id
              0,  # chunk_idx
          )
          .compile()
      )
    logging.info(
        'Compiled extract-chunk function. Took %s seconds.',
        time.time() - time_start,
    )
    return compiled

  @functools.cached_property
  def compiled_push_fn(
      self,
  ) -> Callable[
      [rpa.SamplingState, jax.typing.ArrayLike, int, int],
      tuple[rpa.SamplingState, jax.Array],
  ]:
    """Compiled push function."""
    if jax.config.jax_disable_jit:
      return rpa.SamplingState.push
    logging.info('Compiling push function...')
    time_start = time.time()
    with self.set_mesh():
      compiled = (
          jax.jit(rpa.SamplingState.push, donate_argnames='self')
          .lower(
              self.abstract_sampling_state,
              jax.ShapeDtypeStruct((self.max_seq_len,), jnp.int32),
              0,
              self.max_decode_steps,
          )
          .compile()
      )
    logging.info(
        'Compiled push function. Took %s seconds.', time.time() - time_start
    )
    return compiled

  @functools.cached_property
  def compiled_release_fn(
      self,
  ) -> Callable[[rpa.SamplingState, jax.typing.ArrayLike], rpa.SamplingState]:
    """Compiled release function."""
    if jax.config.jax_disable_jit:
      return rpa.SamplingState.release
    logging.info('Compiling release function...')
    time_start = time.time()
    with self.set_mesh():
      compiled = (
          jax.jit(rpa.SamplingState.release, donate_argnames='self')
          .lower(
              self.abstract_sampling_state,
              jax.ShapeDtypeStruct((self.config.batch_size,), jnp.bool),
          )
          .compile()
      )
    logging.info(
        'Compiled release function. Took %s seconds.', time.time() - time_start
    )
    return compiled

  @functools.cached_property
  def prefix_cache(self) -> prefix_cache_lib.PrefixCache | None:
    """Returns the prefix cache if enabled, otherwise None."""
    if not self.enable_prefix_caching:
      return None
    return prefix_cache_lib.PrefixCache(
        chunk_size=self.abstract_sampling_state.chunk_size,
        max_bytes=300 * 1024**3,  # 300 GiB host-RAM cap.
    )

  def _maybe_inject_prefix_cache(self) -> list[int]:
    """Looks up the prefix cache and injects any hit for prefilling slots.

    Scans every active slot (those with `~is_pad_seq`). For each slot,
    asks the cache how deep a prefix is fully restorable — see
    :meth:`PrefixCache.longest_restorable_prefix`, which enforces both "every
    chunk in `[0, k)` is present" and "every chunk in `[k -
    min_num_chunks_for_window, k)` is complete (every leaf payload
    present)." Then tree-maps :meth:`rpa.SnapshotChunkLeaf.onload`
    over each restored chunk to onload real bytes HBM-ward and
    materialise uninitialised buffers in their place for any
    `ShapeDtypeStruct` leaves (the leaf was unavailable at extract
    time), and injects via the AOT-compiled inject program.

    All hosts run this method in lockstep over identical state, so the
    per-process caches stay in sync.

    Returns:
      A length-`batch_size` list `num_cached_chunks` with the number of
      chunks injected for each slot (0 for free slots, decoding or
      non-chunk-aligned slots, and prefilling slots with no cache hit).
    """
    batch_size = self.sampling_state.batch_size
    num_cached_chunks = [0] * batch_size
    if (cache := self.prefix_cache) is None:
      return num_cached_chunks
    # One host pull of `position`, `input_lens`, `tokens`.
    positions = np.asarray(self.sampling_state.position)
    input_lens = np.asarray(self.sampling_state.input_lens)
    tokens = np.asarray(self.sampling_state.tokens)
    min_num_chunks_for_window = (
        self.abstract_sampling_state.min_num_chunks_for_window
    )
    time_start = time.time()
    total_dispatches = 0
    for slot_id in range(batch_size):
      if (input_len := input_lens[slot_id]) == 0:
        # Free slot: `position` is undefined, so there is nothing to inject.
        continue
      if (position := positions[slot_id]) >= input_len - 1:
        continue
      slot_tokens = tokens[slot_id]
      start_chunk_idx = position // cache.chunk_size
      max_chunks = int(input_len // cache.chunk_size)
      best_k = cache.longest_restorable_prefix(
          slot_tokens, max_chunks, start_chunk_idx, min_num_chunks_for_window
      )
      logging.info(
          'prefix_cache: slot=%d hit=%d/%d chunks (start=%d, input_tokens=%d)',
          slot_id,
          best_k - start_chunk_idx,
          max_chunks,
          start_chunk_idx,
          int(input_len),
      )
      # Inject chunks `[start_chunk_idx, best_k)` in order.
      # TODO: Avoid O(C^2) hashing.
      for chunk_idx in range(start_chunk_idx, best_k):
        token_chunks = slot_tokens[: (chunk_idx + 1) * cache.chunk_size]
        stored = cache.restore(token_chunks)
        assert stored is not None, (
            f'longest_restorable_prefix returned {best_k} but chunk {chunk_idx}'
            ' missing'
        )
        payload = jax.tree_util.tree_map(rpa.SnapshotChunkLeaf.onload, stored)
        self.state['sampling_state'] = self.compiled_inject_chunk_fn(
            self.sampling_state, slot_id, payload
        )
        total_dispatches += 1
      num_cached_chunks[slot_id] = best_k
    logging.info(
        'Injected prefix cache: %d total chunk dispatches in %.2fs',
        total_dispatches,
        time.time() - time_start,
    )
    return num_cached_chunks

  def _maybe_snapshot_prefix_cache(
      self, num_cached_chunks: MutableSequence[int]
  ) -> None:
    """Snapshots the PROMPT PREFIX of every slot that just finished prefill.

    For each active slot (those with `~is_pad_seq`), computes the
    number of chunks now fully represented in the on-device KV and
    snapshots any chunks past `num_cached_chunks[slot_id]`. Updates
    `num_cached_chunks` in place so the next snapshot pass only walks
    newly-ready chunks.

    Per-chunk reads use :meth:`SamplingState.extract_chunk` via the AOT-
    compiled :attr:`compiled_extract_chunk_fn` so the read uses the
    same `shard_map`-based per-shard DMA as the inject path. The
    program returns a pytree of :class:`rpa.SnapshotChunkLeaf` (one per
    DecodeState leaf); each leaf's `available` bool is `True` iff
    that layer's KV for this chunk is still resident
    (`base_col >= 0`). The batcher hands the tree straight to
    :meth:`PrefixCache.snapshot`, which calls
    :meth:`SnapshotChunkLeaf.offload` per leaf — returning the
    host-resident payload for available leaves and just a
    `ShapeDtypeStruct` for unavailable ones (no payload bytes stored,
    saving host RAM).

    Only the PROMPT prefix is cached, and only once per sequence -- right after
    it finishes prefill (`position == input_len - 1`, before any decode).
    Decode-generated KV (`position > input_len - 1`) is deliberately NOT cached.

    Args:
      num_cached_chunks: per-slot count of chunks already cached; updated in
        place so each pass only snapshots newly-ready chunks.
    """
    if (cache := self.prefix_cache) is None:
      return
    is_pad_seq = np.asarray(self.sampling_state.is_pad_seq)
    active_slots = [i for i, pad in enumerate(is_pad_seq) if not pad]
    if not active_slots:
      return
    tokens = np.asarray(self.sampling_state.tokens)
    positions = np.asarray(self.sampling_state.position)
    input_lens = np.asarray(self.sampling_state.input_lens)
    chunk_size = cache.chunk_size
    # Snapshot only chunks whose KV has actually been COMPUTED so far,
    # i.e. logical chunks in `[num_cached_chunks, ready_chunks)` where
    # `ready_chunks = position // chunk_size`. We only reach the per-slot body
    # at `position == input_len - 1`, so this is exactly the prompt prefix.
    #
    # `position` is how far the slot has been prefilled/decoded; chunks
    # beyond it have NOT been processed yet, so their pages are
    # uninitialized/stale -- snapshotting them would cache garbage KV.
    # `extract_chunk` maps the logical index to the correct physical
    # column per layer via `position`, and per-leaf gating inside
    # `SnapshotChunkLeaf.offload` substitutes a `ShapeDtypeStruct`
    # (zero-byte placeholder) for any layer whose chunk has already
    # been evicted by `release_for_window`.
    time_start = time.time()
    n_dispatched = 0
    for slot_id in active_slots:
      # Snapshot a slot's prompt prefix exactly once -- right when it finishes
      # prefill (`position == input_len - 1`, before any decode).
      if positions[slot_id] != input_lens[slot_id] - 1:
        continue
      ready_chunks = int(positions[slot_id]) // chunk_size
      already = num_cached_chunks[slot_id]
      if ready_chunks <= already:
        continue
      slot_tokens = tokens[slot_id]
      for chunk_idx in range(already, ready_chunks):
        token_chunks = slot_tokens[: (chunk_idx + 1) * cache.chunk_size]
        # Skip the (collective) extract+offload when the stored entry
        # is already fully complete (every leaf has a payload) — no
        # extract result could improve it. Partial entries may be
        # filled in by the new snapshot (per-leaf first-writer-wins
        # inside `PrefixCache.snapshot`).
        if cache.is_complete(token_chunks):
          continue
        entry_tree = self.compiled_extract_chunk_fn(
            self.sampling_state, slot_id, chunk_idx
        )
        cache.snapshot(token_chunks, entry_tree)
        n_dispatched += 1
      # Advance the per-slot watermark so the next pass only walks newly
      # ready chunks (and never re-reaches an evicted one).
      num_cached_chunks[slot_id] = ready_chunks
    logging.info(
        'Cached %d prefix chunks in %.2fs.',
        n_dispatched,
        time.time() - time_start,
    )

  def _maybe_pause(
      self,
      pause_event: threading.Event,
      paused_event: threading.Event,
      resume_event: threading.Event,
  ):
    """Pauses the batcher loop if pause_event is set, resuming across hosts."""
    if not sharding.sum_across_hosts(pause_event.is_set()):
      return
    pause_event.set()  # Force synchronous block state across sub-host loops.
    paused_event.set()
    resume_event.wait()
    resume_event.clear()
    paused_event.clear()

  def loop(
      self,
      stop_event: threading.Event,
      pause_event: threading.Event | None = None,
      paused_event: threading.Event | None = None,
      resume_event: threading.Event | None = None,
  ):
    """The batcher loop."""
    self.set_mesh()

    if self.state.get('sampling_state') is None:
      seed = (
          self.decode_seed
          if self.decode_seed is not None
          else int(time.time() * 1000)
      )
      seed = multihost_utils.broadcast_one_to_all(seed)
      logging.info('seed: %s', seed)
      time_start = time.time()
      self.state['sampling_state'] = self.init_sampling_state(
          jax.random.key(seed)
      )
      logging.info(
          'Initialized sampling state. Took %s seconds',
          time.time() - time_start,
      )

    # Per-slot state.
    #   `batch[i]`: `(request, future)` for the active request, or
    #     `None` when the slot is free.
    #   `num_cached_chunks[i]`: number of `num_shards * page_size`-
    #     token chunks of this slot's prefix that are present in the
    #     prefix cache. Returned by :meth:`_maybe_inject_prefix_cache`
    #     each pre-decode pass and bumped by
    #     :meth:`_maybe_snapshot_prefix_cache` after each decode call.
    # Prompt tokens themselves live in `self.sampling_state.tokens`
    # and are read on demand by the inject / snapshot passes.
    batch_size = self.sampling_state.batch_size
    batch = [None] * batch_size  # List of [(request, future) or None]
    while True:
      if sharding.sum_across_hosts(stop_event.is_set()):
        return

      if pause_event is not None:
        assert resume_event is not None
        assert paused_event is not None
        self._maybe_pause(pause_event, paused_event, resume_event)

      # Try to get a request if batch is not full.
      request, future = None, None
      input_tokens = np.empty((0,), dtype=np.int32)
      if not all(batch) and experiment_helper.is_primary_task():
        timeout = 0 if any(batch) else 60
        item = self._try_get_request(
            max_seq_len=self.sampling_state.max_seq_len, timeout=timeout
        )
        if item is not None:
          request, future, input_tokens = item

      input_len = len(input_tokens)
      n = int(sharding.sum_across_hosts(input_len))
      if n > 0:
        input_tokens = np.pad(
            input_tokens, (0, self.sampling_state.max_seq_len - input_len)
        )
        input_tokens = sharding.sum_across_hosts(input_tokens)
        self.state['sampling_state'], slot_id = self.compiled_push_fn(
            self.sampling_state, input_tokens, n, self.max_decode_steps
        )
        slot_id = int(slot_id)
        batch[slot_id] = (request, future)
        continue  # Filled a slot -- try to fill more before prefill/decode.

      if not any(batch):
        continue  # Nothing to do; loop back to the stop check.

      if bool(self.sampling_state.any_prefilling):
        num_cached_chunks = self._maybe_inject_prefix_cache()
        logging.info('Running chunked prefill step...')
        self.state['sampling_state'] = self.compiled_prefill_fn(
            self.sampling_state, self.state['params']
        )
        self._maybe_snapshot_prefix_cache(num_cached_chunks)
        continue  # Admit + prefill more before decoding.
      # Nothing needs prefill -> one baseline decode pass over the full batch.
      logging.info('Running decode function...')
      self.state['sampling_state'] = self.compiled_decode_fn(
          self.sampling_state, self.state['params']
      )

      if logging.vlog_is_on(1):
        logging.vlog(
            1, 'sampling_state.is_pad_seq=%s', self.sampling_state.is_pad_seq
        )
        logging.vlog(
            1, 'sampling_state.has_ended=%s', self.sampling_state.has_ended
        )
        logging.vlog(
            1, 'sampling_state.position=%s', self.sampling_state.position
        )
        logging.vlog(
            1, 'sampling_state.input_lens=%s', self.sampling_state.input_lens
        )
        logging.vlog(1, 'sampling_state.rank=%s', self.sampling_state.rank)
        logging.vlog(
            1,
            'num_used_tokens=%s, max_total_num_tokens=%s',
            self.sampling_state.num_used_tokens,
            self.sampling_state.max_total_num_tokens,
        )
        logging.vlog(
            1,
            'sampling_state.desired_issue_lens=%s',
            self.sampling_state.desired_issue_lens(),
        )
        logging.vlog(
            1,
            'sampling_state.issue_lens=%s',
            self.sampling_state.issue_lens(self.config.batch_size),
        )

      logging.info('Completed decode function...')
      # No post-decode snapshot: the disaggregated schedule snapshots the whole
      # prompt prefix once at end-of-prefill (before decode), and decode-stage
      # output KV is not cached in this schedule.
      completed_mask = (
          ~self.sampling_state.is_pad_seq & self.sampling_state.has_ended
      )
      completed_seqs = self.sampling_state.get(completed_mask)

      is_cancelled = np.array([False] * self.sampling_state.batch_size)
      if experiment_helper.is_primary_task():
        for i, request_future in enumerate(batch):
          if request_future is not None:
            _, future = request_future
            if future is not None and future.cancelled():
              logging.info('Future is cancelled.')
              is_cancelled[i] = True
      is_cancelled = np.astype(sharding.sum_across_hosts(is_cancelled), np.bool)

      logging.info('is_cancelled=%s', is_cancelled)
      logging.info('Releasing sampling state...')
      should_release_mask = completed_mask | jnp.asarray(is_cancelled)
      if np.any(should_release_mask):
        self.state['sampling_state'] = self.compiled_release_fn(
            self.sampling_state, should_release_mask
        )

      if experiment_helper.is_primary_task():
        for seq in completed_seqs:
          seq = {key: value for key, value in seq.items()}
          seq['output_text'] = sampling_lib.chunks_as_text(
              self.input_processor.decode(
                  seq['tokens'][seq['input_len'] :].tolist()
              )
          )
          if parser := getattr(self.lm_format, 'parse', None):
            assistant_marker = getattr(self.lm_format, 'assistant_marker', '')
            text_to_parse = assistant_marker + seq['output_text']
            logging.info('output_text_to_parse=%s', text_to_parse)
            output_messages = parser(text_to_parse)
            seq['output_messages'] = output_messages
          index = seq.pop('index')
          if entry := batch[index]:
            _, future = entry
            if not future.cancelled():
              logging.info('Setting future result.')
              future.get_loop().call_soon_threadsafe(
                  future.set_result,
                  SimplyServiceResponse(
                      code=grpc.StatusCode.OK,
                      result=seq,
                  ),
              )

      for index in np.flatnonzero(should_release_mask):
        batch[index] = None

      if self.prefix_cache is not None and logging.vlog_is_on(1):
        logging.vlog(1, 'prefix_cache.stats=%s', self.prefix_cache.stats())

  def thread(
      self,
      stop_event: threading.Event,
      error_message_queue: queue.Queue[Exception],
      pause_event: threading.Event | None = None,
      paused_event: threading.Event | None = None,
      resume_event: threading.Event | None = None,
  ) -> threading.Thread:
    """Starts the batcher thread."""

    def _batcher_loop():
      try:
        self.loop(
            stop_event,
            pause_event=pause_event,
            paused_event=paused_event,
            resume_event=resume_event,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.exception('Batcher loop failed: %s', e)
        stop_event.set()
        error_message_queue.put(e)

    return threading.Thread(target=_batcher_loop, daemon=True)

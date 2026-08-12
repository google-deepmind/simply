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

import asyncio
from collections.abc import Callable
import dataclasses
import functools
import math
import queue
import threading
import time
from typing import Any, cast

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
  # A MAPPING, not a bare `PyTree`: everything here is `state['params']` /
  # `state['sampling_state']`, which a `PyTree` union cannot be indexed into.
  state: dict[str, Any] = dataclasses.field(default_factory=dict)

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
  # When True, the batcher restores the deepest cached prefix into every
  # prefilling slot and snapshots what each prefill pass made real. The
  # cache is held in **host (CPU) RAM** in-process (offloaded from HBM) --
  # NOT persisted to disk -- and is indexed by the prompt TOKENS in a
  # radix trie, so prompts that share a prefix share its KV chunks. False
  # (default) disables the prefix cache entirely.
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
  def abstract_model_state(self) -> dict[str, Any]:
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

    return cast(dict[str, Any], core_common.eval_abstract_output(_init_fn))

  @property
  def sampling_state(self) -> rpa.SamplingState:
    sampling_state = self.state.get('sampling_state')
    if sampling_state is None:
      raise ValueError('Sampling state is not initialized.')
    return sampling_state

  def update_params(self, params: PyTree):
    self.state['params'] = params

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
        forward_fn=self.decode_model.apply,  # pyrefly: ignore[bad-argument-type]
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
      start: int,
      end: int,
  ) -> rpa.SamplingState:
    """Pure inject function for one num_shards-page chunk.

    Thin wrapper around :meth:`rpa.SamplingState.inject_chunk`, which writes
    exactly `num_shards` cached KV pages -- one per seq-shard -- into the
    live `decode_state.pages` tensor for `slot_id` (per layer) and advances
    `position` to the end of the injected chunk. Has no decode body; the
    caller is expected to issue a subsequent decode dispatch. Multi-chunk
    cache hits are drained as multiple consecutive calls.

    Args:
      sampling_state: the current sampling state (donated).
      slot_id: i32[] scalar (replicated) — the batch slot to inject into.
      payload: pytree with the same structure as `sampling_state.decode_state`
        but each DecodeState leaf is replaced by a bf16/f32 `[num_shards *
        page_size, full_heads, kv_packing, head_dim]` array sharded `(seq, head,
        None, None)` — the token-major KV bytes for the page that DecodeState
        owns.
      start: i32[] scalar (replicated) — start token position to inject.
      end: i32[] scalar (replicated) — end token position to inject.

    Returns:
      The updated `SamplingState` with the chunk's KV injected for
      `slot_id` and its `position` advanced to `end`.
    """
    assert self.prefix_cache is not None
    return sampling_state.inject_chunk(slot_id, payload, start, end)

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
  def abstract_sampling_state(self) -> rpa.SamplingState:
    return cast(
        rpa.SamplingState,
        core_common.eval_abstract_output(
            self.init_sampling_state, jax.random.key(0)
        ),
    )

  def set_mesh(self) -> jax.sharding.set_mesh:
    """Sets the mesh for the current process."""
    return sharding.set_mesh(
        self.config.mesh_shape,  # pyrefly: ignore[bad-argument-type]
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
        forward_fn=self.model.apply,  # pyrefly: ignore[bad-argument-type]
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
  ) -> Callable[[rpa.SamplingState, int, PyTree, int, int], rpa.SamplingState]:
    """Returns the compiled inject-only program.

    The returned callable has signature `(sampling_state, slot_id, payload,
    start, end) -> sampling_state`; `sampling_state` is donated. See
    :meth:`_inject_chunk_body`.

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
              0,  # start
              0,  # end
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
    :meth:`abstract_inject_payload`) and its per-token `written` mask.
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
      return rpa.SamplingState.release  # pyrefly: ignore[bad-return]
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

  def _maybe_restore_from_prefix_cache(self) -> np.ndarray:
    """Restores the deepest cached prefix into every prefilling slot.

    Returns:
      Number of tokens restored from the prefix caching.
    """
    batch_size = self.sampling_state.batch_size
    if (cache := self.prefix_cache) is None:
      return np.zeros(batch_size, dtype=np.int32)
    time_start = time.time()
    positions = np.asarray(self.sampling_state.position).copy()
    input_lens = np.asarray(self.sampling_state.input_lens)
    tokens = np.asarray(self.sampling_state.tokens)
    n_dispatches = 0

    for slot_id in range(batch_size):
      if (input_len := input_lens[slot_id]) == 0:
        continue
      if (position := positions[slot_id]) >= input_len - 1:
        continue
      # Prefill covers `[0, input_len - 1)` -- the last prompt token is
      # consumed by the first decode step -- so a restore may never move the
      # slot past that.
      for tile in cache.restore_chunk_tiles(
          tokens[slot_id], position, input_len - 1
      ):
        payload = rpa.onload_chunk_tree(tile.tree)
        self.state['sampling_state'] = self.compiled_inject_chunk_fn(
            self.sampling_state, slot_id, payload, tile.start, tile.end
        )
        n_dispatches += 1
    logging.info(
        'Restored from prefix cache: %d total chunk dispatches in %.2fs',
        n_dispatches,
        time.time() - time_start,
    )
    after = np.asarray(self.sampling_state.position)
    return after - positions

  def _maybe_evict_prefix_cache(self) -> int:
    """Trims the prefix cache to its byte budget. Returns bytes freed.

    MUST only run with no slot mid-prompt; see the call site.

    Returns:
      Host bytes freed, `0` when there is no cache or it already fits.
    """
    if (cache := self.prefix_cache) is None:
      return 0
    time_start = time.time()
    freed = cache.evict()
    if freed:
      logging.info(
          'prefix_cache: freed %d MiB in %.2fs; %d MiB held',
          freed >> 20,
          time.time() - time_start,
          cache.nbytes >> 20,
      )
    return freed

  def _maybe_snapshot_prefix_cache(self, previous: np.ndarray) -> np.ndarray:
    """Caches what the prefill pass just made ready, `[previous, position)`.

    Args:
      previous: per-slot position BEFORE the prefill pass -- where the cache
        already reaches, so the range this pass added is `[previous,
        position)`.

    Returns:
      Prompt tokens newly cached per slot, as an `int64` array of length
      `batch_size` -- for the response's accounting. It reports what it
      STORED rather than a position delta, because a prefill pass moves the
      position of every slot it touches, including the ones skipped here.
    """
    written = np.zeros(self.sampling_state.batch_size, dtype=np.int32)
    if (cache := self.prefix_cache) is None:
      return written
    time_start = time.time()
    positions = np.asarray(self.sampling_state.position)
    input_lens = np.asarray(self.sampling_state.input_lens)
    tokens = np.asarray(self.sampling_state.tokens)
    chunk_size = cache.chunk_size
    n_dispatches = 0
    for slot_id in range(self.sampling_state.batch_size):
      if input_lens[slot_id] == 0:
        continue
      position = positions[slot_id]
      # Prompt prefix only: stop caching once the slot starts decoding.
      if position > input_lens[slot_id] - 1 or position <= 0:
        continue
      start = previous[slot_id]
      if start >= position:
        continue
      tiles = []
      for chunk_idx in range(
          start // chunk_size, math.ceil(position / chunk_size)
      ):
        chunk_start = chunk_idx * chunk_size
        tiles.append(
            prefix_cache_lib.ChunkTile(
                rpa.offload_chunk_tree(
                    self.compiled_extract_chunk_fn(
                        self.sampling_state, slot_id, chunk_idx
                    )
                ),
                max(chunk_start, start),
                min(chunk_start + chunk_size, position),
            )
        )
        n_dispatches += 1
      # STORE. The pass's end is the resume point; a refusal stops the run
      # short and says so. Nothing is claimed on the cache's behalf either
      # way, so nothing is now false: the next pass covers the same ground.
      slot_tokens = tokens[slot_id]
      reached = cache.store_tiles(slot_tokens, tiles)
      written[slot_id] = reached - start
      if reached < position:
        raise ValueError(
            'Prefix cache refused to store tiles for slot %d from %d to %d'
            % (slot_id, start, position)
        )
    logging.info(
        'Cached %d prefix chunks in %.2fs.',
        n_dispatches,
        time.time() - time_start,
    )
    return written

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
    #   There is NO cursor: a slot's device position IS how much of its
    #     prefix the cache holds, because a snapshot pass caches exactly
    #     what the prefill pass computed -- and where it does not (a store
    #     the cache refused), the next pass finds out by asking rather than
    #     by being told, so nothing here has to remember.
    # Prompt tokens themselves live in `self.sampling_state.tokens`
    # and are read on demand by the inject / snapshot passes.
    batch_size = self.sampling_state.batch_size
    # One slot per batch index: the request that owns it, or None.
    batch: list[tuple[Any, Any] | None] = [None] * batch_size
    cache_read_tokens = np.zeros(batch_size, dtype=np.int64)
    cache_write_tokens = np.zeros(batch_size, dtype=np.int64)
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
      n = int(sharding.sum_across_hosts(input_len))  # pyrefly: ignore[bad-argument-type]
      if n > 0:
        input_tokens = np.pad(
            input_tokens, (0, self.sampling_state.max_seq_len - input_len)
        )
        input_tokens = sharding.sum_across_hosts(input_tokens)  # pyrefly: ignore[bad-argument-type]
        self.state['sampling_state'], slot_id = self.compiled_push_fn(
            self.sampling_state, input_tokens, n, self.max_decode_steps  # pyrefly: ignore[bad-argument-type]
        )
        slot_id = int(slot_id)
        batch[slot_id] = (request, future)
        cache_read_tokens[slot_id] = 0
        cache_write_tokens[slot_id] = 0
        continue  # Filled a slot -- try to fill more before prefill/decode.

      if not any(batch):
        continue  # Nothing to do; loop back to the stop check.

      if bool(self.sampling_state.any_prefilling):
        cache_read_tokens += self._maybe_restore_from_prefix_cache()
        if bool(self.sampling_state.any_prefilling):
          logging.info('Running chunked prefill step...')
          before = np.asarray(self.sampling_state.position)
          self.state['sampling_state'] = self.compiled_prefill_fn(
              self.sampling_state, self.state['params']
          )
          cache_write_tokens += self._maybe_snapshot_prefix_cache(before)

        continue  # Admit + prefill more before decoding.
      # NOTHING IS MID-PROMPT HERE, and that is the whole reason the cache is
      # trimmed at this exact point: every slot is past its prefill, so none
      # of them needs the trie to still hold what it held. A slot part-way
      # through its prompt is the one reader that does, and while one exists
      # the loop never reaches this line.
      self._maybe_evict_prefix_cache()
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
      is_cancelled = np.astype(sharding.sum_across_hosts(is_cancelled), np.bool)  # pyrefly: ignore[no-matching-overload, bad-argument-type]

      logging.info('is_cancelled=%s', is_cancelled)
      logging.info('Releasing sampling state...')
      should_release_mask = completed_mask | jnp.asarray(is_cancelled)
      if np.any(should_release_mask):
        self.state['sampling_state'] = self.compiled_release_fn(
            self.sampling_state, should_release_mask
        )

      if experiment_helper.is_primary_task():
        for completed in completed_seqs:
          seq: dict[str, Any] = dict(completed)
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
          # Per-response prefix-cache attribution (0 when caching is off).
          seq['n_cache_read_tokens'] = int(cache_read_tokens[index])
          seq['n_cache_write_tokens'] = int(cache_write_tokens[index])
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

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
  page_size: int = 128
  temperature: float = 1.0
  top_k: int = -1
  top_p: float = 1.0
  intermediate_steps: int = 1024
  response_asap: bool = False

  @functools.cached_property
  def model(self) -> model_lib.TransformerLM:
    return model_lib.TransformerLM(self.config)

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
    """Returns the abstract model state."""

    def _init_fn():
      params = self.model.init(jax.random.key(0))
      if self.config.activation_dtype_name == 'bfloat16':
        params = jax.tree_util.tree_map(
            lambda x: jnp.astype(x, jnp.bfloat16)
            if x.dtype == jnp.float32
            else x,
            params,
        )
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

  def update_params_from_checkpoint_path(self, ckpt_path: str):
    """Updates the model params from a checkpoint path."""
    if ckpt_format := self.config.init_ckpt_format:
      ckpt_format_cls = checkpoint_lib.CheckpointFormatRegistry.get(ckpt_format)
      ckpt_format = ckpt_format_cls(
          restore_dtype=self.config.activation_dtype_name
      )
    logging.info('ckpt_format: %s', ckpt_format)

    with self.set_mesh():
      model_state = checkpoint_lib.load_checkpoint_from_path(
          ckpt_path,
          self.abstract_model_state,
          ckpt_format=ckpt_format,
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
    if self.response_asap:
      until_fn = lambda state: jnp.any(~state.is_pad_seq & state.has_ended)
    else:
      until_fn = lambda state: jnp.array(False)

    return sampling_state.continue_decode(
        forward_fn=self.model.apply,
        until_fn=until_fn,
        params=params,
        max_num_issue_tokens=self.config.batch_size,  # TODO: Tune this.
        temperature=self.temperature,
        top_k=self.top_k,
        top_p=self.top_p,
        intermediate_steps=self.intermediate_steps,
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
      seed = int(time.time() * 1000)
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

    batch = [None] * self.sampling_state.batch_size
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
        timeout = (
            self.max_queue_timeout
            if any(batch)
            else max(self.max_queue_timeout, 60)
        )
        item = self._try_get_request(
            max_seq_len=self.sampling_state.max_seq_len, timeout=timeout
        )
        if item is not None:
          request, future, input_tokens = item

      input_len = len(input_tokens)
      n = sharding.sum_across_hosts(input_len)

      if n > 0:
        input_tokens = np.pad(
            input_tokens, (0, self.sampling_state.max_seq_len - input_len)
        )
        input_tokens = sharding.sum_across_hosts(input_tokens)
        self.state['sampling_state'], index = self.compiled_push_fn(
            self.sampling_state, input_tokens, n, self.max_decode_steps
        )
        batch[int(index)] = (request, future)
        continue  # Try to fill more slots before decoding.

      if not any(batch):
        continue  # Nothing to do, loops back to stop check.

      # Batch has items, no new request → decode.
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
            self.sampling_state.desired_issue_lens,
        )
        logging.vlog(
            1,
            'sampling_state.issue_lens=%s',
            self.sampling_state.issue_lens(self.config.batch_size),
        )

      logging.info('Completed decode function...')
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
          if request_future := batch[index]:
            _, future = request_future
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

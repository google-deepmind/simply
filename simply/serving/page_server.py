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
r"""Simply gRPC server that uses Ragged Paged Attention.

Start server example:
  JAX_DISABLE_JIT=1 python -m simply.serving.page_server \
    --experiment_config=qwen3_1p7b \
    --lm_format=QwQChat \
    --batch_size=4 \
    --max_seq_len=32 \
    --simply_port=12345 \
    --alsologtostderr

Client query example:
  grpc_cli call localhost:12345 simply.SimplyService/Run 'string_value: "Hello"'
"""

import asyncio
from collections.abc import Sequence
import dataclasses
import functools
import queue
import threading
import time
from typing import Any, NamedTuple

from absl import app
from absl import flags
from absl import logging
import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply import data_lib  # pylint: disable=unused-import
from simply import model_lib
from simply.serving import common
from simply.serving import server_pb2
from simply.serving import server_pb2_grpc
from simply.serving import struct_pb2
from simply.utils import checkpoint_lib
from simply.utils import common as core_common
from simply.utils import experiment_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import pytree
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sampling_lib
from simply.utils import sharding
from simply.utils import tokenization


_SIMPLY_PORT = flags.DEFINE_integer(
    'simply_port', None, 'Port to listen on.', required=True
)

_EXPERIMENT_CONFIG = flags.DEFINE_string(
    'experiment_config',
    None,
    'Experiment config that contains the model config to use.',
    required=True,
)

_MESH_SHAPE = flags.DEFINE_list(
    'mesh_shape', None, 'Mesh shape to use. If none, use default mesh shape.'
)

_CKPT_DIR = flags.DEFINE_string(
    'ckpt_dir', None, 'Path to the checkpoints directory.'
)

_CKPT_STEP = flags.DEFINE_integer(
    'ckpt_step',
    -1,
    'Checkpoint step to use. By default, use the latest checkpoint step.',
)

_CKPT_FORMAT = flags.DEFINE_string(
    'ckpt_format', None, 'Checkpoint format to use. (Optional)'
)

_VOCAB_NAME = flags.DEFINE_string(
    'vocab_name',
    None,
    'Name of the vocab. If not provided, use the vocab name in the experiment'
    ' config.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 1, 'Batch size to use for decoding.'
)

_ACTIVATION_DTYPE = flags.DEFINE_string(
    'activation_dtype', 'bfloat16', 'Dtype of the activation.'
)

_MAX_SEQ_LEN = flags.DEFINE_integer(
    'max_seq_len', 65537, 'Max sequence length for the model.'
)

_MAX_DECODE_STEPS = flags.DEFINE_integer(
    'max_decode_steps',
    np.iinfo(np.int32).max // 2,
    'Max decode steps for the model.',
)

_LM_FORMAT = flags.DEFINE_string(
    'lm_format',
    None,
    'LM format to use. If not provided, use the default LM format in the'
    ' experiment config.',
)


PyTree = core_common.PyTree


class SimplyServiceResponse(NamedTuple):
  code: grpc.StatusCode = grpc.StatusCode.OK
  details: str = ''
  result: Any = None


@dataclasses.dataclass(frozen=True)
class Batcher:
  """The batcher."""

  config: config_lib.BaseExperimentConfig
  lm_format: lm_format_lib.LMFormat
  model_state: PyTree = dataclasses.field(default_factory=dict)

  max_queue_size: int = 4096
  max_queue_timeout: float = 1.0  # seconds

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

  def update_params(self, params: PyTree):
    self.model_state['params'] = params

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

  def decode_fn(
      self, sampling_state: rpa.SamplingState, params: PyTree
  ) -> rpa.SamplingState:
    return sampling_state.continue_decode(
        forward_fn=self.model.apply,
        # TODO: Support intermediate insert.
        until_fn=lambda state: jnp.any(~state.is_pad_seq & state.has_ended),
        params=params,
        max_num_issue_tokens=_BATCH_SIZE.value,  # TODO: Tune this.
    )

  def loop(self, stop_event: threading.Event):
    """The batcher loop."""
    sharding.set_mesh(self.config.mesh_shape)
    seed = int(time.time() * 1000)
    seed = multihost_utils.broadcast_one_to_all(seed)

    page_size = 128
    total_num_pages = (
        (_MAX_SEQ_LEN.value - 1 + page_size - 1)
        // page_size
        * _BATCH_SIZE.value
    )
    logging.info('page_size: %s', page_size)
    logging.info('total_num_pages: %s', total_num_pages)
    max_total_num_tokens = (
        page_size * (total_num_pages - _BATCH_SIZE.value) + _BATCH_SIZE.value
    )
    logging.info('max_total_num_tokens: %s', max_total_num_tokens)
    sampling_state = rpa.SamplingState.create(
        max_total_num_tokens=max_total_num_tokens,
        eos_ids=jnp.asarray(self.input_processor.eos_ids),
        prng_key=jax.random.key(seed),
        decode_state=jax.jit(self.model.init_decode_state, static_argnums=0)(
            _MAX_SEQ_LEN.value
        ),
    )
    experiment_helper.set_notes('Compiling ...')
    logging.info('Compiling decode function...')
    time_start = time.time()
    if jax.config.jax_disable_jit:
      compiled_decode_fn = self.decode_fn
    else:
      compiled_decode_fn = (
          jax.jit(self.decode_fn, donate_argnames='sampling_state')
          .lower(sampling_state, self.model_state['params'])
          .compile()
      )
    logging.info(
        'Compiled decode function. Took %s seconds.', time.time() - time_start
    )
    logging.info('Compiling push function...')
    time_start = time.time()
    if jax.config.jax_disable_jit:
      compiled_push_fn = rpa.SamplingState.push
    else:
      compiled_push_fn = (
          jax.jit(rpa.SamplingState.push, donate_argnames='self')
          .lower(
              sampling_state,
              sampling_state.tokens[0],
              0,
              _MAX_DECODE_STEPS.value,
          )
          .compile()
      )
    logging.info(
        'Compiled push function. Took %s seconds.', time.time() - time_start
    )
    logging.info('Compiling release function...')
    time_start = time.time()
    if jax.config.jax_disable_jit:
      compiled_release_fn = rpa.SamplingState.release
    else:
      compiled_release_fn = (
          jax.jit(rpa.SamplingState.release, donate_argnames='self')
          .lower(sampling_state, sampling_state.is_pad_seq)
          .compile()
      )
    logging.info(
        'Compiled release function. Took %s seconds.', time.time() - time_start
    )
    experiment_helper.set_notes('Ready')

    batch = [None] * sampling_state.batch_size
    while not stop_event.is_set():
      while not all(batch):
        try:
          request, future = self.request_queue.get(
              timeout=self.max_queue_timeout if any(batch) else None
          )
          logging.info('request: %s', request)
          if future.cancelled():
            logging.info('Future is already cancelled.')
            continue
        except queue.Empty:
          break

        try:
          inp = request
          if pytree.tree_is_sequence(inp):
            inp = self.lm_format.format(inp)
          input_chunks = sampling_lib.input_as_chunks(inp)
          logging.info('input_chunks: %s', input_chunks)
          processed_input = self.input_processor.encode(
              input_chunks, sampling_state.max_seq_len
          )
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
        n = len(processed_input.tokens)
        input_tokens = np.pad(
            np.array(processed_input.tokens),
            (0, sampling_state.max_seq_len - n),
        )
        sampling_state, index = compiled_push_fn(
            sampling_state, input_tokens, n, _MAX_DECODE_STEPS.value
        )
        batch[int(index)] = (request, future)

      logging.info('Running decode function...')
      sampling_state = compiled_decode_fn(
          sampling_state, self.model_state['params']
      )

      logging.info('sampling_state.is_pad_seq=%s', sampling_state.is_pad_seq)
      logging.info('sampling_state.has_ended=%s', sampling_state.has_ended)
      logging.info('sampling_state.position=%s', sampling_state.position)
      logging.info('sampling_state.input_lens=%s', sampling_state.input_lens)
      logging.info('sampling_state.rank=%s', sampling_state.rank)
      completed_mask = ~sampling_state.is_pad_seq & sampling_state.has_ended
      completed_seqs = sampling_state.get(completed_mask)
      logging.info('Completed decode function...')

      is_cancelled = [False] * sampling_state.batch_size
      for i, request_future in enumerate(batch):
        if request_future is not None:
          _, future = request_future
          if future.cancelled():
            logging.info('Future is cancelled.')
            is_cancelled[i] = True
            batch[i] = None
      logging.info('is_cancelled=%s', is_cancelled)
      logging.info('Releasing sampling state...')
      sampling_state = compiled_release_fn(
          sampling_state, completed_mask | jnp.array(is_cancelled)
      )

      for seq in completed_seqs:
        seq = {key: value for key, value in seq.items()}
        seq['output_text'] = sampling_lib.chunks_as_text(
            self.input_processor.decode(
                seq['tokens'][seq['input_len'] :].tolist()
            )
        )
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
          batch[index] = None

  def thread(
      self,
      stop_event: threading.Event,
      error_message_queue: queue.Queue[Exception],
  ) -> threading.Thread:
    """Starts the batcher thread."""

    def _batcher_loop():
      try:
        self.loop(stop_event)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception('Batcher loop failed: %s', e)
        stop_event.set()
        error_message_queue.put(e)

    return threading.Thread(target=_batcher_loop, daemon=True)


@dataclasses.dataclass(frozen=True)
class SimplyService(server_pb2_grpc.SimplyService):
  """The Simple service with batching."""

  batcher: Batcher

  @functools.cached_property
  def stop_event(self) -> threading.Event:
    return threading.Event()

  @functools.cached_property
  def error_message_queue(self) -> queue.Queue[Exception]:
    return queue.Queue[Exception]()

  @functools.cached_property
  def batcher_thread(self) -> threading.Thread:
    return self.batcher.thread(self.stop_event, self.error_message_queue)

  async def Run(
      self, request: struct_pb2.Value, context: grpc.aio.ServicerContext
  ) -> struct_pb2.Value:
    if not self.batcher_thread.is_alive():
      raise ValueError(
          'Batcher is not running, please call self.batcher_thread.start().'
      )

    request = common.struct_pb_to_py(request)
    future_response = asyncio.Future[SimplyServiceResponse]()

    try:
      self.batcher.enqueue(request, future_response)
    except queue.Full:
      future_response.set_result(
          SimplyServiceResponse(
              code=grpc.StatusCode.RESOURCE_EXHAUSTED,
              details='Queue is full.',
          )
      )

    def _done_callback(context: grpc.aio.ServicerContext) -> None:
      logging.info('Done callback is called.')
      if context.cancelled():
        future_response.get_loop().call_soon_threadsafe(
            future_response.cancel, 'Future is cancelled.'
        )

    context.add_done_callback(_done_callback)

    response = await future_response
    logging.info('response: %s', response)
    context.set_code(response.code)
    context.set_details(response.details)
    logging.info('response.result: %s', pytree.dump(response.result))
    return common.py_to_struct_pb(response.result)


async def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = config_lib.ExperimentConfigRegistry.get_instance(
      _EXPERIMENT_CONFIG.value
  )

  config_replace_kwargs = {}
  if mesh_shape := _MESH_SHAPE.value:
    mesh_shape = [int(i) for i in mesh_shape]
  else:
    mesh_shape = config_lib.get_default_mesh_shape(config, mode='decode')
  sharding.set_mesh(mesh_shape)
  config_replace_kwargs['mesh_shape'] = mesh_shape

  if vocab_name := _VOCAB_NAME.value:
    config_replace_kwargs['vocab_name'] = vocab_name
  if batch_size := _BATCH_SIZE.value:
    config_replace_kwargs['batch_size'] = batch_size
  if activation_dtype := _ACTIVATION_DTYPE.value:
    config_replace_kwargs['activation_dtype_name'] = activation_dtype
  if checkpoint_dir := _CKPT_DIR.value:
    config_replace_kwargs['init_ckpt_dir'] = checkpoint_dir
    config_replace_kwargs['init_ckpt_step'] = _CKPT_STEP.value
    if (ckpt_format := _CKPT_FORMAT.value) is not None:
      config_replace_kwargs['init_ckpt_format'] = ckpt_format
  page_size = 128
  total_num_pages = (
      (_MAX_SEQ_LEN.value - 1 + page_size - 1) // page_size * _BATCH_SIZE.value
  )
  config_replace_kwargs['global_total_num_pages'] = total_num_pages
  config_replace_kwargs['local_total_num_pages'] = total_num_pages
  config_replace_kwargs['page_size'] = page_size

  if not (lm_format_name := _LM_FORMAT.value):
    lm_format_name = getattr(config, 'lm_format_name')
  decoding_sharding_config = getattr(config, 'decoding_sharding_config', None)
  if decoding_sharding_config is None:
    decoding_sharding_config = config.sharding_config.to_decoding_sharding()

  config = dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
      sharding_config=decoding_sharding_config,
      **config_replace_kwargs,
  )

  service = SimplyService(
      batcher=Batcher(
          config=config,
          lm_format=lm_format_lib.LMFormatRegistry.get_instance(lm_format_name),
      ),
  )

  def _init_fn():
    params = service.batcher.model.init(jax.random.key(0))
    if _ACTIVATION_DTYPE.value == 'bfloat16':
      params = jax.tree_util.tree_map(
          lambda x: jnp.astype(x, jnp.bfloat16), params
      )
    return {'params': params}

  experiment_helper.set_notes('Loading checkpoint ...')
  abstract_state = core_common.eval_abstract_output(_init_fn)
  model_state = checkpoint_lib.load_checkpoint_from_dir(
      config.init_ckpt_dir,
      abstract_state,
      config.init_ckpt_step,
      ckpt_format=config.init_ckpt_format,
  )
  service.batcher.update_params(model_state['params'])
  service.batcher_thread.start()

  server = grpc.aio.server()
  health_pb2_grpc.add_HealthServicer_to_server(
      health.aio.HealthServicer(), server
  )
  server_pb2_grpc.add_SimplyServiceServicer_to_server(service, server)

  service_names = (
      health_pb2.Health.DESCRIPTOR.full_name,
      server_pb2.SimplyService.DESCRIPTOR.full_name,
      reflection.SERVICE_NAME,
  )
  reflection.enable_server_reflection(service_names, server)
  port = server.add_insecure_port(f'[::]:{_SIMPLY_PORT.value}')
  logging.info('listening %s', port)
  await server.start()
  while not service.stop_event.is_set():
    await asyncio.sleep(1)

  while service.error_message_queue:
    raise service.error_message_queue.get()


if __name__ == '__main__':
  app.run(lambda argv: asyncio.run(main(argv)))

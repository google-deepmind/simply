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
r"""Simply gRPC server that uses vanilla decoding method.

Start server example:

python -m simply.serving.vanilla_server -- \
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
from simply.utils import lm_format as lm_format_lib
from simply.utils import pytree
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

_INTERMEDIATE_DECODE_STEPS = flags.DEFINE_integer(
    'intermediate_decode_steps',
    4096,
    'Intermediate decode steps for the model. Though it is optional, it is'
    ' recommended to set it to frame the jit into limited programs.',
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

  batch_size: int = 8
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
  def lm_interface(self) -> model_lib.LMInterface:
    return model_lib.LMInterface(
        self.model,
        params=None,
        input_processor=self.input_processor,
        default_sampling_params=model_lib.SamplingParams(
            max_seq_len=_MAX_SEQ_LEN.value,
            max_decode_steps=_MAX_DECODE_STEPS.value,
            intermediate_decode_steps=_INTERMEDIATE_DECODE_STEPS.value,
        ),
    )

  @functools.cached_property
  def queue(
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
    self.queue.put((request, future), timeout=self.max_queue_timeout)

  def loop(self, stop_event: threading.Event):
    """The batcher loop."""
    sharding.set_mesh(self.config.mesh_shape)
    seed = int(time.time() * 1000)
    seed = multihost_utils.broadcast_one_to_all(seed)
    prng_key = jax.random.key(seed=seed)
    while not stop_event.is_set():
      batch = []
      batched_inputs = []
      while len(batch) < self.batch_size:
        try:
          request, future = self.queue.get(
              timeout=self.max_queue_timeout if batch else None
          )
          logging.info('request: %s', request)
        except queue.Empty:
          break

        try:
          if pytree.tree_is_sequence(request):
            request = self.lm_format.format(request)
          input_chunks = sampling_lib.input_as_chunks(request)
          logging.info('input_chunks: %s', input_chunks)
        except Exception as e:  # pylint: disable=broad-except
          logging.exception('Failed to process input: %s', e)
          future.set_result(
              SimplyServiceResponse(
                  code=grpc.StatusCode.INVALID_ARGUMENT,
                  details=str(e),
              )
          )
          continue
        batched_inputs.append(input_chunks)
        batch.append((request, future))

      logging.info('batch size: %d', len(batch))

      prng_key, subkey = jax.random.split(prng_key)
      sampling_outputs = self.lm_interface.generate(
          batched_inputs,
          params=self.model_state['params'],
          prng_key=subkey,
          scoring_inputs=False,
          batch_size=_BATCH_SIZE.value,
      )
      assert len(sampling_outputs) == len(batched_inputs)

      for (_, future), so in zip(
          batch, sampling_outputs[: len(batch)], strict=True
      ):
        future.get_loop().call_soon_threadsafe(
            future.set_result,
            SimplyServiceResponse(
                code=grpc.StatusCode.OK,
                result=dict(output_text=so[0].output_text),
            ),
        )


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
  def batcher_thread(self):

    def _batcher_loop():
      try:
        self.batcher.loop(self.stop_event)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception('Batcher loop failed: %s', e)
        self.stop_event.set()
        self.error_message_queue.put(e)

    return threading.Thread(target=_batcher_loop)

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

    response = await future_response
    logging.info('response: %s', response)
    context.set_code(response.code)
    context.set_details(response.details)
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

  decoding_sharding_config = getattr(config, 'decoding_sharding_config', None)
  if decoding_sharding_config is None:
    decoding_sharding_config = config.sharding_config.to_decoding_sharding()

  if not (lm_format_name := _LM_FORMAT.value):
    lm_format_name = getattr(config, 'lm_format_name')

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
          batch_size=_BATCH_SIZE.value,
      ),
  )

  def _init_fn():
    params = service.batcher.model.init(jax.random.key(0))
    if _ACTIVATION_DTYPE.value == 'bfloat16':
      params = jax.tree_util.tree_map(
          lambda x: jnp.astype(x, jnp.bfloat16), params
      )
    return {'params': params}

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

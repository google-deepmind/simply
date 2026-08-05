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

from absl import app
from absl import flags
from absl import logging
import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from simply import config_lib
from simply import data_lib  # pylint: disable=unused-import
from simply.serving import common
from simply.serving import common_flags
from simply.serving import page_batcher
from simply.serving import server_pb2
from simply.serving import server_pb2_grpc
from simply.serving import struct_pb2
from simply.utils import checkpoint_lib
from simply.utils import common as core_common
from simply.utils import experiment_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import pytree
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sharding


_SIMPLY_PORT = flags.DEFINE_integer(
    'simply_port', None, 'Port to listen on.', required=True
)

PyTree = core_common.PyTree


def set_notes(notes: str):
  experiment_helper.set_notes(
      notes, should_set=experiment_helper.is_primary_task()
  )


SimplyServiceResponse = common.SimplyServiceResponse


@dataclasses.dataclass(frozen=True)
class SimplyService(server_pb2_grpc.SimplyService):
  """The Simple service with batching."""

  batcher: page_batcher.Batcher

  @functools.cached_property
  def stop_event(self) -> threading.Event:
    return threading.Event()

  @functools.cached_property
  def error_message_queue(self) -> queue.Queue[Exception]:
    return queue.Queue[Exception](1)

  @functools.cached_property
  def batcher_thread(self) -> threading.Thread:
    return self.batcher.thread(self.stop_event, self.error_message_queue)

  async def Run(  # pyrefly: ignore[bad-override]
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

  experiment_helper.setup_work_unit()

  config = config_lib.ExperimentConfigRegistry.get_instance(
      common_flags.EXPERIMENT_CONFIG.value  # type: ignore
  )

  config_replace_kwargs = {}
  if mesh_shape := common_flags.MESH_SHAPE.value:  # type: ignore
    mesh_shape = [int(i) for i in mesh_shape]
  else:
    mesh_shape = config_lib.get_default_mesh_shape(config, mode='decode')
  config_replace_kwargs['mesh_shape'] = mesh_shape

  if vocab_name := common_flags.VOCAB_NAME.value:
    config_replace_kwargs['vocab_name'] = vocab_name
  if batch_size := common_flags.BATCH_SIZE.value:
    config_replace_kwargs['batch_size'] = batch_size
  if activation_dtype := common_flags.ACTIVATION_DTYPE.value:
    config_replace_kwargs['activation_dtype_name'] = activation_dtype
  if checkpoint_dir := common_flags.CKPT_DIR.value:
    config_replace_kwargs['init_ckpt_dir'] = checkpoint_dir
    config_replace_kwargs['init_ckpt_step'] = common_flags.CKPT_STEP.value
    if (ckpt_format := common_flags.CKPT_FORMAT.value) is not None:
      config_replace_kwargs['init_ckpt_format'] = ckpt_format

  if not (lm_format_name := common_flags.LM_FORMAT.value):
    lm_format_name = getattr(config, 'lm_format_name')

  decoding_sharding_config = (
      getattr(config, 'decoding_sharding_config', None)
      or config.sharding_config.to_decoding_sharding()
  )
  config_replace_kwargs['sharding_config'] = decoding_sharding_config
  sharding.set_mesh(
      mesh_shape, axis_names=decoding_sharding_config.mesh_axis_names
  )

  page_size = common_flags.PAGE_SIZE.value
  seq_partition = sharding.get_partition_axis(
      decoding_sharding_config.attn_activation_partition, 1
  )
  num_seq_shards = sharding.get_partition_size(seq_partition)
  global_total_num_pages = (
      rpa.max_num_pages_per_seq_per_shard(
          common_flags.MAX_SEQ_LEN.value, page_size, None, num_seq_shards
      )
      * common_flags.BATCH_SIZE.value
      * num_seq_shards
  )
  local_total_num_pages = (
      rpa.max_num_pages_per_seq_per_shard(
          common_flags.MAX_SEQ_LEN.value,
          page_size,
          config.window_size,
          num_seq_shards,
      )
      * common_flags.BATCH_SIZE.value
      * num_seq_shards
  )
  logging.info(
      'global_total_num_pages=%d, local_total_num_pages=%d',
      global_total_num_pages,
      local_total_num_pages,
  )
  config_replace_kwargs['global_total_num_pages'] = global_total_num_pages
  config_replace_kwargs['local_total_num_pages'] = local_total_num_pages
  config_replace_kwargs['page_size'] = page_size

  config = dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
      **config_replace_kwargs,
  )
  service = SimplyService(
      batcher=page_batcher.Batcher(
          config=config,
          lm_format=lm_format_lib.LMFormatRegistry.get_instance(lm_format_name),
          max_seq_len=common_flags.MAX_SEQ_LEN.value,
          max_decode_steps=common_flags.MAX_DECODE_STEPS.value,
          temperature=common_flags.TEMPERATURE.value,
          top_k=common_flags.TOP_K.value,
          top_p=common_flags.TOP_P.value,
          intermediate_steps=common_flags.INTERMEDIATE_STEPS.value,
          response_asap=common_flags.RESPONSE_ASAP.value,
          enable_prefix_caching=common_flags.ENABLE_PREFIX_CACHING.value,
          max_num_issue_tokens=common_flags.MAX_NUM_ISSUE_TOKENS.value,
      ),
  )

  set_notes('Compiling ...')
  _ = service.batcher.compiled_decode_fn
  _ = service.batcher.compiled_push_fn
  _ = service.batcher.compiled_release_fn

  checkpoint_path = checkpoint_lib.get_checkpoint_path(
      config.init_ckpt_dir, config.init_ckpt_step
  )
  set_notes(f'Loading checkpoint from {checkpoint_path} ...')
  service.batcher.update_params_from_checkpoint_path(
      checkpoint_path,
  )
  set_notes('Ready')
  service.batcher_thread.start()

  if experiment_helper.is_primary_task():
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

  if not service.error_message_queue.empty():
    raise service.error_message_queue.get()


if __name__ == '__main__':
  app.run(lambda argv: asyncio.run(main(argv)))

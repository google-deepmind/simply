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
"""Decoding evaluation on a dataset."""

import asyncio
from collections.abc import Mapping, Sequence
import dataclasses
import json
import logging
import queue
import re
import threading
import time
from typing import Any

from absl import app
from absl import flags
from etils import epath
import grain
import grpc
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
from simply import config_lib
from simply import data_lib
from simply.serving import page_server
from simply.utils import checkpoint_lib
from simply.utils import common
from simply.utils import evaluation_lib
from simply.utils import experiment_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import pytree
from simply.utils import sharding


_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir', None, 'Path to the experiment directory.', required=True
)

_EVALUATION = flags.DEFINE_string(
    'evaluation', None, 'Evaluation to run.', required=True
)

_DATASOURCE_NAME = flags.DEFINE_string(
    'datasource_name',
    None,
    'Name of the dataset to evaluate on.',
    required=True,
)

_SAVE_EVERY_N = flags.DEFINE_integer(
    'save_every_n', 10, 'Save the history every n examples.'
)

_TEMPERATURE = flags.DEFINE_float(
    'temperature', 0.0, 'Temperature for sampling.'
)

_TOP_K = flags.DEFINE_integer('top_k', -1, 'Top-k for sampling.')

_TOP_P = flags.DEFINE_float('top_p', 1.0, 'Top-p for sampling.')

_N_REPEATS = flags.DEFINE_integer(
    'n_repeats', 1, 'Number of times to repeat the dataset.'
)


# pylint: disable=protected-access


def get_last_file(directory: epath.PathLike, pattern: str) -> epath.Path | None:
  """Returns the last file that matches the pattern."""
  last_file = None
  last_id = None
  directory = epath.Path(directory)
  for f in directory.iterdir():
    if m := re.fullmatch(pattern, f.name):
      current_id = int(m.group(1))
      if last_id is None or current_id > last_id:
        last_id = current_id
        last_file = f
  return last_file


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  experiment_helper.setup_work_unit()

  helper = experiment_helper.ExperimentHelper(
      experiment_dir=_EXPERIMENT_DIR.value
  )

  config = config_lib.ExperimentConfigRegistry.get_instance(
      page_server._EXPERIMENT_CONFIG.value
  )

  config_replace_kwargs = {}
  if mesh_shape := page_server._MESH_SHAPE.value:
    mesh_shape = [int(i) for i in mesh_shape]
  else:
    mesh_shape = config_lib.get_default_mesh_shape(config, mode='decode')
  sharding.set_mesh(mesh_shape)
  config_replace_kwargs['mesh_shape'] = mesh_shape

  if vocab_name := page_server._VOCAB_NAME.value:
    config_replace_kwargs['vocab_name'] = vocab_name
  if batch_size := page_server._BATCH_SIZE.value:
    config_replace_kwargs['batch_size'] = batch_size
  if activation_dtype := page_server._ACTIVATION_DTYPE.value:
    config_replace_kwargs['activation_dtype_name'] = activation_dtype
  if checkpoint_dir := page_server._CKPT_DIR.value:
    config_replace_kwargs['init_ckpt_dir'] = checkpoint_dir
    config_replace_kwargs['init_ckpt_step'] = page_server._CKPT_STEP.value
    if (ckpt_format := page_server._CKPT_FORMAT.value) is not None:
      config_replace_kwargs['init_ckpt_format'] = ckpt_format
  page_size = 128
  total_num_pages = (
      (page_server._MAX_SEQ_LEN.value - 1 + page_size - 1)
      // page_size
      * page_server._BATCH_SIZE.value
  )
  config_replace_kwargs['global_total_num_pages'] = total_num_pages
  config_replace_kwargs['local_total_num_pages'] = total_num_pages
  config_replace_kwargs['page_size'] = page_size

  decoding_sharding_config = getattr(config, 'decoding_sharding_config', None)
  if decoding_sharding_config is None:
    decoding_sharding_config = config.sharding_config.to_decoding_sharding()

  if not (lm_format_name := page_server._LM_FORMAT.value):
    lm_format_name = getattr(config, 'lm_format_name')

  config = dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
      sharding_config=decoding_sharding_config,
      **config_replace_kwargs,
  )

  experiment_dir = _EXPERIMENT_DIR.value
  if not experiment_dir:
    raise ValueError('Must specify --experiment_dir.')
  experiment_dir = epath.Path(experiment_dir)

  batcher = page_server.Batcher(
      config=config,
      lm_format=lm_format_lib.LMFormatRegistry.get_instance(lm_format_name),
  )
  helper.save_config_info(config, config.sharding_config)
  experiment_helper.set_notes('Initializing model.')

  def _init_fn():
    params = batcher.model.init(jax.random.key(0))
    if page_server._ACTIVATION_DTYPE.value == 'bfloat16':
      params = jax.tree_util.tree_map(
          lambda x: jnp.astype(x, jnp.bfloat16), params
      )
    return {'params': params}

  experiment_helper.set_notes('Loading checkpoint ...')
  abstract_state = common.eval_abstract_output(_init_fn)
  model_state = checkpoint_lib.load_checkpoint_from_dir(
      config.init_ckpt_dir,
      abstract_state,
      config.init_ckpt_step,
      ckpt_format=config.init_ckpt_format,
  )
  batcher.update_params(model_state['params'])
  stop_event = threading.Event()
  error_message_queue = queue.Queue[Exception]()
  batcher_thread = batcher.thread(stop_event, error_message_queue)
  batcher_thread.start()

  evaluation = evaluation_lib.EvaluationRegistry.get_instance(_EVALUATION.value)

  datasource = data_lib.DataSourceRegistry.get_instance(_DATASOURCE_NAME.value)
  dataset = grain.MapDataset.source(datasource.load())
  dataset = dataset.repeat(_N_REPEATS.value)
  num_total_examples = len(dataset)

  dataiter_state = None
  num_past_examples = 0
  iter_state_path = get_last_file(experiment_dir, r'iter_state_(\d+)\.json')
  if iter_state_path is not None:
    iter_state = pytree.load_pytree_from(iter_state_path)
    dataiter_state = iter_state['dataiter']
    num_past_examples = iter_state['num_past_examples']

  logging.info('dataiter_state=%s', dataiter_state)
  logging.info('num_past_examples=%d', num_past_examples)
  experiment_helper.set_notes(
      f'Starting to decode from example {num_past_examples}.'
  )

  async def query_and_evaluate(example: Mapping[str, Any]) -> Mapping[str, Any]:
    """Queries the server and evaluates the response."""

    request = evaluation.get_messages(example)
    future_response = asyncio.Future[page_server.SimplyServiceResponse]()
    batcher.enqueue(request, future_response)
    response = await future_response
    logging.info('response=%s', response)
    assert response.code == grpc.StatusCode.OK
    response = response.result
    responsed_example = example | dict(lm_request=request, lm_response=response)
    result = evaluation.evaluate(responsed_example, response['output_text'])
    return responsed_example | result

  dataset = dataset.map(lambda x: asyncio.run(query_and_evaluate(x)))
  dataset = dataset.to_iter_dataset(
      grain.ReadOptions(
          num_threads=page_server._BATCH_SIZE.value * 2,
          prefetch_buffer_size=4096,
      )
  )
  dataiter = dataset.__iter__()

  if dataiter_state is not None:
    dataiter.set_state(dataiter_state)

  experiment_helper.set_notes(
      f'Starting to decode from example {num_past_examples}.'
  )
  start_time = time.time()
  num_saved_examples = num_past_examples
  history = []
  for example in dataiter:
    num_past_examples += 1
    logging.info('Completed %d examples', num_past_examples)
    generation_time = time.time() - start_time
    history.append(example | dict(lm_avg_generation_time=generation_time))

    # Save the history if we have processed `save_every_n` examples or we have
    # finished all the epochs.
    if (
        num_past_examples - num_saved_examples >= _SAVE_EVERY_N.value
        or num_past_examples >= num_total_examples
    ):
      logging.info('Saving history %d', num_past_examples)
      history_path = experiment_dir / f'history_{num_past_examples}.jsonl'
      history_tmp_path = history_path.with_suffix('.tmp')
      total_generation_time = 0.0
      with history_tmp_path.open('w') as f:
        for example in history:
          total_generation_time += example['lm_avg_generation_time']
          print(f'{example=}')
          json.dump(pytree.dump(example), f)
          f.write('\n')
      history_path.rmtree(missing_ok=True)
      history_tmp_path.rename(history_path)

      iter_state_path = (
          experiment_dir / f'iter_state_{num_past_examples}.json'
      )
      iter_state_tmp_path = iter_state_path.with_suffix('.tmp')
      pytree.save_pytree_to(
          dict(
              dataiter=dataiter.get_state(),
              num_past_examples=num_past_examples,
          ),
          iter_state_tmp_path,
      )
      iter_state_tmp_path.rename(iter_state_path)

      avg_generation_time = total_generation_time / (
          num_past_examples - num_saved_examples
      )
      helper.set_notes(
          f'Completed {num_past_examples}/{num_total_examples} examples,'
          f' {avg_generation_time:.2f} s/example'
      )
      history = []
      num_saved_examples = num_past_examples

    start_time = time.time()

  def _stats_history(history_path: epath.PathLike) -> Mapping[str, int | float]:
    correct = 0
    total = 0
    total_generation_time = 0.0
    history_path = epath.Path(history_path)
    with history_path.open() as f:
      for x in f:
        example = pytree.load(json.loads(x))
        correct += example['correct']
        total_generation_time += example['lm_avg_generation_time']
        total += 1
    return dict(
        correct=correct,
        total=total,
        total_generation_time=total_generation_time,
    )

  async def _stats_all_history() -> Mapping[str, int | float]:
    history_paths = experiment_dir.glob('history_*.jsonl')
    stat_futures = []
    for history_path in history_paths:
      # Validate the name of history paths.
      logging.info('Loading history_path=%s', history_path)
      stat_futures.append(asyncio.to_thread(_stats_history, history_path))
    results = {}
    for stat_future in asyncio.as_completed(stat_futures):
      stat = await stat_future
      logging.info('stat=%s', stat)
      for k, v in stat.items():
        if k not in results:
          results[k] = v
        else:
          results[k] += v
    return results

  if experiment_helper.is_primary_process():
    results = asyncio.run(_stats_all_history())
    logging.info('results=%s', results)

    correct = results['correct']
    total = results['total']
    total_generation_time = results['total_generation_time']

    experiment_helper.set_notes(
        f'Finished: accuracy is {correct=}/{total=} ='
        f' {correct / total * 100:.2f}%,'
        f' {total_generation_time / total:.2f} s/example'
    )


if __name__ == '__main__':
  app.run(main)

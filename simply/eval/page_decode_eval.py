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
import functools
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
from simply import config_lib
from simply import data_lib
from simply.serving import common as serving_common
from simply.serving import common_flags
from simply.serving import page_batcher
from simply.utils import checkpoint_lib
from simply.utils import common
from simply.utils import evaluation_lib
from simply.utils import experiment_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import pytree
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sharding

PyTree = common.PyTree


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


_N_REPEATS = flags.DEFINE_integer(
    'n_repeats', 1, 'Number of times to repeat the dataset.'
)

_DATA_SHARD_COUNT = flags.DEFINE_integer(
    'data_shard_count',
    1,
    'Number of data shards to split the dataset into (>1 enables sharding).',
)

_DATA_SHARD_INDEX = flags.DEFINE_integer(
    'data_shard_index',
    0,
    'Which data shard this job processes, in [0, data_shard_count-1].',
)

_NUM_EVAL_THREADS = flags.DEFINE_integer(
    'num_eval_threads',
    None,
    'Number of threads to use for evaluation. Defaults to batch_size * 2.',
)

_SAVE_FULL_INFO = flags.DEFINE_bool(
    'save_full_info',
    False,
    'If True, save full information in history, else only save critical info.',
)

_SEED = flags.DEFINE_integer(
    'seed',
    None,
    'Decode-sampling PRNG seed. None (default) seeds from wall-clock time'
    ' (each run differs). Set a fixed int to seed sampling for eval sweeps;'
    ' note batch decode is not fully reproducible even with a fixed seed'
    ' (async queueing). Recorded in final_result.json as "seed".',
)


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

  config = config_lib.ExperimentConfigRegistry.get_instance(
      common_flags.EXPERIMENT_CONFIG.value
  )

  config_replace_kwargs = {}
  if mesh_shape := common_flags.MESH_SHAPE.value:
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
  if common_flags.FFN_WEIGHT_QUANT.value is not None:
    config_replace_kwargs['ffn_weight_quant'] = (
        common_flags.FFN_WEIGHT_QUANT.value
    )
  if common_flags.KV_CACHE_QUANT.value is not None:
    config_replace_kwargs['kv_cache_quant'] = common_flags.KV_CACHE_QUANT.value

  config = dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
      **config_replace_kwargs,
  )

  experiment_dir = _EXPERIMENT_DIR.value
  if not experiment_dir:
    raise ValueError('Must specify --experiment_dir.')
  helper = experiment_helper.ExperimentHelper(
      experiment_dir=experiment_dir,
      is_primary=experiment_helper.is_primary_task(),
  )
  experiment_dir = epath.Path(helper.experiment_dir)
  helper.save_config_info(config, config.sharding_config)

  batcher = page_batcher.Batcher(
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
      decode_seed=_SEED.value,
  )

  helper.set_notes('Compiling ...')
  _ = batcher.compiled_decode_fn
  _ = batcher.compiled_prefill_fn
  _ = batcher.compiled_push_fn
  _ = batcher.compiled_release_fn
  if batcher.prefix_cache is not None:
    _ = batcher.compiled_inject_chunk_fn
    _ = batcher.compiled_extract_chunk_fn

  checkpoint_path = checkpoint_lib.get_checkpoint_path(
      config.init_ckpt_dir, config.init_ckpt_step
  )
  helper.set_notes(f'Loading checkpoint from {checkpoint_path} ...')
  batcher.update_params_from_checkpoint_path(checkpoint_path)

  stop_event = threading.Event()
  error_message_queue = queue.Queue[Exception]()
  batcher_thread = batcher.thread(stop_event, error_message_queue)
  batcher_thread.start()

  if not experiment_helper.is_primary_task():
    batcher_thread.join()
    if not error_message_queue.empty():
      raise error_message_queue.get()
    return

  evaluation = evaluation_lib.EvaluationRegistry.get_instance(_EVALUATION.value)

  datasource = data_lib.DataSourceRegistry.get_instance(_DATASOURCE_NAME.value)
  dataset = data_lib.get_data_source(datasource)
  if _DATA_SHARD_COUNT.value > 1:
    # Shard the dataset: strided slice so shard i gets examples i, i+count,
    # i+2*count, ... (grain's documented sharding idiom, dataset.py:
    # ds[shard_index::shard_count]). Union of all shards covers the dataset
    # exactly once with no overlap.
    dataset = dataset[_DATA_SHARD_INDEX.value :: _DATA_SHARD_COUNT.value]
    logging.info(
        'DATA SHARD %d/%d -> %d examples (strided)',
        _DATA_SHARD_INDEX.value,
        _DATA_SHARD_COUNT.value,
        len(dataset),
    )
  dataset = dataset.repeat(_N_REPEATS.value)
  num_total_examples = len(dataset)

  dataiter_state = None
  num_past_examples = 0
  total_generation_time = 0.0
  iter_state_path = get_last_file(experiment_dir, r'iter_state_(\d+)\.json')
  if iter_state_path is not None:
    iter_state = pytree.load_pytree_from(iter_state_path)
    dataiter_state = iter_state['dataiter']
    num_past_examples = iter_state['num_past_examples']
    total_generation_time = iter_state['total_generation_time']

  logging.info('dataiter_state=%s', dataiter_state)
  logging.info('num_past_examples=%d', num_past_examples)
  helper.set_notes(f'Starting to decode from example {num_past_examples}.')

  async def model_fn(
      messages: Sequence[Mapping[str, Any]], *, index: int = 0
  ) -> Mapping[str, Any]:
    """Posts messages to the in-process batcher; returns the response dict."""
    # DEBUG: phase markers to localize a hang to one span in the path
    # do_POST -> text_model_fn -> model_fn -> batcher.enqueue -> await future.
    # See chatty-bot run 9nkpcrhw for context.
    logging.info(
        '[page_decode_eval.model_fn] ENTER index=%d num_messages=%d',
        index,
        len(messages),
    )
    # `__index__` lets the batcher log per-example timing; it must be on
    # the first message (the framework strips it before tokenisation).
    # Copy each message to a fresh dict so we never mutate the caller's data.
    request: list[dict[str, Any]] = [dict(m) for m in messages]
    if request:
      request[0]['__index__'] = index
    future_response = asyncio.Future[serving_common.SimplyServiceResponse]()
    logging.info(
        '[page_decode_eval.model_fn] about to enqueue index=%d'
        ' (queue depth=%d/%d)',
        index,
        batcher.request_queue.qsize(),
        batcher.max_queue_size,
    )
    batcher.enqueue(request, future_response)
    logging.info(
        '[page_decode_eval.model_fn] enqueued index=%d; awaiting future',
        index,
    )
    response = await future_response
    logging.info(
        '[page_decode_eval.model_fn] future resolved index=%d code=%s',
        index,
        response.code,
    )
    assert response.code == grpc.StatusCode.OK
    return response.result

  async def query_and_evaluate(
      index: int, example: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    """Queries the server and evaluates the response.

    If the registered evaluation defines an async `evaluate_async(example,
    model_fn)` method, we dispatch to it (for multi-turn / agentic evals).
    Otherwise we use the legacy single-turn path: `get_messages` ->
    `batcher.enqueue` -> `evaluate(response_text)`.

    Args:
      index: Zero-based example index within the eval split; threaded into
        `model_fn` so the batcher can log per-example timing.
      example: The raw example dict from the dataset iterator.

    Returns:
      A dict shaped like the legacy `responsed_example`: the input
      `example` merged with at minimum `lm_request`, `lm_response`, and
      any verification fields the evaluation produced. For the agentic
      path the `lm_request` / `lm_response` are populated inside
      `evaluate_async`; for the legacy single-turn path they are added
      here from the synchronous enqueue + evaluate flow.
    """
    if getattr(evaluation, 'in_sandbox_loop', False):
      assert getattr(evaluation, 'evaluate_async')
      logging.info('agentic evaluate_async index=%s', index)
      bound_model_fn = functools.partial(model_fn, index=index)
      example = dict(example)
      example['__index__'] = index
      result = await evaluation.evaluate_async(example, bound_model_fn)
      # evaluate_async returns a dict shaped like the legacy responsed_example
      # (must include 'lm_request' + 'lm_response' for history.jsonl).
      return example | result

    request = evaluation.get_messages(example)
    assert pytree.tree_is_sequence(request)
    logging.info('enqueue index=%s', index)
    response = await model_fn(request, index=index)
    logging.info('response=%s', response)
    responsed_example = example | dict(lm_request=request, lm_response=response)  # pyrefly: ignore[unsupported-operation]
    result = evaluation.evaluate(responsed_example, response['output_text'])
    return responsed_example | result

  dataset = dataset.map_with_index(
      lambda i, x: asyncio.run(query_and_evaluate(i, x))
  )
  num_eval_threads = (
      _NUM_EVAL_THREADS.value
      if _NUM_EVAL_THREADS.value is not None
      else common_flags.BATCH_SIZE.value * 2
  )
  dataset = dataset.to_iter_dataset(
      grain.ReadOptions(
          num_threads=num_eval_threads,
          prefetch_buffer_size=common_flags.BATCH_SIZE.value * 8,
      )
  )
  dataiter = dataset.__iter__()

  if dataiter_state is not None:
    dataiter.set_state(dataiter_state)

  helper.set_notes(f'Starting to decode from example {num_past_examples}.')
  start_time = time.time()
  num_saved_examples = num_past_examples
  history = []
  for example in dataiter:
    num_past_examples += 1
    logging.info('Completed %d examples', num_past_examples)
    generation_time = time.time() - start_time
    total_generation_time += generation_time
    history.append(example)

    # Save the history if we have processed `save_every_n` examples or we have
    # finished all the epochs.
    if (
        num_past_examples - num_saved_examples >= _SAVE_EVERY_N.value
        or num_past_examples >= num_total_examples
    ):
      logging.info('Saving history %d', num_past_examples)
      history_path = experiment_dir / f'history_{num_past_examples}.jsonl'
      history_tmp_path = history_path.with_suffix('.tmp')
      with history_tmp_path.open('w') as f:
        for example_to_save in history:
          if not _SAVE_FULL_INFO.value:
            lm_response = example_to_save['lm_response']
            input_len = lm_response['input_len']
            example_to_save = dict(
                lm_request=example_to_save['lm_request'],
                output_text=lm_response['output_text'],
                input_len=input_len,
                output_len=len(lm_response['tokens']) - input_len,
                correct=example_to_save['correct'],
            )
          json.dump(pytree.dump(example_to_save), f)
          f.write('\n')
      history_path.rmtree(missing_ok=True)
      history_tmp_path.rename(history_path)

      iter_state_path = experiment_dir / f'iter_state_{num_past_examples}.json'
      iter_state_tmp_path = iter_state_path.with_suffix('.tmp')
      pytree.save_pytree_to(
          dict(
              dataiter=dataiter.get_state(),
              num_past_examples=num_past_examples,
              total_generation_time=total_generation_time,
          ),
          iter_state_tmp_path,
      )
      iter_state_tmp_path.rename(iter_state_path)

      avg_generation_time = total_generation_time / num_past_examples
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
    history_path = epath.Path(history_path)
    with history_path.open() as f:
      for x in f:
        example = pytree.load(json.loads(x))
        correct += example['correct']
        total += 1
    return dict(
        correct=correct,
        total=total,
    )

  stop_event.set()

  async def _stats_all_history() -> Mapping[str, int | float]:
    history_paths = experiment_dir.glob('history_*.jsonl')  # pyrefly: ignore[missing-attribute]
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

  results = asyncio.run(_stats_all_history())
  logging.info('results=%s', results)
  correct = results['correct']
  total = results['total']
  helper.set_notes(
      f'Finished: accuracy is {correct=}/{total=} ='
      f' {correct / total * 100:.2f}%,'
      f' {total_generation_time / total:.2f} s/example'
  )

  final_result = {
      'accuracy': correct / total,
      'correct': correct,
      'total': total,
      'avg_generation_time': total_generation_time / total,
      'seed': _SEED.value,
  }
  logging.info('final_result=%s', final_result)
  experiment_dir = epath.Path(helper.experiment_dir)
  with (experiment_dir / 'final_result.json').open('w') as f:
    f.write(json.dumps(final_result, indent=2))

  batcher_thread.join()


if __name__ == '__main__':
  app.run(main)

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
"""Decoding evaluation on a dataset."""

import asyncio
from collections.abc import Mapping, Sequence
import functools
import json
import logging
import re
import time
from typing import Any

from absl import app
from absl import flags
from etils import epath
import grain
import grpc
from simply import data_lib
from simply.eval import model_backends
from simply.serving import common as serving_common
from simply.serving import server_pb2_grpc
from simply.utils import evaluation_lib
from simply.utils import experiment_helper
from simply.utils import pytree

_SERVER_ADDRESS = flags.DEFINE_string(
    'server_address',
    None,
    'Address of the server to evaluate on.',
    required=True,
)

_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir', None, 'Path to the experiment directory.', required=True
)

_MAX_DECODE_STEPS = flags.DEFINE_integer(
    'max_decode_steps',
    None,
    'Per-call output token cap, forwarded to the backend selected by'
    ' --server_address (a backend may map it to its own field or ignore'
    ' it; the built-in gRPC backend ignores it). Unset (default) means'
    ' "defer to the server\'s per-model default"; set a positive value to'
    ' override.',
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

_NUM_EVAL_THREADS = flags.DEFINE_integer(
    'num_eval_threads', 128,
    'Across-example parallelism: maximum number of examples processed in'
    ' parallel by the dataset iterator (passed as grain.ReadOptions.num_threads'
    ' on the `map_with_index(asyncio.run(query_and_evaluate))` stage).'
    ' Effective concurrency in production is bounded below by the number of'
    ' sandbox replicas, since each example needs a sandbox.',
)

# Backend selection is plugin-driven via `model_backends.ModelBackendRegistry`:
# a `<scheme>:...` `--server_address` looks up `<scheme>` in the registry and
# delegates URI parsing + model_fn construction to the registered factory.
# When no scheme matches (empty address, schemeless address, or unknown
# scheme) we fall through to the built-in legacy gRPC stub path. See
# `model_backends.py` for the plugin contract.

_TEMPERATURE = flags.DEFINE_float(
    'temperature', None,
    'Optional sampling temperature. Forwarded to the backend selected by'
    ' --server_address; backends may ignore it.',
)

_TOP_P = flags.DEFINE_float(
    'top_p', None,
    'Optional top-p. Forwarded to the selected backend; may be ignored.',
)

_TOP_K = flags.DEFINE_integer(
    'top_k', None,
    'Optional top-k. Forwarded to the selected backend; may be ignored.',
)


def simply_service_stub(
    server_address: str | None = None,
) -> server_pb2_grpc.SimplyServiceStub:
  """Returns a Simply gRPC stub for `server_address`.

  When `server_address` is empty or None, falls back to work-unit-0 of the
  current experiment (legacy behavior preserved for callers that omit the
  flag and colocate the server in the same launcher experiment).

  Args:
    server_address: BNS / host:port of the Simply server. When falsy, falls
      back to work-unit-0 of the current experiment.
  """
  channel = grpc.insecure_channel(server_address)
  logging.info('Connecting to server %s', server_address)
  grpc.channel_ready_future(channel).result()
  logging.info('Channel to server %s is ready', server_address)
  return server_pb2_grpc.SimplyServiceStub(channel)


@model_backends.register_default
def _simply_grpc_factory(
    server_address: str,
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_decode_steps: int | None,
) -> model_backends.ModelFn:
  """Built-in default `model_backends` factory: legacy Simply gRPC stub.

  Sampling knobs are accepted for API parity but ignored -- the Simply
  server controls its own sampling. The `server_address` (which may be
  empty / a BNS / a host:port / an unknown-scheme URI) is forwarded
  verbatim to `simply_service_stub`, which preserves the work-unit-0
  fallback when empty.

  Args:
    server_address: Full `--server_address` flag value.
    temperature: Unused; accepted for API parity.
    top_p: Unused; accepted for API parity.
    top_k: Unused; accepted for API parity.
    max_decode_steps: Unused; accepted for API parity.

  Returns:
    An async `model_fn(messages, *, index=0)` that proxies one chat turn to
    the configured Simply gRPC server and returns its response dict.
  """
  del temperature, top_p, top_k, max_decode_steps  # unused
  stub = simply_service_stub(server_address)

  async def model_fn(
      messages: Sequence[Mapping[str, Any]], *, index: int = 0
  ) -> Mapping[str, Any]:
    """Posts messages to the remote Simply server; returns the response dict.

    Args:
      messages: Chat-formatted prompt to send to the server.
      index: Per-example index threaded through to the server for logging
        as `__index__` on the first message.

    Returns:
      The parsed Simply server response dict.
    """
    # `__index__` lets the server log per-example timing; it must be on
    # the first message (the framework strips it before tokenisation).
    # Copy each message to a fresh dict so we never mutate the caller's data.
    request: list[dict[str, Any]] = [dict(m) for m in messages]
    if request:
      request[0]['__index__'] = index
    while True:
      try:
        response = serving_common.struct_pb_to_py(
            await asyncio.to_thread(
                stub.Run, serving_common.py_to_struct_pb(request)
            )
        )
        break
      except grpc.RpcError as e:
        logging.error('Failed to query server: %s', e)
        await asyncio.sleep(5)
    return response

  return model_fn


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

  experiment_dir = _EXPERIMENT_DIR.value
  if not experiment_dir:
    raise ValueError('Must specify --experiment_dir.')
  experiment_dir = epath.Path(experiment_dir)
  experiment_dir.mkdir(parents=True, exist_ok=True)

  evaluation = evaluation_lib.EvaluationRegistry.get_instance(_EVALUATION.value)

  datasource = data_lib.DataSourceRegistry.get_instance(_DATASOURCE_NAME.value)
  dataset = data_lib.get_data_source(datasource)
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

  # ---- Build the per-request `model_fn` once, based on --server_address.
  # `model_backends.dispatch` looks up the scheme; unknown / schemeless /
  # empty addresses fall through to the registered default factory (the
  # built-in Simply gRPC stub, registered above via `@register_default`).
  server_address = _SERVER_ADDRESS.value
  # `or 0` keeps pytype happy about absl's `int | None` flag type; backends
  # then map 0 to `None` semantics themselves.
  model_fn = model_backends.dispatch(
      server_address,
      temperature=_TEMPERATURE.value,
      top_p=_TOP_P.value,
      top_k=_TOP_K.value,
      max_decode_steps=(_MAX_DECODE_STEPS.value or 0) or None,
  )
  assert model_fn is not None, (
      'No model backend resolved for --server_address=%r and no default'
      ' backend is registered.' % server_address
  )
  logging.info('Using model backend from --server_address=%r', server_address)

  experiment_helper.set_notes(
      f'Starting to decode from example {num_past_examples}.'
  )

  async def query_and_evaluate(
      index: int, example: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    """Queries the server and evaluates the response.

    If the registered evaluation defines an async `evaluate_async(example,
    model_fn)` method (and opts into the in-sandbox agentic loop via
    `in_sandbox_loop=True`), we dispatch to it for multi-turn / agentic
    evals. Otherwise we use the legacy single-turn path: `get_messages` ->
    `stub.Run` -> `evaluate(response_text)`.

    Args:
      index: Zero-based example index within the eval split; threaded into
        `model_fn` so the server can log per-example timing.
      example: The raw example dict from the dataset iterator.

    Returns:
      A dict shaped like the legacy `responsed_example`: the input
      `example` merged with at minimum `lm_request`, `lm_response`, and
      any verification fields the evaluation produced. For the agentic
      path the `lm_request` / `lm_response` are populated inside
      `evaluate_async`; for the legacy single-turn path they are added
      here from the synchronous Run + evaluate flow.
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
    responsed_example = example | dict(lm_request=request, lm_response=response)  # pyrefly: ignore[unsupported-operation]
    result = evaluation.evaluate(responsed_example, response['output_text'])
    return responsed_example | result

  dataset = dataset.map_with_index(
      lambda i, x: asyncio.run(query_and_evaluate(i, x))
  )
  dataset = dataset.to_iter_dataset(
      grain.ReadOptions(
          num_threads=_NUM_EVAL_THREADS.value, prefetch_buffer_size=4096,
      )
  )
  dataiter = dataset.__iter__()

  if dataiter_state is not None:
    dataiter.set_state(dataiter_state)

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
        for example in history:
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
              total_generation_time=total_generation_time,
          ),
          iter_state_tmp_path,
      )
      iter_state_tmp_path.rename(iter_state_path)

      avg_generation_time = total_generation_time / num_past_examples
      experiment_helper.set_notes(
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

  results = asyncio.run(_stats_all_history())
  logging.info('results=%s', results)
  correct = results['correct']
  total = results['total']

  experiment_helper.set_notes(
      f'Finished: accuracy is {correct=}/{total=} ='
      f' {correct / total * 100:.2f}%,'
      f' {total_generation_time / total:.2f} s/example'
  )

if __name__ == '__main__':
  app.run(main)

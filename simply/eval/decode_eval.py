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
import re
import time
from typing import Any, cast

from absl import app
from absl import flags
from etils import epath
import grain
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply import data_lib
from simply import model_lib
from simply.utils import checkpoint_lib
from simply.utils import common
from simply.utils import evaluation_lib
from simply.utils import experiment_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import pytree
from simply.utils import sampling_lib
from simply.utils import sharding
from simply.utils import tokenization


_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir', None, 'Path to the experiment directory.', required=True
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
    'ckpt_dir', None, 'Path to the checkpoints directory.', required=True
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

_ACTIVATION_DTYPE = flags.DEFINE_enum(
    'activation_dtype',
    'bfloat16',
    ['float32', 'bfloat16'],
    'Dtype of the activation.',
)

_SEED = flags.DEFINE_integer(
    'seed',
    None,
    'Random seed used for sampling. If None, use a random seed.',
)

_PREFILL_SIZE = flags.DEFINE_integer(
    'prefill_size',
    1024,
    'Prefill size for the model. -1 for inferring from the input length. '
    'Though it is optional, it is recommended to'
    ' set it to frame the jit into limited programs.',
)

_MAX_DECODE_STEPS = flags.DEFINE_integer(
    'max_decode_steps', 32768, 'Max decode steps for the model.'
)

_INTERMEDIATE_DECODE_STEPS = flags.DEFINE_integer(
    'intermediate_decode_steps',
    4096,
    'Intermediate decode steps for the model. Though it is optional, it is'
    ' recommended to set it to frame the jit into limited programs.',
)

_LM_FORMAT = flags.DEFINE_string(
    'lm_format', None, 'Format of the LM to be evaluated.', required=True
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
      _EXPERIMENT_CONFIG.value
  )

  if mesh_shape := _MESH_SHAPE.value:
    mesh_shape = [int(i) for i in mesh_shape]
  else:
    mesh_shape = config_lib.get_default_mesh_shape(config, mode='decode')
  sharding.set_default_mesh_shape(
      mesh_shape=mesh_shape, axis_names=config.sharding_config.mesh_axis_names
  )

  config = dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
      activation_dtype_name=_ACTIVATION_DTYPE.value,
  )

  experiment_dir = _EXPERIMENT_DIR.value
  if not experiment_dir:
    raise ValueError('Must specify --experiment_dir.')
  experiment_dir = epath.Path(experiment_dir)

  model = model_lib.TransformerLM(config)
  helper.save_config_info(config, config.sharding_config, model=model)
  helper.set_notes('Initializing model.')

  def _init_fn():
    params = model.init(jax.random.key(0))
    if _ACTIVATION_DTYPE.value == 'bfloat16':
      params = jax.tree_util.tree_map(
          lambda x: jnp.astype(x, jnp.bfloat16), params
      )
    return {'params': params}

  abstract_state = common.eval_abstract_output(_init_fn)
  state = checkpoint_lib.load_checkpoint_from_dir(
      _CKPT_DIR.value,
      abstract_state,
      _CKPT_STEP.value,
      ckpt_format=_CKPT_FORMAT.value,
  )
  params = state['params']

  if not (vocab_name := _VOCAB_NAME.value):
    vocab_name = config.vocab_name
  tokenizer = tokenization.TokenizerRegistry.get_instance(vocab_name)

  default_sampling_params = model_lib.SamplingParams(
      temperature=_TEMPERATURE.value,
      top_k=_TOP_K.value,
      top_p=_TOP_P.value,
      max_decode_steps=_MAX_DECODE_STEPS.value,
      num_samples=1,
      intermediate_decode_steps=_INTERMEDIATE_DECODE_STEPS.value,
  )

  lm_format = lm_format_lib.LMFormatRegistry.get_instance(_LM_FORMAT.value)
  evaluation = evaluation_lib.EvaluationRegistry.get_instance(_EVALUATION.value)

  input_processor = sampling_lib.create_input_processor(
      config,
      vocab=tokenizer,
      bos_id_override=lm_format.bos_id,
      pad_id_override=lm_format.pad_id,
      extra_eos_tokens=lm_format.extra_eos_tokens,
  )
  lm_interface = model_lib.LMInterface(
      model,
      params=params,
      input_processor=input_processor,
      default_sampling_params=default_sampling_params,
  )

  if (seed := _SEED.value) is None:
    seed = int(time.time() * 1000)
    seed = jax.experimental.multihost_utils.broadcast_one_to_all(seed)
  logging.info('seed=%s', seed)
  prng_key = jax.random.key(seed)

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
    prng_key = jnp.asarray(iter_state['prng_key'])

  dataiter = dataset.batch(
      _BATCH_SIZE.value, batch_fn=lambda x: x
  ).to_iter_dataset()
  dataiter = dataiter.__iter__()

  if dataiter_state is not None:
    dataiter.set_state(dataiter_state)

  logging.info('dataiter_state=%s', dataiter_state)
  logging.info('num_past_examples=%d', num_past_examples)
  logging.info('prng_key=%s', prng_key)

  helper.set_notes(f'Starting to decode from example {num_past_examples}.')

  history = []
  total_generation_time = 0.0
  num_saved_examples = num_past_examples
  for batch in dataiter:
    logging.info(
        'Processing batch %d - %d.',
        num_past_examples,
        num_past_examples + len(batch),
    )

    sampling_inputs = []
    for example in batch:
      sampling_inputs.append(evaluation.get_sampling_input(example, lm_format))

    start_time = time.time()
    prng_key, subkey = jax.random.split(prng_key)
    sampling_outputs = lm_interface.generate(
        sampling_inputs,
        prefill_size=_PREFILL_SIZE.value,
        prng_key=subkey,
        batch_size=_BATCH_SIZE.value,
        scoring_inputs=False,
    )
    generation_time = time.time() - start_time
    logging.info(
        'Generated batch %d - %d, used %.2f seconds.',
        num_past_examples,
        num_past_examples + len(batch),
        generation_time,
    )
    total_generation_time += generation_time

    for example, si, so in zip(
        batch, sampling_inputs, sampling_outputs, strict=True
    ):
      assert len(so) == 1
      rewarded_sample = dict(
          **example, lm_request=si, lm_response=so[0].output_text
      )
      rewarded_sample['lm_avg_generation_time'] = (
          generation_time / _BATCH_SIZE.value
      )
      if experiment_helper.is_primary_process():
        result = evaluation.evaluate(example, rewarded_sample['lm_response'])
        result = cast(Mapping[str, Any], result)
        rewarded_sample.update(result)
        history.append(rewarded_sample)
        logging.info(
            'Evaluated example %d, which has %d input tokens and generated %d'
            ' output tokens',
            num_past_examples,
            len(so[0].input_token_ids),
            len(so[0].output_token_ids),
        )
      num_past_examples += 1

    # Save the history if we have processed `save_every_n` examples or we have
    # finished all the epochs.
    if (
        num_past_examples - num_saved_examples >= _SAVE_EVERY_N.value
        or num_past_examples >= num_total_examples
    ):
      if experiment_helper.is_primary_process():
        logging.info('Saving history %d', num_past_examples)
        history_path = experiment_dir / f'history_{num_past_examples}.jsonl'
        history_tmp_path = history_path.with_suffix('.tmp')
        with history_tmp_path.open('w') as f:
          for example in history:
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
                prng_key=np.asarray(jax.random.key_data(prng_key)),
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
      total_generation_time = 0.0
      num_saved_examples = num_past_examples

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

    helper.set_notes(
        f'Finished: accuracy is {correct=}/{total=} ='
        f' {correct / total * 100:.2f}%,'
        f' {total_generation_time / total:.2f} s/example'
    )


if __name__ == '__main__':
  app.run(main)

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
"""Simply a language model."""

# Force tensorflow to ignore the GPUs to avoid conflicts with jax.
# 1. Import TensorFlow FIRST
import tensorflow as tf
# 2. Configure TF's device visibility IMMEDIATELY
# This hides the GPU from TensorFlow and all libraries that use it.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices([], 'GPU')
    print('GPU is now hidden from TensorFlow.')
  except RuntimeError as e:
    print(e)

import dataclasses
import json
import os
import re
from typing import Sequence

from absl import app
from absl import flags
from absl import logging


from simply import config_lib
from simply import data_lib
from simply import model_lib
from simply import rl_lib  # pylint: disable=unused-import
from simply.utils import common
from simply.utils import pytree

_EXPERIMENT_CONFIG = flags.DEFINE_string(
    'experiment_config', None, 'Name of the experiment config.')

_SHARDING_CONFIG = flags.DEFINE_string(
    'sharding_config', 'GSPMDSharding', 'Name of the sharding config.')

_EXPERIMENT_CONFIG_PATH = flags.DEFINE_string(
    'experiment_config_path',
    None,
    'Path to the experiment config file. If provided, experiment_config has to'
    ' be unset.',
)

_SHARDING_CONFIG_PATH = flags.DEFINE_string(
    'sharding_config_path',
    None,
    'Path to the sharding config file. If provided, sharding_config will be'
    ' ignored.',
)

_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir', '/tmp/simply_lm/', 'Path to save the experiment data.')

_MESH_SHAPE = flags.DEFINE_list(
    'mesh_shape',
    None,
    'Shape for the mesh, comma separated integers, e.g. 1,265,1',
)

_DCN_MESH_SHAPE = flags.DEFINE_list(
    'dcn_mesh_shape',
    None,
    'Shape for the dcn mesh, comma separated integers, e.g. 2,1,1',
)

_DECODING_MESH_SHAPE = flags.DEFINE_list(
    'decoding_mesh_shape',
    None,
    'Shape for the decoding mesh, comma separated integers.',
)


def main(argv: Sequence[str]) -> None:
  del argv
  if experiment_config_path := _EXPERIMENT_CONFIG_PATH.value:
    assert (
        not _EXPERIMENT_CONFIG.value
    ), 'experiment_config and experiment_config_path cannot both be set.'
    with open(experiment_config_path, 'r') as f:
      config_dict = json.load(f)
    if 'code_patch' in config_dict:
      for code, code_context in config_dict['code_patch']:
        print(f'Executing under code context: {code_context}')
        context = globals()[code_context]
        print(f'code:\n{code}')
        exec(code, context.__dict__)  # pylint: disable=exec-used
    config = pytree.load_dataclasses(config_dict)
  else:
    config = config_lib.ExperimentConfigRegistry.get_config(
        _EXPERIMENT_CONFIG.value
    )
  if sharding_config_path := _SHARDING_CONFIG_PATH.value:
    with open(sharding_config_path, 'r') as f:
      sharding_config = pytree.load_dataclasses(json.load(f))
  else:
    sharding_config = config_lib.ShardingConfigRegistry.get_config(
        _SHARDING_CONFIG.value
    )
  if dcn_mesh_shape := _DCN_MESH_SHAPE.value:
    dcn_mesh_shape = [int(i) for i in dcn_mesh_shape]

  if mesh_shape := _MESH_SHAPE.value:
    mesh_shape = [int(i) for i in mesh_shape]
  else:
    mesh_shape = config_lib.get_default_mesh_shape(
        config, mode='train', dcn_mesh_shape=dcn_mesh_shape)

  if decoding_mesh_shape := _DECODING_MESH_SHAPE.value:
    decoding_mesh_shape = [int(i) for i in decoding_mesh_shape]
  else:
    decoding_mesh_shape = config_lib.get_default_mesh_shape(
        config, mode='decode', dcn_mesh_shape=dcn_mesh_shape)
  logging.info('mesh_shape: %s', mesh_shape)
  logging.info('decoding_mesh_shape: %s', decoding_mesh_shape)

  if not dataclasses.is_dataclass(config):
    config = common.AttributeDict(config)

  logging.info('config: %s', config)
  logging.info('sharding_config: %s', sharding_config)
  logging.info('mesh_shape: %s', mesh_shape)
  logging.info('dcn_mesh_shape: %s', dcn_mesh_shape)
  run_experiment_fn = model_lib.TrainLoopRegistry.get(config.train_loop_name)
  run_experiment_fn(
      config=config, sharding_config=sharding_config,
      mesh_shape=mesh_shape,
      dcn_mesh_shape=dcn_mesh_shape,
      create_dataset=data_lib.create_dataset,
      experiment_dir=_EXPERIMENT_DIR.value,
      decoding_mesh_shape=decoding_mesh_shape,
  )

if __name__ == '__main__':
  app.run(main)

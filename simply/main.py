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

import dataclasses
import json
import os
import re
from typing import Sequence

from absl import flags
from absl import logging
from etils import epath
from simply import config_lib
from simply import model_lib
from simply import rl_lib  # pylint: disable=unused-import
from simply.utils import common
from simply.utils import experiment_helper
from simply.utils import pytree

from absl import app


_EXPERIMENT_CONFIG = flags.DEFINE_string(
    'experiment_config', None, 'Name of the experiment config.'
)

_SHARDING_CONFIG = flags.DEFINE_string(
    'sharding_config', None, 'Name of the sharding config.'
)

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
    'experiment_dir', '/tmp/simply_lm/', 'Path to save the experiment data.'
)

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


def override_mesh_and_sharding(config):
  """Updates sharding and mesh fields in the config."""
  if sharding_config_path := _SHARDING_CONFIG_PATH.value:
    sharding_config = pytree.load_pytree_from(sharding_config_path)
    config = dataclasses.replace(config, sharding_config=sharding_config)
  elif sharding_config_name := _SHARDING_CONFIG.value:
    sharding_config = config_lib.ShardingConfigRegistry.get_config(
        sharding_config_name
    )
    config = dataclasses.replace(config, sharding_config=sharding_config)

  mesh_axis_names = config.sharding_config.mesh_axis_names

  def parse_mesh_shape_flags(mesh_shape_flags):
    """Parses mesh shape flags into a dict."""
    return dict(zip(mesh_axis_names, map(int, mesh_shape_flags)))

  kwargs = {}
  if dcn_mesh_shape_flags := _DCN_MESH_SHAPE.value:
    kwargs['dcn_mesh_shape'] = parse_mesh_shape_flags(dcn_mesh_shape_flags)
  if mesh_shape_flags := _MESH_SHAPE.value:
    kwargs['mesh_shape'] = parse_mesh_shape_flags(mesh_shape_flags)
  if decoding_mesh_shape_flags := _DECODING_MESH_SHAPE.value:
    kwargs['decoding_mesh_shape'] = parse_mesh_shape_flags(
        decoding_mesh_shape_flags
    )
  if kwargs:
    config = dataclasses.replace(config, **kwargs)

  if config.mesh_shape is None:
    mesh_shape = config_lib.get_default_mesh_shape(
        config, mode='train', dcn_mesh_shape=config.dcn_mesh_shape
    )
    config = dataclasses.replace(config, mesh_shape=mesh_shape)

  if config.decoding_mesh_shape is None:
    decoding_mesh_shape = config_lib.get_default_mesh_shape(
        config, mode='decode', dcn_mesh_shape=config.dcn_mesh_shape
    )
    config = dataclasses.replace(
        config, decoding_mesh_shape=decoding_mesh_shape
    )
  return config


def execute_code_patch(config):
  if not dataclasses.is_dataclass(config):
    config = common.AttributeDict(config)
  if code_patch := getattr(config, 'code_patch', None):
    for code, code_context in code_patch:
      print(f'Executing under code context: {code_context}')
      context = globals()[code_context]
      print(f'code:\n{code}')
      exec(code, context.__dict__)  # pylint: disable=exec-used


def load_experiment_config():
  """Loads the experiment configuration.

  This function loads the experiment configuration from either a specified file
  path or a registered config name. It also applies mesh and sharding overrides
  based on command line flags and executes any code patches defined in the
  config.

  Returns:
    A tuple containing the loaded experiment config and the experiment
    directory.
  """
  if experiment_config_path := _EXPERIMENT_CONFIG_PATH.value:
    if _EXPERIMENT_CONFIG.value:
      logging.warning(
          'experiment_config_path is set. Will ignore experiment_config.'
      )
    with epath.Path(experiment_config_path).open('r') as f:
      config_dict = json.load(f)
    execute_code_patch(config_dict)
    config = pytree.load(config_dict)
  else:
    config = config_lib.ExperimentConfigRegistry.get_config(
        _EXPERIMENT_CONFIG.value
    )
    execute_code_patch(config)
  config = override_mesh_and_sharding(config)
  return config, _EXPERIMENT_DIR.value


def main(argv: Sequence[str]) -> None:
  del argv

  experiment_helper.setup_work_unit()
  config, experiment_dir = load_experiment_config()
  logging.info('config: %s', config)
  run_experiment_fn = model_lib.TrainLoopRegistry.get(config.train_loop_name)
  run_experiment_fn(config=config, experiment_dir=experiment_dir)

if __name__ == '__main__':
  app.run(main)

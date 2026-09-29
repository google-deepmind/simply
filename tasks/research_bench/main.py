# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for the research-bench task (forked from core simply).

Run it from the repository root, one process per seed, each with its own
experiment dir::

    python -m tasks.research_bench.main \\
        --experiment_config=pretrain_bpb_byte \\
        --experiment_dir=gs://<bucket>/<experiment_name>/seed_42 \\
        --config_overlay='{"model_seed": 42, "dataset_seed": 42}'
"""

import json
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from simply import main as main_lib
from tasks.research_bench import checkpoint_lib  # pylint: disable=unused-import
from tasks.research_bench import config_lib
from tasks.research_bench import data_lib  # pylint: disable=unused-import
from tasks.research_bench import model_lib  # pylint: disable=unused-import
from tasks.research_bench import port_eval_lib  # pylint: disable=unused-import
# Imported for registration side-effects: rl_algorithms registers the RL
# algorithm registry and rl_loop registers the self-contained tool-use train
# loop (`research_bench_rl`). The entry point is the single place that pulls in
# every task family's loops; config_lib stays free of loop-implementation deps.
from tasks.research_bench import rl_algorithms  # pylint: disable=unused-import
from tasks.research_bench import rl_loop  # pylint: disable=unused-import

experiment_helper = main_lib.experiment_helper


def load_experiment_config():
  """Core's loader, but `--experiment_config` resolves in the registry.

  Benchmark configs are registered in their own namespace (see
  `config_lib.ExperimentConfigRegistry`), so the name has to be looked up there;
  everything after the lookup -- code patches, the sweep overlay, the mesh and
  sharding overrides -- is core's, unchanged. The `--experiment_config_path`
  branch never touches a registry, so it is delegated to core as-is.

  Returns:
    A tuple of the loaded experiment config and the experiment directory.

  Raises:
    ValueError: if the name is not a research_bench config.
  """
  name = flags.FLAGS['experiment_config'].value
  if flags.FLAGS['experiment_config_path'].value or not name:
    return main_lib.load_experiment_config()
  if config_lib.ExperimentConfigRegistry.get(name, raise_error=False) is None:
    raise ValueError(
        f'--experiment_config={name!r} is not a research_bench config. '
        'Benchmark configs are registered in this package only (see '
        'research_bench/config_lib.py); a config defined in core simply is NOT visible '
        'to this binary, deliberately -- running one would leave the benchmark '
        'loop and produce a metric with no protocol stamp. Available: '
        f'{sorted(config_lib.ExperimentConfigRegistry.keys())}'
    )
  config = config_lib.ExperimentConfigRegistry.get_config(name)
  main_lib.execute_code_patch(config)
  config = main_lib.sweep.overlay(
      config, json.loads(flags.FLAGS['config_overlay'].value)  # pyrefly: ignore[bad-argument-type]
  )
  config = main_lib.override_mesh_and_sharding(config)
  return config, flags.FLAGS['experiment_dir'].value


def main(argv: Sequence[str]) -> None:
  del argv
  import jax  # pylint: disable=g-import-not-at-top
  try:
    jax.distributed.initialize()  # multi-host (TPU/Slurm/MPI auto-detected)
  except ValueError:
    pass  # single-host run: no coordinator needed

  experiment_helper.setup_work_unit()
  config, experiment_dir = load_experiment_config()
  logging.info('config: %s', config)
  logging.info('experiment_dir: %s', experiment_dir)
  loop_name = getattr(config, 'train_loop_name', None) or 'default'
  # For a task with a fixed training-compute cap this also attaches the compute
  # checks + result stamp (see model_lib.resolve_train_loop).
  run_experiment_fn = model_lib.resolve_train_loop(config, loop_name)
  final_result = run_experiment_fn(config=config, experiment_dir=experiment_dir)
  # After the run: record HOW the number was produced (which loop, whether the
  # eval/compute stamps are there) into final_result.json.
  model_lib.record_run_provenance(
      config, final_result, experiment_dir,
      config_name=flags.FLAGS['experiment_config'].value, loop_name=loop_name)


if __name__ == '__main__':
  experiment_helper.set_env_based_flags()
  app.run(main)

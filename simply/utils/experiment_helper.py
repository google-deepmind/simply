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
"""Helper class for experiments."""

import collections
from collections.abc import Sequence
import functools
import json
import logging
from typing import Any, Callable, cast, Mapping

from clu import metric_writers
from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import pytree
import yaml


def is_primary_process() -> bool:
  """Returns if the current process is the primary one."""
  return jax.process_index() == 0


class ExperimentHelper:
  """A utility class that saves all the experiment related data."""

  def __init__(
      self,
      experiment_dir,
      ckpt_interval,
      ckpt_max_to_keep,
      ckpt_keep_period,
      metric_log_interval,
      num_train_steps,
      log_additional_info,
      should_save_ckpt: bool = True,
  ):
    self.experiment_dir = epath.Path(experiment_dir)
    self.ckpt_interval = ckpt_interval
    self.ckpt_max_to_keep = ckpt_max_to_keep
    self.ckpt_keep_period = ckpt_keep_period
    self.num_train_steps = num_train_steps
    self.metric_log_interval = metric_log_interval
    self.log_additional_info = log_additional_info
    self.is_primary = is_primary_process()
    self.should_save_ckpt = should_save_ckpt
    self.should_save_data = self.is_primary and experiment_dir
    self.has_experiment_dir = bool(experiment_dir)
    self.metric_logdir = self.experiment_dir / 'tb_log'
    if self.should_save_data:
      if not self.experiment_dir.exists():
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    self.metric_writer = self.create_metric_writer()
    self.ckpt_mngr, self.ckpt_dir = self.create_ckpt_manager()
    self.metrics_aggregator = MetricsAggregator(
        average_last_n_steps=self.metric_log_interval)

  def write_record(self, record: Mapping[str, Any]):
    for k, v in record.items():
      logging.info('%s: %s', str(k), str(v))

  def save_config_info(self, config, sharding_config, model=None):
    """Save model and config information."""
    if model is not None:
      model_basic_jsons = json.dumps(
          pytree.dump(model, only_dump_basic=True), indent=2
      )
      model_full_jsons = json.dumps(
          pytree.dump(model, only_dump_basic=False), indent=2
      )
    else:
      model_basic_jsons = ''
      model_full_jsons = ''
    experiment_config_jsons = json.dumps(
        pytree.dump(config), indent=2
    )
    sharding_config_jsons = json.dumps(
        pytree.dump(sharding_config), indent=2
    )
    self.metric_writer.write_texts(
        step=0,
        texts={
            'experiment_config': f'```\n{experiment_config_jsons}\n```',
            'sharding_config': f'```\n{sharding_config_jsons}\n```',
            'model_full_jsons': f'```\n{model_full_jsons}\n```',
            'model_basic_jsons': f'```\n{model_basic_jsons}\n```',},
        )
    self.metric_writer.flush()
    if self.should_save_data:
      with (self.experiment_dir / 'experiment_config.json').open('w') as f:
        f.write(experiment_config_jsons)
      with (self.experiment_dir / 'model_basic.json').open('w') as f:
        f.write(model_basic_jsons)
      with (self.experiment_dir / 'model_full.json').open('w') as f:
        f.write(model_full_jsons)

      if sharding_config:
        with (self.experiment_dir / 'sharding_config.json').open('w') as f:
          f.write(sharding_config_jsons)

  def add_metric(self, metric_name, metric_value, **kwargs):
    self.metrics_aggregator.add(metric_name, metric_value, **kwargs)

  def get_aggregated_metrics(self):
    return self.metrics_aggregator.get_aggregated_metrics()

  def should_log_metrics(self, step):
    return (step % self.metric_log_interval == 0 or
            step == (self.num_train_steps - 1))

  def should_log_additional_info(self, step):
    return (
        self.log_additional_info and self.should_log_metrics(step))

  def write_scalars(self, step, scalars):
    self.metric_writer.write_scalars(step, common.get_raw_arrays(scalars))

  def write_texts(self, step, texts):
    self.metric_writer.write_texts(step, texts)

  def flush(self):
    self.metric_writer.flush()

  def create_metric_writer(self):
    """Creates a metric writer."""
    # Create experiment folder with tensorboard log and checkpoint.
    if self.should_save_data:
      if not self.metric_logdir.exists():
        self.metric_logdir.mkdir(parents=True, exist_ok=True)
    writer = metric_writers.create_default_writer(
        logdir=self.metric_logdir,
        just_logging=not self.should_save_data,
        asynchronous=True,)
    return writer

  def create_ckpt_manager(self):
    """Creates a checkpoint manager."""
    ckpt_dir = self.experiment_dir / 'checkpoints'
    # TODO: Use SaveDecisionPolicy and PreservationPolicy.
    if (
        self.ckpt_keep_period
        and (self.ckpt_keep_period % self.ckpt_interval) != 0
    ):
      raise ValueError(
          f'{self.ckpt_keep_period=} must be a multiple of '
          f'{self.ckpt_interval=}. Otherwise, it does not preserve anything.'
      )
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=self.ckpt_interval,
        max_to_keep=self.ckpt_max_to_keep,
        keep_period=self.ckpt_keep_period,
        async_options=ocp.AsyncOptions(timeout_secs=360000),
    )
    if self.has_experiment_dir:
      mngr = ocp.CheckpointManager(ckpt_dir, options=options)
    else:
      mngr = None
    return mngr, ckpt_dir

  def save_state_info(self, state):
    """Save state information."""
    state = common.get_raw_arrays(state)
    params_shape = jax.tree_util.tree_map(
        lambda x: str(x.shape), state['params'])
    logging.info('params shape: %s', params_shape)
    params_sharding = jax.tree_util.tree_map(
        lambda x: str(x.sharding), state['params'])
    logging.info('params sharding: %s', params_sharding)
    num_params = sum(jax.tree.leaves(
        jax.tree_util.tree_map(lambda x: np.prod(x.shape), state['params'])))
    logging.info('num_params: %s M', num_params/1e6)

    param_info_map = jax.tree_util.tree_map(
        lambda x, y: f'{x} :: {y}', params_shape, params_sharding)
    param_info_text = yaml.dump(
        param_info_map, default_flow_style=False, sort_keys=False)
    self.metric_writer.write_texts(
        step=0,
        texts={
            'num_params': f'`{num_params}`',
            'param_info_text': f'```\n{param_info_text}\n```',},)
    self.metric_writer.flush()
    if self.should_save_data:
      with (self.experiment_dir / 'params_info.json').open('w') as f:
        f.write(
            json.dumps(
                {
                    'params_shape': params_shape,
                    'params_sharding': params_sharding,
                    'num_params': int(num_params),
                },
                indent=2,
            )
        )

  def save_ckpt(self, state, step, data=None, force=False):
    # TODO: simplify the conditions for when to save ckpt.
    if not self.should_save_ckpt:
      return
    if self.ckpt_mngr and (
        force or self.ckpt_mngr.should_save(step) or
        step == self.num_train_steps):
      ckpt_lib.save_checkpoint(
          self.ckpt_mngr, state, step, data=data, force=True
      )
      logging.info('Saving checkpoint at step %s.', step)

  def close(self, final_result=None):
    # Ensure all the checkpoints are saved.
    if self.ckpt_mngr: self.ckpt_mngr.close()
    self.metric_writer.close()
    if self.should_save_data and final_result:
      with (self.experiment_dir / 'final_result.json').open('w') as f:
        f.write(json.dumps(final_result, indent=2))


class MetricsAggregator(object):
  """Metrics aggregator."""

  def __init__(self, average_last_n_steps: int = 100):
    assert average_last_n_steps > 0
    self._metrics = collections.defaultdict(collections.deque)
    self._metric_agg_fn_map = {}
    self.average_last_n_steps = average_last_n_steps

  def add(
      self,
      name: str,
      value: np.typing.ArrayLike,
      agg_fn: Callable[
          [Sequence[np.typing.ArrayLike]], np.typing.ArrayLike
      ] = np.mean,
  ) -> None:
    """Adds a metric to the aggregator."""
    if name in self._metrics:
      if (existing_agg_fn := self._metric_agg_fn_map[name]) != agg_fn:
        raise ValueError(
            f'Metric {name} has different aggregation functions: {agg_fn} vs'
            f' {existing_agg_fn}'
        )
    else:
      self._metric_agg_fn_map[name] = agg_fn
    agg_value = agg_fn(value) if np.size(value) > 1 else value
    self._metrics[name].append(agg_value)
    if len(self._metrics[name]) > self.average_last_n_steps:
      self._metrics[name].popleft()

  def reset(self) -> None:
    self._metrics = collections.defaultdict(collections.deque)

  def get_aggregated_metrics(self) -> Mapping[str, np.ndarray]:
    agg_metrics = {}
    for k, vlist in self._metrics.items():
      agg_metrics[k] = self._metric_agg_fn_map[k](vlist)
    return agg_metrics

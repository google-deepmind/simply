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
"""Optimizers for Simply."""

import abc
import dataclasses
import functools
import operator
from typing import Any, ClassVar, Mapping, cast, final

from absl import logging
import jax
import jax.numpy as jnp
from simply.utils import common
from simply.utils import registry
from simply.utils import sharding


PyTree = common.PyTree
Array = common.Array


################################################################################
## Optimizers.


class OptimizerRegistry(registry.RootRegistry):
  """Registry for optimizers."""

  namespace: ClassVar[str] = 'Optimizer'


@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class Optimizer(abc.ABC):
  """Base class for optimizers."""

  @final
  def __post_init__(self):
    if not dataclasses.is_dataclass(self):
      raise ValueError(
          f'OptimizerModule must be a dataclass. {self.__class__.__name__} is'
          ' not.'
      )
    if not OptimizerRegistry.get(self.__class__.__name__):
      raise ValueError(
          'OptimizerModule'
          f' {OptimizerRegistry.fullname(self.__class__.__name__)} is not'
          ' registered.'
      )

  def init(self, params: PyTree) -> PyTree:
    """Initializes the state associated with the optimizer."""

  @abc.abstractmethod
  def apply(self, state: PyTree, grad: PyTree) -> tuple[PyTree, PyTree]:
    """Applies the update rule to the optimizer state and the gradient."""

  def apply_updates(self, state, updates) -> PyTree:
    """Applies the update to the parameters."""
    assert jax.tree.map(lambda x: x.shape, state['params']) == jax.tree.map(
        lambda x: x.shape, updates
    )
    new_params = jax.tree_util.tree_map(
        lambda x, u: x - u, state['params'], updates
    )
    new_params = common.transfer_metadata(state['params'], new_params)
    state['params'] = new_params
    return state


def get_init_steps() -> jax.Array:
  return sharding.with_sharding_constraint(jnp.array(0, dtype=jnp.int32), None)


@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class SGD(Optimizer):
  """Stochastic Gradient Descent Optimizer."""

  def init(self, params):
    state = {}
    state['params'] = params
    state['steps'] = get_init_steps()
    return state

  def apply(self, state, grad):
    return grad, state


@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class Adam(Optimizer):
  """Adam Optimizer."""

  beta1: float = 0.9
  beta2: float = 0.999
  epsilon: float = 1e-6

  def init(self, params):
    state = {}
    state['params'] = params
    state['m'] = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(
            jnp.zeros_like(x), x.sharding),
        params)
    state['v'] = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(
            jnp.zeros_like(x), x.sharding),
        params)
    state['steps'] = get_init_steps()
    return state

  def apply(self, state, grad):
    state['m'] = jax.tree_util.tree_map(
        lambda m, g: m * self.beta1 + g * (1 - self.beta1), state['m'], grad)
    state['v'] = jax.tree_util.tree_map(
        lambda v, g: v * self.beta2 + jnp.square(g) * (1 - self.beta2),
        state['v'], grad)
    update = jax.tree_util.tree_map(
        lambda x, y: (x / (1 - self.beta1 ** (state['steps'] + 1))) /
        (jnp.sqrt(y / (1 - self.beta2 ** (state['steps'] + 1))) + self.epsilon),
        state['m'], state['v'])
    return update, state


@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class Lion(Optimizer):
  """Lion Optimizer."""

  beta1: float = 0.95
  beta2: float = 0.98
  momentum_dtype: jax.typing.DTypeLike = 'bfloat16'

  def init(self, params):
    state = {}
    state['params'] = params
    state['m'] = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(
            jnp.zeros_like(x, dtype=self.momentum_dtype), x.sharding),
        params)
    state['steps'] = get_init_steps()
    return state

  def apply(self, state, grad):
    grad = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=self.momentum_dtype), grad)
    update = jax.tree_util.tree_map(
        lambda m, g: jnp.sign(m * self.beta1 + g * (1 - self.beta1)),
        state['m'], grad)
    state['m'] = jax.tree_util.tree_map(
        lambda v, g: v * self.beta2 + g * (1 - self.beta2),
        state['m'], grad)
    return update, state


################################################################################
## Schedules.


class ScheduleRegistry(registry.RootRegistry):
  """Registry for Schedule."""

  namespace: ClassVar[str] = 'Schedule'


@dataclasses.dataclass(frozen=True)
class Schedule(abc.ABC):
  """Base class for schedules."""

  def __call__(self, steps: Array | int) -> Array:
    """Returns the value of the schedule at the given steps."""
    raise NotImplementedError()


@ScheduleRegistry.register
@dataclasses.dataclass(frozen=True)
class LinearWarmupConstant(Schedule):
  """Constant schedule."""
  value: float
  # warmup_steps=1 means no warmup.
  warmup_steps: int | None = None
  # Specify the warmup fraction instead of the absolute steps.
  warmup_fraction: float | None = None

  def _finalize(self, num_train_steps: Array | int):
    schedule = self
    for fraction_field, steps_field in [
        ('warmup_fraction', 'warmup_steps'),
    ]:
      schedule = replace_fraction(
          schedule, num_train_steps, fraction_field, steps_field)
    return schedule

  def __call__(
      self, steps: Array | int,
      num_train_steps: Array | int | None = None) -> Array:
    if num_train_steps is None:
      schedule = self
    else:
      schedule = self._finalize(num_train_steps)
    return constant_schedule(
        steps, schedule.value, warmup_steps=schedule.warmup_steps)


@ScheduleRegistry.register
@dataclasses.dataclass(frozen=True)
class LinearWarmupCosineDecay(Schedule):
  """Constant schedule."""
  value: float
  end_decay: float = 0.0

  decay_start: int | None = None
  decay_steps: int | None = None
  steps_after_decay: int | None = None

  # Specify the decay start and end fractions instead of the absolute steps.
  decay_start_fraction: float | None = None
  decay_fraction: float | None = None
  fraction_after_decay: float | None = None

  # warmup_steps=1 means no warmup.
  warmup_steps: int | None = None
  # Specify the warmup fraction instead of the absolute steps.
  warmup_fraction: float | None = None

  def _finalize(self, num_train_steps: Array | int):
    """Creates a schedule with absolute steps."""
    schedule = self
    for fraction_field, steps_field in [
        ('decay_start_fraction', 'decay_start'),
        ('decay_fraction', 'decay_steps'),
        ('fraction_after_decay', 'steps_after_decay'),
        ('warmup_fraction', 'warmup_steps'),
    ]:
      schedule = replace_fraction(
          schedule, num_train_steps, fraction_field, steps_field)
    schedule = cast(LinearWarmupCosineDecay, schedule)
    if schedule.decay_steps is None:
      decay_steps = num_train_steps
      if schedule.steps_after_decay is not None:
        decay_steps -= schedule.steps_after_decay
      if schedule.decay_start is not None:
        decay_steps -= schedule.decay_start
      elif schedule.warmup_steps is not None:
        decay_steps -= schedule.warmup_steps
      schedule = dataclasses.replace(schedule, decay_steps=decay_steps)
    else:
      raise ValueError(
          'Cannot specify both steps_after_decay and decay_steps.'
      )
    return schedule

  def __call__(
      self, steps: Array | int,
      num_train_steps: Array | int | None = None) -> Array:
    if num_train_steps is None:
      schedule = self
    else:
      schedule = self._finalize(num_train_steps)
    return cosine_decay_schedule(
        steps, schedule.value, decay_steps=schedule.decay_steps,
        warmup_steps=schedule.warmup_steps,
        end_decay=schedule.end_decay, decay_start=schedule.decay_start)


def replace_fraction(
    schedule: Schedule, num_train_steps: Array | int,
    fraction_field: str, steps_field: str):
  if getattr(schedule, fraction_field, None) is None:
    return schedule
  elif getattr(schedule, steps_field, None) is not None:
    raise ValueError(
        f'steps_field {steps_field} is already specified in {schedule}.')
  else:
    return dataclasses.replace(
        schedule,
        **{
            fraction_field: None,
            steps_field: int(
                num_train_steps * getattr(schedule, fraction_field)),
        }
    )


def create_lr_schedule(config):
  if getattr(config, 'lr_schedule_name', None):
    # Backward compatibility for existing configs that use lr_schedule_name.
    return create_lr_schedule_v0(config)
  else:
    return functools.partial(
        config.lr, num_train_steps=config.num_train_steps)


def create_lr_schedule_v0(config):
  """Creates a learning rate schedule from a experiment config."""
  lr_schedule_config = dict(config.lr_schedule_config)
  lr_schedule_config['val'] = lr_schedule_config.pop('lr')
  if ('decay_start' in lr_schedule_config and
      isinstance(lr_schedule_config['decay_start'], float) and
      lr_schedule_config['decay_start'] > 0 and
      lr_schedule_config['decay_start'] < 1.0):
    lr_schedule_config['decay_start'] = int(
        config.num_train_steps * lr_schedule_config['decay_start'])
  if (('decay_steps' in lr_schedule_config) and
      ('steps_after_decay' in lr_schedule_config)):
    raise ValueError('Cannot specify both decay_steps and steps_after_decay.')
  elif 'steps_after_decay' in lr_schedule_config:
    lr_schedule_config['decay_steps'] = (
        config.num_train_steps - lr_schedule_config['steps_after_decay'])
    if 'decay_start' in lr_schedule_config:
      lr_schedule_config['decay_steps'] -= lr_schedule_config['decay_start']
    elif 'warmup_steps' in lr_schedule_config:
      lr_schedule_config['decay_steps'] -= lr_schedule_config['warmup_steps']
    del lr_schedule_config['steps_after_decay']
  if config.lr_schedule_name == 'cosine_decay':
    return functools.partial(cosine_decay_schedule, **lr_schedule_config)
  elif config.lr_schedule_name == 'constant':
    return functools.partial(constant_schedule, **lr_schedule_config)
  else:
    raise ValueError(
        f'Unknown lr_schedule: {config.lr_schedule_name}')


def cosine_decay_schedule(
    steps, val, decay_steps, warmup_steps=1, end_decay=0.1, decay_start=None):
  """Linear warmup and cosine decay schedule."""
  # Linear warmup.
  steps += 1
  warmup_factor = jnp.minimum(steps, warmup_steps) / warmup_steps
  if decay_start is None:
    decay_start = warmup_steps
  decay_progress = jnp.maximum(0.0, steps - decay_start) / decay_steps
  decay_factor = (
      1 + jnp.cos(jnp.minimum(decay_progress, 1.0) * jnp.pi)) / 2
  actual_val = val * warmup_factor * (
      (1 - end_decay) * decay_factor + end_decay)
  actual_val = sharding.with_sharding_constraint(actual_val, None)
  return actual_val


def constant_schedule(steps, val, warmup_steps=None):
  if warmup_steps is None:
    warmup_steps = 1
  steps += 1
  warmup_factor = jnp.minimum(steps, warmup_steps) / warmup_steps
  actual_val = val * warmup_factor
  actual_val = sharding.with_sharding_constraint(actual_val, None)
  return actual_val


################################################################################
## Early stopping.


class EarlyStop(abc.ABC):
  """Base class for early stopping."""

  @final
  def __post_init__(self):
    if not dataclasses.is_dataclass(self):
      raise ValueError(
          f'EarlyStop must be a dataclass. {self.__class__.__name__} is'
          ' not.'
      )
    if not EarlyStopRegistry.get(self.__class__.__name__):
      raise ValueError(
          'EarlyStop'
          f' {EarlyStopRegistry.fullname(self.__class__.__name__)} is not'
          ' registered.'
      )

  def should_stop(self, step, metrics: Mapping[str, Any]) -> bool:
    """Returns whether the early stopping should be triggered."""
    raise NotImplementedError()


class EarlyStopRegistry(registry.RootRegistry):
  """Registry for early stopping."""

  namespace: ClassVar[str] = 'EarlyStop'


ThresholdDef = tuple[str, str, float]


@EarlyStopRegistry.register
@dataclasses.dataclass(frozen=True)
class SimpleEarlyStop(EarlyStop):
  """Early stopping based on metrics."""

  thresholds: tuple[tuple[int, ThresholdDef], ...] = ()

  def should_stop(self, step, metrics: Mapping[str, Any]) -> bool:
    """Returns whether the early stopping should be triggered."""
    op_dict = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
    }
    for threshold_step, (op, name, val) in self.thresholds:
      if threshold_step <= step:
        if name in metrics:
          metric_val = metrics[name]
          if op_dict[op](metric_val, val):
            logging.info(
                'Early stop triggered by metric %s at step %s'
                ' with value %s.', name, step, metric_val
            )
            return True
        else:
          logging.warning(
              'Early stop metric %s not found in `metrics`.', name
          )
    return False

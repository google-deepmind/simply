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
import collections
import dataclasses
import functools
import operator
from typing import Any, cast, ClassVar, final, Mapping

from absl import logging
import einops
import jax
import jax.numpy as jnp
from simply.utils import common
from simply.utils import registry
from simply.utils import sharding


Counter = collections.Counter
PyTree = common.PyTree
Array = common.Array
AnnotatedArray = common.AnnotatedArray

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


## Muon.
@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class Muon(Optimizer):
  """Implementation of the Muon Optimizer.

  Muon is Scalable for LLM Training: https://arxiv.org/html/2502.16982v1

  """

  learning_rate: float = 1.0
  muon_a: float = 3.4445
  muon_b: float = -4.7750
  muon_c: float = 2.0315
  ns_steps: int = 5
  beta: float = 0.95
  eps: float = 1e-8
  nesterov: bool = True
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  dim_threshold: int = 10000

  def init(self, params):
    """Initializes the optimizer state, using string indices as keys.

    Args:
      params: A dictionary of model parameters (PyTree).

    Returns:
      A dictionary representing the optimizer state.
    """
    def init_muon(p):
      if p.ndim < 2 or max(p.shape) > self.dim_threshold:
        return None
      else:
        sharded_zeros = jax.lax.with_sharding_constraint(
            jnp.zeros_like(p), p.sharding)
        return sharded_zeros
    def init_adam(p):
      if p.ndim < 2 or max(p.shape) > self.dim_threshold:
        sharded_zeros = jax.lax.with_sharding_constraint(
            jnp.zeros_like(p), p.sharding)
        return sharded_zeros
      else:
        return None
    state = {
        'params': params,
        'adam_m': jax.tree_util.tree_map(init_adam, params),
        'adam_v': jax.tree_util.tree_map(init_adam, params),
        'mu': jax.tree_util.tree_map(init_muon, params),
        'steps': get_init_steps(),
    }
    return state

  def apply(self, state, grad):
    # Compute update and state for each leaf
    def _mu(g, mu):
      if mu is not None:
        new_mu = mu * self.beta + g
        return new_mu

    def _adam_m(g, adam_m):
      if adam_m is not None:
        new_adam_m = adam_m * self.adam_b1 + g * (1 - self.adam_b1)
        return new_adam_m

    def _adam_v(g, adam_v):
      if adam_v is not None:
        new_adam_v = adam_v * self.adam_b2 + jnp.square(g) * (
            1 - self.adam_b2
        )
        return new_adam_v

    def _param_update(g, adam_m, adam_v, mu):
      if mu.array is not None:
        mu_ = (self.beta * mu.array + g.array) if self.nesterov else mu.array
        mu_ = self._orthogonalize_via_newton_schulz(mu_, mu.dim_annotation)
        return dataclasses.replace(mu, array=mu_)
      elif adam_v.array is not None and adam_m.array is not None:
        adam_ = (adam_m.array / (1 - self.adam_b1 ** (state['steps'] + 1))) / (
            jnp.sqrt(adam_v.array / (1 - self.adam_b2 ** (state['steps'] + 1)))
            + self.eps
        )
        return dataclasses.replace(adam_m, array=adam_)

    state['adam_m'] = jax.tree_util.tree_map(_adam_m, grad, state['adam_m'])
    state['adam_v'] = jax.tree_util.tree_map(_adam_v, grad, state['adam_v'])
    state['mu'] = jax.tree_util.tree_map(_mu, grad, state['mu'])

    updates = jax.tree_util.tree_map(
        _param_update,
        grad,
        state['adam_m'],
        state['adam_v'],
        state['mu'],
        is_leaf=lambda x: isinstance(x, AnnotatedArray),
    )
    # updates = common.transfer_metadata(state['params'], updates)
    return updates, state

  def _orthogonalize_via_newton_schulz(self, x, dim_annotation):
    """Newton-Schulz orthogonalization."""
    # Handle batch dimensions
    x, recipe = self.merge_repeated_dims(x, dim_annotation)
    # Ensure more columns than rows for efficiency
    transposed = x.shape[-1] < x.shape[-2]
    if transposed:
      x = jnp.einsum('...ij->...ji', x)

    # Normalize
    x_norm = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + self.eps
    x = x / x_norm

    # Newton-Schulz iterations
    for _ in range(self.ns_steps):
      a = jnp.einsum('...ij,...kj->...ik', x, x)
      a_squared = jnp.einsum('...ij,...jk->...ik', a, a)
      b = self.muon_b * a + self.muon_c * a_squared
      x = self.muon_a * x + jnp.einsum('...ij,...jk->...ik', b, x)

    # Restore original orientation and shape
    if transposed:
      x = jnp.einsum('...ji->...ij', x)

    scale = 0.2 * jnp.sqrt(jnp.maximum(x.shape[-1], x.shape[-2]))

    # Reshape back to original shape
    x = self.reconstruct_from_merged(x, recipe)

    return scale * x

  def merge_repeated_dims(self, tensor, dim_annotation):
    """Merges repeated dimensions in a tensor using einops.

    This method identifies dimensions with the same label in `dim_annotation`
    and merges them into a single dimension. It returns the rearranged tensor
    and a recipe to reconstruct the original shape.

    Args:
      tensor: The input tensor.
      dim_annotation: A list of strings or characters annotating each dimension
        of the tensor.

    Returns:
      A tuple containing:
        - merged_tensor: The tensor with repeated dimensions merged.
        - recipe: A dictionary containing information to reverse the merge,
          or None if no dimensions were merged.
    """
    counts = Counter(dim_annotation)

    repeated_labels = [label for label, count in counts.items() if count > 1]
    if len(repeated_labels) > 1:
      raise ValueError(
          'merge_repeated_dims only supports merging one type of repeated'
          f' dimension. Found multiple repeated labels: {repeated_labels}'
      )

    try:
      merge_label = next(iter(repeated_labels))
    except StopIteration:
      return tensor, None

    # Generate unique names for each input axis.
    from_names = []
    label_counts = Counter()
    for label in dim_annotation:
      label_counts[label] += 1
      suffix = str(label_counts[label]) if counts[label] > 1 else ''
      if label == '.':
        name = f'dot{suffix}'
      else:
        name = label + suffix
      from_names.append(name)

    # Build the 'to' pattern.
    to_merge_names = [
        name
        for name, label in zip(from_names, dim_annotation)
        if label == merge_label
    ]
    to_keep_names = [
        name
        for name, label in zip(from_names, dim_annotation)
        if label != merge_label
    ]

    # Store the original shapes of the dimensions that will be merged.
    merged_shapes = [
        s for s, l in zip(tensor.shape, dim_annotation) if l == merge_label
    ]

    # Construct the forward and backward patterns.
    pattern_from = ' '.join(from_names)
    pattern_to = f"{' '.join(to_keep_names)} ({' '.join(to_merge_names)})"
    rearrange_pattern = f'{pattern_from} -> {pattern_to}'

    merged_tensor = einops.rearrange(tensor, rearrange_pattern)

    # The recipe contains everything needed for the reverse operation.
    recipe = {
        'reconstruct_pattern': f'{pattern_to} -> {pattern_from}',
        'merged_shapes': {
            name: shape for name, shape in zip(to_merge_names, merged_shapes)
        },
    }

    return merged_tensor, recipe

  def reconstruct_from_merged(self, merged_tensor, recipe):
    """Reshapes a merged tensor back to its original shape using a recipe."""
    if not recipe:
      return merged_tensor
    reconstructed_tensor = einops.rearrange(
        merged_tensor, recipe['reconstruct_pattern'], **recipe['merged_shapes']
    )

    return reconstructed_tensor

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
class Constant(LinearWarmupConstant):
  value: float
  warmup_steps: int = 1


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

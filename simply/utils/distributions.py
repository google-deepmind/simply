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
"""Util classes for distributions."""

import abc
from collections.abc import Sequence
import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp

from simply.utils import common
from simply.utils import masked
from simply.utils import registry

Shape = int | Sequence[int]
Array = common.Array


class DistributionRegistry(registry.RootRegistry):
  """Registry for distributions."""

  namespace: ClassVar[str] = 'Distribution'


class Distribution(abc.ABC):
  """Base class of Distribution."""

  @abc.abstractmethod
  def entropy(self) -> Array:
    """Computes the entropy of the distribution."""

  @abc.abstractmethod
  def sample(
      self, key: jax.typing.ArrayLike, shape: Shape | None = None
  ) -> Array:
    """Samples random values from the distribution."""

  @abc.abstractmethod
  def prob(self, value: Array) -> Array:
    """Computes the probability of the input value."""

  @abc.abstractmethod
  def log_prob(self, value: Array) -> Array:
    """Computes the log probability of the input value."""


# TODO: Consider to use a functional design.
@DistributionRegistry.register
@dataclasses.dataclass(frozen=True)
class Categorical(Distribution):
  """Categorical distribution."""

  logits: Array

  @property
  def dtype(self) -> jax.typing.DTypeLike:
    return self.logits.dtype

  @functools.cached_property
  def log_probs(self) -> Array:
    return jax.nn.log_softmax(self.logits, axis=-1)

  def entropy(self) -> Array:
    return self._entropy

  def sample(
      self, key: jax.typing.ArrayLike, shape: Shape | None = None
  ) -> Array:
    return jax.random.categorical(key, self.logits, shape=shape)

  def prob(self, value: Array) -> Array:
    return jnp.exp(self.log_prob(value))

  def log_prob(self, value: Array) -> Array:
    return _gather(self.log_probs, value)

  @functools.cached_property
  def _entropy(self) -> Array:
    return -jnp.sum(jnp.exp(self.log_probs) * self.log_probs, axis=-1)


@DistributionRegistry.register
@dataclasses.dataclass(frozen=True)
class MaskedCategorical(Categorical):
  """Masked categorical distribution."""

  mask: Array
  neg_inf: float = -1e7

  @functools.cached_property
  def masked_logits(self) -> Array:
    return masked.masked(self.logits, self.mask, padding_value=self.neg_inf)

  @functools.cached_property
  def masked_log_probs(self) -> Array:
    return jax.nn.log_softmax(self.masked_logits, axis=-1)

  def sample(
      self, key: jax.typing.ArrayLike, shape: Shape | None = None
  ) -> Array:
    return jax.random.categorical(key, self.masked_logits, shape=shape)

  def prob(self, value: Array) -> Array:
    return jnp.exp(self.log_prob(value))

  def log_prob(self, value: Array) -> Array:
    return _gather(self.masked_log_probs, value)

  @functools.cached_property
  def _entropy(self) -> Array:
    return -jnp.sum(
        jnp.exp(self.masked_log_probs) * self.masked_log_probs, axis=-1
    )


def _gather(x: Array, indices: Array) -> Array:
  """Gathers x values along the last dim.

  Args:
    x: The source array.
    indices: The indices of elements to gather. The shape should be
      broadcast-compatible with np.delete(x.shape, -1).

  Returns:
    Array of values gathered from x. The shape is the same as indices.
  """
  # We are using one_hot instead of jnp.take_along_axis because the latter one
  # consumes much more HBM than the former one. One hypothesis is that XLA
  # cannot fuse jnp.take_along_axis with computation of `x` (typically computed
  # by log-softmax), so that `x` would have to be stored in HBM. In contrast,
  # computation of `x` is done on the fly in VMEM and does not go through HBM.
  one_hot = jax.nn.one_hot(indices, num_classes=x.shape[-1], dtype=x.dtype)
  return jnp.sum(one_hot * x, axis=-1)

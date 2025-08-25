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
"""Initializer."""

import abc
import dataclasses
import math
from typing import ClassVar, Sequence, final

import jax
import jax.numpy as jnp
import jax.typing

from simply.utils import registry


@dataclasses.dataclass(frozen=True)
class Initializer(abc.ABC):
  """Initializer."""

  def __post_init__(self):
    if not dataclasses.is_dataclass(self):
      raise ValueError(
          f'Initializer must be a dataclass. {self.__class__.__name__} is not.'
      )
    if not InitializerRegistry.get(self.__class__.__name__):
      raise ValueError(
          'Initializer'
          f' {InitializerRegistry.fullname(self.__class__.__name__)} is not'
          ' registered.'
      )

  @abc.abstractmethod
  def init(
      self,
      prng_key: jax.Array,
      shape: Sequence[int],
      dim_annotation: str,
      dtype: jax.typing.DTypeLike,
  ) -> jax.Array:
    """Initializes an array given.

    Args:
      prng_key: The random number generator key.
      shape: The shape of the array to initialize.
      dim_annotation: a string of 'i', 'o' and '.' with the same length as
        shape, where 'i' indicates input dimension, 'o' indicates output
        dimension and '.' indicates independent dimension.
      dtype: The data type of the array to initialize.

    Returns:
      The initialized array.
    """

  @final
  def __call__(
      self,
      prng_key: jax.Array,
      shape: Sequence[int],
      dim_annotation: str,
      dtype: jax.typing.DTypeLike,
  ) -> jax.Array:
    return self.init(prng_key, shape, dim_annotation, dtype)


class InitializerRegistry(registry.RootRegistry):
  """Registry for initializers.

  Signature:
    init_fn(
        prng_key: jax.Array, shape: Sequence[int],
        dim_annotation: str, dtype: jax.typing.DTypeLike) -> jax.Array
  """

  namespace: ClassVar[str] = 'Initializer'


@InitializerRegistry.register
@dataclasses.dataclass(frozen=True)
class XavierUniformInit(Initializer):
  """Xavier initializer."""

  scale: float = math.sqrt(6.0)

  def init(
      self,
      prng_key: jax.Array,
      shape: Sequence[int],
      dim_annotation: str,
      dtype: jax.typing.DTypeLike,
  ) -> jax.Array:
    """Xavier initializer."""
    assert len(shape) == len(dim_annotation)
    i_dim = math.prod(
        float(d) for c, d in zip(dim_annotation, shape) if c in 'i'
    )
    o_dim = math.prod(
        float(d) for c, d in zip(dim_annotation, shape) if c in 'o'
    )
    scale = jnp.array(self.scale * jax.lax.rsqrt(i_dim + o_dim), dtype=dtype)
    return (
        jax.random.uniform(
            prng_key, shape, dtype=dtype, minval=-1.0, maxval=1.0
        )
        * scale
    )


@InitializerRegistry.register
@dataclasses.dataclass(frozen=True)
class HeNormalInit(Initializer):
  """He normal initializer."""

  scale: float = math.sqrt(2.0)

  def init(
      self,
      prng_key: jax.Array,
      shape: Sequence[int],
      dim_annotation: str,
      dtype: jax.typing.DTypeLike,
  ) -> jax.Array:
    """He normal initializer."""
    assert len(shape) == len(dim_annotation)
    i_dim = math.prod(
        float(d) for c, d in zip(dim_annotation, shape) if c == 'i'
    )
    scale = jnp.array(self.scale * jax.lax.rsqrt(i_dim), dtype=dtype)
    return jax.random.normal(prng_key, shape, dtype=dtype) * scale


@InitializerRegistry.register
@dataclasses.dataclass(frozen=True)
class IdentityInit(Initializer):
  """Identity initializer."""

  def init(
      self,
      prng_key: jax.Array,
      shape: Sequence[int],
      dim_annotation: str,
      dtype: jax.typing.DTypeLike,
  ) -> jax.Array:
    """Identity initializer."""
    del prng_key
    assert len(shape) == len(dim_annotation)
    assert len(shape) == 2
    return jnp.eye(shape[0], dtype=dtype)

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
"""Utilities for masked tensor operations."""

from collections.abc import Sequence
import jax
import jax.numpy as jnp
from simply.utils import common

Array = common.Array

_EPS = 1e-5


def masked(
    x: jax.Array, mask: jax.Array, padding_value: float = 0
) -> jax.Array:
  return jnp.where(mask, x, jnp.astype(padding_value, x.dtype))


def masked_max(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Array:
  return jnp.max(
      x,
      axis=axis,
      keepdims=keepdims,
      where=mask,
      initial=jnp.finfo(x.dtype).min,
  )


def masked_min(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Array:
  return jnp.min(
      x,
      axis=axis,
      keepdims=keepdims,
      where=mask,
      initial=jnp.finfo(x.dtype).max,
  )


def masked_sum(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Array:
  return jnp.sum(x, axis=axis, keepdims=keepdims, where=mask)


def masked_mean(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Array:
  return jnp.sum(x, axis=axis, keepdims=keepdims, where=mask) / jnp.maximum(
      jnp.sum(mask.astype(x.dtype), axis=axis, keepdims=keepdims), _EPS
  )


def masked_var(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
) -> Array:
  _, var = masked_mean_var(x, mask, axis=axis, ddof=ddof, keepdims=keepdims)
  return var


def masked_std(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
) -> Array:
  _, std = masked_mean_std(x, mask, axis=axis, ddof=ddof, keepdims=keepdims)
  return std


def masked_mean_var(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
) -> tuple[Array, Array]:
  """Computes the mean and variance of a masked tensor."""
  mean = masked_mean(x, mask, axis=axis, keepdims=True)
  m2 = masked_sum(jnp.square(x - mean), mask, axis=axis, keepdims=keepdims)
  denorm = jnp.maximum(
      jnp.sum(mask.astype(x.dtype), axis=axis, keepdims=keepdims) - ddof, _EPS
  )
  return jnp.reshape(mean, shape=m2.shape), m2 / denorm


def masked_mean_std(
    x: Array,
    mask: Array,
    axis: int | Sequence[int] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
) -> tuple[Array, Array]:
  mean, var = masked_mean_var(x, mask, axis=axis, ddof=ddof, keepdims=keepdims)
  return mean, jnp.sqrt(var)

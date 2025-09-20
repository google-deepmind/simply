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
"""RunningMoments to track running_mean and running_var."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Scalar = jax.typing.ArrayLike


class RunningMoments(NamedTuple):
  """The running mean and var of a stream of value."""

  momentum: float = -1.0
  m0: Scalar = np.int64(0)
  m1: Scalar = np.float32(0)
  m2: Scalar = np.float32(0)

  @property
  def count(self) -> Scalar:
    return self.m0

  @property
  def mean(self) -> Scalar:
    return jax.lax.select(
        jnp.logical_or(self.momentum < 0, self.m0 == 0),
        self.m1,
        self.m1 / (1 - self.momentum**self.m0),
    )

  @property
  def var(self) -> Scalar:
    return jax.lax.select(
        jnp.logical_or(self.momentum < 0, self.m0 == 0),
        self.m2,
        self.m2 / (1 - self.momentum**self.m0),
    )

  @property
  def std(self) -> Scalar:
    return jnp.sqrt(self.var)


def update(running_moments: RunningMoments, x: Scalar) -> RunningMoments:
  """Update running_moments by adding x and returns the new running_moments."""

  assert jnp.isscalar(x)
  beta, m0, m1, m2 = running_moments

  def welford_fn(
      m0: Scalar, m1: Scalar, m2: Scalar, x: Scalar
  ) -> tuple[Scalar, Scalar, Scalar]:
    # Welford algorithm.
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    delta = x - running_moments.mean
    m0 += 1
    m1 += delta / m0
    m2 += (delta * (x - m1) - m2) / m0
    return m0, m1, m2

  def ema_fn(
      m0: Scalar, m1: Scalar, m2: Scalar, x: Scalar
  ) -> tuple[Scalar, Scalar, Scalar]:
    delta = x - running_moments.mean
    m0 += 1
    m1 = beta * m1 + (1 - beta) * x
    m2 = beta * m2 + (1 - beta) * delta * (x - m1 / (1 - beta**m0))
    return m0, m1, m2

  m0, m1, m2 = jax.lax.cond(beta < 0, welford_fn, ema_fn, m0, m1, m2, x)
  # For numerical stability.
  m2 = jnp.maximum(m2, 0)

  return RunningMoments(beta, m0, m1, m2)

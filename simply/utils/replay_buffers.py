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
"""ReplayBuffers for RL algorithms."""

from collections.abc import Iterator, Sequence
import typing

import jax
import numpy as np

from simply.utils import common
from simply.utils import segment_trees


class ReplayBuffer(Sequence[common.PyTree]):
  """The ReplayBuffer for RL algorithms."""

  def __init__(self, capacity: int, seed: int | None = None) -> None:
    self._capacity = capacity
    self._data = []
    self._cursor = 0
    self._rng = np.random.default_rng(seed)

  def __getitem__(
      self, index: int | Sequence[int] | np.ndarray
  ) -> common.PyTree:
    if np.isscalar(index):
      return self._data[index]

    batch = [self._data[i] for i in index]
    return jax.tree.map(lambda *x: np.stack(x), *batch)

  def __len__(self) -> int:
    return len(self._data)

  def __iter__(self) -> Iterator[common.PyTree]:
    return self.iterator()

  @property
  def capacity(self) -> int:
    return self._capacity

  @property
  def cursor(self) -> int:
    return self._cursor

  def append(self, x: common.PyTree) -> None:
    if len(self) < self._capacity:
      self._data.append(x)
    else:
      self._data[self._cursor] = x
    self._cursor = (self._cursor + 1) % self._capacity

  def extend(self, batch: common.PyTree) -> None:
    batch_size = jax.tree.leaves(batch)[0].shape[0]
    for i in range(batch_size):
      self.append(jax.tree.map(lambda x, i=i: x[i], batch))

  def sample(self, batch_size: int, replace: bool = False) -> common.PyTree:
    assert batch_size <= len(self)
    indices = self._rng.choice(len(self), batch_size, replace=replace)
    return self[indices]

  def iterator(
      self, batch_size: int = 1, shuffle: bool = False
  ) -> Iterator[common.PyTree]:
    size = len(self)
    indices = np.arange(size)
    if shuffle:
      self._rng.shuffle(indices)

    if batch_size == 1:
      for i in range(size):
        yield self[indices[i]]
    else:
      for i in range(0, size, batch_size):
        yield self[indices[i : i + batch_size]]


class PrioritizedReplayBuffer(ReplayBuffer):
  """The PrioritizedReplayBuffer for RL algorithms.

  Reference: https://arxiv.org/abs/1511.05952
  """

  def __init__(
      self, capacity: int, alpha: float, beta: float, seed: int | None = None
  ) -> None:
    super().__init__(capacity, seed)

    self._alpha = alpha
    self._beta = beta
    self._sum_tree = segment_trees.SumSegmentTree(capacity)
    self._min_tree = segment_trees.MinSegmentTree(capacity)
    self._max_priority = 1.0

  @property
  def alpha(self) -> float:
    return self._alpha

  @property
  def beta(self) -> float:
    return self._beta

  @property
  def max_priority(self) -> float:
    return self._max_priority

  def append(self, x: common.PyTree, priority: float | None = None) -> None:
    if priority is None:
      weight = self._max_priority**self.alpha
    else:
      self._max_priority = max(self._max_priority, priority)
      weight = priority**self.alpha

    index = self.cursor
    super().append(x)
    self._sum_tree[index] = weight
    self._min_tree[index] = weight

  def extend(
      self, batch: common.PyTree, priorities: np.ndarray | None = None
  ) -> None:
    if priorities is None:
      weights = self._max_priority**self.alpha
    else:
      self._max_priority = typing.cast(
          float, max(self._max_priority, float(np.max(priorities)))
      )
      weights = priorities**self.alpha

    batch_size = jax.tree.leaves(batch)[0].shape[0]
    indices = np.empty(batch_size, dtype=np.int64)
    for i in range(batch_size):
      indices[i] = self.cursor
      super().append(jax.tree.map(lambda x, i=i: x[i], batch))

    self._sum_tree[indices] = weights
    self._min_tree[indices] = weights

  def sample(
      self, batch_size: int, replace: bool = False
  ) -> tuple[common.PyTree, np.ndarray, np.ndarray]:
    indices, weights = self._sample_indices(batch_size, replace=replace)
    data = self[indices]
    weights = (weights / self._min_tree.min()) ** (-self.beta)
    return data, indices, weights

  def update_priorities(
      self, indices: np.ndarray, priorities: np.ndarray
  ) -> None:
    assert indices.ndim == 1
    assert indices.size == priorities.size

    weights = priorities**self.alpha
    self._sum_tree[indices] = weights
    self._min_tree[indices] = weights
    self._max_priority = typing.cast(
        float, max(self._max_priority, float(np.max(priorities)))
    )

  def _sample_indices(
      self, batch_size: int, replace: bool = False
  ) -> tuple[np.ndarray, np.ndarray]:

    if replace:
      mass = self._rng.uniform(high=self._sum_tree.sum(), size=batch_size)
      indices = self._sum_tree.scan_upper_bound(mass)
      weights = self._sum_tree[indices]
      return typing.cast(np.ndarray, indices), typing.cast(np.ndarray, weights)

    indices = np.empty(batch_size, dtype=np.int64)
    weights = np.empty(batch_size, dtype=self._sum_tree.dtype)
    for i in range(batch_size):
      mass = self._rng.uniform(high=self._sum_tree.sum())
      index = self._sum_tree.scan_upper_bound(mass)
      indices[i] = index
      weights[i] = self._sum_tree[index]
      self._sum_tree[index] = 0.0

    # Recovers the original priorities.
    self._sum_tree[indices] = weights

    return indices, weights

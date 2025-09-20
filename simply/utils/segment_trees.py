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
"""SegmentTrees."""

from collections.abc import Callable

import numpy as np

IndexType = int | np.int32 | np.int64 | np.ndarray
ValueType = float | np.float32 | np.float64 | np.ndarray


class SegmentTree:
  """SegmentTree that supports single element updates and range reductions.

  Reference: https://codeforces.com/blog/entry/18051
  """

  def __init__(
      self,
      size: int,
      reduce_op: Callable[[ValueType, ValueType], ValueType],
      identity_element: float,
      dtype: np.dtype = np.float64,  # pytype: disable=annotation-type-mismatch
  ) -> None:
    self._size = size
    self._capacity = (
        size if (size & (size - 1)) == 0 else (1 << (size.bit_length()))
    )
    self._reduce_op = reduce_op
    self._identity_element = identity_element
    self._data = np.full(2 * self._capacity, identity_element, dtype=dtype)

  def __len__(self) -> int:
    return self._size

  def __getitem__(self, key: IndexType) -> ValueType:
    return self._data[key + self._capacity]

  def __setitem__(self, key: IndexType, value: ValueType) -> None:
    if np.isscalar(key):
      self.update(key, value)
      return

    assert isinstance(key, np.ndarray) and key.ndim == 1
    if np.isscalar(value):
      for i in range(key.size):
        self.update(key[i], value)
    else:
      assert isinstance(value, np.ndarray) and value.shape == key.shape
      for i in range(key.size):
        self.update(key[i], value[i])

  @property
  def capacity(self) -> int:
    return self._capacity

  @property
  def identity_element(self) -> float:
    return self._identity_element

  @property
  def dtype(self) -> np.dtype:  # pytype: disable=annotation-type-mismatch
    return self._data.dtype

  def update(self, key: int, value: ValueType) -> None:
    """Updates data at key to value. The time complexity is O(logN)."""
    reduce = self._reduce_op
    key += self.capacity
    self._data[key] = value
    while key > 1:
      self._data[key >> 1] = reduce(self._data[key], self._data[key ^ 1])
      key >>= 1

  def reduce(self, start: int = 0, end: int | None = None) -> ValueType:
    """Reduces the range [start, end). The time complexity is O(logN)."""
    reduce = self._reduce_op

    l = start
    r = end or len(self)
    if l < 0:
      l += len(self)
    if r < 0:
      r += len(self)

    if l <= 0 and r >= len(self):
      return self._data[1]

    ret = self.identity_element
    l += self.capacity
    r += self.capacity
    while l < r:
      if (l & 1) == 1:
        ret = reduce(ret, self._data[l])
        l += 1
      if (r & 1) == 1:
        r -= 1
        ret = reduce(ret, self._data[r])
      l >>= 1
      r >>= 1

    return ret


class SumSegmentTree(SegmentTree):
  """SegmentTree that maintains the sum."""

  def __init__(
      self,
      size: int,
      dtype: np.dtype = np.float64,  # pytype: disable=annotation-type-mismatch
  ) -> None:
    super().__init__(size, reduce_op=np.add, identity_element=0.0, dtype=dtype)

  def sum(self, start: int = 0, end: int | None = None) -> ValueType:
    return self.reduce(start, end)

  def scan_upper_bound(self, value: ValueType) -> IndexType:
    if np.isscalar(value):
      return self._scan_upper_bound(value)
    else:
      return self._vectorized_scan_upper_bound(value)

  def _scan_upper_bound(self, value: float) -> int:
    """Returns the smallest `i` that sum(arr[0:i]) > value.

    The time complexity is O(logN).

    Args:
      value: The input value to compute scan_upper_bound.

    Returns:
      The smallest `i` that sum(arr[0:i]) > value.
    """
    if value >= self.sum():
      return self._size

    ret = 1
    cur = value
    while ret < self.capacity:
      ret <<= 1
      lvalue = self._data[ret]
      if cur >= lvalue:
        cur -= lvalue
        ret |= 1

    return ret - self.capacity

  def _vectorized_scan_upper_bound(self, value: np.ndarray) -> np.ndarray:
    """The vectorized version of _scan_upper_bound.

    This is adapted from the SegmentTree implementation in Tianshou.
    https://github.com/thu-ml/tianshou/blob/7a2bbe5e71dfe2763bd34470e1a678865124a10a/tianshou/data/utils/segtree.py#L120

    Args:
      value: The value vector to compute scan_upper_bound.

    Returns:
      The index vector of the result.
    """
    assert value.ndim == 1

    ret = np.ones_like(value, dtype=np.int64)
    cur = np.copy(value)
    while ret[0] < self.capacity:
      ret *= 2
      lvalue = self._data[ret]
      mask = cur >= lvalue
      cur -= lvalue * mask
      ret += mask

    return np.where(value >= self.sum(), self._size, ret - self.capacity)


class MinSegmentTree(SegmentTree):
  """SegmentTree that maintains the min value."""

  def __init__(
      self,
      size: int,
      dtype: np.dtype = np.float64,  # pytype: disable=annotation-type-mismatch
  ) -> None:
    super().__init__(
        size,
        reduce_op=np.minimum,
        identity_element=np.finfo(dtype).max,
        dtype=dtype,
    )

  def min(self, start: int = 0, end: int | None = None) -> ValueType:
    return self.reduce(start, end)

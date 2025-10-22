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
"""Unit test segment_trees."""

from absl.testing import absltest
import numpy as np
from simply.utils import segment_trees


class SegmentTreesTest(absltest.TestCase):

  def test_sum_segment_tree(self):
    with self.subTest("Init"):
      segment_tree = segment_trees.SumSegmentTree(10)
      self.assertLen(segment_tree, 10)
      self.assertEqual(segment_tree.capacity, 16)
      self.assertEqual(segment_tree.identity_element, 0)
      indices = np.arange(10)
      self.assertTrue(np.allclose(segment_tree[indices], 0))

      segment_tree = segment_trees.SumSegmentTree(16)
      self.assertLen(segment_tree, 16)
      self.assertEqual(segment_tree.capacity, 16)
      self.assertEqual(segment_tree.identity_element, 0)
      indices = np.arange(16)
      self.assertTrue(np.allclose(segment_tree[indices], 0))

    with self.subTest("Update"):
      size = 10
      segment_tree = segment_trees.SumSegmentTree(size)
      indices = np.arange(size)
      values = np.random.rand(size)

      for i in range(size):
        segment_tree[i] = values[i]
        self.assertEqual(segment_tree[i], values[i])

      segment_tree[indices] = values[0]
      self.assertTrue(np.allclose(segment_tree[indices], values[0]))

      segment_tree[indices] = values
      self.assertTrue(np.allclose(segment_tree[indices], values))

    with self.subTest("Reduce"):
      size = 10
      segment_tree = segment_trees.SumSegmentTree(size)
      indices = np.arange(size)
      values = np.random.rand(size)
      segment_tree[indices] = values

      self.assertAlmostEqual(segment_tree.sum(), np.sum(values))

      for i in range(size):
        self.assertAlmostEqual(segment_tree.sum(i), np.sum(values[i:]))
        self.assertAlmostEqual(
            segment_tree.sum(end=i + 1), np.sum(values[: i + 1])
        )
        if i + 1 < size:
          self.assertAlmostEqual(
              segment_tree.sum(end=-(i + 1)), np.sum(values[: -(i + 1)])
          )

      for l in range(size):
        for r in range(l + 1, size + 1):
          self.assertAlmostEqual(segment_tree.sum(l, r), np.sum(values[l:r]))
          if r < size:
            self.assertAlmostEqual(
                segment_tree.sum(l, r - size),
                np.sum(values[l - size : r - size]),
            )

    with self.subTest("ScanUpperBound"):

      def _scan_upper_bound_ref(
          values: np.ndarray, value: segment_trees.ValueType
      ) -> int:
        cur = 0.0
        for i in range(values.size):
          cur += values[i]
          if cur > value:
            return i
        return values.size

      size = 10
      segment_tree = segment_trees.SumSegmentTree(size)
      indices = np.arange(size)
      values = np.random.rand(size)
      segment_tree[indices] = values

      self.assertEqual(segment_tree.scan_upper_bound(0), 0)
      self.assertEqual(
          segment_tree.scan_upper_bound(np.sum(values) + 1e-8), size
      )
      mass = np.full(100, np.sum(values) + 1e-8)
      self.assertTrue(np.all(segment_tree.scan_upper_bound(mass) == size))

      mass = np.random.uniform(high=np.sum(values), size=100)
      ref = np.asarray(
          [_scan_upper_bound_ref(values, mass[i]) for i in range(mass.size)]
      )
      for i in range(mass.size):
        self.assertEqual(segment_tree.scan_upper_bound(mass[i]), ref[i])
      self.assertTrue(np.all(segment_tree.scan_upper_bound(mass) == ref))

  def test_min_segment_tree(self):
    with self.subTest("Init"):
      segment_tree = segment_trees.MinSegmentTree(10)
      self.assertLen(segment_tree, 10)
      self.assertEqual(segment_tree.capacity, 16)
      self.assertEqual(
          segment_tree.identity_element, np.finfo(segment_tree.dtype).max
      )
      indices = np.arange(10)
      self.assertTrue(
          np.all(segment_tree[indices] == np.finfo(segment_tree.dtype).max)
      )

      segment_tree = segment_trees.MinSegmentTree(16)
      self.assertLen(segment_tree, 16)
      self.assertEqual(segment_tree.capacity, 16)
      self.assertEqual(
          segment_tree.identity_element, np.finfo(segment_tree.dtype).max
      )
      indices = np.arange(16)
      self.assertTrue(
          np.all(segment_tree[indices] == np.finfo(segment_tree.dtype).max)
      )

    with self.subTest("Update"):
      size = 10
      segment_tree = segment_trees.MinSegmentTree(size)
      indices = np.arange(size)
      values = np.random.rand(size)

      for i in range(size):
        segment_tree[i] = values[i]
        self.assertEqual(segment_tree[i], values[i])

      segment_tree[indices] = values[0]
      self.assertTrue(np.allclose(segment_tree[indices], values[0]))

      segment_tree[indices] = values
      self.assertTrue(np.allclose(segment_tree[indices], values))

    with self.subTest("Reduce"):
      size = 10
      segment_tree = segment_trees.MinSegmentTree(size)
      indices = np.arange(size)
      values = np.random.rand(size)
      segment_tree[indices] = values

      self.assertEqual(segment_tree.min(), np.min(values))

      for i in range(size):
        self.assertEqual(segment_tree.min(i), np.min(values[i:]))
        self.assertEqual(segment_tree.min(end=i + 1), np.min(values[: i + 1]))
        if i + 1 < size:
          self.assertEqual(
              segment_tree.min(end=-(i + 1)), np.min(values[: -(i + 1)])
          )

      for l in range(size):
        for r in range(l + 1, size + 1):
          self.assertEqual(segment_tree.min(l, r), np.min(values[l:r]))
          if r < size:
            self.assertEqual(
                segment_tree.min(l - size, r - size),
                np.min(values[l - size : r - size]),
            )


if __name__ == "__main__":
  absltest.main()

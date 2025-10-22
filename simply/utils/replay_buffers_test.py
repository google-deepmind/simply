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
"""Unit test for replay_buffers."""

import typing

from absl.testing import absltest
import jax
import numpy as np
from simply.utils import replay_buffers


class ReplayBuffersTest(absltest.TestCase):

  def test_replay_buffer(self):
    capacity = 10
    data_size = 15
    data = [
        {
            "x": np.asarray([2 * i]),  # 1D array
            "y": np.int64(2 * i),  # 0D scalar
            "z": np.asarray([[2 * i]]),  # 2D array
        }
        for i in range(data_size)
    ]

    with self.subTest("Init"):
      buffer = replay_buffers.ReplayBuffer(capacity)
      self.assertEmpty(buffer)
      self.assertEqual(buffer.capacity, capacity)
      self.assertEqual(buffer.cursor, 0)

    with self.subTest("Append"):
      buffer = replay_buffers.ReplayBuffer(capacity)
      for i, x in enumerate(data):
        buffer.append(x)
        self.assertLen(buffer, min(i + 1, capacity))
        self.assertEqual(buffer.cursor, (i + 1) % capacity)
        self.assertDictEqual(buffer[i % capacity], x)

    with self.subTest("Extend"):
      buffer = replay_buffers.ReplayBuffer(capacity)
      for i in range(0, data_size, 3):
        batch = jax.tree.map(lambda *x: np.stack(x), *data[i : i + 3])
        buffer.extend(batch)
        self.assertLen(buffer, min(i + 3, capacity))
        self.assertEqual(buffer.cursor, (i + 3) % capacity)
        self.assertDictEqual(buffer[i % capacity], data[i])
        self.assertDictEqual(buffer[(i + 1) % capacity], data[i + 1])
        self.assertDictEqual(buffer[(i + 2) % capacity], data[i + 2])

    with self.subTest("Sample"):

      def _test_sample(replace: bool, atol: float) -> None:
        buffer = replay_buffers.ReplayBuffer(capacity)
        buffer.extend({"x": np.expand_dims(np.arange(capacity), -1)})

        num_samples = 20000
        batch_size = 4
        values = np.empty((num_samples, batch_size), dtype=np.int64)
        for i in range(num_samples):
          batch = buffer.sample(batch_size, replace=replace)
          self.assertEqual(batch["x"].shape, (batch_size, 1))
          values[i, :] = batch["x"].flatten()

        probs = np.bincount(values.flatten()) / (num_samples * batch_size)
        self.assertEqual(probs.size, capacity)
        np.testing.assert_allclose(probs, 1.0 / capacity, atol=atol)

      _test_sample(replace=True, atol=5e-3)
      _test_sample(replace=False, atol=5e-3)

    with self.subTest("Iterator"):
      buffer = replay_buffers.ReplayBuffer(capacity)
      buffer.extend({"x": np.expand_dims(np.arange(capacity), -1)})

      # batch_size = 1
      for i, batch in enumerate(buffer):
        self.assertEqual(typing.cast(np.ndarray, batch["x"]).item(), i)

      # batch_size > 1
      batch_size = 4
      indices = []
      for i, batch in enumerate(buffer.iterator(batch_size, shuffle=True)):
        self.assertEqual(
            batch["x"].shape, (min(batch_size, capacity - i * batch_size), 1)
        )
        indices.extend(batch["x"].flatten().tolist())
      self.assertCountEqual(indices, range(capacity))

  def test_prioritized_replay_buffer(self):
    capacity = 10
    data_size = 15
    data = [
        {"x": np.asarray([2 * i]), "y": np.asarray([2 * i + 1])}
        for i in range(data_size)
    ]
    priorities = np.random.uniform(high=2.0, size=data_size)

    with self.subTest("Init"):
      buffer = replay_buffers.PrioritizedReplayBuffer(
          capacity, alpha=0.9, beta=0.2
      )
      self.assertEmpty(buffer)
      self.assertEqual(buffer.capacity, capacity)
      self.assertEqual(buffer.alpha, 0.9)
      self.assertEqual(buffer.beta, 0.2)
      self.assertEqual(buffer.max_priority, 1.0)

    with self.subTest("Append"):
      buffer = replay_buffers.PrioritizedReplayBuffer(
          capacity, alpha=0.9, beta=0.2
      )
      for i, x in enumerate(data):
        buffer.append(x)
        self.assertLen(buffer, min(i + 1, capacity))
        self.assertEqual(buffer.cursor, (i + 1) % capacity)
        self.assertEqual(buffer.max_priority, 1.0)
        self.assertDictEqual(buffer[i % capacity], x)

      buffer = replay_buffers.PrioritizedReplayBuffer(
          capacity, alpha=0.9, beta=0.2
      )
      for i, x in enumerate(data):
        buffer.append(x, priority=priorities[i])
        self.assertLen(buffer, min(i + 1, capacity))
        self.assertEqual(buffer.cursor, (i + 1) % capacity)
        self.assertEqual(
            buffer.max_priority, max(1.0, np.max(priorities[: i + 1]))
        )
        self.assertDictEqual(buffer[i % capacity], x)

    with self.subTest("Extend"):
      buffer = replay_buffers.PrioritizedReplayBuffer(
          capacity, alpha=0.9, beta=0.2
      )
      for i in range(0, data_size, 3):
        batch = jax.tree.map(lambda *x: np.vstack(x), *data[i : i + 3])
        buffer.extend(batch)
        self.assertLen(buffer, min(i + 3, capacity))
        self.assertEqual(buffer.cursor, (i + 3) % capacity)
        self.assertEqual(buffer.max_priority, 1.0)
        self.assertDictEqual(buffer[i % capacity], data[i])
        self.assertDictEqual(buffer[(i + 1) % capacity], data[i + 1])
        self.assertDictEqual(buffer[(i + 2) % capacity], data[i + 2])

      buffer = replay_buffers.PrioritizedReplayBuffer(
          capacity, alpha=0.9, beta=0.2
      )
      for i in range(0, data_size, 3):
        batch = jax.tree.map(lambda *x: np.vstack(x), *data[i : i + 3])
        buffer.extend(batch, priorities=priorities[i : i + 3])
        self.assertLen(buffer, min(i + 3, capacity))
        self.assertEqual(buffer.cursor, (i + 3) % capacity)
        self.assertEqual(
            buffer.max_priority, max(1.0, np.max(priorities[: i + 3]))
        )
        self.assertDictEqual(buffer[i % capacity], data[i])
        self.assertDictEqual(buffer[(i + 1) % capacity], data[i + 1])
        self.assertDictEqual(buffer[(i + 2) % capacity], data[i + 2])

    with self.subTest("Sample"):

      def _test_sample(replace: bool, atol: float) -> None:
        buffer = replay_buffers.PrioritizedReplayBuffer(
            capacity, alpha=1.0, beta=1.0
        )
        buffer.extend(
            {"x": np.expand_dims(np.arange(capacity), axis=-1)},
            priorities=priorities[:capacity],
        )

        num_samples = 20000
        batch_size = 4
        values = np.empty((num_samples, batch_size), dtype=np.int64)

        for i in range(num_samples):
          batch, indices, weights = buffer.sample(batch_size, replace=replace)
          self.assertEqual(batch["x"].shape, (batch_size, 1))
          self.assertEqual(indices.shape, (batch_size,))
          self.assertEqual(weights.shape, (batch_size,))

          values[i, :] = batch["x"].flatten()
          np.testing.assert_allclose(
              weights, np.min(priorities[:capacity]) / priorities[indices]
          )

        probs = np.bincount(values.flatten()) / (num_samples * batch_size)
        self.assertEqual(probs.size, capacity)
        np.testing.assert_allclose(
            probs,
            priorities[:capacity] / np.sum(priorities[:capacity]),
            atol=atol,
        )

      _test_sample(replace=True, atol=5e-3)
      _test_sample(replace=False, atol=0.1)

    with self.subTest("UpdatePriorities"):
      buffer = replay_buffers.PrioritizedReplayBuffer(
          capacity, alpha=1.0, beta=1.0
      )
      buffer.extend({"x": np.arange(capacity)})
      self.assertEqual(buffer.max_priority, 1.0)

      buffer.update_priorities(np.arange(capacity), priorities[:capacity])
      self.assertEqual(
          buffer.max_priority, max(1.0, np.max(priorities[:capacity]))
      )
      _, indices, weights = buffer.sample(capacity)
      np.testing.assert_allclose(
          weights, np.min(priorities[:capacity]) / priorities[indices]
      )


if __name__ == "__main__":
  absltest.main()

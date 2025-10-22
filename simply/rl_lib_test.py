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
"""Unit test for rl_lib.py."""

from absl.testing import absltest
import numpy as np
from simply import rl_lib


class RewardNormalizerTest(absltest.TestCase):

  def test_global(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    example_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    masks = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.bool)

    normalizer = rl_lib.RewardNormalizer.Global()
    normalized_rewards = normalizer.normalize(rewards, example_ids, masks)
    subrewards = rewards[masks]
    expected = (subrewards - np.mean(subrewards)) / np.std(subrewards)
    np.testing.assert_allclose(normalized_rewards[masks], expected)

  def test_by_group(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    example_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    masks = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.bool)

    normalizer = rl_lib.RewardNormalizer.ByGroup()
    normalized_rewards = normalizer.normalize(rewards, example_ids, masks)
    expected = np.array([0, 0, -1, 1])
    np.testing.assert_allclose(normalized_rewards[masks], expected)

    normalized_rewards = normalizer.normalize_by_group(
        rewards, example_ids, masks, std=1.0
    )
    expected = np.array([0, 0, -0.5, 0.5])
    np.testing.assert_allclose(normalized_rewards[masks], expected)


if __name__ == "__main__":
  absltest.main()

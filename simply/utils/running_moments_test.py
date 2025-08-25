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
"""Unit test for RunningMoments."""

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from simply.utils import running_moments


class RunningMomentsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("without_momentum", -1.0), ("with_momentum", 0.99)
  )
  def test_update(self, momentum: float):
    num_values = 10
    moments = running_moments.RunningMoments(momentum)

    x = np.random.rand(num_values)
    if momentum < 0:
      w = None
    else:
      # w = [1, momentum, momentum^2, momentum^3, ...]
      w = np.exp(np.arange(num_values) * np.log(momentum))

    for i in range(num_values):
      moments = running_moments.update(moments, x[i])

      if momentum < 0:
        ref_mean = np.mean(x[: i + 1])
        ref_var = np.var(x[: i + 1])
        ref_std = np.std(x[: i + 1])
      else:
        weights = np.flip(w[: i + 1])
        ref_mean = np.average(x[: i + 1], weights=weights)
        ref_var = np.average(np.square(x[: i + 1] - ref_mean), weights=weights)
        ref_std = np.sqrt(ref_var)

      self.assertEqual(moments.count, i + 1)
      self.assertAlmostEqual(moments.mean, ref_mean, places=5)
      self.assertAlmostEqual(moments.var, ref_var, places=5)
      self.assertAlmostEqual(moments.std, ref_std, places=5)


if __name__ == "__main__":
  absltest.main()

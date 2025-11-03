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
"""Unit test for data_lib.py."""

from absl.testing import absltest
import numpy as np
from simply import data_lib


class DataLibTest(absltest.TestCase):

  def test_add_chat_mask_to_loss(self):
    self.assertEqual(1, 1)
    a = np.array([
        [4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 7, 5, 43, 43, 7],
        [23, 7, 4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 7, 5, 43],
        [7, 4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 7, 5, 43, 23],
        [7, 4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 43, 23, 7, 5],
    ])
    output = data_lib.create_chat_loss_mask(a, mask_start_id=5, mask_end_id=7)

    target_output = np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ])
    self.assertTrue(np.all(output == target_output))


if __name__ == '__main__':
  absltest.main()

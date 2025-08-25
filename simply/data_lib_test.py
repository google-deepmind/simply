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

import numpy as np

from absl.testing import absltest
from simply import data_lib


class DataLibTest(absltest.TestCase):

  def test_add_chat_mask_to_loss(self):
    self.assertEqual(1, 1)
    a = np.array(
        [[4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 7, 5, 43, 43, 7],
         [23, 7, 4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 7, 5, 43],
         [7, 4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 7, 5, 43, 23],
         [7, 4, 21, 32, 7, 5, 21, 32, 43, 7, 4, 21, 32, 43, 23, 7, 5]])
    output = data_lib.create_chat_loss_mask(
        a, mask_start_id=5, mask_end_id=7)

    target_output = np.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
    self.assertTrue(np.all(output == target_output))

  def test_simple_dataloader(self):
    datasource = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Test basic iteration.
    dataloader = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=False,
        num_epochs=1, drop_remainder=False)
    self.assertEqual(dataloader.current_epoch, 0)
    self.assertEqual(dataloader._cursor, 0)
    self.assertEqual(dataloader.num_past_examples, 0)
    iter_dataset = iter(dataloader)
    self.assertEqual(list(iter_dataset),
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 1, 2]])
    dl = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=False,
        num_epochs=1, drop_remainder=True)
    self.assertEqual(list(iter(dl)), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with self.assertRaises(StopIteration):
      next(iter_dataset)

    # Test random seeds.
    dl_1 = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=True,
        num_epochs=1, drop_remainder=False, seed=1)
    dl_2 = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=True,
        num_epochs=1, drop_remainder=False, seed=2)
    self.assertNotEqual(list(iter(dl_1)), list(iter(dl_2)))
    dl_3 = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=True,
        num_epochs=1, drop_remainder=False, seed=1)
    dl_4 = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=True,
        num_epochs=1, drop_remainder=False, seed=1)
    self.assertEqual(list(iter(dl_3)), list(iter(dl_4)))

    # Test num_past_examples.
    dl = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=False,
        num_epochs=1, drop_remainder=True, num_past_examples=2)
    self.assertEqual(list(iter(dl)), [[3, 4, 5], [6, 7, 8]])

    # Test multiple epochs.
    dl = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=False,
        num_epochs=2, drop_remainder=True, num_past_examples=12)
    self.assertEqual(list(iter(dl)), [[3, 4, 5], [6, 7, 8]])
    dl = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=False,
        num_epochs=2, drop_remainder=False, num_past_examples=12)
    self.assertEqual(list(iter(dl)), [[3, 4, 5], [6, 7, 8], [9, 10, 1]])

    # Test multiple epochs.
    dl = data_lib.SimpleDataloader(
        datasource=datasource, batch_size=3, shuffle=False,
        num_epochs=1, drop_remainder=False, num_past_examples=12)
    self.assertEqual(list(iter(dl.repeat(2))),
                     [[3, 4, 5], [6, 7, 8], [9, 10, 1]])


if __name__ == '__main__':
  absltest.main()

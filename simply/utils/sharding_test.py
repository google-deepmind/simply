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
import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
from unittest import mock
from simply.utils import sharding


class ShardingTest(absltest.TestCase):

  def test_mesh_context(self):
    with sharding.mesh_context(mesh_shape=[1, 1, 1], dcn_mesh_shape=[1]):
      self.assertEqual(
          sharding.get_default_mesh().shape, dict(replica=1, data=1, model=1)
      )

  def test_sharding(self):
    _ = sharding.with_sharding_constraint(
        jnp.array([[1, 2], [3, 4]]), ['replica', 'data']
    )
    _ = sharding.with_sharding_constraint(jnp.array([[1, 2], [3, 4]]), None)

    with self.assertRaises(ValueError):
      sharding.with_sharding_constraint(
          jnp.array([[1, 2], [3, 4]]), ['replica']
      )
    with self.assertRaises(ValueError):
      sharding.with_sharding_constraint(jnp.array([[1, 2], [3, 4]]), [None])
    with self.assertRaises(ValueError):
      sharding.with_sharding_constraint(jnp.array(3), [None])

    sharding.with_sharding_constraint(
        jnp.array([1, 2]), sharding.mesh_sharding([['replica', 'data']])
    )

  def test_multihost_sharded(self):
    batch = [1, 2, 3, 4, 5, 6]
    self.assertEqual(sharding.multihost_sharded(batch), batch)
    self.assertEqual(
        sharding.multihost_sharded(batch, process_index=0, process_count=2),
        [1, 2, 3],
    )
    self.assertEqual(
        sharding.multihost_sharded(batch, process_index=1, process_count=2),
        [4, 5, 6],
    )
    self.assertEqual(
        sharding.multihost_sharded(batch, process_index=1, process_count=4),
        [3, 4],
    )
    self.assertEqual(
        sharding.multihost_sharded(batch, process_index=2, process_count=4),
        [5],
    )

  def test_multihost_data(self):
    testdir = os.path.join(self.create_tempdir(), 'test_multihost_data')
    global_data = {'a': 1, 'b': 2}
    local_data = {'c': 1, 'd': [1, 2]}
    with mock.patch.object(
        jax, 'process_count', return_value=2
    ), mock.patch.object(
        jax.experimental.multihost_utils, 'sync_global_devices'
    ):
      with mock.patch.object(jax, 'process_index', return_value=0):
        sharding.MultihostData(
            global_data=global_data, local_data=local_data
        ).snapshot(testdir)
      local_data['d'] = [3]
      with mock.patch.object(jax, 'process_index', return_value=1):
        sharding.MultihostData(
            global_data=global_data, local_data=local_data
        ).save(testdir)
    multihost_data = sharding.MultihostData.load(testdir)
    self.assertEqual(multihost_data.global_data, global_data)
    self.assertEqual(multihost_data.local_data, {'c': 1, 'd': [1, 2, 3]})


if __name__ == '__main__':
  absltest.main()

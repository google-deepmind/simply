# Copyright 2026 The Simply Authors
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
"""Tests for `colocate_async_rl_lib.reshard` that need more than one device.

Two meshes can only disagree about the ORDER they enumerate their devices in
when they hold more than one, and that disagreement is what used to break the
train -> decode param move, so it cannot be covered by the single-device
`colocate_async_rl_lib_test`. The device count must be set before JAX
initializes its backend, hence a separate target.
"""

import os

# Must be set BEFORE `import jax`. JAX initializes its backend on first use
# and the device count cannot be changed afterwards.
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

# pylint: disable=g-import-not-at-top
# Imports below jax-affecting env vars are intentional; do not reorder.
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from simply import colocate_async_rl_lib

# pylint: enable=g-import-not-at-top

_AXIS_NAMES = ('data', 'model')


def _mesh(devices) -> jax.sharding.Mesh:
  return jax.sharding.Mesh(np.asarray(devices).reshape(2, 2), _AXIS_NAMES)


def _sharded(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))


class ReshardAcrossMeshesTest(absltest.TestCase):
  """The train -> decode move, over meshes that order their devices freely."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()[:4]
    self.train_mesh = _mesh(devices)
    # `mesh_utils.create_device_mesh` orders devices per mesh SHAPE, so the
    # decoding mesh generally holds the same devices in a different order.
    self.decoding_mesh = _mesh([devices[i] for i in (0, 2, 1, 3)])

  def _tree_on(self, mesh):
    return {'w': jax.device_put(jnp.arange(4.0), _sharded(mesh))}

  def _target_on(self, mesh):
    return {
        'w': jax.ShapeDtypeStruct((4,), jnp.bfloat16, sharding=_sharded(mesh))
    }

  def _assert_moved(self, resharded, mesh):
    self.assertEqual(resharded['w'].dtype, jnp.bfloat16)
    self.assertEqual(resharded['w'].sharding, _sharded(mesh))
    np.testing.assert_array_equal(
        np.asarray(resharded['w'], np.float32), [0.0, 1.0, 2.0, 3.0]
    )

  def test_a_different_device_order_is_not_an_obstacle(self):
    self.assertNotEqual(
        [d.id for d in self.train_mesh.devices.flat],
        [d.id for d in self.decoding_mesh.devices.flat],
        'the two meshes must disagree for this test to mean anything',
    )
    self._assert_moved(
        colocate_async_rl_lib.reshard(
            self._tree_on(self.train_mesh), self._target_on(self.decoding_mesh)
        ),
        self.decoding_mesh,
    )

  def test_the_same_mesh_still_casts_in_place(self):
    self._assert_moved(
        colocate_async_rl_lib.reshard(
            self._tree_on(self.train_mesh), self._target_on(self.train_mesh)
        ),
        self.train_mesh,
    )


if __name__ == '__main__':
  absltest.main()

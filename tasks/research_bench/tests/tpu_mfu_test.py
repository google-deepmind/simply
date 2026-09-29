# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the local FLOPs-estimate helper."""

from absl.testing import absltest
from tasks.research_bench import tpu_mfu


class MeshDeviceCountTest(absltest.TestCase):

  def test_scale_factor_comes_from_the_mesh_not_the_backend(self):
    # cost_analysis is per-partition, so the factor must be the size of the
    # mesh we compiled on. The estimate pins a TRIVIAL mesh, hence 1.
    self.assertEqual(tpu_mfu._mesh_device_count({'replica': 1, 'data': 1}), 1)
    self.assertEqual(tpu_mfu._mesh_device_count({}), 1)
    self.assertEqual(tpu_mfu._mesh_device_count(None), 1)
    # A real mesh scales by its own size, matching core's train_flops_per_step.
    self.assertEqual(tpu_mfu._mesh_device_count({'data': 4, 'model': 2}), 8)
    # Degenerate sizes never produce a zero or negative factor.
    self.assertEqual(tpu_mfu._mesh_device_count({'data': 0}), 1)


if __name__ == '__main__':
  absltest.main()

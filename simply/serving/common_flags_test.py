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
"""Tests for the auto-wiring of the shared serving common flags."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from simply.serving import common_flags

FLAGS = flags.FLAGS


class CommonFlagsTest(absltest.TestCase):

  def test_covers_every_module_flag_by_name(self):
    # Every flag defined in the module appears in the launch args under its own
    # flag name (and nothing extra is invented).
    module_flag_names = {
        f.name for f in FLAGS.get_flags_for_module(common_flags.__name__)
    }
    self.assertNotEmpty(module_flag_names)
    self.assertEqual(set(common_flags.build_launch_args()), module_flag_names)

  @flagsaver.flagsaver(
      experiment_config='cfg',
      batch_size=8,
      response_asap=True,
      top_k=5,
      temperature=0.7,
  )
  def test_values_pass_through_with_native_types(self):
    args = common_flags.build_launch_args()
    self.assertEqual(args['experiment_config'], 'cfg')
    self.assertEqual(args['batch_size'], 8)
    self.assertIsInstance(args['batch_size'], int)
    self.assertEqual(args['response_asap'], True)
    self.assertIsInstance(args['response_asap'], bool)
    self.assertEqual(args['top_k'], 5)
    self.assertEqual(args['temperature'], 0.7)

  def test_unset_flags_pass_through_as_none(self):
    # Optional flags default to None and must be forwarded as None (matching the
    # previous hand-written dicts), not dropped or stringified.
    with flagsaver.flagsaver(experiment_config='cfg'):
      args = common_flags.build_launch_args()
    self.assertIsNone(args['ckpt_dir'])
    self.assertIsNone(args['ckpt_format'])

  @flagsaver.flagsaver(experiment_config='cfg', mesh_shape=['1', '2', '4'])
  def test_list_flag_is_comma_joined(self):
    args = common_flags.build_launch_args()
    self.assertEqual(args['mesh_shape'], '1,2,4')

  def test_unset_list_flag_is_none(self):
    with flagsaver.flagsaver(experiment_config='cfg'):
      args = common_flags.build_launch_args()
    self.assertIsNone(args['mesh_shape'])


if __name__ == '__main__':
  absltest.main()

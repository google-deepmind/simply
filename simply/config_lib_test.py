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
"""Unit test for config_lib.py."""

from absl.testing import absltest
from simply import config_lib
from simply import model_lib
from simply.utils import module
from simply.utils import pytree


class ConfigLibTest(absltest.TestCase):

  def test_dump_load_baseline_config(self):
    config = config_lib.ExperimentConfigRegistry.get_config(
        'TransformerLMTest'
    )
    sharding_config = config_lib.ShardingConfigRegistry.get_config(
        'GSPMDSharding'
    )
    model_cls = module.ModuleRegistry.get(config.model_name)
    self.assertEqual(model_cls, model_lib.TransformerLM)
    model = model_cls(config, sharding_config)
    py = pytree.dump_dataclasses(model)
    loaded_model = pytree.load_dataclasses(py)
    loaded_py = pytree.dump_dataclasses(loaded_model)
    self.assertEqual(py, loaded_py)


if __name__ == '__main__':
  absltest.main()

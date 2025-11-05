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
from typing import ClassVar

from absl.testing import absltest
from simply.utils import registry


class DummyClassRegistry(registry.RootRegistry):
  """Registry for dummy class."""

  namespace: ClassVar[str] = 'Dummy'


class RegistryTest(absltest.TestCase):

  def test_dummy_registry(self):
    # Save the current registry state
    saved_registry = registry.RootRegistry.registry.copy()
    try:
      registry.RootRegistry.reset()
      @registry.FunctionRegistry.register
      def _dummy_fn():
        return 'hello'

      @DummyClassRegistry.register
      class _DummyClass:
        pass

      self.assertEqual(
          DummyClassRegistry.get('_DummyClass').__name__,
          '_DummyClass',
      )
      self.assertEqual(
          registry.FunctionRegistry.get('_dummy_fn')(), _dummy_fn()
      )
      self.assertSetEqual(
          set(registry.RootRegistry.registry.keys()),
          set(['Dummy:_DummyClass', 'Function:_dummy_fn']),
      )
      self.assertEqual(
          getattr(_DummyClass, '__registered_name__'), 'Dummy:_DummyClass'
      )
      self.assertEqual(
          getattr(_dummy_fn, '__registered_name__'), 'Function:_dummy_fn'
      )
    finally:
      # Restore the original registry state
      registry.RootRegistry.registry = saved_registry


if __name__ == '__main__':
  absltest.main()

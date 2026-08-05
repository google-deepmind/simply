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
"""Tests for the `model_backends` plugin registry."""

from absl.testing import absltest
from simply.eval import model_backends


_KNOBS = dict(temperature=0.7, top_p=0.9, top_k=40, max_decode_steps=128)


class ModelBackendsTest(absltest.TestCase):
  """Dispatch behavior of `ModelBackendRegistry` / `dispatch`."""

  def setUp(self):
    super().setUp()
    # The registry is process-global; snapshot and restore it in place so a
    # test never leaks a backend into another test (or into the real
    # `remote_decode_eval` default registered at import time).
    registry = model_backends.ModelBackendRegistry.registry
    saved = dict(registry)

    def _restore():
      registry.clear()
      registry.update(saved)

    self.addCleanup(_restore)
    registry.clear()
    self.calls = []

  def _factory(self, tag):
    """Returns a factory recording `(tag, uri_body, knobs)` when invoked."""

    def factory(uri_body, **kwargs):
      self.calls.append((tag, uri_body, kwargs))
      return lambda *a, **k: None  # stand-in ModelFn

    return factory

  def test_registered_scheme_gets_uri_body_and_knobs(self):
    model_backends.ModelBackendRegistry.register(
        self._factory('fake'), name='fake'
    )
    model_fn = model_backends.dispatch('fake:host/path:8080', **_KNOBS)
    self.assertIsNotNone(model_fn)
    self.assertEqual(self.calls, [('fake', 'host/path:8080', _KNOBS)])

  def test_unknown_scheme_falls_through_to_default_with_full_address(self):
    model_backends.register_default(self._factory('default'))
    model_fn = model_backends.dispatch('localhost:8500', **_KNOBS)
    self.assertIsNotNone(model_fn)
    # The legacy `host:port` form must reach the default backend intact.
    self.assertEqual(self.calls, [('default', 'localhost:8500', _KNOBS)])

  def test_schemeless_address_falls_through_to_default(self):
    model_backends.register_default(self._factory('default'))
    model_backends.dispatch('/bns/xx/yy/0', **_KNOBS)
    self.assertEqual(self.calls, [('default', '/bns/xx/yy/0', _KNOBS)])

  def test_empty_and_none_address_reach_default_as_empty_string(self):
    model_backends.register_default(self._factory('default'))
    model_backends.dispatch('', **_KNOBS)
    model_backends.dispatch(None, **_KNOBS)
    self.assertEqual(
        self.calls, [('default', '', _KNOBS), ('default', '', _KNOBS)]
    )

  def test_registered_scheme_wins_over_default(self):
    model_backends.register_default(self._factory('default'))
    model_backends.ModelBackendRegistry.register(
        self._factory('fake'), name='fake'
    )
    model_backends.dispatch('fake:body', **_KNOBS)
    self.assertEqual(self.calls, [('fake', 'body', _KNOBS)])

  def test_no_default_registered_returns_none(self):
    self.assertIsNone(model_backends.dispatch('localhost:8500', **_KNOBS))
    self.assertIsNone(model_backends.dispatch(None, **_KNOBS))
    self.assertEmpty(self.calls)

  def test_register_default_returns_the_factory(self):
    factory = self._factory('default')
    self.assertIs(model_backends.register_default(factory), factory)

  def test_unset_knobs_are_passed_through_as_none(self):
    model_backends.register_default(self._factory('default'))
    knobs = dict(temperature=None, top_p=None, top_k=None,
                 max_decode_steps=None)
    model_backends.dispatch('addr', **knobs)
    self.assertEqual(self.calls, [('default', 'addr', knobs)])


if __name__ == '__main__':
  absltest.main()

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
"""Tests for page_batcher.Batcher (prefix-cache retirement).

Focused on `Batcher.clear_prefix_cache()`: the host-side hook rl_lib calls
right after a post-train-step `update_params(...)` to retire the KV prefix
cache (which is content-addressable by tokens only and would otherwise serve
KV cached under the now-stale pre-update weights). We avoid materializing a
real model/mesh: the enabled path injects a fake cache into the
`prefix_cache` cached_property slot, and the disabled path exercises the real
(model-free) `enable_prefix_caching=False` branch.
"""

from absl.testing import absltest
from simply import config_lib
from simply.serving import page_batcher
from simply.utils import lm_format as lm_format_lib


class _FakeCache:
  """Minimal stand-in for PrefixCache recording clear() calls."""

  def __init__(self):
    self.clear_calls = 0

  def clear(self) -> None:
    self.clear_calls += 1


def _make_batcher(enable_prefix_caching: bool) -> page_batcher.Batcher:
  """Builds a Batcher without touching the model/mesh."""
  return page_batcher.Batcher(
      config=config_lib.BaseExperimentConfig(),
      lm_format=lm_format_lib.Pretrain(),
      enable_prefix_caching=enable_prefix_caching,
  )


class ClearPrefixCacheTest(absltest.TestCase):

  def test_enabled_clears_the_cache(self):
    batcher = _make_batcher(enable_prefix_caching=True)
    fake = _FakeCache()
    # Inject the fake into the `prefix_cache` cached_property slot so we don't
    # have to build a real model to materialize the real cache. cached_property
    # reads/writes the instance __dict__ (works even on a frozen dataclass,
    # which only blocks FIELD reassignment, not __dict__ writes).
    batcher.__dict__['prefix_cache'] = fake
    self.assertIs(batcher.prefix_cache, fake)

    batcher.clear_prefix_cache()
    self.assertEqual(fake.clear_calls, 1)
    # Idempotent / repeatable across successive train-step boundaries.
    batcher.clear_prefix_cache()
    self.assertEqual(fake.clear_calls, 2)

  def test_disabled_is_noop(self):
    # Real disabled path: `enable_prefix_caching=False` -> `prefix_cache` is
    # None (no model init), and clear_prefix_cache() must be a safe no-op.
    batcher = _make_batcher(enable_prefix_caching=False)
    self.assertIsNone(batcher.prefix_cache)
    batcher.clear_prefix_cache()  # must not raise.
    self.assertIsNone(batcher.prefix_cache)

  def test_disabled_noop_before_cache_materialized(self):
    # Even if code calls clear before anything ever touched prefix_cache, the
    # disabled branch returns None and clear is a no-op.
    batcher = _make_batcher(enable_prefix_caching=False)
    # Do NOT read batcher.prefix_cache first; call clear directly.
    batcher.clear_prefix_cache()  # must not raise.


if __name__ == '__main__':
  absltest.main()

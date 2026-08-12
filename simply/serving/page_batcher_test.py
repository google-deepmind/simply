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
"""Tests for the prefix-cache drivers in page_batcher.Batcher.

WHAT THESE PROTECT. The batcher carries no cache state at all now: it does
not tell the cache where its coverage ends, it ASKS (`begin_snapshot`
verifies and reports what is really held), and a restore's `start_position`
is a bound rather than a promise. So a cache that has fallen behind a slot is
not representable, and the failure that used to need a flag -- a store that
could not be offloaded -- needs nothing: the next pass asks again, finds the
coverage short and re-stores from there. The slot recovers by itself, which
is what most of this file is about.

The drivers are exercised against a real `PrefixCache` with the device state
faked down to the five fields they read, so the wiring is tested without a
model, a mesh or a TPU. Everything below the drivers lives in
`prefix_cache_test`; everything above them needs real device work.
"""

import dataclasses

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply.serving import page_batcher
from simply.serving import prefix_cache as prefix_cache_lib
from simply.utils import lm_format as lm_format_lib
from simply.utils import ragged_paged_attention as rpa


CHUNK_SIZE = 4


@dataclasses.dataclass(frozen=True)
class _FakeSamplingState:
  """The fields the cache drivers read off the device state, and no more."""

  batch_size: int
  position: np.ndarray
  input_lens: np.ndarray
  tokens: np.ndarray
  is_pad_seq: np.ndarray


def _make_batcher(enable_prefix_caching: bool) -> page_batcher.Batcher:
  """Builds a Batcher without touching the model/mesh."""
  return page_batcher.Batcher(
      config=config_lib.BaseExperimentConfig(),
      lm_format=lm_format_lib.Pretrain(),
      enable_prefix_caching=enable_prefix_caching,
  )


def _entry_tree(value: float) -> dict[str, rpa.SnapshotChunkLeaf]:
  """One chunk's capture, tagged with `value` so its owner is identifiable."""
  return {
      'layer_0': rpa.SnapshotChunkLeaf(
          payload=jnp.full((CHUNK_SIZE, 8), value, dtype=jnp.float32),
          written=jnp.ones((CHUNK_SIZE,), dtype=bool),
      )
  }


def _prefilling_batcher(
    prompts: list[list[int]], positions: list[int]
) -> page_batcher.Batcher:
  """A batcher mid-prefill over `prompts`, with a real cache and no model.

  Args:
    prompts: one token row per slot, all the same length.
    positions: each slot's device position -- how much it has computed.

  Returns:
    The batcher, with a fake `sampling_state` and a fake extract program that
    tags each capture with its slot id.
  """
  batcher = _make_batcher(enable_prefix_caching=True)
  tokens = np.asarray(prompts, dtype=np.int32)
  batch_size = len(prompts)
  batcher.state['sampling_state'] = _FakeSamplingState(  # pyrefly: ignore[unsupported-operation]
      batch_size=batch_size,
      position=np.asarray(positions, dtype=np.int32),
      input_lens=np.full((batch_size,), tokens.shape[1], dtype=np.int32),
      tokens=tokens,
      is_pad_seq=np.zeros((batch_size,), dtype=bool),
  )
  batcher.__dict__['prefix_cache'] = prefix_cache_lib.PrefixCache(
      chunk_size=CHUNK_SIZE
  )
  # `extract_chunk` on the device; here, a capture carrying the slot id so a
  # later read says whose KV it is.
  batcher.__dict__['compiled_extract_chunk_fn'] = (
      lambda state, slot_id, chunk_idx: _entry_tree(float(slot_id))
  )
  # `inject_chunk` on the device: what it does that anything here can see is
  # advance the slot's KV length by the tokens injected.
  def _inject(state, slot_id, payload, start, end):
    del payload, start
    position = np.array(state.position)
    position[slot_id] = end
    return dataclasses.replace(state, position=position)

  batcher.__dict__['compiled_inject_chunk_fn'] = _inject
  return batcher


def _positions(batcher: page_batcher.Batcher) -> np.ndarray:
  """A copy of the slots' device positions, the way the loop reads them."""
  return np.asarray(batcher.sampling_state.position).copy()


class PrefixCacheWiringTest(absltest.TestCase):
  """What the batcher owns on the cache's behalf, without a model."""

  def test_disabled_has_no_cache(self):
    batcher = _make_batcher(enable_prefix_caching=False)
    self.assertIsNone(batcher.prefix_cache)

  def test_the_drivers_are_no_ops_when_caching_is_disabled(self):
    # Both drivers run on every pass of the decode loop whether or not there
    # is a cache, so the disabled path has to be a real no-op -- and "no-op"
    # is now the direct statement: NO POSITION MOVED, nothing was stored, and
    # both drivers report a per-slot ZERO rather than a shape the caller has
    # to special-case (it accumulates the arrays unconditionally).
    batcher = _prefilling_batcher(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], [8]
    )
    batcher.__dict__['prefix_cache'] = None
    before = _positions(batcher)
    np.testing.assert_array_equal(
        batcher._maybe_restore_from_prefix_cache(), [0]  # pylint: disable=protected-access
    )
    np.testing.assert_array_equal(_positions(batcher), before)
    np.testing.assert_array_equal(
        batcher._maybe_snapshot_prefix_cache(np.zeros(1, np.int64)), [0]  # pylint: disable=protected-access
    )
    np.testing.assert_array_equal(_positions(batcher), before)


class PrefixCacheDriverTest(absltest.TestCase):
  """The two drivers, end to end against a real cache.

  Between them they are the cache's only callers in production: one reads the
  deepest cached prefix into a prefilling slot, the other hands back what the
  pass just made real. What is checked here is the accounting they report --
  a restore's report IS the slot's advance -- and that a prefix one slot
  cached is one another slot is offered.
  """

  def test_one_slot_caches_a_prefix_and_another_is_offered_it(self):
    # What the cache is FOR, through the two drivers that use it.
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    batcher = _prefilling_batcher(prompts, [8])
    cache = batcher.prefix_cache
    assert cache is not None
    np.testing.assert_array_equal(
        batcher._maybe_snapshot_prefix_cache(np.zeros(1, np.int64)), [8]  # pylint: disable=protected-access
    )
    # ...and a slot on the same prompt still at position 0 is offered what
    # this one cached -- the restore side of the same guarantee.
    tiles = cache.restore_chunk_tiles(np.asarray(prompts[0]), 0, 11)
    assert tiles
    self.assertEqual(tiles[-1].end, 8)  # the boundary the first pass left

  def test_a_restore_advances_the_slot_by_what_it_injected(self):
    # The measure the response reads, end to end: one slot caches a prefix,
    # a second slot on the same prompt starts from nothing, and the inject
    # driver moves it by exactly the tokens the plan carried. What the driver
    # REPORTS and what the DEVICE did are checked against each other here --
    # the report is derived from the position, so a driver that returned the
    # plan's intent instead of the slot's advance would pass one and fail the
    # other.
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    writer = _prefilling_batcher([prompt], [8])
    np.testing.assert_array_equal(
        writer._maybe_snapshot_prefix_cache(np.zeros(1, np.int64)), [8]  # pylint: disable=protected-access
    )
    cache = writer.prefix_cache
    assert cache is not None

    reader = _prefilling_batcher([prompt], [0])
    reader.__dict__['prefix_cache'] = cache
    before = _positions(reader)
    restored = reader._maybe_restore_from_prefix_cache()  # pylint: disable=protected-access
    after = _positions(reader)
    self.assertEqual(int(after[0] - before[0]), 8, 'the slot did not move')
    np.testing.assert_array_equal(
        restored, after - before, 'the report is not the slot advance'
    )
    np.testing.assert_array_equal(restored, [8])
    # The advance is the plan's tokens, and the plan stops at the boundary
    # the writer's pass left behind.
    tiles = cache.restore_chunk_tiles(np.asarray(prompt), 0, 11)
    assert tiles
    self.assertEqual(sum(tile.num_tokens for tile in tiles), 8)
    self.assertEqual(tiles[-1].end, 8)
    # An immediate second call moves nothing: the slot is now AT the deepest
    # resume point, so there is nothing left to inject -- and the report says
    # zero rather than repeating the eight it restored a moment ago.
    np.testing.assert_array_equal(
        reader._maybe_restore_from_prefix_cache(), [0]  # pylint: disable=protected-access
    )
    np.testing.assert_array_equal(_positions(reader), after)


if __name__ == '__main__':
  absltest.main()

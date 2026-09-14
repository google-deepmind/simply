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
"""Tests for the remote-DMA reshard kernel, in Pallas TPU interpret mode.

Interpret mode simulates TPU memory spaces, remote DMAs and semaphores on CPU
(and can detect data races), so the kernel is exercised without a TPU. The 8
CPU devices have to be requested before JAX initializes its backend, hence the
separate test target.
"""

import collections
import contextlib
import io
import itertools
import math
import os

# Must be set BEFORE `import jax`.
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

# pylint: disable=g-import-not-at-top
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.sharding as js
import numpy as np
from simply.kernels import reshard as reshard_kernel

# pylint: enable=g-import-not-at-top

P = js.PartitionSpec

# Semaphores are pallas_call outputs here (the kernel is split into an issuing
# and a completing phase), and the interpreter cannot NaN-fill a semaphore
# buffer, so allocate zeroed memory.
_INTERPRET = pltpu.InterpretParams(
    detect_races=True, uninitialized_memory='zero'
)


def _mesh(shape, axis_names, order=None):
  devices = np.array(jax.devices())
  if order is not None:
    devices = devices[list(order)]
  return js.Mesh(devices[: math.prod(shape)].reshape(shape), axis_names)


_LAYOUTS = {
    # Every destination tile is built from whole source tiles (or the other way
    # round), so all the boxes have the same shape and the DMA slices are
    # static -- the regime this kernel targets.
    'row_to_column': lambda: (
        (64, 64),
        js.NamedSharding(_mesh((8,), ('x',)), P('x', None)),
        js.NamedSharding(_mesh((8,), ('x',)), P(None, 'x')),
    ),
    'tile_aspect_ratio': lambda: (
        (32, 32),
        js.NamedSharding(_mesh((2, 4), ('a', 'b')), P('a', 'b')),
        js.NamedSharding(_mesh((4, 2), ('a', 'b')), P('a', 'b')),
    ),
    'shard_to_replicate': lambda: (
        (64, 16),
        js.NamedSharding(_mesh((8,), ('x',)), P('x', None)),
        js.NamedSharding(_mesh((8,), ('x',)), P(None, None)),
    ),
    'replicate_to_shard': lambda: (
        (64, 16),
        js.NamedSharding(_mesh((8,), ('x',)), P(None, None)),
        js.NamedSharding(_mesh((8,), ('x',)), P('x', None)),
    ),
    'eight_way_to_four_way': lambda: (
        (64, 16),
        js.NamedSharding(_mesh((8,), ('x',)), P('x', None)),
        js.NamedSharding(_mesh((2, 4), ('a', 'b')), P('b', None)),
    ),
    'shuffled_device_order': lambda: (
        (32, 32),
        js.NamedSharding(_mesh((8,), ('x',)), P('x', None)),
        js.NamedSharding(
            _mesh((8,), ('x',), order=[3, 1, 7, 0, 5, 2, 6, 4]), P('x', None)
        ),
    ),
}

_CASES = tuple((name, name) for name in _LAYOUTS)


class ReshardKernelTest(parameterized.TestCase):

  def tearDown(self):
    super().tearDown()
    pltpu.reset_tpu_interpret_mode_state()

  @parameterized.named_parameters(*_CASES)
  def test_matches_reference(self, layout):
    shape, src, dst = _LAYOUTS[layout]()
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    # The interpreter reports races on stdout; a clean run must stay silent.
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
      out = reshard_kernel.reshard(
          jax.device_put(x, src),
          dst,
          interpret=_INTERPRET,
      )
      out.block_until_ready()
    np.testing.assert_array_equal(np.asarray(out), x)
    self.assertEqual(out.sharding, dst)
    self.assertNotIn('RACE DETECTED', log.getvalue())

  def test_moves_only_the_planned_tiles(self):
    # One DMA per box that has to move, and not one byte more: the plan must
    # match the independently computed minimum (what each destination device
    # needs, minus what it already holds).
    shape = (32, 32)
    src = js.NamedSharding(_mesh((2, 4), ('a', 'b')), P('a', 'b'))
    dst = js.NamedSharding(_mesh((4, 2), ('a', 'b')), P('a', 'b'))
    src_tiling = reshard_kernel._Tiling(shape, src)  # pylint: disable=protected-access
    dst_tiling = reshard_kernel._Tiling(shape, dst)  # pylint: disable=protected-access
    transfers = reshard_kernel._plan_transfers(src_tiling, dst_tiling)  # pylint: disable=protected-access

    def cells(tiling, device_id):
      origin = tiling.offset(device_id)
      ranges = [range(o, o + n) for o, n in zip(origin, tiling.shard_shape)]
      return set(itertools.product(*ranges))

    expected = 0
    for device_id in dst_tiling.tile_index:
      needed = cells(dst_tiling, device_id)
      expected += len(needed - cells(src_tiling, device_id))
    moved = sum(math.prod(t.box) for t in transfers if not t.is_local)
    self.assertEqual(moved, expected)

    # And every destination cell is written exactly once.
    written = collections.Counter()
    for t in transfers:
      origin = dst_tiling.offset(t.dst_device)
      base = [o + d for o, d in zip(origin, t.dst_offset)]
      for cell in itertools.product(
          *[range(b, b + n) for b, n in zip(base, t.box)]
      ):
        written[(t.dst_device, cell)] += 1
    self.assertEqual(set(written.values()), {1})
    self.assertLen(
        written, len(dst_tiling.tile_index) * math.prod(dst_tiling.shard_shape)
    )

  @parameterized.named_parameters(*_CASES)
  def test_offsets_are_aligned_to_the_box(self, layout):
    # The kernel tells Mosaic that every dynamic DMA offset is a multiple of
    # the box, which is an assumption the compiler trusts rather than checks --
    # and interpret mode does not model tiling, so nothing else here would
    # catch a violation. Assert it directly on the plan.
    shape, src, dst = _LAYOUTS[layout]()
    transfers = reshard_kernel._plan_transfers(  # pylint: disable=protected-access
        reshard_kernel._Tiling(shape, src),  # pylint: disable=protected-access
        reshard_kernel._Tiling(shape, dst),  # pylint: disable=protected-access
    )
    for t in transfers:
      for i, size in enumerate(t.box):
        self.assertEqual(t.src_offset[i] % size, 0)
        self.assertEqual(t.dst_offset[i] % size, 0)

  def test_start_and_wait_are_separate_programs(self):
    # The split kernels cannot run under the interpreter (it cannot allocate a
    # semaphore-typed output), so check that they trace and that the future
    # carries the semaphores from one to the other. Running them needs a TPU.
    shape = (32, 32)
    src = js.NamedSharding(_mesh((8,), ('x',)), P('x', None))
    dst = js.NamedSharding(_mesh((8,), ('x',)), P(None, 'x'))
    compiled = reshard_kernel._build(  # pylint: disable=protected-access
        shape, np.dtype(np.float32), src, dst, None, 0
    )
    stacked = jax.ShapeDtypeStruct((8, 4, 32), np.float32)
    buffer = jax.ShapeDtypeStruct((8, 32, 4), np.float32)
    metadata = [
        jax.ShapeDtypeStruct(a.shape, a.dtype)
        for a in compiled.operands.arrays()
    ]
    out, sems = jax.eval_shape(
        compiled.start_program, stacked, buffer, *metadata
    )
    self.assertEqual(out.shape, buffer.shape)
    waited = jax.eval_shape(
        compiled.wait_program, stacked, buffer, *metadata, sems
    )
    self.assertEqual(waited.shape, buffer.shape)

  def test_mixed_box_shapes_are_rejected(self):
    # A destination tile that straddles source tiles unevenly needs DMAs of
    # different shapes, which this prototype does not lower.
    shape = (48,)
    src = js.NamedSharding(_mesh((8,), ('x',)), P('x'))
    dst = js.NamedSharding(_mesh((6,), ('y',)), P('y'))
    with self.assertRaises(NotImplementedError):
      reshard_kernel.reshard(
          jax.device_put(np.zeros(shape, np.float32), src),
          dst,
          interpret=_INTERPRET,
      )


if __name__ == '__main__':
  absltest.main()

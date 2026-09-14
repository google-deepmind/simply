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
"""Resharding by explicit remote DMAs, as a Pallas TPU kernel.

Make it feasiable to reshard efficiently from an abitrary source sharding to
another arbitrary destination sharding without all gather.

Restrictions (`reshard` raises if they are not met; `jax.device_put`
handles the general case):
  * every transfer in the plan must have the same box shape, which holds
    whenever the two tilings' cuts are nested (equal chip counts, evenly
    divisible shapes) -- the DMA slice shapes have to be static;
  * TPU only, and within a single pod: remote DMA cannot cross DCN.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jax.sharding as js
import numpy as np

_MESH_AXIS = 'reshard_device'
# Barrier namespace. Any value works as long as reshards that are in flight at
# the same time do not share one.
_DEFAULT_COLLECTIVE_ID = 27
# counts, send_dev, send_src, send_dst, recv_dev, recv_dst, local_src, local_dst
_NUM_METADATA_ARRAYS = 8


@dataclasses.dataclass(frozen=True)
class _Tiling:
  """Which tile of a global array each device of a sharding holds.

  Everything is derived from the two fields on demand, so a tiling is cheap to
  pass around and safe to reuse: it is hashable and nothing about it can drift
  out of sync with the sharding it describes.

  Attributes:
    shape: the global shape it is applied to, kept as a tuple so that a tiling
      is hashable.
    sharding: the sharding being described.
  """

  shape: Sequence[int]
  sharding: js.NamedSharding

  def __post_init__(self):
    if not self.is_even():
      raise ValueError(
          f'reshard() needs evenly divisible shardings, got {self.sharding}'
          f' for shape {self.shape}'
      )

  @functools.cached_property
  def axes(self) -> tuple[tuple[str, ...], ...]:
    """Returns the mesh axes each array dimension is partitioned over."""
    spec = self.sharding.spec
    entries = []
    for i in range(len(self.shape)):
      entry = spec[i] if i < len(spec) else None
      if entry is None:
        entries.append(())
      elif isinstance(entry, str):
        entries.append((entry,))
      else:
        entries.append(tuple(entry))
    return tuple(entries)

  @functools.cached_property
  def num_shards(self) -> tuple[int, ...]:
    """Returns how many shards each array dimension is cut into."""
    mesh_shape = self.sharding.mesh.shape
    return tuple(math.prod(mesh_shape[a] for a in e) for e in self.axes)

  @functools.cached_property
  def bounds(self) -> tuple[np.ndarray, ...]:
    """Returns the shard boundaries along each array dimension."""
    return tuple(
        np.arange(0, n + 1, step=n // k, dtype=np.int32)
        for n, k in zip(self.shape, self.num_shards)
    )

  @functools.cached_property
  def shard_shape(self) -> tuple[int, ...]:
    """Returns the per-device buffer shape, the same on every device."""
    return tuple(n // k for n, k in zip(self.shape, self.num_shards))

  @functools.cached_property
  def tile_index(self) -> dict[int, tuple[int, ...]]:
    """Returns the tile each device holds, as an index per array dimension."""
    mesh = self.sharding.mesh
    axis_pos = {a: i for i, a in enumerate(mesh.axis_names)}
    tile_index = {}
    for coord, device in np.ndenumerate(mesh.devices):
      index = []
      for entry in self.axes:
        flat = 0
        for axis in entry:
          flat = flat * mesh.shape[axis] + coord[axis_pos[axis]]
        index.append(flat)
      tile_index[device.id] = tuple(index)
    return tile_index

  @functools.cached_property
  def devices(self) -> dict[int, jax.Device]:
    """Returns the devices of the mesh, keyed by device id."""
    return {d.id: d for d in self.sharding.mesh.devices.flat}

  def offset(self, device_id: int) -> tuple[int, ...]:
    """Returns the global offset of the tile held by a device."""
    return tuple(
        int(self.bounds[i][j]) for i, j in enumerate(self.tile_index[device_id])
    )

  def is_even(self) -> bool:
    """Returns whether every dimension divides evenly into its shards."""
    return all(n % k == 0 for n, k in zip(self.shape, self.num_shards))


def _axis_overlaps(src_bounds: np.ndarray, dst_bounds: np.ndarray):
  """For every dst shard of one axis, the src shards it overlaps.

  Args:
    src_bounds: source shard boundaries along the axis.
    dst_bounds: destination shard boundaries along the axis.

  Returns:
    A list indexed by destination shard of `(src_shard, lo, hi)`, where
    `[lo, hi)` is the overlapping global interval. Two-pointer sweep, so the
    work is proportional to the number of overlaps rather than to the product
    of the shard counts.
  """
  overlaps = []
  first, num_src = 0, len(src_bounds) - 1
  for j in range(len(dst_bounds) - 1):
    dst_lo, dst_hi = int(dst_bounds[j]), int(dst_bounds[j + 1])
    current = []
    if dst_hi > dst_lo:
      while first < num_src and int(src_bounds[first + 1]) <= dst_lo:
        first += 1
      i = first
      while i < num_src and int(src_bounds[i]) < dst_hi:
        lo = max(dst_lo, int(src_bounds[i]))
        hi = min(dst_hi, int(src_bounds[i + 1]))
        if hi > lo:
          current.append((i, lo, hi))
        i += 1
    overlaps.append(current)
  # For each destination shard, the source shards it overlaps.
  return overlaps


def _replica_cost(src, dst, load, dst_device, candidate):
  """Ranks a replica of a source tile for a given destination device.

  Closest first, ties broken towards the least loaded replica so that a
  replicated source does not funnel all of its traffic through one device.

  Args:
    src: the source tiling.
    dst: the destination tiling.
    load: elements already assigned to each sender.
    dst_device: the device that needs the box.
    candidate: a device holding the box.

  Returns:
    A sort key; smaller is better.
  """
  return (
      src.devices[candidate].process_index
      != dst.devices[dst_device].process_index,
      load[candidate],
      abs(candidate - dst_device),
  )


@dataclasses.dataclass(frozen=True)
class _Transfer:
  """One box moving from one device to another (possibly to itself)."""

  src_device: int
  dst_device: int
  src_offset: tuple[int, ...]  # In the sender's shard.
  dst_offset: tuple[int, ...]  # In the receiver's shard.
  box: tuple[int, ...]

  @property
  def is_local(self) -> bool:
    return self.src_device == self.dst_device


def _plan_transfers(src: _Tiling, dst: _Tiling) -> list[_Transfer]:
  """Lists the tiles that have to move, one entry per (box, destination).

  A destination device only needs the part of its tile it does not already
  hold, and every such region is served by exactly one source, so the transfers
  add up to the minimum traffic. Overlap factorizes per dimension, so only the
  tile pairs that really overlap are enumerated.

  Args:
    src: tiling the data currently has.
    dst: tiling it should end up with.

  Returns:
    The transfers, with local ones (a device serving itself) included.

  Raises:
    ValueError: if the two tilings describe different global shapes.
  """
  if tuple(src.shape) != tuple(dst.shape):
    raise ValueError(f'shape mismatch: {src.shape} vs {dst.shape}')
  overlaps = [
      _axis_overlaps(src.bounds[i], dst.bounds[i])
      for i in range(len(src.shape))
  ]  # shape -> dst_shard -> src_shard
  holders: dict[tuple[int, ...], list[int]] = collections.defaultdict(list)
  for device_id, index in src.tile_index.items():
    holders[index].append(device_id)
  for candidates in holders.values():
    candidates.sort()
  load: collections.Counter[int] = collections.Counter()

  transfers = []
  for dst_device, dst_index in sorted(dst.tile_index.items()):
    dst_origin = dst.offset(dst_device)
    for combination in itertools.product(
        *[overlaps[i][j] for i, j in enumerate(dst_index)]
    ):
      src_index = tuple(c[0] for c in combination)
      box = tuple(c[2] - c[1] for c in combination)
      if not math.prod(box):
        continue
      candidates = holders[src_index]
      if (
          dst_device in src.tile_index
          and src.tile_index[dst_device] == src_index
      ):
        src_device = dst_device  # Already here, no transfer needed.
      else:
        # Pick the closest replica, breaking ties towards the least loaded one
        # so that a replicated source does not funnel through one device.
        src_device = min(
            candidates,
            key=functools.partial(_replica_cost, src, dst, load, dst_device),
        )
        load[src_device] += math.prod(box)
      src_origin = tuple(
          int(src.bounds[i][src_index[i]]) for i in range(len(src.shape))
      )
      transfers.append(
          _Transfer(
              src_device=src_device,
              dst_device=dst_device,
              src_offset=tuple(
                  c[1] - o for c, o in zip(combination, src_origin)
              ),
              dst_offset=tuple(
                  c[1] - o for c, o in zip(combination, dst_origin)
              ),
              box=box,
          )
      )
  return transfers


class _Operands:
  """Per-device description of the plan, as small int32 arrays.

  Every array has the devices of the flat reshard mesh on the leading axis, so
  `shard_map` hands each device exactly its own row.
  """

  def __init__(
      self, transfers: Sequence[_Transfer], devices: Sequence[jax.Device]
  ):
    if not transfers:
      raise ValueError('nothing to reshard')
    shapes = {t.box for t in transfers}
    if len(shapes) != 1:
      raise NotImplementedError(
          'the DMA kernel needs one box shape for the whole plan, got '
          f'{sorted(shapes)}; fall back to jax.device_put'
      )
    self.box = shapes.pop()
    # The kernel tells Mosaic that every DMA offset is a multiple of the box
    # (see `_box_at`); that follows from the boxes all having one shape, but it
    # is an assumption the compiler trusts, so check it rather than reason it.
    for t in transfers:
      for i, size in enumerate(self.box):
        if t.src_offset[i] % size or t.dst_offset[i] % size:
          raise NotImplementedError(
              f'the DMA kernel needs offsets aligned to the box {self.box},'
              f' got src {t.src_offset} dst {t.dst_offset}; fall back to'
              ' jax.device_put'
          )
    ndim = len(self.box)
    n = len(devices)
    position = {d.id: i for i, d in enumerate(devices)}

    outgoing = collections.defaultdict(list)
    incoming = collections.defaultdict(list)
    local = {}
    for t in transfers:
      if t.is_local:
        local[t.dst_device] = t
      else:
        outgoing[t.src_device].append(t)
        incoming[t.dst_device].append(t)
    self.num_slots = max(
        max((len(v) for v in outgoing.values()), default=0),
        max((len(v) for v in incoming.values()), default=0),
        1,
    )

    k = self.num_slots
    self.counts = np.zeros((n, 3), np.int32)  # sends, recvs, has_local
    self.send_dev = np.zeros((n, k), np.int32)
    self.send_src = np.zeros((n, k, ndim), np.int32)
    self.send_dst = np.zeros((n, k, ndim), np.int32)
    self.recv_dev = np.zeros((n, k), np.int32)
    self.recv_dst = np.zeros((n, k, ndim), np.int32)
    self.local_src = np.zeros((n, ndim), np.int32)
    self.local_dst = np.zeros((n, ndim), np.int32)

    for device, ts in outgoing.items():
      row = position[device]
      self.counts[row, 0] = len(ts)
      for i, t in enumerate(ts):
        self.send_dev[row, i] = position[t.dst_device]
        self.send_src[row, i] = t.src_offset
        self.send_dst[row, i] = t.dst_offset
    for device, ts in incoming.items():
      row = position[device]
      self.counts[row, 1] = len(ts)
      for i, t in enumerate(ts):
        self.recv_dev[row, i] = position[t.src_device]
        self.recv_dst[row, i] = t.dst_offset
    for device, t in local.items():
      row = position[device]
      self.counts[row, 2] = 1
      self.local_src[row] = t.src_offset
      self.local_dst[row] = t.dst_offset

  def arrays(self):
    return (
        self.counts,
        self.send_dev,
        self.send_src,
        self.send_dst,
        self.recv_dev,
        self.recv_dst,
        self.local_src,
        self.local_dst,
    )


def _box_at(ref, offsets, row, box):
  """Returns `ref` sliced to `box` at the offsets held in an SMEM ref.

  Mosaic rejects a dynamic slice of a tiled memref unless the offset is
  provably a multiple of the tile, and an offset loaded from SMEM carries no
  such proof, hence the explicit `pl.multiple_of`. The box is a valid divisor
  because every plan this kernel accepts is built from boxes of one shape laid
  out contiguously; `_Operands` checks that on the actual offsets.

  Args:
    ref: the shard to slice.
    offsets: SMEM ref holding the per-dimension offsets.
    row: index into `offsets` when it holds one entry per DMA slot, else None.
    box: the shape to slice out.

  Returns:
    A view of `ref` covering `box` at those offsets.
  """
  slices = tuple(
      pl.ds(
          pl.multiple_of(
              offsets[(0, i)] if row is None else offsets[(0, row, i)], size
          ),
          size,
      )
      for i, size in enumerate(box)
  )
  return ref.at[(0, *slices)]


def _body(x_ref, out_ref, meta, sems, *, box, num_slots, phase):
  """Issues and/or completes every DMA this device owes.

  Both phases rebuild the same descriptors -- they are only metadata -- so the
  completing phase knows exactly which semaphores to block on.

  Args:
    x_ref: this device's shard of the source array.
    out_ref: this device's shard of the destination array.
    meta: the per-device plan arrays, in `_Operands.arrays()` order.
    sems: DMA semaphores: `num_slots` for sends, then one recv and one local.
    box: shape of every box in the plan.
    num_slots: static upper bound on sends (and receives) per device.
    phase: 'start' to issue the DMAs, 'wait' to complete them, 'fused' for both.
  """
  (
      counts,
      send_dev,
      send_src,
      send_dst,
      recv_dev,
      recv_dst,
      local_src,
      local_dst,
  ) = meta
  num_sends, num_recvs, has_local = counts[0, 0], counts[0, 1], counts[0, 2]
  recv_sem, local_sem = sems.at[num_slots], sems.at[num_slots + 1]

  def send(k):
    return pltpu.make_async_remote_copy(
        src_ref=_box_at(x_ref, send_src, k, box),
        dst_ref=_box_at(out_ref, send_dst, k, box),
        send_sem=sems.at[k],
        recv_sem=recv_sem,
        device_id=(send_dev[0, k],),
        device_id_type=pl.DeviceIdType.MESH,
    )

  def local_copy():
    return pltpu.make_async_copy(
        _box_at(x_ref, local_src, None, box),
        _box_at(out_ref, local_dst, None, box),
        local_sem,
    )

  def receive(r):
    # Only the semaphore and the size matter here; the box never leaves.
    return pltpu.make_async_remote_copy(
        src_ref=_box_at(out_ref, recv_dst, r, box),  # dummy
        dst_ref=_box_at(out_ref, recv_dst, r, box),
        send_sem=sems.at[0],  # dummy
        recv_sem=recv_sem,
        device_id=(recv_dev[0, r],),
        device_id_type=pl.DeviceIdType.MESH,
    )

  if phase in ('start', 'fused'):
    # A remote DMA writes straight into the destination's buffer, so a sender
    # must know that the receiver has entered the kernel. Tell everyone who
    # sends to me that I am ready, then wait for all of my receivers.
    barrier = pltpu.get_barrier_semaphore()
    for r in range(num_slots):

      @pl.when(r < num_recvs)
      def _signal_ready(r=r):
        pl.semaphore_signal(
            barrier,
            device_id=(recv_dev[0, r],),
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(num_sends > 0)
    def _await_receivers():
      pl.semaphore_wait(barrier, num_sends)

    @pl.when(has_local > 0)
    def _start_local():
      local_copy().start()

    for k in range(num_slots):

      @pl.when(k < num_sends)
      def _start_send(k=k):
        send(k).start()

  if phase in ('wait', 'fused'):
    # Every box has the same size, so waiting on the shared receive semaphore
    # once per incoming message is exactly "wait until they have all landed".
    for r in range(num_slots):

      @pl.when(r < num_recvs)
      def _await_recv(r=r):
        receive(r).wait_recv()

    for k in range(num_slots):

      @pl.when(k < num_sends)
      def _await_send(k=k):
        send(k).wait_send()

    @pl.when(has_local > 0)
    def _await_local():
      local_copy().wait()

    # Nobody leaves until its partners are done, so that a following reshard
    # cannot overwrite a buffer that is still being read or filled.
    @functools.partial(pl.run_scoped, done=pltpu.SemaphoreType.REGULAR)
    def _final_barrier(done):
      for k in range(num_slots):

        @pl.when(k < num_sends)
        def _tell_receiver(k=k):
          pl.semaphore_signal(
              done,
              device_id=(send_dev[0, k],),
              device_id_type=pl.DeviceIdType.MESH,
          )

      for r in range(num_slots):

        @pl.when(r < num_recvs)
        def _tell_sender(r=r):
          pl.semaphore_signal(
              done,
              device_id=(recv_dev[0, r],),
              device_id_type=pl.DeviceIdType.MESH,
          )

      @pl.when(num_sends + num_recvs > 0)
      def _await_partners():
        pl.semaphore_wait(done, num_sends + num_recvs)


def _fused_kernel(x_ref, *refs, box, num_slots):
  *meta, out_ref, sems = refs
  _body(
      x_ref,
      out_ref,
      meta,
      sems,
      box=box,
      num_slots=num_slots,
      phase='fused',
  )


def _start_kernel(x_ref, buffer_ref, *refs, box, num_slots):
  del buffer_ref  # Aliased into out_ref.
  *meta, out_ref, sems = refs
  _body(
      x_ref,
      out_ref,
      meta,
      sems,
      box=box,
      num_slots=num_slots,
      phase='start',
  )


def _wait_kernel(x_ref, buffer_ref, *refs, box, num_slots):
  del buffer_ref  # Aliased into out_ref.
  *meta, sems, out_ref = refs
  _body(
      x_ref,
      out_ref,
      meta,
      sems,
      box=box,
      num_slots=num_slots,
      phase='wait',
  )


@dataclasses.dataclass(frozen=True)
class _Compiled:
  """Everything needed to run one (shape, dtype, src, dst) reshard."""

  fused_program: Callable[..., jax.Array]
  start_program: Callable[..., tuple[jax.Array, jax.Array]]
  wait_program: Callable[..., jax.Array]
  operands: _Operands
  mesh: js.Mesh
  devices: tuple[jax.Device, ...]
  src_shard_shape: tuple[int, ...]
  dst_shard_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ReshardFuture:
  """A reshard whose DMAs are in flight; call `wait` to collect it.

  Attributes:
    compiled: the programs and plan this reshard was built from.
    shape: the global shape being resharded.
    dst_sharding: the sharding the result will have.
    source: the source shards, staged on the reshard mesh.
    buffer: the destination buffer the DMAs are landing in.
    semaphores: the DMA semaphores the issuing kernel left behind.
    operands: the per-device plan metadata.
  """

  compiled: _Compiled
  shape: tuple[int, ...]
  dst_sharding: js.NamedSharding
  source: jax.Array
  buffer: jax.Array
  semaphores: jax.Array
  operands: tuple[jax.Array, ...]

  def wait(self) -> jax.Array:
    """Waits for the transfers and returns the resharded array."""
    out = self.compiled.wait_program(
        self.source, self.buffer, *self.operands, self.semaphores
    )
    return _unstack(out, self.shape, self.dst_sharding, self.compiled)


@functools.lru_cache(maxsize=32)
def _build(shape, dtype, src_sharding, dst_sharding, interpret, collective_id):
  """Plans the reshard and builds the issuing and completing programs."""
  src_geom = _Tiling(shape, src_sharding)
  dst_geom = _Tiling(shape, dst_sharding)
  devices = tuple(
      sorted(
          set(src_sharding.mesh.devices.flat)
          | set(dst_sharding.mesh.devices.flat),
          key=lambda d: d.id,
      )
  )
  transfers = _plan_transfers(src_geom, dst_geom)
  operands = _Operands(transfers, devices)
  mesh = js.Mesh(np.array(devices, dtype=object), (_MESH_AXIS,))
  spec = js.PartitionSpec(_MESH_AXIS)

  out_struct = jax.ShapeDtypeStruct((1, *dst_geom.shard_shape), dtype)
  sems_struct = pltpu.SemaphoreType.DMA((operands.num_slots + 2,))
  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  metadata = [pl.BlockSpec(memory_space=pltpu.SMEM)] * _NUM_METADATA_ARRAYS
  sem_spec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  params = pltpu.CompilerParams(collective_id=collective_id)
  config = dict(box=operands.box, num_slots=operands.num_slots)

  # Fused: the semaphores live in scratch memory and never leave the kernel.
  fused_call = pl.pallas_call(
      functools.partial(_fused_kernel, **config),
      out_shape=out_struct,
      in_specs=[any_spec] + metadata,
      out_specs=any_spec,
      scratch_shapes=[sems_struct],
      compiler_params=params,
      interpret=interpret,
  )
  # Split: the semaphores are handed from the issuing to the completing kernel,
  # and the destination buffer is aliased through both of them.
  start_call = pl.pallas_call(
      functools.partial(_start_kernel, **config),
      out_shape=[out_struct, sems_struct],
      in_specs=[any_spec, any_spec] + metadata,
      out_specs=[any_spec, sem_spec],
      input_output_aliases={1: 0},
      compiler_params=params,
      interpret=interpret,
  )
  wait_call = pl.pallas_call(
      functools.partial(_wait_kernel, **config),
      out_shape=out_struct,
      in_specs=[any_spec, any_spec] + metadata + [sem_spec],
      out_specs=any_spec,
      input_output_aliases={1: 0},
      # No collective_id: the completing phase never touches the barrier
      # semaphore, and Mosaic rejects a collective_id without one.
      compiler_params=pltpu.CompilerParams(has_side_effects=True),
      interpret=interpret,
  )

  n_in = 2 + _NUM_METADATA_ARRAYS

  def shard(call, num_inputs, out_specs):
    return jax.jit(
        jax.shard_map(
            call,
            mesh=mesh,
            in_specs=(spec,) * num_inputs,
            out_specs=out_specs,
            check_vma=False,
        )
    )

  return _Compiled(
      fused_program=shard(fused_call, n_in - 1, spec),
      start_program=shard(start_call, n_in, (spec, spec)),
      wait_program=shard(wait_call, n_in + 1, spec),
      operands=operands,
      mesh=mesh,
      devices=devices,
      src_shard_shape=src_geom.shard_shape,
      dst_shard_shape=dst_geom.shard_shape,
  )


def _rows_on_devices(rows, mesh, devices):
  """Places row `i` of `rows` on the `i`-th device of the reshard mesh."""
  sharding = js.NamedSharding(mesh, js.PartitionSpec(_MESH_AXIS))
  index = {d.id: i for i, d in enumerate(devices)}
  return jax.make_array_from_single_device_arrays(
      rows.shape,
      sharding,
      [
          jax.device_put(rows[index[d.id]][None], d)
          for d in sharding.addressable_devices_indices_map(rows.shape)
      ],
  )


def _stack_shards(x, compiled):
  """Views the shards of `x` as one leading-axis-sharded array (no transfer)."""
  sharding = js.NamedSharding(compiled.mesh, js.PartitionSpec(_MESH_AXIS))
  stacked_shape = (len(compiled.devices), *compiled.src_shard_shape)
  shards = {s.device.id: s.data for s in x.addressable_shards}
  local = []
  for device in sharding.addressable_devices_indices_map(stacked_shape):
    shard = shards.get(device.id)
    if shard is None:
      # A device that holds no part of the source; it only receives.
      shard = jnp.zeros(compiled.src_shard_shape, x.dtype, device=device)
    local.append(shard[None, ...])
  return jax.make_array_from_single_device_arrays(
      stacked_shape, sharding, local
  )


def reshard_start(
    x: jax.Array,
    dst_sharding: js.NamedSharding,
    *,
    collective_id: int = _DEFAULT_COLLECTIVE_ID,
    interpret: pltpu.InterpretParams | None = None,
) -> ReshardFuture:
  """Fires every DMA the reshard needs and returns without waiting for them.

  Call `.wait()` on the result to get the resharded array. Between the two the
  transfers proceed in the background, so several reshards can be in flight
  at once and overlapped with compute -- give each one its own `collective_id`.

  Args:
    x: the array to reshard; its sharding must be a `js.NamedSharding`.
    dst_sharding: the target sharding, over any mesh in the same pod.
    collective_id: barrier namespace; must differ between reshards that are in
      flight simultaneously.
    interpret: pass `pltpu.InterpretParams()` to run (and race-check) the kernel
      on CPU instead of a TPU.

  Returns:
    A `ReshardFuture`.

  Raises:
    NotImplementedError: if the plan mixes box shapes, which static DMA slices
      cannot express (fall back to `jax.device_put`).
    ValueError: for unsupported layouts (uneven shards, non-`NamedSharding`).
  """
  if not isinstance(x.sharding, js.NamedSharding):
    raise ValueError(f'reshard() needs a NamedSharding input, got {x.sharding}')
  shape = tuple(x.shape)
  compiled = _build(
      shape, x.dtype, x.sharding, dst_sharding, interpret, collective_id
  )
  source = _stack_shards(x, compiled)
  sharding = js.NamedSharding(compiled.mesh, js.PartitionSpec(_MESH_AXIS))
  buffer = jax.jit(
      lambda: jnp.zeros(
          (len(compiled.devices), *compiled.dst_shard_shape), x.dtype
      ),
      out_shardings=sharding,
  )()
  operands = tuple(
      _rows_on_devices(a, compiled.mesh, compiled.devices)
      for a in compiled.operands.arrays()
  )
  buffer, semaphores = compiled.start_program(source, buffer, *operands)
  return ReshardFuture(
      compiled=compiled,
      shape=shape,
      dst_sharding=dst_sharding,
      source=source,
      buffer=buffer,
      semaphores=semaphores,
      operands=operands,
  )


def _unstack(out, shape, dst_sharding, compiled):
  """Reinterprets the per-device output shards as one `dst_sharding` array."""
  shards = {s.device.id: s.data for s in out.addressable_shards}
  return jax.make_array_from_single_device_arrays(
      shape,
      dst_sharding,
      [
          shards[d.id].reshape(compiled.dst_shard_shape)
          for d in dst_sharding.addressable_devices_indices_map(shape)
      ],
  )


def reshard(
    x: jax.Array,
    dst_sharding: js.NamedSharding,
    *,
    collective_id: int = _DEFAULT_COLLECTIVE_ID,
    interpret: pltpu.InterpretParams | None = None,
) -> jax.Array:
  """Reshards `x` with one remote DMA per tile that has to move.

  Issues and completes the transfers in a single kernel. Use `reshard_start`
  and `ReshardFuture.wait` instead to overlap the transfers with other work.

  Args:
    x: the array to reshard; its sharding must be a `js.NamedSharding`.
    dst_sharding: the target sharding, over any mesh in the same pod.
    collective_id: barrier namespace, see `reshard_start`.
    interpret: pass `pltpu.InterpretParams()` to run (and race-check) the kernel
      on CPU instead of a TPU.

  Returns:
    An array with the same value as `x` and sharding `dst_sharding`.

  Raises:
    NotImplementedError: if the plan mixes box shapes, which static DMA slices
      cannot express (fall back to `jax.device_put`).
    ValueError: for unsupported layouts (uneven shards, non-`NamedSharding`).
  """
  if not isinstance(x.sharding, js.NamedSharding):
    raise ValueError(f'reshard() needs a NamedSharding input, got {x.sharding}')
  shape = tuple(x.shape)
  compiled = _build(
      shape, x.dtype, x.sharding, dst_sharding, interpret, collective_id
  )
  operands = tuple(
      _rows_on_devices(a, compiled.mesh, compiled.devices)
      for a in compiled.operands.arrays()
  )
  out = compiled.fused_program(_stack_shards(x, compiled), *operands)
  return _unstack(out, shape, dst_sharding, compiled)

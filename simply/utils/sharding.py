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
"""Sharding utilities."""

import asyncio
import collections
from collections.abc import Callable, MutableMapping, Sequence
import contextlib
import dataclasses
import functools
import os
import time
from typing import Any

from absl import logging
from etils import epath
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply.utils import common
from simply.utils import pytree


PartitionAnnotation = common.PartitionAnnotation
MESH_CONTEXT_KEY: str = 'mesh_context'
NOT_ANNOTATED = 'NOT_ANNOTATED'


@contextlib.contextmanager
def mesh_context(
    mesh_shape: Sequence[int], dcn_mesh_shape: Sequence[int] | None = None
):
  set_default_mesh_shape(mesh_shape=mesh_shape, dcn_mesh_shape=dcn_mesh_shape)
  try:
    yield common.THREAD_CONTEXT.mesh_context[-1]
  finally:
    common.THREAD_CONTEXT.mesh_context.pop()


def set_default_mesh_shape(
    *,
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int] | None = None,
):
  context = getattr(common.THREAD_CONTEXT, MESH_CONTEXT_KEY, None)
  if context is None:
    context = []
    setattr(common.THREAD_CONTEXT, MESH_CONTEXT_KEY, context)
  context.append(create_mesh(mesh_shape, dcn_mesh_shape))


def get_mesh_shape(num_devices):
  return (num_devices, 1, 1)


def create_mesh(mesh_shape=None, dcn_mesh_shape=None, print_debug_info=False):
  """Creates mesh for the current device set.

  Full replica parallelism is used if mesh_shape is not provided.

  Args:
    mesh_shape: The mesh shape.
    dcn_mesh_shape: The mesh shape for the dcn devices.
    print_debug_info: Whether to print debug info.

  Returns:
    The mesh.
  """
  num_devices = len(jax.devices())
  if mesh_shape is None:
    mesh_shape = get_mesh_shape(num_devices)
  if len(mesh_shape) == 2:
    mesh_shape = (1, *mesh_shape)
  if dcn_mesh_shape and dcn_mesh_shape[0] > 1:
    if print_debug_info:
      print(f'hybrid, ici mesh_shape: {mesh_shape}')
      print(f'hybrid, dcn_mesh_shape: {dcn_mesh_shape}')
    devices = mesh_utils.create_hybrid_device_mesh(
        mesh_shape, dcn_mesh_shape, allow_split_physical_axes=True
    )
  else:
    devices = mesh_utils.create_device_mesh(
        mesh_shape, allow_split_physical_axes=True
    )
  return js.Mesh(devices, axis_names=('replica', 'data', 'model'))


def get_default_mesh(print_debug_info=False):
  """Returns the default mesh for the current device set."""
  context = getattr(common.THREAD_CONTEXT, MESH_CONTEXT_KEY, None)
  if context:
    return context[-1]
  return create_mesh(print_debug_info=print_debug_info)


def mesh_sharding(
    pspec: common.PartitionAnnotation = None,
    mesh: js.Mesh | None = None,
) -> js.Sharding:
  if mesh is None:
    mesh = get_default_mesh()
  if pspec is None:
    return js.NamedSharding(mesh, js.PartitionSpec())
  else:
    return js.NamedSharding(mesh, js.PartitionSpec(*pspec))


def with_sharding_constraint(
    x: jax.Array, partition: PartitionAnnotation | js.Sharding
):
  """An extension of jax.lax.with_sharding_constraint.

  Besides js.Sharding, it also accepts PartitionAnnotation (e.g. [['replica',
  'data'], None]]) as partition input. Plus, it requires partition to have the
  same length as x.ndim if exists, in order to avoid incorrect implicit sharding
  extended annotation.

  Args:
    x: The array.
    partition: The partition annotation.

  Returns:
    The array with sharding constraint.
  """
  if partition is NOT_ANNOTATED:
    return x
  if isinstance(partition, js.Sharding):
    return jax.lax.with_sharding_constraint(x, partition)
  if partition is not None and len(partition) != x.ndim:
    raise ValueError(
        f'If exists, partition ({partition}) must have the same length as'
        f' x.ndim ({x.ndim}).'
    )
  return jax.lax.with_sharding_constraint(x, mesh_sharding(partition))


def reduce_across_hosts(
    in_tree: common.PyTree, reduce_op: Callable[..., jax.Array]
) -> common.PyTree:
  """Reduces data across all hosts."""
  if jax.process_count() == 1:
    return jax.tree.map(np.asarray, in_tree)

  devices: np.ndarray = np.array(jax.devices()).reshape(
      jax.process_count(), jax.local_device_count()
  )
  global_mesh = jax.sharding.Mesh(devices, ('processes', 'local_devices'))
  pspec = jax.sharding.PartitionSpec('processes')

  def pre_jit(x):
    inp = np.expand_dims(x, axis=0)
    return jax.experimental.multihost_utils.host_local_array_to_global_array(
        inp, global_mesh, pspec
    )

  def post_jit(x):
    return jax.device_get(x.addressable_data(0))

  in_tree = jax.tree.map(pre_jit, in_tree)
  out_tree = jax.jit(
      lambda x: jax.tree.map(functools.partial(reduce_op, axis=0), x),
      out_shardings=jax.sharding.NamedSharding(
          global_mesh, jax.sharding.PartitionSpec()
      ),
  )(in_tree)
  return jax.tree.map(post_jit, out_tree)


def sum_across_hosts(in_tree: common.PyTree) -> common.PyTree:
  """Sums data across all hosts."""
  return reduce_across_hosts(in_tree, jnp.sum)


def max_across_hosts(in_tree: common.PyTree) -> common.PyTree:
  """Sums data across all hosts."""
  return reduce_across_hosts(in_tree, jnp.max)


def _local_pytrees_to_global(
    abstract_pytree: common.PyTree,
    local_pytrees: Sequence[common.PyTree],
    num_per_process: np.ndarray,
    global_batch_size: int,
) -> common.PyTree:
  """See pytree_ragged_stack_allgather."""
  process_index = jax.process_index()
  assert len(local_pytrees) == num_per_process[process_index]

  start_indices = np.cumulative_sum(num_per_process, include_initial=True)
  start = min(global_batch_size, start_indices[process_index])
  end = min(global_batch_size, start_indices[process_index + 1])
  logging.info(
      '[pytree_ragged_stack_allgather] slice is (%s, %s] for process %s',
      start,
      end,
      process_index,
  )

  if end > start:
    batched_local_pytree = jax.tree.map(
        lambda *xs: np.stack(xs), *local_pytrees[: end - start]
    )

    def pad_to_global(x):
      pad_widths = [(start, global_batch_size - end)] + [(0, 0)] * (x.ndim - 1)
      return np.pad(x, pad_widths, constant_values=0)

    return jax.tree.map(pad_to_global, batched_local_pytree)
  else:
    return jax.tree.map(
        lambda x: np.zeros((global_batch_size,) + x.shape, dtype=x.dtype),
        abstract_pytree,
    )


def pytree_ragged_stack_allgather(
    abstract_pytree: common.PyTree,
    local_pytrees: Sequence[common.PyTree],
    num_per_process: np.ndarray,
    global_batch_size: int,
) -> common.PyTree:
  """Combines pytrees local to each process into a global one by stacking.

  Args:
    abstract_pytree: Pytree of ShapeDtypeStruct providing the common structure
      of all pytrees to be combined.
    local_pytrees: The pytrees available to the current local process.
    num_per_process: The number of pytrees for each process, needed to
      coordinate how to combine the local pytrees.
    global_batch_size: The final batch size of the resulting output. If the
      total number of pytrees exceeds this amount, later ones will be dropped.

  Returns:
    A stacked pytree with the same shapes as `abstract_pytree` except
    with a leading batch dimension.
  """
  global_pytree = _local_pytrees_to_global(
      abstract_pytree, local_pytrees, num_per_process, global_batch_size
  )
  time_start = time.time()
  global_pytree = sum_across_hosts(global_pytree)
  # Sum may turn some bool into int. Convert it back here.
  global_pytree = jax.tree.map(
      lambda x, y: x.astype(y.dtype), global_pytree, abstract_pytree
  )
  logging.info(
      '[pytree_ragged_stack_allgather] sum_across_hosts took %f seconds',
      time.time() - time_start,
  )
  return global_pytree


def multihost_sharded(
    batch: Sequence[Any], process_index: int = -1, process_count: int = 0
) -> Sequence[Any]:
  """Shards a batch across multiple hosts."""
  if process_index < 0:
    process_index = jax.process_index()
  if process_count <= 0:
    process_count = jax.process_count()
  batch_size = len(batch)
  base_size = batch_size // process_count
  remainder = batch_size % process_count
  start_index = process_index * base_size + min(process_index, remainder)
  end_index = start_index + base_size + (1 if process_index < remainder else 0)
  return batch[start_index:end_index]


def _inner_partition_with_minimum_redundancy(
    shape: tuple[int, ...],
    mesh_axis_sizes: tuple[int, ...],
    cache: MutableMapping[
        tuple[tuple[int, ...], tuple[int, ...]], Sequence[Sequence[int]]
    ],
) -> Sequence[Sequence[int]]:
  """Fits partition to a shape."""
  if (shape, mesh_axis_sizes) in cache:
    return cache[(shape, mesh_axis_sizes)]

  best_placement = [()] * len(shape)
  if not mesh_axis_sizes:
    return best_placement

  placement_value_fn = lambda placement: np.prod(
      [np.prod(p) for p in placement]
  )
  best_value = placement_value_fn(best_placement)

  for i, dim in enumerate(shape):
    for j, axis_size in enumerate(mesh_axis_sizes):
      if dim % axis_size == 0:
        next_mesh_axis_sizes = (*mesh_axis_sizes[:j], *mesh_axis_sizes[j + 1 :])
        next_shape = (*shape[:i], dim // axis_size, *shape[i + 1 :])
        sorted_next_shape, shape_indices = common.sorted_with_indices(
            next_shape
        )
        sorted_next_placement = _inner_partition_with_minimum_redundancy(
            tuple(sorted_next_shape), next_mesh_axis_sizes, cache
        )
        unsorted_placement = list(
            common.unsorted(sorted_next_placement, shape_indices)
        )
        unsorted_placement[i] = (axis_size, *unsorted_placement[i])

        if not next_mesh_axis_sizes:
          return unsorted_placement

        value = placement_value_fn(unsorted_placement)
        if value > best_value:
          best_placement = unsorted_placement
          best_value = value

  return best_placement


def batch_partition_with_minimum_redundancy(
    shapes: Sequence[Sequence[int]],
    mesh_axis_names: Sequence[str],
    mesh_axis_sizes: Sequence[int],
) -> Sequence[common.PartitionAnnotation]:
  """Finds partitions for a batch of shapes with minimum redundancy."""
  mesh_axis_name_index_map = {
      axis_name: index for index, axis_name in enumerate(mesh_axis_names)
  }
  cache = {}
  partition_annotations = []
  for shape in shapes:
    if not shape:
      partition_annotations.append(None)
      continue
    shape_index = [(shape, index) for index, shape in enumerate(shape)]
    sorted_shape, sorted_indices = zip(*sorted(shape_index, reverse=True))
    sorted_axis_sizes = sorted(mesh_axis_sizes, reverse=True)
    sorted_best_placement = _inner_partition_with_minimum_redundancy(
        tuple(sorted_shape), tuple(sorted_axis_sizes), cache=cache
    )
    unsorted_best_placement = [None] * len(shape)
    for index, p in zip(sorted_indices, sorted_best_placement, strict=True):
      unsorted_best_placement[index] = p

    axis_name_map = collections.defaultdict(list)
    for axis_name, axis_size in zip(
        mesh_axis_names, mesh_axis_sizes, strict=True
    ):
      axis_name_map[axis_size].append(axis_name)

    partition_annotation = []
    for axis_placement in unsorted_best_placement:
      assert axis_placement is not None
      axis_partition = []
      for axis_size in axis_placement:
        axis_partition.append(axis_name_map[axis_size][-1])
        axis_name_map[axis_size].pop(-1)
      if not axis_partition:
        axis_partition = None
      elif len(axis_partition) == 1:
        axis_partition = axis_partition[0]
      else:
        axis_partition = sorted(
            axis_partition, key=lambda x: mesh_axis_name_index_map[x]
        )
      partition_annotation.append(axis_partition)
    partition_annotations.append(partition_annotation)
  return partition_annotations


def partition_with_minimum_redundancy(
    shape: Sequence[int],
    mesh_axis_names: Sequence[str],
    mesh_axis_sizes: Sequence[int],
) -> common.PartitionAnnotation:
  return batch_partition_with_minimum_redundancy(
      [shape], mesh_axis_names, mesh_axis_sizes
  )[0]


@dataclasses.dataclass(frozen=True)
class MultihostData:
  """Multihost data.

  It provides save(), snapshot() and load() methods to save and load pytree data
  across multiple hosts effeciently. Note that these methods do not guarantee
  other hosts have completed the same methods at the end.
  """

  global_data: common.PyTree = None
  local_data: common.PyTree = None

  def save(self, save_dir: epath.PathLike):
    """Saves multi-host data."""
    process_index = jax.process_index()
    process_count = jax.process_count()
    save_dir = epath.Path(save_dir)
    if process_index == 0:
      if save_dir.exists():
        save_dir.rmtree(missing_ok=True)
      save_dir.mkdir(parents=True)
    multihost_utils.sync_global_devices('multihost_data.save.start')
    save_global_future = None
    if process_index == 0:
      save_global_future = asyncio.to_thread(
          pytree.save_pytree_to,
          dict(
              data=self.global_data,
              metadata=dict(process_count=process_count),
          ),
          save_dir / 'global.json',
      )
    pytree.save_pytree_to(
        self.local_data, save_dir / f'local_process_{process_index}.json'
    )
    if process_index == 0:
      assert save_global_future is not None
      asyncio.run(save_global_future)

  def snapshot(self, snapshot_dir: epath.PathLike):
    """Snapshots multi-host data."""
    process_index = jax.process_index()
    snapshot_dir = epath.Path(snapshot_dir).resolve()
    tmp_dir = epath.Path(os.fspath(snapshot_dir) + '.tmp')
    self.save(tmp_dir)
    multihost_utils.sync_global_devices('multihost_data.snapshot.saved')
    if process_index == 0:
      if snapshot_dir.exists():
        snapshot_dir.rmtree(missing_ok=True)
      tmp_dir.rename(snapshot_dir)

  @classmethod
  async def load_async(cls, load_dir: epath.PathLike) -> 'MultihostData':
    """Loads multi-host data from local_dir."""
    process_index = jax.process_index()
    process_count = jax.process_count()

    load_dir = epath.Path(load_dir)

    payload = pytree.load_pytree_from(load_dir / 'global.json')
    global_data = payload['data']
    metadata = payload['metadata']

    saved_process_count = metadata['process_count']
    process_indices_to_load = multihost_sharded(
        batch=list(range(saved_process_count)),
        process_index=process_index,
        process_count=process_count,
    )

    local_data_future_list = []
    for process_index_to_load in process_indices_to_load:
      local_data_future_list.append(
          asyncio.to_thread(
              pytree.load_pytree_from,
              load_dir / f'local_process_{process_index_to_load}.json',
          )
      )
    local_data_future_list = await asyncio.gather(*local_data_future_list)

    return cls(
        global_data=global_data,
        local_data=pytree.concatenate_pytrees(local_data_future_list),
    )

  @classmethod
  def load(cls, load_dir: epath.PathLike) -> 'MultihostData':
    """Loads multi-host data from local_dir."""
    return asyncio.run(cls.load_async(load_dir))

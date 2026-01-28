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
import dataclasses
import functools
import os
import time
from typing import Any, Mapping

from absl import logging
import deprecated
from etils import epath
import jax
import jax.core
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply.utils import common
from simply.utils import pytree


PartitionAnnotation = common.PartitionAnnotation
NOT_ANNOTATED = 'NOT_ANNOTATED'
# For backward compatibility.
DEFAULT_AXIS_NAMES = ('replica', 'data', 'model')


@deprecated.deprecated(reason='Use jax.sharding.set_mesh instead.')
def mesh_context(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int] | None = None,
    axis_names: Sequence[str] | None = None,
):
  return set_default_mesh_shape(
      mesh_shape=mesh_shape,
      dcn_mesh_shape=dcn_mesh_shape,
      axis_names=axis_names,
  )


@deprecated.deprecated(reason='Use jax.sharding.set_mesh instead.')
def set_default_mesh_shape(
    mesh_shape: Sequence[int] | Mapping[str, int],
    dcn_mesh_shape: Sequence[int] | Mapping[str, int] | None = None,
    axis_names: Sequence[str] | None = None,
) -> js.set_mesh:
  """Sets the default mesh shape for the current thread context.

  Args:
    mesh_shape: The shape of the mesh. Can be a sequence of integers or a
      mapping from axis names to sizes (missing keys are filled with 1).
    dcn_mesh_shape: The shape of the DCN mesh, if applicable. Can be a sequence
      of integers or a mapping from axis names to sizes (missing keys are filled
      with 1).
    axis_names: The names of the mesh axes. If None, `DEFAULT_AXIS_NAMES` is
      used.

  Returns:
    The set mesh.
  """
  return set_mesh(
      mesh_shape=mesh_shape,
      dcn_mesh_shape=dcn_mesh_shape,
      axis_names=axis_names,
  )


def set_mesh(
    mesh_shape: Sequence[int] | Mapping[str, int],
    dcn_mesh_shape: Sequence[int] | Mapping[str, int] | None = None,
    axis_names: Sequence[str] | None = None,
) -> js.set_mesh:
  return js.set_mesh(
      create_mesh(
          mesh_shape=mesh_shape,
          dcn_mesh_shape=dcn_mesh_shape,
          axis_names=axis_names,
      )
  )


def create_mesh(
    mesh_shape: Sequence[int] | Mapping[str, int] | None = None,
    dcn_mesh_shape: Sequence[int] | Mapping[str, int] | None = None,
    axis_names: Sequence[str] | None = None,
) -> js.Mesh:
  """Creates mesh for the current device set.

  Full replica parallelism is used if mesh_shape is not provided.

  Args:
    mesh_shape: The mesh shape.
    dcn_mesh_shape: The mesh shape for the dcn devices.
    axis_names: The names of the mesh axes.

  Returns:
    The mesh.
  """
  if axis_names is None:
    axis_names = DEFAULT_AXIS_NAMES

  if mesh_shape is None:
    # By default we just do full replica parallelism and assume the first axis
    # is the replica axis.
    mesh_shape = [len(jax.devices())] + [1] * (len(axis_names) - 1)
  if isinstance(mesh_shape, Mapping):
    mesh_shape = [mesh_shape.get(axis_name, 1) for axis_name in axis_names]
  if len(mesh_shape) == 2:
    mesh_shape = (1, *mesh_shape)
  if len(mesh_shape) != len(axis_names):
    raise ValueError(f'{mesh_shape=} does not match {axis_names=}')

  if isinstance(dcn_mesh_shape, Mapping):
    dcn_mesh_shape = [
        dcn_mesh_shape.get(axis_name, 1) for axis_name in axis_names
    ]

  if dcn_mesh_shape and sum(dcn_mesh_shape) > 1:
    logging.info(
        'hybrid, ici_mesh_shape=%s, dcn_mesh_shape=%s',
        mesh_shape,
        dcn_mesh_shape,
    )
    if len(dcn_mesh_shape) != len(axis_names):
      raise ValueError(f'{dcn_mesh_shape=} does not match {axis_names=}')
    devices = mesh_utils.create_hybrid_device_mesh(
        mesh_shape, dcn_mesh_shape, allow_split_physical_axes=True
    )
  else:
    logging.info('non-hybrid, ici_mesh_shape: %s', mesh_shape)
    devices = mesh_utils.create_device_mesh(
        mesh_shape, allow_split_physical_axes=True
    )
  return js.Mesh(devices, axis_names=axis_names)


@deprecated.deprecated(reason='Use jax.sharding.get_mesh instead.')
def get_default_mesh():
  """Returns the default mesh for the current device set."""
  mesh = js.get_abstract_mesh()
  if mesh.empty:
    logging.warning('No mesh is set. Creating a default mesh.')
    mesh = create_mesh()
  return mesh


@deprecated.deprecated(reason='Use named_sharding instead.')
def mesh_sharding(
    pspec: common.PartitionAnnotation = None,
    mesh: js.Mesh | None = None,
) -> js.Sharding:
  if mesh is None:
    return named_sharding(pspec)
  return js.NamedSharding(mesh, partition_spec(pspec))


def named_sharding(
    pspec: common.PartitionAnnotation = None,
) -> js.NamedSharding:
  mesh = js.get_abstract_mesh()
  if mesh.empty:
    logging.warning(
        'No mesh is set. Creating a default mesh for pspec=%s', pspec
    )
    mesh = create_mesh()
  if pspec is None:
    return js.NamedSharding(mesh, js.PartitionSpec())
  return js.NamedSharding(mesh, js.PartitionSpec(*pspec))


def get_array_sharding(array: jax.Array) -> js.Sharding:
  """Returns the sharding of the array."""
  if isinstance(array, jax.core.Tracer):
    if js.get_abstract_mesh().are_all_axes_explicit:
      return jax.typeof(array).sharding
    raise ValueError(
        f'Array {array} is a tracer but not all mesh axes are explicit.'
    )
  return array.sharding


def get_partition_axis(
    partition: PartitionAnnotation, axis: int
) -> str | Sequence[str] | None:
  if partition is None:
    return None
  return partition[axis]


def get_partition_size(partition: str | Sequence[str] | None) -> int:
  if partition is None:
    return 1
  if isinstance(partition, str):
    partition = (partition,)
  axis_sizes = [
      js.get_abstract_mesh().shape[axis] if axis is not None else 1
      for axis in partition
  ]
  return np.prod(axis_sizes, dtype=int)


def partition_spec(
    partition: PartitionAnnotation | js.PartitionSpec,
) -> js.PartitionSpec:
  if isinstance(partition, js.PartitionSpec):
    return partition
  if partition is None:
    return js.PartitionSpec()
  if isinstance(partition, str):
    return js.PartitionSpec(partition)
  return js.PartitionSpec(*partition)


def with_sharding_constraint(
    x: jax.Array,
    partition: js.Sharding | js.PartitionSpec | PartitionAnnotation,
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
  partition = partition_spec(partition)
  if len(partition) > 0 and len(partition) != len(x.shape):  # pylint: disable=g-explicit-length-test
    raise ValueError(
        f'If exists, {partition=} must have the same length as {x.ndim=}.'
    )
  if js.get_abstract_mesh().are_all_axes_explicit:
    return js.reshard(x, partition)
  if js.get_abstract_mesh().empty:
    logging.warning('No mesh is set. Creating a default mesh for array %s', x)
    return jax.lax.with_sharding_constraint(
        x, js.NamedSharding(create_mesh(), partition)
    )
  return jax.lax.with_sharding_constraint(x, partition)


def reduce_across_hosts(
    in_tree: common.PyTree, reduce_op: Callable[..., jax.Array]
) -> common.PyTree:
  """Reduces data across all hosts."""
  if jax.process_count() == 1:
    return jax.tree.map(np.asarray, in_tree)

  devices: np.ndarray = np.array(jax.devices()).reshape(
      jax.process_count(), jax.local_device_count()
  )
  global_mesh = js.Mesh(devices, ('processes', 'local_devices'))
  pspec = js.PartitionSpec('processes')

  def pre_jit(x):
    inp = np.expand_dims(x, axis=0)
    return multihost_utils.host_local_array_to_global_array(
        inp, global_mesh, pspec
    )

  def post_jit(x):
    return jax.device_get(x.addressable_data(0))

  in_tree = jax.tree.map(pre_jit, in_tree)
  with js.set_mesh(global_mesh):
    out_tree = jax.jit(
        lambda x: jax.tree.map(functools.partial(reduce_op, axis=0), x),
        out_shardings=js.PartitionSpec(),
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

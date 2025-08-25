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

from collections.abc import Callable, Sequence
import contextlib
import functools
import json
import os
from typing import Any

from absl import logging
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.sharding as js
import numpy as np


from simply.utils import common
from simply.utils import pytree


PartitionAnnotation = common.PartitionAnnotation
MESH_CONTEXT_KEY: str = 'mesh_context'


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


def save_multihost_data(data: Any, save_dir: str, process_index: int = -1):
  """Saves multi-host data."""
  if process_index < 0:
    process_index = jax.process_index()
  gfile.MakeDirs(save_dir)
  logging.info(
      'Saving multihost data to %s',
      os.path.join(save_dir, f'local_process_{process_index}.json'),
  )
  with open(
      os.path.join(save_dir, f'local_process_{process_index}.json'), 'w'
  ) as f:
    json.dump(pytree.dump_dataclasses(data), f)


def checkpoint_multihost_data(
    data: Any,
    save_dir: str,
    tag: str = 'checkpoint_multihost_data',
    process_index: int = -1,
):
  """Checkpoints multi-host data."""
  if process_index < 0:
    process_index = jax.process_index()
  save_dir = os.path.realpath(save_dir)
  tmp_dir = save_dir + '.tmp'
  save_multihost_data(data, tmp_dir, process_index)
  jax.experimental.multihost_utils.sync_global_devices(tag + '.save')
  if process_index == 0:
    gfile.Rename(tmp_dir, save_dir, overwrite=True)
  jax.experimental.multihost_utils.sync_global_devices(tag + '.finish')


def load_multihost_data(load_dir: str, process_index: int = -1) -> Any:
  """Loads multi-host data."""
  if process_index < 0:
    process_index = jax.process_index()
  path = os.path.join(load_dir, f'local_process_{process_index}.json')
  if not os.path.exists(path):
    return None
  with open(path, 'r') as f:
    data = json.load(f)
    return pytree.load_dataclasses(data)

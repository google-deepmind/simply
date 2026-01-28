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
"""Common utilities.

As a base utility library, it should not depend on any other utils libraries.
"""
import collections
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import functools
import hashlib
import json
import os
import re
import threading
import time
import types
from typing import Any, Callable, ClassVar, TypeAlias, Self, TypeVar, Union

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np

PartitionAnnotation: TypeAlias = None | Sequence[None | str | Sequence[str]]

BasicType: TypeAlias = Union[
    None,
    str,
    int,
    float,
    bool,
    jax.Array,
    'AnnotatedArray',
]

PyTree: TypeAlias = BasicType | Sequence['PyTree'] | Mapping[str, 'PyTree']
Array: TypeAlias = jax.Array | np.ndarray
RawT = TypeVar('RawT', str, np.ndarray[Any, np.dtype])
CacheValue = str | dict[str, Any]


THREAD_CONTEXT = threading.local()


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class AnnotatedArray:
  """A wrapper around Array to annotate its metadata."""
  array: Array
  metadata: types.MappingProxyType[str, Any]

  @functools.cached_property
  def dim_annotation(self) -> str | None:
    return self.metadata.get('dim_annotation', None)

  @functools.cached_property
  def shape(self):
    return self.array.shape

  @functools.cached_property
  def dtype(self):
    return self.array.dtype

  def tree_flatten(self):
    return (self.array,), self.metadata

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return AnnotatedArray(children[0], metadata=aux_data)

  @classmethod
  def create(cls, array: Array, **kwargs):
    # Make metadata immutable.
    return cls(array, metadata=types.MappingProxyType(kwargs))


def get_raw_arrays(tree: PyTree) -> PyTree:
  return jax.tree.map(
      lambda x: x.array if isinstance(x, AnnotatedArray) else x, tree,
      is_leaf=lambda x: isinstance(x, AnnotatedArray))


def transfer_metadata(base_tree: PyTree, target_tree: PyTree):
  """Transfer metadata from base to target."""
  def _transfer_metadata(base, target):
    if isinstance(base, AnnotatedArray):
      if isinstance(target, AnnotatedArray):
        array = target.array
      elif isinstance(target, Array):
        array = target
      elif target is None:
        if base is not None:
          raise ValueError(f'Target is None, but base is not None: {base=}')
        array = target
      else:
        raise ValueError(f'Unsupported target type: {type(target)}')
      return AnnotatedArray.create(array, **base.metadata)
    else:
      return target
  return jax.tree.map(_transfer_metadata, base_tree, target_tree,
                      is_leaf=lambda x: isinstance(x, AnnotatedArray))


class AttributeDict(dict):
  """A simplfied version of ConfigDict."""

  __slots__ = ()
  __setattr__ = dict.__setitem__

  def __getattr__(self, key: str) -> Any:
    if key in self:
      return self[key]
    raise AttributeError(f'{key} not found in {self}')


@dataclasses.dataclass(frozen=True)
class ParameterizedString:
  """Parameterized string.

  One use case is to restore a checkpoint with a set of parameters to restore
  into a single parameter, e.g. stacked_blocks, combined_qkv, etc.

  For example, if the template is '{a}/{b}/{c}', the parameters are
  {'a': ['1', '2'], 'b': ['x'], 'c': ['y', 'z']}, then it could iterate over
  '1/x/y', '1/x/z', '2/x/y', '2/x/z'. The iterated order is determined by the
  order of the parameters in the template and the order of the values in each
  each parameter.
  """

  PARAMETER_RE: ClassVar[re.Pattern[str]] = re.compile(r'{(\w+)}')

  template: str
  parameters: Mapping[str, Sequence[str]]

  def __post_init__(self):
    if set(self.parameters) != set(self.available_parameters):
      raise ValueError(
          'Parameters in the template must match the parameters in the'
          f' parameters. {self.parameters.keys()} vs'
          f' {self.available_parameters}.'
      )

  @classmethod
  def parameter_names(cls, template: str) -> Sequence[str]:
    return cls.PARAMETER_RE.findall(template)

  @functools.cached_property
  def available_parameters(self) -> Sequence[str]:
    return self.parameter_names(self.template)

  def format(self, **kwargs: str) -> str:
    return self.template.format(**kwargs)

  def __iter__(self, **fixed_kwargs: str) -> Iterator[Mapping[str, str]]:
    for pname in self.available_parameters:
      if pname not in fixed_kwargs:
        for value in self.parameters[pname]:
          fixed_kwargs[pname] = value
          yield from self.__iter__(**fixed_kwargs)
        fixed_kwargs.pop(pname)
        return
    yield fixed_kwargs.copy()


def quantize_array(w: Array, symmetric: bool = False):
  if symmetric:
    scale = jnp.max(jnp.abs(w)) / 127
    quant_w = jnp.asarray(jnp.round(w / scale), dtype=jnp.int8)
    result = {'quant_array': quant_w, 'scale': scale}
  else:
    scale = (jnp.max(w) - jnp.min(w)) / 256
    zero_point = (jnp.max(w) + jnp.min(w)) / 2
    quant_w = jnp.asarray(jnp.round((w - zero_point) / scale), dtype=jnp.int8)
    result = {'quant_array': quant_w, 'scale': scale, 'zero_point': zero_point}
  return result


def convert_or_dequantize(
    a: Array | Mapping[str, Array],
    dtype: jax.typing.DTypeLike = 'bfloat16',
):
  """Dequantizes an quantized structure if given, otherwise casts dtype."""
  if isinstance(a, Array):
    return jnp.asarray(a, dtype=dtype)
  quant_w = a['quant_array']
  dequant_w = jnp.asarray(quant_w, dtype=jnp.float32) * (
      a['scale'].astype(jnp.float32)
  )
  if 'zero_point' in a:
    dequant_w += a['zero_point'].astype(jnp.float32)
  return jnp.asarray(dequant_w, dtype=dtype)


def eval_abstract_output(fn: Callable[..., Any], *args, **kwargs) -> PyTree:
  """Returns jax.ShapeDtypeStruct tree for given function."""
  jitted_fn = jax.jit(fn)
  compiled_output = jitted_fn.lower(*args, **kwargs).compile()
  return compiled_output.out_info


def named_partial_fn(
    fn: Callable[..., Any], name: str, **kwargs: Any
) -> Callable[..., Any]:
  """Returns a partial function with the given name."""
  fn = functools.partial(fn, **kwargs)
  fn.__name__ = name
  return fn


def named_jit(
    fn: Callable[..., Any], name: str, **kwargs: Any
) -> Callable[..., Any]:
  """Returns a jitted function with the given name."""
  return jax.jit(named_partial_fn(fn, name, **kwargs))


def convert_rows_to_columns(
    rows: Sequence[Mapping[str, np.typing.ArrayLike]],
) -> Mapping[str, np.ndarray]:
  """Converts a sequence of rows to a column view."""
  column_view = collections.defaultdict(list)
  for row in rows:
    for k, v in row.items():
      column_view[k].append(v)
  return {k: np.array(v) for k, v in column_view.items()}


def convert_columns_to_rows(
    columns: Mapping[str, np.typing.ArrayLike],
) -> Sequence[Mapping[str, Any]]:
  """Converts a column view to a sequence of rows."""
  keys = list(columns.keys())
  if not keys:
    return []
  batch_size = len(columns[keys[0]])
  return [{k: columns[k][i] for k in keys} for i in range(batch_size)]


def find_unused_argpaths(
    func: Callable[[Any], Any], argtree: PyTree
) -> Sequence[jax.tree_util.KeyPath]:
  """Analyzes a JAX function to find args that are not used in the computation.

  Args:
    func: The JAX-compatible function to analyze.
    argtree: Example arguments to trace the function with.

  Returns:
    A Sequence of KeyPaths that indicate which arguments are unused in the
    argtree.
  """
  argpaths, _ = zip(*jax.tree.leaves_with_path(argtree))
  closed_jaxpr = jax.make_jaxpr(func)(argtree)

  invars = set(closed_jaxpr.jaxpr.invars)
  used_invars = set()
  for var in closed_jaxpr.jaxpr.outvars:
    if var in invars:
      try:
        used_invars.add(var)
      except TypeError:
        pass
  for eqn in closed_jaxpr.jaxpr.eqns:
    for var in eqn.invars:
      try:
        used_invars.add(var)
      except TypeError:
        pass

  unused_argpaths = []
  for argpath, var in zip(argpaths, closed_jaxpr.jaxpr.invars, strict=True):
    if var not in used_invars:
      unused_argpaths.append(argpath)

  return unused_argpaths


_TypeVarT: TypeAlias = TypeVar('_TypeVarT')


def sorted_with_indices(
    x: Sequence[_TypeVarT],
    key: Callable[[_TypeVarT], Any] | None = None,
    reverse: bool = False,
) -> tuple[Sequence[_TypeVarT], Sequence[int]]:
  """Returns a sorted sequence with indices."""
  if key is None:
    key_fn = lambda e: e[1]
  else:
    key_fn = lambda e: key(e[1])
  indices, sorted_x = zip(*sorted(enumerate(x), key=key_fn, reverse=reverse))
  return sorted_x, indices


def unsorted(
    sorted_x: Sequence[_TypeVarT], indices: Sequence[int]
) -> Sequence[_TypeVarT]:
  """Returns a unsorted sequence with the given indices."""
  unsorted_x = [None] * len(sorted_x)
  for i, v in zip(indices, sorted_x, strict=True):
    unsorted_x[i] = v
  return unsorted_x


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RaggedArray:
  """A ragged 2d array."""

  data: jax.Array  # [capacity, *subshape]
  lens: jax.Array  # i32[batch_size]

  def __post_init__(self):
    if len(self.lens.shape) != 1:
      raise ValueError(f'Lens must be 1d. {self.lens.shape=}')

  @functools.cached_property
  def is_valid(self) -> jax.Array:  # bool[]
    # User should guarantee this is always true.
    return self.total_length <= self.capacity

  @functools.cached_property
  def total_length(self) -> jax.Array:  # int32[]
    return jnp.sum(self.lens)

  @property
  def ndim(self) -> int:
    return len(self.data.shape) + 1

  @property
  def subshape(self) -> tuple[int, ...]:
    return self.data.shape[1:]

  @property
  def capacity(self) -> int:
    return self.data.shape[0]

  @property
  def dtype(self) -> jax.typing.DTypeLike:
    return self.data.dtype

  @property
  def batch_size(self) -> int:
    return self.lens.shape[0]

  @functools.cached_property
  def row_starts(self) -> jax.Array:  # int32[batch_size]
    return self.row_starts_with_end[:-1]

  @functools.cached_property
  def row_starts_with_end(self) -> jax.Array:  # int32[batch_size+1]
    return jnp.cumulative_sum(self.lens, include_initial=True)

  @functools.cached_property
  def row_ids(self) -> jax.Array:  # int32[capacity]
    # input: a1, a2, a3, b1, c1, c2
    # output: 0,  0,  0,  1,  2,  2, | 2, 2, 2, ...
    # numbers after | are padded with `batch_size - 1``
    return jnp.repeat(
        jnp.arange(self.batch_size),
        self.lens,
        total_repeat_length=self.capacity,
    )

  def row(self, idx: jax.typing.ArrayLike) -> jax.Array:
    """Returns the row at the given index."""
    if jnp.ndim(idx) != 0:
      raise ValueError(f'Row index must be 0d: {jnp.shape(idx)=}')
    return self.data[
        self.row_starts_with_end[idx] : self.row_starts_with_end[idx + 1]
    ]

  @functools.cached_property
  def intra_offset(self) -> jax.Array:  # int32[batch_size]
    # input: a1, a2, a3, b1, c1, c2
    # output: 0,  1,  2,  0,  0,  1, | 2, 3, 4, ...
    # numbers after | are padded as if the last row is infinite.
    return jnp.arange(self.capacity) - self.row_starts[self.row_ids]

  def to_numpy_list(self) -> Sequence[np.ndarray]:
    np_data = np.asarray(self.data)
    lens = np.asarray(self.lens)
    if lens.size == 0:
      return []
    indices = np.cumsum(lens)
    return np.split(np_data, indices, axis=0)[:-1]

  def to_padded_dense(
      self, max_len: int, padding_value: jax.typing.ArrayLike = 0
  ) -> jax.Array:
    """Converts to a padded dense array."""
    if jnp.ndim(padding_value) != 0:
      raise ValueError(f'Padding must be 0d. {jnp.shape(padding_value)=}')

    # 1. Create a 2D grid of column indices: shape (1, max_len)
    # [0, 1, 2, ..., max_len-1]
    col_idx = jnp.arange(max_len)[None, :]

    # 2. Create a mask determining which positions are valid
    # Broadcast compares (1, max_len) vs (batch, 1) -> (batch, max_len)
    mask = col_idx < self.lens[:, None]

    # 4. Calculate the flat index for every cell in the 2D output
    # cell_index = start_of_row + column_index
    # Shape: (batch, 1) + (1, max_len) -> (batch, max_len)
    flat_indices = self.row_starts[:, None] + col_idx

    # 5. Handle Out-of-Bounds safe reading
    # We replace invalid indices with 0 temporarily so the gather doesn't crash.
    # We will overwrite these values with fill_value in the next step anyway.
    safe_indices = jnp.where(mask, flat_indices, 0)

    # 6. Gather data and apply padding
    dense = self.data[safe_indices]
    dense = jnp.where(mask, dense, padding_value)
    return dense

  @classmethod
  def from_numpy_list(cls, np_list: Sequence[np.typing.ArrayLike]) -> Self:
    data = jnp.concatenate([jnp.asarray(x) for x in np_list], axis=0)
    lens = jnp.array([np.shape(x)[0] for x in np_list])
    return RaggedArray(data=data, lens=lens)

  def set_padding_value(self, padding_value: jax.typing.ArrayLike) -> Self:
    if jnp.ndim(padding_value) != 0:
      raise ValueError(f'Padding must be 0d. {jnp.shape(padding_value)=}')
    mask = jnp.arange(self.capacity) < self.total_length
    mask = jnp.expand_dims(mask, np.arange(1, len(self.data.shape)))
    data = jnp.where(mask, self.data, padding_value)
    return RaggedArray(data=data, lens=self.lens)

  def extend_capacity_to(self, capacity: int) -> Self:
    if capacity < self.capacity:
      raise ValueError(
          f'Capacity must be >= {self.capacity}. {capacity=} <'
          f' {self.capacity=}.'
      )
    pad_widths = [(0, 0)] * len(self.subshape)
    data = jnp.pad(self.data, [(0, capacity - self.capacity), *pad_widths])
    return RaggedArray(data=data, lens=self.lens)

  def concat(self, other: Self, capacity: int | None = None) -> Self:
    """Concatenates with another ragged array."""
    if self.batch_size != other.batch_size:
      raise ValueError(
          'All ragged arrays must have the same batch size. Got'
          f' {self.batch_size=} and {other.batch_size=}.'
      )
    if self.dtype != other.dtype:
      raise ValueError(
          'All ragged arrays must have the same dtype. Got'
          f' {self.dtype=} and {other.dtype=}.'
      )
    if self.subshape != other.subshape:
      raise ValueError(
          'All ragged arrays must have the same subshape. Got'
          f' {self.subshape=} and {other.subshape=}.'
      )

    if capacity is None:
      capacity = self.capacity + other.capacity

    z_lens = self.lens + other.lens
    z_starts = jnp.cumulative_sum(z_lens, include_initial=True)

    ragged_z = jax.lax.empty((capacity, *self.subshape), dtype=self.dtype)
    self_target_idx = z_starts[self.row_ids] + self.intra_offset
    other_target_idx = (
        z_starts[other.row_ids] + self.lens[other.row_ids] + other.intra_offset
    )
    ragged_z = ragged_z.at[self_target_idx].set(self.data, mode='drop')
    ragged_z = ragged_z.at[other_target_idx].set(other.data, mode='drop')
    return RaggedArray(data=ragged_z, lens=z_lens)

  def keep_rows(self, row_mask: jax.typing.ArrayLike) -> Self:
    """Keeps the rows that satisfy the row mask."""
    row_mask = jnp.asarray(row_mask)
    if len(row_mask.shape) != 1 or row_mask.dtype != jnp.bool:
      raise ValueError(
          f'Keep mask must be 1d bool: {row_mask.shape=}, {row_mask.dtype=}'
      )
    if row_mask.shape[0] != self.batch_size:
      raise ValueError(f'{jnp.shape(row_mask)=} must match {self.batch_size=}')
    element_keep_mask = row_mask[self.row_ids] & (
        jnp.arange(self.capacity) < self.total_length
    )
    indices = jnp.flatnonzero(
        element_keep_mask, size=self.capacity, fill_value=0
    )
    new_data = self.data[indices]
    new_lens = jnp.where(row_mask, self.lens, 0)
    return RaggedArray(data=new_data, lens=new_lens)

  def keep_last_ncols(self, ncols: int) -> Self:
    """Keeps the last n columns of each row."""
    is_last_n = self.intra_offset >= self.lens[self.row_ids] - ncols
    indices = jnp.flatnonzero(is_last_n, size=self.capacity, fill_value=0)
    return RaggedArray(
        data=self.data[indices], lens=jnp.minimum(self.lens, ncols)
    )


def convert_array_with_abstract(
    x: jax.Array, abstract: jax.ShapeDtypeStruct
) -> jax.Array:
  """Converts an array to the given abstract specified dtype/sharding."""
  if not isinstance(x.sharding, js.NamedSharding):
    raise ValueError(f'Unsupported sharding type: {x.sharding=}')
  if not isinstance(abstract.sharding, js.NamedSharding):
    raise ValueError(f'Unsupported sharding type: {abstract.sharding=}')
  if x.shape != abstract.shape:
    raise ValueError(f'Shape mismatch: {x.shape=} vs {abstract.shape=}')

  if x.sharding.mesh == abstract.sharding.mesh:
    return jax.lax.with_sharding_constraint(
        jnp.astype(x, abstract.dtype), abstract.sharding
    )

  replicated_sharding = x.sharding.update(spec=js.PartitionSpec())
  with js.set_mesh(replicated_sharding.mesh):
    if abstract.dtype.itemsize < x.dtype.itemsize:
      x = jnp.astype(x, abstract.dtype)
    x_replicated = jax.lax.with_sharding_constraint(x, replicated_sharding)
    x_np = np.asarray(x_replicated)

  y = jax.lax.with_sharding_constraint(
      jnp.asarray(x_np, abstract.dtype), abstract.sharding
  )
  return y


def neg_inf(dtype: jax.typing.DTypeLike) -> float:
  if jnp.issubdtype(dtype, jnp.inexact):
    dtype_max = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtype_max = jnp.iinfo(dtype).max
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')
  # NOTE: Gemma uses -0.7 * dtype_max
  return -0.5 * dtype_max


def reduce_same(seq: Sequence[Any]) -> Any:
  """Reduces a list of same values to a single value."""
  first = seq[0]
  for x in seq[1:]:
    if x != first:
      raise ValueError(f'Sequence must be same. {x=} != {first=}')
  return first


def pad_to_len(
    arr: np.ndarray, seq_len: int, pad_value: Any, dtype: Any
) -> np.ndarray:
  """Pads array to target length. Does not truncate."""
  if len(arr) >= seq_len:
    return arr.astype(dtype)
  pad_width = seq_len - len(arr)
  return np.pad(arr, (0, pad_width), constant_values=pad_value).astype(dtype)

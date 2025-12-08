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
from typing import Any, Callable, ClassVar, TypeAlias, TypeVar, Union

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
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

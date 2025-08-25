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
"""Utils that process pytree."""

from collections.abc import MutableMapping, MutableSequence, Sequence, Set
import dataclasses
import re
from typing import Any, Callable, cast

import numpy as np

from simply.utils import registry
from simply.utils.common import PyTree  # pylint: disable=g-importing-member


def tree_value(tree: PyTree, path: str) -> PyTree:
  """Gets tree value by path."""
  if not path:
    return tree
  if path.startswith('/'):
    return tree_value(tree, path[1:])
  if path.startswith('['):
    right_index = path.find(']')
    assert right_index != -1
    index = int(path[1:right_index])
    return tree_value(tree[index], path[right_index + 1 :])

  delimiter = re.search(r'/|\[', path)
  if not delimiter:
    return tree[path]
  return tree_value(tree[path[: delimiter.start()]], path[delimiter.start() :])


def set_tree_value(
    tree: PyTree,
    path: str,
    value: PyTree,
) -> None:
  """Sets value at path in tree."""
  if not path:
    raise ValueError('path must be non-empty.')
  delimiter = re.search(r'/|\[(\d+)\]', path)
  if not delimiter:
    cast(MutableMapping[str, PyTree], tree)[path] = value
    return
  name = path[: delimiter.start()]
  path_remain = path[delimiter.end() :]

  if delimiter.group(0) == '/':
    tree = cast(MutableMapping[str, PyTree], tree)
    if not name:
      set_tree_value(tree, path_remain, value)
      return
    if name not in tree:
      tree[name] = {}
    set_tree_value(tree[name], path_remain, value)
    return

  index = int(delimiter.group(1))
  if name:
    tree = cast(MutableMapping[str, PyTree], tree)
    if name not in tree:
      tree[name] = []
    set_tree_value(tree[name], path[delimiter.start() :], value)
    return

  tree = cast(MutableSequence[PyTree], tree)
  if len(tree) <= index:
    tree.extend([None] * (index + 1 - len(tree)))
  if path_remain:
    if tree[index] is None:
      if path_remain.startswith('/'):
        tree[index] = {}
      elif path_remain.startswith('['):
        tree[index] = []
    set_tree_value(tree[index], path_remain, value)
  else:
    tree[index] = value


def tree_is_mapping(tree: PyTree):
  return all(hasattr(tree, attr) for attr in ('keys', '__getitem__'))


def tree_is_sequence(tree: PyTree):
  return isinstance(tree, Sequence) and not isinstance(tree, str)


def check_trees_match_mapping_keys(trees: Sequence[PyTree], keys: Set[str]):
  """Raises error if trees are not consistent mapping with keys."""
  if not trees:
    return
  for tree in trees:
    if not tree_is_mapping(tree):
      raise ValueError(f'Expect tree to be Mapping: {tree}')
    key_set = set(getattr(tree, 'keys')())
    if keys - key_set or key_set - keys:
      raise ValueError(
          f'Expect tree keys to be {keys}, but got {key_set} instead.'
      )


def check_trees_match_sequence_length(trees: Sequence[PyTree], length: int):
  """Raises error if trees are not consistent sequence with length."""
  if not trees:
    return
  for tree in trees:
    if not tree_is_sequence(tree):
      raise ValueError(f'Expect tree to be Sequence: {tree}')
    if len(tree) != length:
      raise ValueError(
          f'Expect tree length to be {length}, but got {len(tree)} instead.'
      )


def traverse_tree_with_path(
    fn: Callable[..., PyTree], *trees: PyTree, root_path: str = ''
) -> PyTree:
  """Traverses tree with path, with fn applied to each leaf node.

  When multiple trees are provided, all trees should be structurally consistent
  at mapping and sequence level, so that leaf nodes at the same path in all
  trees can be fed into fn(). Otherwise, ValueError will be raised.

  Args:
    fn: The function to call on each leaf node, it should be the signature of
      `fn(*leaves: Leaf, path: str) -> PyTree`.
    *trees: The trees to traverse.
    root_path: The root path of the tree.

  Returns:
    A pytree with each leaf node value replaced by of fn().
  """
  if not trees:
    return None
  first_tree = trees[0]
  if tree_is_mapping(first_tree):
    keys = getattr(first_tree, 'keys')()
    check_trees_match_mapping_keys(trees[1:], set(keys))
    res = {}
    for key in keys:
      path = f'{root_path}/{key}' if root_path else key
      values = [subtree[key] for subtree in trees]
      res[key] = traverse_tree_with_path(fn, *values, root_path=path)
    return res
  if tree_is_sequence(first_tree):
    length = len(first_tree)
    check_trees_match_sequence_length(trees[1:], length)
    res = []
    for i in range(length):
      values = [subtree[i] for subtree in trees]
      res.append(
          traverse_tree_with_path(fn, *values, root_path=f'{root_path}[{i}]')
      )
    return res
  return fn(*trees, root_path)


def load_dataclasses(tree: PyTree) -> Any:
  """Loads dataclasses in a python tree.

  Args:
    tree: The python tree to load dataclasses.

  Returns:
    A tree like object that contains dataclasses if any exists.
  """
  if tree_is_mapping(tree):
    if '__dataclass__' in tree:
      module_cls = registry.RootRegistry.get(tree['__dataclass__'])
      assert dataclasses.is_dataclass(module_cls)
      input_params = {}
      for k in dataclasses.fields(module_cls):
        if k.name not in tree:
          if k.default is not dataclasses.MISSING:
            input_params[k.name] = k.default
          elif k.default_factory is not dataclasses.MISSING:
            input_params[k.name] = k.default_factory()
          else:
            raise ValueError(
                f'Field {k.name} without default value not found in {tree}.'
            )
        else:
          input_params[k.name] = load_dataclasses(tree[k.name])
      return module_cls(**input_params)
    if '__numpy_ndarray_dtype__' in tree:
      dtype = tree['__numpy_ndarray_dtype__']
      return np.asarray(tree['data'], dtype=dtype)
    return {k: load_dataclasses(v) for k, v in getattr(tree, 'items')()}
  if tree_is_sequence(tree):
    return [load_dataclasses(v) for v in tree]
  return tree


def dump_dataclasses(
    tree: Any,
    only_dump_basic: bool = True,
) -> PyTree:
  """Dumps dataclasses to pytree in a tree-like object.

  Args:
    tree: Tree-like object that may contain dataclasses.
    only_dump_basic: If true, only dump basic attributes in dataclasses.

  Returns:
    A pytree that has all dataclasses dumped.
  """
  if dataclasses.is_dataclass(tree):
    res = {}
    registered_name = getattr(tree, '__registered_name__')
    if registered_name:
      res['__dataclass__'] = registered_name

    if only_dump_basic:
      keys = [k.name for k in dataclasses.fields(tree)]
    else:
      keys = tree.__dict__.keys()
    for k in keys:
      v = tree.__dict__[k]
      res[k] = dump_dataclasses(v, only_dump_basic=only_dump_basic)
    return res
  if isinstance(tree, np.ndarray):
    return dict(data=tree.tolist(), __numpy_ndarray_dtype__=str(tree.dtype))
  if tree_is_sequence(tree):
    return [dump_dataclasses(v, only_dump_basic=only_dump_basic) for v in tree]
  if tree_is_mapping(tree):
    # A more generic way to detect mapping structure.
    res = {}
    for k, v in tree.items():
      if any(reserved in k for reserved in '[]/'):
        raise ValueError(f'Key {k} contains reserved character ([]/).')
      res[k] = dump_dataclasses(v, only_dump_basic=only_dump_basic)
    return res
  return tree

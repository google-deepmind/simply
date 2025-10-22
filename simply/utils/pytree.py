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

from collections.abc import Sequence, Set
import dataclasses
import json
import re
from typing import Any, Callable, cast
import warnings

from etils import epath
import jax
import numpy as np
from simply.utils import registry
from simply.utils.common import PyTree  # pylint: disable=g-importing-member


def convert_string_path_to_key_path(path: str) -> jax.tree_util.KeyPath:
  """Converts a string path to a KeyPath."""
  keypath = []
  while path:
    if path.startswith('/'):
      path = path[1:]
    if path.startswith('['):
      right_index = path.find(']')
      assert right_index != -1
      index = int(path[1:right_index])
      keypath.append(jax.tree_util.SequenceKey(index))
      path = path[right_index + 1 :]
    else:
      delimiter = re.search(r'/|\[', path)
      if delimiter is None:
        keypath.append(jax.tree_util.DictKey(path))
        path = ''
      else:
        keypath.append(jax.tree_util.DictKey(path[: delimiter.start()]))
        path = path[delimiter.start() :]
  return jax.tree_util.KeyPath(keypath)


def tree_value(tree: PyTree, path: jax.tree_util.KeyPath | str) -> PyTree:
  """Gets tree value by path."""
  if isinstance(path, str):
    path = convert_string_path_to_key_path(path)

  value = tree
  for item in path:
    if isinstance(item, jax.tree_util.DictKey):
      if item.key not in value:
        raise KeyError(f'{path} does not exist in tree at {item}.')
      value = value[item.key]
    elif isinstance(item, jax.tree_util.SequenceKey):
      if item.idx >= len(value):
        raise KeyError(f'{path} does not exist in tree at {item}.')
      value = value[item.idx]
    else:
      raise KeyError(f'Unsupported key type: {type(item)}:{item}')
  return value


def construct_tree_with_path_value(
    path: jax.tree_util.KeyPath, value: PyTree
) -> PyTree:
  """Constructs a tree with path and value."""
  tree = value
  for item in reversed(path):
    if isinstance(item, jax.tree_util.DictKey):
      tree = {item.key: tree}
    elif isinstance(item, jax.tree_util.SequenceKey):
      next_tree = [None] * (item.idx + 1)
      next_tree[item.idx] = tree
      tree = next_tree
    else:
      raise ValueError(f'Unsupported key type: {type(item)}:{item}')
  return tree


def set_tree_value(
    tree: PyTree,
    path: jax.tree_util.KeyPath | str,
    value: PyTree,
) -> None:
  """Sets value at path in tree."""
  if isinstance(path, str):
    path = convert_string_path_to_key_path(path)

  if not path:
    raise ValueError('path must be non-empty.')

  for i, item in enumerate(cast(jax.tree_util.KeyPath, path)):
    if tree_is_mapping(tree):
      if not isinstance(item, jax.tree_util.DictKey):
        raise ValueError(f'Path item {item} does not match subtree: {tree}')
      if i + 1 == len(path):
        tree[item.key] = value
      elif tree.get(item.key) is None:
        tree[item.key] = construct_tree_with_path_value(path[i + 1 :], value)
        break
      else:
        tree = tree[item.key]
    elif tree_is_sequence(tree):
      if not isinstance(item, jax.tree_util.SequenceKey):
        raise ValueError(f'Path item {item} does not match subtree: {tree}')
      if item.idx >= len(tree):
        tree.extend([None] * (item.idx + 1 - len(tree)))
      if i + 1 == len(path):
        tree[item.idx] = value
      elif tree[item.idx] is None:
        tree[item.idx] = construct_tree_with_path_value(path[i + 1 :], value)
        break
      else:
        tree = tree[item.idx]
    else:
      raise ValueError(f'Cannot access path {path[i:]} in tree {tree}')


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


def tree_leaves_with_tag(tree, tag='loss'):
  """Yields leaves and their paths from a pytree if the path contains a tag.

  Args:
    tree: The pytree to traverse.
    tag: The string tag to search for in the raw path components.

  Yields:
    A tuple of (leaf, path) for each leaf where the tag is present in the path.
  """
  def _get_raw(path):
    raw_path = []
    for p in path:
      if hasattr(p, 'key'):
        raw_path.append(p.key)
      elif hasattr(p, 'idx'):
        raw_path.append(p.idx)
      else:
        raise ValueError(f'Unknown path type {type(p)}')
    return raw_path
  for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
    if tag in _get_raw(path):
      yield leaf, path


def load(jtree: PyTree) -> Any:
  """Loads data objects (dataclasses and numpy arrays) in a json-like tree.

  Args:
    jtree: The json-like tree to load data.

  Returns:
    A tree like object that contains data objects if any exists.
  """
  if tree_is_mapping(jtree):
    if '__dataclass__' in jtree:
      module_cls = registry.RootRegistry.get(jtree['__dataclass__'])
      assert dataclasses.is_dataclass(module_cls)
      input_params = {}
      for k in dataclasses.fields(module_cls):
        if k.name not in jtree:
          if k.default is not dataclasses.MISSING:
            input_params[k.name] = k.default
          elif k.default_factory is not dataclasses.MISSING:
            input_params[k.name] = k.default_factory()
          else:
            raise ValueError(
                f'Field {k.name} without default value not found in {jtree}.'
            )
        else:
          input_params[k.name] = load(jtree[k.name])
      return module_cls(**input_params)
    if '__numpy_ndarray_dtype__' in jtree:
      dtype = jtree['__numpy_ndarray_dtype__']
      return np.asarray(jtree['data'], dtype=dtype)
    return {k: load(v) for k, v in getattr(jtree, 'items')()}
  if tree_is_sequence(jtree):
    return [load(v) for v in jtree]
  return jtree


def dump(ptree: Any, only_dump_basic: bool = True) -> PyTree:
  """Dumps data in pytree into a json-like tree.

  Args:
    ptree: Python tree. May contains dataclasses and numpy arrays.
    only_dump_basic: If true, only dump basic attributes in dataclasses.

  Returns:
    A json-like tree.
  """
  if dataclasses.is_dataclass(ptree):
    res = {}
    registered_name = getattr(ptree, '__registered_name__')
    if registered_name:
      res['__dataclass__'] = registered_name

    if only_dump_basic:
      keys = [k.name for k in dataclasses.fields(ptree)]
    else:
      keys = ptree.__dict__.keys()
    for k in keys:
      v = ptree.__dict__[k]
      res[k] = dump(v, only_dump_basic=only_dump_basic)
    return res
  if isinstance(ptree, np.ndarray):
    return dict(data=ptree.tolist(), __numpy_ndarray_dtype__=str(ptree.dtype))
  if tree_is_sequence(ptree):
    return [dump(v, only_dump_basic=only_dump_basic) for v in ptree]
  if tree_is_mapping(ptree):
    # A more generic way to detect mapping structure.
    res = {}
    for k, v in ptree.items():
      if any(reserved in k for reserved in '[]/'):
        raise ValueError(f'Key {k} contains reserved character ([]/).')
      res[k] = dump(v, only_dump_basic=only_dump_basic)
    return res
  return ptree


def load_dataclasses(jtree: PyTree) -> Any:
  warnings.warn('load_dataclasses() is deprecated. Use load() instead.')
  return load(jtree)


def dump_dataclasses(ptree: Any, only_dump_basic: bool = True) -> PyTree:
  warnings.warn('dump_dataclasses() is deprecated. Use dump() instead.')
  return dump(ptree, only_dump_basic=only_dump_basic)


def concatenate_pytrees(trees: Sequence[PyTree]) -> PyTree:
  """Concatenates multiple pytrees' underlining sequences into one.

  It requires all trees to be structurally consistent at mapping level, so that
  the concatenation can be performed on the underlining sequences. Otherwise,
  ValueError will be raised.

  Args:
    trees: The pytrees to concatenate.

  Returns:
    The concatenated pytree.
  """
  if not trees:
    return None

  first_tree = trees[0]
  if len(trees) == 1:
    return first_tree

  if tree_is_mapping(first_tree):
    concatenated = {}
    keys = set(first_tree.keys())
    for tree in trees:
      if not tree_is_mapping(tree):
        raise ValueError(f'Expect tree to be Mapping: {tree}')
      if set(tree.keys()) != keys:
        raise ValueError(
            f'Expect tree keys to be {keys}, but got {tree.keys()} instead.'
        )
    for key in keys:
      subtrees = [tree[key] for tree in trees]
      concatenated[key] = concatenate_pytrees(subtrees)
    return concatenated

  if tree_is_sequence(first_tree):
    concatenated = []
    for tree in trees:
      if not tree_is_sequence(tree):
        raise ValueError(f'Expect tree to be Sequence: {tree}')
      concatenated.extend(tree)
    return type(first_tree)(concatenated)

  if dataclasses.is_dataclass(first_tree):
    tree_cls = first_tree.__class__
    for tree in trees[1:]:
      if not isinstance(tree, tree_cls):
        raise ValueError(
            f'Expect tree type to be {tree_cls}, but got {type(tree)} instead.'
        )
    input_params = {}
    for field in dataclasses.fields(tree_cls):
      subtrees = [tree.__dict__[field.name] for tree in trees]
      input_params[field.name] = concatenate_pytrees(subtrees)
    return tree_cls(**input_params)

  for tree in trees[1:]:
    if tree != first_tree:
      raise ValueError(
          f'Expect tree to be {first_tree}, but got {tree} instead.'
      )
  return first_tree


def save_pytree_to(tree: Any, path: epath.PathLike):
  """Saves a pytree to a file."""
  path = epath.Path(path)
  with path.open('w') as f:
    json.dump(dump(tree), f)


def load_pytree_from(path: epath.PathLike) -> PyTree:
  """Loads a pytree from a file."""
  path = epath.Path(path)
  with path.open('r') as f:
    return load(json.load(f))

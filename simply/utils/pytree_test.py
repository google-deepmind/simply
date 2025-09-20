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
from collections.abc import Sequence
import dataclasses

from absl.testing import absltest
import numpy as np
from simply.utils import pytree
from simply.utils import registry


@registry.RootRegistry.register
@dataclasses.dataclass
class _A:
  x: int


@registry.RootRegistry.register
@dataclasses.dataclass
class _B(_A):
  y: int = 2


@registry.RootRegistry.register
@dataclasses.dataclass
class _C:
  a: _A
  b: _B = dataclasses.field(default_factory=_B)
  d: Sequence[_A] = dataclasses.field(default_factory=list)


class PyTreeTest(absltest.TestCase):

  def test_tree(self):
    tree = {
        'a': 1,
        'b': [1, 2, {'d': [3, 4]}],
        'c': {
            'd': 1,
            'e': 2,
        },
    }
    self.assertEqual(pytree.tree_value(tree, 'a'), 1)
    self.assertEqual(pytree.tree_value(tree, 'b[1]'), 2)
    self.assertEqual(pytree.tree_value(tree, 'b[2]/d[0]'), 3)
    self.assertEqual(pytree.tree_value(tree, 'c/e'), 2)

  def test_set_tree(self):
    tree = {}
    pytree.set_tree_value(tree, 'a', 10)
    pytree.set_tree_value(tree, 'b[1]', 20)
    pytree.set_tree_value(tree, 'c[1][2]', 4)
    pytree.set_tree_value(tree, 'b[2]/a', 4)
    pytree.set_tree_value(tree, 'b[2]/d[1][2]', 4)
    pytree.set_tree_value(tree, 'b[2]/c[1][0]/e', 4)
    self.assertEqual(
        tree,
        {
            'a': 10,
            'b': [
                None,
                20,
                {'a': 4, 'd': [None, [None, None, 4]], 'c': [None, [{'e': 4}]]},
            ],
            'c': [None, [None, None, 4]],
        },
    )

  def test_tree_type(self):
    self.assertTrue(pytree.tree_is_mapping({'a': 1}))
    self.assertFalse(pytree.tree_is_sequence({'a': 1}))
    self.assertFalse(pytree.tree_is_mapping('abc'))
    self.assertFalse(pytree.tree_is_sequence('abc'))
    self.assertTrue(pytree.tree_is_sequence([1, 2, 3]))
    self.assertTrue(pytree.tree_is_sequence((1, 2, 3)))
    self.assertFalse(pytree.tree_is_mapping(1.0))
    self.assertFalse(pytree.tree_is_sequence(1.0))

  def test_tree_mapping_schema_check(self):
    trees = [{'a': 1, 'b': 2, 'c': 3}, {'a': [1], 'b': [2], 'c': [3]}]
    pytree.check_trees_match_mapping_keys(trees, set(['a', 'b', 'c']))
    with self.assertRaises(ValueError):
      pytree.check_trees_match_mapping_keys(trees, set(['a', 'b', 'c', 'd']))
    with self.assertRaises(ValueError):
      pytree.check_trees_match_mapping_keys(trees, set(['a', 'b']))
    with self.assertRaises(ValueError):
      pytree.check_trees_match_mapping_keys(trees, set(['a', 'e', 'c']))

  def test_tree_sequence_schema_check(self):
    trees = [['a', 'b', 'c'], [1, 2, 3]]
    pytree.check_trees_match_sequence_length(trees, 3)
    with self.assertRaises(ValueError):
      pytree.check_trees_match_sequence_length(trees, 2)
    with self.assertRaises(ValueError):
      pytree.check_trees_match_sequence_length(trees, 4)

  def test_traverse_tree(self):
    tree1 = {
        'a': 1,
        'b': [1, 2, {'d': [3, 4]}],
        'c': {
            'd': 1,
            'e': 2,
        },
    }
    tree2 = {
        'a': 2,
        'b': [3, 4, {'d': [5, 6]}],
        'c': {
            'd': 7,
            'e': 8,
        },
    }
    flatten = []

    def _output(v1, v2, p):
      v = v1 + v2
      flatten.append((p, v))
      return f'{p}={v}'

    res = pytree.traverse_tree_with_path(_output, tree1, tree2)
    self.assertEqual(
        flatten,
        [
            ('a', 3),
            ('b[0]', 4),
            ('b[1]', 6),
            ('b[2]/d[0]', 8),
            ('b[2]/d[1]', 10),
            ('c/d', 8),
            ('c/e', 10),
        ],
    )
    self.assertEqual(
        res,
        {
            'a': 'a=3',
            'b': ['b[0]=4', 'b[1]=6', {'d': ['b[2]/d[0]=8', 'b[2]/d[1]=10']}],
            'c': {
                'd': 'c/d=8',
                'e': 'c/e=10',
            },
        },
    )

  def test_load(self):
    tree1 = {
        'a': {'__dataclass__': '_A', 'x': 1},
        'b': {'__dataclass__': '_B', 'x': 1, 'y': 2},
        '_z': 5,
        '__dataclass__': '_C',
    }
    res1 = pytree.load(tree1)
    self.assertEqual(res1, _C(a=_A(x=1), b=_B(x=1, y=2)))

    tree2 = {
        '__dataclass__': '_C',
        'a': {'__dataclass__': '_B', 'y': 1},
        'b': {'__dataclass__': '_B', 'x': 1, 'y': 2},
        '_z': 5,
    }
    with self.assertRaises(ValueError):
      pytree.load(tree2)
    tree2['a']['x'] = 4
    res2 = pytree.load(tree2)
    self.assertEqual(res2, _C(a=_B(x=4, y=1), b=_B(x=1, y=2)))

    tree3 = {
        '__dataclass__': '_C',
        'a': {'__dataclass__': '_B', 'x': 1},
        'b': {'__dataclass__': '_B', 'x': 1, 'y': 2},
        '_z': 5,
    }
    res3 = pytree.load(tree3)
    self.assertEqual(res3, _C(a=_B(x=1, y=2), b=_B(x=1, y=2)))

  def test_load_dict(self):
    tree1 = {
        'a': {'x': 1},
        'b': {'x': 1, 'y': 2},
        'd': [{'__dataclass__': '_A', 'x': 10}, {'x': 1}],
    }
    res1 = pytree.load(tree1)
    self.assertEqual(res1, {
        'a': {'x': 1},
        'b': {'x': 1, 'y': 2},
        'd': [_A(x=10), {'x': 1}],
    })

  def test_dump(self):
    c = _C(a=_B(x=1), b=_B(x=1, y=2))
    self.assertEqual(
        pytree.dump(c),
        {
            '__dataclass__': '_C',
            'a': {'__dataclass__': '_B', 'x': 1, 'y': 2},
            'b': {'__dataclass__': '_B', 'x': 1, 'y': 2},
            'd': [],
        },
    )

    d = {'ab/': 1}
    with self.assertRaises(ValueError):
      pytree.dump(d)

  def test_ndarray(self):
    c = {
        'x': np.array([1, 2, 3], dtype=np.int32),
        'y': [1, 2, np.array([1, 2])],
    }
    expected_dumped = {
        'x': {'data': [1, 2, 3], '__numpy_ndarray_dtype__': 'int32'},
        'y': [1, 2, {'data': [1, 2], '__numpy_ndarray_dtype__': 'int64'}],
    }
    self.assertEqual(pytree.dump(c), expected_dumped)
    loaded = pytree.load(expected_dumped)
    np.testing.assert_array_equal(loaded['x'], c['x'])
    self.assertLen(loaded['y'], 3)
    np.testing.assert_array_equal(loaded['y'][2], c['y'][2])
    self.assertEqual(loaded['y'][:2], c['y'][:2])

  def test_concatenate_pytrees(self):
    tree1 = _C(a=_A(x=1), b=_B(x=1, y=2), d=[1, 3, 4])
    tree2 = _C(a=_A(x=1), b=_B(x=1, y=2), d=[5, 6])
    self.assertEqual(
        pytree.concatenate_pytrees([tree1, tree2]),
        _C(a=_A(x=1), b=_B(x=1, y=2), d=[1, 3, 4, 5, 6]),
    )
    tree2.b.y = 4
    with self.assertRaises(ValueError):
      pytree.concatenate_pytrees([tree1, tree2])

  def test_save_and_load_pytree(self):
    tree = _C(a=_A(x=1), b=_B(x=1, y=2), d=[1, 3, 4])
    path = self.create_tempfile().full_path
    pytree.save_pytree_to(tree, path)
    self.assertEqual(pytree.load_pytree_from(path), tree)


if __name__ == '__main__':
  absltest.main()

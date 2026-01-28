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
from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply.utils import common


class CommonTest(absltest.TestCase):

  def test_parameterized_string(self):
    tmpl = common.ParameterizedString(
        '{a}/{b}/{c}', dict(a=['1', '2'], b=['x'], c=['y', 'z'])
    )
    self.assertEqual(tmpl.available_parameters, ['a', 'b', 'c'])
    self.assertEqual(
        [tmpl.format(**ps) for ps in tmpl], ['1/x/y', '1/x/z', '2/x/y', '2/x/z']
    )

  def test_simple_quantize(self):
    q = common.quantize_array(jnp.array([-0.8, 1]), symmetric=True)
    self.assertEqual(
        q['quant_array'].tolist(),
        [-102, 127],
    )
    self.assertEqual(q['quant_array'].dtype, jnp.int8)
    self.assertAlmostEqual(q['scale'], 0.00787402)
    self.assertNotIn('zero_point', q)
    v = common.convert_or_dequantize(q, dtype=jnp.bfloat16)
    self.assertSequenceAlmostEqual(v.tolist(), [-0.8, 1], delta=0.01)
    self.assertEqual(v.dtype, jnp.bfloat16)

    q = common.quantize_array(jnp.array([-0.8, 1]), symmetric=False)
    self.assertEqual(
        q['quant_array'].tolist(),
        [-128, 127],
    )
    self.assertAlmostEqual(q['scale'], 0.00703125)
    self.assertAlmostEqual(q['zero_point'], 0.09999999)
    v = common.convert_or_dequantize(q, dtype=jnp.float32)
    self.assertSequenceAlmostEqual(v.tolist(), [-0.8, 1], delta=0.01)

  def test_quantization_calculation(self):
    x = jnp.array(
        [1.2, -0.5, -4.3, 1.2, -3.1, 0.8, 2.4, 5.4], dtype=jnp.float32
    )
    expected_q_deq = jnp.array(
        [1.1905512, -0.5102362, -4.294488, 1.1905512, -3.103937,
         0.807874, 2.3811023, 5.4],
        dtype=jnp.float32)
    expected_q = jnp.array(
        [28, -12, -101, 28, -73, 19, 56, 127], dtype=jnp.int8
    )
    q = common.quantize_array(x, symmetric=True)
    q_deq_x = common.convert_or_dequantize(q, dtype=jnp.float32)
    np.testing.assert_allclose(
        np.array(q['quant_array']), np.array(expected_q), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(q_deq_x), np.array(expected_q_deq), atol=1e-6
    )

  def test_annotated_array(self):
    x = common.AnnotatedArray.create(jnp.array([1, 2]), dim_annotation='io')
    self.assertEqual(x.array.tolist(), [1, 2])
    self.assertEqual(x.metadata, {'dim_annotation': 'io'})
    z = common.AnnotatedArray.create(jnp.array([5, 6]), dim_annotation='oi')
    y = jnp.array([3, 4])
    y1, z1 = common.transfer_metadata((x, x), (y, z))
    print('=' * 50)
    print(f'y1: {y1}')
    print('=' * 50)
    self.assertEqual(y1.metadata, {'dim_annotation': 'io'})
    self.assertEqual(z1.metadata, {'dim_annotation': 'io'})
    self.assertEqual(y1.array.tolist(), y.tolist())
    self.assertEqual(z1.array.tolist(), z.array.tolist())

  def test_find_unused_argpaths(self):
    def _func(tree):
      return tree['a']['x'] + tree['b'][1], tree['d'], jnp.zeros((1,))

    tree = {
        'a': {'x': jnp.array([1, 2]), 'y': jnp.array(0)},
        'b': [jnp.array([1, 2, 3]), jnp.array([3, 4])],
        'd': jnp.array(3),
        'c': jnp.array(4),
    }
    unused_argpaths = common.find_unused_argpaths(_func, tree)
    self.assertEqual(
        unused_argpaths,
        [
            jax.tree_util.KeyPath(
                [jax.tree_util.DictKey('a'), jax.tree_util.DictKey('y')]
            ),
            jax.tree_util.KeyPath(
                [jax.tree_util.DictKey('b'), jax.tree_util.SequenceKey(0)]
            ),
            jax.tree_util.KeyPath([
                jax.tree_util.DictKey('c'),
            ]),
        ],
    )

  def test_sorted_with_indices(self):
    x = [2, 1, 3, 4]
    sorted_x, indices = common.sorted_with_indices(x)
    self.assertSequenceEqual(sorted_x, [1, 2, 3, 4])
    self.assertSequenceEqual(indices, [1, 0, 2, 3])
    unsorted_x = common.unsorted(sorted_x, indices)
    self.assertSequenceEqual(unsorted_x, x)

    sorted_x, indices = common.sorted_with_indices(x, key=lambda e: -e)
    self.assertSequenceEqual(sorted_x, [4, 3, 2, 1])
    self.assertSequenceEqual(indices, [3, 2, 0, 1])
    unsorted_x = common.unsorted(sorted_x, indices)
    self.assertSequenceEqual(unsorted_x, x)

  def test_convert_array_with_abstract(self):
    x = jnp.ones((2, 3), dtype=jnp.float32)
    x = jax.lax.with_sharding_constraint(
        x,
        js.NamedSharding(
            mesh=js.Mesh(jax.devices(), 'y'),
            spec=js.PartitionSpec('y'),
        ),
    )
    a = jax.ShapeDtypeStruct(
        shape=(2, 3),
        dtype=jnp.float32,
        sharding=js.NamedSharding(
            mesh=js.Mesh(jax.devices(), 'x'),
            spec=js.PartitionSpec('x'),
        ),
    )
    z = common.convert_array_with_abstract(x, a)
    self.assertEqual(z.sharding, a.sharding)
    self.assertEqual(z.dtype, a.dtype)
    self.assertEqual(z.shape, a.shape)

  def test_neg_inf(self):
    self.assertAlmostEqual(common.neg_inf(jnp.float32), -1.7014117e38)
    self.assertAlmostEqual(common.neg_inf(jnp.float16), -3.275e4)
    self.assertAlmostEqual(common.neg_inf(jnp.bfloat16), -1.6947657e38)
    self.assertAlmostEqual(common.neg_inf(jnp.int32), -1073741823.5)

  def test_reduce_same(self):
    with self.assertRaises(ValueError):
      common.reduce_same([1, 1, 2])
    with self.assertRaises(IndexError):
      common.reduce_same([])
    self.assertEqual(common.reduce_same([1, 1, 1]), 1)


class RaggedArrayTest(absltest.TestCase):

  def test_ragged_array(self):
    x = [np.array([1, 2, 3]), np.array([4]), np.array([5, 6])]
    ra = common.RaggedArray.from_numpy_list(x)
    jax.tree_util.tree_map(np.testing.assert_array_equal, ra.to_numpy_list(), x)

    self.assertEqual(ra.lens.tolist(), [3, 1, 2])
    self.assertEqual(ra.data.tolist(), [1, 2, 3, 4, 5, 6])

    self.assertEqual(ra.total_length.tolist(), 6)
    self.assertEqual(ra.capacity, 6)
    self.assertEqual(ra.dtype, jnp.int32)
    self.assertEqual(ra.batch_size, 3)
    self.assertEqual(ra.row_ids.tolist(), [0, 0, 0, 1, 2, 2])
    self.assertEqual(ra.intra_offset.tolist(), [0, 1, 2, 0, 0, 1])

    y = common.RaggedArray(
        jnp.array([1, 2, 3, 4, 5, 0, 0, 0], dtype=jnp.float32),
        jnp.array([2, 0, 3]),
    )
    self.assertEqual(y.total_length.tolist(), 5)
    self.assertEqual(y.batch_size, 3)
    self.assertEqual(y.capacity, 8)
    self.assertEqual(y.dtype, jnp.float32)
    self.assertEqual(y.row_ids.tolist(), [0, 0, 2, 2, 2, 2, 2, 2])
    self.assertEqual(y.intra_offset.tolist(), [0, 1, 0, 1, 2, 3, 4, 5])

  def test_ragged_array_invalid_lens(self):
    x = common.RaggedArray(data=jnp.array([1, 2, 3]), lens=jnp.array([2, 1, 3]))
    self.assertFalse(x.is_valid)

  def test_ragged_concat(self):
    x = common.RaggedArray.from_numpy_list(
        [np.array([1, 2, 3]), np.array([4]), np.array([5, 6])]
    )
    y = common.RaggedArray(
        jnp.array([1, 2, 3, 4, 5, 0, 0, 0]), jnp.array([2, 0, 3])
    )
    z = x.concat(y)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        z.to_numpy_list(),
        [
            np.array([1, 2, 3, 1, 2]),
            np.array([4]),
            np.array([5, 6, 3, 4, 5]),
        ],
    )

  def test_nd_ragged(self):
    x = common.RaggedArray.from_numpy_list([
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.empty([0, 3], dtype=np.int32),
        np.array([[7, 8, 9]]),
    ])
    y = common.RaggedArray.from_numpy_list([
        np.array([[10, 11, 12]]),
        np.array([[13, 14, 15], [16, 17, 18]]),
        np.empty([0, 3], dtype=np.int32),
    ])
    z = x.concat(y)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        z.to_numpy_list(),
        [
            np.array([[1, 2, 3], [4, 5, 6], [10, 11, 12]]),
            np.array([[13, 14, 15], [16, 17, 18]]),
            np.array([[7, 8, 9]]),
        ],
    )
    new_z = z.extend_capacity_to(8)
    self.assertEqual(new_z.capacity, 8)
    new_z = new_z.set_padding_value(-1)
    self.assertEqual(
        new_z.data.tolist(),
        [
            [1, 2, 3],
            [4, 5, 6],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [7, 8, 9],
            [-1, -1, -1],
            [-1, -1, -1],
        ],
    )

  def test_padding_value(self):
    x = common.RaggedArray(
        jnp.array([1, 2, 3, 4, 5, 0, 0, 0]), jnp.array([2, 0, 3])
    )
    y = x.set_padding_value(6)
    self.assertEqual(y.data.tolist(), [1, 2, 3, 4, 5, 6, 6, 6])

  def test_to_padded_dense(self):
    x = common.RaggedArray(
        jnp.array([1, 2, 3, 4, 5, 0, 0, 0]), jnp.array([2, 0, 3])
    )
    y = x.to_padded_dense(max_len=4, padding_value=6)
    np.testing.assert_array_equal(
        y, np.array([[1, 2, 6, 6], [6, 6, 6, 6], [3, 4, 5, 6]])
    )

  def test_keep_rows(self):
    x = common.RaggedArray(
        jnp.array([1, 2, 3, 4, 5, 6, 7, 8]), jnp.array([3, 0, 2, 3])
    )
    y = x.keep_rows(jnp.array([False, True, True, False]))
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        y.to_numpy_list(),
        [np.array([]), np.array([]), np.array([4, 5]), np.array([])],
    )

  def test_keep_last_ncols(self):
    x = common.RaggedArray(
        jnp.array([1, 2, 3, 4, 5, 6, 7, 8]), jnp.array([3, 0, 2, 3])
    )
    y = x.keep_last_ncols(2)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        y.to_numpy_list(),
        [np.array([2, 3]), np.array([]), np.array([4, 5]), np.array([7, 8])],
    )

if __name__ == '__main__':
  absltest.main()

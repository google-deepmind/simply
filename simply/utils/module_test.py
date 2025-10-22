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
from simply.utils import initializer
from simply.utils import module
from simply.utils import pytree
from simply.utils import sharding


class EinsumLinearTest(absltest.TestCase):

  def test_dump(self):
    js = pytree.dump(
        module.EinsumLinear(eqn='df,...d->...f', weight_shape=[2, 4])
    )
    self.assertEqual(
        js,
        {
            '__dataclass__': 'Module:EinsumLinear',
            'activation_dtype': 'bfloat16',
            'bias_name': 'b',
            'bias_term': '',
            'eqn': 'df,...d->...f',
            'output_partition': sharding.NOT_ANNOTATED,
            'weight_dtype': 'float32',
            'weight_init': {
                '__dataclass__': 'Initializer:XavierUniformInit',
                'scale': 2.449489742783178,
            },
            'weight_name': 'w',
            'weight_partition': None,
            'weight_dim_annotation': 'io',
            'weight_shape': [2, 4],
            'bias_init': {'__dataclass__': 'Initializer:ZeroInit'},
        }
    )

  def test_classic_linear(self):
    layer = module.EinsumLinear(
        eqn='df,...d->...f',
        bias_term='f',
        weight_shape=[2, 4],
        weight_partition=('data', 'model'),
        output_partition=(('replica', 'data'), None, None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, 'io')
    self.assertSequenceEqual(layer.bias_partition, (None,))
    self.assertEqual(layer.bias_shape, (4,))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (1, 3, 2)))
    self.assertEqual(output.shape, (1, 3, 4))

  def test_classic_linear2(self):
    layer = module.EinsumLinear(
        eqn='df,...f->...d',
        bias_term='d',
        weight_shape=[2, 4],
        weight_partition=('data', 'model'),
        output_partition=(('replica', 'data'), None, 'model'),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, 'oi')
    self.assertSequenceEqual(layer.bias_partition, ('model',))
    self.assertEqual(layer.bias_shape, (2,))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (1, 3, 4)))
    self.assertEqual(output.shape, (1, 3, 2))

  def test_gmoe_linear(self):
    layer = module.EinsumLinear(
        eqn='ndf,...nd->...nf',
        bias_term='nf',
        weight_shape=[2, 3, 4],
        weight_partition=('model', None, 'data'),
        output_partition=(('replica', 'data'), None, 'model', None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, '.io')
    self.assertEqual(layer.bias_shape, (2, 4))
    self.assertSequenceEqual(layer.bias_partition, ('model', None))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (1, 3, 2, 3)))
    self.assertEqual(output.shape, (1, 3, 2, 4))

    layer = module.EinsumLinear(
        eqn='ndf,...nd->...nf',
        weight_shape=[2, 3, 4],
        bias_term='f',
        weight_partition=('model', None, 'data'),
        output_partition=(('replica', 'data'), None, 'model', None),
    )
    layer.setup()
    self.assertSequenceEqual(layer.bias_partition, (None,))
    self.assertEqual(layer.bias_shape, (4,))

  def test_gmoe_linear2(self):
    layer = module.EinsumLinear(
        eqn='ndf,n...d->n...f',
        bias_term='nf',
        weight_shape=[2, 3, 4],
        weight_partition=('model', None, 'data'),
        output_partition=('model', ('replica', 'data'), None, None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, '.io')
    self.assertSequenceEqual(
        layer.bias_partition, ('model', None)
    )
    self.assertEqual(layer.bias_shape, (2, 4))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (2, 1, 3, 3)))
    self.assertEqual(output.shape, (2, 1, 3, 4))

  def test_mha_qkv(self):
    layer = module.EinsumLinear(
        eqn='ndh,...d->n...h',
        bias_term='nh',
        weight_shape=[2, 3, 4],
        weight_partition=('model', 'data', None),
        output_partition=('model', ('replica', 'data'), None, None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, 'oio')
    self.assertSequenceEqual(layer.bias_partition, ('model', None))
    self.assertEqual(layer.bias_shape, (2, 4))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (1, 2, 3)))
    self.assertEqual(output.shape, (2, 1, 2, 4))

  def test_combined_qkv(self):
    batch_size = 2
    seq_len = 3
    model_dim = 4
    n_heads = 5
    per_head_dim = 6
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    x = jax.random.uniform(xk, (batch_size, seq_len, model_dim))
    layer = module.EinsumLinear(
        eqn='sdnh,...d->s...nh',
        bias_term='snh',
        weight_dim_annotation='.ioo',
        weight_shape=[3, model_dim, n_heads, per_head_dim],
        weight_partition=(None, 'data', 'model', None),
        output_partition=(None, ('replica', 'data'), None, 'model', None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, '.ioo')
    self.assertSequenceEqual(layer.bias_partition, (None, 'model', None))
    self.assertEqual(layer.bias_shape, (3, n_heads, per_head_dim))
    self.assertEqual(layer.bias_dim_annotation, '.hh')
    params = layer.init(pk)
    output = layer.apply(params, x)
    self.assertEqual(
        output.shape, (3, batch_size, seq_len, n_heads, per_head_dim))

  def test_mha_o(self):
    layer = module.EinsumLinear(
        eqn='ndh,n...h->...d',
        bias_term='d',
        weight_shape=[2, 3, 4],
        weight_partition=('model', 'data', None),
        output_partition=('model', ('replica', 'data'), None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, 'ioi')
    self.assertSequenceEqual(layer.bias_partition, (None,))
    self.assertEqual(layer.bias_shape, (3,))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (2, 1, 2, 4)))
    self.assertEqual(output.shape, (1, 2, 3))

  def test_gmoe_routing(self):
    layer = module.EinsumLinear(
        eqn='mn,...nd->...md',
        bias_term='m',
        weight_shape=[2, 2],
        weight_partition=(None, 'model'),
        output_partition=(('replica', 'data'), None, 'model', None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, 'oi')
    self.assertSequenceEqual(layer.bias_partition, ('model',))
    self.assertEqual(layer.bias_shape, (2,))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (1, 3, 2, 3)))
    self.assertEqual(output.shape, (1, 3, 2, 3))

    layer = module.EinsumLinear(
        eqn='mn,n...d->m...d',
        bias_term='m',
        weight_shape=[2, 2],
        weight_init=initializer.IdentityInit(),
        weight_partition=('model', None),
        output_partition=('model', ('replica', 'data'), None, None),
    )
    layer.setup()
    self.assertEqual(layer.weight_dim_annotation, 'oi')
    self.assertSequenceEqual(layer.bias_partition, ('model',))
    self.assertEqual(layer.bias_shape, (2,))
    prng_key = jax.random.PRNGKey(0)
    xk, pk = jax.random.split(prng_key)
    params = layer.init(pk)
    output = layer.apply(params, jax.random.uniform(xk, (2, 1, 3, 3)))
    self.assertEqual(output.shape, (2, 1, 3, 3))


if __name__ == '__main__':
  absltest.main()

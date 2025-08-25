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
import math

import jax
import numpy as np

from absl.testing import absltest
from simply.utils import initializer


class InitializerTest(absltest.TestCase):

  def test_xavier_uniform(self):
    init_fn = initializer.XavierUniformInit(3)
    shape = [1, 2, 3, 4]
    prng_key = jax.random.PRNGKey(0)
    w1 = init_fn(prng_key, shape, dim_annotation='ioio', dtype='float32')
    scale = 3. / math.sqrt(3 + 2 * 4)
    w1_expected = jax.random.uniform(
        prng_key,
        shape,
        minval=-scale,
        maxval=scale,
        dtype='float32',
    )
    self.assertTrue(np.allclose(w1, w1_expected))

    w2 = init_fn(prng_key, shape, dim_annotation='..oi', dtype='float32')
    scale = 3.0 / math.sqrt(3 + 4)
    w2_expected = jax.random.uniform(
        prng_key,
        shape,
        minval=-scale,
        maxval=scale,
        dtype='float32',
    )
    self.assertTrue(np.allclose(w2, w2_expected))

  def test_he_normal(self):
    init_fn = initializer.HeNormalInit(3)
    shape = [1, 2, 3, 4]
    prng_key = jax.random.PRNGKey(0)
    w1 = init_fn(prng_key, shape, dim_annotation='ioio', dtype='float32')
    scale = 3. / math.sqrt(3)
    w1_expected = jax.random.normal(prng_key, shape) * scale
    self.assertTrue(np.allclose(w1, w1_expected))

    w2 = init_fn(prng_key, shape, dim_annotation='..ii', dtype='float32')
    scale = 3.0 / math.sqrt(3 * 4)
    w2_expected = jax.random.normal(prng_key, shape) * scale
    self.assertTrue(np.allclose(w2, w2_expected))


if __name__ == '__main__':
  absltest.main()

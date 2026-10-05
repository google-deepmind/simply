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
"""Tests for distributions with zero-probability categories."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special
from simply.utils import distributions


class DistributionNumericsTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
      masked=(False, True),
      compiled=(False, True),
  )
  def test_zero_probability_values_and_gradients(self, dtype, masked, compiled):
    logits = jnp.array([1.0, 0.0, 3.0], dtype=dtype)
    mask = jnp.array([True, False, True])

    def distribution(x):
      if masked:
        return distributions.MaskedCategorical(x, mask, neg_inf=-jnp.inf)
      return distributions.Categorical(jnp.where(mask, x, -jnp.inf))

    def evaluate(x):
      d = distribution(x)
      return d.log_prob(jnp.int32(0)), d.entropy()

    evaluate_fn = jax.jit(evaluate) if compiled else evaluate
    log_prob, entropy = evaluate_fn(logits)
    log_p = np.array([1.0, 3.0]) - special.logsumexp([1.0, 3.0])
    p = np.exp(log_p)
    h = -np.sum(p * log_p)
    tolerance = 2e-2 if dtype == jnp.bfloat16 else 3e-3
    np.testing.assert_allclose(log_prob, log_p[0], rtol=tolerance)
    np.testing.assert_allclose(entropy, h, rtol=tolerance)
    for index, expected in (
        (0, np.array([1.0 - p[0], 0.0, -p[1]])),
        (1, np.array([-p[0] * (log_p[0] + h), 0.0, -p[1] * (log_p[1] + h)])),
    ):
      gradient = jax.grad(lambda x: evaluate(x)[index])
      if compiled:
        gradient = jax.jit(gradient)
      np.testing.assert_allclose(
          gradient(logits), expected, rtol=tolerance, atol=2e-3
      )
    d = distribution(logits)
    self.assertEqual(float(d.log_prob(jnp.int32(1))), -np.inf)
    self.assertEqual(float(d.prob(jnp.int32(1))), 0.0)
    samples = d.sample(jax.random.key(0), shape=(128,))
    self.assertFalse(np.any(np.asarray(samples) == 1))

  @parameterized.parameters(False, True)
  def test_float16_default_mask(self, compiled):
    def evaluate(logits):
      d = distributions.MaskedCategorical(
          logits, jnp.array([True, False, True])
      )
      return d.log_prob(jnp.int32(0)), d.entropy()

    if compiled:
      evaluate = jax.jit(evaluate)
    result = evaluate(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float16))
    self.assertTrue(all(np.isfinite(value) for value in result))

  def test_batched_and_broadcast_gather(self):
    logits = jnp.array([[0.0, -jnp.inf, 1.0], [-jnp.inf, 2.0, 0.0]])
    indices = jnp.array([[0, 1], [2, 2], [1, 0]])
    dist = distributions.Categorical(logits)
    expected = np.asarray(jax.nn.log_softmax(logits))[np.arange(2), indices]
    np.testing.assert_allclose(dist.log_prob(indices), expected)
    np.testing.assert_allclose(
        jax.vmap(lambda x: distributions.Categorical(x).entropy())(logits),
        dist.entropy(),
    )

  def test_invalid_distributions_remain_invalid(self):
    for logits in ([jnp.nan, 0.0], [jnp.inf, 0.0], [-jnp.inf, -jnp.inf]):
      self.assertTrue(
          np.isnan(distributions.Categorical(jnp.array(logits)).entropy())
      )


if __name__ == '__main__':
  absltest.main()

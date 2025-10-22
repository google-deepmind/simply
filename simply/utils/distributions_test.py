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
"""Unit test for distributions."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from simply.utils import distributions
from simply.utils import masked


class DistributionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.prng_key = jax.random.PRNGKey(0)
    self.rtol = 1e-5

  def test_categorical(self):
    logits = np.random.randn(2, 10)
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    m = distributions.Categorical(logits)

    with self.subTest("DType"):
      self.assertEqual(m.dtype, logits.dtype)

    with self.subTest("Entropy"):
      np.testing.assert_allclose(
          m.entropy(), scipy.stats.entropy(probs, axis=-1), rtol=self.rtol
      )

    with self.subTest("Sample"):
      samples = m.sample(self.prng_key)
      self.assertEqual(samples.shape, (2,))

    with self.subTest("BatchedSample"):
      n = 10000
      samples = m.sample(self.prng_key, shape=(n, 2))
      self.assertEqual(samples.shape, (n, 2))
      sample_probs = np.vstack(
          [np.bincount(samples[:, 0]) / n, np.bincount(samples[:, 1]) / n]
      )
      np.testing.assert_allclose(sample_probs, probs, atol=1e-2)

    with self.subTest("Prob"):
      samples = m.sample(self.prng_key)
      sample_probs = m.prob(samples)
      expected_sample_probs = jnp.squeeze(
          jnp.take_along_axis(probs, jnp.expand_dims(samples, -1), axis=-1),
          axis=-1,
      )
      self.assertEqual(sample_probs.shape, samples.shape)
      np.testing.assert_allclose(
          sample_probs, expected_sample_probs, rtol=self.rtol
      )

    with self.subTest("BatchedProb"):
      n = 4
      samples = m.sample(self.prng_key, shape=(n, 2))
      sample_probs = m.prob(samples)
      self.assertEqual(sample_probs.shape, samples.shape)
      np.testing.assert_allclose(
          sample_probs[:, 0],
          jnp.take_along_axis(probs[0], samples[:, 0], axis=-1),
          rtol=self.rtol,
      )
      np.testing.assert_allclose(
          sample_probs[:, 1],
          jnp.take_along_axis(probs[1], samples[:, 1], axis=-1),
          rtol=self.rtol,
      )

    with self.subTest("LogProb"):
      samples = m.sample(self.prng_key)
      sample_log_probs = m.log_prob(samples)
      expected_sample_log_probs = jnp.squeeze(
          jnp.take_along_axis(log_probs, jnp.expand_dims(samples, -1), axis=-1),
          axis=-1,
      )
      self.assertEqual(sample_log_probs.shape, samples.shape)
      np.testing.assert_allclose(sample_log_probs, expected_sample_log_probs)

    with self.subTest("BatchedLogProb"):
      n = 4
      samples = m.sample(self.prng_key, shape=(n, 2))
      sample_log_probs = m.log_prob(samples)
      self.assertEqual(sample_log_probs.shape, samples.shape)
      np.testing.assert_allclose(
          sample_log_probs[:, 0],
          jnp.take_along_axis(log_probs[0], samples[:, 0], axis=-1),
      )
      np.testing.assert_allclose(
          sample_log_probs[:, 1],
          jnp.take_along_axis(log_probs[1], samples[:, 1], axis=-1),
      )

  def test_masked_categorical(self):
    logits = np.random.randn(2, 10)
    p = 0.5
    mask = np.random.rand(*logits.shape) <= p
    # The largest index should be sampled for np.bincount.
    mask[:, -1] = True
    m = distributions.MaskedCategorical(logits, mask)
    masked_probs = jax.nn.softmax(
        masked.masked(logits, mask=mask, padding_value=m.neg_inf), axis=-1
    )
    masked_log_probs = jax.nn.log_softmax(
        masked.masked(logits, mask=mask, padding_value=m.neg_inf), axis=-1
    )

    with self.subTest("DType"):
      self.assertEqual(m.dtype, logits.dtype)

    with self.subTest("Entropy"):
      np.testing.assert_allclose(
          m.entropy(),
          scipy.stats.entropy(masked_probs, axis=-1),
          rtol=self.rtol,
      )

    with self.subTest("Sample"):
      samples = m.sample(self.prng_key)
      self.assertEqual(samples.shape, (2,))

    with self.subTest("BatchedSample"):
      n = 10000
      samples = m.sample(self.prng_key, shape=(n, 2))
      self.assertEqual(samples.shape, (n, 2))
      sample_probs = np.vstack(
          [np.bincount(samples[:, 0]) / n, np.bincount(samples[:, 1]) / n]
      )
      np.testing.assert_allclose(sample_probs, masked_probs, atol=1e-2)

    with self.subTest("Prob"):
      samples = m.sample(self.prng_key)
      sample_probs = m.prob(samples)
      expected_sample_probs = jnp.squeeze(
          jnp.take_along_axis(
              masked_probs, jnp.expand_dims(samples, -1), axis=-1
          ),
          axis=-1,
      )
      self.assertEqual(sample_probs.shape, samples.shape)
      np.testing.assert_allclose(
          sample_probs, expected_sample_probs, rtol=self.rtol
      )

    with self.subTest("BatchedProb"):
      n = 4
      samples = m.sample(self.prng_key, shape=(n, 2))
      sample_probs = m.prob(samples)
      self.assertEqual(sample_probs.shape, samples.shape)
      np.testing.assert_allclose(
          sample_probs[:, 0],
          jnp.take_along_axis(masked_probs[0], samples[:, 0], axis=-1),
          rtol=self.rtol,
      )
      np.testing.assert_allclose(
          sample_probs[:, 1],
          jnp.take_along_axis(masked_probs[1], samples[:, 1], axis=-1),
          rtol=self.rtol,
      )

    with self.subTest("LogProb"):
      samples = m.sample(self.prng_key)
      sample_log_probs = m.log_prob(samples)
      expected_sample_log_probs = jnp.squeeze(
          jnp.take_along_axis(
              masked_log_probs, jnp.expand_dims(samples, -1), axis=-1
          ),
          axis=-1,
      )
      self.assertEqual(sample_log_probs.shape, samples.shape)
      np.testing.assert_allclose(sample_log_probs, expected_sample_log_probs)

    with self.subTest("BatchedLogProb"):
      n = 4
      samples = m.sample(self.prng_key, shape=(n, 2))
      sample_log_probs = m.log_prob(samples)
      self.assertEqual(sample_log_probs.shape, samples.shape)
      np.testing.assert_allclose(
          sample_log_probs[:, 0],
          jnp.take_along_axis(masked_log_probs[0], samples[:, 0], axis=-1),
      )
      np.testing.assert_allclose(
          sample_log_probs[:, 1],
          jnp.take_along_axis(masked_log_probs[1], samples[:, 1], axis=-1),
      )


if __name__ == "__main__":
  absltest.main()

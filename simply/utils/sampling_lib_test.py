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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import pytree
from simply.utils import sampling_lib


class SamplingLibTest(parameterized.TestCase):

  def test_decoding_schedule(self):
    schedule = sampling_lib.DecodingSchedule(
        prefill_size=105,
        begin_position=80,
        end_position=2000,
        chunk_size=100,
    )
    self.assertEqual(80, schedule.get_next_length(0))
    self.assertEqual(105, schedule.get_next_length(80))
    self.assertEqual(205, schedule.get_next_length(105))
    self.assertEqual(2000, schedule.get_next_length(1905))

  def test_sampling_params(self):
    sampling_params = sampling_lib.SamplingParams(
        intermediate_decode_steps=128,
        max_decode_steps=1000,
    )
    schedule = sampling_params.get_decoding_schedule(
        min_input_length=200, max_input_length=400
    )

    self.assertEqual(256, schedule.prefill_size)
    self.assertEqual(199, schedule.begin_position)
    self.assertEqual(1399, schedule.end_position)
    self.assertEqual(128, schedule.chunk_size)

  def test_processed_input_batching(self):
    input1 = sampling_lib.ProcessedInput(
        tokens=[1, 2],
        extra_inputs={'extra_field': np.ones((1, 3))},
    )
    input2 = sampling_lib.ProcessedInput(
        tokens=[1, 2, 3, 4],
        extra_inputs={'extra_field': np.ones((2, 2))},
    )
    input3 = sampling_lib.ProcessedInput(
        tokens=[1, 2, 3],
        extra_inputs={'extra_field': np.ones((3, 1))},
    )

    batch = sampling_lib.ProcessedInputBatch.from_unpadded_inputs(
        [input1, input2, input3]
    )

    expected_extra_field = np.array([
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
    ])

    np.testing.assert_equal(
        batch.tokens,
        np.array([
            [1, 2, 0, 0],
            [1, 2, 3, 4],
            [1, 2, 3, 0],
        ]),
    )
    np.testing.assert_equal(batch.lengths, np.array([2, 4, 3]))
    np.testing.assert_equal(
        batch.extra_inputs['extra_field'], expected_extra_field
    )

  def test_chunk_dump_and_load(self):
    chunk = sampling_lib.Chunk(
        type=sampling_lib.Chunk.Type.TEXT,
        content='chunk',
    )
    tmp_path = self.create_tempfile().full_path
    pytree.save_pytree_to(chunk, tmp_path)
    loaded = pytree.load_pytree_from(tmp_path)
    self.assertEqual(chunk, loaded)

  @parameterized.named_parameters(
      ('top_k=1', 1), ('top_k=5', 5), ('top_k=10', 10), ('top_k=12', 12)
  )
  def test_top_k_mask(self, top_k: int):
    logits = jax.random.normal(jax.random.key(0), (5, 4, 10))
    mask = sampling_lib.top_k_mask(logits, top_k=top_k)
    self.assertEqual(mask.shape, logits.shape)
    self.assertEqual(mask.dtype, jnp.bool)
    np.testing.assert_array_equal(
        jnp.sum(mask, initial=0, axis=-1), min(top_k, 10)
    )
    dtype = logits.dtype
    np.testing.assert_array_less(
        jnp.max(logits, axis=-1, initial=jnp.finfo(dtype).min, where=~mask),
        jnp.min(logits, axis=-1, initial=jnp.finfo(dtype).max, where=mask),
    )

  @parameterized.named_parameters(
      ('top_p=0.0', 0.0),
      ('top_p=0.2', 0.2),
      ('top_p=0.5', 0.5),
      ('top_p=0.8', 0.8),
      ('top_p=1.0', 1.0),
  )
  def test_top_p_mask(self, top_p: float):
    logits = jax.random.normal(jax.random.key(0), (5, 4, 10))
    probs = jax.nn.softmax(logits, axis=-1)
    mask = sampling_lib.top_p_mask(logits, top_p=top_p)
    self.assertEqual(mask.shape, logits.shape)
    self.assertEqual(mask.dtype, jnp.bool)
    np.testing.assert_array_less(
        top_p, jnp.sum(probs, axis=-1, initial=0, where=mask) + 1e-6
    )
    dtype = logits.dtype
    np.testing.assert_array_less(
        jnp.max(logits, axis=-1, initial=jnp.finfo(dtype).min, where=~mask),
        jnp.min(logits, axis=-1, initial=jnp.finfo(dtype).max, where=mask),
    )

  def test_sample_from_logits(self):
    logits = jax.random.normal(jax.random.key(0), (5, 3, 10))
    tokens, logprobs = sampling_lib.sample_from_logits(
        jax.random.key(0), logits, top_k=3, top_p=0.5
    )
    tokens_onehot = jax.nn.one_hot(tokens, 10, dtype=jnp.bool)
    self.assertEqual(tokens.shape, logits.shape[:-1])
    self.assertEqual(logprobs.shape, logits.shape[:-1])
    np.testing.assert_array_equal(
        tokens_onehot & sampling_lib.top_k_mask(logits, top_k=3), tokens_onehot
    )
    np.testing.assert_array_equal(
        tokens_onehot & sampling_lib.top_p_mask(logits, top_p=0.5),
        tokens_onehot,
    )

  def test_compute_log_likelihood(self):
    logits = jax.random.normal(jax.random.key(0), (5, 3, 10))
    tokens, logprobs = sampling_lib.sample_from_logits(
        jax.random.key(0), logits, top_k=3, top_p=0.5
    )
    log_likelihood = sampling_lib.compute_log_likelihood(
        logits, tokens, top_k=3, top_p=0.5
    )
    np.testing.assert_array_equal(log_likelihood, logprobs)


if __name__ == '__main__':
  absltest.main()

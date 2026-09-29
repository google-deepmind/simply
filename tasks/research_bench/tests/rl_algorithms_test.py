# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the modular RL algorithms.

Key property under test: the group-relative advantage is computed in
`advantage()` over the FULL batch (called from build_batch, before the
grad-accum microbatch split), NOT inside compute_loss -- so it is correct
regardless of how train_one_step microbatches the data.
"""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from simply import rl_lib
from tasks.research_bench import rl_algorithms


class _Cfg:
  ppo_clip_eps = 0.2
  kl_coeff = 0.0


class AdvantageTest(absltest.TestCase):
  """The group-relative advantage math (the C1-critical path)."""

  def test_reinforce_group_mean_baseline(self):
    algo = rl_algorithms.ReinforceBaseline(_Cfg())
    # one group of 4: rewards [1,0,0,0], mean=0.25 -> adv = [.75,-.25,-.25,-.25]
    reward = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    ids = np.array([1, 1, 1, 1])
    valid = np.array([True, True, True, True])
    adv = algo.advantage(reward, ids, valid)
    np.testing.assert_allclose(adv, [0.75, -0.25, -0.25, -0.25], atol=1e-6)

  def test_grpo_group_normalized(self):
    algo = rl_algorithms.SimpleGRPO(_Cfg())
    reward = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    ids = np.array([1, 1, 1, 1])
    valid = np.array([True, True, True, True])
    adv = algo.advantage(reward, ids, valid)
    # (r-0.25)/(std+eps); std=0.4330127
    np.testing.assert_allclose(
        adv, (reward - 0.25) / (0.4330127 + 1e-4), rtol=1e-4
    )

  def test_advantage_computed_over_full_batch_not_per_microbatch(self):
    # Two groups; the group baseline must use ALL 4 members of each group.
    # If it were (incorrectly) computed on a 2-row microbatch, the mean would
    # differ. Here we verify the full-batch group mean is used.
    algo = rl_algorithms.ReinforceBaseline(_Cfg())
    reward = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)
    ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    valid = np.ones(8, bool)
    adv = algo.advantage(reward, ids, valid)
    # g1 mean=0.5 -> [.5,.5,-.5,-.5]; g2 mean=0.25 -> [.75,-.25,-.25,-.25]
    np.testing.assert_allclose(
        adv, [0.5, 0.5, -0.5, -0.5, 0.75, -0.25, -0.25, -0.25], atol=1e-6
    )

  def test_invalid_rows_excluded_and_zeroed(self):
    algo = rl_algorithms.SimpleGRPO(_Cfg())
    # last row invalid (padding): must not affect the group mean, adv=0 for it.
    reward = np.array([1.0, 0.0, 0.0, 99.0], np.float32)
    ids = np.array([1, 1, 1, 0])  # padding row has dummy id 0
    valid = np.array([True, True, True, False])
    adv = algo.advantage(reward, ids, valid)
    self.assertEqual(adv[3], 0.0)  # invalid row zeroed
    self.assertNotEqual(adv[0], 0.0)
    # group-1 stats unaffected by the invalid row (mean over [1,0,0]=1/3)
    self.assertAlmostEqual(
        float(adv[0]), (1.0 - 1 / 3) / (np.std([1, 0, 0]) + 1e-4), places=4
    )


class _ConstModel:

  def apply(self, params, input_tokens, **kwargs):
    del kwargs  # unused test stub kwargs.
    b, t = input_tokens.shape
    return jnp.broadcast_to(params[0], (b, t, 5)), {}


def _batch(advantages):
  b, t = len(advantages), 4
  return rl_lib.RLTrainingExampleBatch(
      input_tokens=jnp.zeros((b, t), jnp.int32),
      target_tokens=jnp.ones((b, t), jnp.int32),
      logprobs=jnp.zeros((b, t), jnp.float32),
      target_mask=jnp.ones((b, t), jnp.bool_),
      answer_mask=jnp.ones((b, t), jnp.bool_),
      in_batch_example_id=jnp.arange(b, dtype=jnp.int32) + 1,
      reward=jnp.asarray(advantages, jnp.float32),  # build_batch put adv here
      is_correct=jnp.zeros((b,), jnp.bool_),
      is_valid_for_training=jnp.ones((b,), jnp.bool_),
      ref_logprobs=jnp.zeros((b, t), jnp.float32),
      extra_inputs=None,
  )


class LossTest(absltest.TestCase):
  """compute_loss consumes batch.reward as the (precomputed) advantage."""

  def setUp(self):
    super().setUp()
    self.model = _ConstModel()
    self.params = [jnp.zeros((5,), jnp.float32)]

  def test_grpo_loss_finite_and_has_required_metrics(self):
    algo = rl_algorithms.SimpleGRPO(_Cfg())
    loss, m = algo.compute_loss(
        self.model, self.params, _batch([1.0, -1.0, 0.5, -0.5])
    )
    self.assertTrue(np.isfinite(float(loss)))
    self.assertIn('loss_weight', m)  # required for grad accumulation
    self.assertGreater(float(m['loss_weight']), 0.0)
    self.assertAlmostEqual(float(m['advantage/abs_mean']), 0.75, places=5)

  def test_reinforce_loss_finite(self):
    algo = rl_algorithms.ReinforceBaseline(_Cfg())
    loss, m = algo.compute_loss(self.model, self.params, _batch([0.75, -0.25]))
    self.assertTrue(np.isfinite(float(loss)))
    self.assertIn('loss_weight', m)

  def test_registry_and_hooks(self):
    for name in ['reinforce_baseline', 'simple_grpo']:
      algo = rl_algorithms.RLAlgorithmRegistry.get(name)(_Cfg())
      for hook in (
          'advantage',
          'compute_loss',
          'train_reward',
          'build_batch',
          'sampling_params',
      ):
        self.assertTrue(hasattr(algo, hook))


if __name__ == '__main__':
  absltest.main()

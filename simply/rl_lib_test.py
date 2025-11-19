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
"""Unit test for rl_lib.py."""

from collections.abc import Sequence
import dataclasses
import functools
import pathlib
from typing import TypedDict

from absl.testing import absltest
import numpy as np
import seqio
from simply import config_lib
from simply import data_lib
from simply import rl_lib
from simply.utils import lm_format
from simply.utils import tokenization


_MOCK_VOCAB_NAME = 'mock_vocab'


@lm_format.LMFormatRegistry.register
@dataclasses.dataclass
class MockSimplyV1Chat(lm_format.SimplyV1Chat):
  """Mock LM format for SimplyV1Chat, used in tests."""

  extra_eos_tokens: tuple[str, ...] = ()


class MockDeepScaleRJSONExample(TypedDict):
  """Type definition for a single example in MockDeepScaleRJSONTrain."""

  question: str
  short_answer: str
  answer: str
  uid: str
  id: int


@functools.partial(
    data_lib.DataSourceRegistry.register, name='simply_json:mock_dsr40k_train'
)
@dataclasses.dataclass(frozen=True)
class MockDeepScaleRJSONTrain(data_lib.DeepScaleRJSONTrain):
  """Mock DeepScaleRJSONTrain with 100 examples."""

  def load(self) -> Sequence[MockDeepScaleRJSONExample]:
    examples: list[MockDeepScaleRJSONExample] = [
        {
            'question': f'random question {i}',
            'short_answer': f'random short_answer {i}',
            'answer': f'random answer {i}',
            'uid': f'dsr40k_train-{i}',
            'id': i,
        }
        for i in range(100)
    ]
    return examples[self.example_start_index : self.example_end_index]


class RewardNormalizerTest(absltest.TestCase):

  def test_global(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    example_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    masks = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.bool)

    normalizer = rl_lib.RewardNormalizer.Global()
    normalized_rewards = normalizer.normalize(rewards, example_ids, masks)
    subrewards = rewards[masks]
    expected = (subrewards - np.mean(subrewards)) / np.std(subrewards)
    np.testing.assert_allclose(normalized_rewards[masks], expected)

  def test_by_group(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    example_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    masks = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.bool)

    normalizer = rl_lib.RewardNormalizer.ByGroup()
    normalized_rewards = normalizer.normalize(rewards, example_ids, masks)
    expected = np.array([0, 0, -1, 1])
    np.testing.assert_allclose(normalized_rewards[masks], expected)

    normalized_rewards = normalizer.normalize_by_group(
        rewards, example_ids, masks, std=1.0
    )
    expected = np.array([0, 0, -0.5, 0.5])
    np.testing.assert_allclose(normalized_rewards[masks], expected)


class RunExperimentTest(absltest.TestCase):
  """Tests for RL loop with `lm_rl_test` config."""

  def setUp(self) -> None:
    super().setUp()
    self._mock_vocab = seqio.ByteVocabulary()
    tokenization.TokenizerRegistry.register_value(
        self._mock_vocab, name=_MOCK_VOCAB_NAME
    )

  def tearDown(self):
    super().tearDown()
    tokenization.TokenizerRegistry.unregister(_MOCK_VOCAB_NAME)

  def test_run_experiment_saves_checkpoint(self) -> None:
    num_train_steps = 3
    config = dataclasses.replace(
        config_lib.lm_rl_test(),
        num_train_steps=num_train_steps,
        ckpt_interval=num_train_steps,
        dataset_name='simply_json:mock_dsr40k_train',
        vocab_name=_MOCK_VOCAB_NAME,
        vocab_size=self._mock_vocab.vocab_size,
        # EOS strings must tokenize as a single token. Since the mock vocabulary
        # lacks `SimplyV1Chat.extra_eos_tokens` we override the latter to empty.
        lm_format_name='MockSimplyV1Chat',
    )
    experiment_dir = pathlib.Path(self.create_tempdir().full_path)

    rl_lib.run_experiment(
        config=config,
        sharding_config=config.decoding_sharding_config,
        mesh_shape=(1, 1, 1),
        experiment_dir=str(experiment_dir),
    )

    self.assertTrue(
        (experiment_dir / 'checkpoints' / str(num_train_steps)).exists(),
        msg='Checkpoint directory was not found in the experiment directory.',
    )


if __name__ == '__main__':
  absltest.main()

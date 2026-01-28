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

import dataclasses
import json
import typing

from absl.testing import absltest
import numpy as np

from simply import data_lib
from simply.utils import tokenization


# Mock tokenizer for testing
class MockTokenizer:
  """Byte-based mock tokenizer with special token support.

  Handles special tokens like <reserved_0> through <reserved_9> as single
  tokens. Other text is encoded as UTF-8 bytes (one token per byte).

  Token ID mapping:
    0: pad_id
    1: bos_id
    2: eos_id
    3-255: byte values (text encoded as UTF-8 bytes)
    256-265: <reserved_0> through <reserved_9>
  """
  bos_id = 1
  eos_id = 2
  pad_id = 0

  # Special tokens mapped to IDs starting at 256 (above byte range)
  _special_tokens = {f'<reserved_{i}>': 256 + i for i in range(10)}
  _id_to_special = {v: k for k, v in _special_tokens.items()}

  def encode(self, text):
    tokens = []
    i = 0
    while i < len(text):
      # Check for special tokens
      matched = False
      for special, token_id in self._special_tokens.items():
        if text[i:].startswith(special):
          tokens.append(token_id)
          i += len(special)
          matched = True
          break
      if not matched:
        # Encode single character as UTF-8 bytes
        char_bytes = text[i].encode('utf-8')
        tokens.extend(list(char_bytes))
        i += 1
    return tokens

  def decode(self, ids):
    result = []
    byte_buffer = []
    for token_id in ids:
      if token_id in (self.pad_id, self.bos_id, self.eos_id):
        # Flush byte buffer before skipping
        if byte_buffer:
          result.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
          byte_buffer = []
        continue
      if token_id in self._id_to_special:
        # Flush byte buffer before special token
        if byte_buffer:
          result.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
          byte_buffer = []
        result.append(self._id_to_special[token_id])
      else:
        # Accumulate bytes
        byte_buffer.append(token_id)
    # Flush remaining bytes
    if byte_buffer:
      result.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
    return ''.join(result)


# Register mock tokenizer once
if 'test_tokenizer' not in tokenization.TokenizerRegistry.keys():
  tokenization.TokenizerRegistry.register_value(
      MockTokenizer(), name='test_tokenizer')


################################################################################
# Test Config Validation
################################################################################


class MixtureConfigValidationTest(absltest.TestCase):
  """Tests for MixtureConfig validation logic."""

  def test_requires_datasets(self):
    """Test that MixtureConfig requires at least one dataset."""
    with self.assertRaises(ValueError):
      data_lib.MixtureConfig(datasets=())

  def test_requires_positive_weights(self):
    """Test that MixtureConfig requires positive weights."""
    with self.assertRaises(ValueError):
      data_lib.MixtureConfig(
          datasets=((data_lib.DatasetConfig(source='source'), -0.5),)
      )


################################################################################
# Test Transforms
################################################################################


class TokenizeTransformTest(absltest.TestCase):
  """Tests for TokenizeTransform."""

  def test_basic_tokenization_with_eos(self):
    """Test tokenization adds EOS token."""
    transform = data_lib.TokenizeTransform(
        tokenizer_name='test_tokenizer',
        add_eos=True,
        add_bos=False,
    )
    result = transform.map({'text': 'abc'})

    # 'abc' -> [97, 98, 99], then add eos (2)
    expected = np.array([97, 98, 99, 2], dtype=np.int32)
    np.testing.assert_array_equal(result['tokens'], expected)

  def test_tokenization_with_bos(self):
    """Test tokenization with BOS token."""
    transform = data_lib.TokenizeTransform(
        tokenizer_name='test_tokenizer',
        add_eos=False,
        add_bos=True,
    )
    result = transform.map({'text': 'ab'})

    # 'ab' -> [97, 98], add bos (1) at start
    expected = np.array([1, 97, 98], dtype=np.int32)
    np.testing.assert_array_equal(result['tokens'], expected)

  def test_bytes_input_decoded(self):
    """Test that bytes input is decoded to string."""
    transform = data_lib.TokenizeTransform(
        tokenizer_name='test_tokenizer',
        add_eos=True,
    )
    result = transform.map({'text': b'hi'})

    # b'hi' decoded to 'hi' -> [104, 105], add eos (2)
    expected = np.array([104, 105, 2], dtype=np.int32)
    np.testing.assert_array_equal(result['tokens'], expected)


class NextTokenPredTransformTest(absltest.TestCase):
  """Tests for NextTokenPredTransform."""

  def test_shifts_tokens_correctly(self):
    """Test next-token pred shifts input/target correctly."""
    transform = data_lib.NextTokenPredTransform()
    tokens = np.array([10, 20, 30], dtype=np.int32)
    result = transform.map({'tokens': tokens})

    # decoder_input: [10, 20] (drop last)
    np.testing.assert_array_equal(result['decoder_input_tokens'], [10, 20])
    # decoder_target: [20, 30] (drop first)
    np.testing.assert_array_equal(result['decoder_target_tokens'], [20, 30])
    # loss_weights: all 1s
    np.testing.assert_array_equal(result['decoder_loss_weights'], [1.0, 1.0])

  def test_loss_mask_shifted_correctly(self):
    """Test that loss mask is shifted to align with targets."""
    transform = data_lib.NextTokenPredTransform()
    tokens = np.array([1, 10, 20, 30], dtype=np.int32)
    loss_mask = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    result = transform.map({'tokens': tokens, 'token_loss_mask': loss_mask})

    # loss_weights: [1.0, 1.0, 1.0] (drop first from mask)
    np.testing.assert_array_equal(
        result['decoder_loss_weights'], [1.0, 1.0, 1.0]
    )


class TruncateAndPadTransformTest(absltest.TestCase):
  """Tests for TruncateTransform and PadTransform."""

  def test_pad_short_sequences(self):
    """Test PadTransform pads short sequences."""
    transform = data_lib.PadTransform(seq_len=5, pad_id=0)
    features = {
        'decoder_input_tokens': np.array([1, 2], dtype=np.int32),
        'decoder_loss_weights': np.array([1, 1], dtype=np.float32),
    }
    result = transform.map(features)

    np.testing.assert_array_equal(
        result['decoder_input_tokens'], [1, 2, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        result['decoder_loss_weights'], [1, 1, 0, 0, 0]
    )

  def test_pad_does_not_truncate(self):
    """Test PadTransform does not truncate long sequences."""
    transform = data_lib.PadTransform(seq_len=3, pad_id=0)
    features = {
        'decoder_input_tokens': np.array([1, 2, 3, 4, 5], dtype=np.int32),
    }
    result = transform.map(features)

    # PadTransform should NOT truncate - sequence stays at length 5
    np.testing.assert_array_equal(
        result['decoder_input_tokens'], [1, 2, 3, 4, 5]
    )

  def test_truncate_long_sequences(self):
    """Test TruncateTransform truncates from the left (keeps end)."""
    transform = data_lib.TruncateTransform(seq_len=3)
    features = {
        'decoder_input_tokens': np.array([1, 2, 3, 4, 5], dtype=np.int32),
    }
    result = transform.map(features)

    # Left truncation: keeps [3, 4, 5] (the end)
    np.testing.assert_array_equal(result['decoder_input_tokens'], [3, 4, 5])

  def test_composed_truncate_then_pad(self):
    """Test composing TruncateTransform + PadTransform for fixed length."""
    features = {
        'decoder_input_tokens': np.array([1, 2, 3, 4, 5], dtype=np.int32),
    }
    # Truncate to 3 (from left), then pad to 5
    result = data_lib.TruncateTransform(seq_len=3).map(features)
    result = data_lib.PadTransform(seq_len=5, pad_id=0).map(result)

    # Left truncation keeps [3, 4, 5], then pad adds [0, 0] at end
    np.testing.assert_array_equal(
        result['decoder_input_tokens'], [3, 4, 5, 0, 0]
    )


class ChatFormatTransformTest(absltest.TestCase):
  """Tests for ChatFormatTransform."""

  def test_trainable_roles_creates_loss_mask(self):
    """Test that trainable_roles masks exactly the correct tokens.

    With MockTokenizer and SimplyV1Chat format:
    Structure: <reserved_1>ab<reserved_4><reserved_2>xy<reserved_4>

    Token IDs (MockTokenizer):
      <reserved_1> = 257, <reserved_2> = 258, <reserved_4> = 260
      'a' = 97, 'b' = 98, 'x' = 120, 'y' = 121

    Expected tokens: [257, 97, 98, 260, 258, 120, 121, 260]
    Expected masks:  [0,   0,  0,  0,   0,   1,   1,   1  ]

    - Role markers (<reserved_1>, <reserved_2>) always get mask=0
    - User content ('ab') and user's end marker get mask=0 (not trainable)
    - Assistant content ('xy') and assistant's end marker get mask=1 (trainable)
    """
    transform = data_lib.ChatFormatTransform(
        tokenizer_name='test_tokenizer',
        lm_format_name='SimplyV1Chat',
        add_bos=False,
        trainable_roles=('assistant',),
    )
    conversation = json.dumps([
        {'role': 'user', 'content': 'ab'},
        {'role': 'assistant', 'content': 'xy'},
    ])
    result = transform.map({'conversation': conversation})

    tokens_list = result['tokens'].tolist()
    mask_list = result['token_loss_mask'].tolist()

    # Special token IDs from MockTokenizer
    user_marker_id = 257      # <reserved_1>
    assistant_marker_id = 258  # <reserved_2>
    end_marker_id = 260        # <reserved_4>

    # Expected structure:
    # [user_marker, 'a', 'b', end_marker, asst_marker, 'x', 'y', end_marker]
    expected_tokens = [
        user_marker_id, 97, 98, end_marker_id, assistant_marker_id, 120, 121,
        end_marker_id
    ]
    self.assertEqual(tokens_list, expected_tokens)

    # Verify masks for each position
    # User marker (idx 0) - mask=0
    self.assertEqual(mask_list[0], 0.0, 'User marker should have mask=0')

    # User content 'ab' (idx 1, 2) - mask=0
    self.assertEqual(
        mask_list[1], 0.0, "User content 'a' should have mask=0"
    )
    self.assertEqual(
        mask_list[2], 0.0, "User content 'b' should have mask=0"
    )

    # User's end marker (idx 3) - mask=0
    self.assertEqual(
        mask_list[3], 0.0, "User's end marker should have mask=0"
    )

    # Assistant marker (idx 4) - mask=0
    self.assertEqual(mask_list[4], 0.0, 'Assistant marker should have mask=0')

    # Assistant content 'xy' (idx 5, 6) - mask=1
    self.assertEqual(
        mask_list[5], 1.0, "Assistant content 'x' should have mask=1"
    )
    self.assertEqual(
        mask_list[6], 1.0, "Assistant content 'y' should have mask=1"
    )

    # Assistant's end marker (idx 7) - mask=1
    self.assertEqual(
        mask_list[7], 1.0, "Assistant's end marker should have mask=1"
    )


################################################################################
# Test Data Source
################################################################################


class GetDataSourceTest(absltest.TestCase):
  """Tests for get_data_source function."""

  def test_unknown_source_raises_error(self):
    """Test that getting unknown source raises error."""
    config = data_lib.DatasetConfig(source='nonexistent_source')
    with self.assertRaises(ValueError):
      data_lib.get_data_source(config)


################################################################################
# Test Full Pipeline
################################################################################


@dataclasses.dataclass
class MockExperimentConfig:
  """Mock experiment config for testing."""
  dataset_config: typing.Any
  validation_dataset_config: typing.Any = None
  vocab_name: str = 'test_tokenizer'
  seq_len: int = 8
  batch_size: int = 2
  dataset_seed: int = 42
  prefetch_num_workers: int = 0
  prefetch_per_worker_buffer_size: int = 1
  validation_eval_batch_size: int = 2
  validation_eval_epochs: int = 1


class CreateIterDatasetTest(absltest.TestCase):
  """Tests for create_iter_dataset function."""

  @classmethod
  def setUpClass(cls):
    """Register test sources."""
    super().setUpClass()

    @dataclasses.dataclass(frozen=True)
    class TestSource:
      def __len__(self):
        return 8

      def __getitem__(self, index):
        return {'text': chr(ord('a') + index) * 3}  # 'aaa', 'bbb', ...

    if 'test_source' not in data_lib.DataSourceRegistry.keys():
      data_lib.DataSourceRegistry.register_value(
          TestSource(), name='test_source'
      )

  def test_produces_correct_batch_shape(self):
    """Test that output batches have correct shape."""
    ds_config = data_lib.DatasetConfig(source='test_source')
    config = MockExperimentConfig(dataset_config=ds_config)
    dataset = data_lib.create_iter_dataset(config, training=True)

    batch = next(iter(dataset))
    self.assertEqual(batch['decoder_input_tokens'].shape, (2, 8))
    self.assertEqual(batch['decoder_target_tokens'].shape, (2, 8))
    self.assertEqual(batch['decoder_loss_weights'].shape, (2, 8))

  def test_validation_mode_produces_finite_batches(self):
    """Test validation mode produces finite number of batches."""
    ds_config = data_lib.DatasetConfig(source='test_source')
    config = MockExperimentConfig(
        dataset_config=ds_config,
        validation_dataset_config=ds_config,
    )
    dataset = data_lib.create_iter_dataset(config, training=False)

    batches = list(dataset)
    self.assertLen(batches, 4)  # 8 examples / batch_size 2

  def test_batch_contents_correct(self):
    """Test that batch contents are tokenized correctly."""
    ds_config = data_lib.DatasetConfig(source='test_source')
    config = MockExperimentConfig(
        dataset_config=ds_config,
        validation_dataset_config=ds_config,
    )
    dataset = data_lib.create_iter_dataset(config, training=False)

    batch = next(iter(dataset))
    # First example: 'aaa' -> [97, 97, 97, eos=2]
    # After NextTokenPred: input=[97, 97, 97], target=[97, 97, 2]
    # After padding to seq_len=8
    expected_input = np.array([97, 97, 97, 0, 0, 0, 0, 0], dtype=np.int32)
    expected_target = np.array([97, 97, 2, 0, 0, 0, 0, 0], dtype=np.int32)

    np.testing.assert_array_equal(
        batch['decoder_input_tokens'][0], expected_input
    )
    np.testing.assert_array_equal(
        batch['decoder_target_tokens'][0], expected_target
    )


class MixtureDatasetTest(absltest.TestCase):
  """Tests for mixture datasets."""

  @classmethod
  def setUpClass(cls):
    """Register test sources."""
    super().setUpClass()

    @dataclasses.dataclass(frozen=True)
    class MixSource1:
      def __len__(self):
        return 10

      def __getitem__(self, index):
        return {'text': f'source1_{index}'}

    @dataclasses.dataclass(frozen=True)
    class MixSource2:
      def __len__(self):
        return 10

      def __getitem__(self, index):
        return {'text': f'source2_{index}'}

    if 'mix_source1' not in data_lib.DataSourceRegistry.keys():
      data_lib.DataSourceRegistry.register_value(
          MixSource1(), name='mix_source1'
      )
    if 'mix_source2' not in data_lib.DataSourceRegistry.keys():
      data_lib.DataSourceRegistry.register_value(
          MixSource2(), name='mix_source2'
      )

  def test_mixture_produces_batches(self):
    """Test that mixture dataset produces valid batches."""
    mixture_config = data_lib.MixtureConfig(
        datasets=(
            (data_lib.DatasetConfig(source='mix_source1'), 0.5),
            (data_lib.DatasetConfig(source='mix_source2'), 0.5),
        )
    )
    config = MockExperimentConfig(dataset_config=mixture_config)
    dataset = data_lib.create_iter_dataset(config, training=True)

    batch = next(iter(dataset))
    self.assertIn('decoder_input_tokens', batch)

  def test_pack_before_mix(self):
    """Test pack_before_mix option works."""
    mixture_config = data_lib.MixtureConfig(
        datasets=(
            (data_lib.DatasetConfig(source='mix_source1'), 0.7),
            (data_lib.DatasetConfig(source='mix_source2'), 0.3),
        ),
        pack_before_mix=True,
    )
    config = MockExperimentConfig(dataset_config=mixture_config, seq_len=16)
    dataset = data_lib.create_iter_dataset(config, training=True)

    batch = next(iter(dataset))
    self.assertEqual(batch['decoder_input_tokens'].shape, (2, 16))


if __name__ == '__main__':
  absltest.main()

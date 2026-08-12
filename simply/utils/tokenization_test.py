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
"""Tests for tokenization, focused on token_byte_lengths (bpb support)."""

import math
import os

from absl.testing import absltest
from simply.utils import tokenization

from sentencepiece import SentencePieceTrainer


def _train_spm(tmpdir: str) -> str:
  """Trains a tiny SentencePiece model and returns the .model path."""
  sentences = [
      'the quick brown fox jumps over the lazy dog',
      'hello world this is a small tokenizer test',
      'sentencepiece produces subword pieces for text',
      'bits per byte is a vocabulary independent metric',
      'we accumulate nats and bytes across evaluation batches',
      'lower bits per byte indicates a better language model',
      'training language models requires large amounts of data',
      'transformers use attention mechanisms to model sequences',
      'gradient descent optimizes parameters to minimize loss',
      'validation datasets measure generalization performance',
      'perplexity and cross entropy quantify prediction quality',
      'tokenization splits raw strings into discrete symbol ids',
      'unicode characters may span multiple utf eight bytes',
      'byte fallback pieces represent arbitrary raw bytes safely',
      'padding control and unknown symbols contribute zero bytes',
      'researchers compare architectures across different vocabularies',
  ] * 60
  model_prefix = os.path.join(tmpdir, 'spm')
  corpus_path = os.path.join(tmpdir, 'corpus.txt')
  with open(corpus_path, 'w') as f:
    f.write('\n'.join(sentences))
  SentencePieceTrainer.Train(
      '--input={} --vocab_size=300 --model_prefix={}'
      ' --character_coverage=1.0 --model_type=unigram --pad_id=0 --unk_id=1'
      ' --bos_id=2 --eos_id=3 --byte_fallback=true'.format(
          corpus_path, model_prefix
      )
  )
  return model_prefix + '.model'


class TokenByteLengthsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = _train_spm(self.create_tempdir().full_path)
    self.vocab = tokenization.SimplySentencePieceVocab(model_path)
    self.sp = self.vocab._sp  # pylint: disable=protected-access

  def test_returns_none_for_non_spm_vocab(self):
    test_vocab = tokenization.TestVocab(['a', 'b', 'c'])
    self.assertIsNone(test_vocab.token_byte_lengths())

  def test_length_matches_vocab_size(self):
    byte_lengths = self.vocab.token_byte_lengths()
    self.assertIsNotNone(byte_lengths)
    self.assertLen(byte_lengths, self.sp.GetPieceSize())

  def test_specials_are_zero_bytes(self):
    byte_lengths = self.vocab.token_byte_lengths()
    for special_id in (self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id()):
      if special_id is not None and special_id >= 0:
        self.assertEqual(
            byte_lengths[special_id], 0, msg=f'id={special_id} not 0 bytes'
        )
    # The unknown piece must also contribute 0 bytes.
    self.assertEqual(byte_lengths[self.sp.unk_id()], 0)

  def test_byte_fallback_pieces_are_one_byte(self):
    byte_lengths = self.vocab.token_byte_lengths()
    is_byte = getattr(self.sp, 'IsByte', None)
    self.assertIsNotNone(is_byte, 'expected IsByte predicate on this SPM build')
    saw_byte_piece = False
    for token_id in range(self.sp.GetPieceSize()):
      if is_byte(token_id):
        saw_byte_piece = True
        self.assertEqual(byte_lengths[token_id], 1)
    self.assertTrue(saw_byte_piece, 'expected some byte-fallback pieces')

  def test_roundtrip_bytes_match_encoded_sentence(self):
    # For a sentence tokenized into non-special pieces, the sum of per-piece
    # byte lengths should equal the utf-8 byte length of the decoded text.
    byte_lengths = self.vocab.token_byte_lengths()
    sentence = 'the quick brown fox'
    ids = self.vocab.encode(sentence)
    decoded = self.vocab.decode(ids)
    total_bytes = sum(byte_lengths[i] for i in ids)
    # The summed piece byte-lengths reconstruct the text including the leading
    # space marker ('\u2581' -> ' '), so allow up to one extra leading-space
    # byte relative to the (marker-stripped) decoded string.
    decoded_bytes = len(decoded.encode('utf-8'))
    self.assertIn(total_bytes, (decoded_bytes, decoded_bytes + 1))

  def test_bpb_formula_for_known_nats(self):
    # Sanity-check the bpb arithmetic used in run_eval: given total nats and
    # total bytes, bpb = nats / (ln2 * bytes).
    byte_lengths = self.vocab.token_byte_lengths()
    ids = self.vocab.encode('hello world')
    total_bytes = sum(byte_lengths[i] for i in ids)
    self.assertGreater(total_bytes, 0)
    total_nats = 2.0 * total_bytes  # pretend 2 nats/byte for the check.
    bpb = total_nats / (math.log(2) * total_bytes)
    self.assertAlmostEqual(bpb, 2.0 / math.log(2), places=6)


class ByteVocabTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = tokenization.ByteVocab()

  def test_ids_and_size(self):
    self.assertEqual(self.vocab.pad_id, 0)
    self.assertEqual(self.vocab.eos_id, 1)
    self.assertEqual(self.vocab.bos_id, 2)
    self.assertEqual(self.vocab.vocab_size, 259)

  def test_encode_decode_roundtrip_ascii(self):
    text = 'Hello world, C4!'
    ids = self.vocab.encode(text)
    self.assertEqual(ids, [b + 3 for b in text.encode('utf-8')])
    self.assertEqual(self.vocab.decode(ids), text)

  def test_encode_decode_roundtrip_unicode(self):
    text = 'caf\u00e9 \u2013 na\u00efve \U0001f600'
    ids = self.vocab.encode(text)
    self.assertEqual(self.vocab.decode(ids), text)

  def test_decode_skips_specials(self):
    ids = [self.vocab.bos_id] + self.vocab.encode('hi') + [self.vocab.eos_id]
    self.assertEqual(self.vocab.decode(ids), 'hi')  # pyrefly: ignore[bad-argument-type]

  def test_ids_within_vocab_size(self):
    ids = self.vocab.encode('the quick brown fox \u00e9')
    self.assertTrue(all(0 <= i < self.vocab.vocab_size for i in ids))

  def test_token_byte_lengths(self):
    byte_lengths = self.vocab.token_byte_lengths()
    self.assertIsNotNone(byte_lengths)
    self.assertLen(byte_lengths, 259)
    # Specials contribute 0 bytes.
    self.assertEqual(byte_lengths[0], 0)
    self.assertEqual(byte_lengths[1], 0)
    self.assertEqual(byte_lengths[2], 0)
    # Every byte token contributes exactly 1 byte.
    self.assertTrue(all(byte_lengths[i] == 1 for i in range(3, 259)))

  def test_byte_lengths_sum_matches_utf8(self):
    byte_lengths = self.vocab.token_byte_lengths()
    text = 'caf\u00e9 na\u00efve'
    ids = self.vocab.encode(text)
    total_bytes = sum(byte_lengths[i] for i in ids)
    expected_bytes = len(text.encode('utf-8'))
    self.assertEqual(total_bytes, expected_bytes)


if __name__ == '__main__':
  absltest.main()

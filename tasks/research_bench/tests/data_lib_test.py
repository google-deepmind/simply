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

"""Tests for the OSS-port halves of data_lib: C4 shards + `nanodo_c4`.

Neither asset exists in internal form outside it, so these cover the two
substitutions the port makes (see PORTING_NOTES.md). The C4 tests build their
own shards; the vocab tests need the staged sentencepiece model and skip
cleanly without it.
"""

import os
import pickle

from absl.testing import absltest
import numpy as np
from simply.utils import tokenization
from tasks.research_bench import data_lib


def _write_shard(root: str, split: str, index: int, n: int, docs: list[str]):
  """Writes one `.bin` + `.idx.npy` pair in the agreed shard naming."""
  stem = os.path.join(root, f'c4-{split}.{index:05d}-of-{n:05d}')
  blobs = [d.encode('utf-8') for d in docs]
  with open(f'{stem}.bin', 'wb') as f:
    for b in blobs:
      f.write(b)
  offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.uint64)
  np.save(f'{stem}.idx.npy', offsets)


class C4FileSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path
    _write_shard(self.root, 'train', 0, 2, ['alpha', 'beta'])
    _write_shard(self.root, 'train', 1, 2, ['gamma'])
    _write_shard(self.root, 'validation', 0, 1, ['delta', 'epsilon'])

  def test_reads_documents_in_shard_then_file_order(self):
    # Determinism is the whole point: the eval stream must be the same corpus
    # in the same order on every host of a sweep.
    source = data_lib.C4FileSource(split='train', root=self.root)
    self.assertLen(source, 3)
    self.assertEqual(
        [source[i]['text'] for i in range(3)], ['alpha', 'beta', 'gamma']
    )

  def test_splits_are_separate(self):
    source = data_lib.C4FileSource(split='validation', root=self.root)
    self.assertEqual([source[i]['text'] for i in range(2)],
                     ['delta', 'epsilon'])

  def test_num_shards_truncates_in_file_order(self):
    source = data_lib.C4FileSource(split='train', num_shards=1,
                                   root=self.root)
    self.assertLen(source, 2)

  def test_out_of_range_raises_index_error(self):
    # grain's index sampler relies on IndexError to detect exhaustion.
    source = data_lib.C4FileSource(split='train', root=self.root)
    with self.assertRaises(IndexError):
      _ = source[3]
    with self.assertRaises(IndexError):
      _ = source[-1]

  def test_missing_shards_fail_loudly(self):
    source = data_lib.C4FileSource(split='test', root=self.root)
    with self.assertRaises(FileNotFoundError):
      len(source)

  def test_utf8_documents_survive_the_round_trip(self):
    root = self.create_tempdir().full_path
    docs = ['héllo wörld', '日本語のテキスト', 'emoji 🙂']
    _write_shard(root, 'train', 0, 1, docs)
    source = data_lib.C4FileSource(split='train', root=root)
    self.assertEqual([source[i]['text'] for i in range(len(docs))], docs)

  def test_pickles_without_copying_the_corpus(self):
    # grain pickles the source into every worker process; a cached `np.memmap`
    # would be serialized BY VALUE, i.e. the whole corpus per worker (~12 GB
    # for the 16-shard train split -- it hangs the run).
    source = data_lib.C4FileSource(split='train', root=self.root)
    self.assertLen(source, 3)  # populate the memmap cache
    blob = pickle.dumps(source)
    self.assertLess(len(blob), 4096)
    restored = pickle.loads(blob)
    self.assertEqual(
        [restored[i]['text'] for i in range(3)], ['alpha', 'beta', 'gamma']
    )

  def test_eval_protocol_name_is_location_independent(self):
    # `eval_protocol.validation_sources` is compared across runs by the
    # validator; it must not embed the machine's shard directory.
    source = data_lib.C4FileSource(split='validation', root=self.root)
    self.assertEqual(
        str(source), "C4FileSource(split='validation', num_shards=0)"
    )
    self.assertNotIn(self.root, str(source))
    self.assertIn(self.root, repr(source))  # the full repr still logs it

  def test_registered_under_its_class_name(self):
    # config_lib builds the source directly, but core resolves string sources
    # (and names the tb_log eval tag) through the registry.
    self.assertIsNotNone(
        data_lib.DataSourceRegistry.get('C4FileSource', raise_error=False)
    )


class NanodoC4VocabTest(absltest.TestCase):
  """`nanodo_c4` == the internal cc_all.32000.100extra.bos model.

  `pretrain_bpb_v32k` reports bits-per-byte, which divides by
  `token_byte_lengths()`; an off-by-one there or a shifted id space would
  corrupt the scored metric silently, so both are pinned here.
  """

  def setUp(self):
    super().setUp()
    if not os.path.exists(data_lib.NANODO_C4_VOCAB):
      self.skipTest(
          f'nanodo_c4 vocab not staged at {data_lib.NANODO_C4_VOCAB}; run '
          'setup/prepare_assets.py'
      )
    self.vocab = tokenization.TokenizerRegistry.get_instance('nanodo_c4')

  def test_registered(self):
    self.assertIsInstance(self.vocab, tokenization.SimplySentencePieceVocab)

  def test_piece_count_and_special_ids_match_the_internal_model(self):
    # 32100 public pieces + the `<s>` control piece inserted at index 2.
    self.assertEqual(self.vocab._sp.GetPieceSize(), 32_101)  # pylint: disable=protected-access
    self.assertEqual(self.vocab.bos_id, 2)
    self.assertEqual(self.vocab.pad_id, 0)
    self.assertEqual(self.vocab.eos_id, 1)

  def test_encode_decode_round_trip(self):
    text = 'The quick brown fox jumps over the lazy dog.'
    ids = self.vocab.encode(text)
    self.assertTrue(all(0 <= i < 32_101 for i in ids))
    self.assertEqual(self.vocab.decode(ids), text)

  def test_token_byte_lengths_cover_every_id(self):
    byte_lengths = self.vocab.token_byte_lengths()
    self.assertLen(byte_lengths, 32_101)
    self.assertEqual(byte_lengths[self.vocab.pad_id], 0)
    self.assertEqual(byte_lengths[self.vocab.bos_id], 0)
    self.assertEqual(byte_lengths[self.vocab.eos_id], 0)
    # bpb is meaningless unless the byte lengths add up to the source text.
    # The leading-space form is the one sentencepiece round-trips exactly (it
    # renders the `\u2581` marker as a space).
    text = ' bits per byte'
    self.assertEqual(
        sum(byte_lengths[i] for i in self.vocab.encode(text)),
        len(text.encode('utf-8')),
    )


if __name__ == '__main__':
  absltest.main()

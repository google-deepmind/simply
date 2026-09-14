# Copyright 2026 The Simply Authors
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
"""Tests the Qwen3.8 vocabulary registration.

The encoder itself is not exercised: the released tokenizer files live under
`data_lib.VOCABS_DIR`, which a test environment does not have, so
`HuggingFaceVocab.tokenizer` cannot be built here. Everything this module
actually owns -- the registered name, the guard that survives a re-import, the
resolved path and its environment override -- is checked, and constructing the
vocabulary is safe because `HuggingFaceVocab` reads nothing until its first
use. The encoder is covered where the files exist: `utils/lm_format_test.py`
and `compare_to_hf_reference.py`.
"""

import os
from unittest import mock

from absl.testing import absltest
from simply import data_lib
from simply.utils import tokenization as tokenization_lib
from simply.zoo.qwen3p8 import config_lib
from simply.zoo.qwen3p8.utils import tokenization


class RegistrationTest(absltest.TestCase):

  def test_registered_under_the_name_the_config_asks_for(self):
    self.assertEqual(config_lib.qwen3p8_27b().vocab_name, 'Qwen3.8')
    self.assertEqual(tokenization.VOCAB_NAME, 'Qwen3.8')
    self.assertIn('Qwen3.8', tokenization_lib.TokenizerRegistry.keys())
    self.assertIs(
        tokenization_lib.TokenizerRegistry.get('Qwen3.8'),
        tokenization.qwen3p8_vocab,
    )

  def test_registering_again_is_a_no_op(self):
    # `RootRegistry.register` raises on a duplicate name, so an unguarded
    # registration turns a reload -- or a second import path -- into an
    # import-time crash.
    tokenization.register()
    tokenization.register()
    self.assertIs(
        tokenization_lib.TokenizerRegistry.get('Qwen3.8'),
        tokenization.qwen3p8_vocab,
    )

  def test_registry_builds_a_huggingface_vocab(self):
    vocab = tokenization_lib.TokenizerRegistry.get_instance('Qwen3.8')
    self.assertIsInstance(vocab, tokenization_lib.HuggingFaceVocab)
    self.assertEqual(vocab.vocab_path, tokenization.default_vocab_path())


class VocabPathTest(absltest.TestCase):

  def test_default_path_is_the_vocab_store(self):
    with mock.patch.dict(os.environ):
      os.environ.pop(tokenization.VOCAB_PATH_ENV_VAR, None)
      self.assertEqual(
          tokenization.default_vocab_path(),
          os.path.join(data_lib.VOCABS_DIR, 'Qwen3.8'),
      )

  def test_an_empty_environment_variable_falls_back_to_the_default(self):
    with mock.patch.dict(os.environ, {tokenization.VOCAB_PATH_ENV_VAR: ''}):
      self.assertEqual(
          tokenization.default_vocab_path(),
          os.path.join(data_lib.VOCABS_DIR, 'Qwen3.8'),
      )

  def test_environment_overrides_the_default_and_nothing_is_read(self):
    # Also pins laziness: `HuggingFaceVocab` must not touch the directory
    # until its first use, which is what lets a sandbox build one at all.
    with mock.patch.dict(
        os.environ, {tokenization.VOCAB_PATH_ENV_VAR: '/tmp/qwen3p8-vocab'}
    ):
      self.assertEqual(
          tokenization.default_vocab_path(), '/tmp/qwen3p8-vocab'
      )
      self.assertEqual(
          tokenization.qwen3p8_vocab().vocab_path, '/tmp/qwen3p8-vocab'
      )


if __name__ == '__main__':
  absltest.main()

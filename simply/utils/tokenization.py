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
"""Tokenizers."""

from typing import ClassVar, Generic, Protocol

import seqio

from simply.utils import common
from simply.utils import registry


class TokenizerRegistry(registry.RootRegistry):
  """Tokenizer registry."""
  namespace: ClassVar[str] = 'tokenizer'


class SimplyVocab(Protocol, Generic[common.RawT]):
  pad_id: int | None
  bos_id: int | None
  eos_id: int | None

  def encode(self, text: common.RawT) -> list[int]:
    ...

  def decode(self, token_ids: list[int]) -> common.RawT:
    ...


class TestVocab(SimplyVocab[str]):
  """Test vocab."""

  def __init__(self, vocab_list, bos_id=2, eos_id=-1, pad_id=0, unk_id=3):
    self.bos_id = bos_id
    self.eos_id = eos_id
    self.pad_id = pad_id
    self.unk_id = unk_id
    start_id = max(unk_id, pad_id, eos_id, bos_id) + 1
    self._vocab_dict = dict(
        [(w, (i + start_id)) for i, w in enumerate(vocab_list)])
    self._rev_vocab_dict = {v: k for k, v in self._vocab_dict.items()}

  def encode(self, text: str) -> list[int]:
    return [self._vocab_dict.get(w, self.unk_id) for w in text.split()]

  def decode(self, token_ids: list[int]) -> str:
    return ' '.join([self._rev_vocab_dict.get(i, '<unk>') for i in token_ids])


class SimplySentencePieceVocab(SimplyVocab[str]):
  """Wrapper around seqio.SentencePieceVocabulary."""

  def __init__(self, vocab_path: str):
    self._vocab = seqio.SentencePieceVocabulary(vocab_path)
    self.bos_id = self._vocab.bos_id
    self.pad_id = self._vocab.pad_id
    self.eos_id = self._vocab.eos_id

  def encode(self, text: str) -> list[int]:
    return self._vocab.encode(text)  # pytype: disable=bad-return-type

  def decode(self, token_ids: list[int]) -> str:
    return self._vocab.decode(token_ids)


class HuggingFaceVocab(SimplyVocab[str]):
  """Generic class for HuggingFace vocab."""

  def __init__(self, vocab_path: str):
    try:
      import transformers  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
    except ImportError as exc:
      raise ImportError(
          'HuggingFace vocab requires transformers library, which'
          ' is not included. Please include transformers library and try'
          ' again.'
      ) from exc
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(vocab_path)
    self.pad_id = self.tokenizer.pad_token_id
    self.bos_id = self.tokenizer.bos_token_id
    self.eos_id = self.tokenizer.eos_token_id

  def encode(self, text: str) -> list[int]:
    return self.tokenizer.encode(text, add_special_tokens=False)

  def decode(self, token_ids: list[int]) -> str:
    return self.tokenizer.decode(token_ids)

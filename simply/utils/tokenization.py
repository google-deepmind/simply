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

from collections.abc import Mapping
import functools
import json
from typing import Any, ClassVar, Generic, Protocol, cast

from etils import epath
from simply.utils import common
from simply.utils import registry
import tokenizers

import sentencepiece as spm


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
        [(w, (i + start_id)) for i, w in enumerate(vocab_list)]
    )
    self._rev_vocab_dict = {v: k for k, v in self._vocab_dict.items()}

  def encode(self, text: str) -> list[int]:
    return [self._vocab_dict.get(w, self.unk_id) for w in text.split()]

  def decode(self, token_ids: list[int]) -> str:
    return ' '.join([self._rev_vocab_dict.get(i, '<unk>') for i in token_ids])


class SimplySentencePieceVocab(SimplyVocab[str]):
  """Wrapper around sentencepiece.SentencePieceProcessor."""

  def __init__(self, vocab_path: str):
    self._sp = spm.SentencePieceProcessor()
    self._sp.Load(vocab_path)
    self.bos_id = self._sp.bos_id()
    self.pad_id = self._sp.pad_id()
    self.eos_id = self._sp.eos_id()

  def encode(self, text: str) -> list[int]:
    return self._sp.EncodeAsIds(text)

  def decode(self, token_ids: list[int]) -> str:
    return self._sp.DecodeIds(token_ids)


class HuggingFaceVocab(SimplyVocab[str]):
  """Generic class for HuggingFace vocab."""

  def __init__(self, vocab_path: str):
    self.vocab_path = vocab_path

  @functools.cached_property
  def tokenizer(self) -> tokenizers.Tokenizer:
    vocab_path = epath.Path(self.vocab_path)
    return tokenizers.Tokenizer.from_file(
        (vocab_path / 'tokenizer.json').as_posix()
    )

  @functools.cached_property
  def tokenizer_config(self) -> Mapping[str, Any]:
    vocab_path = epath.Path(self.vocab_path)
    with (vocab_path / 'tokenizer_config.json').open() as f:
      return json.load(f)

  def get_token_id(self, name: str) -> int | None:
    token = self.tokenizer_config[name]
    if token is None:
      return None
    if not isinstance(token, str):
      token = token['content']
    if not isinstance(token, str):
      raise ValueError(f'{token=} is not a string ({name=}).')
    return self.tokenizer.token_to_id(token)

  @functools.cached_property
  def bos_id(self) -> int | None:
    return self.get_token_id('bos_token')

  @functools.cached_property
  def eos_id(self) -> int | None:
    return self.get_token_id('eos_token')

  @functools.cached_property
  def pad_id(self) -> int | None:
    return self.get_token_id('pad_token')

  def encode(self, text: str) -> list[int]:
    encoded = self.tokenizer.encode(text)
    return cast(tokenizers.Encoding, encoded).ids

  def decode(self, token_ids: list[int]) -> str:
    return self.tokenizer.decode(token_ids, skip_special_tokens=False)

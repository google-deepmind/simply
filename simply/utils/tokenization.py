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


# SentencePiece replaces the space (U+0020) with this Unicode "lower one eighth
# block" character (U+2581, '▁') as the visible meta-symbol for a leading space
# in a piece. Ref: SentencePiece README, "Whitespace is treated as a basic
# symbol" (github.com/google/sentencepiece), and the kSpaceSymbol definition in
# its source (src/sentencepiece_processor.cc).
_SPM_SPACE_MARKER = '\u2581'


class TokenizerRegistry(registry.RootRegistry):
  """Tokenizer registry."""

  namespace: ClassVar[str] = 'tokenizer'


class SimplyVocab(Protocol, Generic[common.RawT]):
  """Protocol for a tokenizer vocab: encode/decode + optional byte lengths."""

  pad_id: int | None
  bos_id: int | None
  eos_id: int | None

  def encode(self, text: common.RawT) -> list[int]:
    ...

  def decode(self, token_ids: list[int]) -> common.RawT:
    ...

  def token_byte_lengths(self) -> list[int] | None:
    r"""Returns per-token-id utf-8 byte lengths, or None if not defined.

    Powers the vocab-independent bits-per-byte (bpb) eval metric: `result[id]`
    is the number of utf-8 bytes token `id` contributes to the decoded text
    (special/control/unknown tokens -> 0). Override in subclasses.

    Returns:
      A list of length `vocab_size`, or None for vocabs that do not implement
      it (bpb is then unavailable).
    """
    return None


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
    self._sp.LoadFromSerializedProto(
        epath.Path(vocab_path).read_bytes()
    )
    self.bos_id = self._sp.bos_id()
    self.pad_id = self._sp.pad_id()
    self.eos_id = self._sp.eos_id()

  def encode(self, text: str) -> list[int]:
    return self._sp.EncodeAsIds(text)

  def decode(self, token_ids: list[int]) -> str:
    return self._sp.DecodeIds(token_ids)

  def token_byte_lengths(self) -> list[int]:
    r"""Returns per-token-id utf-8 byte lengths for this SentencePiece vocab.

    Pad, control (e.g. ``<s>``/``</s>``/``<pad>``) and unknown pieces contribute
    0 bytes; byte-fallback pieces (``<0xAB>``) contribute exactly 1 byte; the
    SentencePiece space marker ``\u2581`` is treated as a single space (1 byte).

    Returns:
      A list of length `vocab_size` of per-token-id utf-8 byte lengths.
    """
    sp = self._sp
    byte_lengths: list[int] = []
    for token_id in range(sp.GetPieceSize()):
      if token_id == self.pad_id:
        byte_lengths.append(0)
        continue
      # Guard each predicate: not all sentencepiece bindings expose all of them.
      is_control = getattr(sp, 'IsControl', None)
      is_unknown = getattr(sp, 'IsUnknown', None)
      is_byte = getattr(sp, 'IsByte', None)
      if (is_control is not None and is_control(token_id)) or (
          is_unknown is not None and is_unknown(token_id)
      ):
        byte_lengths.append(0)
        continue
      if is_byte is not None and is_byte(token_id):
        byte_lengths.append(1)
        continue
      piece = sp.IdToPiece(token_id)
      piece = piece.replace(_SPM_SPACE_MARKER, ' ')
      byte_lengths.append(len(piece.encode('utf-8')))
    return byte_lengths


class HuggingFaceVocab(SimplyVocab[str]):
  """Generic class for HuggingFace vocab."""

  def __init__(self, vocab_path: str):
    self.vocab_path = vocab_path

  @functools.cached_property
  def tokenizer(self) -> tokenizers.Tokenizer:
    vocab_path = epath.Path(self.vocab_path)
    return tokenizers.Tokenizer.from_buffer(
        (vocab_path / 'tokenizer.json').read_bytes()
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
  def bos_id(self) -> int | None:  # pyrefly: ignore[bad-override]
    return self.get_token_id('bos_token')

  @functools.cached_property
  def eos_id(self) -> int | None:  # pyrefly: ignore[bad-override]
    return self.get_token_id('eos_token')

  @functools.cached_property
  def pad_id(self) -> int | None:  # pyrefly: ignore[bad-override]
    return self.get_token_id('pad_token')

  def encode(self, text: str) -> list[int]:
    encoded = self.tokenizer.encode(text)
    return cast(tokenizers.Encoding, encoded).ids

  def decode(self, token_ids: list[int]) -> str:
    return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class ByteVocab(SimplyVocab[str]):
  """Raw utf-8 byte-level vocab.

  Layout: reserved special ids first, then the 256 raw utf-8 byte values.
  Defaults: pad=0, eos=1, bos=2, and byte value ``b`` maps to id ``b + 3``.
  ``vocab_size == 256 + num_specials == 259`` by default.

  Encoding is ``list(text.encode('utf-8'))`` shifted by ``_byte_offset``;
  decoding reassembles the bytes and utf-8 decodes (skipping specials, and
  tolerating partial/truncated multi-byte sequences via ``errors='replace'``).

  This is a fixed, deterministic vocab (no model file), so it also powers the
  vocab-independent bits-per-byte (bpb) metric: every byte token contributes
  exactly 1 utf-8 byte and every special contributes 0.
  """

  def __init__(self, pad_id: int = 0, eos_id: int = 1, bos_id: int = 2):
    self.pad_id = pad_id
    self.eos_id = eos_id
    self.bos_id = bos_id
    self._specials = {pad_id, eos_id, bos_id}
    # Number of reserved special-token ids at the front of the id space.
    self._num_specials = 3
    self._byte_offset = self._num_specials
    self.vocab_size = 256 + self._num_specials

  def encode(self, text: str) -> list[int]:
    return [b + self._byte_offset for b in text.encode('utf-8')]

  def decode(self, token_ids: list[int]) -> str:
    out = bytearray()
    for i in token_ids:
      if i in self._specials:
        continue
      b = i - self._byte_offset
      if 0 <= b < 256:
        out.append(b)
    return out.decode('utf-8', errors='replace')

  def token_byte_lengths(self) -> list[int]:
    """Per-token-id utf-8 byte length: specials -> 0, byte tokens -> 1."""
    lengths = [0] * self.vocab_size
    for token_id in range(self.vocab_size):
      if token_id in self._specials:
        continue
      if self._byte_offset <= token_id < self._byte_offset + 256:
        lengths[token_id] = 1
    return lengths

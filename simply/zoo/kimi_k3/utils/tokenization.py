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
r"""Kimi K3 tokenizer: the released tiktoken BPE + K3's special-token table.

The HuggingFace release `moonshotai/Kimi-K3` ships `tiktoken.model` (163,584
base64 `<token> <rank>` lines) and `tokenizer_config.json`, but no
`tokenizer.json` -- so `HuggingFaceVocab` cannot read it. Ids
163,584..163,839 are 256 special tokens (16 named by
`added_tokens_decoder`, the rest `<|reserved_token_N|>`), giving
`vocab_size == 163,840`.

The pre-tokenizer regex uses Rust character-class intersection (`&&`) and
`\p{Han}`, which only the Rust-backed `tiktoken` can execute; there is no
pure-Python fallback.

Vocab file resolution (`default_vocab_path`), in order:
  1. the `vocab_path` constructor argument;
  2. the `SIMPLY_KIMI_K3_VOCAB_PATH` environment variable;
  3. `kimi_k3_tiktoken.model`, the released file vendored beside this module
     (2.8 MB; the K3 license permits redistribution and its notice is vendored
     as `kimi_k3_tiktoken_LICENSE`).
The file is read through `epath`, so CNS, local and runfiles paths all work,
and lazily, so the id constants below cost nothing to import.
"""

import base64
from collections.abc import Iterator
import functools
import os

from etils import epath
from simply.utils import tokenization as tokenization_lib
import tiktoken

# Name under which this vocab is registered (`KimiK3ExperimentConfig.
# vocab_name`).
VOCAB_NAME = 'KimiK3'

VENDORED_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'kimi_k3_tiktoken.model'
)
VOCAB_PATH_ENV_VAR = 'SIMPLY_KIMI_K3_VOCAB_PATH'

NUM_BASE_TOKENS = 163_584
NUM_SPECIAL_TOKENS = 256
VOCAB_SIZE = NUM_BASE_TOKENS + NUM_SPECIAL_TOKENS  # 163,840

# The release chops text before handing it to tiktoken
# (`tokenization_kimi.py:_encode_text_piece`): 400k chars per call to dodge a
# pyo3 panic, and no more than 25k consecutive whitespace / non-whitespace
# characters per call (tiktoken issue #195). The cuts change BPE merges, so
# reproducing them is a wire-format requirement, not an optimization -- a long
# base64 blob or minified JSON in a tool result hits it.
MAX_ENCODE_CHARS = 400_000
MAX_CONSECUTIVE_CHARS = 25_000

# The K3 pre-tokenizer regex, verbatim from the release's
# `tokenization_kimi.py:TikTokenTokenizer.pat_str`.
PAT_STR = '|'.join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

# Special tokens (K3_ARCHITECTURE.md S10 / `tokenizer_config.json`).
BOS = '[BOS]'
EOS = '[EOS]'
END_OF_MSG = '<|end_of_msg|>'
OPEN = '<|open|>'
CLOSE = '<|close|>'
SEP = '<|sep|>'
START_HEADER_ID = '[start_header_id]'
END_HEADER_ID = '[end_header_id]'
EOT = '[EOT]'
MEDIA_BEGIN = '<|media_begin|>'
MEDIA_CONTENT = '<|media_content|>'
MEDIA_END = '<|media_end|>'
MEDIA_PAD = '<|media_pad|>'
OSAGENT_MODE = '<osagent_mode>'
UNK = '[UNK]'
PAD = '[PAD]'

BOS_ID = 163_584
EOS_ID = 163_585
# The generation stop token (`config.json` / `generation_config.json`
# eos_token_id), NOT [EOS]: K3 ends every chat message with it.
END_OF_MSG_ID = 163_586
OPEN_ID = 163_587
CLOSE_ID = 163_588
SEP_ID = 163_589
START_HEADER_ID_ID = 163_590
END_HEADER_ID_ID = 163_591
EOT_ID = 163_593
MEDIA_BEGIN_ID = 163_602
MEDIA_CONTENT_ID = 163_603
MEDIA_END_ID = 163_604
MEDIA_PAD_ID = 163_605
OSAGENT_MODE_ID = 163_649
UNK_ID = 163_838
PAD_ID = 163_839

# The 16 named entries of `tokenizer_config.json:added_tokens_decoder`; every
# other id in [NUM_BASE_TOKENS, VOCAB_SIZE) is `<|reserved_token_<id>|>`.
# TODO: cross-check these against the release's
# `tokenizer_config.json` in a test. They are transcribed rather than read at
# runtime because the ids are a wire format -- the chat format and the decode
# stop condition depend on them, so a checkpoint whose config disagrees should
# fail a test rather than silently retokenize -- but nothing currently proves
# the transcription still matches the release.
NAMED_SPECIAL_TOKENS: dict[str, int] = {
    BOS: BOS_ID,
    EOS: EOS_ID,
    END_OF_MSG: END_OF_MSG_ID,
    OPEN: OPEN_ID,
    CLOSE: CLOSE_ID,
    SEP: SEP_ID,
    START_HEADER_ID: START_HEADER_ID_ID,
    END_HEADER_ID: END_HEADER_ID_ID,
    EOT: EOT_ID,
    MEDIA_BEGIN: MEDIA_BEGIN_ID,
    MEDIA_CONTENT: MEDIA_CONTENT_ID,
    MEDIA_END: MEDIA_END_ID,
    MEDIA_PAD: MEDIA_PAD_ID,
    OSAGENT_MODE: OSAGENT_MODE_ID,
    UNK: UNK_ID,
    PAD: PAD_ID,
}


def reserved_token(token_id: int) -> str:
  return f'<|reserved_token_{token_id}|>'


def _build_special_tokens() -> dict[str, int]:
  by_id = {v: k for k, v in NAMED_SPECIAL_TOKENS.items()}
  return {
      by_id.get(i, reserved_token(i)): i
      for i in range(NUM_BASE_TOKENS, VOCAB_SIZE)
  }


# The full 256-entry special-token table, in id order.
SPECIAL_TOKENS: dict[str, int] = _build_special_tokens()


def default_vocab_path() -> str:
  """The vocab file to read when the caller names none.

  Returns:
    `$SIMPLY_KIMI_K3_VOCAB_PATH`, else the vendored copy.
  """
  return os.environ.get(VOCAB_PATH_ENV_VAR) or VENDORED_VOCAB_PATH


def _split_long_runs(text: str, max_run: int) -> Iterator[str]:
  """Splits `text` so no piece holds > `max_run` same-class chars in a row.

  Character class = whitespace vs non-whitespace. Verbatim behaviour of the
  release's `_split_whitespaces_or_nonwhitespaces`.

  Args:
    text: The text.
    max_run: The maximum run length.

  Yields:
    The pieces, in order; concatenating them rebuilds `text`.
  """
  if len(text) <= max_run:
    yield text
    return
  run_len = 0
  run_is_space = text[0].isspace() if text else False
  start = 0
  for i, char in enumerate(text):
    is_space = char.isspace()
    if run_is_space ^ is_space:
      run_len = 1
      run_is_space = is_space
    else:
      run_len += 1
      if run_len > max_run:
        yield text[start:i]
        start = i
        run_len = 1
  yield text[start:]


def load_mergeable_ranks(vocab_path: str) -> dict[bytes, int]:
  """Reads `tiktoken.model` (base64 `<token> <rank>` lines) via epath.

  Equivalent to `tiktoken.load.load_tiktoken_bpe`, which cannot read CNS.

  Args:
    vocab_path: Path to the released `tiktoken.model`.

  Returns:
    Token bytes -> BPE rank.
  """
  ranks = {}
  for line in epath.Path(vocab_path).read_bytes().splitlines():
    if not line:
      continue
    token, rank = line.split()
    ranks[base64.b64decode(token)] = int(rank)
  return ranks


class KimiK3Vocab(tokenization_lib.SimplyVocab[str]):
  """Kimi K3 vocab: the released tiktoken BPE behind Simply's vocab API.

  Attributes:
    vocab_path: Path the BPE ranks are read from.
    vocab_size: 163,840.
    bos_id: None, not `BOS_ID`: nothing in the release prepends [BOS] (the XTML
      renderer does not, and `TikTokenTokenizer.encode` has no bos flag), and
      Simply's input processors prepend `vocab.bos_id` whenever it is set
      (`utils/sampling_lib.py:BasicTextInputProcessor`), which no `LMFormat` can
      override back to "none". Consequence to know about on the data path:
      `data_lib.DatasetConfig.add_bos` defaults to True but is a silent no-op
      here. Encode `BOS` yourself if a corpus needs it.
    eos_id: [EOS]. Chat generation stops on `end_of_msg_id` instead; see
      `lm_format.KimiK3Chat.extra_eos_tokens`.
    pad_id: [PAD].
    unk_id: [UNK].
    end_of_msg_id: <|end_of_msg|>, the generation stop token.
  """

  def __init__(self, vocab_path: str | None = None):
    self.vocab_path = vocab_path or default_vocab_path()
    self.vocab_size = VOCAB_SIZE
    self.bos_id = None
    self.eos_id = EOS_ID
    self.pad_id = PAD_ID
    self.unk_id = UNK_ID
    self.end_of_msg_id = END_OF_MSG_ID

  @functools.cached_property
  def encoding(self) -> tiktoken.Encoding:
    return tiktoken.Encoding(
        name=os.path.basename(self.vocab_path),
        pat_str=PAT_STR,
        mergeable_ranks=load_mergeable_ranks(self.vocab_path),
        special_tokens=dict(SPECIAL_TOKENS),
    )

  def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
    """Encodes text to token ids.

    Args:
      text: The text.
      allow_special: If True -- the default, matching both the release's
        `TikTokenTokenizer.encode(allow_special_tokens=True)` and
        `HuggingFaceVocab`, which always recognizes added tokens -- a
        special-token spelling in `text` encodes to its special id. That is
        required for prompts rendered as strings by
        `lm_format.KimiK3Chat.format` and for Simply's `extra_eos_tokens`
        contract (each must be exactly one token), and it means corpus text
        containing e.g. `[UNK]` yields a control token on the data path, as in
        the release. Pass False for untrusted content so that a user-supplied
        `<|open|>` stays ordinary text and cannot fabricate a control token;
        `KimiK3Chat.format_tokens` does exactly this, per segment.

    Returns:
      Token ids.
    """
    token_ids: list[int] = []
    for start in range(0, len(text), MAX_ENCODE_CHARS):
      for piece in _split_long_runs(
          text[start : start + MAX_ENCODE_CHARS], MAX_CONSECUTIVE_CHARS
      ):
        token_ids.extend(
            self.encoding.encode(piece, allowed_special='all')
            if allow_special
            else self.encoding.encode(piece, disallowed_special=())
        )
    return token_ids

  def decode(self, token_ids: list[int]) -> str:
    """Decodes ids to text, replacing bytes that no longer form valid utf-8."""
    return self.encoding.decode(token_ids)

  def decode_bytes(self, token_ids: list[int]) -> bytes:
    """Decodes ids to raw bytes: exact even across a split multi-byte char."""
    return self.encoding.decode_bytes(token_ids)

  def token_byte_lengths(self) -> list[int]:
    """Per-token-id utf-8 byte lengths (specials -> 0) for the bpb metric."""
    return [
        0
        if i >= NUM_BASE_TOKENS
        else len(self.encoding.decode_single_token_bytes(i))
        for i in range(VOCAB_SIZE)
    ]


# Registration is guarded because `TokenizerRegistry.register` raises on a
# duplicate name (`utils/registry.py`), which a module reload would hit.
if VOCAB_NAME not in tokenization_lib.TokenizerRegistry.keys():
  tokenization_lib.TokenizerRegistry.register(KimiK3Vocab, name=VOCAB_NAME)

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
"""Registers the Qwen3.8 vocabulary under the name the config asks for.

The release ships a `tokenizer.json`, so there is nothing to implement: core's
`tokenization.HuggingFaceVocab` reads the released tokenizer directory as it
is. This module only says where that directory is and what the vocabulary is
called, because `Qwen38ExperimentConfig.vocab_name` names it by string and
something has to put it in the registry.

The directory is `data_lib.VOCABS_DIR/Qwen3.8` -- a copy of the released
`tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt`. It is
built from `VOCABS_DIR` rather than written out, so the OSS scrub and any
future move of Simply's vocabulary store apply here too, and
`SIMPLY_QWEN3P8_VOCAB_PATH` overrides it for a run that has the files
somewhere else (a local checkout, another cell). Everything is read through
`epath` inside `HuggingFaceVocab`, so all three work.
"""

import os

from simply import data_lib
from simply.utils import tokenization as tokenization_lib

# `Qwen38ExperimentConfig.vocab_name`, and the directory name under
# `VOCABS_DIR`. Public: renaming it invalidates every launch flag.
VOCAB_NAME = 'Qwen3.8'

VOCAB_PATH_ENV_VAR = 'SIMPLY_QWEN3P8_VOCAB_PATH'


def default_vocab_path() -> str:
  """The released tokenizer directory: `$SIMPLY_QWEN3P8_VOCAB_PATH`, or CNS."""
  return os.environ.get(VOCAB_PATH_ENV_VAR) or os.path.join(
      data_lib.VOCABS_DIR, VOCAB_NAME
  )


def qwen3p8_vocab() -> tokenization_lib.HuggingFaceVocab:
  """The registered factory: `TokenizerRegistry.get_instance` calls it bare."""
  return tokenization_lib.HuggingFaceVocab(default_vocab_path())


def register() -> None:
  """Idempotent: `TokenizerRegistry.register` raises on a duplicate name.

  `utils/registry.py` is one process-wide dict, so a module reload -- or any
  second import path into this module -- would otherwise crash at import.
  Deferring to an existing `Qwen3.8` rather than overwriting it is deliberate:
  the name is frozen, and two definitions of it is a bug to be found by the
  duplicate, not papered over here.
  """
  if VOCAB_NAME not in tokenization_lib.TokenizerRegistry.keys():
    tokenization_lib.TokenizerRegistry.register(qwen3p8_vocab, name=VOCAB_NAME)


register()

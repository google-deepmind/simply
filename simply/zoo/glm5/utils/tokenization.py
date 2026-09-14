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
"""GLM-5.2 tokenizer registration, kept inside the plugin.

Importing this module registers the `GLM-5.2` HuggingFace vocab into the shared
`TokenizerRegistry` (the same mechanism `data_lib` uses for the built-in
vocabs), so `config.vocab_name='GLM-5.2'` resolves with no edit to `data_lib`.

GLM-5.2 has no `bos_token` (it uses a `[gMASK]<sop>` document prefix instead),
so `GlmHuggingFaceVocab` tolerates a missing special-token key rather than
raising — kept plugin-local so `tokenization` needs no change.
"""

import os

from simply import data_lib
from simply.utils import tokenization

GLM5P2_VOCAB = os.path.join(data_lib.VOCABS_DIR, 'GLM-5.2')


class GlmHuggingFaceVocab(tokenization.HuggingFaceVocab):
  """HuggingFace vocab that treats a missing special-token key as absent."""

  def get_token_id(self, name: str) -> int | None:
    token = self.tokenizer_config.get(name)
    if token is None:
      return None
    if not isinstance(token, str):
      token = token['content']
    if not isinstance(token, str):
      raise ValueError(f'{token=} is not a string ({name=}).')
    return self.tokenizer.token_to_id(token)


tokenization.TokenizerRegistry.register(
    lambda: GlmHuggingFaceVocab(GLM5P2_VOCAB),
    name='GLM-5.2',
)

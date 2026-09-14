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
"""GLM-5-series `glm_moe_dsa` architecture: self-contained model plugin.

Architecture-scoped (not version-scoped): this package holds the shared
`glm_moe_dsa` model code (MLA attention + sigmoid/`noaux_tc` MoE + shared
expert); each version that shares the architecture is a config entry in
`config_lib.py` (e.g. `glm5p2()`). Importing `simply.zoo.glm5` fires all of the
plugin's registration side effects — the model (`GlmTransformerLM`), the
experiment configs (`glm5p2`), the `GlmMoeDsaFormat` checkpoint converter, the
`GlmChat` LM format, the `InterleavedRoPE` position encoding, and the `GLM-5.2`
tokenizer — so they resolve by name. Nothing GLM-specific lives in core simply;
a launcher only needs `from simply.zoo import glm5` to make GLM-5 available.
"""

from simply.zoo.glm5 import config_lib  # pylint: disable=unused-import
from simply.zoo.glm5 import model_lib  # pylint: disable=unused-import
from simply.zoo.glm5.utils import ckpt_format  # pylint: disable=unused-import
from simply.zoo.glm5.utils import lm_format  # pylint: disable=unused-import
from simply.zoo.glm5.utils import mla  # pylint: disable=unused-import
from simply.zoo.glm5.utils import tokenization  # pylint: disable=unused-import

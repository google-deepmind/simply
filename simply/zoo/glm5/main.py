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
"""`simply/main.py` with the GLM-5-series plugin registered.

Core `simply/main.py` deliberately does not link any zoo plugin, so
`--experiment_config=glm5p2` fails there with
`ValueError: Unknown name: glm5p2`. Run this module instead
(`python -m simply.zoo.glm5.main`, or point a launcher's `--binary` at it): it
imports the GLM plugin modules for their registration side effects and then
reuses core `main` verbatim, exactly as `zoo/kimi_k3/eval/decode_eval.py` does
for K3.
"""

from simply import main as simply_main
from simply.utils import experiment_helper

# Imported for their registration side effects, as `simply/main.py` does for its
# own configs: the `glm5p2` experiment config, the `GlmTransformerLM` module
# (transitively `MLAAttention` + `InterleavedRoPE` + `GlmMoeFeedForward`), the
# `GlmMoeDsaFormat` checkpoint format, the `GlmChat` LM format and the `GLM-5.2`
# vocab. `config_lib` deliberately does not import the model, so this binary is
# where they are pulled together.
from simply.zoo.glm5 import config_lib  # pylint: disable=unused-import
from simply.zoo.glm5 import model_lib  # pylint: disable=unused-import
from simply.zoo.glm5.utils import ckpt_format  # pylint: disable=unused-import
from simply.zoo.glm5.utils import lm_format  # pylint: disable=unused-import
from simply.zoo.glm5.utils import tokenization  # pylint: disable=unused-import

from absl import app

if __name__ == '__main__':
  experiment_helper.set_env_based_flags()
  app.run(simply_main.main)

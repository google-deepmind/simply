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
"""Tests for the GLM-5.2 experiment config and the param tree it sizes.

`glm5p2()` is the registered deployment; the config-structure test builds a
reduced-layer copy (all true per-tensor dims kept) and checks the abstract
param tree so the config and the model code cannot drift.
"""

import dataclasses

from absl.testing import absltest
import orbax.checkpoint as ocp
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import sharding as sharding_lib
from simply.zoo.glm5 import config_lib as glm5_config
from simply.zoo.glm5 import model_lib as glm_model


class Glm5ConfigTest(absltest.TestCase):

  def test_glm5p2_resolves(self):
    cfg = glm5_config.glm5p2()
    self.assertEqual(cfg.model_name, 'GlmTransformerLM')
    self.assertEqual(cfg.n_layers, 78)

  def test_full_config_structure(self):
    """The real GLM-5.2 config builds the expected param-tree shapes.

    Uses a reduced-layer copy of `glm5p2()` (all true per-tensor dims, expert
    count, and `first_k_dense_replace` kept) so the abstract param tree is cheap
    to build; the real depth is asserted separately below.
    """
    self.assertEqual(glm5_config.glm5p2().n_layers, 78)
    n_layers = 5  # dense blocks 0-2, MoE blocks 3-4 (first_k_dense_replace=3).
    cfg = dataclasses.replace(glm5_config.glm5p2(), n_layers=n_layers)
    sharding_lib.set_default_mesh_shape(
        mesh_shape=(1, 1, 1, 1),
        axis_names=cfg.sharding_config.mesh_axis_names,
    )
    model = glm_model.GlmTransformerLM(cfg)
    abstract = common.get_raw_arrays(ckpt_lib.get_abstract_params(model))
    flat = ocp.tree.to_flat_dict(abstract, sep='/')
    # Keys may or may not carry a leading 'params/'; normalize by suffix match.
    by_suffix = {}
    for key, val in flat.items():
      by_suffix[key] = val
      by_suffix[key.removeprefix('params/')] = val

    def shape(key):
      return tuple(by_suffix[key].shape)

    # Embedding / output / final norm (untied).
    self.assertEqual(shape('embed_linear/embed'), (154880, 6144))
    self.assertEqual(shape('embed_linear/w'), (154880, 6144))
    self.assertEqual(shape('final_ln/scale'), (6144,))
    # MLA block shapes (block 0).
    self.assertEqual(shape('block_0/attn/q_a_proj/w'), (6144, 2048))
    self.assertEqual(shape('block_0/attn/q_a_layernorm/scale'), (2048,))
    self.assertEqual(shape('block_0/attn/q_b_proj/w'), (2048, 64, 256))
    self.assertEqual(shape('block_0/attn/kv_a_proj/w'), (6144, 576))
    self.assertEqual(shape('block_0/attn/kv_a_layernorm/scale'), (512,))
    self.assertEqual(shape('block_0/attn/kv_b_proj/w'), (512, 64, 448))
    self.assertEqual(shape('block_0/attn/o_proj/w'), (6144, 64, 256))
    # Dense FFN on the first 3 layers (intermediate=12288).
    self.assertEqual(shape('block_0/ffn/ffn_0/w'), (6144, 12288))
    self.assertEqual(shape('block_2/ffn/ffn_1/w'), (12288, 6144))
    self.assertNotIn('block_0/ffn/router/w', by_suffix)
    # MoE on layer 3+ (256 experts, moe_inter=2048, shared expert + bias).
    self.assertEqual(shape('block_3/ffn/router/w'), (6144, 256))
    self.assertEqual(shape('block_3/ffn/router_correction_bias'), (256,))
    self.assertEqual(shape('block_3/ffn/ffn_0/w'), (256, 6144, 2048))
    self.assertEqual(shape('block_3/ffn/ffn_0_gate/w'), (256, 6144, 2048))
    self.assertEqual(shape('block_3/ffn/ffn_1/w'), (256, 2048, 6144))
    self.assertEqual(shape('block_3/ffn/shared_expert/ffn_0/w'), (6144, 2048))
    self.assertEqual(shape('block_3/ffn/shared_expert/ffn_1/w'), (2048, 6144))
    # Last block is MoE; there is no extra block after it (the HF MTP layer at
    # index num_hidden_layers is dropped on load).
    self.assertIn(f'block_{n_layers - 1}/ffn/router/w', by_suffix)
    self.assertNotIn(f'block_{n_layers}/ffn/router/w', by_suffix)


if __name__ == '__main__':
  absltest.main()

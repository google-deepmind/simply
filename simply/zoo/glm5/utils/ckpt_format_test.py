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
"""HuggingFace -> Simply conversion equivalence for GLM-5.2 (`glm_moe_dsa`).

Generates random HuggingFace-format `glm_moe_dsa` weights, then:
  (a) runs the validated standalone JAX reference (`utils/test_utils.py`), and
  (b) converts the same weights via `GlmMoeDsaFormat` into a Simply
      `GlmTransformerLM` (MLA + sigmoid/noaux_tc MoE + shared expert), runs its
      forward, and asserts the logits match the reference.

The reference (`utils/test_utils.py`) has been validated bit-for-bit (to fp32
rounding, max-abs logit diff ~7e-5) against the real HuggingFace
`GlmMoeDsaForCausalLM` modeling code.
"""

import dataclasses

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import sharding as sharding_lib
from simply.zoo.glm5 import config_lib as glm5_config
from simply.zoo.glm5 import model_lib as glm_model
from simply.zoo.glm5.utils import ckpt_format as glm5_checkpoint
from simply.zoo.glm5.utils import test_utils as ref


def _tiny_cfg() -> ref.Cfg:
  return ref.Cfg(
      hidden=16,
      n_layers=4,
      n_heads=2,
      q_lora_rank=8,
      kv_lora_rank=6,
      qk_nope=12,
      qk_rope=8,
      v_head=20,
      n_routed=4,
      n_shared=1,
      topk=2,
      moe_inter=10,
      inter=14,
      first_k_dense=1,
      routed_scaling=2.5,
      rope_theta=8e6,
      eps=1e-5,
      vocab=10,
  )


def _simply_cfg(c: ref.Cfg):
  return dataclasses.replace(
      glm5_config.glm5p2(),
      model_dim=c.hidden,
      n_heads=c.n_heads,
      n_kv_heads=c.n_heads,
      per_head_dim=c.qk_head,
      n_layers=c.n_layers,
      vocab_size=c.vocab,
      mla_q_lora_rank=c.q_lora_rank,
      mla_kv_lora_rank=c.kv_lora_rank,
      mla_qk_nope_head_dim=c.qk_nope,
      mla_qk_rope_head_dim=c.qk_rope,
      mla_v_head_dim=c.v_head,
      num_experts=c.n_routed,
      num_experts_per_token=c.topk,
      ffn_expand_dim=c.moe_inter,
      dense_ffn_expand_dim=c.inter,
      first_k_dense_replace=c.first_k_dense,
      num_shared_experts=c.n_shared,
      routed_scaling_factor=c.routed_scaling,
      rms_norm_epsilon=c.eps,
      query_scale=float(np.sqrt(c.qk_head)),
      activation_dtype_name='float32',
      gmm_impl='ragged_dot',
  )


def _ref_params_to_hf(c: ref.Cfg, p) -> dict[str, jax.Array]:
  """Maps `glm5_reference` param layout to HuggingFace flat keys."""
  hf = {}
  hf['model.embed_tokens.weight'] = p['embed']
  hf['lm_head.weight'] = p['lm_head']
  hf['model.norm.weight'] = p['final_norm']
  for i, layer in enumerate(p['layers']):
    pre = f'model.layers.{i}.'
    hf[pre + 'input_layernorm.weight'] = layer['input_ln']
    hf[pre + 'post_attention_layernorm.weight'] = layer['post_ln']
    hf[pre + 'self_attn.q_a_proj.weight'] = layer['q_a']
    hf[pre + 'self_attn.q_a_layernorm.weight'] = layer['q_a_ln']
    hf[pre + 'self_attn.q_b_proj.weight'] = layer['q_b']
    hf[pre + 'self_attn.kv_a_proj_with_mqa.weight'] = layer['kv_a']
    hf[pre + 'self_attn.kv_a_layernorm.weight'] = layer['kv_a_ln']
    hf[pre + 'self_attn.kv_b_proj.weight'] = layer['kv_b']
    hf[pre + 'self_attn.o_proj.weight'] = layer['o']
    if i < c.first_k_dense:
      hf[pre + 'mlp.gate_proj.weight'] = layer['gate_proj']
      hf[pre + 'mlp.up_proj.weight'] = layer['up_proj']
      hf[pre + 'mlp.down_proj.weight'] = layer['down_proj']
    else:
      hf[pre + 'mlp.gate.weight'] = layer['router']
      hf[pre + 'mlp.gate.e_score_correction_bias'] = layer['e_bias']
      for e in range(c.n_routed):
        hf[pre + f'mlp.experts.{e}.gate_proj.weight'] = layer['ex_gate'][e]
        hf[pre + f'mlp.experts.{e}.up_proj.weight'] = layer['ex_up'][e]
        hf[pre + f'mlp.experts.{e}.down_proj.weight'] = layer['ex_down'][e]
      hf[pre + 'mlp.shared_experts.gate_proj.weight'] = layer['sh_gate']
      hf[pre + 'mlp.shared_experts.up_proj.weight'] = layer['sh_up']
      hf[pre + 'mlp.shared_experts.down_proj.weight'] = layer['sh_down']
  return hf


class GlmMoeDsaFormatTest(absltest.TestCase):

  def test_forward_matches_reference(self):
    c = _tiny_cfg()
    cfg = _simply_cfg(c)
    sharding_lib.set_default_mesh_shape(
        mesh_shape=(1, 1, 1, 1),
        axis_names=cfg.sharding_config.mesh_axis_names,
    )

    # 1. Random weights in the reference layout (float32).
    p = ref.init_params(c, jax.random.PRNGKey(0), dtype=jnp.float32)
    tokens = jnp.array([[1, 2, 3, 4, 5, 6]])
    ref_logits = ref.forward(c, p, tokens)

    # 2. Save the equivalent HF-format weights and load via GlmMoeDsaFormat.
    hf_state = _ref_params_to_hf(c, p)
    ckpt_dir = self.create_tempdir().full_path
    mngr = ocp.CheckpointManager(ckpt_dir)
    ckpt_lib.save_checkpoint(
        mngr, hf_state, 0, ckpt_format=glm5_checkpoint.GlmMoeDsaFormat()
    )
    mngr.wait_until_finished()

    model = glm_model.GlmTransformerLM(cfg)
    abstract_state = {'params': ckpt_lib.get_abstract_params(model)}
    restored = ckpt_lib.load_checkpoint_from_dir(ckpt_dir, abstract_state)
    restored = common.get_raw_arrays(restored)

    # 3. Run the Simply model forward and compare logits.
    out = model.apply(
        restored['params'],  # pyrefly: ignore[bad-index, unsupported-operation]
        tokens,
        segment_ids=jnp.ones_like(tokens),
        segment_positions=jnp.broadcast_to(
            jnp.arange(tokens.shape[1]), tokens.shape
        ),
    )
    simply_logits = out[0] if isinstance(out, tuple) else out
    simply_logits = np.asarray(simply_logits, dtype=np.float32)
    ref_logits = np.asarray(ref_logits, dtype=np.float32)
    self.assertEqual(simply_logits.shape, ref_logits.shape)
    max_diff = float(np.max(np.abs(simply_logits - ref_logits)))
    self.assertLess(max_diff, 1e-3, msg=f'max abs logit diff = {max_diff}')


if __name__ == '__main__':
  absltest.main()

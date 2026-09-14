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
"""GLM-5.2 (`glm_moe_dsa`) HuggingFace -> Orbax checkpoint converter.

`GlmMoeDsaFormat` is a `CheckpointFormat` (registered) that maps a raw
HF-keyed GLM-5.2 checkpoint into Simply's MLA + GLM-MoE param layout. Kept in
the plugin dir; wired into the core loader purely via
`CheckpointFormatRegistry`, so no core file references GLM.
"""

from collections.abc import Mapping
import dataclasses
import logging
import re

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from simply.utils import checkpoint_lib
from simply.utils import common
from simply.utils import pytree

Array = common.Array
PyTree = common.PyTree
CheckpointFormat = checkpoint_lib.CheckpointFormat
CheckpointFormatRegistry = checkpoint_lib.CheckpointFormatRegistry


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class GlmMoeDsaFormat(CheckpointFormat):
  """GLM-5 / GLM-5.1 / GLM-5.2 (`glm_moe_dsa`) HuggingFace checkpoint format.

  Converts a HuggingFace `GlmMoeDsaForCausalLM` checkpoint (DeepSeek-V3-style
  MLA + sigmoid/noaux_tc MoE with a shared expert) into the Simply param tree
  produced by `model_lib.MLAAttention` + `model_lib.MoEFeedForward` (with
  `router_score_func='sigmoid'`, `router_use_correction_bias=True`,
  `num_shared_experts=1`) and per-layer dense/MoE blocks
  (`first_k_dense_replace=3`).

  HuggingFace Linear convention is `weight=[out, in]`, `y = x @ W.T`; Simply's
  `EinsumLinear` stores the weight in `[in, ...out]` layout, so most projections
  are transposed and reshaped to per-head form.

  The DSA indexer weights (`self_attn.indexer.*`) and the MTP layer (index
  `num_hidden_layers`) are ignored, matching HF (the indexer is a no-op for
  context <= index_topk; the MTP layer is dropped on load).
  """

  def _stack_experts(
      self, flatten_stored_state: Mapping[str, jax.Array], pattern: str
  ) -> jax.Array:
    """Stacks per-expert weights into [num_experts, out, in] (HF layout)."""
    experts = []
    for k, v in flatten_stored_state.items():
      if m := re.fullmatch(pattern, k):
        experts.append((int(m.group(1)), v))
    _, experts = zip(*sorted(experts, key=lambda x: x[0]))
    return jnp.stack(experts, axis=0)

  def transforms(
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    flatten_stored_state = ocp.tree.to_flat_dict(stored_state, sep='/')
    # Infer per-head dims from the target abstract state.
    q_b_w = pytree.tree_value(
        target_abstract_state, 'params/block_0/attn/q_b_proj/w'
    )
    # q_b_proj/w has shape [q_lora_rank, n_heads, qk_head_dim].
    n_heads = q_b_w.shape[1]  # pyrefly: ignore[missing-attribute]
    qk_head_dim = q_b_w.shape[2]  # pyrefly: ignore[missing-attribute]
    kv_b_w = pytree.tree_value(
        target_abstract_state, 'params/block_0/attn/kv_b_proj/w'
    )
    # kv_b_proj/w has shape [kv_lora_rank, n_heads, qk_nope + v_head].
    nope_plus_v = kv_b_w.shape[2]  # pyrefly: ignore[missing-attribute]
    o_w = pytree.tree_value(
        target_abstract_state, 'params/block_0/attn/o_proj/w'
    )
    # o_proj/w has shape [model_dim, n_heads, v_head_dim].
    v_head_dim = o_w.shape[2]  # pyrefly: ignore[missing-attribute]
    qk_nope_head_dim = nope_plus_v - v_head_dim

    abstract_keys = _abstract_keys(target_abstract_state)
    transformed_state = {}
    for k, v in flatten_stored_state.items():
      # Drop the MTP layer (index == num_hidden_layers): any layer whose Simply
      # block does not exist in the target abstract state (e.g. layer 78).
      if lm := re.match(r'model\.layers\.(\d+)\.', k):
        layer = int(lm.group(1))
        if abstract_keys and not any(
            kk.startswith(f'params/block_{layer}/') for kk in abstract_keys
        ):
          continue
      if k == 'model.embed_tokens.weight':
        transformed_state['params/embed_linear/embed'] = v
      elif k == 'lm_head.weight':
        # Output layer weight uses eqn 'vd,...d->...v' => shape [vocab, dim],
        # which is exactly HF `lm_head.weight` ([vocab, hidden]); no transpose.
        transformed_state['params/embed_linear/w'] = v
      elif k == 'model.norm.weight':
        transformed_state['params/final_ln/scale'] = v
      elif m := re.fullmatch(r'model.layers.(\d+).input_layernorm.weight', k):
        transformed_state[f'params/block_{m.group(1)}/pre_ln_0/scale'] = v
      elif m := re.fullmatch(
          r'model.layers.(\d+).post_attention_layernorm.weight', k
      ):
        transformed_state[f'params/block_{m.group(1)}/pre_ln_1/scale'] = v
      # --- MLA attention ---
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.q_a_proj.weight', k
      ):
        # HF [q_lora, model_dim] -> Simply [model_dim, q_lora].
        transformed_state[f'params/block_{m.group(1)}/attn/q_a_proj/w'] = (
            jnp.transpose(v)
        )
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.q_a_layernorm.weight', k
      ):
        transformed_state[
            f'params/block_{m.group(1)}/attn/q_a_layernorm/scale'
        ] = v
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.q_b_proj.weight', k
      ):
        # HF [n_heads*qk_head, q_lora] -> Simply [q_lora, n_heads, qk_head].
        w = jnp.transpose(v)  # [q_lora, n_heads*qk_head]
        w = w.reshape(w.shape[0], n_heads, qk_head_dim)
        transformed_state[f'params/block_{m.group(1)}/attn/q_b_proj/w'] = w
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.kv_a_proj_with_mqa.weight', k
      ):
        # HF [kv_lora+rope, model_dim] -> Simply [model_dim, kv_lora+rope].
        transformed_state[f'params/block_{m.group(1)}/attn/kv_a_proj/w'] = (
            jnp.transpose(v)
        )
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.kv_a_layernorm.weight', k
      ):
        transformed_state[
            f'params/block_{m.group(1)}/attn/kv_a_layernorm/scale'
        ] = v
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.kv_b_proj.weight', k
      ):
        # HF [n_heads*(nope+v), kv_lora] -> Simply [kv_lora, n_heads, nope+v].
        w = jnp.transpose(v)  # [kv_lora, n_heads*(nope+v)]
        w = w.reshape(w.shape[0], n_heads, qk_nope_head_dim + v_head_dim)
        transformed_state[f'params/block_{m.group(1)}/attn/kv_b_proj/w'] = w
      elif m := re.fullmatch(r'model.layers.(\d+).self_attn.o_proj.weight', k):
        # HF [model_dim, n_heads*v_head] -> Simply [model_dim, n_heads, v_head].
        w = v.reshape(v.shape[0], n_heads, v_head_dim)
        transformed_state[f'params/block_{m.group(1)}/attn/o_proj/w'] = w
      elif re.fullmatch(r'model.layers.(\d+).self_attn.indexer\..*', k):
        continue  # DSA indexer not modeled (no-op for context <= index_topk).
      # --- Dense MLP (layers 0..first_k_dense-1) ---
      elif m := re.fullmatch(r'model.layers.(\d+).mlp.gate_proj.weight', k):
        transformed_state[f'params/block_{m.group(1)}/ffn/ffn_0_gate/w'] = (
            jnp.transpose(v)
        )
      elif m := re.fullmatch(r'model.layers.(\d+).mlp.up_proj.weight', k):
        transformed_state[f'params/block_{m.group(1)}/ffn/ffn_0/w'] = (
            jnp.transpose(v)
        )
      elif m := re.fullmatch(r'model.layers.(\d+).mlp.down_proj.weight', k):
        transformed_state[f'params/block_{m.group(1)}/ffn/ffn_1/w'] = (
            jnp.transpose(v)
        )
      # --- MoE router ---
      elif m := re.fullmatch(r'model.layers.(\d+).mlp.gate.weight', k):
        # HF [num_experts, model_dim] -> Simply router [model_dim, num_experts].
        transformed_state[f'params/block_{m.group(1)}/ffn/router/w'] = (
            jnp.transpose(v)
        )
      elif m := re.fullmatch(
          r'model.layers.(\d+).mlp.gate.e_score_correction_bias', k
      ):
        transformed_state[
            f'params/block_{m.group(1)}/ffn/router_correction_bias'
        ] = v
      # --- MoE routed experts (stack per-expert into [E, in, out]) ---
      elif m := re.fullmatch(
          r'model.layers.(\d+).mlp.experts.(\d+).gate_proj.weight', k
      ):
        if m.group(2) == '0':
          w = self._stack_experts(
              flatten_stored_state,
              rf'model.layers.{m.group(1)}.mlp.experts.(\d+).gate_proj.weight',
          )  # [E, moe_inter, model_dim]
          transformed_state[f'params/block_{m.group(1)}/ffn/ffn_0_gate/w'] = (
              jnp.einsum('eoi->eio', w)
          )
      elif m := re.fullmatch(
          r'model.layers.(\d+).mlp.experts.(\d+).up_proj.weight', k
      ):
        if m.group(2) == '0':
          w = self._stack_experts(
              flatten_stored_state,
              rf'model.layers.{m.group(1)}.mlp.experts.(\d+).up_proj.weight',
          )  # [E, moe_inter, model_dim]
          transformed_state[f'params/block_{m.group(1)}/ffn/ffn_0/w'] = (
              jnp.einsum('eoi->eio', w)
          )
      elif m := re.fullmatch(
          r'model.layers.(\d+).mlp.experts.(\d+).down_proj.weight', k
      ):
        if m.group(2) == '0':
          w = self._stack_experts(
              flatten_stored_state,
              rf'model.layers.{m.group(1)}.mlp.experts.(\d+).down_proj.weight',
          )  # [E, model_dim, moe_inter]
          transformed_state[f'params/block_{m.group(1)}/ffn/ffn_1/w'] = (
              jnp.einsum('eoi->eio', w)
          )
      # --- MoE shared expert ---
      elif m := re.fullmatch(
          r'model.layers.(\d+).mlp.shared_experts.gate_proj.weight', k
      ):
        transformed_state[
            f'params/block_{m.group(1)}/ffn/shared_expert/ffn_0_gate/w'
        ] = jnp.transpose(v)
      elif m := re.fullmatch(
          r'model.layers.(\d+).mlp.shared_experts.up_proj.weight', k
      ):
        transformed_state[
            f'params/block_{m.group(1)}/ffn/shared_expert/ffn_0/w'
        ] = jnp.transpose(v)
      elif m := re.fullmatch(
          r'model.layers.(\d+).mlp.shared_experts.down_proj.weight', k
      ):
        transformed_state[
            f'params/block_{m.group(1)}/ffn/shared_expert/ffn_1/w'
        ] = jnp.transpose(v)
      else:
        logging.warning('stored_state[%s] is ignored by %s', k, self.__class__)
    return ocp.tree.from_flat_dict(transformed_state, sep='/')


def _abstract_keys(abstract_state: PyTree) -> set[str]:
  if abstract_state is None:
    return set()
  return set(ocp.tree.to_flat_dict(abstract_state, sep='/').keys())

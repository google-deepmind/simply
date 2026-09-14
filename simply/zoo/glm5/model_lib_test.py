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
"""Model-level parity tests for the assembled GLM-5.2 TransformerLM.

`MlaLatentFullModelTest` runs the tiny `glm5p2` end-to-end: cached step-by-step
decode == full-sequence prefill (with the compact-latent KV cache), and a short
training step (finite grads, loss non-increasing).
"""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply import config_lib
from simply import model_lib
from simply.utils import sharding as sharding_lib
from simply.zoo.glm5 import config_lib as glm5_config
from simply.zoo.glm5 import model_lib as glm_model


class MlaLatentFullModelTest(parameterized.TestCase):
  """End-to-end: tiny glm5p2 TransformerLM with the latent KV cache."""

  def _tiny_config(self, latent):
    cfg = glm5_config.glm5p2()
    cfg = dataclasses.replace(
        cfg,
        model_dim=16,
        n_heads=2,
        n_kv_heads=2,
        per_head_dim=20,  # qk_head = nope12 + rope8
        n_layers=2,
        vocab_size=10,
        mla_q_lora_rank=8,
        mla_kv_lora_rank=6,
        mla_qk_nope_head_dim=12,
        mla_qk_rope_head_dim=8,
        mla_v_head_dim=20,
        num_experts=4,
        num_experts_per_token=2,
        ffn_expand_dim=10,
        dense_ffn_expand_dim=14,
        first_k_dense_replace=1,
        num_shared_experts=1,
        activation_dtype_name='float32',
        gmm_impl='ragged_dot',
        use_scan=False,
        use_remat=False,
        mla_use_latent_kv_cache=latent,
        sharding_config=config_lib.moe_sharding(),
        batch_size=1,
    )
    sharding_lib.set_default_mesh_shape(
        mesh_shape=(1, 1, 1, 1),
        axis_names=cfg.sharding_config.mesh_axis_names,
    )
    return cfg

  def test_full_model_decode_matches_prefill(self):
    cfg = self._tiny_config(latent=True)
    model = glm_model.GlmTransformerLM(cfg)
    params = model.init(jax.random.key(0))
    tokens = np.array([[1, 3, 5, 2, 7, 4]], dtype=np.int32)
    seq = tokens.shape[1]
    # Full prefill (materialized reference model, same params).
    ref_cfg = dataclasses.replace(cfg, mla_use_latent_kv_cache=False)
    ref_model = glm_model.GlmTransformerLM(ref_cfg)
    seg_pos = np.arange(seq)[None]
    full_logits, _ = jax.jit(ref_model.apply)(
        params, tokens, segment_positions=seg_pos
    )
    full_logits = np.asarray(full_logits, np.float32)

    # Step-by-step latent decode: pre-allocate a full-length decode cache and
    # write one token per step (avoids pad_decode_state_to, which assumes the
    # legacy per-block dict layout).
    apply_fn = jax.jit(model.apply)
    decode_state = model.init_decode_state(seq)
    outs = []
    for i in range(seq):
      li, extra = apply_fn(
          params,
          tokens[:, i : i + 1],
          segment_positions=np.array([[i]]),
          decode_state=decode_state,
      )
      decode_state = extra['decode_state']
      outs.append(np.asarray(li, np.float32))
    decoded = np.concatenate(outs, axis=1)
    diff = float(np.max(np.abs(decoded - full_logits)))
    self.assertLess(diff, 2e-3, msg=f'full-model decode vs prefill diff={diff}')

  @parameterized.parameters({'latent': False}, {'latent': True})
  def test_full_model_training_step(self, latent):
    """Tiny glm5p2 fwd+bwd: finite grads + loss decreases over 2 SGD steps."""
    # Proves the MLA change keeps the training path working. With latent=True
    # the differentiable prefill (empty-cache) absorption path is exercised.
    cfg = self._tiny_config(latent=latent)
    cfg = dataclasses.replace(cfg, batch_size=2)
    model = glm_model.GlmTransformerLM(cfg)
    params = model.init(jax.random.key(0))
    seq = 8
    rng = np.random.default_rng(0)
    toks = rng.integers(0, cfg.vocab_size, size=(2, seq + 1))
    batch = {
        'decoder_input_tokens': jnp.asarray(toks[:, :-1], jnp.int32),
        'decoder_target_tokens': jnp.asarray(toks[:, 1:], jnp.int32),
        'decoder_segment_ids': jnp.ones((2, seq), jnp.int32),
        'decoder_positions': jnp.tile(jnp.arange(seq)[None], (2, 1)),
    }

    def loss_fn(p):
      loss, _ = model_lib.compute_train_loss(model, p, batch)
      return loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    losses = []
    lr = 0.1
    for _ in range(3):
      loss, grads = grad_fn(params)
      losses.append(float(loss))
      # All grads finite.
      for g in jax.tree_util.tree_leaves(grads):
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(g)))))
      params = jax.tree_util.tree_map(lambda a, b: a - lr * b, params, grads)
    self.assertTrue(np.isfinite(losses[0]))
    # Loss should not increase over the steps (allow tiny slack).
    self.assertLessEqual(
        losses[-1],
        losses[0] + 1e-3,
        msg=f'loss did not decrease: {losses} (latent={latent})',
    )


if __name__ == '__main__':
  absltest.main()

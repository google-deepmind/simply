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
"""Parity tests for the efficient compact/latent MLA KV cache decode.

Verifies (attention module level):
  * latent absorption attention == materialized (per-head) MLA attention
    (full-sequence / training-prefill path), to tight tolerance.
  * flash-tiled latent attention == the dense single-softmax latent attention,
    across block sizes (the efficient-decode correctness guarantee).
  * cached step-by-step latent decode == full-sequence prefill (decode-vs-
    prefill parity).
  * training smoke: forward+backward through the latent path yields finite
    grads for every MLA parameter.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply import model_lib
from simply.zoo.glm5.utils import mla as mla_lib


def _make_mla(**overrides):
  cfg = dict(
      model_dim=16,
      n_heads=3,
      q_lora_rank=8,
      kv_lora_rank=6,
      qk_nope_head_dim=12,
      qk_rope_head_dim=8,
      v_head_dim=20,
      rms_norm_epsilon=1e-6,
      norm_scale_plus_one=False,
      activation_dtype='float32',
      weight_dtype='float32',
      query_scale=-1.0,
      position_encoding=mla_lib.InterleavedRoPE(max_timescale=8_000_000),
  )
  cfg.update(overrides)
  return mla_lib.MLAAttention(**cfg)


class MlaLatentParityTest(parameterized.TestCase):

  def _run_full(self, mla, params, x, seg_ids, seg_pos):
    out, _ = mla.apply(
        params, x, segment_ids=seg_ids, segment_positions=seg_pos
    )
    return np.asarray(out, np.float32)

  def test_absorption_matches_materialized_prefill(self):
    """Latent path (no cache) == materialized path, full sequence."""
    b, s = 2, 7
    mat = _make_mla(use_latent_kv_cache=False)
    lat = _make_mla(use_latent_kv_cache=True)
    params = mat.init(jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (b, s, 16))
    seg_ids = jnp.ones((b, s), jnp.int32)
    seg_pos = jnp.tile(jnp.arange(s)[None], (b, 1))
    out_mat = self._run_full(mat, params, x, seg_ids, seg_pos)
    out_lat = self._run_full(lat, params, x, seg_ids, seg_pos)
    diff = float(np.max(np.abs(out_mat - out_lat)))
    self.assertLess(diff, 1e-4, msg=f'absorb vs materialized diff={diff}')

  def test_decode_matches_prefill(self):
    """Step-by-step latent decode == full prefill (same tokens)."""
    b, s = 2, 6
    mla = _make_mla(use_latent_kv_cache=True)
    params = mla.init(jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (b, s, 16))
    seg_ids = jnp.ones((b, s), jnp.int32)
    seg_pos = jnp.tile(jnp.arange(s)[None], (b, 1))
    # Full prefill reference (materialized, exact).
    ref_mla = _make_mla(use_latent_kv_cache=False)
    full = self._run_full(ref_mla, params, x, seg_ids, seg_pos)

    apply_fn = jax.jit(mla.apply)
    decode_state = mla.init_decode_state(b, s)
    outs = []
    for i in range(s):
      xi = x[:, i : i + 1, :]
      pi = jnp.full((b, 1), i, jnp.int32)
      si = jnp.ones((b, 1), jnp.int32)
      out_i, extra = apply_fn(
          params,
          xi,
          segment_ids=si,
          segment_positions=pi,
          decode_state=decode_state,
      )
      decode_state = extra['decode_state']
      outs.append(np.asarray(out_i, np.float32))
    decoded = np.concatenate(outs, axis=1)
    diff = float(np.max(np.abs(decoded - full)))
    self.assertLess(diff, 1e-4, msg=f'decode vs prefill diff={diff}')

  @parameterized.parameters({'block_k': 4}, {'block_k': 8}, {'block_k': 512})
  def test_flash_latent_attention_matches_dense(self, block_k):
    """Flash latent attention == dense _latent_attention across block sizes."""
    # The flash decode path (_latent_attention_flash, the default decode
    # attention) tiles the sk axis in block_k blocks with an online
    # running-max/sum softmax so intermediates stay O(block_k) not O(ctx). This
    # asserts it is numerically identical to the dense single-softmax
    # _latent_attention (block_k=4 = many tiny blocks, exercising rescaling).
    mla = _make_mla(use_latent_kv_cache=True)
    # Unwrap AnnotatedArray params to raw arrays (as MLAAttention.apply does
    # before calling _latent_attention / _kv_b_split).
    params = model_lib.get_raw_arrays(mla.init(jax.random.key(0)))
    b, sq, sk = (
        2,
        3,
        37,
    )  # sk not a multiple of block_k -> exercises padding/mask
    h = 3
    nope, rope, klora = 12, 8, 6  # match _make_mla dims
    k = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(k, 4)
    q_nope = jax.random.normal(k1, (b, sq, h, nope))
    q_rope = jax.random.normal(k2, (b, sq, h, rope))
    kv_latent = jax.random.normal(k3, (b, sk, klora))
    k_rope = jax.random.normal(k4, (b, sk, 1, rope))
    # Causal-ish boolean mask [b,1,sq,sk] with some masked keys per query.
    key_pos = jnp.arange(sk)[None, None, None, :]
    q_pos = jnp.arange(sq)[None, None, :, None] + (sk - sq)
    mask = key_pos <= q_pos  # [b?,1,sq,sk] broadcast
    mask = jnp.broadcast_to(mask, (b, 1, sq, sk))
    dense = np.asarray(
        mla._latent_attention(params, q_nope, q_rope, kv_latent, k_rope, mask),
        np.float32,
    )
    flash = np.asarray(
        mla._latent_attention_flash(
            params, q_nope, q_rope, kv_latent, k_rope, mask, block_k=block_k
        ),
        np.float32,
    )
    diff = float(np.max(np.abs(dense - flash)))
    self.assertLess(
        diff,
        1e-4,
        msg=f'flash vs dense latent-attn diff={diff} (block_k={block_k})',
    )

  def test_training_smoke_finite_grads(self):
    """Forward+backward through the latent path yields finite grads."""
    b, s = 2, 5
    mla = _make_mla(use_latent_kv_cache=True)
    params = mla.init(jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (b, s, 16))
    seg_ids = jnp.ones((b, s), jnp.int32)
    seg_pos = jnp.tile(jnp.arange(s)[None], (b, 1))

    def loss_fn(p):
      out, _ = mla.apply(p, x, segment_ids=seg_ids, segment_positions=seg_pos)
      return jnp.mean(out**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    self.assertTrue(np.isfinite(float(loss)))
    leaves = jax.tree_util.tree_leaves(grads)
    self.assertNotEmpty(leaves)
    for g in leaves:
      self.assertTrue(bool(np.all(np.isfinite(np.asarray(g)))))
    # Grads must be non-trivial (flow through kv_a/kv_b/q/rmsnorm).
    total = sum(float(np.sum(np.abs(np.asarray(g)))) for g in leaves)
    self.assertGreater(total, 0.0)


if __name__ == '__main__':
  absltest.main()

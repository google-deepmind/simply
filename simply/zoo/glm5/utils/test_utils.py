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
"""Standalone JAX reference implementation of the GLM-5.2 forward pass.

Dense MLA (indexer skipped, valid for T<=2048), GLM MoE, no MTP. HF Linear:
w=[out,in], y=x@w.T.
Used as the golden reference to validate the Simply port with random weights.
"""

import dataclasses
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class Cfg:
  """GLM-5.2 (`glm_moe_dsa`) architecture hyper-parameters."""

  hidden: int = 6144
  n_layers: int = 78
  n_heads: int = 64
  q_lora_rank: int = 2048
  kv_lora_rank: int = 512
  qk_nope: int = 192
  qk_rope: int = 64
  v_head: int = 256
  n_routed: int = 256
  n_shared: int = 1
  topk: int = 8
  moe_inter: int = 2048
  inter: int = 12288
  first_k_dense: int = 3
  routed_scaling: float = 2.5
  rope_theta: float = 8e6
  eps: float = 1e-5
  vocab: int = 154880

  @property
  def qk_head(self):
    return self.qk_nope + self.qk_rope


def rmsnorm(x, w, eps):
  x32 = x.astype(jnp.float32)
  v = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)
  return (w * v.astype(x.dtype)).astype(x.dtype)


def layernorm(x, w, b, eps=1e-6):
  x32 = x.astype(jnp.float32)
  m = jnp.mean(x32, -1, keepdims=True)
  v = jnp.mean((x32 - m) ** 2, -1, keepdims=True)
  y = (x32 - m) * jax.lax.rsqrt(v + eps)
  return (w * y.astype(x.dtype) + b).astype(x.dtype)


def lin(x, w):  # HF Linear: w=[out,in], y=x@w.T
  return x @ w.T


def silu(x):
  return x * jax.nn.sigmoid(x)


def rope_cos_sin(positions, dim, base):
  # positions: [S]; returns cos,sin of width `dim` (cat[freqs,freqs]).
  half = dim // 2
  inv_freq = base ** (-jnp.arange(0, half, dtype=jnp.float32) * 2.0 / dim)
  freqs = jnp.outer(positions.astype(jnp.float32), inv_freq)  # [S,half]
  emb = jnp.concatenate([freqs, freqs], -1)  # [S,dim]
  return jnp.cos(emb), jnp.sin(emb)


def apply_rope_interleave(x, cos, sin):
  """Applies interleaved RoPE to `x` [..., seq, dim]."""
  # cos/sin have width `dim`; only their first half (the distinct angles) is
  # used. Pairs (x_2i, x_2i+1) are rotated -> [cos-halves || sin-halves].
  dim = x.shape[-1]
  half = dim // 2
  cos_h = cos[..., :half]  # [S,32]
  sin_h = sin[..., :half]
  # broadcast cos/sin over head dim: x is [B,H,S,dim] or [B,1,S,dim]
  cos_h = cos_h[None, None]  # [1,1,S,32]
  sin_h = sin_h[None, None]
  x1 = x[..., 0::2]  # even -> [.,.,S,32]
  x2 = x[..., 1::2]  # odd
  r1 = x1 * cos_h - x2 * sin_h
  r2 = x2 * cos_h + x1 * sin_h
  return jnp.concatenate([r1, r2], -1)


def init_params(cfg: Cfg, key, dtype=jnp.float32):
  """Initializes random reference weights (HuggingFace [out, in] layout)."""
  keys = iter(jax.random.split(key, 10000))

  def rnd(*shape, scale=0.02):
    return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

  params = {}
  params['embed'] = rnd(cfg.vocab, cfg.hidden)
  params['lm_head'] = rnd(cfg.vocab, cfg.hidden)
  params['final_norm'] = rnd(cfg.hidden, scale=1.0) + 1.0
  layers = []
  for i in range(cfg.n_layers):
    layer = {}
    layer['input_ln'] = rnd(cfg.hidden, scale=0.1) + 1.0
    layer['post_ln'] = rnd(cfg.hidden, scale=0.1) + 1.0
    layer['q_a'] = rnd(cfg.q_lora_rank, cfg.hidden)
    layer['q_a_ln'] = rnd(cfg.q_lora_rank, scale=0.1) + 1.0
    layer['q_b'] = rnd(cfg.n_heads * cfg.qk_head, cfg.q_lora_rank)
    layer['kv_a'] = rnd(cfg.kv_lora_rank + cfg.qk_rope, cfg.hidden)
    layer['kv_a_ln'] = rnd(cfg.kv_lora_rank, scale=0.1) + 1.0
    layer['kv_b'] = rnd(
        cfg.n_heads * (cfg.qk_nope + cfg.v_head), cfg.kv_lora_rank
    )
    layer['o'] = rnd(cfg.hidden, cfg.n_heads * cfg.v_head)
    if i < cfg.first_k_dense:
      layer['gate_proj'] = rnd(cfg.inter, cfg.hidden)
      layer['up_proj'] = rnd(cfg.inter, cfg.hidden)
      layer['down_proj'] = rnd(cfg.hidden, cfg.inter)
    else:
      layer['router'] = rnd(cfg.n_routed, cfg.hidden)
      layer['e_bias'] = rnd(cfg.n_routed, scale=0.1)
      layer['ex_gate'] = rnd(cfg.n_routed, cfg.moe_inter, cfg.hidden)
      layer['ex_up'] = rnd(cfg.n_routed, cfg.moe_inter, cfg.hidden)
      layer['ex_down'] = rnd(cfg.n_routed, cfg.hidden, cfg.moe_inter)
      layer['sh_gate'] = rnd(cfg.moe_inter, cfg.hidden)
      layer['sh_up'] = rnd(cfg.moe_inter, cfg.hidden)
      layer['sh_down'] = rnd(cfg.hidden, cfg.moe_inter)
    layers.append(layer)
  params['layers'] = layers
  return params


def mla(cfg, layer, x, cos, sin):
  """Dense Multi-head Latent Attention for one layer."""
  bsz, seq, _ = x.shape
  heads = cfg.n_heads
  q = rmsnorm(lin(x, layer['q_a']), layer['q_a_ln'], cfg.eps)  # [B,S,qlr]
  q = lin(q, layer['q_b']).reshape(bsz, seq, heads, cfg.qk_head)
  q = jnp.transpose(q, (0, 2, 1, 3))  # [B,H,S,qk_head]
  q_pass, q_rot = q[..., : cfg.qk_nope], q[..., cfg.qk_nope :]
  ckv = lin(x, layer['kv_a'])  # [B,S,kv_lora+rope]
  k_pass_c, k_rot = ckv[..., : cfg.kv_lora_rank], ckv[..., cfg.kv_lora_rank :]
  k_pass = rmsnorm(k_pass_c, layer['kv_a_ln'], cfg.eps)
  k_pass = lin(k_pass, layer['kv_b']).reshape(
      bsz, seq, heads, cfg.qk_nope + cfg.v_head
  )
  k_pass = jnp.transpose(k_pass, (0, 2, 1, 3))  # [B,H,S,nope+v]
  k_nope, value = k_pass[..., : cfg.qk_nope], k_pass[..., cfg.qk_nope :]
  k_rot = k_rot.reshape(bsz, 1, seq, cfg.qk_rope)
  q_rot = apply_rope_interleave(q_rot, cos, sin)
  k_rot = apply_rope_interleave(k_rot, cos, sin)
  k_rot = jnp.broadcast_to(k_rot, (bsz, heads, seq, cfg.qk_rope))
  query = jnp.concatenate([q_pass, q_rot], -1)  # [B,H,S,qk_head]
  key = jnp.concatenate([k_nope, k_rot], -1)  # [B,H,S,qk_head]
  scale = cfg.qk_head**-0.5
  attn = jnp.einsum('bhsd,bhtd->bhst', query, key).astype(jnp.float32) * scale
  mask = jnp.tril(jnp.ones((seq, seq), dtype=bool))
  attn = jnp.where(mask[None, None], attn, -1e30)
  attn = jax.nn.softmax(attn, -1).astype(x.dtype)
  out = jnp.einsum('bhst,bhtd->bhsd', attn, value)  # [B,H,S,v_head]
  out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bsz, seq, heads * cfg.v_head)
  return lin(out, layer['o'])


def dense_mlp(layer, x):
  """Gated SiLU dense FFN (first `first_k_dense` layers)."""
  return lin(
      silu(lin(x, layer['gate_proj'])) * lin(x, layer['up_proj']),
      layer['down_proj'],
  )


def moe(cfg, layer, x):
  """Sigmoid/noaux_tc MoE FFN with a shared expert (GLM/DeepSeek-V3)."""
  bsz, seq, dim = x.shape
  xf = x.reshape(-1, dim)
  logits = lin(xf.astype(jnp.float32), layer['router'].astype(jnp.float32))
  scores = jax.nn.sigmoid(logits)  # [N, n_routed]
  sel = scores + layer['e_bias'].astype(jnp.float32)
  _, idx = jax.lax.top_k(sel, cfg.topk)  # [N, topk]
  w = jnp.take_along_axis(scores, idx, axis=1)  # gather from un-biased sigmoid
  w = w / (jnp.sum(w, -1, keepdims=True) + 1e-20)
  w = w * cfg.routed_scaling
  # Densely compute all experts then gather the selected ones (reference).
  # ex_gate/ex_up: [E, moe_inter, hidden]; ex_down: [E, hidden, moe_inter].
  g = jnp.einsum('nh,emh->nem', xf, layer['ex_gate'])  # [N,E,moe_inter]
  u = jnp.einsum('nh,emh->nem', xf, layer['ex_up'])
  h = silu(g) * u
  yall = jnp.einsum('nem,ehm->neh', h, layer['ex_down'])  # [N,E,hidden]
  sel_y = jnp.take_along_axis(yall, idx[:, :, None], axis=1)  # [N,topk,dim]
  moe_out = jnp.sum(sel_y * w[:, :, None].astype(x.dtype), axis=1)  # [N,dim]
  shared = lin(
      silu(lin(xf, layer['sh_gate'])) * lin(xf, layer['sh_up']),
      layer['sh_down'],
  )
  return (moe_out + shared).reshape(bsz, seq, dim)


def forward(cfg, params, tokens):
  """Runs the full dense GLM-5.2 forward pass, returning logits."""
  seq = tokens.shape[1]
  x = params['embed'][tokens]
  positions = jnp.arange(seq)
  cos, sin = rope_cos_sin(positions, cfg.qk_rope, cfg.rope_theta)
  for i, layer in enumerate(params['layers']):
    x = x + mla(cfg, layer, rmsnorm(x, layer['input_ln'], cfg.eps), cos, sin)
    h = rmsnorm(x, layer['post_ln'], cfg.eps)
    x = x + (
        dense_mlp(layer, h) if i < cfg.first_k_dense else moe(cfg, layer, h)
    )
  x = rmsnorm(x, params['final_norm'], cfg.eps)
  return lin(x, params['lm_head'])

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
r"""Regenerates `k3_golden_tiny.npz`, the oracle `hf_equivalence_test` reads.

Deliberately not part of the test suite, for the same reason the fla shims it
loads are not: it runs the UNMODIFIED HuggingFace release code
(`modeling_kimi_linear.py` from `moonshotai/Kimi-K3`, commit 9f62e4e), which
needs torch and transformers >= 5 from a workstation venv. `transformers`
vendors no `kimi_linear`, so the reference cannot be built in-process the way
the other Simply ports build theirs -- hence a checked-in fixture, and hence
this script next to it.

  PYTHONPATH=~/kimi_k3_pylibs python3 gen_k3_golden.py \
      --hf_dir=/tmp/k3hf --out=k3_golden_tiny.npz

Everything is float32 on CPU over a tiny random-weight config that exercises
every K3 text feature (KDA, gated NoPE MLA, AttnRes with 2 snapshots,
LatentMoE + shared experts, SiTU-GLU, a dense layer 0). The script captures,
via forward hooks: the parameters, prefill activations layer by layer, the
`KimiDynamicCache` after prefill, and one incremental decode step with its own
activations and cache. `validate_fixture` then asserts the fixture actually
exercises each feature before it is written -- a fixture that silently stopped
routing to more than one expert would otherwise still "pass".

The fla Triton kernels are GPU-only; the pure-torch replacements come from
`run_hf_reference.py` above, whose semantics were pinned against the fla-core
v0.5.2 sources. `--config_json` is rewritten alongside the npz and is what
`config_lib.config_from_hf` reads in the test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any

import numpy as np
import torch

HF_DIR = '/tmp/k3hf'
# The pure-torch replacements for fla's GPU-only Triton kernels; pass
# --ref_tools to point at your own copy.
REF_TOOLS = os.environ.get('K3_HF_REFERENCE', 'run_hf_reference.py')
WORKDIR = '/tmp/k3golden_work'
SEED = 20260813


def load_by_path(path: str, name: str):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def boot_hf(
    hf_dir: str = HF_DIR, ref_tools: str = REF_TOOLS, workdir: str = WORKDIR
):
  """Installs the fla shims and imports the release modeling module."""
  ref = load_by_path(ref_tools, 'k3_run_hf_reference')
  ref._install_fla_stubs()  # pylint: disable=protected-access
  os.makedirs(workdir, exist_ok=True)
  mk = ref._load_kimi_package(hf_dir, workdir)  # pylint: disable=protected-access

  # transformers >= 5.14 renamed create_causal_mask's `input_embeds` kwarg and
  # dropped `cache_position` (same patch as run_hf_reference.main).
  orig = mk.create_causal_mask

  def _compat(**kw):
    if 'input_embeds' in kw:
      kw['inputs_embeds'] = kw.pop('input_embeds')
    kw.pop('cache_position', None)
    return orig(**kw)

  mk.create_causal_mask = _compat
  _patch_cache_for_transformers5(mk)
  return ref, mk


def _patch_cache_for_transformers5(mk) -> None:
  """Adds the cache API transformers >= 5.14 expects during masking.

  Args:
    mk: the release's `modeling_kimi_linear` module.

  `masking_utils._preprocess_mask_arguments` calls
  `past_key_values.get_query_offset(layer_idx)` (absent from the release's
  `KimiDynamicCache`) and calls `get_mask_sizes` with an INT query length,
  whereas the release signature takes a `cache_position` tensor. Both are
  transformers-version plumbing, not model semantics: the values match
  `DynamicCache`'s (`q_offset = get_seq_length(layer_idx)`,
  `kv_length = q_length + past_seen`, `kv_offset = 0`).
  """
  cache_cls = mk.KimiDynamicCache
  if hasattr(cache_cls, 'get_query_offset'):
    return
  cache_cls.get_query_offset = lambda self, layer_idx=0: self.get_seq_length(
      layer_idx
  )
  orig_sizes = cache_cls.get_mask_sizes

  def get_mask_sizes(self, cache_position, layer_idx):
    if isinstance(cache_position, int):
      return cache_position + self.get_seq_length(layer_idx), 0
    return orig_sizes(self, cache_position, layer_idx)

  cache_cls.get_mask_sizes = get_mask_sizes


# Every K3 text feature is switched on; sizes are the smallest that keep each
# feature meaningful. 8 layers with the release's 3 KDA : 1 MLA pattern
# (kda_layers/full_attn_layers are 1-INDEXED, as in the release config) and
# attn_res_block_size=4 -> AttnRes snapshots at layers 0 and 4 (2 blocks).
TINY_CONFIG = dict(
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=64,
    hidden_act='situ',
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    max_position_embeddings=64,
    use_cache=True,
    tie_word_embeddings=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    # MLA (NoPE, gated)
    q_lora_rank=32,
    kv_lora_rank=16,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    mla_use_nope=True,
    mla_use_output_gate=True,
    # Stable LatentMoE
    num_experts=8,
    num_experts_per_token=2,
    num_shared_experts=2,
    moe_intermediate_size=32,
    routed_expert_hidden_size=64,
    latent_moe_use_norm=True,
    moe_renormalize=True,
    moe_router_activation_func='sigmoid',
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    use_grouped_topk=True,
    num_expert_group=1,
    topk_group=1,
    topk_method='noaux_tc',
    num_nextn_predict_layers=0,
    # AttnRes
    attn_res_block_size=4,
    # KDA
    linear_attn_config=dict(
        kda_layers=[1, 2, 3, 5, 6, 7],
        full_attn_layers=[4, 8],
        head_dim=16,
        num_heads=4,
        short_conv_kernel_size=4,
        gate_lower_bound=-5.0,
        use_full_rank_gate=True,
    ),
)


def make_config(mk, overrides: dict[str, Any] | None = None):
  cfg = mk.KimiLinearConfig(**{**TINY_CONFIG, **(overrides or {})})
  cfg._attn_implementation = 'eager'  # pylint: disable=protected-access
  # transformers >= 5.14 GenerationConfig.from_model_config walks
  # get_text_config() and mistakes linear_attn_config for a sub-config.
  type(cfg).get_text_config = lambda self, **kw: self
  return cfg


def build_model(mk, cfg):
  model = mk.KimiLinearForCausalLM(cfg)
  model.eval()
  # KimiLinearModel.__init__ force-sets flash_attention_2 (GPU only); the
  # attention modules re-read the config at forward time.
  model.config._attn_implementation = 'eager'  # pylint: disable=protected-access
  return model.to(torch.float32)


def init_params(model, seed: int = SEED) -> None:
  """Deterministic, sane-magnitude fill for every parameter.

  Args:
    model: the HF model, filled in place.
    seed: the only thing the fill depends on.

  HF's `_init_weights` leaves `dt_bias` / `e_score_correction_bias` as
  `torch.empty` (uninitialized memory), so the fixture must not rely on it.
  Iterating `sorted(state_dict)` with one generator makes the fill reproducible
  from the seed alone.
  """
  gen = torch.Generator().manual_seed(seed)
  sd = model.state_dict()
  for name in sorted(sd):
    p = sd[name]
    if name.endswith('A_log'):
      v = torch.log(torch.empty_like(p).uniform_(1.0, 16.0, generator=gen))
    elif name.endswith('dt_bias'):
      v = torch.randn(p.shape, generator=gen) * 0.5
    elif name.endswith('e_score_correction_bias'):
      v = torch.randn(p.shape, generator=gen) * 0.1
    elif name.endswith('norm.weight') or name.endswith('layernorm.weight'):
      v = 1.0 + 0.05 * torch.randn(p.shape, generator=gen)
    elif 'conv1d.weight' in name:  # [D, 1, W]
      v = torch.randn(p.shape, generator=gen) * 0.5
    elif name.endswith('embed_tokens.weight'):
      v = torch.randn(p.shape, generator=gen) * 0.05
    elif p.dim() == 2:  # every nn.Linear (incl. the router + res projections)
      v = torch.randn(p.shape, generator=gen) / np.sqrt(p.shape[1])
    else:
      v = torch.randn(p.shape, generator=gen) * 0.05
    p.copy_(v.to(p.dtype))
  assert all(torch.isfinite(v).all() for v in model.state_dict().values())


def make_input_ids(cfg, batch: int = 2, total: int = 17, seed: int = SEED):
  """[batch, total] ids; column `total-1` is the incremental decode token."""
  rng = np.random.RandomState(seed)
  ids = rng.randint(0, cfg.vocab_size, size=(batch, total)).astype(np.int64)
  ids[:, 0] = cfg.bos_token_id
  return torch.from_numpy(ids)


class Capture:
  """Collects activations via forward hooks into `{key: np.ndarray}`."""

  def __init__(self):
    self.out: dict[str, np.ndarray] = {}
    self.prefix = 'act/'
    self._handles = []

  def _put(self, key, value):
    if isinstance(value, torch.Tensor):
      value = value.detach().cpu().numpy()
      if value.dtype == np.float64:
        value = value.astype(np.float32)
    self.out[self.prefix + key] = value

  def _pre(self, key):
    def hook(unused_module, args):
      self._put(key, args[0])

    return hook

  def _post(self, key):
    def hook(unused_module, unused_args, out):
      self._put(key, out)

    return hook

  def attach(self, model):
    """Registers every hook this fixture reads, and returns self."""
    inner = model.model
    h = self._handles
    h.append(inner.embed_tokens.register_forward_hook(self._post('embed')))
    h.append(inner.norm.register_forward_pre_hook(self._pre('final_attn_res')))
    h.append(inner.norm.register_forward_hook(self._post('final_hidden')))
    for i, layer in enumerate(inner.layers):
      p = f'layer{i}/'
      h.append(
          layer.input_layernorm.register_forward_pre_hook(
              self._pre(p + 'attn_in')
          )
      )
      h.append(
          layer.input_layernorm.register_forward_hook(
              self._post(p + 'attn_norm_out')
          )
      )
      h.append(
          layer.self_attn.register_forward_hook(self._post(p + 'attn_out'))
      )
      h.append(
          layer.post_attention_layernorm.register_forward_pre_hook(
              self._pre(p + 'mlp_in')
          )
      )
      h.append(
          layer.post_attention_layernorm.register_forward_hook(
              self._post(p + 'mlp_norm_out')
          )
      )
      ffn = getattr(layer, 'block_sparse_moe', None) or layer.mlp
      h.append(ffn.register_forward_hook(self._post(p + 'mlp_out')))
      if hasattr(layer, 'block_sparse_moe'):

        def router(unused_module, unused_args, out, p=p):
          self._put(p + 'router_topk_ids', out[0].to(torch.int32))
          self._put(p + 'router_topk_weights', out[1])

        h.append(layer.block_sparse_moe.gate.register_forward_hook(router))

      def layer_out(unused_module, unused_args, out, p=p):
        self._put(p + 'prefix_sum', out[0])
        self._put(p + 'block_residual', out[1])

      h.append(layer.register_forward_hook(layer_out))
    return self

  def detach(self):
    for handle in self._handles:
      handle.remove()
    self._handles = []


def dump_cache(cache, cfg, prefix: str, out: dict[str, np.ndarray]) -> None:
  """KDA conv/recurrent states and MLA k/v caches, per layer."""
  for i in range(cfg.num_hidden_layers):
    p = f'{prefix}/layer{i}/'
    if cache.conv_states[i] is not None:
      cq, ck, cv = cache.conv_states[i]
      out[p + 'conv_q'] = cq.numpy().astype(np.float32)
      out[p + 'conv_k'] = ck.numpy().astype(np.float32)
      out[p + 'conv_v'] = cv.numpy().astype(np.float32)
    if cache.recurrent_states[i] is not None:
      out[p + 'recurrent_state'] = (
          cache.recurrent_states[i].numpy().astype(np.float32)
      )
    if cache.key_cache[i] is not None:
      out[p + 'key'] = cache.key_cache[i].numpy().astype(np.float32)
      out[p + 'value'] = cache.value_cache[i].numpy().astype(np.float32)


def build(hf_dir, ref_tools, workdir):
  """Returns (ref, mk, cfg, model, input_ids)."""
  torch.set_grad_enabled(False)
  torch.manual_seed(SEED)
  ref, mk = boot_hf(hf_dir, ref_tools, workdir)
  cfg = make_config(mk)
  model = build_model(mk, cfg)
  init_params(model)
  return ref, mk, cfg, model, make_input_ids(cfg)


def run_all(mk, cfg, model, input_ids) -> dict[str, np.ndarray]:
  """Runs prefill (no cache), prefill (cache) and one decode step."""
  prefill_ids, decode_ids = input_ids[:, :-1], input_ids[:, -1:]
  data: dict[str, np.ndarray] = {}

  cap = Capture().attach(model)
  cap.prefix = 'act/'
  out = model(input_ids=prefill_ids, use_cache=False)
  cap._put('logits', out.logits)  # pylint: disable=protected-access
  data.update(cap.out)
  cap.out = {}

  # Same prefill, but building the cache; must be bit-identical.
  cache = mk.KimiDynamicCache(config=cfg)
  out_c = model(input_ids=prefill_ids, past_key_values=cache, use_cache=True)
  assert torch.equal(out.logits, out_c.logits), (
      'cached and uncached prefill disagree: '
      f'{(out.logits - out_c.logits).abs().max()}'
  )
  cap.out = {}
  dump_cache(cache, cfg, 'cache_prefill', data)

  cap.prefix = 'act_decode/'
  out_d = model(input_ids=decode_ids, past_key_values=cache, use_cache=True)
  cap._put('logits', out_d.logits)  # pylint: disable=protected-access
  data.update(cap.out)
  cap.detach()
  dump_cache(cache, cfg, 'cache_decode', data)

  data['input_ids'] = input_ids.numpy().astype(np.int32)
  data['prefill_input_ids'] = prefill_ids.numpy().astype(np.int32)
  data['decode_input_ids'] = decode_ids.numpy().astype(np.int32)
  return data


def validate_fixture(data: dict[str, np.ndarray], cfg) -> None:
  """Structural assertions: the fixture must actually exercise every feature."""
  for k, v in data.items():
    assert np.isfinite(np.asarray(v, np.float64)).all(), f'non-finite in {k}'
  bs = cfg.attn_res_block_size
  for i in range(cfg.num_hidden_layers):
    p = f'act/layer{i}/'
    # AttnRes: one snapshot per completed block, taken at layer i%bs == 0.
    assert data[p + 'block_residual'].shape[1] == i // bs + 1, (
        i,
        data[p + 'block_residual'].shape,
    )
    is_kda = cfg.is_kda_layer(i)
    assert (f'cache_prefill/layer{i}/recurrent_state' in data) == is_kda
    assert (f'cache_prefill/layer{i}/key' in data) != is_kda
    if is_kda:
      st = data[f'cache_prefill/layer{i}/recurrent_state']
      assert np.abs(st).max() > 1e-3, f'layer {i} recurrent state is ~zero'
      assert not np.array_equal(
          st, data[f'cache_decode/layer{i}/recurrent_state']
      )
    else:
      assert data[f'cache_prefill/layer{i}/key'].shape[2] == 16
      assert data[f'cache_decode/layer{i}/key'].shape[2] == 17
    if i >= cfg.first_k_dense_replace:  # MoE layer
      ids = data[p + 'router_topk_ids']
      assert ids.shape[1] == cfg.num_experts_per_token
      assert ids.min() >= 0 and ids.max() < cfg.num_experts
      assert len(np.unique(ids)) >= 2, f'layer {i} routes to one expert only'
    else:
      assert p + 'router_topk_ids' not in data, f'layer {i} should be dense'
  assert not np.array_equal(
      data['act/logits'][:, -1:], data['act_decode/logits']
  )
  _assert_attn_res_invariants(data, cfg)


def _assert_attn_res_invariants(data, cfg) -> None:
  """The AttnRes wiring documented in README.md, pinned numerically."""
  bs, h = cfg.attn_res_block_size, cfg.hidden_size
  eq = np.array_equal
  for prefix in ('act/', 'act_decode/'):
    embed = data[prefix + 'embed']
    flat = lambda x: x.reshape(-1, h)
    # layer 0: no snapshot to mix against yet, so the sublayer input is raw.
    assert eq(data[prefix + 'layer0/attn_in'], embed)
    assert eq(data[prefix + 'layer0/block_residual'][:, 0], flat(embed))
    for i in range(cfg.num_hidden_layers):
      p = f'{prefix}layer{i}/'
      # The residual stream is `prefix_sum`, reset at every snapshot layer.
      # Summation order matches the model's (fp32 addition is not associative).
      acc = data[p + 'attn_out']
      if i % bs:
        acc = data[f'{prefix}layer{i - 1}/prefix_sum'] + acc
      elif i:  # the snapshot pushed is the incoming prefix_sum
        assert eq(
            data[p + 'block_residual'][:, -1],
            flat(data[f'{prefix}layer{i - 1}/prefix_sum']),
        )
      assert eq(
          data[p + 'prefix_sum'], acc + data[p + 'mlp_out']
      ), f'{p}prefix_sum recurrence'


def param_arrays(model) -> dict[str, np.ndarray]:
  return {
      f'param/{k}': v.detach().cpu().numpy().astype(np.float32)
      for k, v in model.state_dict().items()
  }


def main():
  here = os.path.dirname(os.path.abspath(__file__))
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=os.path.join(here, 'k3_golden_tiny.npz'))
  ap.add_argument(
      '--config_json', default=os.path.join(here, 'k3_golden_tiny_config.json')
  )
  ap.add_argument('--hf_dir', default=HF_DIR)
  ap.add_argument('--ref_tools', default=REF_TOOLS)
  ap.add_argument('--workdir', default=WORKDIR)
  args = ap.parse_args()

  _, mk, cfg, model, input_ids = build(
      args.hf_dir, args.ref_tools, args.workdir
  )
  data = run_all(mk, cfg, model, input_ids)
  validate_fixture(data, cfg)
  data.update(param_arrays(model))

  cfg_doc = {
      'config_kwargs': TINY_CONFIG,  # The contract: rebuilds the config.
      'hf_config': json.loads(cfg.to_json_string()),
      'meta': {
          'seed': SEED,
          'dtype': 'float32',
          'hf_modeling_dir': args.hf_dir,
          # Basename only: where the shims live is the caller's business.
          'fla_shims_from': os.path.basename(args.ref_tools),
          'kda_layers_0indexed': [
              i for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i)
          ],
          'mla_layers_0indexed': [
              i for i in range(cfg.num_hidden_layers) if not cfg.is_kda_layer(i)
          ],
          'moe_layers_0indexed': [
              i
              for i in range(cfg.num_hidden_layers)
              if i >= cfg.first_k_dense_replace
          ],
          'attn_res_snapshot_layers_0indexed': [
              i
              for i in range(cfg.num_hidden_layers)
              if i % cfg.attn_res_block_size == 0
          ],
      },
  }
  with open(args.config_json, 'w') as f:
    json.dump(cfg_doc, f, indent=1, sort_keys=True)
  data['meta/config_json'] = np.array(json.dumps(cfg_doc, sort_keys=True))

  np.savez_compressed(args.out, **data)
  size_mb = os.path.getsize(args.out) / 1e6
  n_param = sum(v.size for k, v in data.items() if k.startswith('param/'))
  print(
      f'wrote {args.out} ({size_mb:.1f} MB, {len(data)} arrays, '
      f'{n_param} parameter values)'
  )
  print(f'wrote {args.config_json}')


if __name__ == '__main__':
  main()

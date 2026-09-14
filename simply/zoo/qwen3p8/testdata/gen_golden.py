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
r"""Regenerates `golden_tiny.npz`, the HuggingFace oracle the tests read.

Deliberately not part of the test suite: it runs the released HuggingFace
model code, which drags torch and transformers in, and this package's whole
dependency set is jax + numpy + Simply core. The only references to it are
`README.md`, `model_lib_test.py`, `utils/test_utils.py` and this docstring.

To regenerate the fixture, run it with a Python that has torch and
transformers installed (this package itself needs neither):

T=simply/zoo/qwen3p8/testdata
python $T/gen_golden.py \
    --out=$T/golden_tiny.npz --config_json=$T/golden_tiny_config.json

The oracle is `transformers.models.qwen3_5`, i.e. the released
`Qwen3_5TextConfig` + `Qwen3_5ForCausalLM` **text tower** (the vision tower is
not modelled by this port), on CPU, in float32, with
`attn_implementation='eager'` and gradients off. Everything about the run is
recorded in `golden_tiny_config.json:meta` and in `meta/config_json` inside
the npz.

The config is a tiny random-weight Qwen3.8: 8 layers on the released 3:1
schedule (GatedDeltaNet x3 then gated attention, twice), grouped-query
attention, an attention output gate, partial rotary, a 4-wide causal depthwise
conv and a chunked delta rule, at the smallest sizes that keep each of them
meaningful. `validate_fixture` asserts that the fixture really exercises them
(a recurrent state that stayed zero, or an attention cache that stopped
growing, would otherwise still "pass") before a single byte is written.

Eight layers rather than the four that would already cover both mixers: at
four, the single attention layer is the last one, so no block ever consumes an
attention layer's residual stream, nothing distinguishes "layer index" from
"index among attention layers", and a scanned stack has a single stage -- the
carry the released 64-layer model always takes would be executed by no test.

The prefill is 20 tokens because the tests load the config with
`linear_attention_chunk_size=8`: 20 = 2 full chunks + a 4-token remainder, so
both the cross-chunk carry and the ragged last chunk are exercised. Two cached
decode steps follow, because one step cannot tell "the decode state was
written correctly" from "it was read correctly once".

What this fixture deliberately does NOT pin, so that a reader does not mistake
it for a complete gate:

  * **the deployed numerics.** It is float32, and `utils/test_utils.py` loads
    it unscanned, unsharded and at `linear_attention_chunk_size=8`; the
    release runs bfloat16, scanned, sharded and at 32. A defect that only
    appears in bf16 (a rounding order in a norm, say) or only under sharding
    is invisible here -- `compare_to_hf_reference.py` on the real weights is
    what covers that shape, and it is not a test target.
  * **the mRoPE section split.** Text-only input gives all three mRoPE rows
    the same position, which makes `apply_interleaved_mrope` an identity for
    any split: the model run cannot see `mrope_section` at all, and only
    `partial_rotary_factor` (16 rotated dims, 48 passed through) is pinned by
    it. The `mrope/` arrays are the release's own answer for three DIFFERENT
    position rows, stored so `utils/rope_test.py` can pin the split against
    the release; they are not part of the model run.
  * **padded batches.** Nothing is padded and no `attention_mask` is passed,
    so segment masking is left to `model_lib_test` against the model itself.
"""

import json
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np
import torch
from transformers.models.qwen3_5 import configuration_qwen3_5 as hf_config_lib
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_modeling_lib

_HERE = os.path.dirname(os.path.abspath(__file__))

_OUT = flags.DEFINE_string(
    'out', os.path.join(_HERE, 'golden_tiny.npz'), 'Where to write the npz.'
)
_CONFIG_JSON = flags.DEFINE_string(
    'config_json',
    os.path.join(_HERE, 'golden_tiny_config.json'),
    'Where to write the config document (the same one goes into the npz).',
)

# The only thing the fixture depends on, besides the release code itself.
SEED = 20260906

BATCH = 2
PREFILL_LEN = 20
DECODE_STEPS = 2
TOTAL_LEN = PREFILL_LEN + DECODE_STEPS

# --- The tiny config --------------------------------------------------------

# Every Qwen3.8 text feature this package implements is switched on, at the
# smallest size that keeps it meaningful:
#   * 8 layers on the released schedule (`full_attention_interval=4`) -> two
#     full groups: attention at 3 and 7, a GatedDeltaNet layer at 4 that
#     consumes an attention layer's residual stream, and two scan stages;
#   * `num_attention_heads=4` over `num_key_value_heads=2` -> GQA with 2
#     query heads per kv head;
#   * `head_dim=64` with `partial_rotary_factor=0.25` -> 16 rotary dims = 8
#     pairs rotated and 48 dims passed through untouched. The
#     `mrope_section=[3, 3, 2]` split is NOT pinned by the model run (see the
#     module docstring and `mrope_oracle`);
#   * `linear_num_value_heads=6` over `linear_num_key_heads=2` -> the delta
#     rule's 3x key/query head repeat;
#   * `linear_conv_kernel_dim=4` -> a conv window wider than one token.
# `mtp_num_hidden_layers=0` and `output_gate_type`/`mamba_ssm_dtype`/
# `attn_output_gate` are carried because `config_lib.config_from_hf` reads
# them; the release code itself hardcodes those choices (see meta.notes).
TINY_CONFIG: dict[str, Any] = dict(
    model_type='qwen3_5_text',
    vocab_size=256,
    hidden_size=128,
    intermediate_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=1024,
    rms_norm_eps=1e-6,
    hidden_act='silu',
    attention_bias=False,
    tie_word_embeddings=False,
    use_cache=True,
    attn_output_gate=True,
    output_gate_type='swish',
    mamba_ssm_dtype='float32',
    mtp_num_hidden_layers=0,
    full_attention_interval=4,
    layer_types=[
        'linear_attention',
        'linear_attention',
        'linear_attention',
        'full_attention',
        'linear_attention',
        'linear_attention',
        'linear_attention',
        'full_attention',
    ],
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    rope_parameters=dict(
        rope_type='default',
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
        mrope_section=[3, 3, 2],
        mrope_interleaved=True,
    ),
)


def make_config(**overrides: Any) -> hf_config_lib.Qwen3_5TextConfig:
  """Builds the released text config from `TINY_CONFIG`.

  Args:
    **overrides: `TINY_CONFIG` entries to replace. Only `mrope_oracle` uses
      them, to build the released rotary geometry; the fixture itself is
      generated from `TINY_CONFIG` unchanged.

  Returns:
    The HF config, with eager attention (the fused kernels are GPU-only).

  Raises:
    ValueError: if the release stopped honouring `TINY_CONFIG`'s schedule.
  """
  cfg = hf_config_lib.Qwen3_5TextConfig(**{**TINY_CONFIG, **overrides})
  cfg._attn_implementation = 'eager'  # pylint: disable=protected-access
  if list(cfg.layer_types or []) != TINY_CONFIG['layer_types']:
    raise ValueError(f'{cfg.layer_types=} is not {TINY_CONFIG["layer_types"]}.')
  return cfg


def build_model(cfg) -> torch.nn.Module:
  """The text tower in eval mode, float32, on CPU.

  Args:
    cfg: the HF config.

  Returns:
    `Qwen3_5ForCausalLM`; the vision tower and the MTP head of the released
    checkpoint are not part of it and are not modelled by this port.
  """
  model = hf_modeling_lib.Qwen3_5ForCausalLM(cfg)
  model.eval()
  return model.to(torch.float32)


def init_params(model: torch.nn.Module, seed: int = SEED) -> None:
  """Deterministic, sane-magnitude fill for every parameter.

  Args:
    model: the HF model, filled in place.
    seed: the only thing the fill depends on.

  Raises:
    ValueError: if two entries of the state dict share storage (a tied
      weight would make the fill order-dependent) or a fill is not finite.

  HF's `_init_weights` leaves `dt_bias` and `A_log` to whatever the module
  constructor put there, and the constructor draws `A_log` from an
  unseeded RNG, so the fixture must not rely on it. Iterating
  `sorted(state_dict)` with one generator makes the fill reproducible from
  the seed alone.

  The two RMSNorm conventions are filled differently on purpose. The release
  stores every `*layernorm.weight`, `q_norm`, `k_norm` and `model.norm` as a
  DELTA -- `Qwen3_5RMSNorm` computes `x_norm * (1 + w)` and initialises `w` to
  zero -- while the GatedDeltaNet output norm `linear_attn.norm.weight` is a
  plain scale (`Qwen3_5RMSNormGated`, initialised to one). Filling the first
  kind around 0 and the second around 1 means a port that applies the wrong
  convention to either is off by ~2x, not by 5%.
  """
  gen = torch.Generator().manual_seed(seed)
  sd = model.state_dict()
  if len({p.data_ptr() for p in sd.values()}) != len(sd):
    raise ValueError('Tied weights: the per-kind fill would be ambiguous.')
  for name in sorted(sd):
    p = sd[name]
    if name.endswith('A_log'):
      v = torch.log(torch.empty_like(p).uniform_(1.0, 16.0, generator=gen))
    elif name.endswith('dt_bias'):
      v = torch.randn(p.shape, generator=gen) * 0.5
    elif name.endswith('linear_attn.norm.weight'):  # Gated RMSNorm: plain `w`.
      v = torch.randn(p.shape, generator=gen) * 0.05 + 1.0
    elif name.endswith('norm.weight') or name.endswith('layernorm.weight'):
      v = torch.randn(p.shape, generator=gen) * 0.05  # RMSNorm: `1 + w`.
    elif 'conv1d.weight' in name:  # [conv_dim, 1, kernel].
      v = torch.randn(p.shape, generator=gen) * 0.5
    elif name.endswith('embed_tokens.weight'):
      v = torch.randn(p.shape, generator=gen) * 0.05
    elif p.dim() == 2:  # Every nn.Linear, including lm_head.
      v = torch.randn(p.shape, generator=gen) / np.sqrt(p.shape[1])
    else:
      v = torch.randn(p.shape, generator=gen) * 0.05
    p.copy_(v.to(p.dtype))
  if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
    raise ValueError('A parameter fill is not finite.')


def make_input_ids(cfg, batch: int = BATCH, total: int = TOTAL_LEN) -> Any:
  """`[batch, total]` ids; the last `DECODE_STEPS` are the decode tokens.

  Args:
    cfg: the HF config, for the vocabulary and the BOS id.
    batch: sequences.
    total: prefill plus decode tokens.

  Returns:
    An int64 tensor. Column 0 is `bos_token_id`; nothing is padded, so the
    fixture says nothing about padded batches (`model_lib_test` covers those
    against the model itself).
  """
  rng = np.random.RandomState(SEED)
  ids = rng.randint(0, cfg.vocab_size, size=(batch, total)).astype(np.int64)
  ids[:, 0] = cfg.bos_token_id
  return torch.from_numpy(ids)


# --- Capture ----------------------------------------------------------------


class Capture:
  """Collects activations via forward hooks into `{key: np.ndarray}`.

  `prefix` is mutable so that the same hooks serve the prefill and each
  decode step: set it, run the model, drain `out`.
  """

  def __init__(self):
    self.out: dict[str, Any] = {}
    self.prefix = 'act/'
    self._handles = []

  def put(self, key: str, value: Any) -> None:
    """Stores one tensor under the current prefix, as float32 numpy."""
    if isinstance(value, tuple):
      value = value[0]  # `self_attn` returns (output, attn_weights).
    if isinstance(value, torch.Tensor):
      value = value.detach().cpu().numpy()
      if value.dtype == np.float64:
        value = value.astype(np.float32)
    self.out[self.prefix + key] = value

  def _pre(self, key):
    def hook(unused_module, args):
      self.put(key, args[0])

    return hook

  def _post(self, key):
    def hook(unused_module, unused_args, out):
      self.put(key, out)

    return hook

  def attach(self, model: torch.nn.Module) -> 'Capture':
    """Registers every hook this fixture reads, and returns self.

    Args:
      model: the `Qwen3_5ForCausalLM` to instrument.

    Returns:
      Self.

    Per layer the fixture stores the input and the output of both sub-layers,
    so a port can be bisected: `block_in` -> `mixer_norm_out` -> `mixer_out`
    -> `mlp_in` -> `mlp_norm_out` -> `mlp_out` -> `block_out`. The two
    residual adds are not hooked because they are asserted exactly in
    `_assert_residual_invariants`.
    """
    inner: Any = model.model
    h = self._handles
    h.append(inner.embed_tokens.register_forward_hook(self._post('embed')))
    h.append(inner.norm.register_forward_pre_hook(self._pre('final_norm_in')))
    h.append(inner.norm.register_forward_hook(self._post('final_hidden')))
    for i, layer in enumerate(inner.layers):
      p = f'layer{i}/'
      h.append(
          layer.input_layernorm.register_forward_pre_hook(
              self._pre(p + 'block_in')
          )
      )
      h.append(
          layer.input_layernorm.register_forward_hook(
              self._post(p + 'mixer_norm_out')
          )
      )
      mixer = getattr(layer, 'linear_attn', None)
      if mixer is None:
        mixer = layer.self_attn
      h.append(mixer.register_forward_hook(self._post(p + 'mixer_out')))
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
      h.append(layer.mlp.register_forward_hook(self._post(p + 'mlp_out')))
      h.append(layer.register_forward_hook(self._post(p + 'block_out')))
    return self

  def detach(self) -> None:
    for handle in self._handles:
      handle.remove()
    self._handles = []


def dump_cache(cache, cfg, prefix: str, out: dict[str, Any]) -> None:
  """Stores the decode state of every layer, by kind.

  Args:
    cache: the `transformers.cache_utils.DynamicCache` after a forward pass.
    cfg: the HF config.
    prefix: `cache_prefill`, `cache_decode` or `cache_decode2`.
    out: the fixture, updated in place.

  GatedDeltaNet layers get `conv_state` `[B, 2*key_dim + value_dim, kernel]`
  (the release keeps the full window, not `kernel - 1` of it) and
  `recurrent_state` `[B, num_v_heads, key_head_dim, value_head_dim]`;
  attention layers get `key`/`value` `[B, n_kv_heads, T, head_dim]`.
  """
  for i in range(cfg.num_hidden_layers):
    p = f'{prefix}/layer{i}/'
    layer = cache.layers[i]
    conv = getattr(layer, 'conv_states', {})
    if conv.get(0) is not None:
      out[p + 'conv_state'] = conv[0].numpy().astype(np.float32)
    recurrent = getattr(layer, 'recurrent_states', {})
    if recurrent.get(0) is not None:
      out[p + 'recurrent_state'] = recurrent[0].numpy().astype(np.float32)
    if getattr(layer, 'keys', None) is not None:
      out[p + 'key'] = layer.keys.numpy().astype(np.float32)
      out[p + 'value'] = layer.values.numpy().astype(np.float32)


def run_all(cfg, model, input_ids) -> dict[str, Any]:
  """Runs prefill (no cache), prefill (cache) and two decode steps.

  Args:
    cfg: the HF config.
    model: the filled model.
    input_ids: `[B, TOTAL_LEN]`.

  Returns:
    The fixture's activations, caches and ids.

  Raises:
    ValueError: if the cached prefill is not bit-identical to the uncached
      one, which would mean the cache path changes the numbers and the
      fixture could not be used to test a cached implementation.
  """
  prefill_ids = input_ids[:, :PREFILL_LEN]
  data: dict[str, Any] = {}

  cap = Capture().attach(model)
  cap.prefix = 'act/'
  out = model(input_ids=prefill_ids, use_cache=False)
  cap.put('logits', out.logits)
  data.update(cap.out)
  cap.out = {}

  # The same prefill, but building the cache; it must be bit-identical.
  cache = hf_modeling_lib.DynamicCache(config=cfg)
  out_c = model(input_ids=prefill_ids, past_key_values=cache, use_cache=True)
  if not torch.equal(out.logits, out_c.logits):
    raise ValueError(
        'Cached and uncached prefill disagree by '
        f'{(out.logits - out_c.logits).abs().max()}.'
    )
  cap.out = {}
  dump_cache(cache, cfg, 'cache_prefill', data)

  for step in range(DECODE_STEPS):
    pos = PREFILL_LEN + step
    suffix = '' if step == 0 else str(step + 1)
    cap.prefix = f'act_decode{suffix}/'
    step_ids = input_ids[:, pos : pos + 1]
    out_d = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
    cap.put('logits', out_d.logits)
    data.update(cap.out)
    cap.out = {}
    dump_cache(cache, cfg, f'cache_decode{suffix}', data)
    data[f'decode{suffix}_input_ids'] = step_ids.numpy().astype(np.int32)
  cap.detach()

  data['input_ids'] = input_ids.numpy().astype(np.int32)
  data['prefill_input_ids'] = prefill_ids.numpy().astype(np.int32)
  return data


# --- The mRoPE side oracle --------------------------------------------------

# Three position rows a text prompt never produces, chosen so that no two
# agree anywhere and all stay inside `max_position_embeddings`: time, a
# far-offset height, and a reversed, widely spaced width. Large offsets are
# the point -- the section split only ever moves the LOWEST-frequency
# channels (see `mrope_oracle`), whose angles are ~1e-6 per position.
_MROPE_ROWS = (
    lambda t: t,
    lambda t: t + 700,
    lambda t: 1023 - 50 * t,
)

# The released rotary geometry (`config.json`: head_dim 256,
# partial_rotary_factor 0.25 -> 32 channels, mrope_section [11, 11, 10]).
# The tiny config's 8 channels are too few for the released proportions to
# truncate anything, so the second oracle is generated at the real one.
_RELEASED_HEAD_DIM = 256
_RELEASED_MROPE_SECTION = [11, 11, 10]


def _rotary_cos_sin(rotary, position_ids, hidden_size: int):
  """`(cos, sin)` for `position_ids`, and the same with the split permuted.

  Args:
    rotary: a release `Qwen3_5TextRotaryEmbedding`.
    position_ids: `[3, B, T]` int64 positions.
    hidden_size: only used to shape the dummy hidden state it reads a dtype
      and a device from.

  Returns:
    `(cos, sin, cos_permuted)`.
  """
  hidden = torch.zeros(BATCH, position_ids.shape[-1], hidden_size)
  cos, sin = rotary(hidden, position_ids)
  section = list(rotary.mrope_section)
  rotary.mrope_section = [section[1], section[2], section[0]]
  cos_permuted, _ = rotary(hidden, position_ids)
  rotary.mrope_section = section
  return cos, sin, cos_permuted


def mrope_oracle(cfg, model) -> dict[str, Any]:
  """The release's mRoPE for three DIFFERENT position rows.

  Args:
    cfg: the HF config.
    model: the built model, for its `rotary_emb` (nothing else is run).

  Returns:
    `mrope/...`: the positions, the `cos`/`sin` the release builds from them
    at the fixture's geometry, and one random query/key pair before and after
    `apply_rotary_pos_emb`; plus `mrope_released/...`, the same `cos`/`sin` at
    the released `head_dim` and `mrope_section`.

  Raises:
    ValueError: if permuting `mrope_section` leaves `cos` unchanged at either
      geometry, i.e. if these arrays would pin nothing.

  The model run itself is text-only, where the three mRoPE rows carry the same
  position and `apply_interleaved_mrope` is an identity for ANY section split.
  So the split -- which channel of the rotary half follows time, height or
  width -- is invisible to `act/` and to the caches, and a port could read
  `mrope_section` backwards and still match the fixture everywhere. These
  arrays are the one thing in the fixture that separates the splits, for
  `utils/rope_test.py` to compare against; they are not produced by, and
  cannot be reproduced by, the text-only model path.

  Be careful what strength is claimed for them: with `3 * section[a]` at or
  past the channel count, the layout degenerates to `channel % 3` and only
  the top few (lowest-frequency) channels can move at all, so a wrong split
  is a ~1e-5-scale difference in `sin`, not a visibly wrong rotation.
  """
  t = torch.arange(PREFILL_LEN, dtype=torch.long)
  rows = torch.stack([row(t) for row in _MROPE_ROWS])
  position_ids = rows[:, None, :].expand(3, BATCH, PREFILL_LEN).contiguous()

  cos, sin, cos_permuted = _rotary_cos_sin(
      model.model.rotary_emb, position_ids, cfg.hidden_size
  )
  if torch.equal(cos, cos_permuted):
    raise ValueError(
        f'mrope_section={list(model.model.rotary_emb.mrope_section)} is inert'
        ' even for distinct position rows; these arrays would pin nothing.'
    )

  released_cfg = make_config(
      head_dim=_RELEASED_HEAD_DIM,
      rope_parameters={
          **TINY_CONFIG['rope_parameters'],
          'mrope_section': _RELEASED_MROPE_SECTION,
      },
  )
  released = hf_modeling_lib.Qwen3_5TextRotaryEmbedding(config=released_cfg)
  cos_r, sin_r, cos_r_permuted = _rotary_cos_sin(
      released, position_ids, released_cfg.hidden_size
  )
  if torch.equal(cos_r, cos_r_permuted):
    raise ValueError('the released mrope_section is inert; see above.')

  gen = torch.Generator().manual_seed(SEED + 1)
  shape = lambda heads: (BATCH, heads, PREFILL_LEN, cfg.head_dim)
  q = torch.randn(shape(cfg.num_attention_heads), generator=gen)
  k = torch.randn(shape(cfg.num_key_value_heads), generator=gen)
  q_out, k_out = hf_modeling_lib.apply_rotary_pos_emb(q, k, cos, sin)
  f32 = lambda x: x.numpy().astype(np.float32)
  return {
      'mrope/position_ids': position_ids.numpy().astype(np.int32),
      'mrope/cos': f32(cos),
      'mrope/sin': f32(sin),
      'mrope/q_in': f32(q),
      'mrope/k_in': f32(k),
      'mrope/q_out': f32(q_out),
      'mrope/k_out': f32(k_out),
      'mrope_released/cos': f32(cos_r),
      'mrope_released/sin': f32(sin_r),
  }


# --- Validation -------------------------------------------------------------


def _decode_prefixes() -> list[str]:
  """`['act_decode/', 'act_decode2/', ...]`, one per decode step."""
  return [
      'act_decode/' if s == 0 else f'act_decode{s + 1}/'
      for s in range(DECODE_STEPS)
  ]


def validate_fixture(data: dict[str, Any], cfg) -> None:
  """Structural assertions: the fixture must exercise every feature.

  Args:
    data: the fixture, before the parameters are added to it.
    cfg: the HF config.

  Raises:
    ValueError: if any invariant fails. Every one of them is something that
      could silently stop being true -- a recurrent state that stayed at
      zero, an attention cache that stopped growing, a decode step that
      re-ran the prefill -- and would leave a fixture that still "passes".
  """
  def check(condition: bool, message: str) -> None:
    if not condition:
      raise ValueError(message)

  for k, v in data.items():
    finite = bool(np.isfinite(np.asarray(v, np.float64)).all())
    check(finite, f'non-finite in {k}')

  types = list(cfg.layer_types)
  check(
      set(types) == {'linear_attention', 'full_attention'},
      f'the fixture must cover both layer kinds; {types=}',
  )
  # Why this shape and not the smaller one that already covers both mixers:
  # see the module docstring. Each of these is a defect class the fixture
  # would stop seeing if someone shrank it back.
  full = [i for i, t in enumerate(types) if t == 'full_attention']
  check(len(full) >= 2, f'need >= 2 attention layers to index them; {types=}')
  after_attention = [types[i + 1] for i in full if i + 1 < len(types)]
  check(
      'linear_attention' in after_attention,
      f'no GatedDeltaNet layer follows an attention layer; {types=}',
  )
  interval = TINY_CONFIG['full_attention_interval']
  check(
      len(types) // interval >= 2,
      f'a scanned stack of {len(types)} layers grouped by {interval} has'
      ' fewer than 2 stages, so the scan carry is never executed',
  )
  for key in ('mrope/cos', 'mrope/q_out', 'mrope_released/cos'):
    check(key in data, f'the mRoPE side oracle is missing {key}')
  cache_names = ['cache_prefill'] + [
      'cache_decode' if s == 0 else f'cache_decode{s + 1}'
      for s in range(DECODE_STEPS)
  ]
  for i, layer_type in enumerate(types):
    is_linear = layer_type == 'linear_attention'
    check(
        (f'cache_prefill/layer{i}/recurrent_state' in data) == is_linear,
        f'layer {i} ({layer_type}) has the wrong kind of decode state',
    )
    check(
        (f'cache_prefill/layer{i}/key' in data) != is_linear,
        f'layer {i} ({layer_type}) has the wrong kind of decode state',
    )
    if is_linear:
      previous = None
      for name in cache_names:
        state = data[f'{name}/layer{i}/recurrent_state']
        conv = data[f'{name}/layer{i}/conv_state']
        check(
            np.abs(state).max() > 1e-3,
            f'{name} layer {i} recurrent state is ~zero',
        )
        check(
            conv.shape[-1] == cfg.linear_conv_kernel_dim,
            f'{name} layer {i} conv window is {conv.shape[-1]}',
        )
        check(
            previous is None or not np.array_equal(state, previous),
            f'layer {i} recurrent state did not change at {name}',
        )
        previous = state
    else:
      for step, name in enumerate(cache_names):
        rows = data[f'{name}/layer{i}/key'].shape[2]
        check(
            rows == PREFILL_LEN + step,
            f'{name} layer {i} has {rows} rows, want {PREFILL_LEN + step}',
        )
        check(
            data[f'{name}/layer{i}/value'].shape[2] == rows,
            f'{name} layer {i} key and value disagree in length',
        )

  last = data['act/logits'][:, -1:]
  for prefix in _decode_prefixes():
    check(
        not np.array_equal(data[prefix + 'logits'], last),
        f'{prefix}logits repeat the last prefill position',
    )
    last = data[prefix + 'logits']
  _assert_residual_invariants(data, cfg)


def _assert_residual_invariants(data: dict[str, Any], cfg) -> None:
  """The block wiring, pinned exactly.

  Args:
    data: the fixture.
    cfg: the HF config.

  Raises:
    ValueError: if a captured tensor is not the sum the release computes.

  `np.array_equal`, not `allclose`: these are the model's own float32 adds in
  the model's own order, so anything but equality means the hooks are reading
  something other than what they claim to.
  """
  for prefix in ['act/'] + _decode_prefixes():
    if not np.array_equal(data[prefix + 'layer0/block_in'], data[
        prefix + 'embed'
    ]):
      raise ValueError(f'{prefix}layer0/block_in is not the embedding')
    for i in range(cfg.num_hidden_layers):
      p = f'{prefix}layer{i}/'
      if not np.array_equal(
          data[p + 'mlp_in'], data[p + 'block_in'] + data[p + 'mixer_out']
      ):
        raise ValueError(f'{p}mlp_in is not block_in + mixer_out')
      if not np.array_equal(
          data[p + 'block_out'], data[p + 'mlp_in'] + data[p + 'mlp_out']
      ):
        raise ValueError(f'{p}block_out is not mlp_in + mlp_out')
      nxt = f'{prefix}layer{i + 1}/block_in'
      if nxt in data and not np.array_equal(data[nxt], data[p + 'block_out']):
        raise ValueError(f'{nxt} is not layer {i}"s output')
    if not np.array_equal(
        data[prefix + 'final_norm_in'],
        data[f'{prefix}layer{cfg.num_hidden_layers - 1}/block_out'],
    ):
      raise ValueError(f'{prefix}final_norm_in is not the last block output')


def param_arrays(model: torch.nn.Module) -> dict[str, Any]:
  """`{'param/<HF state_dict key>': float32 array}`, HF shapes verbatim.

  Args:
    model: the filled model.

  Returns:
    The parameters. They are stored under their HuggingFace names, not
    Simply's, so that the conversion is done by the code under test
    (`utils/ckpt_format.py`) and the fixture cannot bake in a mapping bug.
  """
  return {
      f'param/{k}': v.detach().cpu().numpy().astype(np.float32)
      for k, v in model.state_dict().items()
  }


# --- Provenance -------------------------------------------------------------

_GENERATED_BY = (
    'simply/zoo/qwen3p8/testdata/gen_golden.py'
    ' (run it directly; see its docstring)'
)
_HF_CONFIG_CLASS = (
    'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig'
)
_HF_MODEL_CLASS = (
    'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'
)

# What a reader of the fixture has to know and cannot see in the arrays.
_NOTE_TEXT_ONLY = (
    'Text tower only: the released checkpoint also carries a vision tower and'
    ' an MTP head, which this port drops at restore.'
)
_NOTE_CHUNKS = (
    'The prefill is 20 tokens so that a test loading the config with'
    ' linear_attention_chunk_size=8 crosses two chunk boundaries and ends'
    ' with a 4-token remainder.'
)
_NOTE_DECORATIVE_KEYS = (
    'attn_output_gate, output_gate_type and mamba_ssm_dtype are read by'
    ' config_lib.config_from_hf but not by the release code, which hardcodes'
    ' the sigmoid attention gate, the silu GatedDeltaNet norm gate and'
    ' float32 delta-rule math.'
)
_NOTE_NO_MTP = (
    'mtp_num_hidden_layers=0 is accepted by the release config and builds no'
    ' MTP head.'
)
_NOTE_TWO_NORMS = (
    'Qwen3_5RMSNorm applies (1 + weight); the GatedDeltaNet output norm'
    ' Qwen3_5RMSNormGated applies weight. init_params fills the two around 0'
    ' and around 1 respectively.'
)
_NOTE_POSITIONS = (
    'No attention_mask and no position_ids are passed to the model: nothing'
    ' is padded and the positions are arange(prefill_len) then one position'
    ' per decode step. The release expands them to the three mRoPE rows'
    ' itself, so all three carry the same text position (there is no vision'
    ' input) and apply_interleaved_mrope is an identity for any'
    ' mrope_section: the split is NOT pinned by act/ or by the caches.'
)
_NOTE_MROPE_ORACLE = (
    'mrope/* is a side oracle, not part of the model run: the release rotary'
    ' module applied to three DIFFERENT position rows, with the query/key'
    ' pair before and after apply_rotary_pos_emb. It is what utils/rope_test'
    ' can pin the mrope_section split against; generation fails if permuting'
    ' the split leaves mrope/cos unchanged.'
)
_NOTE_NOT_COVERED = (
    'Deliberately not covered: the deployed numerics. This is float32, and'
    ' utils/test_utils loads it unscanned, unsharded and with'
    ' linear_attention_chunk_size=8, while the release is bfloat16, scanned,'
    ' sharded and 32. Nothing here would see a bf16-only or sharding-only'
    ' defect; compare_to_hf_reference.py on the released weights is what'
    ' covers that shape.'
)
_NOTE_CONV_STATE = (
    'cache_*/layer{i}/conv_state is the conv INPUT window -- the last'
    ' linear_conv_kernel_dim columns of in_proj_qkv(x), before the depthwise'
    ' convolution -- and the release keeps all of them, not kernel - 1.'
)


def config_document(cfg) -> dict[str, Any]:
  """The provenance document, written to the JSON file and into the npz.

  Args:
    cfg: the HF config.

  Returns:
    Exactly three keys: `config_kwargs` (the contract -- what
    `config_lib.config_from_hf` is fed), `hf_config` (what the release's own
    config class made of it) and `meta` (how it was run).
  """
  types = list(cfg.layer_types or [])
  return {
      'config_kwargs': TINY_CONFIG,
      'hf_config': json.loads(cfg.to_json_string()),
      'meta': {
          'seed': SEED,
          'dtype': 'float32',
          'batch_size': BATCH,
          'prefill_len': PREFILL_LEN,
          'decode_steps': DECODE_STEPS,
          'linear_attention_layers_0indexed': [
              i for i, t in enumerate(types) if t == 'linear_attention'
          ],
          'full_attention_layers_0indexed': [
              i for i, t in enumerate(types) if t == 'full_attention'
          ],
          'generated_by': _GENERATED_BY,
          'hf_config_class': _HF_CONFIG_CLASS,
          'hf_model_class': _HF_MODEL_CLASS,
          'attn_implementation': 'eager',
          # `transformers_version` is in `hf_config`; the release puts it
          # there itself.
          'notes': [
              _NOTE_TEXT_ONLY,
              _NOTE_CHUNKS,
              _NOTE_DECORATIVE_KEYS,
              _NOTE_NO_MTP,
              _NOTE_TWO_NORMS,
              _NOTE_POSITIONS,
              _NOTE_MROPE_ORACLE,
              _NOTE_CONV_STATE,
              _NOTE_NOT_COVERED,
          ],
      },
  }


def main(argv) -> None:
  del argv
  torch.set_grad_enabled(False)
  torch.manual_seed(SEED)
  cfg = make_config()
  model = build_model(cfg)
  init_params(model)
  input_ids = make_input_ids(cfg)

  data = run_all(cfg, model, input_ids)
  data.update(mrope_oracle(cfg, model))
  validate_fixture(data, cfg)
  data.update(param_arrays(model))

  cfg_doc = config_document(cfg)
  with open(_CONFIG_JSON.value, 'w') as f:
    json.dump(cfg_doc, f, indent=1, sort_keys=True)
  data['meta/config_json'] = np.array(json.dumps(cfg_doc, sort_keys=True))

  np.savez_compressed(_OUT.value, **data)
  size_mb = os.path.getsize(_OUT.value) / 1e6
  n_param = sum(v.size for k, v in data.items() if k.startswith('param/'))
  print(
      f'wrote {_OUT.value} ({size_mb:.1f} MB, {len(data)} arrays,'
      f' {n_param} parameter values)'
  )
  print(f'wrote {_CONFIG_JSON.value}')


if __name__ == '__main__':
  app.run(main)

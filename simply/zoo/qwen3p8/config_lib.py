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
"""What a Qwen3.8 is, and which Qwen3.8s we run.

`Qwen38ExperimentConfig` is the architecture as data, `config_from_hf` is the
one place a released `config.json` turns into one -- shared by the golden
fixture and the released deployment so the two cannot drift -- and the rest of
the file is the registered deployments and the arithmetic that sizes them.

The architecture is specified by the HuggingFace release `Qwen/Qwen3.8-27B`:
`config.json` declares `model_type: qwen3_5` and loads through
`transformers.models.qwen3_5`. Qwen3.8-27B is the **dense** member of that
family: 64 layers of 3 GatedDeltaNet (`linear_attention`) : 1 gated softmax
attention (`full_attention_interval = 4`), model dim 5120, SwiGLU FFN 17408,
24 query / 4 key-value heads of 256, GatedDeltaNet 16 key / 48 value heads of
128 with a 4-wide causal depthwise conv and a 32-token chunked delta rule,
interleaved partial mRoPE `(11, 11, 10)` on the attention layers only, vocab
248320, 262144 trained positions (`rope_scaling: null`).

`config.json`'s one delta from Qwen3.5 is `output_gate_type: "swish"`, which
names the activation of the **GatedDeltaNet output-norm** gate (silu, which
both this port and HuggingFace already hardcode). The softmax-attention output
gate stays sigmoid. See README.md; the A/B is machine-checked in
`utils/gdn_multi_device_test.py`.

Registered configs (`--experiment_config=<name>`):

  `qwen3p8_27b`        the released 27B, bf16, evaluated on 16 chips.
  `qwen3p8_tiny_test`  a 4-layer toy with the same topology, random init,
                       CPU-runnable.

Only the text backbone is modelled: `config.json` carries a vision tower
(`model.visual.*`, 333 tensors) that text-only inference never enters, and one
multi-token-prediction layer (`mtp.*`) that decoding never runs. Both are
dropped at restore by `utils/ckpt_format.Qwen38Format`.
"""

from collections.abc import Mapping
import dataclasses
import os
from typing import Any

from simply import config_lib
from simply.utils import position_encoding as pe_lib
from simply.zoo.qwen3p8.utils import rope as rope_lib

# --- Layer schedule ---------------------------------------------------------

LINEAR_ATTENTION = 'linear_attention'
FULL_ATTENTION = 'full_attention'


def layer_types(
    n_layers: int, full_attention_interval: int = 4
) -> tuple[str, ...]:
  """The released schedule: every `full_attention_interval`-th layer is full."""
  return tuple(
      FULL_ATTENTION
      if (i + 1) % full_attention_interval == 0
      else LINEAR_ATTENTION
      for i in range(n_layers)
  )


# --- Experiment config ------------------------------------------------------

# `<|endoftext|>`, the released pad token. `text_config.pad_token_id` is null
# and Simply needs one to pad a batch with.
PAD_ID = 248044

# The converted weights: `tools:hf_to_orbax` output, restored through
# `Qwen38Format`. Under `MODELS_DIR` so the OSS scrub rewrites it.
QWEN3P8_27B_CKPT_DIR = os.path.join(
    config_lib.MODELS_DIR, 'qwen3p8/Qwen3.8-27B/ORBAX'
)


@dataclasses.dataclass(frozen=True)
class Qwen38ExperimentConfig(config_lib.BaseExperimentConfig):
  """Qwen3.8: a dense hybrid of GatedDeltaNet and gated softmax attention.

  The fields are the released `text_config`, renamed to Simply's vocabulary
  where Simply already has a name for the thing (`hidden_size` ->`model_dim`,
  `intermediate_size` -> `ffn_expand_dim`, ...). `config_from_hf` is the
  mapping; `config_lib_test.py` checks every leaf of the released config
  against it.
  """

  # The plugins are named by string, as `simply/config_lib.py`
  # does, so that a config can be read without importing a model.
  model_name: str = 'Qwen38HybridLM'
  vocab_name: str = 'Qwen3.8'
  lm_format_name: str = 'Qwen38Chat'
  init_ckpt_format: str = 'Qwen38Format'

  seq_len: int = 262_144
  vocab_size: int = 248_320
  n_heads: int = 24
  n_kv_heads: int = 4
  per_head_dim: int = 256
  ffn_activation: str = 'silu'
  pad_id: int = PAD_ID

  # Hybrid schedule. `layer_types` is authoritative; `full_attention_interval`
  # is the rule that generated it and what `config_from_hf` checks against.
  layer_types: tuple[str, ...] = ()
  full_attention_interval: int = 4

  # Attention. The output gate is the sigmoid one (`attn_output_gate`).
  attn_output_gate: bool = True
  use_qk_norm: bool = True
  position_encoding: pe_lib.PositionEncodingConfig = rope_lib.Qwen38RoPE()

  # GatedDeltaNet.
  linear_num_key_heads: int = 16
  linear_num_value_heads: int = 48
  linear_key_head_dim: int = 128
  linear_value_head_dim: int = 128
  linear_conv_kernel_dim: int = 4
  linear_attention_chunk_size: int = 32
  # `text_config.mamba_ssm_dtype`: HuggingFace casts the delta-rule inputs to
  # float32 in both of its kernels whatever the activation dtype is. bfloat16
  # here costs ~1% relative logit error at 16 tokens and buys ~3x per decode
  # step on CPU.
  gdn_compute_dtype: str = 'float32'

  # Pinned, not inherited: Simply's defaults for these are wrong for Qwen3.8,
  # and wrong silently. `model_lib._UNIMPLEMENTED_FIELDS` refuses the ones a
  # caller could still override into something this port does not implement.
  #
  # Neither soft cap exists in the release; `output_logits_soft_cap = 30.0`
  # (Simply's default) returns `30*tanh(logits/30)`, which is invisible to
  # greedy decoding and moves KL(HF||Simply) from 4.2e-4 to 9.2e-2 -- i.e. it
  # silently flattens every sampled eval (measured KL(HF||Simply) 4.8e-4 with
  # the cap off, 9.2e-2 with Simply's default).
  attn_soft_cap: float = -1.0
  output_logits_soft_cap: float = -1.0
  # `Qwen3_5RMSNorm` is `x_norm * (1 + w)` with `w` initialised to zeros
  # (modeling_qwen3_5.py:723-737), i.e. the checkpoint stores the delta, as
  # Gemma's does. The gated GatedDeltaNet output norm is the exception -- it is
  # `w * x_norm` with `w` initialised to ones -- and `utils/gdn.py` owns that.
  # There is no embedding scale, no post-LN, no per-dim scale and no bias.
  norm_scale_plus_one: bool = True
  embedding_lookup_scale: float | None = None
  use_post_ln: bool = False
  use_per_dim_scale: bool = False
  ffn_use_bias: bool = False
  output_layer_use_bias: bool = False
  use_tied_embedding: bool = False
  # Inference configs; training is not covered by this package. `use_scan` is
  # off because `model_lib.py` has no scanned stack: `eval/decode_eval.py` in
  # core forces `use_scan=False` anyway, so one would be unreachable code over
  # the most intricate part of the model.
  use_scan: bool = False
  use_remat: bool = False
  reset_steps: bool = True

  def resolved_layer_types(self) -> tuple[str, ...]:
    """`layer_types`, or the schedule `full_attention_interval` implies."""
    return self.layer_types or layer_types(
        self.n_layers, self.full_attention_interval
    )


# --- HF config mapping ------------------------------------------------------

# Released `text_config` keys this port does not represent, and why. Checked
# exhaustively by `config_lib_test.py`, which fails when the release grows a
# key that is in neither table.
_HF_KEYS_NOT_REPRESENTED = (
    'attention_dropout',  # Inference-only port.
    'bos_token_id',  # The chat format supplies its own prefix.
    'dtype',  # `activation_dtype_name` is a deployment choice.
    'eos_token_id',  # `lm_format` owns the stop tokens.
    'hidden_act',  # 'silu' is `ffn_activation`'s value, asserted below.
    'initializer_range',  # Inference-only port: weights come from a release.
    'mamba_ssm_dtype',  # -> `gdn_compute_dtype`, asserted below.
    'model_type',  # Asserted below.
    'output_gate_type',  # Asserted below.
    'pad_token_id',  # Null in the release; Simply needs one (`PAD_ID`).
    'use_cache',  # Always on in Simply's decode path.
)


def config_from_hf(
    text_config: Mapping[str, Any], **overrides: Any
) -> Qwen38ExperimentConfig:
  """Builds the Simply config from the released `config.json:text_config`.

  One mapping, shared by the released deployment and the golden fixture, so a
  real checkpoint and the test cannot drift apart.

  Args:
    text_config: The `text_config` block of a released `config.json`.
    **overrides: Applied last, for deployment choices (batch size, dtype, ...).

  Returns:
    The Simply config for that release.

  Raises:
    ValueError: if the release asks for a variant this port does not implement.
  """
  if text_config.get('model_type') not in ('qwen3_5', 'qwen3_5_text'):
    raise ValueError(
        f'Not a Qwen3.5-family text config: {text_config.get("model_type")=}.'
    )
  if text_config.get('output_gate_type', 'swish') != 'swish':
    raise ValueError(
        'Only `output_gate_type: "swish"` is implemented (the GatedDeltaNet'
        ' output-norm gate is silu); got'
        f' {text_config["output_gate_type"]!r}.'
    )
  if text_config.get('hidden_act', 'silu') != 'silu':
    raise ValueError(f'Only SwiGLU FFNs: {text_config["hidden_act"]=}.')
  if text_config.get('mamba_ssm_dtype', 'float32') != 'float32':
    raise ValueError(f'{text_config["mamba_ssm_dtype"]=} is not implemented.')
  if text_config.get('attention_bias'):
    raise ValueError('Attention biases are not implemented.')
  if text_config.get('tie_word_embeddings'):
    raise ValueError('Tied embeddings are not implemented for Qwen3.8.')
  rope = dict(text_config.get('rope_parameters') or {})
  if rope.get('rope_type', 'default') != 'default':
    raise ValueError(f'Only untruncated RoPE: {rope["rope_type"]=}.')
  if not rope.get('mrope_interleaved', False):
    raise ValueError('Only the interleaved mRoPE layout is implemented.')

  n_layers = text_config['num_hidden_layers']
  interval = text_config['full_attention_interval']
  config = Qwen38ExperimentConfig(
      vocab_size=text_config['vocab_size'],
      model_dim=text_config['hidden_size'],
      ffn_expand_dim=text_config['intermediate_size'],
      n_layers=n_layers,
      n_heads=text_config['num_attention_heads'],
      n_kv_heads=text_config['num_key_value_heads'],
      per_head_dim=text_config['head_dim'],
      seq_len=text_config['max_position_embeddings'],
      attn_output_gate=text_config['attn_output_gate'],
      full_attention_interval=interval,
      layer_types=tuple(text_config['layer_types']),
      linear_num_key_heads=text_config['linear_num_key_heads'],
      linear_num_value_heads=text_config['linear_num_value_heads'],
      linear_key_head_dim=text_config['linear_key_head_dim'],
      linear_value_head_dim=text_config['linear_value_head_dim'],
      linear_conv_kernel_dim=text_config['linear_conv_kernel_dim'],
      gdn_compute_dtype=text_config.get('mamba_ssm_dtype', 'float32'),
      rms_norm_epsilon=text_config['rms_norm_eps'],
      position_encoding=rope_lib.Qwen38RoPE(
          max_timescale=rope['rope_theta'],
          rotary_fraction=rope['partial_rotary_factor'],
          mrope_section=tuple(rope['mrope_section']),
      ),
  )
  expected = layer_types(n_layers, interval)
  if config.layer_types != expected:
    raise ValueError(
        'Only the 3:1 GatedDeltaNet:attention schedule is implemented;'
        f' {text_config["layer_types"]=} is not'
        f' full_attention_interval={interval}.'
    )
  return dataclasses.replace(config, **overrides) if overrides else config


# --- Parameter arithmetic ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParamCounts:
  """Parameters of a Qwen3.8, by part."""

  embedding: int
  gated_delta_net: int
  attention: int
  ffn: int
  norms: int
  output: int

  @property
  def total(self) -> int:
    return (
        self.embedding
        + self.gated_delta_net
        + self.attention
        + self.ffn
        + self.norms
        + self.output
    )


def param_counts(config: Qwen38ExperimentConfig) -> ParamCounts:
  """Closed-form parameter count; pinned against `model.init` in the test."""
  if config.ffn_expand_dim is None:
    raise ValueError(
        'Qwen3.8 sizes its FFN explicitly (`intermediate_size`); a config with'
        ' ffn_expand_dim=None would be sized by `expand_factor` instead.'
    )
  types = config.resolved_layer_types()
  n_full = sum(t == FULL_ATTENTION for t in types)
  n_linear = len(types) - n_full
  d = config.model_dim
  key_dim = config.linear_num_key_heads * config.linear_key_head_dim
  value_dim = config.linear_num_value_heads * config.linear_value_head_dim
  # in_proj_qkv + in_proj_z (2 key + 2 value stacks), in_proj_b + in_proj_a,
  # the conv, A_log + dt_bias, the output norm and out_proj.
  gdn = n_linear * (
      d * (2 * key_dim + 2 * value_dim)
      + d * 2 * config.linear_num_value_heads
      + (2 * key_dim + value_dim) * config.linear_conv_kernel_dim
      + 2 * config.linear_num_value_heads
      + config.linear_value_head_dim
      + value_dim * d
  )
  q_dim = config.n_heads * config.per_head_dim
  kv_dim = config.n_kv_heads * config.per_head_dim
  # q (doubled by the output gate), k, v, out, and the q/k head norms.
  attention = n_full * (
      d * ((2 if config.attn_output_gate else 1) * q_dim + 2 * kv_dim)
      + q_dim * d
      + 2 * config.per_head_dim
  )
  ffn = len(types) * 3 * d * config.ffn_expand_dim
  return ParamCounts(
      embedding=config.vocab_size * d,
      gated_delta_net=gdn,
      attention=attention,
      ffn=ffn,
      norms=(2 * len(types) + 1) * d,
      output=0 if config.use_tied_embedding else d * config.vocab_size,
  )


def kv_cache_bytes(
    config: Qwen38ExperimentConfig,
    *,
    batch_size: int,
    max_seq_len: int,
    bytes_per_element: int = 2,
) -> int:
  """Bytes of attention KV cache the non-paged sampler allocates.

  The GatedDeltaNet layers hold a constant-size conv window and recurrent
  state instead, which `gdn_state_bytes` counts; this is the term that grows
  with the context and decides what batch a slice can hold.

  Args:
    config: The deployment.
    batch_size: Sequences in flight.
    max_seq_len: Positions the cache is opened for.
    bytes_per_element: 2 for bf16, 4 for float32.

  Returns:
    Total bytes over all devices; divide by the mesh's `data * model` for the
    per-device figure (the cache is sharded on batch and on kv heads).
  """
  n_full = sum(t == FULL_ATTENTION for t in config.resolved_layer_types())
  return (
      2  # key and value
      * n_full
      * batch_size
      * max_seq_len
      * config.n_kv_heads
      * config.per_head_dim
      * bytes_per_element
  )


def gdn_state_bytes(
    config: Qwen38ExperimentConfig,
    *,
    batch_size: int,
    bytes_per_element: int = 4,
) -> int:
  """Bytes of GatedDeltaNet decode state; constant in the sequence length."""
  types = config.resolved_layer_types()
  n_linear = sum(t == LINEAR_ATTENTION for t in types)
  key_dim = config.linear_num_key_heads * config.linear_key_head_dim
  value_dim = config.linear_num_value_heads * config.linear_value_head_dim
  conv = (2 * key_dim + value_dim) * (config.linear_conv_kernel_dim - 1)
  recurrent = (
      config.linear_num_value_heads
      * config.linear_key_head_dim
      * config.linear_value_head_dim
  )
  return n_linear * batch_size * (conv + recurrent) * bytes_per_element


# --- Configs ----------------------------------------------------------------


def _released_text_config() -> Mapping[str, Any]:
  """The released `text_config`, as data, so `config_from_hf` is the only map."""
  return dict(
      model_type='qwen3_5_text',
      vocab_size=248_320,
      hidden_size=5120,
      intermediate_size=17408,
      num_hidden_layers=64,
      num_attention_heads=24,
      num_key_value_heads=4,
      head_dim=256,
      max_position_embeddings=262_144,
      attn_output_gate=True,
      full_attention_interval=4,
      layer_types=list(layer_types(64)),
      linear_num_key_heads=16,
      linear_num_value_heads=48,
      linear_key_head_dim=128,
      linear_value_head_dim=128,
      linear_conv_kernel_dim=4,
      mamba_ssm_dtype='float32',
      mtp_num_hidden_layers=1,
      mtp_use_dedicated_embeddings=False,
      partial_rotary_factor=0.25,
      rms_norm_eps=1e-6,
      output_gate_type='swish',
      hidden_act='silu',
      tie_word_embeddings=False,
      attention_bias=False,
      rope_parameters=dict(
          rope_type='default',
          rope_theta=10_000_000,
          partial_rotary_factor=0.25,
          mrope_section=[11, 11, 10],
          mrope_interleaved=True,
      ),
  )


@config_lib.ExperimentConfigRegistry.register
def qwen3p8_27b() -> Qwen38ExperimentConfig:
  """The released Qwen3.8-27B: 26.9 B parameters, bf16.

  Meshed `(replica, data, model)`; `model` must divide both the 4 key-value
  heads and the 24 query heads, so it is 1, 2 or 4, and the parallelism above
  that goes on `data` -- e.g. `1,4,4` on 16 chips or `1,8,4` on 32.
  `eval/decode_eval.py` refuses anything else before the restore.

  Returns:
    The released deployment's config.
  """
  return config_from_hf(
      _released_text_config(),
      init_ckpt_dir=QWEN3P8_27B_CKPT_DIR,
      init_ckpt_step=-1,
      sharding_config=config_lib.BaseSharding(),
  )


@config_lib.ExperimentConfigRegistry.register
def qwen3p8_tiny_test() -> Qwen38ExperimentConfig:
  """A 4-layer toy with the released topology: random init, CPU-runnable."""
  return dataclasses.replace(
      qwen3p8_27b(),
      model_dim=256,
      ffn_expand_dim=512,
      n_heads=4,
      n_kv_heads=2,
      per_head_dim=64,
      # Eight layers, not four: two full groups, so a GatedDeltaNet layer
      # consumes an attention layer's residual stream, as it does 15 times in
      # the released model. `testdata/golden_tiny_config.json` is the same
      # shape for the same reason.
      n_layers=8,
      layer_types=layer_types(8),
      linear_num_key_heads=2,
      linear_num_value_heads=6,
      linear_key_head_dim=32,
      linear_value_head_dim=32,
      vocab_size=1024,
      seq_len=128,
      init_ckpt_dir='',
      init_ckpt_format='',
  )

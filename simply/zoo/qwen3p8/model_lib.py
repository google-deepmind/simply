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
"""The Qwen3.8 text backbone: 3 GatedDeltaNet layers : 1 gated attention, x16.

`Qwen38Block` is one layer -- a token mixer chosen by `layer_type` and a SwiGLU
FFN, each behind an RMSNorm and a residual -- and `Qwen38HybridLM` is the
stack, the embedding, the final norm and the untied head. The mixers are in
`utils/gdn.py` and `utils/attn.py`; this file owns the layer schedule and the
**decode protocol**, which is the part a hybrid model gets wrong by default:

* the recurrent mixer has no notion of position, so `apply` refuses a
  `decode_state` without `segment_positions`;
* `sampling_lib` prefills a padded window and then re-feeds the tokens from
  `extra_inputs['prefill_position']` one at a time, so the prefill absorbs only
  the tokens before that position and marks the rest as padding. Without it the
  GatedDeltaNet state that reaches the decode loop is the state after
  `prefill_size` tokens, pad suffix included, instead of after `input_len`, and
  the last prompt token is folded into the recurrence twice;
* `model_lib.pad_block_decode_state` is registered for the GatedDeltaNet state
  at the bottom of this file: the conv window and the recurrent state are
  constant in the sequence length, so growing the batch's horizon is a no-op
  for them, while the attention layers keep core's mapping KV cache and core's
  own padding.

The stack is unrolled. `config.use_scan` is pinned off (`config_lib.py`) and
`eval/decode_eval.py` in core forces it off anyway, so a scanned stack would be
unreachable code over the most intricate part of the model; scanning the
repeating `(GDN, GDN, GDN, attention)` group is exact and would cut the compile,
and it belongs in the change that can measure it.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any, cast

import einops
import jax
from jax import numpy as jnp

from simply import model_lib
from simply.utils import common
from simply.utils import module
from simply.utils import sharding as sharding_lib
from simply.zoo.qwen3p8 import config_lib as qwen3p8_config_lib
from simply.zoo.qwen3p8.utils import attn as attn_lib
from simply.zoo.qwen3p8.utils import gdn as gdn_lib

Array = common.Array
PRNGKey = jax.typing.ArrayLike
# The config is passed as data; typing it would tie this module to the config
# module's identity, which `simply/model_lib.py` avoids too.
SimplyConfig = Any


def block_key(layer_idx: int) -> str:
  """The parameter- and decode-state key of layer `layer_idx`."""
  return f'block_{layer_idx}'


# --- One layer --------------------------------------------------------------


@module.ModuleRegistry.register
@dataclasses.dataclass
class Qwen38Block(module.SimplyModule):
  """One Qwen3.8 layer: a token mixer and a SwiGLU FFN, pre-normed."""

  config: SimplyConfig
  layer_idx: int
  layer_type: str
  sharding_config: SimplyConfig | None = None

  def setup(self) -> None:
    config = self.config
    activation_dtype = config.activation_dtype_name
    # `Qwen3_5RMSNorm` computes `x_norm * (1 + w)` in float32 and rounds once
    # (modeling_qwen3_5.py:723-737), while core's `LayerNorm` rounds to
    # `activation_dtype` *before* the scale. In bfloat16 that quantizes
    # `1 + w` itself: a fixed per-channel error of up to 3.9e-3 that does not
    # average out over the 129 norms of the released model. Asking for float32
    # here buys HuggingFace's order for one elementwise multiply; `apply`
    # restores the caller's dtype, so the layer still hands on bfloat16.
    # (`utils/gdn.py` deliberately does NOT do this: `Qwen3_5RMSNormGated` is
    # the release's other convention -- no `1 + w`, and it rounds first.)
    self.input_layernorm = model_lib.LayerNorm(
        dim=config.model_dim,
        use_bias=False,
        activation_dtype='float32',
        scale_plus_one=config.norm_scale_plus_one,
        epsilon=config.rms_norm_epsilon,
    )
    self.post_attention_layernorm = model_lib.LayerNorm(
        dim=config.model_dim,
        use_bias=False,
        activation_dtype='float32',
        scale_plus_one=config.norm_scale_plus_one,
        epsilon=config.rms_norm_epsilon,
    )
    if self.layer_type == qwen3p8_config_lib.LINEAR_ATTENTION:
      self.token_mixer = gdn_lib.Qwen38GatedDeltaNet(
          model_dim=config.model_dim,
          num_key_heads=config.linear_num_key_heads,
          num_value_heads=config.linear_num_value_heads,
          key_head_dim=config.linear_key_head_dim,
          value_head_dim=config.linear_value_head_dim,
          conv_kernel_dim=config.linear_conv_kernel_dim,
          chunk_size=config.linear_attention_chunk_size,
          activation_dtype=activation_dtype,
          gdn_compute_dtype=config.gdn_compute_dtype,
          rms_norm_epsilon=config.rms_norm_epsilon,
          sharding_config=self.sharding_config,
      )
    elif self.layer_type == qwen3p8_config_lib.FULL_ATTENTION:
      self.token_mixer = attn_lib.Qwen38Attention(
          model_dim=config.model_dim,
          n_heads=config.n_heads,
          n_kv_heads=config.n_kv_heads,
          per_head_dim=config.per_head_dim,
          attn_output_gate=config.attn_output_gate,
          use_qk_norm=config.use_qk_norm,
          activation_dtype=activation_dtype,
          rms_norm_epsilon=config.rms_norm_epsilon,
          position_encoding=config.position_encoding,
          sharding_config=self.sharding_config,
      )
    else:
      raise ValueError(
          f'Unsupported Qwen3.8 layer_type={self.layer_type!r} at layer'
          f' {self.layer_idx}.'
      )
    self.ffn = model_lib.FeedForward(
        model_dim=config.model_dim,
        expand_factor=0,
        sharding_config=self.sharding_config,
        use_gated_activation_in_ffn=True,
        activation_dtype=activation_dtype,
        ffn_expand_dim=config.ffn_expand_dim,
        ffn_use_bias=False,
        ffn_activation=config.ffn_activation,
    )

  def init(self, prng_key: PRNGKey) -> Any:
    mixer_key, ffn_key = jax.random.split(prng_key, num=2)
    return {
        'input_layernorm': self.input_layernorm.init(),
        'post_attention_layernorm': self.post_attention_layernorm.init(),
        'token_mixer': self.token_mixer.init(mixer_key),
        'ffn': self.ffn.init(ffn_key),
    }

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: Any,
      x: Array,
      *,
      segment_ids: Array,
      segment_positions: Array,
      extra_inputs: Any | None = None,
      decode_state: Any | None = None,
  ) -> tuple[Array, Any]:
    inputs_mask = segment_ids != 0
    residual = x
    x_norm = self.input_layernorm.apply(params['input_layernorm'], x)
    mixed, mixer_extra = self.token_mixer.apply(
        params['token_mixer'],
        x_norm,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        extra_inputs=extra_inputs,
        decode_state=decode_state,
    )
    x = self._constrain(residual + mixed)

    residual = x
    x_norm = self.post_attention_layernorm.apply(
        params['post_attention_layernorm'], x
    )
    ffn_out, _ = self.ffn.apply(params['ffn'], x_norm, inputs_mask=inputs_mask)
    x = self._constrain(residual + ffn_out)
    return x, {'decode_state': mixer_extra['decode_state']}

  def _constrain(self, x: Array) -> Array:
    if self.sharding_config is None:
      return x
    return sharding_lib.with_sharding_constraint(
        jnp.asarray(x), self.sharding_config.activation_partition
    )

  def init_decode_state(self, batch_size: int, max_seq_len: int) -> Any:
    return self.token_mixer.init_decode_state(batch_size, max_seq_len)


# --- The stack --------------------------------------------------------------


@module.ModuleRegistry.register
@dataclasses.dataclass
class Qwen38HybridLM(module.SimplyModule):
  """The Qwen3.8 text backbone."""

  config: SimplyConfig
  sharding_config: SimplyConfig | None = None

  def setup(self) -> None:
    config = self.config
    if self.sharding_config is None:
      self.sharding_config = config.sharding_config
    self.embed_linear = module.EmbeddingLinear(
        vocab_size=config.vocab_size,
        dim=config.model_dim,
        weight_partition=self.sharding_config.embed_partition,
        activation_dtype=config.activation_dtype_name,
        embedding_scale_by_sqrt_dim=config.embedding_lookup_scale,
        use_tied_embedding=config.use_tied_embedding,
        use_bias=config.output_layer_use_bias,
    )
    _refuse_unimplemented(config)
    layer_types = config.resolved_layer_types()
    if len(layer_types) != config.n_layers:
      raise ValueError(
          f'{len(layer_types)=} does not match {config.n_layers=}.'
      )
    self.blocks = [
        Qwen38Block(
            config=config,
            layer_idx=idx,
            layer_type=layer_type,
            sharding_config=self.sharding_config,
        )
        for idx, layer_type in enumerate(layer_types)
    ]
    # float32 for the same reason as the block norms; see `Qwen38Block.setup`.
    self.final_ln = model_lib.LayerNorm(
        dim=config.model_dim,
        use_bias=False,
        activation_dtype='float32',
        scale_plus_one=config.norm_scale_plus_one,
        epsilon=config.rms_norm_epsilon,
    )

  def init(self, prng_key: PRNGKey) -> Any:
    params = {}
    prng_key, embed_key = jax.random.split(prng_key, num=2)
    params['embed_linear'] = self.embed_linear.init(embed_key)
    for i, block in enumerate(self.blocks):
      prng_key, block_key_prng = jax.random.split(prng_key, num=2)
      params[block_key(i)] = block.init(block_key_prng)
    params['final_ln'] = self.final_ln.init()
    return params

  def init_decode_state(
      self, max_seq_len: int, batch_size: int | None = None
  ) -> Any:
    """Opens a decode state for every layer.

    Args:
      max_seq_len: Rows the attention layers' KV cache must hold. The
        GatedDeltaNet layers ignore it: their state is constant in the length.
      batch_size: Sequences; `config.batch_size` when omitted, which is what
        core's decode path assumes.

    Returns:
      `{'block_i': <the layer's state>}`.
    """
    if batch_size is None:
      batch_size = self.config.batch_size
    return {
        block_key(i): block.init_decode_state(batch_size, max_seq_len)
        for i, block in enumerate(self.blocks)
    }

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: Any,
      x: Array,
      *,
      segment_ids: Array | None = None,
      segment_positions: Array | None = None,
      extra_inputs: Any = None,
      decode_state: Any = None,
  ) -> tuple[Array, Any]:
    if extra_inputs is None:
      extra_inputs = {}
    extra_input_map = cast(Mapping[str, Any], extra_inputs)
    batch_size, seq_len = x.shape
    if segment_positions is None:
      if decode_state is not None:
        raise ValueError(
            'segment_positions is required when a decode_state is passed: the'
            ' attention cache and the GatedDeltaNet recurrence both restart a'
            ' row whose first non-pad token has position 0, so the 0..T-1'
            ' default would wipe the state on every decode step.'
        )
      segment_positions = einops.repeat(
          jnp.arange(seq_len), 'l -> b l', b=batch_size
      )
    if segment_ids is None:
      segment_ids = jnp.ones_like(segment_positions)

    # `sampling_lib` prefills the whole window with `prefill_position` set and
    # no state, then re-feeds the tokens from that position one at a time. The
    # GatedDeltaNet absorbs tokens into a recurrence in order rather than by
    # position, so both the pad suffix and the re-fed token would be counted
    # twice: absorb only the tokens before `prefill_position` and mark the rest
    # as padding, which is an exact identity for both mixers.
    prefill_position = extra_input_map.get('prefill_position')
    if decode_state is None and prefill_position is not None:
      if jnp.ndim(prefill_position) != 0:
        raise ValueError(
            "extra_inputs['prefill_position'] must be a scalar (core passes"
            f' one); got shape {jnp.shape(prefill_position)}. A per-row value'
            ' would broadcast against the sequence axis and mask nothing.'
        )
      # At the prefill window's length: core's mapping KV cache replaces
      # itself on a multi-token pass, so it must be exactly as wide as the
      # window, and `LMInterface` grows it with `pad_block_decode_state`
      # before it decodes into it.
      decode_state = self.init_decode_state(seq_len, batch_size=batch_size)
      segment_ids = jnp.where(
          jnp.arange(seq_len) < prefill_position, segment_ids, 0
      )

    sharding_config = cast(SimplyConfig, self.sharding_config)
    x = sharding_lib.with_sharding_constraint(
        jnp.asarray(x), sharding_config.data_partition
    )
    segment_ids = sharding_lib.with_sharding_constraint(
        jnp.asarray(segment_ids), sharding_config.data_partition
    )
    segment_positions = sharding_lib.with_sharding_constraint(
        jnp.asarray(segment_positions), sharding_config.data_partition
    )
    x = self.embed_linear.embed(params['embed_linear'], x)

    new_decode_state = {} if decode_state is not None else None
    remat = self.config.use_remat and decode_state is None
    run_kwargs = dict(
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        extra_inputs=extra_inputs,
    )
    if self.config.use_scan:
      raise ValueError(
          'Qwen3.8 has no scanned stack; `config.use_scan` is pinned off.'
      )
    x, states = self._run_blocks(
        params, x, decode_state, remat=remat, **run_kwargs
    )
    if new_decode_state is not None:
      new_decode_state.update(states)

    x = self.final_ln.apply(params['final_ln'], x)
    logits = self.embed_linear.apply(params['embed_linear'], x)
    if self.config.output_logits_soft_cap > 0:
      logits = model_lib.soft_cap(logits, self.config.output_logits_soft_cap)
    extra_output = {}
    if new_decode_state is not None:
      extra_output['decode_state'] = new_decode_state
    return logits, extra_output

  def _apply_fn(self, block: Qwen38Block, remat: bool) -> Any:
    """`block.apply`, rematerialised when this is not a decode pass."""
    if not remat:
      return block.apply
    name = (
        'nothing_saveable'
        if self.config.remat_policy == 'full'
        else self.config.remat_policy
    )
    policy = getattr(jax.checkpoint_policies, name, None)
    if policy is None:
      raise ValueError(
          f'Unknown remat_policy={name!r}; `jax.checkpoint_policies` has no'
          ' such policy, and defaulting to None would silently rematerialise'
          ' everything.'
      )
    return jax.remat(block.apply, policy=policy)

  def _run_blocks(
      self,
      params: Any,
      x: Array,
      decode_state: Any,
      *,
      remat: bool,
      **kwargs: Any,
  ) -> tuple[Array, dict[str, Any]]:
    states = {}
    state_map = cast(Mapping[str, Any], decode_state or {})
    for i, block in enumerate(self.blocks):
      key = block_key(i)
      x, extra = self._apply_fn(block, remat)(
          params[key], x, decode_state=state_map.get(key), **kwargs
      )
      if decode_state is not None:
        states[key] = extra['decode_state']
    return x, states

# Fields of core's `BaseExperimentConfig` that this port ignores, and the value
# they must have for it to be ignoring nothing. Without this, a config built by
# `dataclasses.replace(qwen3p8_27b(), use_moe=True)` constructs fine and quietly
# builds a dense FFN -- exactly the class of silent divergence the package's
# tests are written to prevent.
_UNIMPLEMENTED_FIELDS = (
    ('use_moe', False, 'Qwen3.8-27B is dense; there is no expert FFN here.'),
    ('attn_soft_cap', -1.0, 'The release has no attention soft cap.'),
    ('window_size', 0, 'The release has no sliding-window attention.'),
    ('use_post_ln', False, 'The release norms are pre-norm only.'),
    ('use_per_dim_scale', False, 'The release has no per-dim scale.'),
    ('ffn_use_bias', False, 'No bias anywhere in the release.'),
    ('qkv_use_bias', False, 'No bias anywhere in the release.'),
    ('output_layer_use_bias', False, 'No bias anywhere in the release.'),
    ('use_tied_embedding', False, 'Qwen3.8-27B has an untied head.'),
    ('ffn_weight_quant', '', 'Quantization is not implemented here.'),
    ('kv_cache_quant', '', 'Quantization is not implemented here.'),
)


def _refuse_unimplemented(config: SimplyConfig) -> None:
  """Raises if the config asks for something this port does not implement.

  Args:
    config: The experiment config.

  Raises:
    ValueError: naming the field, the value it must have, and why.
  """
  for name, expected, reason in _UNIMPLEMENTED_FIELDS:
    value = getattr(config, name, expected)
    if value != expected:
      raise ValueError(
          f'{name}={value!r} is not implemented by simply/zoo/qwen3p8;'
          f' it must be {expected!r}. {reason}'
      )


@model_lib.pad_block_decode_state.register
def _pad_gdn_decode_state(
    state: gdn_lib.GatedDeltaNetDecodeState, length_to_pad: int
) -> gdn_lib.GatedDeltaNetDecodeState:
  """The conv window and the recurrent state are constant in the length."""
  del length_to_pad
  return state

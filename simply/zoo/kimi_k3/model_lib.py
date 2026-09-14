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
"""Kimi K3 decoder block and language model.

Assembles the pieces of the K3 text backbone -- KDA / gated MLA token mixers,
Stable LatentMoE channel mixing and Block Attention Residuals -- following
`KimiDecoderLayer._forward_attn_residual` and `KimiLinearModel.forward` in the
HuggingFace release.
"""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Mapping, cast

import einops
import jax
import jax.numpy as jnp
from simply import model_lib
from simply.utils import common
from simply.utils import module
from simply.utils import sharding as sharding_lib
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3.utils import attn_res as attn_res_lib
from simply.zoo.kimi_k3.utils import kda as kda_lib
from simply.zoo.kimi_k3.utils import mla as mla_lib
from simply.zoo.kimi_k3.utils import moe as moe_lib

Array = common.Array
PyTree = common.PyTree
PRNGKey = jax.typing.ArrayLike
SimplyConfig = Any
AttnResState = attn_res_lib.AttnResState


def _block_key(layer_idx: int) -> str:
  """Key of one layer's parameters in the model's tree."""
  return f'block_{layer_idx}'


def num_attn_res_slots(n_layers: int, block_size: int) -> int:
  """Number of block snapshots: one per `block_size` layers, layer 0 included.

  Args:
    n_layers: layers in the stack.
    block_size: layers per AttnRes block.

  Returns:
    The number of snapshot slots the buffer needs.
  """
  return (n_layers + block_size - 1) // block_size


def _attn_res_partition(
    sharding_config: SimplyConfig | None,
) -> common.PartitionAnnotation:
  """`[B, T, slots, D]` annotation for the snapshot buffer.

  Every snapshot is a copy of the residual stream, so the buffer follows the
  activation sharding on the batch, sequence and feature axes and replicates
  the (tiny) slot axis. Keeping the SEQ axis matters: replicating it would
  all-gather `x` on every push and re-shard every mixture output, on a
  seq-sharded mesh at exactly the context lengths the annotation protects.

  Args:
    sharding_config: the config the model runs under, or None to leave the
      buffer unannotated.

  Returns:
    The 4-axis annotation, or None.
  """
  activation = (
      None if sharding_config is None else sharding_config.activation_partition
  )
  if activation is None:
    return None
  batch, seq, model = activation
  return (batch, seq, None, model)


@module.ModuleRegistry.register
@dataclasses.dataclass
class KimiK3Block(module.SimplyModule):
  """One K3 decoder layer: AttnRes read, token mixer, AttnRes read, FFN.

  The residual stream is not a plain sum: `x` carries the partial sum since the
  last snapshot, and both sublayers read a mixture of that partial sum with the
  earlier block snapshots (`attn_res_lib`). On a snapshot layer the incoming
  stream is pushed and the partial sum restarts from the attention output,
  which is what makes the stream reachable only through the mixture.
  """

  config: SimplyConfig
  layer_idx: int
  layer_type: str
  sharding_config: SimplyConfig | None = None

  @property
  def is_snapshot_layer(self) -> bool:
    return self.layer_idx % self.config.attn_res_block_size == 0

  @property
  def is_moe_layer(self) -> bool:
    return (
        self.config.use_moe
        and self.layer_idx >= self.config.first_k_dense_replace
    )

  def setup(self) -> None:
    config = self.config
    activation_dtype = config.activation_dtype_name
    norm_kwargs = dict(
        dim=config.model_dim,
        use_bias=False,
        scale_plus_one=False,
        epsilon=config.rms_norm_epsilon,
        activation_dtype=activation_dtype,
    )
    self.input_layernorm = model_lib.LayerNorm(**norm_kwargs)
    self.post_attention_layernorm = model_lib.LayerNorm(**norm_kwargs)
    self.attn_res_self = attn_res_lib.KimiK3AttnResMix(
        dim=config.model_dim,
        epsilon=config.rms_norm_epsilon,
        activation_dtype=activation_dtype,
    )
    self.attn_res_mlp = attn_res_lib.KimiK3AttnResMix(
        dim=config.model_dim,
        epsilon=config.rms_norm_epsilon,
        activation_dtype=activation_dtype,
    )

    if self.layer_type == k3_config_lib.LINEAR_ATTENTION:
      self.token_mixer = kda_lib.KimiK3DeltaAttention(
          model_dim=config.model_dim,
          num_heads=config.kda_num_heads,
          head_dim=config.kda_head_dim,
          conv_kernel_dim=config.kda_conv_kernel_dim,
          gate_lora_rank=config.kda_gate_lora_rank,
          gate_lower_bound=config.kda_gate_lower_bound,
          chunk_size=config.kda_chunk_size,
          rms_norm_epsilon=config.rms_norm_epsilon,
          activation_dtype=activation_dtype,
          sharding_config=self.sharding_config,
      )
    elif self.layer_type == k3_config_lib.FULL_ATTENTION:
      self.token_mixer = mla_lib.KimiK3MLA(
          model_dim=config.model_dim,
          n_heads=config.n_heads,
          q_lora_rank=config.q_lora_rank,
          kv_lora_rank=config.kv_lora_rank,
          qk_nope_head_dim=config.qk_nope_head_dim,
          qk_rope_head_dim=config.qk_rope_head_dim,
          v_head_dim=config.v_head_dim,
          use_output_gate=config.mla_use_output_gate,
          rms_norm_epsilon=config.rms_norm_epsilon,
          activation_dtype=activation_dtype,
          sharding_config=self.sharding_config,
      )
    else:
      raise ValueError(f'Unsupported Kimi K3 {self.layer_type=}.')

    if self.is_moe_layer:
      self.ffn = moe_lib.KimiK3LatentMoE(
          model_dim=config.model_dim,
          latent_dim=config.routed_expert_latent_dim,
          moe_intermediate_size=config.moe_intermediate_size,
          shared_expert_dim=config.shared_expert_intermediate_size,
          num_experts=config.num_experts,
          num_experts_per_token=config.num_experts_per_token,
          routed_scaling_factor=config.routed_scaling_factor,
          renormalize=config.moe_renormalize,
          use_latent_norm=config.latent_moe_use_norm,
          situ_beta=config.situ_beta,
          situ_linear_beta=config.situ_linear_beta,
          rms_norm_epsilon=config.rms_norm_epsilon,
          activation_dtype=activation_dtype,
          expert_dispatch=config.moe_expert_dispatch,
          gmm_impl=config.gmm_impl,
          expert_parallel_axis=config.moe_expert_parallel_axis,
          sharding_config=self.sharding_config,
      )
    else:
      self.ffn = moe_lib.KimiK3DenseMLP(
          model_dim=config.model_dim,
          expand_dim=config.ffn_expand_dim,
          situ_beta=config.situ_beta,
          situ_linear_beta=config.situ_linear_beta,
          activation_dtype=activation_dtype,
          sharding_config=self.sharding_config,
      )

  def init(self, prng_key: PRNGKey) -> Any:
    mixer_key, ffn_key = jax.random.split(prng_key, num=2)
    return {
        'input_layernorm': self.input_layernorm.init(),
        'post_attention_layernorm': self.post_attention_layernorm.init(),
        'attn_res_self': self.attn_res_self.init(),
        'attn_res_mlp': self.attn_res_mlp.init(),
        'token_mixer': self.token_mixer.init(mixer_key),
        'ffn': self.ffn.init(ffn_key),
    }

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: Any,
      x: Array,
      *,
      attn_res_state: AttnResState,
      segment_ids: Array,
      segment_positions: Array,
      extra_inputs: Any = None,
      decode_state: Any = None,
  ) -> tuple[Array, Any]:
    """Runs the layer; `x` is the partial residual sum since the last snapshot.

    Args:
      params: this layer's parameters.
      x: the partial residual sum since the last snapshot.
      attn_res_state: the AttnRes snapshot buffer.
      segment_ids: `[B, T]`; 0 marks padding.
      segment_positions: `[B, T]` per-token positions.
      extra_inputs: forwarded to the token mixer.
      decode_state: this layer's cache, or None outside decoding.

    Returns:
      The layer output and its extra outputs.
    """
    inputs_mask = segment_ids != 0
    attn_in = self.attn_res_self.apply(
        params['attn_res_self'], x, attn_res_state
    )
    attn_res_state, x = attn_res_lib.maybe_push_snapshot(
        attn_res_state, x, self.is_snapshot_layer
    )

    attn_in = self.input_layernorm.apply(params['input_layernorm'], attn_in)
    mixed, mixer_extra = self.token_mixer.apply(
        params['token_mixer'],
        attn_in,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        extra_inputs=extra_inputs,
        decode_state=decode_state,
    )
    x = x + mixed
    if self.sharding_config is not None:
      x = sharding_lib.with_sharding_constraint(
          jnp.asarray(x), self.sharding_config.activation_partition
      )

    ffn_in = self.attn_res_mlp.apply(params['attn_res_mlp'], x, attn_res_state)
    ffn_in = self.post_attention_layernorm.apply(
        params['post_attention_layernorm'], ffn_in
    )
    ffn_out, ffn_extra = self.ffn.apply(
        params['ffn'], ffn_in, inputs_mask=inputs_mask
    )
    x = x + ffn_out
    if self.sharding_config is not None:
      x = sharding_lib.with_sharding_constraint(
          jnp.asarray(x), self.sharding_config.activation_partition
      )

    extra_output = {
        'attn_res_state': attn_res_state,
        'decode_state': mixer_extra.get('decode_state'),
    }
    if ffn_extra:
      extra_output['ffn'] = ffn_extra
    return x, extra_output

  def init_decode_state(self, batch_size: int, max_seq_len: int) -> Any:
    return self.token_mixer.init_decode_state(batch_size, max_seq_len)


@module.ModuleRegistry.register
@dataclasses.dataclass
class KimiK3LM(module.SimplyModule):
  """The Kimi K3 text backbone."""

  config: SimplyConfig
  sharding_config: SimplyConfig | None = None

  def setup(self) -> None:
    config = self.config
    if self.sharding_config is None:
      self.sharding_config = config.sharding_config
    sharding_config = self.sharding_config
    activation_dtype = config.activation_dtype_name

    self.embed_linear = module.EmbeddingLinear(
        vocab_size=config.vocab_size,
        dim=config.model_dim,
        weight_partition=sharding_config.embed_partition,
        activation_dtype=activation_dtype,
        embedding_scale_by_sqrt_dim=config.embedding_lookup_scale,
        use_tied_embedding=config.use_tied_embedding,
        use_bias=config.output_layer_use_bias,
    )
    self.layer_types = config.resolved_layer_types()
    self.blocks = [
        KimiK3Block(
            config=config,
            layer_idx=i,
            layer_type=layer_type,
            sharding_config=sharding_config,
        )
        for i, layer_type in enumerate(self.layer_types)
    ]
    self.final_attn_res = attn_res_lib.KimiK3AttnResMix(
        dim=config.model_dim,
        epsilon=config.rms_norm_epsilon,
        activation_dtype=activation_dtype,
    )
    self.final_ln = model_lib.LayerNorm(
        dim=config.model_dim,
        use_bias=False,
        scale_plus_one=False,
        epsilon=config.rms_norm_epsilon,
        activation_dtype=activation_dtype,
    )
    self.num_slots = num_attn_res_slots(
        config.n_layers, config.attn_res_block_size
    )

  def init(self, prng_key: PRNGKey) -> Any:
    params = {}
    prng_key, embed_key = jax.random.split(prng_key, num=2)
    params['embed_linear'] = self.embed_linear.init(embed_key)
    per_layer = []
    for block in self.blocks:
      prng_key, block_key = jax.random.split(prng_key, num=2)
      per_layer.append(block.init(block_key))
    params.update(self._layer_tree(per_layer))
    params['final_attn_res'] = self.final_attn_res.init()
    params['final_ln'] = self.final_ln.init()
    return params

  def _layer_tree(self, per_layer: Sequence[Any]) -> dict[str, Any]:
    """Keys one per-layer value each by the layer it belongs to.

    Args:
      per_layer: one value per layer, in layer order.

    Returns:
      The tree to store under the model's top level.
    """
    del self  # The layout is positional.
    return {_block_key(i): v for i, v in enumerate(per_layer)}

  def _run_layer(
      self,
      block: KimiK3Block,
      block_params: Any,
      x: Array,
      attn_res_state: AttnResState,
      block_decode_state: Any,
      *,
      remat: bool,
      segment_ids: Array,
      segment_positions: Array,
      extra_inputs: Any,
  ) -> tuple[Array, dict[str, Any]]:
    """One layer."""
    apply_fn = block.apply
    if remat:
      apply_fn = jax.remat(
          apply_fn,
          policy=getattr(
              jax.checkpoint_policies,
              'nothing_saveable'
              if self.config.remat_policy == 'full'
              else self.config.remat_policy,
              None,
          ),
          static_argnums=(),
      )
    return apply_fn(
        block_params,
        x,
        attn_res_state=attn_res_state,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        extra_inputs=extra_inputs,
        decode_state=block_decode_state,
    )

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
            'segment_positions is required when a decode_state is passed: both'
            ' caches restart a row whose non-pad token has position 0, so the'
            ' 0..T-1 default would wipe the cache on every decode step.'
        )
      segment_positions = einops.repeat(
          jnp.arange(seq_len), 'l -> b l', b=batch_size
      )
    if segment_ids is None:
      segment_ids = jnp.ones_like(segment_positions)

    # `sampling_lib`'s prefill opens the cache with `prefill_position` and no
    # state, then re-feeds tokens `prefill_position ...` one at a time. K3's
    # caches are order-indexed (MLA appends, KDA absorbs into a recurrence), not
    # position-indexed, so re-feeding is a duplicate the state cannot undo:
    # absorb only the tokens before `prefill_position` and mark the rest as
    # padding, which is an exact identity for both mixers.
    prefill_position = extra_input_map.get('prefill_position')
    if decode_state is None and prefill_position is not None:
      decode_state = self.init_decode_state(
          self._decode_cache_len(extra_input_map, seq_len),
          batch_size=batch_size,
      )
      segment_ids = jnp.where(
          jnp.arange(seq_len) < prefill_position, segment_ids, 0
      )

    sharding_config = cast(SimplyConfig, self.sharding_config)
    x = sharding_lib.with_sharding_constraint(
        jnp.asarray(x), sharding_config.data_partition
    )
    x = self.embed_linear.embed(params['embed_linear'], x)

    attn_res_state = attn_res_lib.init_attn_res_state(
        batch_size=x.shape[0],
        seq_len=x.shape[1],
        model_dim=x.shape[2],
        num_slots=self.num_slots,
        dtype=x.dtype,
        partition=_attn_res_partition(sharding_config),
    )

    extra_output = {}
    new_decode_state = {} if decode_state is not None else None
    run_layer = functools.partial(
        self._run_layer,
        remat=self.config.use_remat and decode_state is None,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        extra_inputs=extra_inputs,
    )
    state_map = cast(Mapping[str, Any], decode_state or {})

    def run_unrolled(layers, x, attn_res_state):
      for i in layers:
        key = _block_key(i)
        x, block_extra = run_layer(
            self.blocks[i], params[key], x, attn_res_state, state_map.get(key)
        )
        attn_res_state = block_extra['attn_res_state']
        if new_decode_state is not None:
          new_decode_state[key] = block_extra['decode_state']
        if 'ffn' in block_extra:
          extra_output[key] = {'ffn': block_extra['ffn']}
      return x, attn_res_state

    x, attn_res_state = run_unrolled(range(len(self.blocks)), x, attn_res_state)
    if new_decode_state is not None:
      extra_output['decode_state'] = new_decode_state

    x = self.final_attn_res.apply(params['final_attn_res'], x, attn_res_state)
    x = self.final_ln.apply(params['final_ln'], x)
    if extra_input_map.get('return_hidden_only'):
      return x, extra_output
    if extra_input_map.get('return_last_logits_only'):
      x = x[:, -1:, :]
    logits = self.embed_linear.apply(params['embed_linear'], x)
    if self.config.output_logits_soft_cap > 0:
      logits = model_lib.soft_cap(logits, self.config.output_logits_soft_cap)
    return logits, extra_output

  def init_decode_state(
      self, max_seq_len: int, batch_size: int | None = None
  ) -> Any:
    """Empty caches for `batch_size` rows of up to `max_seq_len` tokens.

    The AttnRes snapshots are not part of this state: the mixture is strictly
    per token, so each call rebuilds them from its own residual stream (shape
    `[B, T, slots, D]`, i.e. `[B, 1, slots, D]` on a decode step).

    Args:
      max_seq_len: cache capacity, in tokens, per row. Only MLA grows with it;
        KDA's state is constant in the sequence length.
      batch_size: number of rows; defaults to the config's batch size.

    Returns:
      The per-layer decode state, keyed as the parameter tree is.
    """
    if batch_size is None:
      batch_size = self.config.batch_size
    return self._layer_tree([
        block.init_decode_state(batch_size, max_seq_len)
        for block in self.blocks
    ])

  def _decode_cache_len(
      self, extra_inputs: Mapping[str, Any], seq_len: int
  ) -> int:
    """Cache capacity to open a decode state with, at prefill.

    The prefill window, not `config.seq_len`: `LMInterface` grows the state to
    each chunk's horizon (`pad_block_decode_state`) before decoding into it, so
    opening at the sampler's *maximum* horizon would reserve the whole 32k-row
    cache for a run that decodes 1k tokens -- 22.6 GiB/device at batch 64 on
    4x4x8, of which two thirds is never written. `TransformerLM` opens its
    KV cache at the prefill length for the same reason
    (`model_lib.updated_decode_state`).

    A caller that drives `apply` itself and does *not* grow the state must say
    how far it intends to decode, via `extra_inputs['decode_max_seq_len']` (a
    Python int, read at trace time); the MLA cache saturates rather than grows,
    so an unannounced overflow would silently drop tokens.

    Args:
      extra_inputs: the call's extra inputs.
      seq_len: tokens in this prefill call.

    Returns:
      The number of cache rows to allocate per sequence.

    Raises:
      ValueError: if `decode_max_seq_len` is smaller than the prefill.
    """
    requested = extra_inputs.get('decode_max_seq_len')
    if requested is None:
      return seq_len
    max_seq_len = int(requested)
    if max_seq_len < seq_len:
      raise ValueError(
          f'A {seq_len}-token prefill does not fit a {max_seq_len}-token decode'
          " cache; raise `extra_inputs['decode_max_seq_len']`. The MLA cache"
          ' saturates instead of growing, which would silently drop tokens.'
      )
    return max_seq_len


@model_lib.pad_block_decode_state.register
def _pad_mla_decode_state(
    state: mla_lib.KimiK3MLADecodeState, length_to_pad: int
) -> mla_lib.KimiK3MLADecodeState:
  """Grows the latent cache to `length_to_pad` rows; never shrinks it.

  `LMInterface` pads the decode state to the sampler's horizon before every
  decode chunk, so honouring it here is what keeps an append from saturating.
  A scanned model hands over the whole group slot at once, `[n_repeats, B, S,
  C]`; the row axis is the second-to-last either way.
  """
  return dataclasses.replace(
      state,
      kv_cache=model_lib.pad_to_along_axis(
          state.kv_cache, length_to_pad, axis=state.kv_cache.ndim - 2
      ),
  )


@model_lib.pad_block_decode_state.register
def _pad_kda_decode_state(
    state: kda_lib.KDADecodeState, length_to_pad: int
) -> kda_lib.KDADecodeState:
  """KDA's conv window and recurrent state are constant in the sequence length."""
  del length_to_pad
  return state

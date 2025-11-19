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
"""All modeling components including architecture, training and inference."""

import collections
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import copy
import dataclasses
import functools
import time
from typing import Any, cast, ClassVar, Self, Tuple
import warnings

from absl import logging
import einops
import jax
from jax.experimental import shard_map
from jax.experimental import xla_metadata
from jax.experimental.pallas.ops.tpu import megablox
from jax.experimental.pallas.ops.tpu import splash_attention
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply import config_lib
from simply import data_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import distributions
from simply.utils import experiment_helper as exp_helper
from simply.utils import initializer
from simply.utils import module
from simply.utils import optimizers as opt_lib
from simply.utils import pytree
from simply.utils import registry
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib
from simply.utils import tokenization

################################################################################
## Type aliases.
Batch = MutableMapping[str, np.ndarray | jnp.ndarray]
DTypeLike = jax.typing.DTypeLike
PRNGKey = jax.typing.ArrayLike
PartitionAnnotation = common.PartitionAnnotation
PyTree = common.PyTree
SimplyConfig = Any
SimplyModule = module.SimplyModule
Array = common.Array
RawT = common.RawT
get_default_mesh = sharding_lib.get_default_mesh
maybe_dequantize_array = common.convert_or_dequantize
mesh_sharding = sharding_lib.mesh_sharding
ExperimentHelper = exp_helper.ExperimentHelper
create_lr_schedule = opt_lib.create_lr_schedule
SamplingParams = sampling_lib.SamplingParams
EinsumLinear = module.EinsumLinear

# All the model parameters are wrapped into AnnotatedArray dataclass.
# Its `array` field holds the raw array and its `metadata` field
# holds the annotations.
# For example, `AnnotatedArray.create(x, metadata_a=1, metadata_b='yy')`
# will save x as the `array` field and the annotation `metadata_a=1` and
# `metadata_b='yy'` to the `metadata` field.
# AnnotatedArray is registered as PyTree node so the raw array will be
# treated as a leaf node unless you use
# `is_leaf=lambda x: isinstance(x, AnnotatedArray)` when traversing the PyTree.
AnnotatedArray = common.AnnotatedArray
# Use the `get_raw_arrays` function below to turn all the AnnotatedArray
# back to raw arrays in a PyTree like `get_raw_arrays(x)`.
get_raw_arrays = common.get_raw_arrays


################################################################################
# Initialization.


def xavier_init(prng_key, shape, dtype, in_dim, out_dim):
  logging.warning(
      'DEPRECATED: xavier_init is deprecated. Use XavierUniformInit instead.'
  )
  scale = jnp.sqrt(6 / (in_dim + out_dim))
  return jax.random.uniform(
      prng_key, shape, dtype=dtype, minval=-1.0, maxval=1.0) * jnp.array(
          scale, dtype=dtype)


################################################################################
# Architecture.


@registry.FunctionRegistry.register
def gelu(x: Array):
  return 0.5 * x * (1.0 + jnp.tanh(
      jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))


@registry.FunctionRegistry.register
def squared_relu(x: Array):
  return jnp.square(jax.nn.relu(x))


registry.FunctionRegistry.register(jax.nn.silu, 'silu')


def soft_cap(x: Array, cap: float):
  cap = jnp.asarray(cap, x.dtype)
  return jnp.asarray(cap * jnp.tanh(x / cap), x.dtype)


@module.ModuleRegistry.register
@dataclasses.dataclass
class LayerNorm(module.SimplyModule):
  """Layer normalization layer (can be also configured as RMSNorm)."""
  dim: int
  axis: int = -1
  use_bias: bool = True  # Set to False if want to use RMSNorm.
  use_scale: bool = True
  # Mixed precision related.
  weight_dtype: DTypeLike = 'float32'
  activation_dtype: DTypeLike = 'bfloat16'
  scale_plus_one: bool = True
  # Sharding related.
  scale_partition: PartitionAnnotation = None
  bias_partition: PartitionAnnotation = None
  # Others.
  epsilon: float = 1e-6

  def init(self, prng_key: PRNGKey | None = None) -> PyTree:
    del prng_key
    assert self.use_bias or self.use_scale
    params = {}
    if self.use_bias:
      params['bias'] = jnp.zeros(self.dim, dtype=self.weight_dtype)
      params['bias'] = sharding_lib.with_sharding_constraint(
          params['bias'], self.bias_partition
      )
      params['bias'] = AnnotatedArray.create(
          params['bias'], dim_annotation='h')
    if self.use_scale:
      params['scale'] = (
          jnp.zeros(self.dim, dtype=self.weight_dtype)
          if self.scale_plus_one
          else jnp.ones(self.dim, dtype=self.weight_dtype)
      )
      params['scale'] = sharding_lib.with_sharding_constraint(
          params['scale'], self.scale_partition
      )
      params['scale'] = AnnotatedArray.create(
          params['scale'], dim_annotation='h')
    return params

  def apply(self, params: PyTree, x: Array) -> Array:
    params = get_raw_arrays(params)
    inputs_dtype = x.dtype
    # Perform reduction in float32 for better stability.
    x = x.astype(jnp.float32)
    if self.use_bias:
      mean = jnp.mean(x, axis=self.axis, keepdims=True)
      x -= mean
    if self.use_scale:
      var = jnp.mean(jnp.square(x), axis=self.axis, keepdims=True)
      x *= jax.lax.rsqrt(var + self.epsilon)
      x = jnp.asarray(x, self.activation_dtype)
      scale = common.convert_or_dequantize(
          params['scale'], dtype=self.activation_dtype
      )
      if self.scale_plus_one:
        x *= scale + jnp.array(1.0, dtype=self.activation_dtype)
      else:
        x *= scale
    x = x.astype(self.activation_dtype)
    if self.use_bias:
      x += common.convert_or_dequantize(
          params['bias'], dtype=self.activation_dtype
      )
    return x.astype(inputs_dtype)


@module.ModuleRegistry.register
@dataclasses.dataclass
class PerDimScale(module.SimplyModule):
  """Layer to scale individual dims of the input."""
  dim: int
  axis: int = -1
  # Mixed precision related.
  weight_dtype: DTypeLike = 'float32'
  activation_dtype: DTypeLike = 'bfloat16'

  def init(self, prng_key: PRNGKey | None = None) -> PyTree:
    params = {}
    params['scale'] = jnp.zeros(self.dim, dtype=self.weight_dtype)
    params['scale'] = AnnotatedArray.create(
        params['scale'], dim_annotation='h')
    return params

  def apply(self, params: PyTree, x: Array) -> Array:
    params = get_raw_arrays(params)
    r_softplus_0 = 1.442695041
    scaling_factor = jnp.array(
        r_softplus_0 / jnp.sqrt(self.dim), dtype=self.activation_dtype)
    scaling_factor *= jax.nn.softplus(params['scale'])
    x *= scaling_factor
    return x


def neg_inf(dtype: DTypeLike) -> float:
  if np.issubdtype(dtype, np.inexact):
    dtype_max = np.finfo(dtype).max
  elif np.issubdtype(dtype, np.integer):
    dtype_max = np.iinfo(dtype).max
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')
  # NOTE: Gemma uses -0.7 * dtype_max
  return -0.5 * dtype_max


def rotary_positional_embedding(
    embedding_mat,
    segment_positions=None,
    min_timescale=1,
    max_timescale=10_000,
    scale_factor=1.0,
):
  embedding_dims = embedding_mat.shape[-1]
  half_embedding_dim = embedding_dims // 2
  fraction = 2 * jnp.arange(0, half_embedding_dim) / embedding_dims
  timescale = min_timescale * (max_timescale / min_timescale)**fraction
  query_segment_pos = segment_positions
  if query_segment_pos is None:
    seq_length = embedding_mat.shape[1]
    query_segment_pos = jnp.arange(
        seq_length, dtype=jnp.float32)[jnp.newaxis, :]
  else:
    query_segment_pos = jnp.asarray(query_segment_pos, dtype=jnp.float32)
  query_segment_pos = query_segment_pos[:, :, jnp.newaxis, jnp.newaxis]
  timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
  sinusoid_inp = query_segment_pos / timescale / scale_factor
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)
  # Convert to float32.
  embedding_dtype = embedding_mat.dtype
  embedding_mat = jnp.asarray(embedding_mat, jnp.float32)
  first_half, second_half = jnp.split(embedding_mat, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  embedding_mat = jnp.concatenate([first_part, second_part], axis=-1)
  # Convert back to original dtype.
  embedding_mat = jnp.asarray(embedding_mat, embedding_dtype)
  return embedding_mat


def updated_decode_state(
    k: Array,
    v: Array,
    segment_positions: Array,
    segment_ids: Array,
    decode_state: PyTree,
    window_size: int = 0,
) -> tuple[Array, Array, Array, Array, PyTree]:
  """Updates decode state when decode_state is not None.

  decode_state caches the key, value, segment_positions, segment_ids for all
  previous decode steps. The input key, value, segment_positions, segment_ids
  will be written at the cache_position in the decode_state caches. decode_state
  also contains 'prefill_position' at prefill stage, which is the position where
  the decoding will starts. This is used to properly truncate the cache to
  corresponding window.

  Args:
    k: The key of this step.
    v: The value of this step.
    segment_positions: The segment positions of this step.
    segment_ids: The segment ids of this step.
    decode_state: The decode state cache mapping to be updated. It is set to
      None when not decoding.
    window_size: The size of the sliding window. If greater than 0, the cache
      will be updated with the sliding window.

  Returns:
    The updated cached k, v, segment_positions, segment_ids, decode_state. They
    contain the information of the current step and previous steps. The returned
    decode_state also contains 'window_size=...' as key to store window_size as
    metadata, so that decoding can use this information to properly truncate the
    cache.
  """
  if decode_state is None:
    return k, v, segment_positions, segment_ids, decode_state

  decode_state = cast(Mapping[str, Any], decode_state)
  new_decode_state = None
  if 'k' in decode_state and 'v' in decode_state:
    k_cache = decode_state['k']
    v_cache = decode_state['v']
    # Assume that we are dealing with one decode step.
    assert segment_positions.shape[1] == 1
    # Assume that all the tokens in the batch share the same position.
    position = segment_positions[0][0]
    cache_position = position
    if window_size > 0:
      cache_position = cache_position % (window_size + 1)
    # Insert the new key and value at the cache_position.
    k = jax.lax.dynamic_update_slice_in_dim(k_cache, k, cache_position, axis=1)
    v = jax.lax.dynamic_update_slice_in_dim(v_cache, v, cache_position, axis=1)
    segment_positions = jax.lax.dynamic_update_slice_in_dim(
        decode_state['segment_positions'],
        segment_positions,
        cache_position,
        axis=1,
    )
    segment_ids = jax.lax.dynamic_update_slice_in_dim(
        decode_state['segment_ids'],
        segment_ids,
        cache_position,
        axis=1,
    )
  elif window_size > 0 and k.shape[1] > window_size + 1:
    # Properly truncate the cache to window_size.
    if (prefill_position := decode_state.get('prefill_position')) is None:
      raise ValueError(
          'prefill_position is required in decode_state when window_size > 0.'
      )

    def _windowized_array(x: Array) -> Array:
      return jax.lax.cond(
          prefill_position < window_size + 1,
          lambda: jax.lax.dynamic_slice_in_dim(x, 0, window_size + 1, axis=1),
          lambda: jnp.roll(
              jax.lax.dynamic_slice_in_dim(
                  x, prefill_position - window_size - 1, window_size + 1, axis=1
              ),
              prefill_position % (window_size + 1),
              axis=1,
          ),
      )

    new_decode_state = dict(
        k=_windowized_array(k),
        v=_windowized_array(v),
        segment_positions=_windowized_array(segment_positions),
        segment_ids=_windowized_array(segment_ids),
    )
  if new_decode_state is None:
    new_decode_state = dict(
        k=k, v=v, segment_positions=segment_positions, segment_ids=segment_ids
    )
  new_decode_state[f'window_size={window_size}'] = None
  return k, v, segment_positions, segment_ids, new_decode_state


def create_mask(
    segment_positions: Array,
    kv_segment_positions: Array,
    segment_ids: Array,
    kv_segment_ids: Array,
    window_size: int = 0,
) -> Array:
  """Create a mask for attention.

  Args:
    segment_positions: The segment positions.
    kv_segment_positions: The segment positions for the key and value.
    segment_ids: The segment ids.
    kv_segment_ids: The segment ids for the key and value.
    window_size: Attends how many tokens ahead (excluding self). Used when
      greater than 0 and use_causal is True.

  Returns:
    The mask in bool of shape [batch_size, seq_len, seq_len], with 1 as
    attendable and 0 as unattendabe.
  """
  kv_len = kv_segment_positions.shape[1]
  masks = []

  # Causal mask.
  a = einops.rearrange(segment_positions, 'b l -> b l 1')
  b = einops.rearrange(kv_segment_positions, 'b l -> b 1 l')
  causal_mask = a >= b
  masks.append(causal_mask)

  # Window mask.
  if window_size > 0 and window_size + 1 < kv_len:
    window_mask = a - b <= window_size
    masks.append(window_mask)

  # Segment mask.
  if segment_ids is not None:
    a = einops.rearrange(segment_ids, '... l -> ... l 1')
    b = einops.rearrange(kv_segment_ids, '... l -> ... 1 l')
    seg_mask = a == b
    masks.append(seg_mask)

  assert masks
  mask = masks[0]
  for m in masks[1:]:
    mask &= m
  return mask


def chunked_local_attn(
    q, k, v, mask, window_size, *, attn_soft_cap=50.0, dtype=jnp.bfloat16
):
  """Chunked local attention.

  It splits the sequence into chunks of size window_size and performs local
  attention within each chunk, i.e. query i-th chunk attends to key/value in
  (i-1)-th and i-th chunks, in order to reduce unnecessary computation.

  Args:
    q: The query in [batch_size, seq_len, num_heads, model_dim].
    k: The key in [batch_size, seq_len, num_heads, model_dim].
    v: The value in [batch_size, seq_len, num_heads, model_dim].
    mask: The mask in [batch_size, num_heads, seq_len, seq_len].
    window_size: The size of the sliding window.
    attn_soft_cap: The soft cap for the attention logits. Does not apply if
      negative.
    dtype: The dtype of the output.

  Returns:
    The output of the attention.
  """
  seq_len = k.shape[1]
  if seq_len % window_size != 0:
    # TODO: Support non-divisible case.
    raise ValueError(
        f'{seq_len=} must be a multiple of {window_size=}.'
    )
  chunked_q = einops.rearrange(q, 'b (c w) ... -> b c w ...', w=window_size)
  chunked_k = einops.rearrange(k, 'b (c w) ... -> b c w ...', w=window_size)
  chunked_v = einops.rearrange(v, 'b (c w) ... -> b c w ...', w=window_size)

  chunked_mask = einops.rearrange(
      mask,
      'b ... (c1 w1) (c2 w2) -> b c1 c2 ... w1 w2',
      w1=window_size,
      w2=window_size,
  )

  # output0: [batch_size, window_size, num_heads, model_dim]
  output0, _ = attn(
      chunked_q[:, 0],
      chunked_k[:, 0],
      chunked_v[:, 0],
      chunked_mask[:, 0, 0],
      attn_soft_cap=attn_soft_cap,
      dtype=dtype,
  )

  # Prepare k/v and mask for concantation of (i-1)-th and i-th chunks.
  # Chunked mask is implemented by taking the diagnal using einsum.
  # chunked_mask0 (current chunk) and chunked_mask1 (previous chunk):
  #   [batch_size, num_chunks-1, num_heads, window_size, window_size]
  chunked_mask0 = jnp.einsum('bcc...->bc...', chunked_mask[:, 1:, 1:])
  chunked_mask1 = jnp.einsum('bcc...->bc...', chunked_mask[:, 1:, :-1])
  # w2_chunked_mask:
  #   [batch_size, num_chunks-1, num_heads, window_size, 2*window_size]
  w2_chunked_mask = jnp.concat([chunked_mask1, chunked_mask0], axis=-1)
  # w2_chunked_k and w2_chunked_v:
  #   [batch_size, num_chunks-1, 2*window_size, num_heads, model_dim]
  w2_chunked_k = jnp.concat([chunked_k[:, :-1], chunked_k[:, 1:]], axis=2)
  w2_chunked_v = jnp.concat([chunked_v[:, :-1], chunked_v[:, 1:]], axis=2)

  # chunked_output1:
  #   [batch_size, num_chunks-1, window_size, num_heads, model_dim]
  chunked_output1, _ = attn(
      chunked_q[:, 1:],
      w2_chunked_k,
      w2_chunked_v,
      w2_chunked_mask,
      attn_soft_cap=attn_soft_cap,
      dtype=dtype,
  )
  # output1: [batch_size, (num_chunks-1)*window_size, num_heads, model_dim]
  output1 = einops.rearrange(chunked_output1, 'b c w ... -> b (c w) ...')

  # output: [batch_size, seq_len, num_heads, model_dim]
  output = jnp.concat([output0, output1], axis=1)
  return output


def attn(q, k, v, mask, *, attn_soft_cap=50.0, dtype='bfloat16'):
  group_axis = 'g' if len(q.shape) > len(k.shape) else ''
  # ...tgnh, ...snh -> ...gnts
  attn_logit_mat = jnp.einsum(
      f'...t{group_axis}hi,...qhi->...{group_axis}htq', q, k
  ).astype(jnp.float32)
  if attn_soft_cap > 0:
    attn_logit_mat = soft_cap(attn_logit_mat, attn_soft_cap)
  # NOTE: leaner and sampler logp diff can come from this process:
  # Sampler may use different seq len against learner
  # a. Intermediate decoding graduately extends seq len
  # b. Sliding window decoding keeps KV seq len as window_size + 1
  # In practice, we do not see it results in significant logp diff.
  attn_logit_mat = jnp.where(
      mask, attn_logit_mat, neg_inf(attn_logit_mat.dtype)
  )
  attn_mat = jax.nn.softmax(attn_logit_mat, axis=-1)
  attn_mat = attn_mat.astype(dtype)
  output = jnp.einsum(
      f'...{group_axis}htq,...qhi->...t{group_axis}hi', attn_mat, v
  )
  return output, attn_mat


@module.ModuleRegistry.register
@dataclasses.dataclass
class FeedForward(module.SimplyModule):
  """The FeedForward block in Transformer."""

  model_dim: int
  expand_factor: int
  sharding_config: SimplyConfig
  use_gated_activation_in_ffn: bool = False
  # Mixed precision related.
  activation_dtype: DTypeLike = 'bfloat16'
  # Below are for experimental usage.
  ffn_expand_dim: int | None = None
  ffn_use_bias: bool = True
  ffn_activation: str = 'gelu'

  @property
  def expand_dim(self) -> int:
    if self.ffn_expand_dim is not None:
      return self.ffn_expand_dim
    return self.expand_factor * self.model_dim

  def setup(self) -> None:
    self.ffn_0 = EinsumLinear(
        eqn='io,...i->...o',
        weight_shape=[self.model_dim, self.expand_dim],
        bias_term='o' if self.ffn_use_bias else '',
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        weight_partition=self.sharding_config.ffn0_partition,
        output_partition=self.sharding_config.ffn0_activation_partition,
    )
    if self.use_gated_activation_in_ffn:
      self.ffn_0_gate = EinsumLinear(
          eqn='io,...i->...o',
          weight_shape=[self.model_dim, self.expand_dim],
          bias_term='o' if self.ffn_use_bias else '',
          # Mixed precision related.
          activation_dtype=self.activation_dtype,
          # Sharding related.
          weight_partition=self.sharding_config.ffn0_partition,
          output_partition=self.sharding_config.ffn0_activation_partition,
      )
    self.ffn_1 = EinsumLinear(
        eqn='io,...i->...o',
        weight_shape=[self.expand_dim, self.model_dim],
        bias_term='o' if self.ffn_use_bias else '',
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        weight_partition=self.sharding_config.ffn1_partition,
        output_partition=self.sharding_config.activation_partition,
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    ffn0_key, ffn0gate_key, ffn1_key = jax.random.split(prng_key, num=3)
    params['ffn_0'] = self.ffn_0.init(ffn0_key)
    if self.use_gated_activation_in_ffn:
      params['ffn_0_gate'] = self.ffn_0_gate.init(ffn0gate_key)
    params['ffn_1'] = self.ffn_1.init(ffn1_key)
    return params

  def apply(
      self,
      params: PyTree,
      x: Array,
      inputs_mask: Array | None = None,
  ) -> tuple[Array, PyTree]:
    del inputs_mask
    extra_output = {}
    projected_x = self.ffn_0.apply(params['ffn_0'], x)
    activation_fn = registry.FunctionRegistry.get(self.ffn_activation)
    if self.use_gated_activation_in_ffn:
      gate = self.ffn_0_gate.apply(params['ffn_0_gate'], x)
      x = jnp.asarray(activation_fn(gate), self.activation_dtype) * projected_x
    else:
      x = jnp.asarray(activation_fn(projected_x), self.activation_dtype)
    x = self.ffn_1.apply(params['ffn_1'], x)
    return x, extra_output


@module.ModuleRegistry.register
@dataclasses.dataclass
class MoEFeedForward(FeedForward):
  """A Mixture-of-Experts FeedForward block."""
  num_experts: int = 8
  num_experts_per_token: int = 2
  expert_capacity_factor: float | None = 1.0
  router_z_loss_weight: float = 0.0
  lbl_loss_weight: float = 0.0
  tile_batch_seq: int = 128
  tile_model_dim: int = 128
  tile_expand_dim: int = 128
  gmm_impl: str = 'ragged_dot'

  def setup(self):
    if self.ffn_use_bias:
      raise ValueError('MoEFeedForward does not support bias in FFN.')
    if self.sharding_config.activation_partition is None:
      router_output_partition = None
    else:
      router_output_partition = (
          self.sharding_config.activation_partition[0],
          self.sharding_config.activation_partition[1],
          None,
      )
    self.router = EinsumLinear(
        eqn='ie,...i->...e',
        weight_shape=[self.model_dim, self.num_experts],
        # Use float32 for router to avoid numerical issues.
        weight_dtype='float32',
        activation_dtype='float32',
        # Sharding related.
        weight_partition=(None, None),
        output_partition=router_output_partition,
    )
    self.ffn0_partition = (
        (None, *self.sharding_config.ffn0_partition)
        if self.sharding_config.ffn0_partition is not None else None
    )
    self.ffn_0 = EinsumLinear(
        eqn='eio,e...i->e...o',
        weight_shape=[self.num_experts, self.model_dim, self.expand_dim],
        weight_dim_annotation='.io',
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        weight_partition=self.ffn0_partition,
    )
    self.ffn1_partition = (
        (None, *self.sharding_config.ffn1_partition)
        if self.sharding_config.ffn1_partition is not None else None
    )
    if self.use_gated_activation_in_ffn:
      self.ffn_0_gate = EinsumLinear(
          eqn='eio,e...i->e...o',
          weight_dim_annotation='.io',
          weight_shape=[self.num_experts, self.model_dim, self.expand_dim],
          # Mixed precision related.
          activation_dtype=self.activation_dtype,
          # Sharding related.
          weight_partition=self.ffn0_partition,
      )
    self.ffn_1 = EinsumLinear(
        eqn='eio,e...i->e...o',
        weight_dim_annotation='.io',
        weight_shape=[self.num_experts, self.expand_dim, self.model_dim],
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        weight_partition=self.ffn1_partition,
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    router_key, ffn0_key, ffn0gate_key, ffn1_key = jax.random.split(
        prng_key, num=4
    )
    params['router'] = self.router.init(prng_key=router_key)
    params['ffn_0'] = self.ffn_0.init(ffn0_key)
    if self.use_gated_activation_in_ffn:
      params['ffn_0_gate'] = self.ffn_0_gate.init(ffn0gate_key)
    params['ffn_1'] = self.ffn_1.init(ffn1_key)
    return params

  def apply(
      self, params: PyTree, x: Array, inputs_mask: Array | None = None
  ) -> PyTree:
    inputs = x
    extra_output = {'loss': {}, 'metric': {}}
    params = get_raw_arrays(params)
    inputs = jnp.where(inputs_mask[..., None], inputs, 0.0)
    # router_logits: [batch_size, seq_len, num_experts]
    router_logits = self.router.apply(params['router'], inputs)
    router_logits = router_logits.astype(jnp.float32)
    # router_probs: [batch_size, seq_len, num_experts]
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    if self.num_experts_per_token == 1:
      # Apply `softmax => topk` when k == 1 to avoid zero gradient
      # on the router logits.
      # selected_router_probs, selected_indices:
      # [batch_size, seq_len, num_experts_per_token]
      selected_router_probs, selected_indices = jax.lax.top_k(
          router_probs, k=self.num_experts_per_token
      )
    else:
      # Perform `topk => softmax` to get a normalized probability distribution.
      # selected_router_logits, selected_indices:
      # [batch_size, seq_len, num_experts_per_token]
      selected_router_logits, selected_indices = jax.lax.top_k(
          router_logits, k=self.num_experts_per_token
      )
      selected_router_probs = jax.nn.softmax(selected_router_logits, axis=-1)
    selected_router_probs = jnp.asarray(
        selected_router_probs, self.activation_dtype
    )
    router_probs = jnp.asarray(router_probs, self.activation_dtype)
    if self.expert_capacity_factor is None:
      outputs, ffn_extra_output = self._apply_sparse_moe(
          params,
          inputs,
          selected_indices=selected_indices,
          selected_weights=selected_router_probs,
          inputs_mask=inputs_mask,
      )
    else:
      outputs, ffn_extra_output = self._apply_dense_moe(
          params,
          inputs,
          selected_indices=selected_indices,
          selected_weights=selected_router_probs,
          inputs_mask=inputs_mask,
      )
    load = ffn_extra_output['load']
    extra_output.update(ffn_extra_output)
    router_entropy = - jnp.sum(router_probs * jnp.where(
        router_probs > 0, jnp.log(router_probs), 0.0), axis=-1)
    extra_output['metric'].update({
        'max_load': jnp.max(load),
        'min_load': jnp.min(load),
        'router_entropy': (
            jnp.mean(router_entropy, where=inputs_mask)
        ),
        'gini': jnp.sum(load ** 2) * self.num_experts - 1,
    })
    if self.lbl_loss_weight > 0:
      if inputs_mask is None:
        inputs_mask = jnp.ones(shape=x.shape[:2], dtype=self.activation_dtype)
      else:
        inputs_mask = jnp.asarray(inputs_mask, dtype=self.activation_dtype)
      importance = jnp.einsum(
          'bse,bs->e', router_probs, inputs_mask
      ) / jnp.sum(inputs_mask)
      lbl_loss = (
          jnp.einsum('i,i->', jax.lax.stop_gradient(load), importance)
          * self.num_experts
      )
      extra_output['metric']['lbl_loss'] = lbl_loss
      extra_output['loss']['lbl_loss'] = lbl_loss * self.lbl_loss_weight
    if self.router_z_loss_weight > 0:
      z_loss = jnp.mean(
          jax.nn.logsumexp(router_logits, axis=-1) ** 2, where=inputs_mask)
      extra_output['metric']['z_loss'] = z_loss
      extra_output['loss']['z_loss'] = z_loss * self.router_z_loss_weight
    outputs = jnp.where(inputs_mask[..., None], outputs, 0.0)
    outputs = sharding_lib.with_sharding_constraint(
        outputs, self.sharding_config.activation_partition
    )
    return outputs, extra_output

  def _apply_sparse_moe(
      self,
      params: PyTree,
      inputs: Array,
      selected_indices: Array,
      selected_weights: Array,
      inputs_mask: Array | None = None,
  ) -> tuple[Array, PyTree]:
    """Apply sparse MoE."""
    logging.info('using sparse moe!')
    if inputs_mask is not None:
      # Mask out the padded tokens by assigning them to an expert that
      # does not exist so that they will be skipped in the sparse matmul.
      selected_indices = jnp.where(
          inputs_mask[..., None], selected_indices, self.num_experts
      )

    @jax.named_scope('gmm')
    def gmm(lhs, rhs, group_sizes, tiling):
      if self.gmm_impl == 'megablox':
        output = megablox.gmm(
            lhs,
            rhs,
            group_sizes=group_sizes,
            tiling=tiling,
            preferred_element_type=self.activation_dtype,
        )
      elif self.gmm_impl == 'ragged_dot':
        with xla_metadata.set_xla_metadata(
            ragged_dot_tiling=','.join([str(t) for t in tiling])):
          output = jax.lax.ragged_dot(
              lhs=lhs,
              rhs=rhs,
              group_sizes=group_sizes,
              preferred_element_type=self.activation_dtype,
          )
      else:
        raise ValueError(f'Unsupported gmm impl: {self.gmm_impl}')
      return output

    if self.sharding_config.activation_partition is None:
      selected_indices_partition = None
    else:
      selected_indices_partition = js.PartitionSpec(
          self.sharding_config.activation_partition[0],
          self.sharding_config.activation_partition[1],
          None)
    selected_weights_partition = selected_indices_partition

    # ffn0_w: [num_experts, model_dim, expand_dim]
    ffn0_w = params['ffn_0']['w']
    ffn0_w = common.convert_or_dequantize(ffn0_w, dtype=self.activation_dtype)
    ffn0_partition = (
        js.PartitionSpec(*self.ffn0_partition)
        if self.ffn0_partition is not None
        else js.PartitionSpec()
    )

    if self.use_gated_activation_in_ffn:
      # ffn0_gate_w: [num_experts, model_dim, expand_dim]
      ffn0_gate_w = params['ffn_0_gate']['w']
      ffn0_gate_w = common.convert_or_dequantize(
          ffn0_gate_w, dtype=self.activation_dtype
      )
      ffn0_gate_partition = ffn0_partition
    else:
      ffn0_gate_w = None
      ffn0_gate_partition = None

    # ffn1_w: [num_experts, expand_dim, model_dim]
    ffn1_w = params['ffn_1']['w']
    ffn1_w = common.convert_or_dequantize(ffn1_w, dtype=self.activation_dtype)
    ffn1_partition = (
        js.PartitionSpec(*self.ffn1_partition)
        if self.ffn1_partition is not None
        else js.PartitionSpec()
    )

    if self.sharding_config.activation_partition is None:
      activation_partition = js.PartitionSpec()
    else:
      activation_partition = js.PartitionSpec(
          *self.sharding_config.activation_partition)

    @jax.shard_map(
        mesh=sharding_lib.get_default_mesh(),
        in_specs=(
            activation_partition,
            ffn0_partition,
            ffn0_gate_partition,
            ffn1_partition,
            selected_indices_partition,
            selected_weights_partition
        ),
        out_specs=(
            activation_partition,
            js.PartitionSpec()),
        # Needed when using megablox.
        check_vma=False)
    def moe_ffn(
        inputs, ffn0_w, ffn0_gate_w, ffn1_w,
        selected_indices, selected_weights):
      batch_size, seq_len, _ = inputs.shape
      # flat_repeated_inputs:
      # [batch_size * seq_len * num_experts_per_token, model_dim]
      flat_repeated_inputs = einops.repeat(
          inputs, 'b s d -> (b s r) d', r=self.num_experts_per_token
      )
      flat_repeated_inputs = jnp.asarray(
          flat_repeated_inputs, self.activation_dtype)
      # flat_selected_indices: [batch_size * seq_len * num_experts_per_token]
      flat_selected_indices = jnp.ravel(selected_indices)
      sort_indices = jnp.argsort(flat_selected_indices)
      unsort_indices = jnp.argsort(sort_indices)
      sorted_inputs = flat_repeated_inputs[sort_indices]
      sorted_expert_indices = flat_selected_indices[sort_indices]
      group_sizes = jnp.bincount(sorted_expert_indices, length=self.num_experts)
      load = jnp.asarray(
          group_sizes / jnp.sum(group_sizes), self.activation_dtype
      )
      load = (
          jax.lax.psum(
              load, axis_name=('replica', 'data', 'model')) /
          jax.lax.psum(
              1, axis_name=('replica', 'data', 'model')))
      m, k = sorted_inputs.shape
      n = ffn0_w.shape[-1]
      ffn0_w = jax.lax.all_gather(ffn0_w, axis_name='data', tiled=True, axis=1)
      ffn0_tiling = (
          min(self.tile_batch_seq, m),
          min(self.tile_model_dim, k),
          min(self.tile_expand_dim, n),
      )
      ffn1_tiling = (ffn0_tiling[0], ffn0_tiling[2], ffn0_tiling[1])
      projected_inputs = gmm(sorted_inputs, ffn0_w, group_sizes, ffn0_tiling)
      activation_fn = registry.FunctionRegistry.get(self.ffn_activation)
      if self.use_gated_activation_in_ffn:
        ffn0_gate_w = jax.lax.all_gather(
            ffn0_gate_w, axis_name='data', tiled=True, axis=1)
        gate = gmm(sorted_inputs, ffn0_gate_w, group_sizes, ffn0_tiling)
        gate = activation_fn(gate)
        middle = jnp.asarray(gate, self.activation_dtype) * projected_inputs
      else:
        middle = jnp.asarray(
            activation_fn(projected_inputs), self.activation_dtype
        )
      ffn1_w = jax.lax.all_gather(ffn1_w, axis_name='data', tiled=True, axis=2)
      sorted_outputs = gmm(middle, ffn1_w, group_sizes, ffn1_tiling)
      outputs = sorted_outputs[unsort_indices]
      # outputs: [batch_size, seq_len, model_dim, num_experts_per_token]
      outputs = einops.rearrange(
          outputs,
          '(b s r) d -> b s d r',
          b=batch_size,
          s=seq_len,
          r=self.num_experts_per_token,
      )
      outputs = jnp.einsum('bsk,bstk->bst', selected_weights, outputs)
      return outputs, load

    outputs, load = moe_ffn(
        inputs,
        ffn0_w,
        ffn0_gate_w,
        ffn1_w,
        selected_indices,
        selected_weights,
    )
    return outputs, {'load': load}

  def _apply_dense_moe(
      self,
      params: PyTree,
      inputs: Array,
      selected_indices: Array,
      selected_weights: Array,
      inputs_mask: Array | None = None,
  ) -> tuple[Array, PyTree]:
    extra_outputs = {}
    batch_size, seq_len, _ = inputs.shape
    num_tokens = batch_size * seq_len
    expert_capacity = max(
        1, int(num_tokens / self.num_experts * self.expert_capacity_factor)
    )
    logging.info('expert_capacity=%s', expert_capacity)
    # selected_indices: [batch_size * seq_len, num_experts_per_token]
    selected_indices = einops.rearrange(selected_indices, 'b s k -> (b s) k')
    selected_weights = einops.rearrange(selected_weights, 'b s k -> (b s) k')
    # inputs: [batch_size * seq_len, model_dim]
    inputs = einops.rearrange(inputs, 'b s d -> (b s) d')

    # selected_onehot: [num_tokens, num_experts_per_token, num_experts]
    selected_onehot = jax.nn.one_hot(
        selected_indices, self.num_experts, dtype=jnp.int32
    )
    if inputs_mask is not None:
      selected_onehot *= einops.rearrange(inputs_mask, 'b s -> (b s) 1 1')

    # dispatch_mask: [num_tokens, num_experts]
    dispatch_mask = jnp.sum(selected_onehot, axis=1)

    # dispatch_position: [num_tokens, num_experts]
    dispatch_position = jnp.cumsum(dispatch_mask, axis=0) * dispatch_mask

    # Mask out the tokens that are out of the expert capacity.
    # position_mask: [num_tokens, num_experts]
    position_mask = jnp.asarray(
        (dispatch_position > 0) & (dispatch_position <= expert_capacity),
        dtype=self.activation_dtype,
    )

    # dispatch_onehot: [num_tokens, num_experts, expert_capacity]
    dispatch_onehot = (
        jax.nn.one_hot(
            dispatch_position - 1, expert_capacity, dtype=self.activation_dtype)
        * position_mask[..., None]
    )
    extra_outputs['load'] = jnp.sum(dispatch_onehot, axis=[0, 2]) / jnp.sum(
        dispatch_onehot
    )

    # selected_weights: [num_tokens, num_experts]
    selected_weights = jnp.einsum(
        'tk,tke->te', selected_weights, selected_onehot
    )

    # dispatch_weights: [num_tokens, num_experts, expert_capacity]
    dispatch_weights = jnp.einsum(
        'te,tec->tec', selected_weights, dispatch_onehot
    )

    # expert_inputs: [num_experts, expert_capacity, model_dim]
    expert_inputs = jnp.einsum('tec,ti->eci', dispatch_onehot, inputs)

    # projected_inputs:
    # [num_experts, expert_capacity, model_dim * expand_factor]
    expert_inputs = jnp.asarray(expert_inputs, self.activation_dtype)
    projected_inputs = self.ffn_0.apply(params['ffn_0'], expert_inputs)
    activation_fn = registry.FunctionRegistry.get(self.ffn_activation)
    if self.use_gated_activation_in_ffn:
      gate = self.ffn_0_gate.apply(params['ffn_0_gate'], expert_inputs)
      gate = jnp.asarray(activation_fn(gate), self.activation_dtype)
      middle = gate * projected_inputs
    else:
      middle = jnp.asarray(
          activation_fn(projected_inputs), self.activation_dtype
      )
    # outputs: [num_experts, expert_capacity, model_dim]
    outputs = self.ffn_1.apply(params['ffn_1'], middle)

    # outputs: [num_tokens, model_dim]
    outputs = jnp.einsum('ecd,tec->td', outputs, dispatch_weights)

    # output: [batch_size, seq_len, model_dim, num_experts_per_token]
    outputs = einops.rearrange(
        outputs, '(b s) d -> b s d', b=batch_size, s=seq_len
    )
    return outputs, extra_outputs


@module.ModuleRegistry.register
@dataclasses.dataclass
class Attention(module.SimplyModule):
  """Standard Multi-head Attention layer."""

  model_dim: int
  n_heads: int
  per_head_dim: int
  use_causal: bool = True
  add_extra_output: bool = False
  qk_norm: LayerNorm | None = None
  use_per_dim_scale: bool = False
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  # Mixed precision related.
  activation_dtype: DTypeLike = 'bfloat16'
  weight_dtype: DTypeLike = 'float32'
  # Sharding related.
  qkv_partition: PartitionAnnotation = None
  o_partition: PartitionAnnotation = None
  attn_activation_partition: PartitionAnnotation = None
  output_partition: PartitionAnnotation = None
  # Decoding related.
  update_kv_cache_in_place: bool = True
  # Experimental flags.
  use_flash_attention: bool = False
  flash_attention_block_size: int = 512
  window_size: int = 0
  use_window_chunk: bool = False
  n_kv_heads: int = 0
  qkv_use_bias: bool = False
  o_use_bias: bool = False
  attn_soft_cap: float = 50.0
  rope_max_timescale: int = 10_000
  rope_scale_factor: float = 1.0
  query_scale: float = -1.0

  def setup(self) -> None:
    if self.use_per_dim_scale:
      self.per_dim_scale = PerDimScale(
          self.per_head_dim,
          weight_dtype=self.weight_dtype,
          activation_dtype=self.activation_dtype,
      )

    if self.n_kv_heads <= 0:
      self.n_kv_heads = self.n_heads
    if self.n_heads % self.n_kv_heads != 0:
      raise ValueError(
          f'n_heads ({self.n_heads}) must be a multiple of n_kv_heads'
          f'({self.n_kv_heads}).'
      )
    q_shape = [self.model_dim, self.n_heads, self.per_head_dim]
    kv_shape = [self.model_dim, self.n_kv_heads, self.per_head_dim]
    qkv_kwargs = {
        'bias_term': 'hd' if self.qkv_use_bias else '',
        'weight_init': self.weight_init,
        'weight_dtype': self.weight_dtype,
        'activation_dtype': self.activation_dtype,
        'output_partition': self.attn_activation_partition,
    }
    self.q_proj = module.EinsumLinear(
        eqn='ihd,...i->...hd',
        weight_shape=q_shape,
        weight_partition=self.qkv_partition,
        **qkv_kwargs,
    )
    self.k_proj = module.EinsumLinear(
        eqn='ihd,...i->...hd',
        weight_shape=kv_shape,
        weight_partition=self.qkv_partition,
        **qkv_kwargs,
    )
    self.v_proj = module.EinsumLinear(
        eqn='ihd,...i->...hd',
        weight_shape=kv_shape,
        weight_partition=self.qkv_partition,
        **qkv_kwargs,
    )
    o_kwargs = qkv_kwargs.copy()
    o_kwargs['bias_term'] = 'i' if self.o_use_bias else ''
    o_kwargs['output_partition'] = self.output_partition
    self.o_proj = module.EinsumLinear(
        eqn='ihd,...hd->...i',
        weight_shape=q_shape,
        weight_partition=self.o_partition,
        **o_kwargs,
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    q_key, k_key, v_key, o_key = jax.random.split(prng_key, num=4)
    params = {}
    params['q_proj'] = self.q_proj.init(q_key)
    params['k_proj'] = self.k_proj.init(k_key)
    params['v_proj'] = self.v_proj.init(v_key)
    params['o_proj'] = self.o_proj.init(o_key)

    if self.qk_norm:
      params['q_norm'] = self.qk_norm.init()
      params['k_norm'] = self.qk_norm.init()

    if self.use_per_dim_scale:
      params['per_dim_scale'] = self.per_dim_scale.init()

    return params

  def apply(
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array,
      segment_positions: Array,
      decode_state: PyTree = None,
  ) -> tuple[Array, PyTree]:
    params = get_raw_arrays(params)
    # x: [batch_size, seq_len, model_dim]
    assert len(x.shape) == 3
    assert x.shape[-1] == self.model_dim
    # q: [batch_size, seq_len, n_heads, per_head_dim]
    q = self.q_proj.apply(params['q_proj'], x)
    # k: [batch_size, seq_len, n_heads, per_head_dim]
    k = self.k_proj.apply(params['k_proj'], x)
    # v: [batch_size, seq_len, n_heads, per_head_dim]
    v = self.v_proj.apply(params['v_proj'], x)

    if self.qk_norm:
      q = self.qk_norm.apply(params['q_norm'], q)
      k = self.qk_norm.apply(params['k_norm'], k)

    q = rotary_positional_embedding(
        q,
        segment_positions=segment_positions,
        max_timescale=self.rope_max_timescale,
        scale_factor=self.rope_scale_factor,
    )
    k = rotary_positional_embedding(
        k,
        segment_positions=segment_positions,
        max_timescale=self.rope_max_timescale,
        scale_factor=self.rope_scale_factor,
    )

    if self.use_per_dim_scale:
      q = self.per_dim_scale.apply(params['per_dim_scale'], q)
    elif self.query_scale > 0:
      q = q / self.query_scale
    else:
      q = q / jnp.sqrt(self.per_head_dim)

    # n_groups = n_heads // n_kv_heads
    # q in [batch_size, seq_len, n_groups, n_kv_heads, per_head_dim]
    # k in [batch_size, seq_len, n_kv_heads, per_head_dim]
    # v in [batch_size, seq_len, n_kv_heads, per_head_dim]

    # Note: g/n_kv_heads order change is for compatibility with Gemma models.
    q = einops.rearrange(
        q,
        '... (n_kv_heads g) h -> ... g n_kv_heads h',
        n_kv_heads=self.n_kv_heads,
    )
    group_sharding = None
    if self.attn_activation_partition:
      group_sharding = (
          *self.attn_activation_partition[:2],
          None,
          *self.attn_activation_partition[2:],
      )

    k, v, kv_segment_positions, kv_segment_ids, decode_state = (
        updated_decode_state(
            k=k,
            v=v,
            segment_positions=segment_positions,
            segment_ids=segment_ids,
            decode_state=decode_state,
            window_size=self.window_size,
        )
    )

    extra_output = dict(decode_state=decode_state)

    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[1]

    # At decoding time (q.shape[1] == 1), we don't use flash attention.
    if self.use_flash_attention and q_seq_len > 1:
      batch_size_axis, seq_len_axis, num_heads_axis, per_head_size_axis = (
          self.attn_activation_partition
      )
      bnlh = js.PartitionSpec(
          batch_size_axis, num_heads_axis, seq_len_axis, per_head_size_axis
      )
      bl = js.PartitionSpec(batch_size_axis, seq_len_axis)

      q = einops.rearrange(
          q,
          'b l g n_kv_heads h -> b (n_kv_heads g) l h ',
          n_kv_heads=self.n_kv_heads,
      )
      k = einops.repeat(
          k,
          'b l n_kv_heads h -> b (n_kv_heads g) l h',
          g=self.n_heads // self.n_kv_heads,
      )
      v = einops.repeat(
          v,
          'b l n_kv_heads h -> b (n_kv_heads g) l h',
          g=self.n_heads // self.n_kv_heads,
      )

      # NOTE: These are static masks, and their behavior are global which can
      # result in some limitations. For example, we cannot mask first/last k
      # tokens for each sequence under packed mode.
      mask = splash_attention.CausalMask((q_seq_len, kv_seq_len))
      if self.window_size > 0 and self.window_size + 1 < kv_seq_len:
        mask &= splash_attention.LocalMask(
            (q_seq_len, kv_seq_len),
            (self.window_size, None),
            offset=0,
        )
      mask = splash_attention.MultiHeadMask([mask] * self.n_heads)

      block_sizes = splash_attention.BlockSizes(
          block_q=self.flash_attention_block_size,
          block_kv=self.flash_attention_block_size,
          block_kv_compute=self.flash_attention_block_size,
          block_q_dkv=self.flash_attention_block_size,
          block_kv_dkv=self.flash_attention_block_size,
          block_kv_dkv_compute=self.flash_attention_block_size,
          block_q_dq=self.flash_attention_block_size,
          block_kv_dq=self.flash_attention_block_size,
      )

      mesh = sharding_lib.get_default_mesh()
      attn_soft_cap = self.attn_soft_cap
      if attn_soft_cap is not None and attn_soft_cap < 0:
        attn_soft_cap = None
      splash_attn_kernel = splash_attention.make_splash_mha(
          mask=mask,
          block_sizes=block_sizes,
          mask_value=neg_inf(np.float32),
          attn_logits_soft_cap=attn_soft_cap,
          head_shards=mesh.shape[num_heads_axis],
          q_seq_shards=1,
      )
      kernel_sharding = splash_attn_kernel.manual_sharding_spec(
          mesh_sharding((num_heads_axis, seq_len_axis))
      )

      @functools.partial(
          shard_map.shard_map,
          mesh=sharding_lib.get_default_mesh(),
          in_specs=(kernel_sharding, bnlh, bnlh, bnlh, bl, bl),
          out_specs=bnlh,
          check_rep=False,
      )
      def flash_attention_fn(
          kernel, query, key, value, q_segment_ids, kv_segment_ids
      ):
        attn_out = jax.vmap(kernel)(
            q=query,
            k=key,
            v=value,
            segment_ids=splash_attention.SegmentIds(
                q=q_segment_ids, kv=kv_segment_ids
            ),
        )
        return attn_out

      output = flash_attention_fn(
          splash_attn_kernel, q, k, v, segment_ids, kv_segment_ids
      )
      output = jnp.swapaxes(output, 1, 2)  # Swap back.
    else:
      mask = create_mask(
          segment_positions=segment_positions,
          kv_segment_positions=kv_segment_positions,
          segment_ids=segment_ids,
          kv_segment_ids=kv_segment_ids,
          window_size=self.window_size,
      )
      # Add the group and head dimension.
      mask = einops.rearrange(mask, 'b l1 l2 -> b 1 1 l1 l2')

      # q: [batch_size, seq_len, n_groups, self.n_kv_heads, self.per_head_dim]
      # k, v: [batch_size, seq_len, self.n_kv_heads, self.per_head_dim]
      if (
          self.use_window_chunk
          and self.window_size > 0
          and self.window_size + 1 < kv_seq_len
          and q_seq_len > 1
      ):
        # We don't do this trick at decoding time (q.shape[1] == 1), as we have
        # better way there.
        output = chunked_local_attn(
            q,
            k,
            v,
            mask,
            self.window_size,
            attn_soft_cap=self.attn_soft_cap,
            dtype=self.activation_dtype,
        )
      else:
        output, attn_mat = attn(
            q,
            k,
            v,
            mask,
            attn_soft_cap=self.attn_soft_cap,
            dtype=self.activation_dtype,
        )
        if self.add_extra_output:
          extra_output['attn_mat'] = attn_mat

      output = sharding_lib.with_sharding_constraint(output, group_sharding)
      output = einops.rearrange(
          output, '... n_groups n_kv_heads i -> ... (n_kv_heads n_groups) i'
      )

    output = sharding_lib.with_sharding_constraint(
        output, self.attn_activation_partition
    )
    output = self.o_proj.apply(params['o_proj'], output)
    return output, extra_output


@module.ModuleRegistry.register
@dataclasses.dataclass
class TransformerBlock(module.SimplyModule):
  """A single transformer block."""

  model_dim: int
  n_heads: int
  per_head_dim: int
  expand_factor: int
  sharding_config: SimplyConfig
  use_rmsnorm: bool = False
  use_pre_ln: bool = True
  use_post_ln: bool = False
  use_post_skip_ln: bool = False
  use_qk_norm: bool = False
  use_gated_activation_in_ffn: bool = False
  use_per_dim_scale: bool = False
  # MoE.
  use_moe: bool = False
  num_experts: int = 0
  num_experts_per_token: int = 0
  expert_capacity_factor: float | None = 0.0
  lbl_loss_weight: float = 0.0
  router_z_loss_weight: float = 0.0
  # Mixed precision related.
  activation_dtype: DTypeLike = 'bfloat16'
  # Below are for experimental usage.
  ffn_expand_dim: int | None = None
  use_flash_attention: bool = False
  flash_attention_block_size: int = 512
  window_size: int = 0
  use_window_chunk: bool = False
  n_kv_heads: int = 0
  ffn_use_bias: bool = True
  qkv_use_bias: bool = False
  o_use_bias: bool = False
  ffn_activation: str = 'gelu'
  norm_scale_plus_one: bool = True
  attn_soft_cap: float = 50.0  # If negative, no softcap.
  rms_norm_epsilon: float = 1e-6
  rope_max_timescale: int = 10_000
  rope_scale_factor: float = 1.0
  query_scale: float = -1.0
  # tile sizes for gmm.
  tile_batch_seq: int = 128
  tile_model_dim: int = 128
  tile_expand_dim: int = 128
  # Implementation of gmm.
  gmm_impl: str = 'ragged_dot'

  @property
  def expand_dim(self) -> int:
    if self.ffn_expand_dim is not None:
      return self.ffn_expand_dim
    return self.expand_factor * self.model_dim

  def setup(self) -> None:
    if self.use_pre_ln:
      self.pre_ln_0 = LayerNorm(
          dim=self.model_dim,
          use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype,
          scale_plus_one=self.norm_scale_plus_one,
          epsilon=self.rms_norm_epsilon,
      )
      self.pre_ln_1 = LayerNorm(
          dim=self.model_dim,
          use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype,
          scale_plus_one=self.norm_scale_plus_one,
          epsilon=self.rms_norm_epsilon,
      )
    if self.use_post_ln:
      self.post_ln_0 = LayerNorm(
          dim=self.model_dim,
          use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype,
          scale_plus_one=self.norm_scale_plus_one,
          epsilon=self.rms_norm_epsilon,
      )
      self.post_ln_1 = LayerNorm(
          dim=self.model_dim,
          use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype,
          scale_plus_one=self.norm_scale_plus_one,
          epsilon=self.rms_norm_epsilon,
      )
    if self.use_post_skip_ln:
      self.post_skip_ln_0 = LayerNorm(
          dim=self.model_dim,
          use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype,
          scale_plus_one=self.norm_scale_plus_one,
          epsilon=self.rms_norm_epsilon,
      )
      self.post_skip_ln_1 = LayerNorm(
          dim=self.model_dim,
          use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype,
          scale_plus_one=self.norm_scale_plus_one,
          epsilon=self.rms_norm_epsilon,
      )

    qk_norm = None
    if self.use_qk_norm:
      qk_norm = LayerNorm(
          dim=self.per_head_dim,
          use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype,
          scale_plus_one=self.norm_scale_plus_one,
          epsilon=self.rms_norm_epsilon,
      )

    self.attn = Attention(
        self.model_dim,
        self.n_heads,
        self.per_head_dim,
        qk_norm=qk_norm,
        use_per_dim_scale=self.use_per_dim_scale,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        qkv_partition=self.sharding_config.attn_qkv_partition,
        o_partition=self.sharding_config.attn_o_partition,
        attn_activation_partition=self.sharding_config.attn_activation_partition,
        output_partition=self.sharding_config.activation_partition,
        # Others.
        use_flash_attention=self.use_flash_attention,
        flash_attention_block_size=self.flash_attention_block_size,
        window_size=self.window_size,
        use_window_chunk=self.use_window_chunk,
        n_kv_heads=self.n_kv_heads,
        qkv_use_bias=self.qkv_use_bias,
        o_use_bias=self.o_use_bias,
        attn_soft_cap=self.attn_soft_cap,
        rope_max_timescale=self.rope_max_timescale,
        rope_scale_factor=self.rope_scale_factor,
        query_scale=self.query_scale,
    )
    if self.use_moe:
      self.ffn = MoEFeedForward(
          num_experts=self.num_experts,
          num_experts_per_token=self.num_experts_per_token,
          expert_capacity_factor=self.expert_capacity_factor,
          router_z_loss_weight=self.router_z_loss_weight,
          lbl_loss_weight=self.lbl_loss_weight,
          model_dim=self.model_dim,
          expand_factor=self.expand_factor,
          use_gated_activation_in_ffn=self.use_gated_activation_in_ffn,
          # Mixed precision related.
          activation_dtype=self.activation_dtype,
          # Sharding related.
          sharding_config=self.sharding_config,
          # Below are for experimental usage.
          ffn_expand_dim=self.ffn_expand_dim,
          ffn_use_bias=self.ffn_use_bias,
          ffn_activation=self.ffn_activation,
          # tile sizes for gmm.
          tile_batch_seq=self.tile_batch_seq,
          tile_model_dim=self.tile_model_dim,
          tile_expand_dim=self.tile_expand_dim,
          # Implementation of gmm.
          gmm_impl=self.gmm_impl,
      )
    else:
      self.ffn = FeedForward(
          model_dim=self.model_dim,
          expand_factor=self.expand_factor,
          use_gated_activation_in_ffn=self.use_gated_activation_in_ffn,
          # Mixed precision related.
          activation_dtype=self.activation_dtype,
          # Sharding related.
          sharding_config=self.sharding_config,
          # Below are for experimental usage.
          ffn_expand_dim=self.ffn_expand_dim,
          ffn_use_bias=self.ffn_use_bias,
          ffn_activation=self.ffn_activation,
      )

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    ffn_key, attn_key = jax.random.split(prng_key, num=2)
    params['ffn'] = self.ffn.init(ffn_key)
    params['attn'] = self.attn.init(attn_key)
    if self.use_pre_ln:
      params['pre_ln_0'] = self.pre_ln_0.init()
      params['pre_ln_1'] = self.pre_ln_1.init()
    if self.use_post_ln:
      params['post_ln_0'] = self.post_ln_0.init()
      params['post_ln_1'] = self.post_ln_1.init()
    if self.use_post_skip_ln:
      params['post_skip_ln_0'] = self.post_skip_ln_0.init()
      params['post_skip_ln_1'] = self.post_skip_ln_1.init()

    return params

  def apply(
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array,
      segment_positions: Array,
      decode_state: PyTree = None,
  ) -> tuple[Array, PyTree]:
    extra_output = {}
    x_res = x
    if self.use_pre_ln:
      x = self.pre_ln_0.apply(params['pre_ln_0'], x)
    x, attn_extra_output = self.attn.apply(
        params['attn'],
        x,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        decode_state=decode_state,
    )
    if self.use_post_ln:
      x = self.post_ln_0.apply(params['post_ln_0'], x)
    x += x_res
    if self.use_post_skip_ln:
      x = self.post_skip_ln_0.apply(params['post_skip_ln_0'], x)
    x = sharding_lib.with_sharding_constraint(
        x, self.sharding_config.activation_partition
    )

    x_res = x
    if self.use_pre_ln:
      x = self.pre_ln_1.apply(params['pre_ln_1'], x)
    # Assumes pad id for segment_ids is 0.
    x, ffn_extra_output = self.ffn.apply(
        params['ffn'], x, inputs_mask=segment_ids != 0)
    if self.use_post_ln:
      x = self.post_ln_1.apply(params['post_ln_1'], x)
    x += x_res
    if self.use_post_skip_ln:
      x = self.post_skip_ln_1.apply(params['post_skip_ln_1'], x)
    x = sharding_lib.with_sharding_constraint(
        x, self.sharding_config.activation_partition
    )

    if decode_state is not None:
      extra_output['decode_state'] = attn_extra_output['decode_state']
    if self.use_moe:
      extra_output['ffn'] = ffn_extra_output
    return x, extra_output


@dataclasses.dataclass
class InputEncoderInterface(module.SimplyModule):
  """Interface for custom input encoding for TransformerLM.

  The primary input is the batched token sequence, just like
  TransformerLM. Additional inputs from extra_inputs may be specified.

  Output is a sequence of embeddings and a mask of where in the input sequence
  they should be substituted:

    embeddings: shape [batch num_embeddings dim]
    embedding_mask: shape [batch input_seq_len]

  The k-th embedding is substituted at the k-th set bit of
  embedding_mask. Any excess entries are ignored.
  """

  @dataclasses.dataclass
  class Output:
    embeddings: common.Array
    embedding_mask: common.Array

  # Name for this input encoder used for naming params. Must be unique among
  # input encoders within a model.
  name: str
  # Keys from extra_input that will be passed to apply().
  extra_input_keys: Tuple[str, ...]

  def apply(
      self, params: common.PyTree, x: common.Array, **kwargs: Mapping[str, Any]
  ) -> 'InputEncoderInterface.Output':
    raise NotImplementedError()


@module.ModuleRegistry.register
@dataclasses.dataclass
class TransformerLM(module.SimplyModule):
  """A decoder-only Transformer."""

  config: SimplyConfig
  sharding_config: SimplyConfig | None = None

  def setup(self) -> None:
    config = self.config
    if self.sharding_config is None:
      self.sharding_config = config_lib.GSPMDSharding()
    sharding_config = self.sharding_config
    self.activation_dtype = config.activation_dtype_name

    self.embed_linear = module.EmbeddingLinear(
        vocab_size=config.vocab_size,
        dim=config.model_dim,
        weight_partition=sharding_config.embed_partition,
        activation_dtype=self.activation_dtype,
        embedding_scale_by_sqrt_dim=config.embedding_lookup_scale,
        use_tied_embedding=config.use_tied_embedding,
        use_bias=config.output_layer_use_bias,
    )
    self.input_encoders: list[InputEncoderInterface] = config.input_encoders
    names = [x.name for x in self.input_encoders]
    assert len(names) == len(
        set(names)
    ), f'Duplicate input encoder name: {names}'

    def _create_transformer_block(pattern):
      rope_max_timescale = config.global_rope_max_timescale
      if pattern == 'local':
        rope_max_timescale = config.local_rope_max_timescale
      rope_scale_factor = config.global_rope_scale_factor
      if pattern == 'local':
        rope_scale_factor = config.local_rope_scale_factor
      return TransformerBlock(
          config.model_dim,
          config.n_heads,
          config.per_head_dim,
          config.expand_factor,
          use_rmsnorm=config.use_rmsnorm,
          use_pre_ln=config.use_pre_ln,
          use_post_ln=config.use_post_ln,
          use_post_skip_ln=config.use_post_skip_ln,
          use_qk_norm=config.use_qk_norm,
          use_per_dim_scale=config.use_per_dim_scale,
          use_gated_activation_in_ffn=config.use_gated_activation_in_ffn,
          ffn_use_bias=config.ffn_use_bias,
          ffn_expand_dim=getattr(config, 'ffn_expand_dim', None),
          # MoE related.
          use_moe=config.use_moe,
          num_experts=config.num_experts,
          expert_capacity_factor=config.expert_capacity_factor,
          num_experts_per_token=config.num_experts_per_token,
          lbl_loss_weight=config.lbl_loss_weight,
          router_z_loss_weight=config.router_z_loss_weight,
          tile_batch_seq=config.tile_batch_seq,
          tile_model_dim=config.tile_model_dim,
          tile_expand_dim=config.tile_expand_dim,
          gmm_impl=config.gmm_impl,
          # Mixed precision related.
          activation_dtype=self.activation_dtype,
          sharding_config=sharding_config,
          # Others.
          use_flash_attention=config.use_flash_attention,
          flash_attention_block_size=config.flash_attention_block_size,
          window_size=config.window_size if pattern == 'local' else 0,
          use_window_chunk=config.use_window_chunk,
          n_kv_heads=config.n_kv_heads,
          qkv_use_bias=config.qkv_use_bias,
          ffn_activation=config.ffn_activation,
          norm_scale_plus_one=config.norm_scale_plus_one,
          attn_soft_cap=config.attn_soft_cap,
          rms_norm_epsilon=config.rms_norm_epsilon,
          rope_max_timescale=rope_max_timescale,
          rope_scale_factor=rope_scale_factor,
          query_scale=config.query_scale,
      )

    self.blocks = []
    for i in range(self.config.n_layers):
      block = _create_transformer_block(
          config.block_attn_pattern[i % len(config.block_attn_pattern)]
      )
      self.blocks.append(block)

    self.final_ln = LayerNorm(
        dim=config.model_dim,
        use_bias=not config.use_rmsnorm,
        activation_dtype=self.activation_dtype,
        scale_plus_one=config.norm_scale_plus_one,
        epsilon=config.rms_norm_epsilon,
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    prng_key, embed_linear_key = jax.random.split(prng_key, num=2)
    params['embed_linear'] = self.embed_linear.init(embed_linear_key)

    input_encoder_params = {}
    for input_encoder in self.input_encoders:
      prng_key, input_enc_key = jax.random.split(prng_key)
      input_encoder_params[input_encoder.name] = input_encoder.init(
          input_enc_key
      )
    if input_encoder_params:
      params['input_encoders'] = input_encoder_params

    for i, block in enumerate(self.blocks):
      prng_key, block_key = jax.random.split(prng_key, num=2)
      params[f'block_{i}'] = block.init(block_key)
    params['final_ln'] = self.final_ln.init()
    return params

  def _replace_embeddings(
      self, orig_embeddings, replacement_embeddings, replacement_mask
  ):
    """Replaces a sequence of embeddings at certain positions.

    Args:
      orig_embeddings: Original embeddings of shape [batch seq_len dim]
      replacement_embeddings: New embeddings of shape [batch num_embeddings
        dim].
      replacement_mask: Mask of where to replace, with shape [batch seq_len].
        Note that due to our implementation, the mask cannot include position 0.

    Returns:
      Array with the same shape as `orig_embeddings` with some entries replaced
      by `replacement_embeddings`. The k-th embedding in
      `replacement_embeddings` is placed at
      the k-th set bit of `replacement_mask`. Excess entries in either
      `replacement_embeddings` or `replacement_mask` are ignored.
    """

    def substitute_embeddings(x, y, mask):
      target_pos = jnp.nonzero(mask, size=y.shape[0])
      first_emb = x[0]
      x = x.at[target_pos, :].set(y)
      return x.at[0].set(first_emb)

    substitute_embeddings_batch = jax.vmap(substitute_embeddings)
    return substitute_embeddings_batch(
        orig_embeddings, replacement_embeddings, replacement_mask
    )

  def apply(
      self,
      params: PyTree,
      x: Array,
      *,
      segment_ids: Array | None = None,
      segment_positions: Array | None = None,
      extra_inputs: Mapping[str, PyTree] | None = None,
      decode_state: PyTree = None,
  ) -> tuple[Array, PyTree]:
    """Transformer forward pass.

    Args:
      params: All the transformer params.
      x: Input token sequence of shape [batch seq_len]
      segment_ids: IDs in case multiple sequences are combined in the input (no
        cross-attention between segments). Defaults to all 1.
      segment_positions: Positions for the tokens, defaults to sequential
        starting from 0.
      extra_inputs: Additional inputs (e.g. images) that can be passed to input
        encoders.
      decode_state: KV cache for decoding.

    Returns:
      A pair of (logits, new decode state).
    """
    input_tokens = x

    if segment_positions is None:
      batch_size, seq_len = x.shape
      segment_positions = einops.repeat(
          jnp.arange(seq_len), 'l -> b l', b=batch_size
      )
    if segment_ids is None:
      segment_ids = jnp.ones_like(segment_positions)
    extra_inputs = extra_inputs or {}

    self.sharding_config = cast(SimplyConfig, self.sharding_config)
    # Add sharding constraints to the inputs.
    x = sharding_lib.with_sharding_constraint(
        x, self.sharding_config.data_partition
    )
    segment_ids = sharding_lib.with_sharding_constraint(
        segment_ids, self.sharding_config.data_partition
    )
    segment_positions = sharding_lib.with_sharding_constraint(
        segment_positions, self.sharding_config.data_partition
    )

    # TODO: Consider removing this conversion. In theory, in can result
    # in larger HBM usage when params.dtype=float32, activation_dtype=bfloat16,
    # as it forces XLA to keep copy of params in bfloat16 at the beginning.
    # In practice, we found, by default, params would be casted to bfloat16
    # no matter what activation_dtype is set.
    def convert_to_lower_bits(x, activation_dtype):
      # Only convert if the activation_dtype is lower bits than the params.
      if x.dtype.itemsize > jnp.dtype(activation_dtype).itemsize:
        return jnp.asarray(x, dtype=activation_dtype)
      else:
        return x

    params = jax.tree_util.tree_map(
        functools.partial(
            convert_to_lower_bits, activation_dtype=self.activation_dtype
        ),
        params,
    )

    x = self.embed_linear.embed(params['embed_linear'], x)

    for input_encoder in self.input_encoders:
      missing_keys = [
          k not in extra_inputs for k in input_encoder.extra_input_keys
      ]
      if all(missing_keys):
        continue
      if any(missing_keys):
        raise ValueError(
            f'Incomplete extra_inputs keys, got {extra_inputs.keys()}, expected'
            f' {input_encoder.extra_input_keys}'
        )

      input_enc_params = params['input_encoders'][input_encoder.name]
      kwargs = {k: extra_inputs[k] for k in input_encoder.extra_input_keys}
      encoder_output = input_encoder.apply(
          input_enc_params, input_tokens, **kwargs
      )
      x = self._replace_embeddings(
          x, encoder_output.embeddings, encoder_output.embedding_mask
      )

    extra_output_list = []
    block_start_index = 0

    global_decode_state = None
    if decode_state is not None:
      global_decode_state = {
          k: v
          for k, v in getattr(decode_state, 'items')()
          if not k.startswith('block_')
      }

    def _completed_block_decode_state(block_decode_state: PyTree) -> PyTree:
      if block_decode_state is None:
        return global_decode_state
      assert global_decode_state is not None
      return block_decode_state | global_decode_state

    if self.config.use_scan:

      def _prepare_stack_list(
          tree: PyTree, n_repeats: int, n_blocks_per_repeat: int = 1
      ) -> Sequence[PyTree]:
        if tree is None:
          return [None] * n_blocks_per_repeat
        tree = cast(Mapping[str, Any], tree)
        block_stack_list = []
        for i in range(n_blocks_per_repeat):
          s = [
              tree.get(f'block_{j * n_blocks_per_repeat + i}', {})
              for j in range(n_repeats)
          ]
          block_stack_list.append(
              jax.tree.map(lambda *x: jnp.stack(x, axis=0), *s),
          )
        return block_stack_list

      n_repeats = len(self.blocks) // len(self.config.block_attn_pattern)
      params_stack_list = _prepare_stack_list(
          params, n_repeats, len(self.config.block_attn_pattern)
      )
      decode_state_stack_list = _prepare_stack_list(
          decode_state, n_repeats, len(self.config.block_attn_pattern)
      )
      # decode_state_stack_list is formatted as:
      # [
      #   {  # block_0_stack
      #     'k': [n_repeats, batch_size, seq_len, n_kv_heads, per_head_dim],
      #     'v': [n_repeats, batch_size, seq_len, n_kv_heads, per_head_dim],
      #     ...
      #   },
      #   {  # block_1_stack
      #     'k': [n_repeats, batch_size, seq_len, n_kv_heads, per_head_dim],
      #     'v': [n_repeats, batch_size, seq_len, n_kv_heads, per_head_dim],
      #     ...
      #   },
      #   ...
      # ]

      def _process_per_repeat(
          inputs: jax.Array, p: tuple[Sequence[PyTree], Sequence[PyTree]]
      ) -> tuple[jax.Array, Sequence[PyTree]]:
        # This function will process a set of blocks that will be repeated
        # multiple times. The number of blocks in this set is determined by the
        # `block_attn_pattern`` in the config. For example, if the pattern is
        # ('global', 'local', 'local'), then this function will process 3 blocks
        # that will be repeated (n_layers // 3) times.
        block_params_list, block_decode_state_list = p
        x = inputs
        block_extra_output_list = []
        for i in range(len(self.config.block_attn_pattern)):
          apply_fn = self.blocks[i].apply
          if self.config.use_remat:
            apply_fn = jax.remat(
                apply_fn,
                policy=getattr(
                    jax.checkpoint_policies, self.config.remat_policy, None
                ),
            )
          block_decode_state = _completed_block_decode_state(
              block_decode_state_list[i]
          )
          x, block_extra_output = apply_fn(
              block_params_list[i],
              x,
              segment_ids=segment_ids,
              segment_positions=segment_positions,
              decode_state=block_decode_state,
          )
          block_extra_output_list.append(block_extra_output)
        return x, block_extra_output_list

      x, extra_output_stack_list = jax.lax.scan(
          _process_per_repeat,
          init=x,
          xs=(params_stack_list, decode_state_stack_list),
          length=n_repeats,
      )

      # extra_output_stack_list is formatted as:
      # [
      #   {  # block_0_stack
      #     'decode_state': {
      #       'k': [n_repeats, batch_size, seq_len, n_kv_heads, per_head_dim],
      #       'v': [n_repeats, batch_size, seq_len, n_kv_heads, per_head_dim],
      #     }
      #   },
      #   {  # block_1_stack
      #     'decode_state': { ... }
      #   },
      #   ...
      # ]
      # We want to flatten n_repeats.
      new_leaves = [
          [] for _ in range(n_repeats * len(self.config.block_attn_pattern))
      ]
      treedef = None
      for i, extra_output_stack in enumerate(extra_output_stack_list):
        leaves, treedef = jax.tree.flatten(extra_output_stack)
        for leaf in leaves:
          for j in range(n_repeats):
            new_leaves[j * len(self.config.block_attn_pattern) + i].append(
                leaf[j]
            )
      assert treedef is not None
      # extra_output_list is formatted as:
      # [
      #   {  # block_0
      #     'decode_state': {
      #       'k': [batch_size, seq_len, n_kv_heads, per_head_dim],
      #       'v': [batch_size, seq_len, n_kv_heads, per_head_dim],
      #     }
      #   },
      #   {  # block_1
      #     'decode_state': { ... }
      #   },
      #   ...
      # ]
      # Later, We want to extract `decode_state` to the root level.
      extra_output_list = [treedef.unflatten(leaf) for leaf in new_leaves]

      block_start_index = n_repeats * len(self.config.block_attn_pattern)

    # Process the remaining blocks that are not in scan.
    for i in range(block_start_index, self.config.n_layers):
      if decode_state is None:
        block_decode_state = None
      else:
        decode_state = cast(Mapping[str, Any], decode_state)
        block_decode_state = _completed_block_decode_state(
            decode_state.get(f'block_{i}')
        )
      x, block_extra_output = self.blocks[i].apply(
          params[f'block_{i}'],
          x,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
          decode_state=block_decode_state,
      )
      extra_output_list.append(block_extra_output)

    extra_output = {}
    for i, extra_output_per_repeat in enumerate(extra_output_list):
      for k, v in extra_output_per_repeat.items():
        if k not in extra_output:
          extra_output[k] = {}
        extra_output[k][f'block_{i}'] = v

    x = self.final_ln.apply(params['final_ln'], x)
    logits = self.embed_linear.apply(params['embed_linear'], x)
    if self.config.output_logits_soft_cap > 0:
      logits = soft_cap(logits, self.config.output_logits_soft_cap)
    return logits, extra_output

  def predict_probs(
      self, params: PyTree, x: Array, temperature: float = 1.0
  ) -> Array:
    logits, _ = self.apply(params, x)
    logits = logits.astype(jnp.float32)
    logits /= temperature
    return jax.nn.softmax(logits, axis=-1)


################################################################################
## Loss and backprop.


def compute_loss(model, params, batch):
  """The base method for loss computation."""
  # inputs: [batch_size, seq_len]
  inputs = batch['decoder_input_tokens']
  # targets: [batch_size, seq_len]
  targets = batch['decoder_target_tokens']
  # loss_weights: [batch_size, seq_len]
  loss_weights = batch.get('decoder_loss_weights', None)
  if loss_weights is None:
    loss_weights = jnp.ones_like(targets, dtype='bool')
  # segment_ids: [batch_size, seq_len]
  segment_ids = batch.get('decoder_segment_ids', None)
  # segment_positions: [batch_size, seq_len]
  segment_positions = batch.get('decoder_positions', None)
  # logits: [batch_size, seq_len, vocab_size]
  logits, model_extra_output = model.apply(
      params, inputs,
      segment_ids=segment_ids, segment_positions=segment_positions)
  # Always use float32 in softmax.
  logits = logits.astype(jnp.float32)
  targets_one_hot = jax.nn.one_hot(targets, logits.shape[-1], axis=-1)
  token_loss = jnp.einsum(
      'blv,blv->bl', targets_one_hot, jax.nn.log_softmax(logits))
  total_loss = - jnp.sum(token_loss * loss_weights)
  total_loss_weight = sharding_lib.with_sharding_constraint(
      jnp.sum(loss_weights), None)
  loss = total_loss / total_loss_weight
  loss = sharding_lib.with_sharding_constraint(loss, None)
  # Compute accuracy.
  pred = jnp.argmax(logits, axis=-1)
  correct = (pred == targets).astype(jnp.float32) * loss_weights
  accuracy = jnp.sum(correct) / total_loss_weight
  accuracy = sharding_lib.with_sharding_constraint(accuracy, None)
  extra_output = {'accuracy': accuracy, 'loss_weight': total_loss_weight}
  if model_extra_output:
    extra_loss, extra_metric_dict = collect_loss_and_metric(model_extra_output)
    extra_output.update(extra_metric_dict)
    loss += extra_loss
  return loss, extra_output


def collect_tag(extra_outputs, tag):
  tag_dict = collections.defaultdict(list)
  for val, path in pytree.tree_leaves_with_tag(extra_outputs, tag):
    name = path[-1].key
    logging.info(
        'Collected %s tag from %s with shape %s and name %s.',
        tag,
        path,
        val.shape,
        name,
    )
    tag_dict[name].append(val)
  tag_val_dict = {}
  for k, v in tag_dict.items():
    if v[0].size != 1:
      raise ValueError(
          f'Tag {tag} has met a non-scalar values for key {k}: {v}')
    tag_val_dict[k] = jnp.mean(jnp.stack(v))
  return tag_val_dict


def collect_loss_and_metric(extra_outputs):
  metric_val_dict = collect_tag(extra_outputs, 'metric')
  loss_dict = collect_tag(extra_outputs, 'loss')
  total_loss = sum(loss_dict.values())
  return total_loss, metric_val_dict


def compute_train_loss(model, params, batch):
  return compute_loss(model, params, batch)


def compute_eval_loss(model, params, batch):
  return compute_loss(model, params, batch)


def compute_distill_loss(
    model,
    params,
    teacher_model,
    teacher_params,
    batch,
    temperature=1.0,
    alpha: float = 1.0,
):
  """Computes the distillation loss between a student and teacher model.

  Args:
    model: The student model.
    params: The student model parameters.
    teacher_model: The teacher model.
    teacher_params: The teacher model parameters.
    batch: The input batch.
    temperature: The temperature scaling factor for the logits. Higher values
      soften the probability distributions, while lower values sharpen them.
    alpha: The weight of the soft loss (KL divergence). The hard loss weight is
      (1 - alpha). alpha = 1.0 means only soft loss is used.

  Returns:
    The distillation loss and a dictionary of extra outputs.
  """
  inputs = batch['decoder_input_tokens']
  loss_weights = batch['decoder_loss_weights']
  segment_ids = batch.get('decoder_segment_ids', None)
  segment_positions = batch.get('decoder_positions', None)
  logits, _ = model.apply(
      params, inputs,
      segment_ids=segment_ids, segment_positions=segment_positions)
  teacher_logits, _ = teacher_model.apply(
      teacher_params, inputs,
      segment_ids=segment_ids, segment_positions=segment_positions)
  # Always use float32 in softmax.
  logits = logits.astype(jnp.float32)
  teacher_logits = teacher_logits.astype(jnp.float32)
  teacher_logits = jax.lax.stop_gradient(teacher_logits)
  # Apply temperature scaling to soften the distributions.
  scaled_teacher_logits = teacher_logits / temperature
  scaled_student_logits = logits / temperature
  # Compute KL divergence: KL(teacher || student) with temp scaling.
  # Scale by temperature^2 to maintain gradient magnitude.
  soft_token_loss = (temperature**2) * jnp.einsum(
      'blv,blv->bl',
      jax.nn.softmax(scaled_teacher_logits),
      jax.nn.log_softmax(scaled_teacher_logits)
      - jax.nn.log_softmax(scaled_student_logits),
  )

  soft_loss = jnp.sum(soft_token_loss * loss_weights) / jnp.sum(loss_weights)
  soft_loss = sharding_lib.with_sharding_constraint(soft_loss, None)

  # Calculate Hard Loss
  targets = batch['decoder_target_tokens']
  targets_one_hot = jax.nn.one_hot(targets, logits.shape[-1], axis=-1)
  hard_token_loss = -jnp.einsum(
      'blv,blv->bl', targets_one_hot, jax.nn.log_softmax(logits)
  )
  hard_loss = jnp.sum(hard_token_loss * loss_weights) / jnp.sum(loss_weights)
  hard_loss = sharding_lib.with_sharding_constraint(hard_loss, None)

  # Combine soft and hard losses
  loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
  loss = sharding_lib.with_sharding_constraint(loss, None)
  # Compute accuracy.
  pred = jnp.argmax(logits, axis=-1)
  targets = jnp.argmax(teacher_logits, axis=-1)
  correct = (pred == targets).astype(jnp.float32) * loss_weights
  accuracy = jnp.sum(correct) / jnp.sum(loss_weights)
  accuracy = sharding_lib.with_sharding_constraint(accuracy, None)
  return loss, {'accuracy': accuracy}


def train_one_step(
    state,
    batch,
    model,
    opt,
    teacher_model=None,
    lr=1e-4,
    grad_accum_steps=-1,
    clip_grad_norm=-1,
    clip_update_norm=-1,
    clip_update_rms=-1,
    clip_local_update_rms=-1,
    weight_decay=-1,
    custom_loss_fn=None,
    add_log_info=False,
    distill_temperature: float = 1.0,
    distill_alpha: float = 1.0,
):
  clip_norm_fn = functools.partial(
      clip_tree_fn, fn=tree_norm, fn_name='norm')
  clip_rms_fn = functools.partial(
      clip_tree_fn, fn=tree_rms, fn_name='rms')

  norm_info_fn = functools.partial(
      compute_tree_info_fn, fn=tree_norm, fn_name='norm')
  rms_info_fn = functools.partial(
      compute_tree_info_fn, fn=tree_rms, fn_name='rms')

  log_dict = {}
  if add_log_info:
    log_dict.update(norm_info_fn(state['params'], name='weights'))
    log_dict.update(rms_info_fn(state['params'], name='weights'))

  def _compute_grad(batch):
    if teacher_model is None:
      loss_fn = (
          compute_train_loss
          if custom_loss_fn is None
          else custom_loss_fn
      )
      (loss, extra_output), grad = jax.value_and_grad(
          loss_fn, argnums=1, has_aux=True)(model, state['params'], batch)
    else:
      if custom_loss_fn is None:
        loss_fn = functools.partial(
            compute_distill_loss,
            temperature=distill_temperature,
            alpha=distill_alpha,
        )
      else:
        loss_fn = custom_loss_fn
      (loss, extra_output), grad = jax.value_and_grad(
          loss_fn, argnums=1, has_aux=True)(
              model, state['params'],
              teacher_model, state['teacher_params'], batch)
    return loss, extra_output, grad

  if grad_accum_steps > 1:
    # Prepare the batch for grad accumulation.
    batch = jax.tree.map(
        lambda x: einops.rearrange(
            x, '(g m) ... -> g m ...',
            g=grad_accum_steps),
        batch)

    # One step of grad accumulation.
    def grad_accum_step_fn(accum_info, minibatch):
      accum_loss, accum_grad = accum_info
      minibatch_loss, minibatch_extra_output, minibatch_grad = _compute_grad(
          minibatch)
      minibatch_loss_weight = minibatch_extra_output['loss_weight']
      accum_grad = jax.tree.map(
          lambda x, y: x + y * minibatch_loss_weight,
          accum_grad, minibatch_grad)
      accum_loss += minibatch_loss * minibatch_loss_weight
      return (accum_loss, accum_grad), minibatch_extra_output

    # Initialize the grad accumulation.
    zero_grad = jax.tree.map(
        lambda x: jnp.zeros_like(x, dtype=jnp.float32), state['params'])

    # Run grad accumulation with `scan``.
    (accum_loss, accum_grad), extra_output = jax.lax.scan(
        grad_accum_step_fn, init=(
            jnp.asarray(0.0, dtype=jnp.float32), zero_grad),
        xs=batch)

    # Calculate the final loss, grad and extra_output.
    loss_weight = extra_output.pop('loss_weight')
    total_loss_weight = jnp.sum(loss_weight, axis=0) + 1e-6
    for k, v in extra_output.items():
      if k.endswith('_max'):
        extra_output[k] = jax.tree.map(lambda x: jnp.max(x, axis=0), v)
      elif k.endswith('_min'):
        extra_output[k] = jax.tree.map(lambda x: jnp.min(x, axis=0), v)
      else:
        extra_output[k] = jax.tree.map(
            lambda x: jnp.sum(x * loss_weight, axis=0) / total_loss_weight, v)
    extra_output['loss_weight'] = total_loss_weight
    loss = accum_loss / total_loss_weight
    grad = jax.tree.map(lambda x: x / total_loss_weight, accum_grad)
  else:
    loss, extra_output, grad = _compute_grad(batch)

  # Log additional info computed by the loss function, for example,
  # prediction accuracy.
  log_dict.update(extra_output)

  log_dict.update(rms_info_fn(grad, name='grad'))
  if clip_grad_norm > 0:
    grad, clip_log_dict = clip_norm_fn(
        grad, name='grad', threshold=clip_grad_norm,
        add_log_info=add_log_info)
    log_dict.update(clip_log_dict)
  else:
    log_dict.update(norm_info_fn(grad, name='grad'))

  update, new_state = opt.apply(state, grad)
  if clip_update_norm > 0:
    update, clip_log_dict = clip_norm_fn(
        update, name='update', threshold=clip_update_norm,
        add_log_info=add_log_info)
    log_dict.update(clip_log_dict)
  else:
    log_dict.update(norm_info_fn(update, name='update'))

  if clip_update_rms > 0 or clip_local_update_rms > 0:
    update, clip_log_dict = clip_rms_fn(
        update, name='update',
        clip_local=clip_local_update_rms > 0,
        threshold=(clip_local_update_rms
                   if clip_local_update_rms > 0 else clip_update_rms),
        add_log_info=add_log_info)
    log_dict.update(clip_log_dict)
  else:
    log_dict.update(rms_info_fn(update, name='update'))

  if weight_decay > 0:
    update = jax.tree_util.tree_map(
        lambda x, y: x + y * weight_decay, update, new_state['params'])
  new_state = opt.apply_updates(
      new_state, jax.tree.map(lambda x: x * lr, update))
  new_state['steps'] += 1
  return loss, new_state, log_dict


def tree_norm(tree):
  flat, _ = jax.tree_util.tree_flatten(tree)
  norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in flat]))
  return norm


def tree_rms(tree):
  flat, _ = jax.tree_util.tree_flatten(tree)
  # Cast to float32 to avoid overflow.
  total_size = sum([jnp.asarray(jnp.size(x), jnp.float32) for x in flat])
  rms = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in flat]) / total_size)
  return rms


def safe_clip(x, val, threshold):
  # `val` can be zero but `threshold` will be non-zero. Following the guide
  # below to avoid NaN in the gradient.
  # https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
  return x / jnp.where(val > threshold, val / threshold, 1.0)


def clip_tree_fn(
    tree, name, threshold, fn, fn_name,
    clip_local=False, add_log_info=False):
  val = local_val = clipped_tree = None
  if add_log_info or not clip_local:
    val = fn(tree)
    clipped_tree = jax.tree_util.tree_map(
        lambda x: safe_clip(x, val, threshold), tree)

  if add_log_info or clip_local:
    local_val = jax.tree_util.tree_map(fn, tree)
    clipped_tree = jax.tree_util.tree_map(
        lambda x, y: safe_clip(x, y, threshold),
        tree, local_val)

  log_dict = {}
  if add_log_info:
    log_dict[f'global_{name}_{fn_name}'] = val
    log_dict[f'local_{name}_{fn_name}'] = local_val
    log_dict[f'global_clipped_{name}_{fn_name}'] = fn(clipped_tree)
    log_dict[f'local_clipped_{name}_{fn_name}'] = jax.tree_util.tree_map(
        fn, clipped_tree)
  return clipped_tree, log_dict


def compute_tree_info_fn(tree, name, fn, fn_name):
  log_dict = {}
  log_dict[f'global_{name}_{fn_name}'] = fn(tree)
  log_dict[f'local_{name}_{fn_name}'] = jax.tree_util.tree_map(fn, tree)
  return log_dict


################################################################################
# Experiment.


class TrainLoopRegistry(registry.RootRegistry):
  """Registry for train loop functions."""
  namespace: ClassVar[str] = 'TrainLoop'


@functools.partial(TrainLoopRegistry.register, name='default')
def run_experiment(
    config,
    sharding_config,
    mesh_shape,
    create_dataset=None,
    # Leave `experiment_dir` as empty string to skip saving experiment data.
    # Useful if no need to save any data and can reduce some overhead.
    experiment_dir='',
    dcn_mesh_shape=None,
    decoding_mesh_shape=None,
):
  if create_dataset is not None:
    warnings.warn('create_dataset is deprecated.')
  del create_dataset
  del decoding_mesh_shape
  logging.info('jax.process_index(): %s', jax.process_index())
  # Setup model, optimizer, initial state, and mesh.
  sharding_lib.set_default_mesh_shape(
      mesh_shape=mesh_shape, dcn_mesh_shape=dcn_mesh_shape)
  helper = ExperimentHelper(
      experiment_dir,
      ckpt_interval=config.ckpt_interval,
      ckpt_max_to_keep=config.ckpt_max_to_keep,
      ckpt_keep_period=config.ckpt_keep_period,
      num_train_steps=config.num_train_steps,
      metric_log_interval=config.tb_log_interval,
      log_additional_info=config.log_additional_info,
  )
  model, extra_output = create_model(config, sharding_config)
  teacher_model = extra_output.get('teacher')
  helper.save_config_info(config, sharding_config, model)
  opt = config.optimizer
  state = get_init_state(
      config, sharding_config, helper.ckpt_mngr, helper.ckpt_dir)
  helper.save_state_info(state)

  # Compile loss, train and learning rate functions.
  @functools.partial(
      jax.jit, donate_argnames=['state'], static_argnames=['add_log_info']
  )
  def train_one_step_fn(state, batch, lr, add_log_info=False):
    return train_one_step(
        state=state,
        batch=batch,
        lr=lr,
        model=model,
        opt=opt,
        grad_accum_steps=config.grad_accum_steps,
        teacher_model=teacher_model,
        clip_grad_norm=config.clip_grad_norm,
        clip_update_norm=config.clip_update_norm,
        clip_local_update_rms=config.clip_local_update_rms,
        weight_decay=config.weight_decay,
        add_log_info=add_log_info,
        distill_temperature=config.distill_temperature,
        distill_alpha=config.distill_alpha,
    )
  lr_fn = common.named_jit(opt_lib.create_lr_schedule(config), 'lr_fn')

  # Prepare datasets.
  logging.info('Initializing dataset.')
  train_set = data_lib.create_iter_dataset(config, training=True)
  logging.info('sharding_config.data_partition: %s',
               sharding_config.data_partition)

  train_iter = iter(train_set)

  train_iter_state = None
  if helper.ckpt_mngr and helper.ckpt_mngr.latest_step() is not None:
    data_state = ckpt_lib.load_data_state_from_dir(
        helper.ckpt_dir, helper.ckpt_mngr.latest_step()
    )
    assert isinstance(data_state, Mapping)
    train_iter_state = data_state.get('train_iter_state', None)
  if train_iter_state is not None:
    train_iter.set_state(train_iter_state)

  # Start training.
  prev_step_timestamp = time.time()
  final_result = {}
  steps = int(state['steps'].addressable_data(0))

  # Create eval_fn for validation set.
  if config.use_validation_set:
    loss_fn = common.named_jit(
        compute_eval_loss, 'validation_loss_fn', model=model
    )
    validation_set = data_lib.create_iter_dataset(
        config, training=False
    )
    eval_fn = functools.partial(
        run_eval,
        eval_set=validation_set,
        num_eval_steps=config.validation_num_eval_steps,
        loss_fn=loss_fn,
    )
  else:
    eval_fn = None
  agg_metrics = {}
  eval_result = {}
  should_early_stop = False
  while steps <= config.num_train_steps and not should_early_stop:
    with jax.profiler.StepTraceAnnotation('train', step_num=steps):
      logging.info('steps: %s', steps)
      helper.save_ckpt(state, steps, data=train_iter.get_state())
      # Run eval every validation_eval_interval steps and at the very end.
      if config.use_validation_set and (
          steps % config.validation_eval_interval == 0
          or steps == config.num_train_steps
      ):
        eval_result = eval_fn(state=state)
        helper.write_scalars(steps, eval_result)
        helper.flush()

      t1 = time.time()
      batch = next(train_iter)
      logging.info('batch=%s', batch)

      batch = build_global_array_from_replicated(
          batch, data_partition=(('replica', 'data'),)
      )
      data_generation_step_time = time.time() - t1

      t1 = time.time()
      lr = lr_fn(state['steps'])
      loss, state, log_dict = train_one_step_fn(
          state=state,
          batch=batch,
          lr=lr,
          add_log_info=helper.should_log_additional_info(steps),
      )
      train_loss = float(loss.addressable_data(0))
      train_step_time = time.time() - t1
      logging.info('train_loss: %s', train_loss)

      if helper.should_log_additional_info(steps):
        # Log batch stats info for debugging purpose.
        batch_stats_info = compute_batch_stats_info(batch)
        logging.info('========== batch_stats_info ==========')
        for k, v in batch_stats_info.items():
          logging.info('%s: %s', k, v)
        log_dict.update(batch_stats_info)

      step_time = time.time() - prev_step_timestamp
      prev_step_timestamp = time.time()

      # Track and log all the metrics.
      if helper.should_log_additional_info(steps):
        helper.add_metric('total_step_time_with_additional_info', step_time)
        helper.add_metric(
            'train_step_time_with_additional_info', train_step_time)
      else:
        helper.add_metric('total_step_time', step_time)
        helper.add_metric('train_step_time', train_step_time)
      helper.add_metric('avg_total_step_time', step_time)
      logging.info('%s secs per step, log_additional_info: %s',
                   step_time, helper.should_log_additional_info(steps))
      helper.add_metric('loss', train_loss)
      helper.add_metric(
          'accuracy', float(log_dict['accuracy'].addressable_data(0)))
      helper.add_metric(
          'data_generation_step_time', data_generation_step_time)

      agg_metrics = helper.get_aggregated_metrics()
      should_early_stop = should_early_stop or (
          config.early_stop and
          config.early_stop.should_stop(
              steps, agg_metrics))
      if helper.should_log_metrics(steps):
        t1 = time.time()
        metrics_dict = dict(
            lr=lr,
            secs_per_step=agg_metrics['avg_total_step_time'],
            steps_per_sec=1 / agg_metrics['avg_total_step_time'],
        )
        metrics_dict.update(agg_metrics)
        metrics_dict.update(flatten_dict(log_dict))
        helper.write_scalars(steps, metrics_dict)
        helper.flush()
        event_write_time = time.time() - t1
        logging.info('%s secs per writing metrics.', event_write_time)
      steps += 1
  final_result['steps'] = steps - 1
  final_result['train_loss'] = float(agg_metrics['loss'])
  final_result['train_accuracy'] = float(agg_metrics['accuracy'])
  if eval_result:
    final_result['validation_loss'] = float(eval_result['eval_loss'])
    final_result['validation_accuracy'] = float(
        eval_result['eval_accuracy'])
  final_result['early_stop'] = should_early_stop
  if should_early_stop: logging.info('Training is early stopped!')
  helper.close(final_result)
  return final_result


def create_model(config, sharding_config):
  if sharding_config is None:
    sharding_config = config_lib.GSPMDSharding()
  if not (model := getattr(config, 'model', None)):
    model_cls = module.ModuleRegistry.get(config.model_name)
    model = model_cls(config, sharding_config=sharding_config)
  teacher_model = None
  if teacher_config := getattr(config, 'teacher', None):
    teacher_model_cls = module.ModuleRegistry.get(teacher_config.model_name)
    teacher_model = teacher_model_cls(
        teacher_config, sharding_config=sharding_config)
  return model, {'teacher': teacher_model}


def build_global_array_from_replicated(
    batch: PyTree, data_partition: PartitionAnnotation = None
) -> PyTree:
  return jax.tree_util.tree_map(
      lambda x: jax.lax.with_sharding_constraint(
          jnp.asarray(x), sharding_lib.mesh_sharding(data_partition)
      ),
      batch,
  )


def build_global_batch_from_sharded(
    batch: PyTree, data_partition: PartitionAnnotation = None
) -> PyTree:
  data_sharding = sharding_lib.mesh_sharding(data_partition)

  def _build_global_array_from_sharded(array: np.ndarray):
    if array.ndim < 1:
      raise ValueError(f'Array {array} must have at least 1 dimension.')
    global_shape = (array.shape[0] * jax.process_count(), *array.shape[1:])
    global_array = jax.make_array_from_process_local_data(
        data_sharding, array, global_shape
    )
    return global_array

  return jax.tree_util.tree_map(_build_global_array_from_sharded, batch)


def get_init_state(config, sharding_config, ckpt_mngr, ckpt_dir):
  model, extra_output = create_model(config, sharding_config)
  teacher_model = extra_output.get('teacher')
  opt = config.optimizer
  prng_key = jax.random.key(config.model_seed)
  if (ckpt_mngr and
      (latest_step := ckpt_mngr.latest_step()) is not None):
    # Continue training from lastest ckpt.
    abstract_state = opt.init(ckpt_lib.get_abstract_params(model))
    state = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir, abstract_state, latest_step
    )
  elif config.init_ckpt_dir:
    # Initialize from a given external ckpt.
    abstract_params = ckpt_lib.get_abstract_params(model)
    logging.info('abstract_params: %s', abstract_params)
    if config.init_ckpt_opt_state:
      # Initialize params and opt state from a given external ckpt.
      abstract_state = opt.init(abstract_params)
      state = ckpt_lib.load_checkpoint_from_dir(
          config.init_ckpt_dir, abstract_state, config.init_ckpt_step,
          ckpt_format=config.init_ckpt_format,
      )
    else:
      # Only initialize params from a given external ckpt.
      abstract_state = {'params': abstract_params}
      state = ckpt_lib.load_checkpoint_from_dir(
          config.init_ckpt_dir,
          abstract_state,
          config.init_ckpt_step,
          ckpt_format=config.init_ckpt_format,
      )
      state = opt.init(state['params'])
    if config.reset_steps:
      state['steps'] = opt_lib.get_init_steps()
  else:  # initialize from scratch.
    state = opt.init(jax.jit(model.init)(prng_key))

  # Add the teacher configuration if specified.
  if teacher_model is not None:
    abstract_teacher_state = {
        'params': ckpt_lib.get_abstract_params(teacher_model)
    }
    teacher_state = ckpt_lib.load_checkpoint_from_dir(
        config.teacher_ckpt_dir,
        abstract_teacher_state,
        config.teacher_ckpt_step,
        ckpt_format=config.teacher_ckpt_format,
    )
    state['teacher_params'] = teacher_state['params']
  return state


def run_eval(eval_set, num_eval_steps, loss_fn, state) -> dict[str, Any]:
  mean_eval_loss = 0.0
  mean_eval_accuracy = 0.0
  # The `loss_weights` is normally the same as `num_tokens`.
  total_weights = 0.0
  total_num_tokens = 0
  eval_start_time = time.time()
  eval_steps = 0
  for eval_steps, eval_batch in enumerate(eval_set):
    if num_eval_steps > 0 and (eval_steps >= num_eval_steps):
      break
    eval_batch = build_global_array_from_replicated(
        eval_batch, (('replica', 'data'),)
    )
    eval_batch_stats_info = compute_batch_stats_info(eval_batch)
    eval_loss, extra_output = loss_fn(
        params=state['params'], batch=eval_batch)
    eval_loss = float(eval_loss.addressable_data(0))
    eval_accuracy = float(
        extra_output['accuracy'].addressable_data(0))
    num_tokens = float(
        eval_batch_stats_info['num_tokens'].addressable_data(0))
    batch_weights = float(
        eval_batch_stats_info['total_weights'].addressable_data(0))
    if total_weights <= 1e-6:
      mean_eval_loss = eval_loss
      mean_eval_accuracy = eval_accuracy
    else:
      weights_ratio = batch_weights / total_weights
      # Iteratively update mean_eval_loss to avoid numerical overflow.
      mean_eval_loss = (
          mean_eval_loss + (eval_loss - mean_eval_loss) * weights_ratio)
      mean_eval_accuracy = (
          mean_eval_accuracy +
          (eval_accuracy - mean_eval_accuracy) * weights_ratio)
    total_weights += batch_weights
    total_num_tokens += num_tokens
  eval_time = time.time() - eval_start_time
  if eval_steps == 0:
    eval_step_time = 0
  else:
    eval_step_time = eval_time / eval_steps
  logging.info(
      '%s secs in validation eval, %s steps, %s secs per step.',
      eval_time, eval_steps, eval_step_time)
  return dict(eval_loss=float(mean_eval_loss),
              eval_accuracy=float(mean_eval_accuracy),
              eval_weights=float(total_weights),
              eval_tokens=int(total_num_tokens),
              eval_time=float(eval_time),
              eval_step_time=int(eval_step_time))


def flatten_dict(d: dict[str, Any]):
  result_dict = {}
  for k, v in d.items():
    if isinstance(v, dict):
      vd = flatten_dict(v)
      for vk, vv in vd.items():
        new_key = k + '/' + vk
        if new_key in result_dict:
          raise ValueError(f'Duplicate key: {vk}')
        else:
          result_dict[new_key] = vv
    else:
      result_dict[k] = v
  return result_dict


@jax.jit
def compute_batch_stats_info(
    batch: Batch,
    pad_id: int = 0) -> Mapping[str, Any]:
  result = {}
  batch_size = batch['decoder_target_tokens'].shape[0]
  result['num_seq'] = batch_size
  seq_len = batch['decoder_target_tokens'].shape[1]
  result['seq_len'] = seq_len

  tokens_per_seq = jnp.sum(
      batch['decoder_target_tokens'] != pad_id, axis=-1, dtype=jnp.float32
  )
  result['num_tokens'] = jnp.sum(tokens_per_seq)
  result['avg_num_tokens_per_seq'] = jnp.mean(tokens_per_seq)
  result['std_num_tokens_per_seq'] = jnp.std(tokens_per_seq)

  ratio_of_nonpad_tokens = tokens_per_seq / seq_len
  result['avg_ratio_nonpad_tokens_per_seq'] = jnp.mean(ratio_of_nonpad_tokens)
  result['std_ratio_nonpad_tokens_per_seq'] = jnp.std(ratio_of_nonpad_tokens)

  loss_weights = batch.get('decoder_loss_weights', None)
  if loss_weights is None:
    loss_weights = jnp.ones((batch_size, seq_len), dtype='bool')

  loss_weights_per_seq = jnp.sum(loss_weights, axis=-1, dtype=jnp.float32)
  result['total_weights'] = jnp.sum(loss_weights_per_seq)
  result['avg_weights_per_seq'] = jnp.mean(loss_weights_per_seq)
  result['std_weights_per_seq'] = jnp.std(loss_weights_per_seq)

  if 'decoder_segment_ids' in batch:
    num_segments = jnp.max(batch['decoder_segment_ids'], axis=-1)
    result['num_segments'] = jnp.sum(num_segments)
    result['avg_num_segments_per_seq'] = jnp.mean(num_segments)
    result['std_num_segments_per_seq'] = jnp.std(num_segments)
    result['avg_segment_length'] = result['num_tokens'] / result['num_segments']
  return result


################################################################################
# Decoding


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class SamplingState:
  """Sampling state.

  It takes `tokens[position]` as input to run LLM forward pass. `decode_state`
  at `position` would be updated during the forward pass. Sampled output tokens
  and scores are put in `tokens[position+1]` and `token_scores[position+1]` when
  the input is `tokens[position]`.
  """

  prng_key: PRNGKey
  decode_state: PyTree  # kv cache in decode_state_length size
  tokens: Array  # [batch, seq_len], seq_len > decode_state_length
  token_logprobs: Array  # [batch, decode_state_length+1], [:, 0] is dummy
  token_scores: Array  # [batch, decode_state_length+1], [:, 0] is dummy
  position: Array  # [], scale, must be > 0 and < decode_state_length
  # when position == decode_state_length, the sampling state is finished.
  input_lens: Array  # [batch, 1], bos counted
  max_decode_steps: Array  # [batch, 1]
  eos_ids: Array  # [n_eos]

  def __post_init__(self):
    assert not self.position.shape
    assert self.input_lens.shape == (self.batch_size, 1)

  @functools.cached_property
  def is_pad_seq(self) -> Array:
    """This sequence is a padding sequence, in [batch, 1]."""
    return self.input_lens == 0

  @functools.cached_property
  def input_tokens(self) -> Array:
    return jax.lax.dynamic_slice_in_dim(self.tokens, self.position, 1, axis=1)

  @functools.cached_property
  def reached_eos(self) -> Array:
    """This position is output and eos, in [batch, 1]."""
    # eos_ids: [n_eos]
    # input_tokens: [batch, 1]
    # output: [batch, n_eos] -> [batch, 1]
    return (self.position >= self.input_lens) & jnp.any(
        self.input_tokens == self.eos_ids, axis=-1, keepdims=True
    )

  @functools.cached_property
  def has_ended(self) -> Array:
    """Returns whether each sequence in the batch is done with generation."""
    return (
        self.is_pad_seq
        | (self.position + 1 - self.input_lens > self.max_decode_steps)
        | self.reached_eos
    )

  @functools.cached_property
  def next_position_is_output(self) -> Array:
    return self.position + 1 >= self.input_lens  # [batch, 1]

  @functools.cached_property
  def next_tokens(self) -> Array:
    return jax.lax.dynamic_slice_in_dim(
        self.tokens, self.position + 1, 1, axis=1
    )

  def updated_tokens(self, output_tokens: Array) -> Array:
    return jax.lax.dynamic_update_slice_in_dim(
        self.tokens, output_tokens, self.position + 1, axis=1
    )

  def updated_token_logprobs(self, output_logprobs: Array) -> Array:
    return jax.lax.dynamic_update_slice_in_dim(
        self.token_logprobs, output_logprobs, self.position + 1, axis=1
    )

  def updated_token_scores(self, output_scores: Array) -> Array:
    return jax.lax.dynamic_update_slice_in_dim(
        self.token_scores, output_scores, self.position + 1, axis=1
    )

  @property
  def decode_state_length(self) -> int:
    return self.token_scores.shape[1] - 1

  @property
  def batch_size(self) -> int:
    return self.tokens.shape[0]

  def pad_to(self, length: int) -> 'SamplingState':
    if length <= self.decode_state_length:
      return self
    tokens = pad_to_along_axis(self.tokens, length + 1, axis=1)
    token_logprobs = pad_to_along_axis(self.token_logprobs, length + 1, axis=1)
    token_scores = pad_to_along_axis(self.token_scores, length + 1, axis=1)
    decode_state = pad_decode_state_to(self.decode_state, length)
    return dataclasses.replace(
        self,
        decode_state=decode_state,
        tokens=tokens,
        token_logprobs=token_logprobs,
        token_scores=token_scores,
    )


@sampling_lib.SamplingRegistry.register
@dataclasses.dataclass(frozen=True)
class SamplingOutput:
  input_chunks: sampling_lib.ChunkSequence
  input_token_ids: list[int]

  output_chunks: sampling_lib.ChunkSequence
  output_token_ids: list[int]

  # Sampling logprobs of the output tokens.
  output_token_logprobs: list[float]
  # Whether the output was truncated before reaching natural eos.
  is_truncated: bool

  # Log probs of the input tokens computed by log_softmax.
  # The score values are not affected by the sampling params.
  input_token_scores: list[float]
  # Log probs of the output tokens computed by log_softmax.
  # The score values are not affected by the sampling params.
  output_token_scores: list[float]

  @functools.cached_property
  def input_text(self) -> str:
    return sampling_lib.chunks_as_text(self.input_chunks)

  @functools.cached_property
  def output_text(self) -> str:
    return sampling_lib.chunks_as_text(self.output_chunks)

  @functools.cached_property
  def sum_output_logprob(self) -> float:
    return np.maximum(
        np.sum(self.output_token_logprobs), neg_inf(np.float32)
    ).item()

  @functools.cached_property
  def avg_output_logprob(self) -> float:
    return np.mean(self.output_token_logprobs).item()

  @functools.cached_property
  def sum_input_score(self) -> float:
    return np.maximum(
        np.sum(self.input_token_scores), neg_inf(np.float32)
    ).item()

  @functools.cached_property
  def avg_input_score(self) -> float:
    return np.mean(self.input_token_scores)

  @functools.cached_property
  def sum_output_score(self) -> float:
    return np.maximum(
        np.sum(self.output_token_scores), neg_inf(np.float32)
    ).item()

  @functools.cached_property
  def avg_output_score(self) -> float:
    return np.mean(self.output_token_scores).item()


@dataclasses.dataclass(frozen=True)
class ScoringParams:
  temperature: float = 1.0
  top_k: int = -1
  top_p: float = 1.0

  @classmethod
  def from_sampling_params(cls, sampling_params: SamplingParams) -> Self:
    return cls(
        temperature=sampling_params.temperature,
        top_k=sampling_params.top_k,
        top_p=sampling_params.top_p,
    )


@dataclasses.dataclass(frozen=True)
class ScoringOutput:
  params: ScoringParams

  input_chunks: sampling_lib.ChunkSequence
  input_token_ids: list[int]

  output_chunks: sampling_lib.ChunkSequence
  output_token_ids: list[int]

  # Log probs of the input tokens computed by log_softmax.
  # The score values are not affected by the sampling params.
  input_token_scores: list[float]

  # Log probs of the input tokens computed by log_softmax.
  # The score values are not affected by the sampling params.
  output_token_scores: list[float]

  @functools.cached_property
  def input_text(self) -> str:
    return sampling_lib.chunks_as_text(self.input_chunks)

  @functools.cached_property
  def output_text(self) -> str:
    return sampling_lib.chunks_as_text(self.output_chunks)

  @functools.cached_property
  def sum_input_score(self) -> float:
    return np.sum(self.input_token_scores).item()

  @functools.cached_property
  def avg_input_score(self) -> float:
    return np.mean(self.input_token_scores).item()

  @functools.cached_property
  def sum_output_score(self) -> float:
    return np.sum(self.output_token_scores).item()

  @functools.cached_property
  def avg_output_score(self) -> float:
    return np.mean(self.output_token_scores).item()


class LMInterface:

  def __init__(
      self,
      model: module.SimplyModule,
      params: PyTree,
      vocab: tokenization.SimplyVocab[str] | None = None,
      input_processor: sampling_lib.InputProcessorInterface | None = None,
      default_sampling_params: SamplingParams | None = None,
      bos_id: int | None = None,
      pad_id: int | None = None,
      extra_eos_ids: Sequence[int] | None = None,
      extra_eos_tokens: Sequence[str] | None = None,
  ) -> None:
    """An interface to interact with a language model.

    Args:
      model: The model to use, for example, a TransformerLM instance.
      params: The `params` to use in `model.apply`.
      vocab: The vocabulary instance to use. Either `vocab` or `input_processor`
        should be specified. If `vocab` is specified, it will be used to a
        instantiate a default input processor for basic text inputs.
      input_processor: The input processor to use, for specialized input
        processing. For basic text inputs, it is enough to specify `vocab`.
      default_sampling_params: Default sampling params for `generate`.
      bos_id: The bos id to use, if not given then it will use the `bos_id`
        field of the `vocab`.
      pad_id: The pad id to use, if not given then it will use the `pad_id`
        field of the `vocab`.
      extra_eos_ids: Extra eos ids to include.
      extra_eos_tokens: Extra eos tokens to include.
    """
    self.model = model
    if input_processor:
      self.input_processor = input_processor
    else:
      assert vocab is not None, 'Must provide one of vocab or input_processor!'
      self.input_processor = sampling_lib.BasicTextInputProcessor(
          vocab,
          bos_id_override=bos_id,
          pad_id_override=pad_id,
          extra_eos_ids=extra_eos_ids,
          extra_eos_tokens=extra_eos_tokens,
      )
    self.default_sampling_params = default_sampling_params or SamplingParams()

    def prefill_fn(
        params: PyTree,
        inputs: Array,
        extra_inputs: Mapping[str, Array],
        position: int,
        return_logits: bool = True,
    ) -> tuple[Array | None, PyTree]:
      logits, extra_output = model.apply(
          params,
          inputs,
          extra_inputs=extra_inputs,
          decode_state={'prefill_position': position},
      )
      if return_logits:
        return logits, extra_output
      return None, extra_output

    self.prefill_fn = jax.jit(prefill_fn, static_argnames=['return_logits'])

    self.decode_fn = jax.jit(
        common.named_partial_fn(
            continue_decode,
            'decode_fn',
            apply_fn=model.apply,
        ),
        donate_argnames='init_sampling_state',
    )
    self.pad_state_to_fn = jax.jit(
        SamplingState.pad_to,
        donate_argnames='self',
        static_argnames=['length'],
    )
    self.model_params = params

  @property
  def eos_ids(self) -> list[int]:
    return self.input_processor.eos_ids

  def generate(
      self,
      input_text: (
          sampling_lib.SamplingInput | Sequence[sampling_lib.SamplingInput]
      ),
      prng_key: int | PRNGKey | None = None,
      params: PyTree = None,
      # TODO: Deprecate in favor of setting directly in
      # SamplingParams.
      prefill_size: int = -1,
      sampling_params: SamplingParams | None = None,
      scoring_params: ScoringParams | None = None,
      include_eos_in_output_text: bool = False,
      scoring_inputs: bool = True,
      batch_size: int | None = None,
  ) -> list[SamplingOutput] | list[list[SamplingOutput]]:
    """Generate samples from a given input text.

    Args:
      input_text: Single input or sequence of inputs to generate samples for.
        Input can be either string or sequence of Chunks.
      prng_key: A PRNGKey or seed for controlling the randomness. The key would
        be released inside, and cannot be reused.
      params: parameters of the model, if None, use the default parameters.
      prefill_size: Prefill size to use for the generation, if set to a
        non-positive value, it will be inferred from sampling params. At prefill
        stage, prefill_size of input tokens (bos counted) will be processed.
        Recommended to set to multiples of 128.
      sampling_params: Sampling params to use for the generation.
      scoring_params: Scoring params to score the input and generated output.
      include_eos_in_output_text: Whether to include the eos token when
        generating the `output_text` field of the sampling outputs. Note that
        even if this is set to `True`, the `vocab.decode` can still skip the eos
        token.
      scoring_inputs: Whether to compute the log likelihood of the input and
        generated output.
      batch_size: The batch size to use for the generation. If not specified,
        the batch size will be inferred from the length of the input text.

    Returns:
      If the `input_text` is a single text string or a single raw sequence,
      returns a list of `SamplingOutput`, else if the `input_text` is a list
      of text strings or a list of raw sequences,
      returns a list of list of `SamplingOutput`.

      The result `SamplingOutput` instances for each `input_text` are ranked by
      the `sort_by` field of the `sampling_params`.

      Note that the eos token and bos token are included in the
      `output_token_ids` and `input_token_ids` field of the `SamplingOutput`,
      but the `input_token_scores` will not include the bos token so its length
      is one less than `input_token_ids`.
    """
    if params is None:
      params = self.model_params

    if prng_key is None:
      seed = int(time.time() * 1000)
      # This is to guarantee all hosts have the same seed.
      seed = jax.experimental.multihost_utils.broadcast_one_to_all(seed)
      prng_key = jax.random.key(seed=seed)
    elif isinstance(prng_key, int):
      prng_key = jax.random.key(seed=prng_key)
    if sampling_params is None:
      sampling_params = self.default_sampling_params
    if prefill_size > 0:
      sampling_params = dataclasses.replace(
          sampling_params, prefill_size=prefill_size
      )

    if scoring_params is None:
      scoring_params = ScoringParams.from_sampling_params(sampling_params)

    is_singleton_input = isinstance(input_text, str)
    if input_text and isinstance(input_text[0], sampling_lib.Chunk):
      is_singleton_input = True

    if is_singleton_input:
      raw_inputs = [sampling_lib.input_as_chunks(input_text)]
    else:
      raw_inputs = [sampling_lib.input_as_chunks(x) for x in input_text]

    unpadded_inputs = [
        self.input_processor.encode(
            x, max_input_len=sampling_params.max_input_len)
        for x in raw_inputs
    ]
    processed_input = sampling_lib.ProcessedInputBatch.from_unpadded_inputs(
        unpadded_inputs, pad_id=self.input_processor.pad_id
    )
    # Compute before padding the batch which may create length zero inputs.
    decoding_schedule = sampling_params.get_decoding_schedule(
        min_input_length=processed_input.min_length,
        max_input_length=processed_input.max_length,
    )

    if batch_size is not None:
      if processed_input.batch_size > batch_size:
        raise ValueError(
            f'Batch size {processed_input.batch_size=} is larger than the'
            f' specified batch size {batch_size=}.'
        )
      if processed_input.batch_size < batch_size:
        processed_input = processed_input.pad_batch_to(batch_size)
        logging.info('processed_input=%s after batch padding', processed_input)

    processed_input = processed_input.pad_to(
        1
        + max(
            decoding_schedule.get_next_length(processed_input.max_length - 1),
            decoding_schedule.prefill_size,
        )
    )
    if sampling_params.num_samples > 1:
      processed_input = processed_input.repeat(sampling_params.num_samples)

    position = decoding_schedule.begin_position
    logits, extra_output = self.prefill_fn(
        params,
        processed_input.token_slice(0, decoding_schedule.prefill_size),
        extra_inputs=processed_input.extra_inputs,
        position=position,
        return_logits=scoring_inputs,
    )
    if scoring_inputs:
      logits = sharding_lib.with_sharding_constraint(
          logits, (('replica', 'data'), 'model', None)
      )

      token_scores = compute_log_likelihood(
          logits,
          processed_input.token_slice(1, decoding_schedule.prefill_size + 1),
          temperature=scoring_params.temperature,
          top_k=scoring_params.top_k,
          top_p=scoring_params.top_p,
      )
    else:
      token_scores = jnp.zeros(
          (processed_input.batch_size, decoding_schedule.prefill_size),
          dtype=jnp.float32,
      )
    del logits  # Release logits to save HBM.
    # For better readability, we add a dummy score for the BOS token, so that
    # i-th score and logprob corresponds to the i-th token.
    token_scores = pad_along_axis(token_scores, (1, 0), axis=1)
    token_logprobs = jnp.zeros_like(token_scores)

    sampling_state = SamplingState(
        prng_key=jnp.copy(prng_key),
        position=jnp.array(position),
        decode_state=extra_output['decode_state'],
        tokens=processed_input.tokens,
        token_logprobs=token_logprobs,
        token_scores=token_scores,
        input_lens=jnp.reshape(processed_input.lengths, [-1, 1]),
        max_decode_steps=einops.repeat(
            jnp.array(sampling_params.max_decode_steps),
            '-> b 1',
            b=processed_input.batch_size,
        ),
        eos_ids=jnp.array(self.input_processor.eos_ids, dtype=jnp.int32),
    )

    # NOTE that `position + 1` is the output position.
    logging.info('position: %d', position)
    logging.info('max_input_len: %d', processed_input.max_length)
    logging.info(
        'sampling_params.max_decode_steps: %d',
        sampling_params.max_decode_steps,
    )
    logging.info(
        'sampling_params.max_seq_len: %d', sampling_params.max_seq_len
    )

    while position < decoding_schedule.end_position:
      sampling_state = self.pad_state_to_fn(
          sampling_state, length=decoding_schedule.get_next_length(position)
      )
      sampling_state = self.decode_fn(
          params=params,
          init_sampling_state=sampling_state,
          temperature=sampling_params.temperature,
          top_k=sampling_params.top_k,
          top_p=sampling_params.top_p,
          scoring_temperature=scoring_params.temperature,
          scoring_top_k=scoring_params.top_k,
          scoring_top_p=scoring_params.top_p,
      )
      position = jax.device_get(sampling_state.position)
      if jax.device_get(jnp.all(sampling_state.has_ended)):
        break
    # Post process the outputs.
    all_raw_token_ids = jax.experimental.multihost_utils.process_allgather(
        sampling_state.tokens, tiled=True
    ).tolist()
    all_raw_token_logprobs = jax.experimental.multihost_utils.process_allgather(
        sampling_state.token_logprobs, tiled=True
    ).tolist()
    all_raw_token_scores = jax.experimental.multihost_utils.process_allgather(
        sampling_state.token_scores, tiled=True
    ).tolist()

    sample_outputs = []
    num_outputs = len(raw_inputs) * sampling_params.num_samples
    for i in range(num_outputs):
      raw_token_ids = all_raw_token_ids[i]
      assert isinstance(raw_token_ids, list)
      assert isinstance(raw_token_ids[0], int)
      raw_token_logprobs = all_raw_token_logprobs[i]
      assert isinstance(raw_token_logprobs, list)
      assert isinstance(raw_token_logprobs[0], float)
      raw_token_scores = all_raw_token_scores[i]
      assert isinstance(raw_token_scores[0], float)
      assert isinstance(raw_token_scores, list)
      input_token_ids = []
      input_token_scores = []
      output_token_ids = []
      output_token_scores = []
      output_token_logprobs = []
      for t, token_id in enumerate(raw_token_ids):
        if t >= min(
            # Ensure python int to prevent overflow.
            int(processed_input.lengths[i])
            + sampling_params.max_decode_steps,
            sampling_params.max_seq_len,
        ):
          break
        if t < processed_input.lengths[i]:
          input_token_ids.append(token_id)
          if t > 0:
            # The first token score is dummy.
            input_token_scores.append(raw_token_scores[t])
        else:
          output_token_ids.append(token_id)
          output_token_scores.append(raw_token_scores[t])
          output_token_logprobs.append(raw_token_logprobs[t])
          if token_id in self.input_processor.eos_ids:
            # Generated eos token can only appear in output_tokens.
            break

      ends_in_eos = (
          output_token_ids
          and output_token_ids[-1] in self.input_processor.eos_ids
      )
      if ends_in_eos and not include_eos_in_output_text:
        output_chunks = self.input_processor.decode(output_token_ids[:-1])
      else:
        output_chunks = self.input_processor.decode(output_token_ids)

      sample_outputs.append(
          SamplingOutput(
              input_chunks=raw_inputs[i // sampling_params.num_samples],
              output_chunks=output_chunks,
              input_token_ids=input_token_ids,
              output_token_ids=output_token_ids,
              output_token_logprobs=output_token_logprobs,
              input_token_scores=input_token_scores,
              output_token_scores=output_token_scores,
              is_truncated=(not ends_in_eos),
          )
      )

    if not is_singleton_input:
      sample_outputs = [
          sample_outputs[i : i + sampling_params.num_samples]
          for i in range(0, len(sample_outputs), sampling_params.num_samples)
      ]

    if sampling_params.sort_by is not None:
      if is_singleton_input:
        sample_outputs.sort(key=lambda x: getattr(x, sampling_params.sort_by))
      else:
        for batch in sample_outputs:
          assert isinstance(batch, list)
          batch.sort(key=lambda x: getattr(x, sampling_params.sort_by))

    return sample_outputs

  def score(
      self,
      input_text: sampling_lib.SamplingInput,
      output_text: sampling_lib.SamplingInput,
      params: PyTree | None = None,
      scoring_params: ScoringParams | None = None,
  ) -> ScoringOutput:
    """Decode on given texts to compute their token scores (loglikelihood).

    Args:
      input_text: Input, which can be either string or Chunks.
      output_text: Output, which can be either string or Chunks.
      params: parameters of the model, if None, use the default parameters.
      scoring_params: parameters of the model.

    Returns:
      The `ScoringOutput` instance.
    """
    if scoring_params is None:
      scoring_params = ScoringParams.from_sampling_params(
          self.default_sampling_params
      )

    input_chunks = sampling_lib.input_as_chunks(input_text)
    output_chunks = sampling_lib.input_as_chunks(output_text)

    # TODO: add more choices for whether and how to have the EOS token
    processed_input_and_output = self.input_processor.encode(
        [*input_chunks, *output_chunks]
    )
    all_tokens = processed_input_and_output.tokens
    all_scores = self.score_tokens(
        all_tokens,
        extra_inputs=processed_input_and_output.extra_inputs,
        scoring_params=scoring_params,
        params=self.model_params if params is None else params,
    )

    processed_input = self.input_processor.encode(input_chunks)
    input_len = len(processed_input.tokens)

    return ScoringOutput(
        params=scoring_params,
        input_chunks=input_chunks,
        input_token_ids=list(all_tokens[:input_len]),
        output_chunks=output_chunks,
        output_token_ids=list(all_tokens[input_len:]),
        input_token_scores=all_scores[: input_len - 1],
        output_token_scores=all_scores[input_len - 1 :],
    )

  def score_tokens(
      self,
      tokens: Sequence[int],
      extra_inputs: Mapping[str, Array] | None = None,
      scoring_params: ScoringParams | None = None,
      params: PyTree | None = None,
  ) -> list[float]:
    """Compute the token scores (loglikelihood) of a list of tokens.

    Args:
      tokens: list of tokens.
      extra_inputs: any extra inputs for TransformerLM
      scoring_params: parameters of the model.
      params: parameters of the model, if None, use the default parameters.

    Returns:
      token_scores: loglikelihood of tokens.
    """
    if scoring_params is None:
      scoring_params = ScoringParams.from_sampling_params(
          self.default_sampling_params
      )
    tokens = np.array(tokens).reshape([1, -1])
    extra_inputs = jax.tree_util.tree_map(
        lambda x: x.expand_dims(axis=0), extra_inputs
    )
    apply_fn = self.model.apply
    logits, _ = jax.jit(apply_fn)(
        self.model_params if params is None else params,
        tokens[:, :-1],
        extra_inputs=extra_inputs,
    )
    token_scores = compute_log_likelihood(
        logits,
        tokens[:, 1:],
        temperature=scoring_params.temperature,
        top_k=scoring_params.top_k,
        top_p=scoring_params.top_p,
    )
    # convert token score arrays to lists, to be consistent with generate
    token_scores = token_scores[0].tolist()
    return token_scores

  def count_num_tokens(self, text: sampling_lib.SamplingInput) -> int:
    processed_input = self.input_processor.encode(
        sampling_lib.input_as_chunks(text)
    )
    return len(processed_input.tokens)


def top_k_mask(logits: Array, top_k: int) -> Array:
  inner_size = logits.shape[-1]
  indices = jnp.argsort(logits, axis=-1, descending=True)
  mask = jnp.arange(inner_size) < top_k
  mask = jnp.broadcast_to(mask, indices.shape)
  _, mask = jax.lax.sort_key_val(indices, mask, dimension=-1)
  return mask


def top_p_mask(logits: Array, top_p: float) -> Array:
  probs = jax.nn.softmax(logits, axis=-1)
  indices = jnp.argsort(logits, axis=-1, descending=True)
  sorted_probs = jnp.take_along_axis(probs, indices, axis=-1)
  cumsum = jnp.cumsum(sorted_probs, axis=-1)
  mask = cumsum - sorted_probs < top_p
  _, mask = jax.lax.sort_key_val(indices, mask, dimension=-1)
  return mask


def sample_from_logits(
    prng_key: PRNGKey,
    logits: Array,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> tuple[Array, Array]:
  """Samples from the last step of the logits.

  Args:
    prng_key: A PRNGKey used as the random key.
    logits: The logits from the model.
    temperature: The temperature for sampling.
    top_k: The maximum number of top tokens to sample from. If top_k == -1,
      results are sampled from the whole vocabulary.
    top_p: The cumulate probabilty of top tokens to sample from.

  Returns:
    The sampled tokens and the corresponding logprobs.
  """
  logits = jnp.astype(logits[:, [-1], :], jnp.float32)

  def greedy_fn(logits: Array) -> tuple[Array, Array]:
    tokens = jnp.argmax(logits, axis=-1)
    logprobs = jnp.zeros(logits.shape[:-1], dtype=logits.dtype)
    return tokens, logprobs

  def simple_sample_fn(logits: Array) -> tuple[Array, Array]:
    logits = logits / temperature
    m = distributions.Categorical(logits)
    tokens = m.sample(prng_key)
    logprobs = m.log_prob(tokens)
    return tokens, logprobs

  def masked_sample_fn(logits: Array) -> tuple[Array, Array]:
    logits = logits / temperature
    mask = jax.lax.cond(
        top_k > 0,
        lambda x: top_k_mask(x, top_k=top_k),
        lambda x: top_p_mask(x, top_p=top_p),
        logits,
    )
    m = distributions.MaskedCategorical(
        logits, mask=mask, neg_inf=neg_inf(logits.dtype)
    )
    tokens = m.sample(prng_key)
    logprobs = m.log_prob(tokens)
    return tokens, logprobs

  def sample_fn(logits: Array) -> tuple[Array, Array]:
    return jax.lax.cond(
        jnp.logical_or(top_k > 0, top_p < 1),
        masked_sample_fn,
        simple_sample_fn,
        logits,
    )

  return jax.lax.cond(temperature == 0, greedy_fn, sample_fn, logits)


def pad_along_axis(
    x: Array, pad_widths: tuple[int, int], axis: int, **kwargs: Any
) -> Array:
  """Pads the given array along the given axis."""
  all_pad_widths = [(0, 0)] * x.ndim
  all_pad_widths[axis] = pad_widths
  pad_fn = jnp.pad if isinstance(x, jax.Array) else np.pad
  return pad_fn(x, all_pad_widths, **kwargs)


def pad_to_along_axis(
    x: Array, pad_widths_to: int, axis: int, **kwargs: Any
) -> Array:
  """Pads the given array along the given axis to the given length."""
  # TODO: This leads to inhomogeneous seq len in the batch.
  if x.shape[axis] >= pad_widths_to:
    return x
  pad_widths = pad_widths_to - x.shape[axis]
  return pad_along_axis(x, (0, pad_widths), axis=axis, **kwargs)


def pad_decode_state_to(d: PyTree, length_to_pad: int) -> PyTree:
  """Pads the given decode state to the given length."""
  assert pytree.tree_is_mapping(d)
  d = cast(MutableMapping[str, Any], d)
  for k, v in d.items():
    if k.startswith('block_'):
      window_sizes = []
      for k2 in v.keys():
        if k2.startswith('window_size='):
          window_sizes.append(int(k2.split('=', 1)[1]))
      if len(window_sizes) > 1:
        raise ValueError(
            f'Expected no more than one window size for {k}: {v}, got'
            f' {window_sizes}'
        )
      window_size = window_sizes[0] if window_sizes else 0
      block_length_to_pad = length_to_pad
      if 0 < window_size and window_size + 1 < length_to_pad:
        # Note that we look back window_size tokens, so including current token,
        # the block length becomes window_size + 1.
        block_length_to_pad = window_size + 1
      for k2, v2 in v.items():
        if isinstance(v2, jax.typing.ArrayLike) and jnp.ndim(v2) >= 2:
          v[k2] = pad_to_along_axis(v2, block_length_to_pad, axis=1)
  return d


# TODO: This function causes OOM when jitted. The root cause is top_p
# masked sampling implementation needs decent number of copies of logits-shaped
# tensors. There are two solutions:
# 1. First locate a threshold and then mask based on threshold. It can cause
#    inconsistent behavior when the threshold number maches multiple logits.
# 2. Microbatching this function.
def compute_log_likelihood(
    logits: Array,
    tokens: Array,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> Array:
  """Computes the log likelyhood in float32."""
  logits = jnp.astype(logits, jnp.float32)

  def greedy_score_fn(logits: Array, tokens: Array) -> Array:
    shape = tokens.shape
    dtype = logits.dtype
    target_tokens = jnp.argmax(logits, axis=-1)
    return jnp.where(
        tokens == target_tokens,
        jnp.zeros(shape, dtype=dtype),
        jnp.full(shape, neg_inf(dtype), dtype=dtype),
    )

  def simple_sample_score_fn(logits: Array, tokens: Array) -> Array:
    # The shape of logits is (batch_size, seq_len, vocab_size).
    m = distributions.Categorical(logits / temperature)
    return m.log_prob(tokens)

  def masked_sample_score_fn(logits: Array, tokens: Array) -> Array:
    # The shape of logits is (batch_size, seq_len, vocab_size).
    logits = logits / temperature
    mask = jax.lax.cond(
        top_k > 0,
        lambda x: top_k_mask(x, top_k=top_k),
        lambda x: top_p_mask(x, top_p=top_p),
        logits,
    )
    m = distributions.MaskedCategorical(
        logits, mask=mask, neg_inf=neg_inf(logits.dtype)
    )
    return m.log_prob(tokens)

  def sample_score_fn(logits: Array, tokens: Array) -> Array:
    return jax.lax.cond(
        jnp.logical_or(top_k > 0, top_p < 1),
        masked_sample_score_fn,
        simple_sample_score_fn,
        logits,
        tokens,
    )

  return jax.lax.cond(
      temperature == 0, greedy_score_fn, sample_score_fn, logits, tokens
  )


def continue_decode(
    apply_fn: Callable[..., Array],
    params: PyTree,
    init_sampling_state: SamplingState,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    scoring_temperature: float = 1.0,
    scoring_top_k: int = -1,
    scoring_top_p: float = 1.0,
) -> SamplingState:

  def body_fn(sampling_state: SamplingState) -> SamplingState:
    # logits: [batch_size, 1, vocab_size]

    logits, extra_output = apply_fn(
        params,
        sampling_state.input_tokens,
        segment_positions=einops.repeat(
            sampling_state.position, '-> b 1', b=sampling_state.batch_size
        ),
        decode_state=sampling_state.decode_state,
    )

    prng_key, key = jax.random.split(sampling_state.prng_key, 2)
    # output_tokens: [batch_size, 1], output_logprobs: [batch_size, 1]
    output_tokens, output_logprobs = sample_from_logits(
        key, logits, temperature=temperature, top_k=top_k, top_p=top_p
    )

    # Three cases:
    # - Sequence is done generating, just output again the current
    #   token (should be eos), so that we will continue detecting
    #   the sequence is done (exception is all-padding sequence which
    #   is detected separately)
    # - Next position is output, use the sampled output token
    # - Next position is input, repeat back the existing input token
    output_tokens = jnp.select(
        [
            sampling_state.has_ended,
            sampling_state.next_position_is_output,
        ],
        [
            sampling_state.input_tokens,
            output_tokens,
        ],
        default=sampling_state.next_tokens,
    )

    def _score_fn(logits: Array, tokens: Array) -> Array:
      return compute_log_likelihood(
          logits,
          tokens,
          temperature=scoring_temperature,
          top_k=scoring_top_k,
          top_p=scoring_top_p,
      )

    scoring_follows_sampling = (
        (scoring_temperature == temperature)
        & (scoring_top_k == top_k)
        & (scoring_top_p == top_p)
    )
    # Only when all next positions are output tokens, scoring reuse is possible.
    output_scores = jax.lax.cond(
        scoring_follows_sampling
        & jnp.all(sampling_state.next_position_is_output),
        lambda *_: output_logprobs,
        _score_fn,
        logits,
        output_tokens,
    )

    # logprobs might be computed for input tokens and extra beyond eos tokens.
    # scores might be computed for extra beyond eos tokens.
    # We have to ignore those values during post-processing.
    return dataclasses.replace(
        sampling_state,
        prng_key=prng_key,
        position=sampling_state.position + 1,
        decode_state=extra_output['decode_state'],
        tokens=sampling_state.updated_tokens(output_tokens),
        token_logprobs=sampling_state.updated_token_logprobs(output_logprobs),
        token_scores=sampling_state.updated_token_scores(output_scores),
    )

  def cond_fn(sampling_state: SamplingState) -> jax.typing.ArrayLike:
    return (
        sampling_state.position < sampling_state.decode_state_length
    ) & ~jnp.all(sampling_state.has_ended)

  final_sampling_state = jax.lax.while_loop(
      cond_fn, body_fn, init_sampling_state
  )
  return final_sampling_state


################################################################################
# Utilities


def get_scaling_info(config, also_print=False, add_attn_flops=False):
  model_cls = module.ModuleRegistry.get(config.model_name)
  model = model_cls(config)
  info_dict = {}
  params = jax.eval_shape(model.init, jax.random.key(0))
  num_params = np.sum(jax.tree_util.tree_leaves(
      jax.tree_util.tree_map(
          lambda x: np.prod(
              np.array(x.shape, dtype=np.float64)), params)), dtype=np.float64)
  num_examples = (
      np.float64(config.batch_size) * config.num_train_steps)
  num_tokens = num_examples * config.seq_len
  num_embedding_params = config.vocab_size * config.model_dim
  num_non_embedding_params = num_params - num_embedding_params
  num_flops = num_params * num_tokens * 6
  num_attn_flops = -1
  if add_attn_flops:
    w = config.window_size
    s = config.seq_len
    # Calculate the number of attention positions that are not masked.
    if w > 0 and w < s:
      attn_count = (w * (w + 1) / 2 + (s - w) * w)
    else:
      attn_count = s * (s + 1) / 2
    num_attn_flops = (
        # 2 for q @ k and attn_score @ v, and 6 for forward and backward pass.
        12 * config.n_layers * config.n_heads * attn_count *
        config.per_head_dim) * num_examples
    num_flops += num_attn_flops
    info_dict['num_attn_flops'] = num_attn_flops

  info_dict['num_examples'] = num_examples
  info_dict['num_params'] = num_params
  info_dict['num_non_embedding_params'] = num_non_embedding_params
  info_dict['num_embedding_params'] = num_embedding_params
  info_dict['embedding_params_ratio'] = num_embedding_params / num_params
  info_dict['num_tokens'] = num_tokens
  info_dict['num_flops'] = num_flops
  if also_print:
    if add_attn_flops:
      print(f'num_attn_flops: {num_attn_flops}')
      print(f'num_attn_flops / num_flops: {num_attn_flops / num_flops}')
    print(f'num_params: {num_params/1e6} M')
    print(f'num_non_embedding_params: {num_non_embedding_params/1e6} M')
    print(f'num_embedding_params: {num_embedding_params/1e6} M')
    print(f'embedding_params_ratio: {num_embedding_params/num_params}')
    print(f'num_tokens: {num_tokens/1e6} M')
    print(f'num_tokens / num_params: {num_tokens / num_params}')
    print(f'num_tokens / num_non_embedding_params: '
          f'{num_tokens / num_non_embedding_params}')
    print(f'num_flops: {num_flops}')
  return info_dict


def quantize_tfm_params(params, symmetric=False):
  params = get_raw_arrays(params)
  if isinstance(params, jnp.ndarray):
    return params
  quant_params = {}
  for key in params:
    if key.startswith('block'):
      quant_params[key] = quantize_tfm_params(
          params[key], symmetric=symmetric
      )
    elif key == 'attn':
      subparams = copy.copy(params[key])
      for subkey in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        subparams[subkey]['w'] = common.quantize_array(
            subparams[subkey]['w'],
            symmetric=symmetric,
        )
      quant_params[key] = subparams
    elif key == 'ffn':
      subparams = copy.copy(params[key])
      for subkey in ['ffn_0', 'ffn_0_gate', 'ffn_1']:
        subparams[subkey]['w'] = common.quantize_array(
            subparams[subkey]['w'],
            symmetric=symmetric,
        )
      quant_params[key] = subparams
    elif (
        key.startswith('embed_linear')
        or key.startswith('final_ln')
        or key.startswith('pre_ln')
        or key.startswith('post_ln')
    ):
      # Leave the embedding linear and layer norm layer unquantized.
      quant_params[key] = params[key]
    else:
      raise ValueError(f'Unknown key: {key}!')
  return quant_params

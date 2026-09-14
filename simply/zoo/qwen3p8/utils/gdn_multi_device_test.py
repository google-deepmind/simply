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
"""Tests for the Qwen3.8 GatedDeltaNet.

The numpy reference below is an independent float64 transcription of the
HuggingFace release code -- transformers v5
`models/qwen3_5/modeling_qwen3_5.py`, which is what `config.json`'s
`model_type: qwen3_5` loads -- one function per HF function, each naming its
upstream in its docstring. It is deliberately a re-derivation from the torch
source rather than a call into `gdn.py`: nothing here imports the module code
except the tests themselves. Its own cross-check is that
`torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`, two
independent formulations of the same recurrence, agree to 1.4e-16 in f64
(`test_the_two_reference_cores_agree`).

Three A/B knobs of `_reference_layer` are the machine-checked record of the
three semantics that are easy to get wrong and silently expensive:
`beta_sigmoid_twice`, `gate_activation` and `norm_plus_one`.
"""

import dataclasses
import functools
import re
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import sharding as sharding_lib
from simply.zoo.qwen3p8.utils import gdn


def setUpModule():
  # Four CPU devices, so that `sharding_lib.with_sharding_constraint` does
  # something: on one device it only checks the RANK
  # (`simply/utils/sharding.py:253`), which is why a wrong axis used to be
  # invisible. This has to run before the backend initializes.
  jax.config.update('jax_num_cpu_devices', 4)

# Small but non-degenerate geometry; the released 27B is D=5120, HK=16, HV=48,
# K=V=128, W=4, C=32. HV / HK = 3 as in the release.
B, T, D, HK, HV, K, V, W, C = 2, 12, 16, 2, 6, 8, 8, 4, 4
KEY_DIM, VALUE_DIM = HK * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
RMS_EPS = 1e-6  # `rms_norm_eps` of the released config.

# Largest |diff| between the f32 module (or its f32 cores) and the f64
# reference over every case here: 4.6e-6 measured, 2e-5 allowed. It is a
# float32 rounding budget, not a fitted number -- the cores agree with the
# reference to 2.6e-7 on their own.
ATOL = 2e-5
# A bfloat16 bound is a quantization bound, not a parity bound: rounding this
# layer's weights and inputs to bfloat16 and running the f64 reference on them
# already moves the output by 0.069 * max|y_ref| (`test_dtypes` holds that run
# to the same bound), and the module measures 0.068 (bf16 activations, f32
# kernel) and 0.051 (both bfloat16). 0.12 leaves headroom for the rounding of
# the intermediates.
BF16_REL = 0.12
# float32 activations through a bfloat16 kernel: nothing is quantized except
# the delta-rule inputs and the two Gram matmuls. Measured 5.8e-03 of
# max|y_ref|; 0.02 leaves 3.4x, and it is 6x tighter than BF16_REL.
BF16_KERNEL_REL = 0.02
# An A/B knob must move the layer output by much more than ATOL or the test
# proves nothing. Measured on the f64 reference: double-sigmoided `beta` moves
# it by 0.81, a sigmoid output gate by 1.79 and a `1 + w` output norm by 2.90,
# against max|y| ~ 2.1-2.9. 1e-3 is four orders below the smallest of them and
# still two above ATOL.
AB_MIN_DIFF = 1e-3
# A `stablehlo.dot_general` whose two OPERANDS are float32 (its result may be
# float32 for a bfloat16 matmul too, which is why the operand tuple is matched).
_F32_OPERANDS = re.compile(r': \(tensor<[0-9x]*f32>, tensor<[0-9x]*f32>\)')


# --------------------------------------------------------------------------
# numpy reference (f64), transcribed from modeling_qwen3_5.py
# --------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))


def _silu(x: np.ndarray) -> np.ndarray:
  """`ACT2FN['silu']`, the `hidden_act` of the release."""
  return np.asarray(x, np.float64) * _sigmoid(x)


def _softplus(x: np.ndarray) -> np.ndarray:
  """`F.softplus`, evaluated in the overflow-safe form."""
  x = np.asarray(x, np.float64)
  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  """`l2norm`: the eps is added to the SUM OF SQUARES, inside the sqrt."""
  x = np.asarray(x, np.float64)
  return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def _apply_mask_to_padding_states(
    hidden_states: np.ndarray, attention_mask: np.ndarray | None
) -> np.ndarray:
  """`apply_mask_to_padding_states`: zeroes the padded rows of `[B, T, D]`."""
  hidden_states = np.asarray(hidden_states, np.float64)
  if attention_mask is None:
    return hidden_states
  return hidden_states * np.asarray(attention_mask, np.float64)[:, :, None]


def _causal_conv1d_fn(
    hidden_states: np.ndarray, weight: np.ndarray
) -> np.ndarray:
  """`causal_conv1d_fn` + silu: `[B, C, T]` in, `[B, C, T]` out.

  `F.conv1d(..., padding=W - 1, groups=C)[:, :, :T]`, i.e. the taps of the
  first W - 1 positions read zeros.

  Args:
    hidden_states: `[B, C, T]` channel-major pre-conv inputs.
    weight: `[C, W]` depthwise filters (HF's `conv1d.weight.squeeze(1)`).

  Returns:
    `[B, C, T]` silu of the convolution.
  """
  hidden_states = np.asarray(hidden_states, np.float64)
  weight = np.asarray(weight, np.float64)
  b, c, t = hidden_states.shape
  kw = weight.shape[-1]
  padded = np.concatenate([np.zeros((b, c, kw - 1)), hidden_states], axis=-1)
  out = np.zeros((b, c, t))
  for i in range(kw):
    out += weight[None, :, i, None] * padded[:, :, i : i + t]
  return _silu(out)


def _causal_conv1d_update(
    hidden_states: np.ndarray, conv_state: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """`causal_conv1d_update` + silu: one token against a cached window.

  Args:
    hidden_states: `[B, C, 1]` the new token.
    conv_state: `[B, C, S]` cached window, oldest first. HF keeps S = W slots
      and never reads the oldest; S = W - 1 gives the same answer.
    weight: `[C, W]` depthwise filters.

  Returns:
    `([B, C, 1], [B, C, S])`: the silu of the convolution and the new window.
  """
  hidden_states = np.asarray(hidden_states, np.float64)
  conv_state = np.asarray(conv_state, np.float64)
  weight = np.asarray(weight, np.float64)
  state_len = conv_state.shape[-1]
  cat = np.concatenate([conv_state, hidden_states], axis=-1)
  new_state = cat[:, :, -state_len:]
  kw = weight.shape[-1]
  out = np.zeros(hidden_states.shape)
  for i in range(kw):
    out[:, :, 0] += weight[None, :, i] * cat[:, :, cat.shape[-1] - kw + i]
  return _silu(out), new_state


def _rms_norm_gated(
    hidden_states: np.ndarray,
    weight: np.ndarray,
    gate: np.ndarray,
    eps: float = RMS_EPS,
    *,
    activation: str = 'silu',
    plus_one: bool = False,
) -> np.ndarray:
  """`Qwen3_5RMSNormGated`: normalize, apply the gain, THEN gate.

  Args:
    hidden_states: `[..., V]` per-head delta-rule output.
    weight: `[V]` gain, restored verbatim (HF initializes it to ones).
    gate: `[..., V]` the `in_proj_z` projection.
    eps: variance epsilon.
    activation: `'silu'` in the release (`output_gate_type: "swish"`);
      `'sigmoid'` is the A/B that must not match.
    plus_one: the `1 + w` convention of `Qwen3_5RMSNorm` (the BLOCK norms), the
      A/B that must not match here.

  Returns:
    The gated, normalized value.
  """
  hidden_states = np.asarray(hidden_states, np.float64)
  variance = np.mean(hidden_states * hidden_states, axis=-1, keepdims=True)
  out = hidden_states / np.sqrt(variance + eps)
  gain = np.asarray(weight, np.float64)
  out = (1.0 + gain) * out if plus_one else gain * out
  act = _silu(gate) if activation == 'silu' else _sigmoid(gate)
  return out * act


def _torch_chunk_gated_delta_rule(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    chunk_size: int = 64,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """`torch_chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=True`.

  Args:
    query: `[B, T, H, K]`, normalized and scaled inside as HF does.
    key: `[B, T, H, K]`.
    value: `[B, T, H, V]`.
    g: `[B, T, H]` log decay.
    beta: `[B, T, H]` delta-rule step size.
    chunk_size: tokens per chunk; T is zero-padded up to a multiple of it.
    initial_state: `[B, H, K, V]` carried state, or None for zeros.

  Returns:
    `([B, T, H, V], [B, H, K, V])`: the outputs and the final state.
  """
  query, key, value = (np.asarray(a, np.float64) for a in (query, key, value))
  g = np.asarray(g, np.float64)
  beta = np.asarray(beta, np.float64)
  query, key = _l2norm(query), _l2norm(key)
  query, key, value = (a.transpose(0, 2, 1, 3) for a in (query, key, value))
  beta = beta.transpose(0, 2, 1)
  g = g.transpose(0, 2, 1)
  b, h, t, k_dim = key.shape
  v_dim = value.shape[-1]
  pad = (chunk_size - t % chunk_size) % chunk_size
  pad_time = lambda a: np.pad(  # pylint: disable=g-long-lambda
      a, ((0, 0), (0, 0), (0, pad)) + ((0, 0),) * (a.ndim - 3)
  )
  query, key, value, beta, g = (
      pad_time(a) for a in (query, key, value, beta, g)
  )
  total = t + pad
  query = query * (k_dim**-0.5)
  v_beta = value * beta[..., None]
  k_beta = key * beta[..., None]
  to_chunks = lambda a: a.reshape(b, h, -1, chunk_size, a.shape[-1])
  query, key, value, k_beta, v_beta = (
      to_chunks(a) for a in (query, key, value, k_beta, v_beta)
  )
  g = g.reshape(b, h, -1, chunk_size)
  upper = np.triu(np.ones((chunk_size, chunk_size), bool), 0)

  g = np.cumsum(g, axis=-1)
  diff = g[..., :, None] - g[..., None, :]
  decay_mask = np.tril(np.exp(np.tril(diff)))
  # The UT transform, as HF runs it: forward substitution on the strictly
  # lower triangle, which inverts `I + tril(k_beta key^T * decay, -1)`.
  attn = np.where(upper, 0.0, -(np.matmul(k_beta, key.swapaxes(-1, -2))
                                * decay_mask))
  for i in range(1, chunk_size):
    row = attn[..., i, :i].copy()
    sub = attn[..., :i, :i].copy()
    attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
  attn = attn + np.eye(chunk_size)

  value = np.matmul(attn, v_beta)
  k_cumdecay = np.matmul(attn, k_beta * np.exp(g)[..., None])
  state = (
      np.zeros((b, h, k_dim, v_dim))
      if initial_state is None
      else np.asarray(initial_state, np.float64).copy()
  )
  out = np.zeros_like(value)
  for i in range(total // chunk_size):
    q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
    attn_intra = np.matmul(q_i, k_i.swapaxes(-1, -2)) * decay_mask[:, :, i]
    v_new = v_i - np.matmul(k_cumdecay[:, :, i], state)
    attn_inter = np.matmul(q_i * np.exp(g[:, :, i])[..., None], state)
    out[:, :, i] = attn_inter + np.matmul(attn_intra, v_new)
    decay_to_end = np.exp(g[:, :, i, -1:] - g[:, :, i])[..., None]
    state = state * np.exp(g[:, :, i, -1])[..., None, None] + np.matmul(
        (k_i * decay_to_end).swapaxes(-1, -2), v_new
    )
  out = out.reshape(b, h, -1, v_dim)[:, :, :t]
  return out.transpose(0, 2, 1, 3), state


def _torch_recurrent_gated_delta_rule(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """`torch_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel=True`.

  Args:
    query: `[B, T, H, K]`, normalized and scaled inside as HF does.
    key: `[B, T, H, K]`.
    value: `[B, T, H, V]`.
    g: `[B, T, H]` log decay.
    beta: `[B, T, H]` delta-rule step size.
    initial_state: `[B, H, K, V]` carried state, or None for zeros.

  Returns:
    `([B, T, H, V], [B, H, K, V])`: the outputs and the final state.
  """
  query, key, value = (np.asarray(a, np.float64) for a in (query, key, value))
  g = np.asarray(g, np.float64)
  beta = np.asarray(beta, np.float64)
  query, key = _l2norm(query), _l2norm(key)
  query, key, value = (a.transpose(0, 2, 1, 3) for a in (query, key, value))
  beta = beta.transpose(0, 2, 1)
  g = g.transpose(0, 2, 1)
  b, h, t, k_dim = key.shape
  v_dim = value.shape[-1]
  query = query * (k_dim**-0.5)
  out = np.zeros((b, h, t, v_dim))
  state = (
      np.zeros((b, h, k_dim, v_dim))
      if initial_state is None
      else np.asarray(initial_state, np.float64).copy()
  )
  for i in range(t):
    q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
    state = state * np.exp(g[:, :, i])[..., None, None]
    kv_mem = (state * k_t[..., None]).sum(axis=-2)
    delta = (v_t - kv_mem) * beta[:, :, i][..., None]
    state = state + k_t[..., None] * delta[..., None, :]
    out[:, :, i] = (state * q_t[..., None]).sum(axis=-2)
  return out.transpose(0, 2, 1, 3), state


def _reference_layer(
    x: np.ndarray,
    w: dict[str, np.ndarray],
    *,
    attention_mask: np.ndarray | None = None,
    chunk_size: int = C,
    conv_state: np.ndarray | None = None,
    recurrent_state: np.ndarray | None = None,
    beta_sigmoid_twice: bool = False,
    gate_activation: str = 'silu',
    norm_plus_one: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """`Qwen3_5GatedDeltaNet.forward` (modeling_qwen3_5.py:433-547).

  Args:
    x: `[B, T, D]` block input.
    w: weights in the Simply param layout (see `_random_weights`).
    attention_mask: `[B, T]` 1/0 mask fed to `apply_mask_to_padding_states`.
    chunk_size: chunk length of the prefill core.
    conv_state: `[B, conv_dim, W - 1]` carried conv window, or None.
    recurrent_state: `[B, HV, K, V]` carried state, or None.
    beta_sigmoid_twice: A/B for PLAN.md defect 5 -- the ragged kernel applied
      `sigmoid` to an already-sigmoided `beta`. Must NOT match.
    gate_activation: `'silu'` is the release (`output_gate_type: "swish"`);
      `'sigmoid'` is the A/B that must NOT match.
    norm_plus_one: the block norms' `1 + w`; must NOT match here.

  Returns:
    `(y, new_conv_state, new_recurrent_state)`.
  """
  x = _apply_mask_to_padding_states(x, attention_mask)
  b, t, _ = x.shape
  decode = conv_state is not None and t == 1

  mixed_qkv = np.matmul(x, np.asarray(w['in_proj_qkv'], np.float64))
  mixed_qkv = mixed_qkv.transpose(0, 2, 1)  # [B, conv_dim, T]
  if decode:
    assert conv_state is not None
    mixed_qkv, new_conv_state = _causal_conv1d_update(
        mixed_qkv, conv_state, w['conv1d']
    )
  else:
    history = (
        np.zeros((b, CONV_DIM, W - 1))
        if conv_state is None
        else np.asarray(conv_state, np.float64)
    )
    extended = np.concatenate([history, mixed_qkv], axis=-1)
    new_conv_state = extended[:, :, -(W - 1) :]
    mixed_qkv = _causal_conv1d_fn(extended, w['conv1d'])[:, :, -t:]
  mixed_qkv = mixed_qkv.transpose(0, 2, 1)  # [B, T, conv_dim]

  query = mixed_qkv[..., :KEY_DIM].reshape(b, t, HK, K)
  key = mixed_qkv[..., KEY_DIM : 2 * KEY_DIM].reshape(b, t, HK, K)
  value = mixed_qkv[..., 2 * KEY_DIM :].reshape(b, t, HV, V)
  query = np.repeat(query, HV // HK, axis=2)
  key = np.repeat(key, HV // HK, axis=2)
  z = np.matmul(x, np.asarray(w['in_proj_z'], np.float64)).reshape(b, t, HV, V)

  beta = _sigmoid(np.matmul(x, np.asarray(w['in_proj_b'], np.float64)))
  if beta_sigmoid_twice:
    beta = _sigmoid(beta)
  a = np.matmul(x, np.asarray(w['in_proj_a'], np.float64))
  g = -np.exp(np.asarray(w['A_log'], np.float64)) * _softplus(
      a + np.asarray(w['dt_bias'], np.float64)
  )

  if attention_mask is not None:
    # Not HF: HF leaves `g` and `beta` alive at padded positions, so its state
    # depends on the padding. `gdn.py` forces an identity update instead, which
    # is what makes a padded prefill equal the unpadded one.
    valid = np.asarray(attention_mask, np.float64)
    query = query * valid[:, :, None, None]
    key = key * valid[:, :, None, None]
    value = value * valid[:, :, None, None]
    g = g * valid[:, :, None]
    beta = beta * valid[:, :, None]

  if decode and recurrent_state is not None:
    core_out, new_state = _torch_recurrent_gated_delta_rule(
        query, key, value, g, beta, recurrent_state
    )
  else:
    core_out, new_state = _torch_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size, recurrent_state
    )
  out = _rms_norm_gated(
      core_out.reshape(-1, V),
      w['norm'],
      z.reshape(-1, V),
      activation=gate_activation,
      plus_one=norm_plus_one,
  )
  y = np.matmul(
      out.reshape(b, t, VALUE_DIM), np.asarray(w['out_proj'], np.float64)
  )
  if attention_mask is not None:
    y = y * np.asarray(attention_mask, np.float64)[:, :, None]
  return y, new_conv_state, new_state


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _random_weights(seed: int = 0) -> dict[str, np.ndarray]:
  """Weights in the layout of the `Qwen38GatedDeltaNet` param tree."""
  rng = np.random.default_rng(seed)
  normal = lambda *shape: rng.normal(size=shape) / np.sqrt(shape[0])
  return {
      'in_proj_qkv': normal(D, CONV_DIM),
      'in_proj_z': normal(D, VALUE_DIM),
      'in_proj_b': normal(D, HV),
      'in_proj_a': normal(D, HV),
      'conv1d': rng.normal(size=(CONV_DIM, W)),
      'A_log': np.log(rng.uniform(1.0, 16.0, size=(HV,))),
      'dt_bias': rng.normal(size=(HV,)),
      'norm': rng.normal(1.0, 0.1, size=(V,)),
      'out_proj': normal(VALUE_DIM, D),
  }


def _params(w: dict[str, np.ndarray]) -> dict[str, Any]:
  f32 = lambda name: jnp.asarray(w[name], jnp.float32)
  return {
      'in_proj_qkv': {'w': f32('in_proj_qkv')},
      'in_proj_z': {'w': f32('in_proj_z')},
      'in_proj_b': {'w': f32('in_proj_b')},
      'in_proj_a': {'w': f32('in_proj_a')},
      'conv1d': {'w': f32('conv1d')},
      'A_log': f32('A_log'),
      'dt_bias': f32('dt_bias'),
      'norm': {'scale': f32('norm')},
      'out_proj': {'w': f32('out_proj')},
  }


class _StubSharding:
  """The `config_lib.BaseSharding` fields the GDN reads, without the dep."""

  activation_partition = (('replica', 'data'), None, 'model')
  ffn0_partition = ('data', 'model')
  ffn1_partition = ('model', 'data')


# replica, data, model -- `sharding_lib`'s own default axis names, and a shape
# in which B and every annotated dimension divide. `sharding_lib.set_mesh` is
# the one way in: it registers the ABSTRACT mesh that
# `with_sharding_constraint` resolves named axes against (a bare `with mesh:`
# does not, and simply then falls back to an all-replica mesh that cannot
# divide the batch), and its mesh has `Auto` axis types, unlike
# `jax.make_mesh`'s default -- under `Explicit` axes `jnp.repeat` refuses to
# trace without an `out_sharding`.
_MESH_SHAPE = (1, 2, 2)


@functools.lru_cache(maxsize=None)
def _layer(
    chunk_size: int = C,
    activation_dtype: str = 'float32',
    gdn_compute_dtype: str = 'float32',
    sharded: bool = False,
    conv_accumulation_dtype: str = '',
) -> gdn.Qwen38GatedDeltaNet:
  """A tiny GDN layer; cached so `_apply_fn` compiles each variant once.

  Args:
    chunk_size: tokens per chunk of the prefill core.
    activation_dtype: dtype of the projections and of the output.
    gdn_compute_dtype: dtype the delta-rule inputs are rounded to.
    sharded: build with `_StubSharding` instead of no annotations.
    conv_accumulation_dtype: empty means "leave the module's own default",
      which is what `test_conv_accumulation_dtype_is_a_knob` compares against.

  Returns:
    The layer.
  """
  layer = gdn.Qwen38GatedDeltaNet(
      model_dim=D,
      num_key_heads=HK,
      num_value_heads=HV,
      key_head_dim=K,
      value_head_dim=V,
      conv_kernel_dim=W,
      chunk_size=chunk_size,
      rms_norm_epsilon=RMS_EPS,
      activation_dtype=activation_dtype,
      gdn_compute_dtype=gdn_compute_dtype,
      sharding_config=_StubSharding() if sharded else None,
  )
  if conv_accumulation_dtype:
    layer = dataclasses.replace(
        layer, conv_accumulation_dtype=conv_accumulation_dtype
    )
  return layer


@functools.lru_cache(maxsize=None)
def _apply_fn(
    chunk_size: int = C,
    activation_dtype: str = 'float32',
    gdn_compute_dtype: str = 'float32',
    sharded: bool = False,
    conv_accumulation_dtype: str = '',
):
  """`layer.apply` under jit; eager dispatch dominates the test's runtime."""
  layer = _layer(
      chunk_size, activation_dtype, gdn_compute_dtype, sharded,
      conv_accumulation_dtype,
  )

  @jax.jit
  def fn(params, x, segment_ids, segment_positions, decode_state):
    return layer.apply(
        params,
        x,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        decode_state=decode_state,
    )

  return fn


# `compute_dtype` is static: a dtype is not an array, and a jitted kernel that
# takes it as a traced argument is PLAN.md defect 4.
_chunk_rule = jax.jit(
    gdn.chunk_gated_delta_rule,
    static_argnames=('chunk_size', 'compute_dtype'),
)
_recurrent_rule = jax.jit(
    gdn.recurrent_gated_delta_rule, static_argnames=('compute_dtype',)
)
_recurrent_step = jax.jit(
    gdn.recurrent_gated_delta_rule_step, static_argnames=('compute_dtype',)
)
_conv = jax.jit(gdn.causal_depthwise_conv)


def _state_of(
    extra: dict[str, gdn.GatedDeltaNetDecodeState | None],
) -> gdn.GatedDeltaNetDecodeState:
  """The updated decode state of an `apply` call that carried one."""
  state = extra['decode_state']
  assert state is not None
  return state


def _to_bfloat16(a: np.ndarray) -> np.ndarray:
  """`a` rounded to bfloat16 and back, to price operand quantization in f64."""
  return np.asarray(jnp.asarray(a, jnp.bfloat16), np.float64)


def _spec(x: jax.Array) -> jax.sharding.PartitionSpec:
  """The `PartitionSpec` of a concrete array; `.sharding` is typed too wide."""
  sharding = x.sharding
  assert isinstance(sharding, jax.sharding.NamedSharding), sharding
  return sharding.spec


def _max_diff(a, b) -> float:
  return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b))))


def _ones(t: int = T) -> jax.Array:
  return jnp.ones((B, t), jnp.int32)


def _positions(t: int = T, batch: int = B) -> jax.Array:
  return jnp.tile(jnp.arange(t), (batch, 1))


class GdnCoreTest(parameterized.TestCase):
  """The free-function cores, against the f64 reference."""

  def _core_inputs(
      self, t: int = T, seed: int = 0
  ) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    query = rng.normal(size=(B, t, HV, K))
    key = rng.normal(size=(B, t, HV, K))
    value = rng.normal(size=(B, t, HV, V))
    a_log = np.log(rng.uniform(1.0, 16.0, size=(HV,)))
    a = rng.normal(size=(B, t, HV))
    g = -np.exp(a_log) * _softplus(a + rng.normal(size=(HV,)))
    beta = _sigmoid(rng.normal(size=(B, t, HV)))
    s0 = rng.normal(size=(B, HV, K, V)) * 0.1
    return query, key, value, g, beta, s0

  def test_the_two_reference_cores_agree(self):
    """The f64 reference's own cross-check: two formulations, one recurrence."""
    query, key, value, g, beta, s0 = self._core_inputs()
    o_chunk, s_chunk = _torch_chunk_gated_delta_rule(
        query, key, value, g, beta, C, s0
    )
    o_rec, s_rec = _torch_recurrent_gated_delta_rule(
        query, key, value, g, beta, s0
    )
    self.assertLess(_max_diff(o_chunk, o_rec), 1e-14)
    self.assertLess(_max_diff(s_chunk, s_rec), 1e-14)

  @parameterized.parameters(4, 5, 8, 16)
  def test_chunked_matches_recurrent_and_reference(self, chunk_size: int):
    """Three ways: chunked core, recurrent core, f64 reference.

    `chunk_size=5` and `16` do not divide T = 12, so they also exercise the
    zero-padded remainder chunk.

    Args:
      chunk_size: tokens per chunk.
    """
    query, key, value, g, beta, s0 = self._core_inputs()
    o_ref, s_ref = _torch_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size, s0
    )
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    args = (f32(query), f32(key), f32(value), f32(g), f32(beta), f32(s0))
    o, s = _chunk_rule(*args, chunk_size=chunk_size)
    o_seq, s_seq = _recurrent_rule(*args)
    with self.subTest('chunked_vs_reference'):
      self.assertLess(_max_diff(o, o_ref), ATOL)
      self.assertLess(_max_diff(s, s_ref), ATOL)
    with self.subTest('chunked_vs_recurrent'):
      self.assertLess(_max_diff(o, o_seq), ATOL)
      self.assertLess(_max_diff(s, s_seq), ATOL)
    with self.subTest('recurrent_vs_reference'):
      self.assertLess(_max_diff(o_seq, o_ref), ATOL)
      self.assertLess(_max_diff(s_seq, s_ref), ATOL)

  def test_chunked_without_initial_state(self):
    query, key, value, g, beta, _ = self._core_inputs(t=17, seed=1)
    o_ref, s_ref = _torch_chunk_gated_delta_rule(query, key, value, g, beta, 8)
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    o, s = _chunk_rule(
        f32(query), f32(key), f32(value), f32(g), f32(beta), chunk_size=8
    )
    self.assertLess(_max_diff(o, o_ref), ATOL)
    self.assertLess(_max_diff(s, s_ref), ATOL)

  def test_step_matches_recurrent(self):
    query, key, value, g, beta, s0 = self._core_inputs(t=3, seed=2)
    o_ref, s_ref = _torch_recurrent_gated_delta_rule(
        query, key, value, g, beta, s0
    )
    s = jnp.asarray(s0, jnp.float32)
    at = lambda a, i: jnp.asarray(a[:, i], jnp.float32)
    outs = []
    for i in range(3):
      o_i, s = _recurrent_step(
          at(query, i), at(key, i), at(value, i), at(g, i), at(beta, i), s
      )
      outs.append(o_i)
    self.assertLess(_max_diff(jnp.stack(outs, axis=1), o_ref), ATOL)
    self.assertLess(_max_diff(s, s_ref), ATOL)

  def test_padding_is_an_identity_update(self):
    """`g = 0, beta = 0` leaves the state exactly where it was."""
    query, key, value, g, beta, s0 = self._core_inputs(t=9, seed=3)
    g_pad, beta_pad = g.copy(), beta.copy()
    g_pad[:, 5:] = 0.0
    beta_pad[:, 5:] = 0.0
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    _, s = _chunk_rule(
        f32(query),
        f32(key),
        f32(value),
        f32(g_pad),
        f32(beta_pad),
        f32(s0),
        chunk_size=4,
    )
    cut = lambda a: f32(a[:, :5])
    _, s_short = _chunk_rule(
        cut(query), cut(key), cut(value), cut(g), cut(beta), f32(s0),
        chunk_size=4,
    )
    self.assertLess(_max_diff(s, s_short), ATOL)

  @parameterized.parameters(
      ((6,),),  # mid-chunk
      ((4,),),  # on a chunk boundary
      ((0,),),  # the whole call is a fresh sequence
      ((3, 4, 9),),  # three segments, one boundary-aligned
  )
  def test_segment_starts_reset_state(self, starts: tuple[int, ...]):
    """Packed segments == the segments run one by one from a zero state."""
    t = 12
    query, key, value, g, beta, s0 = self._core_inputs(t=t, seed=4)
    start = np.zeros((B, t), bool)
    start[:, list(starts)] = True
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    o, s = _chunk_rule(
        f32(query),
        f32(key),
        f32(value),
        f32(g),
        f32(beta),
        f32(s0),
        chunk_size=4,
        segment_start=jnp.asarray(start),
    )
    edges = sorted({0, t, *starts})
    carried = None if 0 in starts else s0
    s_ref = s0  # the loop always runs; this keeps the type checker happy
    for lo, hi in zip(edges[:-1], edges[1:]):
      cut = lambda a: a[:, lo:hi]  # pylint: disable=cell-var-from-loop
      o_ref, s_ref = _torch_recurrent_gated_delta_rule(
          cut(query), cut(key), cut(value), cut(g), cut(beta), carried
      )
      self.assertLess(_max_diff(o[:, lo:hi], o_ref), ATOL)
      carried = None  # every later edge is a reset
    self.assertLess(_max_diff(s, s_ref), ATOL)

  def test_conv_matches_reference_prefill_and_update(self):
    """Prefill, then a cached step; the seam must equal a T + 1 prefill."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(B, T, CONV_DIM))
    w = rng.normal(size=(CONV_DIM, W))
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    y, state = _conv(f32(x), f32(w))
    y_ref = _causal_conv1d_fn(x.transpose(0, 2, 1), w).transpose(0, 2, 1)
    with self.subTest('prefill'):
      self.assertLess(_max_diff(y, y_ref), ATOL)
      self.assertLess(_max_diff(state, x[:, -(W - 1) :].transpose(0, 2, 1)),
                      ATOL)
    step = rng.normal(size=(B, 1, CONV_DIM))
    y_step, state_step = _conv(f32(step), f32(w), state)
    y_step_ref, state_ref = _causal_conv1d_update(
        step.transpose(0, 2, 1), np.asarray(state, np.float64), w
    )
    with self.subTest('cached_step'):
      self.assertLess(_max_diff(y_step, y_step_ref.transpose(0, 2, 1)), ATOL)
      self.assertLess(_max_diff(state_step, state_ref), ATOL)
    y_all = _causal_conv1d_fn(
        np.concatenate([x, step], axis=1).transpose(0, 2, 1), w
    ).transpose(0, 2, 1)
    with self.subTest('seam'):
      self.assertLess(_max_diff(y_step[:, 0], y_all[:, -1]), ATOL)

  def test_conv_window_skips_padding(self):
    """The window carried out of a padded row holds only valid inputs."""
    rng = np.random.default_rng(6)
    n_valid = T - 3
    x = rng.normal(size=(B, T, CONV_DIM))
    x[1, n_valid:] = 9.0  # garbage in the padded suffix of row 1
    w = rng.normal(size=(CONV_DIM, W))
    valid = np.ones((B, T), bool)
    valid[1, n_valid:] = False
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    _, state = _conv(
        f32(x * valid[..., None]), f32(w), valid=jnp.asarray(valid)
    )
    _, state_short = _conv(f32(x[1:2, :n_valid]), f32(w))
    self.assertLess(_max_diff(state[1], state_short[0]), ATOL)

  def test_log_decay_matches_the_reference_and_stays_in_float32(self):
    """`g` is float32 even for bfloat16 activations (HF's `.float()` casts).

    `g` is cumulated over a whole chunk before it is exponentiated, so a
    bfloat16 `softplus` here compounds; the bit-exact comparison is the only
    thing that can see it, because at bfloat16 activations the layer's own
    quantization noise is 20x larger than the effect.
    """
    rng = np.random.default_rng(83)
    a = rng.normal(size=(B, T, HV))
    a_log = np.log(rng.uniform(1.0, 16.0, size=(HV,)))
    dt_bias = rng.normal(size=(HV,))
    f32 = lambda z: jnp.asarray(z, jnp.float32)
    g = gdn.log_decay(f32(a), f32(a_log), f32(dt_bias))
    g_ref = -np.exp(a_log) * _softplus(a + dt_bias)
    self.assertLess(_max_diff(g, g_ref), ATOL)
    self.assertTrue(bool(jnp.all(g <= 0.0)))  # `exp(g)` is a decay, never gain

    bf16 = lambda z: jnp.asarray(z, jnp.bfloat16)
    g_from_bf16 = gdn.log_decay(bf16(a), f32(a_log), f32(dt_bias))
    self.assertEqual(g_from_bf16.dtype, jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(g_from_bf16),
        np.asarray(
            -jnp.exp(f32(a_log))
            * jax.nn.softplus(jnp.asarray(bf16(a), jnp.float32) + f32(dt_bias))
        ),
    )
    # ... and doing it in bfloat16 really is a different answer.
    in_bf16 = -jnp.exp(bf16(a_log)) * jax.nn.softplus(bf16(a) + bf16(dt_bias))
    self.assertGreater(_max_diff(g_from_bf16, in_bf16), 0.0)

  def test_the_chunked_core_never_rounds_the_decay(self):
    """`compute_dtype` rounds q/k/v/beta; `g` stays float32 in both cores."""
    query, key, value, g, beta, s0 = self._core_inputs(seed=17)
    f32 = lambda z: jnp.asarray(z, jnp.float32)
    rounded = lambda z: jnp.asarray(jnp.asarray(z, jnp.bfloat16), jnp.float32)
    args = (f32(query), f32(key), f32(value))
    _, s_exact = _chunk_rule(
        *args, f32(g), f32(beta), f32(s0), chunk_size=C,
        compute_dtype=jnp.bfloat16,
    )
    _, s_round = _chunk_rule(
        *args, rounded(g), f32(beta), f32(s0), chunk_size=C,
        compute_dtype=jnp.bfloat16,
    )
    # If the core rounded `g` itself, these two calls would be identical.
    self.assertGreater(_max_diff(s_exact, s_round), 0.0)
    # ... and keeping it in float32 is the better answer. The comparison is on
    # the STATE: it is what the cumulated decay compounds into, while the
    # outputs' worst element is the same bfloat16-rounded q/k product in both.
    _, s_ref = _torch_chunk_gated_delta_rule(query, key, value, g, beta, C, s0)
    self.assertLess(_max_diff(s_exact, s_ref), _max_diff(s_round, s_ref))

  def test_l2_normalize_matches_reference(self):
    x = np.random.default_rng(8).normal(size=(B, T, HV, K))
    y = gdn.l2_normalize(jnp.asarray(x, jnp.float32))
    self.assertLess(_max_diff(y, _l2norm(x)), ATOL)
    self.assertEqual(y.dtype, jnp.float32)
    # HF normalizes before its kernels cast to f32, so bfloat16 in, bfloat16
    # out (modeling_qwen3_5.py:262-265).
    self.assertEqual(
        gdn.l2_normalize(jnp.asarray(x, jnp.bfloat16)).dtype, jnp.bfloat16
    )


class GdnModuleTest(parameterized.TestCase):
  """The `Qwen38GatedDeltaNet` module."""

  def test_param_tree_shapes(self):
    """The tree `utils/ckpt_format.py` maps the released tensors onto."""
    params = _layer().init(jax.random.PRNGKey(0))
    has_shape = lambda a: hasattr(a, 'shape')
    shapes = jax.tree.map(lambda a: tuple(a.shape), params, is_leaf=has_shape)
    self.assertEqual(
        shapes,
        {
            'in_proj_qkv': {'w': (D, CONV_DIM)},
            'in_proj_z': {'w': (D, VALUE_DIM)},
            'in_proj_b': {'w': (D, HV)},
            'in_proj_a': {'w': (D, HV)},
            'conv1d': {'w': (CONV_DIM, W)},
            'A_log': (HV,),
            'dt_bias': (HV,),
            'norm': {'scale': (V,)},
            'out_proj': {'w': (VALUE_DIM, D)},
        },
    )
    dtypes = jax.tree.map(lambda a: a.dtype, params, is_leaf=has_shape)
    self.assertEqual(dtypes['conv1d']['w'], jnp.float32)
    self.assertEqual(dtypes['A_log'], jnp.float32)
    self.assertEqual(dtypes['dt_bias'], jnp.float32)
    self.assertEqual(dtypes['norm']['scale'], jnp.float32)

  @parameterized.parameters(4, 8, 32)
  def test_apply_matches_reference_layer(self, chunk_size: int):
    w = _random_weights()
    x = np.random.default_rng(7).normal(size=(B, T, D))
    y_ref, conv_ref, s_ref = _reference_layer(x, w, chunk_size=chunk_size)
    y, extra = _apply_fn(chunk_size)(
        _params(w),
        jnp.asarray(x, jnp.float32),
        _ones(),
        _positions(),
        _layer(chunk_size).init_decode_state(B, T),
    )
    self.assertEqual(y.shape, (B, T, D))
    self.assertLess(_max_diff(y, y_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_ref), ATOL)

  @parameterized.parameters(9, 13, 17)
  def test_sequence_length_not_a_multiple_of_the_chunk_size(self, t: int):
    """T % chunk_size != 0 for all three: the remainder chunk is padded."""
    self.assertNotEqual(t % C, 0)
    w = _random_weights(seed=9)
    x = np.random.default_rng(19).normal(size=(B, t, D))
    y_ref, _, s_ref = _reference_layer(x, w)
    y, extra = _apply_fn()(
        _params(w),
        jnp.asarray(x, jnp.float32),
        _ones(t),
        _positions(t),
        _layer().init_decode_state(B, t),
    )
    self.assertLess(_max_diff(y, y_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)

  def test_a_one_token_prefill_matches_the_chunked_reference(self):
    """T = 1 with a fresh state takes the step core, HF the chunked one."""
    w = _random_weights(seed=11)
    x = np.random.default_rng(53).normal(size=(B, 1, D))
    y_ref, conv_ref, s_ref = _reference_layer(x, w)  # chunked, no cache
    y, extra = _apply_fn()(
        _params(w),
        jnp.asarray(x, jnp.float32),
        _ones(1),
        _positions(1),
        _layer().init_decode_state(B, T),
    )
    self.assertLess(_max_diff(y, y_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_ref), ATOL)

  def test_apply_without_decode_state(self):
    w = _random_weights()
    x = np.random.default_rng(7).normal(size=(B, T, D))
    y_ref, _, _ = _reference_layer(x, w)
    y, extra = _apply_fn()(
        _params(w), jnp.asarray(x, jnp.float32), _ones(), _positions(), None
    )
    self.assertIsNone(extra['decode_state'])
    self.assertLess(_max_diff(y, y_ref), ATOL)

  def test_prefill_then_decode_matches_full_prefill(self):
    """The production seam: chunked prefill, then cached single-token steps."""
    w = _random_weights(seed=1)
    params = _params(w)
    x = np.random.default_rng(11).normal(size=(B, T, D))
    apply_fn = _apply_fn()
    positions = _positions()
    y_full, extra_full = apply_fn(
        params,
        jnp.asarray(x, jnp.float32),
        _ones(),
        positions,
        _layer().init_decode_state(B, T),
    )
    prefill = 7
    y_pre, extra = apply_fn(
        params,
        jnp.asarray(x[:, :prefill], jnp.float32),
        _ones(prefill),
        positions[:, :prefill],
        _layer().init_decode_state(B, T),
    )
    steps = [y_pre]
    for i in range(prefill, T):
      y_i, extra = apply_fn(
          params,
          jnp.asarray(x[:, i : i + 1], jnp.float32),
          _ones(1),
          positions[:, i : i + 1],
          _state_of(extra),
      )
      steps.append(y_i)
    y_inc = jnp.concatenate(steps, axis=1)
    self.assertLess(_max_diff(y_inc, y_full), ATOL)
    self.assertLess(
        _max_diff(
            _state_of(extra).recurrent_state,
            _state_of(extra_full).recurrent_state,
        ),
        ATOL,
    )
    self.assertLess(
        _max_diff(
            _state_of(extra).conv_state, _state_of(extra_full).conv_state
        ),
        ATOL,
    )

  def test_padded_prefill_matches_the_unpadded_prompt(self):
    """The regression test for "prefill fed its pad suffix to the recurrence".

    The padded positions carry GARBAGE (a real batch embeds a pad token id, not
    zeros), so this exercises the input masking, the `g`/`beta` masking and the
    conv window rather than relying on zeros to be neutral.
    """
    w = _random_weights(seed=2)
    n_valid = 9
    x = np.random.default_rng(13).normal(size=(B, T, D))
    x[:, n_valid:] = 7.0 * np.random.default_rng(14).normal(
        size=(B, T - n_valid, D)
    )
    segment_ids = np.ones((B, T), np.int32)
    segment_ids[:, n_valid:] = 0
    y, extra = _apply_fn()(
        _params(w),
        jnp.asarray(x, jnp.float32),
        jnp.asarray(segment_ids),
        _positions(),
        _layer().init_decode_state(B, T),
    )
    y_ref, conv_ref, s_ref = _reference_layer(x[:, :n_valid], w)  # no pads
    with self.subTest('equals_the_unpadded_prompt'):
      self.assertLess(_max_diff(y[:, :n_valid], y_ref), ATOL)
      np.testing.assert_array_equal(np.asarray(y[:, n_valid:]), 0.0)
      self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)
      self.assertLess(_max_diff(_state_of(extra).conv_state, conv_ref), ATOL)
    with self.subTest('equals_the_masked_reference'):
      # The same claim through the reference's own mask path, so its
      # `apply_mask_to_padding_states` transcription is exercised too.
      y_masked, _, s_masked = _reference_layer(
          x, w, attention_mask=(segment_ids != 0).astype(np.float64)
      )
      self.assertLess(_max_diff(y, y_masked), ATOL)
      self.assertLess(
          _max_diff(_state_of(extra).recurrent_state, s_masked), ATOL
      )

  def test_an_all_padding_row_leaves_its_state_untouched(self):
    """A decode step on a row that is still padding must be a no-op.

    `sampling_lib` hands one token per step with its ABSOLUTE position, and a
    row that has not started carries `segment_ids == 0` with position 0; the
    segment reset must key off the token being real, never off the position.
    """
    w = _random_weights(seed=4)
    params = _params(w)
    apply_fn = _apply_fn()
    x = np.random.default_rng(23).normal(size=(B, T, D))
    _, extra = apply_fn(
        params,
        jnp.asarray(x, jnp.float32),
        _ones(),
        _positions(),
        _layer().init_decode_state(B, T),
    )
    before = _state_of(extra)
    step = np.random.default_rng(24).normal(size=(B, 1, D))
    segment_ids = np.array([[1], [0]], np.int32)  # row 1 is padding
    positions = np.array([[T], [0]], np.int32)  # ... and carries position 0
    y, extra = apply_fn(
        params, jnp.asarray(step, jnp.float32), jnp.asarray(segment_ids),
        jnp.asarray(positions), before,
    )
    after = _state_of(extra)
    with self.subTest('padding_row_is_frozen'):
      np.testing.assert_array_equal(
          np.asarray(after.recurrent_state[1]),
          np.asarray(before.recurrent_state[1]),
      )
      np.testing.assert_array_equal(
          np.asarray(after.conv_state[1]), np.asarray(before.conv_state[1])
      )
      np.testing.assert_array_equal(np.asarray(y[1]), 0.0)
    with self.subTest('real_row_still_advances'):
      y_ref, conv_ref, s_ref = _reference_layer(
          step[:1],
          w,
          conv_state=np.asarray(before.conv_state[:1], np.float64),
          recurrent_state=np.asarray(before.recurrent_state[:1], np.float64),
      )
      self.assertLess(_max_diff(y[:1], y_ref), ATOL)
      self.assertLess(_max_diff(after.recurrent_state[:1], s_ref), ATOL)
      self.assertLess(_max_diff(after.conv_state[:1], conv_ref), ATOL)

  def test_a_decode_step_at_position_zero_restarts_the_state(self):
    """A real token at position 0 in a cached step is a NEW sequence.

    This is the line that keeps a finished sequence's recurrent state out of
    the next sequence placed in the same batch slot: silent when it breaks, and
    invisible to every value test whose "restart" row is padding (padding takes
    the identity branch instead). Row 0 restarts, row 1 continues.
    """
    w = _random_weights(seed=13)
    params = _params(w)
    apply_fn = _apply_fn()
    x = np.random.default_rng(61).normal(size=(B, T, D))
    _, extra = apply_fn(
        params,
        jnp.asarray(x, jnp.float32),
        _ones(),
        _positions(),
        _layer().init_decode_state(B, T),
    )
    before = _state_of(extra)
    step = np.random.default_rng(62).normal(size=(B, 1, D))
    y, extra = apply_fn(
        params,
        jnp.asarray(step, jnp.float32),
        jnp.ones((B, 1), jnp.int32),  # both rows are REAL tokens
        jnp.asarray([[0], [T]], jnp.int32),  # row 0 restarts, row 1 continues
        before,
    )
    after = _state_of(extra)
    with self.subTest('row_0_starts_from_a_zero_state'):
      y_fresh, _, s_fresh = _reference_layer(step[:1], w)  # no carried state
      self.assertLess(_max_diff(y[:1], y_fresh), ATOL)
      self.assertLess(_max_diff(after.recurrent_state[:1], s_fresh), ATOL)
    with self.subTest('row_1_continues'):
      _, _, s_cont = _reference_layer(
          step[1:],
          w,
          conv_state=np.asarray(before.conv_state[1:], np.float64),
          recurrent_state=np.asarray(before.recurrent_state[1:], np.float64),
      )
      self.assertLess(_max_diff(after.recurrent_state[1:], s_cont), ATOL)
    with self.subTest('the_restart_is_not_free'):
      # Without the reset, row 0 would carry `before`; that is a real change.
      _, _, s_leaked = _reference_layer(
          step[:1],
          w,
          conv_state=np.asarray(before.conv_state[:1], np.float64),
          recurrent_state=np.asarray(before.recurrent_state[:1], np.float64),
      )
      self.assertGreater(_max_diff(s_fresh, s_leaked), AB_MIN_DIFF)

  def test_left_padding_before_a_fresh_sequence(self):
    """Pads BEFORE the first real token, with no conv history to displace.

    This is the one padded shape in which HF's `apply_mask_to_padding_states`
    is load-bearing: the pads sit in the conv taps of the first real tokens, so
    only zeroing the hidden states makes them the zeros a fresh sequence sees.
    (Right padding hides it: a pad never enters a valid token's taps.)
    """
    w = _random_weights(seed=14)
    n_pad = 3
    x = np.random.default_rng(67).normal(size=(B, T, D))
    x[:, :n_pad] = 5.0 * np.random.default_rng(68).normal(size=(B, n_pad, D))
    segment_ids = np.ones((B, T), np.int32)
    segment_ids[:, :n_pad] = 0
    y, extra = _apply_fn()(
        _params(w),
        jnp.asarray(x, jnp.float32),
        jnp.asarray(segment_ids),
        _positions(),  # absolute positions: no segment restart anywhere
        _layer().init_decode_state(B, T),
    )
    y_ref, conv_ref, s_ref = _reference_layer(x[:, n_pad:], w)
    self.assertLess(_max_diff(y[:, n_pad:], y_ref), ATOL)
    np.testing.assert_array_equal(np.asarray(y[:, :n_pad]), 0.0)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_ref), ATOL)

  def test_conv_accumulation_dtype_is_a_knob(self):
    """The conv accumulates in float32 BY DEFAULT, and that is the good one.

    The default is compared against an explicit bfloat16 accumulation, so this
    fails both if the knob stops working and if its default silently becomes
    the activation dtype.
    """
    w = _random_weights(seed=15)
    x = np.random.default_rng(71).normal(size=(B, T, D))
    y_ref, _, _ = _reference_layer(x, w)
    args = (
        _params(w), jnp.asarray(x, jnp.float32), _ones(), _positions(), None,
    )
    y_default = _apply_fn(C, 'bfloat16', 'float32')(*args)[0]
    y_bf16 = _apply_fn(C, 'bfloat16', 'float32', False, 'bfloat16')(*args)[0]
    self.assertGreater(_max_diff(y_default, y_bf16), 0.0)
    self.assertLess(_max_diff(y_default, y_ref), _max_diff(y_bf16, y_ref))

  def test_l2_normalize_accumulates_in_float32(self):
    """The sum of squares is float32 even for bfloat16 input, as fla does."""
    x = jnp.asarray(
        np.random.default_rng(73).normal(size=(B, T, HV, K)), jnp.bfloat16
    )
    x32 = jnp.asarray(x, jnp.float32)
    in_f32 = jnp.asarray(
        x32 * jax.lax.rsqrt(
            jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + gdn.L2NORM_EPS
        ),
        jnp.bfloat16,
    )
    in_bf16 = x * jax.lax.rsqrt(
        jnp.sum(jnp.square(x), axis=-1, keepdims=True)
        + jnp.asarray(gdn.L2NORM_EPS, jnp.bfloat16)
    )
    np.testing.assert_array_equal(
        np.asarray(gdn.l2_normalize(x), np.float32),
        np.asarray(in_f32, np.float32),
    )
    # ... and the bfloat16 accumulation really is a different answer.
    self.assertGreater(_max_diff(in_f32, in_bf16), 0.0)

  def test_packed_segments_match_separate_sequences(self):
    """Two segments packed in one row == the two rows run on their own."""
    w = _random_weights(seed=3)
    params = _params(w)
    cut = 5
    x = np.random.default_rng(17).normal(size=(1, T, D))
    segment_ids = np.ones((1, T), np.int32)
    segment_ids[:, cut:] = 2
    positions = np.concatenate(
        [np.arange(cut), np.arange(T - cut)], axis=0
    )[None]
    y, extra = _apply_fn()(
        params,
        jnp.asarray(x, jnp.float32),
        jnp.asarray(segment_ids),
        jnp.asarray(positions),
        _layer().init_decode_state(1, T),
    )
    y_first, _, _ = _reference_layer(x[:, :cut], w)
    y_second, conv_ref, s_ref = _reference_layer(x[:, cut:], w)
    self.assertLess(_max_diff(y[:, :cut], y_first), ATOL)
    self.assertLess(_max_diff(y[:, cut:], y_second), ATOL)
    self.assertLess(_max_diff(_state_of(extra).recurrent_state, s_ref), ATOL)
    self.assertLess(_max_diff(_state_of(extra).conv_state, conv_ref), ATOL)

  def test_beta_is_sigmoided_exactly_once(self):
    """`beta` gets exactly one sigmoid (PLAN.md defect 5).

    The ragged kernel re-sigmoided an already-sigmoided `beta`:
    `sigmoid(sigmoid(x))` sends 0 to 0.622 instead of 0.5, a 38.8% error at the
    kernel which the output norm attenuates to ~2e-2 RELATIVE at the layer
    output of the real model -- small enough to pass for kernel noise for
    months. At this geometry it moves the output by 0.81 absolute.
    """
    w = _random_weights(seed=5)
    x = np.random.default_rng(29).normal(size=(B, T, D))
    y, _ = _apply_fn()(
        _params(w), jnp.asarray(x, jnp.float32), _ones(), _positions(), None
    )
    y_once, _, _ = _reference_layer(x, w)
    y_twice, _, _ = _reference_layer(x, w, beta_sigmoid_twice=True)
    self.assertLess(_max_diff(y, y_once), ATOL)
    self.assertGreater(_max_diff(y_once, y_twice), AB_MIN_DIFF)
    self.assertGreater(_max_diff(y, y_twice), AB_MIN_DIFF)

  def test_the_output_gate_is_silu_not_sigmoid(self):
    """`output_gate_type: "swish"` names silu; sigmoid is the attention gate."""
    w = _random_weights(seed=6)
    x = np.random.default_rng(31).normal(size=(B, T, D))
    y, _ = _apply_fn()(
        _params(w), jnp.asarray(x, jnp.float32), _ones(), _positions(), None
    )
    y_silu, _, _ = _reference_layer(x, w, gate_activation='silu')
    y_sigmoid, _, _ = _reference_layer(x, w, gate_activation='sigmoid')
    self.assertLess(_max_diff(y, y_silu), ATOL)
    self.assertGreater(_max_diff(y_silu, y_sigmoid), AB_MIN_DIFF)
    self.assertGreater(_max_diff(y, y_sigmoid), AB_MIN_DIFF)

  def test_the_output_norm_has_no_plus_one(self):
    """`Qwen3_5RMSNormGated` is `w * x`; the block norms are the `(1 + w)`s."""
    w = _random_weights(seed=7)
    x = np.random.default_rng(37).normal(size=(B, T, D))
    y, _ = _apply_fn()(
        _params(w), jnp.asarray(x, jnp.float32), _ones(), _positions(), None
    )
    y_plain, _, _ = _reference_layer(x, w, norm_plus_one=False)
    y_plus_one, _, _ = _reference_layer(x, w, norm_plus_one=True)
    self.assertLess(_max_diff(y, y_plain), ATOL)
    self.assertGreater(_max_diff(y_plain, y_plus_one), AB_MIN_DIFF)
    self.assertGreater(_max_diff(y, y_plus_one), AB_MIN_DIFF)

  def test_the_output_norm_rounds_before_the_gain(self):
    """HF `self.weight * hidden_states.to(input_dtype)`: round, THEN gain.

    `Qwen3_5RMSNormGated` rounds the normalized value to the activation dtype
    before multiplying the gain, and takes only `silu(gate)` in float32 -- the
    opposite of `Qwen3_5RMSNorm`, whose `(1 + w)` stays in float32. The two
    orders differ in the last bits, which is exactly what a bfloat16 parity run
    against HuggingFace would trip over.
    """
    layer = _layer(C, 'bfloat16', 'float32')
    rng = np.random.default_rng(47)
    scale = jnp.asarray(rng.normal(1.0, 0.1, size=(V,)), jnp.float32)
    x = jnp.asarray(rng.normal(size=(B * T, V)), jnp.bfloat16)
    y = layer.norm.apply({'scale': scale}, x)
    x32 = jnp.asarray(x, jnp.float32)
    normed = x32 * jax.lax.rsqrt(
        jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + RMS_EPS
    )
    round_then_gain = jnp.asarray(normed, jnp.bfloat16) * jnp.asarray(
        scale, jnp.bfloat16
    )
    gain_then_round = jnp.asarray(normed * scale, jnp.bfloat16)
    self.assertEqual(y.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(y, np.float32), np.asarray(round_then_gain, np.float32)
    )
    # ... and the two orders really are distinguishable here.
    self.assertGreater(_max_diff(round_then_gain, gain_then_round), 0.0)

  @parameterized.named_parameters(
      ('f32_activations_f32_kernel', 'float32', 'float32'),
      ('bf16_activations_f32_kernel', 'bfloat16', 'float32'),
      ('f32_activations_bf16_kernel', 'float32', 'bfloat16'),
      ('bf16_activations_bf16_kernel', 'bfloat16', 'bfloat16'),
  )
  def test_dtypes(self, activation_dtype: str, gdn_compute_dtype: str):
    """Every dtype pair stays within a bfloat16 sanity bound of the f64 ref."""
    w = _random_weights(seed=8)
    x = np.random.default_rng(41).normal(size=(B, T, D))
    y_ref, _, _ = _reference_layer(x, w)
    y, extra = _apply_fn(C, activation_dtype, gdn_compute_dtype)(
        _params(w),
        jnp.asarray(x, jnp.float32),
        _ones(),
        _positions(),
        _layer(C, activation_dtype, gdn_compute_dtype).init_decode_state(B, T),
    )
    self.assertEqual(y.dtype, jnp.dtype(activation_dtype))
    # The state is float32 whatever the activations are: it is carried across
    # every decode step, so its rounding compounds.
    self.assertEqual(_state_of(extra).recurrent_state.dtype, jnp.float32)
    self.assertEqual(_state_of(extra).conv_state.dtype, jnp.float32)
    if activation_dtype == 'float32' and gdn_compute_dtype == 'float32':
      self.assertLess(_max_diff(y, y_ref), ATOL)
      return
    scale = float(np.max(np.abs(y_ref)))
    if activation_dtype == 'float32':
      # Only the kernel is bfloat16; the operands are not quantized.
      self.assertLess(_max_diff(y, y_ref), BF16_KERNEL_REL * scale)
      return
    self.assertLess(_max_diff(y, y_ref), BF16_REL * scale)
    # The real bound: with bfloat16 activations the module must land ON the
    # operand-quantization floor, not merely inside a fixed fraction of the
    # output scale. `y_quantized` is the f64 reference re-run on
    # bfloat16-rounded weights and inputs, so `floor` is what bfloat16 costs
    # before the module does anything; a dtype slip inside the layer shows up
    # as a multiple of it (measured: 0.20 of the floor, 0.26 with the bfloat16
    # kernel).
    y_quantized, _, _ = _reference_layer(
        _to_bfloat16(x), {k: _to_bfloat16(v) for k, v in w.items()}
    )
    floor = _max_diff(y_quantized, y_ref)
    self.assertLess(_max_diff(y, y_quantized), 0.5 * floor)

  def test_float32_kernel_beats_bfloat16_kernel(self):
    """`gdn_compute_dtype` is a real knob: f32 is the faithful one."""
    w = _random_weights(seed=8)
    x = np.random.default_rng(41).normal(size=(B, T, D))
    y_ref, _, _ = _reference_layer(x, w)
    args = (
        _params(w),
        jnp.asarray(x, jnp.float32),
        _ones(),
        _positions(),
        None,
    )
    y_f32, _ = _apply_fn(C, 'float32', 'float32')(*args)
    y_bf16, _ = _apply_fn(C, 'float32', 'bfloat16')(*args)
    self.assertLess(_max_diff(y_f32, y_ref), _max_diff(y_bf16, y_ref))

  @parameterized.parameters(8, 1)  # chunked prefill, then cached decode
  def test_every_float32_dot_asks_for_float32_precision(self, t: int):
    """The state-carrying matmuls must not lower to the default precision.

    On CPU `DEFAULT` is float32, so no value test can see this; on TPU it is a
    bfloat16 pass, which would carry the recurrent state at bfloat16 in decode
    while prefill stayed float32. The four projections are excluded on purpose:
    their operands are bfloat16, for which `DEFAULT` is already HF's behaviour.

    Args:
      t: sequence length; 1 takes the recurrent step, 8 the chunked core.
    """
    layer = _layer(C, 'bfloat16', 'float32')
    w = _random_weights(seed=12)
    x = jnp.asarray(
        np.random.default_rng(59).normal(size=(B, t, D)), jnp.bfloat16
    )
    fn = jax.jit(
        lambda p, u, s: layer.apply(
            p, u, segment_ids=_ones(t), segment_positions=_positions(t),
            decode_state=s,
        )
    )
    text = fn.lower(_params(w), x, layer.init_decode_state(B, t)).as_text()
    dots = re.findall(r'stablehlo\.dot_general[^\n]*', text)
    f32_dots = [d for d in dots if _F32_OPERANDS.search(d)]
    # Non-vacuity: the recurrence is 2 dots per step, the chunked core 8.
    self.assertGreaterEqual(len(f32_dots), 2)
    for dot in f32_dots:
      self.assertIn('precision = [HIGHEST, HIGHEST]', dot)

  def test_decode_state_shapes(self):
    state = _layer().init_decode_state(B, 128)
    self.assertEqual(state.conv_state.shape, (B, CONV_DIM, W - 1))
    self.assertEqual(state.recurrent_state.shape, (B, HV, K, V))
    # The state is constant in `max_seq_len`.
    other = _layer().init_decode_state(B, 4096)
    self.assertEqual(other.conv_state.shape, state.conv_state.shape)

  def test_decode_state_is_a_pytree(self):
    """`jax.tree` flatten/unflatten and a jit round trip, as a cache must."""
    state = _layer().init_decode_state(B, T)
    leaves, treedef = jax.tree.flatten(state)
    self.assertLen(leaves, 2)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    self.assertIsInstance(rebuilt, gdn.GatedDeltaNetDecodeState)
    np.testing.assert_array_equal(
        np.asarray(rebuilt.conv_state), np.asarray(state.conv_state)
    )
    bumped = jax.jit(
        lambda s: gdn.GatedDeltaNetDecodeState(
            conv_state=s.conv_state + 1.0,
            recurrent_state=s.recurrent_state + 2.0,
        )
    )(state)
    self.assertIsInstance(bumped, gdn.GatedDeltaNetDecodeState)
    np.testing.assert_allclose(np.asarray(bumped.conv_state), 1.0)
    np.testing.assert_allclose(np.asarray(bumped.recurrent_state), 2.0)

  def test_sharding_specs_name_the_intended_axes(self):
    """The axis each annotation names, on a real four-device mesh.

    Asserted eagerly: `with_sharding_constraint` then reshards for real, so the
    spec read back is the one the module asked for. Under `jax.jit` XLA
    normalizes the output specs (it drops the size-1 `replica` axis), which
    would make this test unable to tell a wrong axis from a legal rewrite.
    `model` must land on the 48 value heads and on `conv_dim`, never on the
    3-slot conv window or on `value_head_dim`.
    """
    partition = jax.sharding.PartitionSpec
    w = _random_weights(seed=10)
    x = np.random.default_rng(43).normal(size=(B, T, D))
    layer = _layer(C, 'float32', 'float32', True)
    with sharding_lib.set_mesh(_MESH_SHAPE):
      state = layer.init_decode_state(B, T)
      y, extra = layer.apply(
          _params(w),
          jnp.asarray(x, jnp.float32),
          segment_ids=_ones(),
          segment_positions=_positions(),
          decode_state=state,
      )
      new_state = _state_of(extra)
      batch = ('replica', 'data')
      with self.subTest('initial_state'):
        self.assertEqual(
            _spec(state.conv_state), partition(batch, 'model', None)
        )
        self.assertEqual(
            _spec(state.recurrent_state),
            partition(batch, 'model', None, None),
        )
      with self.subTest('updated_state'):
        self.assertEqual(
            _spec(new_state.conv_state), partition(batch, 'model', None)
        )
        self.assertEqual(
            _spec(new_state.recurrent_state),
            partition(batch, 'model', None, None),
        )
      with self.subTest('output'):
        self.assertEqual(_spec(y), partition(batch, None, 'model'))
      with self.subTest('value_is_unchanged_by_sharding'):
        y_ref, _, _ = _reference_layer(x, w)
        self.assertLess(_max_diff(y, y_ref), ATOL)

  def test_the_thin_projections_are_unannotated_not_replicated(self):
    """`output_partition=None` is a REPLICATE constraint, not "unset".

    `sharding_lib.partition_spec(None)` is `PartitionSpec()` and
    `EinsumLinear.apply` applies it unconditionally, so `None` on the
    `[B, T, num_value_heads]` projections would all-gather them on every layer.
    Only the sentinel is the no-op, and no array read back from `apply` can
    show the difference — hence the assertion on the annotation itself.
    """
    layer = _layer(C, 'float32', 'float32', True)
    for name in ('in_proj_b', 'in_proj_a'):
      self.assertIs(
          getattr(layer, name).output_partition, sharding_lib.NOT_ANNOTATED
      )
    for name in ('in_proj_qkv', 'in_proj_z', 'out_proj'):
      self.assertEqual(
          getattr(layer, name).output_partition,
          _StubSharding.activation_partition,
      )
    self.assertIs(layer.norm.scale_partition, sharding_lib.NOT_ANNOTATED)

  def test_rejects_value_heads_that_are_not_a_multiple_of_key_heads(self):
    with self.assertRaises(ValueError):
      gdn.Qwen38GatedDeltaNet(model_dim=D, num_key_heads=4, num_value_heads=6)


if __name__ == '__main__':
  absltest.main()

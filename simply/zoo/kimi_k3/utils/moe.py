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
"""Kimi K3 channel mixing: the dense SiTU-GLU MLP and the Stable LatentMoE.

Transcribed from the HuggingFace release `moonshotai/Kimi-K3`
(`modeling_kimi_linear.py`): `KimiMLP` and `KimiBlockSparseMLP` (the SiTU-GLU
MLPs), `KimiMoEGate` (sigmoid router with a frozen score-correction bias) and
`KimiSparseMoeBlock` (the latent MoE: down-project 7168 -> 3584, run the routed
experts in the latent space, RMSNorm, up-project, add the fused shared expert).

Two things the HF code hides and that are easy to get wrong:

  * the router's `e_score_correction_bias` biases the top-k *selection* only;
    the combine weights are the UNBIASED sigmoid scores at the selected
    experts, renormalized to sum 1 and scaled by `routed_scaling_factor`;
  * SiTU-GLU soft-caps BOTH branches (`beta * tanh(gate / beta) *
    sigmoid(gate)` times `linear_beta * tanh(up / linear_beta)`), so it cannot
    be expressed as one of Simply's registered pointwise `ffn_activation`s --
    `_situ_glu` owns it.

Routed-expert execution has two interchangeable paths, selected by
`KimiK3LatentMoE.expert_dispatch` (they agree numerically; `moe_multi_device_test` pins
them together):

  'gmm'   sort the (token, slot) pairs by expert id and run one grouped
          matmul per projection (`model_lib.gmm`), the production path;
  'dense' run every expert on every token and select with a one-hot combine.
          Dropless, unlike Simply's `MoEFeedForward.ep_method='dense'`, which
          means capacity-based dispatch. E/K times the flops (56x at K3's
          896/16), so it is a reference for tests and toy configs.

Scaling to K3's 896 experts. Dispatching in the LATENT space is the structural
win: the exchanged payload is `K * L` per token instead of `K * D`, half the
bytes of a model-space MoE at the same top-k. The grouped path keeps the expert
weights in Simply's `[E, in, out]` stacked layout with the expert axis annotated
from `ffn0_partition`/`ffn1_partition` and reduces the layer to a sort, a
bincount and three grouped matmuls over that stack -- the unit
`model_lib.MoEFeedForward._apply_sparse_moe` runs inside a `shard_map` with a
`ragged_all_to_all` exchange. Expert parallelism therefore replaces
`_grouped_experts`, and three things have to move with it: the seam must take
`indices` and compute its own per-shard `group_sizes` (the exchange offsets need
an `all_gather` of the local counts), the row buffer must be per shard rather
than the global `[N, L]` flatten, and the latent RMSNorm needs a full-L row, so
it must be pinned replicated over the model axis or folded into the combine
exit. Until then a mesh with an expert axis makes GSPMD replicate the stacks, so
this file is exact at any E but economical only while they fit.
"""

import dataclasses
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.typing
from simply import model_lib
from simply.utils import common
from simply.utils import initializer
from simply.utils import module
from simply.utils import sharding as sharding_lib

Array = common.Array
PyTree = common.PyTree
PartitionAnnotation = common.PartitionAnnotation
PRNGKey = jax.typing.ArrayLike
DTypeLike = jax.typing.DTypeLike
SimplyConfig = Any


def _situ_glu(
    gate: Array, up: Array, beta: float = 4.0, linear_beta: float = 25.0
) -> Array:
  """SiTU-GLU: `soft_cap(gate, b1) * sigmoid(gate) * soft_cap(up, b2)`.

  `model_lib.soft_cap(x, b)` is `b * tanh(x / b)`; it bounds both factors of
  the gated unit, which SwiGLU leaves unbounded (report S2.3.2, Eq. 9).
  Computed in f32 and cast back, as in `SituAndMul.forward`.

  Args:
    gate: pre-activation of the gate branch.
    up: pre-activation of the up branch.
    beta: soft cap of the gate branch (4.0 for K3).
    linear_beta: soft cap of the up branch (25.0 for K3).

  Returns:
    The gated activation, in `gate`'s dtype.
  """
  dtype = gate.dtype
  gate32 = jnp.asarray(gate, jnp.float32)
  up32 = jnp.asarray(up, jnp.float32)
  activated = model_lib.soft_cap(gate32, beta) * jax.nn.sigmoid(gate32)
  capped_up = model_lib.soft_cap(up32, linear_beta)
  return jnp.asarray(activated * capped_up, dtype)


def _dense_annotation(
    partition: PartitionAnnotation,
) -> PartitionAnnotation:
  """Rank-2 `[in, out]` annotation from a possibly rank-3 MoE one.

  MoE sharding configs (`config_lib.moe_sharding`) spell `ffn0_partition` for
  the stacked `[E, in, out]` expert weights; the plain projections of this file
  are `[in, out]` and drop the leading expert axis.

  Args:
    partition: annotation from the config, rank 2 or rank 3.

  Returns:
    The trailing two axes of `partition`, or `partition` itself when it carries
    no axis names (None or `NOT_ANNOTATED`).
  """
  if partition is None or partition is sharding_lib.NOT_ANNOTATED:
    return partition
  return tuple(partition[-2:])


def _expert_annotation(
    partition: PartitionAnnotation,
) -> PartitionAnnotation:
  """Rank-3 `[E, in, out]` annotation from a possibly rank-2 config one."""
  if partition is None or partition is sharding_lib.NOT_ANNOTATED:
    return partition
  if len(partition) == 3:
    return tuple(partition)
  return (None, *partition)


def _config_annotation(
    sharding_config: SimplyConfig, name: str
) -> PartitionAnnotation:
  if sharding_config is None:
    return sharding_lib.NOT_ANNOTATED
  return getattr(sharding_config, name)


@module.ModuleRegistry.register
@dataclasses.dataclass
class KimiK3DenseMLP(module.SimplyModule):
  """Dense SiTU-GLU MLP: `ffn_1(situ_glu(ffn_0_gate(x), ffn_0(x)))`.

  HF `KimiMLP` (`gate_proj`, `up_proj`, `down_proj`). One module serves both
  uses: layer 0's dense FFN (`expand_dim` 33792) and the LatentMoE's shared
  expert, where K3's two shared experts are fused into one MLP of twice the
  routed width (`expand_dim` 6144).

  Attributes:
    model_dim: input/output width.
    expand_dim: hidden width.
    sharding_config: Simply sharding config; `None` leaves everything
      unannotated.
    situ_beta: soft cap of the gate branch.
    situ_linear_beta: soft cap of the up branch.
    activation_dtype: dtype of the projections' matmuls and outputs.
    weight_dtype: dtype of freshly initialized weights.
    weight_init: initializer for the three projections.
  """

  model_dim: int
  expand_dim: int
  sharding_config: SimplyConfig = None
  situ_beta: float = 4.0
  situ_linear_beta: float = 25.0
  activation_dtype: DTypeLike = 'bfloat16'
  weight_dtype: DTypeLike = 'float32'
  weight_init: initializer.Initializer = initializer.XavierUniformInit()

  def setup(self) -> None:
    ffn0_partition = _dense_annotation(
        _config_annotation(self.sharding_config, 'ffn0_partition')
    )
    ffn1_partition = _dense_annotation(
        _config_annotation(self.sharding_config, 'ffn1_partition')
    )
    hidden_partition = _config_annotation(
        self.sharding_config, 'ffn0_activation_partition'
    )
    output_partition = _config_annotation(
        self.sharding_config, 'activation_partition'
    )
    linear = lambda shape, weight_partition, out_partition: (
        module.EinsumLinear(
            eqn='io,...i->...o',
            weight_shape=shape,
            weight_dtype=self.weight_dtype,
            activation_dtype=self.activation_dtype,
            weight_partition=weight_partition,
            output_partition=out_partition,
            weight_init=self.weight_init,
        )
    )
    self.ffn_0_gate = linear(
        [self.model_dim, self.expand_dim], ffn0_partition, hidden_partition
    )
    self.ffn_0 = linear(
        [self.model_dim, self.expand_dim], ffn0_partition, hidden_partition
    )
    self.ffn_1 = linear(
        [self.expand_dim, self.model_dim], ffn1_partition, output_partition
    )

  def init(self, prng_key: PRNGKey) -> PyTree:
    gate_key, up_key, down_key = jax.random.split(prng_key, num=3)
    return {
        'ffn_0_gate': self.ffn_0_gate.init(gate_key),
        'ffn_0': self.ffn_0.init(up_key),
        'ffn_1': self.ffn_1.init(down_key),
    }

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      inputs_mask: Array | None = None,
  ) -> tuple[Array, dict[str, Any]]:
    """Returns `(y, extra_output)`, matching `model_lib.FeedForward.apply`."""
    del inputs_mask  # A dense MLP is position-wise; padding cannot leak.
    p: Any = common.get_raw_arrays(params)
    gate = self.ffn_0_gate.apply(p['ffn_0_gate'], x)
    up = self.ffn_0.apply(p['ffn_0'], x)
    hidden = _situ_glu(
        gate, up, beta=self.situ_beta, linear_beta=self.situ_linear_beta
    )
    return self.ffn_1.apply(p['ffn_1'], hidden), {}


@module.ModuleRegistry.register
@dataclasses.dataclass
class KimiK3LatentMoE(module.SimplyModule):
  """Stable LatentMoE: routed experts in a low-rank latent space.

  HF `KimiSparseMoeBlock` with `routed_expert_hidden_size` set. For input `x`
  `[B, T, D]`::

      indices, weights = route(sigmoid(x_f32 @ router_w), router_bias)
      y = combine(experts(x @ down_proj), indices, weights)   # in R^L
      y = rms_norm(y, latent_norm) @ up_proj + shared_mlp(x)

  The latent projections (`down_proj` / `up_proj`) are shared by all experts,
  so an expert only costs `2 * L * M + M * L` weights instead of `3 * D * M`;
  at K3's 896 experts that is what makes the layer affordable.

  Attributes:
    model_dim: residual width `D` (7168).
    latent_dim: routed-expert latent width `L` (3584).
    moe_intermediate_size: per-expert hidden width `M` (3072).
    num_experts: number of routed experts `E` (896).
    num_experts_per_token: `K` (16).
    shared_expert_dim: width of the fused shared MLP `S` (6144); 0 disables it.
    sharding_config: Simply sharding config; `None` leaves everything
      unannotated.
    routed_scaling_factor: multiplies the combine weights (1.0 for K3).
    renormalize: renormalize the top-k weights to sum 1.
    router_epsilon: added to the renormalization denominator (HF: 1e-20).
    use_latent_norm: apply `latent_norm` before the up-projection.
    rms_norm_epsilon: epsilon of that RMSNorm.
    situ_beta: soft cap of the experts' gate branch.
    situ_linear_beta: soft cap of the experts' up branch.
    activation_dtype: dtype of every matmul below the router.
    weight_dtype: dtype of freshly initialized weights.
    weight_init: initializer for all projections.
    expert_dispatch: 'gmm' (sorted grouped matmul) or 'dense' (all experts on
      all tokens); see the module docstring.
    gmm_impl: `model_lib.gmm` backend for `expert_dispatch='gmm'`, or
      'dense_grouped' for the kernel-free control (`_dense_grouped_matmul`).
    expert_parallel_axis: mesh axis the expert stacks are sharded over ('seq' in
      `config_lib.moe_sharding`). Set it to run the grouped matmuls under a
      `shard_map` instead of letting GSPMD all-gather the stacks
      (`_expert_parallel_matmuls`); requires the token axis not to be sharded
      over it, i.e. the decoding sharding config. **Decode only.** The gradients
      are exact on a CPU mesh at test width (`moe_multi_device_test`), but the path is slower
      than the plain one below ~1 B, where the gather it removes is 0.4% of the
      step.
    tile_batch_seq: grouped-matmul row tile.
    tile_latent_dim: grouped-matmul tile of the latent (contraction) dim.
    tile_expand_dim: grouped-matmul tile of the expert hidden dim.
  """

  model_dim: int
  latent_dim: int
  moe_intermediate_size: int
  num_experts: int
  num_experts_per_token: int
  shared_expert_dim: int = 0
  sharding_config: SimplyConfig = None
  routed_scaling_factor: float = 1.0
  renormalize: bool = True
  router_epsilon: float = 1e-20
  use_latent_norm: bool = True
  rms_norm_epsilon: float = 1e-5
  situ_beta: float = 4.0
  situ_linear_beta: float = 25.0
  activation_dtype: DTypeLike = 'bfloat16'
  weight_dtype: DTypeLike = 'float32'
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  expert_dispatch: Literal['gmm', 'dense'] = 'gmm'
  gmm_impl: str = 'ragged_dot'
  expert_parallel_axis: str | None = None
  tile_batch_seq: int = 128
  tile_latent_dim: int = 128
  tile_expand_dim: int = 128

  def setup(self) -> None:
    self._check_expert_parallel_axis()
    ffn0_partition = _config_annotation(self.sharding_config, 'ffn0_partition')
    ffn1_partition = _config_annotation(self.sharding_config, 'ffn1_partition')
    self.activation_partition = _config_annotation(
        self.sharding_config, 'activation_partition'
    )
    self.latent_partition = self.activation_partition
    self.expert_ffn0_partition = _expert_annotation(ffn0_partition)
    self.expert_ffn1_partition = _expert_annotation(ffn1_partition)

    dense_linear = lambda shape, partition, out_partition: (
        module.EinsumLinear(
            eqn='io,...i->...o',
            weight_shape=shape,
            weight_dtype=self.weight_dtype,
            activation_dtype=self.activation_dtype,
            weight_partition=_dense_annotation(partition),
            output_partition=out_partition,
            weight_init=self.weight_init,
        )
    )
    self.down_proj = dense_linear(
        [self.model_dim, self.latent_dim], ffn0_partition, self.latent_partition
    )
    self.up_proj = dense_linear(
        [self.latent_dim, self.model_dim],
        ffn1_partition,
        self.activation_partition,
    )

    # The expert stacks are declared as `EinsumLinear`s for their param tree,
    # shapes and sharding; both dispatch paths read the raw `[E, in, out]`
    # weights instead of calling `apply`.
    expert_stack = lambda shape, partition: module.EinsumLinear(
        eqn='eio,e...i->e...o',
        weight_shape=shape,
        weight_dim_annotation='.io',
        weight_dtype=self.weight_dtype,
        activation_dtype=self.activation_dtype,
        weight_partition=partition,
        weight_init=self.weight_init,
    )
    self.expert_ffn_0_gate = expert_stack(
        [self.num_experts, self.latent_dim, self.moe_intermediate_size],
        self.expert_ffn0_partition,
    )
    self.expert_ffn_0 = expert_stack(
        [self.num_experts, self.latent_dim, self.moe_intermediate_size],
        self.expert_ffn0_partition,
    )
    self.expert_ffn_1 = expert_stack(
        [self.num_experts, self.moe_intermediate_size, self.latent_dim],
        self.expert_ffn1_partition,
    )
    if self.use_latent_norm:
      # HF `routed_expert_norm` is a `KimiRMSNorm`: normalize in f32, cast to
      # the activation dtype, THEN apply the gain. `LayerNorm` spells that with
      # `activation_dtype` set to the activation dtype; KDA's `o_norm` is the
      # other flavour (gain in f32) and pins the same module to f32.
      self.latent_norm = model_lib.LayerNorm(
          dim=self.latent_dim,
          use_bias=False,
          scale_plus_one=False,
          epsilon=self.rms_norm_epsilon,
          weight_dtype=self.weight_dtype,
          activation_dtype=self.activation_dtype,
      )
    if self.shared_expert_dim:
      self.shared = KimiK3DenseMLP(
          model_dim=self.model_dim,
          expand_dim=self.shared_expert_dim,
          sharding_config=self.sharding_config,
          situ_beta=self.situ_beta,
          situ_linear_beta=self.situ_linear_beta,
          activation_dtype=self.activation_dtype,
          weight_dtype=self.weight_dtype,
          weight_init=self.weight_init,
      )

  def init(self, prng_key: PRNGKey) -> PyTree:
    keys = jax.random.split(prng_key, num=7)
    router_w = self.weight_init(
        keys[0],
        shape=(self.model_dim, self.num_experts),
        dim_annotation='io',
        dtype='float32',
    )
    params = {
        # Router weights and the frozen selection bias stay f32: the sigmoid
        # scores decide routing, and K3 ships the bias as f32.
        'router': {
            'w': common.AnnotatedArray.create(router_w, dim_annotation='io'),
            'bias': common.AnnotatedArray.create(
                jnp.zeros((self.num_experts,), jnp.float32),
                dim_annotation='h',
            ),
        },
        'down_proj': self.down_proj.init(keys[1]),
        'up_proj': self.up_proj.init(keys[2]),
        'experts': {
            'ffn_0_gate': self.expert_ffn_0_gate.init(keys[3]),
            'ffn_0': self.expert_ffn_0.init(keys[4]),
            'ffn_1': self.expert_ffn_1.init(keys[5]),
        },
    }
    if self.use_latent_norm:
      params['latent_norm'] = self.latent_norm.init()
    if self.shared_expert_dim:
      params['shared'] = self.shared.init(keys[6])
    return params

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      inputs_mask: Array | None = None,
  ) -> tuple[Array, dict[str, Any]]:
    """Runs the layer on `x` `[B, T, D]`; returns `(y, extra_output)`."""
    p: Any = common.get_raw_arrays(params)
    if inputs_mask is not None:
      x = jnp.where(inputs_mask[..., None], x, 0.0)
    indices, weights = self._route(p['router'], x)
    if inputs_mask is not None:
      # Padded tokens are routed to a phantom expert id: `bincount(length=E)`
      # drops it, so no expert spends flops on padding and the load metric
      # counts real tokens only. Their rows sort past `sum(group_sizes)`, which
      # both `model_lib.gmm` (it masks megablox's uninitialized tail) and
      # `ragged_dot` leave at zero, and the zero combine weight drops whatever
      # the dense path computes for them.
      indices = jnp.where(inputs_mask[..., None], indices, self.num_experts)
      weights = jnp.where(inputs_mask[..., None], weights, 0.0)

    latent = self.down_proj.apply(p['down_proj'], x)
    batch, seq_len, _ = latent.shape
    flat_latent = jnp.reshape(latent, (batch * seq_len, self.latent_dim))
    flat_indices = jnp.reshape(indices, (-1, self.num_experts_per_token))
    flat_weights = jnp.reshape(weights, (-1, self.num_experts_per_token))
    group_sizes = jnp.bincount(
        jnp.ravel(flat_indices), length=self.num_experts
    ).astype(jnp.int32)

    if self.expert_dispatch == 'gmm':
      combined = self._grouped_experts(
          p['experts'], flat_latent, flat_indices, flat_weights, group_sizes
      )
    elif self.expert_dispatch == 'dense':
      combined = self._dense_experts(
          p['experts'], flat_latent, flat_indices, flat_weights
      )
    else:
      raise ValueError(f'Unsupported {self.expert_dispatch=}.')

    y = jnp.reshape(combined, (batch, seq_len, self.latent_dim)).astype(
        self.activation_dtype
    )
    if self.use_latent_norm:
      y = self.latent_norm.apply(p['latent_norm'], y)
    y = self.up_proj.apply(p['up_proj'], y)
    if self.shared_expert_dim:
      shared, _ = self.shared.apply(p['shared'], x)
      y = y + shared
    if inputs_mask is not None:
      y = jnp.where(inputs_mask[..., None], y, 0.0)
    y = sharding_lib.with_sharding_constraint(
        cast(jax.Array, y), self.activation_partition
    )

    load = group_sizes / jnp.maximum(jnp.sum(group_sizes), 1)
    extra_output = {
        'loss': {},
        'metric': {
            'max_load': jnp.max(load),
            'min_load': jnp.min(load),
            'gini': jnp.sum(jnp.square(load)) * self.num_experts - 1,
        },
    }
    return y, extra_output

  def _route(self, params: Any, x: Array) -> tuple[Array, Array]:
    """Sigmoid router with a bias that only steers the selection.

    HF `KimiMoEGate.forward` (K3 sets `num_expert_group = topk_group = 1`, so
    the group-limited branch there is a structural no-op and is not ported).

    Args:
      params: `{'w': [D, E], 'bias': [E]}`.
      x: `[B, T, D]` activations.

    Returns:
      `(indices, weights)`: `[B, T, K]` int32 expert ids and f32 combine
      weights, taken from the unbiased scores.
    """
    w = common.convert_or_dequantize(params['w'], dtype=jnp.float32)
    bias = common.convert_or_dequantize(params['bias'], dtype=jnp.float32)
    # Pinned to f32 passes: the ambient default would run this matmul in bf16
    # and let near-ties flip the selected experts.
    with jax.default_matmul_precision('float32'):
      logits = jnp.einsum('...d,de->...e', jnp.asarray(x, jnp.float32), w)
    scores = jax.nn.sigmoid(logits)
    _, indices = jax.lax.top_k(scores + bias, self.num_experts_per_token)
    weights = jnp.take_along_axis(scores, indices, axis=-1)
    if self.num_experts_per_token > 1 and self.renormalize:
      weights = weights / (
          jnp.sum(weights, axis=-1, keepdims=True) + self.router_epsilon
      )
    return indices, weights * self.routed_scaling_factor

  def _check_expert_parallel_axis(self) -> None:
    """Expert parallelism needs the rows replicated over the expert axis.

    `_expert_parallel_matmuls` has every shard evaluate its own experts on
    *every* row. If the token axis were sharded over the same mesh axis, each
    shard would hold different rows and `shard_map` would have to all-gather
    the activations to honour `in_specs=P()` -- correct, but a silent cost
    where the point of the path is to remove one.

    Raises:
      ValueError: if the activation partition uses the expert axis.
    """
    axis = self.expert_parallel_axis
    partition = _config_annotation(self.sharding_config, 'activation_partition')
    if not axis or partition is None or partition is sharding_lib.NOT_ANNOTATED:
      return
    used = set()
    for entry in partition:
      # An entry is a single axis name, a tuple of them, or None.
      used.update((entry,) if isinstance(entry, str) else (entry or ()))
    if axis in used:
      raise ValueError(
          f'expert_parallel_axis={axis!r} is also an activation axis'
          f' ({partition}); the expert-parallel grouped matmul assumes the'
          ' rows are replicated over it. Use the decoding sharding config'
          ' (`kimi_k3_decoding_sharding`), which leaves the token axis'
          ' unsharded.'
      )

  def _expert_weights(self, params: Any) -> tuple[Array, Array, Array]:
    """The three `[E, in, out]` expert stacks, dequantized if needed."""
    read = lambda name: common.convert_or_dequantize(
        params[name]['w'], dtype=self.activation_dtype
    )
    return read('ffn_0_gate'), read('ffn_0'), read('ffn_1')

  def _expert_ffn(self, gate: Array, up: Array) -> Array:
    return _situ_glu(
        gate, up, beta=self.situ_beta, linear_beta=self.situ_linear_beta
    )

  def _grouped_experts(
      self,
      params: Any,
      x: Array,
      indices: Array,
      weights: Array,
      group_sizes: Array,
  ) -> Array:
    """Grouped-matmul dispatch, the production path.

    The `[N, K]` routing table is flattened into `N * K` (token, slot) pairs
    and stably sorted by expert id, which makes the rows of one expert a
    contiguous run; `group_sizes[e]` (a bincount, so it needs no sort) gives
    the run lengths, and the three projections become three grouped matmuls.
    The unsort is the inverse permutation, exact for any sort order; stability
    is what makes each run's row order deterministic and equal to HF
    `moe_infer`'s `argsort` dispatch. Experts with zero tokens are empty runs
    -- correct with no special case.

    `sum(group_sizes)` equals the row count unless padded tokens were routed
    to the phantom expert, in which case their rows are the tail of the sorted
    buffer, which `model_lib.gmm` zeroes.

    Args:
      params: `experts/{ffn_0_gate,ffn_0,ffn_1}`.
      x: `[N, L]` latent activations.
      indices: `[N, K]` int32 expert ids.
      weights: `[N, K]` f32 combine weights.
      group_sizes: `[E]` int32 tokens per expert.

    Returns:
      `[N, L]` f32 combined expert outputs.
    """
    w_gate, w_up, w_down = self._expert_weights(params)
    num_tokens, latent_dim = x.shape
    top_k = self.num_experts_per_token
    sort_indices = jnp.argsort(jnp.ravel(indices), stable=True)
    # Row i of the sorted batch is token `sort_indices[i] // K`, gathered
    # rather than materialized by a K-fold repeat.
    rows = jnp.take(x, sort_indices // top_k, axis=0)
    rows = jnp.asarray(rows, self.activation_dtype)

    tiling = (
        common.round_up_to_base(
            min(self.tile_batch_seq, rows.shape[0]), base=8, threshold=8
        ),
        common.round_up_to_base(
            min(self.tile_latent_dim, latent_dim), base=128, threshold=128
        ),
        common.round_up_to_base(
            min(self.tile_expand_dim, self.moe_intermediate_size),
            base=128,
            threshold=128,
        ),
    )
    if self._expert_parallel_shards() > 1:
      out = self._expert_parallel_matmuls(
          rows, group_sizes, (w_gate, w_up, w_down), tiling
      )
    else:
      out = self._expert_matmuls(
          rows, group_sizes, (w_gate, w_up, w_down), tiling
      )

    out = model_lib.permute(out, jnp.argsort(sort_indices))
    out = jnp.reshape(out, (num_tokens, top_k, latent_dim))
    return jnp.einsum('nk,nkl->nl', weights, jnp.asarray(out, jnp.float32))

  def _dense_grouped_matmul(
      self, lhs: Array, rhs: Array, group_sizes: Array
  ) -> Array:
    """`model_lib.gmm` in pure einsum: same value, no kernel, no custom call.

    A bisect instrument, not a production path -- it runs every expert over
    every row and masks, so it costs `num_local_experts` times the flops. It
    exists because every `model_lib.gmm` backend is a TPU custom call
    (`megablox` and `gmm_v2` are Pallas, `ragged_dot` lowers to Mosaic) and
    `gmm_impl='megablox'` cannot even be swapped in as a control ("Mosaic
    kernels cannot be automatically partitioned", which fires whether or not
    the expert-parallel path is on), leaving no way to ask on hardware whether
    a defect is in the kernel or in the surrounding algebra. It is what
    localized the expert-parallel NaN gradient to `ragged_dot`'s
    transpose, and the residue there -- finite gradients but
    the loss NaN by step 10 -- is still open, so keep it selectable.

    Args:
      lhs: `[rows, in]` sorted expert inputs.
      rhs: `[experts, in, out]` weights.
      group_sizes: `[experts]` rows per expert, in the order `lhs` is sorted.

    Returns:
      `[rows, out]`, with rows past `sum(group_sizes)` zero.
    """
    boundaries = jnp.cumsum(group_sizes)
    row = jax.lax.broadcasted_iota(jnp.int32, (lhs.shape[0],), 0)
    # Rows are sorted by expert, so an expert owns a contiguous window.
    expert_of_row = jnp.sum(row[:, None] >= boundaries[None, :], axis=1)
    out = jnp.zeros(
        (lhs.shape[0], rhs.shape[-1]), dtype=jnp.dtype(self.activation_dtype)
    )
    for expert in range(rhs.shape[0]):
      out += jnp.where(
          (expert_of_row == expert)[:, None],
          jnp.einsum('mk,ko->mo', lhs, rhs[expert]),
          0,
      )
    return out

  def _expert_matmuls(
      self,
      rows: Array,
      group_sizes: Array,
      expert_weights: tuple[Array, Array, Array],
      tiling: tuple[int, int, int],
  ) -> Array:
    """The three grouped matmuls of the routed experts, on sorted rows."""
    w_gate, w_up, w_down = expert_weights
    if self.gmm_impl == 'dense_grouped':
      grouped = lambda lhs, rhs, tile: self._dense_grouped_matmul(
          lhs, rhs, group_sizes
      )
    else:
      grouped = lambda lhs, rhs, tile: model_lib.gmm(
          lhs,
          rhs,
          group_sizes,
          tiling=tile,
          gmm_impl=self.gmm_impl,
          activation_dtype=self.activation_dtype,
      )
    hidden = self._expert_ffn(
        grouped(rows, w_gate, tiling), grouped(rows, w_up, tiling)
    )
    return grouped(hidden, w_down, (tiling[0], tiling[2], tiling[1]))

  def _expert_parallel_shards(self) -> int:
    """Size of `expert_parallel_axis` in this mesh; 1 if not applicable."""
    if not self.expert_parallel_axis:
      return 1
    mesh = jax.sharding.get_abstract_mesh()
    if self.expert_parallel_axis not in mesh.axis_names:
      return 1
    return mesh.shape[self.expert_parallel_axis]

  def _expert_parallel_matmuls(
      self,
      rows: Array,
      group_sizes: Array,
      expert_weights: tuple[Array, Array, Array],
      tiling: tuple[int, int, int],
  ) -> Array:
    """`_expert_matmuls` with each shard running only the experts it owns.

    `jax.lax.ragged_dot`'s group dimension needs replication in Shardy's
    sharding rule (`RaggedDotShardingRuleOpInterface` in
    `xla/service/spmd/shardy/extensions/mhlo_extensions.cc`), so GSPMD
    **all-gathers the whole expert stack before every grouped matmul**. In the
    2.8T decode program that is `all-gather(bf16[112, 448, 768]) ->
    bf16[896, 448, 768]` three times in each of 92 MoE layers: 159 GiB of
    all-gather output per device per step, against a 20.2 GiB weight budget,
    and independent of the batch -- which is exactly the measured signature of
    `t_step` being flat in batch size.

    Under `shard_map` no exchange is needed at all, as long as the rows are
    replicated over the expert axis (`kimi_k3_decoding_sharding` does not shard
    the token axis, which `setup` checks): every shard evaluates the experts it
    owns on every row, contributes zeros for the rows it does not own, and a
    `psum` over the expert axis reassembles the answer. The payload is the
    `[rows, latent]` activation instead of the weights.

    Inside the body `ragged_dot` sees only the local groups, whose rows are a
    contiguous window of the globally sorted buffer starting at
    `starts[shard]`; rolling the rows by `-start` puts that window where group
    0 is expected, and the output is rolled back before the psum. Both rolls
    move activations, not weights.

    Args:
      rows: `[N * K, L]` sorted expert inputs, replicated over the expert axis.
      group_sizes: `[E]` int32 rows per expert.
      expert_weights: the three `[E, in, out]` stacks.
      tiling: grouped-matmul tiling.

    Returns:
      `[N * K, L]` expert outputs, identical to `_expert_matmuls`.
    """
    axis = cast(str, self.expert_parallel_axis)
    shards = self._expert_parallel_shards()
    if self.num_experts % shards:
      raise ValueError(
          f'{self.num_experts} experts do not divide over {shards} shards of'
          f' mesh axis {axis!r}.'
      )
    local_experts = self.num_experts // shards
    # First sorted row of each shard's block of experts. Passed in sharded over
    # the expert axis so the body reads its own offset: `jax.lax.axis_index`
    # inside a partially manual `shard_map` lowers to a `partition-id`, which
    # the SPMD partitioner rejects.
    starts = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes)]
    )[::local_experts][:shards]

    def local(rows, group_sizes, starts, w_gate, w_up, w_down):
      offset = starts[0]
      out = self._expert_matmuls(
          jnp.roll(rows, -offset, axis=0),
          group_sizes,
          (w_gate, w_up, w_down),
          tiling,
      )
      # Rows past this shard's groups belong to another shard; `ragged_dot`
      # promises nothing about them and the psum would add whatever it left.
      row = jax.lax.broadcasted_iota(jnp.int32, out.shape, 0)
      out = jnp.where(row < jnp.sum(group_sizes), out, 0)
      # Each row is owned by exactly one shard, so this psum adds one value to
      # `shards - 1` exact zeros: it is exact in any dtype, and an xprof trace
      # of the 2.8T decode step measured it at 6.2% of device time in f32
      # (`all-reduce f32[3200,448]`), which halves in the activation dtype.
      # XLA:CPU aborts on a bf16 all-reduce (it cannot clone the op), and the
      # multi-device equivalence test runs on CPU, so that backend keeps f32.
      reduce_dtype = (
          jnp.float32 if jax.default_backend() == 'cpu' else out.dtype
      )
      return jax.lax.psum(
          jnp.roll(jnp.asarray(out, reduce_dtype), offset, axis=0), axis
      )

    spec = jax.sharding.PartitionSpec
    return jax.shard_map(
        local,
        axis_names={axis},
        in_specs=(
            spec(),
            spec(axis),
            spec(axis),
            spec(axis),
            spec(axis),
            spec(axis),
        ),
        out_specs=spec(),
    )(rows, group_sizes, starts, *expert_weights)

  def _dense_experts(
      self, params: Any, x: Array, indices: Array, weights: Array
  ) -> Array:
    """Every expert on every token; a one-hot combine keeps the top-k.

    `E / K` times the flops of `_grouped_experts` and no gather at all, which
    makes it the reference path: it shares only the weights and the
    activation with the grouped path, so an A/B test covers the whole
    sort/dispatch/unsort machinery. It is not NaN-equivalent to it -- an
    unselected expert's output is killed by a `0 *`, not by never running --
    which SiTU's caps make unreachable in practice.

    Args:
      params: `experts/{ffn_0_gate,ffn_0,ffn_1}`.
      x: `[N, L]` latent activations.
      indices: `[N, K]` int32 expert ids.
      weights: `[N, K]` f32 combine weights.

    Returns:
      `[N, L]` f32 combined expert outputs.
    """
    w_gate, w_up, w_down = self._expert_weights(params)
    x = jnp.asarray(x, self.activation_dtype)
    hidden = self._expert_ffn(
        jnp.einsum('nl,elm->enm', x, w_gate),
        jnp.einsum('nl,elm->enm', x, w_up),
    )
    expert_out = jnp.einsum('enm,eml->enl', hidden, w_down)
    combine = jnp.einsum(
        'nk,nke->ne',
        weights,
        jax.nn.one_hot(indices, self.num_experts, dtype=jnp.float32),
    )
    return jnp.einsum(
        'ne,enl->nl', combine, jnp.asarray(expert_out, jnp.float32)
    )

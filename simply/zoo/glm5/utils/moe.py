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
"""GLM-5.2 (`glm_moe_dsa`) Mixture-of-Experts feed-forward.

`GlmMoeFeedForward` is a thin subclass of the core `MoEFeedForward`. It plugs in
DeepSeek-V3 / GLM `noaux_tc` routing by overriding the single generic `_gate`
seam (sigmoid scores + a selection-only per-expert correction bias, optional
top-k renormalization and gate scaling), and adds an always-on shared expert by
wrapping `apply` (compute the routed output via `super().apply`, then add the
shared FFN). `setup`/`init` extend the base via `super()`. All sparse-dispatch,
load/entropy metrics and lbl/z-loss bookkeeping are inherited unchanged.
"""

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp

from simply import model_lib
from simply.utils import common
from simply.utils import module
from simply.utils import sharding as sharding_lib

Array = common.Array
PyTree = common.PyTree
PRNGKey = jax.typing.ArrayLike
PartitionAnnotation = common.PartitionAnnotation

FeedForward = model_lib.FeedForward
MoEFeedForward = model_lib.MoEFeedForward


def dense_ffn_partition(
    partition: PartitionAnnotation,
) -> PartitionAnnotation:
  """Adapts a (possibly 3D MoE) FFN weight partition to the 2D dense-FFN rank.

  MoE sharding configs annotate the 3D expert weight ``[experts, in, out]``; a
  dense `FeedForward` weight is 2D ``[in, out]``, so drop the leading (expert)
  axis when a 3D partition is supplied. Lets GLM's dense FFNs (the shared expert
  and the `first_k_dense_replace` layers) reuse the MoE sharding config. Kept
  plugin-local so core needs no change.

  Args:
    partition: An FFN weight partition (2D dense, or 3D from a MoE config).

  Returns:
    The partition adapted to the 2D dense-FFN weight rank.
  """
  if partition is not None and len(partition) == 3:
    return tuple(partition[1:])
  return partition


def dense_ffn_sharding(sharding_config):
  """Returns `sharding_config` with its FFN weight partitions squeezed to 2D."""
  return dataclasses.replace(
      sharding_config,
      ffn0_partition=dense_ffn_partition(sharding_config.ffn0_partition),
      ffn1_partition=dense_ffn_partition(sharding_config.ffn1_partition),
  )


@module.ModuleRegistry.register
@dataclasses.dataclass
class GlmMoeFeedForward(MoEFeedForward):
  """MoE with GLM/DeepSeek-V3 sigmoid (`noaux_tc`) routing + shared expert."""

  # Routing variant. 'softmax' (base behavior) or 'sigmoid' (DeepSeek-V3 / GLM
  # `noaux_tc`): scores = sigmoid(logits); an additive per-expert correction
  # bias selects the top-k experts, the gate weights are gathered from the
  # *un-biased* sigmoid scores, optionally renormalized, then multiplied by
  # `routed_scaling_factor`.
  router_score_func: Literal['softmax', 'sigmoid'] = 'sigmoid'
  # Whether the router has a learned per-expert correction bias used only for
  # expert selection (DeepSeek-V3 `e_score_correction_bias`).
  router_use_correction_bias: bool = True
  # Renormalize the top-k gate weights to sum to 1 (DeepSeek `norm_topk_prob`).
  norm_topk_prob: bool = True
  # Scaling on the gate weights (DeepSeek `routed_scaling_factor`).
  routed_scaling_factor: float = 1.0
  # Number of always-on shared experts (DeepSeek `n_shared_experts`). When > 0
  # an additional dense FFN (with expand dim `expand_dim * num_shared_experts`)
  # is applied to every token and added to the routed output.
  num_shared_experts: int = 1

  def setup(self) -> None:
    super().setup()
    if self.num_shared_experts > 0:
      self.shared_expert = FeedForward(
          model_dim=self.model_dim,
          expand_factor=self.expand_factor,
          # Shared expert is a dense FFN: squeeze the 3D MoE partitions to 2D.
          sharding_config=dense_ffn_sharding(self.sharding_config),
          use_gated_activation_in_ffn=self.use_gated_activation_in_ffn,
          activation_dtype=self.activation_dtype,
          ffn_expand_dim=self.expand_dim * self.num_shared_experts,
          ffn_use_bias=False,
          ffn_activation=self.ffn_activation,
          ffn_weight_init=self.ffn_weight_init,
      )

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = dict(super().init(prng_key))  # pyrefly: ignore[no-matching-overload]
    if self.router_use_correction_bias:
      params['router_correction_bias'] = jnp.zeros(
          self.num_experts, dtype=jnp.float32
      )
    if self.num_shared_experts > 0:
      # Deterministic sub-key so shared-expert init is independent of the base
      # split and stable across runs.
      shared_key = jax.random.fold_in(prng_key, 1)
      params['shared_expert'] = self.shared_expert.init(shared_key)
    return params

  def _gate(
      self, router_logits: Array, params: PyTree
  ) -> tuple[Array, Array, Array]:
    """Sigmoid / `noaux_tc` routing (see class docstring); softmax falls back."""
    if self.router_score_func != 'sigmoid':
      return super()._gate(router_logits, params)
    # router_probs (full, for metrics): sigmoid scores.
    router_probs = jax.nn.sigmoid(router_logits)
    selection_scores = router_probs
    if self.router_use_correction_bias:
      # Bias is used ONLY to select the experts, not to weight them.
      selection_scores = router_probs + params['router_correction_bias']  # pyrefly: ignore[bad-index, unsupported-operation]
    _, selected_indices = jax.lax.top_k(
        selection_scores, k=self.num_experts_per_token
    )
    # Gate weights are gathered from the UN-biased sigmoid scores.
    selected_router_probs = jnp.take_along_axis(
        router_probs, selected_indices, axis=-1
    )
    if self.norm_topk_prob:
      selected_router_probs = selected_router_probs / (
          jnp.sum(selected_router_probs, axis=-1, keepdims=True) + 1e-20
      )
    selected_router_probs = selected_router_probs * self.routed_scaling_factor
    return selected_router_probs, selected_indices, router_probs

  def apply(  # pyrefly: ignore[bad-override]
      self,
      params: PyTree,
      x: Array,
      inputs_mask: Array | None = None,
  ) -> PyTree:
    # Routed experts (with GLM `_gate`) + all metrics/losses via the base.
    outputs, extra_output = super().apply(params, x, inputs_mask=inputs_mask)  # pyrefly: ignore[not-iterable]
    if self.num_shared_experts <= 0:
      return outputs, extra_output
    # Always-on shared expert applied to every token, added to the routed
    # output. `super().apply` already zeroed padded positions; re-mask after the
    # add so padded positions stay zero (the shared FFN does not mask), matching
    # a single combined output masking.
    params = model_lib.get_raw_arrays(params)
    shared_out, _ = self.shared_expert.apply(
        params['shared_expert'], x, inputs_mask=inputs_mask  # pyrefly: ignore[bad-index, unsupported-operation]
    )
    outputs = outputs + jnp.asarray(shared_out, outputs.dtype)  # pyrefly: ignore[missing-attribute, unsupported-operation]
    if inputs_mask is not None:
      outputs = jnp.where(inputs_mask[..., None], outputs, 0.0)
    outputs = sharding_lib.with_sharding_constraint(
        outputs, self.sharding_config.activation_partition  # pyrefly: ignore[bad-argument-type]
    )
    return outputs, extra_output

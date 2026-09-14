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
"""Block Attention Residuals (AttnRes) for Kimi K3.

Instead of accumulating one residual stream over depth, each sublayer reads a
softmax mixture over {block snapshots of the stream, the current partial sum}
with a learned per-site pseudo-query (report S2.2; `_apply_attn_res` in
modeling_kimi_linear.py).

The mixture is strictly per token -- no cross-token dependency -- so chunked
prefill and incremental decode are exact.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from simply.utils import common
from simply.utils import module
from simply.utils import sharding as sharding_lib

Array = common.Array
PyTree = common.PyTree
PRNGKey = jax.typing.ArrayLike
AnnotatedArray = common.AnnotatedArray


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AttnResState:
  """Snapshots of the residual stream, and how many slots are filled.

  Attributes:
    snapshots: [batch, seq_len, num_slots, model_dim] block representations,
      written in order; slots >= `num_valid` hold stale values and are masked
      out of the mixture.
    num_valid: scalar int32, the number of snapshots pushed so far.
  """

  snapshots: Array
  num_valid: Array


def init_attn_res_state(
    batch_size: int,
    seq_len: int,
    model_dim: int,
    num_slots: int,
    dtype: jax.typing.DTypeLike = 'float32',
    partition: common.PartitionAnnotation = None,
) -> AttnResState:
  """Allocates the snapshot buffer; `partition` annotates `[B, T, slots, D]`."""
  snapshots = jnp.zeros(
      (batch_size, seq_len, num_slots, model_dim), dtype=dtype
  )
  return AttnResState(
      snapshots=sharding_lib.with_sharding_constraint(snapshots, partition),
      num_valid=jnp.zeros((), dtype=jnp.int32),
  )


def push_snapshot(state: AttnResState, x: Array) -> AttnResState:
  """Appends the current residual stream as the next block representation."""
  snapshot = jnp.asarray(x, state.snapshots.dtype)[:, :, None, :]
  snapshots = jax.lax.dynamic_update_slice_in_dim(
      state.snapshots, snapshot, state.num_valid, axis=2
  )
  return AttnResState(snapshots=snapshots, num_valid=state.num_valid + 1)


def maybe_push_snapshot(
    state: AttnResState, x: Array, push: bool | Array
) -> tuple[AttnResState, Array]:
  """Pushes `x` and restarts the partial sum, under a static or traced `push`.

  A scanned layer group does not know at trace time whether its layers are
  snapshot layers -- that depends on the iteration -- so the push has to be
  expressible as a select. With a Python bool this is exactly the unrolled
  code; with a traced predicate it rewrites the slot it would have written,
  which is a no-op when `push` is false.

  Args:
    state: the snapshot buffer.
    x: the residual partial sum since the last snapshot.
    push: whether this layer is a snapshot layer.

  Returns:
    The new state and the partial sum to carry on with (zero after a push).
  """
  if isinstance(push, bool):
    if not push:
      return state, x
    return push_snapshot(state, x), jnp.zeros_like(x)
  snapshot = jnp.asarray(x, state.snapshots.dtype)[:, :, None, :]
  current = jax.lax.dynamic_slice_in_dim(
      state.snapshots, state.num_valid, 1, axis=2
  )
  state = AttnResState(
      snapshots=jax.lax.dynamic_update_slice_in_dim(
          state.snapshots,
          jnp.where(push, snapshot, current),
          state.num_valid,
          axis=2,
      ),
      num_valid=state.num_valid + push.astype(state.num_valid.dtype),
  )
  return state, jnp.where(push, jnp.zeros_like(x), x)


@module.ModuleRegistry.register
@dataclasses.dataclass
class KimiK3AttnResMix(module.SimplyModule):
  """One AttnRes read site: softmax over snapshots plus the partial sum.

  Scores are `sum_d(RMSNorm(v)_d * scale_d * w_d)` over the candidate vectors
  `v` (RMSNorm without gain, then the site's gain and pseudo-query folded into
  one vector), and the output is the probability-weighted mean of the
  *unnormalized* candidates. Everything is computed in f32; the result is cast
  back to the dtype of the inputs.
  """

  dim: int
  epsilon: float = 1e-5
  weight_dtype: jax.typing.DTypeLike = 'float32'
  activation_dtype: jax.typing.DTypeLike = 'bfloat16'
  weight_partition: common.PartitionAnnotation = None

  def init(self, prng_key: PRNGKey | None = None) -> PyTree:
    del prng_key
    # A zero pseudo-query mixes the candidates uniformly, which is the
    # identity-like starting point for a from-scratch init; real runs load the
    # released weights.
    params = {
        'scale': jnp.ones(self.dim, dtype=self.weight_dtype),
        'w': jnp.zeros(self.dim, dtype=self.weight_dtype),
    }
    return {
        k: AnnotatedArray.create(
            sharding_lib.with_sharding_constraint(v, self.weight_partition),
            dim_annotation='h',
        )
        for k, v in params.items()
    }

  def apply(  # pyrefly: ignore[bad-override]
      self, params: PyTree, x: Array, state: AttnResState
  ) -> Array:
    """Mixes `x` (the current partial sum) with the snapshots in `state`."""
    p: Any = common.get_raw_arrays(params)
    scale = common.convert_or_dequantize(p['scale'], dtype=jnp.float32)
    query = common.convert_or_dequantize(p['w'], dtype=jnp.float32)
    pseudo_query = scale * query

    # RMSNorm followed by a dot with the pseudo-query is
    # `(v . q) * rsqrt(mean(v^2) + eps)`, which avoids materializing an f32
    # copy of the snapshot buffer -- the largest activation in a K3 prefill.
    def score(v: Array) -> Array:
      v32 = jnp.asarray(v, jnp.float32)
      inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(v32), axis=-1) + self.epsilon)
      return jnp.einsum('...d,d->...', v32, pseudo_query) * inv_rms

    num_slots = state.snapshots.shape[2]
    scores = jnp.concatenate(
        [score(state.snapshots), score(x)[:, :, None]], axis=2
    )
    # The partial sum in the last position is always a candidate; snapshot
    # slots only once they have been written.
    slot = jnp.arange(num_slots + 1)
    valid = (slot < state.num_valid) | (slot == num_slots)
    scores = jnp.where(valid, scores, common.neg_inf(scores.dtype))

    probs = jax.nn.softmax(scores, axis=-1)
    mixed = jnp.einsum(
        'bts,btsd->btd',
        probs[:, :, :num_slots],
        state.snapshots,
        preferred_element_type=jnp.float32,
    ) + probs[:, :, num_slots, None] * jnp.asarray(x, jnp.float32)
    return jnp.asarray(mixed, x.dtype)

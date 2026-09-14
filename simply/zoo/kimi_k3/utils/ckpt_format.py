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
"""Checkpoint format for Kimi K3: MXFP4 experts are decoded on restore.

The released routed experts are MXFP4 (`weight_packed` nibbles + E8M0
`weight_scale`), which is 1.45 TiB instead of the 5.06 TiB the same weights
occupy dense -- worth keeping on disk. They are decoded once here, at restore,
so the model only ever sees dense arrays and no dequantization lands in the
per-step graph.
"""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.zoo.kimi_k3.utils import hf_params

PyTree = Any

# `hf_params.MXFP4_LUT` stays a NumPy constant: a library module must not call
# into JAX at import time.
_E2M1_MAGNITUDE = hf_params.MXFP4_LUT


def dequantize_mxfp4(
    packed: jax.Array, scale: jax.Array, dtype: Any = jnp.bfloat16
) -> jax.Array:
  """Decodes packed MXFP4 `[..., out, in // 2]` to dense `[..., in, out]`.

  The packed form keeps the HuggingFace `[out, in]` orientation because nibbles
  cannot be transposed; the transpose to Simply's `[in, out]` happens here.

  `hf_params.dequantize_mxfp4` is the numpy twin, used by the offline converter
  and the golden fixtures; `ckpt_format_test` pins the two to each other. They
  stay separate so that the parameter map keeps its numpy-only dependencies,
  and this one additionally exploits JAX (bitcast scales, broadcast groups).

  Every step is elementwise and stays in `dtype`, so under `jit` the whole chain
  fuses and peak memory is the output. That matters: one released expert stack
  is 19.7 GiB dense, and an intermediate in f32 -- or a `jnp.repeat` of the
  scales to full width -- costs 39.5 GiB each.

  Args:
    packed: u8 nibble pairs; the low nibble is the even input column.
    scale: u8 E8M0 exponents, one per 32 input columns.
    dtype: output dtype. The decoded values are multiples of 0.5 and the scales
      are powers of two, so bf16 is exact for the range the release uses.

  Returns:
    The dense weights `[..., in, out]`, in Simply's orientation.
  """
  nibbles = jnp.stack([packed & 0x0F, packed >> 4], axis=-1)
  nibbles = nibbles.reshape(*packed.shape[:-1], -1)
  magnitude = jnp.asarray(_E2M1_MAGNITUDE, dtype)
  sign = jnp.where(nibbles & 0x08, -1, 1).astype(dtype)
  values = sign * magnitude[nibbles & 0x07]
  # An E8M0 code IS a float32 exponent field, so building the power of two by
  # bit pattern is exact -- `exp2` on a float is not.
  factor = jax.lax.bitcast_convert_type(
      scale.astype(jnp.uint32) << 23, jnp.float32
  ).astype(dtype)
  # Broadcast the per-group scale instead of materializing it at full width.
  grouped = values.reshape(*scale.shape, -1) * factor[..., None]
  return jnp.swapaxes(grouped.reshape(values.shape), -1, -2)


def _target_sharding(target: PyTree) -> jax.sharding.Sharding | None:
  """The sharding of the dense `w` leaf this packed node restores into."""
  if not isinstance(target, dict) or 'w' not in target:
    return None
  raw: Any = common.get_raw_arrays(target)
  return getattr(raw['w'], 'sharding', None)


def dequantize_tree(
    tree: PyTree, dtype: Any = jnp.bfloat16, target: PyTree = None
) -> PyTree:
  """Replaces every MXFP4 node (`hf_params.is_mxfp4_node`) with a `{'w': ...}`.

  Args:
    tree: the restored checkpoint tree.
    dtype: dtype of the decoded weights.
    target: the abstract target tree, if known. Its leaf shardings are used as
      `out_shardings`, which both places the result where it belongs and lets
      XLA free the packed inputs -- the restore runs before Simply's own
      resharding, so without this the decode happens on Orbax's arbitrary
      initial sharding and the transients OOM a 95 GiB device.

  Returns:
    A tree of the same shape with the packed nodes decoded; every other node is
    returned as is.
  """
  if hf_params.is_mxfp4_node(tree):
    # Not donated: the caller's tree still references the packed arrays, and
    # they are only 1/4 of the dense result anyway.
    decode = jax.jit(
        functools.partial(dequantize_mxfp4, dtype=dtype),
        out_shardings=_target_sharding(target),
    )
    return {
        'w': decode(
            tree[hf_params.MXFP4_PACKED_KEY], tree[hf_params.MXFP4_SCALE_KEY]
        )
    }
  if isinstance(tree, dict):
    sub = lambda k: target.get(k) if isinstance(target, dict) else None
    return {k: dequantize_tree(v, dtype, sub(k)) for k, v in tree.items()}
  return tree


@ckpt_lib.CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class KimiK3Format(ckpt_lib.V2Format):
  """Simply-native Kimi K3 tree; decodes MXFP4 expert weights on restore.

  The other HF ports store the raw HF-named tensors and do the whole key/shape
  mapping here, at restore (`ckpt_lib.Qwen2Format`). K3 cannot: the release is
  1.56 TB over 497k tensors, so the mapping would re-run -- and re-materialize
  5.06 TiB of dense experts -- on every job start, over a stored tree with 497k
  Orbax leaves. `utils/hf_convert.py` therefore maps offline into a
  Simply-native tree, and this format does only the thing that needs the target:
  the MXFP4 decode onto the target sharding.
  `V2Format` is core's Simply-native, key-identity format, which is what such a
  tree wants.

  Every field needs a default: `write_checkpoint` bakes an instance built with
  no arguments into the checkpoint metadata, and both ways of getting the
  format back (the metadata, or `config.init_ckpt_format` through
  `registry.get_instance`) construct it the same way.
  """

  expert_dtype: str = 'bfloat16'

  def transforms(
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    state = dequantize_tree(
        stored_state, jnp.dtype(self.expert_dtype), target_abstract_state
    )
    return super().transforms(state, target_abstract_state)

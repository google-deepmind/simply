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
"""Mapping the HuggingFace Kimi K3 release onto the Simply parameter tree.

The HF text backbone stores every projection as a torch `nn.Linear` weight
`[out, in]`, while Simply stores `[in, ...out]` and keeps the head axis
explicit. `KimiK3HfConverter` walks the *target* tree and pulls the HF tensors
it needs, so the same code serves both the in-memory golden test and the
streaming converter for the 1.45 TiB release (whose routed experts are read one
layer at a time).

Routed expert weights ship as MXFP4 (`weight_packed` u8 nibbles +
`weight_scale` E8M0 exponents); `dequantize_mxfp4` reproduces them exactly.
This module owns that node's contract -- the key names, the packing density and
the group size are the constants below -- and `ckpt_format.dequantize_tree` is
the consumer, at restore.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

PyTree = Any
TensorFn = Callable[[str], np.ndarray]

# Simply expert-weight name -> the HF matrix it comes from.
EXPERT_PROJECTIONS = {'ffn_0_gate': 'w1', 'ffn_0': 'w3', 'ffn_1': 'w2'}

# OCP MX FP4 (E2M1) code -> value; the sign is bit 3.
MXFP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# An undecoded routed expert stack is this pair of arrays instead of a dense
# `{'w': ...}` leaf: `MXFP4_CODES_PER_BYTE` 4-bit codes per `packed` byte, one
# E8M0 exponent per `MXFP4_GROUP_SIZE` values along the contraction axis.
MXFP4_PACKED_KEY = 'packed'
MXFP4_SCALE_KEY = 'scale'
MXFP4_NODE_KEYS = frozenset({MXFP4_PACKED_KEY, MXFP4_SCALE_KEY})
MXFP4_CODES_PER_BYTE = 2
MXFP4_GROUP_SIZE = 32

# The multimodal release nests the text backbone under this key prefix; the
# text-only reference does not.
LANGUAGE_MODEL_PREFIX = 'language_model.'


def is_mxfp4_node(node: Any) -> bool:
  """Whether `node` is an undecoded MXFP4 pair rather than a dense leaf."""
  return isinstance(node, Mapping) and node.keys() == MXFP4_NODE_KEYS


def detect_prefix(names: Iterable[str]) -> str:
  """Returns the HF key prefix `names` are stored under."""
  return (
      LANGUAGE_MODEL_PREFIX
      if any(name.startswith(LANGUAGE_MODEL_PREFIX) for name in names)
      else ''
  )


def dequantize_mxfp4(
    packed: np.ndarray, scale: np.ndarray, dtype: Any = np.float32
) -> np.ndarray:
  """Decodes `mxfp4-pack-quantized` weights to dense values.

  `ckpt_format.dequantize_mxfp4` is the JAX twin of this codec, for the decode
  that happens on restore; the two are pinned to each other by
  `ckpt_format_test`. They stay separate because this module is numpy-only, so
  that the offline converter and the golden fixtures can import the parameter
  map without pulling in JAX.

  Args:
    packed: u8 `[..., out, in // 2]`; the LOW nibble is the even input column.
    scale: u8 `[..., out, in // 32]` E8M0 exponents; the factor is `2 ** (scale
      - 127)`.
    dtype: output dtype.

  Returns:
    `[..., out, in]` dequantized weights.
  """
  low = packed & 0x0F
  high = packed >> 4
  nibbles = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
  values = MXFP4_LUT[nibbles & 0x07] * np.where(nibbles & 0x08, -1.0, 1.0)
  group_size = values.shape[-1] // scale.shape[-1]
  factor = np.exp2(scale.astype(np.float32) - 127.0)
  factor = np.repeat(factor, group_size, axis=-1)
  return (values * factor).astype(dtype)


class KimiK3HfConverter:
  """Builds the Simply parameter tree from HF tensors, pulled by name.

  Attributes:
    config: a `config_lib.KimiK3ExperimentConfig`.
    prefix: HF key prefix, as `detect_prefix` reads it off a release.
    dequantize_experts: whether to decode MXFP4 expert weights (`True`) or keep
      the packed pair `{'packed', 'scale'}` in the tree (`False`).
    dtype: dtype of the emitted non-expert arrays; `None` preserves the source
      dtype, which is what the release intends (bf16 matrices, f32 for A_log,
      dt_bias, conv kernels, o_norm and the router bias).
    expert_dtype: dtype of the dequantized routed expert weights; defaults to
      `dtype`.
    layer_types: the token mixer of each layer (`'linear_attention'` for KDA,
      `'full_attention'` for MLA), resolved from `config`.
  """

  def __init__(
      self,
      config: Any,
      prefix: str = '',
      dequantize_experts: bool = True,
      dtype: Any = np.float32,
      expert_dtype: Any = None,
  ):
    self.config = config
    self.prefix = prefix
    self.dequantize_experts = dequantize_experts
    self.dtype = dtype
    self.expert_dtype = expert_dtype or dtype or np.float32
    self.layer_types = config.resolved_layer_types()

  def _cast(self, x: np.ndarray) -> np.ndarray:
    return x if self.dtype is None else x.astype(self.dtype)

  # --- helpers ---------------------------------------------------------------

  def _linear(self, get: TensorFn, name: str, *out_dims: int) -> np.ndarray:
    """Reads an `nn.Linear` weight `[out, in]` as `[in, *out_dims]`."""
    w = self._cast(np.asarray(get(f'{self.prefix}{name}')).T)
    if out_dims:
      w = w.reshape(w.shape[0], *out_dims)
    return w

  def _vector(self, get: TensorFn, name: str, *shape: int) -> np.ndarray:
    v = self._cast(np.asarray(get(f'{self.prefix}{name}')))
    return v.reshape(*shape) if shape else v

  def _attn_res(self, get: TensorFn, norm: str, proj: str) -> PyTree:
    return {
        'scale': self._vector(get, f'{norm}.weight'),
        # The pseudo-query is an `nn.Linear(dim, 1)`, i.e. `[1, dim]`.
        'w': self._vector(get, f'{proj}.weight').reshape(-1),
    }

  # --- per-module conversion -------------------------------------------------

  def convert_kda(self, get: TensorFn, layer: int) -> PyTree:
    cfg = self.config
    h, k = cfg.kda_num_heads, cfg.kda_head_dim
    p = f'model.layers.{layer}.self_attn'
    conv = lambda name: self._cast(  # [H*K, 1, W] -> [H, K, W]
        np.asarray(get(f'{self.prefix}{p}.{name}.weight'))
    ).reshape(h, k, -1)
    return {
        'q_proj': {'w': self._linear(get, f'{p}.q_proj.weight', h, k)},
        'k_proj': {'w': self._linear(get, f'{p}.k_proj.weight', h, k)},
        'v_proj': {'w': self._linear(get, f'{p}.v_proj.weight', h, k)},
        'q_conv': {'w': conv('q_conv1d')},
        'k_conv': {'w': conv('k_conv1d')},
        'v_conv': {'w': conv('v_conv1d')},
        'f_a_proj': {'w': self._linear(get, f'{p}.f_a_proj.weight')},
        'f_b_proj': {'w': self._linear(get, f'{p}.f_b_proj.weight', h, k)},
        'dt_bias': self._vector(get, f'{p}.dt_bias', h, k),
        # A_log ships as [128] with the tail zero-padded; only the first
        # `num_heads` entries are trained (K3_ARCHITECTURE.md S4).
        'a_log': self._vector(get, f'{p}.A_log')[:h],
        'b_proj': {'w': self._linear(get, f'{p}.b_proj.weight')},
        'g_proj': {'w': self._linear(get, f'{p}.g_proj.weight', h, k)},
        'o_norm': {'scale': self._vector(get, f'{p}.o_norm.weight')},
        'o_proj': {
            'w': self._linear(get, f'{p}.o_proj.weight').reshape(h, k, -1)
        },
    }

  def convert_mla(self, get: TensorFn, layer: int) -> PyTree:
    cfg = self.config
    h = cfg.n_heads
    p = f'model.layers.{layer}.self_attn'
    return {
        'q_a_proj': {'w': self._linear(get, f'{p}.q_a_proj.weight')},
        'q_a_norm': {'scale': self._vector(get, f'{p}.q_a_layernorm.weight')},
        'q_b_proj': {
            'w': self._linear(get, f'{p}.q_b_proj.weight', h, cfg.q_head_dim)
        },
        'kv_a_proj': {'w': self._linear(get, f'{p}.kv_a_proj_with_mqa.weight')},
        'kv_a_norm': {'scale': self._vector(get, f'{p}.kv_a_layernorm.weight')},
        'kv_b_proj': {
            'w': self._linear(
                get,
                f'{p}.kv_b_proj.weight',
                h,
                cfg.qk_nope_head_dim + cfg.v_head_dim,
            )
        },
        'g_proj': {
            'w': self._linear(get, f'{p}.g_proj.weight', h, cfg.v_head_dim)
        },
        'o_proj': {
            'w': (
                self._linear(get, f'{p}.o_proj.weight').reshape(
                    h, cfg.v_head_dim, -1
                )
            )
        },
    }

  def _expert_weight(
      self, get: TensorFn, layer: int, name: str, transpose: bool
  ) -> PyTree:
    """Stacks one expert matrix over all experts into `[E, in, out]`.

    MXFP4 weights cannot be transposed while packed, so with
    `dequantize_experts=False` they keep the HF orientation and are emitted as
    `{'packed': [E, out, in // 2], 'scale': [E, out, in // 32]}`.

    Args:
      get: tensor accessor.
      layer: layer index.
      name: `w1` (gate), `w2` (down) or `w3` (up).
      transpose: emit `[E, in, out]` from the HF `[out, in]` layout.

    Returns:
      `{'w': [E, in, out]}`, or the packed pair above when the layer ships
      MXFP4 and `dequantize_experts` is False.
    """
    base = f'{self.prefix}model.layers.{layer}.block_sparse_moe.experts'
    num_experts = self.config.num_experts
    packed_pairs = []
    out = None
    for e in range(num_experts):
      key = f'{base}.{e}.{name}.weight'
      try:
        w = np.asarray(get(key))
      except KeyError:
        packed = np.asarray(get(f'{key}_packed'))
        scale = np.asarray(get(f'{key}_scale'))
        if not self.dequantize_experts:
          packed_pairs.append((packed, scale))
          continue
        w = dequantize_mxfp4(packed, scale, self.expert_dtype)
      w = w.T if transpose else w
      # Fill in place: the real leaves are [896, 3584, 3072], where a
      # list-then-stack would peak at twice the final size.
      if out is None:
        out = np.empty((num_experts, *w.shape), self.expert_dtype)
      out[e] = w
    if packed_pairs:
      return {
          MXFP4_PACKED_KEY: np.stack([s[0] for s in packed_pairs]),
          MXFP4_SCALE_KEY: np.stack([s[1] for s in packed_pairs]),
      }
    return {'w': out}

  def convert_moe(
      self, get: TensorFn, layer: int, include_experts: bool = True
  ) -> PyTree:
    """The layer's MoE block: routed experts, router and shared expert.

    Args:
      get: tensor accessor.
      layer: layer index.
      include_experts: whether to stack the routed expert weights here. A
        streaming converter leaves them out and materializes each stack as its
        own group instead (see `group_paths`).

    Returns:
      The `ffn` sub-tree of the block.
    """
    p = f'model.layers.{layer}.block_sparse_moe'
    experts = {}
    if include_experts:
      experts = {
          'experts': {
              name: self._expert_weight(get, layer, hf_name, transpose=True)
              for name, hf_name in EXPERT_PROJECTIONS.items()
          }
      }
    return {
        **experts,
        'router': {
            'w': self._linear(get, f'{p}.gate.weight'),
            'bias': self._vector(get, f'{p}.gate.e_score_correction_bias'),
        },
        'down_proj': {
            'w': self._linear(get, f'{p}.routed_expert_down_proj.weight')
        },
        'up_proj': {
            'w': self._linear(get, f'{p}.routed_expert_up_proj.weight')
        },
        'latent_norm': {
            'scale': self._vector(get, f'{p}.routed_expert_norm.weight')
        },
        'shared': {
            'ffn_0_gate': {
                'w': self._linear(get, f'{p}.shared_experts.gate_proj.weight')
            },
            'ffn_0': {
                'w': self._linear(get, f'{p}.shared_experts.up_proj.weight')
            },
            'ffn_1': {
                'w': self._linear(get, f'{p}.shared_experts.down_proj.weight')
            },
        },
    }

  def convert_dense_mlp(self, get: TensorFn, layer: int) -> PyTree:
    p = f'model.layers.{layer}.mlp'
    return {
        'ffn_0_gate': {'w': self._linear(get, f'{p}.gate_proj.weight')},
        'ffn_0': {'w': self._linear(get, f'{p}.up_proj.weight')},
        'ffn_1': {'w': self._linear(get, f'{p}.down_proj.weight')},
    }

  def is_moe_layer(self, layer: int) -> bool:
    return self.config.use_moe and layer >= self.config.first_k_dense_replace

  def convert_block_norms(self, get: TensorFn, layer: int) -> PyTree:
    """The layer's normalization gains and AttnRes pseudo-queries."""
    p = f'model.layers.{layer}'
    return {
        'input_layernorm': {
            'scale': self._vector(get, f'{p}.input_layernorm.weight')
        },
        'post_attention_layernorm': {
            'scale': self._vector(get, f'{p}.post_attention_layernorm.weight')
        },
        'attn_res_self': self._attn_res(
            get, f'{p}.self_attention_res_norm', f'{p}.self_attention_res_proj'
        ),
        'attn_res_mlp': self._attn_res(
            get, f'{p}.mlp_res_norm', f'{p}.mlp_res_proj'
        ),
    }

  def convert_token_mixer(self, get: TensorFn, layer: int) -> PyTree:
    if self.layer_types[layer] == 'linear_attention':
      return self.convert_kda(get, layer)
    return self.convert_mla(get, layer)

  def convert_ffn(
      self, get: TensorFn, layer: int, include_experts: bool = True
  ) -> PyTree:
    if self.is_moe_layer(layer):
      return self.convert_moe(get, layer, include_experts=include_experts)
    return self.convert_dense_mlp(get, layer)

  def convert_block(self, get: TensorFn, layer: int) -> PyTree:
    return {
        **self.convert_block_norms(get, layer),
        'token_mixer': self.convert_token_mixer(get, layer),
        'ffn': self.convert_ffn(get, layer),
    }

  def group_paths(
      self, layers: Sequence[int] | None = None
  ) -> list[tuple[str, ...]]:
    """Tree paths of every conversion group, in tree order.

    A group is the unit a streaming converter materializes and then frees; each
    routed expert stack is its own group because one is 20 GiB at bf16. Groups
    nest (`block_3/ffn` contains `block_3/ffn/experts/ffn_0`) but their leaves
    are disjoint, so the full tree is the merge of all groups.

    Args:
      layers: layer subset; all layers by default.

    Returns:
      One path per group, each naming the sub-tree `convert_group` builds.
    """
    layers = range(self.config.n_layers) if layers is None else layers
    paths: list[tuple[str, ...]] = [('embed_linear',)]
    for i in layers:
      block = f'block_{i}'
      paths += [(block,), (block, 'token_mixer'), (block, 'ffn')]
      if self.is_moe_layer(i):
        paths += [
            (block, 'ffn', 'experts', name) for name in EXPERT_PROJECTIONS
        ]
    return paths + [('final_attn_res',), ('final_ln',)]

  def convert_group(self, get: TensorFn, path: Sequence[str]) -> PyTree:
    """Builds one group of `group_paths`, excluding any nested group."""
    match tuple(path):
      case ('embed_linear',):
        return self.convert_embeddings(get)
      case ('final_attn_res',):
        return self._attn_res(
            get, 'model.output_attn_res_norm', 'model.output_attn_res_proj'
        )
      case ('final_ln',):
        return {'scale': self._vector(get, 'model.norm.weight')}
      case (block,) if block.startswith('block_'):
        return self.convert_block_norms(get, int(block[len('block_') :]))
      case (block, 'token_mixer'):
        return self.convert_token_mixer(get, int(block[len('block_') :]))
      case (block, 'ffn'):
        return self.convert_ffn(
            get, int(block[len('block_') :]), include_experts=False
        )
      case (block, 'ffn', 'experts', name) if name in EXPERT_PROJECTIONS:
        return self._expert_weight(
            get,
            int(block[len('block_') :]),
            EXPERT_PROJECTIONS[name],
            transpose=True,
        )
    raise ValueError(f'Not a conversion group: {tuple(path)}')

  def convert_embeddings(self, get: TensorFn) -> PyTree:
    embed = self._cast(
        np.asarray(get(f'{self.prefix}model.embed_tokens.weight'))
    )
    params = {'embed': embed}
    if not self.config.use_tied_embedding:
      # `module.EmbeddingLinear` contracts with `'vd,...d->...v'`, i.e. the
      # output weight keeps HF's `[vocab, dim]` orientation.
      params['w'] = self._cast(np.asarray(get(f'{self.prefix}lm_head.weight')))
    return params

  def convert(self, get: TensorFn) -> PyTree:
    """Returns the full `params` tree for `KimiK3LM`."""
    params = {'embed_linear': self.convert_embeddings(get)}
    for layer in range(self.config.n_layers):
      params[f'block_{layer}'] = self.convert_block(get, layer)
    params['final_attn_res'] = self._attn_res(
        get, 'model.output_attn_res_norm', 'model.output_attn_res_proj'
    )
    params['final_ln'] = {'scale': self._vector(get, 'model.norm.weight')}
    return params


def convert_from_mapping(
    state_dict: Mapping[str, np.ndarray], config: Any, **kwargs: Any
) -> PyTree:
  """Converts an in-memory HF state dict (used by tests)."""
  converter = KimiK3HfConverter(
      config, prefix=detect_prefix(state_dict), **kwargs
  )

  def get(name: str) -> np.ndarray:
    if name not in state_dict:
      raise KeyError(name)
    return state_dict[name]

  return converter.convert(get)

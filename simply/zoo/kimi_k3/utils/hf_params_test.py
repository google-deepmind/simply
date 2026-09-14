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
"""Tests for the HuggingFace -> Simply Kimi K3 parameter mapping.

Three things are pinned here:

  * `dequantize_mxfp4` against a hand-written table and, independently, against
    `ml_dtypes`' own E2M1/E8M0 bit layouts (the LUT and the nibble order were
    verified against real shard bytes in the MXFP4 section of the resource
    report; this keeps that result from drifting);
  * the shape and transpose contract of every module type, against the golden
    fixture, which holds real HF-layout tensors -- including the arithmetic
    identity `x @ W_simply == x @ W_hf.T` that the layouts exist to preserve;
  * the streaming contract (`group_paths` / `convert_group`) and the released
    config reproducing the `KimiK3ExperimentConfig` defaults.
"""

import dataclasses
import json
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3.utils import hf_params

# The HuggingFace oracle lives with the tests that model it; this one reads it
# through the `:k3_golden_tiny` filegroup of the parent package.
_TESTDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testdata')

# `config.json:text_config` of the release (moonshotai/Kimi-K3, commit 9f62e4e),
# trimmed to the fields `config_from_hf` reads. The point of the copy is to
# fail if `KimiK3ExperimentConfig`'s defaults ever stop describing the released
# 2.8T model.
RELEASED_TEXT_CONFIG: dict[str, Any] = {
    'activation_situ_beta': 4.0,
    'activation_situ_linear_beta': 25.0,
    'attn_res_block_size': 12,
    'first_k_dense_replace': 1,
    'hidden_size': 7168,
    'intermediate_size': 33792,
    'kv_lora_rank': 512,
    'latent_moe_use_norm': True,
    'linear_attn_config': {
        'full_attn_layers': list(range(4, 93, 4)) + [93],
        'gate_lower_bound': -5.0,
        'head_dim': 128,
        'kda_layers': [i for i in range(1, 94) if i % 4 and i != 93],
        'num_heads': 96,
        'short_conv_kernel_size': 4,
        'use_full_rank_gate': True,
    },
    'mla_use_nope': True,
    'mla_use_output_gate': True,
    'moe_intermediate_size': 3072,
    'moe_renormalize': True,
    'num_attention_heads': 96,
    'num_experts': 896,
    'num_experts_per_token': 16,
    'num_hidden_layers': 93,
    'num_shared_experts': 2,
    'pad_token_id': 163839,
    'q_lora_rank': 1536,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'rms_norm_eps': 1e-05,
    'routed_expert_hidden_size': 3584,
    'routed_scaling_factor': 1.0,
    'tie_word_embeddings': False,
    'v_head_dim': 128,
    'vocab_size': 163840,
}


def _pack_nibbles(codes: np.ndarray) -> np.ndarray:
  """Packs `[..., in]` 4-bit codes into `[..., in // 2]` bytes, low first."""
  low, high = codes[..., 0::2], codes[..., 1::2]
  return (low | (high << 4)).astype(np.uint8)


def _reference_dequantize(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
  """Dequantizes through `ml_dtypes`' own bit layouts, not through the LUT."""
  codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), np.uint8)
  codes[..., 0::2] = packed & 0x0F
  codes[..., 1::2] = packed >> 4
  values = codes.view(ml_dtypes.float4_e2m1fn).astype(np.float32)
  factors = scale.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
  group = values.shape[-1] // factors.shape[-1]
  return values * np.repeat(factors, group, axis=-1)


class DequantizeMxfp4Test(parameterized.TestCase):

  def test_value_table(self):
    """Every E2M1 code, at scale 2^0, against the hand-written table."""
    codes = np.arange(16, dtype=np.uint8)[None, :]
    packed = _pack_nibbles(codes)
    scale = np.full((1, 1), 127, np.uint8)  # 2 ** (127 - 127) == 1.0
    expected = [
        0,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        6,
        -0.0,
        -0.5,
        -1,
        -1.5,
        -2,
        -3,
        -4,
        -6,
    ]
    got = hf_params.dequantize_mxfp4(packed, scale)
    np.testing.assert_array_equal(got, np.asarray([expected], np.float32))
    # Code 8 is negative zero, not a second positive zero.
    self.assertEqual(np.signbit(got[0, 8]), True)
    self.assertEqual(np.signbit(got[0, 0]), False)

  def test_low_nibble_is_the_even_column(self):
    packed = np.asarray([[0x21]], np.uint8)  # low = 1 (0.5), high = 2 (1.0)
    scale = np.full((1, 1), 127, np.uint8)
    np.testing.assert_array_equal(
        hf_params.dequantize_mxfp4(packed, scale),
        np.asarray([[0.5, 1.0]], np.float32),
    )

  @parameterized.parameters(
      (0, 2.0**-127),  # Not a special value: the smallest legal exponent.
      (100, 2.0**-27),
      (119, 2.0**-8),  # The four codes that actually occur in the release.
      (122, 2.0**-5),
      (127, 1.0),
      (254, 2.0**127),
  )
  def test_e8m0_scale(self, code, factor):
    packed = _pack_nibbles(np.asarray([[2, 2]], np.uint8))  # value 1.0
    scale = np.full((1, 1), code, np.uint8)
    got = hf_params.dequantize_mxfp4(packed, scale)
    np.testing.assert_array_equal(got, np.full((1, 2), factor, np.float32))

  def test_groups_are_contiguous_and_aligned(self):
    """Group k scales columns [32k, 32k + 32); no interleaving."""
    in_dim, groups = 64, 2
    codes = np.full((3, in_dim), 2, np.uint8)  # every value 1.0
    packed = _pack_nibbles(codes)
    scale = np.asarray([[127, 128], [126, 127], [127, 127]], np.uint8)
    got = hf_params.dequantize_mxfp4(packed, scale)
    expected = np.repeat(
        np.asarray([[1.0, 2.0], [0.5, 1.0], [1.0, 1.0]], np.float32),
        in_dim // groups,
        axis=1,
    )
    np.testing.assert_array_equal(got, expected)

  def test_matches_ml_dtypes_on_random_weights(self):
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, size=(7, 16), dtype=np.uint8)
    # 255 is E8M0 NaN, which `np.exp2` renders as +inf; no released tensor
    # contains it (all scales are in 119..122), so the two agree everywhere.
    scale = rng.integers(0, 255, size=(7, 1), dtype=np.uint8)
    np.testing.assert_array_equal(
        hf_params.dequantize_mxfp4(packed, scale),
        _reference_dequantize(packed, scale),
    )

  def test_stacked_and_dtype(self):
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 256, size=(4, 3, 16), dtype=np.uint8)
    scale = rng.integers(100, 130, size=(4, 3, 1), dtype=np.uint8)
    got = hf_params.dequantize_mxfp4(packed, scale, ml_dtypes.bfloat16)
    self.assertEqual(got.shape, (4, 3, 32))
    self.assertEqual(got.dtype, ml_dtypes.bfloat16)
    for e in range(4):
      np.testing.assert_array_equal(
          got[e].astype(np.float32),
          hf_params.dequantize_mxfp4(packed[e], scale[e])
          .astype(ml_dtypes.bfloat16)
          .astype(np.float32),
      )


class SharedContractTest(absltest.TestCase):
  """What `ckpt_format` and the streaming converter read out of this module."""

  def test_prefix_is_detected_only_when_it_is_there(self):
    self.assertEmpty(hf_params.detect_prefix(['model.norm.weight']))
    self.assertEqual(
        hf_params.detect_prefix(
            ['lm_head.weight', 'language_model.model.norm.weight']
        ),
        hf_params.LANGUAGE_MODEL_PREFIX,
    )

  def test_an_mxfp4_node_is_the_packed_pair_and_nothing_else(self):
    array = np.zeros((2, 4), np.uint8)
    packed = {
        hf_params.MXFP4_PACKED_KEY: array,
        hf_params.MXFP4_SCALE_KEY: array,
    }
    self.assertTrue(hf_params.is_mxfp4_node(packed))
    self.assertFalse(hf_params.is_mxfp4_node({'w': array}))
    self.assertFalse(
        hf_params.is_mxfp4_node({hf_params.MXFP4_PACKED_KEY: array})
    )
    self.assertFalse(hf_params.is_mxfp4_node(array))


class GoldenFixtureTest(parameterized.TestCase):
  """The layout contract, against real HF-layout tensors."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    data = np.load(os.path.join(_TESTDATA, 'k3_golden_tiny.npz'))
    with open(os.path.join(_TESTDATA, 'k3_golden_tiny_config.json')) as f:
      meta = json.load(f)
    cls.hf = {
        k[len('param/') :]: data[k]
        for k in data.files
        if k.startswith('param/')
    }
    cls.config = k3_config_lib.config_from_hf(meta['config_kwargs'])
    cls.meta = meta['meta']
    cls.params = hf_params.convert_from_mapping(
        cls.hf, cls.config, dtype=np.float32
    )

  def _expected_shapes(self) -> dict[str, tuple[int, ...]]:
    """The param tree the converter targets, written from the config."""
    c = self.config
    d, h, k = c.model_dim, c.kda_num_heads, c.kda_head_dim
    mla_h, dv = c.n_heads, c.v_head_dim
    e, latent, m = (
        c.num_experts,
        c.routed_expert_latent_dim,
        (c.moe_intermediate_size),
    )
    ffn = c.ffn_expand_dim or 0
    kda = self.meta['kda_layers_0indexed'][0]
    mla = self.meta['mla_layers_0indexed'][0]
    moe = self.meta['moe_layers_0indexed'][0]
    return {
        'embed_linear/embed': (c.vocab_size, d),
        # Untied output weight keeps HF's [vocab, dim]: `EmbeddingLinear`
        # contracts with 'vd,...d->...v'.
        'embed_linear/w': (c.vocab_size, d),
        'final_attn_res/scale': (d,),
        'final_attn_res/w': (d,),
        'final_ln/scale': (d,),
        f'block_{kda}/input_layernorm/scale': (d,),
        f'block_{kda}/attn_res_self/w': (d,),
        f'block_{kda}/token_mixer/q_proj/w': (d, h, k),
        f'block_{kda}/token_mixer/k_proj/w': (d, h, k),
        f'block_{kda}/token_mixer/v_proj/w': (d, h, k),
        f'block_{kda}/token_mixer/q_conv/w': (h, k, c.kda_conv_kernel_dim),
        f'block_{kda}/token_mixer/f_a_proj/w': (d, c.kda_gate_lora_rank),
        f'block_{kda}/token_mixer/f_b_proj/w': (c.kda_gate_lora_rank, h, k),
        f'block_{kda}/token_mixer/dt_bias': (h, k),
        f'block_{kda}/token_mixer/a_log': (h,),
        f'block_{kda}/token_mixer/b_proj/w': (d, h),
        f'block_{kda}/token_mixer/g_proj/w': (d, h, k),
        f'block_{kda}/token_mixer/o_norm/scale': (k,),
        f'block_{kda}/token_mixer/o_proj/w': (h, k, d),
        'block_0/ffn/ffn_0_gate/w': (d, ffn),
        'block_0/ffn/ffn_1/w': (ffn, d),
        f'block_{mla}/token_mixer/q_a_proj/w': (d, c.q_lora_rank),
        f'block_{mla}/token_mixer/q_a_norm/scale': (c.q_lora_rank,),
        f'block_{mla}/token_mixer/q_b_proj/w': (
            c.q_lora_rank,
            mla_h,
            c.q_head_dim,
        ),
        f'block_{mla}/token_mixer/kv_a_proj/w': (
            d,
            c.kv_lora_rank + c.qk_rope_head_dim,
        ),
        f'block_{mla}/token_mixer/kv_a_norm/scale': (c.kv_lora_rank,),
        f'block_{mla}/token_mixer/kv_b_proj/w': (
            c.kv_lora_rank,
            mla_h,
            c.qk_nope_head_dim + dv,
        ),
        f'block_{mla}/token_mixer/g_proj/w': (d, mla_h, dv),
        f'block_{mla}/token_mixer/o_proj/w': (mla_h, dv, d),
        f'block_{moe}/ffn/router/w': (d, e),
        f'block_{moe}/ffn/router/bias': (e,),
        f'block_{moe}/ffn/down_proj/w': (d, latent),
        f'block_{moe}/ffn/up_proj/w': (latent, d),
        f'block_{moe}/ffn/latent_norm/scale': (latent,),
        f'block_{moe}/ffn/experts/ffn_0_gate/w': (e, latent, m),
        f'block_{moe}/ffn/experts/ffn_0/w': (e, latent, m),
        f'block_{moe}/ffn/experts/ffn_1/w': (e, m, latent),
        f'block_{moe}/ffn/shared/ffn_0_gate/w': (
            d,
            c.shared_expert_intermediate_size,
        ),
        f'block_{moe}/ffn/shared/ffn_1/w': (
            c.shared_expert_intermediate_size,
            d,
        ),
    }

  def _leaf(self, path: str) -> np.ndarray:
    node = self.params
    for key in path.split('/'):
      node = node[key]
    return node

  def test_shapes(self):
    for path, shape in self._expected_shapes().items():
      with self.subTest(path):
        self.assertEqual(self._leaf(path).shape, shape)

  def test_tree_has_one_block_per_layer(self):
    self.assertCountEqual(
        list(self.params),
        ['embed_linear', 'final_attn_res', 'final_ln']
        + [f'block_{i}' for i in range(self.config.n_layers)],
    )

  @parameterized.named_parameters(
      ('kda_q_proj', 0, 'token_mixer/q_proj/w', 'self_attn.q_proj.weight'),
      ('kda_g_proj', 0, 'token_mixer/g_proj/w', 'self_attn.g_proj.weight'),
      (
          'kda_f_b_proj',
          0,
          'token_mixer/f_b_proj/w',
          'self_attn.f_b_proj.weight',
      ),
      (
          'mla_q_b_proj',
          3,
          'token_mixer/q_b_proj/w',
          'self_attn.q_b_proj.weight',
      ),
      (
          'mla_kv_b_proj',
          3,
          'token_mixer/kv_b_proj/w',
          'self_attn.kv_b_proj.weight',
      ),
      (
          'mla_kv_a_proj',
          3,
          'token_mixer/kv_a_proj/w',
          'self_attn.kv_a_proj_with_mqa.weight',
      ),
      ('moe_router', 1, 'ffn/router/w', 'block_sparse_moe.gate.weight'),
      (
          'moe_down_proj',
          1,
          'ffn/down_proj/w',
          'block_sparse_moe.routed_expert_down_proj.weight',
      ),
      (
          'moe_shared_up',
          1,
          'ffn/shared/ffn_0/w',
          'block_sparse_moe.shared_experts.up_proj.weight',
      ),
      ('dense_mlp_gate', 0, 'ffn/ffn_0_gate/w', 'mlp.gate_proj.weight'),
  )
  def test_projection_is_the_transpose(self, layer, simply, hf):
    """`x @ W_simply` must equal `x @ W_hf.T` for every `nn.Linear`."""
    w_simply = self._leaf(f'block_{layer}/{simply}')
    w_hf = self.hf[f'model.layers.{layer}.{hf}']
    rng = np.random.default_rng(0)
    x = rng.standard_normal(w_hf.shape[1], dtype=np.float32)
    np.testing.assert_allclose(
        x @ w_simply.reshape(w_simply.shape[0], -1), x @ w_hf.T, atol=1e-5
    )

  def test_output_projection_folds_the_head_axis(self):
    """`o_proj` is `[H, V, D]`, contracting the head axes in order."""
    layer = 0
    w_simply = self._leaf(f'block_{layer}/token_mixer/o_proj/w')
    w_hf = self.hf[f'model.layers.{layer}.self_attn.o_proj.weight']
    h, v, _ = w_simply.shape
    rng = np.random.default_rng(1)
    o = rng.standard_normal((h, v), dtype=np.float32)
    np.testing.assert_allclose(
        np.einsum('hv,hvd->d', o, w_simply), w_hf @ o.reshape(-1), atol=1e-5
    )

  def test_depthwise_conv_splits_the_head_axis(self):
    layer = 0
    for name, hf_name in (('q_conv', 'q_conv1d'), ('v_conv', 'v_conv1d')):
      w = self._leaf(f'block_{layer}/token_mixer/{name}/w')
      hf = self.hf[f'model.layers.{layer}.self_attn.{hf_name}.weight']
      np.testing.assert_array_equal(w, hf.reshape(w.shape))

  def test_a_log_drops_the_zero_padding(self):
    """`A_log` ships padded to the head dim; only `num_heads` are trained."""
    a_log = self._leaf('block_0/token_mixer/a_log')
    hf = self.hf['model.layers.0.self_attn.A_log']
    self.assertEqual(a_log.shape, (self.config.kda_num_heads,))
    np.testing.assert_array_equal(a_log, hf[: self.config.kda_num_heads])

  def test_expert_stacking_order(self):
    """Expert `e` of the stack is HF expert `e`, transposed to `[in, out]`."""
    layer = self.meta['moe_layers_0indexed'][0]
    base = f'model.layers.{layer}.block_sparse_moe.experts'
    for simply, hf in hf_params.EXPERT_PROJECTIONS.items():
      stacked = self._leaf(f'block_{layer}/ffn/experts/{simply}/w')
      for e in range(self.config.num_experts):
        np.testing.assert_array_equal(
            stacked[e],
            self.hf[f'{base}.{e}.{hf}.weight'].T,
            err_msg=f'{simply}[{e}] is not HF {hf} of expert {e}',
        )

  def test_attn_res_projection_is_a_vector(self):
    """The pseudo-query is `nn.Linear(dim, 1)`, i.e. `[1, dim]` in HF."""
    w = self._leaf('block_0/attn_res_self/w')
    hf = self.hf['model.layers.0.self_attention_res_proj.weight']
    self.assertEqual(w.shape, (self.config.model_dim,))
    np.testing.assert_array_equal(w, hf[0])

  def test_language_model_prefix_is_detected(self):
    prefixed = {f'language_model.{k}': v for k, v in self.hf.items()}
    params = hf_params.convert_from_mapping(
        prefixed, self.config, dtype=np.float32
    )
    np.testing.assert_array_equal(
        params['block_0']['token_mixer']['q_proj']['w'],
        self._leaf('block_0/token_mixer/q_proj/w'),
    )

  def test_unprefixed_keys_are_not_found_with_a_prefix(self):
    converter = hf_params.KimiK3HfConverter(
        self.config, prefix='language_model.'
    )
    with self.assertRaises(KeyError):
      converter.convert(lambda name: self.hf[name])

  def test_dtype_none_preserves_the_source(self):
    hf = {k: v.astype(ml_dtypes.bfloat16) for k, v in self.hf.items()}
    hf['model.layers.0.self_attn.A_log'] = self.hf[
        'model.layers.0.self_attn.A_log'
    ]
    params = hf_params.convert_from_mapping(hf, self.config, dtype=None)
    self.assertEqual(
        params['block_0']['token_mixer']['q_proj']['w'].dtype,
        ml_dtypes.bfloat16,
    )
    self.assertEqual(
        params['block_0']['token_mixer']['a_log'].dtype, np.float32
    )
    self.assertEqual(params['embed_linear']['embed'].dtype, ml_dtypes.bfloat16)


class StreamingGroupsTest(absltest.TestCase):
  """`group_paths` / `convert_group`: the contract the converter binary uses."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    data = np.load(os.path.join(_TESTDATA, 'k3_golden_tiny.npz'))
    with open(os.path.join(_TESTDATA, 'k3_golden_tiny_config.json')) as f:
      meta = json.load(f)
    cls.hf = {
        k[len('param/') :]: data[k]
        for k in data.files
        if k.startswith('param/')
    }
    cls.config = k3_config_lib.config_from_hf(meta['config_kwargs'])

  def _flat(self, tree, prefix=()):
    if isinstance(tree, dict):
      for key, value in tree.items():
        yield from self._flat(value, prefix + (key,))
    else:
      yield prefix, tree

  def test_groups_reconstruct_the_full_tree(self):
    converter = hf_params.KimiK3HfConverter(self.config, dtype=np.float32)
    get = lambda name: self.hf[name]
    merged: dict[tuple[str, ...], np.ndarray] = {}
    for path in converter.group_paths():
      for suffix, array in self._flat(converter.convert_group(get, path)):
        leaf = path + suffix
        self.assertNotIn(leaf, merged, f'{leaf} appears in two groups')
        merged[leaf] = array
    expected = dict(self._flat(converter.convert(get)))
    self.assertCountEqual(merged, expected)
    for path, array in expected.items():
      np.testing.assert_array_equal(merged[path], array, err_msg=str(path))

  def test_group_paths_cover_a_layer_subset_only(self):
    converter = hf_params.KimiK3HfConverter(self.config)
    paths = converter.group_paths([1, 2])
    blocks = {p[0] for p in paths if p[0].startswith('block_')}
    self.assertEqual(blocks, {'block_1', 'block_2'})
    self.assertIn(('embed_linear',), paths)
    self.assertIn(('final_ln',), paths)

  def test_unknown_group_raises(self):
    converter = hf_params.KimiK3HfConverter(self.config)
    with self.assertRaises(ValueError):
      converter.convert_group(lambda name: self.hf[name], ('block_0', 'nope'))


class ReleasedConfigTest(absltest.TestCase):

  def test_defaults_describe_the_release(self):
    config = k3_config_lib.config_from_hf(RELEASED_TEXT_CONFIG)
    defaults = k3_config_lib.KimiK3ExperimentConfig()
    self.assertEqual(dataclasses.replace(config, layer_types=()), defaults)
    self.assertEqual(
        config.resolved_layer_types(), defaults.resolved_layer_types()
    )

  def test_layer_schedule(self):
    types = k3_config_lib.config_from_hf(
        RELEASED_TEXT_CONFIG
    ).resolved_layer_types()
    self.assertLen(types, 93)
    self.assertEqual(types.count(k3_config_lib.FULL_ATTENTION), 24)
    self.assertEqual(types[-1], k3_config_lib.FULL_ATTENTION)
    self.assertEqual(types[-2], k3_config_lib.FULL_ATTENTION)
    self.assertEqual(types[0], k3_config_lib.LINEAR_ATTENTION)


if __name__ == '__main__':
  absltest.main()

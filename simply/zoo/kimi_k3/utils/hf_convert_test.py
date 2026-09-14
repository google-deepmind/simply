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
"""End-to-end test for the streaming Kimi K3 checkpoint converter.

Builds a two-layer release in a temp directory -- one KDA layer with a dense
MLP, one MLA layer with an MXFP4 LatentMoE, split over two safetensors shards
with a real `model.safetensors.index.json` and a vision tower to ignore -- and
runs the converter over it. The synthetic release states the HF layout
independently of `hf_params.py`, so a rename or a transposition on either side
fails here.
"""

import collections
import json
import os
import struct
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import ml_dtypes
import numpy as np
from simply.utils import checkpoint_lib
from simply.utils import common
from simply.zoo.kimi_k3 import config_lib as k3_config_lib
from simply.zoo.kimi_k3 import model_lib as k3_model_lib
from simply.zoo.kimi_k3.utils import hf_convert
from simply.zoo.kimi_k3.utils import hf_params

_PREFIX = 'language_model.'

# A two-layer K3: layer 0 is KDA + dense MLP, layer 1 is MLA + LatentMoE.
TINY_TEXT_CONFIG: dict[str, Any] = {
    'activation_situ_beta': 4.0,
    'activation_situ_linear_beta': 25.0,
    'attn_res_block_size': 2,
    'first_k_dense_replace': 1,
    'hidden_size': 32,
    'intermediate_size': 48,
    'kv_lora_rank': 4,
    'latent_moe_use_norm': True,
    'linear_attn_config': {
        'full_attn_layers': [2],
        'gate_lower_bound': -5.0,
        'head_dim': 8,
        'kda_layers': [1],
        'num_heads': 2,
        'short_conv_kernel_size': 4,
        'use_full_rank_gate': True,
    },
    'mla_use_nope': True,
    'mla_use_output_gate': True,
    'moe_intermediate_size': 32,
    'moe_renormalize': True,
    'num_attention_heads': 2,
    'num_experts': 3,
    'num_experts_per_token': 2,
    'num_hidden_layers': 2,
    'num_shared_experts': 2,
    'q_lora_rank': 8,
    'qk_nope_head_dim': 8,
    'qk_rope_head_dim': 4,
    'rms_norm_eps': 1e-05,
    'routed_expert_hidden_size': 64,
    'routed_scaling_factor': 1.0,
    'tie_word_embeddings': False,
    'v_head_dim': 8,
    'vocab_size': 16,
}


def _rng():
  return np.random.default_rng(20260813)


def _bf16(rng, *shape) -> np.ndarray:
  return rng.standard_normal(shape, dtype=np.float32).astype(ml_dtypes.bfloat16)


def _f32(rng, *shape) -> np.ndarray:
  return rng.standard_normal(shape, dtype=np.float32)


def _mxfp4(rng, out_dim: int, in_dim: int, group: int = 32):
  """One routed expert matrix as the release stores it."""
  packed = rng.integers(0, 256, size=(out_dim, in_dim // 2), dtype=np.uint8)
  # E8M0 codes in the range the release actually uses (2^-8 .. 2^-5).
  scale = rng.integers(
      119, 123, size=(out_dim, in_dim // group), dtype=np.uint8
  )
  return packed, scale


def hf_tensors(config) -> dict[str, np.ndarray]:
  """The HF-layout state dict of the tiny release, `[out, in]` throughout."""
  rng = _rng()
  c = config
  d, v = c.model_dim, c.vocab_size
  kda_h, kda_k = c.kda_num_heads, c.kda_head_dim
  h, dv = c.n_heads, c.v_head_dim
  latent, m = c.routed_expert_latent_dim, c.moe_intermediate_size
  shared = c.shared_expert_intermediate_size
  t: dict[str, np.ndarray] = {
      'model.embed_tokens.weight': _bf16(rng, v, d),
      'lm_head.weight': _bf16(rng, v, d),
      'model.norm.weight': _bf16(rng, d),
      'model.output_attn_res_norm.weight': _bf16(rng, d),
      'model.output_attn_res_proj.weight': _bf16(rng, 1, d),
  }
  for layer in range(c.n_layers):
    p = f'model.layers.{layer}'
    for name in (
        'input_layernorm',
        'post_attention_layernorm',
        'self_attention_res_norm',
        'mlp_res_norm',
    ):
      t[f'{p}.{name}.weight'] = _bf16(rng, d)
    for name in ('self_attention_res_proj', 'mlp_res_proj'):
      t[f'{p}.{name}.weight'] = _bf16(rng, 1, d)
    if layer == 0:  # KDA
      for name in ('q_proj', 'k_proj', 'v_proj', 'g_proj'):
        t[f'{p}.self_attn.{name}.weight'] = _bf16(rng, kda_h * kda_k, d)
      for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
        t[f'{p}.self_attn.{name}.weight'] = _f32(rng, kda_h * kda_k, 1, 4)
      t[f'{p}.self_attn.f_a_proj.weight'] = _bf16(rng, c.kda_gate_lora_rank, d)
      t[f'{p}.self_attn.f_b_proj.weight'] = _bf16(
          rng, kda_h * kda_k, c.kda_gate_lora_rank
      )
      t[f'{p}.self_attn.b_proj.weight'] = _bf16(rng, kda_h, d)
      t[f'{p}.self_attn.dt_bias'] = _f32(rng, kda_h * kda_k)
      # A_log ships padded to the head dim; only `num_heads` are trained.
      a_log = _f32(rng, kda_k)
      a_log[kda_h:] = 0.0
      t[f'{p}.self_attn.A_log'] = a_log
      t[f'{p}.self_attn.o_norm.weight'] = _f32(rng, kda_k)
      t[f'{p}.self_attn.o_proj.weight'] = _bf16(rng, d, kda_h * kda_k)
      t[f'{p}.mlp.gate_proj.weight'] = _bf16(rng, c.ffn_expand_dim, d)
      t[f'{p}.mlp.up_proj.weight'] = _bf16(rng, c.ffn_expand_dim, d)
      t[f'{p}.mlp.down_proj.weight'] = _bf16(rng, d, c.ffn_expand_dim)
      continue
    # MLA
    t[f'{p}.self_attn.q_a_proj.weight'] = _bf16(rng, c.q_lora_rank, d)
    t[f'{p}.self_attn.q_a_layernorm.weight'] = _bf16(rng, c.q_lora_rank)
    t[f'{p}.self_attn.q_b_proj.weight'] = _bf16(
        rng, h * c.q_head_dim, c.q_lora_rank
    )
    t[f'{p}.self_attn.kv_a_proj_with_mqa.weight'] = _bf16(
        rng, c.kv_lora_rank + c.qk_rope_head_dim, d
    )
    t[f'{p}.self_attn.kv_a_layernorm.weight'] = _bf16(rng, c.kv_lora_rank)
    t[f'{p}.self_attn.kv_b_proj.weight'] = _bf16(
        rng, h * (c.qk_nope_head_dim + dv), c.kv_lora_rank
    )
    t[f'{p}.self_attn.g_proj.weight'] = _bf16(rng, h * dv, d)
    t[f'{p}.self_attn.o_proj.weight'] = _bf16(rng, d, h * dv)
    # LatentMoE
    q = f'{p}.block_sparse_moe'
    t[f'{q}.gate.weight'] = _bf16(rng, c.num_experts, d)
    t[f'{q}.gate.e_score_correction_bias'] = _f32(rng, c.num_experts)
    t[f'{q}.routed_expert_down_proj.weight'] = _bf16(rng, latent, d)
    t[f'{q}.routed_expert_up_proj.weight'] = _bf16(rng, d, latent)
    t[f'{q}.routed_expert_norm.weight'] = _f32(rng, latent)
    t[f'{q}.shared_experts.gate_proj.weight'] = _bf16(rng, shared, d)
    t[f'{q}.shared_experts.up_proj.weight'] = _bf16(rng, shared, d)
    t[f'{q}.shared_experts.down_proj.weight'] = _bf16(rng, d, shared)
    for e in range(c.num_experts):
      for name, (out_dim, in_dim) in (
          ('w1', (m, latent)),
          ('w3', (m, latent)),
          ('w2', (latent, m)),
      ):
        packed, scale = _mxfp4(rng, out_dim, in_dim)
        t[f'{q}.experts.{e}.{name}.weight_packed'] = packed
        t[f'{q}.experts.{e}.{name}.weight_scale'] = scale
  prefixed = {_PREFIX + k: v for k, v in t.items()}
  # The multimodal wrapper the text backbone must skip, not silently drop.
  prefixed['vision_tower.layers.0.proj.weight'] = _bf16(rng, 4, 4)
  prefixed['mm_projector.weight'] = _bf16(rng, 4, d)
  return prefixed


def write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
  """Writes `<u64 header length><json header><data>`."""
  codes = {
      np.dtype(v).name: k for k, v in hf_convert.SAFETENSORS_DTYPES.items()
  }
  header, blobs, offset = {}, [], 0
  for name, array in tensors.items():
    array = np.ascontiguousarray(array)
    header[name] = {
        'dtype': codes[array.dtype.name],
        'shape': list(array.shape),
        'data_offsets': [offset, offset + array.nbytes],
    }
    blobs.append(array.tobytes())
    offset += array.nbytes
  blob = json.dumps(header).encode()
  blob += b' ' * (-len(blob) % 8)
  with open(path, 'wb') as f:
    f.write(struct.pack('<Q', len(blob)))
    f.write(blob)
    for chunk in blobs:
      f.write(chunk)


def write_release(hf_dir: str, config, tensors: dict[str, np.ndarray]) -> None:
  """Writes config.json, two shards and the index over them."""
  shards = {
      'model-00001-of-000002.safetensors': {
          k: v for k, v in tensors.items() if '.layers.1.' not in k
      },
      'model-00002-of-000002.safetensors': {
          k: v for k, v in tensors.items() if '.layers.1.' in k
      },
  }
  for shard, contents in shards.items():
    write_safetensors(os.path.join(hf_dir, shard), contents)
  weight_map = {}
  for shard, contents in shards.items():
    for name in contents:
      weight_map[name] = shard
  with open(os.path.join(hf_dir, 'model.safetensors.index.json'), 'w') as f:
    json.dump(
        {
            'metadata': {'total_size': sum(v.nbytes for v in tensors.values())},
            'weight_map': weight_map,
        },
        f,
    )
  with open(os.path.join(hf_dir, 'config.json'), 'w') as f:
    json.dump({'model_type': 'kimi_k3', 'text_config': TINY_TEXT_CONFIG}, f)
  del config


class ReleaseFixture(parameterized.TestCase):
  """A written tiny release plus the objects the converter builds from it."""

  def setUp(self):
    super().setUp()
    self.hf_dir = self.create_tempdir('hf').full_path
    self.config = k3_config_lib.config_from_hf(TINY_TEXT_CONFIG)
    self.tensors = hf_tensors(self.config)
    write_release(self.hf_dir, self.config, self.tensors)
    self.index = hf_convert.read_index(self.hf_dir)

  def converter(self, **kwargs):
    return hf_convert.build_converter(self.config, self.index, **kwargs)

  def expected_params(self, **kwargs):
    """The reference tree, built in memory by the same mapping."""
    return hf_params.convert_from_mapping(self.tensors, self.config, **kwargs)

  def expected_leaves(
      self, dequantize_experts: bool = False
  ) -> dict[hf_convert.Path, np.ndarray]:
    """The reference tree, flat, keyed the way the plan keys it."""
    tree = self.expected_params(
        dtype=None,
        expert_dtype=ml_dtypes.bfloat16,
        dequantize_experts=dequantize_experts,
    )
    return {
        ('params',) + tuple(path): array
        for path, array in hf_convert.flatten_tree(tree)
    }


class IndexTest(ReleaseFixture):

  def test_every_tensor_is_indexed(self):
    self.assertCountEqual(self.index.shard_of, self.tensors)
    self.assertEmpty(self.index.missing_shards)
    self.assertEqual(
        self.index.total_size, sum(v.nbytes for v in self.tensors.values())
    )

  def test_headers_give_shape_dtype_and_range(self):
    for name, array in self.tensors.items():
      source = self.index[name]
      self.assertEqual(source.shape, array.shape, name)
      self.assertEqual(source.dtype, array.dtype, name)
      self.assertEqual(source.nbytes, array.nbytes, name)

  def test_reader_returns_the_bytes_verbatim(self):
    with hf_convert.ShardReader(self.hf_dir) as reader:
      for name in list(self.tensors)[::7]:
        np.testing.assert_array_equal(
            reader.read(self.index[name]), self.tensors[name], err_msg=name
        )

  def test_missing_shard_is_reported_not_guessed(self):
    os.remove(os.path.join(self.hf_dir, 'model-00002-of-000002.safetensors'))
    index = hf_convert.read_index(self.hf_dir)
    self.assertEqual(
        index.missing_shards, ('model-00002-of-000002.safetensors',)
    )
    with self.assertRaises(KeyError):
      _ = index[f'{_PREFIX}model.layers.1.self_attn.q_a_proj.weight']

  def test_truncated_shard_is_detected(self):
    shard = os.path.join(self.hf_dir, 'model-00002-of-000002.safetensors')
    with open(shard, 'r+b') as f:
      f.truncate(os.path.getsize(shard) - 1)
    index = hf_convert.read_index(self.hf_dir)
    self.assertEqual(
        index.truncated_shards, ('model-00002-of-000002.safetensors',)
    )

  def test_config_round_trips_through_the_release(self):
    self.assertEqual(hf_convert.load_config(self.hf_dir), self.config)

  def test_prefix_is_detected(self):
    self.assertEqual(self.converter().prefix, _PREFIX)


class PlanTest(ReleaseFixture):

  @parameterized.parameters('mxfp4', 'bfloat16')
  def test_plan_matches_the_in_memory_conversion(self, expert_dtype):
    converter = self.converter(expert_dtype=expert_dtype)
    plan = hf_convert.build_plan(
        converter, self.index, range(self.config.n_layers)
    )
    expected = self.expected_leaves(dequantize_experts=expert_dtype != 'mxfp4')
    self.assertCountEqual([leaf.path for leaf in plan.leaves], expected)
    for leaf in plan.leaves:
      self.assertEqual(leaf.shape, expected[leaf.path].shape, leaf.name)
      self.assertEqual(leaf.dtype, expected[leaf.path].dtype, leaf.name)

  def test_expert_sources_cover_every_expert(self):
    converter = self.converter()
    plan = hf_convert.build_plan(converter, self.index, [1])
    group = ('block_1', 'ffn', 'experts', 'ffn_0_gate')
    sources = plan.group_sources[group]
    self.assertLen(sources, 2 * self.config.num_experts)
    for e in range(self.config.num_experts):
      base = f'{_PREFIX}model.layers.1.block_sparse_moe.experts.{e}.w1.weight'
      self.assertIn(f'{base}_packed', sources)
      self.assertIn(f'{base}_scale', sources)
    # Expert-major order, so the read-ahead predicts the converter's next read.
    self.assertIn('.experts.0.', sources[0])
    self.assertIn('.experts.0.', sources[1])
    self.assertIn('.experts.1.', sources[2])

  def test_every_source_exists_and_nothing_is_dropped_silently(self):
    converter = self.converter()
    plan = hf_convert.build_plan(
        converter, self.index, range(self.config.n_layers)
    )
    for name in plan.sources:
      self.assertIn(name, self.index)
    unconsumed = set(self.index.shard_of) - set(plan.sources)
    self.assertCountEqual(
        unconsumed, ['vision_tower.layers.0.proj.weight', 'mm_projector.weight']
    )
    self.assertEmpty(hf_convert.dry_run_report(plan, self.index, converter))

  def test_layer_subset(self):
    plan = hf_convert.build_plan(self.converter(), self.index, [1])
    blocks = {
        leaf.path[1] for leaf in plan.leaves if leaf.path[1].startswith('block')
    }
    self.assertEqual(blocks, {'block_1'})

  def test_the_default_keeps_the_packed_pair(self):
    converter = self.converter()
    plan = hf_convert.build_plan(converter, self.index, [1])
    packed = {
        leaf.path[-1]: leaf
        for leaf in plan.leaves
        if leaf.path[2:5] == ('ffn', 'experts', 'ffn_0_gate')
    }
    self.assertCountEqual(packed, ['packed', 'scale'])
    e = self.config.num_experts
    latent, m = self.config.routed_expert_latent_dim, (
        self.config.moe_intermediate_size
    )
    self.assertEqual(packed['packed'].shape, (e, m, latent // 2))
    self.assertEqual(packed['scale'].shape, (e, m, latent // 32))
    self.assertEqual(packed['packed'].dtype, np.uint8)

  @parameterized.parameters(
      ('', (0, 1)),
      ('1', (1,)),
      ('0-1', (0, 1)),
      ('1,0', (0, 1)),
  )
  def test_parse_layers(self, spec, expected):
    self.assertEqual(hf_convert.parse_layers(spec, 2), expected)

  def test_parse_layers_rejects_out_of_range(self):
    with self.assertRaises(ValueError):
      hf_convert.parse_layers('0-4', 2)


class ConvertEndToEndTest(ReleaseFixture):

  def _convert(self, out_dir, layers=None, options=None, **kwargs):
    converter = self.converter(**kwargs)
    layers = range(self.config.n_layers) if layers is None else layers
    plan = hf_convert.build_plan(converter, self.index, layers)
    with hf_convert.ShardReader(self.hf_dir) as reader:
      materializer = hf_convert.GroupMaterializer(
          converter, plan, self.index, reader
      )
      hf_convert.write_checkpoint(
          plan,
          materializer.leaf,
          out_dir,
          step=0,
          options=options or hf_convert.WriteOptions(),
      )
      materializer.close()
    return plan

  @parameterized.parameters('mxfp4', 'bfloat16')
  def test_written_tree_matches_the_in_memory_conversion(self, expert_dtype):
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    plan = self._convert(out_dir, expert_dtype=expert_dtype)
    restored = hf_convert.restore_leaves(out_dir, 0, plan)
    expected = self.expected_leaves(dequantize_experts=expert_dtype != 'mxfp4')
    self.assertCountEqual(restored, expected)
    for path, array in expected.items():
      np.testing.assert_array_equal(
          restored[path], array, err_msg='/'.join(path)
      )

  def test_experts_are_dequantized_exactly(self):
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    plan = self._convert(out_dir, layers=[1], expert_dtype='bfloat16')
    leaf = next(
        leaf
        for leaf in plan.leaves
        if leaf.path[2:] == ('ffn', 'experts', 'ffn_1', 'w')
    )
    got = hf_convert.restore_leaves(out_dir, 0, plan, [leaf])[leaf.path]
    base = f'{_PREFIX}model.layers.1.block_sparse_moe.experts'
    for e in range(self.config.num_experts):
      want = hf_params.dequantize_mxfp4(
          self.tensors[f'{base}.{e}.w2.weight_packed'],
          self.tensors[f'{base}.{e}.w2.weight_scale'],
          ml_dtypes.bfloat16,
      ).T
      np.testing.assert_array_equal(got[e], want, err_msg=f'expert {e}')

  def test_verify_passes_on_a_fresh_conversion(self):
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    plan = self._convert(out_dir)
    converter = self.converter()
    with hf_convert.ShardReader(self.hf_dir) as reader:
      materializer = hf_convert.GroupMaterializer(
          converter, plan, self.index, reader
      )
      problems = hf_convert.verify(
          plan, materializer.leaf, out_dir, 0, samples=6
      )
      materializer.close()
    self.assertEmpty(problems)

  def test_verify_a_single_layer_of_a_full_checkpoint(self):
    """The post-write smoke check: --verify_only --verify_layers=0."""
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    plan = self._convert(out_dir)
    leaves = hf_convert.leaves_of_layers(plan, [0])
    blocks = {leaf.path[1] for leaf in leaves}
    self.assertEqual(
        blocks, {'block_0', 'embed_linear', 'final_attn_res', 'final_ln'}
    )
    converter = self.converter()
    with hf_convert.ShardReader(self.hf_dir) as reader:
      materializer = hf_convert.GroupMaterializer(
          converter, plan, self.index, reader
      )
      problems = hf_convert.verify(
          plan, materializer.leaf, out_dir, 0, samples=0, leaves=leaves
      )
      materializer.close()
    self.assertEmpty(problems)

  def test_leaf_stats_flag_nothing_on_healthy_tensors(self):
    plan = hf_convert.build_plan(self.converter(), self.index, [1])
    scale = next(leaf for leaf in plan.leaves if leaf.path[-1] == 'scale')
    stats = hf_convert.leaf_stats(scale, np.full(scale.shape, 120, np.uint8))
    self.assertIn('NaN(0xFF)=0', stats)
    norm = next(leaf for leaf in plan.leaves if leaf.path[-1] == 'w')
    stats = hf_convert.leaf_stats(norm, np.ones(norm.shape, ml_dtypes.bfloat16))
    self.assertIn('nonfinite 0', stats)

  def test_verify_reports_a_corrupted_leaf(self):
    """A single sample always checks the largest leaf: the expert stack."""
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    plan = self._convert(out_dir)
    biggest = max(plan.leaves, key=lambda leaf: leaf.nbytes)
    problems = hf_convert.verify(
        plan,
        lambda leaf: np.zeros(leaf.shape, leaf.dtype),
        out_dir,
        0,
        samples=1,
    )
    self.assertLen(problems, 1)
    self.assertIn(biggest.name, problems[0])

  def test_format_tag_is_kimi_k3(self):
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    self._convert(out_dir, layers=[0])
    fmt = hf_convert.read_checkpoint_format(out_dir, 0)
    self.assertEqual(type(fmt).__name__, 'KimiK3Format')

  def test_contradictory_expert_flags_are_rejected(self):
    with self.assertRaises(ValueError):
      self.converter(expert_dtype='mxfp4', dequantize_experts=True)

  def test_tiny_write_window_still_converts(self):
    """A window smaller than one leaf must drain, not deadlock or corrupt."""
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    plan = self._convert(
        out_dir,
        options=hf_convert.WriteOptions(
            chunk_bytes=4096, max_inflight_bytes=1, max_rss_bytes=0
        ),
    )
    restored = hf_convert.restore_leaves(out_dir, 0, plan)
    expected = self.expected_params(dtype=None, expert_dtype=ml_dtypes.bfloat16)
    np.testing.assert_array_equal(
        restored[('params', 'embed_linear', 'embed')],
        expected['embed_linear']['embed'],
    )

  def test_mxfp4_mode_round_trips_the_packed_bytes(self):
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    plan = self._convert(out_dir, layers=[1])
    leaves = {
        leaf.path[-1]: leaf
        for leaf in plan.leaves
        if leaf.path[2:5] == ('ffn', 'experts', 'ffn_0')
    }
    restored = hf_convert.restore_leaves(
        out_dir, 0, plan, list(leaves.values())
    )
    base = f'{_PREFIX}model.layers.1.block_sparse_moe.experts'
    for e in range(self.config.num_experts):
      np.testing.assert_array_equal(
          restored[leaves['packed'].path][e],
          self.tensors[f'{base}.{e}.w3.weight_packed'],
      )
      np.testing.assert_array_equal(
          restored[leaves['scale'].path][e],
          self.tensors[f'{base}.{e}.w3.weight_scale'],
      )

  def test_simply_restores_it_into_the_model_param_tree(self):
    """The whole point: Simply's own loader must produce the model's params."""
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    self._convert(out_dir)
    model = k3_model_lib.KimiK3LM(
        config=self.config, sharding_config=self.config.sharding_config
    )
    abstract = {'params': checkpoint_lib.get_abstract_params(model)}
    # `load_checkpoint_from_path` re-wraps leaves as `AnnotatedArray`.
    restored: Any = common.get_raw_arrays(
        checkpoint_lib.load_checkpoint_from_path(
            os.path.join(out_dir, '0'), abstract
        )['params']
    )
    expected = self.expected_params(
        dtype=None, expert_dtype=ml_dtypes.bfloat16, dequantize_experts=True
    )
    jax.tree.map(
        lambda got, want: np.testing.assert_allclose(
            np.asarray(got, np.float32),
            np.asarray(want, np.float32),
            rtol=1e-6,
        ),
        restored,
        expected,
    )
    # The elementwise comparison above already rejects a transposed stack, but
    # state the contract the transpose exists for: the restored expert is
    # `[E, in, out]` and contracts on `in`.
    expert = 2
    got = np.asarray(
        restored['block_1']['ffn']['experts']['ffn_0_gate']['w'][expert],
        np.float32,
    )
    base = f'{_PREFIX}model.layers.1.block_sparse_moe.experts.{expert}.w1'
    hf = hf_params.dequantize_mxfp4(
        self.tensors[f'{base}.weight_packed'],
        self.tensors[f'{base}.weight_scale'],
    )
    self.assertEqual(got.shape, hf.T.shape)
    x = np.random.default_rng(3).standard_normal(hf.shape[1], dtype=np.float32)
    np.testing.assert_allclose(x @ got, x @ hf.T, rtol=1e-5, atol=1e-4)

  def test_orbax_serializes_in_sorted_key_order(self):
    """The order the whole memory story is phrased in, measured not assumed."""
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    converter = self.converter()
    plan = hf_convert.build_plan(converter, self.index, [1])
    served: list[hf_convert.Path] = []
    with hf_convert.ShardReader(self.hf_dir) as reader:
      materializer = hf_convert.GroupMaterializer(
          converter, plan, self.index, reader
      )

      def materialize(leaf):
        served.append(leaf.path)
        return materializer.leaf(leaf)

      hf_convert.write_checkpoint(
          plan, materialize, out_dir, 0, hf_convert.WriteOptions()
      )
      materializer.close()
    self.assertEqual(served, sorted(served))
    # Not plan order: a group is interrupted by the nested groups of its own
    # subtree, which is what the materializer has to survive.
    self.assertNotEqual(served, [leaf.path for leaf in plan.leaves])

  def test_one_expert_stack_at_a_time(self):
    """The memory bound, driven by the real write order rather than the plan."""
    out_dir = os.path.join(self.create_tempdir('out').full_path, 'ckpt')
    converter = self.converter()
    plan = hf_convert.build_plan(converter, self.index, [1])
    with hf_convert.ShardReader(self.hf_dir) as reader:
      materializer = hf_convert.GroupMaterializer(
          converter, plan, self.index, reader
      )
      hf_convert.write_checkpoint(
          plan, materializer.leaf, out_dir, 0, hf_convert.WriteOptions()
      )
      # One build per group, and nothing retained at the end.
      self.assertEqual(materializer.groups_rebuilt, 0)
      self.assertLen(converter.group_paths([1]), materializer.groups_built)
      self.assertEqual(materializer.resident_bytes, 0)
      materializer.close()
    group_bytes = collections.Counter()
    for leaf in plan.leaves:
      group_bytes[leaf.group] += leaf.nbytes

    def with_ancestors(group: hf_convert.Path) -> int:
      return sum(
          nbytes
          for path, nbytes in group_bytes.items()
          if group[: len(path)] == path
      )

    # The documented bound: one group plus the groups it is nested in. It has
    # to be well under the whole tree, or this asserts nothing.
    bound = max(with_ancestors(group) for group in group_bytes)
    self.assertLess(bound, plan.output_bytes)
    self.assertLessEqual(materializer.peak_resident_bytes, bound)


if __name__ == '__main__':
  absltest.main()

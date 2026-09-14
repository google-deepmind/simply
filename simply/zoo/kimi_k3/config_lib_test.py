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
"""Tests for the Kimi K3 config type, the experiment configs and their HBM arithmetic.

The parameter formula in `k3_config_lib.param_counts` is checked against
`jax.eval_shape(model.init)` on the small configs -- where the abstract tree is
cheap -- and then used at 2.8T scale, so the released totals are backed by the
same code the budget table is.
"""

import dataclasses
import json
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from simply import config_lib
from simply import model_lib
from simply.utils import common
from simply.utils import sharding as sharding_lib
from simply.zoo.kimi_k3 import config_lib as k3_config_lib

# `k3_config_lib` registers no modules of its own -- `create_model` below needs
# `KimiK3LM` in the module registry, which importing the model is what does.
from simply.zoo.kimi_k3 import model_lib as k3_model_lib  # pylint: disable=unused-import

_TESTDATA = os.path.join(os.path.dirname(__file__), 'testdata')

GIB = 2**30
MESH_AXES = ('replica', 'data', 'seq', 'model')
# Released `linear_attn_config.full_attn_layers`, 0-indexed.
EXPECTED_MLA_LAYERS = tuple(range(3, 92, 4)) + (92,)
# Every config this module registers; the list is derived so that a new one
# cannot be added without meeting `test_registered_and_well_formed`.
REGISTERED = tuple(
    sorted(
        name
        for name in config_lib.ExperimentConfigRegistry.keys()
        if name.startswith('kimi_k3')
    )
)
# 2x2x4: 16 chips = 32 devices, the smallest slice the sharding divides.
MESH_2X2X4 = {'replica': 1, 'data': 2, 'seq': 4, 'model': 4}


def one_layer_config() -> k3_config_lib.KimiK3ExperimentConfig:
  """One KDA + one MLA layer at full width, random init.

  Real `model_dim`, expert count, expert width and vocab, so the expensive
  program shapes (the 896-way grouped matmul, the 96-head KDA scan, the
  163840-wide head) are the production ones while `jax.eval_shape` stays cheap.
  Layer 0 is dense and layer 1 is MoE, so both FFN kinds appear.

  Returns:
    `kimi_k3_decode()` cut down to those two layers.
  """
  return dataclasses.replace(
      k3_config_lib.kimi_k3_decode(),
      n_layers=2,
      layer_types=(
          k3_config_lib.LINEAR_ATTENTION,
          k3_config_lib.FULL_ATTENTION,
      ),
      batch_size=8,
      seq_len=4096,
      mesh_shape=MESH_2X2X4,
      decoding_mesh_shape=MESH_2X2X4,
      init_ckpt_format='',  # Random init: nothing loads these weights.
  )


def setUpModule():
  # Rank-1 mesh over the single test device: the modules annotate every big
  # tensor, and `with_sharding_constraint` validates the annotation's rank.
  sharding_lib.set_mesh({name: 1 for name in MESH_AXES}, axis_names=MESH_AXES)


def _abstract_params(config) -> Any:
  model, _ = model_lib.create_model(config)
  return jax.eval_shape(model.init, jax.random.PRNGKey(0))


def _count(params: Any) -> int:
  return sum(int(x.size) for x in jax.tree.leaves(params))


class ConfigTest(parameterized.TestCase):

  def test_every_registered_config_is_covered(self):
    self.assertEqual(
        set(REGISTERED),
        {
            'kimi_k3_2p8t',
            'kimi_k3_decode',
            'kimi_k3_decode_ep',
            'kimi_k3_tiny_test',
        },
    )

  @parameterized.parameters(*REGISTERED)
  def test_registered_and_well_formed(self, name):
    config = config_lib.ExperimentConfigRegistry.get_config(name)
    self.assertIsInstance(config, k3_config_lib.KimiK3ExperimentConfig)
    self.assertEqual(config.model_name, 'KimiK3LM')
    self.assertEqual(config.vocab_name, 'KimiK3')
    self.assertEqual(config.lm_format_name, 'KimiK3Chat')
    self.assertEqual(config.input_processor_name, 'KimiK3InputProcessor')
    self.assertFalse(config.use_scan)  # Opt-in; see `use_scan`.
    self.assertLen(config.resolved_layer_types(), config.n_layers)

  def test_checkpoint_is_the_converter_output_and_user_supplied(self):
    config = k3_config_lib.kimi_k3_2p8t()
    # No path: a 1.45 TiB conversion lives wherever its owner has quota, so the
    # config carries the format and the caller passes `--ckpt_dir`.
    self.assertEmpty(config.init_ckpt_dir)
    # `convert_hf_checkpoint.py` tags the tree it writes `KimiK3Format`:
    # `V2Format` plus the MXFP4 expert decode on restore.
    self.assertEqual(config.init_ckpt_format, 'KimiK3Format')
    self.assertEqual(config.init_ckpt_step, -1)
    self.assertEmpty(k3_config_lib.kimi_k3_tiny_test().init_ckpt_dir)

  def test_decode_is_the_documented_deployment(self):
    config = k3_config_lib.kimi_k3_decode()
    self.assertEqual(config.batch_size, 64)
    self.assertEqual(config.seq_len, 32768)


class LayerScheduleTest(absltest.TestCase):

  def test_matches_released_linear_attn_config(self):
    config = k3_config_lib.kimi_k3_2p8t()
    types = config.resolved_layer_types()
    mla = tuple(
        i for i, t in enumerate(types) if t == k3_config_lib.FULL_ATTENTION
    )
    self.assertEqual(mla, EXPECTED_MLA_LAYERS)
    self.assertLen(mla, 24)
    self.assertEqual(types.count(k3_config_lib.LINEAR_ATTENTION), 69)
    # The last two layers are both MLA -- 93 = 23 * 4 + 1.
    self.assertEqual(types[-2:], (k3_config_lib.FULL_ATTENTION,) * 2)

  def test_attn_res_slots(self):
    config = k3_config_lib.kimi_k3_2p8t()
    self.assertEqual(config.attn_res_block_size, 12)
    self.assertEqual(-(-config.n_layers // config.attn_res_block_size), 8)


class ParamCountTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tiny', k3_config_lib.kimi_k3_tiny_test),
      ('one_layer', one_layer_config),
  )
  def test_formula_matches_eval_shape(self, config_fn):
    config = config_fn()
    counts = k3_config_lib.param_counts(config)
    self.assertEqual(counts.total, _count(_abstract_params(config)))

  def test_model_builds_and_has_the_documented_shapes(self):
    config = one_layer_config()
    params = _abstract_params(config)
    shapes = {
        'embed_linear/embed': (163840, 7168),
        # `EmbeddingLinear` contracts 'vd,...d->...v', so the untied head
        # keeps HF's [vocab, dim] orientation rather than transposing it.
        'embed_linear/w': (163840, 7168),
        'block_0/token_mixer/q_proj/w': (7168, 96, 128),
        'block_0/token_mixer/o_proj/w': (96, 128, 7168),
        'block_0/token_mixer/f_b_proj/w': (128, 96, 128),
        'block_0/ffn/ffn_0/w': (7168, 33792),
        'block_1/token_mixer/q_b_proj/w': (1536, 96, 192),
        'block_1/token_mixer/kv_a_proj/w': (7168, 576),
        'block_1/token_mixer/kv_b_proj/w': (512, 96, 256),
        'block_1/ffn/router/w': (7168, 896),
        'block_1/ffn/down_proj/w': (7168, 3584),
        'block_1/ffn/experts/ffn_0_gate/w': (896, 3584, 3072),
        'block_1/ffn/experts/ffn_1/w': (896, 3072, 3584),
        'block_1/ffn/shared/ffn_0/w': (7168, 6144),
        'block_1/attn_res_self/w': (7168,),
    }
    leaves = jax.tree_util.tree_flatten_with_path(
        params, is_leaf=lambda x: isinstance(x, common.AnnotatedArray)
    )[0]
    flat = {'/'.join(str(k.key) for k in path): leaf for path, leaf in leaves}
    for name, shape in shapes.items():
      self.assertIn(name, flat)
      self.assertEqual(flat[name].shape, shape, name)

  def test_released_model_eval_shape_is_2p8t(self):
    """The real 93-layer tree, abstractly -- shapes only, nothing allocated."""
    config = k3_config_lib.kimi_k3_2p8t()
    params = _abstract_params(config)
    self.assertLen([k for k in params if k.startswith('block_')], 93)
    total = _count(params)
    self.assertBetween(total, 2.7e12, 2.9e12)
    self.assertEqual(total, k3_config_lib.param_counts(config).total)

  def test_released_model_is_2p8t_and_104b_activated(self):
    counts = k3_config_lib.param_counts(k3_config_lib.kimi_k3_2p8t())
    self.assertBetween(counts.total, 2.7e12, 2.9e12)
    self.assertBetween(counts.activated, 100e9, 110e9)
    # The routed experts are the whole model: 2.72 of 2.78 T.
    self.assertBetween(counts.routed_experts / counts.total, 0.97, 0.99)
    self.assertAlmostEqual(counts.total / 1e12, 2.7795, places=3)
    self.assertAlmostEqual(counts.activated / 1e9, 104.2, places=1)

  def test_docstring_per_layer_numbers(self):
    config = k3_config_lib.kimi_k3_2p8t()
    one = lambda **kw: k3_config_lib.param_counts(
        dataclasses.replace(config, **kw)
    ).total
    embed = 2 * config.vocab_size * config.model_dim + 3 * config.model_dim
    kda_layer = dataclasses.replace(
        config,
        n_layers=1,
        layer_types=(k3_config_lib.LINEAR_ATTENTION,),
        first_k_dense_replace=1,
        ffn_expand_dim=0,
    )
    mla_layer = dataclasses.replace(
        kda_layer, layer_types=(k3_config_lib.FULL_ATTENTION,)
    )
    norms = 6 * config.model_dim
    kda = k3_config_lib.param_counts(kda_layer).total - embed - norms
    mla = k3_config_lib.param_counts(mla_layer).total - embed - norms
    self.assertAlmostEqual(kda / 1e6, 443.74, places=2)
    self.assertAlmostEqual(mla / 1e6, 232.20, places=2)
    # 92 MoE layers of 29.785 B, of which 29.595 B is the routed stack.
    moe = one(n_layers=93) - one(n_layers=92)  # Layer 92 is MoE (and MLA).
    self.assertAlmostEqual((moe - mla - norms) / 1e9, 29.785, places=3)
    self.assertAlmostEqual(3 * 896 * 3584 * 3072 / 1e9, 29.595, places=3)
    self.assertAlmostEqual(3 * 7168 * 33792 / 1e6, 726.7, places=1)
    self.assertAlmostEqual(embed / 1e9, 2.349, places=3)


class HbmBudgetTest(parameterized.TestCase):

  def test_unit_costs(self):
    config = k3_config_lib.kimi_k3_2p8t()
    per_token = k3_config_lib.hbm_budget(
        config,
        {'replica': 1, 'data': 1, 'seq': 1, 'model': 1},
        batch_size=1,
        max_seq_len=1024,
        prefill_chunk_tokens=1024,
    )
    self.assertAlmostEqual(per_token.mla_cache / 1024 / 1024, 27.0, places=2)
    self.assertAlmostEqual(
        per_token.attn_res_snapshots / 1024 / 1024, 112.0, places=2
    )
    self.assertAlmostEqual(
        per_token.attn_res_mixture / 1024 / 1024, 252.0, places=2
    )
    # KDA state is constant in the sequence length -- that is its point.
    self.assertAlmostEqual(per_token.kda_state / 2**20, 414.0, places=1)
    self.assertAlmostEqual(per_token.kda_conv_state / 2**20, 38.8, places=1)

  def test_bf16_weights_fit_on_4x4x8(self):
    budget = k3_config_lib.hbm_budget(
        k3_config_lib.kimi_k3_2p8t(), k3_config_lib.MESH_4X4X8
    )
    self.assertEqual(budget.devices, 256)
    self.assertAlmostEqual(budget.weights / GIB, 20.22, places=2)
    # ...and mxfp4-packed experts would be 3.6x smaller again.
    packed = k3_config_lib.hbm_budget(
        k3_config_lib.kimi_k3_2p8t(),
        k3_config_lib.MESH_4X4X8,
        weight_bytes_per_param=4.25 / 8,
    )
    self.assertLess(packed.weights / GIB, 7.0)

  @parameterized.named_parameters(
      ('eval_32k', 64, 32768, 33.55),
      ('eval_131k', 32, 131072, 37.01),
  )
  def test_eval_deployments_fit(self, batch_size, seq_len, expected_gib):
    config = dataclasses.replace(
        k3_config_lib.kimi_k3_decode(), batch_size=batch_size, seq_len=seq_len
    )
    budget = k3_config_lib.hbm_budget(config, k3_config_lib.MESH_4X4X8)
    self.assertAlmostEqual(budget.total / GIB, expected_gib, places=1)
    self.assertTrue(budget.fits())
    # Half the HBM stays free for XLA temporaries and the logits.
    self.assertLess(budget.total, k3_config_lib.HBM_BYTES_PER_DEVICE / 2)

  def test_batch_must_be_sharded(self):
    """The `data` axis is 8 because a replicated batch does not fit."""
    no_dp = {'replica': 1, 'data': 1, 'seq': 32, 'model': 8}
    budget = k3_config_lib.hbm_budget(k3_config_lib.kimi_k3_decode(), no_dp)
    self.assertGreater(budget.total, k3_config_lib.HBM_BYTES_PER_DEVICE)

  def test_unchunked_prefill_does_not_fit(self):
    """AttnRes forces chunked prefill: 8x the residual stream, in f32."""
    budget = lambda chunk: k3_config_lib.hbm_budget(
        k3_config_lib.kimi_k3_decode(),
        k3_config_lib.MESH_4X4X8,
        max_seq_len=131072,
        prefill_chunk_tokens=chunk,
    )
    self.assertFalse(budget(131072).fits())
    self.assertGreater(budget(131072).total / GIB, 135)
    self.assertTrue(budget(8192).fits())
    self.assertLess(budget(8192).total / GIB, 60)
    # The f32 mixture temporary dominates the bf16 snapshot buffer 2.25:1.
    chunked = budget(8192)
    self.assertAlmostEqual(
        chunked.attn_res_mixture / chunked.attn_res_snapshots, 2.25, places=6
    )

  def test_batch_sharding_the_kda_state_saves_6gib(self):
    """Why `utils/kda.py`'s state follows the activations' batch axis."""
    kwargs = dict(
        config=k3_config_lib.kimi_k3_decode(),
        mesh_shape=k3_config_lib.MESH_4X4X8,
    )
    sharded = k3_config_lib.hbm_budget(**kwargs)
    replicated = k3_config_lib.hbm_budget(
        **kwargs, kda_state_batch_sharded=False
    )
    saved = (
        replicated.kda_state
        + replicated.kda_conv_state
        - sharded.kda_state
        - sharded.kda_conv_state
    )
    self.assertAlmostEqual(saved / GIB, 6.19, places=1)
    self.assertAlmostEqual(sharded.kda_state / GIB, 0.81, places=2)


class ShardingTest(absltest.TestCase):

  def test_annotation_ranks_match_the_tensors_they_describe(self):
    """`with_sharding_constraint` raises on a rank mismatch; catch it here."""
    ranks = {
        'ffn0_partition': 3,  # Expert stack [E, in, out]; dense uses [-2:].
        'ffn1_partition': 3,
        'attn_qkv_partition': 3,  # [D, H, head_dim]
        'attn_o_partition': 3,
        'embed_partition': 2,  # [V, D]
        'attn_activation_partition': 4,  # [B, T, H, head_dim]
        'activation_partition': 3,  # [B, T, D]
        'ffn0_activation_partition': 3,
        'logits_partition': 3,
        'data_partition': 2,  # [B, T]
    }
    for sharding in (
        k3_config_lib.kimi_k3_2p8t().sharding_config,
        k3_config_lib.kimi_k3_decoding_sharding(),
    ):
      self.assertEqual(sharding.mesh_axis_names, MESH_AXES)
      for field, rank in ranks.items():
        self.assertLen(getattr(sharding, field), rank, field)

  def test_decoding_sharding_keeps_tp_and_drops_the_token_shard(self):
    decoding = k3_config_lib.kimi_k3_decoding_sharding()
    self.assertEqual(
        decoding.activation_partition, (('replica', 'data'), None, 'model')
    )
    self.assertEqual(decoding.data_partition, (('replica', 'data'), None))
    # Weight placement is core's `moe_sharding()`, which the prefill/training
    # config uses unchanged: EP on 'seq', TP on 'model'.
    train = config_lib.moe_sharding()
    for field in (
        'ffn0_partition',
        'ffn1_partition',
        'attn_qkv_partition',
        'attn_o_partition',
        'embed_partition',
    ):
      self.assertEqual(getattr(decoding, field), getattr(train, field), field)

  def test_expert_stack_is_sharded_over_every_axis(self):
    config = k3_config_lib.kimi_k3_decode()
    mesh = k3_config_lib.MESH_4X4X8
    ffn0 = config.sharding_config.ffn0_partition
    self.assertEqual(ffn0, ('seq', 'data', 'model'))
    per_device = (
        config.num_experts
        // mesh['seq']
        * config.routed_expert_latent_dim
        // mesh['data']
        * config.moe_intermediate_size
        // mesh['model']
    )
    self.assertEqual(per_device, 112 * 448 * 768)

  def test_every_module_accepts_the_expert_parallel_annotations(self):
    """Regression: the rank-3 expert `ffn0_partition` must not leak.

    Modules that reuse `ffn0_partition` for their own rank-2 projections must
    take its last two entries; `with_sharding_constraint` raises on a rank
    mismatch, so building the model under `moe_sharding()` is the test.
    `utils/kda.py`'s `b_proj/w [7168, 96]` was the one place this went wrong.
    """
    config = one_layer_config()
    self.assertEqual(
        config.sharding_config.ffn0_partition, ('seq', 'data', 'model')
    )
    params = _abstract_params(config)
    self.assertEqual(
        params['block_0']['token_mixer']['b_proj']['w'].shape, (7168, 96)
    )


class ConfigFromHfTest(absltest.TestCase):

  def test_matches_golden_fixture_config(self):
    with open(os.path.join(_TESTDATA, 'k3_golden_tiny_config.json')) as f:
      meta = json.load(f)
    config = k3_config_lib.config_from_hf(meta['config_kwargs'])
    self.assertEqual(config.n_layers, 8)
    self.assertEqual(config.model_dim, 128)
    self.assertEqual(config.attn_res_block_size, 4)
    self.assertEqual(config.num_experts, 8)
    self.assertEqual(config.routed_expert_latent_dim, 64)
    mla = [
        i
        for i, t in enumerate(config.resolved_layer_types())
        if t == k3_config_lib.FULL_ATTENTION
    ]
    self.assertEqual(mla, meta['meta']['mla_layers_0indexed'])

  def test_rejects_rotary_variants(self):
    with open(os.path.join(_TESTDATA, 'k3_golden_tiny_config.json')) as f:
      hf = dict(json.load(f)['config_kwargs'])
    hf['mla_use_nope'] = False
    with self.assertRaises(ValueError):
      k3_config_lib.config_from_hf(hf)


if __name__ == '__main__':
  absltest.main()

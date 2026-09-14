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
"""The config is the model's contract with the release; this pins it.

Three things are checked. **Fidelity**: every leaf of the released
`text_config` (`testdata/qwen3p8_27b_text_config.json`, a copy of
`Qwen/Qwen3.8-27B`'s) is in exactly one of three tables -- mapped to a Simply
field and asserted equal, deliberately different with the intended value
asserted, or not represented with a reason -- so a key the release adds cannot
pass unnoticed. **Discrimination**: `config_from_hf` raises on each variant
this port does not implement, one test per variant, because a silently ignored
field is the failure mode that costs a week. **Arithmetic**: the closed-form
`param_counts` is checked against `jax.eval_shape(model.init)` on a small
config and then trusted at 27B.
"""

import dataclasses
import json
import math
import os
from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import jax

from simply import config_lib as simply_config_lib
from simply.utils import module
from simply.utils import sharding as sharding_lib
from simply.zoo.qwen3p8 import config_lib
from simply.zoo.qwen3p8 import model_lib

_TESTDATA = os.path.join(os.path.dirname(__file__), 'testdata')

# Every config this module registers; derived from the registry so that a new
# one cannot be added without meeting the checks below.
_REGISTERED = tuple(
    name
    for name in simply_config_lib.ExperimentConfigRegistry.keys()
    if name.startswith('qwen3p8')
)

# Released key -> the Simply field that carries it. `None` means the value is
# checked by a dedicated assertion below rather than by attribute equality.
_MAPPED = {
    'attn_output_gate': 'attn_output_gate',
    'full_attention_interval': 'full_attention_interval',
    'head_dim': 'per_head_dim',
    'hidden_size': 'model_dim',
    'intermediate_size': 'ffn_expand_dim',
    'layer_types': None,
    'linear_conv_kernel_dim': 'linear_conv_kernel_dim',
    'linear_key_head_dim': 'linear_key_head_dim',
    'linear_num_key_heads': 'linear_num_key_heads',
    'linear_num_value_heads': 'linear_num_value_heads',
    'linear_value_head_dim': 'linear_value_head_dim',
    'max_position_embeddings': 'seq_len',
    'num_attention_heads': 'n_heads',
    'num_hidden_layers': 'n_layers',
    'num_key_value_heads': 'n_kv_heads',
    'rms_norm_eps': 'rms_norm_epsilon',
    'rope_parameters': None,
    'tie_word_embeddings': 'use_tied_embedding',
    'attention_bias': None,
    'vocab_size': 'vocab_size',
}


class _Difference(NamedTuple):
  """A released value this port deliberately does not adopt."""

  released: Any
  ours: Any
  why: str


# Released key -> what the release says, what this port does, and why.
_DELIBERATELY_DIFFERENT = {
    'mtp_num_hidden_layers': _Difference(
        released=1,
        ours=0,
        why=(
            'The multi-token-prediction head is never run by the decode path;'
            ' its tensors are dropped at restore.'
        ),
    ),
    'pad_token_id': _Difference(
        released=None,
        ours=config_lib.PAD_ID,
        why=(
            'The release leaves it null; Simply pads batches, so it uses'
            ' <|endoftext|> (= the eos/bos id).'
        ),
    ),
}

# Released key -> why this port does not carry it as a field.
_NOT_REPRESENTED = {
    'attention_dropout': 'Inference-only port.',
    'bos_token_id': 'The chat format supplies the prefix.',
    'dtype': '`activation_dtype_name` is a deployment choice.',
    'eos_token_id': 'The chat format owns the stop tokens.',
    'hidden_act': "Asserted to be 'silu' = `ffn_activation`.",
    'initializer_range': 'Inference-only port: weights come from a release.',
    'mamba_ssm_dtype': 'Asserted to be float32 = `gdn_compute_dtype`.',
    'model_type': 'Asserted by `config_from_hf`.',
    'mtp_use_dedicated_embeddings': 'Only read when there is an MTP head.',
    'output_gate_type': (
        "Asserted to be 'swish', which names the GatedDeltaNet output-norm"
        ' gate (silu, hardcoded in both implementations), not the attention'
        ' gate (sigmoid).'
    ),
    'partial_rotary_factor': 'Duplicated inside `rope_parameters`.',
    'use_cache': 'Simply always caches when decoding.',
}


def _released_text_config() -> dict[str, Any]:
  with open(os.path.join(_TESTDATA, 'qwen3p8_27b_text_config.json')) as f:
    return json.load(f)


class ReleasedConfigFidelityTest(parameterized.TestCase):
  """`qwen3p8_27b` against the released `text_config`, key by key."""

  def setUp(self):
    super().setUp()
    self.hf = _released_text_config()
    self.config = config_lib.qwen3p8_27b()

  def test_every_released_key_is_accounted_for(self):
    tables = set(_MAPPED) | set(_DELIBERATELY_DIFFERENT) | set(
        _NOT_REPRESENTED
    )
    self.assertEmpty(
        set(self.hf) - tables,
        'The release grew a key that no table in this test mentions; map it,'
        ' or record why it is not represented.',
    )
    self.assertEmpty(
        tables - set(self.hf),
        'A table mentions a key the release does not have.',
    )

  def test_mapped_keys_match(self):
    for hf_key, field in _MAPPED.items():
      if field is None:
        continue
      with self.subTest(hf_key):
        self.assertEqual(getattr(self.config, field), self.hf[hf_key])

  def test_layer_schedule_matches(self):
    self.assertEqual(
        self.config.resolved_layer_types(), tuple(self.hf['layer_types'])
    )
    self.assertLen(self.config.resolved_layer_types(), 64)
    self.assertEqual(
        sum(
            t == config_lib.FULL_ATTENTION
            for t in self.config.resolved_layer_types()
        ),
        16,
    )

  def test_rope_matches(self):
    rope = self.hf['rope_parameters']
    encoding = self.config.position_encoding
    self.assertEqual(encoding.max_timescale, rope['rope_theta'])
    self.assertEqual(encoding.rotary_fraction, rope['partial_rotary_factor'])
    self.assertEqual(encoding.mrope_section, tuple(rope['mrope_section']))
    # 0.25 * 256 = 64 rotary dims = 32 pairs = sum(11, 11, 10).
    self.assertEqual(
        sum(encoding.mrope_section),
        int(self.config.per_head_dim * rope['partial_rotary_factor']) // 2,
    )

  def test_deliberate_differences(self):
    for hf_key, difference in _DELIBERATELY_DIFFERENT.items():
      with self.subTest(hf_key):
        self.assertEqual(self.hf[hf_key], difference.released, difference.why)
    # The values this port uses instead, against their own justification
    # rather than against the table that states them.
    self.assertEqual(config_lib.PAD_ID, self.hf['eos_token_id'])
    self.assertEqual(config_lib.PAD_ID, self.hf['bos_token_id'])
    self.assertEqual(self.config.pad_id, config_lib.PAD_ID)
    # No MTP head: nothing in the parameter tree is named for one.
    self.assertEmpty(
        [k for k in dataclasses.asdict(self.config) if k.startswith('mtp')]
    )

  def test_not_represented_values_are_the_ones_assumed(self):
    self.assertEqual(self.hf['hidden_act'], self.config.ffn_activation)
    self.assertEqual(self.hf['mamba_ssm_dtype'], self.config.gdn_compute_dtype)
    self.assertEqual(self.hf['output_gate_type'], 'swish')
    self.assertFalse(self.hf['attention_bias'])
    self.assertEqual(self.hf['model_type'], 'qwen3_5_text')

  def test_simply_defaults_that_would_be_wrong_are_pinned(self):
    """The inherited defaults that silently change the numbers."""
    # 30.0 would return 30*tanh(logits/30): invisible to greedy decoding,
    # KL(HF||Simply) 4.2e-4 -> 9.2e-2 for every sampled eval.
    self.assertEqual(self.config.output_logits_soft_cap, -1.0)
    self.assertEqual(self.config.attn_soft_cap, -1.0)
    # `Qwen3_5RMSNorm` is (1 + w) with w initialised to zeros.
    self.assertTrue(self.config.norm_scale_plus_one)
    self.assertIsNone(self.config.embedding_lookup_scale)
    self.assertFalse(self.config.use_post_ln)
    self.assertFalse(self.config.use_per_dim_scale)
    self.assertTrue(self.config.use_qk_norm)
    self.assertFalse(self.config.ffn_use_bias)
    self.assertFalse(self.config.output_layer_use_bias)
    self.assertFalse(self.config.use_tied_embedding)

  def test_config_from_hf_reproduces_the_registered_config(self):
    """The registered config is `config_from_hf` of the released file."""
    from_file = config_lib.config_from_hf(
        self.hf,
        init_ckpt_dir=config_lib.QWEN3P8_27B_CKPT_DIR,
        init_ckpt_step=-1,
        sharding_config=simply_config_lib.BaseSharding(),
    )
    self.assertEqual(from_file, self.config)


class ConfigFromHfDiscriminationTest(parameterized.TestCase):
  """Every variant this port does not implement must raise, not be ignored."""

  @parameterized.named_parameters(
      ('attention_gate_activation', dict(output_gate_type='sigmoid')),
      ('attention_bias', dict(attention_bias=True)),
      ('ffn_activation', dict(hidden_act='gelu')),
      ('gdn_dtype', dict(mamba_ssm_dtype='bfloat16')),
      ('model_type', dict(model_type='qwen3')),
      ('tied_embeddings', dict(tie_word_embeddings=True)),
  )
  def test_unimplemented_variant_raises(self, patch):
    hf = _released_text_config() | patch
    with self.assertRaises(ValueError):
      config_lib.config_from_hf(hf)

  @parameterized.named_parameters(
      ('scaled_rope', dict(rope_type='linear')),
      ('non_interleaved_mrope', dict(mrope_interleaved=False)),
  )
  def test_unimplemented_rope_raises(self, patch):
    hf = _released_text_config()
    hf['rope_parameters'] = hf['rope_parameters'] | patch
    with self.assertRaises(ValueError):
      config_lib.config_from_hf(hf)

  def test_off_schedule_layer_types_raises(self):
    hf = _released_text_config()
    hf['layer_types'] = list(hf['layer_types'])
    hf['layer_types'][0] = config_lib.FULL_ATTENTION
    with self.assertRaises(ValueError):
      config_lib.config_from_hf(hf)

  def test_overrides_are_applied_last(self):
    config = config_lib.config_from_hf(
        _released_text_config(),
        n_layers=8,
        layer_types=config_lib.layer_types(8),
    )
    self.assertEqual(config.n_layers, 8)


class RegisteredConfigsTest(parameterized.TestCase):

  def test_the_frozen_names_are_registered(self):
    self.assertIn('qwen3p8_27b', _REGISTERED)

  @parameterized.parameters(*_REGISTERED)
  def test_registered_and_well_formed(self, name):
    config = simply_config_lib.ExperimentConfigRegistry.get_config(name)
    self.assertIsInstance(config, config_lib.Qwen38ExperimentConfig)
    self.assertLen(config.resolved_layer_types(), config.n_layers)
    self.assertEqual(config.model_name, 'Qwen38HybridLM')
    self.assertIsNotNone(module.ModuleRegistry.get(config.model_name))
    self.assertEqual(config.vocab_name, 'Qwen3.8')
    self.assertEqual(config.lm_format_name, 'Qwen38Chat')
    self.assertEqual(config.pad_id, config_lib.PAD_ID)

  def test_the_released_config_points_at_converted_weights(self):
    config = config_lib.qwen3p8_27b()
    self.assertEqual(config.init_ckpt_format, 'Qwen38Format')
    self.assertTrue(config.init_ckpt_dir.endswith('Qwen3.8-27B/ORBAX'))
    self.assertStartsWith(config.init_ckpt_dir, simply_config_lib.MODELS_DIR)

  def test_the_tiny_config_needs_no_weights(self):
    config = config_lib.qwen3p8_tiny_test()
    self.assertEmpty(config.init_ckpt_dir)
    self.assertEmpty(config.init_ckpt_format)


class ArithmeticTest(parameterized.TestCase):
  """The closed forms, against `jax.eval_shape` and against the release."""

  def test_param_counts_match_model_init(self):
    config = dataclasses.replace(
        config_lib.qwen3p8_tiny_test(),
        sharding_config=simply_config_lib.BaseSharding(),
    )
    with sharding_lib.set_mesh(
        {'replica': 1, 'data': 1, 'model': 1},
        axis_names=('replica', 'data', 'model'),
    ):
      model = model_lib.Qwen38HybridLM(config)
      shapes = jax.eval_shape(model.init, jax.random.PRNGKey(0))
    counted = sum(
        int(math.prod(leaf.shape)) for leaf in jax.tree.leaves(shapes)
    )
    self.assertEqual(config_lib.param_counts(config).total, counted)

  def test_the_released_model_is_27b(self):
    counts = config_lib.param_counts(config_lib.qwen3p8_27b())
    self.assertAlmostEqual(counts.total / 1e9, 26.9, delta=0.3)
    # 48 GatedDeltaNet layers dominate the non-FFN parameters.
    self.assertGreater(counts.gated_delta_net, counts.attention)

  def test_kv_cache_bytes_is_64kib_per_token(self):
    config = config_lib.qwen3p8_27b()
    per_token = config_lib.kv_cache_bytes(
        config, batch_size=1, max_seq_len=1024
    ) // 1024
    self.assertEqual(per_token, 64 * 1024)

  def test_gdn_state_is_constant_in_the_context(self):
    config = config_lib.qwen3p8_27b()
    self.assertEqual(
        config_lib.gdn_state_bytes(config, batch_size=8),
        8 * config_lib.gdn_state_bytes(config, batch_size=1),
    )


if __name__ == '__main__':
  absltest.main()

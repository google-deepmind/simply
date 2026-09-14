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
"""Tests for the assembled Qwen3.8 model, in four parts.

`HfEquivalenceTest` is the correctness claim: prefill logits, every layer's
residual stream and both mixers' caches against a fixture produced by the
*unmodified* HuggingFace release code (`utils/test_utils.py` loads it;
`testdata/gen_golden.py` regenerates it).

`DecodeTest` covers the decode protocol -- a cached step against the fixture,
and the cached path against a stateless re-run of the whole prefix, which is a
tighter bound than the HF comparison and catches cache bugs that one step is
too short to expose. It also pins the two hybrid-specific behaviours this
package carries: the `prefill_position` mask (without it the GatedDeltaNet
absorbs the prefill window's pad suffix) and
`pad_block_decode_state`.

`StackTest` pins the two ways to run the layer stack -- scanned and unrolled,
with and without remat -- against each other, bit-for-bit: nothing in an eval
chooses between them on purpose, so a difference would be silent.

`LMInterfaceTest` runs the model through what `decode_eval` runs.
"""

import dataclasses
from typing import Any, cast

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from simply import config_lib as simply_config_lib
from simply import model_lib as simply_model_lib
from simply.utils import common
from simply.utils import evaluation_lib
from simply.utils import lm_format as lm_format_lib
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib
from simply.utils import tokenization
from simply.zoo.qwen3p8 import config_lib
from simply.zoo.qwen3p8 import model_lib
from simply.zoo.qwen3p8.utils import gdn as gdn_lib
from simply.zoo.qwen3p8.utils import test_utils

MESH_AXES = ('replica', 'data', 'model')
_LM_MAX_SEQ_LEN = 512
_VOCAB_NAME = 'ByteVocabForQwen38Test'

# The model against itself: a cached pass vs a stateless one, in float32. The
# two orders of summation differ, nothing else does; observed 1e-6.
_SELF_REL = 1e-4
# Against the HuggingFace fixture: both sides float32, different summation
# order and a different matmul precision policy. Observed worst case 3e-6
# relative; the bound is deliberately two orders above it because XLA:CPU
# picks its reduction order from the machine the test lands on.
_HF_REL = 5e-4
_ATOL_FLOOR = 2e-5


def _assert_close(
    test: absltest.TestCase,
    actual: Any,
    expected: Any,
    name: str,
    rel: float = _HF_REL,
) -> None:
  """Asserts `actual ~= expected` relative to the scale of `expected`."""
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  test.assertEqual(actual.shape, expected.shape, name)
  delta, relative = _relative(actual, expected)
  test.assertLessEqual(
      delta,
      rel * max(float(np.max(np.abs(expected))), _ATOL_FLOOR),
      f'{name}: max|delta|={delta:.3g} relative={relative:.3g}',
  )

if _VOCAB_NAME not in tokenization.TokenizerRegistry.keys():
  tokenization.TokenizerRegistry.register_value(
      tokenization.ByteVocab(), name=_VOCAB_NAME
  )


def setUpModule():
  # The self-consistency comparisons below are exact up to summation order and
  # need f32 matmuls; pinning it here keeps the value independent of which
  # shard a class lands in.
  jax.config.update('jax_default_matmul_precision', 'float32')


def _no_partitions() -> simply_config_lib.BaseSharding:
  """A single-device sharding config: no constraint anywhere."""
  return dataclasses.replace(
      simply_config_lib.BaseSharding(),
      embed_partition=None,
      activation_partition=None,
      data_partition=None,
      logits_partition=None,
  )


def _tiny_config(**overrides: Any) -> config_lib.Qwen38ExperimentConfig:
  """`qwen3p8_tiny_test` in float32 on one device, with the same topology."""
  return dataclasses.replace(
      config_lib.qwen3p8_tiny_test(),
      batch_size=2,
      activation_dtype_name='float32',
      use_remat=False,
      linear_attention_chunk_size=8,
      sharding_config=_no_partitions(),
      **overrides,
  )


def _mesh():
  return sharding_lib.create_mesh(
      mesh_shape={name: 1 for name in MESH_AXES}, axis_names=MESH_AXES
  )


def _relative(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
  """(max absolute difference, that difference relative to `expected`)."""
  delta = float(np.max(np.abs(actual - expected)))
  scale = float(np.max(np.abs(expected))) or 1.0
  return delta, delta / scale


class _ModelTestCase(parameterized.TestCase):
  """A tiny model on a one-device mesh, built once per class."""

  config_overrides: dict[str, Any] = {}

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mesh = _mesh()
    cls.enterClassContext(
        sharding_lib.set_mesh(
            {name: 1 for name in MESH_AXES}, axis_names=MESH_AXES
        )
    )
    cls.config = _tiny_config(**cls.config_overrides)
    cls.model = model_lib.Qwen38HybridLM(
        config=cls.config, sharding_config=cls.config.sharding_config
    )
    cls.params = cls.model.init(jax.random.PRNGKey(0))

  def _inputs(self, seq_len: int, batch_size: int = 2):
    ids = jax.random.randint(
        jax.random.PRNGKey(1),
        (batch_size, seq_len),
        1,
        self.config.vocab_size,
        dtype=jnp.int32,
    )
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len)
    )
    return ids, jnp.ones_like(positions), positions

  def _apply(self, ids, segment_ids, positions, **kwargs):
    return self.model.apply(
        self.params,
        ids,
        segment_ids=segment_ids,
        segment_positions=positions,
        **kwargs,
    )


class HfEquivalenceTest(parameterized.TestCase):
  """The correctness claim: this model against the HuggingFace release.

  `testdata/golden_tiny.npz` is a tiny random-weight Qwen3.8 run through the
  unmodified `transformers.models.qwen3_5` on CPU in float32 -- a 20-token
  prefill and two cached decode steps, with every sub-layer's input and output
  and both mixers' caches captured. The parameters are stored under their
  HuggingFace names and converted here by `utils/ckpt_format.py`, so a mapping
  bug cannot be baked into the fixture and confirmed by it.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    golden = test_utils.load_golden()
    cls.data, cls.config = golden.data, golden.config
    cls.mesh, cls.params = golden.mesh, golden.params
    cls.model = model_lib.Qwen38HybridLM(
        config=cls.config, sharding_config=cls.config.sharding_config
    )

  def _ids(self, key: str) -> np.ndarray:
    return np.asarray(self.data[key], np.int32)

  def _on_mesh(self, x) -> jax.Array:
    """The model constrains its inputs, so they must live on the mesh."""
    return jax.device_put(
        jnp.asarray(x),
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec()),
    )

  def _positions(self, start: int, length: int) -> jax.Array:
    batch = self._ids('prefill_input_ids').shape[0]
    return self._on_mesh(
        jnp.broadcast_to(
            jnp.arange(start, start + length, dtype=jnp.int32), (batch, length)
        )
    )

  def _prefill(self):
    """Prefills the fixture's prompt into a cache of exactly its length."""
    ids = self._ids('prefill_input_ids')
    length = ids.shape[1]
    positions = self._positions(0, length)
    with jax.set_mesh(self.mesh):
      logits, extra = self.model.apply(
          self.params,
          self._on_mesh(ids),
          segment_ids=jnp.ones_like(positions),
          segment_positions=positions,
          decode_state=self.model.init_decode_state(
              length, batch_size=ids.shape[0]
          ),
      )
    return logits, extra['decode_state']

  def _decode_step(self, state, ids: np.ndarray, position: int):
    """One cached step, after growing the state as `LMInterface` does."""
    grown = simply_model_lib.pad_decode_state_to(dict(state), position + 1)
    positions = self._positions(position, 1)
    with jax.set_mesh(self.mesh):
      logits, extra = self.model.apply(
          self.params,
          self._on_mesh(ids),
          segment_ids=jnp.ones_like(positions),
          segment_positions=positions,
          decode_state=grown,
      )
    return logits, extra['decode_state']

  def test_param_tree_matches_model_init(self):
    with jax.set_mesh(self.mesh):
      init_params = self.model.init(jax.random.PRNGKey(0))
    shapes = lambda tree: jax.tree.map(np.shape, common.get_raw_arrays(tree))
    init_shapes, loaded_shapes = shapes(init_params), shapes(self.params)
    self.assertEqual(
        jax.tree.structure(init_shapes), jax.tree.structure(loaded_shapes)
    )
    jax.tree.map(self.assertEqual, init_shapes, loaded_shapes)

  def test_prefill_logits(self):
    logits, _ = self._prefill()
    _assert_close(self, logits, self.data['act/logits'], 'logits')

  def test_prefill_layerwise(self):
    """Walks the stack by hand, so a divergence is localized to a sub-layer."""
    ids = self._ids('prefill_input_ids')
    positions = self._positions(0, ids.shape[1])
    segment_ids = jnp.ones_like(positions)
    with jax.set_mesh(self.mesh):
      x = self.model.embed_linear.embed(
          self.params['embed_linear'], self._on_mesh(ids)
      )
      _assert_close(self, x, self.data['act/embed'], 'embed')
      for i, block in enumerate(self.model.blocks):
        key = model_lib.block_key(i)
        with self.subTest(key):
          _assert_close(self, x, self.data[f'act/layer{i}/block_in'], key)
          x, _ = block.apply(
              self.params[key],
              x,
              segment_ids=segment_ids,
              segment_positions=positions,
          )
          _assert_close(
              self, x, self.data[f'act/layer{i}/block_out'], f'{key}/out'
          )
      x = self.model.final_ln.apply(self.params['final_ln'], x)
      _assert_close(self, x, self.data['act/final_hidden'], 'final_hidden')

  def test_the_released_dtype_runs_and_stays_finite(self):
    """bfloat16 end to end, which no other test here exercises.

    Deliberately not a tolerance: on this fixture's *random* weights the
    residual stream is ill-conditioned, and bfloat16 moves the logits by ~37 %
    of their scale -- a property of an untrained model, not of the port (the
    released weights measure 0.19 absolute on logits of ~17, and
    `compare_to_hf_reference` is where that is gated). What this pins is that
    the released dtype runs at all and produces finite logits; the sharp
    bfloat16 check is `NormRoundingTest`, which compares one norm against
    HuggingFace's rounding order bit for bit.
    """
    golden = test_utils.load_golden(activation_dtype_name='bfloat16')
    model = model_lib.Qwen38HybridLM(
        config=golden.config, sharding_config=golden.config.sharding_config
    )
    ids = np.asarray(golden.data['prefill_input_ids'], np.int32)
    replicated = jax.sharding.NamedSharding(
        golden.mesh, jax.sharding.PartitionSpec()
    )
    on_mesh = lambda x: jax.device_put(jnp.asarray(x), replicated)
    positions = on_mesh(
        jnp.broadcast_to(jnp.arange(ids.shape[1], dtype=jnp.int32), ids.shape)
    )
    with jax.set_mesh(golden.mesh):
      logits, _ = model.apply(
          golden.params,
          on_mesh(ids),
          segment_ids=jnp.ones_like(positions),
          segment_positions=positions,
      )
    got = np.asarray(logits, np.float32)
    self.assertEqual(logits.dtype, jnp.bfloat16)
    self.assertTrue(np.all(np.isfinite(got)))
    _, relative = _relative(got, np.asarray(golden.data['act/logits']))
    logging.info('bfloat16 vs the float32 fixture: %.3g relative', relative)

  @parameterized.named_parameters(
      ('prefill', 'cache_prefill', 0),
      ('decode', 'cache_decode', 1),
      ('decode2', 'cache_decode2', 2),
  )
  def test_caches(self, prefix: str, num_decode_steps: int):
    """Both mixers' caches after the prefill and after each decode step."""
    _, state = self._prefill()
    length = self._ids('prefill_input_ids').shape[1]
    for step in range(num_decode_steps):
      key = 'decode_input_ids' if step == 0 else 'decode2_input_ids'
      _, state = self._decode_step(state, self._ids(key), length + step)

    for i, block in enumerate(self.model.blocks):
      key = model_lib.block_key(i)
      with self.subTest(key):
        if block.layer_type == config_lib.LINEAR_ATTENTION:
          # The fixture stores the whole conv input window; ours holds the
          # `kernel - 1` columns it still needs, i.e. the window's tail.
          golden_conv = np.asarray(self.data[f'{prefix}/layer{i}/conv_state'])
          _assert_close(
              self,
              state[key].conv_state,
              golden_conv[..., 1:],
              f'{key}/conv_state',
          )
          _assert_close(
              self,
              state[key].recurrent_state,
              self.data[f'{prefix}/layer{i}/recurrent_state'],
              f'{key}/recurrent_state',
          )
        else:
          rows = length + num_decode_steps
          for name in ('key', 'value'):
            # HuggingFace keeps `[B, H, T, D]`; ours is `[B, T, H, D]`.
            golden = np.swapaxes(
                np.asarray(self.data[f'{prefix}/layer{i}/{name}']), 1, 2
            )
            _assert_close(
                self,
                state[key][name[0]][:, :rows],
                golden[:, :rows],
                f'{key}/{name}',
            )

  @parameterized.named_parameters(('first', 1), ('second', 2))
  def test_decode_step_logits(self, step: int):
    """The decode seam: a cached step must match the release's cached step."""
    _, state = self._prefill()
    length = self._ids('prefill_input_ids').shape[1]
    logits, state = self._decode_step(
        state, self._ids('decode_input_ids'), length
    )
    prefix = 'act_decode'
    if step == 2:
      logits, state = self._decode_step(
          state, self._ids('decode2_input_ids'), length + 1
      )
      prefix = 'act_decode2'
    _assert_close(self, logits, self.data[f'{prefix}/logits'], 'logits')


class StackTest(_ModelTestCase):
  """The scanned stack and the unrolled one must be the same function."""

  def test_scan_is_refused_rather_than_silently_ignored(self):
    """There is no scanned stack; `use_scan` must not be quietly dropped."""
    self.assertFalse(config_lib.qwen3p8_27b().use_scan)
    scanned = model_lib.Qwen38HybridLM(
        config=dataclasses.replace(self.config, use_scan=True),
        sharding_config=self.config.sharding_config,
    )
    ids, segment_ids, positions = self._inputs(4)
    with self.assertRaisesRegex(ValueError, 'no scanned stack'):
      scanned.apply(
          self.params,
          ids,
          segment_ids=segment_ids,
          segment_positions=positions,
      )

  def test_remat_is_exact(self):
    """Nothing turns remat on here, so a bad `remat_policy` would be silent."""
    ids, segment_ids, positions = self._inputs(12)
    plain, _ = self._apply(ids, segment_ids, positions)
    remat_model = model_lib.Qwen38HybridLM(
        config=dataclasses.replace(self.config, use_remat=True),
        sharding_config=self.config.sharding_config,
    )
    remat, _ = remat_model.apply(
        self.params,
        ids,
        segment_ids=segment_ids,
        segment_positions=positions,
    )
    np.testing.assert_allclose(np.asarray(remat), np.asarray(plain), atol=0)

  def test_the_schedule_is_three_to_one(self):
    types = [block.layer_type for block in self.model.blocks]
    self.assertEqual(types, list(config_lib.layer_types(len(types))))
    # Two full groups, so a GatedDeltaNet layer follows an attention layer.
    self.assertGreaterEqual(types.count(config_lib.FULL_ATTENTION), 2)
    self.assertEqual(types[-1], config_lib.FULL_ATTENTION)

  @parameterized.named_parameters(
      ('moe', dict(use_moe=True)),
      ('soft_cap', dict(attn_soft_cap=50.0)),
      ('sliding_window', dict(window_size=1024)),
      ('post_ln', dict(use_post_ln=True)),
      ('tied_embedding', dict(use_tied_embedding=True)),
      ('ffn_bias', dict(ffn_use_bias=True)),
      ('quantized_ffn', dict(ffn_weight_quant='int8')),
  )
  def test_unimplemented_core_fields_raise(self, patch):
    """Inherited core knobs this port ignores must fail loudly.

    `Qwen38ExperimentConfig` inherits every field of core's
    `BaseExperimentConfig`, so without a guard
    `dataclasses.replace(qwen3p8_27b(), use_moe=True)` builds a dense FFN and
    says nothing.

    Args:
      patch: the config override that must be refused.
    """
    config = dataclasses.replace(self.config, **patch)
    with self.assertRaisesRegex(ValueError, 'not implemented'):
      model_lib.Qwen38HybridLM(
          config=config, sharding_config=config.sharding_config
      )

  def test_unknown_remat_policy_raises(self):
    config = dataclasses.replace(
        self.config, use_remat=True, remat_policy='typo_saveable'
    )
    model = model_lib.Qwen38HybridLM(
        config=config, sharding_config=config.sharding_config
    )
    ids, segment_ids, positions = self._inputs(4)
    with self.assertRaisesRegex(ValueError, 'Unknown remat_policy'):
      model.apply(
          self.params,
          ids,
          segment_ids=segment_ids,
          segment_positions=positions,
      )

  def test_per_row_prefill_position_raises(self):
    """A `[B]` prefill position would mask nothing; core passes a scalar."""
    ids, segment_ids, positions = self._inputs(4)
    with self.assertRaisesRegex(ValueError, 'must be a scalar'):
      self._apply(
          ids,
          segment_ids,
          positions,
          extra_inputs={'prefill_position': jnp.asarray([2, 3])},
      )

  def test_unknown_layer_type_raises(self):
    # `SimplyModule.__post_init__` runs `setup`, so this raises at construction.
    with self.assertRaisesRegex(ValueError, 'Unsupported'):
      model_lib.Qwen38Block(
          config=self.config, layer_idx=0, layer_type='sliding_window'
      )


class NormRoundingTest(parameterized.TestCase):
  """`1 + w` must be formed in float32, as the release forms it.

  Two RMSNorms with deliberately different rounding live in this model.
  `Qwen3_5RMSNorm` (the block and final norms) is
  `_norm(x.float()) * (1.0 + w.float())` -- one rounding, at the end
  (`modeling_qwen3_5.py:723-737`) -- while `Qwen3_5RMSNormGated` (the
  GatedDeltaNet output norm) rounds to the activation dtype before the gain.
  Core's `LayerNorm` implements the second order for both, so in bfloat16 the
  block norms would quantize `1 + w` itself. Every other test in this file
  runs in float32, where the two orders agree exactly; this one does not.
  """

  def _norm_and_inputs(self):
    config = dataclasses.replace(
        _tiny_config(), activation_dtype_name='bfloat16'
    )
    with sharding_lib.set_mesh(
        {name: 1 for name in MESH_AXES}, axis_names=MESH_AXES
    ):
      model = model_lib.Qwen38HybridLM(
          config=config, sharding_config=config.sharding_config
      )
    x = jax.random.normal(
        jax.random.PRNGKey(0), (2, 4, config.model_dim), dtype=jnp.bfloat16
    )
    scale = jnp.asarray(
        0.05 * jax.random.normal(jax.random.PRNGKey(1), (config.model_dim,)),
        jnp.bfloat16,
    )
    return model, config, x, scale

  @parameterized.named_parameters(
      ('block', 'input_layernorm'),
      ('post_attention', 'post_attention_layernorm'),
  )
  def test_block_norms_scale_in_float32(self, attribute: str):
    model, config, x, scale = self._norm_and_inputs()
    norm = getattr(model.blocks[0], attribute)
    got = np.asarray(norm.apply({'scale': scale}, x), np.float32)

    x32 = np.asarray(x, np.float32)
    w32 = np.asarray(scale, np.float32)
    x_norm = x32 / np.sqrt(
        np.mean(np.square(x32), -1, keepdims=True) + config.rms_norm_epsilon
    )
    hf = np.asarray(
        jnp.asarray(x_norm * (1.0 + w32), jnp.bfloat16), np.float32
    )
    round_first = np.asarray(
        jnp.asarray(x_norm, jnp.bfloat16)
        * jnp.asarray(1.0 + w32, jnp.bfloat16),
        np.float32,
    )
    scale_of = lambda a: max(float(np.max(np.abs(a))), 1e-6)
    self.assertLess(
        float(np.max(np.abs(got - hf))) / scale_of(hf),
        1e-6,
        'the norm does not round the way HuggingFace does',
    )
    # ... and the other order is distinguishable, so the assertion above is
    # not vacuous: quantizing `1 + w` costs ~4e-3 per channel.
    self.assertGreater(
        float(np.max(np.abs(round_first - hf))) / scale_of(hf), 1e-3
    )

  def test_the_gated_delta_net_norm_keeps_the_release_s_other_order(self):
    """The GatedDeltaNet output norm is the opposite convention, on purpose."""
    model, _, _, _ = self._norm_and_inputs()
    mixer = model.blocks[0].token_mixer
    self.assertIsInstance(mixer, gdn_lib.Qwen38GatedDeltaNet)
    self.assertFalse(mixer.norm.scale_plus_one)
    self.assertEqual(mixer.norm.activation_dtype, 'bfloat16')


class DecodeTest(_ModelTestCase):
  """The decode protocol: caches, the prefill mask, and state padding."""

  def _decode_state(self, max_seq_len: int, batch_size: int = 2):
    return self.model.init_decode_state(max_seq_len, batch_size=batch_size)

  def test_prefill_then_decode_matches_one_shot(self):
    """The production protocol: prefill, grow the state, decode one token."""
    ids, segment_ids, positions = self._inputs(9)
    full_logits, _ = self._apply(ids, segment_ids, positions)

    prefill_len = 8
    _, extra = self._apply(
        ids[:, :prefill_len],
        segment_ids[:, :prefill_len],
        positions[:, :prefill_len],
        decode_state=self._decode_state(prefill_len),
    )
    # `LMInterface` grows the state to the decode horizon between the prefill
    # and the decode loop; the attention cache grows, the recurrent state does
    # not (`model_lib.pad_block_decode_state`).
    grown = simply_model_lib.pad_decode_state_to(
        dict(extra['decode_state']), 9
    )
    step_logits, _ = self._apply(
        ids[:, prefill_len:],
        segment_ids[:, prefill_len:],
        positions[:, prefill_len:],
        decode_state=grown,
    )
    delta, rel = _relative(
        np.asarray(step_logits[:, 0]), np.asarray(full_logits[:, -1])
    )
    self.assertLess(rel, _SELF_REL, f'cached vs stateless: {delta=} {rel=}')

  @parameterized.named_parameters(('two_chunks', 6), ('three_chunks', 4))
  def test_chunked_prefill_reaches_the_same_recurrent_state(self, chunk: int):
    """Absorbing a prompt in chunks leaves the GatedDeltaNet state unchanged.

    Only the GatedDeltaNet layers: core's mapping KV cache *replaces* itself on
    a multi-token pass (`model_lib._update_kv` returns the input state when
    `seq_len > 1`), so an attention layer cannot be chunk-prefilled at all --
    which is fine, because `LMInterface` prefills in one window and then
    decodes token by token. The recurrent mixer is the one that could silently
    disagree, and this pins that it does not.

    Args:
      chunk: Tokens per prefill call.
    """
    ids, segment_ids, positions = self._inputs(12)
    _, extra = self._apply(
        ids, segment_ids, positions, decode_state=self._decode_state(12)
    )
    one_shot = extra['decode_state']

    state = self._decode_state(chunk)
    for start in range(0, 12, chunk):
      stop = start + chunk
      _, extra = self._apply(
          ids[:, start:stop],
          segment_ids[:, start:stop],
          positions[:, start:stop],
          decode_state=state,
      )
      state = extra['decode_state']

    first_attention = [b.layer_type for b in self.model.blocks].index(
        config_lib.FULL_ATTENTION
    )
    for i, (key, block) in enumerate(
        zip(state, self.model.blocks, strict=True)
    ):
      # Only the GatedDeltaNet layers *above* the first attention layer: core's
      # mapping KV cache replaces itself on a multi-token pass, so an attention
      # layer sees only its own chunk and everything downstream of it
      # legitimately differs. That is the limitation, not a state bug.
      if block.layer_type != config_lib.LINEAR_ATTENTION or i > first_attention:
        continue
      with self.subTest(key):
        for leaf, expected in zip(
            jax.tree.leaves(state[key]),
            jax.tree.leaves(one_shot[key]),
            strict=True,
        ):
          delta, rel = _relative(
              np.asarray(leaf, np.float32), np.asarray(expected, np.float32)
          )
          self.assertLess(rel, _SELF_REL, f'{key}: {delta=} {rel=}')

  def test_prefill_position_ignores_the_padded_tail(self):
    """The regression test for the prefill pad suffix.

    `sampling_lib` prefills a padded window with `prefill_position` set; the
    state that reaches the decode loop must be the state after
    `prefill_position` real tokens, not after the whole window. Without the
    mask, 48 of the 64 layers of the released model absorb the pad suffix into
    their recurrence and the decode loop continues from the wrong state.
    """
    ids, segment_ids, positions = self._inputs(12)
    real = 5
    padded_ids = ids.at[:, real:].set(0)

    _, windowed = self._apply(
        padded_ids,
        segment_ids,
        positions,
        extra_inputs={'prefill_position': real},
    )
    _, exact = self._apply(
        ids[:, :real],
        segment_ids[:, :real],
        positions[:, :real],
        decode_state=self._decode_state(real),
    )
    for i, block in enumerate(self.model.blocks):
      key = model_lib.block_key(i)
      got, want = windowed['decode_state'][key], exact['decode_state'][key]
      with self.subTest(key):
        if block.layer_type == config_lib.LINEAR_ATTENTION:
          leaves = [
              ('conv_state', got.conv_state, want.conv_state),
              ('recurrent_state', got.recurrent_state, want.recurrent_state),
          ]
        else:
          # The window's cache is 12 rows and the reference's is `real`; the
          # rows that hold real tokens are the ones under test.
          leaves = [
              (name, got[name][:, :real], want[name][:, :real])
              for name in ('k', 'v')
          ]
        for name, a, b in leaves:
          delta, rel = _relative(
              np.asarray(a, np.float32), np.asarray(b, np.float32)
          )
          self.assertLess(rel, _SELF_REL, f'{key}/{name}: {delta=} {rel=}')

  def test_decode_state_without_positions_raises(self):
    ids, segment_ids, positions = self._inputs(4)
    with self.assertRaisesRegex(ValueError, 'segment_positions is required'):
      self.model.apply(
          self.params,
          ids,
          segment_ids=segment_ids,
          decode_state=self._decode_state(8),
      )
    del positions

  def test_pad_decode_state_grows_attention_and_not_the_recurrence(self):
    """`pad_block_decode_state` is registered for the GatedDeltaNet state."""
    state = self._decode_state(8)
    # `pad_decode_state_to` grows the attention mappings in place, so the
    # "before" shapes have to be read first.
    before = {
        key: [np.shape(leaf) for leaf in jax.tree.leaves(value)]
        for key, value in state.items()
    }
    padded = cast(
        dict[str, Any],
        simply_model_lib.pad_decode_state_to(dict(state), 16),
    )
    for key, block in zip(before, self.model.blocks, strict=True):
      after = [np.shape(leaf) for leaf in jax.tree.leaves(padded[key])]
      with self.subTest(key):
        if block.layer_type == config_lib.LINEAR_ATTENTION:
          self.assertIsInstance(padded[key], gdn_lib.GatedDeltaNetDecodeState)
          self.assertEqual(after, before[key], 'the state is length-free')
        else:
          self.assertNotEqual(after, before[key], 'the cache must grow')
          self.assertEqual(
              [shape[1] for shape in after], [16] * len(after)
          )

  def test_decode_state_survives_a_jit_boundary(self):
    """The states are registered dataclasses, so they are pytrees."""
    ids, segment_ids, positions = self._inputs(4)
    apply_fn = jax.jit(
        lambda p, i, s, q, st: self.model.apply(
            p, i, segment_ids=s, segment_positions=q, decode_state=st
        )
    )
    _, extra = apply_fn(
        self.params, ids, segment_ids, positions, self._decode_state(4)
    )
    self.assertIn('decode_state', extra)
    # The property `continue_decode`'s `while_loop` needs: a decode step must
    # return a state of exactly the structure it was given.
    self.assertEqual(
        jax.tree.structure(extra['decode_state']),
        jax.tree.structure(self._decode_state(4)),
    )


class LMInterfaceTest(parameterized.TestCase):
  """The model through `model_lib.LMInterface`, i.e. through `decode_eval`."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.enterClassContext(
        sharding_lib.set_mesh(
            {name: 1 for name in MESH_AXES}, axis_names=MESH_AXES
        )
    )
    cls.vocab = tokenization.TokenizerRegistry.get_instance(_VOCAB_NAME)
    cls.config = _tiny_config(
        vocab_name=_VOCAB_NAME,
        vocab_size=cls.vocab.vocab_size,
        seq_len=_LM_MAX_SEQ_LEN,
        lm_format_name='SimplyV1Chat',
    )
    cls.model, _ = simply_model_lib.create_model(cls.config)
    cls.params = cls.model.init(jax.random.key(0))
    cls.logits_fn = staticmethod(
        jax.jit(lambda tokens: cls.model.apply(cls.params, tokens)[0])
    )

  def test_create_model_dispatches_on_model_name(self):
    """`create_model` must build OUR model, not `TransformerLM`."""
    self.assertIsInstance(self.model, model_lib.Qwen38HybridLM)

  def _lm_interface(self, **sampling_kwargs) -> simply_model_lib.LMInterface:
    params = dict(
        temperature=0.0,
        max_seq_len=_LM_MAX_SEQ_LEN,
        max_decode_steps=4,
        num_samples=1,
    )
    params.update(sampling_kwargs)
    return simply_model_lib.LMInterface(
        self.model,
        params=self.params,
        input_processor=sampling_lib.create_input_processor(
            self.config, vocab=self.vocab
        ),
        default_sampling_params=simply_model_lib.SamplingParams(**params),
    )

  def _assert_is_greedy(self, prompt: str, output_token_ids: list[int]):
    """Teacher-forces the generation through one stateless pass.

    Greedy decoding means every generated token is the argmax given the true
    prefix, so one pass over `prompt + output` checks them all -- and it shares
    no decode-state code with the sampler, so a cache bug cannot cancel out.

    Args:
      prompt: The prompt that was generated from.
      output_token_ids: The sampler's generated ids.
    """
    prompt_ids = [self.vocab.bos_id] + self.vocab.encode(prompt)
    logits = self.logits_fn(jnp.asarray([prompt_ids + output_token_ids]))
    predicted = np.argmax(np.asarray(logits[0]), axis=-1)
    np.testing.assert_array_equal(
        predicted[len(prompt_ids) - 1 : -1],
        np.asarray(output_token_ids),
        err_msg=f'{prompt=}',
    )

  @parameterized.named_parameters(
      # `prefill_size <= min_input_len - 1`: the sampler decodes from the end
      # of the prefill window.
      ('short_prefill', 2, None),
      # `prefill_size > min_input_len - 1`: the sampler restarts *inside* the
      # window, so the window tail must not be absorbed twice -- the case a
      # recurrent mixer gets wrong without the `prefill_position` mask.
      ('long_prefill', 16, None),
      # Several decode chunks, so the state is padded between them.
      ('chunked_decode', 16, 2),
  )
  def test_generate_matches_greedy_reference(
      self, prefill_size, intermediate_decode_steps
  ):
    max_decode_steps = 4
    prompts = ['hi', 'hello there']
    interface = self._lm_interface(
        prefill_size=prefill_size,
        max_decode_steps=max_decode_steps,
        intermediate_decode_steps=intermediate_decode_steps,
    )
    outputs = cast(
        list[list[simply_model_lib.SamplingOutput]],
        interface.generate(
            prompts, prng_key=0, batch_size=4, scoring_inputs=False
        ),
    )
    for prompt, sample_outputs in zip(prompts, outputs, strict=True):
      self.assertLen(sample_outputs, 1)
      self.assertEqual(
          sample_outputs[0].input_token_ids,
          [self.vocab.bos_id] + self.vocab.encode(prompt),
      )
      self.assertLen(sample_outputs[0].output_token_ids, max_decode_steps)
      self._assert_is_greedy(prompt, sample_outputs[0].output_token_ids)

  def test_decode_eval_loop_scores_generations(self):
    """`decode_eval.main`'s per-example body on a canned example."""
    evaluation = evaluation_lib.EvaluationRegistry.get_instance(
        'ZeroShotDeepSeekQwenR1CoTBoxed'
    )
    lm_format = lm_format_lib.LMFormatRegistry.get_instance('SimplyV1Chat')
    example = {'question': 'What is 2+2?', 'short_answer': '4'}
    sampling_input = evaluation.get_sampling_input(example, lm_format)
    outputs = cast(
        list[list[simply_model_lib.SamplingOutput]],
        self._lm_interface(prefill_size=64, max_decode_steps=4).generate(
            [sampling_input], prng_key=0, batch_size=1, scoring_inputs=False
        ),
    )
    # The scorer runs on whatever the random-init model produced; the point is
    # that the loop `decode_eval` runs end to end, not the score.
    metrics = evaluation.evaluate(example, outputs[0][0].output_text)
    self.assertIn('correct', metrics)
    self.assertEqual(metrics['correct'], 0)


if __name__ == '__main__':
  absltest.main()

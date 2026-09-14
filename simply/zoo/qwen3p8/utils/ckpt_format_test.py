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
"""Spec test for the released Qwen3.8 -> Simply checkpoint mapping.

This is the only thing standing between a released-checkpoint rename and a
silently mis-loaded 27B: core fills a parameter it could not find in the
checkpoint from the abstract state and only `logging.warning`s, so
"every leaf is filled" has to be asserted here.

The test is written as a specification, not as a copy of `ckpt_format.py`:

  * `testdata/qwen3p8_27b_tensor_names.json` is ground truth, not a
    transcription: name and shape of all 1199 tensors of the real converted
    27B, read straight off the checkpoint's Orbax `state/_METADATA` (see its
    `provenance`). `ReleasedGeometryTest` pushes exactly those tensors through
    the mapping at the released geometry, and `hf_tensor_shapes` -- the
    config-derived formula the small fixtures are built from -- is checked
    against it name by name and shape by shape.
  * `_expected_mapping` re-derives where each tensor lands, from the two rules
    Simply follows (`EinsumLinear` is `(in, out)` where `nn.Linear` is
    `(out, in)`; head dims stay separate) rather than from the mapping table.
  * the transposes, the head splits and the conv squeeze are additionally
    pinned elementwise on hand-built arrays, so a self-consistent pair of
    wrong reshapes cannot pass.

The model is `qwen3p8_tiny_test`: the released topology at 1/20th the width,
which keeps the whole file a few seconds on CPU.

What this file cannot catch, so that nobody over-trusts it:

  * a *shared* wrong belief in the two same-shape pairs that are named by hand
    on both sides: `mlp.gate_proj`/`mlp.up_proj` (`ffn_0_gate`/`ffn_0`) and
    `embed_tokens.weight`/`lm_head.weight` (both `(vocab, model_dim)`).
    `test_the_embedding_and_the_head_are_not_swapped` closes the second
    against `EmbeddingLinear`'s own semantics; the first is only closed by the
    golden-numerics gate in `model_lib_test.py`.

Every other same-shape pair (`k`/`v`, `q_norm`/`k_norm`, `in_proj_a`/
`in_proj_b`, the two block norms) is driven by a capture group on both sides,
so no one-sided edit can swap them.
"""

import collections
from collections.abc import Mapping
import dataclasses
import functools
import json
import os
import re
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from simply import model_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import sharding as sharding_lib
from simply.zoo.qwen3p8 import config_lib
from simply.zoo.qwen3p8 import model_lib as qwen3p8_model_lib  # pylint: disable=unused-import  Registers `Qwen38HybridLM`, which `config.model_name` names.
from simply.zoo.qwen3p8.utils import ckpt_format

Config = config_lib.Qwen38ExperimentConfig

# Ground truth, read off the released checkpoint; see its `provenance`.
_TESTDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testdata')
_RELEASED_TENSORS_PATH = os.path.join(
    _TESTDATA, 'qwen3p8_27b_tensor_names.json'
)


@functools.cache
def released_tensors() -> Mapping[str, tuple[int, ...]]:
  """Name -> shape of every tensor of the converted Qwen3.8-27B."""
  with open(_RELEASED_TENSORS_PATH) as f:
    released = json.load(f)
  return {name: tuple(shape) for name, shape in released['tensors'].items()}


@functools.cache
def released_provenance() -> Mapping[str, Any]:
  """Where `released_tensors` came from, and what it should contain."""
  with open(_RELEASED_TENSORS_PATH) as f:
    return json.load(f)['provenance']


def released_text_tensors() -> dict[str, tuple[int, ...]]:
  """The released tensors this text model is supposed to restore."""
  return {
      name: shape
      for name, shape in released_tensors().items()
      if not name.startswith(ckpt_format.DROPPED_PREFIXES)
  }


# HF gated-MLP projection -> Simply `FeedForward` leaf.
_MLP_TO_FFN = {'gate': 'ffn_0_gate', 'up': 'ffn_0', 'down': 'ffn_1'}


# --- The released tensor list, derived from the config ----------------------


def _layer_shapes(c: Config, layer_type: str) -> dict[str, tuple[int, ...]]:
  """HF tensor suffix -> shape, for one decoder layer of `layer_type`."""
  if c.ffn_expand_dim is None:
    raise ValueError('Qwen3.8 sizes its FFN with `ffn_expand_dim`.')
  key_dim = c.linear_num_key_heads * c.linear_key_head_dim
  value_dim = c.linear_num_value_heads * c.linear_value_head_dim
  conv_dim = 2 * key_dim + value_dim
  shapes: dict[str, tuple[int, ...]] = {
      'input_layernorm.weight': (c.model_dim,),
      'post_attention_layernorm.weight': (c.model_dim,),
      'mlp.gate_proj.weight': (c.ffn_expand_dim, c.model_dim),
      'mlp.up_proj.weight': (c.ffn_expand_dim, c.model_dim),
      'mlp.down_proj.weight': (c.model_dim, c.ffn_expand_dim),
  }
  if layer_type == config_lib.LINEAR_ATTENTION:
    shapes.update({
        'linear_attn.in_proj_qkv.weight': (conv_dim, c.model_dim),
        'linear_attn.in_proj_z.weight': (value_dim, c.model_dim),
        'linear_attn.in_proj_b.weight': (
            c.linear_num_value_heads,
            c.model_dim,
        ),
        'linear_attn.in_proj_a.weight': (
            c.linear_num_value_heads,
            c.model_dim,
        ),
        'linear_attn.conv1d.weight': (conv_dim, 1, c.linear_conv_kernel_dim),
        'linear_attn.A_log': (c.linear_num_value_heads,),
        'linear_attn.dt_bias': (c.linear_num_value_heads,),
        'linear_attn.norm.weight': (c.linear_value_head_dim,),
        'linear_attn.out_proj.weight': (c.model_dim, value_dim),
    })
  else:
    # q_proj is twice as wide as usual: the query and the attention output
    # gate are one tensor, `[q | gate]` per head. `attn_output_gate=False` is
    # a variant neither the release nor `utils/attn.py` has.
    shapes.update({
        'self_attn.q_proj.weight': (
            2 * c.n_heads * c.per_head_dim,
            c.model_dim,
        ),
        'self_attn.k_proj.weight': (c.n_kv_heads * c.per_head_dim, c.model_dim),
        'self_attn.v_proj.weight': (c.n_kv_heads * c.per_head_dim, c.model_dim),
        'self_attn.o_proj.weight': (c.model_dim, c.n_heads * c.per_head_dim),
        'self_attn.q_norm.weight': (c.per_head_dim,),
        'self_attn.k_norm.weight': (c.per_head_dim,),
    })
  return shapes


def hf_tensor_shapes(c: Config) -> dict[str, tuple[int, ...]]:
  """Every text tensor the released checkpoint holds, name -> shape."""
  shapes: dict[str, tuple[int, ...]] = {
      'model.language_model.embed_tokens.weight': (c.vocab_size, c.model_dim),
      'model.language_model.norm.weight': (c.model_dim,),
      'lm_head.weight': (c.vocab_size, c.model_dim),
  }
  for i, layer_type in enumerate(c.resolved_layer_types()):
    for suffix, shape in _layer_shapes(c, layer_type).items():
      shapes[f'model.language_model.layers.{i}.{suffix}'] = shape
  return shapes


def mtp_tensor_shapes(c: Config) -> dict[str, tuple[int, ...]]:
  """The released one-layer multi-token-prediction head; this port drops it."""
  shapes: dict[str, Any] = {
      # `fc` consumes [norm(embedding) ; norm(hidden)].
      'mtp.fc.weight': (c.model_dim, 2 * c.model_dim),
      'mtp.pre_fc_norm_embedding.weight': (c.model_dim,),
      'mtp.pre_fc_norm_hidden.weight': (c.model_dim,),
      'mtp.norm.weight': (c.model_dim,),
  }
  for suffix, shape in _layer_shapes(c, config_lib.FULL_ATTENTION).items():
    shapes[f'mtp.layers.0.{suffix}'] = shape
  return shapes


def _synthesize(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, jax.Array]:
  """Distinct, non-symmetric values per tensor, so mis-mappings show up."""
  out = {}
  for i, (name, shape) in enumerate(sorted(shapes.items())):
    size = int(np.prod(shape))
    out[name] = jnp.reshape(
        jnp.arange(size, dtype=jnp.float32) / size + i, shape
    )
  return out


# --- Where each tensor should land, re-derived independently ----------------


def _expected_mapping(
    c: Config, name: str, value: np.ndarray
) -> tuple[str, np.ndarray] | None:
  """Re-derives, independently of `ckpt_format`, where one HF tensor lands.

  Simply stores `EinsumLinear` weights as `(in, out)` where `nn.Linear` is
  `(out, in)`, and keeps the head dim separate: `(model_dim, heads, head_dim)`.
  Norm scales, the embedding table, the LM head and the GatedDeltaNet per-head
  scalars are stored verbatim.

  Args:
    c: The config the tensors were synthesized from.
    name: The HF tensor name.
    value: The HF tensor.

  Returns:
    (Simply parameter path, expected value), or None if the tensor has no
    counterpart in the Simply text model.
  """
  if name.startswith(('model.visual.', 'mtp.')):
    return None
  if name == 'lm_head.weight':
    return 'params/embed_linear/w', value

  key = name.removeprefix('model.language_model.')
  if key == 'embed_tokens.weight':
    return 'params/embed_linear/embed', value
  if key == 'norm.weight':
    return 'params/final_ln/scale', value

  m = re.fullmatch(r'layers\.(\d+)\.(.+)', key)
  assert m is not None, name
  block, rest = f'params/block_{m[1]}', m[2]
  mixer = f'{block}/token_mixer'

  if m := re.fullmatch(
      r'(input_layernorm|post_attention_layernorm)\.weight', rest
  ):
    return f'{block}/{m[1]}/scale', value
  if m := re.fullmatch(r'mlp\.(gate|up|down)_proj\.weight', rest):
    return f'{block}/ffn/{_MLP_TO_FFN[m[1]]}/w', value.T
  if m := re.fullmatch(r'self_attn\.([qk])_norm\.weight', rest):
    return f'{mixer}/{m[1]}_norm/scale', value
  if rest == 'self_attn.q_proj.weight':
    return f'{mixer}/q_proj/w', value.reshape(
        c.n_heads, 2 * c.per_head_dim, c.model_dim
    ).transpose(2, 0, 1)
  if m := re.fullmatch(r'self_attn\.([kv])_proj\.weight', rest):
    return f'{mixer}/{m[1]}_proj/w', value.reshape(
        c.n_kv_heads, c.per_head_dim, c.model_dim
    ).transpose(2, 0, 1)
  if rest == 'self_attn.o_proj.weight':
    return f'{mixer}/o_proj/w', value.reshape(
        c.model_dim, c.n_heads, c.per_head_dim
    )
  if m := re.fullmatch(
      r'linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)'
      r'\.weight',
      rest,
  ):
    return f'{mixer}/{m[1]}/w', value.T
  if rest == 'linear_attn.conv1d.weight':  # (channels, 1, taps).
    return f'{mixer}/conv1d/w', value[:, 0, :]
  if m := re.fullmatch(r'linear_attn\.(A_log|dt_bias)', rest):
    return f'{mixer}/{m[1]}', value
  if rest == 'linear_attn.norm.weight':
    return f'{mixer}/norm/scale', value
  raise AssertionError(f'the test table maps no tensor named {name}')


def _set_default_mesh() -> None:
  """A one-device mesh.

  `Qwen2Format._split_head` constrains the sharding of what it reshapes, which
  needs a mesh, and every array it is handed must have been created under the
  same one.
  """
  sharding_lib.set_default_mesh_shape(
      mesh_shape=(1, 1, 1, 1),
      axis_names=('replica', 'data', 'seq', 'model'),
  )


def _model(config: Config) -> Any:
  """The Qwen3.8 `config.model_name` names, built as the eval binary does."""
  _set_default_mesh()
  model, _ = model_lib.create_model(config)
  return model


def _abstract_state(config: Config) -> Any:
  """The parameter tree of `config`, as shapes and dtypes only."""
  model = _model(config)
  # `get_abstract_params` is `jax.eval_shape` over `model.init`; the raw form
  # (no `AnnotatedArray` wrappers) is what core hands to `transforms`.
  return common.get_raw_arrays({'params': ckpt_lib.get_abstract_params(model)})


@functools.cache
def _tiny_fixture() -> tuple[Config, Any, Mapping[str, jax.Array]]:
  """The config, its abstract parameter tree and a synthetic release.

  Built once: nothing mutates it, and rebuilding the tree per test method
  dominates the runtime of the file.

  Returns:
    (config, abstract state, the stored checkpoint keyed by HF tensor name).
  """
  _set_default_mesh()
  config = dataclasses.replace(
      config_lib.qwen3p8_tiny_test(), init_ckpt_dir=''
  )
  # Real vision names at stand-in sizes: the drop is by name, and the
  # released tower is 333 tensors of up to 23 M values.
  vision = {
      name: (4, 3)
      for name in sorted(released_tensors())
      if name.startswith('model.visual.')
  }
  # One pass, so that no two tensors anywhere hold the same values.
  stored = _synthesize(
      hf_tensor_shapes(config) | mtp_tensor_shapes(config) | vision
  )
  return config, _abstract_state(config), stored


class ReleasedTensorListTest(absltest.TestCase):
  """The config derives exactly the tensor list the release actually has."""

  def test_the_config_derives_the_released_tensor_list(self):
    config = config_lib.qwen3p8_27b()
    derived = hf_tensor_shapes(config) | mtp_tensor_shapes(config)
    released = {
        name: shape
        for name, shape in released_tensors().items()
        if not name.startswith('model.visual.')
    }
    self.assertEqual(derived, released)

  def test_the_ground_truth_file_is_what_its_provenance_says(self):
    counts = released_provenance()['counts']
    tensors = released_tensors()
    self.assertLen(tensors, counts['total'])
    for prefix, key in (('model.visual.', 'visual'), ('mtp.', 'mtp')):
      self.assertLen(
          [name for name in tensors if name.startswith(prefix)], counts[key]
      )
    self.assertLen(released_text_tensors(), counts['text'])

  def test_every_layer_is_one_of_the_two_kinds(self):
    """Nothing in the release is outside the 3:1 GatedDeltaNet:attention rule."""
    mixers = collections.Counter(
        'linear_attn' if '.linear_attn.' in name else 'self_attn'
        for name in released_text_tensors()
        if '.linear_attn.' in name or '.self_attn.' in name
    )
    types = collections.Counter(config_lib.qwen3p8_27b().resolved_layer_types())
    self.assertEqual(mixers['linear_attn'], 9 * types['linear_attention'])
    self.assertEqual(mixers['self_attn'], 6 * types['full_attention'])


class ReleasedGeometryTest(absltest.TestCase):
  """The mapping, run on the real released tensor list at the real geometry.

  `jax.eval_shape` traces `transforms` over the 1199 released tensors without
  allocating any of the 27B's 54 GB: what it returns is exactly the names and
  shapes a restore would produce. It is one test method rather than four
  because building the 64-layer abstract tree and tracing the mapping over it
  is the slowest thing in this file, and every assertion below is about that
  one result.
  """

  def test_the_released_27b_maps_onto_every_parameter(self):
    config = config_lib.qwen3p8_27b()
    abstract_state = _abstract_state(config)
    stored = {
        name: jax.ShapeDtypeStruct(shape, jnp.bfloat16)
        for name, shape in released_tensors().items()
    }

    # No warning: the only tensors without a target are the vision tower and
    # the MTP head, and those are dropped deliberately, by name.
    with self.assertNoLogs(level='WARNING'):
      restored = jax.eval_shape(
          lambda s: ckpt_format.Qwen38Format().transforms(s, abstract_state),
          stored,
      )

    got = ocp.tree.to_flat_dict(restored, sep='/')
    want = ocp.tree.to_flat_dict(abstract_state, sep='/')
    with self.subTest('every parameter is filled'):
      self.assertEmpty(sorted(set(want) - set(got)))
    with self.subTest('nothing the model cannot hold is produced'):
      self.assertEmpty(sorted(set(got) - set(want)))
    with self.subTest('one released tensor per parameter'):
      # Two tensors landing on one parameter would hide one of them.
      self.assertLen(got, len(released_text_tensors()))
    with self.subTest('shapes'):
      for path, abstract in want.items():
        self.assertEqual(got[path].shape, abstract.shape, msg=f'at {path}')


class MappingTest(parameterized.TestCase):
  """Every released text tensor lands on exactly one leaf, and vice versa."""

  def setUp(self):
    super().setUp()
    self.config, self.abstract_state, self.stored = _tiny_fixture()

  def _transform(self) -> dict[str, Any]:
    restored = ckpt_format.Qwen38Format().transforms(
        self.stored, self.abstract_state
    )
    return ocp.tree.to_flat_dict(restored, sep='/')

  def _convert(self, stored: Mapping[str, Any]) -> dict[str, Any]:
    """`convert`, not `transforms`: a subset does not cover the model."""
    converted = ckpt_format.Qwen38Format().convert(
        stored, ckpt_format.target_from_abstract_state(self.abstract_state)
    )
    return ocp.tree.to_flat_dict(converted, sep='/')

  def _want(self) -> dict[str, Any]:
    return ocp.tree.to_flat_dict(self.abstract_state, sep='/')

  def test_every_simply_leaf_is_filled(self):
    got, want = self._transform(), self._want()
    self.assertEmpty(
        sorted(set(want) - set(got)), msg='Simply parameters left unfilled'
    )
    self.assertEmpty(
        sorted(set(got) - set(want)), msg='parameters not in the Simply model'
    )

  def test_every_released_tensor_lands_where_it_should(self):
    expected = {}
    for name, array in self.stored.items():
      if mapped := _expected_mapping(self.config, name, np.asarray(array)):
        path, value = mapped
        expected[path] = value

    got, want = self._transform(), self._want()
    self.assertCountEqual(got.keys(), expected.keys())
    # Two tensors mapping to one path would collapse on both sides and hide.
    self.assertLen(got, sum(
        _expected_mapping(self.config, name, np.asarray(array)) is not None
        for name, array in self.stored.items()
    ))
    for path, value in expected.items():
      self.assertEqual(
          got[path].shape, want[path].shape, msg=f'shape mismatch at {path}'
      )
      np.testing.assert_array_equal(
          np.asarray(got[path]), value, err_msg=f'wrong transform for {path}'
      )

  @parameterized.named_parameters(
      ('vision', 'model.visual.'), ('mtp', 'mtp.')
  )
  def test_dropped_families_are_dropped_silently(self, prefix: str):
    """Deliberate: neither has a counterpart, so neither may warn."""
    self.assertIn(prefix, ckpt_format.DROPPED_PREFIXES)
    dropped = {k: v for k, v in self.stored.items() if k.startswith(prefix)}
    self.assertNotEmpty(dropped)
    with self.assertNoLogs(level='WARNING'):
      self.assertEmpty(self._convert(dropped))

  def test_an_unrecognised_tensor_warns(self):
    """The tail that would otherwise hide a released rename."""
    name = 'model.language_model.layers.0.linear_attn.in_proj_qkvz.weight'
    with self.assertLogs(level='WARNING') as logs:
      self.assertEmpty(
          self._convert({name: jnp.zeros((4, self.config.model_dim))})
      )
    self.assertIn('in_proj_qkvz', '\n'.join(logs.output))

  def test_a_layer_the_deployment_does_not_have_is_skipped(self):
    """Restoring the 64-layer release into a 4-layer model must not invent.

    Silently, and by the rule matching: a warning here would mean the rule
    table no longer recognises the tensor, which is a different bug.
    """
    name = 'model.language_model.layers.60.mlp.up_proj.weight'
    with self.assertNoLogs(level='WARNING'):
      self.assertEmpty(self._convert({name: jnp.zeros((2, 3))}))

  def test_transforms_rejects_a_checkpoint_that_leaves_a_parameter_unfilled(
      self,
  ):
    """Core would only `logging.warning` and run on uninitialised weights."""
    incomplete = {
        name: value
        for name, value in self.stored.items()
        if 'in_proj_qkv' not in name
    }
    with self.assertRaisesRegex(ValueError, 'no tensor in the checkpoint'):
      ckpt_format.Qwen38Format().transforms(incomplete, self.abstract_state)

  def test_the_embedding_and_the_head_are_not_swapped(self):
    """Both are `(vocab, model_dim)`, so only their use tells them apart.

    `EmbeddingLinear` looks tokens up in `embed` and projects to logits with
    `w`; a swap survives every shape and key assertion in this file.
    """
    params = ckpt_format.convert_from_mapping(self.stored, self.config)
    embed_linear = _model(self.config).embed_linear
    table = self.stored['model.language_model.embed_tokens.weight']
    head = self.stored['lm_head.weight']

    rows = embed_linear.embed(params['embed_linear'], jnp.asarray([[3, 7]]))
    np.testing.assert_array_equal(
        np.asarray(rows[0]), np.asarray(table[jnp.asarray([3, 7])], rows.dtype)
    )
    # A one-hot input picks one column of the head, exactly even in bfloat16.
    one_hot = jnp.eye(self.config.model_dim)[2][None, None, :]
    logits = embed_linear.apply(params['embed_linear'], one_hot)
    np.testing.assert_array_equal(
        np.asarray(logits[0, 0]), np.asarray(head[:, 2], logits.dtype)
    )

  @parameterized.named_parameters(
      # `config.init_ckpt_format` -- what `decode_eval` passes.
      ('by_name', ckpt_format.FORMAT_NAME),
      # No format: core reads the one stamped into the checkpoint metadata.
      ('from_metadata', ''),
  )
  def test_restore_from_checkpoint_fills_every_param(self, name: str):
    """The production path, not just `transforms`."""
    ckpt_dir = self.create_tempdir()
    manager = ocp.CheckpointManager(ckpt_dir.full_path)
    ckpt_lib.save_checkpoint(
        manager, self.stored, 0, ckpt_format=ckpt_format.Qwen38Format()
    )
    manager.wait_until_finished()

    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir.full_path, self.abstract_state, ckpt_format=name
    )

    got = ocp.tree.to_flat_dict(common.get_raw_arrays(restored), sep='/')
    want = self._want()
    self.assertCountEqual(got.keys(), want.keys())
    for path, abstract in want.items():
      # A leaf the loader failed to fill is passed through as its abstract
      # `ShapeDtypeStruct`.
      self.assertNotIsInstance(
          got[path], jax.ShapeDtypeStruct, msg=f'{path} was not restored'
      )
      self.assertEqual(got[path].shape, abstract.shape, msg=f'at {path}')


class ConvertFromMappingTest(absltest.TestCase):
  """The offline entry point: the same mapping, described by the config."""

  def setUp(self):
    super().setUp()
    self.config, self.abstract_state, self.stored = _tiny_fixture()

  def test_matches_the_restore_path(self):
    from_config = ckpt_format.convert_from_mapping(self.stored, self.config)
    from_target = ckpt_format.Qwen38Format().transforms(
        self.stored, self.abstract_state
    )['params']
    flat_config = ocp.tree.to_flat_dict(from_config, sep='/')
    flat_target = ocp.tree.to_flat_dict(from_target, sep='/')
    self.assertCountEqual(flat_config.keys(), flat_target.keys())
    for path, value in flat_config.items():
      np.testing.assert_array_equal(
          np.asarray(value), np.asarray(flat_target[path]), err_msg=path
      )

  def test_accepts_the_text_only_prefix_and_numpy_arrays(self):
    """`Qwen3_5ForCausalLM` names its tensors `model.*`, not the release's."""
    renamed = {
        name.replace('model.language_model.', 'model.'): np.asarray(value)
        for name, value in self.stored.items()
    }
    converted = ckpt_format.convert_from_mapping(
        renamed, self.config, dtype=jnp.bfloat16
    )
    flat = ocp.tree.to_flat_dict(converted, sep='/')
    self.assertIn('block_0/token_mixer/in_proj_qkv/w', flat)
    self.assertEqual(flat['embed_linear/embed'].dtype, jnp.bfloat16)

  def test_drops_the_leaves_of_the_other_kind_of_mixer(self):
    """`wants` is layer-type aware: a GDN tensor at a full-attention layer."""
    full = self.config.resolved_layer_types().index(config_lib.FULL_ATTENTION)
    name = f'model.language_model.layers.{full}.linear_attn.in_proj_z.weight'
    converted = ckpt_format.convert_from_mapping(
        {name: jnp.zeros((4, self.config.model_dim))}, self.config
    )
    self.assertEmpty(ocp.tree.to_flat_dict(converted, sep='/'))


class TransformTest(absltest.TestCase):
  """The reshapes, pinned elementwise on hand-built arrays."""

  def setUp(self):
    super().setUp()
    self.fmt = ckpt_format.Qwen38Format()
    _set_default_mesh()

  def _target(self, **leaves: tuple[int, ...]) -> Any:
    """An abstract tree holding just `params/block_0/token_mixer/<name>/w`."""
    return {
        'params': {
            'block_0': {
                'token_mixer': {
                    name: {'w': jax.ShapeDtypeStruct(shape, jnp.float32)}
                    for name, shape in leaves.items()
                }
            }
        }
    }

  def _transform(self, name: str, value: Any, target: Any) -> np.ndarray:
    out = self.fmt.transforms({name: value}, target)
    flat = ocp.tree.to_flat_dict(out, sep='/')
    self.assertLen(flat, 1)
    return np.asarray(next(iter(flat.values())))

  def test_linear_weights_are_transposed(self):
    value = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (out, in).
    got = self._transform(
        'model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
        value,
        self._target(in_proj_qkv=(3, 2)),
    )
    np.testing.assert_array_equal(
        got, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    )

  def test_depthwise_conv_is_squeezed(self):
    value = jnp.reshape(jnp.arange(6.0), (3, 1, 2))  # (channels, 1, taps).
    got = self._transform(
        'model.language_model.layers.0.linear_attn.conv1d.weight',
        value,
        self._target(conv1d=(3, 2)),
    )
    np.testing.assert_array_equal(got, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

  def test_query_and_gate_stay_head_major(self):
    """HF row block of head `h` is `[q(head_dim) | gate(head_dim)]`.

    Getting this wrong -- splitting into `(2, heads, head_dim)` -- keeps every
    shape right and swaps half of the queries with half of the gates, so it is
    pinned elementwise here rather than by shape.
    """
    n_heads, head_dim, model_dim = 3, 2, 4
    rows = 2 * n_heads * head_dim
    value = jnp.reshape(jnp.arange(float(rows * model_dim)), (rows, model_dim))
    got = self._transform(
        'model.language_model.layers.0.self_attn.q_proj.weight',
        value,
        self._target(q_proj=(model_dim, n_heads, 2 * head_dim)),
    )
    self.assertEqual(got.shape, (model_dim, n_heads, 2 * head_dim))
    for head in range(n_heads):
      for j in range(2 * head_dim):
        for i in range(model_dim):
          self.assertEqual(
              got[i, head, j],
              value[head * 2 * head_dim + j, i],
              msg=f'{head=} {j=} {i=}',
          )

  def test_output_projection_splits_the_second_axis(self):
    n_heads, head_dim, model_dim = 3, 2, 4
    value = jnp.reshape(
        jnp.arange(float(model_dim * n_heads * head_dim)),
        (model_dim, n_heads * head_dim),
    )
    got = self._transform(
        'model.language_model.layers.0.self_attn.o_proj.weight',
        value,
        self._target(o_proj=(model_dim, n_heads, head_dim)),
    )
    np.testing.assert_array_equal(
        got, np.asarray(value).reshape(model_dim, n_heads, head_dim)
    )

  def test_a_target_is_required(self):
    with self.assertRaisesRegex(ValueError, 'target_abstract_state'):
      self.fmt.transforms(
          {'model.language_model.layers.0.self_attn.q_proj.weight': jnp.zeros(
              (4, 2)
          )},
          None,
      )


class RegistrationTest(absltest.TestCase):

  def test_registered_under_the_name_the_config_asks_for(self):
    name = config_lib.qwen3p8_27b().init_ckpt_format
    self.assertEqual(name, ckpt_format.FORMAT_NAME)
    self.assertIs(
        ckpt_lib.CheckpointFormatRegistry.get(name),
        ckpt_format.Qwen38Format,
    )

  def test_constructs_with_no_arguments(self):
    # `save_checkpoint` bakes a no-argument instance into the metadata.
    self.assertIsInstance(
        ckpt_lib.CheckpointFormatRegistry.get_instance(
            ckpt_format.FORMAT_NAME
        ),
        ckpt_format.Qwen38Format,
    )


if __name__ == '__main__':
  absltest.main()

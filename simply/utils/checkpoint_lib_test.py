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
import dataclasses
import json

from absl import flags
from absl.testing import absltest
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from simply import config_lib
from simply import model_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import pytree
from simply.utils import sharding as sharding_lib


from importlib.resources import files, as_file
import simply.utils.testdata as testdata


def load_state(json_file):
  data = json.loads((files(testdata) / json_file).read_text(encoding='utf-8'))
  state = jax.tree_util.tree_map(
      jnp.array, data, is_leaf=lambda x: isinstance(x, list)
  )
  return state


class CheckpointFormatTest(absltest.TestCase):

  def lm_test_config(self):
    return dataclasses.replace(
        config_lib.lm_test(),
        model_dim=4,
        expand_factor=2,
        n_heads=2,
        n_layers=2,
        per_head_dim=4,
        vocab_size=2,
        use_per_dim_scale=False,
        output_layer_use_bias=False,
    )

  def setUp(self):
    super().setUp()
    self.expected_state = load_state('ckpt_expected_format.json')
    model = model_lib.TransformerLM(self.lm_test_config())
    self.expected_abstract_state = model.config.optimizer.init(
        ckpt_lib.get_abstract_params(model)
    )

  def test_restore_legacy_format(self):
    legacy_state = load_state('ckpt_legacy_format.json')
    ckpt_dir = self.create_tempdir()
    mngr = ocp.CheckpointManager(ckpt_dir.full_path)
    mngr.save(0, legacy_state, args=ocp.args.StandardSave(legacy_state))
    mngr.wait_until_finished()

    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir.full_path,
        self.expected_abstract_state,
    )
    restored = common.get_raw_arrays(restored)
    pytree.traverse_tree_with_path(
        lambda actual, expected, path: self.assertAlmostEqual(
            actual.tolist(), expected.tolist(), msg=f'Mismatch at {path}'
        ),
        restored,
        self.expected_state,
    )

  def test_restore_gemma2_format(self):
    gemma2_state = load_state('ckpt_gemma2_format.json')
    ckpt_dir = self.create_tempdir()
    mngr = ocp.CheckpointManager(ckpt_dir.full_path)
    ckpt_lib.save_checkpoint(
        mngr, gemma2_state, 0, ckpt_format=ckpt_lib.Gemma2Format()
    )
    mngr.wait_until_finished()

    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir.full_path, self.expected_abstract_state
    )
    restored = common.get_raw_arrays(restored)
    pytree.traverse_tree_with_path(
        lambda actual, expected, path: self.assertAlmostEqual(
            actual.tolist(), expected.tolist(), msg=f'Mismatch at {path}'
        ),
        restored,
        self.expected_state,
    )

  def test_restore_gemma2_transpose_format(self):
    gemma2_state = load_state('ckpt_gemma2_transpose_format.json')
    ckpt_dir = self.create_tempdir()
    mngr = ocp.CheckpointManager(ckpt_dir.full_path)
    ckpt_lib.save_checkpoint(
        mngr, gemma2_state, 0, ckpt_format=ckpt_lib.Gemma2TransposeFormat()
    )
    mngr.wait_until_finished()

    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir.full_path, self.expected_abstract_state
    )
    restored = common.get_raw_arrays(restored)
    pytree.traverse_tree_with_path(
        lambda actual, expected, path: self.assertAlmostEqual(
            actual.tolist(), expected.tolist(), msg=f'Mismatch at {path}'
        ),
        restored,
        self.expected_state,
    )

  def test_restore_with_format(self):
    gemma2_state = load_state('ckpt_gemma2_format.json')
    ckpt_dir = self.create_tempdir()
    mngr = ocp.CheckpointManager(ckpt_dir.full_path)
    ckpt_lib.save_checkpoint(  # Save incorrect format.
        mngr, gemma2_state, 0, ckpt_format=ckpt_lib.LegacyFormat()
    )
    mngr.wait_until_finished()

    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir.full_path,
        self.expected_abstract_state,
        ckpt_format='Gemma2Format',
    )
    restored = common.get_raw_arrays(restored)
    pytree.traverse_tree_with_path(
        lambda actual, expected, path: self.assertAlmostEqual(
            actual.tolist(), expected.tolist(), msg=f'Mismatch at {path}'
        ),
        restored,
        self.expected_state,
    )


class QwenFormatTest(absltest.TestCase):

  @classmethod
  def qwen2_test_config(cls):
    return dataclasses.replace(
        config_lib.deepseek_qwen2_1p5b(),
        model_dim=4,
        ffn_expand_dim=8,
        n_heads=2,
        n_layers=2,
        per_head_dim=2,
        vocab_size=2,
    )

  @classmethod
  def qwen3_moe_test_config(cls):
    return dataclasses.replace(
        config_lib.qwen3_30b_a3b(),
        use_qk_norm=False,
        model_dim=4,
        ffn_expand_dim=8,
        n_heads=2,
        n_kv_heads=2,
        n_layers=1,
        per_head_dim=2,
        vocab_size=2,
        use_moe=True,
        num_experts=2,
        num_experts_per_token=1,
    )

  def setUp(self):
    super().setUp()
    self.expected_state = load_state('ckpt_expected_qwen2_format.json')
    model = model_lib.TransformerLM(self.qwen2_test_config())
    self.expected_abstract_state = {
        'params': ckpt_lib.get_abstract_params(model)
    }

  def test_restore_qwen2_format(self):
    qwen2_state = load_state('ckpt_qwen2_format.json')
    ckpt_dir = self.create_tempdir()
    mngr = ocp.CheckpointManager(ckpt_dir.full_path)
    ckpt_lib.save_checkpoint(
        mngr, qwen2_state, 0, ckpt_format=ckpt_lib.Qwen2Format()
    )
    mngr.wait_until_finished()

    model = model_lib.TransformerLM(self.qwen2_test_config())
    expected_abstract_state = {
        'params': ckpt_lib.get_abstract_params(model)
    }
    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir.full_path, expected_abstract_state
    )
    restored = common.get_raw_arrays(restored)

    expected_state = load_state('ckpt_expected_qwen2_format.json')
    pytree.traverse_tree_with_path(
        lambda actual, expected, path: self.assertAlmostEqual(
            actual.tolist(), expected.tolist(), msg=f'Mismatch at {path}'
        ),
        restored,
        expected_state,
    )

  def test_restore_qwen3_moe_format(self):
    qwen3_moe_state = load_state('ckpt_qwen3_moe_format.json')
    ckpt_dir = self.create_tempdir()
    mngr = ocp.CheckpointManager(ckpt_dir.full_path)
    ckpt_lib.save_checkpoint(
        mngr, qwen3_moe_state, 0, ckpt_format=ckpt_lib.Qwen2Format()
    )
    mngr.wait_until_finished()

    config = self.qwen3_moe_test_config()
    sharding_lib.set_default_mesh_shape(
        mesh_shape=(1, 1, 1, 1),
        axis_names=config.sharding_config.mesh_axis_names)
    model = model_lib.TransformerLM(config)
    expected_abstract_state = {
        'params': ckpt_lib.get_abstract_params(model)
    }
    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir.full_path, expected_abstract_state
    )
    restored = common.get_raw_arrays(restored)

    expected_state = load_state('ckpt_expected_qwen3_moe_format.json')
    pytree.traverse_tree_with_path(
        lambda actual, expected, path: self.assertAlmostEqual(
            actual.tolist(), expected.tolist(), msg=f'Mismatch at {path}'
        ),
        restored,
        expected_state,
    )


class CheckpointLibTest(absltest.TestCase):

  def test_dump_format(self):
    js = pytree.dump(ckpt_lib.LegacyFormat())
    self.assertEqual(
        js,
        {
            '__dataclass__': 'CheckpointFormat:LegacyFormat',
            'restore_dtype': None,
        },
    )


class V2FormatQuantizedRestoreTest(absltest.TestCase):
  """Restores a training checkpoint into a params-only, FFN-quantized target.

  A training checkpoint carries optimizer state (`m` / `v` / `steps`) next to
  `params` and stores the routed-expert weights as plain full-precision arrays,
  while a serving target asks for `params` only and for `{quant_array, scale}`
  in place of each expert weight. Neither of the two quantized leaves has a
  same-named source, so they used to come back as `jax.ShapeDtypeStruct`, which
  is not a valid JAX type and crashes the first forward pass.
  """

  def test_load_training_checkpoint_into_quantized_target(self):
    num_experts, k, n, n_blocks = 4, 8, 6, 2
    param_keys = (
        'embed_linear/w',
        'transformer/block_0/attention/attention/q_proj/w',
        'transformer/block_0/routed_ffw/router/w',
    )
    ffn_keys = (
        'transformer/block_0/routed_ffw/ffn_0/w',
        'transformer/block_0/routed_ffw/ffn_1/w',
    )

    # What training writes: params + optimizer state, expert weights plain.
    rng = np.random.RandomState(0)
    flat_stored = {'steps': jnp.asarray(20, dtype=jnp.int32)}
    for key in param_keys + ffn_keys:
      shape = (num_experts, k, n) if key in ffn_keys else (3, 4)
      value = jnp.asarray(rng.randn(*shape).astype(np.float32))
      flat_stored[f'params/{key}'] = value
      flat_stored[f'm/{key}'] = jnp.zeros_like(value)
      flat_stored[f'v/{key}'] = jnp.ones_like(value)
    ckpt_dir = self.create_tempdir().full_path
    with ocp.CheckpointManager(ckpt_dir) as mngr:
      ckpt_lib.save_checkpoint(
          mngr,
          ocp.tree.from_flat_dict(flat_stored, sep='/'),
          20,
          ckpt_format=ckpt_lib.V2Format(),
      )
      mngr.wait_until_finished()

    # What serving asks for: no optimizer state, expert weights int4-quantized.
    replicated = jax.sharding.NamedSharding(
        sharding_lib.create_mesh(), jax.sharding.PartitionSpec()
    )
    flat_target = {
        f'params/{key}': jax.ShapeDtypeStruct(
            (3, 4), jnp.float32, sharding=replicated
        )
        for key in param_keys
    }
    for key in ffn_keys:
      flat_target[f'params/{key}/quant_array'] = jax.ShapeDtypeStruct(
          (num_experts, k, n), jnp.int4, sharding=replicated
      )
      flat_target[f'params/{key}/scale'] = jax.ShapeDtypeStruct(
          (num_experts, n_blocks, n), jnp.float32, sharding=replicated
      )

    # No explicit format: the loader reads `V2Format` off the ckpt metadata.
    restored = ckpt_lib.load_checkpoint_from_dir(
        ckpt_dir, ocp.tree.from_flat_dict(flat_target, sep='/')
    )

    flat = ocp.tree.to_flat_dict(common.get_raw_arrays(restored), sep='/')
    self.assertCountEqual(flat_target, flat)
    for key, value in flat.items():
      self.assertIsInstance(value, jax.Array, msg=f'left abstract: {key}')
    for key in param_keys:
      np.testing.assert_allclose(
          np.asarray(flat[f'params/{key}'], dtype=np.float32),
          np.asarray(flat_stored[f'params/{key}'], dtype=np.float32),
          rtol=1e-6,
      )
    for key in ffn_keys:
      quant = np.asarray(flat[f'params/{key}/quant_array'].astype(jnp.float32))
      scale = np.asarray(flat[f'params/{key}/scale'])
      dequant = (
          quant.reshape(num_experts, n_blocks, k // n_blocks, n)
          * scale[:, :, None, :]
      ).reshape(num_experts, k, n)
      original = np.asarray(flat_stored[f'params/{key}'], dtype=np.float32)
      rel_error = np.abs(dequant - original).max() / np.abs(original).max()
      self.assertLess(rel_error, 0.25, msg=f'int4 restore blew up at {key}')

if __name__ == '__main__':
  absltest.main()

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
import orbax.checkpoint as ocp
from simply import config_lib
from simply import model_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import pytree


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


class Qwen2FormatTest(absltest.TestCase):

  def qwen2_test_config(self):
    return dataclasses.replace(
        config_lib.deepseek_qwen2_1p5b(),
        model_dim=4,
        ffn_expand_dim=8,
        n_heads=2,
        n_layers=2,
        per_head_dim=2,
        vocab_size=2,
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


class CheckpointLibTest(absltest.TestCase):

  def test_dump_format(self):
    js = pytree.dump(ckpt_lib.LegacyFormat())
    self.assertEqual(js, {'__dataclass__': 'CheckpointFormat:LegacyFormat'})


if __name__ == '__main__':
  absltest.main()

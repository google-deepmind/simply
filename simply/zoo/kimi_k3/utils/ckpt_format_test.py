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
"""Tests for the Kimi K3 checkpoint format (MXFP4 decode on restore)."""

from typing import Any, cast

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import common
from simply.zoo.kimi_k3 import model_lib as k3_model_lib
from simply.zoo.kimi_k3.utils import ckpt_format
from simply.zoo.kimi_k3.utils import hf_params
from simply.zoo.kimi_k3.utils import test_utils


def _random_mxfp4(num_experts, out_features, in_features, seed=0):
  rng = np.random.default_rng(seed)
  packed = rng.integers(
      0, 256, size=(num_experts, out_features, in_features // 2), dtype=np.uint8
  )
  scale = rng.integers(
      110,
      140,
      size=(num_experts, out_features, in_features // 32),
      dtype=np.uint8,
  )
  return packed, scale


class DequantizeTest(absltest.TestCase):

  def test_matches_the_numpy_decoder_and_transposes(self):
    packed, scale = _random_mxfp4(2, 8, 64)
    expected = hf_params.dequantize_mxfp4(packed, scale, np.float32)
    got = ckpt_format.dequantize_mxfp4(
        jnp.asarray(packed), jnp.asarray(scale), jnp.float32
    )
    self.assertEqual(got.shape, (2, 64, 8))
    np.testing.assert_array_equal(
        np.asarray(got), np.swapaxes(expected, -1, -2)
    )

  def test_values_are_exact_in_bfloat16(self):
    """E2M1 codes times a power of two are representable in bf16."""
    packed, scale = _random_mxfp4(1, 4, 32, seed=1)
    expected = hf_params.dequantize_mxfp4(packed, scale, np.float32)
    got = ckpt_format.dequantize_mxfp4(
        jnp.asarray(packed), jnp.asarray(scale), jnp.bfloat16
    )
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.swapaxes(expected, -1, -2)
    )


class BitExactnessTest(absltest.TestCase):

  def test_bfloat16_decode_equals_the_float32_decode(self):
    """Decoded values are multiples of 0.5 times a power of two."""
    packed, scale = _random_mxfp4(2, 16, 128, seed=3)
    in_f32 = ckpt_format.dequantize_mxfp4(
        jnp.asarray(packed), jnp.asarray(scale), jnp.float32
    )
    in_bf16 = ckpt_format.dequantize_mxfp4(
        jnp.asarray(packed), jnp.asarray(scale), jnp.bfloat16
    )
    np.testing.assert_array_equal(
        np.asarray(in_bf16, np.float32), np.asarray(in_f32)
    )

  def test_jit_and_eager_agree(self):
    packed, scale = _random_mxfp4(1, 8, 64, seed=4)
    eager = ckpt_format.dequantize_mxfp4(
        jnp.asarray(packed), jnp.asarray(scale)
    )
    jitted = jax.jit(ckpt_format.dequantize_mxfp4)(
        jnp.asarray(packed), jnp.asarray(scale)
    )
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


class TreeTest(absltest.TestCase):

  def test_only_packed_nodes_are_rewritten(self):
    packed, scale = _random_mxfp4(1, 4, 32, seed=2)
    tree = {
        'experts': {
            'ffn_0': {
                'packed': jnp.asarray(packed),
                'scale': jnp.asarray(scale),
            }
        },
        'router': {'w': jnp.ones((3, 4)), 'bias': jnp.zeros((4,))},
    }
    out = ckpt_format.dequantize_tree(tree)
    self.assertEqual(set(out['experts']['ffn_0']), {'w'})
    self.assertEqual(out['experts']['ffn_0']['w'].shape, (1, 32, 4))
    self.assertIs(out['router']['w'], tree['router']['w'])


class TransformsTest(absltest.TestCase):
  """`transforms` hands the loader a tree shaped like the target."""

  def test_transforms_leaves_an_unrolled_target_alone(self):
    key = jax.random.key(0)
    unrolled = k3_model_lib.KimiK3LM(config=test_utils.tiny_config())
    stored = cast(dict[str, Any], common.get_raw_arrays(unrolled.init(key)))
    target = jax.eval_shape(lambda: stored)
    got = ckpt_format.KimiK3Format().transforms(stored, target)
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(target))


if __name__ == '__main__':
  absltest.main()

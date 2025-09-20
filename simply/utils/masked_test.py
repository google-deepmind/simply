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
"""Unit test for masked."""

import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from simply.utils import masked


def _make_random_input_and_mask(
    size: int | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
  if isinstance(size, int):
    size = (size,)
  x = np.random.rand(*size).astype(np.float32)
  mask = np.random.randint(2, size=size).astype(np.bool)
  return x, mask


class MaskedTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("padding_value_0", 0), ("padding_value_1", 1)
  )
  def test_masked(self, padding_value: float):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = masked.masked(
        jnp.asarray(x), mask=jnp.asarray(mask), padding_value=padding_value
    )
    self.assertEqual(masked_x.shape, x.shape)
    self.assertTrue(np.all(masked_x == np.where(mask, x, padding_value)))

  @parameterized.named_parameters(
      ("axis_none", None, True),
      ("axis_0", 0, False),
      ("axis_1", 1, True),
      ("axis_all", (0, 1), False),
  )
  def test_masked_max(self, axis: int | tuple[int, ...] | None, keepdims: bool):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_max = masked.masked_max(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        keepdims=keepdims,
    )
    ref_masked_max = masked_x.max(axis=axis, keepdims=keepdims)

    self.assertEqual(masked_max.shape, ref_masked_max.shape)
    self.assertTrue(np.all(np.asarray(masked_max) == ref_masked_max))

  @parameterized.named_parameters(
      ("axis_none", None, True),
      ("axis_0", 0, False),
      ("axis_1", 1, True),
      ("axis_all", (0, 1), False),
  )
  def test_masked_min(self, axis: int | tuple[int, ...] | None, keepdims: bool):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_min = masked.masked_min(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        keepdims=keepdims,
    )
    ref_masked_min = masked_x.min(axis=axis, keepdims=keepdims)

    self.assertEqual(masked_min.shape, ref_masked_min.shape)
    self.assertTrue(np.all(np.asarray(masked_min) == ref_masked_min))

  @parameterized.named_parameters(
      ("axis_none", None, True),
      ("axis_0", 0, False),
      ("axis_1", 1, True),
      ("axis_all", (0, 1), False),
  )
  def test_masked_sum(self, axis: int | tuple[int, ...] | None, keepdims: bool):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_sum = masked.masked_sum(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        keepdims=keepdims,
    )
    ref_masked_sum = masked_x.sum(axis=axis, keepdims=keepdims)

    self.assertEqual(masked_sum.shape, ref_masked_sum.shape)
    self.assertTrue(np.allclose(masked_sum, ref_masked_sum))

  @parameterized.named_parameters(
      ("axis_none", None, True),
      ("axis_0", 0, False),
      ("axis_1", 1, True),
      ("axis_all", (0, 1), False),
  )
  def test_masked_mean(
      self, axis: int | tuple[int, ...] | None, keepdims: bool
  ):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_mean = masked.masked_mean(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        keepdims=keepdims,
    )
    ref_masked_mean = masked_x.mean(axis=axis, keepdims=keepdims)

    self.assertEqual(masked_mean.shape, ref_masked_mean.shape)
    self.assertTrue(np.allclose(masked_mean, ref_masked_mean))

  @parameterized.named_parameters(
      ("axis_none", None, 0, True),
      ("axis_0", 0, 1, False),
      ("axis_1", 1, 0, True),
      ("axis_all", (0, 1), 1, False),
  )
  def test_masked_var(
      self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool
  ):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_var = masked.masked_var(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        ddof=ddof,
        keepdims=keepdims,
    )
    ref_masked_var = masked_x.var(axis=axis, ddof=ddof, keepdims=keepdims)

    self.assertEqual(masked_var.shape, ref_masked_var.shape)
    self.assertTrue(np.allclose(masked_var, ref_masked_var))

  @parameterized.named_parameters(
      ("axis_none", None, 0, True),
      ("axis_0", 0, 1, False),
      ("axis_1", 1, 0, True),
      ("axis_all", (0, 1), 1, False),
  )
  def test_masked_std(
      self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool
  ):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_std = masked.masked_std(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        ddof=ddof,
        keepdims=keepdims,
    )
    ref_masked_std = masked_x.std(axis=axis, ddof=ddof, keepdims=keepdims)

    self.assertEqual(masked_std.shape, ref_masked_std.shape)
    self.assertTrue(np.allclose(masked_std, ref_masked_std))

  @parameterized.named_parameters(
      ("axis_none", None, 0, True),
      ("axis_0", 0, 1, False),
      ("axis_1", 1, 0, True),
      ("axis_all", (0, 1), 1, False),
  )
  def test_masked_mean_var(
      self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool
  ):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_mean, masked_var = masked.masked_mean_var(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        ddof=ddof,
        keepdims=keepdims,
    )
    ref_masked_mean = masked_x.mean(axis=axis, keepdims=keepdims)
    ref_masked_var = masked_x.var(axis=axis, ddof=ddof, keepdims=keepdims)

    self.assertEqual(masked_mean.shape, ref_masked_mean.shape)
    self.assertTrue(np.allclose(masked_mean, ref_masked_mean))

    self.assertEqual(masked_var.shape, ref_masked_var.shape)
    self.assertTrue(np.allclose(masked_var, ref_masked_var))

  @parameterized.named_parameters(
      ("axis_none", None, 0, True),
      ("axis_0", 0, 1, False),
      ("axis_1", 1, 0, True),
      ("axis_all", (0, 1), 1, False),
  )
  def test_masked_mean_std(
      self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool
  ):
    size = (10, 20)
    x, mask = _make_random_input_and_mask(size)
    masked_x = np.ma.array(x, mask=~mask)
    masked_mean, masked_std = masked.masked_mean_std(
        jnp.asarray(x),
        jnp.asarray(mask),
        axis=axis,
        ddof=ddof,
        keepdims=keepdims,
    )
    ref_masked_mean = masked_x.mean(axis=axis, keepdims=keepdims)
    ref_masked_std = masked_x.std(axis=axis, ddof=ddof, keepdims=keepdims)

    self.assertEqual(masked_mean.shape, ref_masked_mean.shape)
    self.assertTrue(np.allclose(masked_mean, ref_masked_mean))

    self.assertEqual(masked_std.shape, ref_masked_std.shape)
    self.assertTrue(np.allclose(masked_std, ref_masked_std))


if __name__ == "__main__":
  absltest.main()

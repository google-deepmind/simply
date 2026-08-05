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
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
from simply.utils import common
from simply.utils import optimizers as opt_lib
from simply.utils import pytree


class OptimizerTest(absltest.TestCase):

  def assert_almost_equal(
      self, actual: common.PyTree, expected: common.PyTree, delta: float = 1e-5
  ):

    def _assert_tensor_almost_equal(
        actual: jax.typing.ArrayLike, expected: jax.typing.ArrayLike, path: str
    ):
      if hasattr(actual, 'tolist'):
        actual = actual.tolist()  # pyrefly: ignore[bad-assignment]
      if hasattr(expected, 'tolist'):
        expected = expected.tolist()  # pyrefly: ignore[bad-assignment]
      self.assertAlmostEqual(
          actual, expected, delta=delta, msg=f'Mismatch at {path}'
      )

    pytree.traverse_tree_with_path(
        _assert_tensor_almost_equal, actual, expected
    )

  def test_dump(self):
    js = pytree.dump(opt_lib.SGD())
    # `skip_updates_grad_nans` is a field on the base `Optimizer` dataclass
    # (default False; opt-in NaN-skip wrapper around `apply`); it shows up
    # in the dumped shape of every subclass.
    self.assertEqual(
        js,
        {'__dataclass__': 'Optimizer:SGD', 'skip_updates_grad_nans': False},
    )

  def test_init_step(self):
    step = opt_lib.get_init_steps()
    self.assertEqual(step.tolist(), 0)

  def test_sgd(self):
    opt = opt_lib.SGD()
    state = opt.init({'a': jnp.array(1.0)})
    self.assert_almost_equal(state, {'params': {'a': 1.0}, 'steps': 0})
    grad, state = opt.apply(state, {'a': jnp.array(2.0)})
    self.assert_almost_equal(grad, {'a': 2.0})
    self.assert_almost_equal(state, {'params': {'a': 1.0}, 'steps': 0})

  def test_adam(self):
    opt = opt_lib.Adam()
    state = opt.init({'a': jnp.array(1.0)})
    self.assert_almost_equal(
        state,
        {'params': {'a': 1.0}, 'm': {'a': 0.0}, 'v': {'a': 0.0}, 'steps': 0},
    )

    grad, state = opt.apply(state, {'a': jnp.array(2.0)})
    self.assert_almost_equal(grad, {'a': 1.0})
    self.assert_almost_equal(
        state,
        {'params': {'a': 1.0}, 'm': {'a': 0.2}, 'v': {'a': 0.004}, 'steps': 0},
    )

  def test_lion(self):
    opt = opt_lib.Lion()
    state = opt.init({'a': jnp.array(1.0)})
    self.assert_almost_equal(
        state, {'params': {'a': 1.0}, 'm': {'a': 0.0}, 'steps': 0}
    )

    grad, state = opt.apply(state, {'a': jnp.array(2.0)})
    self.assert_almost_equal(grad, {'a': 1.0})
    self.assert_almost_equal(
        state, {'params': {'a': 1.0}, 'm': {'a': 0.04004}, 'steps': 0}
    )

  def test_apply_with_nan_skip_reverts_state_on_nan_grad(self):
    """NaN-containing grad: ``apply_with_nan_skip`` is a no-op.

    The post-step ``m`` / ``v`` / ``params`` equal their pre-step values,
    the returned update is exactly zero, and the ``nan_skips`` counter
    bumps by 1. Exercises Adam (non-trivial m, v leaves) with the
    skipper opted in.
    """
    opt = opt_lib.Adam(skip_updates_grad_nans=True)
    state = opt.init({'a': jnp.array(1.0)})
    # Run one clean step first so m / v are non-zero (makes the revert
    # check non-trivial) and the ``nan_skips`` counter is materialised.
    _, state = opt.apply_with_nan_skip(state, {'a': jnp.array(2.0)})
    pre_m = float(state['m']['a'])  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    pre_v = float(state['v']['a'])  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    pre_params = float(state['params']['a'])  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    pre_skips = int(state['nan_skips'])  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    # Now a bad-grad step.
    upd, new_state = opt.apply_with_nan_skip(state, {'a': jnp.array(jnp.nan)})
    self.assertEqual(float(upd['a']), 0.0)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    self.assertEqual(float(new_state['m']['a']), pre_m)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    self.assertEqual(float(new_state['v']['a']), pre_v)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    self.assertEqual(float(new_state['params']['a']), pre_params)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]
    self.assertEqual(int(new_state['nan_skips']), pre_skips + 1)  # pyrefly: ignore[bad-argument-type, bad-index, unsupported-operation]

  def test_schedule_backward_compatibility_constant(self):

    @dataclasses.dataclass(frozen=True)
    class MockConfigV0:
      num_train_steps: int = 100
      lr_schedule_name: str = 'constant'
      lr_schedule_config: tuple[tuple[str, Any], ...] = (
          ('lr', 1e-3),
          ('warmup_steps', 10),
      )

    @dataclasses.dataclass(frozen=True)
    class MockConfigV1:
      num_train_steps: int = 100
      lr: opt_lib.Schedule = opt_lib.LinearWarmupConstant(
          value=1e-3,
          warmup_steps=10,
      )
    config0 = MockConfigV0()
    lr_fn_0 = opt_lib.create_lr_schedule(config0)
    config1 = MockConfigV1()
    lr_fn_1 = opt_lib.create_lr_schedule(config1)
    for steps in [0, 101]:
      self.assertEqual(lr_fn_0(steps).tolist(), lr_fn_1(steps).tolist())

  def test_schedule_backward_compatibility_cosine_decay(self):

    @dataclasses.dataclass(frozen=True)
    class MockConfigV0:
      num_train_steps: int = 100
      lr_schedule_name: str = 'cosine_decay'
      lr_schedule_config: tuple[tuple[str, Any], ...] = (
          ('lr', 1e-3),
          ('warmup_steps', 10),
          ('steps_after_decay', 10),
          ('end_decay', 0.1),
      )

    @dataclasses.dataclass(frozen=True)
    class MockConfigV1:
      num_train_steps: int = 100
      lr: opt_lib.Schedule = opt_lib.LinearWarmupCosineDecay(
          value=1e-3,
          warmup_steps=10,
          steps_after_decay=10,
          end_decay=0.1,
      )
    config0 = MockConfigV0()
    lr_fn_0 = opt_lib.create_lr_schedule(config0)
    config1 = MockConfigV1()
    lr_fn_1 = opt_lib.create_lr_schedule(config1)
    for steps in [0, 101]:
      self.assertEqual(lr_fn_0(steps).tolist(), lr_fn_1(steps).tolist())

  def test_schedule_backward_compatibility_cosine_decay_fraction(self):

    @dataclasses.dataclass(frozen=True)
    class MockConfigV0:
      num_train_steps: int = 100
      lr_schedule_name: str = 'cosine_decay'
      lr_schedule_config: tuple[tuple[str, Any], ...] = (
          ('lr', 1e-3),
          ('warmup_steps', 10),
          ('steps_after_decay', 10),
          ('decay_start', 20),
          ('end_decay', 0.1),
      )

    @dataclasses.dataclass(frozen=True)
    class MockConfigV1:
      num_train_steps: int = 100
      lr: opt_lib.Schedule = opt_lib.LinearWarmupCosineDecay(
          value=1e-3,
          warmup_fraction=0.1,
          decay_start_fraction=0.2,
          fraction_after_decay=0.1,
          end_decay=0.1,
      )
    config0 = MockConfigV0()
    lr_fn_0 = opt_lib.create_lr_schedule(config0)
    config1 = MockConfigV1()
    lr_fn_1 = opt_lib.create_lr_schedule(config1)
    for steps in [0, 101]:
      self.assertEqual(lr_fn_0(steps).tolist(), lr_fn_1(steps).tolist())


if __name__ == '__main__':
  absltest.main()

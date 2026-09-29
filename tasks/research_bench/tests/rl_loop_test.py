# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the task-agnostic research-bench RL loop's held-out eval aggregation.

`_aggregate_eval` is task-agnostic: the scored `eval_accuracy` is the mean
per-example accuracy (avg@k) over the FULL eval set (pooled / micro-averaged),
and -- if examples carry an optional 'category' field -- per-category accuracies
are reported as diagnostics only. WHICH examples are in the eval set (e.g. BFCL
excluding its abstention category) is a property of each task's eval DATA
SOURCE, not of this function.
"""

import dataclasses
import json
import os
from typing import Any

os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '')
    + ' --xla_force_host_platform_device_count=4'
)

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import jax.sharding as js  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from tasks.research_bench import config_lib  # pylint: disable=g-import-not-at-top
from tasks.research_bench import rl_loop  # pylint: disable=g-import-not-at-top


@dataclasses.dataclass(frozen=True)
class _ReshardCfg:
  """Just the field `_make_decode_resharder` reads."""

  decode_reshard: str = 'jit'


def _ex(category=None):
  """A minimal eval example (optional 'category' field for diagnostics)."""
  return {'category': category} if category is not None else {}


class AggregateEvalTest(absltest.TestCase):

  def test_scored_metric_is_mean_over_all_examples(self):
    examples = [_ex(), _ex(), _ex(), _ex()]
    res = rl_loop._aggregate_eval(examples, [1.0, 1.0, 1.0, 0.0])
    self.assertAlmostEqual(res['eval_accuracy'], 0.75)
    self.assertEqual(res['n_scored'], 4)  # every example is scored

  def test_micro_average_not_macro(self):
    # 1x cat_a (correct) + 3x cat_b (2 correct). The scored metric pools ALL
    # examples: 3/4 = 0.75 (NOT the macro-average (1.0 + 2/3)/2 = 0.833 that
    # would over-weight the tiny category).
    examples = [_ex('a'), _ex('b'), _ex('b'), _ex('b')]
    res = rl_loop._aggregate_eval(examples, [1.0, 1.0, 1.0, 0.0])
    self.assertAlmostEqual(res['eval_accuracy'], 0.75)
    self.assertAlmostEqual(res['by_category']['b']['accuracy'], 2.0 / 3.0)
    self.assertEqual(res['by_category']['a']['n'], 1)

  def test_perfect_solver_scores_one(self):
    res = rl_loop._aggregate_eval([_ex('a'), _ex('b')], [1.0, 1.0])
    self.assertEqual(res['eval_accuracy'], 1.0)

  def test_avg_at_k_fraction_is_carried_through(self):
    # A per-example avg@k of 0.5 (half the k samples correct) contributes 0.5.
    res = rl_loop._aggregate_eval([_ex(), _ex()], [0.5, 0.5])
    self.assertAlmostEqual(res['eval_accuracy'], 0.5)

  def test_examples_without_category_go_to_one_bucket(self):
    res = rl_loop._aggregate_eval([_ex(), _ex()], [1.0, 0.0])
    self.assertEqual(list(res['by_category']), ['all'])
    self.assertEqual(res['by_category']['all']['n'], 2)
    self.assertAlmostEqual(res['by_category']['all']['accuracy'], 0.5)

  def test_empty_eval_set_no_crash(self):
    res = rl_loop._aggregate_eval([], [])
    self.assertEqual(res['eval_accuracy'], 0.0)
    self.assertEqual(res['n_scored'], 0)

  def test_eval_protocol_records_the_scored_setup(self):
    # The eval protocol block must pin down everything the scored number depends
    # on, including Evaluation constructor args (few_shot / system_message are
    # ordinary dataclass fields, so two runs with different ones are not
    # comparable).

    @dataclasses.dataclass(frozen=True)
    class _Eval:
      few_shot: str = 'SHOT'
      partial_credit: bool = False
      nested: tuple[int, ...] = (1, 2)

    class _Ds:
      source = 'simply:bfcl_live_eval'

    @dataclasses.dataclass(frozen=True)
    class _Cfg:
      validation_datasets: tuple[Any, ...] = (_Ds(),)
      vocab_name: str = 'Qwen3'
      eval_temperature: float = 0.0
      eval_num_samples: int = 1
      eval_max_decode_steps: int = 160
      eval_max_input_len: int = 3072
      eval_prefill_size: int = 3072
      validation_eval_batch_size: int = 256
      activation_dtype_name: str = 'bfloat16'
      decoding_quant_scheme: str = 'bfloat16'

    got = rl_loop._eval_protocol(  # pylint: disable=protected-access
        _Cfg(), _Eval(), 'Pretrain', ('\n\n',), 1351
    )
    self.assertEqual(got['evaluation'], '_Eval')
    self.assertEqual(got['evaluation_args']['few_shot'], 'SHOT')
    self.assertFalse(got['evaluation_args']['partial_credit'])
    # Non-scalar ctor args are recorded by type, never dropped silently.
    self.assertEqual(got['evaluation_args']['nested'], '<tuple>')
    self.assertEqual(got['eval_source'], 'simply:bfcl_live_eval')
    self.assertEqual(got['n_scored'], 1351)
    self.assertEqual(got['lm_format_name'], 'Pretrain')
    self.assertEqual(got['eval_max_input_len'], 3072)
    # Must be JSON-serializable: it is written into final_result.json.
    json.dumps(got)

  def test_train_loop_registered(self):
    # Importing rl_loop registers the task-agnostic RL train loop that main.py
    # dispatches to via the config's train_loop_name.
    self.assertIsNotNone(rl_loop.TrainLoopRegistry.get('research_bench_rl'))


class DecodeResharderTest(absltest.TestCase):
  """Every `decode_reshard` implementation must produce the same params.

  `jit` is the default and the fast one; `per_array` is the original and the
  reference. They differ in how the move is executed, never in what it
  produces.
  """

  def _params_and_target(self):
    devices = np.array(jax.devices()[:4])
    axis_names = ('replica', 'data', 'model')
    train_mesh = js.Mesh(devices.reshape(1, 4, 1), axis_names=axis_names)
    decode_mesh = js.Mesh(devices.reshape(4, 1, 1), axis_names=axis_names)
    with js.set_mesh(train_mesh):
      params = {
          'w': jax.device_put(
              jnp.arange(32, dtype=jnp.float32).reshape(8, 4),
              js.NamedSharding(train_mesh, js.PartitionSpec('data')),
          ),
          'b': jax.device_put(
              jnp.arange(4, dtype=jnp.float32),
              js.NamedSharding(train_mesh, js.PartitionSpec()),
          ),
      }
    target = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(
            x.shape,
            jnp.bfloat16,
            sharding=js.NamedSharding(decode_mesh, js.PartitionSpec()),
        ),
        params,
    )
    return params, decode_mesh, target

  def test_implementations_agree(self):
    params, decode_mesh, target = self._params_and_target()
    outputs = {}
    for mode in ('per_array', 'device_put', 'jit'):
      reshard = rl_loop._make_decode_resharder(  # pylint: disable=protected-access
          dataclasses.replace(_ReshardCfg(), decode_reshard=mode),
          decode_mesh,
          target,
      )
      out = reshard(params)
      jax.tree_util.tree_map(
          lambda x: self.assertEqual(x.dtype, jnp.bfloat16), out
      )
      outputs[mode] = jax.tree_util.tree_map(np.asarray, out)
    for mode in ('device_put', 'jit'):
      for key in outputs['per_array']:
        np.testing.assert_array_equal(
            outputs['per_array'][key],
            outputs[mode][key],
            err_msg=f'{mode}/{key}',
        )

  def test_unknown_mode_is_rejected(self):
    params, decode_mesh, target = self._params_and_target()
    del params
    with self.assertRaises(ValueError):
      rl_loop._make_decode_resharder(  # pylint: disable=protected-access
          dataclasses.replace(_ReshardCfg(), decode_reshard='nope'),
          decode_mesh,
          target,
      )


class ExperimentHelperTest(absltest.TestCase):
  """Guards the RL loop against core-API drift it cannot see on CPU."""

  def test_every_task_config_builds_a_helper(self):
    # `run_experiment` needs a checkpoint and an accelerator, so this is the
    # only part of its setup a unit test can reach -- and it is where a kwarg
    # core does not accept (the internal `write_to_datatable`) used to hide,
    # failing minutes into a TPU run instead of here.
    for name in ('rl_gemma3_1b', 'rl_qwen2p5_math_1p5b', 'rl_bfcl_qwen3_0p6b',
                 'rl_bfcl_gemma3_1b', 'port_falcon_h1_0p5b',
                 'port_recurrentgemma_2b'):
      config = config_lib.ExperimentConfigRegistry.get_config(name)
      helper = rl_loop.make_experiment_helper(config, '/tmp/rb_helper')
      self.assertEqual(helper.experiment_dir, '/tmp/rb_helper')
      self.assertEqual(helper.num_train_steps, config.num_train_steps)


if __name__ == '__main__':
  absltest.main()

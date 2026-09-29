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

"""End-to-end test of the eval entry point with a STUB model, on CPU.

It runs the real `eval_main.main` -> `page_decode_eval.main` pipeline (data
source -> `LcbEval.evaluate_async` -> sandboxed grading -> history ->
`final_result.json`) with the TPU-bound parts replaced: the page batcher is a
stub that answers each request with a canned program, and the checkpoint load
is a no-op. Everything the task's metric depends on -- which problems, which
grader, how `accuracy` is computed and written -- is the shipped code.

The canned model answers ~half the problems correctly, so the expected
`accuracy` is known exactly.
"""

import json
import os
import threading
from unittest import mock

from absl import flags
from absl.testing import absltest
import grpc
from simply import config_lib as core_config_lib
from simply import data_lib as core_data_lib
from simply.eval import page_decode_eval
from simply.serving import common as serving_common
from tasks.research_bench import eval_main
from tasks.research_bench import lcb_data_lib


_STDIN_PROBLEM = dict(
    question_id='p{i}',
    question_title='add one',
    question_content='Read n from stdin and print n + 1.',
    platform='atcoder',
    contest_id='c1',
    contest_date='2024-10-01T00:00:00',
    difficulty='easy',
    starter_code='',
    func_name='',
    public_test_cases=json.dumps(
        [dict(input='1\n', output='2\n', testtype='stdin')]
    ),
    private_test_cases=json.dumps(
        [dict(input='41\n', output='42\n', testtype='stdin')]
    ),
)

_CORRECT = 'print(int(input()) + 1)'
_WRONG = 'print(0)'


class _StubBatcher:
  """Stands in for `page_batcher.Batcher`: canned responses, no accelerator."""

  responses: list[str] = []
  max_queue_size = 4096

  def __init__(self, **kwargs):
    del kwargs
    self._n = 0
    self._lock = threading.Lock()
    self.request_queue = _StubQueue()
    self.prefix_cache = None
    self.compiled_decode_fn = None
    self.compiled_prefill_fn = None
    self.compiled_push_fn = None
    self.compiled_release_fn = None

  def update_params_from_checkpoint_path(self, path):
    del path

  def thread(self, stop_event, error_queue):
    del error_queue
    return threading.Thread(target=stop_event.wait, daemon=True)

  def enqueue(self, request, future):
    del request
    with self._lock:
      text = self.responses[self._n % len(self.responses)]
      self._n += 1
    # The real batcher resolves the future from its own thread; here the
    # caller's event loop is the current one, so setting it directly is what
    # `await future` needs.
    future.set_result(
        serving_common.SimplyServiceResponse(
            code=grpc.StatusCode.OK,
            result=dict(output_text=text, tokens=[], input_len=0),
        )
    )


class _StubQueue:

  def qsize(self):
    return 0


class EvalMainStubTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_problems = 6
    data_dir = self.create_tempdir('data').full_path
    lcb_path = os.path.join(data_dir, 'lcb.jsonl')
    with open(lcb_path, 'w') as f:
      for i in range(self.num_problems):
        row = dict(_STDIN_PROBLEM, question_id=f'p{i}')
        f.write(json.dumps(row) + '\n')
    self.enter_context(
        _registered(
            core_data_lib.DataSourceRegistry,
            'simply_json:livecodebench_stub',
            lambda: lcb_data_lib.LiveCodeBenchV5(
                path=lcb_path, expected_count=self.num_problems
            ),
        )
    )
    self.enter_context(
        _registered(
            core_config_lib.ExperimentConfigRegistry,
            'lcb_stub_cpu',
            _stub_config,
        )
    )
    # Alternate correct / wrong answers -> a known accuracy of exactly 0.5.
    _StubBatcher.responses = [
        f'```python\n{_CORRECT}\n```',
        f'```python\n{_WRONG}\n```',
    ]
    self.enter_context(
        mock.patch.object(page_decode_eval.page_batcher, 'Batcher',
                          _StubBatcher)
    )
    self.enter_context(
        mock.patch.object(
            page_decode_eval.checkpoint_lib, 'get_checkpoint_path',
            lambda ckpt_dir, step: 'stub-checkpoint',
        )
    )

  def test_writes_final_result_json(self):
    experiment_dir = self.create_tempdir('run').full_path
    _run_eval_main(
        experiment_dir=experiment_dir,
        experiment_config='lcb_stub_cpu',
        evaluation='LcbBaseline',
        datasource_name='simply_json:livecodebench_stub',
        seed=42,
        n_repeats=1,
    )

    with open(os.path.join(experiment_dir, 'final_result.json')) as f:
      result = json.load(f)
    self.assertEqual(result['total'], self.num_problems)
    self.assertEqual(result['correct'], self.num_problems // 2)
    self.assertAlmostEqual(result['accuracy'], 0.5)
    self.assertEqual(result['seed'], 42)
    self.assertGreater(result['avg_generation_time'], 0.0)

  def test_n_repeats_multiplies_the_sample_count(self):
    experiment_dir = self.create_tempdir('run_repeats').full_path
    _run_eval_main(
        experiment_dir=experiment_dir,
        experiment_config='lcb_stub_cpu',
        evaluation='LcbBaseline',
        datasource_name='simply_json:livecodebench_stub',
        seed=43,
        n_repeats=2,
    )

    with open(os.path.join(experiment_dir, 'final_result.json')) as f:
      result = json.load(f)
    self.assertEqual(result['total'], 2 * self.num_problems)
    self.assertEqual(result['seed'], 43)


def _stub_config():
  """A tiny CPU config; the model is never built (the batcher is a stub)."""
  return core_config_lib.BaseExperimentConfig(
      model_dim=8, n_heads=1, n_layers=1, per_head_dim=8, vocab_size=32
  )


class _registered:
  """Context manager that adds one entry to a registry and removes it after."""

  def __init__(self, registry, name, factory):
    self._registry, self._name, self._factory = registry, name, factory

  def __enter__(self):
    self._registry.register(self._factory, name=self._name)
    return self._factory

  def __exit__(self, *exc):
    self._registry.unregister(self._name)


def _run_eval_main(**flag_values) -> None:
  """Runs `eval_main.main` with the given flags, restoring them afterwards."""
  import jax  # Local: keep module import time free of the JAX import.

  overrides = dict(
      # Whatever CPU device count this process was started with (other test
      # modules ask XLA for several host devices).
      mesh_shape=['1', '1', str(jax.device_count())],
      lm_format='QwenV2Chat',
      batch_size=2,
      max_seq_len=128,
      num_eval_threads=2,
      save_every_n=2,
      temperature=0.6,
      top_p=0.95,
      top_k=20,
      **flag_values,
  )
  saved = {name: flags.FLAGS[name].value for name in overrides}
  try:
    for name, value in overrides.items():
      flags.FLAGS[name].value = value
    eval_main.main([''])
  finally:
    for name, value in saved.items():
      flags.FLAGS[name].value = value


if __name__ == '__main__':
  absltest.main()

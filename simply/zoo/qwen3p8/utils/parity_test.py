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
"""Tests for the released-weights parity gate.

The gate itself can only run against the 27B checkpoint and a golden dumped
from HuggingFace torch, so what is checked here is everything it computes from
those two: the statistics against hand-computed values, the hidden-state
alignment against HF's documented convention, the thresholds on both sides of
each bound, and -- against a stub model whose forward pass is three additions
-- the report, the exit code and the order in which `run` does things.
"""

import builtins
import dataclasses
import json
import math
import os
import re
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from simply.zoo.qwen3p8.utils import parity


def _hide_numbers(line: str) -> str:
  """Blanks the measured values, keeping the labels, bounds and layout."""
  line = re.sub(r'in \d+\.\d s$', 'in # s', line)
  line = re.sub(r'(?<=\w)=\s*-?[\d.][\w.+-]*', '=#', line)  # Numbers only.
  line = re.sub(r'^(  \[(?:PASS|FAIL)\] \w+): \S+ vs ', r'\1: # vs ', line)
  return re.sub(r'^  h\[\s*\d+\]', '  h[#]', line)


def _kl(ref_logits: list[float], got_logits: list[float]) -> float:
  """KL(softmax(ref) || softmax(got)), transcribed from the definition."""
  p = [math.exp(v) for v in ref_logits]
  p = [v / sum(p) for v in p]
  q = [math.exp(v) for v in got_logits]
  q = [v / sum(q) for v in q]
  return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))


class DiffStatsTest(parameterized.TestCase):

  def test_identical_inputs_are_exactly_equal_and_cosine_does_not_exceed_one(
      self,
  ):
    # The float64 reduction exists for this: in float32 a vocab-sized dot
    # product reports cosine > 1 for identical inputs.
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(7, 4096)) * 20.0).astype(np.float32)
    stats = parity.diff_stats(x, x)
    self.assertEqual(stats['max_abs'], 0.0)
    self.assertEqual(stats['mean_abs'], 0.0)
    self.assertEqual(stats['rel_fro'], 0.0)
    self.assertLessEqual(stats['cosine'], 1.0)
    self.assertAlmostEqual(stats['cosine'], 1.0, places=12)

  def test_matches_hand_computed_values(self):
    got = np.array([[1.0, 2.0], [3.0, 4.0]])
    ref = np.array([[1.0, 2.0], [3.0, 5.0]])
    stats = parity.diff_stats(got, ref)
    self.assertAlmostEqual(stats['max_abs'], 1.0)
    self.assertAlmostEqual(stats['mean_abs'], 0.25)  # one diff of 1 over 4.
    self.assertAlmostEqual(stats['rel_fro'], 1.0 / math.sqrt(39.0), places=12)
    self.assertAlmostEqual(
        stats['cosine'], 34.0 / math.sqrt(30.0 * 39.0), places=12
    )

  @parameterized.named_parameters(
      ('orthogonal', [1.0, 0.0], [0.0, 1.0], 0.0),
      ('antiparallel', [1.0, 2.0], [-1.0, -2.0], -1.0),
      ('scaled', [1.0, 2.0], [3.0, 6.0], 1.0),
  )
  def test_cosine_is_direction_only(self, got, ref, expected):
    stats = parity.diff_stats(np.array([got]), np.array([ref]))
    self.assertAlmostEqual(stats['cosine'], expected, places=12)

  def test_zero_reference_gives_nan_cosine_rather_than_dividing_by_zero(self):
    stats = parity.diff_stats(np.zeros((2, 3)), np.zeros((2, 3)))
    self.assertTrue(math.isnan(stats['cosine']))
    self.assertEqual(stats['rel_fro'], 0.0)  # The 1e-12 floor in the denom.


class LogitsStatsTest(parameterized.TestCase):

  def test_identical_logits_have_no_divergence(self):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 32)).astype(np.float32) * 10.0
    stats = parity.logits_stats(x, x)
    self.assertEqual(stats['top1_agreement'], 1.0)
    self.assertAlmostEqual(stats['kl_mean'], 0.0, places=15)
    self.assertAlmostEqual(stats['kl_max'], 0.0, places=15)
    self.assertEqual(stats['top5_agreement_last_pos'], 1.0)

  def test_kl_matches_the_definition_and_is_reported_as_mean_and_max(self):
    ref = np.array([[0.0, math.log(2.0), math.log(4.0)], [0.0, 0.0, 0.0]])
    got = np.array([[0.0, 0.0, math.log(2.0)], [0.0, 0.0, 0.0]])
    expected = _kl(list(ref[0]), list(got[0]))
    self.assertGreater(expected, 0.0)
    stats = parity.logits_stats(got, ref)
    self.assertAlmostEqual(stats['kl_max'], expected, places=12)
    self.assertAlmostEqual(stats['kl_mean'], expected / 2.0, places=12)

  def test_kl_is_measured_on_distributions_not_on_raw_logits(self):
    # A per-row constant is invisible to softmax; without the log-softmax it
    # would not be.
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(4, 9))
    got = ref + np.array([[0.0], [5.0], [-3.0], [100.0]])
    stats = parity.logits_stats(got, ref)
    self.assertAlmostEqual(stats['kl_mean'], 0.0, places=12)
    self.assertEqual(stats['top1_agreement'], 1.0)
    self.assertEqual(stats['max_abs'], 100.0)  # ... but the diff is not.

  def test_one_flipped_argmax_costs_exactly_one_position(self):
    ref = np.array([[3.0, 1.0], [3.0, 1.0], [3.0, 1.0], [3.0, 1.0]])
    got = ref.copy()
    got[2] = [1.0, 3.0]
    stats = parity.logits_stats(got, ref)
    self.assertEqual(stats['top1_agreement'], 0.75)
    self.assertGreater(stats['kl_mean'], 0.0)

  def test_top5_agreement_reads_the_last_position_only(self):
    descending = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    ascending = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ref = np.array([descending, ascending])
    got = np.array([ascending, ascending])
    # Row 0 disagrees on four of its top five; only row 1 (identical) counts.
    self.assertEqual(
        parity.logits_stats(got, ref)['top5_agreement_last_pos'], 1.0
    )
    self.assertEqual(parity.logits_stats(got, ref)['top1_agreement'], 0.5)

  def test_top5_agreement_saturates_below_one_for_a_tiny_vocabulary(self):
    # A wart worth knowing before quoting the number: the denominator is 5
    # whatever the vocabulary is, so a 3-token vocabulary caps at 0.6 even for
    # identical logits. The released vocabulary is 248320 wide.
    x = np.array([[1.0, 2.0, 3.0]])
    self.assertEqual(parity.logits_stats(x, x)['top5_agreement_last_pos'], 0.6)

  def test_carries_the_diff_stats_through(self):
    got = np.array([[1.0, 2.0], [3.0, 4.0]])
    ref = np.array([[1.0, 2.0], [3.0, 5.0]])
    stats = parity.logits_stats(got, ref)
    for key, value in parity.diff_stats(got, ref).items():
      self.assertEqual(stats[key], value)


class AlignHiddensTest(parameterized.TestCase):
  """`align_hiddens` maps 66 Simply hiddens onto HF's 65 for the released 27B.

  Off by one, every cosine here would still be ~0.999+ (the residual stream
  barely moves per layer) and the gate would still pass; hence the mapping,
  not the count, is what these assert.
  """

  def _stacks(self, num_blocks: int) -> tuple[list[np.ndarray],
                                              list[np.ndarray]]:
    """Our hiddens and HF's, from the same made-up residual stream."""
    embeddings = np.full((2, 3), 100.0)
    block_out = [np.full((2, 3), 100.0 + i + 1) for i in range(num_blocks)]
    final_norm = np.full((2, 3), -1.0)
    ours = [embeddings, *block_out, final_norm]
    # HF appends the *input* of each layer, then the normed final output.
    hf = [embeddings, *block_out[:-1], final_norm]
    return ours, hf

  def test_maps_every_hidden_to_the_hf_slot_it_belongs_in(self):
    ours, hf = self._stacks(num_blocks=64)  # The released depth.
    self.assertLen(ours, 66)
    self.assertLen(hf, 65)
    aligned = parity.align_hiddens(ours, len(hf))
    self.assertLen(aligned, 65)
    for i, (got, want) in enumerate(zip(aligned, hf, strict=True)):
      np.testing.assert_array_equal(got, want, err_msg=f'at hidden {i}')

  @parameterized.named_parameters(
      # Every plausible off-by-one against the correct mapping.
      ('drop_the_final_norm', lambda h, n: list(h[:n])),
      ('keep_the_unnormed_last_block', lambda h, n: [*h[: n - 1], h[-2]]),
      ('drop_the_embeddings', lambda h, n: list(h[1 : n + 1])),
      ('shift_by_one', lambda h, n: [*h[1:n], h[-1]]),
  )
  def test_rejects_off_by_one_reorderings(self, wrong_fn):
    ours, hf = self._stacks(num_blocks=8)
    wrong = wrong_fn(ours, len(hf))
    self.assertLen(wrong, len(hf))  # A count check would not notice.
    aligned = parity.align_hiddens(ours, len(hf))
    self.assertFalse(
        all(np.array_equal(a, w) for a, w in zip(aligned, wrong, strict=True)),
        'align_hiddens agrees with an off-by-one mapping',
    )

  def test_an_off_by_one_is_invisible_to_the_cosine_check(self):
    # Why the mapping is asserted directly: a residual stream that grows by 1%
    # per layer still cosines above the 0.999 gate when shifted by one layer.
    rng = np.random.default_rng(3)
    stream = [rng.normal(size=(4, 16))]
    for _ in range(8):
      stream.append(stream[-1] + 0.01 * rng.normal(size=(4, 16)))
    hf = stream[:-1]
    ours = [*stream, stream[-1] * 1.0]
    shifted = ours[1 : len(hf) + 1]
    worst = min(
        parity.diff_stats(a, b)['cosine']
        for a, b in zip(shifted, hf, strict=True)
    )
    self.assertGreater(worst, 0.999)

  @parameterized.named_parameters(('same_length', 10), ('much_shorter', 3))
  def test_any_other_reference_length_is_a_plain_truncation(self, num_ref):
    ours, _ = self._stacks(num_blocks=8)  # 10 hiddens.
    aligned = parity.align_hiddens(ours, num_ref)
    self.assertLen(aligned, num_ref)
    for got, want in zip(aligned, ours[:num_ref], strict=True):
      np.testing.assert_array_equal(got, want)


class EvaluateChecksTest(parameterized.TestCase):

  def _report(self, **kwargs: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        'logits': {'max_abs': 0.1875, 'kl_mean': 4.79e-4,
                   'top1_agreement': 1.0}
    }
    report.update(kwargs)
    return report

  def test_the_released_measurements_pass_the_shipped_defaults(self):
    # The five numbers the README quotes, against the flag defaults.
    report = self._report(
        hidden=[{'cosine': 0.999926}, {'cosine': 0.9999}],
        greedy={'compared_steps': 16, 'match': True},
    )
    checks = parity.evaluate_checks(report, parity.Thresholds())
    self.assertEqual(
        [c['name'] for c in checks],
        [
            'logits_max_abs',
            'logits_kl_mean',
            'top1_agreement',
            'min_hidden_cosine',
            'greedy_match_16_steps',
        ],
    )
    self.assertTrue(all(c['passed'] for c in checks))

  @parameterized.named_parameters(
      # (stat, value on the failing side, value on the passing side).
      ('max_abs', 'max_abs', 0.5000001, 0.5),
      ('kl_mean', 'kl_mean', 5.0001e-3, 5e-3),
      ('top1', 'top1_agreement', 0.9899999, 0.99),
  )
  def test_each_logits_bound_fails_on_the_wrong_side_only(
      self, stat, bad, good
  ):
    name = {'max_abs': 'logits_max_abs', 'kl_mean': 'logits_kl_mean',
            'top1_agreement': 'top1_agreement'}[stat]
    for value, expected in ((bad, False), (good, True)):
      report = self._report()
      report['logits'][stat] = value
      checks = {
          c['name']: c
          for c in parity.evaluate_checks(report, parity.Thresholds())
      }
      self.assertEqual(checks[name]['passed'], expected, f'{stat}={value}')
      self.assertEqual(checks[name]['value'], value)

  @parameterized.named_parameters(
      ('below', 0.9989999, False), ('at', 0.999, True), ('above', 0.9999, True)
  )
  def test_the_hidden_cosine_bound_is_on_the_worst_layer(self, worst, expected):
    report = self._report(
        hidden=[{'cosine': 1.0}, {'cosine': worst}, {'cosine': 0.99999}]
    )
    check = parity.evaluate_checks(report, parity.Thresholds())[3]
    self.assertEqual(check['name'], 'min_hidden_cosine')
    self.assertEqual(check['value'], worst)
    self.assertEqual(check['passed'], expected)

  def test_the_hidden_check_is_absent_without_per_layer(self):
    checks = parity.evaluate_checks(self._report(), parity.Thresholds())
    self.assertNotIn('min_hidden_cosine', [c['name'] for c in checks])

  @parameterized.named_parameters(('match', True), ('mismatch', False))
  def test_the_greedy_check_follows_the_comparison(self, match):
    report = self._report(greedy={'compared_steps': 16, 'match': match})
    check = parity.evaluate_checks(report, parity.Thresholds())[-1]
    self.assertEqual(check['name'], 'greedy_match_16_steps')
    self.assertEqual(check['passed'], match)

  def test_require_greedy_match_false_drops_the_check(self):
    report = self._report(greedy={'compared_steps': 16, 'match': False})
    checks = parity.evaluate_checks(
        report, parity.Thresholds(require_greedy_match=False)
    )
    self.assertLen(checks, 3)
    self.assertTrue(all(c['passed'] for c in checks))

  def test_a_zero_step_greedy_comparison_would_pass_vacuously(self):
    # Which is what `check_greedy_reference` exists to prevent; see
    # `RunTest.test_a_golden_without_a_continuation_raises_before_the_restore`.
    report = self._report(greedy={'compared_steps': 0, 'match': True})
    check = parity.evaluate_checks(report, parity.Thresholds())[-1]
    self.assertEqual(check['name'], 'greedy_match_0_steps')
    self.assertTrue(check['passed'])

  def test_relaxed_thresholds_accept_the_64_token_chat_golden(self):
    # The measurements the module docstring records for the longer prompt.
    report = self._report()
    report['logits'].update({'max_abs': 4.8, 'top1_agreement': 0.97})
    self.assertFalse(
        all(c['passed'] for c in parity.evaluate_checks(
            report, parity.Thresholds()))
    )
    relaxed = parity.Thresholds(max_abs_diff=5.0, min_top1=0.96)
    self.assertTrue(
        all(c['passed'] for c in parity.evaluate_checks(report, relaxed))
    )


class GreedyContinueTest(absltest.TestCase):
  """Cache-free greedy decoding on a fixed-length buffer."""

  def setUp(self):
    super().setUp()
    self.buffers: list[np.ndarray] = []

  def _forward_fn(self, params: Any, ids: Any) -> Any:
    """Predicts `token + 1` at every position, and records what it was fed."""
    del params
    ids = np.asarray(ids)
    self.buffers.append(ids.copy())
    vocab = 10
    logits = np.zeros(ids.shape + (vocab,), dtype=np.float32)
    for b in range(ids.shape[0]):
      for t in range(ids.shape[1]):
        logits[b, t, (ids[b, t] + 1) % vocab] = 1.0
    return jnp.asarray(logits)

  def test_generates_from_the_position_before_the_one_it_writes(self):
    got = parity.greedy_continue(
        self._forward_fn, None, np.array([2, 3, 4]), num_steps=4, pad_id=9
    )
    np.testing.assert_array_equal(got, [5, 6, 7, 8])

  def test_feeds_a_constant_shape_buffer_padded_past_the_current_step(self):
    prompt = np.array([2, 3, 4])
    parity.greedy_continue(
        self._forward_fn, None, prompt, num_steps=3, pad_id=7
    )
    self.assertLen(self.buffers, 3)  # One forward per step, one compilation.
    for step, buf in enumerate(self.buffers):
      self.assertEqual(buf.shape, (1, 6))
      np.testing.assert_array_equal(buf[0, :3], prompt)
      np.testing.assert_array_equal(buf[0, 3 : 3 + step], [5, 6][:step])
      np.testing.assert_array_equal(buf[0, 3 + step :], [7] * (3 - step))

  def test_returns_nothing_for_zero_steps(self):
    got = parity.greedy_continue(
        self._forward_fn, None, np.array([2]), num_steps=0, pad_id=0
    )
    self.assertEmpty(got)
    self.assertEmpty(self.buffers)


@dataclasses.dataclass(frozen=True)
class _StubConfig:
  """The five config fields `run` reads."""

  init_ckpt_dir: str = '/tmp/not-a-checkpoint'
  output_logits_soft_cap: float | None = None
  vocab_name: str = 'unused'
  pad_id: int = 5  # Not 0: see `_StubModel.apply`.


class _StubEmbedLinear:
  """Untied embed/unembed so a stub can predict `token + 1`."""

  def embed(self, params: Any, ids: Any) -> Any:
    return params['embed'][ids]

  def apply(self, params: Any, x: Any) -> Any:
    return jnp.einsum('btd,vd->btv', x, params['unembed'])


class _StubBlock:

  def apply(self, params: Any, x: Any, **kwargs: Any) -> tuple[Any, None]:
    del kwargs  # segment_ids / segment_positions / extra_inputs.
    return x + params['delta'], None


class _StubLayerNorm:

  def apply(self, params: Any, x: Any) -> Any:
    return x * params['scale']


class _StubModel:
  """A model whose forward pass is `num_blocks` additions and a scaling."""

  def __init__(self, num_blocks: int):
    self.embed_linear = _StubEmbedLinear()
    self.blocks = [_StubBlock() for _ in range(num_blocks)]
    self.final_ln = _StubLayerNorm()

  def apply(self, params: Any, ids: Any) -> tuple[Any, None]:
    logits, _ = parity.forward_per_layer(self, params, ids)
    # Token 0 is the "wrong padding" sentinel: it never occurs in a golden or
    # in a continuation, so seeing one means the greedy buffer was padded with
    # something other than `config.pad_id`, and the prediction collapses.
    poison = jnp.any(ids == 0).astype(logits.dtype)
    return logits - 10.0 * poison * jnp.arange(logits.shape[-1]), None


_VOCAB = 8  # Wide enough that a 4-step continuation never wraps to token 0.
_NUM_BLOCKS = 3


def _stub_params() -> dict[str, Any]:
  """Predicts `token + 1`: the unembedding is the embedding shifted by one."""
  eye = np.eye(_VOCAB, dtype=np.float32)
  return {
      'embed_linear': {
          'embed': jnp.asarray(eye),
          'unembed': jnp.asarray(np.roll(eye, 1, axis=0)),
      },
      **{
          f'block_{i}': {'delta': jnp.asarray(
              np.full(_VOCAB, 0.01 * (i + 1), dtype=np.float32)
          )}
          for i in range(_NUM_BLOCKS)
      },
      'final_ln': {'scale': jnp.asarray(np.float32(2.0))},
  }


def _stub_forward(input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The stub's logits and HF-convention hidden states, in NumPy."""
  eye = np.eye(_VOCAB, dtype=np.float32)
  x = eye[input_ids]
  hiddens = [x]
  for i in range(_NUM_BLOCKS):
    x = x + np.float32(0.01 * (i + 1))
    hiddens.append(x)
  final = x * np.float32(2.0)
  logits = final @ np.roll(eye, 1, axis=0).T
  return logits, np.stack([*hiddens[:-1], final])  # HF drops the last block.


class RunTest(parameterized.TestCase):
  """End-to-end over a stub model: the report, the checks and the exit code."""

  def _golden(self, decode_steps: int = 0, **overrides: Any) -> str:
    input_ids = np.array([1, 2, 3], dtype=np.int64)  # HF dumps int64.
    logits, hidden_states = _stub_forward(input_ids)
    arrays: dict[str, Any] = {
        'input_ids': input_ids,
        'logits': logits,
        'hidden_states': hidden_states,
        'decoded': np.array(
            [(3 + i) % _VOCAB for i in range(1, decode_steps + 1)],
            dtype=np.int64,
        ),
    }
    arrays.update(overrides)
    path = os.path.join(
        self.create_tempdir().full_path, f'golden_{decode_steps}.npz'
    )
    np.savez(path, **arrays)
    return path

  def _run(self, path: str, **option_overrides: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    options = parity.Options(ref_path=path, decode_text=False,
                             **option_overrides)
    code = parity.run(
        options,
        lambda: (_StubModel(_NUM_BLOCKS), _stub_params(), _StubConfig()),
        lines.append,
    )
    return code, lines

  def test_a_matching_model_passes_with_exit_code_zero(self):
    code, lines = self._run(self._golden())
    self.assertEqual(code, 0, '\n'.join(lines))
    self.assertEqual(lines[-1], 'PASS')
    self.assertIn('  [PASS] top1_agreement: 1.0 vs >= 0.99', lines)
    self.assertIn(
        'per-layer hidden states (HF convention: 0=embeddings,'
        ' i=out(block_{i-1}), last=final norm):',
        lines,
    )
    # 4 hidden states for 3 blocks: the alignment ran and the strict zip held.
    self.assertLen([l for l in lines if l.startswith('  h[')], 4)

  def test_a_wrong_model_fails_with_exit_code_one(self):
    logits, hidden_states = _stub_forward(np.array([1, 2, 3], dtype=np.int64))
    logits = logits.copy()
    logits[2] = logits[2][::-1] * 3.0  # A different argmax at one position.
    path = self._golden(logits=logits, hidden_states=hidden_states)
    code, lines = self._run(path)
    self.assertEqual(code, 1)
    self.assertEqual(
        lines[-1],
        'FAIL: logits_max_abs, logits_kl_mean, top1_agreement',
    )

  def test_the_greedy_continuation_is_compared_and_gated(self):
    code, lines = self._run(self._golden(decode_steps=4), greedy_steps=4)
    self.assertEqual(code, 0, '\n'.join(lines))
    self.assertIn('greedy match over 4/4 steps: True', lines)
    code, lines = self._run(
        self._golden(decode_steps=4, decoded=np.array([0, 0, 0, 0])),
        greedy_steps=4,
    )
    self.assertEqual(code, 1)
    self.assertEqual(lines[-1], 'FAIL: greedy_match_4_steps')

  def test_the_json_report_holds_the_schema_the_readme_quotes(self):
    output_path = os.path.join(self.create_tempdir().full_path, 'report.json')
    code, _ = self._run(
        self._golden(decode_steps=2), greedy_steps=2, output_path=output_path
    )
    self.assertEqual(code, 0)
    with open(output_path) as f:
      report = json.load(f)
    self.assertContainsSubset(
        [
            'experiment_config',
            'ckpt_dir',
            'activation_dtype',
            'ref_path',
            'output_logits_soft_cap',
            'input_ids',
            'forward_seconds',
            'logits',
            'logits_per_layer_path',
            'logits_simply_path_delta',
            'hidden',
            'top10_last_pos',
            'greedy',
            'checks',
            'passed',
        ],
        report.keys(),
    )
    self.assertTrue(report['passed'])
    self.assertEqual(report['greedy']['compared_steps'], 2)
    self.assertEqual(report['greedy']['requested_steps'], 2)
    self.assertContainsSubset(
        ['max_abs', 'mean_abs', 'rel_fro', 'cosine', 'top1_agreement',
         'kl_mean', 'kl_max', 'top5_agreement_last_pos'],
        report['logits'].keys(),
    )

  def test_per_layer_false_skips_the_hidden_states(self):
    code, lines = self._run(self._golden(), per_layer=False)
    self.assertEqual(code, 0)
    self.assertEmpty([l for l in lines if l.startswith('  h[')])
    self.assertNotIn('min_hidden_cosine', '\n'.join(lines))

  def test_a_golden_without_a_continuation_raises_before_the_restore(self):
    # The vacuous match: with an empty reference the comparison is over zero
    # steps and reports True (EvaluateChecksTest pins that), so this must fail
    # before the multi-minute checkpoint restore is even attempted.
    path = self._golden(decoded=np.zeros(0, dtype=np.int64))
    builds = []

    def _build():
      builds.append(1)
      raise AssertionError('the checkpoint restore must not be reached')

    options = parity.Options(ref_path=path, greedy_steps=16)
    with self.assertRaisesRegex(ValueError, 'carries no reference cont'):
      parity.run(options, _build, lambda line: None)
    self.assertEmpty(builds)

  def test_a_golden_missing_the_continuation_key_is_read_as_having_none(self):
    ref = parity.load_reference(self._golden_without_decoded())
    self.assertEmpty(ref.decoded)
    with self.assertRaises(ValueError):
      parity.check_greedy_reference(ref, 16)
    parity.check_greedy_reference(ref, 0)  # Must not raise.

  def _golden_without_decoded(self) -> str:
    path = os.path.join(self.create_tempdir().full_path, 'no_decoded.npz')
    input_ids = np.array([1, 2, 3], dtype=np.int64)
    logits, hidden_states = _stub_forward(input_ids)
    np.savez(
        path, input_ids=input_ids, logits=logits, hidden_states=hidden_states
    )
    return path

  def test_a_golden_without_a_continuation_still_runs_without_greedy_steps(
      self,
  ):
    code, _ = self._run(self._golden_without_decoded())
    self.assertEqual(code, 0)

  def test_load_reference_reads_what_the_dump_writes(self):
    ref = parity.load_reference(self._golden(decode_steps=3))
    np.testing.assert_array_equal(ref.input_ids, [1, 2, 3])
    self.assertEqual(ref.input_ids.dtype, np.int32)  # The golden is int64.
    self.assertEqual(ref.logits.shape, (3, _VOCAB))
    self.assertEqual(ref.hidden_states.shape, (_NUM_BLOCKS + 1, 3, _VOCAB))
    self.assertLen(ref.decoded, 3)

  def test_a_short_reference_is_compared_over_the_overlap_only(self):
    output_path = os.path.join(self.create_tempdir().full_path, 'short.json')
    code, lines = self._run(
        self._golden(decode_steps=2), greedy_steps=4, output_path=output_path
    )
    self.assertEqual(code, 0, chr(10).join(lines))
    self.assertIn('greedy match over 2/4 steps: True', lines)
    self.assertIn('  [PASS] greedy_match_2_steps: True vs is True', lines)
    with open(output_path) as f:
      greedy = json.load(f)['greedy']
    self.assertEqual(greedy['compared_steps'], 2)
    self.assertEqual(greedy['requested_steps'], 4)
    self.assertEqual(greedy['got'], [4, 5, 6, 7])  # All four were generated.

  def test_the_greedy_buffer_is_padded_with_the_configs_pad_id(self):
    # `_StubModel` collapses onto token 0 if it ever sees one, and token 0 is
    # neither in the prompt nor in the continuation, so this only holds if
    # `run` padded the buffer with `_StubConfig.pad_id`.
    code, lines = self._run(self._golden(decode_steps=4), greedy_steps=4)
    self.assertEqual(code, 0, chr(10).join(lines))
    self.assertIn(f'  got: {[4, 5, 6, 7]}', lines)

  def test_the_json_report_is_rewritten_on_every_line(self):
    # "Partial results survive a crash / a kill": a 27B run is nine minutes.
    output_path = os.path.join(self.create_tempdir().full_path, 'partial.json')
    options = parity.Options(
        ref_path=self._golden(), decode_text=False, output_path=output_path
    )

    def _print_then_die(line: str) -> None:
      if line.startswith('checks:'):
        raise KeyboardInterrupt('killed mid-report')

    with self.assertRaises(KeyboardInterrupt):
      parity.run(
          options,
          lambda: (_StubModel(_NUM_BLOCKS), _stub_params(), _StubConfig()),
          _print_then_die,
      )
    with open(output_path) as f:
      partial = json.load(f)
    self.assertIn('hidden', partial)  # Everything measured before the kill.
    self.assertNotIn('checks', partial)

  def test_the_default_printer_flushes(self):
    # The gate is watched through nohup; unflushed output arrives at the end.
    with mock.patch.object(builtins, 'print') as mock_print:
      parity.run(
          parity.Options(ref_path=self._golden(), per_layer=False),
          lambda: (_StubModel(_NUM_BLOCKS), _stub_params(), _StubConfig()),
      )
    mock_print.assert_called_with('PASS', flush=True)

  def test_the_printed_report_is_the_one_the_readme_quotes(self):
    # Everything but the measured numbers: the labels, their order and widths,
    # the stat names, the bound strings and the PASS/FAIL markers. The values
    # themselves are covered by the stats tests above.
    path = self._golden(decode_steps=2)
    _, lines = self._run(path, greedy_steps=2)
    self.maxDiff = None
    diff_keys = 'max_abs=# mean_abs=# rel_fro=# cosine=#'
    logit_keys = (
        diff_keys
        + ' top1_agreement=# kl_mean=# kl_max=# top5_agreement_last_pos=#'
    )
    self.assertEqual(
        [_hide_numbers(l) for l in lines],
        [
            '==== Qwen3.8 real-weight parity vs HF golden ====',
            'experiment_config : qwen3p8_27b',
            'ckpt_dir          : /tmp/not-a-checkpoint',
            'activation_dtype  : bfloat16',
            f'ref_path          : {path}',
            'output_logits_soft_cap: None',
            'input_ids         : [1, 2, 3]',
            'model.apply forward in # s',
            f'logits: {logit_keys}',
            'per-layer forward in # s',
            f'logits_per_layer_path: {logit_keys}',
            'logits_simply_path_delta (model.apply vs per-layer, both'
            f' Simply): {diff_keys}',
            'top-10 reference logits at the last position:',
            *['  token=# ref=# model_apply=# per_layer=#'] * _VOCAB,
            'per-layer hidden states (HF convention: 0=embeddings,'
            + ' i=out(block_{i-1}), last=final norm):',
            *['  h[#] max_abs=# mean_abs=# rel_fro=# cos=#']
            * (_NUM_BLOCKS + 1),
            'greedy continuation in # s',
            'greedy match over 2/2 steps: True',
            '  got: [4, 5]',
            '  ref: [4, 5]',
            'checks:',
            '  [PASS] logits_max_abs: # vs <= 0.5',
            '  [PASS] logits_kl_mean: # vs <= 0.005',
            '  [PASS] top1_agreement: # vs >= 0.99',
            '  [PASS] min_hidden_cosine: # vs >= 0.999',
            '  [PASS] greedy_match_2_steps: # vs is True',
            'PASS',
        ],
    )


if __name__ == '__main__':
  absltest.main()

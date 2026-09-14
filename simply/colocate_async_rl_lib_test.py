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
"""Unit test for colocate_async_rl_lib.py."""

# Same reason as in colocate_async_rl_lib.py: `PyTree`-typed values and the
# optional `Sample.reward` are not statically indexable/comparable.

from collections.abc import Sequence
import dataclasses
import functools
import itertools
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from simply import colocate_async_rl_lib
from simply import config_lib
from simply import model_lib
from simply.serving import page_batcher

_VOCAB_SIZE = 32


def _sequence(
    length: int,
    *,
    input_len: int = 1,
    advantage: float = 0.0,
    seed: int = 0,
) -> colocate_async_rl_lib.TrainSequence:
  """A `[length]` sequence of random tokens, trainable after `input_len`."""
  rng = np.random.default_rng(seed)
  return colocate_async_rl_lib.TrainSequence.from_completion(
      tokens=rng.integers(1, _VOCAB_SIZE, size=length).astype(np.int32),
      logprobs=rng.normal(size=length).astype(np.float32),
      input_len=input_len,
      advantage=advantage,
  )


def _sample(
    length: Any = 4,
    *,
    input_len: Any = 1,
    advantage: Any = 1.0,
    reward: Any = None,
    correct: Any = None,
    valid: Any = True,
    prompt_index: Any = 0,
    seed: Any = 0,
    truncated: Any = False,
    **kwargs: Any,
) -> colocate_async_rl_lib.Sample:
  del kwargs
  return colocate_async_rl_lib.Sample(
      sequence=_sequence(
          int(length),
          input_len=int(input_len),
          advantage=float(advantage),
          seed=seed,
      ),
      reward=reward,
      correct=correct,
      is_valid_for_training=valid,
      prompt_index=prompt_index,
      truncated=truncated,
  )


def _seq(
    sample: colocate_async_rl_lib.Sample,
) -> colocate_async_rl_lib.TrainSequence:
  assert sample.sequence is not None
  return sample.sequence


def _group(k: int, num_correct: int) -> list[colocate_async_rl_lib.Sample]:
  """A scored group of `k` rollouts, the first `num_correct` of them right."""
  return [
      _sample(4, reward=float(i < num_correct), correct=i < num_correct)
      for i in range(k)
  ]


def _segment_runs(segment_ids: Any) -> list[tuple[int, int, int]]:
  """Splits a row's `segment_ids` into `(segment_id, start, end)` runs."""
  runs = []
  start = 0
  for i in range(1, len(segment_ids) + 1):
    if i == len(segment_ids) or segment_ids[i] != segment_ids[start]:
      runs.append((int(segment_ids[start]), start, i))
      start = i
  return runs


class TrainSequenceTest(absltest.TestCase):

  def test_from_completion_marks_the_tail_after_the_prompt(self):
    tokens = np.arange(5, dtype=np.int32)
    logprobs = np.arange(5, dtype=np.float32) * -0.5
    sequence = colocate_async_rl_lib.TrainSequence.from_completion(
        tokens=tokens, logprobs=logprobs, input_len=2, advantage=1.5
    )
    self.assertLen(sequence, 5)
    # Index-aligned with `tokens`; index 0 is the un-predicted BOS.
    np.testing.assert_array_equal(sequence.tokens, tokens)
    np.testing.assert_array_equal(sequence.logprobs, logprobs)
    np.testing.assert_array_equal(
        sequence.mask, [False, False, True, True, True]
    )
    self.assertIsNotNone(sequence.advantages)
    assert sequence.advantages is not None
    np.testing.assert_allclose(sequence.advantages, [0.0, 0.0, 1.5, 1.5, 1.5])
    self.assertEqual(sequence.tokens.dtype, np.int32)
    self.assertEqual(sequence.advantages.dtype, np.float32)

  def test_from_completion_with_no_output_has_nothing_trainable(self):
    sequence = _sequence(4, input_len=4, advantage=1.0)
    np.testing.assert_array_equal(sequence.mask, False)
    self.assertIsNotNone(sequence.advantages)
    assert sequence.advantages is not None
    np.testing.assert_array_equal(sequence.advantages, 0.0)

  def test_with_advantage_restamps_only_the_trainable_span(self):
    sequence = _sequence(5, input_len=2, advantage=1.5)
    restamped = sequence.with_advantage(-0.25)
    self.assertIsNotNone(restamped.advantages)
    assert restamped.advantages is not None
    np.testing.assert_allclose(
        restamped.advantages, [0.0, 0.0, -0.25, -0.25, -0.25]
    )
    # Everything else is untouched.
    np.testing.assert_array_equal(restamped.tokens, sequence.tokens)
    np.testing.assert_array_equal(restamped.logprobs, sequence.logprobs)
    np.testing.assert_array_equal(restamped.mask, sequence.mask)

  def test_train_sequence_is_none_when_there_is_nothing_to_train(self):
    self.assertIsNone(_sample(4, input_len=4).train_sequence())
    self.assertIsNone(_sample(4, input_len=1, valid=False).train_sequence())
    self.assertIsNone(
        _sample(4, input_len=1, advantage=0.0).train_sequence()
    )
    sample = _sample(4, input_len=1)
    self.assertIs(sample.train_sequence(), sample.sequence)


class PackTest(absltest.TestCase):
  """`TrainBatch.pack` packs sequences end to end into rows."""

  def test_packs_several_sequences_into_one_row(self):
    batch = colocate_async_rl_lib.TrainBatch.create(
        batch_size=2, max_seq_len=9
    )
    self.assertEqual(batch.max_seq_len, 10)
    first = _sequence(4, input_len=1, advantage=1.0, seed=1)
    second = _sequence(3, input_len=2, advantage=-2.0, seed=2)
    self.assertTrue(batch.pack(first))
    self.assertTrue(batch.pack(second))

    np.testing.assert_array_equal(batch.lens, [7, 0])
    np.testing.assert_array_equal(
        batch.segment_ids[0], [1, 1, 1, 1, 2, 2, 2, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        batch.segment_positions[0], [0, 1, 2, 3, 0, 1, 2, 0, 0, 0]
    )
    for field in ('tokens', 'logprobs', 'mask', 'advantages'):
      row = getattr(batch, field)[0]
      np.testing.assert_array_equal(row[0:4], getattr(first, field))
      np.testing.assert_array_equal(row[4:7], getattr(second, field))
      np.testing.assert_array_equal(row[7:], np.zeros_like(row[7:]))
    # The untouched row stays zero.
    for leaf in jax.tree.leaves(batch):
      np.testing.assert_array_equal(leaf[1], np.zeros_like(leaf[1]))

  def test_the_cross_segment_prediction_pair_is_masked_out(self):
    # The loss pairs input `tokens[i]` with target `tokens[i + 1]`, so the
    # boundary between two segments creates a pair that crosses sequences. It
    # is dropped iff the first token of every non-first segment is unmasked.
    batch = colocate_async_rl_lib.TrainBatch.create(
        batch_size=1, max_seq_len=15
    )
    for i, length in enumerate([6, 5, 4]):
      self.assertTrue(batch.pack(_sequence(length, input_len=2, seed=i)))
    for segment_id, start, _ in _segment_runs(batch.segment_ids[0]):
      if segment_id > 1:
        self.assertFalse(
            batch.mask[0, start],
            msg=f'segment {segment_id} starts trainable at {start}',
        )
    # Equivalently, on the target-aligned view no trained token is predicted
    # from a token of another segment.
    inputs_segment = batch.segment_ids[0, :-1]
    targets_segment = batch.segment_ids[0, 1:]
    trained = batch.mask[0, 1:]
    np.testing.assert_array_equal(
        inputs_segment[trained], targets_segment[trained]
    )

  def test_spills_to_the_next_row_and_reports_a_full_batch(self):
    batch = colocate_async_rl_lib.TrainBatch.create(
        batch_size=2, max_seq_len=9
    )
    self.assertTrue(batch.pack(_sequence(7, seed=1)))
    # 7 + 5 > 10, so this one starts the next row.
    self.assertTrue(batch.pack(_sequence(5, seed=2)))
    np.testing.assert_array_equal(batch.lens, [7, 5])
    # Back to the first row with room.
    self.assertTrue(batch.pack(_sequence(3, seed=3)))
    np.testing.assert_array_equal(batch.lens, [10, 5])
    np.testing.assert_array_equal(batch.segment_ids[0, 7:], 2)
    # No row can take 6 more tokens.
    self.assertFalse(batch.pack(_sequence(6, seed=4)))
    np.testing.assert_array_equal(batch.lens, [10, 5])

  def test_raises_when_a_sequence_cannot_fit_an_empty_row(self):
    batch = colocate_async_rl_lib.TrainBatch.create(
        batch_size=2, max_seq_len=4
    )
    with self.assertRaisesRegex(ValueError, 'exceeds the row length'):
      batch.pack(_sequence(6))


class CreateTrainBatchesTest(absltest.TestCase):

  def test_packs_sequences_into_micro_batches(self):
    sequences = [
        _sequence(6, input_len=2, seed=1),
        _sequence(5, input_len=2, seed=4),
    ]
    micro_batches = colocate_async_rl_lib.create_train_batches(
        sequences, rows_per_micro_batch=2, max_seq_len=10, is_primary=True
    )
    self.assertLen(micro_batches, 1)
    batch = micro_batches[0]
    self.assertEqual(batch.tokens.shape, (2, 11))
    # Both surviving sequences fit the first row (6 + 5 = 11).
    np.testing.assert_array_equal(batch.lens, [11, 0])
    np.testing.assert_array_equal(batch.tokens[0, :6], sequences[0].tokens)
    np.testing.assert_array_equal(batch.tokens[0, 6:], sequences[1].tokens)

  def test_opens_a_new_micro_batch_when_the_last_one_is_full(self):
    sequences = [_sequence(6, input_len=2, seed=i) for i in range(5)]
    micro_batches = colocate_async_rl_lib.create_train_batches(
        sequences, rows_per_micro_batch=2, max_seq_len=11, is_primary=True
    )
    # 12-token rows hold two sequences each: 4 sequences fill a micro-batch.
    self.assertLen(micro_batches, 2)
    for micro_batch in micro_batches:
      self.assertEqual(micro_batch.tokens.shape, (2, 12))
    np.testing.assert_array_equal(micro_batches[0].lens, [12, 12])
    np.testing.assert_array_equal(micro_batches[1].lens, [6, 0])

  def test_longest_sequence_goes_first(self):
    sequences = [
        _sequence(4, input_len=1, seed=1),
        _sequence(8, input_len=1, seed=2),
    ]
    (micro_batch,) = colocate_async_rl_lib.create_train_batches(
        sequences, rows_per_micro_batch=1, max_seq_len=11, is_primary=True
    )
    # Row contents follow decreasing length, not the input order.
    np.testing.assert_array_equal(
        micro_batch.tokens[0, :8], sequences[1].tokens
    )
    np.testing.assert_array_equal(
        micro_batch.tokens[0, 8:12], sequences[0].tokens
    )
    np.testing.assert_array_equal(micro_batch.segment_ids[0, :8], 1)
    np.testing.assert_array_equal(micro_batch.segment_ids[0, 8:], 2)

  def test_only_the_current_micro_batch_is_filled(self):
    # Rows of 12 tokens, one row per micro-batch. Longest first: 7 opens the
    # first micro-batch, 6 does not fit next to it so it opens the second, 5
    # joins the 6 (11 tokens), and 4 no longer fits the second -- even though
    # the FIRST still has 5 free tokens. Fill is sequential, so the 4 opens a
    # third micro-batch rather than going back.
    sequences = [_sequence(n, input_len=1, seed=n) for n in (7, 6, 5, 4)]
    micro_batches = colocate_async_rl_lib.create_train_batches(
        sequences, rows_per_micro_batch=1, max_seq_len=11, is_primary=True
    )
    self.assertLen(micro_batches, 3)
    np.testing.assert_array_equal(micro_batches[0].lens, [7])
    np.testing.assert_array_equal(micro_batches[1].lens, [11])
    np.testing.assert_array_equal(micro_batches[2].lens, [4])
    room = micro_batches[0].max_seq_len - int(micro_batches[0].lens[0])
    self.assertGreaterEqual(room, len(sequences[3]))

  def test_empty_sequences_yields_no_micro_batch(self):
    self.assertEmpty(
        colocate_async_rl_lib.create_train_batches(
            [],
            rows_per_micro_batch=2,
            max_seq_len=10,
            is_primary=True,
        )
    )


class NormalizeGroupRewardsTest(absltest.TestCase):

  def _advantages(self, sample: colocate_async_rl_lib.Sample) -> np.ndarray:
    advantages = _seq(sample).advantages
    assert advantages is not None
    return advantages

  def test_group_baseline_is_stamped_on_every_trainable_token(self):
    group = [
        _sample(5, input_len=2, reward=1.0, seed=1),
        _sample(5, input_len=2, reward=0.0, seed=2),
    ]
    (normalized,) = colocate_async_rl_lib.normalize_group_rewards(
        [group], 'ByGroup'
    )
    # The baseline lands on the ADVANTAGE, (r - 0.5) / std([1, 0]) = +/- 1;
    # `reward` keeps the raw evaluator score the records and stats report.
    self.assertEqual(normalized[0].reward, 1.0)
    self.assertEqual(normalized[1].reward, 0.0)
    np.testing.assert_allclose(
        self._advantages(normalized[0]), [0.0, 0.0, 1.0, 1.0, 1.0], atol=1e-5
    )
    np.testing.assert_allclose(
        self._advantages(normalized[1]), [0.0, 0.0, -1.0, -1.0, -1.0], atol=1e-5
    )
    np.testing.assert_array_equal(_seq(normalized[0]).mask, _seq(group[0]).mask)

  def test_std_normalization(self):
    group = [
        _sample(4, input_len=1, reward=1.0),
        _sample(4, input_len=1, reward=0.0),
    ]
    out = colocate_async_rl_lib.normalize_group_rewards([group], 'ByGroup')[0]
    np.testing.assert_allclose(
        self._advantages(out[0])[_seq(out[0]).mask], 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        self._advantages(out[1])[_seq(out[1]).mask], -1.0, atol=1e-5
    )

  def test_degenerate_group_gets_zero_advantage_but_keeps_its_mask(self):
    group = [
        _sample(4, input_len=1, reward=1.0, advantage=7.0, seed=1),
        _sample(4, input_len=1, reward=1.0, advantage=7.0, seed=2),
    ]
    (normalized,) = colocate_async_rl_lib.normalize_group_rewards(
        [group], 'ByGroup'
    )
    for sample in normalized:
      self.assertEqual(sample.reward, 1.0)
      np.testing.assert_array_equal(self._advantages(sample), 0.0)
      np.testing.assert_array_equal(
          _seq(sample).mask, [False, True, True, True]
      )

  def test_invalid_and_unscored_samples_are_left_alone(self):
    invalid = _sample(4, reward=5.0, advantage=7.0, valid=False, seed=1)
    unscored = _sample(4, reward=None, advantage=7.0, seed=2)
    nan_reward = _sample(4, reward=float('nan'), advantage=7.0, seed=3)
    group = [_sample(4, reward=1.0, seed=4), invalid, unscored, nan_reward]
    (normalized,) = colocate_async_rl_lib.normalize_group_rewards(
        [group], 'ByGroup'
    )
    self.assertEqual(normalized[1], invalid)
    self.assertEqual(normalized[2], unscored)
    self.assertEqual(normalized[3], nan_reward)
    # A single valid reward is its own baseline: zero advantage.
    np.testing.assert_array_equal(self._advantages(normalized[0]), 0.0)

  def test_empty_method_is_a_no_op(self):
    group = [_sample(4, reward=1.0, advantage=7.0)]
    (normalized,) = colocate_async_rl_lib.normalize_group_rewards([group], '')
    self.assertEqual(normalized[0], group[0])

  def test_unknown_method_raises(self):
    with self.assertRaisesRegex(ValueError, 'group baseline'):
      colocate_async_rl_lib.normalize_group_rewards(
          [[_sample(4, reward=1.0)]], 'Global'
      )

  def test_verdict_only_samples_pass_through(self):
    # Validation-eval samples carry no sequence; normalization must not choke
    # on them (and has nothing to stamp).
    verdict = colocate_async_rl_lib.Sample(correct=True, reward=1.0)
    (normalized,) = colocate_async_rl_lib.normalize_group_rewards(
        [[verdict, colocate_async_rl_lib.Sample(correct=False, reward=0.0)]],
        'ByGroup',
    )
    for sample in normalized:
      self.assertIsNone(sample.sequence)
      self.assertIsNone(sample.train_sequence())


class ComputeGroupStatsTest(absltest.TestCase):

  def test_pass_at_1_and_pass_at_k(self):
    groups = [
        [
            _sample(4, reward=1.0, correct=True),
            _sample(4, reward=0.0, correct=False),
        ],
        [
            _sample(4, reward=0.0, correct=False),
            _sample(4, reward=0.0, correct=False),
        ],
    ]
    stats = colocate_async_rl_lib.compute_group_stats(groups)
    self.assertAlmostEqual(stats['pass@1'], 0.25)
    self.assertAlmostEqual(stats['pass@k'], 0.5)

  def test_unscored_samples_are_ignored(self):
    self.assertEmpty(colocate_async_rl_lib.compute_group_stats([]))
    self.assertEmpty(
        colocate_async_rl_lib.compute_group_stats([[_sample(4, reward=1.0)]])
    )
    self.assertEmpty(
        colocate_async_rl_lib.compute_group_stats(
            [[_sample(4, reward=float('nan'), correct=True)]]
        )
    )

  def test_lengths_and_truncation_stats(self):
    s1 = _sample(10, input_len=3, reward=1.0, correct=True, truncated=False)
    s2 = _sample(15, input_len=5, reward=0.0, correct=False, truncated=True)
    stats = colocate_async_rl_lib.compute_group_stats([[s1, s2]])
    self.assertAlmostEqual(stats['pass@1'], 0.5)
    self.assertAlmostEqual(stats['pass@k'], 1.0)
    self.assertAlmostEqual(stats['truncated'], 0.5)
    self.assertAlmostEqual(stats['input_len/mean'], 4.0)
    self.assertAlmostEqual(stats['input_len/min'], 3.0)
    self.assertAlmostEqual(stats['input_len/max'], 5.0)
    self.assertAlmostEqual(stats['seq_len/mean'], 12.5)
    self.assertAlmostEqual(stats['seq_len/min'], 10.0)
    self.assertAlmostEqual(stats['seq_len/max'], 15.0)
    self.assertAlmostEqual(stats['response_len/mean'], 8.5)
    self.assertAlmostEqual(stats['response_len/min'], 7.0)
    self.assertAlmostEqual(stats['response_len/max'], 10.0)
    self.assertAlmostEqual(stats['response_len/mean/correct'], 7.0)
    self.assertAlmostEqual(stats['response_len/mean/incorrect'], 10.0)


class ColocateAsyncRLLibTest(absltest.TestCase):

  def test_seq_to_sample_reads_batcher_fields(self):
    seq = {
        'tokens': np.arange(5, dtype=np.int32),
        'input_len': 2,
        'logprobs': np.arange(5, dtype=np.float32),
        'truncated': True,
        'output_text': 'hello',
    }
    sample = colocate_async_rl_lib._seq_to_sample(seq, prompt_index=3, step=7)
    np.testing.assert_array_equal(_seq(sample).tokens, seq['tokens'])
    # Logprobs stay index-aligned with the tokens they score.
    np.testing.assert_array_equal(_seq(sample).logprobs, seq['logprobs'])
    np.testing.assert_array_equal(
        _seq(sample).mask, [False, False, True, True, True]
    )
    self.assertIsNone(_seq(sample).advantages)
    self.assertTrue(sample.truncated)
    self.assertEqual(sample.output_text, 'hello')
    self.assertEqual(sample.output_messages, ())
    self.assertEqual(sample.prompt_index, 3)
    self.assertEqual(sample.step, 7)

  def test_train_loop_is_registered(self):
    self.assertIs(
        model_lib.TrainLoopRegistry.get('colocate_async_rl'),
        colocate_async_rl_lib.rl_experiment,
    )

  def test_the_batcher_exposes_the_programs_the_loop_precompiles(self):
    # `rl_experiment` asks for each of these before it starts the sampling
    # thread; a rename in `page_batcher` should not wait for a TPU run.
    for name in (
        'compiled_prefill_fn',
        'compiled_decode_fn',
        'compiled_push_fn',
        'compiled_release_fn',
        'compiled_inject_chunk_fn',
        'compiled_extract_chunk_fn',
        'prefix_cache',
    ):
      self.assertTrue(hasattr(page_batcher.Batcher, name), name)


class MoveTreeTest(absltest.TestCase):
  """Host offloading utilities (`move_tree` and friends)."""

  def test_offload_onload_roundtrip(self):
    tree = {
        'params': {'w': jnp.arange(8.0).reshape(2, 4)},
        'steps': jnp.array(3),
    }
    # The move consumes its argument, so the values are read off first.
    expected = jax.tree.map(np.asarray, tree)
    host_tree = colocate_async_rl_lib.offload_to_host(tree)
    for leaf in jax.tree.leaves(host_tree):
      self.assertEqual(leaf.sharding.memory_kind, 'pinned_host')
    restored = colocate_async_rl_lib.onload_from_host(host_tree)
    for leaf in jax.tree.leaves(restored):
      self.assertEqual(leaf.sharding.memory_kind, 'device')
    np.testing.assert_array_equal(
        restored['params']['w'], expected['params']['w']
    )
    np.testing.assert_array_equal(restored['steps'], expected['steps'])

  def test_the_move_frees_each_source_as_its_transfer_is_issued(self):
    # THE POINT OF THE CHANGE. The deletes happen inside the loop, before
    # anything blocks; what this pins is that dropping the handle that early
    # does not corrupt the copy still reading it.
    tree = {'a': jnp.arange(4.0), 'b': {'c': jnp.ones((2, 2))}}
    expected = jax.tree.map(np.asarray, tree)
    moved = colocate_async_rl_lib.move_tree(tree, 'pinned_host')
    self.assertTrue(
        all(leaf.is_deleted() for leaf in jax.tree.leaves(tree)),
        'a source outlived the transfer that reads it',
    )
    jax.block_until_ready(moved)
    jax.tree.map(
        np.testing.assert_array_equal, jax.tree.map(np.asarray, moved), expected
    )

  def test_a_leaf_already_in_the_target_memory_is_left_alone(self):
    # `device_put` to the memory kind a buffer is already in hands back the
    # same buffer, so moving it and then freeing the source would free the
    # result.
    tree = {'w': jnp.arange(4.0)}
    moved = colocate_async_rl_lib.move_tree(tree, 'device')
    self.assertFalse(tree['w'].is_deleted())
    np.testing.assert_array_equal(moved['w'], [0.0, 1.0, 2.0, 3.0])

  def test_prng_key_survives_the_roundtrip(self):
    # The move handles extended dtypes with no `key_data` special casing, so
    # this is what says a PRNG key can sit in pinned host memory.
    key = jax.random.key(0)
    expected_data = np.asarray(jax.random.key_data(key))
    expected_draw = np.asarray(jax.random.normal(key, (4,)))
    tree = {'key': key, 'w': jnp.arange(4.0)}
    host_tree = colocate_async_rl_lib.offload_to_host(tree)
    self.assertEqual(host_tree['key'].sharding.memory_kind, 'pinned_host')
    restored = colocate_async_rl_lib.onload_from_host(host_tree)
    self.assertEqual(restored['key'].sharding.memory_kind, 'device')
    np.testing.assert_array_equal(
        jax.random.key_data(restored['key']), expected_data
    )
    # The restored key is still usable as a key, and draws the same numbers.
    np.testing.assert_array_equal(
        jax.random.normal(restored['key'], (4,)), expected_draw
    )

  def test_reshard_converts_dtype(self):
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ('x',)),
        jax.sharding.PartitionSpec(),
    )
    tree = {'w': jax.device_put(jnp.arange(4.0, dtype=jnp.float32), sharding)}
    target = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(
            x.shape, jnp.bfloat16, sharding=x.sharding
        ),
        tree,
    )
    resharded = colocate_async_rl_lib.reshard(tree, target)
    self.assertEqual(resharded['w'].dtype, jnp.bfloat16)
    self.assertEqual(resharded['w'].sharding, sharding)
    np.testing.assert_array_equal(
        np.asarray(resharded['w'], dtype=np.float32), [0.0, 1.0, 2.0, 3.0]
    )


# Packing must not change the loss, so the loss must not depend on how the
# sequences are grouped into rows.
_LOSS_KWARGS = dict(
    kl_coeff=0.01,
    ppo_clip_eps_low=0.2,
    ppo_clip_eps_high=0.2,
    policy_ratio_cap=10.0,
    max_abs_advantage=10.0,
)
_MAX_SEQ_LEN = 15  # Rows hold `_MAX_SEQ_LEN + 1` tokens.
_EQUIVALENCE_SAMPLES = (
    dict(length=11, input_len=4, advantage=1.5),
    dict(length=7, input_len=2, advantage=-0.75),
    dict(length=5, input_len=1, advantage=0.25),
    dict(length=3, input_len=1, advantage=-2.0),
)


@functools.cache
def _tiny_model() -> tuple[Any, Any]:
  """Returns `(params, loss_and_grad_fn)` for a small CPU `TransformerLM`."""
  config = dataclasses.replace(
      config_lib.BaseExperimentConfig(),
      model_dim=16,
      per_head_dim=4,
      n_heads=2,
      n_layers=2,
      expand_factor=2,
      vocab_size=_VOCAB_SIZE,
      seq_len=_MAX_SEQ_LEN,
      use_scan=True,
      use_remat=False,
      use_flash_attention=False,
      activation_dtype_name='float32',
  )
  model = model_lib.TransformerLM(config)
  loss_and_grad_fn = jax.jit(
      jax.value_and_grad(
          functools.partial(
              colocate_async_rl_lib.compute_ppo_loss, model, **_LOSS_KWARGS
          ),
          has_aux=True,
      )
  )
  return model.init(jax.random.key(0)), loss_and_grad_fn


def _combine_micro_batches(
    loss_and_grad_fn: Any,
    params: Any,
    micro_batches: Sequence[colocate_async_rl_lib.TrainBatch],
) -> tuple[jax.Array, Any, list[float]]:
  """Averages micro-batches exactly the way `train_step` does."""
  accum_grad = jax.tree.map(jnp.zeros_like, params)
  weighted_loss = jnp.zeros((), dtype=jnp.float32)
  total_weight = jnp.zeros((), dtype=jnp.float32)
  weights = []
  for micro_batch in micro_batches:
    (loss, aux), grad = loss_and_grad_fn(params, micro_batch)
    weight = aux['loss_weight']

    def _accumulate(accum, g, weight=weight):
      return accum + g * weight

    accum_grad = jax.tree.map(_accumulate, accum_grad, grad)
    weighted_loss += loss * weight
    total_weight += weight
    weights.append(float(weight))
  return (
      weighted_loss / total_weight,
      jax.tree.map(lambda x: x / total_weight, accum_grad),
      weights,
  )


def _max_abs_diff(tree_a: Any, tree_b: Any) -> float:
  leaves = jax.tree.leaves(
      jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), tree_a, tree_b)
  )
  return max(float(leaf) for leaf in leaves)


class PackedLossEquivalenceTest(absltest.TestCase):
  """Packing rows must not move the loss or the gradient."""

  def setUp(self):
    super().setUp()
    self.params, self.loss_and_grad_fn = _tiny_model()
    self.samples = [
        _sample(seed=i, **kwargs)
        for i, kwargs in enumerate(_EQUIVALENCE_SAMPLES)
    ]
    self.sequences = [
        s
        for sample in self.samples
        if (s := sample.train_sequence()) is not None
    ]
    # One sequence per row: one `create_train_batches` call per sequence.
    self.unpacked = [
        colocate_async_rl_lib.create_train_batches(
            [sequence],
            rows_per_micro_batch=1,
            max_seq_len=_MAX_SEQ_LEN,
            is_primary=True,
        )[0]
        for sequence in self.sequences
    ]
    for micro_batch in self.unpacked:
      np.testing.assert_array_equal(micro_batch.segment_ids.max(), 1)
    self.loss, self.grad, weights = _combine_micro_batches(
        self.loss_and_grad_fn, self.params, self.unpacked
    )
    self.weight = sum(weights)

  def _assert_matches_unpacked(self, packed_loss, packed_grad):
    loss_diff = abs(float(packed_loss) - float(self.loss))
    grad_diff = _max_abs_diff(packed_grad, self.grad)
    detail = (
        f'unpacked_loss={float(self.loss):.8f} '
        f'packed_loss={float(packed_loss):.8f} '
        f'max|dloss|={loss_diff:.3e} max|dgrad|={grad_diff:.3e}'
    )
    np.testing.assert_allclose(
        packed_loss, self.loss, rtol=1e-4, atol=1e-5, err_msg=detail
    )
    unpacked_leaves = jax.tree.leaves(self.grad)
    packed_leaves = jax.tree_util.tree_flatten_with_path(packed_grad)[0]
    for (path, packed_leaf), unpacked_leaf in zip(
        packed_leaves, unpacked_leaves, strict=True
    ):
      np.testing.assert_allclose(
          packed_leaf,
          unpacked_leaf,
          rtol=1e-4,
          atol=1e-5,
          err_msg=f'grad mismatch at {jax.tree_util.keystr(path)}: {detail}',
      )

  def test_single_packed_micro_batch_matches_unpacked(self):
    micro_batches = colocate_async_rl_lib.create_train_batches(
        self.sequences,
        rows_per_micro_batch=2,
        max_seq_len=_MAX_SEQ_LEN,
        is_primary=True,
    )
    self.assertLen(micro_batches, 1)
    self.assertGreater(int(micro_batches[0].segment_ids.max()), 1)
    packed_loss, packed_grad, weights = _combine_micro_batches(
        self.loss_and_grad_fn, self.params, micro_batches
    )
    self.assertEqual(sum(weights), self.weight)
    self._assert_matches_unpacked(packed_loss, packed_grad)

  def test_several_packed_micro_batches_match_unpacked(self):
    micro_batches = colocate_async_rl_lib.create_train_batches(
        self.sequences,
        rows_per_micro_batch=1,
        max_seq_len=_MAX_SEQ_LEN,
        is_primary=True,
    )
    self.assertGreater(len(micro_batches), 1)
    packed_loss, packed_grad, weights = _combine_micro_batches(
        self.loss_and_grad_fn, self.params, micro_batches
    )
    # The micro-batches carry different token counts, so the weighting matters.
    self.assertNotEqual(min(weights), max(weights))
    self.assertEqual(sum(weights), self.weight)
    self._assert_matches_unpacked(packed_loss, packed_grad)


_LOGIT_VOCAB = 5
# The band is 0.2 both ways, which is [log 0.8, log 1.2] = [-0.223, +0.182] in
# log space.
_BAND_KWARGS = dict(
    ppo_clip_eps_low=0.2,
    ppo_clip_eps_high=0.2,
    # Off: it caps the ratio unconditionally, which would mask the clip's own
    # effect on the gradient.
    policy_ratio_cap=None,
)


class _LogitsModel:
  """Stand-in model whose params ARE the logits, so grads are wrt logits."""

  def apply(self, params, inputs, *, segment_ids=None, segment_positions=None):
    del inputs, segment_ids, segment_positions
    return params['logits'], None


def _logit_params_and_batch(
    logp_diffs: Sequence[float],
    advantages: Sequence[float],
    seed: int = 0,
) -> tuple[Any, colocate_async_rl_lib.TrainBatch]:
  """A one-row batch whose per-token `logpi - logpi_old` is `logp_diffs`.

  Args:
    logp_diffs: Target log-ratio for each trainable token.
    advantages: Advantage for each trainable token.
    seed: Seed for the logits and the token ids.

  Returns:
    `(params, batch)` for `_LogitsModel`; the sampler log-probs are set to
    `logpi - logp_diff`, so the loss sees exactly the requested log-ratios.
  """
  rng = np.random.default_rng(seed)
  n = len(logp_diffs)
  tokens = rng.integers(0, _LOGIT_VOCAB, size=n + 1).astype(np.int32)
  logits = rng.normal(size=(1, n, _LOGIT_VOCAB)).astype(np.float32)
  logpi = np.asarray(
      jax.nn.log_softmax(jnp.asarray(logits), axis=-1)[
          0, np.arange(n), tokens[1:]
      ]
  )

  def _row(tail: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=tail.dtype), tail])[None]

  batch_cls = colocate_async_rl_lib.TrainBatch
  return {'logits': jnp.asarray(logits)}, batch_cls(
      tokens=tokens[None],
      logprobs=_row(logpi - np.asarray(logp_diffs, dtype=np.float32)),
      mask=_row(np.ones(n, dtype=np.bool_)),
      advantages=_row(np.asarray(advantages, dtype=np.float32)),
      segment_ids=_row(np.ones(n, dtype=np.int32)),
      segment_positions=np.arange(n + 1, dtype=np.int32)[None],
      lens=np.array([n + 1], dtype=np.int32),
  )


def _logit_loss_and_grad(params, batch, **kwargs):
  """Returns `((loss, aux), d loss / d logits)`."""
  return jax.value_and_grad(
      functools.partial(colocate_async_rl_lib.compute_ppo_loss, _LogitsModel()),
      has_aux=True,
  )(params, batch, **{**_BAND_KWARGS, **kwargs})


class ValidationEvalSampleTest(absltest.TestCase):
  """Verdict-only samples: scored for stats, never trainable."""

  def test_verdict_only_samples_feed_stats_but_not_training(self):
    # Validation eval builds `Sample(correct=..., reward=...)` with no
    # sequence; those must score like any group and never reach a batch.
    groups = [
        [
            colocate_async_rl_lib.Sample(correct=True, reward=1.0),
            colocate_async_rl_lib.Sample(correct=False, reward=0.0),
        ],
        [
            colocate_async_rl_lib.Sample(correct=False, reward=0.0),
            colocate_async_rl_lib.Sample(correct=False, reward=0.0),
        ],
    ]
    samples = list(itertools.chain.from_iterable(groups))
    for sample in samples:
      self.assertIsNone(sample.sequence)
      self.assertIsNone(sample.train_sequence())
      # `with_advantage` is a no-op rather than a crash.
      self.assertIs(sample.with_advantage(1.0), sample)

    stats = colocate_async_rl_lib.compute_group_stats(groups)
    self.assertAlmostEqual(stats['pass@1'], 0.25)
    self.assertAlmostEqual(stats['pass@k'], 0.5)

    training_sequences = [
        s for sample in samples if (s := sample.train_sequence()) is not None
    ]
    self.assertEmpty(
        colocate_async_rl_lib.create_train_batches(
            training_sequences,
            rows_per_micro_batch=2,
            max_seq_len=15,
            is_primary=True,
        )
    )


class ReshardTreeTest(absltest.TestCase):
  """Every leaf of the tree lands on its target dtype and sharding."""

  def _tree_and_target(self, dtype=jnp.bfloat16):
    # A real NamedSharding: `convert_array_with_abstract`, which `reshard`
    # maps over the tree, accepts nothing else.
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ('data',))
    named = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    tree = jax.tree.map(
        lambda x: jax.device_put(x, named),
        {'a': jnp.arange(4.0), 'b': jnp.ones((2, 2))},
    )
    target = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, dtype, sharding=named), tree
    )
    return tree, target

  def test_the_whole_tree_is_converted(self):
    tree, target = self._tree_and_target()
    resharded = colocate_async_rl_lib.reshard(tree, target)
    jax.tree.map(
        lambda got, want: self.assertEqual(
            (got.dtype, got.sharding), (want.dtype, want.sharding)
        ),
        resharded,
        target,
    )
    jax.tree.map(
        lambda got, src: np.testing.assert_array_equal(
            np.asarray(got, np.float32), np.asarray(src, np.float32)
        ),
        resharded,
        tree,
    )


if __name__ == '__main__':
  absltest.main()

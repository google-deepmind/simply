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
"""Colocate Async RL training loop with ragged paged attention.

This module implements an RL training loop that fuses continuous sampling
(using ragged paged attention via the `Batcher` from `page_batcher.py`)
with training, connected by host offloading.

Architecture
------------
*  A **sampling thread** continuously pushes prompts into the ragged paged
   attention batch, runs `continue_decode` (with `intermediate_steps`),
   harvests completed sequences, evaluates rewards, and pushes
   `Sample` groups into a thread-safe `completed_queue`.
*  The **main thread** pops from `completed_queue`, checks the collection
   condition (`train_batch_size` total valid samples), and when met:
   1. Pauses the sampling thread.
   2. Offloads the sampling state to host memory.
   3. Runs a training step (PPO/GRPO).
   4. Converts decoding params from training params.
   5. Onloads the sampling state.
   6. Resumes the sampling thread.

Loop steps
----------
1. **Sample** — Sampling thread pushes prompts and decodes.
2. **Check collection** — Main thread checks completed_queue.
3. **Train** — Main thread runs the packed micro-batch step.
4. **Offload training states** — To host CPU.
5. **Convert params** — Training → decoding format.
6. **Resume sampling** — Onload sampling state, signal sampling thread.
"""

import asyncio
from collections.abc import Mapping, Sequence
from concurrent import futures
import dataclasses
import functools
import queue
import threading
import time
from typing import Any, Literal, TypeAlias

from absl import logging
import grpc
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
from simply import data_lib
from simply import model_lib
from simply.serving import page_batcher
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import distributions
from simply.utils import experiment_helper as exp_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import masked
from simply.utils import pytree
from simply.utils import ragged_paged_attention as rpa
from simply.utils import sharding as sharding_lib

_PREFILL_CHUNK_TOKENS = 4096

PyTree: TypeAlias = Any
ExperimentHelper = exp_helper.ExperimentHelper
TrainLoopRegistry = model_lib.TrainLoopRegistry


# =============================================================================
# Host offloading/onloading utilities
# =============================================================================


def move_tree(
    tree: PyTree, memory_kind: Literal['device', 'pinned_host']
) -> PyTree:
  """Moves every leaf of `tree` to `memory_kind`, one transfer per buffer."""
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  moved: list[Any] = []
  for leaf in leaves:
    # A leaf already there is left alone: `device_put` would hand back the
    # same buffer, and the delete below would then free the "destination".
    if (
        not isinstance(leaf, jax.Array)
        or leaf.sharding.memory_kind == memory_kind
    ):
      moved.append(leaf)
      continue
    moved.append(
        jax.device_put(leaf, leaf.sharding.with_memory_kind(memory_kind))
    )
    # Dropped HERE, not after the caller blocks: `PjRtBuffer::Delete` drops
    # this handle and frees the memory lazily once every async operation using
    # the buffer has completed, and never blocks for one (pjrt_client.h), so
    # the transfer just issued still reads it. Deferring these deletes to the
    # end of the tree measured 2.5x slower on the way in.
    leaf.delete()
  return jax.tree_util.tree_unflatten(treedef, moved)


def offload_to_host(tree: PyTree) -> PyTree:
  return move_tree(tree, 'pinned_host')


def onload_from_host(tree: PyTree) -> PyTree:
  return move_tree(tree, 'device')


def reshard(tree: PyTree, abstract_target: PyTree) -> PyTree:
  """Casts and reshards a pytree onto its target shardings, leaf by leaf.

  Delegates to `common.convert_array_with_abstract`, as `rl_lib` does: it casts
  under a sharding constraint when the leaf already lives on the target mesh,
  and otherwise casts under the SOURCE mesh and `device_put`s across. No single
  `jax.jit` ever spans both meshes, so the two are free to enumerate their
  devices in different orders -- `mesh_utils.create_device_mesh` orders them
  per mesh SHAPE, so a training and a decoding mesh generally do.

  Args:
    tree: Source pytree whose leaves are to be resharded. RAW arrays: run
      `common.get_raw_arrays` first if they may be `AnnotatedArray`-wrapped.
    abstract_target: PyTree of `jax.ShapeDtypeStruct`s matching the source
      pytree structure, carrying the destination shardings and dtypes.

  Returns:
    A new PyTree with the same structure whose leaves are `jax.Array`s on
    the destination sharding.
  """
  return jax.tree.map(common.convert_array_with_abstract, tree, abstract_target)


@dataclasses.dataclass(frozen=True)
class TrainSequence:
  """One contiguous served sequence and what training should do with it.

  `mask` is carried explicitly rather than inferred from `advantages != 0`,
  because a zero advantage is a legitimate trained value -- a degenerate GRPO
  group (every rollout scored the same) baselines to exactly 0 on every token,
  and those tokens must still count in `loss_weight`.

  Attributes:
    tokens: Token ids, `[n]`.
    logprobs: Sampler log-probs, `[n]`; entry i scores `tokens[i]` (i >= 1).
    mask: True where the token is trainable (model output, not prompt), `[n]`.
    advantages: Per-token advantage, `[n]`; 0 off the trainable span.
  """

  tokens: np.ndarray
  logprobs: np.ndarray
  mask: np.ndarray
  advantages: np.ndarray | None = None

  def __len__(self) -> int:
    return len(self.tokens)

  @classmethod
  def from_completion(
      cls,
      *,
      tokens: np.ndarray,
      logprobs: np.ndarray,
      input_len: int,
      advantage: float | None = None,
  ) -> 'TrainSequence':
    """Builds a sequence whose trainable span is the tail after `input_len`.

    Args:
      tokens: Token ids `[n]`.
      logprobs: Sampler log-probs `[n]`, index-aligned with `tokens` (entry 0 is
        a dummy for the un-predicted BOS).
      input_len: Number of prompt tokens; everything after is the output.
      advantage: Scalar advantage to broadcast over the output span.

    Returns:
      The `TrainSequence`.
    """
    mask = np.zeros(len(tokens), dtype=np.bool)
    mask[input_len:] = True
    kwargs = dict(
        tokens=np.asarray(tokens, dtype=np.int32),
        logprobs=np.asarray(logprobs, dtype=np.float32),
        mask=mask,
    )
    if advantage is not None:
      kwargs['advantages'] = np.where(mask, advantage, 0.0).astype(np.float32)
    return cls(**kwargs)

  def with_advantage(self, advantage: float) -> 'TrainSequence':
    """Returns a copy whose trainable span carries `advantage`."""
    return dataclasses.replace(
        self, advantages=np.where(self.mask, advantage, 0.0).astype(np.float32)
    )


@dataclasses.dataclass(frozen=True)
class Sample:
  """One served completion and what the evaluator decided about it.

  Attributes:
    sequence: The served sequence; its trainable span is the tail after the
      prompt (see `TrainSequence.from_completion`). `None` for samples that only
      carry a verdict, e.g. validation rollouts.
    reward: Scalar reward assigned by the evaluator (`None` until eval).
    correct: Whether the sample was judged correct by the evaluator.
    is_valid_for_training: Whether the sample is eligible for training (set
      during collection from the reward / truncation checks).
    prompt_index: Index of the prompt within the current RL step.
    step: Training step at which this sample was collected.
    start_step: Training step at which the rollout was submitted. A rollout that
      spans a param update has `step > start_step`, which means its tokens were
      sampled by more than one policy version while its `logprobs` record
      whatever produced each token -- so its `logp_diff` mixes versions.
    truncated: Whether the rollout stopped without sampling an EOS.
    output_text: Decoded output text (diagnostics / history only).
    output_messages: Decoded output as a sequence of chunk mappings.
  """

  sequence: TrainSequence | None = None
  reward: float | None = None
  correct: bool | None = None
  is_valid_for_training: bool = True
  prompt_index: int = 0
  step: int = 0
  start_step: int = 0
  truncated: bool = False
  output_text: str = ''
  output_messages: Sequence[Mapping[str, Any]] = ()

  def update_with_evaluation_result(self, eval_result) -> 'Sample':
    return dataclasses.replace(
        self,
        correct=eval_result['correct'],
        reward=eval_result['reward'],
    )

  def with_advantage(self, advantage: float) -> 'Sample':
    """Returns a copy whose trainable span carries `advantage`."""
    if self.sequence is None:
      return self
    return dataclasses.replace(
        self, sequence=self.sequence.with_advantage(advantage)
    )

  def train_sequence(self) -> TrainSequence | None:
    """The sequence to emit, or None when there is nothing to train on."""
    if (
        self.sequence is None
        or not self.is_valid_for_training
        or self.sequence.advantages is None
    ):
      return None
    if not np.any(np.where(self.sequence.mask, self.sequence.advantages, 0.0)):
      # TODO: This has impact on the normalized total loss (effectively
      # inflates the gradients). Should better study its effect.
      return None
    return self.sequence


def _seq_to_sample(
    seq: Mapping[str, Any],
    prompt_index: int,
    step: int = 0,
    start_step: int = 0,
) -> Sample:
  """Converts a completed sequence dict from the batcher to a `Sample`.

  Args:
    seq: Dict from the batcher with keys `tokens`, `input_len`, `logprobs`,
      `truncated`, `output_text` and optionally `output_messages`. `logprobs`
      is the log-likelihood of the token that was actually emitted, i.e. the
      `logpi_old` the PPO ratio is measured against.
    prompt_index: Index of the prompt within the current RL step.
    step: Current training step.
    start_step: Training step at which the rollout was submitted.

  Returns:
    A `Sample` whose advantage is still 0: the reward is only known after
    evaluation, and `normalize_group_rewards` stamps the group-baselined
    value.
  """
  return Sample(
      sequence=TrainSequence.from_completion(
          tokens=np.asarray(seq['tokens'], dtype=np.int32),
          logprobs=np.asarray(seq['logprobs'], dtype=np.float32),
          input_len=int(seq['input_len']),
      ),
      prompt_index=prompt_index,
      step=step,
      start_step=start_step,
      truncated=bool(seq['truncated']),
      output_text=seq['output_text'],
      output_messages=seq.get('output_messages', ()),
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(slots=True)
class TrainBatch:
  """Minimal GRPO training batch of PACKED rows, assembled host-side.

  Attributes:
    tokens: `[B, L]` token ids.
    logprobs: `[B, L]` sampler log-probs, index-aligned with `tokens`.
    mask: `[B, L]` trainable-token mask.
    advantages: `[B, L]` per-token advantage.
    segment_ids: `[B, L]` 1-based segment index within the row, 0 on padding.
    segment_positions: `[B, L]` position within the segment.
    lens: `[B]` fill level of each row.
  """

  tokens: common.Array
  logprobs: common.Array
  mask: common.Array
  advantages: common.Array
  segment_ids: common.Array
  segment_positions: common.Array
  lens: common.Array

  @classmethod
  def create(
      cls, batch_size: int, max_seq_len: int
  ) -> 'TrainBatch':
    """Returns an EMPTY batch of `batch_size` rows of `max_seq_len + 1`."""
    l = max_seq_len + 1
    return cls(
        tokens=np.zeros((batch_size, l), dtype=np.int32),
        logprobs=np.zeros((batch_size, l), dtype=np.float32),
        mask=np.zeros((batch_size, l), dtype=np.bool_),
        advantages=np.zeros((batch_size, l), dtype=np.float32),
        segment_ids=np.zeros((batch_size, l), dtype=np.int32),
        segment_positions=np.zeros((batch_size, l), dtype=np.int32),
        lens=np.zeros((batch_size,), dtype=np.int32),
    )

  @property
  def max_seq_len(self) -> int:
    """The maximum sequence length of the batch."""
    return self.tokens.shape[1]

  def pack(self, sequence: TrainSequence) -> bool:
    """Appends one sequence to the first row with room, IN PLACE.

    Args:
      sequence: The sequence to pack; `len(sequence) <= L`.

    Returns:
      True if the sequence was written, False if no row had room left.

    Raises:
      ValueError: If the sequence cannot fit an empty row -- that is a
        misconfiguration (`train_max_seq_len` too small), not a data point --
        or if its first token is trainable.
    """
    n = len(sequence)
    if n > self.max_seq_len:
      raise ValueError(
          f'sequence of {n} tokens exceeds the row length {self.max_seq_len}.'
      )
    if sequence.mask[0]:
      raise ValueError('the first token of a sequence cannot be trainable.')
    rows = np.flatnonzero(self.lens + n <= self.max_seq_len)
    if rows.size == 0:
      return False
    i = int(rows[0])
    start = int(self.lens[i])
    end = start + n
    self.tokens[i, start:end] = sequence.tokens
    self.logprobs[i, start:end] = sequence.logprobs
    self.mask[i, start:end] = sequence.mask
    self.advantages[i, start:end] = sequence.advantages
    self.segment_ids[i, start:end] = (
        self.segment_ids[i, start - 1] + 1 if start else 1
    )
    self.segment_positions[i, start:end] = np.arange(n, dtype=np.int32)
    self.lens[i] = end
    return True


def compute_ppo_loss(
    model,
    params: PyTree,
    batch: TrainBatch,
    kl_coeff: float = 0.0,
    ppo_clip_eps_high: float = 0.2,
    ppo_clip_eps_low: float = 0.2,
    policy_ratio_cap: float | None = 10.0,
    use_kl_correction: bool = False,
    max_abs_advantage: float | None = None,
    use_policy_logp_as_sampler_logp: bool = False,
) -> tuple[float, dict[str, Any]]:
  """Compute PPO loss."""
  inputs = batch.tokens[:, :-1]
  targets = batch.tokens[:, 1:]
  answer_mask = batch.mask[:, 1:]
  logprobs = batch.logprobs[:, 1:]
  advantages = batch.advantages[:, 1:]

  logits, _ = model.apply(
      params,
      inputs,
      segment_ids=batch.segment_ids[:, :-1],
      segment_positions=batch.segment_positions[:, :-1],
  )
  logits = jnp.astype(logits, jnp.float32)
  m = distributions.Categorical(logits)

  logpi = masked.masked(m.log_prob(targets), mask=answer_mask)
  logpi_old = masked.masked(logprobs, mask=answer_mask)
  logpi_ref = logpi_old

  # K3 estimator from http://joschu.net/blog/kl-approx.html.
  logr = masked.masked(logpi_ref - logpi, mask=answer_mask)
  kl = masked.masked(jnp.expm1(logr) - logr, mask=answer_mask)

  adv = jnp.astype(advantages, jnp.float32)
  if max_abs_advantage is not None:
    adv = jnp.clip(adv, -max_abs_advantage, max_abs_advantage)
  adv = jax.lax.stop_gradient(masked.masked(adv, mask=answer_mask))

  if use_policy_logp_as_sampler_logp:
    logpi_old = jax.lax.stop_gradient(logpi)

  logp_diff = masked.masked(logpi - logpi_old, mask=answer_mask)
  abs_logp_diff = jnp.abs(logp_diff)

  ratio = masked.masked(jnp.exp(logp_diff), mask=answer_mask)
  if policy_ratio_cap is not None:
    assert policy_ratio_cap > 1.0 + ppo_clip_eps_high
    ratio = jnp.minimum(ratio, policy_ratio_cap)
  clipped_ratio = masked.masked(
      jnp.clip(ratio, 1.0 - ppo_clip_eps_low, 1.0 + ppo_clip_eps_high),
      mask=answer_mask,
  )

  surr1 = masked.masked(ratio * adv, mask=answer_mask)
  surr2 = masked.masked(clipped_ratio * adv, mask=answer_mask)
  per_token_ppo_loss = masked.masked(
      -jnp.minimum(surr1, surr2), mask=answer_mask
  )

  loss = masked.masked_mean(per_token_ppo_loss, mask=answer_mask)
  if kl_coeff > 0:
    kl_loss = masked.masked_mean(kl, mask=answer_mask)
    if use_kl_correction:
      kl_loss = masked.masked_mean(
          jax.lax.stop_gradient(ratio) * kl, mask=answer_mask
      )
    loss += kl_coeff * kl_loss

  loss = sharding_lib.with_sharding_constraint(loss, None)

  entropy = jax.lax.stop_gradient(
      masked.masked_mean(m.entropy(), mask=answer_mask)
  )
  entropy = sharding_lib.with_sharding_constraint(entropy, None)

  kl_divergence = jax.lax.stop_gradient(
      masked.masked_mean(kl, mask=answer_mask)
  )
  kl_divergence = sharding_lib.with_sharding_constraint(kl_divergence, None)

  policy_ratio = jax.lax.stop_gradient(
      masked.masked_mean(ratio, mask=answer_mask)
  )
  policy_ratio = sharding_lib.with_sharding_constraint(policy_ratio, None)
  policy_ratio_max = sharding_lib.with_sharding_constraint(
      jax.lax.stop_gradient(masked.masked_max(ratio, mask=answer_mask)), None
  )
  policy_ratio_min = sharding_lib.with_sharding_constraint(
      jax.lax.stop_gradient(masked.masked_min(ratio, mask=answer_mask)), None
  )

  return loss, {
      'entropy': entropy,
      'kl_divergence': kl_divergence,
      'policy_ratio/mean': policy_ratio,
      'policy_ratio/max': policy_ratio_max,
      'policy_ratio/min': policy_ratio_min,
      'loss_weight': jnp.sum(answer_mask),
      'logp_diff_abs/mean': masked.masked_mean(abs_logp_diff, mask=answer_mask),
      'logp_diff_abs/max': masked.masked_max(abs_logp_diff, mask=answer_mask),
  }


def create_train_batches(
    training_sequences: Sequence[TrainSequence],
    rows_per_micro_batch: int,
    max_seq_len: int,
    is_primary: bool,
) -> list[TrainBatch]:
  """Packs training sequences into micro-batches and broadcasts them from task 0.

  Only the primary task holds the training sequences, so it decides how many
  micro-batches there are (that count follows how densely the data packs),
  broadcasts it, and then broadcasts each micro-batch. Their shape is fixed,
  so the training step compiles once however many there are.

  Args:
    training_sequences: Training sequences to train on.
    rows_per_micro_batch: Rows in each micro-batch.
    max_seq_len: Row length (a row holds `max_seq_len + 1` tokens).
    is_primary: Whether this process is the primary task.

  Returns:
    The micro-batches, replicated on all hosts.
  """
  micro_batches: list[TrainBatch] = []
  if is_primary:
    # Longest first (it packs the rows denser), filling the current
    # micro-batch until it cannot take the next sequence, then opening a new
    # one -- so how many there are follows the data.
    for sequence in sorted(training_sequences, key=len, reverse=True):
      if not micro_batches or not micro_batches[-1].pack(sequence):
        micro_batches.append(
            TrainBatch.create(rows_per_micro_batch, max_seq_len)
        )
        micro_batches[-1].pack(sequence)
  count = int(
      jax.experimental.multihost_utils.broadcast_one_to_all(
          np.int32(len(micro_batches)), is_source=is_primary
      )
  )
  out = []
  for i in range(count):
    micro_batch = (
        micro_batches[i]
        if is_primary
        else TrainBatch.create(rows_per_micro_batch, max_seq_len)
    )
    out.append(
        jax.experimental.multihost_utils.broadcast_one_to_all(
            micro_batch, is_source=is_primary
        )
    )
  return out


def _trainable(sample: Sample) -> bool:
  """Whether a sample carries a usable reward and may be trained on."""
  return (
      sample.reward is not None
      and not np.isnan(sample.reward)
      and sample.is_valid_for_training
  )


def normalize_group_rewards(
    groups: Sequence[Sequence[Sample]],
    normalize_reward_method: str,
) -> list[list[Sample]]:
  """Group-baselines each group's rewards, host-side, in plain numpy.

  For each group, each trainable sample's advantage is computed as
  `(r - mean) / max(std, 1e-5)` over the valid rewards of its own group.
  `Sample.reward` retains the unbaselined evaluator score.

  Args:
    groups: One group per prompt.
    normalize_reward_method: Only `'ByGroup'` is implemented; the empty string
      means no normalization.

  Returns:
    New groups whose valid samples carry the baselined advantage.

  Raises:
    ValueError: For any method other than `''` or `'ByGroup'`.
  """
  if not normalize_reward_method:
    return [list(g) for g in groups]
  if normalize_reward_method != 'ByGroup':
    raise ValueError(
        'normalize_group_rewards implements the per-prompt group baseline'
        f' only; got normalize_reward_method={normalize_reward_method!r}.'
    )

  new_groups: list[list[Sample]] = []
  for group in groups:
    rewards = np.asarray(
        [
            float(s.reward)
            for s in group
            if s.reward is not None and _trainable(s)
        ],
        dtype=np.float32,
    )
    if rewards.size == 0:
      new_groups.append(list(group))
      continue
    baseline = float(rewards.mean())
    # Population std, matching the upstream masked `np_safe_std`. A degenerate
    # group (every rollout scored the same) has std 0 and numerator 0, so the
    # floor only keeps the division finite: the advantage is exactly zero,
    # which is what GRPO should say about a group with no contrast.
    scale = max(float(rewards.std()), 1e-5)
    new_group = []
    for sample in group:
      if not _trainable(sample) or sample.reward is None:
        new_group.append(sample)
        continue
      advantage = (float(sample.reward) - baseline) / scale
      new_group.append(sample.with_advantage(advantage))
    new_groups.append(new_group)
  return new_groups


def compute_group_stats(
    groups: Sequence[Sequence[Sample]],
) -> Mapping[str, Any]:
  """Computes pass@1, pass@k, length, and truncation stats from Sample groups.

  Since only task 0 collects samples, no cross-host aggregation is needed.

  Args:
    groups: One group per prompt.

  Returns:
    Mapping containing `pass@1`, `pass@k`, `truncated`,
    `<input|seq|response>_len/<mean|max|min>`, and
    `response_len/mean/<correct|incorrect>`, or an empty dict if nothing was
    scored.
  """
  corrects = []
  pass_at_k = []
  truncated = []
  input_lens = []
  seq_lens = []
  # Response length by verdict: under a token-mean loss a rollout's weight IS
  # its length, so "wrong rollouts are longer" is the same statement as "the
  # update's negative mass dominates" (see `advantage_mass_stats`).
  response_by_verdict: dict[bool, list[int]] = {True: [], False: []}
  for group in groups:
    for s in group:
      if s.sequence is None:
        continue
      truncated.append(bool(s.truncated))
      # The trainable span starts after the prompt, so the first True in the
      # mask is the prompt length.
      mask = s.sequence.mask
      input_lens.append(int(np.argmax(mask)) if mask.any() else len(mask))
      seq_lens.append(len(s.sequence))
      if s.correct is not None:
        response_by_verdict[bool(s.correct)].append(
            seq_lens[-1] - input_lens[-1]
        )
    group_corrects = [
        s.correct
        for s in group
        if s.correct is not None
        and s.reward is not None
        and not np.isnan(s.reward)
    ]
    if group_corrects:
      corrects.extend(group_corrects)
      pass_at_k.append(any(group_corrects))

  if not corrects:
    return {}

  stats = {
      'pass@1': float(np.mean(corrects)),
      'pass@k': float(np.mean(pass_at_k)),
  }
  if seq_lens:
    inp = np.asarray(input_lens)
    seq = np.asarray(seq_lens)
    resp = seq - inp
    stats['truncated'] = float(np.mean(truncated))
    for name, v in (
        ('input_len', inp),
        ('seq_len', seq),
        ('response_len', resp),
    ):
      stats[f'{name}/mean'] = float(v.mean())
      stats[f'{name}/max'] = float(v.max())
      stats[f'{name}/min'] = float(v.min())
    for verdict, lengths in response_by_verdict.items():
      if lengths:
        label = 'correct' if verdict else 'incorrect'
        stats[f'response_len/mean/{label}'] = float(np.mean(lengths))
  return stats


def _wait_until(
    predicate: Any,
    description: str,
    *,
    threads: Sequence[threading.Thread],
    error_queue: queue.Queue[Exception],
    timeout: float = 1800.0,
    poll_interval: float = 0.01,
) -> None:
  """Blocks until `predicate()` holds, failing fast if a worker died.

  Args:
    predicate: Called repeatedly; the wait ends when it returns True.
    description: What is being waited for, used in error messages.
    threads: Threads whose death makes the wait unsatisfiable.
    error_queue: Queue the worker threads report their exceptions on; a queued
      exception is re-raised here.
    timeout: Seconds to wait before giving up.
    poll_interval: Seconds between polls.

  Raises:
    RuntimeError: If one of `threads` died, or `timeout` elapsed.
  """
  deadline = time.time() + timeout
  while not predicate():
    if not error_queue.empty():
      raise error_queue.get()
    for thread in threads:
      if not thread.is_alive():
        raise RuntimeError(
            f'{thread.name} died while waiting for {description}.'
        )
    if time.time() > deadline:
      raise RuntimeError(
          f'Timed out after {timeout}s waiting for {description}.'
      )
    time.sleep(poll_interval)


# =============================================================================
# Main colocate async RL training loop
# =============================================================================


_COLLECTION_TIMEOUT_SECONDS = 180.0  # cap on one collection's wall clock


@functools.partial(TrainLoopRegistry.register, name='colocate_async_rl')
def rl_experiment(config, experiment_dir=''):
  """Colocate Async RL training loop with ragged paged attention and host offloading.

  Uses `page_batcher.Batcher` for sampling on a dedicated thread, and the
  main thread for training with PPO/GRPO.

  Args:
    config: The experiment config (RLExperimentConfig or subclass).
    experiment_dir: The directory for saving experiment data.

  Returns:
    A dict of final results.

  Raises:
    RuntimeError: If the sampling thread or sample-and-evaluate thread dies
      unexpectedly while the main loop is collecting groups.
  """
  logging.info('jax.process_index(): %s', jax.process_index())

  # =========================================================================
  # Setup: training model, optimizer, state, mesh
  # =========================================================================
  sharding_lib.set_mesh(
      mesh_shape=config.mesh_shape,
      dcn_mesh_shape=config.dcn_mesh_shape,
      axis_names=config.sharding_config.mesh_axis_names,
  )
  helper = ExperimentHelper(
      experiment_dir,
      ckpt_interval=config.ckpt_interval,
      ckpt_max_to_keep=config.ckpt_max_to_keep,
      ckpt_keep_period=config.ckpt_keep_period,
      num_train_steps=config.num_train_steps,
      metric_log_interval=config.tb_log_interval,
      log_additional_info=config.log_additional_info,
      should_save_ckpt=config.should_save_ckpt,
      is_primary=exp_helper.is_primary_task(),
  )
  model, _ = model_lib.create_model(config, config.sharding_config)
  helper.save_config_info(config, config.sharding_config, model)
  opt = config.optimizer

  # =========================================================================
  # Compile training functions, before the state is initialized: lowering
  # against the ABSTRACT state `get_init_state` is about to produce keeps HBM
  # empty while XLA compiles.
  # =========================================================================
  t1 = time.time()
  abstract_state: PyTree = common.eval_abstract_output(
      model_lib.get_init_state_fn(config, model)
  )

  # How many micro-batches a step runs follows how densely the data packs;
  # only their shape is fixed.
  row_len = config.train_max_seq_len + 1
  grad_accum_steps = max(1, config.grad_accum_steps)
  if config.train_batch_size % grad_accum_steps:
    raise ValueError(
        f'train_batch_size={config.train_batch_size} is not divisible by '
        f'grad_accum_steps={grad_accum_steps}.'
    )
  rows_per_micro_batch = config.train_batch_size // grad_accum_steps
  logging.info(
      'micro-batch: %d rows x %d tokens', rows_per_micro_batch, row_len
  )
  ppo_loss_fn = functools.partial(
      compute_ppo_loss,
      kl_coeff=config.kl_coeff,
      ppo_clip_eps_high=config.ppo_clip_eps_high or config.ppo_clip_eps,
      ppo_clip_eps_low=config.ppo_clip_eps_low or config.ppo_clip_eps,
      policy_ratio_cap=config.policy_ratio_cap,
      max_abs_advantage=config.max_abs_advantage,
      use_policy_logp_as_sampler_logp=config.use_policy_logp_as_sampler_logp,
  )

  accum_shardings = jax.tree.map(lambda x: x.sharding, abstract_state['params'])

  @functools.partial(jax.jit, out_shardings=accum_shardings)
  def zero_grad_fn(params):
    return jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)

  # One micro-batch: its gradient is folded into `accum_grad` (donated, so the
  # accumulator is updated in place) inside the same program, which keeps only
  # one gradient tree beyond the accumulator alive.
  @functools.partial(
      jax.jit,
      donate_argnames=['accum_grad'],
      out_shardings=(accum_shardings, None, None),
  )
  def micro_grad_step_fn(params, batch, accum_grad):
    loss, extra_output, grad = model_lib.compute_grads(
        params=params,
        batch=batch,
        model=model,
        custom_loss_fn=ppo_loss_fn,
    )
    weight = extra_output['loss_weight']
    accum_grad = jax.tree.map(lambda a, g: a + g * weight, accum_grad, grad)
    return accum_grad, loss * weight, extra_output

  @functools.partial(jax.jit, donate_argnames=['state', 'accum_grad'])
  def apply_grads_fn(state, accum_grad, weighted_loss, total_weight, lr):
    return model_lib.apply_grads(
        state=state,
        grad=jax.tree.map(lambda x: x / total_weight, accum_grad),
        loss=weighted_loss / total_weight,
        model=model,
        opt=opt,
        lr=lr,
        clip_grad_norm=config.clip_grad_norm,
        clip_update_norm=config.clip_update_norm,
        clip_local_update_rms=config.clip_local_update_rms,
        weight_decay=config.weight_decay,
    )

  lr_fn = common.named_jit(model_lib.create_lr_schedule(config), 'lr_fn')

  # AOT pre-compile
  helper.set_notes('Pre-compiling train functions ...')
  _tc = time.time()
  _accum_grad_avals = jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32, sharding=x.sharding),
      abstract_state['params'],
  )
  compiled_zero_grad_fn = zero_grad_fn.lower(abstract_state['params']).compile()
  compiled_micro_grad_step_fn = micro_grad_step_fn.lower(
      abstract_state['params'],
      TrainBatch.create(rows_per_micro_batch, config.train_max_seq_len),
      _accum_grad_avals,
  ).compile()
  # The weighted loss and the total weight the loop hands over are f32
  # scalars: it accumulates them from f32 zeros.
  _scalar = jax.ShapeDtypeStruct((), jnp.float32)
  compiled_apply_grads_fn = apply_grads_fn.lower(
      abstract_state,
      _accum_grad_avals,
      _scalar,  # weighted_loss
      _scalar,  # total_weight
      _scalar,  # lr
  ).compile()
  logging.info(
      'AOT pre-compiled the train step in %.1fs (%d rows x %d tokens).',
      time.time() - _tc,
      rows_per_micro_batch,
      row_len,
  )
  del _accum_grad_avals, _scalar

  def train_step(state, micro_batches, lr):
    """Runs one update over pre-packed micro-batches.

    Compiled once and called N times: the micro-batch count follows the data,
    their shape does not.

    Args:
      state: Training state, donated to the update.
      micro_batches: The packed micro-batches of this update.
      lr: Learning rate for this step.

    Returns:
      `(loss, new_state, log_dict)`, or `(None, state, {})` when there is
      nothing to train on. Every log value is token-weighted over the
      micro-batches, except the `_max`/`_min` ones, which reduce instead.
    """
    if not micro_batches:
      logging.warning('No trainable token in this batch; skipping the update.')
      return None, state, {}

    accum_grad = compiled_zero_grad_fn(state['params'])
    weighted_loss = jnp.zeros((), dtype=jnp.float32)
    total_weight = jnp.zeros((), dtype=jnp.float32)
    aux: dict[str, Any] = {}
    for micro_batch in micro_batches:
      accum_grad, micro_loss, extra_output = compiled_micro_grad_step_fn(
          state['params'], micro_batch, accum_grad
      )
      weight = extra_output.pop('loss_weight')
      weighted_loss += micro_loss
      total_weight += weight
      for k, v in extra_output.items():
        if k not in aux:
          aux[k] = v if k.endswith(('_max', '_min')) else v * weight
        elif k.endswith('_max'):
          aux[k] = jnp.maximum(aux[k], v)
        elif k.endswith('_min'):
          aux[k] = jnp.minimum(aux[k], v)
        else:
          aux[k] += v * weight

    loss, state, log_dict = compiled_apply_grads_fn(
        state, accum_grad, weighted_loss, jnp.maximum(total_weight, 1e-6), lr
    )
    log_dict.update({
        k: v if k.endswith(('_max', '_min')) else v / total_weight
        for k, v in aux.items()
    })
    log_dict['loss_weight'] = total_weight
    return loss, state, log_dict

  dt = time.time() - t1
  logging.info('%s secs used for compiling train, loss and lr functions.', dt)

  # =========================================================================
  # Prepare decoding config and page Batcher
  # =========================================================================
  decoding_sharding_config = (
      config.decoding_sharding_config or config.sharding_config
  )
  decoding_config = dataclasses.replace(
      config,
      use_scan=False,
      use_remat=False,
      mesh_shape=config.decoding_mesh_shape or config.mesh_shape,
      sharding_config=decoding_sharding_config,
  )

  page_size = config.page_size or 128
  max_seq_len = config.train_max_seq_len + 1
  batch_size = config.batch_size

  lm_format = lm_format_lib.LMFormatRegistry.get(config.lm_format_name)()
  # The batcher derives its stop tokens from the format alone, so fold in the
  # config-level ones (`rl_lib.run_experiment` does the same for its sampler).
  lm_format = dataclasses.replace(
      lm_format,
      extra_eos_tokens=tuple(
          set(config.extra_eos_tokens) | set(lm_format.extra_eos_tokens)
      ),
  )
  evaluation = config.evaluation
  # One worker per decode slot: a pass can complete at most `batch_size`
  # sequences, and more threads than that only add scheduler overhead.
  evaluation_executor = futures.ThreadPoolExecutor(max_workers=batch_size)

  batcher = page_batcher.Batcher(
      config=decoding_config,
      lm_format=lm_format,
      max_seq_len=max_seq_len,
      temperature=config.sampling_temperature,
      max_decode_steps=config.sampling_max_decode_steps,
      intermediate_steps=config.sampling_intermediate_decode_steps,
      # Chunked-prefill budget: how many NEW tokens one prefill pass issues.
      max_num_issue_tokens=_PREFILL_CHUNK_TOKENS,
      max_queue_size=batch_size,
      max_queue_timeout=None,
  )

  # The page pool MUST be sized under the DECODE mesh: `get_partition_size`
  # reads the AMBIENT mesh, not the sharding config, and the ambient mesh here
  # is the TRAINING one. Sized under training, `num_seq_shards` is wrong and
  # `local_total_num_pages` can come out at half of what the decode program
  # needs. Constructing the batcher first is what makes the decode mesh
  # available: it is a frozen dataclass whose expensive members are all
  # `cached_property`, so nothing is realized until the first decode.
  with batcher.set_mesh():
    # TODO: Offload some of the following logics to page batcher.
    seq_partition = sharding_lib.get_partition_axis(
        decoding_sharding_config.attn_activation_partition, 1
    )
    num_seq_shards = sharding_lib.get_partition_size(seq_partition)
    global_total_num_pages = (
        rpa.max_num_pages_per_seq_per_shard(
            max_seq_len, page_size, None, num_seq_shards
        )
        * batch_size
        * num_seq_shards
    )
    local_total_num_pages = (
        rpa.max_num_pages_per_seq_per_shard(
            max_seq_len, page_size, config.window_size, num_seq_shards
        )
        * batch_size
        * num_seq_shards
    )
  logging.info(
      'num_seq_shards=%d, global_total_num_pages=%d, local_total_num_pages=%d',
      num_seq_shards,
      global_total_num_pages,
      local_total_num_pages,
  )

  # The page counts are part of the model config (`model_lib` reads them when
  # it builds each block's paged KV cache), so they have to go back into the
  # batcher's config now that they are known.
  decoding_config = dataclasses.replace(
      decoding_config,
      global_total_num_pages=global_total_num_pages,
      local_total_num_pages=local_total_num_pages,
      page_size=page_size,
  )
  batcher = dataclasses.replace(batcher, config=decoding_config)

  # The batcher compiles its programs on first use, which would land inside the
  # first rollout of the first timed step; it lowers them against abstract
  # params, so this needs no state either. Prefill is a program of its own
  # (chunked prefill issues `max_num_issue_tokens` new tokens per pass, decode
  # runs one query position), so both have to be asked for here.
  helper.set_notes('Pre-compiling decode functions ...')
  precompile_start = time.time()
  with batcher.set_mesh():
    _ = batcher.compiled_prefill_fn
    _ = batcher.compiled_decode_fn
    _ = batcher.compiled_push_fn
    _ = batcher.compiled_release_fn
    if batcher.prefix_cache is not None:
      _ = batcher.compiled_inject_chunk_fn
      _ = batcher.compiled_extract_chunk_fn
  logging.info(
      'AOT pre-compiled the batcher decode functions in %.1fs.',
      time.time() - precompile_start,
  )

  # =========================================================================
  # Initialize the training state, on the shapes and shardings the executables
  # above were compiled for.
  # =========================================================================
  helper.set_notes('Initializing training state ...')
  state = model_lib.get_init_state(
      config, config.sharding_config, helper.ckpt_mngr, helper.ckpt_dir
  )
  helper.save_state_info(state)

  train_iter_state = None
  if (
      helper.ckpt_mngr
      and (latest_step := helper.ckpt_mngr.latest_step()) is not None
  ):
    data_state = ckpt_lib.load_data_state_from_dir(helper.ckpt_dir, latest_step)
    assert isinstance(data_state, Mapping)
    train_iter_state = data_state.get('train_iter_state', None)

  # Load initial params into batcher (convert from training params).
  with batcher.set_mesh():
    # `get_raw_arrays` because `model.init` wraps every parameter in an
    # `AnnotatedArray` and `reshard` pairs the two trees with `jax.tree.map`,
    # which flattens the target only as far as the SOURCE's leaves.
    abstract_decoding_params = common.get_raw_arrays(
        common.eval_abstract_output(
            lambda: jax.tree.map(
                lambda x: jnp.astype(x, config.decoding_quant_scheme),
                batcher.model.init(jax.random.key(0)),
            )
        )
    )
    decoding_params = reshard(
        common.get_raw_arrays(state['params']), abstract_decoding_params
    )
    batcher.update_params(decoding_params)
    del decoding_params

  eval_enabled = (
      bool(config.validation_datasets) and config.validation_eval_interval > 0
  )
  if not eval_enabled:
    logging.info(
        'Held-out eval disabled (interval=%s, %d validation datasets); '
        'judging on train-side pass@1.',
        config.validation_eval_interval,
        len(config.validation_datasets),
    )

  # =========================================================================
  # Prepare datasets
  # =========================================================================
  start_steps = int(state['steps'])
  logging.info('Initializing dataset.')
  train_set = data_lib.create_iter_dataset(config, training=True)
  assert config.batch_mode == data_lib.BATCH_NONE
  train_set = train_set.map_with_index(lambda i, x: dict(__index__=i, **x))

  train_iter = iter(train_set)
  if train_iter_state is not None:
    logging.info('Restoring training iter state: %s.', train_iter_state)
    train_iter.set_state(train_iter_state)  # pyrefly: ignore[bad-argument-type]

  # =========================================================================
  # Thread coordination
  # =========================================================================
  completed_queue: queue.Queue[list[Sample]] = queue.Queue()
  pause_requested = threading.Event()
  pause_done = threading.Event()
  resume_requested = threading.Event()
  stop_event = threading.Event()
  error_queue: queue.Queue[Exception] = queue.Queue(1)

  # =========================================================================
  # Offload training state to host BEFORE sampling starts.
  #
  # The training params/opt-state will sit in pinned host memory while the
  # sampling thread uses device HBM for KV cache, activations, and the
  # batcher's decoding-mesh params. We onload right before the train step
  # and re-offload as soon as the post-step batcher param update is done.
  # =========================================================================
  logging.info('Offloading training state to host (initial).')
  initial_offload_start = time.time()
  state_host = offload_to_host(state)
  jax.block_until_ready(state_host)
  del state
  initial_offload_time = time.time() - initial_offload_start
  logging.info(
      'Offloaded initial training state. Took %s seconds.',
      initial_offload_time,
  )

  sampling_thread = batcher.thread(
      stop_event=stop_event,
      error_message_queue=error_queue,
      pause_event=pause_requested,
      paused_event=pause_done,
      resume_event=resume_requested,
  )
  sampling_thread.start()
  logging.info('Started RL sampling thread.')

  async def sample_one_and_evaluate(
      example: Mapping[str, Any], messages: Any
  ) -> Sample:
    """Enqueues one decode and runs reward evaluation once it completes.

    Wrapping a single sample+evaluation into its own coroutine ensures that
    reward evaluation for an early-finishing sample starts as soon as that
    sample is done, without waiting for sibling samples of the same prompt.

    Args:
      example: Raw dataset example (forwarded to the evaluator).
      messages: Prompt messages produced from `example`.

    Returns:
      The completed `Sample` (with reward populated if eval succeeded).
    """
    loop = asyncio.get_running_loop()
    f = loop.create_future()
    batcher.enqueue(messages, f)
    response = await f

    if response.code != grpc.StatusCode.OK:
      raise ValueError(f'Batcher decode error: {response.details}')

    seq = response.result
    logging.info('seq=%s', seq)
    sample = _seq_to_sample(seq, prompt_index=example.get('__index__', 0))
    logging.info('sample output_text=%s', sample.output_text)

    try:
      reward = await loop.run_in_executor(
          evaluation_executor,
          evaluation.evaluate,
          example,
          sample.output_text,
      )
      sample = sample.update_with_evaluation_result(reward)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Reward evaluation failed: %s', e)

    return sample

  async def process_one(
      example: Mapping[str, Any],
      num_samples: int,
      destination_queue: queue.Queue[list[Sample]] | None = None,
  ) -> list[Sample]:
    """Schedules this prompt's k sample+eval coroutines.

    Args:
      example: Raw dataset example.
      num_samples: Number of rollouts to sample.
      destination_queue: Optional queue to put the completed group of samples.

    Returns:
      List of completed `Sample`s for this prompt.
    """
    loop = asyncio.get_running_loop()
    messages = evaluation.get_messages(example)
    logging.info('messages=%s', messages)

    if example.get('extra_inputs'):
      raise NotImplementedError(
          'extra_inputs are not carried by the packed training schema.'
      )

    sample_tasks = [
        loop.create_task(sample_one_and_evaluate(example, messages))
        for _ in range(num_samples)
    ]
    group = list(await asyncio.gather(*sample_tasks))
    if destination_queue is not None:
      destination_queue.put(group)

    logging.info(
        'process_one: group size=%d for index=%s',
        len(group),
        example.get('__index__', 0),
    )
    return group

  def rollout_producer_thread_fn():
    if not helper.is_primary:
      return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pending_tasks = set()
    semaphore = asyncio.Semaphore(batch_size)

    async def iterate_data():
      while not stop_event.is_set():
        await semaphore.acquire()
        example = next(train_iter)
        logging.info('example=%s', example)

        task = loop.create_task(
            process_one(
                example,
                num_samples=config.num_samples_per_example,
                destination_queue=completed_queue,
            )
        )

        task.add_done_callback(lambda _: semaphore.release())
        # Register add() before the discard callback so a task that
        # completes immediately doesn't try to discard before it's added.
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    try:
      loop.run_until_complete(iterate_data())
      if pending_tasks:
        loop.run_until_complete(asyncio.wait(pending_tasks, timeout=60))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Rollout producer failed: %s', e)
      error_queue.put(e)
      stop_event.set()
    finally:
      loop.close()

  rollout_producer_thread = threading.Thread(
      target=rollout_producer_thread_fn, daemon=True
  )
  rollout_producer_thread.start()
  logging.info('Started rollout producer thread.')

  # On non-primary tasks the rollout producer thread exits immediately (only
  # task 0 drives the data), so it is not a liveness signal there.
  live_threads = [sampling_thread]
  if helper.is_primary:
    live_threads.append(rollout_producer_thread)

  def run_eval() -> tuple[dict[str, float], int]:
    """Scores the held-out sets through the batcher."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _eval() -> tuple[dict[str, float], int]:
      scalars: dict[str, float] = {}
      per_ds_pass_at_1: dict[str, float] = {}
      num_rollouts = 0
      # One sample per prompt per epoch: at the default single epoch this is a
      # plain single-pass pass@1, and pass@k only exists when there are k
      # epochs to be correct in at least one of.
      k = max(1, config.validation_eval_epochs)

      for ds_cfg in config.validation_datasets:
        source = ds_cfg.source
        if isinstance(source, str):
          source = data_lib.DataSourceRegistry.get_instance(source)
        name = getattr(source, 'name', None) or type(source).__name__
        eval_set = data_lib.create_iter_dataset(
            config, training=False, ds_config=ds_cfg
        )
        assert config.batch_mode == data_lib.BATCH_NONE
        eval_set = eval_set.map_with_index(lambda i, x: dict(__index__=i, **x))
        score_tasks = [process_one(item, num_samples=k) for item in eval_set]
        logging.info(
            'validation_datasets[%r]: %d prompts', name, len(score_tasks)
        )

        groups = await asyncio.gather(*score_tasks)
        num_rollouts += sum(len(g) for g in groups)
        stats = compute_group_stats(groups)
        if not stats:
          continue
        per_ds_pass_at_1[name] = stats['pass@1']
        scalars[f'{name}/eval_pass@1'] = stats['pass@1']
        if k > 1:
          scalars[f'{name}/eval_pass@{k}'] = stats['pass@k']

      if len(per_ds_pass_at_1) == 1:
        only = next(iter(per_ds_pass_at_1))
        for metric in ('eval_pass@1', f'eval_pass@{k}'):
          if f'{only}/{metric}' in scalars:
            scalars[metric] = scalars[f'{only}/{metric}']

      return scalars, num_rollouts

    try:
      return loop.run_until_complete(_eval())
    finally:
      loop.close()

  # =========================================================================
  # Main training loop
  # =========================================================================
  train_max_seq_len = config.train_max_seq_len
  steps = start_steps
  # Background scalar writes with a flush every step: the inline path blocked
  # the loop on TensorBoard.
  # Every automatic reaction reads the median over this many steps: one bad
  # batch of groups moves a step's KL two orders of magnitude.

  stats: Mapping[str, Any] = {}
  should_early_stop = False
  final_result: dict[str, Any] = {}
  final_result['eval_accuracy_history'] = []

  while steps <= config.num_train_steps and not should_early_stop:
    if (
        eval_enabled
        and helper.is_primary
        and (
            steps % config.validation_eval_interval == 0
            or steps == config.num_train_steps
        )
    ):
      helper.set_notes(f'{steps=}, evaluating')
      eval_start = time.time()
      eval_scalars, eval_rollouts = run_eval()
      eval_time = time.time() - eval_start
      logging.info(
          'Eval for step %d took %.1fs over %d rollouts: %s',
          steps,
          eval_time,
          eval_rollouts,
          eval_scalars,
      )
      if eval_scalars:
        helper.write_scalars(steps, eval_scalars)
        helper.flush()
        if 'eval_accuracy' in eval_scalars:
          final_result['eval_accuracy'] = eval_scalars['eval_accuracy']
          final_result['eval_accuracy_history'].append(
              eval_scalars['eval_accuracy']
          )
        should_early_stop = should_early_stop or (
            config.early_stop
            and config.early_stop.should_stop(steps, eval_scalars)
        )
      if should_early_stop:
        break

    start_time = time.time()
    # `start_time` marks the beginning of the training iteration (excluding
    # eval). It serves as the baseline for:
    # - `sampling_time`: The duration spent collecting rollouts from the
    #   concurrent sampling thread (including queue waits and status updates).
    # - `total_time`: The full training iteration (collection, pause, HBM
    #   offload/onload, training step, and parameter conversion).
    helper.set_notes(f'{steps=}, collecting')

    # =====================================================================
    # Step 1 & 2: Collect completed sample groups from the sampling thread
    # =====================================================================
    groups: list[list[Sample]] = []
    training_sequences: list[TrainSequence] = []
    num_sampled = 0
    if helper.is_primary:
      deadline = (
          start_time + _COLLECTION_TIMEOUT_SECONDS
          if _COLLECTION_TIMEOUT_SECONDS > 0
          else None
      )
      while len(training_sequences) < config.train_batch_size:
        if not error_queue.empty():
          raise error_queue.get()
        if not sampling_thread.is_alive():
          raise RuntimeError('Sampling thread died unexpectedly.')
        if not rollout_producer_thread.is_alive():
          raise RuntimeError('Rollout producer thread died unexpectedly.')
        if deadline is not None and time.time() > deadline:
          logging.warning(
              'Collection budget of %ss elapsed with %d/%d trainable samples; '
              'training on what has been collected.',
              _COLLECTION_TIMEOUT_SECONDS,
              len(training_sequences),
              config.train_batch_size,
          )
          if len(training_sequences) > 0:
            break

        helper.set_notes(
            f'{steps=}, collecting'
            f' {len(training_sequences)}/{config.train_batch_size} trainable'
        )

        try:
          group = completed_queue.get(timeout=1.0)
          logging.info('group size=%d', len(group))
        except queue.Empty:
          continue

        validated_group = []
        for sample in group:
          is_nan_reward = sample.reward is not None and np.isnan(sample.reward)
          is_invalid = is_nan_reward or (
              config.filter_truncated and sample.truncated
          )
          validated_group.append(
              dataclasses.replace(sample, is_valid_for_training=not is_invalid)
          )
        num_sampled += len(validated_group)

        normalized_group = normalize_group_rewards(
            [validated_group], config.normalize_reward_method
        )[0]
        groups.append(normalized_group)

        for sample in normalized_group:
          if (train_sequence := sample.train_sequence()) is not None:
            training_sequences.append(train_sequence)

      # From the TOP of the iteration: it covers the status note above and
      # every queue wait, all of it concurrent with the sampler.
      sampling_time = time.time() - start_time
      logging.info(
          'Sampling time: %s sec, sampled %d samples, kept %d.',
          sampling_time,
          num_sampled,
          len(training_sequences),
      )

      # =====================================================================
      # Pause sampling thread for training
      # =====================================================================
      logging.info('Requesting sampling thread to pause...')
      pause_requested.set()

      # Stats after pause to overlap the time.
      stats = compute_group_stats(groups)
      for k, v in stats.items():
        helper.add_metric(k, v)

    pause_start = time.time()
    _wait_until(
        pause_done.is_set,
        'the sampling thread to pause',
        threads=live_threads,
        error_queue=error_queue,
    )
    pause_time = time.time() - pause_start
    logging.info('Sampling thread paused. Took %s seconds.', pause_time)

    logging.info('Offloading sampling state to host...')
    offload_start = time.time()
    with batcher.set_mesh():
      sampling_state_host = offload_to_host(batcher.sampling_state)
      batcher.state['sampling_state'] = None
      jax.block_until_ready(sampling_state_host)
    offload_time = time.time() - offload_start
    logging.info('Offloaded sampling state. Took %s seconds.', offload_time)

    # Free the batcher's decoding params from device HBM during training.
    # Updated decoding params will be resharded from the training state
    # after the train step.
    if batcher.state.get('params') is not None:
      batcher.state['params'] = None

    # =====================================================================
    # Step 3: TRAIN
    # =====================================================================
    helper.set_notes(f'{steps=}, training')
    train_start_time = time.time()

    # Onload training state from host into device HBM for this step. The
    # training mesh is the ambient one; only decode regions are scoped.
    # Everything dispatched above must have LANDED before the training state
    # claims the HBM it is freeing.
    onload_train_start = time.time()
    state = onload_from_host(state_host)
    del state_host
    jax.block_until_ready(state)
    onload_train_time = time.time() - onload_train_start
    logging.info('Onloaded training state. Took %s seconds.', onload_train_time)

    train_iter_state = train_iter.get_state()

    # `onload_train` is the DISPATCH; the 31.4 GB lands here, on the first host
    # read of the onloaded state. Booked where it is paid rather than forced
    # earlier with a barrier, which would serialise it against the line above.
    steps = int(state['steps'])
    # Drain first: a checkpoint is the point a reader trusts, so the
    # scalars up to it must already be on disk.
    helper.save_ckpt(state, steps, data={'train_iter_state': train_iter_state})

    # Only the primary holds the samples; the micro-batches it packs are
    # broadcast to every host.
    micro_batches = create_train_batches(
        training_sequences,
        rows_per_micro_batch=rows_per_micro_batch,
        max_seq_len=train_max_seq_len,
        is_primary=helper.is_primary,
    )
    with jax.profiler.StepTraceAnnotation('train', step_num=steps):
      lr = lr_fn(state['steps'])
      loss, state, log_dict = train_step(state, micro_batches, lr)
    if loss is not None:
      helper.add_metric('loss', float(loss))
      for key, value in log_dict.items():
        helper.add_metric(key, float(value))

    train_step_time = time.time() - train_start_time
    logging.info('train_step_time: %s sec', train_step_time)

    # =====================================================================
    # Step 4 & 5: Convert params for decoding, update batcher
    # =====================================================================
    convert_start = time.time()

    with batcher.set_mesh():
      raw_params = common.get_raw_arrays(state['params'])
      # How many transfers the per-leaf path would issue: bytes/second means
      # nothing without it (5.2 GB in one collective is not 5.2 GB in N).
      # The realised path beside the requested one: 1 when the whole tree was
      # cast by ONE program, 0 when a leaf forced the per-leaf fallback.
      decoding_params = reshard(raw_params, abstract_decoding_params)
      del raw_params
      batcher.update_params(decoding_params)
      del decoding_params

    convert_time = time.time() - convert_start
    logging.info('Converted decoding params. Took %s seconds.', convert_time)

    new_steps = int(state['steps'])

    # =====================================================================
    # Offload training state back to host before resuming sampling.
    # =====================================================================
    logging.info('Offloading training state to host...')
    offload_train_start = time.time()
    state_host = offload_to_host(state)
    del state
    jax.block_until_ready(state_host)
    offload_train_time = time.time() - offload_train_start
    logging.info(
        'Offloaded training state. Took %s seconds.', offload_train_time
    )

    # =====================================================================
    # Step 6: Onload sampling state, resume sampling thread
    # =====================================================================
    logging.info('Onloading sampling state...')
    onload_start = time.time()
    with batcher.set_mesh():
      sampling_state = onload_from_host(sampling_state_host)
      del sampling_state_host

    batcher.state['sampling_state'] = jax.block_until_ready(sampling_state)
    del sampling_state
    onload_time = time.time() - onload_start
    logging.info('Onloaded sampling state. Took %s seconds.', onload_time)

    # Resume the sampling thread. The batcher clears `pause_done` only once
    # it has observed the resume, and waiting for that ack is what keeps the
    # hosts in lockstep: without it the next iteration's pause wait returns on
    # the stale flag, and this thread would start issuing training programs
    # while the sampling thread is still issuing decode programs -- the two
    # program streams then desync across hosts and the TPU runtime aborts with
    # a launch-id mismatch.
    pause_requested.clear()
    resume_requested.set()
    _wait_until(
        lambda: not pause_done.is_set(),
        'the sampling thread to acknowledge the resume',
        threads=live_threads,
        error_queue=error_queue,
    )
    logging.info('Resumed sampling thread.')

    # =====================================================================
    # Logging and metrics
    # =====================================================================

    agg_metrics = helper.get_aggregated_metrics()
    # Every reaction reads the MEDIAN over a window of steps, never one step.
    should_early_stop = should_early_stop or (
        config.early_stop and config.early_stop.should_stop(steps, agg_metrics)
    )
    # Keep the cadence stride-safe: `steps` need not land on the exact
    # multiples `helper.should_log_metrics` tests for.
    if steps % config.tb_log_interval == 0 or steps >= config.num_train_steps:
      metrics_dict = dict(lr=lr)
      metrics_dict.update(agg_metrics)
      metrics_dict.update(pytree.to_flat_dict(log_dict, sep='/'))
      helper.write_scalars(steps, metrics_dict)
      helper.flush()

    total_time = time.time() - start_time
    helper.add_metric('total_time', total_time)

    steps = new_steps

  # =========================================================================
  # Shutdown
  # =========================================================================
  stop_event.set()
  if pause_requested.is_set():
    resume_requested.set()  # Unblock sampling thread if paused.
  sampling_thread.join(timeout=30)
  rollout_producer_thread.join(timeout=30)
  evaluation_executor.shutdown(wait=False)

  final_result['train_accuracy'] = float(stats.get('accuracy', 0.0))
  final_result['early_stop'] = should_early_stop
  if should_early_stop:
    logging.info('Training is early stopped!')
  helper.close(final_result)
  return final_result

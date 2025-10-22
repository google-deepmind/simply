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
"""Components for RL training."""

import abc
import collections
from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
import contextlib
import dataclasses
import functools
import time
from typing import Any, cast

from absl import logging
import einops
import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply import model_lib
from simply import tool_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import distributions
from simply.utils import evaluation_lib as eval_lib
from simply.utils import experiment_helper as exp_helper
from simply.utils import lm_format as lm_format_lib
from simply.utils import masked
from simply.utils import registry
from simply.utils import replay_buffers
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib
from simply.utils import tokenization


Array = jax.Array | np.ndarray
Batch = model_lib.Batch
PyTree = common.PyTree
TrainLoopRegistry = model_lib.TrainLoopRegistry
ExperimentHelper = exp_helper.ExperimentHelper


def compute_logprobs(
    model,
    params: common.PyTree,
    batch: dict[str, Array],
    microbatch_size: int | None = None,
) -> Array:
  """Computes the logprobs of the decoder tokens."""

  def _compute_logprobs(microbatch: dict[str, Array]) -> Array:
    inputs = microbatch['input_tokens']
    targets = microbatch['target_tokens']
    segment_ids = microbatch.get('decoder_segment_ids', None)
    segment_positions = microbatch.get('decoder_positions', None)
    mask = microbatch['answer_masks']
    logits, _ = model.apply(
        params,
        inputs,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
    )
    logits = jnp.astype(logits, jnp.float32)
    m = distributions.Categorical(logits)
    logprobs = masked.masked(m.log_prob(targets), mask=mask)
    return logprobs

  batch_size = batch['input_tokens'].shape[0]
  num_microbatches = 1
  if microbatch_size is not None and microbatch_size > 0:
    assert batch_size % microbatch_size == 0
    num_microbatches = batch_size // microbatch_size

  if num_microbatches == 1:
    return _compute_logprobs(batch)

  microbatches = jax.tree.map(
      lambda x: einops.rearrange(x, '(g b) ... -> g b ...', g=num_microbatches),
      batch,
  )
  _, logprobs = jax.lax.scan(
      lambda _, microbatch: (None, _compute_logprobs(microbatch)),
      init=None,
      xs=microbatches,
      length=num_microbatches,
  )
  logprobs = einops.rearrange(
      logprobs, 'g b ... -> (g b) ...', g=num_microbatches
  )
  return logprobs


def np_safe_mean(x, where):
  return np.sum(x, where=where) / np.maximum(np.sum(where), 1e-5)


def np_safe_weighted_mean(x, w):
  normed_w = w / np.sum(w)
  return np.nansum(x * normed_w)


def np_safe_std(x, where):
  mean = np_safe_mean(x, where=where)
  d2 = (x - mean) ** 2
  var = np.sum(d2, where=where) / np.maximum(np.sum(where), 1e-5)
  return np.sqrt(var)


def compute_stats(
    rewarded_completed_batch: Mapping[int, Sequence[Mapping[str, Any]]],
    lm_interface: model_lib.LMInterface,
    evaluation: eval_lib.Evaluation,
) -> dict[str, np.ndarray]:
  stats_rows = []
  pass_at_k_corrects = []
  pass_at_k_eval_masks = []
  for rewarded_per_prompt_batch in rewarded_completed_batch.values():
    corrects = []
    eval_masks = []
    for rewarded_per_response in rewarded_per_prompt_batch:
      stats = {}
      so = rewarded_per_response['lm_sampling_output']
      stats['seq_len'] = len(so.output_token_ids) + len(so.input_token_ids) - 1
      stats['prompt_len'] = len(so.input_token_ids) - 1
      stats['response_len'] = len(so.output_token_ids)
      stats['truncated'] = so.output_token_ids[-1] not in lm_interface.eos_ids
      stats['reward'] = rewarded_per_response['reward']
      stats['correct'] = rewarded_per_response['correct']
      stats['eval_mask'] = not np.isnan(rewarded_per_response['reward'])
      stats['train_sample_mask'] = rewarded_per_response['train_sample_mask']
      for reward_type in getattr(evaluation, 'reward_types', ()):
        stats[f'is_reward_type/{reward_type}'] = (
            reward_type in rewarded_per_response.get('reward_types', [])
        )
      stats_rows.append(stats)
      corrects.append(stats['correct'])
      eval_masks.append(stats['eval_mask'])
    pass_at_k_corrects.append(np.any(corrects))
    pass_at_k_eval_masks.append(np.any(eval_masks))
  stats_columns = common.convert_rows_to_columns(stats_rows)
  logging.info('stats_columns: %s', jax.tree.map(np.shape, stats_columns))
  eval_mask = stats_columns['eval_mask']
  pass_at_k_corrects = np.array(pass_at_k_corrects)
  pass_at_k_eval_masks = np.array(pass_at_k_eval_masks)
  stats = {
      'seq_len/mean': np_safe_mean(stats_columns['seq_len'], where=eval_mask),
      'seq_len/max': np.max(
          stats_columns['seq_len'], where=eval_mask, initial=0
      ),
      'seq_len/min': np.min(
          stats_columns['seq_len'],
          where=eval_mask,
          initial=np.iinfo(np.int32).max,
      ),
      'prompt_len/mean': np_safe_mean(
          stats_columns['prompt_len'], where=eval_mask
      ),
      'prompt_len/max': np.max(
          stats_columns['prompt_len'], where=eval_mask, initial=0
      ),
      'prompt_len/min': np.min(
          stats_columns['prompt_len'],
          where=eval_mask,
          initial=np.iinfo(np.int32).max,
      ),
      'response_len/mean': np_safe_mean(
          stats_columns['response_len'], where=eval_mask
      ),
      'response_len/max': np.max(
          stats_columns['response_len'], where=eval_mask, initial=0
      ),
      'response_len/min': np.min(
          stats_columns['response_len'],
          where=eval_mask,
          initial=np.iinfo(np.int32).max,
      ),
      'truncated': np_safe_mean(stats_columns['truncated'], where=eval_mask),
      'reward': np_safe_mean(stats_columns['reward'], where=eval_mask),
      'accuracy': np_safe_mean(stats_columns['correct'], where=eval_mask),
      'pass_at_k': np_safe_mean(pass_at_k_corrects, where=pass_at_k_eval_masks),
      'pass_at_k_eval_count': np.sum(pass_at_k_eval_masks),
      'eval_count': np.sum(eval_mask),
      'train_count': np.sum(stats_columns['train_sample_mask']),
  }
  # TODO: Ideally we should implement reward_by_type.
  for reward_type in getattr(evaluation, 'reward_types', ()):
    is_reward_type = stats_columns[f'is_reward_type/{reward_type}']
    stats.update({
        f'reward/{reward_type}': np_safe_mean(
            stats_columns['reward'], where=is_reward_type & eval_mask
        ),
        f'accuracy/{reward_type}': np_safe_mean(
            stats_columns['correct'], where=is_reward_type & eval_mask
        ),
        f'eval_count/{reward_type}': np.sum(is_reward_type & eval_mask),
        f'train_count/{reward_type}': np.sum(
            is_reward_type & stats_columns['train_sample_mask']
        ),
    })
  logging.info('stats: %s', stats)
  stats = jax.experimental.multihost_utils.process_allgather(stats, tiled=True)

  eval_count = stats['eval_count']
  formatted_stats = {
      'seq_len/mean': np_safe_weighted_mean(stats['seq_len/mean'], eval_count),
      'seq_len/max': np.max(stats['seq_len/max']),
      'seq_len/min': np.min(stats['seq_len/min']),
      'prompt_len/mean': np_safe_weighted_mean(
          stats['prompt_len/mean'], eval_count
      ),
      'prompt_len/max': np.max(stats['prompt_len/max']),
      'prompt_len/min': np.min(stats['prompt_len/min']),
      'truncated': np_safe_weighted_mean(stats['truncated'], eval_count),
      'response_len/mean': np_safe_weighted_mean(
          stats['response_len/mean'], eval_count
      ),
      'response_len/max': np.max(stats['response_len/max']),
      'response_len/min': np.min(stats['response_len/min']),
      'reward': np_safe_weighted_mean(stats['reward'], eval_count),
      'accuracy': np_safe_weighted_mean(stats['accuracy'], eval_count),
      'pass_at_k': np_safe_weighted_mean(
          stats['pass_at_k'], stats['pass_at_k_eval_count']
      ),
      'eval_count': np.sum(eval_count),
      'train_sample_ratio': np.sum(stats['train_count']) / np.sum(eval_count),
  }
  for reward_type in getattr(evaluation, 'reward_types', ()):
    eval_count = stats[f'eval_count/{reward_type}']
    formatted_stats.update({
        f'reward/{reward_type}': np_safe_weighted_mean(
            stats[f'reward/{reward_type}'], eval_count
        ),
        f'accuracy/{reward_type}': np_safe_weighted_mean(
            stats[f'accuracy/{reward_type}'], eval_count
        ),
        f'eval_count/{reward_type}': np.sum(eval_count),
        f'train_count/{reward_type}': np.sum(
            stats[f'train_count/{reward_type}']
        ),
    })
  return formatted_stats


class RewardNormalizerRegistry(registry.RootRegistry):
  namespace: str = 'RewardNormalizer'


class RewardNormalizer:

  class Base(abc.ABC):

    @abc.abstractmethod
    def normalize(
        self, rewards: np.ndarray, example_ids: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
      """Normalizes the rewards given they are grouped by example_ids.

      Args:
        rewards: 1D array of rewards of the samples.
        example_ids: 1D array of example ids of the samples, same ids are next
          to each other.
        masks: The masks of the samples.

      Returns:
        The normalized 1D array of rewards.
      """
      raise NotImplementedError()

  @RewardNormalizerRegistry.register
  class Global(Base):

    def normalize(
        self, rewards: np.ndarray, example_ids: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
      mean_reward = np_safe_mean(rewards, where=masks)
      std_reward = np_safe_std(rewards, where=masks)
      return (rewards - mean_reward) / np.maximum(std_reward, 1e-5)

  @RewardNormalizerRegistry.register
  class ByGroup(Base):

    def normalize_by_group(
        self,
        rewards: np.ndarray,
        example_ids: np.ndarray,
        masks: np.ndarray,
        std: np.ndarray | None = None,
    ) -> np.ndarray:
      new_rewards = []
      # TODO: Explore more efficient ways to implement this instead of
      # this for loop.
      i = 0
      while i < rewards.shape[0]:
        j = i + 1
        while j < rewards.shape[0] and example_ids[j] == example_ids[i]:
          j += 1
        group_rewards = rewards[i:j]
        group_masks = masks[i:j]
        mean_reward = np_safe_mean(group_rewards, where=group_masks)
        if std is None:
          std_reward = np_safe_std(group_rewards, where=group_masks)
        else:
          std_reward = std
        for k in range(i, j):
          new_rewards.append(
              (rewards[k] - mean_reward) / np.maximum(std_reward, 1e-5)
          )
        i = j
      return np.array(new_rewards)

    def normalize(
        self, rewards: np.ndarray, example_ids: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
      return self.normalize_by_group(rewards, example_ids, masks)


def local_batch_to_global(
    local_batch: Mapping[str, np.ndarray],
    mask_name: str,
    num_valids: np.ndarray,
    global_batch_size: int,
) -> dict[str, np.ndarray]:
  """Converts a local batch from different hosts to a global batch.

  The sizes of local_batch at different hosts are not necessarily the same. This
  function creates a global_batch_size buffer and copies the filtered local
  batch to the buffer at corresponding positions based on num_valids across all
  hosts, and then sums the buffer across all hosts. When total number of valid
  samples is larger than global_batch_size, the extra samples from last hosts
  would be dropped.

  Args:
    local_batch: A batch of data from the local devices.
    mask_name: The name of the mask in the batch, which is used to identify
      valid samples.
    num_valids: The number of valid samples in the batch across all hosts.
    global_batch_size: The size of the global batch.

  Returns:
    Global batch with size of global_batch_size, which is aggregated from
    local batches across all hosts.
  """
  local_mask = local_batch[mask_name]
  logging.info('num_valids=%s', num_valids)
  logging.info('jax.process_index()=%s', jax.process_index())
  if len(num_valids.shape) < 1:
    num_valids = np.array([num_valids])
  local_num_valids = num_valids[jax.process_index()]
  logging.info('local_num_valid=%s', local_num_valids)
  logging.info('local_mask=%s', local_mask)
  local_valid_batch = {}
  for k, v in local_batch.items():
    v = v[local_mask]
    logging.info('local_valid_batch[%s].shape=%s', k, v.shape)
    assert v.shape[0] == local_num_valids
    local_valid_batch[k] = v
    if np.any(np.isnan(v)):
      raise ValueError(f'local_valid_batch[{k}] contains NaNs: {v}')

  start_indices = np.cumulative_sum(num_valids, include_initial=True)
  logging.info('start_indices=%s', start_indices)
  start_index = start_indices[jax.process_index()]
  global_batch = {}
  if start_index >= global_batch_size:
    for k, v in local_valid_batch.items():
      global_batch[k] = np.zeros(
          (global_batch_size,) + v.shape[1:], dtype=v.dtype
      )
  else:
    for k, v in local_valid_batch.items():
      local_batch_size = min(global_batch_size - start_index, v.shape[0])
      global_batch[k] = model_lib.pad_along_axis(
          v[:local_batch_size],
          (start_index, global_batch_size - local_batch_size - start_index),
          axis=0,
      )

  logging.info(
      'global_batch.shape(before sum_across_hosts)=%s',
      jax.tree.map(np.shape, global_batch),
  )
  time_start = time.time()
  global_batch = sharding_lib.sum_across_hosts(global_batch)
  global_batch = cast(dict[str, np.ndarray], global_batch)
  logging.info('sum_across_hosts took %f seconds', time.time() - time_start)

  logging.info(
      'global_batch.shape(after sum_across_hosts)=%s',
      jax.tree.map(np.shape, global_batch),
  )
  for k, v in global_batch.items():
    global_batch[k] = v.astype(local_batch[k].dtype)

  global_mask = global_batch[mask_name]
  logging.info('global_mask=%s', global_mask)
  expected_global_mask = np.arange(global_batch_size) < np.sum(num_valids)
  np.testing.assert_array_equal(global_mask, expected_global_mask)
  return global_batch


def create_train_batch(
    rewarded_batch: Mapping[int, Sequence[Mapping[str, Any]]],
    num_valid_samples: np.ndarray,
    train_batch_size: int,
    max_seq_len: int = 1024,
    normalize_reward_method: str = '',
    ref_params: PyTree | None = None,
    compute_logprobs_fn: Callable[..., Array] | None = None,
) -> Mapping[str, Array]:
  """Creates a batch of data for training.

  Args:
    rewarded_batch: A batch of per-prompt rewarded example batches.
    num_valid_samples: The number of valid samples in the batch across all
      hosts.
    train_batch_size: The size of the train batch.
    max_seq_len: The maximum length for each sequence.
    normalize_reward_method: How to normalize reward.
    ref_params: The params of the reference model.
    compute_logprobs_fn: A function to compute logprobs.

  Returns:
    A dict of arrays as a batch of data. It includes the following items:
      input_tokens: shape = (batch_size, max_seq_len)
      target_tokens: shape = (batch_size, max_seq_len)
      target_masks: shape = (batch_size, max_seq_len)
      logprobs: shape = (batch_size, max_seq_len)
      answer_masks: shape = (batch_size, max_seq_len)
      verdicts: shape = (batch_size, 1)
      rewards: shape = (batch_size, 1)
      sample_masks: shape = (batch_size, 1)
      ref_logprobs: shape = (batch_size, max_seq_len)
      **extra_inputs: extra inputs from the rewarded batch.
  """
  local_train_rows = []
  for rewarded_batch_per_prompt in rewarded_batch.values():
    for rewarded_per_response in rewarded_batch_per_prompt:
      # Add everything to train_batch first.
      so = rewarded_per_response['lm_sampling_output']
      all_token_ids = so.input_token_ids + so.output_token_ids
      logging.info('all_token_ids_len=%s', len(all_token_ids))
      all_token_scores = so.input_token_scores + so.output_token_scores
      assert len(all_token_ids) == len(all_token_scores) + 1
      if hasattr(so, 'answer_mask'):
        answer_mask = np.array(so.answer_mask[1:])
        assert len(all_token_ids) == len(answer_mask) + 1
      else:
        answer_mask = np.concatenate([
            np.zeros(len(so.input_token_ids) - 1, dtype=np.bool),
            np.ones(len(so.output_token_ids), dtype=np.bool),
        ])
      example_per_response = dict(
          input_tokens=np.array(all_token_ids[:-1]),
          target_tokens=np.array(all_token_ids[1:]),
          logprobs=np.array(all_token_scores),
          target_masks=np.ones(len(all_token_ids) - 1, dtype=np.bool),
          answer_masks=answer_mask,
      )
      extra_inputs = rewarded_per_response.get('extra_inputs', {})
      example_per_response.update(extra_inputs)
      for k, v in example_per_response.items():
        if k not in extra_inputs:
          example_per_response[k] = model_lib.pad_to_along_axis(
              v, max_seq_len, axis=0
          )
      example_per_response.update(
          dict(
              in_batch_example_ids=(
                  # The index starts from 0. Plus 1 to distinguish padding.
                  rewarded_per_response['in_batch_example_index'] + 1
              ),
              rewards=rewarded_per_response['reward'],
              verdicts=rewarded_per_response['correct'],
              train_sample_masks=rewarded_per_response['train_sample_mask'],
          )
      )
      local_train_rows.append(example_per_response)

  local_train_batch = common.convert_rows_to_columns(local_train_rows)
  global_train_batch = local_batch_to_global(
      local_train_batch,
      mask_name='train_sample_masks',
      num_valids=num_valid_samples,
      global_batch_size=train_batch_size,
  )

  if normalize_reward_method:
    global_train_batch['rewards'] = RewardNormalizerRegistry.get_instance(
        normalize_reward_method
    ).normalize(
        global_train_batch['rewards'],
        global_train_batch['in_batch_example_ids'],
        global_train_batch['train_sample_masks'],
    )

  for k, v in global_train_batch.items():
    if v.ndim == 1:
      global_train_batch[k] = np.expand_dims(v, axis=-1)

  if ref_params is not None and compute_logprobs_fn is not None:
    ref_logprobs = compute_logprobs_fn(
        params=ref_params, batch=global_train_batch
    )
    global_train_batch['ref_logprobs'] = (
        jax.experimental.multihost_utils.process_allgather(
            ref_logprobs, tiled=True
        )
    )

  return global_train_batch


def compute_return(reward: Array, mask: Array, gamma: float = 1.0) -> Array:
  """Computes the discounted return."""

  if gamma == 1.0:
    ret = jnp.flip(jnp.cumsum(jnp.flip(reward, axis=-1), axis=-1), axis=-1)
    return masked.masked(ret, mask=mask)

  def _update_fn(g: Array, r: Array) -> tuple[Array, Array]:
    g = r + gamma * g
    return g, g

  batch_size, seq_len = reward.shape
  _, ret = jax.lax.scan(
      _update_fn,
      init=jnp.zeros(batch_size),
      xs=reward.T,
      length=seq_len,
      reverse=True,
  )
  return masked.masked(ret.T, mask=mask)


def compute_ppo_loss(
    model,
    params: common.PyTree,
    batch: dict[str, Array],
    gamma: float = 1.0,
    kl_coeff: float = 0.001,
    use_grpo: bool = False,
    ppo_clip_eps_high: float = 0.2,
    ppo_clip_eps_low: float = 0.2,
    policy_ratio_cap: float | None = 10.0,
    normalize_advantage: bool = True,
    max_abs_advantage: float | None = 10.0,
    use_policy_logp_as_sampler_logp: bool = False,
) -> tuple[float, dict[str, Any]]:
  """Compute PPO loss."""
  # TODO: Consider unified field names.
  inputs = batch['input_tokens']
  targets = batch['target_tokens']
  segment_ids = batch.get('decoder_segment_ids', None)
  segment_positions = batch.get('decoder_positions', None)

  standard_keys = [
      'input_tokens',
      'target_tokens',
      'decoder_segment_ids',
      'decoder_positions',
      'target_masks',
      'answer_masks',
      'train_sample_masks',
      'rewards',
      'logprobs',
      'ref_logprobs',
      'in_batch_example_ids',
      'verdicts',
  ]
  extra_inputs = {k: v for k, v in batch.items() if k not in standard_keys}
  if not extra_inputs:
    extra_inputs = None

  target_mask = batch['target_masks']
  answer_mask = batch['answer_masks']  # (batch_size, max_seq_len)
  sample_mask = batch['train_sample_masks']  # (batch_size, 1)
  reward = batch['rewards']  # (batch_size, 1)
  assert sample_mask.ndim == 2
  assert reward.ndim == 2

  answer_mask = answer_mask * sample_mask

  seq_len = jnp.sum(target_mask, axis=-1)

  logits, _ = model.apply(
      params,
      inputs,
      segment_ids=segment_ids,
      segment_positions=segment_positions,
      extra_inputs=extra_inputs,
  )
  logits = jnp.astype(logits, jnp.float32)
  m = distributions.Categorical(logits)

  logpi = masked.masked(m.log_prob(targets), mask=answer_mask)
  logpi_old = masked.masked(batch['logprobs'], mask=answer_mask)
  logpi_ref = masked.masked(batch['ref_logprobs'], mask=answer_mask)

  if use_grpo:
    # K3 estimator from http://joschu.net/blog/kl-approx.html.
    logr = masked.masked(logpi_ref - logpi, mask=answer_mask)
    kl = masked.masked(jnp.expm1(logr) - logr, mask=answer_mask)
  else:
    kl = masked.masked(logpi - logpi_ref, mask=answer_mask)

  index = jnp.arange(kl.shape[0])
  if use_grpo:
    if gamma == 1.0:
      adv = reward * jnp.astype(answer_mask, reward.dtype)
    else:
      step_reward = jnp.zeros_like(logpi)
      step_reward = step_reward.at[index, seq_len - 1].add(jnp.squeeze(reward))
      adv = compute_return(step_reward, mask=answer_mask, gamma=gamma)
  else:
    step_reward = jax.lax.stop_gradient(-kl_coeff * kl)
    step_reward = step_reward.at[index, seq_len - 1].add(jnp.squeeze(reward))
    adv = compute_return(step_reward, mask=answer_mask, gamma=gamma)

  if normalize_advantage:
    mean, std = masked.masked_mean_std(adv, mask=answer_mask)
    adv = masked.masked((adv - mean) / (std + 1e-5), mask=answer_mask)

  if max_abs_advantage is not None:
    adv = jnp.clip(adv, -max_abs_advantage, max_abs_advantage)

  adv = jax.lax.stop_gradient(masked.masked(adv, mask=answer_mask))

  if use_policy_logp_as_sampler_logp:
    # In pure on-policy learning, we may take logpi as logpi_old to avoid logp
    # diff that may be caused by sharding diff.
    logpi_old = jax.lax.stop_gradient(logpi)

  logp_diff = masked.masked(logpi - logpi_old, mask=answer_mask)
  abs_logp_diff = jnp.abs(logp_diff)

  ratio = masked.masked(jnp.exp(logp_diff), mask=answer_mask)
  if policy_ratio_cap is not None:
    # Applies dual-clip PPO. https://arxiv.org/abs/1912.09729.
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
  if use_grpo:
    kl_loss = masked.masked_mean(kl, mask=answer_mask)
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


def decoding_mesh_context(
    decoding_mesh_shape: Sequence[int] | None = None,
    dcn_mesh_shape: Sequence[int] | None = None,
):
  if decoding_mesh_shape is None:
    return contextlib.nullcontext()
  return sharding_lib.mesh_context(
      mesh_shape=decoding_mesh_shape, dcn_mesh_shape=dcn_mesh_shape
  )


def mesh_in_params(params: common.PyTree) -> js.Mesh | None:
  """Returns the mesh in params."""
  leaves = jax.tree_util.tree_leaves(params)
  if leaves and hasattr(leaves[0].sharding, 'mesh'):
    return leaves[0].sharding.mesh
  return None


@functools.partial(jax.jit, static_argnames=['dtype'])
def tree_convert_dtype(x: PyTree, dtype: jax.typing.DTypeLike):
  return jax.tree_util.tree_map(lambda x: x.astype(dtype), x)


# TODO: Move this to serving_lib.py.
def prepare_params_for_decoding(
    params: common.PyTree,
    abstract_decoding_params: common.PyTree = None,
    quant_scheme: str = 'bfloat16',
):
  """Quantizes params and then reshards them to the current mesh."""
  # Convert to bfloat16 to reduce allgather cost.
  if quant_scheme in ['bfloat16', 'float32']:
    params = tree_convert_dtype(params, quant_scheme)
  if abstract_decoding_params:
    if sharding_lib.get_default_mesh() != mesh_in_params(params):
      params = jax.experimental.multihost_utils.process_allgather(
          params, tiled=True
      )
    params = jax.tree_util.tree_map(
        lambda x, y: sharding_lib.with_sharding_constraint(x, y.sharding),
        params,
        abstract_decoding_params,
    )
  return params


@functools.partial(model_lib.TrainLoopRegistry.register, name='rl')
def run_experiment(
    config,
    sharding_config,
    mesh_shape,
    create_dataset,
    # Leave `experiment_dir` as empty string to skip saving experiment data.
    # Useful if no need to save any data and can reduce some overhead.
    experiment_dir='',
    dcn_mesh_shape=None,
    decoding_mesh_shape=None,
):
  logging.info('jax.process_index(): %s', jax.process_index())
  # Setup model, optimizer, initial state, and mesh.
  sharding_lib.set_default_mesh_shape(
      mesh_shape=mesh_shape, dcn_mesh_shape=dcn_mesh_shape
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
  )
  model, _ = model_lib.create_model(config, sharding_config)
  helper.save_config_info(config, sharding_config, model)
  opt = config.optimizer
  state = model_lib.get_init_state(
      config, sharding_config, helper.ckpt_mngr, helper.ckpt_dir
  )
  helper.save_state_info(state)
  train_iter_state = None
  if helper.ckpt_mngr and helper.ckpt_mngr.latest_step() is not None:
    data_state = ckpt_lib.load_data_state_from_dir(
        helper.ckpt_dir, helper.ckpt_mngr.latest_step()
    )
    assert isinstance(data_state, Mapping)
    train_iter_state = data_state.get('train_iter_state', None)

  if helper.ckpt_mngr and helper.ckpt_mngr.latest_step() is not None:
    # Continue training from lastest ckpt, so we load ref_params from init_ckpt.
    abstract_params = ckpt_lib.get_abstract_params(model)
    abstract_state = {'params': abstract_params}
    ref_state = ckpt_lib.load_checkpoint_from_dir(
        config.init_ckpt_dir,
        abstract_state,
        config.init_ckpt_step,
        ckpt_format=config.init_ckpt_format,
    )
    ref_params = ref_state['params']
  else:
    ref_params = state['params']

  ref_params = tree_convert_dtype(ref_params, config.ref_params_dtype)

  # Compile loss, train and learning rate functions.
  t1 = time.time()

  @functools.partial(jax.jit, static_argnames=['add_log_info'])
  def train_one_step_fn(state, batch, lr, add_log_info=False):
    return model_lib.train_one_step(
        state=state,
        batch=batch,
        lr=lr,
        model=model,
        opt=opt,
        custom_loss_fn=functools.partial(
            compute_ppo_loss,
            gamma=config.gamma,
            kl_coeff=config.kl_coeff,
            use_grpo=config.use_grpo,
            ppo_clip_eps_high=config.ppo_clip_eps_high or config.ppo_clip_eps,
            ppo_clip_eps_low=config.ppo_clip_eps_low or config.ppo_clip_eps,
            policy_ratio_cap=config.policy_ratio_cap,
            normalize_advantage=config.normalize_advantage,
            max_abs_advantage=config.max_abs_advantage,
            use_policy_logp_as_sampler_logp=config.use_policy_logp_as_sampler_logp,
        ),
        grad_accum_steps=config.grad_accum_steps,
        clip_grad_norm=config.clip_grad_norm,
        clip_update_norm=config.clip_update_norm,
        clip_local_update_rms=config.clip_local_update_rms,
        weight_decay=config.weight_decay,
        add_log_info=add_log_info,
    )

  # Compute logprobs is using training sharding, so it follows the same
  # microbatch size as training.
  compute_logprobs_microbatch_size = None
  if config.grad_accum_steps > 1:
    compute_logprobs_microbatch_size = (
        config.train_batch_size // config.grad_accum_steps
    )
  compute_logprobs_fn = common.named_jit(
      compute_logprobs,
      'compute_logprobs_fn',
      model=model,
      # Microbatch size is at example level.
      microbatch_size=compute_logprobs_microbatch_size,
  )

  lr_fn = common.named_jit(model_lib.create_lr_schedule(config), 'lr_fn')
  dt = time.time() - t1
  logging.info('%s secs used for compiling train, loss and lr functions.', dt)

  # Prepare datasets.
  start_steps = int(state['steps'].addressable_data(0))
  logging.info('Initializing dataset.')
  train_set, eval_set = create_dataset(config)
  if train_iter_state is not None:
    logging.info('Restoring training iter state: %s.', train_iter_state)
    train_set.set_state(train_iter_state)
  eval_iter_init_state = eval_set.get_state() if eval_set is not None else None

  logging.info(
      'sharding_config.data_partition: %s', sharding_config.data_partition
  )

  evaluation = config.evaluation
  # Just set max_workers to be a large enough number. As we do multi-host
  # reward computation, we actually only need a few number of workers.
  evaluation_executor = futures.ThreadPoolExecutor(
      max_workers=config.batch_size * config.num_samples_per_example
  )
  tool_executor = tool_lib.create_tool_executor(config)
  tokenizer = tokenization.TokenizerRegistry.get(config.vocab_name)()
  sampling_params = model_lib.SamplingParams(
      temperature=config.sampling_temperature,
      max_decode_steps=config.sampling_max_decode_steps,
      intermediate_decode_steps=config.sampling_intermediate_decode_steps,
      max_seq_len=config.train_max_seq_len + 1,
      max_input_len=config.sampling_max_input_len,
      num_samples=config.num_samples_per_example,
      sort_by=None,
  )
  lm_format = lm_format_lib.LMFormatRegistry.get(config.lm_format_name)()
  model_for_decoding, _ = model_lib.create_model(
      dataclasses.replace(
          config,
          use_scan=False,
          use_remat=False,
          activation_dtype_name=config.activation_dtype_name,
      ),
      config.decoding_sharding_config or sharding_config,
  )
  extra_eos_tokens = list(
      set(config.extra_eos_tokens) | set(lm_format.extra_eos_tokens)
  )
  input_processor = sampling_lib.create_input_processor(
      config,
      vocab=tokenizer,
      bos_id_override=lm_format.bos_id,
      pad_id_override=lm_format.pad_id,
      extra_eos_tokens=extra_eos_tokens,
  )
  with decoding_mesh_context(decoding_mesh_shape, dcn_mesh_shape):
    lm_interface = model_lib.LMInterface(
        model_for_decoding,
        params=None,
        vocab=tokenizer,
        input_processor=input_processor,
        bos_id=lm_format.bos_id,
        pad_id=lm_format.pad_id,
        default_sampling_params=sampling_params,
        extra_eos_tokens=extra_eos_tokens,
    )
    abstract_decoding_params = None
    if decoding_mesh_shape or config.decoding_sharding_config:
      # That means we need to do reshard.
      abstract_decoding_params = ckpt_lib.get_abstract_params(
          model_for_decoding
      )

  seed = config.model_seed
  train_batch_size = config.train_batch_size
  train_max_seq_len = config.train_max_seq_len
  num_train_steps_per_batch = config.num_train_steps_per_batch
  replay_buffer_size = train_batch_size * num_train_steps_per_batch
  replay_buffer = replay_buffers.ReplayBuffer(replay_buffer_size)

  if num_train_steps_per_batch > 1 and config.use_policy_logp_as_sampler_logp:
    raise ValueError(
        'use_policy_logp_as_sampler_logp is not supported when off-policy '
        'learning can happen (i.e. num_train_steps_per_batch > 1).'
    )

  # Start training.
  steps = start_steps
  train_iter = iter(train_set)
  prng_key = jax.random.key(seed=seed)
  stats = {}
  should_early_stop = False
  final_result = {}
  final_result['eval_accuracy_history'] = []
  while steps <= config.num_train_steps and not should_early_stop:
    train_iter_state = train_set.get_state()
    logging.info('train_iter_state=%s', train_iter_state)
    start_time = time.time()
    with jax.profiler.StepTraceAnnotation('sampling'), decoding_mesh_context(
        decoding_mesh_shape, dcn_mesh_shape
    ):
      decoding_params = prepare_params_for_decoding(
          params=state['params'],
          abstract_decoding_params=abstract_decoding_params,
          quant_scheme=config.decoding_quant_scheme,
      )
      prepare_decoding_params_time = time.time() - start_time
      print(
          f'Prepare decoding params time: {prepare_decoding_params_time} secs.'
      )
      helper.add_metric(
          'prepare_decoding_params_time', prepare_decoding_params_time
      )

      num_valid_samples_array = np.zeros(jax.process_count(), dtype=np.int32)
      num_nan_samples_array = np.zeros(jax.process_count(), dtype=np.int32)
      num_truncated_array = np.zeros(jax.process_count(), dtype=np.int32)
      rewarded_completed_batch = collections.defaultdict(list)
      rewarded_pending_batch = []
      in_batch_example_index = 0
      max_num_samples_per_train_batch = (
          config.max_num_samples_per_train_batch or train_batch_size
      )
      while np.sum(num_valid_samples_array) < train_batch_size:
        if (
            in_batch_example_index * config.num_samples_per_example
            >= max_num_samples_per_train_batch
        ):
          # We have sampled this number of samples, no need to do further
          # sampling.
          break
        example_batch = next(train_iter)
        logging.info('example_batch_len=%s', len(example_batch))
        for i, example in enumerate(example_batch):
          # NOTE: We should assume example is immutable to avoid data cache
          # pollution.
          lm_request = list(
              sampling_lib.input_as_chunks(
                  lm_format.format(evaluation.get_messages(example))
              )
          )
          extra_inputs = example.get('extra_inputs', {})
          for extra_input_key in extra_inputs:
            lm_request.append(
                sampling_lib.Chunk(
                    type=sampling_lib.Chunk.Type.ARRAY,
                    content=example['extra_inputs'][extra_input_key],
                )
            )
          example_batch[i] = example | dict(
              lm_request=lm_request,
              steps=steps,
              in_batch_example_index=in_batch_example_index,
              extra_inputs=extra_inputs,
          )
          in_batch_example_index += 1
        print(f'example_batch: {example_batch}')

        prng_key, subkey = jax.random.split(prng_key)
        if tool_executor:
          sampling_outputs = tool_executor.sample_with_tool(
              lm_interface,
              lm_format,
              [e['lm_request'] for e in example_batch],
              prng_key=subkey,
              params=decoding_params,
              prefill_size=config.sampling_prefill_size,
              max_turns=config.max_turns,
              max_tool_response_len=config.sampling_max_tool_response_len,
          )
        else:
          sampling_outputs = lm_interface.generate(
              [e['lm_request'] for e in example_batch],
              prng_key=subkey,
              params=decoding_params,
              prefill_size=config.sampling_prefill_size,
              scoring_inputs=False,
          )

        # At this point, each process only processes a part of the batch, i.e.
        # per-process batch or local batch.
        assert config.batch_size % jax.process_count() == 0
        batch_size_per_process = config.batch_size // jax.process_count()
        process_start_index = jax.process_index() * batch_size_per_process

        def _sharded(xs):
          assert len(xs) == config.batch_size
          return xs[
              process_start_index : process_start_index + batch_size_per_process
          ]

        for example, so_per_prompt in zip(
            _sharded(example_batch), _sharded(sampling_outputs), strict=True
        ):
          assert len(so_per_prompt) == config.num_samples_per_example
          for so in so_per_prompt:
            per_response_example = dict(lm_sampling_output=so) | example
            per_response_example['reward_future'] = evaluation_executor.submit(
                evaluation.evaluate, per_response_example, so.output_text
            )
            rewarded_pending_batch.append(per_response_example)

        must_wait = (
            in_batch_example_index * config.num_samples_per_example
            >= max_num_samples_per_train_batch
        )
        reward_start_time = time.time()

        new_rewarded_pending_batch = []
        for rewarded_per_response in rewarded_pending_batch:
          # At the last batch of sampling, we wait for all evaluations to be
          # collected. Though a non-waiting strategy might be more efficient,
          # it may result in some stability issue when the evaluation servers
          # are down.
          if must_wait or rewarded_per_response['reward_future'].done():
            reward_future = rewarded_per_response.pop('reward_future')
            rewarded_per_response.update(reward_future.result())
            rewarded_completed_batch[
                rewarded_per_response['in_batch_example_index']
            ].append(rewarded_per_response)
          else:
            new_rewarded_pending_batch.append(rewarded_per_response)
        rewarded_pending_batch = new_rewarded_pending_batch

        if must_wait:
          jax.experimental.multihost_utils.sync_global_devices(
              'wait_for_reward'
          )
          reward_time = time.time() - reward_start_time
          logging.info('non_overlapping_reward_time: %s', reward_time)
          helper.add_metric('non_overlapping_reward_time', reward_time)

        for rewarded_per_prompt_batch in rewarded_completed_batch.values():
          corrects = []
          for rewarded_per_response in rewarded_per_prompt_batch:
            # NOTE: reward_result is only available for particular configs.
            if reward_result := rewarded_per_response.get('reward_result'):
              # TODO: Consider limit logging frequency.
              logging.info(
                  'reward=%s, correct=%s, reward_types=%s,'
                  ' COT/cot_generation_length=%s,'
                  ' COT/non_cot_generation_length=%s',
                  reward_result.reward,
                  reward_result.is_correct,
                  reward_result.reward_by_type,
                  reward_result.metrics.get('COT/cot_generation_length'),
                  reward_result.metrics.get('COT/non_cot_generation_length'),
              )
            rewarded_per_response['train_sample_mask'] = True
            if np.isnan(rewarded_per_response['reward']):
              rewarded_per_response['train_sample_mask'] = False
            elif (
                rewarded_per_response['lm_sampling_output'].output_token_ids[-1]
                not in lm_interface.eos_ids
            ):
              # truncated
              if config.filter_truncated:
                rewarded_per_response['train_sample_mask'] = False
            if (
                tool_executor
                and config.filter_throttled
                and rewarded_per_response['lm_sampling_output'].is_throttled
            ):
              rewarded_per_response['train_sample_mask'] = False
            corrects.append(rewarded_per_response['correct'])

        num_truncated = 0
        num_valid_samples = 0
        num_nan_samples = 0
        for rewarded_per_prompt_batch in rewarded_completed_batch.values():
          for rewarded_per_response in rewarded_per_prompt_batch:
            num_valid_samples += rewarded_per_response['train_sample_mask']
            num_nan_samples += np.isnan(rewarded_per_response['reward'])
            num_truncated += (
                rewarded_per_response['lm_sampling_output'].output_token_ids[-1]
                not in lm_interface.eos_ids
            )

        num_valid_samples_array = (
            jax.experimental.multihost_utils.process_allgather(
                num_valid_samples, tiled=True
            )
        )
        num_nan_samples_array = (
            jax.experimental.multihost_utils.process_allgather(
                num_nan_samples, tiled=True
            )
        )
        num_truncated_array = (
            jax.experimental.multihost_utils.process_allgather(
                num_truncated, tiled=True
            )
        )

    del decoding_params
    # TODO: May also want to cancel sub-eval threads?
    for rewarded_per_response in rewarded_pending_batch:
      rewarded_per_response.pop('reward_future').cancel()

    jax.experimental.multihost_utils.sync_global_devices('wait_for_sampling')
    sampling_time = time.time() - start_time
    sampling_time_per_sample = sampling_time / train_batch_size
    print(f'Sampling time total: {sampling_time} sec')
    print(f'Sampling time per sample: {sampling_time_per_sample} sec')
    helper.add_metric('sampling_time', sampling_time)
    helper.add_metric('sampling_time_per_sample', sampling_time_per_sample)

    write_record_start_time = time.time()
    for rewarded_per_prompt_batch in rewarded_completed_batch.values():
      for rewarded_per_response in rewarded_per_prompt_batch:
        helper.write_record(
            dict(
                steps=rewarded_per_response['steps'],
                in_batch_example_index=rewarded_per_response[
                    'in_batch_example_index'
                ],
                reward=rewarded_per_response['reward'],
                correct=rewarded_per_response['correct'],
                lm_sampling_output_text=rewarded_per_response[
                    'lm_sampling_output'
                ].output_text,
                lm_request=rewarded_per_response['lm_request'],
            )
        )
    write_record_time = time.time() - write_record_start_time
    jax.experimental.multihost_utils.sync_global_devices(
        'wait_for_write_record'
    )
    logging.info('write_record_time: %s', write_record_time)
    helper.add_metric('write_record_time', write_record_time)

    stats = compute_stats(rewarded_completed_batch, lm_interface, evaluation)

    # TODO: This may not be correct when log interval > 1.
    for k, v in stats.items():
      helper.add_metric(k, v)

    logging.info(
        'rewarded_completed_batch_len: %s', len(rewarded_completed_batch)
    )

    train_batch = create_train_batch(
        rewarded_completed_batch,
        num_valid_samples=num_valid_samples_array,
        train_batch_size=train_batch_size,
        max_seq_len=train_max_seq_len,
        normalize_reward_method=config.normalize_reward_method,
        ref_params=ref_params,
        compute_logprobs_fn=compute_logprobs_fn,
    )

    helper.add_metric(
        'effective_train_batch_size', np.sum(train_batch['train_sample_masks'])
    )

    for k, v in train_batch.items():
      helper.add_metric(f'train_batch_isnan/{k}', np.sum(np.isnan(v)))

    logging.info('train_batch: %s', jax.tree.map(np.shape, train_batch))

    replay_buffer.extend(train_batch)
    print(f'len(replay_buffer): {len(replay_buffer)}')

    if len(replay_buffer) >= replay_buffer_size:
      train_start_time = time.time()
      for batch in replay_buffer.iterator(train_batch_size, shuffle=True):
        logging.info('batch: %s', jax.tree.map(np.shape, batch))
        steps = int(state['steps'].addressable_data(0))
        print(f'steps: {steps}')
        assert train_iter_state is not None
        helper.save_ckpt(
            state, steps, data={'train_iter_state': train_iter_state}
        )

        # TODO: Merge this process with xm decode eval script.
        if config.use_validation_set and (
            steps % config.validation_eval_interval == 0
            or steps == config.num_train_steps
        ):
          logging.info('Starting eval at step %d.', steps)
          eval_start_time = time.time()
          eval_sampling_params = dataclasses.replace(
              sampling_params, num_samples=1
          )

          with decoding_mesh_context(decoding_mesh_shape, dcn_mesh_shape):
            decoding_params = prepare_params_for_decoding(
                params=state['params'],
                abstract_decoding_params=abstract_decoding_params,
                quant_scheme=config.decoding_quant_scheme,
            )
            prng_key, subkey = jax.random.split(prng_key)
            eval_verdicts = []
            eval_set.set_state(eval_iter_init_state)
            eval_steps = 0
            for eval_batch in eval_set.repeat(config.validation_eval_epochs):
              if (
                  config.validation_num_eval_steps > 0
                  and eval_steps >= config.validation_num_eval_steps
              ):
                break
              eval_prompt_batch = []
              for example in eval_batch:
                lm_request = list(
                    sampling_lib.input_as_chunks(
                        lm_format.format(evaluation.get_messages(example))
                    )
                )
                extra_inputs = example.get('extra_inputs', {})
                for extra_input_key in extra_inputs:
                  lm_request.append(
                      sampling_lib.Chunk(
                          type=sampling_lib.Chunk.Type.ARRAY,
                          content=example['extra_inputs'][extra_input_key],
                      )
                  )
                eval_prompt_batch.append(lm_request)
              if tool_executor:
                eval_sampling_outputs = tool_executor.sample_with_tool(
                    lm_interface,
                    lm_format,
                    eval_prompt_batch,
                    prng_key=subkey,
                    params=decoding_params,
                    sampling_params=eval_sampling_params,
                    prefill_size=config.sampling_prefill_size,
                    max_turns=config.max_turns,
                    max_tool_response_len=config.sampling_max_tool_response_len,
                )
              else:
                eval_sampling_outputs = lm_interface.generate(
                    eval_prompt_batch,
                    prng_key=subkey,
                    params=decoding_params,
                    sampling_params=eval_sampling_params,
                    prefill_size=config.sampling_prefill_size,
                )
              for example, eval_so in zip(
                  eval_batch, eval_sampling_outputs, strict=True
              ):
                eval_verdicts.extend([
                    evaluation.evaluate(example, so.output_text)['correct']
                    for so in eval_so
                ])
              eval_steps += 1
            del decoding_params
            eval_accuracy = np.sum(eval_verdicts) / len(eval_verdicts)
            final_result['eval_accuracy'] = float(eval_accuracy)
            final_result['eval_accuracy_history'].append(eval_accuracy)
            helper.write_scalars(steps, {'eval_accuracy': eval_accuracy})
            should_early_stop = should_early_stop or (
                config.early_stop and
                config.early_stop.should_stop(
                    steps, {'eval_accuracy': eval_accuracy}))
            helper.flush()
            eval_time = time.time() - eval_start_time
            logging.info(
                'Completed eval at step %s, used %d secs.', steps, eval_time
            )

        train_step_start_time = time.time()
        with jax.profiler.StepTraceAnnotation('train', step_num=steps):
          lr = lr_fn(state['steps'])
          loss, state, log_dict = train_one_step_fn(state, batch, lr=lr)

        loss = float(loss.addressable_data(0))
        helper.add_metric('loss', loss)
        train_step_time = time.time() - train_step_start_time
        print(f'train_step_time: {train_step_time} sec')
        helper.add_metric('train_step_time', train_step_time)

        entropy = log_dict['entropy'].addressable_data(0)
        print(f'entropy: {entropy}')
        helper.add_metric('entropy', entropy)

        agg_metrics = helper.get_aggregated_metrics()
        should_early_stop = should_early_stop or (
            config.early_stop and
            config.early_stop.should_stop(
                steps, agg_metrics))
        if helper.should_log_metrics(steps):
          log_start_time = time.time()
          metrics_dict = dict(lr=lr)
          print(f'agg_metrics: {agg_metrics}')
          metrics_dict.update(agg_metrics)
          metrics_dict.update(model_lib.flatten_dict(log_dict))
          helper.write_scalars(steps, metrics_dict)
          helper.flush()
          event_write_time = time.time() - log_start_time
          logging.info('%s secs per writing metrics.', event_write_time)

      training_time = time.time() - train_start_time
      print(f'Training time: {training_time} sec')
      helper.add_metric('training_time', training_time)

    steps = int(state['steps'].addressable_data(0))
    print(f'{steps} train steps passed.')
    total_time = time.time() - start_time
    print(f'Total time: {total_time} sec')
    helper.add_metric('total_time', total_time)
  final_result['train_accuracy'] = float(stats.get('accuracy', 0.0))
  final_result['early_stop'] = should_early_stop
  if should_early_stop: logging.info('Training is early stopped!')
  helper.close(final_result)
  return final_result

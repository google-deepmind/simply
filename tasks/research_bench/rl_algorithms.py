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

"""Modular RL algorithms for the research-bench RL tasks (task-agnostic).

An RL algorithm here is a SELF-CONTAINED, SWAPPABLE MODULE that owns the whole
learning recipe. The custom training loop in `rl_loop.py` drives a fixed
skeleton (sample -> reward -> build batch -> loss -> optimizer update -> FIXED
held-out eval) and delegates each *algorithm* stage to an `RLAlgorithm` object
via these hooks:

  * sampling_params(config)          -> how rollouts are drawn (temperature,
                                        num_samples, decode length, ...).
  * train_reward(example, response)  -> the TRAINING reward (shapeable: partial
                                        credit, format/length bonuses, ...).
  * build_batch(rewarded_batch, ...) -> advantages / normalization / filtering
                                        (returns an
                                        rl_lib.RLTrainingExampleBatch).
  * compute_loss(model, params, batch) -> (loss, metrics), the objective.

The HELD-OUT EVAL is fixed by the loop (`config.validation_evaluation` on the
held-out split) and is NOT an algorithm hook -- it is the scored ground truth
and must not be changed. In particular its decoding is built from the config's
`eval_*` fields alone (`rl_loop._held_out_eval` constructs its own
`SamplingParams`), so `sampling_params()` here only controls exploration.

To add your own algorithm, subclass `RLAlgorithm`, override the stage(s) you
want (typically `compute_loss`, optionally `build_batch` / `train_reward` /
`sampling_params`), and register it with `@functools.partial(
RLAlgorithmRegistry.register, name='my_algo')`. Select it via
`config.rl_algorithm` (a plain config field read by the loop). Implement a NEW
module rather than reusing another algorithm with different arguments.

Config fields these algorithms + the loop actually read: `rl_algorithm`,
`ppo_clip_eps`, `kl_coeff`, `use_ref_params` / `ref_params_dtype`, the
`sampling_*` / `train_max_seq_len` / `num_samples_per_example` sampler fields,
`batch_size` / `train_batch_size` / `grad_accum_steps`, the optimizer / `lr` /
`weight_decay` / clipping fields, and the `eval_*` + `validation_*` eval fields.
Other core `RLExperimentConfig` algorithm switches (ppo_clip_eps_high/low,
policy_ratio_cap, normalize_advantage, max_abs_advantage, use_grpo, gamma,
normalize_reward_method, filter_truncated, num_train_steps_per_batch,
use_policy_logp_as_sampler_logp, ...) are core-`rl`-loop-only: setting them here
has NO effect -- implement the behaviour in your subclass instead.

Two reference algorithms are provided, deliberately as independent
implementations:
  * 'reinforce_baseline' -- REINFORCE with a per-prompt (group) mean baseline.
  * 'simple_grpo'        -- minimal GRPO/PPO-clip with group-normalized
                            advantages + a KL-to-reference penalty.
"""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from simply import model_lib
from simply import rl_lib
from simply.utils import distributions
from simply.utils import masked
from simply.utils import registry


class RLAlgorithmRegistry(registry.RootRegistry):
  """Registry of RL algorithms (see module docstring)."""

  namespace: str = 'RLAlgorithm'  # pyrefly: ignore[bad-override]


class RLAlgorithm:
  """Base RL algorithm: sensible defaults for every stage but the loss.

  Subclasses typically override `compute_loss`; `sampling_params`,
  `train_reward`, and `build_batch` have defaults that reproduce a standard
  on-policy RLVR setup so simple algorithms stay short.
  """

  def __init__(self, config):
    self.config = config

  # ---- stage 1: sampling ----------------------------------------------------
  def sampling_params(self) -> model_lib.SamplingParams:
    """Decoding params for the TRAINING rollouts only.

    These never reach the scored held-out eval: `rl_loop._held_out_eval` builds
    its own `SamplingParams` from scratch out of the config's `eval_*` fields,
    so overriding this (temperature, top_k/top_p, lengths, #samples) changes
    exploration only and cannot move the scored metric.

    Returns:
      The `SamplingParams` used for on-policy rollout generation.
    """
    c = self.config
    return model_lib.SamplingParams(
        temperature=c.sampling_temperature,
        max_decode_steps=c.sampling_max_decode_steps,
        intermediate_decode_steps=c.sampling_intermediate_decode_steps,
        max_seq_len=c.train_max_seq_len + 1,
        max_input_len=c.sampling_max_input_len,
        num_samples=c.num_samples_per_example,
        sort_by=None,
        # Pad decode buffers to a reusable grid to avoid recompiling across
        # varying prompt lengths.
        decode_buffer_multiple=c.sampling_decode_buffer_multiple,
    )

  # ---- stage 2: TRAINING reward (shapeable; NOT the scored eval) -------------
  def train_reward(self, example: Any, response: str) -> dict[str, Any]:
    """Reward used to train on rollouts. Default: the task's configured reward.

    Uses `config.evaluation` -- the task's training-reward Evaluation (e.g. the
    verifiable BFCL AST checker, or a boxed-answer math grader). Override to
    shape the training signal (partial credit, format/length bonuses, curriculum
    weighting, anti-hacking penalties, ...). Whatever you return, the HELD-OUT
    eval (`config.validation_evaluation`) is unchanged and remains the scored
    ground truth.

    Args:
      example: the raw training example (task-specific record).
      response: the model's decoded rollout text.

    Returns:
      A reward dict from the configured Evaluation. It MUST contain 'reward'
      (float; NaN marks the rollout invalid / excluded from training) and
      'correct' (bool-ish; logged as `train_accuracy`) -- both are required by
      `rl_lib.RewardedSample.update_with_evaluation_result`. Any other key is
      free-form and eval-specific.
    """
    return self._reward_eval.evaluate(example, response)

  @functools.cached_property
  def _reward_eval(self):
    # The task's configured TRAINING reward Evaluation. Required: every
    # research-bench RL task sets `config.evaluation` (shaping it, e.g. via
    # partial_credit, takes effect here).
    ev = getattr(self.config, 'evaluation', None)
    if ev is None:
      raise ValueError(
          'config.evaluation (the training-reward Evaluation) must be set for '
          'the default RLAlgorithm.train_reward.'
      )
    return ev

  # ---- stage 3: batch construction + GLOBAL advantage ------------------------
  def advantage(self, reward, example_id, is_valid):
    """Per-sequence advantage from raw rewards, computed over the FULL batch.

    MUST be computed here (over all rollouts) rather than in `compute_loss`,
    because `train_one_step` splits the batch into grad-accum microbatches and
    calls the loss on ONE microbatch at a time -- a group-relative baseline
    computed inside the loss would only see a fraction of each rollout group and
    be wrong. Default: no-op (advantage = reward). Subclasses override.

    Args:
      reward: float np.ndarray [B] of raw per-sequence rewards.
      example_id: int np.ndarray [B] grouping rollouts by prompt.
      is_valid: bool np.ndarray [B] of training-valid rows (exclude padding).

    Returns:
      float np.ndarray [B] of per-sequence advantages.
    """
    del example_id, is_valid  # unused by the base no-op; subclasses use them.
    return reward

  def build_batch(
      self,
      rewarded_batch,
      num_valid_samples,
      train_batch_size: int,
      max_seq_len: int,
      ref_params=None,
      compute_logprobs_fn=None,
  ) -> rl_lib.RLTrainingExampleBatch:
    """Assemble the training batch, baking the GLOBAL advantage into batch.reward.

    We build the batch with RAW rewards (normalize_reward_method=''), then
    overwrite `batch.reward` with the algorithm's per-sequence advantage
    computed over the WHOLE batch. `compute_loss` then reads `batch.reward` as
    the (already group-relative) advantage -- microbatch-/shuffle-invariant.
    Override to filter samples / add a replay buffer. When filtering, do NOT
    drop entries from `rewarded_batch` before calling `create_train_batch`
    unless you also shrink `num_valid_samples` to match (it asserts
    `len(valid rows) == num_valid_samples[process_index]`); the safe idiom is to
    flip `is_valid_for_training` to False on the rollouts you want to exclude
    and recount. Also note `create_train_batch` keeps only the first
    `train_batch_size` valid rows and silently drops the rest.

    Args:
      rewarded_batch: rollouts with per-sequence rewards attached.
      num_valid_samples: number of training-valid rollouts in the batch.
      train_batch_size: target padded training batch size.
      max_seq_len: max sequence length for the packed batch.
      ref_params: optional reference-policy params (for KL / ref logprobs).
      compute_logprobs_fn: optional fn to compute per-token logprobs.

    Returns:
      An `RLTrainingExampleBatch` whose `reward` field holds the global
      per-sequence advantage.
    """
    batch = rl_lib.create_train_batch(
        rewarded_batch,
        num_valid_samples,
        train_batch_size=train_batch_size,
        max_seq_len=max_seq_len,
        normalize_reward_method='',  # keep raw; we compute advantage below.
        ref_params=ref_params,
        compute_logprobs_fn=compute_logprobs_fn,
    )
    adv = self.advantage(
        np.asarray(batch.reward, dtype=np.float32),
        np.asarray(batch.in_batch_example_id),
        np.asarray(batch.is_valid_for_training, dtype=bool),
    )
    return dataclasses.replace(
        batch, reward=jnp.asarray(adv, dtype=jnp.float32)
    )

  # ---- stage 4: loss --------------------------------------------------------
  def compute_loss(self, model, params, batch):
    """The objective: returns (loss, metrics).

    Called by `model_lib.train_one_step` as `custom_loss_fn` -- once per
    grad-accum microbatch, under `jax.value_and_grad(..., has_aux=True)`.

    Args:
      model: the training model (call `model.apply(params, ...)`).
      params: the current parameters (differentiated w.r.t.).
      batch: an `rl_lib.RLTrainingExampleBatch`; `batch.reward` already holds
        the per-sequence ADVANTAGE produced by `build_batch`.

    Returns:
      (loss, metrics). `metrics` MUST contain 'loss_weight': the weight used to
      combine microbatches under gradient accumulation (`jnp.sum(answer_mask)`
      for a token-mean loss); core pops it in `model_lib.train_one_step`, so a
      missing key raises KeyError on the first update. Every other key is
      optional and is logged by the loop ('entropy', 'kl_divergence',
      'advantage/abs_mean', ... in the reference algorithms below); values must
      be scalars to show up in tensorboard.
    """
    raise NotImplementedError


# --------------------------- shared loss primitives ---------------------------
def _current_logprobs(model, params, batch):
  """Per-token log-prob of sampled targets under the CURRENT policy + entropy."""
  answer_mask = batch.answer_mask * jnp.expand_dims(
      batch.is_valid_for_training, axis=-1
  )
  logits, _ = model.apply(
      params,
      batch.input_tokens,
      segment_ids=None,
      segment_positions=None,
      extra_inputs=batch.extra_inputs,
  )
  logits = jnp.astype(logits, jnp.float32)
  dist = distributions.Categorical(logits)
  logpi = masked.masked(dist.log_prob(batch.target_tokens), mask=answer_mask)  # pyrefly: ignore[bad-argument-type]
  return logpi, answer_mask, dist


def _group_mean_std_np(reward, example_id, is_valid):
  """Per-prompt mean/std of reward over VALID rows only, broadcast to [B] (numpy).

  Computed over the full batch (in build_batch, before the grad-accum split), so
  the group baseline sees every rollout in a prompt group. Invalid/padding rows
  are excluded from the stats and get advantage 0.

  Args:
    reward: float array [B] of per-sequence rewards.
    example_id: int array [B] grouping rollouts by prompt.
    is_valid: bool array [B] marking training-valid rows.

  Returns:
    (mean, std, valid): per-row group mean and std (both [B]) and the bool
    valid mask [B].
  """
  reward = np.asarray(reward, dtype=np.float64)
  valid = np.asarray(is_valid, dtype=bool)
  same = (example_id[:, None] == example_id[None, :]) & valid[None, :]
  cnt = np.maximum(same.sum(-1), 1.0)
  mean = (same @ reward) / cnt
  var = (same @ (reward * reward)) / cnt - mean * mean
  std = np.sqrt(np.maximum(var, 0.0))
  return mean, std, valid


@functools.partial(RLAlgorithmRegistry.register, name='reinforce_baseline')
class ReinforceBaseline(RLAlgorithm):
  """REINFORCE with a per-prompt (group) mean baseline.

  advantage = reward - group_mean_reward   (computed over the full batch)
  loss      = -E_tokens[ logpi(a|s) * stop_grad(advantage) ]
  """

  def advantage(self, reward, example_id, is_valid):
    mean, _, valid = _group_mean_std_np(reward, example_id, is_valid)
    return np.where(valid, reward - mean, 0.0)

  def compute_loss(self, model, params, batch):
    logpi, answer_mask, dist = _current_logprobs(model, params, batch)
    # batch.reward already holds the group-relative advantage (build_batch).
    advantage = jax.lax.stop_gradient(batch.reward.astype(jnp.float32))
    per_token = -logpi * advantage[:, None]
    loss = masked.masked_mean(per_token, mask=answer_mask)
    entropy = jax.lax.stop_gradient(
        masked.masked_mean(dist.entropy(), mask=answer_mask)
    )
    metrics = {
        'loss': loss,
        'loss_weight': jnp.sum(answer_mask),  # for grad accumulation.
        'advantage/abs_mean': jnp.mean(jnp.abs(advantage)),
        'entropy': entropy,
    }
    return loss, metrics


@functools.partial(RLAlgorithmRegistry.register, name='simple_grpo')
class SimpleGRPO(RLAlgorithm):
  """Minimal GRPO / PPO-clip objective (self-contained).

  advantage = (reward - group_mean) / (group_std + eps)   (full-batch)
  ratio     = exp(logpi - logpi_old)
  loss      = -E[min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv)] + beta * KL
  KL        = K3 estimator of KL(pi || pi_ref).

  Reads `ppo_clip_eps` and `kl_coeff` from the config.

  NOTE on `logpi_old` (= `batch.logprobs`): those are the SAMPLER's token
  logprobs, i.e. computed under the rollout distribution AFTER temperature /
  top-k / top-p are applied (see `sampling_lib.sample_from_logits`). With the
  baseline `sampling_temperature=1.0` (and no top-k/top-p) that is exactly the
  policy logprob and the ratio starts at 1. If you sample at a different
  temperature (or truncate the distribution), `logpi_old` is NOT log pi_theta
  and the importance ratio is biased on the very first, still on-policy update;
  for strictly on-policy training use `logpi_old = stop_gradient(logpi)`
  instead.
  """

  def advantage(self, reward, example_id, is_valid):
    mean, std, valid = _group_mean_std_np(reward, example_id, is_valid)
    return np.where(valid, (reward - mean) / (std + 1e-4), 0.0)

  def compute_loss(self, model, params, batch):
    clip_eps = float(self.config.ppo_clip_eps)
    kl_coeff = float(self.config.kl_coeff)
    logpi, answer_mask, dist = _current_logprobs(model, params, batch)
    logpi_old = masked.masked(batch.logprobs, mask=answer_mask)
    if batch.ref_logprobs is not None:
      logpi_ref = masked.masked(batch.ref_logprobs, mask=answer_mask)
    else:
      logpi_ref = logpi_old

    # batch.reward already holds the group-normalized advantage (build_batch).
    advantage = jax.lax.stop_gradient(batch.reward.astype(jnp.float32))
    adv_t = advantage[:, None]

    ratio = jnp.exp(logpi - jax.lax.stop_gradient(logpi_old))
    surrogate = jnp.minimum(
        ratio * adv_t, jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
    )
    pg_loss = -masked.masked_mean(surrogate, mask=answer_mask)

    logr = masked.masked(logpi_ref - logpi, mask=answer_mask)
    kl = masked.masked_mean(jnp.expm1(logr) - logr, mask=answer_mask)

    loss = pg_loss + kl_coeff * kl
    entropy = jax.lax.stop_gradient(
        masked.masked_mean(dist.entropy(), mask=answer_mask)
    )
    metrics = {
        'loss': loss,
        'loss_weight': jnp.sum(answer_mask),
        'pg_loss': pg_loss,
        'kl_divergence': jax.lax.stop_gradient(kl),
        'advantage/abs_mean': jnp.mean(jnp.abs(advantage)),
        'policy_ratio/mean': jax.lax.stop_gradient(
            masked.masked_mean(ratio, mask=answer_mask)
        ),
        'entropy': entropy,
    }
    return loss, metrics

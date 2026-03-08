# Automated AI Research with Agents

You can use agents like
[Google Antigravity](https://antigravity.google/),
[Claude Code](https://docs.anthropic.com/en/docs/claude-code), or
[Gemini CLI](https://github.com/google-gemini/gemini-cli) to run
automated research experiments on this codebase. The agent reads the
code, designs experiments, runs training, analyzes results, and
iterates autonomously.

Below is an example prompt you can paste into your agent.
It runs a small research study using the test
config, which trains a tiny model on CPU in under a minute.
Check the section below for running more meaningful tasks on google
cloud (or locally if you have GPUs available).

## Prerequisites

```bash
pip install ".[tfds]"
```

The pretraining example uses IMDB reviews (auto-downloaded by TFDS).
The GCP RL example below needs GSM8K: `python setup/setup_assets.py --datasets-only`.

## CPU example: Optimizer Search

Paste into your agent from the repo root directory:

~~~
You are an AI research agent.
Your task is to design and benchmark new optimizers for training a small transformer language model.

Research question: Can you design an optimizer that outperforms Adam on this training task?

Setup:
- Read CLAUDE.md / AGENTS.md / GEMINI.md for a codebase overview.
- Read simply/utils/optimizers.py carefully.
  Each optimizer is a frozen dataclass registered via @OptimizerRegistry.register,
  with init(params) and apply(state, grad) -> (update, state) methods.
  The existing optimizers are SGD, Adam, Lion, and Muon.
- Read simply/config_lib.py to find the `lm_test` config.
  It trains a tiny transformer on IMDB reviews in ~30 seconds on CPU.
  The optimizer is set via the `optimizer` field on the config.

How to run each experiment:
1. Add your new optimizer class to simply/utils/optimizers.py, following the same pattern as Adam or Lion.
2. Add a new experiment config to simply/config_lib.py that uses your optimizer.
   Follow the existing pattern: register a function that calls the base config
   and overrides the optimizer field with dataclasses.replace().
3. Run: python -m simply.main --experiment_config <your_config_name> --experiment_dir /tmp/<exp_name> --alsologtostderr
4. Read /tmp/<exp_name>/final_result.json for train_loss and train_accuracy.

Experiment plan:
1. First, run the Adam baseline and record the final loss.
2. In multiple research iterations, propose novel optimizers, tune their hyperparameters, and run experiments to compare with the baseline.
   Don't just reimplement known optimizers — invent new update rules,
   combine ideas in original ways, or explore unconventional approaches.
3. In each iteration, you can propose and run 3 experiments, wait for results and then start the next iteration based on the results of finished experiments. Keep running new iterations until you have finished 15 experiments or proposed 10 new optimizers.

After all experiments, write a report with a results table,
a brief description of each optimizer you designed,
and analysis of which ideas worked and why.
Save to /tmp/optimizer_report.md.
~~~

## Scaling to Google Cloud

The same prompts work at real scale on GCP TPUs. Below are concrete
examples — see [docs/gcloud.md](gcloud.md) for full TPU setup,
multi-host training, and preemption handling.

### Optimizer search on C4 pretraining

Use `flops2e17_tfm41m_c4_l2048` (41M params, ~4k steps) or
`flops2e16_tfm15m_c4_l2048` (15M params, ~1.7k steps) as the base
config instead of `lm_test`. These pretrain small transformers on C4
with seq_len=2048 and run on a single v5litepod-1 (1 chip). They
are sized to complete in minutes rather than hours, making them
practical for running many optimizer comparisons.

~~~
You are an AI research agent.
Your task is to design and benchmark new optimizers for pretraining a transformer language model.

Research question: Can you design an optimizer that outperforms Adam for LLM pretraining?

Setup:
- Read CLAUDE.md / AGENTS.md / GEMINI.md for a codebase overview.
- Read docs/gcloud.md for TPU setup, multi-host training, and preemption handling.
- Read simply/utils/optimizers.py carefully.
  Each optimizer is a frozen dataclass registered via @OptimizerRegistry.register,
  with init(params) and apply(state, grad) -> (update, state) methods.
  The existing optimizers are SGD, Adam, Lion, and Muon.
- Read simply/config_lib.py to find the scaling law configs.
  Use `flops2e17_tfm41m_c4_l2048` (41M params, 8 layers, ~4k steps, batch_size=80)
  as your base config. It pretrains a transformer on C4 with seq_len=2048
  and includes validation.
  The optimizer is set via the `optimizer` field on the config.

How to run each experiment:
1. Add your new optimizer class to simply/utils/optimizers.py, following the same pattern as Adam or Lion.
2. Add a new experiment config to simply/config_lib.py that uses your optimizer.
   Follow the existing pattern: register a function that calls flops2e17_tfm41m_c4_l2048()
   and overrides the optimizer field with dataclasses.replace().
3. Run on TPU: python -m simply.main --experiment_config <your_config_name> --experiment_dir gs://<bucket>/experiments/<exp_name> --alsologtostderr
4. Read results from gs://<bucket>/experiments/<exp_name>/final_result.json
   and monitor training curves via TensorBoard:
   tensorboard --logdir gs://<bucket>/experiments/<exp_name>

Experiment plan:
1. First, run the Adam baseline and record the final loss.
2. In multiple research iterations, propose novel optimizers, tune their hyperparameters, and run experiments to compare with the baseline.
   Don't just reimplement known optimizers — invent new update rules,
   combine ideas in original ways, or explore unconventional approaches.
3. In each iteration, you can propose and run 3 experiments, wait for results and then start the next iteration based on the results of finished experiments. Keep running new iterations until you have finished 15 experiments or proposed 10 new optimizers.

After all experiments, write a report with a results table,
a brief description of each optimizer you designed,
and analysis of which ideas worked and why.
Save to /tmp/glcoud_optimizer_report.md.
~~~

### RL algorithm search on Gemma 2B + GSM8K

Use `gemma2_2b_gsm8k_0shot_rl` as the base config instead of
`lm_rl_test`. This post-trains Gemma 2B with GRPO on GSM8K math
problems (batch_size=16, 8 samples per example). Requires a
v5litepod-16 (4 hosts, 16 chips).

~~~
You are an AI research agent.
Your task is to design and benchmark new RL algorithms for post-training Gemma 2B on math reasoning.

Research question: Can you design an RL loss function that improves on GRPO for training Gemma 2B on GSM8K?

Setup:
- Read CLAUDE.md / AGENTS.md / GEMINI.md for a codebase overview.
- Read docs/gcloud.md for TPU setup, multi-host training, and preemption handling.
- Read simply/rl_lib.py carefully, especially the compute_ppo_loss function.
  It computes the RL loss and is the core algorithm —
  it handles advantage estimation, KL penalties, and PPO clipping.
  It is passed as custom_loss_fn to model_lib.train_one_step
  in the rl train loop (run_experiment in rl_lib.py).
  The function signature is:
    compute_ppo_loss(model, params, batch, ...) -> (loss, metrics)
  where batch is an RLTrainingExampleBatch with fields like
  input_tokens, target_tokens, reward, logprobs, ref_logprobs, etc.
- Read simply/config_lib.py to find the `gemma2_2b_gsm8k_0shot_rl` config.
  It post-trains Gemma 2B with GRPO on GSM8K (16 examples, 8 samples each).
  The RL algorithm is configured via fields like
  use_grpo, kl_coeff, ppo_clip_eps, normalize_advantage, gamma, etc.
  Validation evaluates accuracy on the GSM8K test set every 100 steps.

How to run each experiment:
1. Implement your new RL loss function in simply/rl_lib.py
   following the same signature as compute_ppo_loss.
   Wire it into the rl train loop by adding a config field to select between algorithms.
2. Add a new experiment config to simply/config_lib.py that uses your algorithm.
   Follow the existing pattern: register a function that calls gemma2_2b_gsm8k_0shot_rl()
   and overrides the relevant fields with dataclasses.replace().
3. Run on TPU: python -m simply.main --experiment_config <your_config_name> --experiment_dir gs://<bucket>/experiments/<exp_name> --alsologtostderr
4. Read results from gs://<bucket>/experiments/<exp_name>/final_result.json.
   Key metrics: accuracy (fraction correct on GSM8K test), kl_divergence, entropy, policy_ratio.
   Monitor via: tensorboard --logdir gs://<bucket>/experiments/

Experiment plan:
1. First, run the GRPO baseline with the default config for ~500 steps and record the results.
2. In multiple research iterations, propose novel RL algorithm variants, tune their hyperparameters, and run experiments to compare with the baseline.
   Don't just reimplement known algorithms — invent new loss formulations,
   combine ideas in original ways, or explore unconventional approaches.
3. In each iteration, you can propose and run 3 experiments, wait for results and then start the next iteration based on the results of finished experiments. Keep running new iterations until you have finished 15 experiments or proposed 10 new algorithms.

After all experiments, write a report with a results table,
a brief description of each algorithm you designed,
and analysis of which ideas worked and why.
Save to /tmp/gcloud_rl_algorithm_report.md.
~~~

### TPU sizing reference

| Config | Params | TPU type | Hosts | Chips |
|--------|--------|----------|-------|-------|
| `lm_test` | Tiny | CPU or v5litepod-1 | 1 | 1 |
| `flops2e16_tfm15m_c4_l2048` | 15M | v5litepod-1 | 1 | 1 |
| `flops2e17_tfm41m_c4_l2048` | 41M | v5litepod-1 | 1 | 1 |
| `flops1e18_tfm111m_c4_l2048` | 111M | v5litepod-1 | 1 | 1 |
| `flops1e19_tfm338m_c4_l2048` | 338M | v5litepod-8 | 2 | 8 |
| `gemma2_2b_*` | 2.6B | v5litepod-16 | 4 | 16 |

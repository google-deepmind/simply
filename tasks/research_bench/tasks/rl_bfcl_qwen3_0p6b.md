---
task_id: rl_bfcl_qwen3_0p6b
time_limit: 5 hours
hardware: "Cloud TPU `v6e-4` (4 chips, 1 host)."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=rl_bfcl_qwen3_0p6b --experiment_dir=<your gs:// experiment dir>"
---

Turn a fixed PRETRAINED-only checkpoint (Qwen3-0.6B-Base) into a working
tool-use / function-calling model, scored under a FIXED BFCL-live AST eval on
the call-required (relevance) examples.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  Baseline config: `rl_bfcl_qwen3_0p6b` in
    `tasks/research_bench/config_lib.py` —
    naive GRPO from the Qwen3-0.6B-Base PRETRAINED checkpoint, run through the
    `research_bench_rl` loop (see Research Surface). Training reward and the
    scored eval are the SAME ported Berkeley Function-Calling Leaderboard (BFCL)
    AST checker (parse the generated call; check the function name + every
    argument value against the ground-truth allowed-value set).

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
Assets (datasets, tokenizers, checkpoints) are not in the repo: fetch the ones
this task needs, then mirror them so the Cloud TPU VMs can read them
(`../ASSETS.md` lists sizes, times and licence steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=rl_bfcl_qwen3_0p6b
python -m tasks.research_bench.setup.prepare_assets --task=rl_bfcl_qwen3_0p6b \
    --gcs-bucket=gs://<your-bucket>
```

A baseline run, locally on CPU and tiny, to check your wiring before you spend
TPU time:

```bash
python -m tasks.research_bench.main --experiment_config=rl_bfcl_qwen3_0p6b \
    --experiment_dir=/tmp/rl_bfcl_qwen3_0p6b_smoke \
    --config_overlay='{"num_train_steps": 5}' --alsologtostderr
```

## Objective

-   **Metric:** `eval_accuracy` on the FIXED BFCL live eval, restricted to the
    **call-required** (relevance) examples — the 1351 `live_simple` /
    `live_multiple` / `live_parallel` / `live_parallel_multiple` cases that MUST
    emit a correct function call (the `live_irrelevance` abstention category is
    excluded, so "always abstain" scores 0). Per-example AST correctness is
    micro-averaged, pass@1 (greedy), reported in each run's `final_result.json`
    as `eval_accuracy` (value at the LAST checkpoint; full curve in
    `eval_accuracy_history`, per-category breakdown in
    `eval_accuracy_by_category`). HIGHER is better. YOU decide how many
    steps/phases to train; the last checkpoint is scored.
-   **Reference baseline:** an internal reference run (3-seed naive-GRPO baseline of the
    unchanged `rl_bfcl_qwen3_0p6b`), mean `eval_accuracy` ≈ **0.254** (per-seed
    0.281 / 0.242 / 0.241), ~40-45 min/run on 4 chips, with the held-out eval
    at `validation_eval_batch_size=128` (see the launch command below).

> **Where the reference numbers come from.** Every number quoted above as a
> reference was measured with the original internal scaffolding on internal TPU
> hardware. This Cloud port keeps the same code, configs and eval, but some
> assets differ (see `../ASSETS.md`), so absolute values can shift a little.
> `../BASELINES.md` lists what this port measures on Cloud TPU for the same
> baseline configs; prefer those when judging your own runs.


## Research Surface (what you can change)

The ENTIRE training recipe can be changed, via the provided RL scaffolding
(`rl_algorithms.py` — subclass `RLAlgorithm` and register your own;
`simple_grpo` / `reinforce_baseline` are the provided baselines). Concretely,
you may change any training method(s) (RL, supervised, self-improvement, ...),
the optimizer, LR schedule, hyperparameters, curriculum, number of steps/phases,
the prompt format used DURING training, and the training data.

**Data:** you may train ONLY on (a) and (b) below:

-   **(a) Data shipped with the scaffolding:** any data source registered under
    a `simply:` name in the checked-in `simply/data_lib.py` or
    `tasks/research_bench/data_lib.py` (as of the base changelist), restricted to
    its training splits (the BFCL `bfcl_nonlive_train` split is the baseline
    default) — used as provided, or filtered, reformatted, or mixed. Registered
    evaluation sets must not be used for training: anything named `*_test` or
    `*_eval`, and benchmark sets such as MATH500, AIME, GPQA, and MMLU. Data
    sources you add yourself count only if they read exclusively from (a) or
    (b). No other dataset may be used, including files you find on the
    filesystem (e.g. unregistered folders under `simply`'s `DATASETS_DIR`), elsewhere in the
    codebase, or on the internet.
-   **(b) Data you create during this session,** either with programs you write
    (e.g. procedurally generated examples whose ground-truth calls are computed
    by your code, or programmatic transformations of (a)), or by sampling from
    the Qwen3-0.6B-Base model or models you train from it (e.g. rejection
    sampling, self-training).

Both (a) and (b) are subject to two rules: (1) **no eval contamination** — do
not train on, few-shot from, or otherwise expose the BFCL `bfcl_live_eval`
problems, and do not derive any training data from them; (2) **no outputs from
stronger models** — do not use any model more capable than the Qwen3-0.6B-Base
model, including yourself, to write, rewrite, label, filter, score, or judge
training data or rewards. Code and templates you write are fine as long as the
ground-truth calls are computed by your code; training examples or ground-truth
calls that you write out by hand are not. Data in (a) is approved as provided,
regardless of how it was created.

## Eval Setup (FIXED; tampering is disqualified)

Evaluation runs INLINE during the training job (there is no separate eval job):
every `validation_eval_interval` steps the loop evaluates the current policy on
the FIXED BFCL live call-required set (1351 problems, 1 greedy sample each,
pass@1) and scores a response correct only if the parsed function call matches
the ground truth under the BFCL AST checker. The submitted metric is the
accuracy at the LAST checkpoint. The following are FIXED and must not be
modified:

-   the base model — the Qwen3-0.6B-Base **PRETRAINED** checkpoint (not an
    IT/chat checkpoint, not another model);
-   the scored eval — the eval class (`BFCLFunctionCallEvaluation`) and its AST
    checker, the eval dataset (`simply:bfcl_live_eval`, call-required examples),
    and the eval decoding, which is decoupled from training:
    `eval_temperature=0.0` (greedy), `eval_num_samples=1` (pass@1),
    `eval_max_input_len=3072` (long tool schemas are not front-truncated).
    Changing the training sampler must not, and cannot, move the scored eval;
-   the AST checker itself (reward == metric); do not edit `tool_use_eval.py`;
-   the vocab (Qwen3).

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Data rules:** as above — train only on data shipped with the scaffolding
    and on data you create yourself; no BFCL live-eval contamination; and no
    outputs from models more capable than the base (including yourself).
-   **Time limit:** complete all work within the time limit — the final
    submission job must finish before the time limit expires, so plan ahead.

## Submission Candidate Run

The final submission is the GCS experiment dir of a 3-seed sweep run. Launch your best
recipe with a command like the one below:

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=rl_bfcl_qwen3_0p6b \
  --experiment_config=<your-config-name> \
  --experiment_name="rl_bfcl_qwen3_0p6b_<your-run-name>" \
  --bucket=gs://<your-bucket> --seeds=42,43,44 \
  --config-overlay='{"validation_eval_batch_size": 128, "sampling_decode_buffer_multiple": 128, "eval_decode_buffer_multiple": 128}' \
  --tpu-type=v6e-4 --zone=<your-zone> [--spot]
```

The `--config-overlay` above holds the decode-buffer (and eval batch) settings
the reference runs used; `--task` applies them by default, they are spelled out
here so you can see what the baseline ran with.

Check the plan without touching GCP by appending `--dry-run`; watch a running
sweep with `launch_gcp status|logs|collect --bucket=... --experiment_name=...`.

The candidate run is valid only if all 3 runs (seeds 42/43/44) start from the
fixed Qwen3-0.6B-Base PT checkpoint, respect the Data rules, and are scored
under the fixed BFCL-live call-required AST eval, each writing an
`eval_accuracy`.

At the end of your research, write `submission.json` at the root of your
submitted experiment dir (`gs://<bucket>/<experiment_name>/submission.json`)
and repeat it in your final answer, with these keys:

*   `experiment_dir`: The GCS experiment dir of your best submission candidate
    run (e.g.
    `"gs://my-bucket/rl_bfcl_qwen3_0p6b_final"`).
*   `eval_accuracy_all`: The eval accuracy values across the 3 seeds (e.g.
    `"[0.0, 0.0, 0.0]"`).
*   `eval_accuracy_avg`: The mean eval accuracy value across the 3 seeds (e.g.
    `"0.0"`).
*   `summary`: Brief description of the final run training recipe.

## Scoring

-   **Raw metric** = mean `eval_accuracy` (BFCL-live call-required AST
    correctness, last checkpoint) over the 3 seeds (higher is better).

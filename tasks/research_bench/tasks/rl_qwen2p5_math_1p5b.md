---
task_id: rl_qwen2p5_math_1p5b
time_limit: 5 hours
hardware: "Cloud TPU `v6e-4` (4 chips, 1 host)."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=rl_qwen2p5_math_1p5b --experiment_dir=<your gs:// experiment dir>"
---

Improve the RL training recipe for a fixed math-pretrained base checkpoint
(Qwen2.5-Math-1.5B) so it solves HARD competition-math problems (MATH500 levels
4-5), scored under a FIXED avg@8 boxed eval.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  Baseline config: `rl_qwen2p5_math_1p5b` in
    `tasks/research_bench/config_lib.py` —
    naive GRPO from the Qwen2.5-Math-1.5B PRETRAINED checkpoint (R1-Zero-like:
    no KL, no reference policy), run through the `research_bench_rl` loop (see
    Research Surface), and scored on the FIXED MATH500 levels-4-5 eval with
    avg@8.

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
Assets (datasets, tokenizers, checkpoints) are not in the repo: fetch the ones
this task needs, then mirror them so the Cloud TPU VMs can read them
(`../ASSETS.md` lists sizes, times and licence steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=rl_qwen2p5_math_1p5b
python -m tasks.research_bench.setup.prepare_assets --task=rl_qwen2p5_math_1p5b \
    --gcs-bucket=gs://<your-bucket>
```

A baseline run, locally on CPU and tiny, to check your wiring before you spend
TPU time:

```bash
python -m tasks.research_bench.main --experiment_config=rl_qwen2p5_math_1p5b \
    --experiment_dir=/tmp/rl_qwen2p5_math_1p5b_smoke \
    --config_overlay='{"num_train_steps": 5}' --alsologtostderr
```

## Objective

-   **Metric:** `eval_accuracy` = **avg@8** (8 samples/problem at temperature
    0.6, mean correctness) on **MATH500 levels 4-5** (the 262 hardest problems),
    boxed-answer grading with symbolic equivalence. Reported in each run's
    `final_result.json` as `eval_accuracy` (the value at the LAST checkpoint;
    the full curve is in `eval_accuracy_history`). HIGHER is better. YOU decide
    how many steps/phases to train; the last checkpoint is scored.
-   **Reference baseline:** an internal reference run (3-seed shipped-config baseline of
    the unchanged `rl_qwen2p5_math_1p5b`), mean `eval_accuracy` ≈ **0.345**
    (per-seed 0.342 / 0.360 / 0.332), ~**2 h**/run on 4 chips.

> **Where the reference numbers come from.** Every number quoted above as a
> reference was measured with the original internal scaffolding on internal TPU
> hardware. This Cloud port keeps the same code, configs and eval, but some
> assets differ (see `../ASSETS.md`), so absolute values can shift a little.
> `../BASELINES.md` lists what this port measures on Cloud TPU for the same
> baseline configs; prefer those when judging your own runs.


## Research Surface (what you can change)

The ENTIRE training recipe can be changed, via the provided RL scaffolding.
Concretely, you may change the RL algorithm (advantage / loss / clipping / KL /
normalization / importance sampling), reward design and shaping, the optimizer,
LR schedule, hyperparameters, curriculum, number of steps/phases, the prompt
format used DURING training, and the training data.

**Data:** you may train ONLY on (a) and (b) below:

-   **(a) Data shipped with the scaffolding:** any data source registered under
    a `simply:` name in the checked-in `simply/data_lib.py` or
    `tasks/research_bench/data_lib.py` (as of the base changelist), restricted to
    its training splits (DeepScaleR-40K math train, `simply:dsr40k_train`, is
    the baseline default) — used as provided, or filtered, reformatted, or
    mixed. Registered evaluation sets must not be used for training: anything
    named `*_test` or `*_eval`, and benchmark sets such as MATH500, AIME, GPQA,
    and MMLU. Data sources you add yourself count only if they read exclusively
    from (a) or (b). No other dataset may be used, including files you find on
    the filesystem (e.g. unregistered folders under `simply`'s `DATASETS_DIR`), elsewhere
    in the codebase, or on the internet.
-   **(b) Data you create during this session,** either with programs you write
    (e.g. procedurally generated examples whose answers are computed by your
    code, or programmatic transformations of (a)), or by sampling from the
    Qwen2.5-Math-1.5B base model or models you train from it (e.g. rejection
    sampling, self-training).

Both (a) and (b) are subject to two rules: (1) **no contamination of the MATH500
test set** — do not train on, few-shot from, or otherwise expose the MATH500
problems, and do not derive any training data from them; (2) **no outputs from
stronger models** — do not use any model more capable than the Qwen2.5-Math-1.5B
base model, including yourself, to write, rewrite, label, filter, score, or
judge training data or rewards. Code and templates you write are fine as long as
the answers are computed by your code; training examples or answers that you
write out by hand are not. Data in (a) is approved as provided, regardless of
how it was created.

## Eval Setup (FIXED; tampering is disqualified)

Evaluation runs INLINE during the training job. Every `validation_eval_interval`
steps the loop evaluates the current policy on ALL of MATH500 levels 4-5 (262
problems) with **avg@8**: 8 samples per problem at temperature 0.6, scoring the
mean fraction correct (a boxed answer is correct iff it is symbolically
equivalent to the reference). The submitted metric is the value at the LAST
checkpoint. The following are FIXED and must not be modified:

-   the base model — the Qwen2.5-Math-1.5B **PRETRAINED** checkpoint (not the
    Instruct checkpoint, not another model);
-   the scored eval — the eval class (`ZeroShotBoxedInQuestionEvaluation`), the
    eval dataset (`simply:math500_test_l45`, all 262 level-4/5 problems), and
    the eval decoding, which is decoupled from training: `eval_num_samples=8`
    (avg@8) at `eval_temperature=0.6`. Changing the training sampler (including
    its temperature) must not, and cannot, move the scored eval;
-   the vocab (`Qwen2.5`).

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Data rules:** as above — train only on data shipped with the scaffolding
    and on data you create yourself; no MATH500 contamination; and no outputs
    from models more capable than the base (including yourself).
-   **Time limit:** complete all work within the time limit — the final
    submission job must finish before the time limit expires, so plan ahead.

## Submission Candidate Run

The final submission is the GCS experiment dir of a 3-seed sweep run. Launch your best
recipe with a command like the one below:

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=rl_qwen2p5_math_1p5b \
  --experiment_config=<your-config-name> \
  --experiment_name="rl_qwen2p5_math_1p5b_<your-run-name>" \
  --bucket=gs://<your-bucket> --seeds=42,43,44 \
  --config-overlay='{"sampling_decode_buffer_multiple": 128, "eval_decode_buffer_multiple": 128}' \
  --tpu-type=v6e-4 --zone=<your-zone> [--spot]
```

The `--config-overlay` above holds the decode-buffer (and eval batch) settings
the reference runs used; `--task` applies them by default, they are spelled out
here so you can see what the baseline ran with.

Check the plan without touching GCP by appending `--dry-run`; watch a running
sweep with `launch_gcp status|logs|collect --bucket=... --experiment_name=...`.

The candidate run is valid only if all 3 runs (seeds 42/43/44) start from the
fixed Qwen2.5-Math-1.5B PT checkpoint, respect the Data rules, and are scored
under the fixed MATH500 L4-5 avg@8 eval, each writing an `eval_accuracy`.

At the end of your research, write `submission.json` at the root of your
submitted experiment dir (`gs://<bucket>/<experiment_name>/submission.json`)
and repeat it in your final answer, with these keys:

*   `experiment_dir`: The GCS experiment dir of your best submission candidate
    run (e.g.
    `"gs://my-bucket/rl_qwen2p5_math_1p5b_final"`).
*   `eval_accuracy_all`: The eval accuracy values across the 3 seeds (e.g.
    `"[0.0, 0.0, 0.0]"`).
*   `eval_accuracy_avg`: The mean eval accuracy value across the 3 seeds (e.g.
    `"0.0"`).
*   `summary`: Brief description of the final run RL algorithm / reward recipe.

## Scoring

-   **Raw metric** = mean `eval_accuracy` (MATH500 L4-5, avg@8, last checkpoint)
    over the 3 seeds (higher is better).

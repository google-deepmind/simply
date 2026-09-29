---
task_id: pretrain_optimizer_ttt
time_limit: 5 hours
hardware: "Cloud TPU `v6e-4` (4 chips, 1 host)."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=pretrain_optimizer_ttt --experiment_dir=<your gs:// experiment dir>"
---

Design novel optimizers to train a fixed 41M-parameter C4 language model to
target held-out C4 validation losses in *fewer training steps* than a fixed
reference.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  Baseline config: `pretrain_optimizer_ttt` in
    `tasks/research_bench/config_lib.py`
    (41M model, model_dim=256, 8 layers, 100k `vb100864_openmix_v1` vocab, C4,
    H=1200 steps, WSD schedule, tuned-AdamW baseline optimizer).

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
Assets (datasets, tokenizers, checkpoints) are not in the repo: fetch the ones
this task needs, then mirror them so the Cloud TPU VMs can read them
(`../ASSETS.md` lists sizes, times and licence steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=pretrain_optimizer_ttt
python -m tasks.research_bench.setup.prepare_assets --task=pretrain_optimizer_ttt \
    --gcs-bucket=gs://<your-bucket>
```

> **Two substitutions in this port.** (1) C4 comes from the `allenai/c4` `en`
> release repacked into `.bin`/`.idx.npy` shards, not TFDS. (2) The internal
> 100864-piece tokenizer is not public; see `../ASSETS.md` for what this port
> uses. Both move the absolute loss scale, so the four loss targets and the
> scoring anchors are re-derived here — `pretrain_optimizer_ttt_anchor` is the
> registered config that defines them (see `../validator/ANCHORS.md`).

A baseline run, locally on CPU and tiny, to check your wiring before you spend
TPU time:

```bash
python -m tasks.research_bench.main --experiment_config=pretrain_optimizer_ttt \
    --experiment_dir=/tmp/pretrain_optimizer_ttt_smoke \
    --config_overlay='{"num_train_steps": 5}' --alsologtostderr
```

## Objective

-   **Metric:** Time-to-target speedup measured on the held-out C4 validation
    split (HIGHER is better). Each run logs its full held-out C4 val-loss curve
    to `final_result.json` as `validation_loss_curve`; the metric is computed
    from the whole curve, not just the final value.
-   **Reference baseline:** the unchanged `pretrain_optimizer_ttt` baseline,
    3-seed sweep on `v6e-4`, mean speedup = **1.575** (per-seed 1.526 / 1.624 /
    1.574), ~5 min/run. The weak-Adam anchor that defines the targets is the
    registered `pretrain_optimizer_ttt_anchor` config; see `../BASELINES.md`.

> **These reference numbers were measured by this Cloud port**, not inherited
> from the internal suite: the task's loss targets move with the tokenizer, and
> this port ships a different one (see `../ASSETS.md`). `../BASELINES.md` has
> the runs they come from, and `../validator/ANCHORS.md` the arithmetic that
> turns them into the scoring anchors.


## Research Surface (what you can change)

You may invent novel optimizers or implement known ones (e.g., AdEMAMix,
Adam/AdamW, Lion, MARS, Muon and variants, muP-style scaling, PSGD/Kron,
Shampoo, SOAP, Sophia, etc.), as well as the LR value and schedule (e.g., warmup
length, decay shape/onset, WSD/cosine/constant), and any other optimizer
hyperparameters.

## Eval Setup (FIXED; tampering is disqualified)

The following are FIXED and must not be modified: the model (size, architecture,
depth, width), the C4 training datasets, the number of training steps
(**H=1200**), the **batch size (80)**, the sequence length (2048), the vocab
(`vb100864_openmix_v1`, vocab_size 100864), and the C4 validation eval dataset
and settings. The final submission must use the 3-seed (42/43/44) sweep shown
below and write a valid `validation_loss_curve` to `final_result.json`.

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Compute:** There is no separate FLOPs cap. Because the model, step count,
    batch size, and sequence length are all fixed, every submission processes
    the exact same number of tokens (and roughly the same compute) per run. The
    time-to-target metric measures data efficiency at this fixed setup.
-   **Time limit:** complete all work within the time limit — the final
    submission job must finish before the time limit expires, so plan ahead.

## Submission Candidate Run

The final submission is the GCS experiment dir of a 3-seed sweep run. Launch your best
recipe with a command like the one below:

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=pretrain_optimizer_ttt \
  --experiment_config=<your-config-name> \
  --experiment_name="pretrain_optimizer_ttt_<your-run-name>" \
  --bucket=gs://<your-bucket> --seeds=42,43,44 \
  --tpu-type=v6e-4 --zone=<your-zone> [--spot]
```

Check the plan without touching GCP by appending `--dry-run`; watch a running
sweep with `launch_gcp status|logs|collect --bucket=... --experiment_name=...`.

The candidate run is valid only if all 3 runs (seeds 42/43/44) adhere to the
fixed model, data, step budget (H=1200), and evaluation setup.

At the end of your research, write `submission.json` at the root of your
submitted experiment dir (`gs://<bucket>/<experiment_name>/submission.json`)
and repeat it in your final answer, with these keys:

*   `experiment_dir`: The GCS experiment dir of your best submission candidate
    run (e.g.
    `"gs://my-bucket/pretrain_optimizer_ttt_final"`).
*   `speedup_all`: The speedup values across the 3 seeds (e.g. `"[0.0, 0.0, 0.0]"`).
*   `speedup_avg`: The mean speedup value across the 3 seeds (e.g. `"0.0"`).
*   `summary`: Brief description of the final run algorithm / recipe.

## Scoring

-   **Raw metric (time-to-target speedup):** Four loss targets are fixed from a
    frozen weak-Adam anchor curve's held-out C4 validation loss at 40%, 60%,
    80%, and 100% of the 1200-step budget (**5.4272**, **4.9969**, **4.7037**,
    and **4.5901** at steps 480, 720, 960, and 1200 -- the 3-seed mean curve of
    `pretrain_optimizer_ttt_anchor`, re-measured for this port). For each seed, the
    per-target speedup is `anchor_step / first_crossing_step` (via linear
    interpolation on your `validation_loss_curve`). A target you never reach
    contributes speedup 0. The run's raw metric is the average over the four
    targets; the submission's raw metric is the **mean over the 3 seeds**
    (higher is better).

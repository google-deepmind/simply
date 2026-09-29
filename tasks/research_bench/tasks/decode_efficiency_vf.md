---
task_id: decode_efficiency_vf
time_limit: 5 hours
hardware: "Cloud TPU `v6e-8` (8 chips, 1 host) — FIXED, do not change: the metric is wall-clock, so it is only comparable on one TPU type."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=decode_efficiency_vf --experiment_dir=<your gs:// experiment dir>"
---

Make a FIXED thinking LLM decode FASTER without losing accuracy. For
`qwen3_30b_a3b_thinking_2507` on the AIME-2025 math set, minimize the per-sample
generation time (`avg_generation_time`) while keeping accuracy above a fixed
gate. The model, dataset, correctness scorer, seed and hardware are all fixed;
the research surface is the entire decoding pipeline that turns prompts into
answers, with NO training.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  This is an EVAL-ONLY task (page decode-eval; NO training): launch the
    `:xm_eval` binary, not the training `:main`. Each run writes
    `seed_<seed>/final_result.json` with `accuracy`, `avg_generation_time` and
    `seed`.
3.  **Reference baseline:** the unchanged pipeline, i.e. the Submission
    Candidate Run command below with no decode changes.

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
This is an EVAL-ONLY task: the entry point is
`tasks.research_bench.eval_main` (decode-eval), not the training `main`.
Assets are not in the repo: fetch the ones this task needs, then mirror them so
the Cloud TPU VMs can read them (`../ASSETS.md` lists sizes, times and licence
steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=decode_efficiency_vf
python -m tasks.research_bench.setup.prepare_assets --task=decode_efficiency_vf \
    --gcs-bucket=gs://<your-bucket>
```


## Objective

-   **Metric:** `avg_generation_time` (seconds), reported in each run's
    `final_result.json`. It is the total generation wall-clock divided by
    (num_problems × n_repeats) — a throughput measure. **LOWER is better.**
-   **Accuracy gate:** a run counts ONLY if its `accuracy` — micro-averaged over
    the 30 problems × `n_repeats`=4 samples (i.e. avg@4) — is **at or above
    0.75**. Any speedup that drops accuracy below the gate does not count (it
    scores 0).
-   **Reference baseline:** an internal reference run (unchanged pipeline, seed 42 x3),
    `avg_generation_time` ≈ **18.4s** at `accuracy` ≈ **0.78**, ~52 min/run on
    8 chips.

> **Where the reference numbers come from.** Every number quoted above as a
> reference was measured with the original internal scaffolding on internal TPU
> hardware. This Cloud port keeps the same code, configs and eval, but some
> assets differ (see `../ASSETS.md`), so absolute values can shift a little.
> `../BASELINES.md` lists what this port measures on Cloud TPU for the same
> baseline configs; prefer those when judging your own runs.


## Research Surface (what you can change)

You may redesign **anything about the decoding pipeline** — everything about
*how* the fixed model turns each prompt into its answer — as long as the
accuracy gate is met and the Fixed items below are respected. There is no
prescribed list of techniques; the whole decode path is yours.

## Eval Setup (FIXED; tampering is disqualified)

-   **The model:** `qwen3_30b_a3b_thinking_2507` — its weights and architecture.
    No other model, no weight edits, no distillation/replacement.
-   **The task + scorer:** the dataset `simply:aime25`, the evaluation
    `ZeroShotDeepSeekQwenR1CoTBoxed` and its correctness scorer, and the
    accuracy gate. Do not modify or bypass how correctness is judged, and do not
    shrink/alter the problem set.
-   **The averaging protocol:** `n_repeats = 4` (avg@4). It defines both the
    accuracy estimate and the `avg_generation_time` normalization, so it is
    fixed. Every one of the `30 × 4` samples must be generated independently —
    do not reuse or cache one sample's output for its repeats.
-   **The measurement harness:** the timing and result-reporting code in
    `page_decode_eval.py` (where `avg_generation_time` is measured and
    `final_result.json` is written). Optimize the decoding, not the measurement:
    all generation work must happen inside the timed region.
-   **The seed:** `--seeds=42`, so every submission is measured on the same
    decode draw.
-   **The hardware:** an 8-chip slice. Do NOT change
    the TPU type or the chip count. `avg_generation_time` is a throughput metric
    and is not comparable across hardware, so a run on any other platform or
    chip count is invalid.

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Fixed items:** as above — the model, the dataset, the scorer and its gate,
    `n_repeats`, the measurement harness, the seed and the hardware.
-   **Time limit:** complete all work within the time limit — the final
    submission job must finish before the time limit expires, so plan ahead.

## Submission Candidate Run

Launch your best decoder with the command below (any decode changes you like,
same fixed items):

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=decode_efficiency_vf \
  --experiment_name="decode_efficiency_vf_<your-run-name>" \
  --bucket=gs://<your-bucket> --seeds=42 \
  --tpu-type=v6e-8 --zone=<your-zone> [--spot]
```

`--task` supplies the FIXED eval flags for you: `--experiment_config=qwen3_30b_a3b_thinking_2507 --lm_format=QwQChat --evaluation=ZeroShotDeepSeekQwenR1CoTBoxed --datasource_name=simply:aime25 --temperature=0.6 --top_p=0.95 --top_k=20 --batch_size=40 --n_repeats=4 --max_seq_len=32000 --mesh_shape=1,1,2,4`.

Check the plan without touching GCP by appending `--dry-run`; watch a running
sweep with `launch_gcp status|logs|collect --bucket=... --experiment_name=...`.

The candidate run is valid only if it uses the fixed model / dataset /
eval-scorer / `n_repeats`=4 / `--seeds=42` on 8 chips, and its `accuracy` ≥
0.75.

At the end of your research, write `submission.json` at the root of your
submitted experiment dir (`gs://<bucket>/<experiment_name>/submission.json`)
and repeat it in your final answer, with these keys:

*   `experiment_dir`: The GCS experiment dir of your best submission candidate
    run (e.g.
    `"gs://my-bucket/decode_efficiency_vf_final"`).
*   `avg_generation_time`: The average generation time in seconds (e.g.
    `"0.0"`).
*   `accuracy`: The micro-averaged accuracy value (e.g. `"0.0"`).
*   `summary`: Brief description of the decode optimizations you applied.

## Scoring

-   **Raw metric** = `avg_generation_time` of the submitted run (lower is
    better), valid ONLY if that run's `accuracy` ≥ 0.75; otherwise the
    submission scores 0.

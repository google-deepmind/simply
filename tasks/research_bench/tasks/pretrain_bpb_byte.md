---
task_id: pretrain_bpb_byte
time_limit: 5 hours
hardware: "Cloud TPU `v6e-4` (4 chips, 1 host)."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=pretrain_bpb_byte --experiment_dir=<your gs:// experiment dir>"
---

Minimize C4 validation bits-per-byte (`validation_bpb`) for a byte-level LLM
under a fixed training-compute budget.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  Baseline config: `pretrain_bpb_byte` in
    `tasks/research_bench/config_lib.py`
    (byte vocab `byte256`, C4, ~1.9M params).

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
Assets (datasets, tokenizers, checkpoints) are not in the repo: fetch the ones
this task needs, then mirror them so the Cloud TPU VMs can read them
(`../ASSETS.md` lists sizes, times and licence steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=pretrain_bpb_byte
python -m tasks.research_bench.setup.prepare_assets --task=pretrain_bpb_byte \
    --gcs-bucket=gs://<your-bucket>
```

> **C4 in this port.** TFDS has no public C4, so the corpus is the `allenai/c4`
> `en` release repacked into `.bin`/`.idx.npy` shards (`setup/build_c4.py`,
> read by `data_lib.C4FileSource`). Same documents, deterministic order, but
> not TFDS's shard order — absolute bpb is comparable *within* this port only.

Estimate the FLOPs of a recipe before you launch it:

```bash
python -c "from tasks.research_bench import tpu_mfu, config_lib; \
print(tpu_mfu.training_flops_xla_1cpu(\
config_lib.ExperimentConfigRegistry.get_config('<your-config-name>')))"
```

A baseline run, locally on CPU and tiny, to check your wiring before you spend
TPU time:

```bash
python -m tasks.research_bench.main --experiment_config=pretrain_bpb_byte \
    --experiment_dir=/tmp/pretrain_bpb_byte_smoke \
    --config_overlay='{"num_train_steps": 5}' --alsologtostderr
```

## Objective

-   **Metric:** `validation_bpb` on the C4 validation split (LOWER is better),
    reported as `validation_bpb` in each run's `final_result.json`.
-   **Reference baseline:** an internal reference run (3-seed sweep), mean
    `validation_bpb` = **1.8432**, `training_flops_xla` ≈ **6.120e15**, ~12-18
    min/run on 2 chips.

> **Where the reference numbers come from.** Every number quoted above as a
> reference was measured with the original internal scaffolding on internal TPU
> hardware. This Cloud port keeps the same code, configs and eval, but some
> assets differ (see `../ASSETS.md`), so absolute values can shift a little.
> `../BASELINES.md` lists what this port measures on Cloud TPU for the same
> baseline configs; prefer those when judging your own runs.


## Research Surface (what you can change)

You are free to change the training recipe (architecture, optimizer/algorithm,
hyperparameters, data mixture, model size, steps, etc.).

You may source, mix, and reformat ANY training data. But exposing the test set
during training, or any other kind of test data contamination, is strictly
forbidden.

## Eval Setup (FIXED; tampering is disqualified)

The following are FIXED and must not be modified: the C4 validation dataset, the
`validation_bpb` metric, the FLOPs accounting, and the vocab (`byte256`,
vocab_size 259). The final submission must use the 3-seed (42/43/44) sweep shown
below.

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Compute budget:** `training_flops_xla` ≤ **9.180e15** (= 1.5 × baseline).
    Each run writes its `training_flops_xla` (global XLA HLO FLOPs) to
    `final_result.json`; the validator checks it. Estimate cheaply before
    launching with `tpu_mfu.training_flops_xla_1cpu(config)` -- use it for
    reference and *relative* comparison only; the estimate differs from the
    actual value computed in the accelerator training job.
-   **Faithful compute accounting (the FLOP budget must reflect the real
    compute):**
    -   **No `scan` anywhere in the train step (`use_scan=False`).** XLA's FLOP
        cost model counts a `jax.lax.scan` body only once regardless of its trip
        count, so any scanned repetition understates its compute. Do not use
        `scan` (or an equivalent loop construct) to stack layers or to repeat
        any other part of the train step; the baseline sets `use_scan=False` and
        a submission that re-enables it, or that introduces its own scan/loop,
        is rejected.
    -   **Standard XLA ops only (`use_flash_attention=False`; no custom
        kernels).** Express the training step in standard XLA operations.
        Pallas/Mosaic custom kernels (e.g. flash/splash attention, or
        hand-written matmul/attention kernels) appear to the FLOP cost model as
        opaque `custom-call`s with zero counted FLOPs, so routing compute
        through them understates `training_flops_xla`; such runs are rejected.
-   **Single training run:** the submitted run must train from scratch within
    its own `training_flops_xla` budget. Multi-stage training is not allowed: do
    not warm-start the submission from a checkpoint you trained earlier
    (`init_ckpt_dir` / `teacher_ckpt_dir`), since the reported FLOPs only cover
    the submitted run.
-   **Time limit:** complete all work within the time limit — the final
    submission job must finish before the time limit expires, so plan ahead.

## Submission Candidate Run

The final submission is the GCS experiment dir of a 3-seed sweep run. Launch your best
recipe with a command like the one below:

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=pretrain_bpb_byte \
  --experiment_config=<your-config-name> \
  --experiment_name="pretrain_bpb_byte_<your-run-name>" \
  --bucket=gs://<your-bucket> --seeds=42,43,44 \
  --tpu-type=v6e-4 --zone=<your-zone> [--spot]
```

Check the plan without touching GCP by appending `--dry-run`; watch a running
sweep with `launch_gcp status|logs|collect --bucket=... --experiment_name=...`.

The candidate run is valid only if all 3 runs (seeds 42/43/44) are within budget
and use the fixed eval.

At the end of your research, write `submission.json` at the root of your
submitted experiment dir (`gs://<bucket>/<experiment_name>/submission.json`)
and repeat it in your final answer, with these keys:

*   `experiment_dir`: The GCS experiment dir of your best submission candidate
    run (e.g.
    `"gs://my-bucket/pretrain_bpb_byte_final"`).
*   `training_flops_xla_all`: The training FLOPs values across the 3 seeds (e.g.
    `"[0.0, 0.0, 0.0]"`).
*   `validation_bpb_all`: The validation bpb values across the 3 seeds (e.g.
    `"[0.0, 0.0, 0.0]"`).
*   `validation_bpb_avg`: The mean validation bpb value across the 3 seeds (e.g.
    `"0.0"`).
*   `summary`: Brief description of the final run algorithm / recipe.

## Scoring

-   **Raw metric** = mean `validation_bpb` across the 3 seeds (lower is better).

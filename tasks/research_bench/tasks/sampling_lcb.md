---
task_id: sampling_lcb
time_limit: 10 hours
hardware: "Cloud TPU `v6e-4` (4 chips, 1 host)."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=sampling_lcb --experiment_dir=<your gs:// experiment dir>"
---

Design a test-time SAMPLING / DECODING algorithm that maximizes coding accuracy
for a FIXED model (Qwen3-4B in non-thinking mode) on LiveCodeBench v5, with NO
training. The model and the correctness scorer are fixed; the research surface
is the entire decoding process that turns the model into one final program per
problem.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  Baseline eval: `LcbBaseline` in
    `tasks/research_bench/lcb_sampling_lib.py`
    — one sample per problem, graded on the full LiveCodeBench test suite. Data
    source `simply_json:livecodebench_v5` (167 problems, tests carried per
    example); grading runs in the gVisor code-execution sandbox. This is an
    EVAL-ONLY task: launch the `:xm_eval` binary (page decode-eval), not the
    training `:main`.

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
This is an EVAL-ONLY task: the entry point is
`tasks.research_bench.eval_main` (decode-eval), not the training `main`.
Assets are not in the repo: fetch the ones this task needs, then mirror them so
the Cloud TPU VMs can read them (`../ASSETS.md` lists sizes, times and licence
steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=sampling_lcb
python -m tasks.research_bench.setup.prepare_assets --task=sampling_lcb \
    --gcs-bucket=gs://<your-bucket>
```


## Objective

-   **Metric:** `accuracy` (pass@1) on LiveCodeBench v5 (167 problems), reported
    in each run's `final_result.json` as `accuracy`. A problem is correct iff
    the ONE final program you emit for it passes ALL (public + private) tests
    via the fixed executor. HIGHER is better.
-   **Reference baseline:** `LcbBaseline` (single sample) on 4 chips,
    expected 3-seed mean **~0.376** pooled over repeats (the fixed `qwen3_4b` +
    `QwenV2Chat` non-thinking base, full 167-problem LiveCodeBench v5);
    an internal reference run is one such run, whose own draw was 0.371. ~**25-30
    min**/run and ~30 min for the 3-seed sweep.
-   **This metric is not exactly reproducible.** The decoder batches
    asynchronously, so repeating an identical run at the same seeds can shift
    the score by a few problems. The scored value is the 3-seed mean; judge a
    change by whether it is larger than that run-to-run movement.

> **Where the reference numbers come from.** Every number quoted above as a
> reference was measured with the original internal scaffolding on internal TPU
> hardware. This Cloud port keeps the same code, configs and eval, but some
> assets differ (see `../ASSETS.md`), so absolute values can shift a little.
> `../BASELINES.md` lists what this port measures on Cloud TPU for the same
> baseline configs; prefer those when judging your own runs.


## Research Surface (what you can change)

Everything about HOW you produce the final program. Subclass `LcbEval` and
override ONLY `decode(example, model_fn, ctx) -> final_code`: draw as many
samples as you like at any temperature / top_p / top_k; prompt however you like.
`model_fn(messages)` decodes a response; `ctx.run_public_tests(code)` runs a
candidate against the PUBLIC tests only and returns per-test pass/fail — a
legitimate decoding signal you MAY use.

There is NO explicit sample / compute cap; the wall-clock limit is the governor
(scale test-time compute too far and the submission run will not finish in
time).

## Eval Setup (FIXED; tampering is disqualified)

The correctness scorer is the single source of truth and must not be
reimplemented or bypassed:

-   `lcb_sampling_lib._LcbGrader.score_final_answer` grades your final program
    on the FULL LiveCodeBench test suite via the UNCHANGED executor
    (`_test_livecodebench_problem`, tests_used=ALL). The framework calls it on
    the answer you return; you never call it yourself.
-   The base model (`qwen3_4b`), the vocab / `QwenV2Chat` lm_format, the dataset
    (`simply_json:livecodebench_v5`), and the `LcbEval.evaluate_async` pipeline
    (you decode ONE answer → framework grades ALL tests → writes `correct`) are
    FIXED.

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Fixed:** the base model, the dataset, and the correctness scorer
    (full-suite grading). Do not modify or bypass the grader, the executor, or
    the ALL-tests final grade.
-   **No private-test contamination.** The private tests exist ONLY for the
    final grade. Do NOT use them as a decoding signal — do not read
    `example['auxiliary']['private_test_cases']`, and do not call the full-suite
    grader during decoding. Using the PUBLIC tests (`ctx.run_public_tests`) is
    allowed.
-   **Time limit:** complete all work within the time limit — the final
    submission run must finish before the limit; plan ahead.

## Submission Candidate Run

The final submission is the GCS experiment dir of a 3-seed sweep (seeds
MUST be 42/43/44, one run each). Launch your best decoder with a command like:

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=sampling_lcb \
  --experiment_name="sampling_lcb_<your-run-name>" \
  --bucket=gs://<your-bucket> --seeds=42,43,44 \
  --extra-flag=evaluation=<YourEvaluationSubclass> \
  --tpu-type=v6e-4 --zone=<your-zone> [--spot]
```

`--task` supplies the FIXED eval flags for you: `--experiment_config=qwen3_4b --lm_format=QwenV2Chat --datasource_name=simply_json:livecodebench_v5 --top_p=0.95 --temperature=0.6 --top_k=20 --batch_size=48 --n_repeats=1 --max_seq_len=12000 --mesh_shape=1,1,4 --num_eval_threads=96`.

Check the plan without touching GCP by appending `--dry-run`; watch a running
sweep with `launch_gcp status|logs|collect --bucket=... --experiment_name=...`.

The launcher creates ONE experiment dir with one run per seed (each writes
`seed_<seed>/final_result.json` with its `accuracy` and `seed`). Valid only if all
three seeds (42/43/44) are present, use the fixed model, and are graded by the
fixed full-suite scorer.

At the end of your research, write `submission.json` at the root of your
submitted experiment dir (`gs://<bucket>/<experiment_name>/submission.json`)
and repeat it in your final answer, with these keys:

*   `experiment_dir`: The GCS experiment dir of your best submission candidate
    run (e.g.
    `"gs://my-bucket/sampling_lcb_final"`).
*   `accuracy_all`: The accuracy values across the 3 seeds (e.g. `"[0.0, 0.0, 0.0]"`).
*   `accuracy_avg`: The mean accuracy value across the 3 seeds (e.g. `"0.0"`).
*   `summary`: Brief description of the final decoding algorithm.

## Scoring

-   **Raw metric** = mean `accuracy` (full-suite pass@1) over the 3 seeds
    (higher is better).

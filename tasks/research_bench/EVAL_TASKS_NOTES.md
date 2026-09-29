# Eval-only tasks: `sampling_lcb` + `decode_efficiency_vf`

Port notes for the two tasks that run the page decode-eval path
(`eval_main.py` -> `simply.eval.page_decode_eval.main`), i.e. no training:
load a FIXED checkpoint, decode the eval set, write `final_result.json`.

| | `sampling_lcb` | `decode_efficiency_vf` |
|---|---|---|
| model (FIXED) | `qwen3_4b`, `QwenV2Chat` (non-thinking) | `qwen3_30b_a3b_thinking_2507`, `QwQChat` |
| data (FIXED) | `simply_json:livecodebench_v5`, 167 problems | `simply:aime25`, 30 problems x `n_repeats=4` |
| evaluation | `LcbBaseline` (or your `LcbEval` subclass) | `ZeroShotDeepSeekQwenR1CoTBoxed` (FIXED) |
| metric | `accuracy` (pass@1, full test suite), higher better | `avg_generation_time` (s), lower better, gated on `accuracy >= 0.75` |
| GCP hardware | v6e-4, `--mesh_shape=1,1,4` | v6e-8, `--mesh_shape=1,1,8` |
| research surface | `LcbEval.decode(example, model_fn, ctx)` | the whole decode pipeline |

---

## !! This code EXECUTES UNTRUSTED MODEL-GENERATED PROGRAMS !!

`sampling_lcb` is graded by running the model's Python on your machine
(`code_exec_lib.py`). The internal task ran it inside **gVisor** (a userspace
kernel that interposes on the whole syscall surface). There is no gVisor in the
OSS path, so the replacement is **weaker**:

| launcher | picked when | network | host filesystem | escapes the kernel? |
|---|---|---|---|---|
| `bwrap` (bubblewrap) | installed + unprivileged user namespaces work | **denied** (netns) | **read-only**, private `/tmp`, `/proc`, PID ns | not prevented |
| `unshare` (util-linux) | bubblewrap missing | **denied** (netns) | **full host FS, writable** | not prevented |
| `none` | nothing else probes OK | **reachable** | **full host FS, writable** | not prevented |

Always enforced, on every launcher: hard wall-clock timeout (10 s/test, whole
process group SIGKILLed), `RLIMIT_CPU`, `RLIMIT_AS` (16 GiB), `RLIMIT_FSIZE`
(caps runaway stdout, which is redirected to a file), `RLIMIT_CORE=0`, a fresh
throw-away cwd, `HOME`/`TMPDIR` pointed into it, a scrubbed environment
(`PATH`, no credentials, no `PYTHON*`), `python -I`, and a soft+hard rlimit pair
so the program cannot raise its own limits.

**Consequences you must accept before running it:**

* Run graded LCB evals on a **disposable VM** (the Cloud TPU worker), never on a
  workstation holding credentials. A kernel exploit, or simply `unshare`/`none`
  plus `rm -rf ~`, is not stopped.
* Install bubblewrap on the TPU VM (`sudo apt-get install -y bubblewrap`) --
  it is the only launcher that makes the host filesystem read-only. Verify from
  the log: the first line of every eval run is
  `[code_exec] sandbox=... ; available=[...]`.
* Pin the launcher with `SIMPLY_CODE_EXEC_SANDBOX=bwrap` if you want a run to
  **fail loudly** rather than silently degrade to `none`.
* `decode_efficiency_vf` does NOT execute model code (its scorer is a boxed-
  answer string match), so this warning applies to `sampling_lcb` only.

---

## Files

| file | what it is |
|---|---|
| `eval_main.py` | entry point; imports the LCB modules for registration, prints the sandbox report, defers to the UNCHANGED `page_decode_eval.main` |
| `lcb_data_lib.py` | `simply_json:livecodebench_v5` data source (167 problems, tests carried per example) |
| `lcb_sampling_lib.py` | the FIXED grading harness (`_LcbGrader`, `LcbEval`) + the `LcbBaseline` decoder |
| `code_exec_lib.py` | the sandboxed program runner that replaces gVisor/CEO |
| `tests/lcb_*_test.py`, `tests/code_exec_lib_test.py` | CPU pytest suite, no accelerator, no network |

## How to run

```bash
# sampling_lcb (v6e-4), one process per seed; seeds 42/43/44 for a submission.
python -m tasks.research_bench.eval_main \
    --experiment_config=qwen3_4b --lm_format=QwenV2Chat \
    --evaluation=LcbBaseline \
    --datasource_name=simply_json:livecodebench_v5 \
    --temperature=0.6 --top_p=0.95 --top_k=20 \
    --batch_size=48 --n_repeats=1 --max_seq_len=12000 \
    --mesh_shape=1,1,4 --num_eval_threads=96 --seed=42 \
    --experiment_dir=gs://<bucket>/<experiment_name>/seed_42

# decode_efficiency_vf (v6e-8), seed 42 only.
python -m tasks.research_bench.eval_main \
    --experiment_config=qwen3_30b_a3b_thinking_2507 --lm_format=QwQChat \
    --evaluation=ZeroShotDeepSeekQwenR1CoTBoxed --datasource_name=simply:aime25 \
    --temperature=0.6 --top_p=0.95 --top_k=20 \
    --batch_size=40 --n_repeats=4 --max_seq_len=32000 \
    --mesh_shape=1,1,8 --seed=42 \
    --experiment_dir=gs://<bucket>/<experiment_name>/seed_42
```

`final_result.json` (written by the unchanged harness) is
`{accuracy, correct, total, avg_generation_time, seed}`.

Every flag the task prompts quote already exists: `--experiment_config
--lm_format --mesh_shape --batch_size --max_seq_len --temperature --top_p
--top_k --page_size --enable_prefix_caching --ffn_weight_quant
--kv_cache_quant` come from `simply/serving/common_flags.py`; `--evaluation
--datasource_name --experiment_dir --n_repeats --num_eval_threads --seed
--save_every_n --save_full_info --data_shard_{count,index}` from
`simply/eval/page_decode_eval.py`. `eval_main.py` adds none of its own.

**Writing a decoder**: subclass `LcbEval`, override ONLY `decode`, and register
it with `@evaluation_lib.EvaluationRegistry.register` **in a module
`eval_main.py` imports** (simplest: next to `LcbBaseline` in
`lcb_sampling_lib.py`); then pass `--evaluation=<YourClass>`. Inside `decode`,
`ctx.run_public_tests(code)` / `await ctx.run_public_tests_async(code)` /
`await ctx.run_public_tests_many(codes)` are the sanctioned public-test signal.

**Smoke run on a fraction of the set** (same registered data source, so no
separate "small" dataset can be mistaken for the graded one): add
`--data_shard_count=20 --data_shard_index=0` -> 9 of the 167 problems. A
submission run must have neither flag (`total` in `final_result.json` must be
167).

## Deviations from the internal scaffolding (and why)

1. **No work-unit subdir.** The internal `eval.py` appended a work-unit dir
   because one XID held all seeds. Here the launcher gives each seed its own
   `--experiment_dir` (`.../seed_<seed>/`), so `final_result.json` lands
   directly in it. Validators must look for `seed_<seed>/final_result.json`,
   not `wu_<wid>/final_result.json`.
2. **No internal-sandbox wiring** (`_wire_gvisor_loader` is gone) -- replaced by
   `code_exec_lib`, see the warning above.
3. **LiveCodeBench loader.** The internal scaffolding pulled the problems
   through an internal module from internal storage. Here they are staged to disk by
   `setup/build_datasets.py::build_livecodebench()`. The source file is
   **byte-identical** to the internal one -- HF `livecodebench/code_generation_lite`
   `test5.jsonl`, 557,699,297 bytes,
   sha256 `7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214`,
   167 lines -- so this is the same benchmark, not a substitute. Staged shape
   (agreed with the assets owner):

   ```
   $SIMPLY_DATASETS/livecodebench/livecodebench_v5.jsonl   # 167 lines, upstream order
   {question_id, question_title, question_content, platform, contest_id,
    contest_date, difficulty, starter_code, func_name,
    public_test_cases,      # JSON string: [{input, output, testtype}, ...]
    private_test_cases}     # JSON string, decoded from base64+zlib+pickle
   ```

   Counts asserted at build AND load time: 167 problems, 441 public tests,
   6099 private tests, first `question_id == 'abc374_c'`. Decoded private tests
   are ~1.1 GiB, hence JSONL + an mmap'd line index (nothing is unpickled at
   load time; `pickle` appears only in the staging script).
4. **The executor is re-implemented, not re-invented.** The CMS
   `_test_livecodebench_problem` is internal, so `lcb_sampling_lib` reproduces
   its behaviour: the same 30-line import preamble +
   `sys.setrecursionlimit(6*10**5)`, the same `stdin` and `functional` drivers
   (including the `Solution().<func>(args)` call, the `json.loads` of the
   expected value with the `repr()` fix for quoted strings, and exit code 24
   for a wrong answer), the same verdict taxonomy
   (timeout / syntax / wrong-output / execution-error), the same
   whitespace-insensitive, 1e-6-float-tolerant line comparison
   (`approx_float_matching=True`, as the internal grader passed), the same
   "empty stdout is a failure" rule, the same stop-at-first-failure loop, and
   the same response normalisation (`remove_thoughts` ->
   `extract_last_code_block` -> `add_function_call_if_missing`).
   Residual differences that could move a verdict on a handful of problems:
   the interpreter is the run's own Python 3.12 venv (the internal grader used a
   prebuilt docker image), and the sandbox launch overhead is ~60 ms
   instead of a container start.
5. **`<think>`/`</think>` added to the thought-stripping pairs.** The internal
   list only had Gemini's control tokens; the Qwen chat models emit
   `<think>...</think>`, which must not end up inside the graded program.
   (`sampling_lcb` runs Qwen3-4B in non-thinking mode, so this is a safety net,
   not a behaviour change for the baseline.)

## Anchors and reference numbers

* `sampling_lcb`: score anchors `a=0.3896 -> 0`, `b=0.7056 -> 1`; internal
  `LcbBaseline` reference ~**0.376** (3-seed mean). `accuracy` is a property of
  the model + decoder + grader only, so these carry over to Cloud TPU **as long
  as the checkpoint is the same Qwen3-4B**. Re-measure the baseline once on
  v6e-4 to confirm; a large gap means a checkpoint/format problem, not a
  decoding one.
* `decode_efficiency_vf`: **the metric is hardware-calibrated and its anchors
  do NOT carry over.** `avg_generation_time` is wall-clock per sample on a
  specific accelerator; the internal anchors (`a=16.2718 -> 0`,
  `b=5.1540 -> 1`, reference ~18.4 s at accuracy ~0.78) were measured on
  an internal 8-chip slice. On v6e-8 the unchanged pipeline will produce a
  different number. **Someone must run the unchanged baseline on v6e-8 and
  re-anchor a and b from it** (e.g. keep the internal ratio b/a ~ 0.317, or set
  b to the best measured optimisation), and the task prompt must say the
  anchors were re-measured on Cloud TPU. The accuracy gate (0.75) is hardware
  independent and stays.

## What the task prompts must warn the agent about

1. **Untrusted code execution** (sampling_lcb): the grader runs model output;
   the run must happen on the disposable TPU VM, and the first log line reports
   the isolation actually in force.
2. **Submission layout**: `gs://<bucket>/<experiment_name>/seed_<seed>/final_result.json`,
   one process per seed (42/43/44 for sampling_lcb, 42 only for
   decode_efficiency_vf) -- not an XID, and no `wu_<wid>/`.
3. **No private-test contamination**: `ctx.run_public_tests(code)` is allowed;
   the private tests are not merely forbidden, they are **absent** from the
   example `decode` receives, and `DecodeContext` exposes no full-suite grader.
   Reaching around that (re-reading the dataset file from `decode`) is
   disqualifying.
4. **Wall clock, not samples, is the governor** (sampling_lcb): there is no
   sample cap, but every extra candidate costs decode time *and* sandbox CPU on
   the same host. A best-of-N decoder that public-tests every candidate can
   saturate the host CPU and slow decoding; cap the sandbox pool with
   `SIMPLY_CODE_EXEC_WORKERS` if so.
5. **decode_efficiency_vf**: `n_repeats=4`, `--seeds=42`, the fixed chip count
   and the measurement harness in `page_decode_eval.py` are all FIXED; all
   generation work must stay inside the timed region, and each of the 30x4
   samples must be generated independently (no reusing one sample for its
   repeats). Note that prefix caching (`--enable_prefix_caching`) caches the
   *prompt* KV, which is a legitimate decode optimisation, but caching a
   *response* across repeats is not.
6. **Baseline first**: `LcbBaseline` / the unchanged decode pipeline should be
   run once before optimising, to confirm the checkpoint and the environment
   reproduce the reference number.

## Verification performed (CPU, no accelerator)

* `pytest tasks/research_bench/tests/{code_exec_lib,lcb_data_lib,lcb_sampling_lib,lcb_eval_main}_test.py`
  -> 51 passed in ~16 s. Covers: timeouts/crashes/OOM/output floods graded as
  failures not exceptions; no network and read-only host FS under bubblewrap;
  correct program accepted and wrong program rejected on the FULL suite (both
  `stdin` and `functional` problems); `run_public_tests` per-test results;
  private tests absent from what `decode` sees; the loader's shape, prompt,
  slicing and count checks on a synthetic fixture and on the real 167-problem
  file.
* Real LCB data, real `eval_main` -> `page_decode_eval` pipeline, stub model
  over the first 8 LiveCodeBench v5 problems with hand-written solutions for 3
  of them:
  `{"accuracy": 0.375, "correct": 3, "total": 8, "avg_generation_time": 5.44, "seed": 42}`.
* Sandbox throughput on a 128-vCPU host: 441 public-test executions in 2.7 s
  and 167 full-suite grades of a failing program in 3.2 s (32 threads). Grading
  is not the bottleneck; decoding is.
* Same pipeline on the decode_efficiency_vf side (`simply:aime25`,
  `ZeroShotDeepSeekQwenR1CoTBoxed`, `n_repeats=4`, stub model answering 4/5
  correctly): `{"accuracy": 0.8, "correct": 96, "total": 120, ...}` -- i.e. the
  30x4 micro-average and the gate quantity are computed as the task defines
  them.

Still to do on real hardware (owner: the TPU launcher): one `LcbBaseline` run
on v6e-4 (~25-30 min expected) and one unchanged `decode_efficiency_vf` run on
v6e-8 to re-anchor `avg_generation_time`.

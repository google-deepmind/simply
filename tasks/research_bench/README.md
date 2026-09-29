# Simply Research Bench (Cloud port)

Eleven research tasks that measure how well an autonomous agent can do real LLM
research: pretraining under a compute cap, optimizer design, RL post-training,
porting a published architecture from a spec, and test-time decoding. Each task
fixes an objective, an evaluation and a set of rules, and ships a validator that
turns a finished multi-seed run into a normalized score.

This is a port of an internal Google benchmark suite to open-source simply on
Google Cloud: the internal build, launcher and storage stack become
`python -m ...`, Cloud TPU VMs and GCS. What is scored, what is frozen and how
it is scored are unchanged; [`PORTING_NOTES.md`](PORTING_NOTES.md) lists every
deviation and why.

## The tasks

| Task | Objective | Metric (direction) | Anchors a→0, b→1 | TPU | Reference baseline |
|---|---|---|---|---|---|
| [`pretrain_bpb_v32k`](tasks/pretrain_bpb_v32k.md) | 6.0M 32k-subword LM on C4, FLOPs ≤ 1.811e16 | `validation_bpb` (lower) | 1.3790 / 1.2296 | v6e-4 | 1.4322 |
| [`pretrain_bpb_byte`](tasks/pretrain_bpb_byte.md) | 1.9M byte-level LM on C4, FLOPs ≤ 9.180e15 | `validation_bpb` (lower) | 1.7519 / 1.2607 | v6e-4 | 1.8432 |
| [`pretrain_optimizer_ttt`](tasks/pretrain_optimizer_ttt.md) | new optimizer for a fixed 41M C4 LM, H=1200 | `time_to_target_speedup` (higher) | 1.6848 / 3.4168 | v6e-4 | 1.575 |
| [`rl_gemma3_1b`](tasks/rl_gemma3_1b.md) | Gemma-3-1B PT → GSM8K, 0-shot boxed | `eval_accuracy` (higher) | 0.0572 / 0.3015 | v6e-4 | ~0.00 |
| [`rl_qwen2p5_math_1p5b`](tasks/rl_qwen2p5_math_1p5b.md) | Qwen2.5-Math-1.5B → MATH500 L4-5, avg@8 | `eval_accuracy` (higher) | 0.4273 / 0.5780 | v6e-4 | 0.345 |
| [`rl_bfcl_qwen3_0p6b`](tasks/rl_bfcl_qwen3_0p6b.md) | Qwen3-0.6B → BFCL-Live tool calling | `eval_accuracy` (higher) | 0.2607 / 0.5174 | v6e-4 | 0.254 |
| [`rl_bfcl_gemma3_1b`](tasks/rl_bfcl_gemma3_1b.md) | Gemma-3-1B PT → BFCL-Live tool calling | `eval_accuracy` (higher) | 0.0759 / 0.4839 | v6e-4 | 0.032 |
| [`sampling_lcb`](tasks/sampling_lcb.md) | test-time decoding for a fixed Qwen3-4B on LiveCodeBench v5 | `accuracy` (higher) | 0.3896 / 0.7056 | v6e-4 | 0.376 |
| [`decode_efficiency_vf`](tasks/decode_efficiency_vf.md) | decode Qwen3-30B-A3B-Thinking on AIME-25 faster, accuracy ≥ 0.75 | `avg_generation_time` (lower) | 16.2718 / 5.1540 | v6e-8 | (re-measure) |
| [`port_falcon_h1_0p5b`](tasks/port_falcon_h1_0p5b.md) | implement Falcon-H1-0.5B (attention + Mamba-2) from a spec | `eval_accuracy` (higher) | 0.4317 / 0.9584 | v6e-4 | stage-1 ≈ 0.60 |
| [`port_recurrentgemma_2b`](tasks/port_recurrentgemma_2b.md) | implement RecurrentGemma-2B (Griffin RG-LRU) from a spec | `eval_accuracy` (higher) | 0.0000 / 0.5423 | v6e-4 | stage-1 ≈ 0.12 |

`score = clamp((a − raw) / (a − b), 0, 1.2)` for lower-is-better metrics, and
the mirror image for higher-is-better; 0 means "no better than the shipped
baseline", 1 means "reached the headroom estimate". Reference baselines are the
internal measurements ([`validator/ANCHORS.md`](validator/ANCHORS.md) says where
each number comes from); what this port measures on Cloud TPU is in
[`BASELINES.md`](BASELINES.md).

## Quick start

```bash
cd /path/to/simply                      # everything runs from the repo root
pip install -e ".[tpu,assets,math-eval]"

# 1. unit tests — no accelerator, no assets
python -m pytest tasks/research_bench/tests tasks/research_bench/validator -q

# 2. assets for the task you picked (see ASSETS.md for sizes and times)
python -m tasks.research_bench.setup.prepare_assets --task=pretrain_bpb_byte
python -m tasks.research_bench.setup.prepare_assets --task=pretrain_bpb_byte \
    --gcs-bucket=gs://<your-bucket>     # mirror for Cloud TPU VMs

# 3. the baseline, locally on CPU, tiny (sanity only, not a result)
python -m tasks.research_bench.main --experiment_config=pretrain_bpb_byte \
    --experiment_dir=/tmp/rb_smoke --config_overlay='{"num_train_steps": 5}' \
    --alsologtostderr

# 4. the scored run: one Cloud TPU VM per seed, results in GCS
python -m tasks.research_bench.launch.launch_gcp \
    --task=pretrain_bpb_byte --experiment_name=bpb_byte_baseline \
    --bucket=gs://<your-bucket> --seeds=42,43,44 \
    --tpu-type=v6e-4 --zone=<your-zone> --spot

# 5. score it
python -m tasks.research_bench.validator.run_validator \
    --task=pretrain_bpb_byte --experiment_dir=gs://<your-bucket>/bpb_byte_baseline
```

[`GCLOUD.md`](GCLOUD.md) covers project setup, quota, zones, spot preemption and
the SSH-less transport. [`ASSETS.md`](ASSETS.md) covers every dataset,
tokenizer and checkpoint each task needs, including the two that need a
HuggingFace licence acceptance.

## Layout

```
tasks/<task_id>.md      the task prompt handed to the agent (11 of them)
config_lib.py           the baseline config per task; add yours next to it
data_lib.py             C4, BFCL/ToolACE, MATH500-L4/5, porting tokenizers
model_lib.py            fixed-compute train loop: FLOP accounting + integrity checks
rl_loop.py              RL train loop with the inline held-out eval
main.py / eval_main.py  training and eval-only entry points
launch/                 launch_gcp.py: run a seed sweep on Cloud TPU VMs
validator/              scoring: task_specs.py, validator.py, run_validator.py
setup/                  prepare_assets.py and friends
tests/                  CPU unit tests
```

## What has actually been run on Cloud TPU

The port is not a paper exercise; this is the evidence behind it, all on
`v6e` VMs driven by `launch/launch_gcp.py`.

| What | Evidence |
|---|---|
| `pretrain_bpb_byte`, full 3-seed baseline | mean `validation_bpb` **1.83456** vs the internal 1.8432 (−0.47%), validator `valid: True` |
| `pretrain_bpb_v32k`, full 3-seed baseline | mean **1.43710** vs the internal 1.4322 (+0.34%), validator `valid: True` |
| `pretrain_optimizer_ttt`, anchor + baseline sweeps | four loss targets and the a/b anchors re-derived for this port ([`BASELINES.md`](BASELINES.md)) |
| `sampling_lcb`, decode smoke (9 of 167 problems) | Qwen3-4B decodes on TPU, the sandboxed grader runs (`sandbox=bwrap`), `final_result.json` written |
| `rl_gemma3_1b`, 2-step smoke with the full GSM8K eval | Gemma-3-1B PT restores, GRPO rollouts run on TPU, `eval_accuracy` + `eval_protocol` written |
| Checkpoint restores | all staged checkpoints restore with **0 re-initialised branches** and published parameter counts (`prepare_assets check-restore`) |
| Unit tests | 220 across the suite, CPU only, no assets required |

Not yet run here: full baselines for the four RL and two porting tasks (their
configs, assets and loops are verified, but a baseline is 20 min - 2 h per
seed), and `decode_efficiency_vf` — whose 30B checkpoint shards and loads on
`v6e-8` but whose run never survived spot preemption, and whose wall-clock
anchors therefore still have to be re-measured before it can be scored. See
[`BASELINES.md`](BASELINES.md).

## Rules that are the same for every task

* **A submission is a 3-seed sweep** (seeds 42/43/44, one run each) in one GCS
  experiment dir. The validator rejects a submission with a missing, repeated
  or mismatched seed.
* **The eval is frozen.** Each run stamps an `eval_protocol` into
  `final_result.json`; the validator matches it against the task's pinned
  fields (dataset, number of scored examples, decoding settings, vocab).
* **Compute accounting is frozen** for the two capped pretraining tasks: no
  `scan` in the train step, no custom (Pallas/Mosaic) kernels, and the run's own
  `training_flops_xla` must be under the cap. The run writes a
  `compute_integrity` block and the validator checks it.
* **Your own work, your own run**: the submitted runs must be ones you launched
  in your own session.

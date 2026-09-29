# Baselines measured by this port on Cloud TPU

The numbers each task prompt quotes as its *reference baseline* were measured
internally, with the original scaffolding and internal TPU hardware. This file
is the Cloud-TPU counterpart: the same baseline configs, unchanged, run through
`launch/launch_gcp.py` on Cloud TPU VMs with the assets
[`ASSETS.md`](ASSETS.md) describes. Use these when judging your own runs; use
the internal numbers only to see whether the port landed in the same place.

Why they can differ at all — the three substitutions in
[`PORTING_NOTES.md`](PORTING_NOTES.md): C4 read in release-file order instead of
TFDS's hash-shuffled order, a C4-trained 100864-piece tokenizer in place of the
internal one (`pretrain_optimizer_ttt` only), and different accelerators.
Everything else — model, optimizer, steps, batch, sequence length, eval split,
metric and FLOP accounting — is the internal configuration.

## Method

```bash
python -m tasks.research_bench.launch.launch_gcp \
    --task=<task_id> --experiment_name=baseline_<task_id> \
    --bucket=gs://<bucket> --seeds=42,43,44 --tpu-type=v6e-4 --zone=<zone> --spot
python -m tasks.research_bench.validator.run_validator \
    --task=<task_id> --experiment_dir=gs://<bucket>/baseline_<task_id>
```

No config overrides beyond the per-task defaults `--task` supplies.

## Results

| Task | Metric | Per-seed (42/43/44) | Mean | Internal reference | TPU | Wall-clock/run |
|---|---|---|---|---|---|---|
| `pretrain_bpb_byte` | `validation_bpb` | 1.85625 / 1.84032 / 1.80712 | **1.83456** | 1.8432 (−0.47%) | v6e-4 | ~4-5 min |
| `pretrain_bpb_v32k` | `validation_bpb` | 1.43835 / 1.43606 / 1.43687 | **1.43710** | 1.4322 (+0.34%) | v6e-4 | ~5-6 min |
| `pretrain_optimizer_ttt` | `time_to_target_speedup` | 1.5263 / 1.6237 / 1.5737 | **1.5746** | 1.277 (different targets) | v6e-4 | ~5 min |

### Measured FLOPs run a few percent higher here than internally — plan for it

Same configs, measured `training_flops_xla`, identical across seeds:

| Task | this port (v6e-4) | internal | delta | % of cap |
|---|---|---|---|---|
| `pretrain_bpb_byte` | 6.44633e15 | 6.120e15 | +5.3% | 70.2% of 9.180e15 |
| `pretrain_bpb_v32k` | 1.24984e16 | 1.207e16 | +3.6% | 69.0% of 1.811e16 |

The cap is enforced against the value **your run measures**, so a recipe tuned
to more than ~95% of the cap on the internal numbers can trip it here.

`tpu_mfu.training_flops_xla_1cpu` estimates 5.914e15 for the same config — 0.92x
the TPU measurement. Use it for *relative* comparisons only, exactly as the task
prompts say.

## `pretrain_optimizer_ttt` loss targets

The task scores a *ratio*, so its four loss targets have to come from a run in
this port rather than from the internal curve. `pretrain_optimizer_ttt_anchor`
is the registered weak-Adam anchor; the mean of its three seeds'
`validation_loss_curve` at steps 480 / 720 / 960 / 1200 defines the targets that
`validator/task_specs.TTT_TARGETS` must hold, and the tuned baseline above then
fixes the `a` / `b` anchors (see [`validator/ANCHORS.md`](validator/ANCHORS.md)
for the arithmetic).

Measured 2026-09-29, `gs://.../baseline_ttt_anchor`, v6e-4, seeds 42/43/44
(`training_flops_xla` 3.68628e16, ~5 min/seed):

| Fraction of H=1200 | Step | Anchor loss (mean of 3 seeds) |
|---|---|---|
| 40% | 480 | 5.4272 |
| 60% | 720 | 4.9969 |
| 80% | 960 | 4.7037 |
| 100% | 1200 | 4.5901 |

Those four numbers are `validator/task_specs.TTT_TARGETS`. Scored against them,
the tuned baseline measures **1.5746** (above), which sets the anchors by the
internal calibration's ratios: `a = 1.070 x 1.5746 = 1.6848`,
`b = 2.028 x a = 3.4168`. The internal a/b (1.3662 / 2.7709) do not apply here
because the targets are raw losses under a different tokenizer.

The anchor run itself scores 0.917 on average (1.006 / 1.012 / 0.735), not
exactly 1.0: targets are the *mean* curve, so a seed that ends fractionally
above the 1200-step target never crosses it and contributes 0 for that target.
That cliff is inherent to the metric (a target you never reach scores 0) and
affects only runs sitting at anchor level, which score 0 after the affine
transform anyway.

## Smoke runs (not baselines) that prove the other task families run here

| Task | What ran | Result |
|---|---|---|
| `rl_gemma3_1b` | `num_train_steps=2`, full 1319-example GSM8K eval, no config overrides | Gemma-3-1B PT restored from the staged ORBAX, GRPO rollouts + boxed rewards on v6e-4, `eval_accuracy=0.0` (a PT model rarely emits `\boxed{}` — the internal baseline is ~0.00 too), full `eval_protocol` stamp |
| `sampling_lcb` | `LcbBaseline` over 9 of the 167 problems | `accuracy=0.5556`, 20.8 s/example, grader ran in the `bwrap` sandbox |

Timing these imply, which the internal references do not: an RL seed here is
**eval-dominated** (~8 min per full GSM8K eval on v6e-4, run every
`validation_eval_interval` steps), and a full `sampling_lcb` seed is ~1 h on
v6e-4 rather than the internal 25-30 min.

## Not yet measured here

* The four RL tasks and the two porting tasks: the configs and assets are in
  place, but their baselines are 20 min - 2 h per seed and were not part of this
  campaign. Their prompts quote the internal references.
* `sampling_lcb`: hardware-independent metric, so the internal ~0.376 should
  carry; one `LcbBaseline` run on `v6e-4` would confirm it.
* `decode_efficiency_vf`: **attempted on `v6e-8`, not measured** — and the
  reason is capacity, not code. The 42.5 GiB Qwen3-30B-A3B-Thinking checkpoint
  shards and restores on a real `1,1,2,4` mesh across 8 chips, sparse MoE with
  expert parallelism initialises, ragged-paged attention autotunes and the page
  batcher compiles — but ~30 min of unavoidable setup (apt+pip, a 45.6 GB asset
  mirror, checkpoint load + compile) does not fit inside the observed v6e-8
  **spot** lifetimes of 6-13 min, and v6e-8 was spot-only in the project used
  here. Three nodes, three preemptions, no decode.
  To finish it: one **on-demand** v6e-8 for ~75 min. Then `base` is the measured
  `avg_generation_time` and the anchors follow the internal calibration's
  ratios, `a = 0.884 x base`, `b = 0.280 x base`; the accuracy gate (≥ 0.75) is
  hardware-independent and stands. Until then its shipped anchors
  (16.2718 / 5.1540) belong to the internal 8-chip reference slice and **must not** be used
  to score a Cloud run.
  Practical notes for whoever does it: TPU VM boot disks are not sizable
  (`tpu-vm create` has no disk-size flag, ~97 GB), so mirror selectively
  (`--assets-include=models/Qwen3-30B-A3B-Thinking-2507,vocabs,datasets/aime`).

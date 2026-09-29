# Where the scoring anchors come from

`score = clamp((a - raw) / (a - b), 0, 1.2)` for lower-is-better tasks, mirrored
otherwise (`task_specs.TaskSpec.score`). This file records, per task, where `a`
and `b` came from and what would invalidate them. The numbers themselves live in
`task_specs.py` and are the single source of truth; nothing here is computed at
runtime.

## How the anchors were set

* `a` (score 0) is a value **better than the shipped baseline** in every task, so
  re-running the unchanged baseline config scores 0 (see the "internal baseline"
  column below).
* `b` (score 1) is the strong-submission target.
* Both were calibrated once on the internal 2026-07-27 pilot round (several
  agents per task, 3 seeds each) so that the weakest pilot submission lands near
  0.1 and the best near 0.7-0.9. The hand-verified `(raw -> score)` points from
  that round are pinned as regression tests in
  `validator_test.py::test_anchors_reproduce_the_published_scores`; they are the
  operational definition of "the anchors are unchanged".
* **The anchors were measured on internal hardware with internal assets and have
  NOT been re-measured for the open-source port**, except where the port status
  section below says otherwise. `pretrain_optimizer_ttt` is the one task whose
  a/b are already port-measured: its metric is a ratio against loss targets
  that had to be re-derived here, so the numbers come from Cloud-TPU baselines
  with only the internal calibration's *ratios* preserved.
* The Cloud-TPU baseline runs behind all of this -- per-seed values, wall-clock,
  buckets and method -- are in [`../BASELINES.md`](../BASELINES.md), which this
  file deliberately does not duplicate.

## Per task

| task | metric (dir) | a -> 0 | b -> 1 | internal baseline (the run `a` beats) |
|---|---|---|---|---|
| pretrain_bpb_v32k | validation_bpb (lower) | 1.3790 | 1.2296 | internal reference sweep, mean 1.4322, ~1.207e16 FLOPs |
| pretrain_bpb_byte | validation_bpb (lower) | 1.7519 | 1.2607 | internal reference sweep, mean 1.8432, ~6.120e15 FLOPs |
| pretrain_optimizer_ttt | time_to_target_speedup (higher) | 1.6848 | 3.4168 | port-measured: `baseline_pretrain_optimizer_ttt`, mean speedup 1.5746 (v6e-4, 2026-09-29; the internal sweep measured 1.277 against different targets) |
| rl_gemma3_1b | eval_accuracy (higher) | 0.0572 | 0.3015 | internal reference sweep, mean ~0.0010 (per-seed 0.0000 / 0.0030 / 0.0000) |
| rl_qwen2p5_math_1p5b | eval_accuracy (higher) | 0.4273 | 0.5780 | internal reference sweep, mean ~0.345 |
| rl_bfcl_qwen3_0p6b | eval_accuracy (higher) | 0.2607 | 0.5174 | internal reference sweep, mean ~0.254 |
| rl_bfcl_gemma3_1b | eval_accuracy (higher) | 0.0759 | 0.4839 | internal reference sweep, mean ~0.032 |
| sampling_lcb | accuracy (higher) | 0.3896 | 0.7056 | `LcbBaseline`, expected 3-seed mean ~0.376 (the internal sweep drew 0.371) |
| decode_efficiency_vf | avg_generation_time (lower), gate accuracy>=0.75 | 16.2718 | 5.1540 | NOT re-measured: dropped from the internal bundle, see below |
| port_falcon_h1_0p5b | eval_accuracy (higher) | 0.4317 | 0.9584 | a faithful stage-1 port scores ~0.68 under this harness; port_reference 0.60 |
| port_recurrentgemma_2b | eval_accuracy (higher) | 0.00 | 0.5423 | a faithful stage-1 port reproduces the published ~0.134; port_reference 0.12 |

Where a Cloud-TPU run of the same unchanged config exists, it is in
[`../BASELINES.md`](../BASELINES.md); the "internal baseline" column stays the
internal measurement so the two can be compared.

Baselines were re-measured for the hardware-homogenized internal task bundle
(2 chips for the pretrain tasks, 4 chips for the RL and sampling tasks). The
anchors themselves did NOT move with that re-measurement. The internal sweeps
are provenance only; they are not something the open-source validator can read.
`port_reference` is advisory: a stage-1 port below it warns, never fails
(`validator.validate`, step 4).

## What invalidates an anchor

An anchor is only comparable to runs produced under the same conditions. Any of
the following requires re-measuring `a` and `b` for the affected task:

* **A different asset**: vocabulary, tokenizer, training corpus, eval set, or
  base checkpoint. Bits-per-BYTE is vocab-normalized and survives a +-1 piece
  difference; raw validation LOSS (pretrain_optimizer_ttt) survives nothing.

  A *silently* different model counts here too: `simply.model_lib`
  re-initialises the parameter branches a checkpoint does not cover and carries
  on, so a half-mapped checkpoint trains a partly random model and reports a
  plausible bad metric. The four RL tasks pin their base checkpoint, so the
  validator reads `seed_<seed>/log.txt` and fails such a run rather than
  scoring it against anchors that assume the pinned model; on the two porting
  tasks the model class is the agent's own, so the same finding is a warning
  (`TaskSpec.checkpoint_policy`).
* **A different eval protocol.** Pinned per task in
  `task_specs.TASKS[...].protocol` precisely so this cannot happen silently:
  changing `n_scored`, the eval source, the temperature or the sample count
  changes the metric's meaning, and the validator fails the run instead of
  scoring it against a stale anchor.
* **Different hardware, for a time-based metric.** `decode_efficiency_vf`'s
  `avg_generation_time` is seconds on the topology it was measured on
  (an internal 8-chip slice). A different accelerator moves both anchors -- this has
  already happened in this port, see the port status below. Nothing else in the
  suite is wall-clock based.
* **A changed compute cap.** The two pretrain tasks' anchors assume the runs fit
  `flops_cap`; raising the cap makes `b` reachable by brute force.

  The caps themselves are unchanged in this port, but the *same config* measures
  a few percent more compute on v6e than internally -- `pretrain_bpb_byte`
  6.44633e15 vs 6.120e15 (+5.3%, 70.2% of its cap), `pretrain_bpb_v32k`
  1.24984e16 vs 1.207e16 (+3.6%, 69.0%). The validator enforces the cap against
  the value the submission itself reports, so a recipe tuned to ~95% of the cap
  on the internal numbers can trip it here. See
  [`../BASELINES.md`](../BASELINES.md).

## Port status (open-source suite)

* **decode_efficiency_vf**: STALE, re-measurement required BEFORE the task is
  scored here. `avg_generation_time` is wall-clock seconds, so it is a property
  of the accelerator as much as of the decoder: the internal bundle DROPPED this
  task when it homogenized hardware, for exactly that reason. The Cloud port
  pins it to `v6e-8`, which is not the internal slice the anchors were measured on,
  so a=16.2718 / b=5.1540 (and the reference ~18.4 s at accuracy ~0.78) do not
  transfer. Re-measure by running the unchanged pipeline on `v6e-8` (seed 42)
  for the new reference `base`, then keep the internal calibration's ratios:
  a = 0.884 * base, b = 0.280 * base (internally base=18.4 s). The accuracy
  gate (>=0.75) is hardware-independent and stands. See
  `task_specs.DECODE_ANCHOR_HARDWARE`.
* **pretrain_bpb_v32k**: `nanodo_c4` is reproduced bit-exactly from the public
  t5-data sentencepiece model plus one `<s>` control piece (32101 pieces,
  bos_id=2), so the protocol stamp and the anchors stand as measured.
* **pretrain_optimizer_ttt**: RE-DERIVED for this port on 2026-09-29, by the
  recipe above `task_specs.TTT_TARGETS`. The internal 100k vocab
  `vb100864_openmix_v1` has no public twin, so the port trains its own
  100864-piece C4 SPM (`task_specs.TTT_TARGETS_VOCAB = 'c4_spm100864'`), which
  moves the loss scale and invalidated the internal targets. New numbers, all
  from v6e-4 seeds 42/43/44 (`gs://<your-bucket>/baseline_ttt_anchor` and
  `.../baseline_pretrain_optimizer_ttt`):

  | | 480 | 720 | 960 | 1200 |
  |---|---|---|---|---|
  | anchor mean loss = `TTT_TARGETS` | 5.4272 | 4.9969 | 4.7037 | 4.5901 |

  Scored against those, the tuned baseline measures `base` = 1.5746 (per-seed
  1.5263 / 1.6237 / 1.5737), so `a = 1.070 * base = 1.6848` and
  `b = 2.028 * a = 3.4168` -- the internal ratios, applied to a port-measured
  baseline.

  Worth knowing before anyone "fixes" it: the anchor run scores **0.917** mean
  (1.006 / 1.012 / 0.735), not exactly 1.0. The targets are the *mean* curve, so
  a seed that ends fractionally above the 1200-step target never crosses it and
  contributes 0 for that target. The cliff is inherent to a time-to-target
  metric (a target you never reach scores 0) and only bites runs sitting at
  anchor level, which score 0 after the affine transform anyway.
* **The two bpb tasks**: anchors carried over unchanged, and a Cloud-TPU run of
  the unchanged baseline config lands on the internal number --
  `pretrain_bpb_byte` 1.83456 here vs 1.8432 internally (-0.47%),
  `pretrain_bpb_v32k` 1.43710 vs 1.4322 (+0.34%). That is the evidence the
  anchors transfer; see [`../BASELINES.md`](../BASELINES.md).
* **The four RL tasks, the two porting tasks and `sampling_lcb`**: anchors
  carried over unchanged from the internal suite and NOT yet confirmed by a
  Cloud-TPU baseline (20 min - 2 h per seed, not part of the port campaign).
  `sampling_lcb`'s metric is hardware-independent, so the internal ~0.376
  should carry; one `LcbBaseline` run would confirm it. Re-measure any of them
  if a run of the unchanged baseline config on GCP does not land near the
  "internal baseline" column above.

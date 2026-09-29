# Porting notes: internal -> OSS/GCP

This package is a port of the internal Research Bench scaffolding. The
goal is a small, reviewable diff: file names, class/function names, structure and
comments are kept unless they stopped being true. This file lists every place
the port behaves differently from the internal source, and why.

Anything listed under **Substituted asset** changes the numbers a run produces,
so the affected reference anchors must be re-measured (or explicitly flagged
"internal reference, not re-measured") in the task `.md`.

## 1. Removed internal-only machinery

| Internal | OSS | Where |
|---|---|---|
| the internal multiprocessing entry point | `absl.app.run` (+ `jax.distributed.initialize()` for multi-host TPU VMs, as core `simply/main.py` does) | `main.py` |
| `_per_work_unit_dir()` (work-unit subdir + "must be an internal-storage path" guard) | deleted — a submission is one process per seed, each launched with its own `--experiment_dir` | `main.py` |
| the internal `tensorboard` event-accumulator import | the public `tensorboard.backend.event_processing.event_accumulator` (same API; `tensorboard` is already a core dependency) | `model_lib.py` |
| internal-storage model paths | `SIMPLY_MODELS`-rooted paths via `os.path.join(core.MODELS_DIR, ...)`, as in OSS `simply/config_lib.py` | `config_lib.py`, `data_lib.py` |
| internal build / "work unit" / storage wording in docstrings | reworded to the OSS equivalent | several |

`os.path.join` replaces the internal `MODELS_DIR + '...'` string concatenation
so a `SIMPLY_MODELS` without a trailing slash cannot silently produce a wrong
path.

## 2. Run layout and provenance

A submission is a GCS experiment dir, not an XID:

    gs://<bucket>/<experiment_name>/seed_<seed>/final_result.json

`main.py` writes into `--experiment_dir` exactly as given; the launcher gives
each seed its own directory. Because the seeds no longer come from an
internal sweep, `record_run_provenance` additionally writes `model_seed` and
`dataset_seed` into the `run_provenance` block, so the validator can verify
that three seed dirs really ran three distinct seeds instead of trusting their
names.

## 3. Substituted asset: C4 (all pretraining tasks)

Core simply reads C4 as `TFDSSource(name='c4:3.1.0')`. TFDS has no public
mirror of C4 and building it needs the full CommonCrawl pipeline, so the port
reads the `allenai/c4` `en` release repacked into flat `.bin` blobs +
`.idx.npy` uint64 offset tables (`setup/build_c4.py`, read by
`data_lib.C4FileSource`; `config_lib.C4_TRAIN` / `C4_VALIDATION`).

* Same corpus, same documents, deterministic order — but the *file* order of
  the release, not TFDS's key-hash shuffle. Absolute bpb is therefore only
  comparable within this port.
* `SIMPLY_C4_DIR` points at a local directory or a `gs://` prefix; remote
  shards are mirrored into `SIMPLY_C4_CACHE_DIR` before being memory-mapped.
* Shard naming (agreed with the asset builder):
  `c4-{split}.{i:05d}-of-{n:05d}.bin` + `.idx.npy`, where `n` is the *release*
  file count (1024 train / 8 validation), not how many shards are staged.
* Side effect: core names each validation dataset's tb_log scalars after the
  source, so the held-out val-loss tag moved from `c4:3.1.0/eval_loss` to
  `C4FileSource/eval_loss`. `model_lib._val_loss_tag(config)` derives it from
  the config instead of hardcoding it, so `pretrain_optimizer_ttt` still finds
  its `validation_loss_curve`.

**Re-measure:** `pretrain_bpb_v32k` and `pretrain_bpb_byte` anchors
(`a`/`b`), and the `pretrain_optimizer_ttt` anchor curve.

## 4. Substituted asset: the `nanodo_c4` vocabulary (`pretrain_bpb_v32k`)

The internal 32k reference vocabulary
(`GetPieceSize()==32101`, hence the task's `vocab_size=32_101`) is
stripped out of core's vocab table, so this package registers
`nanodo_c4` itself (`data_lib.NANODO_C4_VOCAB`, env override
`SIMPLY_NANODO_C4_VOCAB`).

The file is **regenerated** from the public T5 model
`gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model` (32100 pieces,
`bos_id() == -1`) by inserting one `<s>` CONTROL piece at index 2 — giving
`bos_id=2`, `unk_id=3`, 32101 pieces. `setup/prepare_assets.py` asserts the
result is byte-identical to the internal file (sha256
`15d8dfa11996e0d1e8645897ec4d70b1dec6a7cb6d87df0daf560a3cde479006`, confirmed
on the staged asset), so this is a *reconstruction*, not an approximation: tokenization, id space and model shape are the internal
ones and **no re-measurement is needed for the vocab** (the C4 order in §3
still applies).

Without the inserted BOS piece this task would be silently wrong, not just
different: core's `DatasetConfig.add_bos` defaults to `True`, so the raw public
model's `bos_id == -1` would prepend token id **-1** to every document.

## 5. Substituted asset: `vb100864_openmix_v1` (`pretrain_optimizer_ttt`)

**No public twin exists.** The port keeps every model shape — in particular
`vocab_size=100_864`, so the embedding/logit matmuls and hence the per-step
FLOPs are exactly the internal ones, which is what a fixed-compute
optimizer-efficiency task measures — and swaps only the tokenizer, to the 32k
`nanodo_c4` model above. The choice is a single constant,
`config_lib.TTT_VOCAB_NAME`.

Consequences:

* Token ids `>= 32101` are unreachable: ~68k embedding rows stay at
  initialization and the softmax has ~68k never-target classes.
* The absolute validation loss is **not** comparable to the internal reference.
  The `a`/`b` anchors (1.3662 / 2.7709) and the frozen `TTT_TARGETS` loss
  values must be re-derived on Cloud TPU.

## 6. The time-to-target anchor is now a registered config

`time_to_target_speedup` is `anchor_step / first_crossing_step` averaged over
four loss targets taken at 40/60/80/100% of the 1200-step budget from a "weak
Adam" anchor curve. The internal anchor's hyperparameters were not shipped with
the scaffolding, and §5 moves the loss scale anyway, so the port registers the
anchor **by name**: `pretrain_optimizer_ttt_anchor`.

It shares `_ttt_base()` with the baseline — same model, data, 1200 steps, batch
80, seq 2048, vocab — and differs only in the optimizer: the untuned recipe
inherited from core `flops2e17_tfm41m_c4_l2048` (plain Adam, a ~7x smaller LR
fitted for a 4140-step horizon, cosine decay to 10%), weak precisely because it
is untuned for this horizon. To re-derive the targets: run this config and read
`validation_loss_curve` at steps 480 / 720 / 960 / 1200.

## 7. Checkpoints that must be converted

`rl_bfcl_qwen3_0p6b` used an internal Qwen3-0.6B-**Base** ORBAX export. The
public `Qwen3-0.6B` release on the simply model zoo is POST-TRAINED and
truncates under the `Pretrain` format, so `setup/setup_assets.py` must convert
`Qwen/Qwen3-0.6B-Base` from HF; the config reads
`$SIMPLY_MODELS/Qwen3-0.6B-Base/ORBAX`.

The two porting tasks keep their internal layout under
`$SIMPLY_MODELS/research_bench/{Falcon-H1-0.5B-Base,RecurrentGemma-2B}/{ORBAX,VOCAB}`.

## 8. Unchanged on purpose

The compute-integrity machinery is the anti-cheating core of the benchmark and
is ported verbatim: `FixedComputeConfig.fixed_compute_cap_flops`, the
`research_bench_pretrain` loop's `use_scan` / `use_flash_attention` / `tpu_custom_call`
checks, the `compute_integrity` and `eval_protocol` stamps,
`record_run_provenance`, and `tpu_mfu.training_flops_xla_1cpu`. So are the RL
loop, the RL algorithms, the BFCL evaluation and the porting-task stubs.

`tpu_mfu.training_flops_xla_1cpu` (1-CPU XLA cost-model estimate) on this port:

| config | 1-CPU estimate | internal TPU `training_flops_xla` | ratio | cap |
|---|---|---|---|---|
| `pretrain_bpb_byte` | 5.914e15 | 6.120e15 | 0.966 | 9.180e15 |
| `pretrain_bpb_v32k` | 1.187e16 | 1.207e16 | 0.983 | 1.811e16 |

(The internal TPU numbers are `cap / 1.5`. The estimate is deliberately not
exact — CPU and TPU pick different kernels — but it is within 3.5% here, so it
remains a safe pre-launch budget check.)

## 9. Per-run overrides the reference RL/port baselines were measured with

The reference RL and port runs are launched with
`sampling_decode_buffer_multiple=128` and `eval_decode_buffer_multiple=128`
(plus `validation_eval_batch_size=128` for `rl_bfcl_qwen3_0p6b`), applied by
the launcher through `--config_overlay`, not baked into the configs. Both
fields are research_bench `RLConfig` fields (not core ones) and both survive the port:
`rl_algorithms.SimpleGRPO.sampling_params` reads the first,
`rl_loop._held_out_eval` the second, and both feed core
`SamplingParams.decode_buffer_multiple`, which exists in OSS
`simply/utils/sampling_lib.py`. They pad decode buffers to a fixed grid so
`decode_fn` is not recompiled per prompt-length; the config default stays 0
(off) exactly as internally, so a run WITHOUT the overlay is slower but
numerically the same on the eval side.

## 10. Eval-set cardinalities reproduce the internal ones

The ported data sources, read against the OSS-staged datasets, yield exactly
the `n_scored` the internal task specs pin -- 1319 (gsm8k test), 262
(`MATH500TestL45Source`, the level-4/5 filter) and 1351
(`BFCLLiveEvalSource`, call-required categories only, i.e. `live_irrelevance`
excluded). Verified by reproducing `rl_loop._eval_protocol` for all six
RL/port configs; it is a pure function of the config, the `Evaluation`
dataclass and `len(eval_ds)`, so it needs no checkpoint, and the resulting
stamps are the golden fixtures in `validator/testdata/`.

This is the strongest single piece of evidence that the data port is faithful,
and the validator pins `n_scored`, so it is also the loudest failure mode: if a
dataset is ever re-staged with a different filter or split, every submission on
that task is rejected rather than silently scored against a different eval set.
Re-stage deliberately, and re-run the fixtures if you do.

## 11. For the task `.md` prompts

Things an agent following a task prompt has to be told, which used to be
implicit in the internal setup:

* Build: no internal build system. From the repo root,
  `pip install -e ".[tfds,assets]"`, then
  `python -m tasks.research_bench.main --experiment_config=<id>
  --experiment_dir=gs://<bucket>/<name>/seed_42
  --config_overlay='{"model_seed": 42, "dataset_seed": 42}'`.
* Assets come from `setup/` and live under
  `SIMPLY_MODELS` / `SIMPLY_DATASETS` / `SIMPLY_VOCABS`; C4 additionally honours
  `SIMPLY_C4_DIR` (local dir or `gs://` prefix).
* Local pre-launch FLOPs check:
  `python -c "from tasks.research_bench import tpu_mfu, config_lib;
  print(tpu_mfu.training_flops_xla_1cpu(
  config_lib.ExperimentConfigRegistry.get_config('<id>')))"`.
* Configs resolve in this package's private registry only; a core simply config
  name is deliberately not runnable through this entry point.
* The C4 and `vb100864_openmix_v1` substitutions above, with the re-measurement
  caveat, belong in `pretrain_bpb_*` / `pretrain_optimizer_ttt`.
* `pretrain_optimizer_ttt` should point at `pretrain_optimizer_ttt_anchor` as
  the reproducible definition of the loss targets.

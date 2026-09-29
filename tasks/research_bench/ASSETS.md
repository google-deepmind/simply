# Research Bench — assets

Everything the 11 tasks need, where it comes from, and how to get it. One
script does all of it:

```bash
cd <repo root>                       # the simply repo; run everything from here

# everything one task needs (what the task prompts tell you to run)
python -m tasks.research_bench.setup.prepare_assets --task=rl_gemma3_1b

# everything, for all 11 tasks (~85 GiB)
python -m tasks.research_bench.setup.prepare_assets all

# the ONE command that checks an existing cache: re-reads every staged asset,
# re-counts every dataset, re-hashes the vocabs, and re-opens every checkpoint.
# Non-zero exit means something is missing or has drifted.
python -m tasks.research_bench.setup.prepare_assets verify

# pre-flight before spending accelerator time: actually restore each task's
# checkpoint on CPU and assert 0 re-initialised parameter branches.
python -m tasks.research_bench.setup.prepare_assets check-restore --task=rl_gemma3_1b

# after copying the cache ANYWHERE (e.g. onto a TPU VM): recompute every
# file's crc32c and compare with the manifest. Catches a corrupt copy that
# plain `verify` cannot see.
python -m tasks.research_bench.setup.prepare_assets verify --hash
```

`--task=<task_id>` works for all 11 task ids (`pretrain_bpb_v32k`,
`pretrain_bpb_byte`, `pretrain_optimizer_ttt`, `rl_gemma3_1b`,
`rl_qwen2p5_math_1p5b`, `rl_bfcl_qwen3_0p6b`, `rl_bfcl_gemma3_1b`,
`sampling_lcb`, `decode_efficiency_vf`, `port_falcon_h1_0p5b`,
`port_recurrentgemma_2b`); it is repeatable, and `task <id> [<id>...]` is the
same thing spelled positionally.

Assets land in the canonical simply cache and are found with no path flags:

| env var           | default                       | holds |
|-------------------|-------------------------------|-------|
| `SIMPLY_MODELS`   | `~/.cache/simply/models/`     | ORBAX checkpoints |
| `SIMPLY_DATASETS` | `~/.cache/simply/datasets/`   | eval/train data, C4 shards |
| `SIMPLY_VOCABS`   | `~/.cache/simply/vocabs/`     | tokenizers |

`--gcs-bucket gs://<bucket>` additionally mirrors what the command staged to
`gs://<bucket>/assets/{models,datasets,vocabs}` and prints the environment a
Cloud TPU VM should export.

## Support matrix

**bit-exact** = provably the same bytes the internal task reads; **available** =
the published artifact the internal one was built from; **substituted** = no
public twin, we build a replacement (affected reference numbers must be
re-measured).

| task | assets | where from | staged size | stage time | licence | status |
|---|---|---|---:|---:|---|---|
| `pretrain_bpb_v32k` | C4 (repacked), `nanodo_c4` | `allenai/c4`, t5-data | 23.8 GiB | 5 min | ODC-BY / Apache-2.0 | **available** (vocab bit-exact; C4 same corpus, different shard order) |
| `pretrain_bpb_byte` | C4 (repacked) | `allenai/c4` | 23.8 GiB | 5 min | ODC-BY | **available** (`byte256` is code-only) |
| `pretrain_optimizer_ttt` | C4, `vb100864_openmix_v1` | `allenai/c4` + trained here | 23.8 GiB + 2 MiB | 5 min + ~15 min | ODC-BY | **substituted** (100864-piece SPM trained on C4) |
| `rl_gemma3_1b` | gsm8k, Gemma-3 vocab, Gemma-3-1B-PT | simply-datasets, `google/gemma-3-1b-pt` | 1.5 GiB | 12 s | MIT / **Gemma Terms (gated)** | **available** |
| `rl_qwen2p5_math_1p5b` | MATH-500 L4-5, DeepScaleR-40K, Qwen2.5 vocab, Qwen2.5-Math-1.5B | HF | 2.3 GiB | 9 s | MIT / Apache-2.0 | **available** |
| `rl_bfcl_qwen3_0p6b` | BFCL, ToolACE, Qwen3 vocab, Qwen3-0.6B-Base | HF | 0.9 GiB | 8 s | Apache-2.0 | **available** |
| `rl_bfcl_gemma3_1b` | BFCL, ToolACE, Gemma-3 vocab, Gemma-3-1B-PT | HF | 1.5 GiB | 12 s | Apache-2.0 / **Gemma Terms (gated)** | **available** |
| `sampling_lcb` | LiveCodeBench v5, Qwen3 vocab, Qwen3-4B | HF | 7.0 GiB | 35 s | CC-BY / Apache-2.0 | **available** (LCB data bit-exact) |
| `decode_efficiency_vf` | AIME-2025, Qwen3 vocab, Qwen3-30B-A3B-Thinking-2507 | HF | 42.5 GiB | ~6 min | Apache-2.0 | **available** |
| `port_falcon_h1_0p5b` | gsm8k, Falcon-H1-0.5B-Base + tokenizer | HF | 0.8 GiB | 5 s | Falcon-LLM licence | **available** |
| `port_recurrentgemma_2b` | gsm8k, RecurrentGemma-2B + tokenizer | HF | 4.8 GiB | 25 s | **Gemma Terms (gated)** | **available** |

Sizes are the task-specific assets; the C4 corpus and the small shared
datasets are counted once. Licences: `allenai/c4` is ODC-BY (CommonCrawl terms
apply), the t5-data vocab and the Qwen/ToolACE artifacts are Apache-2.0, gsm8k
and MATH-500 are MIT, BFCL is Apache-2.0, LiveCodeBench is CC-BY-4.0, Falcon-H1
is under TII's Falcon-LLM licence, and the two Gemma checkpoints are under the
Gemma Terms of Use, which is why they are gated and why we do **not**
redistribute the converted versions.

Nothing is blocked. Two HF repos are gated behind a click-through licence
(`google/gemma-3-1b-pt`, `google/recurrentgemma-2b`); see
[Gated repos](#gated-repos). Every asset below was actually fetched, converted
and verified — sizes and wall-clock are measured, not estimated.

## CLI

```
python -m tasks.research_bench.setup.prepare_assets [--gcs-bucket gs://B] [--dry-run] <command>

  c4               download + repack the C4 shards
                     --num-train-shards N (default 32)  --no-validation
                     --workers N  --keep-json
  vocabs           stage the tokenizers
                     --only nanodo_c4 gemma3 qwen2p5 qwen3 openmix_substitute
                     --train-openmix  --openmix-sentences N (default 10_000_000)
  datasets         stage the eval/train datasets
                     --only gsm8k math500 deepscaler aime bfcl toolace livecodebench
  models           stage the ORBAX base checkpoints
                     --only <spec name>...  --max-gib X
  task <task_id>..  stage exactly what one or more tasks need (all flags above)
                     `--task=<task_id>` (repeatable) is the same thing and
                     needs no subcommand
  all              everything, for all 11 tasks
  verify           STRUCTURAL check of a staged cache: counts, shapes,
                     layout, formats. Fast, non-zero on drift.
                     --hash    also recompute every file's crc32c and compare
                               with the manifest; with --gcs-bucket, also
                               against the mirrored objects' metadata
  check-restore    restore each task's checkpoint on CPU through the real
                     code path and assert 0 re-initialised parameter branches;
                     non-zero on a real mismatch. `--task=<id>` to narrow,
                     `--restore-max-gib N` to bound the RAM (default 16)
  plan             offline per-task table of what is and is not staged
  mirror           copy the WHOLE staged cache to --gcs-bucket
  gemma3-selftest  prove the Gemma-3 HF->flax mapping against the shipped 270M ORBAX
```

`--gcs-bucket`, `--dry-run`, `--force` and the rest may be given on either
side of the subcommand. `--force` rebuilds an asset that is already staged —
the supported way to replace a copy that failed `verify --hash`.
Every subcommand is **idempotent and resumable**: an asset already present and
passing its check is skipped, so re-running after an interruption costs only
the missing pieces. `--gcs-bucket` on a build subcommand mirrors *only what
that command staged*; `mirror` pushes everything.

Typical single-task use:

```bash
python -m tasks.research_bench.setup.prepare_assets task rl_bfcl_qwen3_0p6b \
    --gcs-bucket gs://my-bucket
```

Helper modules (importable, each with its own `check()`): `asset_lib.py`
(cache paths, GCS mirror, manifest), `build_c4.py`, `build_vocabs.py`,
`build_datasets.py`, `build_models.py`. `build_c4.py` is also a standalone CLI.

Every staged asset is appended to
`$SIMPLY_DATASETS/research_bench_manifest.json` with its source, size and
build time; that manifest drives `plan` and `mirror`.

## Measured cost

128-core workstation, ~900 MiB/s to HuggingFace and to GCS. Total **83.5 GiB**
staged, ~14 min wall clock for everything except the SPM training.

| asset | GiB | build s | notes |
|---|---:|---:|---|
| `Qwen3-30B-A3B-Thinking-2507` | 42.49 | ~360 | already ORBAX on HF |
| `c4_bin` (32 train + 8 val) | 23.84 | 301 | download + repack, 16 workers |
| `Qwen3-4B` | 5.87 | 18 | already ORBAX on HF |
| `RecurrentGemma-2B` | 4.81 | 25 | HF -> ORBAX |
| `Qwen2.5-Math-1.5B` | 2.24 | 8 | HF -> ORBAX |
| `GEMMA-3.0-1B-PT-ORBAX` | 1.45 | 11 | HF -> flax names -> ORBAX |
| `livecodebench` | 1.12 | 15 | 557 MB download, tests decoded |
| `Qwen3-0.6B-Base` | 0.87 | 4 | HF -> ORBAX |
| `Falcon-H1-0.5B-Base` | 0.76 | 4 | HF -> ORBAX |
| datasets (gsm8k, math500, deepscaler, aime, bfcl, toolace) | 0.06 | 5 | |
| vocabs | 0.02 | 2 | |

The HF snapshot cache (`~/.cache/huggingface`) holds another ~100 GiB of source
safetensors; it can be deleted after conversion.

## C4

`c4:3.0.1` is not publicly downloadable through TFDS (building it needs the
full CommonCrawl pipeline), so `build_c4.py` fetches the `allenai/c4` `en`
release — the same documents TFDS `c4/en` is built from — and repacks each
`json.gz` into the flat format `data_lib.C4FileSource` reads:

```
c4-{split}.{i:05d}-of-{n:05d}.bin      utf-8 text of the shard's documents,
                                       concatenated in file order, no separator
c4-{split}.{i:05d}-of-{n:05d}.idx.npy  np.save'd uint64 offsets, len == ndocs+1,
                                       offsets[0]==0, offsets[-1]==filesize
```

`n` is the *release* file count (1024 train / 8 validation), not how many
shards you staged. Shards are written `.part` then `os.replace`d, so a glob
never sees a half-written file. `build_c4.verify()` asserts
`offsets[-1] == os.path.getsize(bin)` per shard and utf-8-decodes the first
documents of the first and last shard of each split.

Staged under `$SIMPLY_DATASETS/c4_bin` (override with `SIMPLY_C4_DIR`):

| split | shards | docs | bytes |
|---|---:|---:|---:|
| train | 32 of 1024 | 11,402,153 | 24,714,195,573 |
| validation | **8 of 8 (full)** | **364,608** | 788,527,410 |

364,608 validation documents is exactly the TFDS `c4/en` validation
cardinality — a cheap cross-check that this is the right corpus. Each train
shard is 356,317 documents / ~772 MB.

Decode spot-checks: `c4-train.00000` doc 0 begins `'Beginners BBQ Class Taking
Place in Missoula!\nDo you want to...'`; `c4-validation.00000` doc 0 begins
`'The woman who died after falling from a bridge over the A21...'`.

### How much C4 is enough

| vocab | bytes/token (measured, C4 val) | tokens/train shard | shards for the task budget |
|---|---:|---:|---|
| `nanodo_c4` (32101) | 4.287 | 180 M | 1.39 (budget 225.0 M tokens) |
| `byte256` (259) | 1.0 | 772 M | 0.32 (budget 225.0 M tokens) |
| `spm-100864-c4-r100-v1` (100864) | 4.878 | 158 M | 1.24 (budget 196.6 M tokens) |

Budgets: `pretrain_bpb_*` = 1717 steps x batch 64 x 2048 = **225.0 M tokens**;
`pretrain_optimizer_ttt` = 1200 x 80 x 2048 = **196.6 M tokens**. **Two train
shards already cover every pretraining task**, with or without the 3-seed
sweep (each seed reads <5% of the staged data, so no wrapping). The default of
32 shards exists only so a recipe that raises the token budget, enlarges the
batch, or wants more shuffling headroom does not have to re-download;
`--num-train-shards 4` is a fine low-disk setting.

### Deviation from the internal task

The `.bin` shards hold exactly the documents of the `allenai/c4` `en` release
in that release's **file order**. TFDS `c4/en` shuffles by key hash when it
writes its shards, so the *document order* differs. The corpus and the
validation set are the same; absolute `validation_bpb` is comparable within
this port, but the internal anchors (1.3790 -> 1.2296 and 1.7519 -> 1.2607)
are "internal reference, not re-measured" until a baseline is run here.

## Vocabs

| registered name | file under `SIMPLY_VOCABS` | pieces | status |
|---|---|---:|---|
| `nanodo_c4` | `cc_all.32000.100extra.bos.model` | 32101 | **bit-exact** |
| `byte256` | — (`tokenization.ByteVocab`) | 259 | code only |
| `vb262144_gemma3` | `gemma3_cleaned_262144_v2.spiece.model` | 262144 | available |
| `Qwen2.5` | `Qwen2.5/{tokenizer,tokenizer_config}.json` | 151k | available |
| `Qwen3` | `Qwen3/{tokenizer,tokenizer_config}.json` | 151k | available |
| `vb100864_openmix_v1` -> `c4_spm100864` | `spm-100864-c4-r100-v1.model` | 100864 | **substituted** |

### `nanodo_c4` — reproducible bit-for-bit

`pretrain_bpb_v32k` pins `vocab_size=32_101`, matching the internal 32k
reference vocabulary. Provenance chain, entirely public:

```
https://storage.googleapis.com/t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model
  sha256 839ffa4b9afae8d77834a88b87781849aa021975d6063dec6085633fcaf7171c
  794,346 bytes - 32100 pieces, pad=0 eos=1 unk=2, NO bos
    |  insert one CONTROL piece '<s>' (score 0.0) at index 2,
    v  leaving trainer_spec untouched
  sha256 15d8dfa11996e0d1e8645897ec4d70b1dec6a7cb6d87df0daf560a3cde479006
  794,360 bytes - 32101 pieces, pad=0 eos=1 bos=2 unk=3
```

The second hash is **byte-identical to the internal file** (verified by reading
the internal file locally; only the derivation and the expected hash ship).
`build_nanodo_c4()` asserts both hashes, so a drifted download fails loudly.
Every public id >= 2 shifts by +1 in the derived vocab.

Two details worth knowing:

* `trainer_spec.bos_id` stays `-1`. SentencePiece resolves bos/eos/unk/pad from
  the *piece strings*, not from `trainer_spec`; setting `bos_id=2` there yields
  a semantically equivalent but **not** byte-identical file.
* The vocab has no byte fallback, so text outside its training distribution
  does not round-trip (`round_trip_unicode: false` in `verify` is expected, not
  a failure). `token_byte_lengths()` returns 0 bytes for pad/bos/eos/unk.

Both files are staged: the raw 32100-piece download as
`cc_all.32000.100extra.sentencepiece.model` and the derived 32101-piece
`cc_all.32000.100extra.bos.model`, which is the one `nanodo_c4` points at
(env override `SIMPLY_NANODO_C4_VOCAB`).

### `vb262144_gemma3`, `Qwen2.5`, `Qwen3`, `byte256`

`vb262144_gemma3` is `google/gemma-3-1b-pt`'s `tokenizer.model` (sha256
`1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c`, 262144
pieces) — **identical** to the `tokenizer.model` bundled with
`GEMMA-3.0-270M-PT-ORBAX` in `unkindledmonkey/simply-models`, so the OSS mirror
and the gated Gemma repo agree.

The Qwen tokenizers are staged as HuggingFace tokenizer directories from
`Qwen/Qwen2.5-Math-1.5B` and `Qwen/Qwen3-0.6B-Base`; the OSS repo's
`setup/setup_assets.py --vocabs-only` produces the same thing from the mirror
repo, either works. `byte256` needs no file.

### `vb100864_openmix_v1` (substituted, registered as `c4_spm100864`)

The port registers the substitute under its own name, `c4_spm100864`
(`config_lib.TTT_VOCAB_NAME`), rather than shadowing `vb100864_openmix_v1`, so
the two can never be confused in a config dump.

The internal vocab is `spm-100864-open_mix_v1-reserved_100-02272024.model`, a
100864-piece SentencePiece model trained on Google's internal OpenMix corpus.
**OpenMix is not public and there is no published twin** — this is the one
genuine substitution in the suite.

| option | fidelity | cost | verdict |
|---|---|---|---|
| (a) keep `vocab_size=100864`, tokenize with the public 32k T5 SPM | shape + FLOPs preserved, but ~68k rows/classes dead and bytes/token drops to 4.29 | free | rejected: changes the loss scale, wastes 68% of the embedding |
| (b) train a 100864-piece SPM on C4 | same size, same corpus as the task trains on | ~13 min (below) | **chosen** |
| (c) find a public ~100k SPM | none is exactly 100864 | — | rejected |

Recipe (`build_vocabs.build_openmix_substitute`): unigram,
`vocab_size=100864`, `character_coverage=0.9999`, `byte_fallback=True`,
`pad/eos/bos/unk = 0/1/2/3` (same id layout as `nanodo_c4`), 10 M sentences
sampled from the staged C4 train shards with a deterministic stride,
`shuffle_input_sentence=True`, `train_extremely_large_corpus=True`, all cores.

Pilots on this box (128 cores, 236 GB):

| vocab_size | sentences | chars | wall clock | peak RSS |
|---:|---:|---:|---:|---:|
| 8000 | 250 k | 63 M | 39 s | 1.5 GiB |
| 8000 | 1 M | 252 M | 123 s | 5.6 GiB |
| 32000 | 1 M | 252 M | 116 s | 5.6 GiB |

Cost is essentially **independent of `vocab_size`** (a larger target vocab means
*fewer* EM/prune rounds down from the ~1 M seed pieces) and slightly sub-linear
in corpus size (4x corpus -> 3.1x time, 3.7x RAM). Extrapolated to 10 M
sentences / 2.5 G chars: **~13 min, ~55 GiB** — far under the 2-hour budget,
hence "just train it".

`train_extremely_large_corpus=True` is mandatory above ~2^31 characters;
without it the trainer aborts with *"Input corpus too large"*. It is also why
the actual run cost more than the pilots predicted: the pilots ran with the
32-bit suffix array, the real run with the 64-bit one.

**Result** (`prepare_assets vocabs --train-openmix`, then `verify`):

```
path    $SIMPLY_VOCABS/spm-100864-c4-r100-v1.model   (+ .vocab alongside)
sha256  5c6fc1e1d535ae3e7803d3eb04f03c926875f3ee8a1fc5bbcfe666673259a10e
size    2,010,602 bytes
pieces  100864          exactly the config's vocab_size, no dead rows
ids     pad=0 eos=1 bos=2 unk=3
round-trip  ASCII ok, Unicode ok (byte_fallback makes it lossless)
cost    1503.8 s wall, 92.1 GiB peak RSS, 128 threads
```

| vocab | bytes/token on held-out C4 (2000 random validation docs) |
|---|---:|
| internal `spm-100864-open_mix_v1` | **4.547** |
| this substitute | **4.878** |
| `nanodo_c4` (32101), for scale | 4.287 |

The substitute is ~7% *more* token-efficient on C4 than the internal vocab,
which is what you would expect: it is trained on C4 itself, OpenMix is a
broader mixture. Same `vocab_size`, so **parameter count and per-step FLOPs are
identical** to the internal config and the fixed-compute accounting carries
over untouched. The token budget is unaffected: 196.6 M tokens x 4.878 B/tok =
959 MB = **1.24 train shards** of the 32 staged.

**Consequence for the task**: `pretrain_optimizer_ttt`'s metric is a
time-to-target speedup against a frozen weak-Adam anchor curve measured with
the same vocab, so a different (but same-size, same-corpus) tokenizer moves the
anchor and the candidate together. The anchors (1.3662 -> 2.7709) are still
"internal reference, not re-measured": the anchor curve must be re-run with
this vocab before the score means anything.

## Model checkpoints

Each checkpoint is staged as step-numbered directories under **the exact path
its config's `init_ckpt_dir` points at**:

```
$SIMPLY_MODELS/<ckpt_dir>/1/{_CHECKPOINT_METADATA,state,metadata}
```

`init_ckpt_step=-1` makes `checkpoint_lib.get_checkpoint_path` pick the highest
numeric step dir under `ckpt_dir`. The porting tasks also get
`$SIMPLY_MODELS/<dest>/VOCAB/` with the published tokenizer, which is what
`data_lib.FALCON_H1_VOCAB` / `RECURRENTGEMMA_VOCAB` point at.

> **`<ckpt_dir>` is not uniform — do not derive it.** Most core constants end
> in `/ORBAX` (`Qwen3-4B/ORBAX`, `Qwen2.5-Math-1.5B/ORBAX`, ...) but the Gemma
> family does not: `core.GEMMA3_1B_PT_CKPT_DIR` is `GEMMA-3.0-1B-PT-ORBAX`
> itself, so its step dirs sit **directly** under the model directory. Staging
> Gemma one level deeper fails at restore with
> `No checkpoint found in .../GEMMA-3.0-1B-PT-ORBAX`. Every `ModelSpec`
> therefore states `ckpt_dir` literally and names the `config_name` that pins
> it, and `prepare_assets verify` fails loudly when the two disagree or when
> the staged directory has no numeric step child.

| task | `ckpt_dir` under `SIMPLY_MODELS` (holds `1/`) | source | route | tensors | on disk |
|---|---|---|---|---:|---:|
| `rl_gemma3_1b`, `rl_bfcl_gemma3_1b` | `GEMMA-3.0-1B-PT-ORBAX` **(no `/ORBAX`)** | `google/gemma-3-1b-pt` (gated) | flax remap | 288 | 1.45 GiB |
| `rl_qwen2p5_math_1p5b` | `Qwen2.5-Math-1.5B/ORBAX` | `Qwen/Qwen2.5-Math-1.5B` | `hf_to_orbax` | 338 | 2.24 GiB |
| `rl_bfcl_qwen3_0p6b` | `Qwen3-0.6B-Base/ORBAX` | `Qwen/Qwen3-0.6B-Base` | `hf_to_orbax` | 310 | 0.87 GiB |
| `sampling_lcb` | `Qwen3-4B/ORBAX` | `unkindledmonkey/simply-models` | download | 398 | 5.87 GiB |
| `decode_efficiency_vf` | `Qwen3-30B-A3B-Thinking-2507/ORBAX` | `unkindledmonkey/simply-models` | download | 18867 | 42.49 GiB |
| `port_falcon_h1_0p5b` | `research_bench/Falcon-H1-0.5B-Base/ORBAX` (+ `../VOCAB`) | `tiiuae/Falcon-H1-0.5B-Base` | `hf_to_orbax` | 579 | 0.76 GiB |
| `port_recurrentgemma_2b` | `research_bench/RecurrentGemma-2B/ORBAX` (+ `../VOCAB`) | `google/recurrentgemma-2b` (gated) | `hf_to_orbax` | 484 | 4.81 GiB |

`prepare_assets verify` re-reads each one and checks three things, any of
which would otherwise only surface on an accelerator:

1. the staged path equals the registered config's `init_ckpt_dir`;
2. that path has at least one numeric step directory (otherwise
   `last_checkpoint_step` returns -1 and the run dies with "No checkpoint
   found");
3. the format recorded in the checkpoint metadata matches the format the
   config asks for (`Gemma3pFormat`, `Qwen2Format`, `FalconH1Format`,
   `RecurrentGemmaFormat`).

Current state — all seven green:

```
gemma3_1b_pt                steps=['1'] GEMMA-3.0-1B-PT-ORBAX                         config_match=True fmt_ok=True
qwen2p5_math_1p5b           steps=['1'] Qwen2.5-Math-1.5B/ORBAX                       config_match=True fmt_ok=True
qwen3_0p6b_base             steps=['1'] Qwen3-0.6B-Base/ORBAX                         config_match=True fmt_ok=True
qwen3_4b                    steps=['1'] Qwen3-4B/ORBAX                                config_match=True fmt_ok=True
qwen3_30b_a3b_thinking_2507 steps=['1'] Qwen3-30B-A3B-Thinking-2507/ORBAX             config_match=True fmt_ok=True
falcon_h1_0p5b_base         steps=['1'] research_bench/Falcon-H1-0.5B-Base/ORBAX config_match=True fmt_ok=True
recurrentgemma_2b           steps=['1'] research_bench/RecurrentGemma-2B/ORBAX   config_match=True fmt_ok=True
```

The Gemma-3-1B checkpoint was additionally restored end-to-end on CPU through
the real path (`get_config('rl_gemma3_1b')` -> `model_lib.create_model` ->
`load_checkpoint_from_dir(..., 'Gemma3pFormat')`): **340/340 parameter leaves
restored as real arrays, 0 re-initialised branches, 0.9999 B parameters**, and
`embed_linear/w`, `block_0/attn/q_proj/w`, `block_0/attn/o_proj/w` are exact
against the HF safetensors after the bf16->f32 upcast. The
zero-re-initialised-branches part matters: `model_lib` silently re-initialises
missing branches on a structural mismatch, so a wrong layout can yield a
*partly random* model without raising.

### `check-restore`: proving the checkpoint *fits* the model

`verify` proves a checkpoint is **where** the config looks. `check-restore`
proves it **fits**:

```bash
python -m tasks.research_bench.setup.prepare_assets check-restore            # all 11
python -m tasks.research_bench.setup.prepare_assets check-restore --task=rl_gemma3_1b
```

**Why it exists.** `model_lib` does not fail when a checkpoint is missing
parameter branches — it re-initialises them and says so only at INFO level:

```
Checkpoint structural mismatch detected (N valid arrays vs M leaves);
initializing missing branches and overlaying checkpoint
```

A half-mapped checkpoint therefore trains a **partly random model** and reports
a plausible-looking bad metric with no error anywhere. On a research bench that
is the worst possible failure: it looks like a weak baseline, not a bug.
`check-restore` reproduces the real path (`create_model` ->
`load_checkpoint_from_dir`) and makes the branch count the verdict —
**`reinitialised_leaves` must be 0**.

Output, as shipped:

```
pretrain_bpb_v32k        no checkpoint to restore (trains from scratch)
pretrain_bpb_byte        no checkpoint to restore (trains from scratch)
pretrain_optimizer_ttt   no checkpoint to restore (trains from scratch)
rl_gemma3_1b             gemma3_1b_pt: OK  340/340 leaves restored, 0 re-initialised, 0.9999B params [Gemma3pFormat]
rl_qwen2p5_math_1p5b     qwen2p5_math_1p5b: OK  338/338 leaves restored, 0 re-initialised, 1.5437B params [Qwen2Format]
rl_bfcl_qwen3_0p6b       qwen3_0p6b_base: OK  310/310 leaves restored, 0 re-initialised, 0.5960B params [Qwen2Format]
rl_bfcl_gemma3_1b        gemma3_1b_pt: OK  340/340 leaves restored, 0 re-initialised, 0.9999B params [Gemma3pFormat]
sampling_lcb             qwen3_4b: OK  398/398 leaves restored, 0 re-initialised, 4.0225B params [Qwen2Format]
decode_efficiency_vf     qwen3_30b_a3b_thinking_2507: SKIPPED, ~57.0 GiB > --restore-max-gib 16.0
port_falcon_h1_0p5b      falcon_h1_0p5b_base: PORT NOT IMPLEMENTED YET (expected)
port_recurrentgemma_2b   recurrentgemma_2b: PORT NOT IMPLEMENTED YET (expected)
```

The restored parameter counts match the published models (0.9999 B, 1.5437 B,
0.5960 B, 4.0225 B), which is a second, independent check that the conversion
kept every tensor.

Outcomes and what they mean:

* **`no checkpoint to restore`** — the three pretraining tasks train from
  scratch. Not a failure.
* **`PORT NOT IMPLEMENTED YET (expected)`** — `FalconH1LM` / `RecurrentGemmaLM`
  are stubs *as shipped*; writing them **is** the task. The staged checkpoint
  is fine (its tensors are there precisely to be inspected). Exit code stays 0;
  rerun the check once the port exists and it becomes a real pass/fail.
* **`NOT CHECKABLE ON CPU`** — the config's sharding needs a mesh axis a single
  CPU device cannot provide (the 30B MoE, if you force it with
  `--restore-max-gib 0`). An environment limit, not a bad checkpoint; run the
  check on the accelerator, and note `verify` still covers that checkpoint's
  layout and format. Exit code stays 0.
* **`SKIPPED`** — over `--restore-max-gib`. Exit code stays 0.
* **`FAILED`** — a genuine mismatch. Exit code is non-zero, so
  `check-restore` drops into a launch script as a pre-flight gate.

Negative control (Gemma-3 config pointed at a Qwen checkpoint) to show the
check actually bites:

```
status failed  leaves 340  restored 0  reinitialised 340
detail: 340 of 340 parameter leaves were NOT in the checkpoint; a run would
silently re-initialise them ... and train a partly random model.
```

`--restore-max-gib` (default 16) bounds the cost: a CPU restore materialises
every parameter in **float32**, so a checkpoint needs roughly 4x its bf16 size
in RAM. Qwen3-30B-A3B-Thinking-2507 is skipped by default for that reason;
`--restore-max-gib 0` forces it (~120 GB RAM).

### What `unkindledmonkey/simply-models` already has

ORBAX, downloadable verbatim: `Qwen3-0.6B` (post-trained), `Qwen3-4B`,
`Qwen3-4B-Instruct-2507`, `Qwen3-4B-Thinking-2507`, `Qwen3-30B-A3B`,
`Qwen3-30B-A3B-Instruct-2507`, `Qwen3-30B-A3B-Thinking-2507`,
`DeepSeek-R1-Distill-Qwen-1.5B`, `GEMMA-2.0-2B-PT-ORBAX`,
`GEMMA-3.0-270M-PT-ORBAX`.

Of the seven checkpoints these tasks need, **two** are there (Qwen3-4B,
Qwen3-30B-A3B-Thinking-2507). The other five are converted from public HF
repos by `prepare_assets`; we deliberately do **not** push the converted
checkpoints back to that repo, so the suite depends on a reproducible
conversion rather than on a mirror (and the Gemma weights cannot be
redistributed at all).

> **`Qwen3-0.6B` in that repo is the POST-TRAINED (instruct/chat) model.**
> `rl_bfcl_qwen3_0p6b` fixes the *pretrained* `Qwen3-0.6B-Base` — a different
> checkpoint, which is **not** in the repo. `prepare_assets` converts it from
> `Qwen/Qwen3-0.6B-Base` into `$SIMPLY_MODELS/Qwen3-0.6B-Base/ORBAX`. Pointing
> the task at `$SIMPLY_MODELS/Qwen3-0.6B/ORBAX` would silently run a
> post-trained model against a baseline calibrated on the base model.

Unrelated to these tasks but worth noting: the repo's `GEMMA-3.0-270M-PT-ORBAX`
puts its checkpoint in a `gemma-3-270m/` subdirectory instead of a numeric step
dir, so `checkpoint_lib.last_checkpoint_step` cannot find it. Rename it to `1/`
if you ever use `core.gemma3_270m()`.

### Gemma-3: why `hf_to_orbax` alone is not enough

`simply/tools/hf_to_orbax.py` writes the safetensors tensor names verbatim and
only *records* the format name; the renaming happens in
`CheckpointFormat.transforms` at restore time. That is fine for `Qwen2Format`,
which un-maps `model.layers.N....`. It does **not** work for Gemma-3:
`Gemma3pFormat` un-maps the DeepMind **flax** names
(`transformer/layer_N/attn/q_einsum/w`, ...), which the HF export does not use.
Running `hf_to_orbax --format=Gemma3pFormat` on `google/gemma-3-1b-pt` produces
a checkpoint whose every tensor `Gemma3pFormat` logs as "ignored".

`build_models.gemma3_flax_state()` does the renaming at conversion time:

| HF | flax | shape |
|---|---|---|
| `model.embed_tokens.weight` | `transformer/embedder/input_embedding` | (V, D) |
| `model.norm.weight` | `transformer/final_norm/scale` | (D,) |
| `input_layernorm` / `post_attention_layernorm` / `pre_feedforward_layernorm` / `post_feedforward_layernorm` | `pre_attention_norm` / `post_attention_norm` / `pre_ffw_norm` / `post_ffw_norm` | (D,) |
| `self_attn.q_norm` / `k_norm` | `attn/_query_norm` / `attn/_key_norm` | (H,) |
| `self_attn.q_proj.weight` (N*H, D) | `attn/q_einsum/w` | (N, D, H) |
| `self_attn.{k,v}_proj.weight` (K*H, D) | `attn/kv_einsum/w` | (2, K, D, H) |
| `self_attn.o_proj.weight` (D, N*H) | `attn/attn_vec_einsum/w` | (N, H, D) |
| `mlp.{gate,up}_proj.weight` (F, D) | `mlp/gating_einsum/w` | (2, F, D) |
| `mlp.down_proj.weight` (D, F) | `mlp/linear/w` | (F, D) |

The mapping is not guessed. `prepare_assets gemma3-selftest` downloads
`google/gemma-3-270m` plus the `GEMMA-3.0-270M-PT-ORBAX` checkpoint simply
already ships, runs the mapping on the HF weights and asserts the result is
bit-identical:

```
gemma3 mapping self-test OK: 200 tensors bit-identical to
  .../GEMMA-3.0-270M-PT-ORBAX/gemma-3-270m
```

The staged Gemma-3-1B checkpoint was also read back and compared with its
source safetensors: **288 tensors, 0 mismatched**, embedding
`bfloat16 (262144, 1152)`. 340 HF tensors collapse to 288 flax tensors (26 k/v
merges + 26 gate/up merges); 1,999,771,904 bytes of weights in, the same out.

### Gated repos

`google/gemma-3-1b-pt` and `google/recurrentgemma-2b` are `gated: manual`.
User steps:

1. Sign in on huggingface.co, open the model page, accept the licence (Gemma
   Terms of Use). Approval is immediate.
2. `pip install -U huggingface_hub && huggingface-cli login` (or export
   `HF_TOKEN=hf_...`).
3. Re-run `prepare_assets models --only gemma3_1b_pt recurrentgemma_2b`.

Without step 1 the download fails with `GatedRepoError`. Both were fetched and
converted successfully here with an accepted licence, so gating is the only
hurdle — the Gemma-3-1B conversion **was actually run and verified** (11.3 s,
1.45 GiB).

### The two porting-task checkpoints

`port_falcon_h1_0p5b` and `port_recurrentgemma_2b` ship `FalconH1Format` /
`RecurrentGemmaFormat` as deliberate stubs — writing the transform *is* the
task. The staged ORBAX therefore holds the **published tensors under their
original names**, which is exactly what the task tells the participant to
inspect:

```
research_bench/Falcon-H1-0.5B-Base/ORBAX/1/state
  579 tensors, e.g. model.layers.0.mamba.{A_log,D,conv1d.weight,...}
research_bench/RecurrentGemma-2B/ORBAX/1/state
  484 tensors, e.g. model.layers.0.{channel_pre_norm.weight,...}
```

`build_models.convert_hf_to_orbax` imports
`tasks.research_bench.checkpoint_lib` before invoking `hf_to_orbax`, so
the stub formats resolve and get recorded in the checkpoint metadata.

## Datasets

Every builder asserts its example count and field schema, and
`build_datasets.check()` re-reads each staged file the way the task's data
source will. Verified:

> **These cardinalities are load-bearing.** The validator pins them, and the
> task prompts quote them as the size of the FIXED held-out eval. If a
> re-staged dataset changes any one of **1319** (gsm8k test), **262** (MATH-500
> L4-5), **1351** (BFCL live call-required), **167** (LiveCodeBench v5), **30**
> (AIME 2025), 7473 (gsm8k train), 40315 (DeepScaleR), 8937 (ToolACE), then
> every submission for the affected task is rejected. That is why each builder
> asserts its count at build time rather than trusting the upstream repo, and
> why `prepare_assets verify` re-checks them. An upstream re-release that moves
> a count is a **code change** here (bump the constant, re-measure the
> anchors), not a silent data refresh.

| file under `SIMPLY_DATASETS` | rows | read by | source |
|---|---:|---|---|
| `gsm8k/gsm8k.json` | 7473 train / **1319** test | `simply:gsm8k_{train,test}` | `unkindledmonkey/simply-datasets` |
| `math500/test.json` | 500 (**262** at level 4-5) | `simply:math500_test_l45` | `HuggingFaceH4/MATH-500` |
| `deepscaler/deepscaler.json` | 40315 | `simply:dsr40k_train` | `agentica-org/DeepScaleR-Preview-Dataset` |
| `aime/aime_v3.json` | 1035 (**30** in 2025) | `simply:aime25` | `unkindledmonkey/simply-datasets` |
| `tooluse_rlvr/bfcl/<category>.json` | see below | `simply:bfcl_*` | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` |
| `tooluse_rlvr/toolace/toolace_bfcl.json` | 8937 | `simply:toolace_train` | `Team-ACE/ToolACE` |
| `livecodebench/livecodebench_v5.jsonl` | **167** | `simply_json:livecodebench_v5` | `livecodebench/code_generation_lite` |

`math500/test.json` is the unmodified 500-problem test split; the L4-5
restriction happens at read time in `simply:math500_test_l45`, and the count is
asserted to be exactly **262**. `aime/aime_v3.json` is the upstream archive
copied verbatim (it already contains AIME 2025 I+II); if a future archive drops
them, the builder appends them from `opencompass/AIME2025`.

### BFCL

Staged from the official HF mirror of the gorilla repo's BFCL **v3** data
(`gorilla-llm/Berkeley-Function-Calling-Leaderboard`, revision
`61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`). GitHub `main` has moved to v4 and
renamed the categories (`simple` -> `simple_python`, ...), so the HF mirror is
the stable source for the v3 category set the task fixes.

Each `<category>.json` is a JSON array of
`{id, question, function, ground_truth}` — the upstream question record with
the matching `possible_answer` merged in — which is exactly what the ported
`BFCLSource` reads (it json-dumps the nested fields itself and maps a missing
`ground_truth` to `''` for the abstention categories).

| category | rows | ground truth |
|---|---:|---|
| `simple` / `multiple` / `parallel` / `parallel_multiple` | 400 / 200 / 200 / 200 | yes |
| `irrelevance` | 240 | none (abstention) |
| `live_simple` / `live_multiple` / `live_parallel` / `live_parallel_multiple` | 258 / 1053 / 16 / 24 | yes |
| `live_irrelevance` | 882 | none (abstention) |

The four call-required live categories sum to **1351**, the exact size of the
FIXED held-out eval in `rl_bfcl_*`; `live_irrelevance` is **882**, matching the
count in the internal data_lib docstring. Non-live train split = 1240.

One upstream quirk: in `live_multiple` the question id
`live_multiple_1052-79-0` is answered under the id `live_multiple_1052-279-0`.
The builder pairs questions and answers **by line order** (what the upstream
BFCL evaluator does) and logs the mismatch instead of dropping the example.

### ToolACE

`Team-ACE/ToolACE` is a multi-turn chat dataset whose assistant turns are
`[Func(arg=value), ...]` call lists. The builder keeps rows with a single
leading user turn followed by a pure call list over functions declared in the
system prompt, and converts each reference call into the BFCL allowed-value set
`{arg: [value]}`. Result: **8937** rows (the internal docstring says "~8.5k").

The call list is split by hand rather than with `ast.parse`, because ToolACE
tool names contain spaces (`Market Trends API(...)`) and are not valid Python —
an `ast`-only parser silently keeps only 4775 rows.

### LiveCodeBench v5

`sampling_lcb` scores 167 problems. That set is exactly
`livecodebench/code_generation_lite`'s **`test5.jsonl`** (the v4->v5 increment,
contest dates 2024-09-22 .. 2025-01-04) — not the cumulative `release_v5` (880
problems) and not a date filter across files. The file is **byte-identical to
the internal copy** the task reads (557,699,297 bytes, sha256
`7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214`, 167 lines);
the builder asserts that hash before doing anything.

Staged as **JSONL in upstream file order** (first `question_id` = `abc374_c`),
one object per line, all values `str`:

```
question_id, question_title, question_content, platform, contest_id,
contest_date, difficulty, starter_code,
func_name            = json.loads(row['metadata']).get('func_name', '')
public_test_cases    = json.dumps(decoded list)
private_test_cases   = json.dumps(decoded list)
```

The private tests arrive base64 -> zlib -> pickle -> json encoded; the builder
decodes them once, so nothing downstream ever unpickles. Decoded they total
1.12 GiB — hence JSONL, so a loader can index line offsets and parse one
problem at a time. Verified totals: **167 problems, 441 public tests, 6099
private tests**.

## GCS mirroring

```bash
python -m tasks.research_bench.setup.prepare_assets all --gcs-bucket gs://my-bucket
# or, for an already-staged cache:
python -m tasks.research_bench.setup.prepare_assets mirror --gcs-bucket gs://my-bucket
```

Layout mirrors the three cache roots 1:1:

```
gs://<bucket>/assets/models/...     <->  $SIMPLY_MODELS
gs://<bucket>/assets/datasets/...   <->  $SIMPLY_DATASETS
gs://<bucket>/assets/vocabs/...     <->  $SIMPLY_VOCABS
```

`mirror_to_gcs` rsyncs with `--delete-unmatched-destination-objects`, so the
bucket is a true mirror: restaging an asset in a different layout removes the
old objects instead of leaving a ghost tree for a TPU VM to pull down. Use the
same flag when rsyncing onto a VM that already holds an older copy.

The script prints what a TPU VM should run:

```bash
gcloud storage rsync --recursive gs://<bucket>/assets/models   $HOME/.cache/simply/models
gcloud storage rsync --recursive gs://<bucket>/assets/datasets $HOME/.cache/simply/datasets
gcloud storage rsync --recursive gs://<bucket>/assets/vocabs   $HOME/.cache/simply/vocabs
export SIMPLY_MODELS=$HOME/.cache/simply/models/
export SIMPLY_DATASETS=$HOME/.cache/simply/datasets/
export SIMPLY_VOCABS=$HOME/.cache/simply/vocabs/
```

For the three pretraining tasks only the vocabs and the C4 shards are needed:

```bash
gcloud storage rsync --recursive gs://<bucket>/assets/vocabs $HOME/.cache/simply/vocabs
gcloud storage rsync --recursive gs://<bucket>/assets/datasets/c4_bin \
    $HOME/.cache/simply/datasets/c4_bin
export SIMPLY_VOCABS=$HOME/.cache/simply/vocabs/
export SIMPLY_C4_DIR=$HOME/.cache/simply/datasets/c4_bin
```

### Local copy vs reading from GCS

**Default: rsync to local disk.** Orbax can restore straight from
`gs://` (`SIMPLY_MODELS=gs://<bucket>/assets/models/`), and that is the
fallback when disk is scarce, but re-reading 42.5 GiB over the network on every
start is slower and flakier than copying it once. The C4 `.bin` shards have no
choice: they are `np.memmap`ed, so they must be local (the ported `data_lib`
copies them down when `SIMPLY_C4_DIR` is a `gs://` prefix, but an explicit
rsync first is faster and more predictable).

**Stage only what the task needs.** The full cache is 83.5 GiB; nothing needs
all of it. Use `--task=<task_id>` (or `--only`) so a 1.9M-parameter
`pretrain_bpb_byte` run does not pull down a 42.5 GiB MoE checkpoint:

| task | rsync | TPU VM boot disk |
|---|---:|---:|
| `pretrain_bpb_*` | `assets/datasets/c4_bin` + `assets/vocabs` (24 GiB, or 1.6 GiB with `--num-train-shards 2`) | 100 GB is plenty |
| `rl_*`, `port_*` | that task's model + datasets (1–5 GiB) | 100 GB |
| `sampling_lcb` | Qwen3-4B + LCB (7 GiB) | 100 GB |
| `decode_efficiency_vf` | **Qwen3-30B-A3B-Thinking-2507, 42.5 GiB** | size the v6e-8 boot disk **>= 150 GB**, or set `SIMPLY_MODELS=gs://<bucket>/assets/models/` and read it in place |

**Verified**: the whole 83.5 GiB cache was mirrored to
`gs://<your-bucket>/assets/` at ~926 MiB/s — 3 min 12 s for models/datasets/vocabs, ~30 s for the 23.8 GiB of C4 shards.

On a corp workstation `gcloud` needs
`export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)`;
`asset_lib.gcloud_env()` sets that automatically and is a no-op elsewhere.

## Integrity: `verify --hash`

> **A checkpoint can be the right size, the right layout, the right format, and
> still be corrupt — and size+mtime sync will never re-fetch it.**

This is not hypothetical. A v6e-8 run of `decode_efficiency_vf` died on a
silently corrupt *local copy* of the 30B checkpoint. The GCS mirror was
byte-exact against the staging cache (354 objects, 45,622,596,984 bytes on both
sides); the damage was done by `gcloud storage`'s **sliced object download** on
the VM — a resumed slice is not checksum-validated end to end, the file ends up
the correct size, and `rsync` (which compares size and mtime) then considers it
up to date forever. Orbax eventually failed with
`Truncated Zstd-compressed stream` inside one expert's weight. Every structural
check passed the whole time.

So the manifest records a **per-file crc32c** for every staged asset
(578 files across the 20 assets, 80 KiB of manifest). crc32c rather than
sha256 because it is what GCS stores in object metadata, which lets the same
recorded value be checked three ways:

```bash
# 1. local bytes vs the manifest
python -m tasks.research_bench.setup.prepare_assets verify --hash

# 2. + the mirrored objects, read from GCS metadata WITHOUT downloading them
python -m tasks.research_bench.setup.prepare_assets verify --hash \
    --gcs-bucket gs://my-bucket
```

```
dataset_livecodebench          local     1 files    1.1 GiB  gcs     1 objects  OK
gemma3_1b_pt                   local    12 files    1.4 GiB  gcs    12 objects  OK
qwen3_30b_a3b_thinking_2507    local   354 files   42.5 GiB  gcs   354 objects  OK
c4_bin                         local    80 files   23.8 GiB  gcs    80 objects  OK
...
```

Cost: **66 s for the whole 83.5 GiB cache** (the C `google_crc32c` runs at
~5.8 GB/s, so it is disk-bound). It needs `google-crc32c`, which
`pip install -e ".[assets]"` pulls in; `--hash` raises an ImportError naming
the package if it is missing. That is cheap enough to run after every copy
but too slow for the default path, hence the flag. Plain `verify` stays
structural and says so in its own output:

```
verify: STRUCTURAL ONLY (counts, shapes, layout, formats). It does NOT read the
bytes, so it cannot see a truncated or corrupted file that still has the right
size. Run `verify --hash` after copying the cache anywhere.
```

Demonstrated end to end — flip **one byte** in a staged file, leaving its size
unchanged (exactly the incident's signature):

```
corrupted 1 byte at offset 223532, size unchanged: True
verify            -> passes, exit 0            # structural checks see nothing
verify --hash     -> dataset_math500  FAILED
                       corrupt: test.json
                     exit 1
```

**On a TPU VM**, do both:

```bash
# belt: make gcloud validate the whole object instead of slicing it
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0
gcloud storage rsync --recursive --delete-unmatched-destination-objects \
    gs://<bucket>/assets/models $HOME/.cache/simply/models
# braces: prove the bytes arrived
python -m tasks.research_bench.setup.prepare_assets verify --hash
```

(The launcher sets the sliced-download threshold itself; `verify --hash` is
what proves it worked.)

**When `--hash` fails, re-run that asset's builder with `--force`** — it
rewrites the bytes *and* re-records their crc32c, so the manifest can only ever
describe bytes a builder produced. There is deliberately no flag that just
re-stamps the manifest from disk: its only real use would be to make a failing
check pass, which would record the corruption as the expected value.

```
verify --hash                              -> dataset_math500 FAILED
datasets --only math500 --force            -> staged (436.6 KiB in 0.6s)
verify --hash                              -> exit 0
```

`--force` works the same way for every group (`c4`, `vocabs`, `datasets`,
`models`): it ignores the "already staged" early-out and rebuilds.

## Nothing internal is redistributed

Two internal files were read **locally, for verification only**; neither is
copied into the repo or into any bucket:

* the internal 32k reference vocabulary — used to prove
  the derived `nanodo_c4` is byte-identical. What ships is the derivation from
  the public t5-data download plus the expected sha256.
* the internal 100864-piece OpenMix vocabulary
  — used only to measure its bytes/token on held-out C4 (**4.547**) as the
  reference the substitute is compared against.

## Decisions

Settled; do not re-litigate without a written reason.

1. **C4 document order.** The repacked corpus is the right documents in
   `allenai/c4` file order, not TFDS's hash-shuffled order, so absolute
   `validation_bpb` is not directly comparable to the internal runs.
   **Decision: re-measure.** The Cloud-TPU baseline campaign (`bpb_byte`,
   `bpb_v32k`, `ttt` anchor + `ttt`, 3 seeds each on v6e-4) publishes the
   numbers in `BASELINES.md`. The internal anchors stay as published, labelled
   "internal reference".
2. **`vb100864_openmix_v1` substitute.** Re-measuring the weak-Adam anchor
   curve with the shipped tokenizer is **in scope** and part of that same
   campaign; `config_lib.TTT_VOCAB_NAME` is `c4_spm100864`, so the anchor and
   the candidate are measured with the same vocab.
3. **Do not publish converted checkpoints to `unkindledmonkey/simply-models`.**
   It is not ours to publish into, and it would make the suite depend on a
   mirror instead of on a reproducible conversion. Every conversion here runs
   in under 30 s from a public repo.
4. **Ship the script, not the weights** — same reasoning, and the Gemma
   weights cannot be redistributed at all under the Gemma Terms.
5. **Mirror layout and staging**: rsync to local disk is the default; reading
   checkpoints straight from `gs://` is documented as the fallback. Stage only
   what a task needs (see the disk note under [GCS mirroring](#gcs-mirroring)).


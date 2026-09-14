# Kimi K3 in Simply

<!-- The tables below do not fit in 80 columns and cannot be wrapped. -->
<!-- disableFinding(LINE_OVER_80) -->

The text backbone of [Kimi K3](https://arxiv.org/abs/2607.24653)
(`moonshotai/Kimi-K3`, commit `9f62e4e`): 2.78 T parameters, 104 B activated,
93 layers, 1M-token context.

## Architecture

| Component | K3 | Where |
|---|---|---|
| Token mixing | 3 Kimi Delta Attention layers : 1 gated NoPE MLA, last two layers MLA (69 KDA / 24 MLA) | `utils/kda.py`, `utils/mla.py` |
| Depth mixing | Block Attention Residuals, block size 12 (8 snapshots + the running prefix sum) | `utils/attn_res.py` |
| Channel mixing | Stable LatentMoE: 896 routed experts of latent width 3584, top-16, 2 shared experts, SiTU-GLU | `utils/moe.py` |
| Position | none anywhere: MLA is NoPE, KDA is recurrent | -- |
| Weights | routed experts MXFP4 (E2M1 nibbles + E8M0 scales), everything else bf16/f32 | `utils/ckpt_format.py` |

The package is laid out as Simply itself is: the top level holds only what
Simply's own top level holds. `config_lib.py` is the config type, the mapping
from a released `config.json` and the registered deployments with the
arithmetic that sizes them; `model_lib.py` assembles the layer stack and owns
the decode protocol; `convert_hf_checkpoint.py` and `eval/decode_eval.py` are
the two entry points, each flags and `main` over a module in `utils/` or core.
Everything
they are built from is in `utils/`, where Simply keeps the same class of module
(`moe_lib.py`, `quant.py`, `position_encoding.py`): the four mixers, the
tokenizer, the chat format, the `KimiK3Format` checkpoint plugin, the
HuggingFace parameter map and converter, and the GPQA scorer. The parameters
they add up to:

| part                              | per layer | x layers | total        |
| --------------------------------- | --------- | -------- | ------------ |
| KDA token mixer                   | 443.74 M  | 69       |  30.62 B     |
| MLA token mixer                   | 232.20 M  | 24       |   5.57 B     |
| routed experts [896,3584,3072] x3 | 29.595 B  | 92       |   2.7227 T   |
| MoE latent + shared + router      | 190 M     | 92       |  17.47 B     |
| dense MLP (layer 0)               | 726.7 M   | 1        |   0.73 B     |
| embedding + untied head           |           |          |   2.35 B     |
| **total**                         |           |          | **2.7795 T** |
| **activated / token**             |           |          | **104.2 B**  |

Activated excludes the `[V, D]` embedding table (one row per token) and
includes the untied head; 16 of 896 experts fire per token.

## Correctness

`model_lib_test` runs the assembled model against
`testdata/k3_golden_tiny.npz`, a fixture produced by the *unmodified*
HuggingFace release code (`modeling_kimi_linear.py`, with fla's GPU-only ops
replaced by their torch references) on a tiny random-weight config that
exercises every K3 feature. It compares logits, every layer's residual stream,
the AttnRes snapshots and the KDA/MLA caches, for both prefill and a cached
decode step; the worst relative error is 6.4e-5 in f32. The same target covers
the decode protocol and the `LMInterface` path.

`transformers` vendors no `kimi_linear`, so unlike the other Simply ports this
one cannot build its reference in-process. `testdata/gen_k3_golden.py`
regenerates the fixture instead; it is run directly rather than as part of the
test suite (it needs torch and the release code) and it records how it ran in
`testdata/k3_golden_tiny_config.json:meta`.

Each module additionally has its own test beside it against an independent
NumPy transcription of the HF code (`utils/kda_test`, `utils/mla_test`,
`utils/moe_multi_device_test`), and the tokenizer and chat format are golden-tested against
the released encoder (`utils/tokenization_test`, `utils/lm_format_test`).

```shell
pytest simply/zoo/kimi_k3
```

## Getting the weights

The tokenizer needs nothing: the released 163,840-token
`kimi_k3_tiktoken.model` is vendored here (2.8 MB, with Moonshot's notice
beside it as `kimi_k3_tiktoken_LICENSE`), and `$SIMPLY_KIMI_K3_VOCAB_PATH`
overrides it. The weights are another matter.

Download the release (1.56 TB, ~1 h at 470 MB/s), then convert:

```shell
HF=$HOME/kimi_k3_weights
seq -f "model-%05g-of-000096.safetensors" 1 96 | xargs -P 6 -I{} \
  curl -sSL --retry 8 -C - \
  "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/{}" -o "$HF/{}"

python -m simply.zoo.kimi_k3.convert_hf_checkpoint \
  --hf_dir=$HF --out_dir=$CKPT --step=0 --verify --alsologtostderr
```

The weights are yours, not the package's: the configs carry only the checkpoint
*format*, and every run takes the path as `--ckpt_dir=$CKPT` (a 1.45 TiB
conversion lives wherever its owner has quota). `decode_eval` refuses to
start without one rather than failing at the restore, 15 minutes in.

The converter streams the safetensors, keeps the routed experts MXFP4-packed
(1.45 TiB written instead of 5.06 TiB, and ~12 h of dequantization avoided) and
tags the checkpoint `KimiK3Format`, which decodes and transposes those leaves
on restore. `--dry_run` reports the target tree and proves every tensor is
accounted for; `--verify_only --verify_layers=0` re-reads one layer out of a
written checkpoint and compares it bytewise with the source shards.

## Running it

`simply/eval/decode_eval.py` links only Simply's own config registry, so run
this package's `eval/decode_eval.py`, which is that binary plus the K3
registrations:

```shell
python -m simply.zoo.kimi_k3.eval.decode_eval \
  --experiment_config=kimi_k3_decode_ep \
  --mesh_shape=1,8,8,4 --batch_size=64 \
  --ckpt_dir=$CKPT \
  --experiment_dir=$HOME/kimi_k3_eval \
  --evaluation=ZeroShotDeepSeekQwenR1CoTBoxed \
  --datasource_name=simply:gsm8k_test \
  --k3_jit_cache_dir=$HOME/.cache/simply/k3_jit_cache
```

The released model fits a 4x4x8 slice (128 chips = 256 devices) with mesh
`1,8,8,4` = 1 replica / 8 FSDP-DP / 8 expert-parallel / 4 tensor-parallel:
20.2 GiB/device of weights, 33.6 GiB/device at batch 64 and 32k context. The
mesh must be passed explicitly -- the default mesh leaves `seq=1`, which shards
the experts 32-way and does not fit. `config_lib.hbm_budget()` answers this
for any batch, context and mesh without allocating a chip:

| deployment                      | weights | MLA $ | KDA state | AttnRes | total |
| ------------------------------- | ------- | ----- | --------- | ------- | ----- |
| B=64,  32k (`kimi_k3_decode`)   |   20.22 |  6.75 |      0.89 |    5.69 |  33.6 |
| B=32, 131k                      |   20.22 | 13.50 |      0.44 |    2.84 |  37.0 |
| B=64,  32k, mesh (1,1,32,8)     |   20.22 | 54.00 |      3.54 |   22.75 | 100.5 |
| B=64, 131k, prefilled unchunked |   20.22 | 27.00 |      0.89 |   91.00 | 139.1 |

GiB per device, bf16 weights and MLA cache, f32 KDA state, 8192-token prefill
chunks. The batch size must be a multiple of `replica * data` (8 in the
recommended mesh); `decode_eval` refuses anything else before the
15-minute startup.

Two things cost more than they look. `kimi_k3_decode_ep` is **5.4x faster per
decode step than the plain path, measured on 256 v5p chips at 2.8 T with
identical accuracy**: it runs the routed grouped
matmuls under a `shard_map` instead of letting GSPMD all-gather the expert
stacks before each one. And cold compile is 10-45 minutes per program shape
for 93 unrolled layers, so pass a persistent JIT cache. Scanning the repeating
`(KDA, KDA, MLA, KDA)` group instead of unrolling it would cut that to 24 s,
and it is exact -- but it cost +124% step time on the same hardware, so it is
not here and lands once that regression is understood.

To sample without an eval harness, K3 goes through Simply's own interface --
`model_lib.create_model(config)` then `model_lib.LMInterface(...).generate(...)`,
which is what `decode_eval` does and what `model_lib_test.LMInterfaceTest`
sets up in 30 lines. The package ships no sampler of its own. There is no server here:
`serving/vanilla_server.py` needs the multi-host and mesh work that follows
separately.

### Results on the released weights

| benchmark | score | how |
|---|---|---|
| GPQA-Diamond (198) | **88.4%** (175/198) strict, against the report's 93.5 | the paper's protocol: reasoning effort max, T=1.0, top_p=0.95, pass@1. Two passes, a 16,384-token cap and 32,768 for the 13 answers it truncated -- re-running the whole set at 32,768 is the same thing in one pass. 10 answers still never parse, 4 of them because K3 exceeds 32k output tokens. Score it with `--evaluation=KimiK3GPQADiamond`; the stock GPQA evaluation reads K3's `\boxed{\text{(B) }...}` as wrong and returns 14.65% |
| GSM8K test (1,319) | **96.74%** (1276/1319) | greedy, batch 64, 1024 decode steps, 2 h 13 min on a 4x4x16 slice |
| MATH500 (500) | **88.2%** (441/500) | greedy, 3072 decode steps; 2.6% of responses near the cap |

## Known gaps

* No paged decode: the page batcher needs an `rpa.DecodeState` for the KDA
  recurrent state and a paged latent cache for MLA. Until then, decoding uses
  the non-paged sampler, whose shared prefill window makes mixed-length batches
  expensive -- prefer a small batch and concurrency from replicas.
* The routed experts are dequantized to bf16 at restore. With expert
  parallelism on, the routed matmul is weight-bandwidth-bound, so fewer bits
  per weight is the next decode optimisation -- but **not** via
  `gmm_impl='gmm_v2'`: that kernel's `rhs_scale` multiplies an *integer* `rhs`,
  and MXFP4's `{0,.5,1,1.5,2,3,4,6}` LUT is not a uniform grid (it needs 12
  int4 magnitudes; signed int4 has 7). Every MXFP4 value *is* exact in
  `float8_e4m3`, and MXFP4's per-32-element group scale is exactly `gmm_v2`'s
  `rhs_scale` block layout, so **fp8 rhs + the E8M0 scale is bit-exact** and
  worth 16 -> 9 bits/weight (1.8x). Do not fold the scale into the fp8 exponent:
  the released scales run to 2^-17, below e4m3's subnormals.
* Training is not covered here: the configs, the checkpoint format and the
  gradients of the expert-parallel path are inference-shaped.
* Vision (MoonViT-V2) is not implemented; the converter preserves the weights
  but text-only inference never enters that path.

# Qwen3.8 in Simply

<!-- The tables below do not fit in 80 columns and cannot be wrapped. -->
<!-- disableFinding(LINE_OVER_80) -->

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): 26.9 B parameters,
64 layers, 262 144-token context, dense. The released `config.json` declares
`model_type: qwen3_5` and loads through `transformers.models.qwen3_5`.

## Architecture

| Component | Qwen3.8-27B | Where |
|---|---|---|
| Token mixing | 3 GatedDeltaNet (`linear_attention`) : 1 gated attention, `full_attention_interval = 4` (48 GDN / 16 attention) | `utils/gdn.py`, `utils/attn.py` |
| GatedDeltaNet | 16 key / 48 value heads x 128, 4-wide causal depthwise conv, chunked delta rule, float32 kernel (`mamba_ssm_dtype`) | `utils/gdn.py` |
| Attention | 24 query / 4 key-value heads x 256, q/k RMSNorm, **sigmoid** output gate | `utils/attn.py` |
| Position | interleaved partial mRoPE, sections (11, 11, 10), `partial_rotary_factor` 0.25, theta 1e7, on the attention layers only; the GatedDeltaNet is recurrent | `utils/rope.py` |
| Channel mixing | dense SwiGLU, 17408 | core `model_lib.FeedForward` |
| Weights | bf16, untied head, vocab 248 320 | `utils/ckpt_format.py` |

`output_gate_type: "swish"` is the release's one declared delta from Qwen3.5
and it names the **GatedDeltaNet output-norm** gate (silu, which HuggingFace
and this port both hardcode), *not* the attention output gate, which stays
sigmoid. Both are machine-checked: `utils/gdn_multi_device_test.py` fails if the norm gate
becomes sigmoid, `utils/attn_multi_device_test.py` fails if the attention gate becomes silu
(it moves the layer output by 88.7, against that test's 1e-4 tolerance).

The chunk size of the delta rule (32 here, 64 in HuggingFace) is not part of
the released config and does not change the result; it trades the scan length
against the chunk matmul.

The package is laid out as Simply itself is, and as `zoo/kimi_k3` is: the top
level holds only the config, the model and the entry points; everything they
are built from is in `utils/`.

## Correctness

`model_lib_test` runs the assembled model against `testdata/golden_tiny.npz`, a
fixture produced by the *unmodified* HuggingFace release code
(`transformers.models.qwen3_5`, transformers 5.16.1) on a
tiny random-weight config with the released topology: 8 layers as
`3 x GDN + attention` twice (so a GatedDeltaNet layer consumes an attention
layer's residual stream, as 15 of them do in the released model), 20 prefill
tokens crossing two chunk boundaries with a remainder, then **two** cached
decode steps. It compares the prefill logits, the residual stream entering and
leaving every layer, the logits of each decode step, and both mixers' caches
after the prefill and after each step, at a 5e-4 relative bound in float32. Parameters are stored under their *HuggingFace* names and
converted by `utils/ckpt_format.py`, so a mapping bug cannot be baked into the
fixture and confirmed by it. `testdata/gen_golden.py` regenerates the fixture
and its docstring says how; it is run directly rather than as part of the test
suite, because it needs torch and `transformers`, which nothing else here
depends on.

What the fixture cannot see, because a reader should not have to derive it:
it is float32, unsharded and chunk-8 where the release is bfloat16, sharded
and chunk-32; nothing in it is padded; its key and value head dims are equal,
so a transpose of the recurrent state is invisible (`utils/gdn_multi_device_test.py` covers
that); and its mRoPE section split is numerically inert, because HuggingFace
expands 2-D positions into three identical rows for text-only input
(`utils/rope_test.py` covers the split against three distinct rows, and the
fixture carries the release's own rotary output as a side oracle). A
bfloat16-only defect is invisible to it -- one was found in review, in the
rounding order of the residual-stream norms, and it is now pinned directly by
`model_lib_test.NormRoundingTest`.

Each module additionally has its own test beside it against an independent
float64 NumPy transcription of the HuggingFace code, and each of those tests
carries the A/B controls that make it non-vacuous -- the numbers below are the
tests' own measurements, logged when they run:

| target | agreement with the oracle | A/B controls that must fail |
|---|---|---|
| `utils:gdn_multi_device_test` | 2.0e-7 to 1.0e-6 relative in float32 across the chunked core, the recurrent step, the prefill/decode seam, the chunk remainder and padded rows; in bfloat16 the error is the operand quantization and no more | sigmoid output gate (1.79); `(1 + w)` on the gated norm (2.90); `beta` sigmoided twice (0.81); the gain applied before the rounding; the decay computed in the activation dtype; every float32 `dot_general` must ask for float32 precision (asserted from the lowered StableHLO, prefill *and* decode) |
| `utils:attn_multi_device_test` | prefill 4.561e-6 max abs (1.7e-7 of the output's scale), cached decode step 4.6e-6, packed segments 6.9e-6, padded batch 1.431e-6 | wrong gate (88.7), no q/k norm (14.7), wrong GQA grouping (29.5), core's default soft cap (1.9e-2), an infinite mask value, the released `rms_norm_epsilon`, and the concrete sharding spec of the KV cache, the output and the projections on a 4-device mesh |
| `utils:rope_test` | 1.5e-7 at the released geometry, 9.2e-3 at position 262143 in bfloat16 | cos/sin rounded to the activation dtype (HuggingFace's own order) must be measurably *worse* against the float64 oracle |
| `utils:ckpt_format_test` | all 1199 tensors of the *real* converted 27B (`testdata/qwen3p8_27b_tensor_names.json`, read off the checkpoint's Orbax metadata) map onto the 64-layer parameter tree under `jax.eval_shape`: every parameter filled, nothing extra, every shape equal, and the vision tower and the MTP head provably the only drops | renaming one released tensor; swapping the same-shape `gate_proj`/`up_proj` and `embed_tokens`/`lm_head` pairs |
| `utils:parity_test` | the released-weights gate's own statistics against hand-computed KL, cosine, top-1 and relative Frobenius; the 66 -> 65 HuggingFace hidden-state mapping element by element at the released depth; the printed report, the JSON schema and the exit code, against a stub model | four off-by-one hidden alignments; a reversed KL; a KL on raw logits; each bound made strict at its own boundary; the greedy step read one position late; a golden with no continuation, which must fail *before* the checkpoint restore rather than report a vacuous match |
| `utils:lm_format_test` | character-identical to `testdata/chat_template.jinja` rendered by jinja2, over 65 cases | a tool call inside a parameter value must not fabricate a call; an EOS-terminated turn must keep its tool call |

On the released 27B, `compare_to_hf_reference` restores the converted
checkpoint through the standard path and checks it against a golden dumped from
HuggingFace torch. It is a gate, not a report: with `--per_layer` and
`--greedy_steps=16` (neither is on by default) it exits 1 unless
`max|dlogit| <= 0.5`, `mean KL(HF||Simply) <= 5e-3`, top-1 `>= 0.99`, every
per-layer hidden cosine `>= 0.999`, and the greedy continuation matches. Two
honest limits: the canonical golden's prompt is 5 tokens, which is shorter than
one GatedDeltaNet chunk, and `--greedy_steps` re-runs the stateless forward on
a growing prefix rather than decoding from the cache, so the cached decode path
is covered by the fixture and not by this gate. A second golden, the 64-token
chat prompt, crosses a chunk boundary and legitimately needs looser bounds
(`max|dlogit|` 4.8, top-1 0.97 at 64 tokens); pass them explicitly rather than
loosening the defaults.

```shell
python -m simply.zoo.qwen3p8.compare_to_hf_reference \
    --ref_path=$GOLDEN --experiment_config=qwen3p8_27b \
    --activation_dtype=bfloat16 --per_layer --greedy_steps=16
```

Measured on the released 27B (one CPU device, bf16 activations, mesh 1,1,1;
restore 143 s, 9 min wall in total):

| check | measured | threshold |
|---|---|---|
| logits max abs diff | **0.1875** (1.5 bf16 ulp at logit magnitude ~17, where the spacing is 0.125) | <= 0.5 |
| mean KL(HF \|\| Simply) | **4.79e-4** | <= 5e-3 |
| top-1 agreement | **1.0** | >= 0.99 |
| min per-layer hidden cosine (65 states) | **0.999926** | >= 0.999 |
| 16-token greedy continuation | **16/16 identical** | must match |

```shell
pytest simply/zoo/qwen3p8
```

## Getting the weights

The released safetensors convert with the stock converter, which writes the raw
HuggingFace tensor names; `utils/ckpt_format.Qwen38Format`, named by
`config.init_ckpt_format`, maps them at restore and drops the vision tower and
the multi-token-prediction head.

```shell
python -m simply.tools.hf_to_orbax \
    --input_path=$HF --output_path=$CKPT
```

`config_lib.QWEN3P8_27B_CKPT_DIR` points at an already-converted copy under
`config_lib.MODELS_DIR`, so `--experiment_config=qwen3p8_27b` needs no
`--ckpt_dir`; pass one to override it. The tokenizer is the released
HuggingFace tokenizer, read from `data_lib.VOCABS_DIR/Qwen3.8` and registered
as the vocab `Qwen3.8` (`$SIMPLY_QWEN3P8_VOCAB_PATH` overrides the directory).

## Running it

`simply/eval/decode_eval.py` links only Simply's own config registry, so run
this package's `eval/decode_eval.py`, which is that binary plus the
Qwen3.8 registrations, a JIT-cache flag and two preflight checks: a checkpoint
directory that is not there, and a mesh the model cannot use (a batch the batch
axes cannot split, or a `model` axis that does not divide the 4 key-value
heads). Both otherwise fail 15 minutes in, after the restore and the first
compile.

Do **not** pass `--enable_prefix_caching` (it only exists on the paged driver,
which this package does not use): the prefix cache snapshots attention KV only,
so a cache hit would skip the recurrence of 48 of the 64 layers.

```shell
python -m simply.zoo.qwen3p8.eval.decode_eval \
  --experiment_config=qwen3p8_27b --lm_format=Qwen38Chat \
  --evaluation=ZeroShotDeepSeekQwenR1CoTBoxed --datasource_name=simply:aime25 \
  --mesh_shape=1,8,4 \
  --batch_size=64 --prefill_size=256 \
  --max_seq_len=66560 --max_decode_steps=65536 \
  --intermediate_decode_steps=65536 \
  --n_repeats=2 --temperature=1.0 --top_p=0.95 --top_k=20 \
  --experiment_dir=$HOME/qwen3p8_decode_eval \
  --qwen3p8_jit_cache_dir=$HOME/.cache/simply/qwen3p8_jit_cache
```

Two flags are not decoration. `--intermediate_decode_steps` equal to the decode
budget makes the sampler compile **two** programs instead of one per
decode-buffer size (seventeen of them at a 65 536-token budget), and
`--qwen3p8_jit_cache_dir=...` turns a rerun's compile into about a minute.

The whole eval is one batch: the non-paged sampler decodes a batch in lockstep
until every row has finished, so 60 samples in one batch of 64 cost what the
longest one costs, and two batches of 32 would cost twice that.

### Results on the released weights

| benchmark | score | how |
|---|---|---|
| AIME-25 avg@2 (30 problems x 2) | **56/60 = 93.3%** | the command above: 3 h 36 m on 32 chips (216 s/example), thinking mode at `reasoning_effort=xhigh`, T=1.0 / top_p 0.95 / top_k 20. 3 of the 60 samples hit the 65 536-token cap without closing `\boxed{}`; **56 of the 57 that finished are correct (98.2%)**. Two problems account for all four misses. |

The published third-party AIME-25 number for this model is 93.3% (avg@4,
T=0.6, <= 32 k tokens, a different protocol). The same configuration run
against the *shared* Qwen3.5 implementation through the paged driver scored
55/60 and 57/60 on two occasions; at n=60 the standard error is 3.2 points, so
these are the same measurement. Report the truncation count with the score:
it is the term that moves when the decode budget changes (at 32 k it was 21.7%
truncated and 82.5% correct).

## What this package does not have, and why

Everything upstream that is unreachable *here* was not ported. Each omission,
with the config field or call site that would make it reachable:

| omitted | why it cannot be reached |
|---|---|
| MoE (~460 lines: the expert FFN, the router, expert parallelism, the MoE checkpoint entries) | Qwen3.8-27B is dense: `grep -i expert` is empty on the released `config.json`, on `testdata/qwen3p8_27b_text_config.json` and on the 1 199-tensor safetensors index. `use_moe` is inherited from core's config, so `Qwen38HybridLM.setup` **refuses** it (`_refuse_unimplemented`) rather than quietly building a dense FFN. |
| Multi-token prediction (~370 lines of module + ~180 of checkpoint mapping) | The release declares one MTP layer; this port has no field for it, nothing in the decode path calls such a head, and `mtp.*` is dropped at restore. |
| The vision tower | Text-only inference never enters it; its 333 `model.visual.*` tensors are dropped at restore. |
| The ragged / paged GatedDeltaNet path and its three vendored kernels (~1 130 lines) | Nothing here can call it -- see "Known gaps" below. |
| Paged attention (`total_num_pages`, `page_size`, `lens`) | Same. |
| `utils/hf_convert.py`, `utils/hf_params.py`, `convert_hf_checkpoint.py` (kimi_k3 has all three) | The stock `simply/tools/hf_to_orbax.py` converts this model unchanged; only the restore-time format is model-specific. |
| `utils/evaluation.py` (kimi_k3 has one) | Qwen3.8 answers AIME in `\boxed{}`, which the stock `ZeroShotDeepSeekQwenR1CoTBoxed` reads; the AIME-25 result below is that scorer's, unedited. kimi_k3 needs its own because K3 writes `\boxed{\text{(B) ...}}`. |
| A vendored tokenizer (kimi_k3 vendors 2.8 MB of tiktoken) | The released HuggingFace tokenizer loads through core's `HuggingFaceVocab`. |
| Attention soft cap, sliding window, the non-mRoPE rotary branch, the ungated and un-q/k-normed attention variants, quantization, biases, tied embeddings, post-LN, per-dim scale | Absent from the release. Every field that would select one raises at model construction (`model_lib._UNIMPLEMENTED_FIELDS`, `Qwen38Attention.setup`, `Qwen38RoPE.__post_init__`) instead of being silently ignored -- these are all fields this package inherits from core's `BaseExperimentConfig`, so silence is the default failure mode. |
| Training | The configs, the checkpoint format and this package's tests are inference-shaped. `use_scan` is pinned off for the same reason (core's `eval/decode_eval.py` forces it off anyway, so a scanned stack would be unreachable code over the most intricate part of the model), and `Qwen38Chat.format_tokens` raises rather than emitting a template-less conversation for SFT. |
| `repetition_penalty` / `presence_penalty` (upstream, on the linear decode state) | No producer: the flag chain that would set them does not exist in `eval/decode_eval.py`. |

## The ragged/paged defects, and where each one lives now

Seven defects were found in the shared Qwen3.5 implementation's ragged/paged
path during the work that preceded this package. This package is dense and non-paged, so most of them are unreachable
*by construction* rather than fixed. Nothing was silently dropped:

| # | defect | status here |
|---|---|---|
| 1 | Model construction hardcoded `TransformerLM`, so a hybrid config silently built a dense transformer | **fixed upstream**: core's `eval/decode_eval.py` builds through `model_lib.create_model` -> `ModuleRegistry.get(config.model_name)`. `model_lib_test.LMInterfaceTest.test_create_model_dispatches_on_model_name` pins it. Still hardcoded in `serving/page_batcher.py:106,110` and `serving/vanilla_server.py:160` -- a core bug to file, and a prerequisite for any paged follow-up. |
| 2 | Prefill fed its padding suffix to the recurrent mixer | **carried here, and strictly stronger than the core fix**: `Qwen38HybridLM.apply` absorbs only the tokens before `extra_inputs['prefill_position']` and marks the rest as padding, which also kills the double-fold of the last prompt token that the core-side fix left open. Pinned by `DecodeTest.test_prefill_position_ignores_the_padded_tail` and by `LMInterfaceTest.test_generate_matches_greedy_reference[long_prefill]`. |
| 3 | Ragged conv1d crashed on bf16 activations (f32 conv state vs bf16 weights) | Unreachable (no ragged path). The dense conv accumulates in float32 by the same rule (`conv_accumulation_dtype`), and `utils/gdn_multi_device_test.py` runs the layer in both activation dtypes. |
| 4 | `compute_dtype` was a traced argument of a jitted kernel | Unreachable: that kernel is not here. `gdn_compute_dtype` is a Python string on the module, used at trace time. |
| 5 | `beta` was sigmoided twice on the ragged path (38.8 % kernel error) | Unreachable, and pinned anyway: `utils/gdn_multi_device_test.py` compares against the NumPy oracle, which applies the sigmoid exactly once, and the double-sigmoid variant fails it. |
| 6 | The GDN `distribution` was built from counts, not sequence-index ranges | Unreachable (no ragged kernel, no `lens`). |
| 6b | The CPU reference paged attention has the same count-vs-range bug (`utils/ragged_paged_attention.py:937`) | Still live in core at head; a bug to file, and a prerequisite for testing any paged follow-up on CPU. |
| 7 | The decode-only fast path dropped state updates when the token buffer was wider than the row count | Unreachable (that is the ragged kernel's fast path). |
| -- | `pad_to`: a hybrid decode state must survive the sampler growing the batch's horizon | **carried here**: core's `model_lib.pad_block_decode_state` singledispatch (which did not exist when the defect was found) is registered for `GatedDeltaNetDecodeState` as a no-op, and the attention layers keep core's mapping cache. Pinned by `DecodeTest.test_pad_decode_state_grows_attention_and_not_the_recurrence`. |
| -- | Prefix caching silently corrupts a hybrid model (it snapshots attention KV only) | Not reachable from here (the flag belongs to the paged driver) and not fixable from here: a `Batcher.__post_init__` guard in `serving/page_batcher.py` is a core bug to file. "Running it" above says not to pass the flag. |

Two more defects were found while building this package, both TPU-only and
invisible to any CPU test:

* the GatedDeltaNet's single-token decode step traced its einsums outside the
  `jax.default_matmul_precision('float32')` context that prefill runs in, so
  the recurrence would have been carried at bf16 precision on TPU while prefill
  ran at f32 (proven from the lowered StableHLO: prefill `[HIGHEST, HIGHEST]`,
  decode `[DEFAULT, DEFAULT]`). Fixed by moving the context inside the step.
* `output_partition=None` on a `module.EinsumLinear` is a *replicate*
  constraint, not "unset" (only the field's own `NOT_ANNOTATED` default is a
  no-op), so the upstream port -- and `zoo/kimi_k3/utils/mla.py` -- all-gather
  every projection output before the next constraint re-shards it. This package
  never passes `None`: `utils/attn.py` leaves the field at its `NOT_ANNOTATED`
  default and `utils/gdn.py` passes `NOT_ANNOTATED` explicitly where it wants
  no constraint.

## Known gaps

* **No paged decode.** `serving/page_batcher.py` hardcodes `TransformerLM`, its
  prefix cache cannot represent a recurrent state, and core's CPU reference
  paged attention still builds `distribution` from counts, so a paged path
  shipped here would be untestable and dead on arrival. `zoo/kimi_k3` documents
  the same gap for the same reason. Making it work needs, in core:
  `page_batcher`/`vanilla_server` to build the model through
  `model_lib.create_model`, `utils/ragged_paged_attention.py:937` to use
  `q.lens.shape[0]`, and a `Batcher.__post_init__` guard against prefix caching
  on a hybrid config; and, here, the three pure-JAX ragged GatedDeltaNet
  kernels (~850 lines) plus the ragged branches of `utils/gdn.py`.
* **Chunked prefill is decode-loop only.** Core's mapping KV cache replaces
  itself on a multi-token pass (`model_lib._update_kv`), so an attention layer
  cannot absorb a prompt in several calls; `LMInterface` does not need it. The
  GatedDeltaNet state *is* chunk-safe and
  `DecodeTest.test_chunked_prefill_reaches_the_same_recurrent_state` pins that.
* **No server.** `serving/vanilla_server.py` has the same `TransformerLM`
  hardcode.
* **Training is not covered**: no optimizer state, no gradient tests, and the
  parameter layout is chosen for restore-and-decode.

## Layout

```
README.md  config_lib.py(+test)  model_lib.py(+test)
           compare_to_hf_reference.py
           eval/decode_eval.py
           testdata/{gen_golden.py, golden_tiny.npz, golden_tiny_config.json,
                     chat_template.jinja, qwen3p8_27b_text_config.json,
                     qwen3p8_27b_tensor_names.json}
           utils/{gdn.py, attn.py, rope.py, ckpt_format.py, lm_format.py,
                  tokenization.py, parity.py, test_utils.py}  (+ _test.py for
                  each but test_utils)
```

Its only dependencies are Simply's own core and third-party libraries (jax,
numpy, einops, orbax, absl, jinja2 in one test); there is no dependency on
`simply/experimental/**` and none on `simply/kernels` beyond the one core
`simply/model_lib.py` already carries.

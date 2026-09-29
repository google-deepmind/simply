---
task_id: port_recurrentgemma_2b
time_limit: 5 hours
hardware: "Cloud TPU `v6e-4` (4 chips, 1 host)."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=port_recurrentgemma_2b --experiment_dir=<your gs:// experiment dir>"
---

Implement RecurrentGemma-2B, an open-weight **Griffin**-architecture language
model, in simply from an ARCHITECTURE SPECIFICATION (below), load its published
weights, and reproduce its reported GSM8K score under a FIXED eval. If time
permits, in a second stage, post-train the reproduced model to score higher.
RecurrentGemma replaces most self-attention with a **linear recurrence
(RG-LRU)** interleaved with local (sliding-window) attention -- a recurrent
primitive simply does not currently have.

You are given a self-contained architecture spec (PART A) with every
hyperparameter and scalar constant the model needs. You are NOT given any
reference implementation of this model, and you must not consult one (see
Constraints). You MAY inspect the provided checkpoint's stored tensors to
determine each tensor's shape and storage layout.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  Baseline config: `port_recurrentgemma_2b` in
    `tasks/research_bench/config_lib.py`. It
    is a unified RL(GRPO)+inline-eval run with `num_train_steps=0` by default
    (so a default run just loads the model and evaluates). The config references
    a model class `RecurrentGemmaLM` (in `tasks/research_bench/model_lib.py`) and a
    checkpoint format `RecurrentGemmaFormat` (in `tasks/research_bench/checkpoint_lib.py`)
    that are **stubs you must implement** -- running the baseline as shipped
    raises `NotImplementedError` pointing you here. The checkpoint dir, vocab,
    lm_format, and the FIXED eval are provided and must not be changed (see Eval
    / Constraints).

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
Assets (datasets, tokenizers, checkpoints) are not in the repo: fetch the ones
this task needs, then mirror them so the Cloud TPU VMs can read them
(`../ASSETS.md` lists sizes, times and licence steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=port_recurrentgemma_2b
python -m tasks.research_bench.setup.prepare_assets --task=port_recurrentgemma_2b \
    --gcs-bucket=gs://<your-bucket>
```

> **Gated checkpoint.** RecurrentGemma-2B is converted from the HuggingFace
> release, which requires accepting Google's licence and an `HF_TOKEN`;
> `prepare_assets` tells you exactly what to do if it is missing.

A baseline run, locally on CPU and tiny, to check your wiring before you spend
TPU time:

```bash
python -m tasks.research_bench.main --experiment_config=port_recurrentgemma_2b \
    --experiment_dir=/tmp/port_recurrentgemma_2b_smoke \
    --config_overlay='{"num_train_steps": 5}' --alsologtostderr
```

## What you implement

-   `RecurrentGemmaLM` (the `model_name` the config selects) in the research_bench
    `model_lib.py`, per PART A. A self-contained decoder-only model; you may
    reuse simply's building blocks (embeddings, attention, RMSNorm, linear
    layers, RoPE) from `simply.model_lib` / `simply.utils`.
-   `RecurrentGemmaFormat` in `tasks/research_bench/checkpoint_lib.py`: a checkpoint
    transform that maps the published checkpoint's stored tensors onto your
    model's parameter tree.
-   Nothing else: the config, vocab, lm_format, inline eval, and
    `final_result.json` emission are all provided.

PART A specifies the model mathematically; making incremental single-token
decoding match a full-sequence forward pass is part of the port.

## Objective

-   **Metric:** `eval_accuracy` on the FULL GSM8K test set (1319 problems) under
    the FIXED 5-shot strict-match eval (single sample at temperature 0.4),
    reported in each run's `final_result.json` as `eval_accuracy` (value at the
    LAST checkpoint). HIGHER is better.
-   **Stage 1 (port):** with `num_train_steps=0`, a CORRECT implementation loads
    the published weights and reproduces the reported number (~0.134).
-   **Stage 2 (optional, improve):** raise `num_train_steps` and design a
    post-training recipe (any method/optimizer/schedule/data, subject to
    Constraints) to lift `eval_accuracy` above the base reproduction. Same
    config, same inline eval, same metric.

> **Where the reference numbers come from.** Every number quoted above as a
> reference was measured with the original internal scaffolding on internal TPU
> hardware. This Cloud port keeps the same code, configs and eval, but some
> assets differ (see `../ASSETS.md`), so absolute values can shift a little.
> `../BASELINES.md` lists what this port measures on Cloud TPU for the same
> baseline configs; prefer those when judging your own runs.


## Eval (FIXED; do not change)

Evaluation runs INLINE (there is no separate eval job): the run evaluates the
current model on the FULL GSM8K test set (1319 problems) with a 5-shot
strict-match protocol (canonical chain-of-thought exemplars, strict anchored
answer extraction), drawing a SINGLE sample per problem at temperature 0.4, and
writes `eval_accuracy` (last checkpoint) to `final_result.json`. The evaluation
class, the few-shot prompt construction, the answer-extraction, the lm_format,
the vocab, and the validation wiring in the baseline config are FIXED and must
not be modified. (The model's published reference GSM8K score is 13.4 under
5-shot strict match; a correct port lands ~13-14% at temperature 0.4, ~15.9% at
greedy. The temperature-0.4 single-sample metric has ~0.8pt/seed variance; the
3-seed mean tightens it.)

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Fixed:** the target model (RecurrentGemma-2B, the provided checkpoint),
    the vocab (`RecurrentGemma`), the checkpoint format name
    (`RecurrentGemmaFormat`), and the final eval (5-shot strict-match, single
    sample at temperature 0.4, full GSM8K test) + its validation wiring.
-   **Implement, do not import:** you must implement the architecture yourself
    from PART A. You MAY inspect the provided checkpoint's stored tensors
    (shapes, storage layout). You must NOT consult ANY reference implementation
    of this architecture -- neither an external PyTorch/HF/JAX port nor any
    RG-LRU / linear-recurrence / Griffin / SSM implementation -- and NOT any
    implementation reachable on the internet or in this checkout (this
    includes, non-exhaustively, the `recurrentgemma` reference library and any
    recurrent/SSM sequence-layer library). Do not copy another party's port.
-   **Stage-2 data:** for post-training you may use ONLY (a) data shipped with
    the scaffolding — any data source registered under a `simply:` name in the
    checked-in `simply/data_lib.py` or
    `tasks/research_bench/data_lib.py` (as of the base changelist), restricted to
    its training splits, as provided or filtered, reformatted, or mixed — and
    (b) data you create during this session with programs you write (e.g.
    procedurally generated problems whose answers your code computes) or by
    sampling from RecurrentGemma-2B or models you train from it. Registered
    evaluation sets must not be used for training: anything named `*_test` or
    `*_eval`, and benchmark sets such as MATH500, AIME, GPQA, and MMLU. Data
    sources you add yourself count only if they read exclusively from (a) or
    (b). No other dataset may be used, including files you find on the
    filesystem (e.g. unregistered folders under `simply`'s `DATASETS_DIR`), elsewhere in the
    codebase, or on the internet. Rules: (1) **no GSM8K TEST contamination**,
    and (2) **no outputs from stronger models** — do not use any model more
    capable than RecurrentGemma-2B, including yourself, to write, rewrite,
    label, filter, score, or judge training data or rewards; code and templates
    you write are fine as long as the answers are computed by your code, but
    examples or answers you write out by hand are not. Data in (a) is approved
    as provided, regardless of how it was created. The GSM8K TRAIN split is the
    natural default.
-   **Hardware:** RecurrentGemma-2B (2.6B params + a multi-GB fp32 tied
    embedding) needs a MODEL-SHARDED mesh: the baseline config shards 4-way on
    the `model` axis (`mesh_shape` model=4), which fits a 4-chip slice; a
    replicated (model=1) embedding OOMs even a 4-chip slice. A 2-chip slice
    also works, with `--mesh_shape=1,1,2 --decoding_mesh_shape=1,1,2`; a
    single chip OOMs. Every larger slice spans more than one host and
    the provided training loop runs single-host, so a multi-host Stage-2 setup
    is yours to implement.
-   **Time limit:** complete all work within the time limit — the final
    submission job must finish before the time limit expires, so plan ahead.

## Submission

Launch each run as a 3-seed sweep (seeds MUST be 42/43/44):

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=port_recurrentgemma_2b \
  --experiment_config=<your-config-name> \
  --experiment_name="port_recurrentgemma_2b_<your-run-name>" \
  --bucket=gs://<your-bucket> --seeds=42,43,44 \
  --config-overlay='{"sampling_decode_buffer_multiple": 128, "eval_decode_buffer_multiple": 128}' \
  --tpu-type=v6e-4 --zone=<your-zone> [--spot]
```

The `--config-overlay` above holds the decode-buffer (and eval batch) settings
the reference runs used; `--task` applies them by default, they are spelled out
here so you can see what the baseline ran with.

Check the plan without touching GCP by appending `--dry-run`; watch a running
sweep with `launch_gcp status|logs|collect --bucket=... --experiment_name=...`.

Your submission has **two runs**, each launched with the command above:

-   **`port_run` (required):** your Stage-1 pure port -- the baseline config
    with your implemented model at `num_train_steps=0`. This anchors that your
    port faithfully reproduces RecurrentGemma-2B's reported GSM8K.
-   **`candidate_run` (scored):** the run whose accuracy is scored. If you
    post-trained (Stage 2), this is your best post-training run; if you only
    ported, set it equal to `port_run` (and `eval_accuracy_*` =
    `port_accuracy_*`).

Each run is one experiment dir with 3 seed runs (42/43/44), starting from the fixed
RecurrentGemma-2B checkpoint, respecting the Constraints, scored under the fixed
5-shot strict-match full-test eval, each writing `eval_accuracy`.

At the end of your research, write `submission.json` at the root of your
submitted experiment dir (`gs://<bucket>/<experiment_name>/submission.json`)
and repeat it in your final answer, with these keys:

*   `port_experiment_dir`: The GCS experiment dir of your Stage-1 pure port run
    (e.g. `"111111111"`).
*   `port_accuracy_all`: The port accuracy values across the 3 seeds (e.g.
    `"[0.0, 0.0, 0.0]"`).
*   `port_accuracy_avg`: The mean port accuracy value across the 3 seeds (e.g.
    `"0.0"`).
*   `experiment_dir`: The GCS experiment dir of your scored candidate run
    (e.g. `"gs://my-bucket/port_recurrentgemma_2b_final"`).
*   `eval_accuracy_all`: The candidate eval accuracy values across the 3 seeds
    (e.g. `"[0.0, 0.0, 0.0]"`).
*   `eval_accuracy_avg`: The mean candidate eval accuracy value across the 3
    seeds (e.g. `"0.0"`).
*   `summary`: Brief description of the port (and any post-training recipe).

## Scoring

-   **Raw metric** = mean `eval_accuracy` of `candidate_run` (5-shot
    strict-match, single sample at temperature 0.4, full GSM8K test, last
    checkpoint) over the 3 seeds. Higher is better.
-   **Port validity:** `port_run` must be a genuine `num_train_steps=0` port
    that faithfully reproduces the model's reported GSM8K; the score counts only
    if the port is valid. # PART A — ARCHITECTURE SPECIFICATION:
    RecurrentGemma-2B (Griffin)

RecurrentGemma-2B is a decoder-only language model built on the **Griffin**
architecture. It is a stack of pre-normalized residual blocks. Unlike a pure
Transformer, most blocks replace self-attention with a **linear recurrence
(RG-LRU)**; the remaining blocks use **local (sliding-window) multi-query
attention**. The two block types are interleaved on a fixed repeating pattern.
Every block also carries a gated-GeGLU MLP sublayer. Input and output embeddings
are tied, embeddings are scaled by sqrt(D), and the final logits are
soft-capped.

This spec gives the mathematical architecture and its hyperparameters. It does
NOT spell out the exact tensor storage layouts/orientations, numeric-precision
choices, or incremental-decoding (single-token) behavior — determine those from
the published checkpoint and from cheap local checks against a full-sequence
forward pass. All scalar constants below are properties of this specific model.

--------------------------------------------------------------------------------

## A.1 Global hyperparameters

| symbol            | meaning                    | value                       |
| ----------------- | -------------------------- | --------------------------- |
| D                 | model width (embedding /   | 2560                        |
:                   : residual width)            :                             :
| L                 | number of residual blocks  | ?                           |
:                   : (layers)                   :                             :
| —                 | temporal block pattern     | cycle(RECURRENT, RECURRENT, |
:                   : (period 3)                 : ATTENTION)                  :
| —                 | recurrent / attention      | ? (from L and the pattern)  |
:                   : layer split + indices      :                             :
| V                 | vocabulary size            | ?                           |
| ε                 | RMSNorm epsilon            | 1e-6                        |
| C_logit           | final-logit soft-cap value | 30.0                        |
| —                 | embedding scale            | sqrt(D)                     |
| —                 | tied input/output          | yes                         |
:                   : embeddings                 :                             :
| **Attention**     |                            |                             |
| H                 | query heads                | ?                           |
| H_kv              | key/value heads            | 1                           |
:                   : (multi-query attention)    :                             :
| d_head            | attention head dim         | ?                           |
| —                 | RoPE fraction (fraction of | 0.5 (half of each head      |
:                   : d_head rotated)            : rotated)                    :
| θ                 | RoPE base (max wavelength) | 10000                       |
| W                 | local attention window     | 2048                        |
:                   : size                       :                             :
| **RG-LRU branch** |                            |                             |
| D_lru             | RG-LRU width               | ?                           |
| G                 | RG-LRU gate blocks         | ?                           |
:                   : (block-diagonal "heads")   :                             :
| d_blk             | per-block gate width (=    | ?                           |
:                   : D_lru / G)                 :                             :
| k_conv            | depthwise causal conv1d    | ?                           |
:                   : kernel width               :                             :
| —                 | RG-LRU recurrent state     | per-channel scalar, size    |
:                   :                            : D_lru                       :
| c_a               | recurrence log-decay       | -8.0                        |
:                   : constant                   :                             :
| **MLP**           |                            |                             |
| F                 | MLP inner width (per       | ?                           |
:                   : branch)                    :                             :

Values marked `?` are omitted from this spec AND from the baseline config (they
hold a `-1` sentinel there): recover each from the provided checkpoint tensor
shapes. All are unambiguously recoverable from the values that ARE given -- `D`
labels the model-width axis of every 2-D weight; `H_kv = 1` (MQA) makes
`k_proj`'s output exactly `d_head`, then `H = q_proj_out / d_head`; `D_lru`,
`k_conv` and `G` come from the recurrent branch's linear / conv / block-diagonal
gate shapes (`d_blk = D_lru / G`); `F` from the MLP weight; and `L` + the
per-block tensors give the block split. See A.7.

No attention soft-cap, no attention biases on q/k/v (bias on the attention
output projection only), no dropout. The RG-LRU recurrence in this model is
purely REAL (no imaginary/complex component).

--------------------------------------------------------------------------------

## A.2 Fixed scalar constants of the model

These are constants of the architecture (not trained), applied at fixed points:

-   **Embedding scale** sqrt(D) ≈ 50.5964, multiplying the token-embedding
    lookup output (A.3).
-   **Attention query scale** d_head^(-1/2), multiplying the q·kᵀ logits (A.4).
-   **Logit soft-cap** C_logit = 30.0 (A.3).
-   **RG-LRU log-decay constant** c_a = -8.0, appearing in log a = c_a · r ·
    softplus(Λ) (A.5.3).

--------------------------------------------------------------------------------

## A.3 Normalization, embeddings, residual skeleton

**RMSNorm (unit-offset scale).** With a learned per-channel scale s ∈ ℝ^D and
reduction over the width axis:

```
RMSNorm(x) = ( x / sqrt(mean(x²) + ε) ) ⊙ (1 + s) ,   ε = 1e-6
```

Note the **(1 + s)** convention: the learned scale is stored as an offset from 1
(initialized at 0, so an untrained norm is the identity scaling). mean(x²) is
the mean of squares over the D channels.

**Embeddings (tied).** A single embedding table E ∈ ℝ^{V×D} is used for both
input lookup and the output head.

```
h⁰ = E[token] · sqrt(D)                    # input encode, scaled by sqrt(D)
logits_raw = h_final · Eᵀ                  # output decode, tied weights
```

**Residual block skeleton (pre-norm, two sublayers).** Each of the L blocks
receives residual-stream input x and applies a temporal sublayer (either the
RG-LRU branch A.5 or the attention branch A.4) followed by an MLP sublayer A.6,
each with its own pre-norm and each added back into the residual stream:

```
raw          = x
u            = RMSNorm_temporal(raw)               # pre-norm for temporal sublayer
t            = TemporalBlock(u)                    # RG-LRU (A.5) OR LocalAttention (A.4)
residual     = t + raw                             # first residual add
v            = RMSNorm_channel(residual)           # pre-norm for MLP sublayer
x_out        = MLP(v) + residual                   # second residual add (A.6)
```

The choice of TemporalBlock for layer i follows the A.1 pattern (RECURRENT,
RECURRENT, ATTENTION, repeating).

**Final projection.** After all L blocks:

```
h_final   = RMSNorm_final(x_L)
logits_raw = h_final · Eᵀ
logits    = C_logit · tanh(logits_raw / C_logit)   # soft-cap, C_logit = 30
```

--------------------------------------------------------------------------------

## A.4 Local attention branch (multi-query, partial RoPE, sliding window)

Applied to the pre-normed input u (shape B×T×D). Multi-query attention: H query
heads share a single (H_kv = 1) key head and a single value head. Projections
(no bias on q/k/v; bias on the output projection):

```
q = W_q · u        # D → H·d_head ;   reshape to B×T×H×d_head
k = W_k · u        # D → d_head   ;   reshape to B×T×1×d_head  (single KV head)
v = W_v · u        # D → d_head   ;   reshape to B×T×1×d_head
```

**Partial RoPE (rotate-half over the first half of each head).** RoPE is applied
to the first d_head/2 of the d_head dimensions of q and k; the remaining
d_head/2 dimensions pass through unchanged. Let z ∈ ℝ^{d_head/2} be the
first-half slice of a head vector at absolute position p. Split z into two
halves of width h = d_head/4, z = [zₐ ; z_b](zₐ = z[0:h], z_b = z[h:2h]). For
j = 0..h-1:

```
freq_j  = θ^(-(2j)/(2h)) = θ^(-j/h)          # θ = 10000
angle   = p · freq_j
zₐ'_j   = zₐ_j · cos(angle) - z_b_j · sin(angle)
z_b'_j  = z_b_j · cos(angle) + zₐ_j · sin(angle)
```

The rotated head vector is
[zₐ' ; z_b' ; z[2h:d_head]](the second half of the head is copied through). This
is the "rotate-half" (paired dim j with dim j+h) convention, restricted to the
first half of the head.

**Attention (causal + sliding window).** The single KV head is shared by all H
query heads (broadcast k, v across heads). With query position p_q and key
position p_k:

```
logits[t,s] = (q_t · k_s) · d_head^(-1/2)                          # query scale, A.2
allowed(p_q, p_k)  ⇔  (p_k ≤ p_q)  AND  (p_q ≤ p_k + W)            # causal AND within window
masked logits = logits where allowed else −∞
probs = softmax(masked logits, over key axis)                     # softmax computed in float32
ctx_t = Σ_s probs[t,s] · v_s                                      # per query head
out   = W_o · concat_heads(ctx)  + b_o                            # (H·d_head) → D, with bias
```

So each query attends to key positions p_k with 0 ≤ p_q − p_k ≤ W (i.e. the
current token plus up to W = 2048 preceding tokens). W_q,W_k,W_v have no bias;
W_o has a bias.

--------------------------------------------------------------------------------

## A.5 Recurrent (RG-LRU) branch

This is the novel primitive. Applied to the pre-normed input u (shape B×T×D).
The branch has two parallel sub-branches whose outputs are multiplied, then
projected out.

### A.5.1 Recurrent block dataflow

```
# gate ("y") branch: GeGLU gate
y  = GeLU( W_ly · u  + b_ly )                       # D → D_lru, with bias

# main ("x") branch: conv → RG-LRU
x  = W_lx · u  + b_lx                               # D → D_lru, with bias
x  = DepthwiseCausalConv1d(x)                       # width k_conv, A.5.2
x  = RG_LRU(x)                                      # A.5.3

# join and project out
o  = x ⊙ y                                          # elementwise multiply (the GeGLU gating)
out = W_lo · o + b_lo                               # D_lru → D, with bias
```

GeLU here is the **tanh-approximate** GeLU:

```
GeLU(a) = 0.5 · a · (1 + tanh( sqrt(2/π) · (a + 0.044715 · a³) ))
```

All three linear maps (W_ly, W_lx, W_lo) have biases.

### A.5.2 Depthwise causal conv1d (width k_conv)

A per-channel (depthwise) causal 1-D convolution over the D_lru channels, kernel
width k_conv, with a per-channel bias. Let the kernel be w ∈ ℝ^{k_conv×D_lru}
(w[j, c] for tap j = 0..k_conv−1, channel c) and bias b ∈ ℝ^{D_lru}. For each
output position t and channel c (with left zero-padding for t − (k_conv−1) < 0):

```
conv(x)[t, c] = b[c] + Σ_{j=0}^{k_conv−1}  w[j, c] · x[t − (k_conv−1) + j, c]
```

i.e. the LAST tap w[k_conv−1,·] multiplies the current token x[t], w[k_conv−2,·]
multiplies x[t−1], …, w[0,·] multiplies x[t−(k_conv−1)]. The convolution is
causal (no look-ahead) and mixes only within a channel (no cross-channel
mixing).

### A.5.3 RG-LRU gates and recurrence (the definition)

Let the conv output be the RG-LRU input x_t ∈ ℝ^{D_lru} at each time t.
Learnable components:

-   Two **block-diagonal linear** gate maps, each ℝ^{D_lru} → ℝ^{D_lru} with G
    diagonal blocks (block size d_blk = D_lru / G) plus per-block bias. A
    block-diagonal linear map reshapes its input into G contiguous blocks x =
    [x⁽¹⁾;…;x⁽ᴳ⁾](each x⁽ᵍ⁾ ∈ ℝ^{d_blk}), applies an independent d_blk×d_blk
    matrix and d_blk-bias per block, and concatenates: BD(x)⁽ᵍ⁾ = x⁽ᵍ⁾ · W⁽ᵍ⁾ +
    b⁽ᵍ⁾. (Equivalently a D_lru×D_lru matrix constrained to a G-block-diagonal
    sparsity pattern.)
-   A per-channel learned vector Λ ∈ ℝ^{D_lru} (the recurrence "a" parameter).

The two gates and the per-channel decay a_t are, elementwise per channel:

```
i_t   = sigmoid( BD_input(x_t) )                    # input gate,      ∈ (0,1)^{D_lru}
r_t   = sigmoid( BD_recur(x_t) )                    # recurrence gate, ∈ (0,1)^{D_lru}
log a_t = c_a · r_t ⊙ softplus(Λ)                   # c_a = -8.0 ; softplus(z)=log(1+e^z) ≥ 0
a_t   = exp( log a_t )                              # per-channel decay, ∈ (0,1]
```

Because softplus(Λ) ≥ 0 and c_a = −8 < 0, log a_t ≤ 0 and a_t ∈ (0,1].

The input is gated by i_t and multiplied by a stability normalizer sqrt(1 −
a_t²):

```
x̂_t  = x_t ⊙ i_t ⊙ sqrt( 1 − a_t² )                # gated + normalized input
```

The state is a **per-channel scalar** recurrence (one scalar per channel, NOT an
outer-product / matrix state). With hidden state h_t ∈ ℝ^{D_lru}, h_{-1} = 0:

```
h_t = a_t ⊙ h_{t-1} + x̂_t                          # elementwise linear recurrence
y_t = h_t                                           # RG-LRU output is the state itself
```

This is an elementwise (diagonal) first-order linear recurrence, applied
independently to each of the D_lru channels. A sequential (per-timestep) scan is
a correct and acceptable implementation; no associative/parallel/Pallas kernel
is required for correctness.

Note on the "heads": G refers ONLY to the block-diagonal structure of the two
gate projections (BD_input, BD_recur). The decay a_t, the normalizer, the gated
input, and the recurrence itself all act per-channel over the full D_lru width;
the recurrent state is not partitioned into heads.

**Sequence/document boundary reset.** At a position that begins a new sequence
or document (position index 0 within its segment), the recurrence is reset: the
previous-state contribution is dropped (a_t treated as 0 for that step) and the
normalizer is treated as 1 for that step, so h at a segment start equals its
gated input x_t ⊙ i_t. For a single un-packed sequence this simply means h
starts from 0 and the first token uses normalizer 1.

### A.5.4 Numerical note on the normalizer

The factor sqrt(1 − a_t²) has an unbounded derivative as a_t² → 1 (i.e. as log
a_t → 0). A faithful implementation must bound/clip the gradient of this square
root to keep training stable in low precision; the forward value is the ordinary
square root.

--------------------------------------------------------------------------------

## A.6 MLP sublayer (gated GeGLU)

Applied to the pre-normed input v (A.3). Two independent up-projections D → F
with biases, a GeLU-gated product, and a down-projection F → D with bias:

```
g   = GeLU( W_gate · v + b_gate )                   # D → F ; tanh-approx GeLU (as in A.5.1)
a   = W_up · v + b_up                               # D → F
MLP(v) = W_down · ( g ⊙ a ) + b_down               # F → D, with bias
```

(The two up-projections may be stored together as a single stacked weight of
shape 2×D×F with a 2×F bias; g uses the "gate" half, a uses the "up" half.)

--------------------------------------------------------------------------------

## A.7 Parameter inventory (logical)

Per **recurrent** layer:

-   RMSNorm_temporal scale ∈ ℝ^D
-   Recurrent branch:
    -   W_ly (D→D_lru) + bias ; W_lx (D→D_lru) + bias ; W_lo (D_lru→D) + bias
    -   conv1d kernel ∈ ℝ^{k_conv×D_lru} + bias ∈ ℝ^ {D_lru}
    -   RG-LRU: Λ ∈ ℝ^{D_lru} ; BD_input {W: G×d_blk×d_blk, b: G×d_blk} ;
        BD_recur {W: G×d_blk×d_blk, b: G×d_blk}
-   RMSNorm_channel scale ∈ ℝ^D
-   MLP: W_gate (D→F)+bias, W_up (D→F)+bias, W_down (F→D)+bias

Per **attention** layer:

-   RMSNorm_temporal scale ∈ ℝ^D
-   Attention branch: W_q (D→H·d_head), W_k (D→d_head), W_v (D→d_head) [no
    bias]; W_o (H·d_head→D) + bias
-   RMSNorm_channel scale ∈ ℝ^D
-   MLP: W_gate (D→F)+bias, W_up (D→F)+bias, W_down (F→D)+bias

Global:

-   Embedding table E ∈ ℝ^{V×D} (tied — used for both input encode and output
    head)
-   RMSNorm_final scale ∈ ℝ^D

There is no separate (untied) output head, no attention soft-cap parameter, and
no positional-embedding parameters (RoPE is parameter-free). Map these logical
roles onto the published checkpoint's tensors yourself; determine each tensor's
exact shape and storage orientation by inspecting the checkpoint.

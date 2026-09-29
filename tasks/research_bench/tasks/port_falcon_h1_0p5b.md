---
task_id: port_falcon_h1_0p5b
time_limit: 5 hours
hardware: "Cloud TPU `v6e-4` (4 chips, 1 host)."
base_commit: <the repo commit your session starts from>
validator:
  command: "python -m tasks.research_bench.validator.run_validator
    --task=port_falcon_h1_0p5b --experiment_dir=<your gs:// experiment dir>"
---

Implement a recently-released open-weight language model,
**Falcon-H1-0.5B-Base**, in simply from an ARCHITECTURE SPECIFICATION (below),
load its published weights, and reproduce its reported GSM8K score under a FIXED
eval. If time permits, in a second stage, post-train the reproduced model to
score higher. Falcon-H1 is a *parallel hybrid* decoder: every layer runs a
Transformer attention branch AND a Mamba-2 state-space branch in parallel and
sums them -- a state-space primitive simply does not currently have.

You are given a self-contained architecture spec (PART A) plus every numeric
constant the model needs: the muP multipliers and all hyperparameters are
provided as fields of the baseline config, so you do NOT have to derive them.
You are NOT given any reference implementation of this model, and you must not
consult one (see Constraints). You MAY inspect the provided checkpoint's stored
tensors to determine each tensor's shape and storage layout.

## Setup

1.  Confirm the scaffolding runs: `python -m pytest tasks/research_bench/tests -q` (CPU, no accelerator needed).
2.  Baseline config: `port_falcon_h1_0p5b` in
    `tasks/research_bench/config_lib.py`. It
    is a unified RL(GRPO)+inline-eval run with `num_train_steps=0` by default
    (so a default run just loads the model and evaluates). The config references
    a model class `FalconH1LM` (in `tasks/research_bench/model_lib.py`) and a checkpoint
    format `FalconH1Format` (in `tasks/research_bench/checkpoint_lib.py`) that are **stubs
    you must implement** -- running the baseline as shipped raises
    `NotImplementedError` pointing you here. The checkpoint dir, vocab,
    lm_format, and the FIXED eval are provided and must not be changed (see Eval
    / Constraints).

Everything runs from the repository root, after
`pip install -e ".[tpu,assets,math-eval]"`.
Assets (datasets, tokenizers, checkpoints) are not in the repo: fetch the ones
this task needs, then mirror them so the Cloud TPU VMs can read them
(`../ASSETS.md` lists sizes, times and licence steps):

```bash
python -m tasks.research_bench.setup.prepare_assets --task=port_falcon_h1_0p5b
python -m tasks.research_bench.setup.prepare_assets --task=port_falcon_h1_0p5b \
    --gcs-bucket=gs://<your-bucket>
```

A baseline run, locally on CPU and tiny, to check your wiring before you spend
TPU time:

```bash
python -m tasks.research_bench.main --experiment_config=port_falcon_h1_0p5b \
    --experiment_dir=/tmp/port_falcon_h1_0p5b_smoke \
    --config_overlay='{"num_train_steps": 5}' --alsologtostderr
```

## What you implement

-   `FalconH1LM` (the `model_name` the config selects) in the research_bench
    `model_lib.py`, per PART A. A self-contained decoder-only model; you may
    reuse simply's building blocks (embeddings, attention, RMSNorm, linear
    layers, RoPE) from `simply.model_lib` / `simply.utils`.
-   `FalconH1Format` in `tasks/research_bench/checkpoint_lib.py`: a checkpoint transform
    that maps the published checkpoint's stored tensors onto your model's
    parameter tree.
-   Nothing else: the config, vocab, lm_format, inline eval, and
    `final_result.json` emission are all provided.

PART A specifies the model mathematically; making incremental single-token
decoding match a full-sequence forward pass is part of the port.

## Objective

-   **Metric:** `eval_accuracy` on the FULL GSM8K test set (1319 problems) under
    the FIXED 5-shot strict-match eval (greedy, pass@1), reported in each run's
    `final_result.json` as `eval_accuracy` (value at the LAST checkpoint).
    HIGHER is better.
-   **Stage 1 (port):** with `num_train_steps=0`, a CORRECT implementation loads
    the published weights and scores **~0.68** under this eval harness.
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
current model on the FULL GSM8K test set (1319 problems, greedy, pass@1) with a
5-shot strict-match protocol (the protocol the published number uses), and
writes `eval_accuracy` (last checkpoint) to `final_result.json`. The evaluation
class, the few-shot prompt construction, the answer-extraction, the lm_format,
the vocab, and the validation wiring in the baseline config are FIXED and must
not be modified. (The model's published reference GSM8K score is 60.20 under
5-shot strict match. This harness is not identical to the one that produced that
figure and reads a few points higher: a correct port scores ~0.68 here, measured
across independent implementations. Treat ~0.68, not 0.60, as the target -- a
port sitting a few points below it still has a defect.)

## Constraints (violations are disqualified)

-   **Your own work, your own run.** The submitted experiment dir must hold runs **you launched yourself during this session**. Do not read, copy or
    build on another agent's workspace, snapshots, experiments, run outputs or
    artifacts. The shipped scaffolding, core `simply`, the baseline runs
    referenced in this document, and published literature are all fair game —
    another agent's solution is not.
-   **Fixed:** the target model (Falcon-H1-0.5B-**Base**, the provided
    checkpoint), the vocab (`FalconH1`), the checkpoint format name
    (`FalconH1Format`), and the final eval (5-shot strict-match, greedy, full
    GSM8K test) + its validation wiring.
-   **Implement, do not import:** you must implement the architecture yourself
    from PART A. You MAY inspect the provided checkpoint's stored tensors
    (shapes, storage layout). You must NOT consult ANY reference implementation
    of this architecture -- neither an external PyTorch/HF/JAX port nor any
    Mamba/SSM/state-space-model implementation -- and NOT any implementation
    reachable on the internet or in this checkout. Do not copy another party's port.
-   **Stage-2 data:** for post-training you may use ONLY (a) data shipped with
    the scaffolding — any data source registered under a `simply:` name in the
    checked-in `simply/data_lib.py` or
    `tasks/research_bench/data_lib.py` (as of the base changelist), restricted to
    its training splits, as provided or filtered, reformatted, or mixed — and
    (b) data you create during this session with programs you write (e.g.
    procedurally generated problems whose answers your code computes) or by
    sampling from Falcon-H1-0.5B or models you train from it. Registered
    evaluation sets must not be used for training: anything named `*_test` or
    `*_eval`, and benchmark sets such as MATH500, AIME, GPQA, and MMLU. Data
    sources you add yourself count only if they read exclusively from (a) or
    (b). No other dataset may be used, including files you find on the
    filesystem (e.g. unregistered folders under `simply`'s `DATASETS_DIR`), elsewhere in the
    codebase, or on the internet. Rules: (1) **no GSM8K TEST contamination**,
    and (2) **no outputs from stronger models** — do not use any model more
    capable than Falcon-H1-0.5B, including yourself, to write, rewrite, label,
    filter, score, or judge training data or rewards; code and templates you
    write are fine as long as the answers are computed by your code, but
    examples or answers you write out by hand are not. Data in (a) is approved
    as provided, regardless of how it was created. The GSM8K TRAIN split is the
    natural default.
-   **Time limit:** complete all work within the time limit — the final
    submission job must finish before the time limit expires, so plan ahead.

## Submission

Launch each run as a 3-seed sweep (seeds MUST be 42/43/44):

```
python -m tasks.research_bench.launch.launch_gcp \
  --task=port_falcon_h1_0p5b \
  --experiment_config=<your-config-name> \
  --experiment_name="port_falcon_h1_0p5b_<your-run-name>" \
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
    port is faithful.
-   **`candidate_run` (scored):** the run whose accuracy is scored. If you
    post-trained (Stage 2), this is your best post-training run; if you only
    ported, set it equal to `port_run` (and `eval_accuracy_*` =
    `port_accuracy_*`).

Each run is one experiment dir with 3 seed runs (42/43/44), starting from the fixed
Falcon-H1-0.5B-Base checkpoint, respecting the Constraints, scored under the
fixed 5-shot strict-match full-test eval, each writing `eval_accuracy`.

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
    (e.g. `"gs://my-bucket/port_falcon_h1_0p5b_final"`).
*   `eval_accuracy_all`: The candidate eval accuracy values across the 3 seeds
    (e.g. `"[0.0, 0.0, 0.0]"`).
*   `eval_accuracy_avg`: The mean candidate eval accuracy value across the 3
    seeds (e.g. `"0.0"`).
*   `summary`: Brief description of the port (and any post-training recipe).

## Scoring

-   **Raw metric** = mean `eval_accuracy` of `candidate_run` (5-shot
    strict-match, full GSM8K test, last checkpoint) over the 3 seeds. Higher is
    better.
-   **Port validity:** `port_run` must be a genuine `num_train_steps=0` port
    that faithfully reproduces the model's reported GSM8K; the score counts only
    if the port is valid. # PART A -- ARCHITECTURE SPECIFICATION: Falcon-H1-0.5B
    (Base)

Falcon-H1-0.5B is a decoder-only language model. Every layer is a **parallel
hybrid block**: the (pre-normalized) layer input is fed to BOTH a Transformer
self-attention branch AND a Mamba-2 state-space (SSM) branch; the two branch
outputs are summed into the residual, followed by a gated-MLP sublayer.

The published Falcon-H1-0.5B-Base weights are provided to you as the fixed
checkpoint referenced by the baseline config (`init_ckpt_dir`); you load them
directly.

## A.1 Global hyperparameters

symbol             | meaning                           | value
------------------ | --------------------------------- | -------------------
D                  | model width                       | 1024
L                  | number of layers                  | ?
V                  | vocabulary size                   | ?
**Attention**      |                                   |
H                  | query heads                       | 8
H_kv               | key/value heads (GQA)             | 2
d_head             | attention head dim                | 64
θ                  | RoPE base (theta)                 | 100000000000 (1e11)
**Mamba-2 branch** |                                   |
d_ssm              | SSM inner ("intermediate") width  | ?
n_h                | SSM heads                         | ?
d_h                | SSM head dim (= d_ssm / n_h)      | ?
n_grp              | SSM groups                        | 1
d_state            | SSM state size N                  | 128
k_conv             | SSM depthwise conv1d kernel width | ?
**MLP**            |                                   |
F                  | MLP inner width (per branch)      | ?
**Norm**           |                                   |
ε                  | RMSNorm epsilon                   | 1e-5

Values marked `?` are intentionally omitted from this spec AND from the baseline
config (they hold a `-1` sentinel there): recover each from the provided
checkpoint (tensor shapes). All are unambiguously recoverable from the values
that ARE given -- `D` anchors which axis of each 2-D weight is the model width;
`d_head` splits the attention projections into heads; `n_grp` and `d_state` pin
the Mamba conv-block split; and `d_h = d_ssm / n_h`. See A.7.

Embeddings are NOT tied (separate output head). No attention soft-cap, no logit
soft-cap. RoPE uses the "rotate-half" convention applied over the FULL head_dim.

## A.2 Scalar / per-channel multipliers (muP-style; fixed constants of the model)

Falcon-H1 applies fixed multiplicative constants at specific points (constants
of the architecture, not trained). The set and where each applies:

-   an embedding multiplier scaling the token-embedding output (A.3);
-   an lm-head multiplier scaling the final logits (A.3);
-   a key multiplier scaling the attention K projection output (A.4);
-   an attention-in multiplier scaling the attention branch input, and an
    attention-out multiplier scaling the attention branch output (A.3);
-   an ssm-in multiplier scaling the SSM branch input, and an ssm-out multiplier
    scaling the SSM branch output (A.3);
-   two MLP multipliers (one on the gate pre-activation, one on the down-proj
    output) (A.6);
-   five "ssm (zxbcdt)" multipliers: a per-channel vector applied to the SSM
    input-projection output, constant within each of its five contiguous
    sections [z (gate), x, B, C, dt] respectively — one scalar per section,
    broadcast over that section's channels (A.5.1). The NUMERIC VALUES of all of
    these are properties of this specific model and are PROVIDED to you as
    fields of the baseline config (you do not need to derive them). All
    multipliers are plain linear scalings and may be folded into the adjacent
    weights if you prefer, as long as the numerics match.

## A.3 Normalization, embeddings, residual skeleton

-   RMSNorm with a learned per-channel scale s ∈ ℝ^D, plain convention (no unit
    offset): RMSNorm(x) = x / sqrt(mean(x²) + ε) · s (mean over the width axis).
-   Embedding: h⁰ = embed_lookup(token) · embedding_multiplier.
-   Per layer with input x: u = RMSNorm_input(x) # ONE shared pre-norm feeding
    BOTH branches a = Attention(u · attn_in_mult) · attn_out_mult # A.4 m =
    SSM(u) · ssm_out_mult # A.5 (SSM also scales its input by ssm_in_mult) r =
    x + a + m # PARALLEL: both branch outputs summed into residual v =
    RMSNorm_pre_ff(r) x_out = r + MLP(v) # A.6
-   After L layers: h = RMSNorm_final(x_L); logits = OutputHead(h) ·
    lm_head_mult.

## A.4 Attention branch (GQA + full RoPE, causal)

Grouped-query attention, no biases on q/k/v/o: q = reshape(Wq(u)) → B×H×T×d_head
k = reshape(Wk(u)) · key_multiplier → B×H_kv×T×d_head # note the K scaling v =
reshape(Wv(u)) → B×H_kv×T×d_head apply RoPE (rotate-half, base θ) to the FULL
d_head of q and k. Each KV head is shared by H/H_kv = 4 query heads
(repeat/broadcast K,V). attn = softmax( q·kᵀ · d_head^(-1/2) + causal_mask ) · v
; concat heads → Wo. No sliding window (full causal), no soft-cap, no dropout.
(Wq: D→H·d_head; Wk,Wv: D→H_kv·d_head; Wo: H·d_head→D.)

## A.5 Mamba-2 (SSD) branch

Given the shared pre-normed input u. The branch scales its input by ssm_in_mult
(A.5.1). Learnable components (all bias-free linear maps except the conv, which
has a per-channel bias):

-   an input projection mapping D → P, where P = d_ssm + conv_dim + n_h,
    conv_dim = d_ssm + 2·n_grp·d_state;
-   a depthwise causal conv1d with kernel width k_conv over the conv_dim
    channels (one independent length-k_conv filter per channel), with a
    per-channel bias;
-   SSM parameters A_log ∈ ℝ^{n_h}, D_skip ∈ ℝ^{n_h}, dt_bias ∈ ℝ^{n_h};
-   an output projection mapping d_ssm → D. This model uses NO gated-RMSNorm
    inside the branch; gating is plain SiLU (A.5.4).

### A.5.1 Input projection, mup, split

s = InputProj(u · ssm_in_mult) # → P channels s = s ⊙ mup # per-channel mup
vector, A.2 split s along the channel axis into: z = s[:d_ssm] # gate, d_ssm
channels xBC = s[d_ssm : d_ssm+conv_dim] # conv_dim channels dt =
s[d_ssm+conv_dim :] # n_h channels The mup section order [z, x, B, C, dt] lines
up with this split because xBC = [x (d_ssm), B (n_grp·d_state), C
(n_grp·d_state)] concatenated.

### A.5.2 Short causal conv over xBC, then split

xBC = SiLU( DepthwiseCausalConv1d(xBC) ) # over conv_dim channels split xBC
into: x = xBC[:d_ssm]; B = xBC[d_ssm : d_ssm+n_grp·d_state]; C =
xBC[d_ssm+n_grp·d_state :] # B,C each have n_grp·d_state DepthwiseCausalConv1d:
per-channel causal convolution of width k_conv; zero left-pad, keep the first T
outputs.

### A.5.3 Selective state-space recurrence (per head)

Reshape into heads: x → B×T×n_h×d_h; with n_grp=1 the single (B,C) group is
shared by all n_h heads (broadcast B,C to every head): B,C → B×T×n_h×d_state.
Per-head continuous parameters: A = -exp(A_log) # ℝ^{n_h}, negative scalar per
head dt = softplus(dt + dt_bias) # ℝ^{B×T×n_h} Discretize and scan. For head
index and time t, with state S_t ∈ ℝ^{d_h×d_state} initialized S_{-1}=0: dA_t =
exp(dt_t · A) # scalar per head, ∈(0,1) S_t = dA_t · S_{t-1} + (dt_t · x_t) ⊗
B_t # outer product: (d_h)⊗(d_state) y_t = S_t · C_t # contract state axis →
ℝ^{d_h} y_t = y_t + D_skip · x_t # per-head skip (D_skip scalar) Collect y over
heads → B×T×d_ssm. (Equivalently: each (head, channel-of-d_h, state-of-d_state)
entry is an independent first-order linear recurrence with decay dA_t and input
dt_t·x_t·B_t; output reads out via C_t. The published implementation computes
this equivalently via a chunked scan, but the plain recurrence above is the
definition; a sequential scan is acceptable and you do not need a fast kernel.)

### A.5.4 Gate and output

y = y · SiLU(z) # plain SiLU gating (no norm) out = OutputProj(y) # d_ssm → D

## A.6 MLP sublayer (SwiGLU with multipliers)

Bias-free linear maps Wgate, Wup (D→F) and Wdown (F→D): MLP(v) = ( SiLU(
Wgate(v) · mlp_gate_mult ) ⊙ Wup(v) ) · Wdown , then · mlp_down_mult i.e. the
gate pre-activation is scaled by the MLP gate multiplier and the down-proj
output by the MLP down multiplier (A.2). SiLU(a)=a·sigmoid(a).

## A.7 Parameter inventory (logical, per layer)

Each layer: {input_layernorm scale; attention {Wq, Wk, Wv, Wo}; mamba
{input-proj, conv1d kernel, conv1d bias, A_log, D_skip, dt_bias, output-proj};
pre_ff_layernorm scale; mlp {Wgate, Wup, Wdown}}. Global: {embedding; final_norm
scale; output head (untied)}. Map these logical roles onto the published
checkpoint's tensors yourself; determine each tensor's exact shape and storage
orientation by inspecting the checkpoint.

# Running Simply on Google Cloud

This guide walks you through running Simply experiments on Google Cloud
TPU VMs, from initial setup through monitoring and collecting results.
It covers both single-host and multi-host configurations. For NVIDIA
GPUs, jump to [Running on GPU](#running-on-gpu).

## Prerequisites

- A GCP project with TPU quota (check IAM & Admin > Quotas)
- Billing enabled on the project
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- The Simply codebase cloned locally

### TPU Types

| Type | Hosts | Chips | Use case |
|------|-------|-------|----------|
| v5litepod-1 | 1 | 1 | Smoke tests, tiny models |
| v5litepod-8 | 2 | 8 | Small RL runs |
| v5litepod-16 | 4 | 16 | Full RL training (e.g. Gemma 2B) |

## 1. One-Time GCloud Setup

Set your project ID and preferred zone as shell variables:

```bash
PROJECT=your-project-id
ZONE=us-central1-a
BUCKET=gs://${PROJECT}-simply
```

### Enable APIs

```bash
gcloud services enable tpu.googleapis.com --project=$PROJECT
```

### VPC Network

If your project doesn't already have a default VPC:

```bash
gcloud compute networks create default \
    --project=$PROJECT --subnet-mode=auto
gcloud compute networks subnets update default \
    --region=us-central1 \
    --enable-private-ip-google-access \
    --project=$PROJECT
```

### Cloud NAT

If your VMs use internal-only IPs (no external IP), they need Cloud
NAT to reach the internet for pip installs and downloading assets:

```bash
gcloud compute routers create simply-router \
    --region=us-central1 \
    --network=default \
    --project=$PROJECT
gcloud compute routers nats create simply-nat \
    --router=simply-router \
    --region=us-central1 \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges \
    --project=$PROJECT
```

### Firewall Rules

Allow SSH access:

```bash
gcloud compute firewall-rules create allow-ssh \
    --network=default \
    --allow=tcp:22,icmp \
    --project=$PROJECT
```

### Service Account Permissions

The default compute service account needs roles for TPU management
and GCS access:

```bash
SA="$(gcloud iam service-accounts list \
    --project=$PROJECT \
    --filter='email:compute@developer.gserviceaccount.com' \
    --format='value(email)')"

for ROLE in roles/tpu.admin \
            roles/compute.instanceAdmin.v1 \
            roles/iam.serviceAccountUser \
            roles/storage.admin; do
  gcloud projects add-iam-policy-binding $PROJECT \
      --member="serviceAccount:$SA" --role="$ROLE"
done
```

### GCS Bucket

Create a bucket for code, assets, and experiment results:

```bash
gcloud storage buckets create $BUCKET \
    --location=us-central1 --project=$PROJECT
```

## 2. Preparing Code and Assets

### Upload Code

Package and upload the Simply codebase to GCS:

```bash
cd /path/to/simply
tar --exclude='.git' --exclude='__pycache__' \
    -czf /tmp/simply.tar.gz .
gcloud storage cp /tmp/simply.tar.gz $BUCKET/code/
```

### Upload Model Checkpoints

Model checkpoints are large (~10 GB for the whole HuggingFace repo).
Download them locally first, then upload to GCS:

```bash
# Download locally (models + datasets); --vocabs-only fetches just the
# tokenizers (~100 MB) if all you need is a from-scratch training config.
pip install ".[assets]"   # huggingface_hub, used by setup_assets.py
python setup/setup_assets.py

# Upload to GCS (example for Gemma 2B)
gcloud storage cp -r ~/.cache/simply/models/GEMMA-2.0-2B-PT-ORBAX \
    $BUCKET/models/
gcloud storage cp -r ~/.cache/simply/vocabs/ $BUCKET/vocabs/
gcloud storage cp -r ~/.cache/simply/datasets/ $BUCKET/datasets/
```

## 3. Creating a TPU VM

### Single-Host (v5litepod-1)

```bash
TPU_NAME=simply-test
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=v5litepod-1 \
    --version=tpu-ubuntu2204-base \
    --project=$PROJECT \
    --preemptible
```

### Multi-Host (v5litepod-8, v5litepod-16, etc.)

Same command, just change `--accelerator-type`:

```bash
TPU_NAME=simply-pod
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=v5litepod-16 \
    --version=tpu-ubuntu2204-base \
    --project=$PROJECT \
    --preemptible
```

Multi-host creates multiple worker VMs (e.g. v5litepod-16 = 4
workers with 4 chips each).

### Preemptible vs On-Demand

Use `--preemptible` for lower cost. Preemptible VMs can be reclaimed
at any time. See [Preemption Handling](#9-preemption-handling) for
retry strategies.

## 4. Setting Up the TPU VM

### SSH into the VM

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT \
    --worker=0
```

### Install Python 3.12

TPU VMs ship with an older Python, but Simply requires 3.12+
(`requires-python` in `pyproject.toml`):

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
```

### Download Code from GCS

```bash
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz
```

### Virtual Environment and Dependencies

The `tpu` extra pulls `jax[tpu]` (libtpu comes from PyPI, so no `-f`
index is needed):

```bash
python3.12 -m venv /tmp/simply_venv
source /tmp/simply_venv/bin/activate
cd /tmp/simply
pip install ".[tpu,tfds,gcloud]"
```

### Set Asset Paths

Simply loads models, datasets, and vocabs via `epath` which supports
GCS paths natively. Point the environment variables directly at your
GCS bucket -- no need to download assets locally:

```bash
export SIMPLY_MODELS=$BUCKET/models/
export SIMPLY_DATASETS=$BUCKET/datasets/
export SIMPLY_VOCABS=$BUCKET/vocabs/
```

## 5. Running Experiments

### Single-Host

SSH in and run directly:

```bash
cd /tmp/simply
source /tmp/simply_venv/bin/activate
export SIMPLY_MODELS=$BUCKET/models/
export SIMPLY_DATASETS=$BUCKET/datasets/
export SIMPLY_VOCABS=$BUCKET/vocabs/

# `lm_smoke_test` is the same model on synthetic data and a byte vocab:
# no asset or dataset download, a few seconds -- use it to check the TPU
# is visible before starting a real run.
python3 -m simply.main \
    --experiment_config lm_test \
    --experiment_dir /tmp/exp_1 \
    --alsologtostderr
```

### Multi-Host

For multi-host pods (v5litepod-8+), the command must run on **all
workers simultaneously**. Simply's `main.py` calls
`jax.distributed.initialize()` at startup, which coordinates across
workers.

**Step 1: Warm up SSH keys** (required before `--worker=all`):

```bash
NUM_WORKERS=4  # v5litepod-16 has 4 workers
for w in $(seq 0 $((NUM_WORKERS - 1))); do
  gcloud compute tpus tpu-vm ssh $TPU_NAME \
      --zone=$ZONE --project=$PROJECT \
      --worker=$w \
      --command="echo 'Worker $w SSH OK'" 2>&1 || true
  sleep 2
done
```

**Step 2: Run on all workers**:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT \
    --worker=all \
    --command="
cd /tmp/simply
source /tmp/simply_venv/bin/activate
export SIMPLY_MODELS=$BUCKET/models/
export SIMPLY_DATASETS=$BUCKET/datasets/
export SIMPLY_VOCABS=$BUCKET/vocabs/
python3 -m simply.main \
    --experiment_config gemma2_2b_gsm8k_seqlen2k_bs16x16_rl \
    --experiment_dir $BUCKET/experiments/my_exp \
    --alsologtostderr
"
```

### Using Config Files

Instead of registered config names, you can pass a JSON config file:

```bash
python3 -m simply.main \
    --experiment_config_path /path/to/config.json \
    --experiment_dir /tmp/exp_1 \
    --alsologtostderr
```

### Experiment Directory

You can use either a local path or a GCS path for `--experiment_dir`:

- **Local path** (`/tmp/exp_1`): Fast writes, but data is lost if
  the VM is preempted. Upload results to GCS manually after training.
- **GCS path** (`gs://my-bucket/experiments/exp_1`): Checkpoints
  and TensorBoard logs are saved directly to GCS and survive
  preemption. Required for multi-host checkpointing (each host has
  its own local filesystem, so Orbax cannot coordinate checkpoint
  saves to a local path).

For multi-host or preemptible runs, prefer a GCS experiment directory:

```bash
python3 -m simply.main \
    --experiment_config gemma2_2b_gsm8k_seqlen2k_bs16x16_rl \
    --experiment_dir gs://my-bucket/experiments/my_exp \
    --alsologtostderr
```

If using a local path, upload results to GCS after training:

```bash
gcloud storage cp -r /tmp/exp_1 $BUCKET/experiments/
```

## 6. Example: Gemma 2B GSM8K RL

This example trains Gemma 2B on GSM8K using RL (GRPO) on a
v5litepod-16. The experiment config `gemma2_2b_gsm8k_seqlen2k_bs16x16_rl`
(defined in `simply/config_lib.py`) sets:

- 5000 training steps
- `LinearWarmupConstant(value=1e-7)` learning rate
- batch of 16 questions x 16 samples, `train_max_seq_len=2048`
- Checkpoints every 40 steps
- Gemma 2B weights from `$SIMPLY_MODELS/GEMMA-2.0-2B-PT-ORBAX` and the
  `vb256128_gemma2` vocab (`$SIMPLY_VOCABS/gemma2_tokenizer.model`)

```bash
TPU_NAME=simply-pod
NUM_WORKERS=4

# Create TPU
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE --accelerator-type=v5litepod-16 \
    --version=tpu-ubuntu2204-base \
    --project=$PROJECT --preemptible

# Warm up SSH keys
for w in $(seq 0 $((NUM_WORKERS - 1))); do
  gcloud compute tpus tpu-vm ssh $TPU_NAME \
      --zone=$ZONE --project=$PROJECT \
      --worker=$w \
      --command="echo 'Worker $w OK'" 2>&1 || true
  sleep 2
done

# Setup all workers (run on all)
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT \
    --worker=all \
    --command="
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
python3.12 -m venv /tmp/simply_venv
source /tmp/simply_venv/bin/activate
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz
pip install -q '.[tpu,tfds,gcloud]'
"

# Run experiment (GCS for assets and experiment dir)
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT \
    --worker=all \
    --ssh-flag="-o ServerAliveInterval=30" \
    --command="
cd /tmp/simply
source /tmp/simply_venv/bin/activate
export SIMPLY_MODELS=$BUCKET/models/
export SIMPLY_DATASETS=$BUCKET/datasets/
export SIMPLY_VOCABS=$BUCKET/vocabs/
python3 -m simply.main \
    --experiment_config gemma2_2b_gsm8k_seqlen2k_bs16x16_rl \
    --experiment_dir $BUCKET/experiments/gemma2b_gsm8k \
    --alsologtostderr 2>&1
"
```

## 7. Common Gotchas

### `jax.distributed.initialize()` Required for Multi-Host

Without this call before any JAX operations, each host only sees its
local chips and the experiment will silently hang. Simply's `main.py`
already includes this call, but if you write custom scripts, add it
before any `jax.*` calls:

```python
import jax
jax.distributed.initialize()
```

### `grad_accum_steps` for OOM

The RL training loop materializes full logits tensors during
`compute_logprobs_fn`: shape `bf16[batch/chips, seq_len, vocab_size]`.
For Gemma 2B (vocab_size=256128), this is ~4 GB per microbatch.

Set `grad_accum_steps` to 2 or higher to shrink the microbatch; the
gradient is mathematically identical. It is not set by the RL configs,
so pass it on the command line:

```bash
python3 -m simply.main \
    --experiment_config gemma2_2b_gsm8k_seqlen2k_bs16x16_rl \
    --config_overlay '{"grad_accum_steps": 2}' \
    --experiment_dir $BUCKET/experiments/my_exp --alsologtostderr
```

### SSH Key Warmup for Multi-Host

`--worker=all` can fail if SSH keys haven't been exchanged with each
worker. Always warm up keys first by SSHing into each worker
individually (see the multi-host example above).

### `--worker=all` Buffers Output

`--worker=all` buffers ALL output from ALL workers until the command
completes. For long-running training, this means you see nothing
until it finishes (or is preempted). SSH into individual workers for
real-time monitoring (see Monitoring below).

### Multi-Host Checkpoints Require Shared Filesystem

On multi-host TPU pods, each host has its own local `/tmp`. Orbax
checkpoints require all hosts to coordinate directory creation, which
fails on local paths. Use a GCS path as `--experiment_dir` for
multi-host runs, or set `should_save_ckpt=False` in the config if
you don't need checkpoints.

## 8. Monitoring

### Single-Worker SSH Probe

SSH into a specific worker to check if training is running:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT \
    --worker=0 \
    --command="ps aux | grep 'simply.main' | grep -v grep"
```

### TensorBoard

If using a GCS experiment directory, you can view TensorBoard logs
directly:

```bash
tensorboard --logdir gs://my-bucket/experiments/my_exp
```

For local experiment directories, download the logs first:

```bash
gcloud storage cp -r $BUCKET/experiments/my_exp /tmp/
tensorboard --logdir /tmp/my_exp
```

### Key Metrics for RL Experiments

- `accuracy` - fraction of correct answers
- `pass_at_k` - fraction of questions with at least 1 correct answer
  out of `num_samples_per_example` samples
- `entropy` - token-level entropy (should decrease during RL)
- `learning_rate` - verify it's not decaying to 0

## 9. Preemption Handling

Preemptible TPU VMs can be reclaimed at any time. Use a bastion VM
with a retry loop to automatically recreate the TPU and resume
training.

### Bastion VM Pattern

A bastion VM is a lightweight VM (e.g. e2-small) that runs a startup
script to manage the TPU lifecycle. It creates the TPU, sets it up,
runs the experiment, and retries on preemption.

Save the following as `bastion_retry.sh`, replacing the variables at
the top with your own values:

```bash
#!/bin/bash
# bastion_retry.sh - Startup script for a bastion VM

TPU_NAME=simply-pod
ZONE=us-central1-a
PROJECT=your-project-id
BUCKET=gs://your-bucket-name
ACCEL_TYPE=v5litepod-16
MAX_ATTEMPTS=10
EXPERIMENT_CONFIG=gemma2_2b_gsm8k_seqlen2k_bs16x16_rl
EXPERIMENT_DIR=$BUCKET/experiments/my_experiment
NUM_WORKERS=4

SETUP_CMD="
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
python3.12 -m venv /tmp/simply_venv
source /tmp/simply_venv/bin/activate
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz
pip install -q '/tmp/simply[tpu,tfds,gcloud]'
"

RUN_CMD="
cd /tmp/simply
source /tmp/simply_venv/bin/activate
export SIMPLY_MODELS=$BUCKET/models/
export SIMPLY_DATASETS=$BUCKET/datasets/
export SIMPLY_VOCABS=$BUCKET/vocabs/
python3 -m simply.main \
    --experiment_config $EXPERIMENT_CONFIG \
    --experiment_dir $EXPERIMENT_DIR \
    --alsologtostderr 2>&1
"

for attempt in \$(seq 1 $MAX_ATTEMPTS); do
  echo "=== Attempt \$attempt/$MAX_ATTEMPTS ==="

  # Create TPU
  echo "Creating TPU $TPU_NAME..."
  gcloud compute tpus tpu-vm create $TPU_NAME \
      --zone=$ZONE --accelerator-type=$ACCEL_TYPE \
      --version=tpu-ubuntu2204-base \
      --project=$PROJECT --preemptible \
      2>&1 || { echo "Create failed, retrying..."; sleep 60; continue; }

  # Warm up SSH keys
  for w in \$(seq 0 \$((NUM_WORKERS - 1))); do
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE --project=$PROJECT \
        --worker=\$w \
        --command="echo 'Worker \$w OK'" 2>&1 || true
    sleep 2
  done

  # Setup all workers
  echo "Setting up workers..."
  gcloud compute tpus tpu-vm ssh $TPU_NAME \
      --zone=$ZONE --project=$PROJECT \
      --worker=all \
      --ssh-flag="-o ServerAliveInterval=30" \
      --command="$SETUP_CMD" 2>&1

  # Run experiment
  echo "Starting experiment..."
  gcloud compute tpus tpu-vm ssh $TPU_NAME \
      --zone=$ZONE --project=$PROJECT \
      --worker=all \
      --ssh-flag="-o ServerAliveInterval=30" \
      --command="$RUN_CMD" 2>&1
  EXIT_CODE=\$?

  # Cleanup TPU
  gcloud compute tpus tpu-vm delete $TPU_NAME \
      --zone=$ZONE --project=$PROJECT --quiet 2>&1

  if [ \$EXIT_CODE -eq 0 ]; then
    echo "=== Experiment completed successfully ==="
    break
  fi
  echo "Attempt \$attempt failed (exit code \$EXIT_CODE). Retrying..."
  sleep 60
done
```

Deploy the bastion VM:

```bash
gcloud compute instances create bastion \
    --zone=$ZONE --machine-type=e2-small \
    --project=$PROJECT \
    --network=default --scopes=cloud-platform \
    --metadata-from-file=startup-script=bastion_retry.sh
```

Monitor via serial port output:

```bash
gcloud compute instances get-serial-port-output bastion \
    --zone=$ZONE --project=$PROJECT
```

Because the experiment directory is on GCS, checkpoints survive
preemption. When the bastion recreates the TPU and restarts the
experiment, training resumes from the latest checkpoint
automatically.

## 10. Cleanup

```bash
# Delete TPU VM
gcloud compute tpus tpu-vm delete $TPU_NAME \
    --zone=$ZONE --project=$PROJECT --quiet

# Delete bastion VM (if used)
gcloud compute instances delete bastion \
    --zone=$ZONE --project=$PROJECT --quiet
```

The GCS bucket, VPC, NAT, and firewall rules persist across
experiments and don't need to be recreated.

## Running on GPU

Simply's model, training loop and (non-paged) decoding are plain
JAX/XLA and run unmodified on NVIDIA GPUs. The hand-written Pallas
kernels are TPU-only, so a few config options must stay off; see
[GPU feature support](#gpu-feature-support) below.

### Install

```bash
# Simply requires Python 3.12+; the cu129 Deep Learning VM images ship it.
python3.12 -m venv ~/simply_venv && source ~/simply_venv/bin/activate
cd /path/to/simply
pip install ".[gpu,tfds,math-eval,assets]"  # gpu extra = jax[cuda13]
python -c "import jax; print(jax.devices())"
```

`jax[cuda13]` needs an NVIDIA driver at least as new as CUDA 13's
(R580+). On an older driver install the CUDA 12 build instead
(`pip install -U "jax[cuda12]"`); see the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html).

### Creating a GPU VM

GPUs come attached to the machine type for the accelerator-optimized
families, so no `--accelerator` flag is needed: `g2-standard-24`
(2x L4), `a2-highgpu-1g` (1x A100 40GB), `a3-highgpu-8g` (8x H100).
Deep Learning VM images ship the driver and CUDA toolkit
pre-installed; list the families and pick one whose driver is new
enough for your JAX build:

```bash
gcloud compute images list --project=deeplearning-platform-release \
    --filter="family~'common-cu'" --format="value(family)" | sort -u
```

```bash
PROJECT=your-project-id
ZONE=us-central1-a

gcloud compute instances create simply-gpu \
    --project=$PROJECT --zone=$ZONE \
    --machine-type=g2-standard-24 \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"

gcloud compute ssh simply-gpu --zone=$ZONE --project=$PROJECT
```

GPU quota is per-region and per-GPU-type (IAM & Admin > Quotas,
e.g. `NVIDIA_L4_GPUS`). Add `--provisioning-model=SPOT` for
preemptible pricing.

### Training smoke test

```bash
nvidia-smi
# Synthetic data, byte vocab: no download, ~10 s.
python -m simply.main --experiment_config lm_smoke_test \
    --experiment_dir /tmp/exp_0 --alsologtostderr

# The real thing: IMDB via TFDS + the Qwen3 tokenizer.
python setup/setup_assets.py --vocabs-only   # Qwen3 tokenizer, ~100 MB
python -m simply.main \
    --experiment_config lm_test \
    --experiment_dir /tmp/exp_1 \
    --alsologtostderr
```

`lm_test` downloads the TFDS `imdb_reviews` dataset on first run, so
the VM needs outbound internet (external IP or Cloud NAT). Everything
else is identical to the TPU flow above: `SIMPLY_MODELS`,
`SIMPLY_DATASETS` and `SIMPLY_VOCABS` may point at `gs://` paths, and
`--experiment_dir` may be a GCS path.

### GPU feature support

| Feature | Config field | GPU |
|---------|--------------|-----|
| Transformer training + eval (dense) | -- | works (XLA) |
| Non-paged sampling / `simply.eval.decode_eval` | -- | works (XLA) |
| MoE grouped matmul | `gmm_impl='ragged_dot'` | works (XLA `jax.lax.ragged_dot`) |
| Flash attention | `use_flash_attention=True` | **unsupported** (Pallas TPU splash attention) |
| MoE megablox / quantized matmul | `gmm_impl='megablox'`, `'gmm_v2'` | **unsupported** (Pallas TPU) |
| Weight-only quantized MoE decode | `ffn_weight_quant='int8'/'int4'` | **unsupported** (forces `gmm_v2`) |
| Paged attention / paged serving | `page_size>0`, `simply/serving/page_server.py`, `simply.eval.page_decode_eval` | works, slowly (reference implementation, see below) |
| KV cache quantization | `kv_cache_quant` | works, slowly (same reference path) |
| Cross-chip resharding kernel | `simply/kernels/reshard.py` | **unsupported** (TPU remote DMA) |

The unsupported rows fail loudly when the kernel is lowered (Mosaic
errors such as `Unsupported TPU device kind`); they never degrade
silently into a slow fallback.

Paged attention is the exception: off TPU,
`utils/ragged_paged_attention.py` calls
`kernels.ragged_paged_attention.ref_ragged_paged_attention`, a plain
XLA implementation that is correct but costs
`O(max_num_seqs * max_num_tokens * pages_per_seq * page_size)` -- it
masks whole page allocations instead of slicing each sequence's KV.
Paged serving and paged eval therefore run on GPU (and CPU), but
expect a large slowdown versus the TPU kernel; `vanilla_server` and
`decode_eval` are the faster non-paged route.

Registered configs that turn on TPU-only features need an override on
GPU. All `deepseek_qwen2_*_rl`, `gemma3_12b_it_dsr40k_b2k_l10k_rl` and
`gemma2_2b_gsm8k_seqlen2k_bs32x16_rl` set `use_flash_attention=True`:

```bash
python -m simply.main \
    --experiment_config deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl \
    --config_overlay '{"use_flash_attention": false}' \
    --experiment_dir /tmp/exp_2 --alsologtostderr
```

The Qwen3 MoE configs (`qwen3_30b_a3b`, `qwen3_235b_a22b`, and their
`_thinking_2507` variants) set `gmm_impl='megablox'`; override with
`--config_overlay '{"gmm_impl": "ragged_dot"}'`. The
`*_colocate_async_rl` configs set `use_flash_attention=True` as well,
so they need the same flash-attention override; their paged sampler
then runs through the reference implementation above.

Activations default to bfloat16 (`activation_dtype_name`), which needs
Ampere or newer (A100/L4/H100). On older cards (T4/V100) run with
`--config_overlay '{"activation_dtype_name": "float32"}'`.

### Multi-GPU sharding

Sharding is expressed as a `(replica, data, model)` mesh over
`jax.devices()`, so it is accelerator-agnostic. On one host with N
GPUs:

```bash
# FSDP-style: shard params and batch over 8 GPUs
python -m simply.main --experiment_config lm_test \
    --mesh_shape 1,8,1 \
    --config_overlay '{"batch_size": 8, "model_dim": 64, "n_heads": 4, "per_head_dim": 16}' \
    --experiment_dir /tmp/exp_3 --alsologtostderr

# 2-way data x 4-way tensor parallel
python -m simply.main --experiment_config lm_test \
    --mesh_shape 1,2,4 \
    --config_overlay '{"batch_size": 8, "model_dim": 64, "n_heads": 4, "per_head_dim": 16}' \
    --experiment_dir /tmp/exp_4 --alsologtostderr
```

(`lm_test` is deliberately tiny -- `model_dim=8`, `n_heads=2` -- so the
overlay widens it enough to be shardable. Real configs need no such
override.)

Sizing rules (`config_lib.get_default_mesh_shape` picks the mesh when
`--mesh_shape` is omitted: `data = gcd(model_dim, num_devices)` and
`replica = rest` for training):

- `batch_size` must be divisible by `replica * data`.
- `data` shards `model_dim` (and the batch); `model` shards `n_heads`,
  the FFN expansion dim and the vocab (see `BaseSharding` in
  `simply/config_lib.py`), so each must divide the axis size.
- Prefer the `data` (FSDP) axis first; add `model` parallelism only
  when one GPU cannot hold the parameters and optimizer state.

Multi-node GPU is *not* covered by the TPU instructions:
`simply/main.py` calls `jax.distributed.initialize()` with no
arguments, which only works when JAX can auto-detect the cluster
(Slurm, Open MPI, mpi4py or Kubernetes). On plain GCE VMs the call
raises `ValueError` and Simply falls back to a single-host run, so use
one multi-GPU VM (up to 8 GPUs) or launch the job under Slurm/MPI/GKE.

## Running on GKE with XPK

As an alternative to managing TPU VMs directly, you can run Simply
on GKE (Google Kubernetes Engine) clusters with TPU node pools using
[XPK](https://github.com/AI-Hypercomputer/xpk). XPK handles Docker
image building, job scheduling, and multi-host coordination
automatically.

### Prerequisites

- A GKE cluster with a TPU node pool already provisioned
- [XPK](https://github.com/AI-Hypercomputer/xpk) installed
  (`pip install xpk`)
- Docker installed and authenticated to push to GCR/Artifact
  Registry
- `kubectl` configured for your cluster
  (`gcloud container clusters get-credentials ...`)

### Setting Environment Variables

Set these once per shell session (or in your shell profile).
Replace the values with your own project, cluster, and bucket:

```bash
export PROJECT=your-gcp-project-id
export CLUSTER=your-gke-cluster-name
export ZONE=us-central1
export TPUTYPE=v4-8
export BUCKET=your-gcs-bucket-name
```

### Building the Docker Image

Simply provides a Dockerfile at `scripts/Dockerfile.simply` that
pre-installs JAX with TPU support and all Simply dependencies:

```bash
cd /path/to/simply

# Build the image
docker build -f scripts/Dockerfile.simply \
    -t gcr.io/$PROJECT/simply-jax-tpu:latest .

# Push to your project's container registry
docker push gcr.io/$PROJECT/simply-jax-tpu:latest
```

The Dockerfile installs dependencies in a separate layer for fast
rebuilds. When the workload starts, the launch script runs
`uv pip install --system .` inside the container to install Simply
itself from the source tree copied by XPK's `--script-dir` flag.

### Launching a Workload with a Registered Config

The simplest way to launch is with a registered config name. `lm_test`
is a tiny from-scratch transformer, so no checkpoint upload is needed;
`--config_overlay` (everything after `--` is forwarded to
`simply.main`) turns checkpointing off and shortens the run:

```bash
./scripts/launch_gke.sh \
    --config lm_test \
    --project $PROJECT \
    --cluster $CLUSTER \
    --zone $ZONE \
    --tpu-type $TPUTYPE \
    --image gcr.io/$PROJECT/simply-jax-tpu:latest \
    -- --config_overlay '{"num_train_steps": 20, "should_save_ckpt": false}'
```

On a large slice also raise `batch_size` in the overlay: it must be
divisible by `replica * data` of the mesh (see
[Multi-GPU sharding](#multi-gpu-sharding) for the same rules).

To preview the XPK command without submitting, add `--dry-run`.
`--train-steps N` is a shorthand for a `num_train_steps` overlay and
cannot be combined with your own `--config_overlay`.

#### Common Options

| Flag | Env Variable | Default | Description |
|------|-------------|---------|-------------|
| `--zone ZONE` | `SIMPLY_XPK_ZONE` | `us-central1` | GCP zone/region |
| `--tpu-type TYPE` | `SIMPLY_XPK_TPU_TYPE` | `v4-8` | TPU accelerator |
| `--num-slices N` | `SIMPLY_XPK_NUM_SLICES` | `1` | Number of slices |
| `--priority PRI` | `SIMPLY_XPK_PRIORITY` | `medium` | Priority |
| `--train-steps N` | | config's value | Override `num_train_steps` |
| `--name NAME` | | auto | Custom workload name |
| `--spot` | | (default) | Use spot instances |
| `--on-demand` | | | Use on-demand instances |
| `--dry-run` | | | Print xpk command only |

### Profiling with XProf

To collect XProf traces, pass `--profile` and set a GCS bucket for
trace storage:

```bash
export SIMPLY_XPK_GCS_BUCKET=gs://my-bucket/profiles

./scripts/launch_gke.sh --config lm_test --profile \
    --project $PROJECT --cluster $CLUSTER \
    --image gcr.io/$PROJECT/simply-jax-tpu:latest
```

This sets `JAX_PROFILER_LOG_DIR` inside the container and saves
worker logs to the GCS bucket. By default, profiling starts after
5 warmup steps and captures 3 steps (configurable via
`--profile-warmup` and `--profile-steps`).

### Managing Workloads

List, monitor, and delete workloads:

```bash
# List all simply-* workloads on the cluster
./scripts/launch_gke.sh \
    --project $PROJECT --cluster $CLUSTER --zone $ZONE \
    --list

# Stream logs from a running workload
./scripts/launch_gke.sh \
    --project $PROJECT --cluster $CLUSTER --zone $ZONE \
    --logs simply-lm-test-0311

# Delete a workload
./scripts/launch_gke.sh \
    --project $PROJECT --cluster $CLUSTER --zone $ZONE \
    --delete simply-lm-test-0311
```

You can also use `kubectl` directly for lower-level diagnostics:

```bash
# List pods for a workload
kubectl get pods \
    -l "jobset.sigs.k8s.io/jobset-name=simply-lm-test-0311"

# Check container logs (replace POD_NAME with actual pod name)
kubectl logs POD_NAME --all-containers 2>&1 | tail -50

# Describe a pod for event details (image pull errors, scheduling)
kubectl describe pod POD_NAME
```

### Docker Access

The launch script checks for Docker access and falls back to
`sg docker` if the current user isn't in the docker group. If
Docker is not accessible at all, run:

```bash
sudo usermod -aG docker $USER && newgrp docker
```

### GKE Troubleshooting

#### Container can't find local files

XPK workloads run inside containers with their own filesystem.
Local paths like `/tmp/config.json` on your machine are not
accessible. Upload config files and assets to GCS and use
`gs://` paths.

#### `ModuleNotFoundError` for a Python package

If a package is missing in the container, add it to
`scripts/Dockerfile.simply`, rebuild, push, and relaunch. The image
covers the `tfds`, `gcloud`, `assets`, `math-eval`, `serving` and `zoo`
extras; anything
else your data pipeline or reward function imports has to be added
there.
#### `Found incomplete checkpoint` / Orbax validation error

Orbax uses `commit_success.txt` marker files to validate
checkpoints on GCS. HuggingFace-hosted checkpoints don't include
this file. Create it manually:

```bash
touch /tmp/commit_success.txt
gcloud storage cp /tmp/commit_success.txt \
    gs://$BUCKET/path/to/checkpoint/1/commit_success.txt
```

#### `FileNotFoundError: tokenizer_config.json`

The launch script downloads Qwen3 tokenizer files at runtime.
If you use a different tokenizer, you may need to add a similar
download step to `launch_gke.sh` or pre-bake the tokenizer files
into the Docker image.


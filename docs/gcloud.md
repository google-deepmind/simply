# Running Simply on Google Cloud TPUs

This guide walks you through running Simply experiments on Google Cloud
TPU VMs, from initial setup through monitoring and collecting results.
It covers both single-host and multi-host configurations.

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

Model checkpoints are large (several GB). Download them locally
first, then upload to GCS:

```bash
# Download locally
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

### Install Python 3.11

TPU VMs ship with Python 3.10, but Simply requires 3.11+ (uses
`typing.Self`):

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

### Virtual Environment and Dependencies

```bash
python3.11 -m venv /tmp/simply_venv
source /tmp/simply_venv/bin/activate
pip install -U 'jax[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
pip install google-cloud-storage  # for TensorBoard gs:// support
```

### Download Code from GCS

```bash
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz
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
    --experiment_config gemma2_2b_gsm8k_2k_rl_16 \
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
    --experiment_config gemma2_2b_gsm8k_2k_rl_16 \
    --experiment_dir gs://my-bucket/experiments/my_exp \
    --alsologtostderr
```

If using a local path, upload results to GCS after training:

```bash
gcloud storage cp -r /tmp/exp_1 $BUCKET/experiments/
```

## 6. Example: Gemma 2B GSM8K RL

This example trains Gemma 2B on GSM8K using RL (GRPO) on a
v5litepod-16. The experiment config `gemma2_2b_gsm8k_2k_rl_16`
(defined in `simply/config_lib.py`) sets:

- 2000 training steps
- `LinearWarmupConstant(value=1e-7)` learning rate
- `grad_accum_steps=2` to avoid OOM on logprobs
- Checkpoints every 20 steps

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
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
python3.11 -m venv /tmp/simply_venv
source /tmp/simply_venv/bin/activate
pip install -q -U 'jax[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz
pip install -q -r requirements.txt
pip install -q google-cloud-storage
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
    --experiment_config gemma2_2b_gsm8k_2k_rl_16 \
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

Set `grad_accum_steps=2` (or higher) to halve the microbatch size.
The gradient is mathematically identical.

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
EXPERIMENT_CONFIG=gemma2_2b_gsm8k_2k_rl_16
EXPERIMENT_DIR=$BUCKET/experiments/my_experiment
NUM_WORKERS=4

SETUP_CMD="
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
python3.11 -m venv /tmp/simply_venv
source /tmp/simply_venv/bin/activate
pip install -q -U 'jax[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz
pip install -q -r /tmp/simply/requirements.txt
pip install -q google-cloud-storage
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

## 11. Running on GKE with XPK

As an alternative to managing TPU VMs directly, you can run Simply
on GKE (Google Kubernetes Engine) clusters with TPU node pools using
[XPK](https://github.com/google/xpk). XPK handles Docker image
building, job scheduling, and multi-host coordination automatically.

### Prerequisites

- A GKE cluster with TPU node pools (created via `xpk cluster
  create` or manually)
- [XPK](https://github.com/google/xpk) installed (`pip install xpk`)
- Docker installed and accessible
- `kubectl` configured to access your cluster

### Building the Docker Image

Simply provides a Dockerfile at `scripts/Dockerfile.simply` that
pre-installs JAX with TPU support and all Simply dependencies:

```bash
# Build the image
docker build -f scripts/Dockerfile.simply \
    -t gcr.io/$PROJECT/simply-jax-tpu:latest .

# Push to your project's container registry
docker push gcr.io/$PROJECT/simply-jax-tpu:latest
```

The Dockerfile installs dependencies in a separate layer for fast
rebuilds. When the workload starts, the launch script runs
`pip install .` inside the container to install Simply itself from
the source tree copied by XPK.

### Launching a Workload

Use `scripts/launch_gke.sh` to submit training jobs:

```bash
./scripts/launch_gke.sh --config qwen3_0p6b \
    --project my-project \
    --cluster my-cluster \
    --zone us-central1 \
    --tpu-type v4-8 \
    --image gcr.io/my-project/simply-jax-tpu:latest
```

The script builds a Docker image (overlaying the Simply source onto
the base image), pushes it, and submits the workload via
`xpk workload create`. XPK uses its `--script-dir` flag to copy the
Simply repo into the container's working directory.

#### Required Flags

| Flag | Env Variable | Description |
|------|-------------|-------------|
| `--config CONFIG` | | Simply experiment config name |
| `--project PROJECT` | `SIMPLY_XPK_PROJECT` | GCP project ID |
| `--cluster CLUSTER` | `SIMPLY_XPK_CLUSTER` | GKE cluster name |
| `--image IMAGE` | `SIMPLY_XPK_IMAGE` | Base Docker image |

#### Common Options

| Flag | Env Variable | Default | Description |
|------|-------------|---------|-------------|
| `--zone ZONE` | `SIMPLY_XPK_ZONE` | `us-central1` | GCP zone/region |
| `--tpu-type TYPE` | `SIMPLY_XPK_TPU_TYPE` | `v4-8` | TPU accelerator type |
| `--num-slices N` | `SIMPLY_XPK_NUM_SLICES` | `1` | Number of TPU slices |
| `--priority PRI` | `SIMPLY_XPK_PRIORITY` | `medium` | Workload priority |
| `--name NAME` | | auto | Custom workload name |
| `--spot` | | (default) | Use spot/preemptible instances |
| `--on-demand` | | | Use on-demand instances |
| `--dry-run` | | | Print the xpk command without running |

#### Passing Simply Arguments

Pass additional flags to `simply.main` after `--`:

```bash
./scripts/launch_gke.sh --config lm_test \
    --project my-project --cluster my-cluster \
    --image gcr.io/my-project/simply-jax-tpu:latest \
    -- --mesh_shape 1,4,32 --experiment_dir gs://my-bucket/exp1
```

### Using Environment Variables

To avoid repeating flags, set environment variables in your shell
profile:

```bash
export SIMPLY_XPK_PROJECT=my-project
export SIMPLY_XPK_CLUSTER=my-cluster
export SIMPLY_XPK_ZONE=us-central1
export SIMPLY_XPK_IMAGE=gcr.io/my-project/simply-jax-tpu:latest

# Now just specify the config:
./scripts/launch_gke.sh --config qwen3_0p6b --tpu-type v4-8
```

### Profiling with XProf

To collect XProf traces, pass `--profile` and set a GCS bucket for
trace storage:

```bash
export SIMPLY_XPK_GCS_BUCKET=gs://my-bucket/profiles

./scripts/launch_gke.sh --config qwen3_0p6b --profile \
    --project my-project --cluster my-cluster \
    --image gcr.io/my-project/simply-jax-tpu:latest
```

This sets `JAX_PROFILER_LOG_DIR` inside the container and saves
worker logs to the GCS bucket. By default, profiling starts after
5 warmup steps and captures 3 steps (configurable via
`--profile-warmup` and `--profile-steps`).

### Managing Workloads

List, monitor, and delete workloads:

```bash
# List all simply-* workloads on the cluster
./scripts/launch_gke.sh --list

# Stream logs from a running workload
./scripts/launch_gke.sh --logs simply-qwen3-0p6b-0310

# Delete a workload
./scripts/launch_gke.sh --delete simply-qwen3-0p6b-0310
```

You can also use `kubectl` directly for more detailed inspection:

```bash
# View logs from a specific worker
kubectl logs -l jobset.sigs.k8s.io/jobset-name=simply-qwen3-0p6b-0310 \
    --all-containers --tail=100

# Check pod status
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=simply-qwen3-0p6b-0310
```

### Docker Access

The launch script checks for Docker access and falls back to
`sg docker` if the current user isn't in the docker group. If
Docker is not accessible at all, run:

```bash
sudo usermod -aG docker $USER && newgrp docker
```

## Future Work

- **GPU VMs** -- A100/H100 setup

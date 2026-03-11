# Simply on GCloud: Quickstart

A minimal guide to run your first Simply experiment on a GCloud TPU.
For multi-host TPU pods, preemption handling, and other advanced
topics, see the [full guide](docs/gcloud.md).

## Prerequisites

- A GCP project with TPU quota and billing enabled
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- The Simply codebase cloned locally

## Step 1: One-Time GCloud Setup

```bash
PROJECT=your-project-id
ZONE=us-central1-a
BUCKET=gs://${PROJECT}-experiments
```

Run these once to set up your project:

```bash
# Enable TPU API
gcloud services enable tpu.googleapis.com --project=$PROJECT

# Create GCS bucket for code and assets
gcloud storage buckets create $BUCKET \
    --location=us-central1 --project=$PROJECT

# VPC with private Google access (skip if you already have one)
gcloud compute networks create default \
    --project=$PROJECT --subnet-mode=auto
gcloud compute networks subnets update default \
    --region=us-central1 \
    --enable-private-ip-google-access \
    --project=$PROJECT

# Cloud NAT (needed if VMs have no external IP)
gcloud compute routers create simply-router \
    --region=us-central1 --network=default --project=$PROJECT
gcloud compute routers nats create simply-nat \
    --router=simply-router --region=us-central1 \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges --project=$PROJECT

# Firewall: allow SSH
gcloud compute firewall-rules create allow-ssh \
    --network=default --allow=tcp:22,icmp --project=$PROJECT

# Service account permissions
SA="$(gcloud iam service-accounts list --project=$PROJECT \
    --filter='email:compute@developer.gserviceaccount.com' \
    --format='value(email)')"
for ROLE in roles/tpu.admin roles/compute.instanceAdmin.v1 \
            roles/iam.serviceAccountUser roles/storage.admin; do
  gcloud projects add-iam-policy-binding $PROJECT \
      --member="serviceAccount:$SA" --role="$ROLE"
done
```

## Step 2: Upload Code and Assets to GCS

```bash
# Package and upload code
cd /path/to/simply
tar --exclude='.git' --exclude='__pycache__' \
    -czf /tmp/simply.tar.gz .
gcloud storage cp /tmp/simply.tar.gz $BUCKET/code/

# Download model assets locally, then upload to GCS
python setup/setup_assets.py
gcloud storage cp -r ~/.cache/simply/models/GEMMA-2.0-2B-PT-ORBAX \
    $BUCKET/models/
gcloud storage cp -r ~/.cache/simply/vocabs/ $BUCKET/vocabs/
gcloud storage cp -r ~/.cache/simply/datasets/ $BUCKET/datasets/
```

## Step 3: Create a TPU VM

```bash
TPU_NAME=simply-test
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=v5litepod-1 \
    --version=tpu-ubuntu2204-base \
    --project=$PROJECT \
    --preemptible
```

## Step 4: Set Up the TPU VM

SSH in:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT --worker=0
```

Then on the VM:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download code from GCS
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz

# Install deps (uv sync uses the lockfile)
uv sync --extra tpu --extra tfds --extra gcloud

# Activate venv
source .venv/bin/activate

# Point Simply at GCS assets
export SIMPLY_MODELS=$BUCKET/models/
export SIMPLY_DATASETS=$BUCKET/datasets/
export SIMPLY_VOCABS=$BUCKET/vocabs/
```

Replace `$BUCKET` with the actual value (e.g.
`gs://your-project-id-experiments`) since the shell variable won't
persist across SSH sessions.

## Step 5: Run an Experiment

```bash
cd /tmp/simply
source .venv/bin/activate
export SIMPLY_MODELS=$BUCKET/models/
export SIMPLY_DATASETS=$BUCKET/datasets/
export SIMPLY_VOCABS=$BUCKET/vocabs/

python3 -m simply.main \
    --experiment_config lm_test \
    --experiment_dir /tmp/exp_1 \
    --alsologtostderr
```

## Step 6: Clean Up

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME \
    --zone=$ZONE --project=$PROJECT --quiet
```

## What Next?

See the [full guide](docs/gcloud.md) for:

- **Multi-host TPU pods** (v5litepod-8/16) -- SSH key warmup,
  `--worker=all`, `jax.distributed.initialize()`
- **Preemption handling** -- bastion VM with auto-retry loop
- **Monitoring** -- TensorBoard, SSH probes, serial port logs
- **Common gotchas** -- OOM fixes, SSH issues

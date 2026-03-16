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
# Install Python 3.12 (Simply requires 3.12+)
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Create venv and install deps
python3.12 -m venv /tmp/simply_venv
source /tmp/simply_venv/bin/activate
pip install -U 'jax[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Download code from GCS
gcloud storage cp $BUCKET/code/simply.tar.gz /tmp/
mkdir -p /tmp/simply && cd /tmp/simply
tar xzf /tmp/simply.tar.gz
pip install ".[tpu,tfds,gcloud]"

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
source /tmp/simply_venv/bin/activate
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

## (Optional) Running on GKE with XPK

For GKE clusters with TPU node pools, you can use
[XPK](https://github.com/AI-Hypercomputer/xpk) to launch Simply
training workloads. This section walks through the full workflow.
For advanced topics (details, profiling, troubleshooting), see the
[full guide](docs/gcloud.md#11-running-on-gke-with-xpk).

### Prerequisites

- A GKE cluster with a TPU node pool already provisioned
- [XPK](https://github.com/AI-Hypercomputer/xpk) installed
  (`pip install xpk`)
- Docker installed and authenticated to push to GCR/Artifact
  Registry
- `kubectl` configured for your cluster
  (`gcloud container clusters get-credentials ...`)

### Quick Commands

Set these once per shell session. Replace the values with your own
project and cluster details.

```bash
export PROJECT=your-gcp-project-id
export CLUSTER=your-gke-cluster-name
export ZONE=us-central1
export TPUTYPE=tpu7x-128
```

Build the Simply base image and push it to your project's container
registry. This image has JAX with TPU support and all Simply
dependencies pre-installed.

```bash
cd /path/to/simply

# Build the Docker image
docker build -f scripts/Dockerfile.simply \
    -t gcr.io/$PROJECT/simply-jax-tpu:latest .

# Push to Google Container Registry
docker push gcr.io/$PROJECT/simply-jax-tpu:latest
```

Use the launcher script to submit the workload via XPK. The
`lm_test_gke_training` config is a small model designed for
testing GKE training -- it uses no checkpoint and disables
checkpoint saving, so no GCS asset upload is needed.

```bash
./scripts/launch_gke.sh \
    --config lm_test_gke_training \
    --project $PROJECT \
    --cluster $CLUSTER \
    --zone $ZONE \
    --tpu-type $TPUTYPE \
    --image gcr.io/$PROJECT/simply-jax-tpu:latest
```

The script packages the Simply source directory (via XPK's
`--script-dir`), installs it inside the container, downloads the
tokenizer, and starts training.

To preview the XPK command without submitting, add `--dry-run`.

To monitor and delete the workloads, you can use the following
commands:

```bash
# List all Simply workloads on the cluster
./scripts/launch_gke.sh \
    --project $PROJECT --cluster $CLUSTER --zone $ZONE \
    --list

# Stream logs for a specific workload
./scripts/launch_gke.sh \
    --project $PROJECT --cluster $CLUSTER --zone $ZONE \
    --logs JOB_NAME

# Delete a workload when done
./scripts/launch_gke.sh \
    --project $PROJECT --cluster $CLUSTER --zone $ZONE \
    --delete JOB_NAME
```

## What Next?

See the [full guide](docs/gcloud.md) for:

- **Multi-host TPU pods** (v5litepod-8/16) -- SSH key warmup,
  `--worker=all`, `jax.distributed.initialize()`
- **Preemption handling** -- bastion VM with auto-retry loop
- **Monitoring** -- TensorBoard, SSH probes, serial port logs
- **Common gotchas** -- OOM fixes, SSH issues

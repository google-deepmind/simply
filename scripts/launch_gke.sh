#!/bin/bash
# launch_gke.sh — Launch Simply training on GKE TPU via XPK
#
# Usage:
#   ./scripts/launch_gke.sh --config <CONFIG> [OPTIONS] [-- SIMPLY_ARGS...]
#
# Examples:
#   # Baseline run:
#   ./scripts/launch_gke.sh --config lm_test \
#       --project my-project --cluster my-cluster --zone us-central1 \
#       -- --experiment_dir gs://my-bucket/exp1
#
#   # With XProf profiling:
#   ./scripts/launch_gke.sh --config lm_test --profile \
#       --project my-project --cluster my-cluster --zone us-central1
#
#   # Using environment variables:
#   export SIMPLY_XPK_PROJECT=my-project
#   export SIMPLY_XPK_CLUSTER=my-cluster
#   export SIMPLY_XPK_ZONE=us-central1
#   ./scripts/launch_gke.sh --config lm_test
#
#   # Dry run:
#   ./scripts/launch_gke.sh --config lm_test --dry-run
#
#   # List/logs/delete:
#   ./scripts/launch_gke.sh --list
#   ./scripts/launch_gke.sh --logs simply-lm-test-0310
#   ./scripts/launch_gke.sh --delete simply-lm-test-0310

set -euo pipefail

# ------------------------------------------
# Defaults
# ------------------------------------------
PROJECT="${SIMPLY_XPK_PROJECT:-}"
CLUSTER="${SIMPLY_XPK_CLUSTER:-}"
ZONE="${SIMPLY_XPK_ZONE:-us-central1}"
TPU_TYPE="${SIMPLY_XPK_TPU_TYPE:-v4-8}"
NUM_SLICES="${SIMPLY_XPK_NUM_SLICES:-1}"
PRIORITY="${SIMPLY_XPK_PRIORITY:-medium}"
BASE_IMAGE="${SIMPLY_XPK_IMAGE:-}"
CAPACITY="--spot"

WORKLOAD_NAME=""
CONFIG=""
PROFILE=false
PROFILE_STEPS="3"
PROFILE_WARMUP="5"
NUM_TRAIN_STEPS=20
SIMPLY_ARGS=""
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GCS_BUCKET="${SIMPLY_XPK_GCS_BUCKET:-}"

# ------------------------------------------
# Helper functions
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage: launch_gke.sh --config <CONFIG> [OPTIONS] [-- SIMPLY_ARGS...]
       launch_gke.sh --list | --logs <name> | --delete <name>

Launch Simply training on GKE TPU via XPK.

Required:
  --config CONFIG         Simply experiment config name (e.g., lm_test)
  -p, --project PROJECT   GCP project (or set SIMPLY_XPK_PROJECT)
  -c, --cluster CLUSTER   GKE cluster (or set SIMPLY_XPK_CLUSTER)

Options:
  --profile               Enable XProf trace collection
  --profile-warmup N      Steps before profiling starts (default: 5)
  --profile-steps N       Number of steps to profile (default: 3)
  --train-steps N         Total training steps (default: 20)
  -n, --name NAME         Workload name (default: auto-generated)
  -z, --zone ZONE         GCP zone              (default: us-central1)
  -t, --tpu-type TYPE     TPU type              (default: v4-8)
  --num-slices N          Number of slices       (default: 1)
  --image IMAGE           Base Docker image
  --priority PRIORITY     Workload priority      (default: medium)
  --spot                  Use spot instances      (default)
  --on-demand             Use on-demand instances
  --dry-run               Print the xpk command without running it

Management:
  --list                  List all simply-* workloads on the cluster
  --logs NAME             Stream logs for a workload
  --delete NAME           Delete a workload

Extra simply.main args after '--':
  e.g., -- --mesh_shape 1,8,16,1 --experiment_dir gs://bucket/exp1
USAGE
    exit "${1:-0}"
}

gen_name() {
    local config_short
    config_short="$(echo "$1" | tr '_' '-')"
    local ts
    ts="$(date +%m%d)"
    local name="simply-${config_short}-${ts}"
    echo "${name:0:40}"
}

do_list() {
    echo "=== Listing Simply workloads on ${CLUSTER} ==="
    xpk workload list \
        --cluster "${CLUSTER}" \
        --project "${PROJECT}" \
        --zone "${ZONE}" 2>&1 \
        | grep -E "^(Jobset|simply-|-)" || true
}

do_logs() {
    local name="$1"
    echo "=== Streaming logs for ${name} ==="
    kubectl logs \
        -l "jobset.sigs.k8s.io/jobset-name=${name}" \
        --all-containers --follow --tail=200 2>/dev/null \
    || kubectl logs \
        -l "jobset.sigs.k8s.io/jobset-name=${name}" \
        --all-containers --tail=200 2>/dev/null \
    || echo "No logs found. The job may not have started yet."
}

do_delete() {
    local name="$1"
    echo "=== Deleting workload ${name} ==="
    xpk workload delete \
        --cluster "${CLUSTER}" \
        --project "${PROJECT}" \
        --zone "${ZONE}" \
        --workload "${name}"
}

# ------------------------------------------
# Parse arguments
# ------------------------------------------
ACTION="launch"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            ACTION="list"; shift ;;
        --logs)
            ACTION="logs"; WORKLOAD_NAME="$2"; shift 2 ;;
        --delete)
            ACTION="delete"; WORKLOAD_NAME="$2"; shift 2 ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --profile)
            PROFILE=true; shift ;;
        --profile-warmup)
            PROFILE_WARMUP="$2"; shift 2 ;;
        --profile-steps)
            PROFILE_STEPS="$2"; shift 2 ;;
        --train-steps)
            NUM_TRAIN_STEPS="$2"; shift 2 ;;
        -n|--name)
            WORKLOAD_NAME="$2"; shift 2 ;;
        -c|--cluster)
            CLUSTER="$2"; shift 2 ;;
        -p|--project)
            PROJECT="$2"; shift 2 ;;
        -z|--zone)
            ZONE="$2"; shift 2 ;;
        -t|--tpu-type)
            TPU_TYPE="$2"; shift 2 ;;
        --num-slices)
            NUM_SLICES="$2"; shift 2 ;;
        --image)
            BASE_IMAGE="$2"; shift 2 ;;
        --priority)
            PRIORITY="$2"; shift 2 ;;
        --spot)
            CAPACITY="--spot"; shift ;;
        --on-demand)
            CAPACITY="--on-demand"; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            usage 0 ;;
        --)
            shift; SIMPLY_ARGS="$*"; break ;;
        -*)
            echo "Error: Unknown option: $1" >&2; usage 1 ;;
        *)
            echo "Error: Unexpected argument: $1" >&2; usage 1 ;;
    esac
done

# ------------------------------------------
# Handle management actions
# ------------------------------------------
case "${ACTION}" in
    list)   do_list; exit 0 ;;
    logs)   do_logs "${WORKLOAD_NAME}"; exit 0 ;;
    delete) do_delete "${WORKLOAD_NAME}"; exit 0 ;;
esac

# ------------------------------------------
# Validate inputs for launch
# ------------------------------------------
if [ -z "${CONFIG}" ]; then
    echo "Error: --config is required." >&2
    usage 1
fi

if [ -z "${PROJECT}" ]; then
    echo "Error: --project (or SIMPLY_XPK_PROJECT) is required." >&2
    usage 1
fi

if [ -z "${CLUSTER}" ]; then
    echo "Error: --cluster (or SIMPLY_XPK_CLUSTER) is required." >&2
    usage 1
fi

if [ -z "${BASE_IMAGE}" ]; then
    echo "Error: --image (or SIMPLY_XPK_IMAGE) is required." >&2
    echo "Build one with: docker build" \
        "-f scripts/Dockerfile.simply -t <tag> ." >&2
    exit 1
fi

# Generate workload name if not provided
if [ -z "${WORKLOAD_NAME}" ]; then
    WORKLOAD_NAME="$(gen_name "${CONFIG}")"
fi

# ------------------------------------------
# Build the Simply command
# ------------------------------------------
INSTALL_CMD="uv pip install --system --no-cache . 2>/dev/null"

# Download vocab/tokenizer files that Simply needs at runtime.
# Qwen3 tokenizer is fetched from HuggingFace into the default
# vocabs directory (~/.cache/simply/vocabs/Qwen3/).
VOCAB_SETUP="python3 -c \"import os; from huggingface_hub import hf_hub_download; d=os.path.expanduser('~/.cache/simply/vocabs/Qwen3'); os.makedirs(d,exist_ok=True); [hf_hub_download('Qwen/Qwen3-0.6B',f,local_dir=d) for f in ['tokenizer.json','tokenizer_config.json']]\""

SIMPLY_CMD="python3 -u -m simply.main"
SIMPLY_CMD="${SIMPLY_CMD} --experiment_config=${CONFIG}"
SIMPLY_CMD="${SIMPLY_CMD} --alsologtostderr"

if [ -n "${SIMPLY_ARGS}" ]; then
    SIMPLY_CMD="${SIMPLY_CMD} ${SIMPLY_ARGS}"
fi

if [ "${PROFILE}" = true ] && [ -n "${GCS_BUCKET}" ]; then
    PROFILE_DIR="${GCS_BUCKET}/${WORKLOAD_NAME}"
    LOG_SAVE="gsutil cp /tmp/simply_output.log"
    LOG_SAVE="${LOG_SAVE} ${PROFILE_DIR}/worker-\$(hostname).log"
    LOG_SAVE="${LOG_SAVE} 2>/dev/null || true"
    PROFILE_ENV="JAX_PROFILER_LOG_DIR=${PROFILE_DIR}"
    PROFILE_ENV="${PROFILE_ENV} SIMPLY_PROFILE_START_STEP=${PROFILE_WARMUP}"
    PROFILE_ENV="${PROFILE_ENV} SIMPLY_PROFILE_END_STEP=$((PROFILE_WARMUP + PROFILE_STEPS))"
    FULL_CMD="${INSTALL_CMD} && ${VOCAB_SETUP}"
    FULL_CMD="${FULL_CMD} && (${PROFILE_ENV} ${SIMPLY_CMD}"
    FULL_CMD="${FULL_CMD} 2>&1 | tee /tmp/simply_output.log;"
    FULL_CMD="${FULL_CMD} ${LOG_SAVE})"
elif [ "${PROFILE}" = true ]; then
    echo "Warning: --profile requires SIMPLY_XPK_GCS_BUCKET." \
        "Profiling disabled." >&2
    FULL_CMD="${INSTALL_CMD} && ${VOCAB_SETUP} && ${SIMPLY_CMD}"
else
    FULL_CMD="${INSTALL_CMD} && ${VOCAB_SETUP} && ${SIMPLY_CMD}"
fi

# ------------------------------------------
# Launch
# ------------------------------------------
echo "============================================"
echo "  Simply — XPK Training Launch"
echo "============================================"
echo "  Workload:  ${WORKLOAD_NAME}"
echo "  Config:    ${CONFIG}"
echo "  Cluster:   ${CLUSTER}"
echo "  Project:   ${PROJECT}"
echo "  Zone:      ${ZONE}"
echo "  TPU Type:  ${TPU_TYPE}"
echo "  Slices:    ${NUM_SLICES}"
echo "  Priority:  ${PRIORITY}"
echo "  Capacity:  ${CAPACITY/--/}"
echo "  Image:     ${BASE_IMAGE}"
echo "  Profile:   ${PROFILE}"
if [ "${PROFILE}" = true ] && [ -n "${GCS_BUCKET}" ]; then
echo "  Prof Dir:  ${PROFILE_DIR}"
fi
echo "  Simply:    ${SIMPLY_DIR}"
echo "  Command:   ${SIMPLY_CMD}"
echo "============================================"

XPK_CMD=(
    xpk workload create
    --cluster "${CLUSTER}"
    --project "${PROJECT}"
    --zone "${ZONE}"
    --workload "${WORKLOAD_NAME}"
    --base-docker-image "${BASE_IMAGE}"
    --script-dir "${SIMPLY_DIR}"
    --tpu-type="${TPU_TYPE}"
    --num-slices="${NUM_SLICES}"
    --priority="${PRIORITY}"
    "${CAPACITY}"
    --command "${FULL_CMD}"
)

if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo "[DRY RUN] Would execute:"
    echo "  xpk workload create \\"
    echo "    --cluster ${CLUSTER} \\"
    echo "    --project ${PROJECT} \\"
    echo "    --zone ${ZONE} \\"
    echo "    --workload ${WORKLOAD_NAME} \\"
    echo "    --base-docker-image ${BASE_IMAGE} \\"
    echo "    --script-dir ${SIMPLY_DIR} \\"
    echo "    --tpu-type=${TPU_TYPE} \\"
    echo "    --num-slices=${NUM_SLICES} \\"
    echo "    --priority=${PRIORITY} \\"
    echo "    ${CAPACITY} \\"
    echo "    --command \"${FULL_CMD}\""
    echo ""
    exit 0
fi

echo ""
echo "=== Submitting workload ==="

if docker info &>/dev/null; then
    "${XPK_CMD[@]}"
elif sg docker -c "docker info" &>/dev/null; then
    echo "(Using 'sg docker' for Docker access)"
    sg docker -c "$(printf '%q ' "${XPK_CMD[@]}")"
else
    echo "Error: Docker is not accessible." >&2
    echo "  sudo usermod -aG docker \$USER && newgrp docker" >&2
    exit 1
fi

echo ""
echo "============================================"
echo "  Workload submitted: ${WORKLOAD_NAME}"
echo "============================================"
echo ""
echo "Monitor:"
echo "  ./scripts/launch_gke.sh --project $PROJECT --cluster $CLUSTER --zone $ZONE --list"
echo "  ./scripts/launch_gke.sh --project $PROJECT --cluster $CLUSTER --zone $ZONE --logs ${WORKLOAD_NAME}"
echo ""
echo "Cleanup:"
echo "  ./scripts/launch_gke.sh --project $PROJECT --cluster $CLUSTER --zone $ZONE --delete ${WORKLOAD_NAME}"

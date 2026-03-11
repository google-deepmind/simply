#!/bin/bash

# Example command for installing dependencies.
# Requires uv: https://docs.astral.sh/uv/
# Install uv if not available:
#   curl -LsSf https://astral.sh/uv/install.sh | sh

# JAX installation is environment-specific (CPU, GPU, TPU). Check the official JAX installation guide at https://docs.jax.dev/en/latest/installation.html.
echo "Select JAX installation type:"
echo "1) CPU"
echo "2) GPU"
echo "3) TPU"
read -p "Enter your choice (1-3): " jax_type

case $jax_type in
  1)
    echo "Installing for CPU..."
    uv sync
    ;;
  2)
    echo "Installing for GPU..."
    uv sync --extra gpu
    ;;
  3)
    echo "Installing for TPU..."
    uv sync --extra tpu
    ;;
  *)
    echo "Invalid choice. Skipping installation."
    ;;
esac

# Example command for local run.
# Add "export JAX_DISABLE_JIT=True;" to disable `jit` for easier debugging.
# Change "lm_test" to other experiment config names in `config_lib.py` to run other experiments.
EXP=local_test_1; rm -rf /tmp/${EXP}; python -m simply.main --experiment_config lm_test --experiment_dir /tmp/${EXP} --alsologtostderr

# Example command for checking learning curves with tensorboard.
tensorboard --logdir /tmp/${EXP}
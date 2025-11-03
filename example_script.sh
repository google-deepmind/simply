#!/bin/bash

# Example command for installing dependencies.
# JAX installation is environment-specific (CPU, GPU, TPU). Check the official JAX installation guide at https://docs.jax.dev/en/latest/installation.html.
echo "Select JAX installation type:"
echo "1) CPU"
echo "2) GPU"
echo "3) TPU"
read -p "Enter your choice (1-3): " jax_type

case $jax_type in
  1)
    echo "Installing JAX for CPU..."
    pip install -U jax
    ;;
  2)
    echo "CUDA versions supported by JAX can be found at https://docs.jax.dev/en/latest/installation.html#cuda-cudnn-installation."
    read -p "Enter your CUDA version (e.g., cuda12): " cuda_version
    echo "Installing JAX for GPU with $cuda_version..."
    pip install -U "jax[$cuda_version]"
    ;;
  3)
    echo "Installing JAX for TPU..."
    pip install -U "jax[tpu]"
    ;;
  *)
    echo "Invalid choice. Skipping JAX installation."
    ;;
esac

pip install -r requirements.txt

# Example command for local run.
# Add "export JAX_DISABLE_JIT=True;" to disable `jit` for easier debugging.
# Change "lm_test" to other experiment config names in `config_lib.py` to run other experiments.
EXP=local_test_1; rm -rf /tmp/${EXP}; python -m simply.main --experiment_config lm_test --experiment_dir /tmp/${EXP} --alsologtostderr

# Example command for checking learning curves with tensorboard.
tensorboard --logdir /tmp/${EXP}
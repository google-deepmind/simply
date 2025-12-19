# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simply is a minimal JAX-based research codebase for LLM training and inference. It emphasizes minimal abstractions for rapid iteration on frontier research. The codebase supports Gemma, Qwen, and DeepSeek model families with multi-host distributed training.

## Common Commands

### Installation
```bash
# Install JAX (environment-specific)
pip install -U jax              # CPU
pip install -U "jax[cuda13]"    # GPU
pip install -U "jax[tpu]"       # TPU

# Install other dependencies
pip install -r requirements.txt

# Download models and datasets
python setup/setup_assets.py
```

### Running Experiments
```bash
# Local test run
python -m simply.main --experiment_config lm_test --experiment_dir /tmp/exp_1 --alsologtostderr

# Debug mode (disable JIT for printing arrays)
export JAX_DISABLE_JIT=True
python -m simply.main --experiment_config lm_no_scan_test --experiment_dir /tmp/exp_1 --alsologtostderr

# TensorBoard monitoring
tensorboard --logdir /tmp/exp_1
```

### Testing
```bash
# Run all tests
pytest simply/

# Run specific test file
pytest simply/model_lib_test.py

# Run specific test
pytest simply/model_lib_test.py::ModelTest::test_forward_pass
```

## Architecture

### Core Modules (simply/)
- **main.py** - Entry point for training runs
- **config_lib.py** - Experiment and sharding configurations via registries
- **model_lib.py** - LLM architectures (Attention, TransformerBlock, TransformerLM, MoE)
- **data_lib.py** - Data pipeline setup using SeqIO and Grain
- **rl_lib.py** - RL training components (reward normalization, batching)
- **tool_lib.py** - Tool use and execution framework

### Utilities (simply/utils/)
- **module.py** - SimplyModule base class with registry pattern
- **common.py** - AnnotatedArray wrapper for metadata, PyTree types
- **checkpoint_lib.py** - Orbax-based checkpoint management
- **sharding.py** - Multi-host sharding patterns (FSDP, TP, Expert Parallelism)
- **sampling_lib.py** - Sampling schedules and input processing
- **optimizers.py** - Adam, AdamW, SGD with learning rate schedules

### Key Design Patterns

**Registry Pattern**: All extensible components use dataclass + registry decorator:
```python
@SomeRegistry.register
@dataclasses.dataclass
class MyComponent:
    param: int
```

Registries include: `ExperimentConfigRegistry`, `ShardingConfigRegistry`, `ModuleRegistry`, `OptimizerRegistry`, `TrainLoopRegistry`, `ToolRegistry`, `TokenizerRegistry`

**AnnotatedArray**: Model parameters are wrapped in `AnnotatedArray` for sharding annotations and metadata tracking throughout the codebase.

**Configuration-Driven**: Experiments are defined via registered configs in `config_lib.py`. Use `--experiment_config <name>` to select, or `--experiment_config_path` for external config files.

### Environment Variables
- `SIMPLY_MODELS` - Model checkpoint directory (default: `~/.cache/simply/models/`)
- `SIMPLY_DATASETS` - Dataset directory (default: `~/.cache/simply/datasets/`)
- `SIMPLY_VOCABS` - Vocabulary directory (default: `~/.cache/simply/vocabs/`)
- `JAX_DISABLE_JIT` - Set to `True` to disable JIT for debugging

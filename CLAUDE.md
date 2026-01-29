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

## Data Pipeline (data_lib.py)

The Grain-native data pipeline is in `data_lib.py`. Key concepts:

### Registries
- **`DataSourceRegistry`** - Raw data sources with `__len__`/`__getitem__` methods.
  Used via `DatasetConfig(source='name')`.
- **`DataConfigRegistry`** - Config dataclasses (for serialization) and preset
  config factory functions. Use `dataset_config='preset_name'` for string shorthand.

### Configuration Classes
- **`TFDSSourceConfig`** - TFDS dataset name and split
- **`HFDatasetSourceConfig`** - HuggingFace datasets (name, split, subset)
- **`ArrayRecordSourceConfig`** - ArrayRecord files (supports glob patterns)
- **`DatasetConfig`** - Single dataset configuration
- **`MixtureConfig`** - Multiple datasets with weights

### DatasetConfig Fields
```python
@dataclasses.dataclass
class DatasetConfig:
  source: str | TFDSSourceConfig | HFDatasetSourceConfig | ArrayRecordSourceConfig
  lm_format_name: str | None = 'Pretrain'  # None = raw
  packing: str = 'concat_split'
  data_key: str = 'text'
  tokenizer_name: str | None = None
  add_eos: bool = True
  add_bos: bool = False
  trainable_roles: tuple[str, ...] | None = None  # For chat: roles in loss
```

### lm_format_name (Tokenization Control)
- `None` - Raw: skip tokenization, data passed through as-is
- `'Pretrain'` - TokenizeTransform + NextTokenPredTransform
- `'SimplyV1Chat'` etc. - ChatFormatTransform + NextTokenPredTransform

### packing (Packing Control)
- `'concat_split'` - Concatenate then split (best throughput, for pretraining)
- `'first_fit'` - Bin packing preserving boundaries (for chat/SFT)
- `'pad_or_truncate'` - Simple pad/truncate (for validation)
- `'none'` - No packing (for raw data)

**Note**: Validation mode automatically overrides packing to `'pad_or_truncate'`.

### MixtureConfig
```python
MixtureConfig(
  datasets=((DatasetConfig(...), 0.7), (DatasetConfig(...), 0.3)),
  pack_before_mix=False,  # False = mix then pack, True = pack then mix
)
```

### String Shorthand for dataset_config

`dataset_config` can be a string that looks up a preset from `DataConfigRegistry`:

```python
# Register a preset
@DataConfigRegistry.register
def c4_train_pt():
    return DatasetConfig(
        source=TFDSSourceConfig(name='c4:3.1.0', split='train'),
        lm_format_name='Pretrain',
    )

# Use in experiment config
dataset_config = 'c4_train_pt'
```

### Key Learnings
- **Token IDs from tokenizer**: `bos_id`, `eos_id`, `pad_id` come from tokenizer
- **Grain mp_prefetch**: Handles 0 workers gracefully (no-op)
- **Registry patterns**: Use `Registry.register_value()` for instances
  (call with `Registry.get_instance(name)`)
- **Chat tokenization**: `LMFormat.format_tokens()` is the single source of truth
  for chat formatting + tokenization. `ChatFormatTransform` delegates to it.
- **ArrayRecord glob**: `ArrayRecordSourceConfig` supports glob patterns
  (expanded before passing to `ArrayRecordDataSource`)
- **Registry serialization**: `pytree.dump()/load()` serializes dataclass instances
  using `__registered_name__`. Factory functions in same registry are fine - they
  return instances that serialize correctly (same pattern as ExperimentConfigRegistry).

## Code Style

- **Line length**: Keep lines under 80 characters
- **Hanging indents**: Use 4 spaces for continuation lines. When arguments to
  a function call or items in a list/dictionary span multiple lines, indent
  continuation lines by exactly 4 spaces relative to the start of the line
  that begins the statement.

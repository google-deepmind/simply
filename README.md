Disclaimer and Personal Statements of Tech Transfers from USA to PRC!

U.S. EAR restricts AI exports (e.g., model weights >10^25 FLOP, ECCN 4E091) to PRC/Entity List.
Check license needs: bis.doc.gov/ear. Report issues to compliance.

Even if not controlled, I protest unauthorized sharing—protect U.S. security, in accordance with my rights to 1st Amendment.

<!-- mdlint off(LINE_OVER_80) -->
# Simply: Minimal Code for Frontier LLM Research in JAX

*Simply* is a minimal and scalable research codebase in JAX, designed for rapid iteration on frontier research in LLM and other autoregressive models.

- *Quick to [fork and hack](#getting-started)* for fast iteration. You should be able to implement your research ideas (e.g., new architecture, optimizer, training loss, etc) in a few hours.
- *Minimal abstractions and dependencies* for a simple and self-contained codebase. Learn [Jax](https://jax.readthedocs.io/en/latest/index.html) (if you haven't), and you are ready to read and hack the code.
- That's it, *simply* [get started](#getting-started) with hacking now :)

## Getting started
### Example commands

#### Local test for debug
```shell
EXP=simply_local_test_1; rm -rf /tmp/${EXP}; python -m simply.main --experiment_config lm_test --experiment_dir /tmp/${EXP} --alsologtostderr
```
Or if you want to debug by printing arrays like normal python code, you can disable `jit` and `use_scan` using the command below.

```shell
export JAX_DISABLE_JIT=True; EXP=simply_local_test_1; rm -rf /tmp/${EXP}; python -m simply.main --experiment_config lm_no_scan_test --experiment_dir /tmp/${EXP} --alsologtostderr
```

#### Running on Google Cloud TPUs
See the [GCloud Quickstart](gcloud_quickstart.md) to run your first experiment on a Cloud TPU, or the [full GCloud guide](docs/gcloud.md) for multi-host training, preemption handling, and monitoring.

## Dependencies

The main dependencies are:
[Jax](https://jax.readthedocs.io/en/latest/index.html) for model and training.
[Orbax](https://orbax.readthedocs.io/en/latest/) for checkpoint management.
[Grain](https://github.com/google/grain) for data pipeline.

Install dependencies:

```bash
# JAX installation is environment-specific. See https://docs.jax.dev/en/latest/installation.html
# CPU:
pip install -U jax
# GPU:
pip install -U "jax[cuda13]"
# TPU:
pip install -U "jax[tpu]"

# Install simply and its dependencies:
pip install .
# With optional dependencies:
pip install ".[tfds]"       # for TensorFlow Datasets
pip install ".[math-eval]"  # for simply/utils/math_eval.py
pip install ".[dev]"        # for testing (pytest)
```

## Setup Model Checkpoints and Datasets

Download datasets and model checkpoints in format supported by Simply from HuggingFace:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download both models and datasets
python setup/setup_assets.py

# Or download only models/datasets
python setup/setup_assets.py --models-only
python setup/setup_assets.py --datasets-only
```

This will download models to `~/.cache/simply/models/` and datasets to `~/.cache/simply/datasets/`. You can customize locations with `--models-dir` and `--datasets-dir` flags, or set environment variables `SIMPLY_MODELS` and `SIMPLY_DATASETS`. (Currently we only included a few datasets and models for testing, and will add more soon.)

## Citation

If you find *Simply* helpful, please cite the following BibTeX:

```
@misc{Liang2025Simply,
  author       = {Chen Liang and Da Huang and Chengrun Yang and Xiaomeng Yang and Andrew Li and Xinchen Yan and {Simply Contributors}},
  title        = {{Simply: an experiment to accelerate and automate AI research}},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/google-deepmind/simply}
}
```

Contributors list:
Alex Zhai, [Xingjian Zhang](https://github.com/xingjian-zhang), [Jiaxi Tang](https://github.com/graytowne), [Lizhang Chen](https://github.com/L-z-Chen), [Ran Tian](https://github.com/tianran)

## License

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

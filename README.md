<!-- mdlint off(LINE_OVER_80) -->
# Simply: Minimal Code for End-to-End Frontier LLM Research

*Simply* is a minimal and scalable research codebase in JAX, designed as an environment where both humans and AI agents can rapidly iterate on frontier LLM research.

- *Quick to [fork and hack](#getting-started)* for fast iteration. We aim at minimizing the time to implement new ideas (e.g., optimizer, training loss, RL algorithms, etc) by humans and AI agents.
- *Minimal abstractions and dependencies* for a simple and self-contained codebase. Learn [Jax](https://jax.readthedocs.io/en/latest/index.html), and you are ready to read and hack the code.
- *An environment for automated AI research* — An AI agent, which can itself be powered by an LLM served with *Simply*, can read the code, propose new ideas, run experiments, and iterate autonomously or under the guidance of human researchers. See [automated AI research with agents](#automated-ai-research-with-agents) for some simple examples. More on the way.
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

#### Automated AI research with agents

You can use agents like [Google Antigravity](https://antigravity.google/), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), or [Gemini CLI](https://github.com/google-gemini/gemini-cli) to run automated research experiments. For example, paste the following prompt into your agent from the repo root to have it design and benchmark new optimizers on a toy setting:

```
You are an AI research agent.
Design and benchmark new optimizers for training a small transformer.
Read simply/utils/optimizers.py to understand the interface.
First, run the Adam baseline using the lm_test config and record the final loss.
In multiple research iterations, propose novel optimizers, tune their hyperparameters, and run experiments to compare with the baseline.
In each iteration, you can propose and run 3 experiments, wait for results and then start the next iteration based on the results of finished experiments. Keep running new iterations until you have finished 15 experiments or proposed 10 new optimizers.
Write a report to /tmp/optimizer_report.md.
```

See the [full guide](docs/automated_ai_research_example.md) for more interesting examples including RL algorithm search for post-training. Have fun playing around with different prompts and interacting with the agent for longer research.

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

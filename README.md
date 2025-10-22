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
EXP=simply_local_test_1; rm -rf /tmp/${EXP}; python -m simply.main --experiment_config TransformerLMTest --experiment_dir /tmp/${EXP} --alsologtostderr
```
Or if you want to debug by printing arrays like normal python code, you can disable `jit` and `use_scan` using the command below.

```shell
export JAX_DISABLE_JIT=True; EXP=simply_local_test_1; rm -rf /tmp/${EXP}; python -m simply.main --experiment_config TransformerLMTestNoScan --experiment_dir /tmp/${EXP} --alsologtostderr
```

## Dependencies

The main dependencies are:
[Jax](https://jax.readthedocs.io/en/latest/index.html) for model and training.
[Orbax](https://orbax.readthedocs.io/en/latest/) for checkpoint management.
[SeqIO](https://github.com/google/seqio) for data pipeline.

Install dependencies:

```bash
# JAX installation is environment-specific. See https://docs.jax.dev/en/latest/installation.html
# CPU:
pip install -U jax
# GPU:
pip install -U "jax[cuda13]"
# TPU:
pip install -U "jax[tpu]"
# Other dependencies:
pip install -r requirements.txt
```

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

## License

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

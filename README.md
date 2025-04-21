<!-- mdlint off(LINE_OVER_80) -->
# Simply

*Simply* is a minimal and scalable research codebase for quickly iterating ideas on autoregressive models. 

- *Quick to [fork and hack](#getting-started)* for fast iteration. You should be able to implement your research ideas (e.g., new architecture, optimizer, training loss, etc) in a few hours.
- *Minimal abstractions and dependencies* for a simple and self-contained codebase. Learn [Jax](https://jax.readthedocs.io/en/latest/index.html) (if you haven't), and you are ready to read and hack the code.
- That's it, *simply* [get started](#getting-started) with hacking now :)

This is an initial release and under active development. More updates on the way, stay tuned! Contributions, suggestions, and feedback are very welcome!

## Getting started
### Example commands

#### Local test for debug
```shell
EXP=simply_local_test_1; rm -rf /tmp/${EXP}; python main.py --experiment_config TransformerLMTest --experiment_dir /tmp/${EXP} --alsologtostderr
```
Or if you want to debug by printing arrays like normal python code, you can disable `jit` and `use_scan` using the command below.

```shell
export JAX_DISABLE_JIT=True; EXP=simply_local_test_1; rm -rf /tmp/${EXP}; python main.py --experiment_config TransformerLMTestNoScan --experiment_dir /tmp/${EXP} --alsologtostderr
```

## Dependencies

The main dependencies are:
[Jax](https://jax.readthedocs.io/en/latest/index.html) for model and training.
[Orbax](https://orbax.readthedocs.io/en/latest/) for checkpoint management.
[SeqIO](https://github.com/google/seqio) for data pipeline.

Install dependencies:
```
pip install -r requirements.txt
```

## Disclaimer

This is not an official Google Product.

## License

Unless explicitly noted otherwise, everything is released under the Apache2 license.

See the LICENSE file for the full license text.

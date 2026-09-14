# Evaluation, Decoding and Serving

Three entry points consume a trained checkpoint:

| Entry point | What it does |
|-------------|--------------|
| `simply.eval.decode_eval` | Batch decode + score a dataset in-process |
| `simply.serving.vanilla_server` | gRPC server, vanilla (non-paged) decoding |
| `simply.serving.page_server` | gRPC server, paged attention (TPU kernel) |
| `simply.eval.remote_decode_eval` | Same evals as `decode_eval`, against a server |

`decode_eval`, `vanilla_server` and `remote_decode_eval` are plain
JAX/XLA and run on CPU, GPU and TPU. `page_server` uses the Pallas
paged-attention kernel on TPU and a much slower reference
implementation elsewhere (see
[GPU feature support](gcloud.md#gpu-feature-support)).

## Prerequisites

```bash
pip install ".[tfds,math-eval,serving,assets]"  # +[zoo] for simply/zoo models
python setup/gen_protos.py                  # gRPC stubs, servers only
python setup/setup_assets.py                # checkpoints + tokenizers
python setup/setup_assets.py --datasets-only  # gsm8k / aime json files
```

Registered names used below come from the code, not from this doc:
`data_lib.DataSourceRegistry` (`simply:gsm8k_test`,
`simply:math500_test`, `simply:aime25`, ...),
`evaluation_lib.EvaluationRegistry` (`FewShotGSM8KEvaluation`,
`ZeroShotCoTBoxedInQuestionEvaluation`, ...) and
`lm_format.LMFormatRegistry` (`Pretrain`, `QwQChat`, `QwenV2Chat`,
`GemmaV2Chat`, ...).

## Offline decode eval

`--experiment_config` supplies the model architecture; model configs
such as `qwen3_4b` already point `init_ckpt_dir` at
`$SIMPLY_MODELS/<model>`, so `--ckpt_dir` is only needed to evaluate
your own training run.

```bash
python -m simply.eval.decode_eval \
    --experiment_config qwen3_4b \
    --experiment_dir /tmp/eval_qwen3_4b_gsm8k \
    --lm_format QwQChat \
    --evaluation ZeroShotCoTBoxedInQuestionEvaluation \
    --datasource_name simply:gsm8k_test \
    --batch_size 8 \
    --prefill_size 1024 \
    --max_decode_steps 1024 \
    --alsologtostderr
```

Evaluating a checkpoint written by `simply.main`:

```bash
python -m simply.eval.decode_eval \
    --experiment_config lm_test \
    --experiment_dir /tmp/eval_1 \
    --ckpt_dir /tmp/exp_1/checkpoints \
    --lm_format Pretrain \
    --evaluation FewShotGSM8KEvaluation \
    --datasource_name simply:gsm8k_train4 \
    --max_decode_steps 8 --prefill_size 128 --max_seq_len 256 \
    --alsologtostderr
```

Results are appended to `<experiment_dir>/history_<n>.jsonl` and the
run is resumable: `iter_state_<n>.json` records how far it got.
`--prefill_size` and `--intermediate_decode_steps` bound the number of
distinct jitted programs; leave them set for predictable compile times.

## gRPC server and remote eval

```bash
python -m simply.serving.vanilla_server \
    --experiment_config qwen3_4b \
    --lm_format QwQChat \
    --batch_size 4 \
    --max_seq_len 2048 \
    --max_decode_steps 512 \
    --simply_port 12345 \
    --alsologtostderr
```

Query it with any gRPC client (the stubs come from
`setup/gen_protos.py`):

```python
import grpc
from simply.serving import server_pb2_grpc, struct_pb2

stub = server_pb2_grpc.SimplyServiceStub(
    grpc.insecure_channel('localhost:12345'))
print(stub.Run(struct_pb2.Value(string_value='Hello world')))
```

Or run a full evaluation against the server -- same
`--evaluation`/`--datasource_name` registry names as `decode_eval`,
with `--num_eval_threads` client-side concurrency:

```bash
python -m simply.eval.remote_decode_eval \
    --server_address localhost:12345 \
    --experiment_dir /tmp/eval_remote \
    --evaluation ZeroShotCoTBoxedInQuestionEvaluation \
    --datasource_name simply:gsm8k_test \
    --max_decode_steps 1024 \
    --num_eval_threads 16 \
    --alsologtostderr
```

`page_server` takes the same flags plus `--page_size` (and the
`--ffn_weight_quant` / `--kv_cache_quant` options from
`simply/serving/common_flags.py`). Use it on TPU: off TPU it works but
falls back to the reference attention implementation.

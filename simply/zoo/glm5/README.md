# GLM-5.2 (`glm_moe_dsa`) in Simply

Self-link: this doc covers loading the open-weight
[GLM-5 / GLM-5.1 / GLM-5.2](https://github.com/zai-org/GLM-5) (744B-A40B,
HuggingFace architecture `GlmMoeDsaForCausalLM`, `model_type="glm_moe_dsa"`)
checkpoint into Simply to run inference efficiently.

## Package scope: architecture, not version

This package (`simply/zoo/glm5/`) is scoped to the **GLM-5-series
`glm_moe_dsa` architecture** (MLA attention + sigmoid/`noaux_tc` MoE + shared
expert), following the HuggingFace/vLLM convention of organizing by architecture
and expressing versions as config:

*   A new GLM version that **shares** this architecture (e.g. a 5.3 post-trained
    on the same base) is added as another config function in `config.py`
    (`glm5pN()`), reusing the model code — not a new folder.
*   A genuinely **different** architecture (e.g. GLM-5.3-Flash / `glm5_next`:
    KDA + mHC residual + multimodal) belongs in a **separate sibling plugin**
    (`zoo/glm5_next/`) that imports the shared primitives (`MLAAttention`, etc.)
    from here, rather than a version-suffixed folder.
## Architecture

GLM-5.2 is a DeepSeek-V3-style model:

| Component         | Value                                                   |
| ----------------- | ------------------------------------------------------- |
| hidden_size       | 6144                                                    |
| num_hidden_layers | 78 (+1 MTP layer, not modeled)                          |
| attention         | **MLA** (q_lora_rank=2048, kv_lora_rank=512,            |
:                   : qk_nope=192, qk_rope=64, v_head=256), 64 heads          :
| softmax scale     | 1/sqrt(256) (no mscale; `rope_type="default"`)          |
| RoPE              | interleaved, theta=8e6                                  |
| MoE               | 256 routed + 1 shared expert, top-8, sigmoid `noaux_tc` |
:                   : routing, routed_scaling=2.5                             :
| dense layers      | first 3 layers (`first_k_dense_replace=3`),             |
:                   : intermediate=12288                                      :
| moe intermediate  | 2048                                                    |
| norm              | RMSNorm, eps=1e-5                                       |
| vocab             | 154880, untied embeddings                               |
| DSA indexer       | lightning indexer + IndexShare (see below)              |

### What is and isn't modeled

*   **MLA attention** — modeled (`model_lib.MLAAttention`).
*   **GLM MoE routing** (sigmoid + `e_score_correction_bias` selection-only +
    renormalize + `routed_scaling_factor` + always-on shared expert) — modeled
    (`model_lib.MoEFeedForward` with `router_score_func='sigmoid'`).
*   **DSA "lightning indexer" (sparse top-k attention) + IndexShare** — **NOT**
    modeled. For context length T <= `index_topk` (2048), the indexer selects
    all causal keys, so dense causal MLA attention is numerically exact. For
    longer contexts the indexer becomes an inference-efficiency optimization (it
    does not change short-context outputs). The indexer weights are dropped on
    conversion.
*   **MTP layer** (index 78, next-token-prediction for speculative decoding) —
    **NOT** modeled; dropped on load, matching the HuggingFace implementation.

The forward pass has been validated bit-for-bit (to fp32 rounding, max-abs logit
diff ~7e-5) against the real HuggingFace `glm_moe_dsa` modeling code via a
standalone reference (`utils/glm5_reference.py`), and the end-to-end
conversion + Simply forward is covered by `utils/glm5_format_test.py`.

### Efficient decode

Decode uses the memory-efficient DeepSeek compact-latent MLA KV cache
(`use_latent_kv_cache`, on by default in `glm5p2`): only the kv-LoRA latent
(`kv_lora_rank=512`) plus one shared MQA RoPE key (`qk_rope=64`) are cached --
576 elements/token, ~57x smaller than the materialized 64-head K/V -- and `kv_b`
is folded into the query/context by absorption at attention time. Attention over
the cache is computed by a flash-style, KV-tiled, online-softmax kernel
(`latent_flash_decode`) that never materializes the full `[heads, q, kv]` score
tensor, so per-token decode cost stays ~flat with context instead of growing.
Both are bit-exact (within fp tolerance) with the materialized prefill math;
this is enforced by `model_lib_mla_test.py` (absorption-vs-materialized,
flash-vs-dense, and decode-vs-prefill parity). Training/prefill (empty cache)
uses the materialized einsum MLA path.

## Converting the checkpoint

Use the **BF16** HuggingFace release (not FP8):

```shell
python -m simply.tools.hf_to_orbax \
    --input_path=${HF_DIR}/GLM-5.2/ \
    --output_path=${CKPT_DIR}/GLM-5.2/ORBAX/ \
    --format=GlmMoeDsaFormat
```

The converted ORBAX checkpoint maps onto the param tree produced by the `glm5p2`
config. Also place the GLM tokenizer (`tokenizer.json`, `tokenizer_config.json`)
under the Simply vocab dir as `GLM-5.2` (registered in `data_lib.py`).

NOTE: at 744B the simple all-in-memory conversion in `hf_to_orbax.py` needs a
large host; for production conversion, stream shards (the format's `transforms`
is per-tensor and stacks experts lazily).

## Running in Simply

The experiment config is `glm5p2` (`config.py`), which sets `use_mla=True`,
the `mla_*` dims, `router_score_func='sigmoid'`,
`router_use_correction_bias=True`, `num_shared_experts=1`,
`routed_scaling_factor=2.5`, `first_k_dense_replace=3`, the efficient-decode
flags (`mla_use_latent_kv_cache=True`, `mla_latent_flash_decode=True`), and
`use_scan=False` (blocks are heterogeneous: dense + MoE). It loads the
pre-converted ORBAX checkpoint as `init_ckpt_format='V2Format'` (set
`init_ckpt_format='GlmMoeDsaFormat'` and point `init_ckpt_dir` at a raw HF
checkpoint to convert on load instead).

Point `GLM5P2_CKPT_DIR` (in `config.py`) at the converted ORBAX directory,
then launch like any other Simply model.

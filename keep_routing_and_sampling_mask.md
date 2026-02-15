# Keep Routing + Keep Sampling Mask: Implementation Guide

## Problem

MoE models activate only a subset of experts per token via a learned router.
During RL training, discrepancies between sampling (inference) and training
(different sharding, numerical precision, or policy updates) can cause
different expert routing for identical inputs, destabilizing optimization.
Similarly, top-p/top-k sampling masks out low-probability tokens, but the
current policy may mask differently, violating importance sampling
assumptions.

**Keep Routing**: Capture expert routing indices during sampling; replay
them during training so the same expert parameters are optimized.

**Keep Sampling Mask**: Capture the top-p/top-k boolean mask during
sampling; apply it when computing log-probs for the current policy during
training, ensuring identical action subspaces.

Both features are **opt-in** (disabled by default) and fully
backward-compatible with existing code paths. All 82 existing tests pass
with no regressions.

---

## Files Modified

| File | Summary |
|------|---------|
| `simply/config_lib.py` | 3 new fields on `RLExperimentConfig` |
| `simply/model_lib.py` | MoE capture/replay, SamplingState fields, continue_decode, LMInterface |
| `simply/utils/sampling_lib.py` | `sample_from_logits` returns mask; 2 new helper functions |
| `simply/rl_lib.py` | Padding, MaskedCategorical in loss, LMInterface construction |

---

## 1. Config Changes (`config_lib.py`)

Add three fields to `RLExperimentConfig`, after
`use_policy_logp_as_sampler_logp`:

```python
# Keep Routing: replay MoE routing indices from sampling during
# training.
keep_routing: bool = False
# Keep Sampling Mask: replay top-p/top-k masks from sampling
# during training.
keep_sampling_mask: bool = False
# Max number of token indices to store per position for
# keep_sampling_mask. Should be >= effective mask size (top_k, or
# upper bound for top_p).
max_sampling_mask_size: int = 256
```

---

## 2. Keep Routing: Model Changes (`model_lib.py`)

### 2a. MoEFeedForward.apply() — capture + forced routing

**Signature change**: Add `forced_routing_indices: Array | None = None`.

**Logic change**: After computing `router_logits` and `router_probs`, add
a branch:

```python
if forced_routing_indices is not None:
    # Use forced indices for expert selection, but still compute
    # router weights normally so the router gets gradients.
    selected_indices = forced_routing_indices
    if self.num_experts_per_token == 1:
        selected_router_probs = jnp.take_along_axis(
            router_probs, selected_indices, axis=-1)
    else:
        selected_router_logits = jnp.take_along_axis(
            router_logits, selected_indices, axis=-1)
        selected_router_probs = jax.nn.softmax(
            selected_router_logits, axis=-1)
elif self.num_experts_per_token == 1:
    # existing softmax=>topk path
    ...
else:
    # existing topk=>softmax path
    ...
```

**Always emit routing indices** (after the if/elif/else):

```python
extra_output['routing_indices'] = selected_indices
```

### 2b. TransformerBlock.apply() — thread per-block routing

Only for MoE blocks, read `_block_routing_indices` from `extra_inputs`
and pass to FFN. Non-MoE blocks keep the original call signature:

```python
if self.use_moe:
    forced_routing = None
    if extra_inputs:
        forced_routing = extra_inputs.get('_block_routing_indices')
    x, ffn_extra_output = self.ffn.apply(
        params['ffn'], x, inputs_mask=segment_ids != 0,
        forced_routing_indices=forced_routing)
else:
    x, ffn_extra_output = self.ffn.apply(
        params['ffn'], x, inputs_mask=segment_ids != 0)
```

### 2c. TransformerLM.apply() — inject routing per block

**Scan path**: Use the existing `_prepare_stack_list` to stack routing
indices across repeats. Add routing as a third element to the scan `xs`:

```python
routing_indices_dict = (extra_inputs or {}).get('routing_indices')
routing_stack_list = _prepare_stack_list(
    routing_indices_dict, n_repeats,
    len(self.config.block_attn_pattern))
```

Update `_process_per_repeat` to accept the third element and inject
`_block_routing_indices` into `block_extra_inputs` when present:

```python
block_params_list, block_decode_state_list, block_routing_list = p
...
for i in range(len(self.config.block_attn_pattern)):
    block_extra_inputs = extra_inputs
    if block_routing_list[i] not in (None, {}):
        block_extra_inputs = {
            **(extra_inputs or {}),
            '_block_routing_indices': block_routing_list[i],
        }
    ...
```

Update `jax.lax.scan` to pass the new xs:

```python
xs=(params_stack_list, decode_state_stack_list, routing_stack_list)
```

**Non-scan path**: Extract per-block routing and inject per iteration:

```python
ri_dict = (extra_inputs or {}).get('routing_indices')
for i in range(block_start_index, self.config.n_layers):
    ...
    block_extra_inputs = extra_inputs
    if ri_dict is not None:
        ri = ri_dict.get(f'block_{i}')
        if ri is not None:
            block_extra_inputs = {
                **(extra_inputs or {}),
                '_block_routing_indices': ri,
            }
    ...
```

**Note on ragged paged attention**: Ragged paged attention requires
`use_scan=False`, so it uses the non-scan path. The non-scan path
correctly threads routing per block, so keep_routing works for ragged
paged attention with no additional changes.

---

## 3. Keep Routing + Keep Sampling Mask: SamplingState (`model_lib.py`)

### 3a. New fields

```python
# Keep Routing: per-block routing indices accumulated during decoding.
# dict: {'block_i': Array [batch, decode_state_length+1, n_experts_per_tok]}
routing_indices: PyTree | None = None
# Keep Sampling Mask: compact indices of allowed tokens per position.
# [batch, decode_state_length+1, max_sampling_mask_size], int32
sampling_mask_indices: Array | None = None
```

### 3b. New methods

```python
def updated_routing_indices(self, step_routing):
    if self.routing_indices is None:
        return None
    return jax.tree.map(
        lambda buf, val: jax.lax.dynamic_update_slice_in_dim(
            buf, val, self.position + 1, axis=1),
        self.routing_indices, step_routing)

def updated_sampling_mask_indices(self, step_mask_indices):
    if self.sampling_mask_indices is None:
        return None
    return jax.lax.dynamic_update_slice_in_dim(
        self.sampling_mask_indices, step_mask_indices,
        self.position + 1, axis=1)
```

### 3c. Update pad_to()

Pad both new fields along axis=1 when present:

```python
routing_indices = self.routing_indices
if routing_indices is not None:
    routing_indices = jax.tree.map(
        lambda x: pad_to_along_axis(x, length + 1, axis=1),
        routing_indices)
sampling_mask_indices = self.sampling_mask_indices
if sampling_mask_indices is not None:
    sampling_mask_indices = pad_to_along_axis(
        sampling_mask_indices, length + 1, axis=1)
# Include both in dataclasses.replace(...)
```

---

## 4. continue_decode() (`model_lib.py`)

Detect features from init state (no new parameters needed):

```python
keep_routing = init_sampling_state.routing_indices is not None
keep_sampling_mask = init_sampling_state.sampling_mask_indices is not None
```

In `body_fn`, after `apply_fn()`:

**Sampling**: Call `sample_from_logits(..., return_mask=keep_sampling_mask)`.
Handle 2-tuple vs 3-tuple return.

**Routing accumulation**:

```python
if keep_routing:
    step_routing = {}
    ffn = extra_output.get('ffn', {})
    for k, v in ffn.items():
        if isinstance(v, dict) and 'routing_indices' in v:
            step_routing[k] = v['routing_indices']
    updated_routing = sampling_state.updated_routing_indices(step_routing)
```

**Mask accumulation**:

```python
if keep_sampling_mask:
    step_mask_indices = sampling_lib.mask_to_indices(
        sample_mask, max_sampling_mask_size)
    updated_mask_indices = sampling_state.updated_sampling_mask_indices(
        step_mask_indices)
```

Pass both to `dataclasses.replace(sampling_state, ...,
routing_indices=updated_routing,
sampling_mask_indices=updated_mask_indices)`.

---

## 5. LMInterface (`model_lib.py`)

### 5a. __init__()

Add parameters:

```python
keep_routing: bool = False,
keep_sampling_mask: bool = False,
max_sampling_mask_size: int = 256,
```

Store as `self.keep_routing`, etc.

### 5b. generate() — buffer initialization

After computing `token_scores` and before creating `SamplingState`:

**Routing buffers** (when `self.keep_routing` and model is MoE):

```python
routing_indices = {}
for i in range(model_config.n_layers):
    if block_i is MoE:
        routing_indices[f'block_{i}'] = jnp.zeros(
            (batch_size, decode_state_length + 1, n_experts_per_tok),
            dtype=jnp.int32)
```

Then fill prefill routing from `extra_output['ffn']`:

```python
prefill_ffn = extra_output.get('ffn', {})
for block_key, block_ffn in prefill_ffn.items():
    if 'routing_indices' in block_ffn and block_key in routing_indices:
        ri = block_ffn['routing_indices']
        routing_indices[block_key] = jax.lax.dynamic_update_slice_in_dim(
            routing_indices[block_key], ri, 0, axis=1)
```

**Sampling mask buffer** (when `self.keep_sampling_mask`):

```python
sampling_mask_indices = jnp.full(
    (batch_size, decode_state_length + 1, self.max_sampling_mask_size),
    vocab_size, dtype=jnp.int32)  # vocab_size is the sentinel value
```

Pass both to `SamplingState(...)`.

### 5c. generate() — post-processing

After the decode loop, allgather both:

```python
if sampling_state.routing_indices is not None:
    all_routing = jax.tree.map(
        lambda x: np.asarray(
            jax.experimental.multihost_utils.process_allgather(x, tiled=True)),
        sampling_state.routing_indices)

if sampling_state.sampling_mask_indices is not None:
    all_sampling_mask = np.asarray(
        jax.experimental.multihost_utils.process_allgather(
            sampling_state.sampling_mask_indices, tiled=True))
```

Per sample `i`, slice and include in `ProcessedInput.extra_inputs`:

```python
total_len = len(input_token_ids) + len(output_token_ids)
sample_extra_inputs = dict(unpadded_inputs[input_index].extra_inputs or {})
if all_routing is not None:
    sample_routing = {k: v[i, :total_len, :] for k, v in all_routing.items()}
    sample_extra_inputs['routing_indices'] = sample_routing
if all_sampling_mask is not None:
    sample_extra_inputs['sampling_mask_indices'] = all_sampling_mask[i, :total_len, :]
sample_processed_input = dataclasses.replace(
    unpadded_inputs[input_index], extra_inputs=sample_extra_inputs or None)
```

Use `sample_processed_input` instead of `unpadded_inputs[input_index]`
when constructing `SamplingOutput`.

---

## 6. Sampling Helpers (`utils/sampling_lib.py`)

### 6a. sample_from_logits() — return mask

Add `return_mask: bool = False` parameter.

All internal functions (`greedy_fn`, `simple_sample_fn`,
`masked_sample_fn`) now return a 3-tuple `(tokens, logprobs, mask)`.
For greedy and simple sampling, the mask is `jnp.zeros(logits.shape,
dtype=jnp.bool_)` (all-False = no masking constraint applied).

At the end:

```python
result = jax.lax.cond(...)  # always returns 3-tuple now
if return_mask:
    return result
return result[0], result[1]
```

This preserves backward compatibility — callers that don't pass
`return_mask` still get a 2-tuple.

### 6b. New helper functions

```python
def mask_to_indices(mask, max_size):
    """Bool mask [..., vocab] -> indices [..., max_size].
    Unset positions padded with vocab_size (out-of-range sentinel)."""
    vocab = mask.shape[-1]
    keyed = jnp.where(mask, jnp.arange(vocab), vocab)
    sorted_indices = jnp.sort(keyed, axis=-1)
    return sorted_indices[..., :max_size]


def indices_to_mask(indices, vocab_size):
    """Indices [..., max_size] -> bool mask [..., vocab_size].
    Indices >= vocab_size are treated as padding (ignored).
    Positions where ALL indices are padding (no mask recorded)
    are treated as unmasked (all-True)."""
    one_hot = jax.nn.one_hot(indices, vocab_size + 1, dtype=jnp.bool_)
    mask = jnp.any(one_hot[..., :vocab_size], axis=-2)
    no_constraint = ~jnp.any(mask, axis=-1, keepdims=True)
    return mask | no_constraint
```

These are invertible: `indices_to_mask(mask_to_indices(m, k), V)` recovers
`m` when `k >= number of True entries in m`. Positions with no masking
(all-False mask from greedy/simple sampling) round-trip through all-sentinel
indices and are reconstructed as all-True (no constraint).

---

## 7. RL Pipeline (`rl_lib.py`)

### 7a. RLTrainingExampleBatch.pad_sequences()

Pad routing and mask arrays in `extra_inputs` along the sequence axis:

```python
extra_inputs = self.extra_inputs
if extra_inputs:
    extra_inputs = dict(extra_inputs)
    if 'routing_indices' in extra_inputs:
        extra_inputs['routing_indices'] = {
            k: model_lib.pad_to_along_axis(v, to_length, axis=0)
            for k, v in extra_inputs['routing_indices'].items()
        }
    if 'sampling_mask_indices' in extra_inputs:
        extra_inputs['sampling_mask_indices'] = (
            model_lib.pad_to_along_axis(
                extra_inputs['sampling_mask_indices'],
                to_length, axis=0))
# Include extra_inputs=extra_inputs in dataclasses.replace(...)
```

**Note**: The routing arrays in `extra_inputs` are per-example (no batch
dimension) at this point — they have shape `[seq_len, n_experts_per_tok]`
— so we pad along `axis=0`. Same for `sampling_mask_indices` with shape
`[seq_len, max_sampling_mask_size]`.

### 7b. compute_ppo_loss() — MaskedCategorical

After obtaining `logits` from `model.apply()`:

```python
sampling_mask_indices = (batch.extra_inputs or {}).get(
    'sampling_mask_indices')
if sampling_mask_indices is not None:
    vocab_mask = sampling_lib.indices_to_mask(
        sampling_mask_indices, logits.shape[-1])
    m = distributions.MaskedCategorical(
        logits, mask=vocab_mask,
        neg_inf=common.neg_inf(logits.dtype))
else:
    m = distributions.Categorical(logits)
```

### 7c. compute_logprobs() — same MaskedCategorical logic

Same pattern as compute_ppo_loss: check for
`sampling_mask_indices` in `extra_inputs`, use `MaskedCategorical` when
present.

### 7d. run_experiment() — pass config flags to LMInterface

```python
lm_interface = model_lib.LMInterface(
    ...,
    keep_routing=getattr(config, 'keep_routing', False),
    keep_sampling_mask=getattr(config, 'keep_sampling_mask', False),
    max_sampling_mask_size=getattr(config, 'max_sampling_mask_size', 256),
)
```

---

## 8. Data Flow Summary

```
Sampling (generate)
  |
  +--> prefill_fn() returns extra_output with routing per MoE block
  |    --> written into routing_indices buffers at positions [0, prefill_size)
  |
  +--> continue_decode() loop:
  |    - model.apply() returns extra_output['ffn']['block_i']['routing_indices']
  |    - sample_from_logits(return_mask=True) returns sampling mask
  |    - routing accumulated via dynamic_update_slice_in_dim
  |    - mask converted to compact indices via mask_to_indices, accumulated
  |
  +--> Post-processing: allgather, slice per sample
       --> stored in SamplingOutput.processed_input.extra_inputs:
           {'routing_indices': {'block_0': array, ...},
            'sampling_mask_indices': array}

Training (RL)
  |
  +--> create_train_batch():
  |    - extra_inputs flow through from SamplingOutput.processed_input
  |    - pad_sequences() pads routing/mask along sequence axis
  |    - pytree_ragged_stack_allgather() stacks across processes
  |
  +--> model.apply(extra_inputs=batch.extra_inputs):
  |    - TransformerLM extracts 'routing_indices' from extra_inputs
  |    - Per-block routing injected as '_block_routing_indices'
  |    - MoEFeedForward uses forced_routing_indices instead of router's topk
  |    - Router still computes logits/probs normally (gets gradients)
  |
  +--> compute_ppo_loss() / compute_logprobs():
       - Extracts 'sampling_mask_indices' from extra_inputs
       - Converts to bool mask via indices_to_mask()
       - Uses MaskedCategorical for log_prob computation
       - Ensures identical action space as during sampling
```

---

## 9. Memory Overhead

| Feature | Per sample (60 MoE layers, 8 experts/tok) | Batch of 128 @ 32k tokens |
|---------|-------------------------------------------|---------------------------|
| Routing | 60 * 32k * 8 * 4B = ~60 MB | ~7.5 GB |
| Sampling Mask | 32k * 256 * 4B = ~32 MB | ~4 GB |
| **Total** | ~92 MB | ~11.5 GB |

---

## 10. Usage

Enable in any `RLExperimentConfig`:

```python
config = dataclasses.replace(
    config,
    keep_routing=True,          # replay MoE routing from sampling
    keep_sampling_mask=True,    # replay top-p/top-k mask from sampling
    max_sampling_mask_size=256, # adjust if using top_k > 256
)
```

No other changes needed — the pipeline handles everything automatically.

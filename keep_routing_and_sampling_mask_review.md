# Keep Routing & Keep Sampling Mask: Review Notes

This document captures the design review, iterative improvements, bugs
found, and future recommendations from the code review session. It
complements `keep_routing_and_sampling_mask.md` (the implementation
guide) with the "why" behind each decision.

---

## 1. Review Summary

The implementation was reviewed over 6 iterations. All 330 existing
tests pass throughout. One correctness bug was found and fixed. Several
code quality improvements were made.

### Changes Made During Review

| Iteration | Change | Rationale |
|-----------|--------|-----------|
| 1 | Always unpack 3-tuple from `sample_from_logits` in `continue_decode` | Eliminated awkward if/else unpacking; XLA DCE handles unused mask |
| 1 | Replaced `\` line continuation with parenthesized expression | Python style: prefer parens over backslash |
| 1 | Removed unused `vocab_size` variable in `sample_from_logits` | Dead code |
| 2 | Extracted `_make_categorical()` helper in `rl_lib.py` | Deduplicated identical 8-line blocks in `compute_logprobs` and `compute_ppo_loss` |
| 3 | Added comment on `not in (None, {})` check | Clarified non-obvious sentinel semantics |
| 3 | Changed `getattr(config, 'keep_routing', False)` to `config.keep_routing` | These are proper dataclass fields; defensive `getattr` was unnecessary |
| 4 | Extracted `model_config` and `buf_shape` in `generate()` | Eliminated deeply nested `getattr` chains; reduced repetition |
| 4 | Simplified prefill routing write loop | Iterate over `routing_indices` keys with chained `.get()` instead of filtering items |
| 5 | **Bug fix**: greedy/simple mask changed from all-True to all-False | Critical correctness fix (see Section 2) |
| 5 | Added no-constraint fallback in `indices_to_mask` | Handles all-sentinel positions as "allow all tokens" |
| 6 | Routing dict comprehension in post-processing | Replaced 4-line loop with 3-line comprehension |

---

## 2. Bug Found: All-True Mask Truncation

### The Problem

When greedy or simple sampling (no top-k/top-p) is used with
`keep_sampling_mask=True`, the original code returned
`jnp.ones(logits.shape, dtype=jnp.bool_)` — an all-True mask meaning
"all tokens allowed." This mask then flows through:

```
all-True mask (vocab_size=128k)
  → mask_to_indices(mask, max_size=256)
  → [0, 1, 2, ..., 255]  (first 256 indices only!)
  → indices_to_mask([0..255], vocab_size=128k)
  → True for tokens 0-255, False for tokens 256-128k
  → MaskedCategorical INCORRECTLY masks out most of the vocabulary
```

This would silently produce wrong log_prob values during training,
causing incorrect policy gradients.

### The Fix

Two changes:

1. **`sample_from_logits`**: `greedy_fn` and `simple_sample_fn` now
   return `jnp.zeros(logits.shape, dtype=jnp.bool_)` (all-False)
   instead of all-True. Semantics: all-False = "no masking constraint
   was applied."

2. **`indices_to_mask`**: Added fallback for positions where all
   indices are sentinel (>= vocab_size):
   ```python
   no_constraint = ~jnp.any(mask, axis=-1, keepdims=True)
   return mask | no_constraint
   ```
   All-sentinel → all-False mask → detected → expanded to all-True.

### Verification

```python
# All-False (no constraint) round-trips correctly:
mask = jnp.zeros((1, 10), dtype=jnp.bool_)
indices = mask_to_indices(mask, max_size=4)   # [10, 10, 10, 10]
result = indices_to_mask(indices, vocab_size=10)  # all-True ✓

# Partial mask round-trips correctly:
mask = jnp.array([[True, False, True, ...]])
indices = mask_to_indices(mask, max_size=4)
result = indices_to_mask(indices, vocab_size=10)  # matches original ✓
```

### Lesson

When compressing data through a lossy representation (bool mask →
fixed-size index array), edge cases where the original data exceeds
the representation capacity must be handled explicitly. The all-True
mask has `vocab_size` True entries but `max_size << vocab_size`, so
it cannot round-trip through the index representation.

---

## 3. Key Design Decisions

### 3a. Why `extra_inputs` for threading routing?

The routing indices flow from sampling output into training via
`extra_inputs`, the existing "optional side-channel" dict. Two keys
are used:

- `'routing_indices'`: `{'block_0': array, ...}` — dict of per-block
  routing, keyed by block name
- `'_block_routing_indices'`: per-block routing injected by
  `TransformerLM` into each block's `extra_inputs`

The `_` prefix on `_block_routing_indices` follows the convention for
internal-only keys that shouldn't leak to user code.

**Alternative considered**: Adding a new parameter to `model.apply()`.
Rejected because it would change the core forward pass signature,
affecting all callers.

### 3b. Why compact index representation for masks?

Storing the full bool mask `[batch, seq_len, vocab_size]` would be
prohibitively large for large vocabularies (e.g., 128k tokens).
The compact index representation `[batch, seq_len, max_size]` with
`max_size=256` is ~500x smaller.

The tradeoff: lossy if the actual mask has more True entries than
`max_size`. In practice, `top_k` is typically small (≤256) and
`top_p` retains a small fraction of the vocabulary.

### 3c. Why detect features from `init_sampling_state`?

In `continue_decode`, instead of adding new boolean parameters, the
code detects whether routing/mask buffers are active by checking
`init_sampling_state.routing_indices is not None`. This is cleaner
because:

- No new parameters to thread through `jax.jit` calls
- The presence of buffers IS the source of truth
- Adding a new buffer doesn't require changing `continue_decode`'s
  signature

### 3d. Why `not in (None, {})` for routing check?

In the scan path, `_prepare_stack_list` returns:
- `[None, None, ...]` when routing is disabled entirely
- `[stacked_array, {}, ...]` when routing is enabled but some
  blocks are not MoE (the `{}` comes from `tree.get(key, {})`
  default on missing blocks, stacked via `jax.tree.map`)

Both `None` and `{}` mean "no routing for this block." The check
`not in (None, {})` handles both cases. A comment was added to
explain this.

### 3e. Why always `return_mask=True` in `continue_decode`?

The original code conditionally unpacked 2 or 3 values:
```python
if keep_sampling_mask:
    tokens, logprobs, mask = sample_result
else:
    tokens, logprobs = sample_result
```

This was simplified to always unpack 3 values. When
`keep_sampling_mask=False`, the mask is unused and XLA's dead code
elimination removes it. The simplified code is easier to read and
avoids branching on the return type.

---

## 4. Codebase Patterns Learned

### 4a. Registry pattern

All extensible components use `@Registry.register` + `@dataclass`.
Config fields with defaults are the standard extension point. The
new `keep_routing`, `keep_sampling_mask`, `max_sampling_mask_size`
fields follow this pattern.

### 4b. `extra_inputs` as side-channel

The `extra_inputs` dict flows through the entire model stack:
`TransformerLM.apply` → `TransformerBlock.apply` → `FeedForward.apply`.
It's the established pattern for passing optional per-forward data
without changing core signatures. Keys like `'prefill_position'`
already use this pattern.

### 4c. Scan path vs non-scan path

`TransformerLM` supports two execution modes:
- **Scan**: `jax.lax.scan` over repeated block patterns. Parameters
  and decode state are stacked across repeats via
  `_prepare_stack_list`. Routing indices use the same stacking.
- **Non-scan**: Direct per-block loop. Used by ragged paged
  attention. Per-block routing is injected directly.

Both paths must be updated when adding per-block data.

### 4d. SamplingState as JAX PyTree

`SamplingState` is `@jax.tree_util.register_dataclass` + frozen.
It's carried through `jax.lax.while_loop`, so its tree structure
must be fixed at trace time. New fields with `None` default work
because JAX treats `None` as a valid leaf.

### 4e. Multi-host allgather

After decoding, per-device data is gathered via
`jax.experimental.multihost_utils.process_allgather(x, tiled=True)`.
The `tiled=True` flag means the data is already sharded across
devices and should be concatenated (not replicated). Both routing
and mask buffers follow the same allgather pattern.

### 4f. Line length and style

- 80-character line limit
- 4-space hanging indents for continuation lines
- 2-space indentation for nested blocks
- Comments above code, not inline
- `(extra_inputs or {}).get('key')` for safe dict access

---

## 5. Extensibility Assessment

### Current State: Ad-hoc

Adding a new per-token buffer (e.g., attention entropy) requires
~14 modifications across 3 files:

```
config_lib.py:
  1. Config field

model_lib.py (10 locations):
  2. SamplingState field
  3. SamplingState.updated_*() method
  4. SamplingState.pad_to() conditional block
  5. LMInterface.__init__() parameter
  6. LMInterface.__init__() storage
  7. LMInterface.generate() buffer initialization
  8. LMInterface.generate() SamplingState construction
  9. LMInterface.generate() allgather
 10. LMInterface.generate() per-sample extraction
 11. continue_decode() feature detection
 12. continue_decode() body_fn extraction + accumulation

rl_lib.py (2 locations):
 13. RLTrainingExampleBatch.pad_sequences()
 14. run_experiment() LMInterface construction
```

### Recommended Improvement: Unified `extra_buffers`

Consolidate `routing_indices` and `sampling_mask_indices` into a
single `extra_buffers: PyTree | None` field on `SamplingState`.

```python
@dataclasses.dataclass
class SamplingState:
    ...
    extra_buffers: PyTree | None = None
    # e.g., {
    #     'routing_indices': {'block_0': Array, ...},
    #     'sampling_mask_indices': Array,
    #     'attention_entropy': {'block_0': Array, ...},
    # }
```

This would make generic:
- `pad_to()` — single `jax.tree.map` over `extra_buffers`
- `updated_extra_buffers()` — single method with
  `dynamic_update_slice_in_dim`
- Allgather — single `jax.tree.map` block
- Per-sample extraction — single dict comprehension

Remaining buffer-specific logic (always needed):
- Buffer initialization (shape depends on buffer type)
- Data extraction from `extra_output` in `body_fn`
- Consumption during training

This would reduce new-buffer touchpoints from ~14 to ~5.

---

## 6. Memory Overhead Reference

| Feature | Per sample | Formula |
|---------|-----------|---------|
| Routing | `n_moe_layers * seq_len * n_experts_per_tok * 4B` | 60 layers, 32k tokens, 8 experts → ~60 MB |
| Sampling Mask | `seq_len * max_sampling_mask_size * 4B` | 32k tokens, 256 size → ~32 MB |
| **Total** | | ~92 MB per sample |

For a batch of 128 samples at 32k tokens: ~11.5 GB total.

---

## 7. Testing Notes

- All 330 existing tests pass with no modifications needed
- The features are opt-in (disabled by default), so existing tests
  exercise the `None` / disabled paths
- No new tests were added for the features themselves
- The `mask_to_indices` / `indices_to_mask` roundtrip was verified
  manually for: all-False mask, partial mask, and top-k style mask

### Recommended Future Tests

1. End-to-end test: enable `keep_routing=True` on a small MoE model,
   verify that routing indices from sampling match those used in
   training forward pass
2. End-to-end test: enable `keep_sampling_mask=True` with `top_k`,
   verify log_probs match between sampling and training
3. Edge case: `max_sampling_mask_size < top_k` — verify graceful
   degradation (truncation, not crash)
4. Edge case: non-MoE model with `keep_routing=True` — verify no-op
5. Multi-host: verify allgather correctness for routing/mask buffers

---

## 8. Files in This Branch

| File | Purpose |
|------|---------|
| `keep_routing_and_sampling_mask.md` | Implementation guide — what each change does |
| `keep_routing_and_sampling_mask_review.md` | This file — review notes, bugs, learnings, recommendations |
| `simply/config_lib.py` | 3 new config fields |
| `simply/model_lib.py` | Core implementation (~210 lines added) |
| `simply/rl_lib.py` | Training-side integration (~30 lines added) |
| `simply/utils/sampling_lib.py` | Sampling helpers (~40 lines added) |

# Ragged Paged Attention (RPA)

This document explains the Ragged Paged Attention mechanism in Simply, covering
its core components, the lifecycle of a query, and how outputs are generated.

## Overview

Ragged Paged Attention is a memory-efficient attention mechanism designed for
high-throughput serving of Large Language Models (LLMs). It manages Key-Value
(KV) caches using non-contiguous memory "pages" (similar to OS virtual memory),
allowing flexible allocation and handling of ragged (variable-length) batches.

## Key Benefits

*   **Memory Efficiency**: Eliminates memory fragmentation by allocating KV
    cache in non-contiguous pages, similar to OS virtual memory.
*   **High Throughput**: Enables continuous batching and efficient
    variable-length processing (ragged attention), maximizing TPU utilization.
*   **Mixed Prefill and Decode**: Seamlessly batches new prompts (prefill) with
    ongoing generations (decode) in the same forward pass, reducing scheduling
    latency and improving overall system efficiency.
*   **Flexible Scaling**: Decouples logical sequence length from physical memory
    allocation, allowing for longer contexts and dynamic generation.

## Key Components

### 1. `DecodeState` (Physical Memory Management)

Defined in `utils/ragged_paged_attention.py`.

*   **Role**: Manages the physical KV cache memory.
*   **Structure**:
    *   `pages`: A large pre-allocated tensor storing all KV blocks
        `[total_num_pages, page_size, num_kv_heads * 2 // kv_packing,
        kv_packing, padded_per_head_dim]`.
    *   `page_indices`: Maps logical sequence positions to physical page indices
        `[batch_size, max_num_pages_per_seq_per_shard]`.
    *   `available_page_indices`: A stack of free page indices.
*   **Key Operations**:
    *   `allocate`: Assigns new pages to sequences that need more space.
    *   `insert`: Writes new K/V embeddings into the allocated pages.
    *   `release`: Frees pages when a sequence finishes.
    *   `update_decode_state_and_compute_attn`: Updates decode state and
        computes attention.

### 2. `SamplingState` (Logical Batch Management)

Defined in `utils/ragged_paged_attention.py`.

*   **Role**: Manages the logical state of the current batch of requests.
*   **Structure**:
    *   `tokens`: Stores the generated tokens for each request.
    *   `position`: Current token position for each request.
    *   `input_lens`: Length of the prompt/prefix for each request.
    *   `rank`: Queue system to determine which requests get processed.
*   **Key Operations**:
    *   `push`: Adds new requests to the batch.
    *   `get`: Retrieves generated tokens and scores.
    *   `release`: Removes finished requests from the batch.

### 3. `Batcher` & `SimplyService` (Orchestration)

`Batcher` is defined in `serving/page_batcher.py`, `SimplyService` in
`serving/page_server.py`.

*   **Role**: Handles the server loop, gRPC interface, and coordination between
    the model and the RPA state.
*   **Structure**:
    *   `request_queue`: Buffers incoming user queries.
    *   `loop`: The main background thread that continuously runs the decoding
        steps.

## Life of a Query

Here describes how a query travels through the system from request to
completion.

### 1. Request Arrival

A client sends a `Run` request via gRPC to `SimplyService`.

*   The request is put into `Batcher.request_queue`.
*   A `Future` is created to await the result.

### 2. Batch Scheduling & Ingestion

The `Batcher.loop` continuously checks for new requests and free slots in the
`SamplingState`.

*   **Check Availability**: If there are free slots (i.e., `is_pad_seq` is true
    for some batch indices), new requests are popped from the queue.
*   **Tokenization**: The prompt text is tokenized (via `input_processor`).
*   **Push to State**: `SamplingState.push` assigns the request to a specific
    batch index (slot).
    *   It initializes `tokens`, `input_lens`, and resets `position` to 0.
    *   The request enters the decoding queue.

### 3. Execution Phase (The Decoding Loop)

The Loop runs `compiled_decode_fn` (wrapping `model.apply` and RPA logic)
repeatedly.

*   **Token Selection (`issue_lens`)**:

    *   Before running the model, `SamplingState.issue_lens` determines how many
        tokens to process for each sequence.
    *   **Priority**: It respects the `rank` (requests with smaller rank indices
        are prioritized).
    *   **Capacity**: It fills the available batch capacity (e.g.,
        `max_num_issue_tokens`) greedily.
    *   **Mixed Batching**: It automatically handles both prefill (issuing all
        prompt tokens) and decoding (issuing 1 token) in the same step.
    *   **Ragged Gathering**: The selected tokens are gathered into a ragged
        tensor via `ragged_issue_tokens`, forming the actual input for the
        forward pass.

*   **KV Allocation**:

    *   `DecodeState.allocate` checks if sequences need new pages for the next
        token(s) and grabs them from `available_page_indices`.

*   **Model Forward Pass**:

    *   The model computes Query, Key, and Value embeddings.
    *   **KV Insertion**: `DecodeState.insert` writes the new K/V data into the
        physical pages mapped by `page_indices`.
    *   **Attention**: `ragged_paged_attention` kernel computes attention using
        the paged KV cache.

*   **Sampling**:

    *   The model outputs logits.
    *   New tokens are sampled (greedy, top-k, etc.).
    *   `SamplingState` updates the `tokens` array with the new token.
    *   `position` is incremented.

### 4. Output Generation & Completion

After each step, the `Batcher` checks for completion.

*   **Check Status**: `completed_mask` identifies sequences that have hit the
    EOS token or max length.
*   **Retrieval**: `SamplingState.get(completed_mask)` extracts the full
    generated sequence:
    *   It slices `tokens` based on `input_len` and current `position`.
    *   Decodes token IDs back to text.
*   **Response**: The result text is set on the `Future`, completing the gRPC
    call for the client.

### 5. Cleanup

*   **Release**: `SamplingState.release` (and `DecodeState.release`) is called
    for completed sequences.
    *   Physical pages are returned to `available_page_indices`.
    *   The batch slot is marked as padding (`is_pad_seq=True`), making it
        available for a new request.

## Sliding windows: holes, breakpoints and prefix caching

A windowed layer's paged KV cache is deliberately **incomplete**, and anything
that reads `pages` directly (prefix caching, disaggregated prefill) has to know
the rule.

### What the kernel writes back

Newly issued KV is written into `pages` from the query block that runs the
*last* query of the pass (`bq_idx == num_bq - 1`), and only over the BKV blocks
that block visits. With `sliding_window` narrower than what a pass issues, the
sliding-window skip starts *above* the first BKV block holding new KV, so the
KV under it is computed, window-masked and then dropped -- a **hole** inside
`kv_lens`.

That is intentional and cheap. The only KV a later query can legitimately read
is the window of the pass boundary, and the write-back block always covers it:

> After a pass that ends at `kv_len == P`, the paged cache is correct over
> `[P - window_size, P)`.

Everything below the hole is out of window for every query that will ever run,
so it is score-masked; the kernel additionally **zeroes V** for those keys
(`flash_attention_step1_qk_softmax`) so that a never-written page holding
uninitialised `NaN` cannot leak into the output via `0 * NaN = NaN` in the PV
matmul. Global layers (`window_size is None`) have no skip and therefore no
holes.

### Pass boundaries and per-token validity

A pass boundary is the only position at which a windowed layer's KV says
anything trustworthy, and what it says is an EXTENT rather than a yes/no.

*   `DecodeState.chunk_written_mask(chunk_idx, position)` returns, per token,
    which of a chunk's KV is real when captured at `position`:
    `[max(chunk_start, position - window_size), min(chunk_end, position))`.
    A global layer is the same rule with an infinite window, so nothing here
    special-cases the two kinds. It is what `written` carries in
    `DecodeState.extract_chunk`.
*   Captures of the same chunk taken at different boundaries cover different
    tokens and **union** (`rpa.merge_chunk_trees`), per leaf and per token.
    Nothing is overwritten, so the outcome does not depend on which prompt
    ran first, and a chunk that no single pass can capture whole (any
    `window < chunk`) can still become whole by accumulation. A capture that
    adds nothing is skipped before the extract is even issued.
*   Whether a position can be resumed from is **whether the cache holds a
    resume point there**, meaning "a pass ended here", and reading it needs no
    window arithmetic: at a pass boundary the writer has captured every chunk
    the window touches, so coverage holds by construction, and eviction that
    only ever truncates AT a resume point, plus union-only merging, mean it
    cannot be lost afterwards. A node keeps its resume points in
    `PrefixNode.resumables`, a sorted sparse list of `(position, key)` --
    PRESENCE IS THE BIT, so there is no disabled-but-present state to test
    for. `PrefixNode.deepest_resumable(limit)` is the one query (a `bisect`
    over a handful of entries), and `PrefixNode.resumable_positions` lists
    them.
*   The batcher snapshots after **every** prefill pass. Passes are *not*
    aligned to the chunk grid: a boundary lands wherever the scheduler's
    token budget ran out.
*   `Batcher._maybe_snapshot_prefix_cache` therefore walks FORWARD from a
    persistent per-slot token frontier, building one `ChunkTile(tree, start,
    end)` per chunk the slot has newly crossed -- the last of them ending AT
    `position`, mid-chunk -- and handing the run to `PrefixCache.store_tiles`,
    which navigates, creates what is missing, fills, and marks `position` as
    a resume point. So the next prompt sharing the prefix resumes there rather
    than at the chunk boundary below it. Offload (HBM->host) happens in the
    batcher, via `rpa.offload_chunk_tree`, before the tile is handed over. One
    capture per chunk is enough for every later boundary of that slot, because
    `p >= p1` implies `p - W >= p1 - W`: the need only shrinks.
*   Reading back is `PrefixCache.restore_chunk_tiles(tokens, start, end)`,
    which returns the tiles to inject for the deepest resume point at or below
    `end`; the caller `rpa.onload_chunk_tree`s each one and issues
    `DecodeState.inject_chunk(slot_id, payload, start, end)`. That signature
    is the whole contract: absolute token positions, the slot gains exactly
    `end - start` tokens, and the payload is the chunk `start` falls in.
*   The cache indexes those positions in a path-compressed (radix) trie over
    **tokens** -- an edge is a run of tokens, a node exists at a branch point
    or where a store split one, and children are keyed by the first token of
    their edge. There is no hashing: a descent is a dict lookup plus one
    vectorised edge comparison per node. Payloads stay chunk-granular, with
    one at every chunk boundary below a stored position, because a restore
    injects every chunk along the path.

### Eviction

The cache is capped in host RAM (`PrefixCache.max_bytes`) and `evict()` trims
to it. What it ranks is **resume points, not nodes**: `PrefixCache.lru` is an
`OrderedDict` keyed per resume point, whose value is the token prefix that
ends there -- so the node to reclaim from is DERIVED (`navigate_to_known`) when
the time comes and never stored, and a split or a fold leaves nothing stale.

One rule, applied to whatever the order hands over, coldest first: the popped
resume point stops being one. What that gives back depends on where it sat.

*   If it was its node's deepest, the tokens above the next resume point down
    are unreachable, and the node is cut back to that one -- or dropped whole
    when there is none. Cuts therefore land ON a resume point, which is what
    keeps every surviving prefix whole and restorable.
*   If a hotter resume point sits above it, popping frees nothing yet; the
    entry leaves the order all the same, so the loop always makes progress.
    The consequence is worth knowing: a prompt whose deepest stop stays hot
    has its shallow stops spent for nothing, and then goes WHOLE when the
    deepest is finally reached.

A chunk straddling a node boundary is ONE buffer shared by every node whose
interval touches it, so freeing is by reachability, not by position -- when the
keeper hands its straddler to a child, the walk goes straight up the ancestors
that still end in that chunk and repoints them, and only then is the buffer
really gone.

Injected chunks that are *not* trustable (a windowed layer's older chunks) are
stored as `ShapeDtypeStruct` placeholders and re-injected as uninitialised
buffers; `release_for_window` evicts them before they can feed into attention,
and the V mask makes any leftover harmless.

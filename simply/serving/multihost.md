# Multihost Page Server

This document describes how the `page_server.py` handles multihost
(multi-process JAX) serving, where TPU workloads are distributed across
multiple JAX processes.

## Overview

In a multihost setup, all JAX processes participate in collective operations
(e.g., `jax.jit`-compiled functions), but only **task 0** (the primary
task) handles gRPC serving and request management. The key challenge is
keeping all processes in lockstep for JAX collective operations while allowing
task 0 to independently manage request I/O.

## Architecture

```
Task 0 (Primary)                 Other Tasks (Workers)
┌──────────────────────┐         ┌──────────────────────┐
│  gRPC Server         │         │                      │
│  Request Queue       │         │                      │
│  Future Management   │         │                      │
│  ──────────────────  │         │                      │
│  Batcher Loop        │◄───────►│  Batcher Loop        │
│  (collective JAX)    │sum_hosts│  (collective JAX)    │
└──────────────────────┘         └──────────────────────┘
```

### Process Roles

*   **Task 0 (Primary)**: Runs the gRPC server, manages the request queue,
    encodes inputs, sets futures with results. Identified by
    `experiment_helper.is_primary_task()`.
*   **Non-primary tasks**: Participate only in JAX collective operations
    (`push`, `decode`, `release`). They have no access to request/future
    objects.

## Batcher Loop: Two-Phase Design

The batcher loop on each iteration has two phases: **request acquisition**
(task 0 only) and **collective execution** (all processes).

### Phase 1: Request Acquisition (Task 0 Only)

Task 0 calls `_try_get_request()`, which internally:

1.  Pops a request from the queue (with a deadline-based timeout).
2.  Skips cancelled futures and failed encodings by retrying internally.
3.  Returns `(request, future, input_tokens)` on success, or `None` on timeout.

All retry logic (cancelled futures, encoding errors) is **local to task 0**.
No collective operations occur during this phase.

### Phase 2: Collective Execution (All Processes)

After task 0 finishes request acquisition:

1.  `sum_across_hosts(input_len)`: Task 0 contributes the token count; all
    others contribute 0. The sum gives every process the correct value.
    *   If `n == 0`: No valid request. All processes either `break` (to decode
        pending items) or `continue` (heartbeat when idle).
    *   If `n > 0`: A valid request exists. Proceed to push.
2.  `sum_across_hosts(input_tokens)`: Task 0 contributes padded tokens; others
    contribute zeros. The sum propagates the tokens to all processes.
3.  `compiled_push_fn(...)`: All processes collectively push the new sequence.
4.  `batch[index] = (request, future)`: All processes record the batch entry.
    Non-primary tasks store `(None, None)`.

### Why `sum_across_hosts` instead of `broadcast_one_to_all`

`broadcast_one_to_all` assumes global process 0 is the source. With task-based
gating (`is_primary_task()`), the leader process may not be process 0.
`sum_across_hosts` is source-agnostic: since only task 0 contributes non-zero
values, the sum produces the correct result on all processes regardless of which
process index task 0 maps to.

### Key Invariant

> **Every path through the inner loop body reaches the same
> `sum_across_hosts` calls in the same order on all processes.**
>
> The `break`/`continue` divergence only happens *after* the collective op
> (using its result), never *before* it.

This invariant prevents multihost deadlocks.

### Batch State Consistency

All processes maintain the `batch` list. Task 0 stores `(request, future)`
tuples; non-primary tasks store `(None, None)` tuples. Since both are
truthy, `any(batch)` and `all(batch)` evaluate identically on all processes,
ensuring consistent `break`/`continue` decisions.

## Post-Decode Result Handling

After `compiled_decode_fn` runs (collectively on all processes):

1.  **Completion check**: `completed_mask` and `completed_seqs` are computed
    from the `SamplingState` (readable on all processes).
2.  **Cancellation sync**: Task 0 checks futures for cancellation;
    `sum_across_hosts` propagates `is_cancelled` to all processes.
3.  **Release** (collective): All processes call `compiled_release_fn` together.
4.  **Result setting** (task 0 only): Task 0 decodes output tokens, sets
    future results, guarded by `is_primary_task()`.
5.  **Batch cleanup**: All processes set `batch[index] = None` for released
    sequences, maintaining batch state consistency.

## Heartbeat Mechanism

When idle (no pending requests), task 0 uses a long timeout (60 seconds)
instead of blocking forever. This ensures:

*   All processes periodically sync via `sum_across_hosts`, preventing the
    TPU runtime from marking nodes as dead.
*   The `_try_get_request` method uses deadline-based timeouts, so a stream of
    cancelled/invalid requests doesn't indefinitely delay the heartbeat.

## Startup Sequence (`main()`)

All processes participate in:

1.  Configuration setup and mesh initialization.
2.  Checkpoint loading (`update_params_from_checkpoint_path`).
3.  Starting the batcher thread.

Only task 0 additionally starts the gRPC server or runs evaluation logic.

## Graceful Shutdown

The outer loop is `while True`. The **only** stop check is inside the inner
loop, at the `n == 0` path where the batcher idles:

```python
if n == 0:
    if sharding.sum_across_hosts(stop_event.is_set()):
        return
    ...
```

This is necessary because the inner `while not all(batch)` loop is unreachable
from the outer loop when the batch is empty (`continue` stays in the inner loop,
and `all([None, ...])` is always `False`).

### Shutdown Flow

1.  Caller sets `stop_event` on task 0.
2.  **Caller calls `batcher_thread.join()`** — this is critical because the
    batcher thread is a daemon thread. Without `join()`, task 0's main thread
    exits, killing the daemon thread before it syncs the stop signal.
3.  Task 0's batcher returns from `_try_get_request` within
    `max_queue_timeout` (1 second).
4.  `sum_across_hosts(stop_event.is_set())` propagates a non-zero value to all
    processes.
5.  All batcher threads `return` together.
6.  Non-primary tasks' `batcher_thread.join()` returns, processes exit.

## Common Pitfalls

*   **Never add collective ops inside `continue`/`break` paths.** This causes
    some processes to skip the op while others wait, resulting in deadlock.
*   **Keep `batch` in sync across all processes.** Use `(None, None)` on
    non-primary tasks so truthiness checks match.
*   **Guard future access.** Only task 0 has real `Future` objects. Always
    wrap `.cancelled()`, `.set_result()`, etc. in `is_primary_task()` checks.
*   **Always `join()` the batcher thread after `stop_event.set()`.** The batcher
    thread is daemon — without `join()`, the process exits before the stop
    signal is synced, leaving other processes hanging at collective ops.

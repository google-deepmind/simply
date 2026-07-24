

## Building on prior runs (situational)

When the task or operator references a prior run id (a short token such as `k7m2p9qx`, e.g. "build on run k7m2p9qx" or "follow up on run a1b2c3d4"), the inspection tools accept an optional `run_id` that reads that run's data directly. Recommended sequence:

1. `view_run_report(run_id=…)` — the end-of-run report compresses what the run did, what it achieved, and where it struggled.
2. `session_list(run_id=…)`, then `session_summary(run_id=…, session_id=…)` for phase-level narratives (the root session, if any, is `main-agent`).
3. `session_search(run_id=…, query=…)` to find specific topics (matches both raw events and observer summaries).
4. `session_steps(run_id=…, …)` for full event-level drill-down when summaries lack detail.



## Tool usage strategy

- **Prefer specialized tools over bash.** When a purpose-built tool exists (file view/edit, full-text search, session inspection, recall), use it instead of shelling out — the results are cleaner and cheaper.
- **Learn on the fly.** If a task needs an unfamiliar tool, CLI, skill, path, or resource, use `recall_search` to find a relevant skill or lesson *before* proceeding. Never declare you can't do something before checking your skills.
- **Don't busy-poll a long job — background it and wait to be woken.** For work whose timing you don't control (a long build/test/experiment, a file appearing, a metric crossing a threshold), launch a detached background script that captures output and notifies you on completion, then `await_event` to park until it reports back. The script reaches you via `"$AMPLIO_NOTIFY" notify` (delivered to your own session; pass a `<session_id>` first arg to target another, or pipe a long body via `"$AMPLIO_NOTIFY" notify -`). Canonical one-shot recipe (a single detached subshell runs the command, saves stdout/stderr to the run's scratch dir, and notifies with the exit code + paths when it finishes):

  ```bash
  J="$AMPLIO_ARTIFACT_DIR/job-$(date +%s)"
  ( <your long command> >"$J.out" 2>"$J.err"; "$AMPLIO_NOTIFY" notify "job done (exit $?): out=$J.out err=$J.err" ) &
  ```

  Then call `await_event` to wait for it efficiently. Periodic/conditional variant: loop with a `sleep` inside the subshell and notify only when your condition holds (e.g. `while ! grep -q DONE "$LOG"; do sleep 10; done; "$AMPLIO_NOTIFY" notify "ready"`). A notification always reaches you: it wakes you if you're parked in `await_event`, and revives you if you've since gone idle or concluded. When woken, read the captured output with `view_file`.

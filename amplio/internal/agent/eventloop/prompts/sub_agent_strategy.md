

## Sub-agent strategy

Use `spawn_agent` to delegate work. It returns immediately and the sub-agent runs in parallel, delivering its final result as a message when it finishes.

- **Workspace mode.** `share` (default): the sub-agent works in the *same* workspace — best for read-only or information-gathering tasks (avoid concurrent edits). `link`: an isolated linked workspace (git/jj worktree or CitC link) with independent file edits but shared history — best for spawning several parallel coding attempts that must not clobber each other.
- **Share via the artifact dir.** All sub-agents share the run's artifact directory; use it as scratch space — e.g. a parent-written plan read by every child, and per-child result files read back by the parent.
- **Coordinate, don't poll.** Use `send_message` (always naming the target session id) to talk to another session, `await_event` to park until a child finishes or the operator messages you, and `session_peek` only when you suspect a child is stuck. Don't expect results the instant you spawn.
- **Act and wait in one turn.** `await_event` can be called in the *same* step as the action that will produce the event — e.g. `spawn_agent` + `await_event`, `send_message` + `await_event`, or launching a background `bash` job + `await_event`. This blocks the turn until the work reports back, instead of spending a separate turn just to wait.
- **A sub-agent can be re-engaged in any state.** `send_message` reaches it whether it is live (running or parked) or stopped (`idle`/`concluded`/`crashed`/`cancelled`); if it has stopped, your message revives it with its full prior context intact (a status like `concluded` just means it finished its last turn, not that it's gone). Prefer this to re-spawning a fresh agent when you want to ask a follow-up question or hand more work to a child with existing context.

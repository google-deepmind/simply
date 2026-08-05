# Session Lifecycle

How an agent session is born, works, parks, ends, and is revived. Builds on the
step/turn model in [`step_model.md`](step_model.md).

A **session** is one agent's event stream plus a **status**. A **run** is a tree
of sessions (a root agent and the sub-agents it spawned). Sessions run as
in-process goroutines; a per-run registry maps `session_id` → a live handle
(notify + cancel). A session with no live goroutine is **cold** — it lives only
as DB rows until something revives it.

## Status (logical)

Status is the session's *logical* state. It does **not** encode whether a
goroutine is currently alive — that's a physical optimization (see
[Hot vs cold](#hot-vs-cold-physical-only)). Every non-`ongoing` status is
restartable; *how* is defined by the [matrix](#what-advances-a-session).

| Status | Meaning |
| --- | --- |
| `ongoing` | actively working in the loop |
| `awaiting` | parked via `await_event` — actively anticipating new events |
| `idle` | parked after a bare no-tool-call turn (interactive agents only) |
| `concluded` | finished — submitted its result (autonomous agents only) |
| `crashed` | stopped by an uncaught error, durably recorded |
| `cancelled` | explicitly stopped by an operator or parent |
Normal "parked" states:

- An **interactive agent** (chatbot) parks as **`idle`** (turn done, waiting for
  the user) or **`awaiting`** (a background task is running, standing by). A
  conversation is never `concluded`.
- An **autonomous agent** ends as **`concluded`** (its final message *is* its
  result) or parks as **`awaiting`** (waiting on sub-agents). It never enters
  `idle`.

A no-tool-call turn means "I have nothing more to do right now," and what it
produces depends on the agent's nature (see
[Termination](#termination-the-no-tool-call-turn)): interactive agent becomes `idle`; autonomous agent pass on the conclusion and becomes `concluded`.

### Two notions of "crashed"

- **`crashed` status** — durably recorded, which (via the atomic
  [child-result](#child-results-exactly-once) write) means the parent was already
  notified. A crashed *sub-agent* is never revived spontaneously — only by an
  `Input` follow-up (e.g. a parent retry). A crashed *root* is revived by
  [Recover](#recover) on restart.
- **`ongoing`-but-dead** — status is still `ongoing` but no goroutine exists (a
  hard process crash before any transition was recorded). [Recover](#recover)
  revives these; since nothing was recorded or notified, re-attempting is correct.

## Hot vs cold (physical only)

After a no-tool-call **park** (`idle`), or while `awaiting`, the goroutine lingers
for an idle timeout (default 30 min) as an optimization. (A `concluded` autonomous
agent does **not** linger — its goroutine exits immediately; a later `Input`
revives it cold.)

- **Hot** — goroutine still alive/registered; revival is an instant in-memory
  wake with context intact.
- **Cold** — goroutine exited; revival respawns from DB state.

**Hot vs cold changes only the *mechanism* (wake vs respawn), never the
*outcome*.** The same event produces the same logical result whether the session
is hot or cold; the 30-minute window is purely a latency optimization. (The one
operation genuinely *about* physical liveness is [Recover](#recover), which
respawns logically-active-but-physically-dead sessions.)

## What advances a session

A session is moved by one of **two stream-event classes** (events materialized in
its stream) or the **Cancel** action. The run-level [Recover](#recover) operation
is separate — it merely *produces* events of the `Input` class.

`Classify(evt)` sorts every stream event into one of two classes by whether it
should revive a *dormant* session:

| Class | Revives a dormant session? | Examples |
|---|---|---|
| **Notice** | no — persists; only a live `awaiting` session reacts | `ChildResultEvent` verdict `crashed`/`cancelled`; self/system writes (Assistant, ToolResult, Compaction, `SystemEvent` markers) |
| **Input** | yes — restarts any non-`ongoing` session (but see the environment-notification exception below) | `UserEvent` (operator/user input); `MessageEvent` — agent (`send_message`) **or** environment (`$AMPLIO_NOTIFY`); `ChildResultEvent` verdict `concluded`; `RecoverEvent` |

**Every event is written to the session's stream regardless of the target's
state** — the class only decides whether it *revives* a dormant session. A
`Notice` to a terminal session just persists (seen only if the session is later
restarted).

**Environment notifications (`$AMPLIO_NOTIFY`) are `Input`-class, but do NOT
revive a *terminal* session.** They revive a *parked* recipient (`idle`/
`awaiting`), so a background job's completion notice reaches an agent that's
merely waiting even if it forgot to `await_event`. They do **not** revive a
`concluded`/`crashed`/`cancelled` session: the environment must not resurrect an
agent that has deliberately finished (or was deliberately stopped), and reviving
a terminal just to trivially ack a stale/runaway notifier churns phases, run
reports, and — via a re-posted `child_result` — the parent chatbot. A terminal
session is still restartable by a `UserEvent`, an agent `send_message` /
`child_result`, or `Recover` — only a raw env notification is gated. This
exception is enforced at the wake path (`runtime.NewCommitNotifier`, against a
wake-path-private `envUnrevivableStatuses` set — deliberately NOT a general
"terminal" predicate, since that notion is overloaded and this policy needs to be
tunable on its own, e.g. `crashed`), not in `Classify` (which stays a pure
function of the event):
the env message is still appended (persisted) to the terminal session's stream,
it just doesn't respawn it, so it's seen on the next genuine restart. A
runaway/forgotten notifier remains recoverable: `notify` stamps the caller's pid
onto the sender (e.g. `environment (pid=12345)`), so a revived agent can identify
and `kill` the offending script.

**Env notifications are capped per step** (`server.maxEnvNoticesPerStep`, 50).
Since they persist even when they cannot wake anyone, an abandoned poller would
otherwise append for as long as it runs — and every notice it appends is also
replayed into the model's context on the next turn. The *step* is the budget
window, which needs no session state to be the right one: a working session
advances its step every turn, so 50 is a generous per-turn allowance; a finished
session never advances again, so the same rule becomes a lifetime cap. Over the
cap, `POST /notify` answers **429** instead of appending, with the fixed token
`env_notice_capped` as the body (a script, not a model, is reading it — so no
prose and no live count, which would defeat an exact match). Since `amplio
notify` maps non-2xx to exit code 3 with the body on stderr, a stray loop can
detect this precisely and stop itself.

Relatedly, since a terminal session can no longer be env-rescued, an autonomous
agent that ends a turn with **no tool calls AND empty content** (a likely
*accidental* conclusion — the model reasoned but forgot to act) is nudged back
into the loop rather than concluding on nothing, bounded by a small in-memory
counter so a degenerate model can't loop forever (see
`eventloop.maxConcludeNudges`).

**Operator/user input is a `UserEvent`, not a `MessageEvent`.** `MessageEvent` is
agent-to-agent (`send_message`) or environment (`$AMPLIO_NOTIFY`); rendering user
input as a `MessageEvent` would tempt a chatbot to "reply" to the user via
`send_message`. The web/CLI persists typed input directly as a `UserEvent` (a plain
user turn).

The matrix — "restart" = wake (hot) or respawn (cold); same outcome:

| State | Input | Notice | Cancel |
|---|---|---|---|
| `ongoing`† | queue; the loop reads it | queue; the loop reads it | stop + mark cancelled + cascade |
| `awaiting` | wake → process | **wake → process** | stop + mark cancelled + cascade |
| `idle` | restart → process | persist | stop + mark cancelled + cascade |
| `concluded` | restart → process | persist | NOP (terminal) |
| `crashed` | restart → process | persist | NOP (terminal) |
| `cancelled` | restart → process | persist | NOP (terminal) |

† `ongoing` means *alive and working*. An `ongoing`-but-dead session is a
crashed-process case revived by [Recover](#recover) (or the next `Input`).

Reading the matrix:

- **Every non-`ongoing` state restarts on an `Input`** — including `cancelled`.
  You bring a session back by **sending it a message** (see
  [Restart vs Recover](#restart-vs-recover)).
- **`awaiting` is the only state where a `Notice` wakes it.** `await_event` opted
  into "any event," so an awaiting parent *does* wake on a child crash (it wants
  to know). Every other parked state ignores `Notice` — if you're anticipating an
  event, `await` it.
- **A `Notice` never revives a dormant session.** This is what stops a crashing
  child from hammering a dormant parent back to life: the report persists and is
  seen on the next genuine restart.
- **`concluded`, `crashed`, `cancelled` are identical here** — all terminal, all
  restartable by `Input`. They differ only in their **verdict** label and in
  whether [Recover](#recover) touches them (only a `crashed` root). `idle` is the
  one non-terminal parked state, hence the only one Cancel acts on.

## Restart vs Recover

Two different ways a session comes back — don't conflate them:

- **Restart** (session-level, reactive): a non-`ongoing` session revives because
  an **`Input`** arrived for it — typically a follow-up **message**. There is no
  "resume this session" button; you restart a session by messaging it. Any
  non-`ongoing` session (`idle`/`concluded`/`crashed`/`cancelled`) can be
  restarted this way.
- **Recover** (run/system-level, proactive): on a process restart, or when an
  operator resumes a **run**, the system re-spawns the run's *active spine*. It is
  never aimed at a single dormant session. See [Recover](#recover).

## Cancel

Cancellation is **canceller-driven, not cooperative**: whoever cancels — the
`session_cancel` tool, an operator, or a concluding parent cascading to its
children — does all the work; the target agent just stops.

A single `cancelSession(sid, reason)` primitive, applied recursively:

1. if already terminal (`concluded`/`crashed`/`cancelled`) → **NOP**;
2. in **one atomic write** (`TerminateAndNotifyParent`, carrying `reason`): set
   status `cancelled`, write a `cancelled` `SystemEvent` marker to the session's own
   stream, and notify the parent. The terminal write fires the session-handler for
   the **parent only** (the `child_result`), never for the target itself — so it
   never self-wakes the session being stopped;
3. recurse into each active child (with a reason such as `"parent <id> concluded"`);
4. if a goroutine is live, **ctx-cancel** it so it stops.

The agent loop contains **no** cancellation logic — it treats a cancelled `ctx`
(parked in `await`, mid-LLM-call, or between iterations) as "stop now, status
already set, return." A parked goroutine's wait is simply interrupted; nothing is
ever "woken so it can cancel itself."

**Reasons make cancellation observable and recoverable.** The `reason` flows into
the parent's `child_result(cancelled, …)` and onto the cancelled session's own
stream. So when a parent concludes while sub-agents are still running, they are
cancelled with `"parent concluded"`, and the parent's stream records exactly what
was cancelled and why. If that was a *mistake*, an operator restarts the parent
(a follow-up message); the parent sees the cancelled-children records and can
**re-spawn** them.

Tradeoffs (accepted): ctx-cancel abandons in-flight work immediately — fine for
cancel, since the work is being discarded (tools must keep writes atomic /
ctx-safe). A cancel landing exactly as a session concludes is a benign race (both
mean "ended"); cancelling mid-`spawn` can briefly miss a just-created grandchild
([Recover](#recover) catches the orphan). And the recovery above **re-spawns**
children (fresh) — a cancelled child's progress is lost. Progress-preserving
revival would need a future agent-callable `resume_child` (a parent sending a
`RecoverEvent` to its own child); not built.

## Recover

`Recover` is the narrow, run/system-level crash-recovery mechanism — it has
nothing to do with restarting an individual dormant session (that's an `Input`).
It runs when the process restarts, or when an operator resumes a **run** (the
`amplio headless resume <run-id>` command), and it re-spawns only the run's **active
spine**:

- `ongoing`-but-dead sessions (were working),
- `awaiting`-but-dead sessions (were waiting; re-arm them), and
- a `crashed` **root** (no parent to retry it).

It does **not** touch `idle`/`concluded`/`cancelled`/`crashed`-sub-agent — those
are settled or dormant and come back only via a reactive `Input`.

Mechanically, Recover appends a **`RecoverEvent`** to each spine session.
`RecoverEvent` is an `Input`-class event, so it flows through the normal respawn
machinery (re-enter the loop with status `ongoing`) and doubles as the visible
"resumed" marker the LLM sees. It is **only** produced by Recover — never by
normal agent flow. (Subtlety from crash recovery: Recover **skips** the
`RecoverEvent` when the stream is already "at rest" — crashed after the final
reply but before the status flip — to avoid a redundant LLM round-trip; see
[Crash recovery](#crash-recovery).)

## Termination: the no-tool-call turn

An agent ends its active work by producing a turn with **no tool calls**:

- **Autonomous agent → `concluded`.** The turn's text *is* the result: it is
  posted to the parent (the return value) and the agent cascade-cancels its owned
  children (with a `"parent concluded"` [reason](#cancel)).
- **Interactive agent (chatbot) → `idle`.** It rests until the next user message, or any background sub-agent's result.

Waiting is itself a tool (`await_event` → `awaiting`), so a bare no-tool-call turn
is unambiguous: the agent is neither acting nor waiting, so it's done. 
Model it as a function: `spawn` = async call, `await_event` = await, a no-tool
turn (autonomous) = `return <text>`, `child_result` = the returned value.

## Cascade on exit

| Exit | Owned children | Posts a result to its own parent |
|---|---|---|
| conclude (autonomous no-tool turn) | cascade-cancel (`cancelSession`, reason `"parent concluded"`) | yes — the return value |
| cancel | cascade-cancel (`cancelSession`) | yes — `cancelled` |
| crash | **kept alive** | yes — `crashed` (so an awaiting parent unblocks) |
| idle (interactive no-tool turn) | untouched | no |

**Crash keeps children alive.** A soft-crashed parent's children keep running and
post their results/crashes durably into the parent's stream; the parent consumes
them when it restarts. (A hard process crash leaves children `ongoing`-but-dead,
which Recover revives.) This stays coherent because the event stream — not
goroutine liveness — is the source of truth.

## Child results (exactly-once)

A child can stop and restart many times, so a parent's stream may hold many
`child_result`s for the same child. To keep delivery exactly-once **per
transition**:

- A terminal transition (`conclude`/`crashed`/`cancelled`) writes the
  `ChildResultEvent` into the parent's stream **in the same DB transaction** as
  the child's own status change (`TerminateAndNotifyParent`; a root agent skips
  the notify). The single SQLite DB makes this atomic, and the child posts from
  its own terminate path — no separate cross-goroutine hand-off.
- The spawner's runner is only a **fallback**: if the child goroutine exits with
  status still `ongoing` (a panic that bypassed `terminate`), the runner performs
  the same atomic crash-and-notify. On the normal path it posts nothing.
- **Status is the dedupe token.** A child already in a terminal status was, by
  that atomicity, already notified; Recover/reconcile notifies only
  `ongoing`-but-dead children. No "has the parent seen this?" query is needed.
- `ChildResultEvent.Verdict` carries the outcome so the parent can branch
  (concluded → use result; crashed → maybe retry; cancelled → don't).

A child that crashes on its own (parent alive) still delivers a
`child_result(crashed)` so an awaiting parent unblocks rather than deadlocking.

## `await_event`

`await_event` parks the session until a *new* event arrives or a timeout fires.
It is **counter-based and level-triggered**, not a bare edge wait, so it is
immune to missed and spurious wakeups:

1. Snapshot the session's event counter **first** (`before`).
2. **Short-circuit**: if a peer event already landed at the current step during
   the just-finished LLM call, return immediately. In amplio's step model the
   `AssistantEvent` sits at the call step and peer events at `current_step`, so
   the check is `event_count(step >= current_step) >= 1`. (Snapshotting *before*
   this check closes the gap race.)
3. Otherwise park as `awaiting` and wait until the counter advances past
   `before`, or the timeout expires (mandatory, default 300 s). Re-check the
   counter on each wake so spurious/self wakes don't return early.
4. On a normal **wake**, restore `ongoing` — but re-read the status first and skip
   the restore if it is terminal: a cancel may have interrupted the wait (via ctx)
   *or* flipped the status via a racing notify. Never clobber a terminal status
   (the canceller already set `cancelled`) with `ongoing`.

`await_event` is **exclusive** (the only tool call in its turn) so no same-turn
self-write pollutes the counter. It wakes on *any* event (the `awaiting` row of
the matrix); the `idle` park, by contrast, only advances on `Input` events.

## Notification fabric

Waking a live session is **synchronous and lossless**: a commit-time hook runs
inside the event append (after the row commits, outside the store lock) and, for
the target session:

- if **live** → bumps its waiter counter. The parked logic then decides whether to
  actually advance: `await` advances on any event; the `idle` park advances only
  on an `Input` event (re-sleeping otherwise).
- if **cold** → respawns iff the event is `Input` (`Classify`).

A dropped notification would mean a hang-until-timeout, so this path must never
drop. The waiter primitive is a monotonic counter guarded by a mutex plus a
broadcast channel that is closed-and-replaced on each notify; `WaitAfter`
snapshots the channel under the lock and selects on `{channel, timer, ctx}`.

## Spawn

When an agent spawns a child:

1. The child's `session_id` is allocated synchronously (a unique name reserved
   in memory — see below), so the parent gets the id back immediately.
2. The child's goroutine then creates its own DB session row and registry entry
   in `Run`/`bootstrap` (not pre-created in `spawn`). This leaves a tiny
   in-process window — between `spawn` returning the id and the child goroutine
   reaching `bootstrap`'s `CreateSession` — during which the session row does not
   yet exist, so an event targeting the child (e.g. a `send_message`) fails the
   append rather than landing. The window is a microsecond-scale goroutine
   scheduling gap that the normal flow never hits: the parent only learns the id
   from the tool result and acts on it a full LLM turn later, by which point the
   child has long since bootstrapped. The failed-append-in-the-gap case is an
   **accepted trade-off by design** (it fails loudly rather than mis-filing the
   event); closing it fully would mean splitting bootstrap across two goroutines
   for no practical benefit.
3. The child's goroutine runs with a context **independent of the parent's
   tool/loop context**, so the parent finishing, idle-exiting, or crashing does
   not cancel the child. Cancellation reaches the child only via `cancelSession`
   (which ctx-cancels through the registry handle).

The child's `session_id` is a unique nickname from the run's in-process **name
allocator**: a mutex-guarded used-set, seeded once from the DB so cold
(concluded/crashed/idle) names are never reused. Reserving the picked name in
memory at allocation time removes the pick→create race without a retry loop.

## Crash Recovery

The per-session **loop re-entry** mechanics — run whenever a cold session respawns,
whether via an `Input` restart or a [Recover](#recover) — repair crash artifacts and
are derived purely from DB state (see the table in
[`step_model.md`](step_model.md#crash-resume)). They are no-ops after a clean park:

- **Don't double-advance.** On the first iteration of a resumed run, advance the
  step unless resuming mid-LLM-call (`current_step > 0` and no `AssistantEvent`
  at it) — then call the LLM at the existing step so the `AssistantEvent` lands
  where the dead predecessor intended.
- **Repair orphan tool calls.** Synthesize a `ToolResultEvent` for every
  `tool_call` id with no matching result (crash mid-tool-execution), or the next
  LLM call is rejected for unmatched `tool_use`. Idempotent.
- **One-shot rest detection.** If the stream tail is already a no-tool
  `AssistantEvent` at resume (crashed after the final reply but before the status
  flip), finalize without a redundant LLM call — and [Recover](#recover) skips the
  `RecoverEvent` in this case.
- **The "resumed" marker is the `RecoverEvent`** ([Recover](#recover) only). It is
  a user-role event (not a system message) because providers require the last
  pre-assistant message to be a user/tool-result message.

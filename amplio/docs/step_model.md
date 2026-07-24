# Step Model

This document describes the step boundary invariant for the event loop agent.

## Core Invariant

**The AssistantEvent is the LAST event of a step** (together with the tool
results event).

The step counter (`Session.current_step`) is bumped BEFORE LLM generation
starts. The AssistantEvent (and its tool results) are written at the
previous step (`current_step - 1` at call time, the "call step"). Any
events that arrive during generation are stamped with `current_step` (the
bumped value), placing them at the NEXT step.

This eliminates the need for within-step event re-arrangement.

## Step Layout

```
Step 0: [bootstrap events]           ← permanent (survives compaction)
Step 1: [input events, AssistantEvent, ToolResults]  ← compactible
Step 2: [input events, AssistantEvent, ToolResults]  ← compactible
...
```

Each step T contains:
- **Input events** (0 or more): messages, child results, or other events
  that arrived before or triggered the LLM call.
- **AssistantEvent** (exactly 1): the LLM's response, written at the call
  step after generation completes.
- **ToolResultEvents** (0 or more): results of tool calls from the
  AssistantEvent, pinned to the same step. Each result is written the instant
  its tool finishes, so within a step they are stored in tool **completion**
  order, not call order; each carries its `tool_call_id`, and context
  reconstruction associates results to calls by that id, so the stored order
  doesn't matter. This allows a blocking `await_event` to be called within 
  the same step (e.g. `send_message` + `await_event`), but without having
  `await_event` blocking the completion of other normal toolcall results.

Step 0 is special: it contains only bootstrap events (system prompt,
optional task, session marker) and never has an AssistantEvent.

The `AssistantEvent` represent's the agent's response to everything happened
within and before that step. Notably, in an autonomous agentic loop, the agent
mostly respond to tool call results, which are logically grouped to the
previous step.

## Lifecycle Trace

### Autonomous Agent (with task)

```
Bootstrap: write events at step 0
  bump current_step → 1

Iteration 1:
  LLM starts (context: step 0), bump current_step → 2
  LLM finishes → AssistantEvent(tools) at step 1
  Tool results pinned at step 1

Iteration 2:
  LLM starts (context: step 0+1), bump current_step → 3
  LLM finishes → AssistantEvent(no tools) at step 2
  Park at current_step = 3
```

### Chatbot (no task)

```
Bootstrap: write events at step 0
  bump current_step → 1
  Park (no task)

User message arrives → step 1
  Wake
  LLM starts (context: step 0+1), bump current_step → 2
  LLM finishes → AssistantEvent at step 1
  Park at current_step = 2
```

The first user message is at step 1 (compactible), NOT step 0 (permanent).
No arbitrary first question gets elevated to bootstrap status.

### During-Generation Message (Race Case)

```
LLM generating (call step = T, current_step = T+1)
  Message arrives → stamped at T+1 (current_step)
  LLM finishes → AssistantEvent at T

Step T: [AssistantEvent]     ← clean, no interleaved message
Step T+1: [MessageEvent]    ← ready for next LLM call
```

No within-step re-arrangement needed. The step bump before generation
creates the boundary automatically.

## Crash Resume

The step model makes crash recovery deterministic from DB state alone:

| Crash point | DB state | Detection | Resume action |
|---|---|---|---|
| During LLM generation | current_step=T+1, step T has no AssistantEvent | `step T has no AssistantEvent` | Re-do LLM call for step T |
| During tool execution | step T has AssistantEvent, missing ToolResults | Orphan tool_call IDs with no matching ToolResult | Insert placeholder ToolResults, continue |
| After tools, before next call | step T complete | Step T has AssistantEvent + all ToolResults | Normal next iteration |
| During park | current_step=T, step T-1 complete | Session status = completed | Re-enter park or check for pending messages |

## Implementation Notes

- `AppendEvent` writes at `current_step` (for input events).
- Each loop iteration bumps the step FIRST, then loads context bounded to
  `step <= callStep` (`callStep = newStep - 1`). Bumping first freezes the
  call-step window — every new event then lands at the bumped step — so no event
  can slip into the call step after context was gathered.
- AssistantEvent and ToolResults are written at the call step (an explicit step
  parameter); events arriving after the bump land at the bumped step.
- The within-step order is: input events (chronological) → AssistantEvent
  → ToolResults. Tool results are appended per-tool as each finishes (so a
  slow/blocking tool like `await_event` doesn't hide the others' completion),
  hence they are stored in completion order. This is fine because results are
  matched to their tool calls by `tool_call_id` when context is rebuilt for the
  next LLM call — order is immaterial to the provider.
- Compaction keeps step 0 events across all generations. Steps 1+ are
  filtered by `generation == current_generation`.

// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package eventloop

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"amplio/internal/agent"
	"amplio/internal/blob"
	"amplio/internal/cli"
	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/imageutil"
	"amplio/internal/llm"
	"amplio/internal/session"
	"amplio/internal/tool"
	"amplio/internal/util"
	"amplio/internal/workspace"
)

const DefaultIdleTimeout = 30 * time.Minute

// maxConcludeNudges bounds how many times an autonomous agent is nudged back
// into the loop after an empty no-tool turn (a likely-accidental conclusion)
// before the empty conclusion is allowed through. Small so a degenerate model
// wastes at most this many extra calls; see concludeNudgeText.
const maxConcludeNudges = 2

// emptyAssistantPlaceholder is projected into the LLM request in place of the
// (empty) content of a degenerate assistant turn — no text, no tool calls — so
// the message is non-empty. Some providers reject an empty assistant message.
// Instead of dropping it, we replace it with a placeholder so that the
// conversation flows more naturally (a nudge may follow this empty message).
const emptyAssistantPlaceholder = "(empty response)"

const concludeNudgeText = "[system] Your previous turn produced no tool calls and no message text, " +
	"so it does not look like a deliberate completion. If the task is genuinely finished, " +
	"reply with a concise final summary (no tool calls) — that summary IS your result. " +
	"Otherwise, continue working with the appropriate tool calls."

// compactionFraming prefixes a CompactionEvent when it is projected into the
// LLM context, telling the model to treat the summary as its own memory of
// already-completed work and continue. Applied here (projection time), not
// stored on the event, so it never leaks into observer/inspect/search renders.
const compactionFraming = "This is a faithful summary of work you have ALREADY COMPLETED. " +
	"Read it as your memory of what happened, then proceed with whatever work remains.\n\n"

// Config holds the per-agent-instance configuration for an EventLoopAgent.
//
// The shared, per-run object graph (Store, LLM, Registry, Workspace,
// Broadcaster, system-tier providers, run id, AGENTS.md) lives on the *agent.Env
// passed alongside this Config to New — it is built once from the raw RunConfig
// at run start and shared by pointer across every agent in the run. Config
// carries only what differs per agent instance, grouped below into three levels.
type Config struct {
	// ── Init params: identify this specific invocation ──
	ParentID  string
	SessionID string
	// Task is scoped for the entire session, lands in the bootstrap events
	// (step 0), and survive compaction, thus always visible in the context
	// of all generations. Used for autonomous agent run.
	Task string
	// FirstMessage, if set, is written as a UserEvent at step 1 right after
	// bootstrap. One canonical use case is the initial message of an
	// interactive run. Unlike `Task`, because this lands in step 1, it
	// does *not* carry over to future generations (unless the compaction
	// summary explicitly resurface it).
	FirstMessage string
	// Handle, if non-nil, is the registry slot the launcher already claimed for
	// this session via Registry.RegisterAndContext (synchronously, before spawning
	// this goroutine, so the run-registry instance can't be orphaned by a
	// concurrent RemoveIfEmpty). Run() uses it directly and the launcher's release
	// owns Unregister. Nil for a direct caller / test, where Run() registers its
	// own handle and owns the lifecycle.
	Handle *session.Handle

	// ── Type profile: constant for an agent type; set by the factory and
	//    treated as read-only per type (Go can't attach constants to a type, so
	//    these live here rather than on a class — do not vary them per instance). ──
	//
	// AgentType is the registry name this agent was created under (e.g.
	// "standard_agent"). Stored on the session so RespawnSession can rebuild it.
	AgentType string
	// Interactive marks a chatbot-style agent: a bare no-tool-call turn parks as
	// idle (waiting for the next user message) instead of concluding. Autonomous
	// agents (the default) conclude on a no-tool turn.
	Interactive bool
	// IdleTimeout is how long to wait for follow-up after completing (0 = default).
	IdleTimeout time.Duration
	// CLITools are optional external command-line tools this agent may run via
	// bash. At bootstrap, the currently-available subset is described to the
	// agent; absent ones are omitted (worker agents opt in; ephemeral helper
	// loops leave it nil).
	CLITools []cli.Tool

	// ── Instance-computed: assembled by the factory for this agent ──
	//
	// SystemPrompt is the role-appropriate prompt the factory assembled from
	// shared snippets (and, for the chatbot, a per-instance root/sidecar choice).
	SystemPrompt string
	// Tools is the toolset the factory wired for this agent (closures may capture
	// the agent's own session id / handle).
	Tools []*tool.Tool
	// InitialRecall, if set, returns a "skills relevant to this task" block seeded
	// as a step-0 SystemEvent for non-empty tasks (skill recall). nil = none.
	InitialRecall func(ctx context.Context, task string) string
	// BlobStore persists tool-result attachment bytes (images) on disk so they
	// stay out of the event log, and reloads them when rebuilding the LLM
	// context. nil disables image attachments (text results still flow).
	BlobStore *blob.Store
}

// EventLoopAgent is the canonical LLM-in-a-loop agent implementation.
type EventLoopAgent struct {
	env    *agent.Env // shared per-run object graph (services, providers, run id)
	cfg    Config     // per-agent-instance configuration
	handle *session.Handle
	logger *slog.Logger
	// callStep is the call step of the turn currently in flight, so a crash is
	// recorded at the turn's step (where its events would have gone), not the
	// bumped current step. -1 outside a turn (pre-loop / between turns) → the
	// failure marker falls back to the session's current step.
	callStep int
	// concludeNudges counts consecutive accidental-conclusion nudges (an empty
	// no-tool turn from an autonomous agent). In-memory and bounded by
	// maxConcludeNudges so a degenerate model can't loop forever; reset to 0 on
	// any substantive turn. Not persisted — it only bounds attempts within one
	// live loop (a cold respawn is a fresh, legitimate retry).
	concludeNudges int
}

// New builds an agent from the shared run env and this agent's instance config.
// env carries the per-run object graph (constructed from the raw RunConfig at
// run start) and is shared by pointer across all agents in the run; cfg carries
// the per-instance data (identity, type profile, factory-assembled tools/prompt).
func New(env *agent.Env, cfg Config) *EventLoopAgent {
	return &EventLoopAgent{
		env:      env,
		cfg:      cfg,
		logger:   slog.With("session", cfg.SessionID, "run", env.RunID),
		callStep: -1,
	}
}

func (a *EventLoopAgent) SessionID() string { return a.cfg.SessionID }

// Handle returns the session handle. Only valid after Run() creates it.
// Used by coordination tools that need the wake channel.
// For tools constructed before Run(), use SetHandle to inject later.
func (a *EventLoopAgent) Handle() *session.Handle { return a.handle }

// Run executes the agent's main loop.
//
// Registration model. A launcher (RunManager / the spawn tool) normally claims
// the registry slot SYNCHRONOUSLY via Registry.RegisterAndContext before it
// spawns this goroutine, and hands the resulting handle in via Config.Handle
// (the launcher's deferred release owns Unregister). This is what keeps the
// run-registry instance from being orphaned by a concurrent RemoveIfEmpty (see
// RegisterAndContext). When Config.Handle is nil — a direct caller / test that
// invokes Run() itself — Run falls back to registering its own handle (deriving
// a cancelable ctx) and owns the Unregister. Either way `ctx` is the cancelable
// context whose cancel lives in a.handle, so Interrupt/CancelAll stop this loop.
func (a *EventLoopAgent) Run(ctx context.Context) error {
	if a.cfg.Handle != nil {
		// Pre-registered by the launcher: ctx is already the cancelable context
		// whose cancel is stored in this handle; release (Unregister + cancel) is
		// the launcher's responsibility.
		a.handle = a.cfg.Handle
	} else {
		// Direct caller (no launcher): register our own slot and own its lifecycle.
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		defer cancel()
		a.handle = session.NewHandle(cancel)
		if err := a.env.Registry.Register(a.cfg.SessionID, a.handle); err != nil {
			return fmt.Errorf("register session: %w", err)
		}
		defer a.env.Registry.Unregister(a.cfg.SessionID)
	}

	sess, err := a.env.Store.GetSession(ctx, a.env.RunID, a.cfg.SessionID)
	fresh := err != nil
	if fresh {
		// Fresh session — create it and seed bootstrap events.
		if err := a.bootstrap(ctx); err != nil {
			return fmt.Errorf("bootstrap: %w", err)
		}
		// Re-read session to check if bootstrap parked it (no task).
		sess, err = a.env.Store.GetSession(ctx, a.env.RunID, a.cfg.SessionID)
		if err != nil {
			return fmt.Errorf("read session after bootstrap: %w", err)
		}
	}

	// Crash recovery: an existing (non-fresh) session may carry artifacts from a
	// prior life. Repair orphan tool calls, finalize an at-rest stream, and
	// decide whether the first loop iteration advances the step. This runs on
	// EVERY cold respawn / restart (a follow-up Input or a Recover), not only on
	// process restart, so the repair is never skipped.
	needsAdvance := true
	if !fresh {
		handled, na, rerr := a.reconcileResume(ctx, sess)
		if rerr != nil {
			return a.recordFailure(ctx, rerr)
		}
		if handled {
			return nil // at-rest autonomous conclude (or other terminal finalize)
		}
		needsAdvance = na
		// reconcileResume may have parked an at-rest interactive session as idle.
		sess, err = a.env.Store.GetSession(ctx, a.env.RunID, a.cfg.SessionID)
		if err != nil {
			return a.recordFailure(ctx, fmt.Errorf("re-read session: %w", err))
		}
	}

	// A non-ongoing session is being (re)started: freshly bootstrapped with no
	// task, or revived by an Input (the notifier only respawns cold sessions on
	// an Input; Recover only re-spawns the active spine). Every non-ongoing
	// status is restartable. Short-circuit if the reviving Input is already
	// pending (the cold-respawn case); otherwise wait for one.
	if sess.Status != db.SessionOngoing {
		if !a.hasPendingInput(ctx) {
			woken, err := a.waitForFollowUp(ctx)
			if err != nil {
				a.logger.Debug("context cancelled while parked, exiting")
				return nil
			}
			if !woken {
				a.logger.Debug("idle timeout, exiting goroutine")
				return nil
			}
		}
		a.logger.Debug("resuming session", "from_status", sess.Status)
		if err := a.env.Store.UpdateSessionStatus(ctx, a.env.RunID, a.cfg.SessionID, db.SessionOngoing); err != nil {
			return a.recordFailure(ctx, fmt.Errorf("resume session: %w", err))
		}
	}

	return a.loop(ctx, needsAdvance)
}

// reconcileResume performs one-shot crash recovery when entering an existing
// (non-fresh) session, derived purely from DB state (see docs/step_model.md).
// It returns handled=true if it finalized the session (the caller should
// return). Otherwise it returns needsAdvance for the first loop iteration.
func (a *EventLoopAgent) reconcileResume(ctx context.Context, sess *db.SessionRecord) (handled, needsAdvance bool, err error) {
	events, err := a.env.Store.GetEvents(ctx, a.env.RunID, a.cfg.SessionID,
		db.EventFilter{CurrentContextOnly: true})
	if err != nil {
		return false, false, fmt.Errorf("load events: %w", err)
	}

	// Repair orphan tool calls (idempotent): synthesize a ToolResultEvent for
	// every tool_call with no matching result, or the next LLM call is rejected
	// for unmatched tool_use. Written at the declaring AssistantEvent's step so
	// results stay grouped with it.
	if err := a.repairOrphanToolCalls(ctx, events); err != nil {
		return false, false, fmt.Errorf("repair orphans: %w", err)
	}

	// Finalize any complete-but-unfinalized step. After orphan repair every
	// declared tool call has a result, so each step bearing an AssistantEvent is
	// complete. Because results are now appended per-tool (not in one finalizing
	// write), a crash can leave a fully-resulted step unfinalized — results
	// durable, MarkStepFinalized never reached. Finalizing the latest such step
	// here makes recovery deterministic rather than relying on a later turn to
	// advance the summarizer cursor past it. max() keeps it idempotent.
	maxAssistantStep := 0
	for _, rec := range events {
		if _, ok := rec.Event.(*event.AssistantEvent); ok && rec.Step > maxAssistantStep {
			maxAssistantStep = rec.Step
		}
	}
	if maxAssistantStep > sess.LastFinalizedStep {
		if err := a.env.Store.MarkStepFinalized(ctx, a.env.RunID, a.cfg.SessionID, maxAssistantStep); err != nil {
			return false, false, fmt.Errorf("finalize recovered step: %w", err)
		}
	}

	// One-shot rest detection: a no-tool AssistantEvent tail means the prior life
	// produced its final reply but crashed before flipping status. Finalize
	// without a redundant LLM call (which would also be rejected for a trailing
	// assistant message). (Orphans, if any, made the tail a tool result instead,
	// so this and the repair above are mutually exclusive.)
	if tail := tailAssistantAtRest(events); tail != nil {
		if a.cfg.Interactive {
			// Chatbot finished its reply but crashed before parking. Go idle and
			// wait for the next message (handled by the caller's wait); do NOT
			// re-call the LLM.
			a.logger.Debug("resume: stream at rest, parking idle")
			if err := a.env.Store.UpdateSessionStatus(ctx, a.env.RunID, a.cfg.SessionID, db.SessionIdle); err != nil {
				return false, false, fmt.Errorf("idle session: %w", err)
			}
			return false, true, nil
		}
		// Accidental-conclusion guard, on the RECOVERY path. Append the nudge
		// instead and re-enter the loop: that also makes the tail a user event,
		// which is what the provider requires before the next assistant turn
		// (the reason this branch avoids re-calling the LLM in the first place).
		if isDegenerateTurn(tail.Content) && a.concludeNudges < maxConcludeNudges {
			a.concludeNudges++
			a.logger.Debug("resume: at-rest turn is empty; nudging instead of concluding", "nudge", a.concludeNudges)
			if _, aErr := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
				&event.UserEvent{Content: concludeNudgeText}); aErr != nil {
				return false, false, fmt.Errorf("append resume conclude nudge: %w", aErr)
			}
			return false, true, nil
		}
		// Autonomous: the no-tool reply is the result. Conclude without re-asking.
		a.logger.Debug("resume: stream at rest, concluding")
		if cErr := a.conclude(tail.Content); cErr != nil {
			// Don't claim handled-success on a failed conclude: that would exit the
			// goroutine with the session still ongoing and the parent never
			// notified (an awaiting parent would hang). Return the error (raw, like
			// the other returns here) so the caller routes it through recordFailure,
			// which records the crash + notifies the parent — or, if conclude failed
			// because the DB itself is broken (errors.Is(db.ErrStore)), leaves it
			// ongoing-but-dead for Recover.
			return false, false, fmt.Errorf("resume at-rest conclude: %w", cErr)
		}
		return true, false, nil
	}

	// Advance the first iteration unless resuming mid-LLM-call: a real turn step
	// at current_step-1 with no AssistantEvent means the dead predecessor
	// advanced the step but never wrote its assistant — re-run the LLM at that
	// step instead of double-bumping.
	needsAdvance = sess.CurrentStep <= 1 || hasAssistantAtStep(events, sess.CurrentStep-1)
	return false, needsAdvance, nil
}

// repairOrphanToolCalls appends a placeholder ToolResultEvent for every tool_call
// id (across the current context) that has no matching result, at the step of the
// AssistantEvent that declared it. Idempotent.
func (a *EventLoopAgent) repairOrphanToolCalls(ctx context.Context, events []db.EventRecord) error {
	declaredStep := map[string]int{}
	hasResult := map[string]bool{}
	for _, rec := range events {
		switch e := rec.Event.(type) {
		case *event.AssistantEvent:
			for _, tc := range e.ToolCalls {
				declaredStep[tc.ID] = rec.Step
			}
		case *event.ToolResultEvent:
			hasResult[e.ToolCallID] = true
		}
	}
	orphans := make([]string, 0)
	for id := range declaredStep {
		if !hasResult[id] {
			orphans = append(orphans, id)
		}
	}
	sort.Strings(orphans) // deterministic insertion order
	for _, id := range orphans {
		evt := &event.ToolResultEvent{
			ToolCallID: id,
			Content:    "[Tool execution interrupted by process crash; no result available.]",
			IsError:    true,
		}
		if err := a.env.Store.AppendEventAtStep(ctx, a.env.RunID, a.cfg.SessionID, declaredStep[id], evt); err != nil {
			return err
		}
	}
	return nil
}

// isDegenerateTurn reports whether a no-tool turn's text is a non-answer, and
// so should be nudged rather than accepted as an autonomous agent's result.
//
// Empty is the obvious case. The placeholder echo is the learned one: because
// we project a degenerate turn into the context AS the placeholder, some weak models
// see "(empty response)" as a legitimate thing for it to say and repeats it —
// which would otherwise pass the guard, since it is not empty. Treating the
// echo as empty costs nothing (a model that genuinely wants to emit only that
// string has nothing to say either) and closes the loop the placeholder opened.
func isDegenerateTurn(content string) bool {
	trimmed := strings.TrimSpace(content)
	return trimmed == "" || trimmed == emptyAssistantPlaceholder
}

// tailAssistantAtRest returns the trailing event iff it is a no-tool
// AssistantEvent (the "natural rest" predicate). events must be ordered.
func tailAssistantAtRest(events []db.EventRecord) *event.AssistantEvent {
	if len(events) == 0 {
		return nil
	}
	last := events[len(events)-1].Event
	if event.IsNoToolAssistant(last) {
		return last.(*event.AssistantEvent)
	}
	return nil
}

// hasAssistantAtStep reports whether any AssistantEvent sits at the given step.
func hasAssistantAtStep(events []db.EventRecord, step int) bool {
	for _, rec := range events {
		if rec.Step != step {
			continue
		}
		if _, ok := rec.Event.(*event.AssistantEvent); ok {
			return true
		}
	}
	return false
}

// bootstrap creates the session, seeds step-0 events, and advances to step 1.
// After bootstrap, current_step=1 so that messages arriving during park
// land at step 1 (compactible), not step 0 (permanent bootstrap).
func (a *EventLoopAgent) bootstrap(ctx context.Context) error {
	now := time.Now().UTC()

	// Persist this session's OWN workspace so resume reconstructs it from the
	// session row (sub-agents may have their own linked workspaces), not from
	// run config. Best-effort: a marshal failure logs and leaves it absent, and
	// reconstruction falls back to the run-config workspace.
	var meta map[string]any
	if a.env.Workspace != nil {
		if blob, err := workspace.Marshal(a.env.Workspace); err != nil {
			a.logger.Warn("marshal workspace for session metadata failed", "error", err)
		} else {
			meta = map[string]any{workspace.SessionMetadataKey: string(blob)}
		}
	}

	err := a.env.Store.CreateSession(ctx, db.SessionRecord{
		RunID:     a.env.RunID,
		SessionID: a.cfg.SessionID,
		AgentType: a.cfg.AgentType,
		Task:      a.cfg.Task,
		Status:    db.SessionOngoing,
		ParentID:  a.cfg.ParentID,
		Metadata:  meta,
		CreatedAt: now,
	})
	if err != nil {
		return err
	}

	// Step 0: bootstrap events (permanent, survive compaction).
	//
	// Ordering matters: everything that FRAMES the work is seeded ahead of the
	// Task, so the Task is the last bootstrap turn and reads as the distinct
	// "here's what you're asked to do" message rather than being sandwiched
	// between system context. The projection layer (buildMessages) further folds
	// the leading run of system events into the provider's native system slot, so
	// the Task becomes the first actual user turn. The one exception is
	// initial_recall, which is seeded AFTER the task — it's task-derived search
	// results, so it reads correctly as a follow-on to the task.

	// New-session marker first: pure framing ("a fresh agent session begins").
	newSessionContent := fmt.Sprintf("New agent session %q started at %s.",
		a.cfg.SessionID, util.FormatLocalISO(now))
	if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
		&event.SystemEvent{Content: newSessionContent, Marker: event.MarkerNewSession}); err != nil {
		return err
	}
	if a.cfg.SystemPrompt != "" {
		if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
			&event.SystemEvent{Content: a.cfg.SystemPrompt, Marker: event.MarkerSystemPrompt}); err != nil {
			return err
		}
	}
	// Optional CLI tools: describe the currently-available subset (probed now,
	// so a tool installed after server start is picked up by new sessions). The
	// agent invokes them via bash; absent tools are omitted entirely. Seeded
	// before the task so the agent sees its toolbox ahead of the work.
	if body := cli.BootstrapSnippet(a.cfg.CLITools); body != "" {
		if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
			&event.SystemEvent{Content: body, Marker: event.MarkerCLITools}); err != nil {
			return err
		}
	}
	// Workspace hint: tell the agent about ITS OWN working tree (path, VCS,
	// link/share provenance, alias) — the one place a live alias is resolved.
	// Seeded for every fresh session, ahead of the task.
	if a.env.Workspace != nil {
		if body := a.env.Workspace.Describe(ctx, a.cfg.ParentID); body != "" {
			if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
				&event.SystemEvent{Content: body, Marker: event.MarkerWorkspace}); err != nil {
				return err
			}
		}
	}
	// Operator AGENTS.md instructions — one SystemEvent containing whatever
	// the producer (server / headless CLI) combined at run-start from the
	// global (<data-dir>/AGENTS.md) and root-workspace (<ws>/AGENTS.md)
	// sources, persisted on the run's RunConfig. Read straight from the run
	// record (the raw-config layer) rather than carrying a copy on Env: this
	// is a fresh-session-only bootstrap step, so the one GetRun is cheap and
	// keeps Env a pure object graph. Every agent in the run (including linked
	// sub-agents) reads the same persisted snapshot. (If a sub-agent ever
	// needs its own rules, the operator can put them in <data-dir>/AGENTS.md
	// so every agent sees them.)
	if run, err := a.env.Store.GetRun(ctx, a.env.RunID); err != nil {
		return fmt.Errorf("read run config for AGENTS.md: %w", err)
	} else if run.Config.AgentsMD != "" {
		if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
			&event.SystemEvent{Content: run.Config.AgentsMD, Marker: event.MarkerAgentsMD}); err != nil {
			return err
		}
	}
	// The Task: seeded LAST among the bootstrap events so it's the distinct
	// "here's the work" turn, immediately followed only by its own recall.
	if a.cfg.Task != "" {
		if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
			&event.UserEvent{Content: a.cfg.Task}); err != nil {
			return err
		}
		// Seed task-relevant skills so the agent sees them on its first turn.
		// Kept AFTER the task: these are search results derived from the task.
		if a.cfg.InitialRecall != nil {
			if content := a.cfg.InitialRecall(ctx, a.cfg.Task); content != "" {
				if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
					&event.SystemEvent{Content: content, Marker: event.MarkerInitialRecall}); err != nil {
					return err
				}
			}
		}
	}

	// Advance to step 1 so messages during park (and the opening message below)
	// land at step 1 (compactible), not step 0 (permanent bootstrap).
	if _, err := a.env.Store.AdvanceStep(ctx, a.env.RunID, a.cfg.SessionID); err != nil {
		return fmt.Errorf("advance past bootstrap: %w", err)
	}

	// Opening message for a chat run (generic: appended for any agent if set).
	if a.cfg.FirstMessage != "" {
		if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
			&event.UserEvent{Content: a.cfg.FirstMessage}); err != nil {
			return err
		}
	}

	// Park immediately only when there's nothing to respond to: no task AND no
	// opening message. Autonomous (has task) and seeded chat runs proceed into
	// the loop; an empty interactive run waits to be woken by the first message.
	if a.cfg.Task == "" && a.cfg.FirstMessage == "" {
		a.logger.Debug("no task or opening message, parking until first message")
		if err := a.env.Store.UpdateSessionStatus(ctx, a.env.RunID, a.cfg.SessionID, db.SessionIdle); err != nil {
			return err
		}
	}
	return nil
}

// loop is the main agent loop.
//
// Step model: each iteration bumps current_step BEFORE the LLM call.
// The AssistantEvent and ToolResults are written at the "call step"
// (current_step before the bump). Events arriving during LLM generation
// land at current_step (after the bump), placing them at the next step.
//
// needsAdvance is the first-iteration advance decision from reconcileResume:
// false only when resuming a mid-LLM-call crash, where the dead predecessor
// already advanced and we must call the LLM at the existing call step rather
// than double-bumping. Every subsequent iteration advances.
func (a *EventLoopAgent) loop(ctx context.Context, needsAdvance bool) error {
	toolMap := tool.ByName(a.cfg.Tools)
	toolDefs := tool.Defs(a.cfg.Tools)

	// consecutiveCompactions guards against a pathological compact→retry→compact
	// loop (e.g. a context limit so small even the post-compaction context can't
	// fit). Reset after any successful LLM call.
	consecutiveCompactions := 0

	for {
		// ctx cancellation = shutdown OR a canceller-driven cancel. Either way
		// stop and return: shutdown preserves the status (recovered later); a
		// cancel already set it cancelled. The loop has no cancel logic of its own.
		if ctx.Err() != nil {
			a.logger.Debug("context cancelled, stopping")
			return nil
		}

		// No valid call step yet this iteration: a failure during step
		// determination records at the current step, not a stale turn's step.
		a.callStep = -1

		// Determine the call step. Normally we bump current_step FIRST, before
		// gathering context: that freezes the call-step window (once current_step
		// advances, every new event lands at the bumped step, so the <= callStep
		// set can't change) and closes the gather/advance gap. On a mid-LLM-call
		// resume we instead reuse the predecessor's already-advanced step (don't
		// double-bump): the call step is current_step-1.
		var callStep, newStep int
		if needsAdvance {
			ns, err := a.env.Store.AdvanceStep(ctx, a.env.RunID, a.cfg.SessionID)
			if err != nil {
				return a.recordFailure(ctx, fmt.Errorf("advance step: %w", err))
			}
			newStep, callStep = ns, ns-1
		} else {
			s, err := a.env.Store.GetSession(ctx, a.env.RunID, a.cfg.SessionID)
			if err != nil {
				return a.recordFailure(ctx, fmt.Errorf("read session: %w", err))
			}
			newStep, callStep = s.CurrentStep, s.CurrentStep-1
		}
		needsAdvance = true   // every subsequent iteration advances
		a.callStep = callStep // a crash this turn records at its call step
		a.logger.Debug("calling LLM", "call_step", callStep, "current_step", newStep)

		// Load context up to and including the call step (bootstrap + current
		// generation, step <= callStep). Events at the bumped step — arrivals
		// during/after now — are excluded here and picked up by the
		// during-generation check after the LLM call.
		events, err := a.env.Store.GetEvents(ctx, a.env.RunID, a.cfg.SessionID,
			db.EventFilter{CurrentContextOnly: true, EndStep: &callStep})
		if err != nil {
			return a.recordFailure(ctx, fmt.Errorf("load events: %w", err))
		}
		systemPrompt, messages := a.buildMessages(events)

		// Call LLM.
		resp, err := a.callLLM(ctx, llm.Request{
			SystemPrompt: systemPrompt,
			Messages:     messages,
			Tools:        toolDefs,
			// RunID-qualified so distinct runs don't collide on one cache namespace.
			SessionID: a.env.RunID + "/" + a.cfg.SessionID,
		}, callStep)
		if err != nil {
			// Reactive compaction: ONLY a provider-call error reaches here, so
			// this is the one site that may be a context-window overflow. If the
			// fast judge confirms it (and we haven't just compacted), summarize
			// the prior context and retry the SAME turn against the compacted
			// context (needsAdvance=false reuses this call step). Every other
			// failure — DB errors, ctx cancellation, tool errors — flows straight
			// to recordFailure untouched.
			if ctx.Err() == nil && consecutiveCompactions < maxConsecutiveCompactions &&
				a.tryCompact(ctx, err, callStep) {
				consecutiveCompactions++
				needsAdvance = false
				continue
			}
			return a.recordFailure(ctx, fmt.Errorf("llm call: %w", err))
		}
		consecutiveCompactions = 0

		// AssistantEvent for the CALL STEP (before the bump).
		assistantEvt := &event.AssistantEvent{
			Content:       resp.Content,
			Thoughts:      resp.Thoughts,
			ToolCalls:     convertToolCalls(resp.ToolCalls),
			StopReason:    resp.StopReason,
			ProviderExtra: resp.ProviderExtra,
			Usage: &event.Usage{
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				TotalTokens:      resp.Usage.TotalTokens,
				CacheReadTokens:  resp.Usage.CacheReadTokens,
				CacheWriteTokens: resp.Usage.CacheWriteTokens,
			},
		}

		// Tool calls: write the assistant first (so a crash mid-execution can
		// recover the declared calls via orphan repair), then execute and append
		// each result the moment its tool finishes, and finally mark the step
		// finalized. Appending per-result (rather than one terminal write) keeps
		// timestamps honest and stops a slow, blocking tool (await_event) from
		// hiding the others' completion; recovery mends any missing results per
		// tool-call id and finalizes the step (see reconcileResume).
		if len(resp.ToolCalls) > 0 {
			a.concludeNudges = 0 // substantive turn — reset the accidental-conclusion guard
			if err := a.env.Store.AppendEventAtStep(ctx, a.env.RunID, a.cfg.SessionID, callStep, assistantEvt); err != nil {
				return a.recordFailure(ctx, fmt.Errorf("record assistant: %w", err))
			}
			if err := a.executeAndAppendToolResults(ctx, callStep, resp.ToolCalls, toolMap); err != nil {
				return a.recordFailure(ctx, fmt.Errorf("append tool results: %w", err))
			}
			if err := a.env.Store.MarkStepFinalized(ctx, a.env.RunID, a.cfg.SessionID, callStep); err != nil {
				return a.recordFailure(ctx, fmt.Errorf("finalize tool step: %w", err))
			}
			continue
		}

		// No tool calls: the assistant is the step's only event — write and
		// finalize it atomically.
		if err := a.env.Store.FinalizeStep(ctx, a.env.RunID, a.cfg.SessionID, callStep, []event.Event{assistantEvt}); err != nil {
			return a.recordFailure(ctx, fmt.Errorf("finalize turn: %w", err))
		}

		// No tool calls — but events may have arrived during generation (they
		// land at the bumped step). The session is still ongoing, so ANY queued
		// event counts here, Notice or Input alike: react to it rather than
		// finishing. The Notice/Input distinction only gates whether a *dormant*
		// session is revived (see docs/session_lifecycle.md) — an ongoing loop
		// always processes what's queued. So e.g. a child_result(crashed) that
		// lands as we're about to conclude pulls us back for one more turn to see
		// it, which is intended.
		nextStepCount, _ := a.env.Store.GetEventCount(ctx, a.env.RunID, a.cfg.SessionID,
			db.EventFilter{StartStep: &newStep})
		if nextStepCount > 0 {
			a.logger.Debug("events arrived during generation, continuing")
			continue
		}

		// A bare no-tool-call turn means "done for now". What it produces
		// depends on the agent's nature (see docs/session_lifecycle.md).
		if !a.cfg.Interactive {
			// Accidental-conclusion guard: an autonomous agent's final turn text IS
			// its result, so an EMPTY message with no tool calls is almost always a
			// degenerate turn (the model reasoned but forgot to act), not a real
			// conclusion. Nudge it back into the loop instead of concluding on
			// nothing — bounded by maxConcludeNudges (in-memory, reset on any
			// substantive turn) so a degenerate model can't loop forever. The empty
			// AssistantEvent stays persisted (step-model invariant); eventToMessage
			// projects it as a non-empty placeholder so the replayed context is
			// provider-safe and the nudge that follows reads coherently.
			if isDegenerateTurn(resp.Content) && a.concludeNudges < maxConcludeNudges {
				a.concludeNudges++
				a.logger.Debug("empty no-tool turn; nudging instead of concluding", "nudge", a.concludeNudges)
				if _, err := a.env.Store.AppendEvent(ctx, a.env.RunID, a.cfg.SessionID,
					&event.UserEvent{Content: concludeNudgeText}); err != nil {
					return a.recordFailure(ctx, fmt.Errorf("append conclude nudge: %w", err))
				}
				continue
			}
			// Autonomous: conclude — the turn's text is the result.
			a.logger.Debug("no tool calls, concluding")
			return a.conclude(resp.Content)
		}

		// Interactive (chatbot): park idle and wait for the next user input.
		a.logger.Debug("no tool calls, parking idle")
		if err := a.env.Store.UpdateSessionStatus(ctx, a.env.RunID, a.cfg.SessionID, db.SessionIdle); err != nil {
			return a.recordFailure(ctx, fmt.Errorf("idle session: %w", err))
		}
		woken, err := a.waitForFollowUp(ctx)
		if err != nil {
			a.logger.Debug("context cancelled while idle, exiting")
			return nil
		}
		if !woken {
			a.logger.Debug("idle timeout, exiting goroutine")
			return nil
		}
		a.logger.Debug("woken from idle, resuming")
		if err := a.env.Store.UpdateSessionStatus(ctx, a.env.RunID, a.cfg.SessionID, db.SessionOngoing); err != nil {
			return a.recordFailure(ctx, fmt.Errorf("resume session: %w", err))
		}
	}
}

// waitForFollowUp parks an idle or just-restarted session until a new Input
// event arrives or the idle timeout expires. It returns true if an Input is
// pending (resume), false on timeout (exit cold; a later Input respawns the
// goroutine), or an error if ctx was cancelled.
//
// Input-gated (the idle row of the matrix): a Notice landing while parked is
// ignored — the waiter bumps, we re-check, find no Input, and re-sleep. The
// counter is snapshotted before the DB check so an Input arriving in the gap
// still wakes the WaitAfter. (An await_event park, by contrast, advances on any
// event.)
func (a *EventLoopAgent) waitForFollowUp(ctx context.Context) (bool, error) {
	timeout := a.cfg.IdleTimeout
	if timeout <= 0 {
		timeout = DefaultIdleTimeout
	}
	deadline := time.Now().Add(timeout)
	for {
		before := a.handle.Counter()
		if a.hasPendingInput(ctx) {
			return true, nil
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return false, nil
		}
		after, err := a.handle.WaitAfter(ctx, before, remaining)
		if err != nil {
			return false, err
		}
		if after == before {
			return false, nil // timed out
		}
		// Woke on a notify; loop to re-check for an Input (re-sleep on a Notice).
	}
}

// hasPendingInput reports whether an Input-class event is already sitting at or
// past the session's current step (events arriving while parked land at the
// current step). Used to short-circuit both the idle wait and the Run-entry wait.
func (a *EventLoopAgent) hasPendingInput(ctx context.Context) bool {
	sess, err := a.env.Store.GetSession(ctx, a.env.RunID, a.cfg.SessionID)
	if err != nil {
		return false
	}
	step := sess.CurrentStep
	events, err := a.env.Store.GetEvents(ctx, a.env.RunID, a.cfg.SessionID,
		db.EventFilter{StartStep: &step})
	if err != nil {
		return false
	}
	for _, rec := range events {
		if db.IsInput(rec.Event) {
			return true
		}
	}
	return false
}

// --- Message building ---

// buildMessages projects events into an LLM request. It returns the CONTIGUOUS
// LEADING run of system events folded into a single systemPrompt (for the
// provider's native system slot), plus the remaining messages in order.
//
// Why fold only the leading run: most providers (Gemini, Vertex-Anthropic) have
// no mid-conversation system role and demote a system message to a `user` turn.
// With the bootstrap ordering (all framing ahead of the Task), that used to bury
// the Task among ~5 pseudo-user system turns. Hoisting the leading system run
// into the native system slot makes the Task the FIRST user turn — undiluted —
// on every provider. System events that appear AFTER the first non-system
// message (e.g. task-derived initial_recall, or a mid-run marker) keep their
// original position and role handling; only the leading cluster is special.
//
// Banners are preserved (SystemEvent.ToText): the marker headers signal the kind
// of each system segment and we keep them for consistency with the trajectory.
func (a *EventLoopAgent) buildMessages(events []db.EventRecord) (systemPrompt string, messages []llm.Message) {
	var sys []string
	leading := true
	for _, rec := range events {
		msg := a.eventToMessage(rec.Event)
		if msg == nil {
			continue
		}
		if leading && msg.Role == llm.RoleSystem {
			sys = append(sys, msg.Content)
			continue
		}
		leading = false // first non-system message ends the leading cluster
		messages = append(messages, *msg)
	}
	return strings.Join(sys, "\n\n"), messages
}

func (a *EventLoopAgent) eventToMessage(evt event.Event) *llm.Message {
	switch e := evt.(type) {
	case *event.SystemEvent:
		return &llm.Message{Role: llm.RoleSystem, Content: e.ToText()}
	case *event.UserEvent:
		return &llm.Message{Role: llm.RoleUser, Content: e.Content}
	case *event.AssistantEvent:
		// Empty content + no tool calls (the degenerate turn from the accidental-
		// conclusion path): substitute a non-empty placeholder so the message is
		// provider-safe (Gemini rejects empty parts) and stays visible for the nudge.
		// ProviderExtra is preserved so thought signatures / thinking blocks carry
		// over — the whole point of keeping the turn (dropping them is API-accepted
		// but needlessly breaks the reasoning chain).
		content := e.Content
		if strings.TrimSpace(content) == "" && len(e.ToolCalls) == 0 {
			content = emptyAssistantPlaceholder
		}
		return &llm.Message{
			Role:          llm.RoleAssistant,
			Content:       content,
			ToolCalls:     eventToolCallsToLLM(e.ToolCalls),
			ProviderExtra: e.ProviderExtra,
		}
	case *event.ToolResultEvent:
		return &llm.Message{
			Role:        llm.RoleToolResult,
			Content:     e.Content,
			ToolCallID:  e.ToolCallID,
			Attachments: a.loadAttachments(e.Attachments),
		}
	case *event.CompactionEvent:
		// Frame the summary for the CONSUMING LLM at projection time (not baked
		// into the event), so it reads the summary as its own memory and keeps
		// going. Kept out of CompactionEvent.ToText so the observer/inspect/search
		// renderings stay clean. Mirrors how Claude Code injects a separate
		// "continue from here" message after compaction.
		//
		// Projected as RoleUser (not RoleSystem): after a compaction whose
		// boundary is at the tip of the conversation, the new-generation context
		// is only step-0 bootstrap (all system) followed by this event. If this
		// were RoleSystem, buildMessages would fold it into the leading system
		// cluster and return zero messages, which providers reject
		// ("messages: Field required"). As a user turn it ends the leading
		// cluster and becomes the first real message.
		return &llm.Message{Role: llm.RoleUser, Content: compactionFraming + e.ToText()}
	case *event.MessageEvent:
		return &llm.Message{Role: llm.RoleUser, Content: e.ToText()}
	case *event.ChildResultEvent:
		return &llm.Message{Role: llm.RoleUser, Content: e.ToText()}
	case *event.RecoverEvent:
		return &llm.Message{Role: llm.RoleUser, Content: e.ToText()}
	default:
		return nil
	}
}

func eventToolCallsToLLM(tcs []event.ToolCall) []llm.ToolCall {
	if len(tcs) == 0 {
		return nil
	}
	result := make([]llm.ToolCall, len(tcs))
	for i, tc := range tcs {
		result[i] = llm.ToolCall{ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments}
	}
	return result
}

func convertToolCalls(tcs []llm.ToolCall) []event.ToolCall {
	if len(tcs) == 0 {
		return nil
	}
	result := make([]event.ToolCall, len(tcs))
	for i, tc := range tcs {
		result[i] = event.ToolCall{ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments}
	}
	return result
}

// callLLM runs the LLM call. For interactive agents with a broadcaster it
// streams, emitting each token delta as an ephemeral stream_chunk for the live
// UI, and returns the accumulated response. Otherwise it makes a blocking call.
// The persisted AssistantEvent is identical either way; chunks are preview-only.
func (a *EventLoopAgent) callLLM(ctx context.Context, req llm.Request, callStep int) (*llm.Response, error) {
	if !a.cfg.Interactive || a.env.Broadcaster == nil {
		return a.env.LLM.Call(ctx, req)
	}
	stream, err := a.env.LLM.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	for stream.Next() {
		ev := stream.Event()
		a.env.Broadcaster.Chunk(a.env.RunID, a.cfg.SessionID, callStep, ev.DeltaText, ev.DeltaThoughts)
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	return stream.Response(), nil
}

// executeAndAppendToolResults runs the step's tool calls and appends each
// result to the store the instant its tool finishes, instead of batching them
// into one terminal write. This keeps result timestamps honest and lets the
// chat UI show each tool completing independently — a fast tool (send_message)
// is no longer hidden behind a slow, blocking one (await_event) sharing the
// step. The caller marks the step finalized once this returns.
//
// ExecuteAll surfaces tool errors as result content, so the only failures here
// are store-append errors. Persistence is serialized via mu, so it's safe
// regardless of the blob store's own concurrency; the tools still run in
// parallel. Attachment bytes are written to the blob store and replaced with
// content-key references before persistence.
func (a *EventLoopAgent) executeAndAppendToolResults(ctx context.Context, callStep int, calls []llm.ToolCall, toolMap map[string]*tool.Tool) error {
	var mu sync.Mutex
	var appendErr error
	tool.ExecuteAll(ctx, calls, toolMap, func(r tool.CallResult) {
		mu.Lock()
		defer mu.Unlock()
		if appendErr != nil {
			return
		}
		evt := &event.ToolResultEvent{
			Content:    r.Result.Content,
			ToolCallID: r.ToolCallID,
			IsError:    r.Result.IsError,
		}
		for _, att := range r.Result.Attachments {
			if ref, ok := a.storeAttachment(att); ok {
				evt.Attachments = append(evt.Attachments, ref)
			}
		}
		if err := a.env.Store.AppendEventAtStep(ctx, a.env.RunID, a.cfg.SessionID, callStep, evt); err != nil {
			appendErr = err
		}
	})
	return appendErr
}

// storeAttachment writes an attachment's bytes to the run blob store and returns
// a persistable reference (key+mime+size, no bytes). Returns ok=false — dropping
// the image but keeping the text result — when there's no blob store or the
// write fails. Oversized images are downscaled first (see clampImage).
func (a *EventLoopAgent) storeAttachment(att event.Attachment) (event.Attachment, bool) {
	if len(att.Data) == 0 {
		return event.Attachment{}, false
	}
	if a.cfg.BlobStore == nil {
		a.logger.Warn("dropping tool-result attachment: no blob store", "mime", att.MimeType)
		return event.Attachment{}, false
	}
	data, mime := a.clampImage(att.Data, att.MimeType)
	key, err := a.cfg.BlobStore.Put(data)
	if err != nil {
		a.logger.Error("dropping tool-result attachment: blob write failed", "error", err, "mime", mime)
		return event.Attachment{}, false
	}
	return event.Attachment{
		MimeType:   mime,
		BlobKey:    key,
		Size:       len(data),
		SourceHint: att.SourceHint,
	}, true
}

// clampImage downscales an image attachment whose pixel dimensions exceed the
// provider limit (providers reject by dimension, not byte size). Non-image
// attachments pass through untouched. On any decode/encode failure it keeps the
// ORIGINAL bytes and logs — a clamp problem must never drop the image or fail
// the turn; a too-large image failing later at the API is strictly better than
// silently losing it here.
func (a *EventLoopAgent) clampImage(data []byte, mime string) ([]byte, string) {
	if !strings.HasPrefix(mime, "image/") {
		return data, mime
	}
	out, outMime, resized, err := imageutil.Clamp(data, mime, imageutil.DefaultMaxDim)
	if err != nil {
		a.logger.Warn("image clamp failed; storing original", "error", err, "mime", mime, "bytes", len(data))
		return data, mime
	}
	if resized {
		a.logger.Info("downscaled oversized image attachment",
			"from_bytes", len(data), "to_bytes", len(out),
			"from_mime", mime, "to_mime", outMime, "max_dim", imageutil.DefaultMaxDim)
	}
	return out, outMime
}

// loadAttachments resolves persisted attachment references into provider-ready
// image attachments by reading bytes from the blob store. Missing or unreadable
// blobs are skipped (logged) so a lost image never breaks the turn.
func (a *EventLoopAgent) loadAttachments(refs []event.Attachment) []llm.Attachment {
	if len(refs) == 0 || a.cfg.BlobStore == nil {
		return nil
	}
	var out []llm.Attachment
	for _, ref := range refs {
		if ref.BlobKey == "" {
			continue
		}
		data, err := a.cfg.BlobStore.ReadAll(ref.BlobKey)
		if err != nil {
			a.logger.Warn("skipping attachment: blob unreadable", "key", ref.BlobKey, "error", err)
			continue
		}
		out = append(out, llm.Attachment{
			MimeType:   ref.MimeType,
			Base64Data: base64.StdEncoding.EncodeToString(data),
		})
	}
	return out
}

// conclude ends an autonomous agent: its final turn's text is the result. It
// cascade-cancels any owned children, then atomically sets concluded and posts
// the result to the parent. Uses a background context so a late ctx cancellation
// can't tear down the terminal write.
func (a *EventLoopAgent) conclude(result string) error {
	a.cascadeCancelChildren("parent " + a.cfg.SessionID + " concluded")
	return a.env.Store.TerminateAndNotifyParent(
		context.Background(), a.env.RunID, a.cfg.SessionID, a.cfg.ParentID, db.SessionConcluded, result, nil, -1)
}

// cascadeCancelChildren canceller-drives a recursive cancel of each active child
// (CancelSession recurses into grandchildren and skips already-terminal ones).
func (a *EventLoopAgent) cascadeCancelChildren(reason string) {
	children, err := a.env.Store.GetChildSessions(
		context.Background(), a.env.RunID, a.cfg.SessionID)
	if err != nil {
		return
	}
	for _, child := range children {
		// Best-effort: a child that won't cancel (e.g. its terminal write failed)
		// is logged but doesn't block this agent's own conclusion. Recover will
		// reconcile any session left ongoing-but-dead.
		if err := session.CancelSession(a.env.Store, a.env.Registry, a.env.RunID, child.SessionID, reason); err != nil {
			a.logger.Warn("cascade cancel child failed", "child", child.SessionID, "error", err)
		}
	}
}

// recordFailure marks the session crashed and notifies the parent, atomically
// recording the error on its own stream. Children are kept alive (no cascade) so
// they resume under the parent on restart.
func (a *EventLoopAgent) recordFailure(ctx context.Context, err error) error {
	a.logger.Error("agent failure", "error", err)
	// A cancelled ctx means shutdown or a canceller-driven cancel: "stop now,
	// status already handled, return" — never record a crash (that would clobber
	// a cancelled status and double-notify the parent).
	if ctx.Err() != nil {
		a.logger.Debug("context cancelled during failure; not crashing")
		return nil
	}
	// Skip DB writes if the error itself is from a DB operation — writing to a
	// broken DB would just produce another error. The session stays ongoing and
	// is recovered as ongoing-but-dead on the next reconcile. Every Store error
	// wraps db.ErrStore (tagged at the store boundary), so this classification
	// is exact — no error-message matching.
	if errors.Is(err, db.ErrStore) {
		return err
	}
	_ = a.env.Store.TerminateAndNotifyParent(context.Background(), a.env.RunID, a.cfg.SessionID,
		a.cfg.ParentID, db.SessionCrashed, err.Error(),
		&event.SystemEvent{Content: err.Error(), Marker: event.MarkerError}, a.callStep)
	return err
}

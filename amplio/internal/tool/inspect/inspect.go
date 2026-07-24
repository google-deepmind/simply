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

// Package inspect provides tools for inspecting sessions and events.
package inspect

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/tool"
	"amplio/internal/toolsummary"
	"amplio/internal/util"
)

// Deps holds the dependencies needed by inspection tools.
type Deps struct {
	Store db.Store
	RunID string
}

// run returns the run to inspect: the override (a cross-run target supplied via
// the tool's optional run_id arg) when non-empty, else the current run.
func (d *Deps) run(override string) string {
	if override != "" {
		return override
	}
	return d.RunID
}

// Observation kinds mirrored from the observer.
const (
	phaseSummaryKind = "phase_summary"
	stepSummaryKind  = "step_summary"
)

// --- session_list ---

// sessionInactiveWindow bounds how long a finished (terminal) session counts as
// "active" after its last status change. Live sessions (ongoing/awaiting) are
// always active regardless of age; this only governs how long
// concluded/crashed/cancelled ones stay rendered in full. Long (1h) because
// session_list is an on-demand orientation tool, not a per-turn feed.
const sessionInactiveWindow = time.Hour

type sessionListParams struct {
	RunID string `json:"run_id,omitempty" jsonschema_description:"Optional: a prior run's id to list sessions for instead of the current run."`
}

// sessionActive reports whether a session is "active": live (ongoing/awaiting)
// regardless of age, or terminal but having changed status within
// sessionInactiveWindow. Liveness is judged by STATUS, not timestamp, so a
// long-running agent (still ongoing) never counts as old.
func sessionActive(s db.SessionRecord, now time.Time) bool {
	if !db.SessionTerminalStatuses[s.Status] {
		return true // ongoing / awaiting / idle: live
	}
	return now.Sub(s.StatusChangedAt) <= sessionInactiveWindow
}

func SessionList(deps *Deps) *tool.Tool {
	return &tool.Tool{
		Name: "session_list",
		Description: "List sessions in a run (the current run, or a prior run via run_id) as an " +
			"indented tree (parent → children, children in spawn order). Active sessions show " +
			"full detail (status, agent type, step, task); long-finished ones are condensed to a " +
			"one-line [inactive] summary. Use session_summary for what a finished session did.",
		ParamType: &sessionListParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[sessionListParams](args)
			if errResult != nil {
				return errResult, nil
			}
			runID := deps.run(params.RunID)
			sessions, err := deps.Store.ListSessions(ctx, runID)
			if err != nil {
				return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
			}
			if len(sessions) == 0 {
				return &tool.Result{Content: fmt.Sprintf("No sessions in run %s.", runID)}, nil
			}
			return &tool.Result{Content: renderSessionTree(runID, sessions, time.Now())}, nil
		},
	}
}

// renderSessionTree formats sessions as an indented tree. ListSessions returns
// rows ordered by created_at ASC, so children appear in spawn order under each
// parent. EVERY session is shown (so the agent never loses track of a spawned
// child, including crashed ones), but a long-finished session is condensed to a
// single [inactive] line with the task omitted — drill in with session_summary
// for details.
func renderSessionTree(runID string, sessions []db.SessionRecord, now time.Time) string {
	byID := make(map[string]db.SessionRecord, len(sessions))
	for _, s := range sessions {
		byID[s.SessionID] = s
	}
	// Second pass (sessions stay in created_at order): a session is a child if its
	// parent is present in this set, else a root (top-level, or parent out of set).
	children := make(map[string][]string) // parentID -> child session ids (created_at order)
	var roots []string
	for _, s := range sessions {
		if _, ok := byID[s.ParentID]; s.ParentID != "" && ok {
			children[s.ParentID] = append(children[s.ParentID], s.SessionID)
		} else {
			roots = append(roots, s.SessionID)
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Sessions in run %s:\n", runID)
	var walk func(id string, depth int)
	walk = func(id string, depth int) {
		s := byID[id]
		indent := strings.Repeat("  ", depth+1)
		if sessionActive(s, now) {
			fmt.Fprintf(&b, "%s%s: agent=%s status=%s step=%d created=%s status_changed=%s\n",
				indent, s.SessionID, s.AgentType, s.Status, s.CurrentStep,
				util.FormatLocalISO(s.CreatedAt), util.FormatLocalISO(s.StatusChangedAt))
			if s.Task != "" {
				fmt.Fprintf(&b, "%s  task=%q\n", indent, util.TruncateRunes(s.Task, 80))
			}
		} else {
			// Condensed: keep identity + status + when it ended so the tree stays
			// complete and crashed children stay visible; omit the task (the
			// longest field) — session_summary is the drill-in.
			fmt.Fprintf(&b, "%s%s: agent=%s status=%s step=%d status_changed=%s [inactive]\n",
				indent, s.SessionID, s.AgentType, s.Status, s.CurrentStep,
				util.FormatLocalISO(s.StatusChangedAt))
		}
		for _, c := range children[id] {
			walk(c, depth+1)
		}
	}
	for _, r := range roots {
		walk(r, 0)
	}
	return b.String()
}

// --- session_summary ---

type sessionSummaryParams struct {
	SessionID string `json:"session_id" jsonschema:"required" jsonschema_description:"Session ID to summarize"`
	RunID     string `json:"run_id,omitempty" jsonschema_description:"Optional: a prior run's id to inspect instead of the current run."`
}

// SessionSummary returns a compact, layered view of what a session did: closed
// phase summaries, then per-step summaries for the steps after the last closed
// phase, then the raw tail of not-yet-summarized events. The fastest way to
// understand a session without reading all its raw events.
func SessionSummary(deps *Deps) *tool.Tool {
	return &tool.Tool{
		Name:        "session_summary",
		Description: "View a compact layered summary of a session: closed phase summaries, recent per-step summaries, and the raw latest events. Use this before drilling into raw steps.",
		ParamType:   &sessionSummaryParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[sessionSummaryParams](args)
			if errResult != nil {
				return errResult, nil
			}
			out, err := LayeredSummary(ctx, deps.Store, deps.run(params.RunID), params.SessionID, -1)
			if err != nil {
				return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
			}
			if strings.TrimSpace(out) == "" {
				return &tool.Result{Content: fmt.Sprintf("No summary yet for session %s (it may be too new; use session_steps for raw events).", params.SessionID)}, nil
			}
			return &tool.Result{Content: fmt.Sprintf("Summary of session %s:\n\n%s", params.SessionID, out)}, nil
		},
	}
}

// LayeredSummary renders a compact, bounded, layered view of a session's work
// through throughStep (use a negative throughStep for the whole session): an
// optional prior-compaction anchor, closed phase summaries, per-step summaries
// for steps after the last closed phase, and the raw tail of not-yet-summarized
// events. It is the curated context shared by the session_summary tool and by
// context compaction — keeping a summarizer's input small even when the raw
// event stream is too large to fit.
func LayeredSummary(ctx context.Context, store db.Store, runID, sessionID string, throughStep int) (string, error) {
	sess, err := store.GetSession(ctx, runID, sessionID)
	if err != nil {
		return "", err
	}
	if throughStep < 0 {
		throughStep = sess.CurrentStep
	}

	// Scan current-context events <= throughStep to anchor on the task and the
	// most recent prior compaction summary. Cheap: in-memory string work, not an
	// LLM call — only the rendered (compressed) portions below cost tokens.
	events, err := store.GetEvents(ctx, runID, sessionID, db.EventFilter{
		CurrentContextOnly: true, EndStep: &throughStep,
	})
	if err != nil {
		return "", err
	}
	task, priorSummary, priorStep := scanAnchors(events)

	var b strings.Builder
	if task != "" {
		fmt.Fprintf(&b, "TASK:\n%s\n\n", truncate(task, 2000))
	}
	if priorSummary != "" {
		fmt.Fprintf(&b, "EARLIER CONTEXT (already-summarized memory through step %d):\n%s\n\n", priorStep, priorSummary)
	}

	// Everything at or below the prior compaction is captured by priorSummary.
	floor := priorStep

	// Render the always-open final phase first — its lower bound (openLo) tells us
	// which phases are still CLOSED (end_step <= openLo) vs. folded into the open
	// phase (the force-reopened last one).
	openBlock, openLo, err := FormatOpenPhase(ctx, store, runID, sessionID, throughStep, 0)
	if err != nil {
		return "", err
	}

	// Closed phase summaries with floor < end_step <= openLo.
	phases, err := store.GetObservations(ctx, runID, db.ObsFilter{Kind: phaseSummaryKind, SessionID: sessionID})
	if err != nil {
		return "", err
	}
	wrotePhases := false
	for _, rec := range phases {
		end := obsInt(rec.Data, "end_step")
		if end <= floor || end > openLo {
			continue
		}
		if !wrotePhases {
			b.WriteString("PHASES:\n")
			wrotePhases = true
		}
		title, _ := rec.Data["title"].(string)
		summary, _ := rec.Data["summary"].(string)
		fmt.Fprintf(&b, "\n--- steps %d-%d: %s ---\n%s\n", obsInt(rec.Data, "start_step"), end, title, summary)
	}

	if strings.TrimSpace(openBlock) != "" {
		fmt.Fprintf(&b, "\nOPEN PHASE (recent activity, steps %d+):\n%s", openLo+1, openBlock)
	}

	return b.String(), nil
}

// openPhaseTailN is the default number of trailing steps rendered with their full
// expanded body (see expandStep) even when they carry a step summary.
const openPhaseTailN = 20

// openPhaseBodyCap bounds each rendered event body so a persisted trace (the
// compaction augmentation) can't be blown up by one giant paste. Generous enough
// that normal messages render "in full".
const openPhaseBodyCap = 2000

// FormatOpenPhase renders the session's final, ALWAYS-open phase through
// throughStep (negative = current step) as a per-step activity trace, and
// returns it plus openLo — the exclusive lower step bound of that phase (so a
// caller can render the phases at or below it as closed).
//
// Force-open rule: the open phase is the region after the last CLOSED phase. If
// steps exist after the last closed phase, that region is the open phase; if
// everything through throughStep is already phased, the LAST phase is re-opened
// so there is always a visible open phase.
//
// Each step renders as a header line ("step N: <step summary>", or "step N:" when
// unsummarized) optionally followed by an expanded body. The body is shown per
// expandStep (no usable summary, recent, external input, or a no-tool-call
// assistant reply). In the body, user/agent/environment/child inputs and
// assistant messages render in full (capped), tool calls collapse to one-line
// verb/target briefs (one per line), and tool results are omitted entirely.
func FormatOpenPhase(ctx context.Context, store db.Store, runID, sessionID string, throughStep, tailN int) (block string, openLo int, err error) {
	if tailN <= 0 {
		tailN = openPhaseTailN
	}
	sess, err := store.GetSession(ctx, runID, sessionID)
	if err != nil {
		return "", 0, err
	}
	if throughStep < 0 {
		throughStep = sess.CurrentStep
	}

	events, err := store.GetEvents(ctx, runID, sessionID, db.EventFilter{
		CurrentContextOnly: true, EndStep: &throughStep,
	})
	if err != nil {
		return "", 0, err
	}

	// floor: everything at or below the latest compaction is subsumed by its
	// summary, so the open phase never reaches below it.
	floor := 0
	for _, rec := range events {
		if _, ok := rec.Event.(*event.CompactionEvent); ok && rec.Step > floor {
			floor = rec.Step
		}
	}

	// Compute openLo from the closed-phase boundaries above the floor.
	phases, err := store.GetObservations(ctx, runID, db.ObsFilter{Kind: phaseSummaryKind, SessionID: sessionID})
	if err != nil {
		return "", 0, err
	}
	var phaseEnds []int
	for _, rec := range phases {
		if end := obsInt(rec.Data, "end_step"); end > floor && end <= throughStep {
			phaseEnds = append(phaseEnds, end)
		}
	}
	sort.Ints(phaseEnds)
	openLo = floor
	if n := len(phaseEnds); n > 0 {
		if throughStep > phaseEnds[n-1] {
			openLo = phaseEnds[n-1] // real steps follow the last closed phase
		} else if n >= 2 {
			openLo = phaseEnds[n-2] // all phased: re-open the last phase
		} // single phase == throughStep: re-open from the floor
	}

	// Step summaries for the open range, keyed by step.
	stepSum := map[int]string{}
	if throughStep > openLo {
		lo := openLo + 1
		recs, err := store.GetObservations(ctx, runID, db.ObsFilter{
			Kind: stepSummaryKind, SessionID: sessionID, StartStep: &lo, EndStep: &throughStep,
		})
		if err != nil {
			return "", 0, err
		}
		for _, rec := range recs {
			if rec.Step != nil {
				s, _ := rec.Data["summary"].(string)
				stepSum[*rec.Step] = s
			}
		}
	}

	// Group open-range events by step (events are ordered step ASC, created ASC).
	byStep := map[int][]db.EventRecord{}
	var steps []int
	for _, rec := range events {
		if rec.Step <= openLo || rec.Step > throughStep {
			continue
		}
		if _, seen := byStep[rec.Step]; !seen {
			steps = append(steps, rec.Step)
		}
		byStep[rec.Step] = append(byStep[rec.Step], rec)
	}
	sort.Ints(steps)

	var b strings.Builder
	for _, step := range steps {
		recs := byStep[step]
		// A failed-summarization sentinel is not a usable summary: drop it from the
		// header and expand the step so the raw events show instead.
		summary := stepSum[step]
		hasSummary := summary != "" && !db.IsSummarizationFailure(summary)
		if hasSummary {
			fmt.Fprintf(&b, "step %d: %s\n", step, summary)
		} else {
			fmt.Fprintf(&b, "step %d:\n", step)
		}
		if expandStep(recs, step, throughStep, tailN, hasSummary) {
			renderStepBody(&b, recs)
		}
	}
	return b.String(), openLo, nil
}

// expandStep reports whether a step should render its full body (not just the
// summary header). We expand when there is no usable summary, when the step is
// within the last tailN steps, when it carries an external input (operator/
// agent/environment message, concluded child, recover marker), or when it holds
// a no-tool-call assistant reply — the operator-facing summary of work, worth
// surfacing verbatim.
func expandStep(recs []db.EventRecord, step, throughStep, tailN int, hasSummary bool) bool {
	if !hasSummary || step > throughStep-tailN {
		return true
	}
	for _, rec := range recs {
		if db.IsInput(rec.Event) {
			return true
		}
		if a, ok := rec.Event.(*event.AssistantEvent); ok &&
			len(a.ToolCalls) == 0 && strings.TrimSpace(a.Content) != "" {
			return true
		}
	}
	return false
}

// renderStepBody writes the economical multiline body for one step's events:
// inputs and assistant messages in full (capped), tool calls as one-line briefs,
// tool results omitted, compaction skipped (shown as the preamble).
func renderStepBody(b *strings.Builder, recs []db.EventRecord) {
	for _, rec := range recs {
		switch e := rec.Event.(type) {
		case *event.AssistantEvent:
			if txt := strings.TrimSpace(e.Content); txt != "" {
				writeBody(b, "assistant", txt)
			}
			for _, tc := range e.ToolCalls {
				fmt.Fprintf(b, "    · %s\n", toolsummary.Brief(tc.Name, tc.Arguments))
			}
		case *event.UserEvent:
			writeBody(b, "user", e.Content)
		case *event.MessageEvent:
			label := "message"
			if e.Sender != "" {
				label = "message from " + e.Sender
			}
			writeBody(b, label, e.Content)
		case *event.ChildResultEvent:
			writeBody(b, fmt.Sprintf("child %s %s", e.ChildSessionID, e.Verdict), e.Content)
		case *event.RecoverEvent:
			writeBody(b, "recover", e.Content)
		case *event.SystemEvent:
			label := "system"
			if e.Marker != "" {
				label = "system:" + e.Marker
			}
			writeBody(b, label, e.Content)
			// ToolResultEvent and CompactionEvent are intentionally omitted.
		}
	}
}

// writeBody writes a labeled, indented, multiline block. The label prefixes the
// first line; continuation lines keep the same indent. Content is capped to keep
// a persisted trace bounded.
func writeBody(b *strings.Builder, label, content string) {
	content = strings.TrimRight(content, "\n")
	if content == "" {
		fmt.Fprintf(b, "    [%s]\n", label)
		return
	}
	content = truncate(content, openPhaseBodyCap)
	for i, ln := range strings.Split(content, "\n") {
		if i == 0 {
			fmt.Fprintf(b, "    [%s] %s\n", label, ln)
		} else {
			fmt.Fprintf(b, "    %s\n", ln)
		}
	}
}

// scanAnchors extracts the originating task (the step-0 user event) and the most
// recent prior compaction summary (with its step) from an ordered event slice.
func scanAnchors(events []db.EventRecord) (task, priorSummary string, priorStep int) {
	for _, rec := range events {
		switch e := rec.Event.(type) {
		case *event.UserEvent:
			if rec.Step == 0 && task == "" {
				task = e.Content
			}
		case *event.CompactionEvent:
			priorSummary, priorStep = e.Content, rec.Step
		}
	}
	return task, priorSummary, priorStep
}

// truncate clips s to at most n bytes, appending an ellipsis when it cuts.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func obsInt(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	}
	return 0
}

// --- session_steps ---

type sessionStepsParams struct {
	SessionID string `json:"session_id" jsonschema:"required" jsonschema_description:"Session ID to inspect"`
	StartStep *int   `json:"start_step,omitempty" jsonschema_description:"Start step (inclusive)"`
	EndStep   *int   `json:"end_step,omitempty" jsonschema_description:"End step (inclusive)"`
	RunID     string `json:"run_id,omitempty" jsonschema_description:"Optional: a prior run's id to inspect instead of the current run."`
}

func SessionSteps(deps *Deps) *tool.Tool {
	return &tool.Tool{
		Name:        "session_steps",
		Description: "View events in a session, optionally filtered by step range. Shows event type and content summary per step.",
		ParamType:   &sessionStepsParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[sessionStepsParams](args)
			if errResult != nil {
				return errResult, nil
			}
			events, err := deps.Store.GetEvents(ctx, deps.run(params.RunID), params.SessionID, db.EventFilter{
				StartStep: params.StartStep,
				EndStep:   params.EndStep,
			})
			if err != nil {
				return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
			}
			if len(events) == 0 {
				return &tool.Result{Content: fmt.Sprintf("No events in session %s for the given range.", params.SessionID)}, nil
			}

			var b strings.Builder
			fmt.Fprintf(&b, "Events in session %s", params.SessionID)
			if params.StartStep != nil || params.EndStep != nil {
				fmt.Fprintf(&b, " (steps")
				if params.StartStep != nil {
					fmt.Fprintf(&b, " %d", *params.StartStep)
				}
				fmt.Fprint(&b, "-")
				if params.EndStep != nil {
					fmt.Fprintf(&b, "%d", *params.EndStep)
				}
				fmt.Fprint(&b, ")")
			}
			fmt.Fprintln(&b, ":")

			for _, e := range events {
				content := e.Event.ToText()
				if len(content) > 200 {
					content = content[:200] + "..."
				}
				// Collapse newlines for compact display.
				content = strings.ReplaceAll(content, "\n", " ")
				fmt.Fprintf(&b, "  step=%d type=%-12s %s\n", e.Step, e.Event.EventType(), content)
			}
			return &tool.Result{Content: b.String()}, nil
		},
	}
}

// --- session_peek ---

type sessionPeekParams struct {
	SessionID  string `json:"session_id" jsonschema:"required" jsonschema_description:"Session ID to peek at"`
	LastNSteps int    `json:"last_n_steps,omitempty" jsonschema_description:"Number of recent steps to show (default 2, max 10)"`
	RunID      string `json:"run_id,omitempty" jsonschema_description:"Optional: a prior run's id to inspect instead of the current run."`
}

func SessionPeek(deps *Deps) *tool.Tool {
	return &tool.Tool{
		Name:        "session_peek",
		Description: "View the most recent events in a session (last N steps).",
		ParamType:   &sessionPeekParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[sessionPeekParams](args)
			if errResult != nil {
				return errResult, nil
			}

			runID := deps.run(params.RunID)
			sess, err := deps.Store.GetSession(ctx, runID, params.SessionID)
			if err != nil {
				return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
			}

			n := params.LastNSteps
			if n <= 0 {
				n = 2
			}
			if n > 10 {
				n = 10
			}

			startStep := sess.CurrentStep - n
			if startStep < 0 {
				startStep = 0
			}
			events, err := deps.Store.GetEvents(ctx, runID, params.SessionID, db.EventFilter{
				StartStep: &startStep,
			})
			if err != nil {
				return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
			}
			if len(events) == 0 {
				return &tool.Result{Content: fmt.Sprintf("No events in session %s.", params.SessionID)}, nil
			}

			var b strings.Builder
			fmt.Fprintf(&b, "Recent events in %s (steps %d-%d):\n", params.SessionID, startStep, sess.CurrentStep)
			for _, e := range events {
				content := e.Event.ToText()
				if len(content) > 500 {
					content = content[:500] + "..."
				}
				content = strings.ReplaceAll(content, "\n", " ")
				fmt.Fprintf(&b, "  step=%d type=%-12s %s\n", e.Step, e.Event.EventType(), content)
			}
			if b.Len() > 32000 {
				return &tool.Result{Content: b.String()[:32000] + "\n[...truncated...]"}, nil
			}
			return &tool.Result{Content: b.String()}, nil
		},
	}
}

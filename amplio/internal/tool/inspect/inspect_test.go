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

package inspect

import (
	"context"
	"strings"
	"testing"
	"time"

	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/event"
)

func seedSession(t *testing.T, s db.Store) (string, string) {
	t.Helper()
	ctx := context.Background()
	runID := db.NewRunID()
	must(t, s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()}))
	must(t, s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()}))

	// step 0 bootstrap: system prompt + task.
	must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "SYS", Marker: event.MarkerSystemPrompt})))
	must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "TASK: build the thing"})))
	// steps 1..5.
	for i, content := range []string{"a1-explore", "a2-explore", "a3-explore", "a4 SECRET_PATH=/data/foo", ""} {
		_, err := s.AdvanceStep(ctx, runID, "s1")
		must(t, err)
		if i == 4 {
			must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "raw tail message"})))
			continue
		}
		must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: content})))
	}

	// Observations: a closed phase over steps 1-3, a step summary for step 4.
	mkStep := func(n int) *int { return &n }
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "phase-s1-3", RunID: runID, Kind: "phase_summary", SessionID: "s1", Step: mkStep(3),
		Data:      map[string]any{"title": "Exploration", "summary": "explored the codebase", "start_step": 1, "end_step": 3},
		CreatedAt: time.Now().UTC(),
	}))
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "step-s1-4", RunID: runID, Kind: "step_summary", SessionID: "s1", Step: mkStep(4),
		Data:      map[string]any{"summary": "ran experiment a4"},
		CreatedAt: time.Now().UTC(),
	}))
	must(t, s.SetLastSummarizedStep(ctx, runID, "s1", 4))
	must(t, s.SetLastPhasedStep(ctx, runID, "s1", 3))
	return runID, "s1"
}

func TestRenderSessionTree(t *testing.T) {
	now := time.Now()
	mk := func(id, parent, status, task string, statusAge time.Duration) db.SessionRecord {
		return db.SessionRecord{
			SessionID: id, ParentID: parent, Status: status, AgentType: "standard_agent",
			Task: task, StatusChangedAt: now.Add(-statusAge),
		}
	}
	// Tree: root(ongoing) -> midA(concluded long ago) -> leaf(ongoing)
	//                     -> crashedKid(crashed long ago, leaf)
	// created_at order is the slice order (ListSessions guarantees it).
	sessions := []db.SessionRecord{
		mk("root", "", db.SessionOngoing, "root task", 0),
		mk("midA", "root", db.SessionConcluded, "midA task here", 3*time.Hour),
		mk("leaf", "midA", db.SessionOngoing, "leaf task here", 0),
		mk("crashedKid", "root", db.SessionCrashed, "crashed task here", 3*time.Hour),
	}

	out := renderSessionTree("run1", sessions, now)

	// EVERY session is shown — including the long-finished + crashed ones (the
	// agent must never lose track of a spawned child). No hidden footer.
	for _, want := range []string{"root", "midA", "leaf", "crashedKid"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing session %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "hidden") {
		t.Errorf("nothing should be hidden now:\n%s", out)
	}

	// Inactive sessions are condensed: tagged [inactive] and their task omitted;
	// active sessions keep their task.
	if !strings.Contains(out, "[inactive]") {
		t.Errorf("long-finished sessions should be tagged [inactive]:\n%s", out)
	}
	if strings.Contains(out, "midA task here") || strings.Contains(out, "crashed task here") {
		t.Errorf("inactive session tasks should be omitted:\n%s", out)
	}
	if !strings.Contains(out, "leaf task here") || !strings.Contains(out, "root task") {
		t.Errorf("active session tasks should be shown:\n%s", out)
	}

	// Tree order: leaf (under midA) renders after midA; crashedKid after that.
	if strings.Index(out, "leaf") < strings.Index(out, "midA") {
		t.Errorf("leaf should render under midA:\n%s", out)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// appended discards the event id AppendEvent now returns, leaving just the
// error so it composes with must(t, ...) in the common test case.
func appended(_ string, err error) error { return err }

func openMem(t *testing.T) db.Store {
	t.Helper()
	s, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestLayeredSummary_Layers(t *testing.T) {
	s := openMem(t)
	runID, sid := seedSession(t, s)

	out, err := LayeredSummary(context.Background(), s, runID, sid, -1)
	if err != nil {
		t.Fatal(err)
	}

	// Task, the closed phase summary, the step-4 summary (as the open-phase
	// header), and the raw tail are all present.
	for _, want := range []string{"TASK: build the thing", "explored the codebase", "ran experiment a4", "raw tail message"} {
		if !strings.Contains(out, want) {
			t.Errorf("layered summary missing %q\n---\n%s", want, out)
		}
	}
	// Steps folded into a CLOSED phase (1-3) are represented only by the phase
	// summary — their raw assistant turns must not leak.
	if strings.Contains(out, "a2-explore") {
		t.Errorf("layered summary leaked raw content of a closed phase\n---\n%s", out)
	}
	// Steps in the OPEN phase within the last N show their economical body even
	// when summarized: step 4's raw text is shown under its summary header.
	if !strings.Contains(out, "SECRET_PATH") {
		t.Errorf("recent summarized open-phase step should show its body\n---\n%s", out)
	}
}

func TestLayeredSummary_PriorCompactionAnchored(t *testing.T) {
	ctx := context.Background()
	s := openMem(t)
	runID, sid := seedSession(t, s)

	// Compact through step 3: steps 1-3 fold into the summary; steps 4-5 carry.
	if _, err := s.CompactContext(ctx, runID, sid, 3, "PRIOR MEMORY: I explored and found X"); err != nil {
		t.Fatal(err)
	}

	out, err := LayeredSummary(ctx, s, runID, sid, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "EARLIER CONTEXT") || !strings.Contains(out, "PRIOR MEMORY: I explored and found X") {
		t.Errorf("expected prior compaction anchored as EARLIER CONTEXT\n---\n%s", out)
	}
	// The pre-boundary phase (steps 1-3) is now folded into the prior summary and
	// must not be re-rendered.
	if strings.Contains(out, "explored the codebase") {
		t.Errorf("pre-boundary phase should be subsumed by prior summary\n---\n%s", out)
	}
	// Post-boundary tail survives.
	if !strings.Contains(out, "raw tail message") {
		t.Errorf("post-boundary tail missing\n---\n%s", out)
	}
}

func TestCrossRunInspection(t *testing.T) {
	s := openMem(t)
	runA, _ := seedSession(t, s)
	runB, _ := seedSession(t, s)
	deps := &Deps{Store: s, RunID: runA}
	ctx := context.Background()

	list := SessionList(deps)
	// Default → current run (A), not B.
	if r := list.ParseAndExecute(ctx, `{}`); !strings.Contains(r.Content, runA) || strings.Contains(r.Content, runB) {
		t.Errorf("default list should target runA, got:\n%s", r.Content)
	}
	// run_id override → run B.
	if r := list.ParseAndExecute(ctx, `{"run_id":"`+runB+`"}`); !strings.Contains(r.Content, runB) {
		t.Errorf("cross-run list should target runB, got:\n%s", r.Content)
	}

	// session_summary honors run_id (reads B's task).
	sum := SessionSummary(deps)
	if r := sum.ParseAndExecute(ctx, `{"session_id":"s1","run_id":"`+runB+`"}`); r.IsError || !strings.Contains(r.Content, "build the thing") {
		t.Errorf("cross-run summary failed: %+v", r)
	}
}

// mkStepPtr returns a pointer to n (observation step key helper).
func mkStepPtr(n int) *int { return &n }

// seedOpenPhaseSession builds a session with a closed phase over steps 1-3 and
// open activity in steps 4-8, exercising tool calls, an external message, and a
// mix of summarized / unsummarized steps.
func seedOpenPhaseSession(t *testing.T, s db.Store) (string, string) {
	t.Helper()
	ctx := context.Background()
	runID := db.NewRunID()
	must(t, s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()}))
	must(t, s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()}))

	must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "SYS", Marker: event.MarkerSystemPrompt})))
	must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "TASK: do work"})))

	adv := func() { _, err := s.AdvanceStep(ctx, runID, "s1"); must(t, err) }
	append2 := func(step int, e event.Event) {
		must(t, s.AppendEventAtStep(ctx, runID, "s1", step, e))
	}

	// steps 1-3: closed phase (plain assistant turns).
	for _, c := range []string{"phase-a1", "phase-a2", "phase-a3"} {
		adv()
		must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: c})))
	}
	// step 4: assistant with a bash tool call + its (omitted) result.
	adv()
	append2(4, &event.AssistantEvent{
		Content:   "exploring the tree",
		ToolCalls: []event.ToolCall{{ID: "t1", Name: "bash", Arguments: `{"command":"grep -n needle file.go"}`}},
	})
	append2(4, &event.ToolResultEvent{ToolCallID: "t1", Content: "RESULT_BODY_SHOULD_BE_OMITTED"})
	// step 5: assistant with a view_file tool call.
	adv()
	append2(5, &event.AssistantEvent{
		ToolCalls: []event.ToolCall{{ID: "t2", Name: "view_file", Arguments: `{"path":"internal/foo/bar.go"}`}},
	})
	// step 6: an inbound environment message (external input).
	adv()
	append2(6, &event.MessageEvent{Content: "job done xid=123", Sender: "watcher", SenderType: event.SenderTypeEnvironment})
	// steps 7-8: plain assistant turns.
	for _, c := range []string{"more-work-7", "final-8"} {
		adv()
		append2(stepOf(t, s, runID), &event.AssistantEvent{Content: c})
	}

	// Closed phase over steps 1-3.
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "ph-1-3", RunID: runID, Kind: "phase_summary", SessionID: "s1", Step: mkStepPtr(3),
		Data:      map[string]any{"title": "Setup", "summary": "closed phase work", "start_step": 1, "end_step": 3},
		CreatedAt: time.Now().UTC(),
	}))
	// Step summaries for steps 4 and 5 (6-8 intentionally unsummarized).
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "st-4", RunID: runID, Kind: "step_summary", SessionID: "s1", Step: mkStepPtr(4),
		Data:      map[string]any{"summary": "searched for needle"},
		CreatedAt: time.Now().UTC(),
	}))
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "st-5", RunID: runID, Kind: "step_summary", SessionID: "s1", Step: mkStepPtr(5),
		Data:      map[string]any{"summary": "opened bar.go"},
		CreatedAt: time.Now().UTC(),
	}))
	must(t, s.SetLastSummarizedStep(ctx, runID, "s1", 5))
	must(t, s.SetLastPhasedStep(ctx, runID, "s1", 3))
	return runID, "s1"
}

// stepOf returns the session's current step (for appending at the just-advanced step).
func stepOf(t *testing.T, s db.Store, runID string) int {
	t.Helper()
	sess, err := s.GetSession(context.Background(), runID, "s1")
	must(t, err)
	return sess.CurrentStep
}

func TestFormatOpenPhase_DisplayModes(t *testing.T) {
	s := openMem(t)
	runID, sid := seedOpenPhaseSession(t, s)

	// tailN=2: only steps 7,8 are "recent". Step 4 & 5 are summarized and NOT
	// recent → header only; step 6 has an external input → body forced.
	block, openLo, err := FormatOpenPhase(context.Background(), s, runID, sid, -1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if openLo != 3 {
		t.Errorf("openLo = %d, want 3 (last closed phase end)", openLo)
	}

	// Closed-phase steps are excluded entirely.
	if strings.Contains(block, "phase-a2") || strings.Contains(block, "step 3:") {
		t.Errorf("closed-phase content leaked into open phase\n---\n%s", block)
	}
	// Summarized headers present for all open steps.
	for _, want := range []string{"step 4: searched for needle", "step 5: opened bar.go", "step 6:", "step 7:", "step 8:"} {
		if !strings.Contains(block, want) {
			t.Errorf("missing header %q\n---\n%s", want, block)
		}
	}
	// Step 4 is summarized and not within the last 2 → header only, no body.
	if strings.Contains(block, "exploring the tree") {
		t.Errorf("non-recent summarized step should not show its body\n---\n%s", block)
	}
	// Tool RESULT bodies are always omitted.
	if strings.Contains(block, "RESULT_BODY_SHOULD_BE_OMITTED") {
		t.Errorf("tool result body must be omitted\n---\n%s", block)
	}
	// Step 6 carries an external input → body forced, message shown in full.
	if !strings.Contains(block, "job done xid=123") {
		t.Errorf("step with external input should show its body\n---\n%s", block)
	}
	// Recent step 7/8 (within tailN) show their assistant body.
	if !strings.Contains(block, "more-work-7") || !strings.Contains(block, "final-8") {
		t.Errorf("recent steps should show their body\n---\n%s", block)
	}
}

func TestFormatOpenPhase_ToolCallBriefs(t *testing.T) {
	s := openMem(t)
	runID, sid := seedOpenPhaseSession(t, s)

	// Large tailN so step 4's body (with the bash tool call) renders.
	block, _, err := FormatOpenPhase(context.Background(), s, runID, sid, -1, 100)
	if err != nil {
		t.Fatal(err)
	}
	// bash → verb/target brief; view_file → "view_file bar.go".
	if !strings.Contains(block, "· search needle") {
		t.Errorf("bash tool call should render as a verb/target brief\n---\n%s", block)
	}
	if !strings.Contains(block, "· view_file bar.go") {
		t.Errorf("view_file tool call should render as a one-line brief\n---\n%s", block)
	}
	// The raw JSON arguments must not leak.
	if strings.Contains(block, "\"command\"") || strings.Contains(block, "internal/foo/bar.go") {
		t.Errorf("raw tool-call arguments leaked\n---\n%s", block)
	}
}

func TestFormatOpenPhase_ForceReopenWhenAllPhased(t *testing.T) {
	ctx := context.Background()
	s := openMem(t)
	runID, sid := seedOpenPhaseSession(t, s)

	// Close a SECOND phase over steps 4-8, so everything is phased.
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "ph-4-8", RunID: runID, Kind: "phase_summary", SessionID: "s1", Step: mkStepPtr(8),
		Data:      map[string]any{"title": "Investigation", "summary": "second phase", "start_step": 4, "end_step": 8},
		CreatedAt: time.Now().UTC(),
	}))
	must(t, s.SetLastPhasedStep(ctx, runID, sid, 8))

	// With all steps phased, the LAST phase (4-8) is force-reopened.
	block, openLo, err := FormatOpenPhase(ctx, s, runID, sid, -1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if openLo != 3 {
		t.Errorf("openLo = %d, want 3 (reopen last phase 4-8)", openLo)
	}
	if !strings.Contains(block, "step 4:") || !strings.Contains(block, "step 8:") {
		t.Errorf("reopened phase should render its steps 4-8\n---\n%s", block)
	}
}

// A failed-summarization sentinel is not shown as the header; the step is
// expanded to its raw body instead.
func TestFormatOpenPhase_FailedSummarizationExpands(t *testing.T) {
	ctx := context.Background()
	s := openMem(t)
	runID, sid := seedOpenPhaseSession(t, s)

	// Overwrite step 4's summary with the failure sentinel.
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "st-4-fail", RunID: runID, Kind: "step_summary", SessionID: sid, Step: mkStepPtr(4),
		Data:      map[string]any{"summary": db.SummarizationFailedPrefix + " LLM call failed: boom"},
		CreatedAt: time.Now().UTC(),
	}))

	// tailN=1 so step 4 is neither recent nor input-bearing: only the failure
	// marker could force expansion.
	block, _, err := FormatOpenPhase(ctx, s, runID, sid, -1, 1)
	if err != nil {
		t.Fatal(err)
	}
	// The failure sentinel must NOT appear as a header.
	if strings.Contains(block, db.SummarizationFailedPrefix) {
		t.Errorf("failed-summarization sentinel should not be shown\n---\n%s", block)
	}
	// Instead, step 4's raw body is expanded.
	if !strings.Contains(block, "exploring the tree") || !strings.Contains(block, "· search needle") {
		t.Errorf("failed-summarization step should be expanded to its body\n---\n%s", block)
	}
}

// A no-tool-call assistant reply (operator-facing summary) is always expanded,
// even when summarized and not recent.
func TestFormatOpenPhase_NoToolAssistantExpands(t *testing.T) {
	ctx := context.Background()
	s := openMem(t)
	runID := db.NewRunID()
	must(t, s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()}))
	must(t, s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()}))
	must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "TASK: x"})))
	// step 1: a no-tool-call assistant reply (a summary to the operator).
	_, err := s.AdvanceStep(ctx, runID, "s1")
	must(t, err)
	must(t, appended(s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "Here is my summary of the work: I did X and Y."})))
	// steps 2-3: plain tool-using turns to push step 1 out of the recent window.
	for i := 0; i < 2; i++ {
		_, err := s.AdvanceStep(ctx, runID, "s1")
		must(t, err)
		must(t, s.AppendEventAtStep(ctx, runID, "s1", stepOf(t, s, runID), &event.AssistantEvent{
			ToolCalls: []event.ToolCall{{ID: "x", Name: "bash", Arguments: `{"command":"ls"}`}},
		}))
	}
	// Summarize step 1 (so it is NOT expanded by the no-summary rule).
	must(t, s.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "st-1", RunID: runID, Kind: "step_summary", SessionID: "s1", Step: mkStepPtr(1),
		Data:      map[string]any{"summary": "replied to operator"},
		CreatedAt: time.Now().UTC(),
	}))
	must(t, s.SetLastSummarizedStep(ctx, runID, "s1", 1))

	// tailN=1 so step 1 is not recent; only the no-tool-call rule forces it open.
	block, _, err := FormatOpenPhase(ctx, s, runID, "s1", -1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(block, "step 1: replied to operator") {
		t.Errorf("summarized step should keep its header\n---\n%s", block)
	}
	if !strings.Contains(block, "Here is my summary of the work") {
		t.Errorf("no-tool-call assistant reply should be expanded verbatim\n---\n%s", block)
	}
}

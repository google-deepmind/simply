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

package sqlite

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/event"
)

func openTestStore(t *testing.T) db.Store {
	t.Helper()
	store, err := Open(":memory:")
	if err != nil {
		t.Fatalf("open test store: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}

func TestCreateAndGetRun(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	run := db.RunRecord{
		RunID:     db.NewRunID(),
		Config:    config.RunConfig{Task: "do something", LLM: "vertex:claude-opus-4-7"},
		CreatedAt: time.Now().UTC(),
	}
	if err := s.CreateRun(ctx, run); err != nil {
		t.Fatal(err)
	}

	// Re-creating the same id is a UNIQUE violation the manager retries on;
	// verify the real driver error is classified by db.IsUniqueViolation.
	if err := s.CreateRun(ctx, run); !db.IsUniqueViolation(err) {
		t.Fatalf("duplicate CreateRun error = %v; want a UNIQUE violation", err)
	}

	got, err := s.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if got.RunID != run.RunID {
		t.Errorf("RunID: got %q, want %q", got.RunID, run.RunID)
	}
	if got.Config.Task != "do something" {
		t.Errorf("Config.Task: got %q", got.Config.Task)
	}
	if got.Config.LLM != "vertex:claude-opus-4-7" {
		t.Errorf("Config.LLM: got %q", got.Config.LLM)
	}
}

func TestListRuns(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	for i := range 5 {
		_ = s.CreateRun(ctx, db.RunRecord{
			RunID:     db.NewRunID(),
			CreatedAt: time.Now().UTC().Add(time.Duration(i) * time.Second),
		})
	}

	runs, hasMore, err := s.ListRuns(ctx, db.ListRunsOpts{Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 3 {
		t.Errorf("got %d runs, want 3", len(runs))
	}
	if !hasMore {
		t.Error("expected hasMore=true")
	}

	// Keyset pagination: continue after the last run of page 1; expect the
	// remaining 2.
	last := runs[len(runs)-1]
	runs2, hasMore2, err := s.ListRuns(ctx, db.ListRunsOpts{
		Limit:       10,
		Before:      last.CreatedAt,
		BeforeRunID: last.RunID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(runs2) != 2 {
		t.Errorf("offset page: got %d runs, want 2", len(runs2))
	}
	if hasMore2 {
		t.Error("expected hasMore=false for final page")
	}
	// No overlap between page 1 (first 3) and the offset=3 page.
	seen := map[string]bool{runs[0].RunID: true, runs[1].RunID: true, runs[2].RunID: true}
	for _, r := range runs2 {
		if seen[r.RunID] {
			t.Errorf("offset page re-returned run %s from page 1", r.RunID)
		}
	}
}

// Runs list newest-first regardless of the starred flag (starring is a UI
// filter, not a sort key); keyset pagination walks that single global order
// without duplicates or skips.
func TestListRunsOrderAndPagination(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	base := time.Now().UTC()
	// 5 runs, oldest→newest by created_at. Star the OLDEST two (r0, r1) to prove
	// starring does NOT reorder them.
	ids := make([]string, 5)
	for i := range 5 {
		ids[i] = db.NewRunID()
		_ = s.CreateRun(ctx, db.RunRecord{
			RunID:     ids[i],
			Starred:   i < 2,
			CreatedAt: base.Add(time.Duration(i) * time.Second),
		})
	}

	all, _, err := s.ListRuns(ctx, db.ListRunsOpts{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 5 {
		t.Fatalf("got %d runs, want 5", len(all))
	}
	// Pure newest-first: r4, r3, r2, r1, r0 — the starred oldest two stay last.
	want := []string{ids[4], ids[3], ids[2], ids[1], ids[0]}
	for i, id := range want {
		if all[i].RunID != id {
			t.Errorf("pos %d = %s, want %s (order: %v)", i, all[i].RunID, id, want)
		}
	}

	// Paginate the same ordering with the keyset cursor: each page continues after
	// the previous page's last run. Must equal `all` with no dupes/skips.
	var paged []string
	var before time.Time
	var beforeID string
	for range 10 { // bound the loop; 5 runs / 2 per page = 3 pages
		p, more, err := s.ListRuns(ctx, db.ListRunsOpts{Limit: 2, Before: before, BeforeRunID: beforeID})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range p {
			paged = append(paged, r.RunID)
		}
		if !more || len(p) == 0 {
			break
		}
		before, beforeID = p[len(p)-1].CreatedAt, p[len(p)-1].RunID
	}
	if len(paged) != 5 {
		t.Fatalf("paged total = %d, want 5", len(paged))
	}
	for i, id := range want {
		if paged[i] != id {
			t.Errorf("paged pos %d = %s, want %s", i, paged[i], id)
		}
	}
}

// Keyset pagination is gap/dup-free even when runs share the exact same
// created_at (down to the millisecond) — the run_id tiebreaker makes the ordering
// total. This is the case OFFSET pagination handled fine but a created_at-only
// keyset would skip/duplicate at a page boundary landing on the tie.
func TestListRuns_KeysetTieBreak(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	// 5 runs with an IDENTICAL created_at → every row ties on the sort key.
	ts := time.Now().UTC()
	for range 5 {
		if err := s.CreateRun(ctx, db.RunRecord{RunID: db.NewRunID(), CreatedAt: ts}); err != nil {
			t.Fatal(err)
		}
	}

	// Page through 2 at a time; each run must appear exactly once.
	seen := map[string]int{}
	var before time.Time
	var beforeID string
	for range 10 { // bound the loop
		p, more, err := s.ListRuns(ctx, db.ListRunsOpts{Limit: 2, Before: before, BeforeRunID: beforeID})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range p {
			seen[r.RunID]++
		}
		if !more || len(p) == 0 {
			break
		}
		before, beforeID = p[len(p)-1].CreatedAt, p[len(p)-1].RunID
	}
	if len(seen) != 5 {
		t.Fatalf("saw %d distinct runs across pages, want 5 (seen=%v)", len(seen), seen)
	}
	for id, n := range seen {
		if n != 1 {
			t.Errorf("run %s returned %d times across pages, want exactly 1", id, n)
		}
	}
}

// ListRuns search (title / run_id / task / workspace), starred, and grade filters
// compose (AND) with each other and paginate over the true matching set.
func TestListRuns_SearchAndFilters(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	base := time.Now().UTC()

	mk := func(id string, i int, r db.RunRecord) {
		r.RunID = id
		r.CreatedAt = base.Add(time.Duration(i) * time.Second)
		if err := s.CreateRun(ctx, r); err != nil {
			t.Fatal(err)
		}
	}
	mk("run-alpha", 0, db.RunRecord{Title: "Refactor the parser", Config: config.RunConfig{Task: "rewrite optimizer", Workspace: "/home/me/proj-a"}, Starred: true, Grade: 5})
	mk("run-beta", 1, db.RunRecord{Title: "Fix the bug", Config: config.RunConfig{Task: "parser crash on empty", Workspace: "/home/me/proj-b"}, ReportGrade: 3})
	mk("run-gamma", 2, db.RunRecord{Title: "Docs", Config: config.RunConfig{Task: "update readme", Workspace: "/home/me/proj-a"}})

	ids := func(runs []db.RunRecord) []string {
		out := make([]string, len(runs))
		for i, r := range runs {
			out[i] = r.RunID
		}
		return out
	}
	search := func(opts db.ListRunsOpts) []string {
		opts.Limit = 100
		runs, _, err := s.ListRuns(ctx, opts)
		if err != nil {
			t.Fatal(err)
		}
		return ids(runs)
	}
	eq := func(name string, got, want []string) {
		if !slices.Equal(got, want) {
			t.Errorf("%s: got %v, want %v", name, got, want)
		}
	}

	// Title match.
	eq("title 'refactor'", search(db.ListRunsOpts{Search: "refactor"}), []string{"run-alpha"})
	// Task (in config_json) match — "parser" is in beta's task, alpha's title.
	// Newest-first: beta (i=1) before alpha (i=0).
	eq("'parser' title+task", search(db.ListRunsOpts{Search: "parser"}), []string{"run-beta", "run-alpha"})
	// run_id match.
	eq("run_id 'gamma'", search(db.ListRunsOpts{Search: "gamma"}), []string{"run-gamma"})
	// Workspace (in config_json) match — proj-a is alpha + gamma; newest-first.
	eq("workspace 'proj-a'", search(db.ListRunsOpts{Search: "proj-a"}), []string{"run-gamma", "run-alpha"})
	// Case-insensitive.
	eq("case-insensitive", search(db.ListRunsOpts{Search: "REFACTOR"}), []string{"run-alpha"})
	// Multi-term AND: both must match (title 'fix' + task 'parser' => beta only).
	eq("multi-term AND", search(db.ListRunsOpts{Search: "fix parser"}), []string{"run-beta"})
	// No match.
	eq("no match", search(db.ListRunsOpts{Search: "nonexistent"}), []string{})

	// Starred only.
	eq("starred", search(db.ListRunsOpts{StarredOnly: true}), []string{"run-alpha"})
	// Grade: effective grade = human else report. alpha=5, beta=3(report), gamma=0.
	eq("grade 5", search(db.ListRunsOpts{GradeFilter: 5}), []string{"run-alpha"})
	eq("grade 3 (report)", search(db.ListRunsOpts{GradeFilter: 3}), []string{"run-beta"})
	eq("ungraded", search(db.ListRunsOpts{GradeFilter: db.GradeUngraded}), []string{"run-gamma"})

	// Compose: workspace 'proj-a' AND starred => alpha only (gamma is unstarred).
	eq("search + starred", search(db.ListRunsOpts{Search: "proj-a", StarredOnly: true}), []string{"run-alpha"})

	// A '%' in the query is a literal, not a wildcard (escapeLike).
	eq("literal percent", search(db.ListRunsOpts{Search: "100%"}), []string{})
}

func TestSessionCRUD(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})

	sess := db.SessionRecord{
		RunID:     runID,
		SessionID: "swift-fox",
		AgentType: "standard_agent",
		Task:      "do stuff",
		Status:    db.SessionOngoing,
		Metadata:  map[string]any{"key": "value"},
		CreatedAt: time.Now().UTC(),
	}
	if err := s.CreateSession(ctx, sess); err != nil {
		t.Fatal(err)
	}

	got, err := s.GetSession(ctx, runID, "swift-fox")
	if err != nil {
		t.Fatal(err)
	}
	if got.AgentType != "standard_agent" {
		t.Errorf("AgentType: %q", got.AgentType)
	}
	if got.Metadata["key"] != "value" {
		t.Errorf("Metadata: %v", got.Metadata)
	}

	// MergeSessionMetadata.
	if err := s.MergeSessionMetadata(ctx, runID, "swift-fox", map[string]any{"new_key": 42}); err != nil {
		t.Fatal(err)
	}
	got, _ = s.GetSession(ctx, runID, "swift-fox")
	if got.Metadata["key"] != "value" {
		t.Error("original key lost after merge")
	}
	if got.Metadata["new_key"] != float64(42) { // JSON numbers are float64
		t.Errorf("new_key: %v", got.Metadata["new_key"])
	}

	// ListSessions.
	sessions, err := s.ListSessions(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if len(sessions) != 1 {
		t.Errorf("got %d sessions", len(sessions))
	}
}

func TestEventAppendAndRead(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})

	// Append a system event.
	if _, err := s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "hello", Marker: "bootstrap"}); err != nil {
		t.Fatal(err)
	}
	// Append an assistant event.
	if _, err := s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "response"}); err != nil {
		t.Fatal(err)
	}

	events, err := s.GetEvents(ctx, runID, "s1", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
	if events[0].Event.EventType() != "system" {
		t.Errorf("first event type: %q", events[0].Event.EventType())
	}
	// Verify column fields were injected.
	if sys, ok := events[0].Event.(*event.SystemEvent); ok {
		if sys.Marker != "bootstrap" {
			t.Errorf("Marker: %q", sys.Marker)
		}
	}

	// Event count.
	count, err := s.GetEventCount(ctx, runID, "s1", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Errorf("count: %d", count)
	}

	// Tail event.
	tail, err := s.GetTailEvent(ctx, runID, "s1")
	if err != nil {
		t.Fatal(err)
	}
	if tail == nil || tail.Event.EventType() != "assistant" {
		t.Error("tail should be the assistant event")
	}
}

func TestAdvanceStepAndCurrentContext(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})

	// Step 0: bootstrap events.
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "prompt"})

	// Advance to step 1.
	newStep, err := s.AdvanceStep(ctx, runID, "s1")
	if err != nil {
		t.Fatal(err)
	}
	if newStep != 1 {
		t.Errorf("new step: %d", newStep)
	}

	// Step 1 events.
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "reply"})

	// All events = 2.
	all, _ := s.GetEvents(ctx, runID, "s1", db.EventFilter{})
	if len(all) != 2 {
		t.Errorf("all events: %d", len(all))
	}
}

func TestCountEnvNotices(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})

	env := func(body string) *event.MessageEvent {
		return &event.MessageEvent{Content: body, Sender: "environment (pid=1)", SenderType: event.SenderTypeEnvironment}
	}
	// Two env notices at the current step, plus three events that must NOT count:
	// an agent-to-agent message (same event type, different sender), a user event,
	// and an assistant turn. The cap is a budget for the environment alone.
	_, _ = s.AppendEvent(ctx, runID, "s1", env("job done"))
	_, _ = s.AppendEvent(ctx, runID, "s1", env("other job done"))
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.MessageEvent{Content: "hi", Sender: "sib", SenderType: event.SenderTypeAgent})
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "carry on"})
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "ok"})

	n, err := s.CountEnvNotices(ctx, runID, "s1")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("count = %d, want 2 (only environment messages at the current step)", n)
	}

	// The budget is per STEP: taking the next turn resets it, which is what makes
	// the same cap a per-turn allowance on a working session and a lifetime cap on
	// a finished one (whose step never advances again).
	if _, err := s.AdvanceStep(ctx, runID, "s1"); err != nil {
		t.Fatal(err)
	}
	if n, err = s.CountEnvNotices(ctx, runID, "s1"); err != nil || n != 0 {
		t.Errorf("after AdvanceStep: count = %d, err = %v; want 0", n, err)
	}

	// An unknown session is 0, not an error: the subselect yields NULL, and the
	// caller's append reports the missing session on its own terms.
	if n, err = s.CountEnvNotices(ctx, runID, "nope"); err != nil || n != 0 {
		t.Errorf("unknown session: count = %d, err = %v; want 0, nil", n, err)
	}
}

func TestCompactContext(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})

	// Gen 0 events: step 0 bootstrap, step 1 a completed turn, step 2 a fresh
	// input that arrived after the call boundary (no response yet).
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "prompt"})
	_, _ = s.AdvanceStep(ctx, runID, "s1")
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "old reply"})
	_, _ = s.AdvanceStep(ctx, runID, "s1")
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "fresh input"})

	// Compact everything through step 1; the step-2 fresh input is kept verbatim.
	newGen, err := s.CompactContext(ctx, runID, "s1", 1, "summary of old context")
	if err != nil {
		t.Fatal(err)
	}
	if newGen != 1 {
		t.Errorf("new gen: %d", newGen)
	}

	// Current context (step 0 OR generation 1), ordered by step:
	//   step 0 prompt        (gen 0, kept via step=0 clause)
	//   step 1 compaction    (gen 1, the summary at the boundary)
	//   step 2 fresh input   (carried gen 0 -> gen 1)
	// The step-1 "old reply" (gen 0) is dropped.
	current, _ := s.GetEvents(ctx, runID, "s1", db.EventFilter{CurrentContextOnly: true})
	gotTypes := make([]string, len(current))
	for i, e := range current {
		gotTypes[i] = fmt.Sprintf("step%d/%s", e.Step, e.Event.EventType())
	}
	want := []string{"step0/system", "step1/compaction", "step2/user"}
	if !reflect.DeepEqual(gotTypes, want) {
		t.Errorf("current context = %v, want %v", gotTypes, want)
	}

	// The carried fresh input must be at the new generation and unchanged.
	last := current[len(current)-1]
	if last.Generation != newGen {
		t.Errorf("carried event generation = %d, want %d", last.Generation, newGen)
	}
	if ue, ok := last.Event.(*event.UserEvent); !ok || ue.Content != "fresh input" {
		t.Errorf("carried event = %+v, want UserEvent{fresh input}", last.Event)
	}
}

func TestSearchEvents(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})

	_, _ = s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "initialize the quantum flux capacitor"})
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "analyzing the data pipeline"})

	results, err := s.SearchEvents(ctx, runID, "quantum", db.SearchOpts{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Errorf("search results: %d, want 1", len(results))
	}
}

func TestRunAnnotationFields(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	if err := s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}

	// Fresh run: empty overlay, updated_at seeded to created_at.
	run, err := s.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if run.Title != "" || run.Note != "" || run.Starred || run.Archived ||
		run.Grade != 0 || run.ReportGrade != 0 {
		t.Errorf("fresh overlay not empty: %+v", run)
	}
	if run.UpdatedAt.IsZero() {
		t.Error("updated_at should be seeded on create")
	}

	if err := s.UpdateRunTitle(ctx, runID, "my title"); err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateRunNote(ctx, runID, "a note"); err != nil {
		t.Fatal(err)
	}
	if err := s.SetRunStarred(ctx, runID, true); err != nil {
		t.Fatal(err)
	}
	if err := s.SetRunGrade(ctx, runID, 4); err != nil {
		t.Fatal(err)
	}
	if err := s.SetRunReportGrade(ctx, runID, 2); err != nil {
		t.Fatal(err)
	}
	if err := s.SetRunArchived(ctx, runID, true); err != nil {
		t.Fatal(err)
	}

	run, _ = s.GetRun(ctx, runID)
	if run.Title != "my title" || run.Note != "a note" || !run.Starred || !run.Archived {
		t.Errorf("overlay after updates: %+v", run)
	}
	// Human grade overrides the cached report grade, but both round-trip.
	if run.Grade != 4 || run.ReportGrade != 2 {
		t.Errorf("grade fields after updates: grade=%d report_grade=%d", run.Grade, run.ReportGrade)
	}

	// Archived runs are hidden by default, shown with ShowArchived.
	hidden, _, err := s.ListRuns(ctx, db.ListRunsOpts{})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range hidden {
		if r.RunID == runID {
			t.Error("archived run should be excluded by default")
		}
	}
	shown, _, err := s.ListRuns(ctx, db.ListRunsOpts{ShowArchived: true})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range shown {
		if r.RunID == runID {
			found = true
		}
	}
	if !found {
		t.Error("archived run should appear with ShowArchived=true")
	}

	// Toggle archive off → reappears in the default listing.
	if err := s.SetRunArchived(ctx, runID, false); err != nil {
		t.Fatal(err)
	}
	def, _, _ := s.ListRuns(ctx, db.ListRunsOpts{})
	found = false
	for _, r := range def {
		if r.RunID == runID {
			found = true
		}
	}
	if !found {
		t.Error("un-archived run should reappear in default listing")
	}
}

func TestLessonCRUD(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	lesson := db.LessonRecord{
		LessonID:    "lesson-1",
		Title:       "How to debug",
		Description: "Debugging tips",
		Body:        "Use print statements",
		Embedding:   []float32{0.1, 0.2, 0.3},
		EmbedderID:  "text-embedding-005",
		SourceRunID: "run-abc",
	}
	if err := s.InsertLesson(ctx, lesson); err != nil {
		t.Fatal(err)
	}

	got, err := s.GetLesson(ctx, "lesson-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "How to debug" {
		t.Errorf("Title: %q", got.Title)
	}
	if len(got.Embedding) != 3 {
		t.Fatalf("Embedding len: %d", len(got.Embedding))
	}
	// Float comparison with tolerance.
	if got.Embedding[0] < 0.09 || got.Embedding[0] > 0.11 {
		t.Errorf("Embedding[0]: %f", got.Embedding[0])
	}

	// Increment counters.
	_ = s.IncrementLessonLoadCount(ctx, "lesson-1", 3)
	_ = s.AddToLessonScore(ctx, "lesson-1", 10)
	got, _ = s.GetLesson(ctx, "lesson-1")
	if got.LoadedCount != 3 {
		t.Errorf("LoadedCount: %d", got.LoadedCount)
	}
	if got.Score != 10 {
		t.Errorf("Score: %d", got.Score)
	}

	// created_at / updated_at populated on insert.
	if got.CreatedAt.IsZero() || got.UpdatedAt.IsZero() {
		t.Errorf("timestamps not set: created=%v updated=%v", got.CreatedAt, got.UpdatedAt)
	}

	// UpdateLesson replaces content, preserves counters + id, bumps updated_at.
	if err := s.UpdateLesson(ctx, db.LessonRecord{
		LessonID:    "lesson-1",
		Title:       "How to debug v2",
		Description: "Better debugging tips",
		Body:        "Use a debugger",
		Embedding:   []float32{0.4, 0.5},
		EmbedderID:  "text-embedding-005",
	}); err != nil {
		t.Fatal(err)
	}
	upd, err := s.GetLesson(ctx, "lesson-1")
	if err != nil {
		t.Fatal(err)
	}
	if upd.Title != "How to debug v2" || upd.Body != "Use a debugger" || len(upd.Embedding) != 2 {
		t.Errorf("content not replaced: %+v", upd)
	}
	if upd.LoadedCount != 3 || upd.Score != 10 {
		t.Errorf("counters not preserved: loaded=%d score=%d", upd.LoadedCount, upd.Score)
	}
	if !upd.CreatedAt.Equal(got.CreatedAt) {
		t.Errorf("created_at changed: %v -> %v", got.CreatedAt, upd.CreatedAt)
	}
	if upd.UpdatedAt.Before(got.UpdatedAt) {
		t.Errorf("updated_at not advanced: %v -> %v", got.UpdatedAt, upd.UpdatedAt)
	}

	// ListAll.
	all, err := s.ListAllLessons(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 1 {
		t.Errorf("ListAll: %d", len(all))
	}
}

// On a migrated DB the updated_at column was added via ALTER (no default), so a
// row can carry NULL. The reads (GetLesson / ListAllLessons) must tolerate that
// rather than 500 on "converting NULL to string". Simulate by forcing NULL.
func TestLesson_NullUpdatedAtTolerated(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "lessons.db")

	// Pre-create a Lesson table in the OLD shape: updated_at added by ALTER, hence
	// nullable with no default — exactly what a migrated pre-updated_at DB has.
	// Open() then sees the table already exists (CREATE TABLE IF NOT EXISTS is a
	// no-op; migrate's ALTER is a duplicate-column no-op), so the column stays
	// nullable and an insert can leave it NULL — the bug's real precondition.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.ExecContext(ctx, `CREATE TABLE Lesson (
		lesson_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
		body TEXT NOT NULL DEFAULT '', embedding BLOB, embedder_id TEXT NOT NULL DEFAULT '',
		source_run_id TEXT, score INTEGER NOT NULL DEFAULT 0, loaded_count INTEGER NOT NULL DEFAULT 0,
		created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))`); err != nil {
		t.Fatal(err)
	}
	if _, err := raw.ExecContext(ctx, `ALTER TABLE Lesson ADD COLUMN updated_at TEXT`); err != nil {
		t.Fatal(err)
	}
	// Pre-seed a row with NULL updated_at, then drain so the backfill in migrate()
	// (which runs once in Open) doesn't mask the read-path test: re-NULL after Open.
	raw.Close()

	store, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })

	if err := store.InsertLesson(ctx, db.LessonRecord{
		LessonID: "l-null", Title: "T", Description: "d", Body: "b", EmbedderID: "e",
	}); err != nil {
		t.Fatal(err)
	}
	// Force NULL via a fresh raw connection to mimic a pre-backfill migrated row
	// (Insert now stamps updated_at, so we must clear it explicitly to test reads).
	raw2, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer raw2.Close()
	if _, err := raw2.ExecContext(ctx,
		`UPDATE Lesson SET updated_at = NULL WHERE lesson_id = ?`, "l-null"); err != nil {
		t.Fatal(err)
	}
	s := store

	if _, err := s.GetLesson(ctx, "l-null"); err != nil {
		t.Fatalf("GetLesson with NULL updated_at: %v", err)
	}
	all, err := s.ListAllLessons(ctx)
	if err != nil {
		t.Fatalf("ListAllLessons with NULL updated_at: %v", err)
	}
	if len(all) != 1 {
		t.Fatalf("ListAllLessons len = %d, want 1", len(all))
	}
}

func TestListRunsWithSessions(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	now := time.Now().UTC()
	runA := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runA, CreatedAt: now})
	// Root session (no parent).
	_ = s.CreateSession(ctx, db.SessionRecord{
		RunID: runA, SessionID: "main-agent", AgentType: "standard_agent",
		Status: db.SessionOngoing, CreatedAt: now,
	})
	// Child session (has parent — should NOT appear in root sessions).
	_ = s.CreateSession(ctx, db.SessionRecord{
		RunID: runA, SessionID: "swift-fox", AgentType: "standard_agent",
		Status: db.SessionOngoing, ParentID: "main-agent", CreatedAt: now,
	})

	// A second run, so the json_each batch fetch is exercised with >1 id.
	runB := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runB, CreatedAt: now.Add(time.Second)})
	_ = s.CreateSession(ctx, db.SessionRecord{
		RunID: runB, SessionID: "brave-owl", AgentType: "standard_agent",
		Status: db.SessionOngoing, CreatedAt: now.Add(time.Second),
	})

	results, _, err := s.ListRunsWithSessions(ctx, db.ListRunsOpts{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d runs, want 2", len(results))
	}

	roots := map[string][]db.SessionRecord{}
	for _, r := range results {
		roots[r.Run.RunID] = r.RootSessions
	}
	if got := roots[runA]; len(got) != 1 || got[0].SessionID != "main-agent" {
		t.Errorf("run A root sessions = %+v, want [main-agent]", got)
	}
	if got := roots[runB]; len(got) != 1 || got[0].SessionID != "brave-owl" {
		t.Errorf("run B root sessions = %+v, want [brave-owl]", got)
	}
}

func TestCreateRunStampsLastSeen(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	// Invariant: even with a ZERO CreatedAt, a new run gets a non-NULL
	// last_seen_at (defaulted to now), so it starts "seen" and last_seen_at is
	// NULL only for legacy/dismissed rows.
	if err := s.CreateRun(ctx, db.RunRecord{RunID: "r-zero"}); err != nil {
		t.Fatal(err)
	}
	got, err := s.GetRun(ctx, "r-zero")
	if err != nil {
		t.Fatal(err)
	}
	if got.LastSeenAt.IsZero() {
		t.Error("new run with zero CreatedAt must still have a non-NULL last_seen_at")
	}
	if got.CreatedAt.IsZero() {
		t.Error("new run with zero CreatedAt must still have a non-NULL created_at")
	}
}

func TestRunCountsAndStatusFilterAndMarkSeen(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()

	// Helper: a run created in the PAST with one non-chatbot root. The root starts
	// ongoing; if finalStatus differs, we transition it via UpdateSessionStatus,
	// which bumps status_changed_at to ~now (AFTER created_at) — the realistic
	// "agent finished after the run was created" shape the updates predicate keys
	// on. createdOffset is negative (minutes ago) to keep ordering deterministic.
	mk := func(id, finalStatus string, createdOffset time.Duration) {
		created := now.Add(createdOffset)
		if err := s.CreateRun(ctx, db.RunRecord{RunID: id, CreatedAt: created}); err != nil {
			t.Fatal(err)
		}
		if err := s.CreateSession(ctx, db.SessionRecord{
			RunID: id, SessionID: "main-agent", AgentType: "standard_agent",
			Status: db.SessionOngoing, CreatedAt: created,
		}); err != nil {
			t.Fatal(err)
		}
		if finalStatus != db.SessionOngoing {
			if err := s.UpdateSessionStatus(ctx, id, "main-agent", finalStatus); err != nil {
				t.Fatal(err)
			}
		}
	}
	// ongoing → active, no update (never transitioned). concluded/crashed → their
	// groups + update (terminal transition after create).
	mk("run-active", db.SessionOngoing, -1*time.Minute)
	mk("run-done", db.SessionConcluded, -2*time.Minute)
	mk("run-failed", db.SessionCrashed, -3*time.Minute)
	// An archived run that would otherwise be active — must be excluded everywhere.
	mk("run-arch", db.SessionOngoing, -4*time.Minute)
	if err := s.SetRunArchived(ctx, "run-arch", true); err != nil {
		t.Fatal(err)
	}

	// Counts: active = run-active (run-arch excluded); updates = the two terminal
	// runs whose root status changed after creation.
	c, err := s.RunCounts(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if c.Active != 1 {
		t.Errorf("active = %d, want 1 (archived excluded)", c.Active)
	}
	if c.Updates != 2 {
		t.Errorf("updates = %d, want 2 (concluded+crashed)", c.Updates)
	}

	// Status filter on the list.
	check := func(filter string, wantIDs ...string) {
		t.Helper()
		runs, _, err := s.ListRuns(ctx, db.ListRunsOpts{Limit: 50, StatusFilter: filter})
		if err != nil {
			t.Fatal(err)
		}
		got := map[string]bool{}
		for _, r := range runs {
			got[r.RunID] = true
		}
		if len(got) != len(wantIDs) {
			t.Errorf("filter %q: got %d runs %v, want %v", filter, len(got), keysOf(got), wantIDs)
			return
		}
		for _, id := range wantIDs {
			if !got[id] {
				t.Errorf("filter %q: missing %s (got %v)", filter, id, keysOf(got))
			}
		}
	}
	check(db.RunFilterActive, "run-active")
	check(db.RunFilterDone, "run-done")
	check(db.RunFilterFailed, "run-failed")
	check(db.RunFilterUpdates, "run-done", "run-failed")

	// MarkRunSeen clears run-done from updates (last_seen_at advances past its
	// status_changed_at), leaving only run-failed.
	if err := s.MarkRunSeen(ctx, "run-done"); err != nil {
		t.Fatal(err)
	}
	c2, err := s.RunCounts(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Updates != 1 {
		t.Errorf("updates after mark-seen = %d, want 1", c2.Updates)
	}
	check(db.RunFilterUpdates, "run-failed")

	// MarkRunUnseen puts it back: an operator who looked at a run but isn't done
	// with it finds it again the same way they found it the first time. It must
	// restore the badge, not merely fail to clear it — last_seen_at rewinds to
	// created_at, so a NULL (which reads as SEEN here) would do the opposite.
	if err := s.MarkRunUnseen(ctx, "run-done"); err != nil {
		t.Fatal(err)
	}
	c3, err := s.RunCounts(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if c3.Updates != 2 {
		t.Errorf("updates after mark-unseen = %d, want 2", c3.Updates)
	}
	check(db.RunFilterUpdates, "run-done", "run-failed")
}

// A legacy run with NULL last_seen_at (added by ALTER, never stamped) must not
// permanently suppress the has-updates badge. The migration backfills it to
// seen-as-of-now: its PAST terminal transition stays quiet (no flood of old
// runs), but a FUTURE transition still badges. Without the backfill the run
// deadlocks (NULL → never badges → client never markSeens → stays NULL).
func TestMigrate_BackfillsNullLastSeen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "seen.db")

	// Seed a run + concluded root with NULL last_seen_at via a raw connection
	// (the store API always stamps, so we forge the legacy shape directly). The
	// Open() below runs migrate(), which should backfill the NULL.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.ExecContext(ctx, schema); err != nil {
		t.Fatal(err)
	}
	past := "2020-01-01T00:00:00.000Z" // long before "now"
	if _, err := raw.ExecContext(ctx,
		`INSERT INTO Run (run_id, created_at, updated_at, last_seen_at) VALUES (?, ?, ?, NULL)`,
		"legacy", past, past); err != nil {
		t.Fatal(err)
	}
	if _, err := raw.ExecContext(ctx,
		`INSERT INTO Session (run_id, session_id, agent_type, status, created_at, status_changed_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		"legacy", "main-agent", "standard_agent", db.SessionConcluded, past, past); err != nil {
		t.Fatal(err)
	}
	raw.Close()

	store, err := Open(path) // runs migrate() → backfill
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })

	// The legacy run's OLD (pre-upgrade) terminal transition must NOT badge:
	// backfilling to now() keeps the historical backlog quiet.
	c, err := store.RunCounts(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if c.Updates != 0 {
		t.Fatalf("updates = %d, want 0 (legacy backlog must stay quiet after backfill)", c.Updates)
	}

	// But the row is no longer NULL — a FUTURE transition can badge. Timestamps
	// are millisecond-precision, so sleep a hair to ensure status_changed_at lands
	// strictly AFTER the backfilled last_seen_at (in production they're seconds+
	// apart; this only matters for the sub-ms test).
	time.Sleep(2 * time.Millisecond)
	if err := store.UpdateSessionStatus(ctx, "legacy", "main-agent", db.SessionCrashed); err != nil {
		t.Fatal(err)
	}
	c2, err := store.RunCounts(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Updates != 1 {
		t.Fatalf("updates after a post-backfill transition = %d, want 1 (NULL deadlock broken)", c2.Updates)
	}
}

func keysOf(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestCustomModels(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if specs, err := s.ListCustomModels(ctx); err != nil || len(specs) != 0 {
		t.Fatalf("empty list: %v, %v", specs, err)
	}
	for _, spec := range []string{"vertex:a", "vertex:b", "vertex:a"} { // last is dup
		if err := s.AddCustomModel(ctx, spec); err != nil {
			t.Fatal(err)
		}
	}
	specs, err := s.ListCustomModels(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 2 || specs[0] != "vertex:a" || specs[1] != "vertex:b" {
		t.Fatalf("list = %v, want [vertex:a vertex:b] (dedup, insert order)", specs)
	}
	if err := s.RemoveCustomModel(ctx, "vertex:a"); err != nil {
		t.Fatal(err)
	}
	specs, _ = s.ListCustomModels(ctx)
	if len(specs) != 1 || specs[0] != "vertex:b" {
		t.Fatalf("after remove = %v, want [vertex:b]", specs)
	}
}

func TestStatusChangedAt(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC(),
	})

	// Initial status_changed_at should be set (same as created_at).
	sess, _ := s.GetSession(ctx, runID, "s1")
	if sess.StatusChangedAt.IsZero() {
		t.Error("initial StatusChangedAt should be set")
	}
	initialTime := sess.StatusChangedAt

	// Update status — status_changed_at should advance.
	time.Sleep(10 * time.Millisecond)
	_ = s.UpdateSessionStatus(ctx, runID, "s1", db.SessionConcluded)
	sess, _ = s.GetSession(ctx, runID, "s1")
	if !sess.StatusChangedAt.After(initialTime) {
		t.Error("StatusChangedAt should have advanced after status change")
	}
}

func TestTerminateAndNotifyParent(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "parent", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "child", ParentID: "parent", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})

	if err := s.TerminateAndNotifyParent(ctx, runID, "child", "parent", db.SessionConcluded, "the result", nil, -1); err != nil {
		t.Fatal(err)
	}

	child, _ := s.GetSession(ctx, runID, "child")
	if child.Status != db.SessionConcluded {
		t.Errorf("child status: %q, want concluded", child.Status)
	}

	evts, _ := s.GetEvents(ctx, runID, "parent", db.EventFilter{})
	var found *event.ChildResultEvent
	for _, e := range evts {
		if cr, ok := e.Event.(*event.ChildResultEvent); ok {
			found = cr
		}
	}
	if found == nil {
		t.Fatal("parent missing ChildResultEvent")
	}
	if found.Verdict != db.SessionConcluded || found.Content != "the result" || found.ChildSessionID != "child" {
		t.Errorf("child result: %+v", found)
	}

	// Root (no parent): status flips, selfEvent lands on its own stream, no
	// child_result anywhere.
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "root", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})
	if err := s.TerminateAndNotifyParent(ctx, runID, "root", "", db.SessionCrashed, "boom",
		&event.SystemEvent{Content: "boom", Marker: event.MarkerError}, -1); err != nil {
		t.Fatal(err)
	}
	root, _ := s.GetSession(ctx, runID, "root")
	if root.Status != db.SessionCrashed {
		t.Errorf("root status: %q, want crashed", root.Status)
	}
	rootEvts, _ := s.GetEvents(ctx, runID, "root", db.EventFilter{})
	var sawErr bool
	for _, e := range rootEvts {
		if se, ok := e.Event.(*event.SystemEvent); ok && se.Marker == event.MarkerError {
			sawErr = true
		}
	}
	if !sawErr {
		t.Error("root missing self error marker")
	}

	// Dedupe: terminating an already-terminal session is a no-op — status stays,
	// and no second child_result is written to the parent.
	if err := s.TerminateAndNotifyParent(ctx, runID, "child", "parent", db.SessionCancelled, "again", nil, -1); err != nil {
		t.Fatal(err)
	}
	child, _ = s.GetSession(ctx, runID, "child")
	if child.Status != db.SessionConcluded {
		t.Errorf("dedupe: child status became %q, want concluded", child.Status)
	}
	var crCount int
	evts, _ = s.GetEvents(ctx, runID, "parent", db.EventFilter{})
	for _, e := range evts {
		if _, ok := e.Event.(*event.ChildResultEvent); ok {
			crCount++
		}
	}
	if crCount != 1 {
		t.Errorf("dedupe: parent has %d child_results, want 1", crCount)
	}
}

// TestTerminateSelfEventStep verifies selfEventStep pins the marker's step: a
// non-negative value overrides the session's current step (so a crash lands at
// the failed turn's call step, not the bumped current step).
func TestTerminateSelfEventStep(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "a", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})
	// Advance the session a couple of steps so current_step != the pinned step.
	_, _ = s.AdvanceStep(ctx, runID, "a") // -> 1
	_, _ = s.AdvanceStep(ctx, runID, "a") // -> 2 (current_step)

	const pinned = 1
	if err := s.TerminateAndNotifyParent(ctx, runID, "a", "", db.SessionCrashed, "boom",
		&event.SystemEvent{Content: "boom", Marker: event.MarkerError}, pinned); err != nil {
		t.Fatal(err)
	}

	evts, _ := s.GetEvents(ctx, runID, "a", db.EventFilter{})
	var step = -1
	for _, e := range evts {
		if se, ok := e.Event.(*event.SystemEvent); ok && se.Marker == event.MarkerError {
			step = e.Step
		}
	}
	if step != pinned {
		t.Errorf("error marker at step %d, want pinned %d", step, pinned)
	}
}

func TestStoreEvents(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	// Drain events in background.
	var received []db.StoreEvent
	done := make(chan struct{})
	go func() {
		defer close(done)
		for evt := range s.Events() {
			received = append(received, evt)
		}
	}()

	runID := db.NewRunID()
	_ = s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()})
	_ = s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()})
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "test"})
	_ = s.UpdateSessionStatus(ctx, runID, "s1", db.SessionConcluded)

	s.Close()
	<-done

	// Should have: SessionCreated, EventAppended, SessionStatusChanged.
	if len(received) != 3 {
		t.Errorf("received %d events, want 3", len(received))
		for _, e := range received {
			t.Logf("  %T", e)
		}
	}
}

// TestPutSkillVectors_PrunesStale verifies that PutSkillVectors replaces the
// full per-model set: a name present in an earlier Put but absent from a later
// Put is deleted (not left to resurrect via GetSkillVectors/LoadCached). See M28.
func TestPutSkillVectors_PrunesStale(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	const model = "test-embedder"

	first := []db.SkillVector{
		{Name: "alpha", ContentHash: "h1", Vector: []float32{1, 0}},
		{Name: "beta", ContentHash: "h2", Vector: []float32{0, 1}},
	}
	if err := s.PutSkillVectors(ctx, model, first); err != nil {
		t.Fatalf("first put: %v", err)
	}

	// Second scan no longer includes "beta" (removed from the source tree) and
	// adds "gamma"; "alpha" is unchanged.
	second := []db.SkillVector{
		{Name: "alpha", ContentHash: "h1", Vector: []float32{1, 0}},
		{Name: "gamma", ContentHash: "h3", Vector: []float32{1, 1}},
	}
	if err := s.PutSkillVectors(ctx, model, second); err != nil {
		t.Fatalf("second put: %v", err)
	}

	got, err := s.GetSkillVectors(ctx, model)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	names := map[string]bool{}
	for _, v := range got {
		names[v.Name] = true
	}
	if names["beta"] {
		t.Error("stale skill \"beta\" was not pruned")
	}
	if !names["alpha"] || !names["gamma"] {
		t.Errorf("expected alpha+gamma to remain, got %v", names)
	}
	if len(got) != 2 {
		t.Errorf("got %d vectors, want 2", len(got))
	}
}

// TestPutSkillVectors_EmptySet clears all rows for a model when given no vectors
// (e.g. all skills removed). Guards against the prune query mishandling an empty
// keep-set.
func TestPutSkillVectors_EmptySet(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	const model = "test-embedder"

	if err := s.PutSkillVectors(ctx, model, []db.SkillVector{
		{Name: "alpha", ContentHash: "h1", Vector: []float32{1, 0}},
	}); err != nil {
		t.Fatalf("put: %v", err)
	}
	if err := s.PutSkillVectors(ctx, model, nil); err != nil {
		t.Fatalf("put empty: %v", err)
	}
	got, err := s.GetSkillVectors(ctx, model)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d vectors, want 0 after empty put", len(got))
	}
}

// --- time / helper round-trips (N5) ---

func TestB2I(t *testing.T) {
	if b2i(true) != 1 {
		t.Errorf("b2i(true) = %d, want 1", b2i(true))
	}
	if b2i(false) != 0 {
		t.Errorf("b2i(false) = %d, want 0", b2i(false))
	}
}

func TestFormatParseTimeRoundTrip(t *testing.T) {
	// formatTime -> parseTime is lossless at millisecond resolution (the stored
	// format's precision). Sub-millisecond input is truncated by the format, so
	// compare at ms granularity.
	in := time.Date(2026, 6, 23, 12, 34, 56, 789_000_000, time.UTC)
	got := parseTime(formatTime(in))
	if !got.Equal(in) {
		t.Errorf("round-trip = %v, want %v", got, in)
	}
	// Zero time maps to "" and back to zero.
	if formatTime(time.Time{}) != "" {
		t.Errorf("formatTime(zero) = %q, want empty", formatTime(time.Time{}))
	}
	if !parseTime("").IsZero() {
		t.Error("parseTime(\"\") should be zero")
	}
	// A malformed timestamp degrades to zero (logged), never panics.
	if !parseTime("not-a-time").IsZero() {
		t.Error("parseTime(garbage) should be zero")
	}
}

// N4: an event appended WITHOUT an explicit created_at gets its timestamp from
// the schema's strftime DEFAULT. Verify that DEFAULT format parses back through
// parseTime (i.e. the Go timeFormat layout matches strftime('%Y-%m-%dT%H:%M:%fZ')),
// since every event read depends on this round-trip.
func TestSchemaDefaultTimestampParses(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()
	runID := db.NewRunID()
	if err := s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing}); err != nil {
		t.Fatal(err)
	}
	before := time.Now().Add(-2 * time.Second)
	if _, err := s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "hi"}); err != nil {
		t.Fatal(err)
	}

	events, err := s.GetEvents(ctx, runID, "s1", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	ts := events[0].CreatedAt
	if ts.IsZero() {
		t.Fatal("schema-DEFAULT created_at parsed to zero — strftime format mismatch with timeFormat")
	}
	if ts.Before(before) || ts.After(time.Now().Add(2*time.Second)) {
		t.Errorf("created_at %v not within the expected window (DEFAULT not 'now'?)", ts)
	}
}

// TestMigrateDropsLegacyUserColumn simulates an OLD DB whose Run table still has
// the vestigial `user NOT NULL` column, then runs migrate() and verifies the
// column is dropped and a normal insert (which no longer lists `user`) works.
// This is the upgrade path that a naive schema-only removal would break.
func TestMigrateDropsLegacyUserColumn(t *testing.T) {
	raw, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()      //nolint:errcheck
	raw.SetMaxOpenConns(1) // pin the in-memory DB to one connection

	// Old-style Run table: includes `user NOT NULL` (no default), like a DB
	// created before the column was removed. Minimal columns + a legacy row.
	if _, err := raw.Exec(`CREATE TABLE Run (
		run_id TEXT PRIMARY KEY,
		user TEXT NOT NULL,
		config_json TEXT,
		created_at TEXT
	)`); err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(`INSERT INTO Run (run_id, user, config_json, created_at) VALUES ('old1', 'alice', '{}', '2026-01-01T00:00:00.000Z')`); err != nil {
		t.Fatal(err)
	}

	// Applying the current schema is a no-op for the existing table (CREATE IF
	// NOT EXISTS); migrate() must bring it up to date — including dropping user.
	if _, err := raw.Exec(schema); err != nil {
		t.Fatal(err)
	}
	if err := migrate(raw); err != nil {
		t.Fatalf("migrate on legacy DB: %v", err)
	}

	// The user column must be gone.
	rows, err := raw.Query(`SELECT name FROM pragma_table_info('Run')`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close() //nolint:errcheck
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		if name == "user" {
			t.Fatal("Run.user column should have been dropped by migrate()")
		}
	}

	// The legacy row survives the drop, and a current-shape insert (no `user`)
	// succeeds against the migrated table.
	store := db.Tag(&sqliteStore{sqlDB: raw, events: make(chan db.StoreEvent, 8)})
	if err := store.CreateRun(context.Background(), db.RunRecord{RunID: "new1", CreatedAt: time.Now().UTC()}); err != nil {
		t.Fatalf("CreateRun after migrate: %v", err)
	}
	if _, err := store.GetRun(context.Background(), "old1"); err != nil {
		t.Errorf("legacy run lost after migrate: %v", err)
	}
}

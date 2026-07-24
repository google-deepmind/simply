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

package critic

import (
	"context"
	"sync"
	"testing"

	"amplio/internal/db"
)

// recordingTracker captures Register/Unregister pairs so tests can assert
// the Finalizer wraps its work — including when the body short-circuits or
// errors. The id allocator is intentionally trivial (monotonic per-instance)
// since assertions check the sequence, not the absolute values.
type recordingTracker struct {
	mu       sync.Mutex
	next     uint64
	events   []string // "register:run:kind" / "unregister:run"
	liveByID map[uint64]string
}

func newRecordingTracker() *recordingTracker {
	return &recordingTracker{liveByID: make(map[uint64]string)}
}

func (r *recordingTracker) Register(runID, kind, subject string) uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.next++
	r.liveByID[r.next] = runID
	r.events = append(r.events, "register:"+runID+":"+kind)
	return r.next
}

func (r *recordingTracker) Unregister(id uint64) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	runID, ok := r.liveByID[id]
	if !ok {
		r.events = append(r.events, "unregister-unknown")
		return ""
	}
	delete(r.liveByID, id)
	r.events = append(r.events, "unregister:"+runID)
	return runID
}

func (r *recordingTracker) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string{}, r.events...)
}

func (r *recordingTracker) liveCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.liveByID)
}

func TestFinalizer_AutoTriggerAndWatermark(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "do the thing")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 3)
	addPhase(t, store, "r", "main-agent", 1, 3, "Phase", "did work", nil)

	fin := NewFinalizer(store, submitMock{summary: "v1 summary"}, nil, nil, nil)

	fin.OnMainAgentConcluded(ctx, "r")
	all, err := AllReports(ctx, store, "r")
	must(t, err)
	if len(all) != 1 || all[0].Version != 1 || all[0].Summary != "v1 summary" {
		t.Fatalf("after first trigger: %+v, want one v1", all)
	}

	// Idempotent: nothing advanced → no new version.
	fin.OnMainAgentConcluded(ctx, "r")
	all, err = AllReports(ctx, store, "r")
	must(t, err)
	if len(all) != 1 {
		t.Fatalf("watermark not respected: %d reports, want 1", len(all))
	}

	// Advance the main-agent past the debounce threshold and conclude another
	// iteration → version 2. seedSession's AdvanceStep loop makes it easy to
	// bump the counter by an arbitrary amount.
	var endStep int
	for i := 0; i < ReportSkipMinSteps; i++ {
		s, err := store.AdvanceStep(ctx, "r", "main-agent")
		must(t, err)
		endStep = s
	}
	addPhase(t, store, "r", "main-agent", 4, endStep, "More", "more work", nil)

	fin.OnMainAgentConcluded(ctx, "r")
	all, err = AllReports(ctx, store, "r")
	must(t, err)
	if len(all) != 2 || all[1].Version != 2 {
		t.Fatalf("second iteration: %+v, want v1+v2", all)
	}
}

// A small delta on top of an existing report is debounced by
// ReportSkipMinSteps: the auto trigger returns without writing a new
// iteration, so a stray-notify reactivation that adds only a handful of steps
// can't produce a bad-grade report per reactivation.
func TestFinalizer_DebouncesSubThresholdDelta(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "do the thing")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 3)
	addPhase(t, store, "r", "main-agent", 1, 3, "Phase", "did work", nil)

	fin := NewFinalizer(store, submitMock{summary: "v1"}, nil, nil, nil)
	fin.OnMainAgentConcluded(ctx, "r")
	if all, _ := AllReports(ctx, store, "r"); len(all) != 1 {
		t.Fatalf("seed v1: got %d reports, want 1", len(all))
	}

	// Advance just BELOW the threshold: (ReportSkipMinSteps - 1) new steps
	// on top of the v1 watermark. Auto trigger must NOT write a v2.
	for i := 0; i < ReportSkipMinSteps-1; i++ {
		_, err := store.AdvanceStep(ctx, "r", "main-agent")
		must(t, err)
	}
	fin.OnMainAgentConcluded(ctx, "r")
	if all, _ := AllReports(ctx, store, "r"); len(all) != 1 {
		t.Fatalf("below threshold: got %d reports, want 1 (deferred)", len(all))
	}

	// One more step brings the delta to exactly the threshold → a v2 lands,
	// covering the whole accumulated span since v1 (no data lost).
	_, err := store.AdvanceStep(ctx, "r", "main-agent")
	must(t, err)
	fin.OnMainAgentConcluded(ctx, "r")
	all, _ := AllReports(ctx, store, "r")
	if len(all) != 2 || all[1].Version != 2 {
		t.Fatalf("at threshold: %+v, want v1+v2", all)
	}
}

// The operator-triggered Generate honors the same debounce, returning the
// previous report with deferred=true so the HTTP handler can respond 200
// (deferred) rather than 201 (created). First-report-always-lands: when there
// is no prior report, delta is not consulted.
func TestFinalizer_GenerateHonorsDebounce(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "do the thing")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 3)
	addPhase(t, store, "r", "main-agent", 1, 3, "Phase", "did work", nil)

	fin := NewFinalizer(store, submitMock{summary: "v1"}, nil, nil, nil)

	// prev == nil: always generates, even for a short run.
	rep, deferred, err := fin.Generate(ctx, "r")
	must(t, err)
	if deferred || rep == nil || rep.Version != 1 {
		t.Fatalf("first Generate: rep=%+v deferred=%v, want v1 not deferred", rep, deferred)
	}

	// Advance below the threshold and call again → deferred, previous returned.
	for i := 0; i < ReportSkipMinSteps-1; i++ {
		_, err := store.AdvanceStep(ctx, "r", "main-agent")
		must(t, err)
	}
	rep, deferred, err = fin.Generate(ctx, "r")
	must(t, err)
	if !deferred || rep == nil || rep.Version != 1 {
		t.Fatalf("sub-threshold Generate: rep=%+v deferred=%v, want v1 deferred", rep, deferred)
	}
	if all, _ := AllReports(ctx, store, "r"); len(all) != 1 {
		t.Fatalf("deferred Generate wrote a new report: %d, want 1", len(all))
	}

	// One more step crosses the threshold → a real v2 gets generated and
	// returned with deferred=false.
	_, err = store.AdvanceStep(ctx, "r", "main-agent")
	must(t, err)
	rep, deferred, err = fin.Generate(ctx, "r")
	must(t, err)
	if deferred || rep == nil || rep.Version != 2 {
		t.Fatalf("at-threshold Generate: rep=%+v deferred=%v, want v2 not deferred", rep, deferred)
	}
}

func TestFinalizer_InteractiveNoAutoButOperator(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "chat", "")
	seedSession(t, store, "chat", "chatty-bot", "chatbot", db.SessionIdle, 2)
	addPhase(t, store, "chat", "chatty-bot", 1, 2, "Chat", "discussed the task", nil)
	addStep(t, store, "chat", "chatty-bot", 1, "asked about the task", "progressing")
	addStep(t, store, "chat", "chatty-bot", 2, "answered the operator", "progressing")

	fin := NewFinalizer(store, submitMock{summary: "chat report"}, nil, nil, nil)

	// No main-agent → auto trigger is a no-op.
	fin.OnMainAgentConcluded(ctx, "chat")
	all, err := AllReports(ctx, store, "chat")
	must(t, err)
	if len(all) != 0 {
		t.Fatalf("interactive auto-report should not fire: %+v", all)
	}

	// Operator on-demand generates against the chatbot (it's the subject session
	// when there's no main-agent).
	rep, deferred, err := fin.Generate(ctx, "chat")
	must(t, err)
	if deferred {
		t.Fatalf("first Generate deferred unexpectedly (prev == nil path)")
	}
	if rep.Version != 1 {
		t.Fatalf("version = %d, want 1", rep.Version)
	}
	if s := findState(rep.Sessions, "chatty-bot"); s == nil || s.CurrentStep != 2 {
		t.Fatalf("chatbot not in snapshot: %+v", rep.Sessions)
	}
	if len(rep.Phases) != 1 {
		t.Fatalf("chatbot phase not aggregated: %+v", rep.Phases)
	}
}

// Both entry points must register the in-flight ephemeral as soon as they
// own the per-run lock, and unregister after — including when the body
// short-circuits (Generate's "nothing advanced" branch, OnMainAgentConcluded's
// watermark branch). Without this, the UI's "Generating report…" indicator
// would silently miss short-circuit attempts, leaving stale state visible.
func TestFinalizer_TracksInflight(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "do the thing")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 3)
	addPhase(t, store, "r", "main-agent", 1, 3, "Phase", "did work", nil)

	tracker := newRecordingTracker()
	fin := NewFinalizer(store, submitMock{summary: "v1"}, nil, nil, tracker)

	// First auto trigger: register → real generation → unregister.
	fin.OnMainAgentConcluded(ctx, "r")
	if tracker.liveCount() != 0 {
		t.Errorf("after auto trigger: live = %d, want 0", tracker.liveCount())
	}
	ev := tracker.snapshot()
	if len(ev) < 2 || ev[0] != "register:r:report" || ev[1] != "unregister:r" {
		t.Errorf("auto: events = %v; want register/unregister pair first", ev)
	}

	// Second auto trigger short-circuits via watermark — but still must
	// register/unregister so the UI sees a brief "checking" state.
	fin.OnMainAgentConcluded(ctx, "r")
	if tracker.liveCount() != 0 {
		t.Errorf("after watermark short-circuit: live = %d, want 0", tracker.liveCount())
	}
	if got := len(tracker.snapshot()); got != 4 { // 2 from first call + 2 from this one
		t.Errorf("event count after second trigger: %d, want 4", got)
	}

	// Manual Generate also wraps, even when it short-circuits to prev.
	_, _, err := fin.Generate(ctx, "r")
	if err != nil {
		t.Fatal(err)
	}
	if tracker.liveCount() != 0 {
		t.Errorf("after manual short-circuit: live = %d, want 0", tracker.liveCount())
	}
	if got := len(tracker.snapshot()); got != 6 {
		t.Errorf("event count after manual: %d, want 6", got)
	}
}

// A nil tracker is a documented call site (headless CLI: no UI to observe).
// The Finalizer must fall through to a no-op without panicking.
func TestFinalizer_NilTrackerIsNoOp(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "task")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 2)
	addPhase(t, store, "r", "main-agent", 1, 2, "P", "w", nil)

	fin := NewFinalizer(store, submitMock{summary: "v1"}, nil, nil, nil)
	// Must not panic on nil dereference inside the wrap.
	fin.OnMainAgentConcluded(ctx, "r")
	_, _, err := fin.Generate(ctx, "r")
	if err != nil {
		t.Fatal(err)
	}
}

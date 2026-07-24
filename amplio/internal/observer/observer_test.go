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

package observer

import (
	"context"
	"strings"
	"sync"
	"testing"

	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/event"
	"amplio/internal/llm"
)

// fakeFast returns the same valid step-summary JSON for every call (stateless →
// race-safe across workers).
type fakeFast struct{}

func (fakeFast) Call(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{Content: `{"summary":"did the thing","status_tag":"progressing"}`}, nil
}
func (fakeFast) Stream(context.Context, llm.Request) (llm.Stream, error) { return nil, nil }
func (fakeFast) ModelID() string                                         { return "fake-fast" }
func (fakeFast) MaxTokens() int                                          { return 1000 }

// fakeHQ returns a valid phase-summary JSON. end_step=999 is always out of range
// → clamped to the chunk's last step, so phases cover the whole chunk.
type fakeHQ struct{}

func (fakeHQ) Call(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{Content: `{"title":"a phase","summary":"did work across steps","end_step":999}`}, nil
}
func (fakeHQ) Stream(context.Context, llm.Request) (llm.Stream, error) { return nil, nil }
func (fakeHQ) ModelID() string                                         { return "fake-hq" }
func (fakeHQ) MaxTokens() int                                          { return 1000 }

const testRunID = "r"

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func newStore(t *testing.T) db.Store {
	t.Helper()
	store, err := sqlite.Open(":memory:")
	must(t, err)
	t.Cleanup(func() { _ = store.Close() })
	return store
}

// finalizeSteps creates a session with the given status and finalizes steps
// 1..n, each carrying an assistant event of `chars` characters.
func finalizeSteps(t *testing.T, store db.Store, sid, status string, n, chars int) {
	t.Helper()
	ctx := context.Background()
	must(t, store.CreateSession(ctx, db.SessionRecord{
		RunID: testRunID, SessionID: sid, AgentType: "standard_agent", Status: status,
	}))
	for step := 1; step <= n; step++ {
		must(t, store.FinalizeStep(ctx, testRunID, sid, step,
			[]event.Event{&event.AssistantEvent{Content: strings.Repeat("x", chars)}}))
	}
}

func obsByStep(t *testing.T, store db.Store, sid, kind string) map[int]db.ObservationRecord {
	t.Helper()
	recs, err := store.GetObservations(context.Background(), testRunID, db.ObsFilter{Kind: kind, SessionID: sid})
	must(t, err)
	out := map[int]db.ObservationRecord{}
	for _, r := range recs {
		if r.Step != nil {
			out[*r.Step] = r
		}
	}
	return out
}

func runObserver(t *testing.T, store db.Store) {
	t.Helper()
	obs := New(store, fakeFast{}, fakeHQ{}, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	obs.Start(ctx)
	obs.Stop(ctx) // sweep + drain
}

// --- step worker ---

func TestObserver_StepSummaries(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	must(t, store.CreateRun(ctx, db.RunRecord{RunID: "r"}))
	finalizeSteps(t, store, "main-agent", db.SessionOngoing, 3, 20)

	runObserver(t, store)

	steps := obsByStep(t, store, "main-agent", "step_summary")
	for _, n := range []int{1, 2, 3} {
		rec, ok := steps[n]
		if !ok {
			t.Fatalf("missing step_summary for step %d (have %v)", n, steps)
		}
		if rec.Data["summary"] != "did the thing" {
			t.Errorf("step %d summary = %v", n, rec.Data["summary"])
		}
		if rec.CharCount <= 0 {
			t.Errorf("step %d char_count = %d, want > 0", n, rec.CharCount)
		}
	}
	sess, err := store.GetSession(ctx, "r", "main-agent")
	must(t, err)
	if sess.LastSummarizedStep != 3 {
		t.Errorf("last_summarized_step = %d, want 3", sess.LastSummarizedStep)
	}
}

func TestObserver_EventDriven(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	must(t, store.CreateRun(ctx, db.RunRecord{RunID: "r"}))
	must(t, store.CreateSession(ctx, db.SessionRecord{
		RunID: "r", SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionOngoing,
	}))

	obs := New(store, fakeFast{}, fakeHQ{}, 2)
	ctx2, cancel := context.WithCancel(ctx)
	defer cancel()
	obs.Start(ctx2)
	for step := 1; step <= 3; step++ { // finalize while live → StepFinalized signals
		must(t, store.FinalizeStep(ctx, "r", "main-agent", step,
			[]event.Event{&event.AssistantEvent{Content: "work"}}))
	}
	obs.Stop(ctx2)

	if got := obsByStep(t, store, "main-agent", "step_summary"); len(got) != 3 {
		t.Fatalf("got %d step summaries, want 3", len(got))
	}
}

// --- phase worker ---

func TestObserver_PhaseForceCloseOnSettle(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	must(t, store.CreateRun(ctx, db.RunRecord{RunID: "r"}))
	// Concluded session, well below the 200k char threshold.
	finalizeSteps(t, store, "main-agent", db.SessionConcluded, 2, 20)

	runObserver(t, store)

	phases := obsByStep(t, store, "main-agent", "phase_summary")
	if len(phases) != 1 {
		t.Fatalf("got %d phase summaries, want 1 (force-close on conclude): %v", len(phases), phases)
	}
	rec, ok := phases[2] // end_step clamped to last step
	if !ok {
		t.Fatalf("phase summary not closed at step 2: %v", phases)
	}
	if rec.Data["title"] != "a phase" {
		t.Errorf("phase title = %v", rec.Data["title"])
	}
	sess, err := store.GetSession(ctx, "r", "main-agent")
	must(t, err)
	if sess.LastPhasedStep != 2 {
		t.Errorf("last_phased_step = %d, want 2", sess.LastPhasedStep)
	}
}

func TestObserver_PhaseNotForceClosedForIdle(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	must(t, store.CreateRun(ctx, db.RunRecord{RunID: "r"}))
	// Idle (interactive park) below threshold: must NOT force-close.
	finalizeSteps(t, store, "chatty-bot", db.SessionIdle, 2, 20)

	runObserver(t, store)

	if phases := obsByStep(t, store, "chatty-bot", "phase_summary"); len(phases) != 0 {
		t.Fatalf("idle session got %d phase summaries, want 0", len(phases))
	}
	// Steps are still summarized.
	if steps := obsByStep(t, store, "chatty-bot", "step_summary"); len(steps) != 2 {
		t.Errorf("got %d step summaries, want 2", len(steps))
	}
}

func TestObserver_PhaseThresholdClose(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	must(t, store.CreateRun(ctx, db.RunRecord{RunID: "r"}))
	// Ongoing session, but 3 × 100k chars > 200k threshold → phase closes
	// without force.
	finalizeSteps(t, store, "main-agent", db.SessionOngoing, 3, 100_000)

	runObserver(t, store)

	phases := obsByStep(t, store, "main-agent", "phase_summary")
	if len(phases) != 1 {
		t.Fatalf("got %d phase summaries, want 1 (threshold close): %v", len(phases), phases)
	}
	if _, ok := phases[3]; !ok {
		t.Errorf("phase not closed at step 3: %v", phases)
	}
}

func TestListPendingSessions(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	must(t, store.CreateRun(ctx, db.RunRecord{RunID: "r"}))

	// s1: step-pending (finalized 1..2, summarized 0).
	finalizeSteps(t, store, "s1", db.SessionOngoing, 2, 20)
	// s2: phase-pending (summarized but not phased).
	finalizeSteps(t, store, "s2", db.SessionOngoing, 1, 20)
	must(t, store.SetLastSummarizedStep(ctx, "r", "s2", 1))
	// s3: fully caught up.
	finalizeSteps(t, store, "s3", db.SessionOngoing, 1, 20)
	must(t, store.SetLastSummarizedStep(ctx, "r", "s3", 1))
	must(t, store.SetLastPhasedStep(ctx, "r", "s3", 1))

	pend, err := store.ListPendingSessions(ctx)
	must(t, err)
	got := map[string]bool{}
	for _, p := range pend {
		got[p.SessionID] = true
	}
	if !got["s1"] || !got["s2"] || got["s3"] || len(got) != 2 {
		t.Fatalf("pending = %v, want {s1, s2}", got)
	}
}

// --- finalizer hook ---

func runObserverWithFinalizer(t *testing.T, store db.Store) []string {
	t.Helper()
	var mu sync.Mutex
	var calls []string
	obs := New(store, fakeFast{}, fakeHQ{}, 2)
	obs.SetFinalizer(func(_ context.Context, runID string) {
		mu.Lock()
		calls = append(calls, runID)
		mu.Unlock()
	})
	ctx := context.Background()
	obs.Start(ctx)
	obs.Stop(ctx) // sweep + drain: the hook has fired by the time this returns
	mu.Lock()
	defer mu.Unlock()
	return append([]string(nil), calls...)
}

func TestObserver_FinalizeHookOnMainAgentConclude(t *testing.T) {
	store := newStore(t)
	must(t, store.CreateRun(context.Background(), db.RunRecord{RunID: testRunID}))
	finalizeSteps(t, store, "main-agent", db.SessionConcluded, 1, 20)

	calls := runObserverWithFinalizer(t, store)
	if len(calls) == 0 || calls[0] != testRunID {
		t.Fatalf("finalizer calls = %v, want [%s]", calls, testRunID)
	}
}

func TestObserver_FinalizeHookSkipsOngoingAndNonRoot(t *testing.T) {
	store := newStore(t)
	must(t, store.CreateRun(context.Background(), db.RunRecord{RunID: testRunID}))
	// Ongoing main-agent (not concluded) + a concluded non-root session: neither
	// should fire the hook.
	finalizeSteps(t, store, "main-agent", db.SessionOngoing, 1, 20)
	finalizeSteps(t, store, "sub-1", db.SessionConcluded, 1, 20)

	if calls := runObserverWithFinalizer(t, store); len(calls) != 0 {
		t.Fatalf("finalizer calls = %v, want none", calls)
	}
}

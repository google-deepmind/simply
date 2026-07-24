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

package coordination

import (
	"context"
	"strings"
	"testing"
	"time"

	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/event"
	"amplio/internal/llm"
	"amplio/internal/session"
	"amplio/internal/tool"
)

const (
	testRunID     = "run-1"
	testSessionID = "swift-fox"
)

// setup builds an in-memory store with one ongoing session at current_step=1,
// a registry with a live handle, and a commit listener that wakes the live
// session on every append (mirroring runtime.NewCommitNotifier for live sessions).
func setup(t *testing.T) (*Deps, *session.Handle) {
	t.Helper()
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRunID}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRunID, SessionID: testSessionID, AgentType: "event_loop", Status: db.SessionOngoing,
	}); err != nil {
		t.Fatal(err)
	}
	// Advance to step 1 so appended events land at a non-zero "current step".
	if _, err := store.AdvanceStep(ctx, testRunID, testSessionID); err != nil {
		t.Fatal(err)
	}

	reg := session.NewRegistry()
	handle := session.NewHandle(func() {})
	if err := reg.Register(testSessionID, handle); err != nil {
		t.Fatal(err)
	}
	store.SetCommitListener(func(_, sid string, _ event.Event) {
		reg.Notify(sid)
	})

	return &Deps{Store: store, RunID: testRunID, Registry: reg}, handle
}

func TestSessionCancel_Recursive(t *testing.T) {
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	runID := "run-cancel"
	_ = store.CreateRun(ctx, db.RunRecord{RunID: runID})
	// p (root) → c (child) → g (grandchild), all ongoing, no live goroutines.
	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "p", Status: db.SessionOngoing})
	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "c", ParentID: "p", Status: db.SessionOngoing})
	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "g", ParentID: "c", Status: db.SessionOngoing})

	deps := &Deps{Store: store, RunID: runID, Registry: session.NewRegistry()}
	if _, err := SessionCancel(deps).Execute(ctx, []byte(`{"session_id":"c","reason":"stop it"}`)); err != nil {
		t.Fatal(err)
	}

	// c and its descendant g are cancelled; the parent p is untouched.
	for _, id := range []string{"c", "g"} {
		s, _ := store.GetSession(ctx, runID, id)
		if s.Status != db.SessionCancelled {
			t.Errorf("%s status = %q, want cancelled", id, s.Status)
		}
	}
	if s, _ := store.GetSession(ctx, runID, "p"); s.Status != db.SessionOngoing {
		t.Errorf("parent p status = %q, want ongoing (untouched)", s.Status)
	}

	// p received a child_result(cancelled) for c.
	evts, _ := store.GetEvents(ctx, runID, "p", db.EventFilter{})
	var got *event.ChildResultEvent
	for _, e := range evts {
		if cr, ok := e.Event.(*event.ChildResultEvent); ok && cr.ChildSessionID == "c" {
			got = cr
		}
	}
	if got == nil || got.Verdict != db.SessionCancelled {
		t.Errorf("parent missing child_result(cancelled) for c: %+v", got)
	}

	// c carries the cancel marker on its own stream.
	cEvts, _ := store.GetEvents(ctx, runID, "c", db.EventFilter{})
	var marker bool
	for _, e := range cEvts {
		if se, ok := e.Event.(*event.SystemEvent); ok && se.Marker == event.MarkerCancelled {
			marker = true
		}
	}
	if !marker {
		t.Error("c missing cancelled marker on its own stream")
	}
}

// await_event is no longer Exclusive, so it can share a step with other tools.
// Through the real ExecuteAll path, two concurrent awaits must both park and
// both wake on a single event — the Waiter broadcasts (channel close) to every
// blocked waiter.
func TestAwaitEvent_MultipleWaitersWake(t *testing.T) {
	deps, handle := setup(t)
	await := AwaitEvent(deps, testSessionID, func() *session.Handle { return handle })
	toolMap := map[string]*tool.Tool{await.Name: await}
	calls := []llm.ToolCall{
		{ID: "w1", Name: "await_event", Arguments: `{"timeout":5}`},
		{ID: "w2", Name: "await_event", Arguments: `{"timeout":5}`},
	}

	resCh := make(chan []tool.CallResult, 1)
	go func() { resCh <- tool.ExecuteAll(context.Background(), calls, toolMap, nil) }()

	time.Sleep(50 * time.Millisecond) // let both park
	if _, err := deps.Store.AppendEvent(context.Background(), testRunID, testSessionID,
		&event.MessageEvent{Content: "ping", Sender: "peer", SenderType: event.SenderTypeAgent}); err != nil {
		t.Fatal(err)
	}

	select {
	case results := <-resCh:
		if len(results) != 2 {
			t.Fatalf("got %d results, want 2", len(results))
		}
		for _, r := range results {
			if strings.Contains(r.Result.Content, "must be the only tool call") {
				t.Fatalf("await_event still exclusive: %q", r.Result.Content)
			}
			if !strings.Contains(r.Result.Content, "Awakened") {
				t.Errorf("waiter %s not awakened: %q", r.ToolCallID, r.Result.Content)
			}
		}
	case <-time.After(3 * time.Second):
		t.Fatal("awaits did not wake on event")
	}
}

func runAwait(t *testing.T, deps *Deps, handle *session.Handle, argsJSON string) string {
	t.Helper()
	tl := AwaitEvent(deps, testSessionID, func() *session.Handle { return handle })
	res, err := tl.Execute(context.Background(), []byte(argsJSON))
	if err != nil {
		t.Fatalf("await execute: %v", err)
	}
	return res.Content
}

// A peer event already at the current step must short-circuit immediately —
// even though it already bumped the waiter counter (folded into the snapshot),
// the DB short-circuit catches it. This is the core race the rewrite fixes.
func TestAwaitEvent_ShortCircuitOnExistingEvent(t *testing.T) {
	deps, handle := setup(t)
	if _, err := deps.Store.AppendEvent(context.Background(), testRunID, testSessionID,
		&event.MessageEvent{Content: "already here", Sender: "peer", SenderType: event.SenderTypeAgent}); err != nil {
		t.Fatal(err)
	}
	got := runAwait(t, deps, handle, `{"timeout":5}`)
	if !strings.Contains(got, "returning immediately") {
		t.Errorf("expected short-circuit, got: %q", got)
	}
}

func TestAwaitEvent_Timeout(t *testing.T) {
	deps, handle := setup(t)
	got := runAwait(t, deps, handle, `{"timeout":0.05}`)
	if !strings.Contains(got, "Timed out") {
		t.Errorf("expected timeout, got: %q", got)
	}
}

// An event arriving while parked must wake the waiter via the commit listener.
func TestAwaitEvent_WakesOnEventDuringWait(t *testing.T) {
	deps, handle := setup(t)
	resCh := make(chan string, 1)
	go func() {
		resCh <- runAwait(t, deps, handle, `{"timeout":5}`)
	}()

	// Let the await park (snapshot + short-circuit pass, then WaitAfter).
	time.Sleep(50 * time.Millisecond)
	if _, err := deps.Store.AppendEvent(context.Background(), testRunID, testSessionID,
		&event.MessageEvent{Content: "ping", Sender: "peer", SenderType: event.SenderTypeAgent}); err != nil {
		t.Fatal(err)
	}

	select {
	case got := <-resCh:
		if !strings.Contains(got, "Awakened") {
			t.Errorf("expected wake, got: %q", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("await did not wake on event")
	}

	sess, err := deps.Store.GetSession(context.Background(), testRunID, testSessionID)
	if err != nil {
		t.Fatal(err)
	}
	if sess.Status != db.SessionOngoing {
		t.Errorf("status should be restored to ongoing, got %q", sess.Status)
	}
}

// The agent's own tool results commit back at the call step T (= CurrentStep-1)
// as each tool finishes, firing a notify that wakes the parked WaitAfter. That
// wake must NOT end the await: the DB is the source of truth and only events at
// step >= T+1 count. The await must re-check, find nothing at T+1, and re-sleep,
// then wake for real on a later peer event. This is the regression the rewrite
// guards against.
func TestAwaitEvent_IgnoresOwnStepToolResult(t *testing.T) {
	deps, handle := setup(t)
	// CurrentStep is 1 (T+1); the call step T is 0.
	const callStep = 0

	resCh := make(chan string, 1)
	go func() {
		resCh <- runAwait(t, deps, handle, `{"timeout":5}`)
	}()
	time.Sleep(50 * time.Millisecond) // let the await park

	// A tool result from the await's own step lands back at T=0 and fires a
	// commit/notify — this must spuriously wake but not satisfy the await.
	if err := deps.Store.AppendEventAtStep(context.Background(), testRunID, testSessionID, callStep,
		&event.ToolResultEvent{Content: "own result", ToolCallID: "x"}); err != nil {
		t.Fatal(err)
	}

	// Give the woken loop time to re-check and re-sleep; it must still be parked.
	time.Sleep(150 * time.Millisecond)
	select {
	case got := <-resCh:
		t.Fatalf("await ended on its own step-T tool result: %q", got)
	default:
	}

	// A real peer event at T+1 (plain append lands at CurrentStep=1) wakes it.
	if _, err := deps.Store.AppendEvent(context.Background(), testRunID, testSessionID,
		&event.MessageEvent{Content: "ping", Sender: "peer", SenderType: event.SenderTypeAgent}); err != nil {
		t.Fatal(err)
	}
	select {
	case got := <-resCh:
		if !strings.Contains(got, "Awakened") {
			t.Errorf("expected wake on peer event, got: %q", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("await did not wake on the real peer event")
	}
}

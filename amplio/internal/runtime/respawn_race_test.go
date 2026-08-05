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

package runtime

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"amplio/internal/agent"
	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/event"
	"amplio/internal/llm"
	"amplio/internal/session"
	"amplio/internal/workspace/plain"
)

// parkAgent mimics the real eventloop's registration timing: on Run() it waits
// at a gate (standing in for RespawnSession's slow pre-Register work + Run's own
// pre-Register work), THEN registers a handle in the session registry and parks
// in a waiter until notified or ctx-cancelled. It counts how many goroutines
// ran and how many actually registered.
type parkAgent struct {
	sid      string
	reg      *session.Registry
	handle   *session.Handle // Config.Handle from the launcher (pre-registered slot)
	gate     chan struct{}   // released by the test to let Run proceed past its slow window
	runs     *int32          // incremented on every Run() entry (goroutine count)
	awakened chan string     // one send per goroutine that was notified while parked
}

func (a *parkAgent) SessionID() string { return a.sid }

func (a *parkAgent) Run(ctx context.Context) error {
	atomic.AddInt32(a.runs, 1)
	// Mirror the real eventloop's registration model: the launcher already claimed
	// the slot (RegisterAndContext) and handed us the handle via Config; ctx is the
	// slot's cancelable context and Unregister is the launcher's (release's) job.
	// A direct caller with no handle registers its own (kept for completeness).
	h := a.handle
	if h == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		defer cancel()
		h = session.NewHandle(cancel)
		if err := a.reg.Register(a.sid, h); err != nil {
			return err
		}
		defer a.reg.Unregister(a.sid)
	}
	// Simulate the slow window between the launcher's slot claim and this goroutine
	// actually parking: wait until the test opens the gate (or ctx is cancelled).
	select {
	case <-a.gate:
	case <-ctx.Done():
		return nil
	}
	// Park like await_event: block on the waiter until notified or cancelled.
	before := h.Counter()
	if _, werr := h.WaitAfter(ctx, before, 2*time.Second); werr == nil {
		a.awakened <- a.sid
	}
	return nil
}

// TestRespawnDoesNotRivalLiveParkedSession is the Mechanism-A regression test:
// a session with a LIVE goroutine parked in await must be WOKEN by an
// Input-class commit (Notify), never respawned into a SECOND goroutine.
func TestRespawnDoesNotRivalLiveParkedSession(t *testing.T) {
	var runs int32
	gate := make(chan struct{})
	awakened := make(chan string, 8)

	agentType := "park_race_agent"
	agent.Register(agentType, func(env *agent.Env, cfg *agent.Config) (agent.Agent, error) {
		return &parkAgent{
			sid:      cfg.SessionID,
			reg:      env.Registry,
			handle:   cfg.Handle,
			gate:     gate,
			runs:     &runs,
			awakened: awakened,
		}, nil
	})

	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()
	runID := "run-race"
	sid := "sess-A"
	if err := store.CreateRun(ctx, db.RunRecord{RunID: runID}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: sid, AgentType: agentType, Status: db.SessionAwaiting,
	}); err != nil {
		t.Fatal(err)
	}

	runReg := NewRunRegistry()
	mgr := NewRunManager(store, func(string) (llm.Provider, error) { return &llm.MockProvider{Model: "m"}, nil }, runReg, plain.Factory)
	store.SetCommitListener(NewCommitNotifier(runReg, mgr.RespawnSession, mgr.SessionStatus))

	// (1) Respawn A. Its goroutine starts and blocks at the gate BEFORE
	// registering — exactly the pre-Register window where the bug bites.
	mgr.RespawnSession(runID, sid)
	waitFor(t, func() bool { return atomic.LoadInt32(&runs) == 1 }, "first goroutine to start")

	// (2) While A is in that window, an Input-class event commits for the
	// session (a child_result / message). The notifier must NOT spawn a rival:
	// with the fix, RespawnSession reserves the slot synchronously, so the slot
	// is already registered and the second respawn is a no-op (and a later
	// Notify wakes the live goroutine).
	if _, err := store.AppendEvent(ctx, runID, sid,
		&event.MessageEvent{Content: "wake up", Sender: "child", SenderType: event.SenderTypeAgent}); err != nil {
		t.Fatal(err)
	}

	// Give any (buggy) rival goroutine time to spawn.
	time.Sleep(100 * time.Millisecond)

	// (3) Let the parked goroutine(s) proceed past the gate.
	close(gate)

	// Exactly ONE goroutine must ever run for this session.
	waitFor(t, func() bool { return atomic.LoadInt32(&runs) >= 1 }, "goroutine count")
	time.Sleep(200 * time.Millisecond)
	if n := atomic.LoadInt32(&runs); n != 1 {
		t.Fatalf("expected exactly 1 goroutine for the session, got %d (spurious respawn created a rival)", n)
	}

	// And the single live goroutine must have been WOKEN by the Input via Notify
	// (not left to time out) — proving the Input took the wake path, not respawn.
	select {
	case got := <-awakened:
		if got != sid {
			t.Fatalf("awakened %q, want %q", got, sid)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the live goroutine was never woken by the Input (Notify missed it)")
	}
}

func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for !cond() {
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		case <-time.After(2 * time.Millisecond):
		}
	}
}

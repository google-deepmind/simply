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

package eventstream

import "testing"

func drainPrime(t *testing.T, s *Subscription) {
	t.Helper()
	select {
	case ev := <-s.C():
		if ev.Kind != KindRefetchAll {
			t.Fatalf("first event = %q, want refetch_all prime", ev.Kind)
		}
	default:
		t.Fatal("subscription not primed with refetch_all")
	}
}

func TestBus_RunScopedFanout(t *testing.T) {
	bus := NewBus()
	a := bus.Subscribe("run-a")
	defer a.Close()
	all := bus.Subscribe("") // dashboard
	defer all.Close()
	drainPrime(t, a)
	drainPrime(t, all)

	bus.Publish(RunEvent{Kind: KindSessionBump, RunID: "run-a", SessionID: "s1"})
	bus.Publish(RunEvent{Kind: KindSessionBump, RunID: "run-b", SessionID: "s2"})

	// run-a subscriber sees only run-a.
	got := <-a.C()
	if got.RunID != "run-a" {
		t.Errorf("run-a sub got %q", got.RunID)
	}
	select {
	case ev := <-a.C():
		t.Errorf("run-a sub leaked event for %q", ev.RunID)
	default:
	}

	// dashboard sees both.
	if (<-all.C()).RunID != "run-a" || (<-all.C()).RunID != "run-b" {
		t.Error("dashboard did not see both runs in order")
	}
}

// Empty-RunID events are global (e.g. workspace_alias): they reach every
// subscriber, including per-run ones that wouldn't normally match by id.
func TestBus_GlobalEventFansOutToPerRunSubs(t *testing.T) {
	bus := NewBus()
	a := bus.Subscribe("run-a")
	defer a.Close()
	all := bus.Subscribe("")
	defer all.Close()
	drainPrime(t, a)
	drainPrime(t, all)

	bus.Publish(RunEvent{Kind: KindWorkspaceAlias, NumericID: 42, Alias: "foo"})

	// Both subscribers receive it.
	if ev := <-a.C(); ev.Kind != KindWorkspaceAlias || ev.Alias != "foo" {
		t.Errorf("per-run sub did not receive global event: %+v", ev)
	}
	if ev := <-all.C(); ev.Kind != KindWorkspaceAlias || ev.Alias != "foo" {
		t.Errorf("dashboard sub did not receive global event: %+v", ev)
	}
}

func TestBus_OverflowFlagsRefetch(t *testing.T) {
	bus := &Bus{subs: make(map[*Subscription]struct{}), depth: 2}
	s := bus.Subscribe("r") // prime fills slot 1 of 2
	defer s.Close()
	for range 5 {
		bus.Publish(RunEvent{Kind: KindSessionBump, RunID: "r"})
	}
	if !s.TakeOverflow() {
		t.Fatal("expected overflow flag after exceeding buffer")
	}
	if s.TakeOverflow() {
		t.Fatal("overflow flag should clear after TakeOverflow")
	}
}

func TestBus_CloseStopsDelivery(t *testing.T) {
	bus := NewBus()
	s := bus.Subscribe("r")
	drainPrime(t, s)
	s.Close()
	bus.Publish(RunEvent{Kind: KindSessionBump, RunID: "r"}) // must not panic / deliver
	select {
	case ev := <-s.C():
		t.Errorf("closed subscription received %q", ev.Kind)
	default:
	}
}

func TestBusBroadcaster_Chunk(t *testing.T) {
	bus := NewBus()
	s := bus.Subscribe("r")
	defer s.Close()
	drainPrime(t, s)

	b := NewBusBroadcaster(bus)
	b.Chunk("r", "s1", 3, "", "")           // empty → dropped
	b.Chunk("r", "s1", 3, "hello", "think") // published

	select {
	case ev := <-s.C():
		if ev.Kind != KindStreamChunk || ev.TextDelta != "hello" || ev.ThoughtsDelta != "think" || ev.Step != 3 {
			t.Errorf("unexpected chunk: %+v", ev)
		}
	default:
		t.Fatal("expected a stream_chunk event")
	}
	select {
	case ev := <-s.C():
		t.Errorf("empty chunk should not publish: %+v", ev)
	default:
	}
}

func TestNoOpBroadcaster(t *testing.T) {
	NoOpBroadcaster{}.Chunk("r", "s", 1, "x", "y") // must not panic
}

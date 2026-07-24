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
	"sync"
	"testing"
)

func TestEphemeralAgentRegistry_RegisterListUnregister(t *testing.T) {
	r := NewEphemeralAgentRegistry()
	id1 := r.Register("run-a", "report", "")
	id2 := r.Register("run-b", "report", "")
	id3 := r.Register("run-a", "title", "")

	if got := len(r.ForRun("run-a")); got != 2 {
		t.Errorf("run-a in-flight = %d; want 2", got)
	}
	if got := len(r.ForRun("run-b")); got != 1 {
		t.Errorf("run-b in-flight = %d; want 1", got)
	}
	if got := len(r.ForRun("run-missing")); got != 0 {
		t.Errorf("missing run = %d; want 0", got)
	}

	if removed := r.Unregister(id1); removed != "run-a" {
		t.Errorf("Unregister(id1) returned %q; want run-a", removed)
	}
	if got := len(r.ForRun("run-a")); got != 1 {
		t.Errorf("run-a after unregister id1 = %d; want 1", got)
	}

	// The remaining run-a entry is "title", confirming we deleted the right
	// one (not just the first iteration order picked up).
	if got := r.ForRun("run-a"); len(got) != 1 || got[0].Kind != "title" {
		t.Errorf("remaining run-a entry = %+v; want kind=title", got)
	}

	r.Unregister(id2)
	r.Unregister(id3)

	if got := len(r.ForRun("run-a")); got != 0 {
		t.Errorf("run-a after all unregistered = %d; want 0", got)
	}
}

func TestEphemeralAgentRegistry_UnregisterUnknownIsNoOp(t *testing.T) {
	r := NewEphemeralAgentRegistry()
	if removed := r.Unregister(9999); removed != "" {
		t.Errorf("Unregister(unknown) returned %q; want empty", removed)
	}
	// Mixed: real register, then unregister an unknown id, then unregister
	// the real one. The unknown call must not corrupt state.
	id := r.Register("run-a", "report", "")
	r.Unregister(9999)
	if got := len(r.ForRun("run-a")); got != 1 {
		t.Errorf("after spurious unregister, run-a = %d; want 1", got)
	}
	r.Unregister(id)
	if got := len(r.ForRun("run-a")); got != 0 {
		t.Errorf("after real unregister, run-a = %d; want 0", got)
	}
}

func TestEphemeralAgentRegistry_NotifyFiresOutsideLock(t *testing.T) {
	r := NewEphemeralAgentRegistry()
	// A notify hook that itself reads from the registry would deadlock if
	// invoked while we hold the write mutex — verify we don't.
	var got []string
	var mu sync.Mutex
	r.SetOnChange(func(ag EphemeralAgent, active bool) {
		live := r.ForRun(ag.RunID) // would deadlock if called under r.mu
		mu.Lock()
		got = append(got, ag.RunID)
		mu.Unlock()
		_ = live
		_ = active
	})
	id := r.Register("run-a", "report", "")
	r.Unregister(id)
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 2 || got[0] != "run-a" || got[1] != "run-a" {
		t.Errorf("notify hook fired with %v; want [run-a, run-a]", got)
	}
}

func TestEphemeralAgentRegistry_ConcurrentAccess(t *testing.T) {
	// Race-detector smoke: many goroutines register / list / unregister at
	// the same time. Mostly here so `go test -race` catches future
	// regressions if someone changes the locking.
	r := NewEphemeralAgentRegistry()
	const n = 100
	ids := make(chan uint64, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for i := range n {
		go func(i int) {
			defer wg.Done()
			id := r.Register("run", "report", "")
			_ = r.ForRun("run")
			if i%2 == 0 {
				r.Unregister(id)
				return
			}
			ids <- id
		}(i)
	}
	wg.Wait()
	close(ids)
	for id := range ids {
		r.Unregister(id)
	}
	if got := len(r.ForRun("run")); got != 0 {
		t.Errorf("residual entries: %d; want 0", got)
	}
}

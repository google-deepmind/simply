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
	"sync/atomic"
	"time"
)

// EphemeralAgent describes one in-flight background worker that is NOT
// represented as a session in the DB. It does NOT occupy a session ID,
// can NOT perform session related operations such as spawn sub-agents,
// send or receive messages. It is NOT recovered upon server crash. This
// mechanism is designed to perform logically "simple" operations that
// still benefit from having an agentic loop. It behaves like a function
// call rather than a crash resistent agent session. Canonical examples
// are critic agent that generates the run reports, and compaction agent
// that summarize the contexts.
//
// Even though the EphemeralAgent execution are not persisted in DB, we
// do use an in-memory registry to keep track of their life cycle. This
// is mostly for UX. e.g. for the user to see why chatbot is not responding
// (as compaction is going on), or run report is missing (generation is
// in progress).
type EphemeralAgent struct {
	// ID is a process-monotonic unique identifier returned by Register and
	// passed back to Unregister. The UI never sees this — it's purely
	// caller-side bookkeeping so concurrent ephemerals of the same kind on
	// the same run can be deregistered independently. Per-run mutex in
	// callers (e.g., critic.Finalizer.lockFor) makes that rare today, but
	// the registry doesn't depend on caller serialization.
	ID    uint64
	RunID string
	Kind  string // "report", "compaction"; future: "title", "autorater", …
	// Subject is what the work targets, NOT the worker's own identity (an
	// ephemeral agent has no session of its own). Empty = run-level work (e.g.
	// the critic report covers the whole run); set = the session the work acts
	// on (e.g. compaction names the session being compacted), so a session UI
	// can show the indicator in the right place.
	Subject   string
	StartedAt time.Time // UTC; UI renders "elapsed time" relative to now
}

// EphemeralAgentRegistry is the process-global, per-run set of in-flight
// EphemeralAgents. Safe for concurrent use. Zero value is NOT usable — call
// NewEphemeralAgentRegistry.
type EphemeralAgentRegistry struct {
	mu     sync.RWMutex
	byRun  map[string]map[uint64]EphemeralAgent
	nextID atomic.Uint64
	// onChange is notified on every register/unregister with the affected agent
	// and whether it just became active (true=register, false=unregister), so a
	// listener can publish a targeted live signal without re-querying.
	onChange func(ag EphemeralAgent, active bool)
}

// NewEphemeralAgentRegistry constructs an empty registry. Wire the onChange
// hook later with SetOnChange (typically: publish KindEphemeralAgents on the
// global event bus so all open SSE subscribers see the change).
func NewEphemeralAgentRegistry() *EphemeralAgentRegistry {
	return &EphemeralAgentRegistry{
		byRun: make(map[string]map[uint64]EphemeralAgent),
	}
}

// SetOnChange installs the (single) notify hook. Replaces any prior hook.
// Pass nil to clear. The hook is invoked OUTSIDE the registry mutex so
// callbacks (typically bus publishes) can't deadlock with concurrent
// Register/Unregister/ForRun calls.
func (r *EphemeralAgentRegistry) SetOnChange(fn func(ag EphemeralAgent, active bool)) {
	r.mu.Lock()
	r.onChange = fn
	r.mu.Unlock()
}

// Register records an in-flight ephemeral for runID and returns its id.
// Always pair with a deferred Unregister so a caller panic still cleans up:
//
//	id := registry.Register(runID, "report")
//	defer registry.Unregister(id)
//	// …do the long-running work…
func (r *EphemeralAgentRegistry) Register(runID, kind, subject string) uint64 {
	id := r.nextID.Add(1)
	ag := EphemeralAgent{ID: id, RunID: runID, Kind: kind, Subject: subject, StartedAt: time.Now().UTC()}
	r.mu.Lock()
	if r.byRun[runID] == nil {
		r.byRun[runID] = make(map[uint64]EphemeralAgent)
	}
	r.byRun[runID][id] = ag
	notify := r.onChange
	r.mu.Unlock()
	if notify != nil {
		notify(ag, true)
	}
	return id
}

// Unregister removes the entry with the given id. No-op when the id is
// unknown — safe for defensive `defer Unregister(id)` patterns where Register
// might not have run (e.g., early return). Returns the runID that owned the
// entry, or "" when nothing was removed (helps tests assert outcomes).
func (r *EphemeralAgentRegistry) Unregister(id uint64) string {
	r.mu.Lock()
	var found string
	var removed EphemeralAgent
	for rid, m := range r.byRun {
		if ag, ok := m[id]; ok {
			removed = ag
			delete(m, id)
			if len(m) == 0 {
				delete(r.byRun, rid)
			}
			found = rid
			break
		}
	}
	notify := r.onChange
	r.mu.Unlock()
	if notify != nil && found != "" {
		notify(removed, false)
	}
	return found
}

// ForRun returns a snapshot of in-flight ephemerals for runID. The slice is
// owned by the caller and safe to use after the call (no aliasing). Order
// is unspecified; UI sorts by Kind+StartedAt when it cares.
func (r *EphemeralAgentRegistry) ForRun(runID string) []EphemeralAgent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m := r.byRun[runID]
	if len(m) == 0 {
		return nil
	}
	out := make([]EphemeralAgent, 0, len(m))
	for _, ag := range m {
		out = append(out, ag)
	}
	return out
}

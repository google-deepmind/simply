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

	"amplio/internal/session"
)

// RunRegistry maps run IDs to per-run session registries.
// The two-level structure (run → sessions) avoids composite keys and gives
// each run its own isolated coordination namespace.
type RunRegistry struct {
	mu   sync.RWMutex
	runs map[string]*session.Registry
}

func NewRunRegistry() *RunRegistry {
	return &RunRegistry{runs: make(map[string]*session.Registry)}
}

// GetOrCreate returns the session registry for a run, creating one if
// it doesn't exist.
func (rr *RunRegistry) GetOrCreate(runID string) *session.Registry {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	if reg, ok := rr.runs[runID]; ok {
		return reg
	}
	reg := session.NewRegistry()
	rr.runs[runID] = reg
	return reg
}

// Get returns the session registry for a run, or nil if not tracked.
func (rr *RunRegistry) Get(runID string) *session.Registry {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	return rr.runs[runID]
}

// Remove removes a run's registry. Called when all sessions have exited
// and the run is no longer active.
func (rr *RunRegistry) Remove(runID string) {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	delete(rr.runs, runID)
}

// RemoveIfEmpty atomically removes a run's registry only if it currently has no
// registered sessions, and reports whether it removed it.
//
// This must be used instead of a separate IsEmpty()+Remove() from the
// goroutine-exit cleanup path: checking emptiness and deleting under a single
// hold of rr.mu closes a TOCTOU where a concurrent respawn/spawn registers a
// new session into the (same) registry between the check and the delete — which
// would orphan a live session (its registry no longer reachable via Get, so the
// commit-notifier can't wake it and respawns it cold, duplicating the
// goroutine). Holding rr.mu across reg.IsEmpty() also serializes against Get,
// the wake path's lookup.
//
// The atomicity here is necessary but NOT sufficient on its own: the same
// orphaning would happen if RemoveIfEmpty ran while a run's registry was
// legitimately empty because a launcher had captured the instance (via
// GetOrCreate) but not yet registered its session. Every launcher closes that
// window by claiming the slot SYNCHRONOUSLY via Registry.RegisterAndContext
// before spawning its goroutine, keeping the instance non-empty across the slow
// pre-launch work (see RegisterAndContext).
func (rr *RunRegistry) RemoveIfEmpty(runID string) bool {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	reg, ok := rr.runs[runID]
	if !ok {
		return false
	}
	if !reg.IsEmpty() {
		return false
	}
	delete(rr.runs, runID)
	return true
}

// IsRunActive reports whether a run has any registered sessions.
func (rr *RunRegistry) IsRunActive(runID string) bool {
	rr.mu.RLock()
	reg, ok := rr.runs[runID]
	rr.mu.RUnlock()
	if !ok {
		return false
	}
	return !reg.IsEmpty()
}

// ActiveRunIDs returns all run IDs that have at least one registered session.
func (rr *RunRegistry) ActiveRunIDs() []string {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	var ids []string
	for id, reg := range rr.runs {
		if !reg.IsEmpty() {
			ids = append(ids, id)
		}
	}
	return ids
}

// CancelAll cancels all sessions across all runs.
func (rr *RunRegistry) CancelAll() {
	rr.mu.RLock()
	regs := make([]*session.Registry, 0, len(rr.runs))
	for _, reg := range rr.runs {
		regs = append(regs, reg)
	}
	rr.mu.RUnlock()
	for _, reg := range regs {
		reg.CancelAll()
	}
}

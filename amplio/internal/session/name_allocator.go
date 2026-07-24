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

package session

import (
	"context"
	"fmt"
	"sync"

	"amplio/internal/db"
	"amplio/internal/nickname"
)

// NameAllocator hands out unique session nicknames within a single run.
//
// It is the in-process serialization point for name allocation. Because amplio
// runs a whole run in one process, reserving a picked name in memory (under the
// mutex, before the DB insert) eliminates the pick→insert race without any
// retry loop or DB-level collision detection. The used-set is lazily seeded
// once from the DB (every existing session id in the run), so cold (concluded/
// crashed/idle) names are never reused.
//
// One allocator is shared by all sessions in a run; concurrent spawns serialize
// on it and therefore receive distinct names.
type NameAllocator struct {
	store db.Store
	runID string

	mu     sync.Mutex
	used   map[string]bool
	seeded bool
}

// NewNameAllocator returns an allocator for a run. The used-set is seeded lazily
// from the DB on the first Next call.
func NewNameAllocator(store db.Store, runID string) *NameAllocator {
	return &NameAllocator{store: store, runID: runID}
}

// Next returns a fresh nickname that is unique within the run and reserves it
// in-memory before returning, so a concurrent Next cannot hand out the same name.
func (a *NameAllocator) Next(ctx context.Context) (string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.seeded {
		sessions, err := a.store.ListSessions(ctx, a.runID)
		if err != nil {
			return "", fmt.Errorf("seed name allocator: %w", err)
		}
		a.used = make(map[string]bool, len(sessions)+8)
		for _, s := range sessions {
			a.used[s.SessionID] = true
		}
		a.seeded = true
	}

	name := nickname.PickUnique(a.used, nil)
	a.used[name] = true
	return name, nil
}

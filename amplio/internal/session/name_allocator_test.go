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
	"sync"
	"testing"

	"amplio/internal/db"
	"amplio/internal/db/sqlite"
)

func allocStore(t *testing.T, runID string) db.Store {
	t.Helper()
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if err := store.CreateRun(context.Background(), db.RunRecord{RunID: runID}); err != nil {
		t.Fatal(err)
	}
	return store
}

// Names already in the DB (including cold/concluded ones) are never handed out.
func TestNameAllocator_SeedsFromDBAndExcludesExisting(t *testing.T) {
	ctx := context.Background()
	runID := "run-1"
	store := allocStore(t, runID)
	// A concluded (cold, unregistered) session — its name must not be reused.
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: "swift-fox", Status: db.SessionConcluded,
	}); err != nil {
		t.Fatal(err)
	}

	a := NewNameAllocator(store, runID)
	seen := map[string]bool{"swift-fox": true}
	for i := 0; i < 50; i++ {
		name, err := a.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if seen[name] {
			t.Fatalf("allocator reused a name: %q", name)
		}
		seen[name] = true
	}
}

// Concurrent Next calls serialize on the allocator and never collide.
func TestNameAllocator_ConcurrentDistinct(t *testing.T) {
	ctx := context.Background()
	runID := "run-1"
	store := allocStore(t, runID)
	a := NewNameAllocator(store, runID)

	const n = 100
	var wg sync.WaitGroup
	var mu sync.Mutex
	names := make(map[string]bool)
	dup := false
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			name, err := a.Next(ctx)
			if err != nil {
				return
			}
			mu.Lock()
			if names[name] {
				dup = true
			}
			names[name] = true
			mu.Unlock()
		}()
	}
	wg.Wait()

	if dup {
		t.Error("allocator handed out a duplicate name under concurrency")
	}
	if len(names) != n {
		t.Errorf("got %d distinct names, want %d", len(names), n)
	}
}

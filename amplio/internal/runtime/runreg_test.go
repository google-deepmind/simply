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
	"sync"
	"testing"
)

// TestRegisterAndContext_KeepsRegistryFromBeingOrphaned is the fix-confirming
// regression for the true root cause of the duplicate-tool_result bug:
// run-registry INSTANCE ORPHANING.
//
// The bug's shape (original code): a launcher did
//
//	reg := runReg.GetOrCreate(runID) // instance RegA
//	... slow pre-Register work (workspace.Validate does I/O) ...
//	go func(){ ... Run() eventually reg.Register(session) ... }()
//
// so RegA stayed EMPTY across the slow work. A concurrent goroutine-exit defer
// calling RemoveIfEmpty(runID) saw RegA empty and DELETED it from the map. The
// launcher then registered its LIVE handle into the now-orphaned RegA, while the
// commit notifier's Get(runID) returned a DIFFERENT/nil instance — so Notify
// missed the live session and a rival respawn spawned a second goroutine ->
// duplicate tool_result -> 400.
//
// The fix (RegisterAndContext) claims the slot SYNCHRONOUSLY right after
// GetOrCreate, before the slow work, so RegA is non-empty at the vulnerable
// moment. This test asserts exactly that invariant: after the claim, a concurrent
// RemoveIfEmpty is a no-op, the run keeps its ORIGINAL registry instance, and the
// session stays visible via Get + IsRegistered.
func TestRegisterAndContext_KeepsRegistryFromBeingOrphaned(t *testing.T) {
	runReg := NewRunRegistry()
	const (
		runID     = "run-1"
		sessionID = "sess-1"
	)

	// A launcher's first two steps: capture the instance, then claim the slot.
	regA := runReg.GetOrCreate(runID)
	_, _, release, ok := regA.RegisterAndContext(context.Background(), sessionID)
	if !ok {
		t.Fatal("RegisterAndContext: ok=false, want a winning claim")
	}
	defer release()

	// A concurrent goroutine-exit cleanup runs RemoveIfEmpty during the (now
	// safe) pre-launch window. It MUST be a no-op because the slot is claimed.
	if removed := runReg.RemoveIfEmpty(runID); removed {
		t.Fatal("RemoveIfEmpty deleted a registry that has a claimed (live) session")
	}

	// The run must still map to the SAME instance we claimed into — not a fresh
	// one a later GetOrCreate would create (which is what orphaned the live one).
	if got := runReg.Get(runID); got != regA {
		t.Fatalf("Get(runID) = %p, want the original claimed instance %p (registry was orphaned)", got, regA)
	}

	// And the session is visible through that instance, so the commit notifier's
	// Get(runID).Notify(sessionID) can find and wake it.
	if !runReg.Get(runID).IsRegistered(sessionID) {
		t.Fatal("claimed session not visible via Get(runID).IsRegistered — Notify would miss it")
	}
	if !runReg.Get(runID).Notify(sessionID) {
		t.Fatal("Notify(sessionID) returned false for a claimed (live) session — would trigger a rival respawn")
	}
}

// TestRegisterAndContext_KeepsRegistryFromBeingOrphaned_Concurrent stresses the
// same invariant under concurrency: many launch-shaped sequences (GetOrCreate +
// RegisterAndContext) racing goroutine-exit RemoveIfEmpty calls. No claimed
// session may ever end up orphaned (its run mapping to a different instance /
// losing visibility). Run under `go test -race`.
func TestRegisterAndContext_KeepsRegistryFromBeingOrphaned_Concurrent(t *testing.T) {
	const iters = 300
	for i := 0; i < iters; i++ {
		runReg := NewRunRegistry()
		const runID, sessionID = "run", "sess"

		// The launch: capture instance + claim synchronously.
		regA := runReg.GetOrCreate(runID)
		if _, _, _, ok := regA.RegisterAndContext(context.Background(), sessionID); !ok {
			t.Fatalf("iter %d: claim lost unexpectedly", i)
		}

		var wg sync.WaitGroup
		wg.Add(1)
		// A concurrent cleanup attempt (goroutine-exit defer).
		go func() {
			defer wg.Done()
			runReg.RemoveIfEmpty(runID)
		}()
		wg.Wait()

		// Post-condition: the claimed session is never orphaned.
		if got := runReg.Get(runID); got != regA || !got.IsRegistered(sessionID) {
			t.Fatalf("iter %d: claimed session orphaned (Get=%p regA=%p registered=%v)",
				i, got, regA, got != nil && got.IsRegistered(sessionID))
		}
	}
}

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

package workspace

import (
	"sync"
	"testing"
)

func TestLinkLocks_SerializesSameKey(t *testing.T) {
	l := NewLinkLocks()
	var counter int
	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			release := l.Acquire("repo")
			defer release()
			counter++ // guarded by the keyed lock; -race verifies no concurrent access
		}()
	}
	wg.Wait()
	if counter != 50 {
		t.Errorf("counter = %d, want 50", counter)
	}
	if got := l.size(); got != 0 {
		t.Errorf("size after release = %d, want 0 (entries should be reclaimed)", got)
	}
}

func TestLinkLocks_IndependentKeys(t *testing.T) {
	l := NewLinkLocks()
	// Different keys must not block each other; acquiring both on one goroutine
	// would deadlock (test hang) if they shared a lock.
	r1 := l.Acquire("a")
	r2 := l.Acquire("b")
	if got := l.size(); got != 2 {
		t.Errorf("size = %d, want 2", got)
	}
	r1()
	r2()
	if got := l.size(); got != 0 {
		t.Errorf("size after release = %d, want 0", got)
	}
}

func TestLinkLocks_ReleaseIdempotent(t *testing.T) {
	l := NewLinkLocks()
	release := l.Acquire("k")
	release()
	release() // must be a no-op, not a double-unlock panic
	if got := l.size(); got != 0 {
		t.Errorf("size = %d, want 0", got)
	}
}

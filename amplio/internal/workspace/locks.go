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

import "sync"

// LinkLocks serializes linked-workspace creation per shared repo (keyed by the
// main repo's identity), preventing the concurrent-creation races some VCSes
// exhibit — notably jj's op-head corruption when two `workspace add`s race on
// the same repo. Concurrent commits and normal tool use are unaffected; only
// creation is serialized.
//
// In-process only, which is sufficient: a turn's tool calls (e.g. several
// link-mode spawns) run as parallel goroutines in ONE amplio process, so that
// fan-out is the real race. A cross-process race (two amplio invocations sharing
// a repo) is rare-squared and surfaces as a clean VCS error from the create
// command, not corruption.
//
// Entries are reference-counted and removed when the last holder/waiter
// releases, so the map stays bounded to currently-active repos even in a
// long-lived server hosting many runs across many repos.
type LinkLocks struct {
	guard sync.Mutex
	locks map[string]*refLock
}

type refLock struct {
	mu   sync.Mutex
	refs int
}

// NewLinkLocks returns an empty LinkLocks.
func NewLinkLocks() *LinkLocks {
	return &LinkLocks{locks: make(map[string]*refLock)}
}

// Acquire blocks until it holds the lock for key, then returns a release func
// the caller MUST call (typically via defer) exactly once. Different keys never
// contend; the same key serializes. The returned func both unlocks and drops the
// entry's refcount, deleting it from the map when no holder/waiter remains.
//
// Acquire is not context-cancellable: the wait is bounded (creation is seconds,
// serialized per repo). A caller that needs cancellation should check ctx after
// acquiring, before doing work.
func (l *LinkLocks) Acquire(key string) (release func()) {
	l.guard.Lock()
	rl, ok := l.locks[key]
	if !ok {
		rl = &refLock{}
		l.locks[key] = rl
	}
	rl.refs++ // mark interest under the guard so the entry can't be reclaimed
	l.guard.Unlock()

	rl.mu.Lock()

	var once sync.Once
	return func() {
		once.Do(func() {
			rl.mu.Unlock()
			l.guard.Lock()
			rl.refs--
			if rl.refs == 0 {
				delete(l.locks, key)
			}
			l.guard.Unlock()
		})
	}
}

// size reports the number of live lock entries (test-only).
func (l *LinkLocks) size() int {
	l.guard.Lock()
	defer l.guard.Unlock()
	return len(l.locks)
}

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
	"errors"
	"fmt"
	"sync"
	"time"

	"amplio/internal/db"
	"amplio/internal/event"
)

// Waiter is a per-session notification primitive: a monotonic counter plus a
// broadcast channel that is closed-and-replaced on each notify. It is the Go
// equivalent of a condition variable + counter, supporting WaitAfter with a
// timeout and context cancellation.
type Waiter struct {
	mu      sync.Mutex
	counter uint64
	ch      chan struct{}
}

func newWaiter() *Waiter {
	return &Waiter{ch: make(chan struct{})}
}

func (w *Waiter) snapshot() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.counter
}

func (w *Waiter) notify() {
	w.mu.Lock()
	w.counter++
	close(w.ch)
	w.ch = make(chan struct{})
	w.mu.Unlock()
}

// waitAfter blocks until the counter advances past last, the timeout elapses,
// or ctx is cancelled. It returns the counter at wake time: equal to last
// means it timed out without a notify; greater than last means a notify
// arrived. timeout <= 0 waits until notify or ctx cancellation.
func (w *Waiter) waitAfter(ctx context.Context, last uint64, timeout time.Duration) (uint64, error) {
	var timeoutCh <-chan time.Time
	if timeout > 0 {
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		timeoutCh = timer.C
	}
	for {
		w.mu.Lock()
		if w.counter != last {
			c := w.counter
			w.mu.Unlock()
			return c, nil
		}
		ch := w.ch
		w.mu.Unlock()

		select {
		case <-ch:
			// Notified — loop and re-read the counter under the lock.
		case <-timeoutCh:
			return w.snapshot(), nil
		case <-ctx.Done():
			return w.snapshot(), ctx.Err()
		}
	}
}

// Handle is a registered session's coordination surface: its notification
// waiter plus a cancel func that stops the session's goroutine.
//
// The cancel is fixed at construction and never changes: RegisterAndContext
// creates the cancelable ctx and the handle together, registers synchronously,
// and only THEN does the caller launch the goroutine — so the cancel is fully
// set before the slot is ever visible to a concurrent Interrupt/CancelAll. No
// post-registration mutation, hence no atomic needed.
type Handle struct {
	waiter *Waiter
	cancel context.CancelFunc
}

// NewHandle creates a Handle with a fresh notification waiter and a fixed cancel.
func NewHandle(cancel context.CancelFunc) *Handle {
	return &Handle{waiter: newWaiter(), cancel: cancel}
}

// Counter snapshots the handle's notification counter. Capture this before the
// action that may produce events, then pass it to WaitAfter.
func (h *Handle) Counter() uint64 { return h.waiter.snapshot() }

// WaitAfter blocks until the counter advances past last, the timeout elapses,
// or ctx is cancelled. Returns the counter at wake time (== last means timed
// out without a notify).
func (h *Handle) WaitAfter(ctx context.Context, last uint64, timeout time.Duration) (uint64, error) {
	return h.waiter.waitAfter(ctx, last, timeout)
}

// Registry maps session IDs to handles within a single run.
// Safe for concurrent use.
type Registry struct {
	mu       sync.RWMutex
	sessions map[string]*Handle
}

func NewRegistry() *Registry {
	return &Registry{sessions: make(map[string]*Handle)}
}

func (r *Registry) Register(sessionID string, h *Handle) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.sessions[sessionID]; exists {
		return fmt.Errorf("session %q already registered", sessionID)
	}
	r.sessions[sessionID] = h
	return nil
}

// RegisterAndContext claims a session's slot SYNCHRONOUSLY and returns the
// pieces a launcher needs to run the session's goroutine safely:
//
//   - ctx: a cancelable child of parent that the goroutine should loop on;
//   - h: the registered Handle (its cancel is ctx's cancel — so Interrupt /
//     CancelAll on this slot stop exactly this ctx);
//   - release: cancels ctx and unregisters the slot; the goroutine defers it.
//
// It returns ok=false (and does nothing) if the slot is already taken — the
// caller should treat that as "already live" and NOT launch a rival goroutine.
//
// This is the single registration path for every session launch (fresh run,
// respawn, spawned child). Registering the slot BEFORE the caller spawns its
// goroutine — and before any slow pre-launch work — makes the run's registry
// instance non-empty at the vulnerable moment, so a concurrent goroutine-exit
// RemoveIfEmpty can never observe it empty and orphan it (the root cause of the
// duplicate-tool_result bug). Because ctx and its cancel are created together
// here and installed in the handle before the slot is visible, a cancel that
// races (or precedes) the goroutine start is latched on ctx and observed by the
// goroutine as soon as it runs — no adopt window, no lost cancel.
func (r *Registry) RegisterAndContext(parent context.Context, sessionID string) (ctx context.Context, h *Handle, release func(), ok bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.sessions[sessionID]; exists {
		return nil, nil, nil, false
	}
	ctx, cancel := context.WithCancel(parent)
	h = NewHandle(cancel)
	r.sessions[sessionID] = h
	release = func() {
		cancel()
		r.Unregister(sessionID)
	}
	return ctx, h, release, true
}

func (r *Registry) Unregister(sessionID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.sessions, sessionID)
}

// Notify wakes a registered session by bumping its waiter counter. It reports
// whether the session was registered (live): a false return tells the caller
// the session is cold and may need a respawn instead.
func (r *Registry) Notify(sessionID string) bool {
	r.mu.RLock()
	h, ok := r.sessions[sessionID]
	r.mu.RUnlock()
	if !ok {
		return false
	}
	h.waiter.notify()
	return true
}

// Interrupt ctx-cancels a live session's goroutine, if registered. It does NOT
// change DB status — that is the caller's responsibility (see CancelSession).
func (r *Registry) Interrupt(sessionID string) {
	r.mu.RLock()
	h, ok := r.sessions[sessionID]
	r.mu.RUnlock()
	if ok {
		h.cancel()
	}
}

// CancelAll interrupts all session goroutines in this registry (e.g. on process
// shutdown). It only ctx-cancels; it does not change DB status.
func (r *Registry) CancelAll() {
	r.mu.RLock()
	handles := make([]*Handle, 0, len(r.sessions))
	for _, h := range r.sessions {
		handles = append(handles, h)
	}
	r.mu.RUnlock()
	for _, h := range handles {
		h.cancel()
	}
}

// CancelSession is the canceller-driven cancel primitive, applied recursively
// (see docs/session_lifecycle.md). Whoever cancels does all the work; the target
// agent just stops when its ctx is interrupted. It is safe on a cold (no
// goroutine) or already-terminal session.
//
// For the session and each active descendant it: (1) atomically sets status
// cancelled, writes a "cancelled" SystemEvent marker to the session's own stream,
// and notifies the parent (TerminateAndNotifyParent — which fires the commit
// handler for the parent only, never self-waking the target); then (2) interrupts
// the live goroutine, if any. An already-terminal session is skipped (NOP), which
// also dedupes the recursive cascade.
//
// It returns an error only if the target's own terminal write failed — in that
// case the goroutine is deliberately NOT interrupted, so the session stays
// genuinely ongoing (alive) and a retry or Recover can re-attempt cleanly,
// rather than being left ongoing-but-dead with a parent that was never notified.
// The recursive child cascade is best-effort: a child's failure is logged and
// folded into the returned error but never blocks the parent's cancellation.
func CancelSession(store db.Store, reg *Registry, runID, sessionID, reason string) error {
	sess, err := store.GetSession(context.Background(), runID, sessionID)
	if err != nil {
		return fmt.Errorf("cancel %s: read session: %w", sessionID, err)
	}
	if db.SessionTerminalStatuses[sess.Status] {
		return nil // already terminal: NOP (also dedupes the recursive cascade)
	}
	// -1: an async cancel marker belongs at the session's current step.
	if err := store.TerminateAndNotifyParent(context.Background(), runID, sessionID, sess.ParentID,
		db.SessionCancelled, reason, &event.SystemEvent{Content: reason, Marker: event.MarkerCancelled}, -1); err != nil {
		// The terminal write failed: do NOT interrupt the goroutine. Leaving it
		// alive keeps the session truthfully ongoing (not ongoing-but-dead), so
		// the canceller can retry — interrupting now would strand the session and
		// its awaiting parent (which never got a child_result).
		return fmt.Errorf("cancel %s: %w", sessionID, err)
	}

	// Cascade into children (best-effort: never let a child failure stop us from
	// having cancelled the target, but surface it so the caller can log/retry).
	var childErrs []error
	children, err := store.GetChildSessions(context.Background(), runID, sessionID)
	if err != nil {
		childErrs = append(childErrs, fmt.Errorf("list children of %s: %w", sessionID, err))
	}
	for _, child := range children {
		if cErr := CancelSession(store, reg, runID, child.SessionID, "parent "+sessionID+" cancelled"); cErr != nil {
			childErrs = append(childErrs, cErr)
		}
	}

	if reg != nil {
		reg.Interrupt(sessionID)
	}
	return errors.Join(childErrs...)
}

// IsRegistered reports whether a session has an active goroutine.
func (r *Registry) IsRegistered(sessionID string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.sessions[sessionID]
	return ok
}

// IsEmpty reports whether the registry has no sessions.
func (r *Registry) IsEmpty() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.sessions) == 0
}

// SessionIDs returns all registered session IDs.
func (r *Registry) SessionIDs() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make([]string, 0, len(r.sessions))
	for id := range r.sessions {
		ids = append(ids, id)
	}
	return ids
}

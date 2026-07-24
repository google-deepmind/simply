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

package observer

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/llm"
)

// DefaultWorkers bounds concurrent cross-session summarization. Per-session
// work stays serialized (a session is processed by at most one worker at a
// time) to preserve step order and avoid duplicate LLM calls.
const DefaultWorkers = 3

const stepSummaryKind = "step_summary"

type sessionKey struct {
	runID     string
	sessionID string
}

// Observer is a single, process-global summarizer. It reacts to StepFinalized
// signals (the fast path) and to a durable catch-up sweep over the pending-set
// partial index (crash recovery + final drain). All state lives in the DB; the
// observer holds only an in-memory work set.
type Observer struct {
	store   db.Store
	llmFast llm.Provider
	llmHQ   llm.Provider
	workers int
	logger  *slog.Logger

	// finalizer, if set, is invoked (synchronously, on the worker) after a
	// session's summaries are caught up, when that session is the concluded
	// main-agent. It generates the run report. It must be idempotent — it's
	// re-fired by the crash-recovery sweep. nil = no report generation.
	finalizer func(ctx context.Context, runID string)

	mu      sync.Mutex
	cond    *sync.Cond
	pending map[sessionKey]struct{} // wants (re)processing
	active  map[sessionKey]struct{} // currently being processed
	closed  bool
	wg      sync.WaitGroup
}

// SetFinalizer installs the run-report hook, fired when the main-agent concludes
// (and re-checked by the recovery sweep). Set once at startup before Start.
func (o *Observer) SetFinalizer(fn func(ctx context.Context, runID string)) { o.finalizer = fn }

func New(store db.Store, llmFast, llmHQ llm.Provider, workers int) *Observer {
	if workers <= 0 {
		workers = DefaultWorkers
	}
	o := &Observer{
		store:   store,
		llmFast: llmFast,
		llmHQ:   llmHQ,
		workers: workers,
		pending: make(map[sessionKey]struct{}),
		active:  make(map[sessionKey]struct{}),
		logger:  slog.Default().With("component", "observer"),
	}
	o.cond = sync.NewCond(&o.mu)
	return o
}

// Start registers the StepFinalized hook, launches the worker pool, and sweeps
// the durable pending set for catch-up. ctx bounds worker lifetime: on
// cancellation workers abandon in-flight work (a later sweep re-summarizes it).
func (o *Observer) Start(ctx context.Context) {
	o.store.SetStepFinalizedListener(func(runID, sessionID string, _ int) {
		o.enqueue(sessionKey{runID, sessionID})
	})
	// Settle trigger: force-close the trailing phase when a session reaches a
	// terminal status. process() applies the precise !IsSpine && !idle rule, so
	// a crashed root coarsely enqueued here just does normal work.
	o.store.SetSessionStatusChangedListener(func(runID, sessionID, newStatus string) {
		switch newStatus {
		case db.SessionConcluded, db.SessionCancelled, db.SessionCrashed:
			o.enqueue(sessionKey{runID, sessionID})
		}
	})
	for range o.workers {
		o.wg.Add(1)
		go o.worker(ctx)
	}
	// Wake blocked workers/Stop when ctx is cancelled so they can observe it.
	go func() {
		<-ctx.Done()
		o.mu.Lock()
		o.cond.Broadcast()
		o.mu.Unlock()
	}()
	o.sweep(ctx)
}

// Stop captures any final finalized-but-unqueued steps, drains the work set, and
// waits for workers to exit. Aborts the drain if ctx is cancelled.
func (o *Observer) Stop(ctx context.Context) {
	o.sweep(ctx)
	o.mu.Lock()
	for (len(o.pending) > 0 || len(o.active) > 0) && ctx.Err() == nil {
		o.cond.Wait()
	}
	o.closed = true
	o.cond.Broadcast()
	o.mu.Unlock()
	o.wg.Wait()
}

func (o *Observer) enqueue(key sessionKey) {
	o.mu.Lock()
	if !o.closed {
		o.pending[key] = struct{}{}
		o.cond.Broadcast()
	}
	o.mu.Unlock()
}

func (o *Observer) worker(ctx context.Context) {
	defer o.wg.Done()
	for {
		o.mu.Lock()
		key, ok := o.takePending()
		for !ok && !o.closed && ctx.Err() == nil {
			o.cond.Wait()
			key, ok = o.takePending()
		}
		if !ok {
			o.mu.Unlock()
			return // closed or cancelled with nothing takeable
		}
		o.active[key] = struct{}{}
		o.mu.Unlock()

		o.process(ctx, key)

		o.mu.Lock()
		delete(o.active, key)
		o.cond.Broadcast() // wake peers waiting to re-take this key, and Stop's drain
		o.mu.Unlock()
	}
}

// takePending returns a pending key not currently being processed, removing it
// from the pending set. Caller holds o.mu.
func (o *Observer) takePending() (sessionKey, bool) {
	for k := range o.pending {
		if _, busy := o.active[k]; busy {
			continue
		}
		delete(o.pending, k)
		return k, true
	}
	return sessionKey{}, false
}

// process is a session's full work unit: catch up step summaries, then close any
// due phases. Per-session serialization (the active set) gives phases their
// required in-order processing and the fresh cursors after the step pass.
func (o *Observer) process(ctx context.Context, key sessionKey) {
	o.catchUpSteps(ctx, key)
	o.closePhases(ctx, key)
	o.maybeFinalize(ctx, key)
}

// maybeFinalize fires the run-report hook, but ONLY when processing the
// main-agent session itself: sub-agents and the chatbot never trigger it, even
// while they emit events after the main-agent concludes (their work units have a
// different key.sessionID). It runs after summaries are caught up (the report
// reads them, falling back to raw events for any tail the summarizer hasn't
// reached). The hook (and the Finalizer behind it) is idempotent on the
// main-agent step watermark under a per-run lock, so repeated fires (a sweep
// re-enqueue, the startup backfill) yield at most one report per iteration.
// Report generation writes only the run_report observation, which doesn't
// re-enter the observer — no feedback loop. The serve hook runs generation in its
// own goroutine, so this never occupies a summarizer worker.
func (o *Observer) maybeFinalize(ctx context.Context, key sessionKey) {
	if o.finalizer == nil || key.sessionID != config.RootAgentSessionID {
		return
	}
	sess, err := o.store.GetSession(ctx, key.runID, key.sessionID)
	if err != nil || sess.Status != db.SessionConcluded {
		return
	}
	o.finalizer(ctx, key.runID)
}

// catchUpSteps summarizes a session's pending steps (last_summarized,
// last_finalized], bumping the consumer cursor after each. On error it stops and
// leaves the session dirty for the next signal or sweep to retry.
func (o *Observer) catchUpSteps(ctx context.Context, key sessionKey) {
	sess, err := o.store.GetSession(ctx, key.runID, key.sessionID)
	if err != nil {
		o.warn(ctx, "get session", key, 0, err)
		return
	}
	for step := sess.LastSummarizedStep + 1; step <= sess.LastFinalizedStep; step++ {
		if ctx.Err() != nil {
			return
		}
		if err := o.summarizeOneStep(ctx, key.runID, key.sessionID, step); err != nil {
			o.warn(ctx, "summarize step", key, step, err)
			return
		}
		if err := o.store.SetLastSummarizedStep(ctx, key.runID, key.sessionID, step); err != nil {
			o.warn(ctx, "bump cursor", key, step, err)
			return
		}
	}
}

// summarizeOneStep reads a step's events, summarizes them, and writes the
// idempotent step_summary observation. char_count is stamped here so the phase
// trigger's sum works for degraded rows too.
func (o *Observer) summarizeOneStep(ctx context.Context, runID, sessionID string, step int) error {
	recs, err := o.store.GetEvents(ctx, runID, sessionID, db.EventFilter{StartStep: &step, EndStep: &step})
	if err != nil {
		return fmt.Errorf("read events: %w", err)
	}
	events := make([]event.Event, len(recs))
	for i, r := range recs {
		events[i] = r.Event
	}
	payload := summarizeStep(ctx, o.llmFast, sessionID, step, events)
	sp := step
	return o.store.AppendObservation(ctx, db.ObservationRecord{
		ObsID:     stepSummaryObsID(sessionID, step),
		RunID:     runID,
		Kind:      stepSummaryKind,
		SessionID: sessionID,
		Step:      &sp,
		CharCount: eventsCharCount(events),
		Data:      payload,
		CreatedAt: time.Now().UTC(),
	})
}

// sweep enqueues every session with un-summarized finalized steps (O(dirty) via
// the partial index).
func (o *Observer) sweep(ctx context.Context) {
	pending, err := o.store.ListPendingSessions(ctx)
	if err != nil {
		if ctx.Err() == nil {
			o.logger.Warn("observer: sweep failed", "error", err)
		}
		return
	}
	for _, p := range pending {
		o.enqueue(sessionKey{p.RunID, p.SessionID})
	}
}

func (o *Observer) warn(ctx context.Context, msg string, key sessionKey, step int, err error) {
	if ctx.Err() != nil {
		return // suppress noise from shutdown-cancelled work
	}
	o.logger.Warn("observer: "+msg+" failed",
		"run_id", key.runID, "session_id", key.sessionID, "step", step, "error", err)
}

func eventsCharCount(events []event.Event) int {
	n := 0
	for _, ev := range events {
		n += len(ev.ToText())
	}
	return n
}

func stepSummaryObsID(sessionID string, step int) string {
	return fmt.Sprintf("%s-%s-%d", stepSummaryKind, sessionID, step)
}

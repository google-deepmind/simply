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

package critic

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	"amplio/internal/skills"
)

// InflightTracker is the minimal surface the Finalizer needs to advertise its
// in-flight report generations to the UI. The server wires this to
// runtime.EphemeralAgentRegistry; the headless CLI passes nil (no UI to
// observe), which falls through to a no-op so call sites stay nil-safe.
//
// Defined here (not imported from runtime) to avoid an `agent/critic →
// runtime` import; runtime's registry satisfies the interface implicitly.
type InflightTracker interface {
	Register(runID, kind, subject string) uint64
	Unregister(id uint64) string
}

type noopTracker struct{}

func (noopTracker) Register(string, string, string) uint64 { return 0 }
func (noopTracker) Unregister(uint64) string               { return "" }

// InflightKindReport is the Kind value the Finalizer uses when registering
// itself. Exported so the UI side has one source of truth (the frontend's
// "is report generating?" check filters on this constant's value).
const InflightKindReport = "report"

// ReportSkipMinSteps is the minimum main-agent step delta required for a new
// report iteration. Below this, both the auto-trigger (OnMainAgentConcluded)
// and the operator-trigger (Generate) defer, returning the previous report
// unchanged. Never applied when there is no previous report — a run's first
// report always lands, regardless of length.
//
// The threshold exists to debounce trivial re-conclusions after the main-agent
// is revived by a stray environment notification (see docs/session_lifecycle.md
// on Input-class events reviving `concluded` sessions). Such reactivations
// typically add 1-3 steps of "nothing to do here" before the agent re-concludes,
// which would otherwise produce a bad-grade report per reactivation.
//
// Deferrals do NOT advance any watermark: the next attempt's delta is measured
// from the last real report, so accumulated small deltas eventually cross the
// threshold and land in a single report — no data is lost, only debounced.
const ReportSkipMinSteps = 30

// Finalizer owns report generation triggering and idempotency. It is shared by
// the observer (auto trigger on main-agent conclude + crash recovery) and the
// server (operator on-demand). Generation per run is serialized by a per-run
// mutex so a live trigger and an operator click can't double-generate.
type Finalizer struct {
	store       db.Store
	hq          llm.Provider    // shared system-tier HQ provider (report generation)
	skillIndex  *skills.Index   // shared skill recall corpus; nil = no skill recall
	lessonIndex *lessons.Index  // shared lesson recall corpus; nil = no lesson recall
	tracker     InflightTracker // UI visibility for in-flight generations

	mu    sync.Mutex
	locks map[string]*sync.Mutex // per-run generation lock
}

// NewFinalizer constructs a Finalizer. hq is the shared system-tier HQ provider
// the report agent runs on — a System property, like the observer's tiers, built
// once at startup, not a per-run override. tracker may be nil — the server wires
// runtime.EphemeralAgentRegistry, headless callers pass nil (UI-less).
func NewFinalizer(store db.Store, hq llm.Provider, skillIndex *skills.Index, lessonIndex *lessons.Index, tracker InflightTracker) *Finalizer {
	if tracker == nil {
		tracker = noopTracker{}
	}
	return &Finalizer{
		store:       store,
		hq:          hq,
		skillIndex:  skillIndex,
		lessonIndex: lessonIndex,
		tracker:     tracker,
		locks:       make(map[string]*sync.Mutex),
	}
}

// lockFor returns the per-run generation mutex (created on first use). The map
// grows by one entry per run ever finalized — negligible, never cleaned up.
func (f *Finalizer) lockFor(runID string) *sync.Mutex {
	f.mu.Lock()
	defer f.mu.Unlock()
	m := f.locks[runID]
	if m == nil {
		m = &sync.Mutex{}
		f.locks[runID] = m
	}
	return m
}

func (f *Finalizer) deps(run *db.RunRecord) Deps {
	return Deps{Store: f.store, HQ: f.hq, SkillIndex: f.skillIndex, LessonIndex: f.lessonIndex, CWD: run.Config.Workspace}
}

// OnMainAgentConcluded generates a report for an autonomous run whose main-agent
// has concluded, unless (a) the latest report already covers its current step,
// or (b) the delta since the latest report is below ReportSkipMinSteps (see
// that const for the debounce rationale). Safe to call from the live trigger
// and from crash recovery (idempotent on the main-agent step watermark);
// errors are logged, not returned. A run with no concluded main-agent (e.g.
// interactive-only) is a no-op.
func (f *Finalizer) OnMainAgentConcluded(ctx context.Context, runID string) {
	mu := f.lockFor(runID)
	mu.Lock()
	defer mu.Unlock()

	// Advertise the in-flight generation to the UI as soon as we hold the
	// per-run lock — even the pre-flight checks below (ListSessions,
	// LatestReport) take time on a heavily-loaded server, and surfacing the
	// "Generating report…" state right at the start avoids a brief window
	// where the UI says "no report yet" between conclude and registration.
	// Deferred Unregister survives panics inside the LLM call below.
	id := f.tracker.Register(runID, InflightKindReport, "")
	defer f.tracker.Unregister(id)

	sessions, err := f.store.ListSessions(ctx, runID)
	if err != nil {
		slog.Error("report: list sessions failed", "run_id", runID, "error", err)
		return
	}
	main := findSession(sessions, config.RootAgentSessionID)
	if main == nil || main.Status != db.SessionConcluded {
		return // not an autonomous run at rest
	}
	prev, err := LatestReport(ctx, f.store, runID)
	if err != nil {
		slog.Error("report: load latest failed", "run_id", runID, "error", err)
		return
	}
	if prev != nil {
		prevStep := prev.SessionStep(config.RootAgentSessionID)
		if prevStep >= main.CurrentStep {
			return // already covered
		}
		if main.CurrentStep-prevStep < ReportSkipMinSteps {
			// Debounce trivial re-conclusions (e.g. stray environment notifies
			// reviving a concluded agent for a handful of steps). The delta will
			// accumulate against the same watermark on each subsequent try and
			// eventually cross the threshold, at which point a real report covers
			// everything since prev — nothing is lost, only deferred.
			slog.Info("report: deferring — delta below threshold",
				"run_id", runID, "prev_version", prev.Version,
				"prev_step", prevStep, "current_step", main.CurrentStep,
				"threshold", ReportSkipMinSteps)
			return
		}
	}
	run, err := f.store.GetRun(ctx, runID)
	if err != nil {
		slog.Error("report: get run failed", "run_id", runID, "error", err)
		return
	}
	deps := f.deps(run)
	report, err := GenerateReport(ctx, deps, runID, prev)
	if err != nil {
		slog.Error("report: generate failed", "run_id", runID, "error", err)
		return
	}
	if err := writeReport(ctx, f.store, runID, report); err != nil {
		slog.Error("report: write failed", "run_id", runID, "error", err)
		return
	}
	slog.Info("generated run report", "run_id", runID, "version", report.Version)
	f.maybeMine(ctx, deps, runID, report)
	f.maybeAttribute(ctx, deps, runID)
}

// Generate produces a report on operator demand against the current snapshot,
// regardless of main-agent status (works for interactive runs too). It returns
// the latest report unchanged with deferred=true when the delta since it is
// below ReportSkipMinSteps — the caller (HTTP handler) surfaces the deferral
// so the UI can explain why no new iteration was produced. On a run with no
// prior report, always generates (first-report-always-lands). Serialized per
// run.
func (f *Finalizer) Generate(ctx context.Context, runID string) (report *RunReport, deferred bool, err error) {
	mu := f.lockFor(runID)
	mu.Lock()
	defer mu.Unlock()

	// Same as OnMainAgentConcluded: surface the in-flight state to the UI
	// from the moment we own the run, regardless of whether the call
	// short-circuits on the delta check below.
	id := f.tracker.Register(runID, InflightKindReport, "")
	defer f.tracker.Unregister(id)

	sessions, err := f.store.ListSessions(ctx, runID)
	if err != nil {
		return nil, false, err
	}
	prev, err := LatestReport(ctx, f.store, runID)
	if err != nil {
		return nil, false, err
	}
	if prev != nil && !advancedByAtLeast(prev, subjectSessions(sessions), ReportSkipMinSteps) {
		return prev, true, nil
	}
	run, err := f.store.GetRun(ctx, runID)
	if err != nil {
		return nil, false, err
	}
	deps := f.deps(run)
	report, err = GenerateReport(ctx, deps, runID, prev)
	if err != nil {
		return nil, false, err
	}
	if err := writeReport(ctx, f.store, runID, report); err != nil {
		return nil, false, err
	}
	f.maybeMine(ctx, deps, runID, report)
	f.maybeAttribute(ctx, deps, runID)
	return report, false, nil
}

// maybeAttribute scores the lessons loaded during the run (once per (run, lesson),
// via per-lesson sentinels in AttributeLessons). Reads loaded lessons from the
// phase summaries' recall_engagements. Called under the per-run lock; best-effort.
func (f *Finalizer) maybeAttribute(ctx context.Context, deps Deps, runID string) {
	n, err := AttributeLessons(ctx, deps, runID)
	if err != nil {
		slog.Warn("lesson attribution failed", "run_id", runID, "error", err)
		return
	}
	if n > 0 {
		slog.Info("attributed lessons", "run_id", runID, "scored", n)
	}
}

// maybeMine runs lesson mining for an iteration once, after its report. Guarded
// by a per-version sentinel so it can't double-mine. Called under the per-run
// lock (so mining is serialized per run); best-effort (failures are logged).
func (f *Finalizer) maybeMine(ctx context.Context, deps Deps, runID string, report *RunReport) {
	if deps.LessonIndex == nil || !deps.LessonIndex.IsBuilt() {
		return
	}
	sentinel := fmt.Sprintf("%s-%d", lessonsMinedKind, report.Version)
	recs, err := f.store.GetObservations(ctx, runID, db.ObsFilter{Kind: lessonsMinedKind})
	if err != nil {
		slog.Warn("mine: read sentinel failed", "run_id", runID, "error", err)
		return
	}
	for _, r := range recs {
		if r.ObsID == sentinel {
			return // already mined this iteration
		}
	}
	n, err := MineLessons(ctx, deps, runID, report)
	if err != nil {
		slog.Warn("lesson mining failed", "run_id", runID, "version", report.Version, "error", err)
		return
	}
	if err := f.store.AppendObservation(ctx, db.ObservationRecord{
		ObsID: sentinel, RunID: runID, Kind: lessonsMinedKind,
		Data: map[string]any{"inserted": n}, CreatedAt: time.Now().UTC(),
	}); err != nil {
		slog.Warn("mine: write sentinel failed", "run_id", runID, "error", err)
	}
	if n > 0 {
		slog.Info("mined lessons", "run_id", runID, "version", report.Version, "inserted", n)
	}
}

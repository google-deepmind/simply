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
	"sort"
	"strings"
	"time"

	"amplio/internal/db"
)

// lessonAttributionKind marks a (run, lesson) pair as scored. The sentinel id is
// lesson_attribution-<lesson_id>, so each lesson is attributed at most once per
// run (even if loaded across multiple iterations).
const lessonAttributionKind = "lesson_attribution"

const lessonHandlePrefix = "lesson:"

// verdictDelta maps a lesson verdict to a score change. Asymmetric on purpose: a
// false "harmful" hurts the corpus more than a false "unhelpful", so harmful is
// penalized hardest. Verdicts are produced by the phase summarizer (which saw the
// raw events); attribution just applies the deltas — no LLM call here.
var verdictDelta = map[string]int{
	"helpful":   1,
	"neutral":   0,
	"unhelpful": -1,
	"harmful":   -3,
}

// AttributeLessons applies the score deltas for lessons the agent loaded during
// the run, at most once per (run, lesson). The per-lesson verdicts were produced
// inline by the phase summarizer (lesson_verdicts on each phase_summary, judged
// against the raw chunk events); this is a pure DB pass with NO LLM call. When a
// lesson is judged in multiple phases, the MOST-RECENT phase's verdict wins (the
// phase with the most accumulated context). Returns the number scored.
func AttributeLessons(ctx context.Context, deps Deps, runID string) (int, error) {
	if deps.LessonIndex == nil {
		return 0, nil // lessons system off
	}
	phases, err := deps.Store.GetObservations(ctx, runID, db.ObsFilter{Kind: phaseSummaryKind})
	if err != nil {
		return 0, fmt.Errorf("get phase summaries: %w", err)
	}
	verdicts := collectLessonVerdicts(phases)
	if len(verdicts) == 0 {
		return 0, nil
	}

	attributed, err := attributedIDs(ctx, deps.Store, runID)
	if err != nil {
		return 0, err
	}

	// Deterministic order so the pass is reproducible.
	ids := make([]string, 0, len(verdicts))
	for id := range verdicts {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	scored := 0
	for _, id := range ids {
		if attributed[id] {
			continue
		}
		if _, err := deps.Store.GetLesson(ctx, id); err != nil {
			continue // deleted/missing → nothing to score
		}
		verdict := verdicts[id]
		delta := verdictDelta[verdict]
		if delta != 0 {
			if err := deps.Store.AddToLessonScore(ctx, id, delta); err != nil {
				slog.Warn("attribute: add score failed", "lesson_id", id, "error", err)
				continue // don't sentinel a failed score; resume retries
			}
		}
		if err := deps.Store.AppendObservation(ctx, db.ObservationRecord{
			ObsID: lessonAttributionKind + "-" + id, RunID: runID, Kind: lessonAttributionKind,
			Data: map[string]any{"verdict": verdict, "delta": delta}, CreatedAt: time.Now().UTC(),
		}); err != nil {
			slog.Warn("attribute: write sentinel failed", "lesson_id", id, "error", err)
		}
		scored++
	}
	return scored, nil
}

// rawVerdict is a phase_summary lesson_verdicts entry.
type rawVerdict struct{ handle, verdict string }

// phaseVerdicts pulls the lesson_verdicts list out of a phase_summary's Data blob.
// Missing/old rows (no key) yield nil — back-compat for phase summaries written
// before inline verdicts existed.
func phaseVerdicts(m map[string]any) []rawVerdict {
	raw, ok := m["lesson_verdicts"].([]any)
	if !ok {
		return nil
	}
	out := make([]rawVerdict, 0, len(raw))
	for _, item := range raw {
		vm, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, rawVerdict{handle: obsStr(vm, "handle"), verdict: obsStr(vm, "verdict")})
	}
	return out
}

// lessonIDFromHandle returns the bare lesson id for a "lesson:<id>" handle, or ""
// for a skill/other handle. A handle with no recognized prefix is treated as a
// bare lesson id (some models drop the prefix).
func lessonIDFromHandle(handle string) string {
	h := strings.TrimSpace(handle)
	if h == "" || strings.HasPrefix(h, "skill:") {
		return ""
	}
	return strings.TrimPrefix(h, lessonHandlePrefix)
}

// collectLessonVerdicts walks the run's phases in order and returns the verdict
// per lesson id, MOST-RECENT phase winning (by session_id, end_step). Only
// recognized verdicts are kept; unknown/empty are ignored.
func collectLessonVerdicts(phases []db.ObservationRecord) map[string]string {
	ordered := make([]db.ObservationRecord, len(phases))
	copy(ordered, phases)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].SessionID != ordered[j].SessionID {
			return ordered[i].SessionID < ordered[j].SessionID
		}
		return obsStep(ordered[i]) < obsStep(ordered[j])
	})
	out := map[string]string{}
	for _, p := range ordered {
		for _, v := range phaseVerdicts(p.Data) {
			id := lessonIDFromHandle(v.handle)
			if id == "" {
				continue
			}
			if _, ok := verdictDelta[v.verdict]; !ok {
				continue // unknown verdict; skip rather than default
			}
			out[id] = v.verdict // later phase overwrites earlier (most-recent wins)
		}
	}
	return out
}

func obsStep(rec db.ObservationRecord) int {
	if rec.Step != nil {
		return *rec.Step
	}
	return 0
}

func attributedIDs(ctx context.Context, store db.Store, runID string) (map[string]bool, error) {
	recs, err := store.GetObservations(ctx, runID, db.ObsFilter{Kind: lessonAttributionKind})
	if err != nil {
		return nil, err
	}
	out := make(map[string]bool, len(recs))
	for _, r := range recs {
		if id, ok := strings.CutPrefix(r.ObsID, lessonAttributionKind+"-"); ok {
			out[id] = true
		}
	}
	return out, nil
}

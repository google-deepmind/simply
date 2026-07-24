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
	"testing"
	"time"

	"amplio/internal/db"
	"amplio/internal/embed"
)

// seedVerdictLesson inserts a lesson and a phase_summary (steps 1-2 of main-agent)
// carrying an inline lesson_verdicts entry for it — the attribution input (the
// phase summarizer produces verdicts inline; attribution is a pure DB pass).
func seedVerdictLesson(t *testing.T, store db.Store, emb embed.Embedder, lessonID, verdict string) {
	t.Helper()
	ctx := context.Background()
	vecs, err := emb.Embed(ctx, []string{"a useful lesson"})
	must(t, err)
	must(t, store.InsertLesson(ctx, db.LessonRecord{
		LessonID: lessonID, Title: "L", Description: "a useful lesson", Body: "do the thing",
		Embedding: vecs[0], EmbedderID: emb.ModelID(),
	}))
	seedRun(t, store, "r", "task")
	must(t, store.CreateSession(ctx, db.SessionRecord{
		RunID: "r", SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionConcluded,
	}))
	addPhaseWithVerdicts(t, store, 1, 2, "Work", "did work",
		[]map[string]any{{"handle": "lesson:" + lessonID, "verdict": verdict, "reason": "acted on it"}})
}

// addPhaseWithVerdicts writes a phase_summary observation carrying a
// lesson_verdicts list (the attribution input).
func addPhaseWithVerdicts(t *testing.T, store db.Store, start, end int, title, summary string, verdicts []map[string]any) {
	t.Helper()
	const sid = "main-agent"
	step := end
	must(t, store.AppendObservation(context.Background(), db.ObservationRecord{
		ObsID:     fmt.Sprintf("phase_summary-%s-%d", sid, end),
		RunID:     "r",
		Kind:      "phase_summary",
		SessionID: sid,
		Step:      &step,
		Data: map[string]any{
			"title": title, "summary": summary, "start_step": start, "end_step": end,
			"lesson_verdicts": verdicts,
		},
		CreatedAt: time.Now().UTC(),
	}))
}
func TestAttributeLessons_HelpfulAndIdempotent(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	seedVerdictLesson(t, store, emb, "L1", "helpful")
	ix := builtLessonIndex(t, store, emb)
	deps := Deps{Store: store, LessonIndex: ix}

	n, err := AttributeLessons(ctx, deps, "r")
	must(t, err)
	if n != 1 {
		t.Fatalf("scored = %d, want 1", n)
	}
	l, err := store.GetLesson(ctx, "L1")
	must(t, err)
	if l.Score != 1 {
		t.Fatalf("score = %d, want +1 (helpful)", l.Score)
	}

	// Re-running is a no-op (per-(run,lesson) sentinel).
	n2, err := AttributeLessons(ctx, deps, "r")
	must(t, err)
	if n2 != 0 {
		t.Fatalf("re-attribute scored %d, want 0", n2)
	}
	if l2, _ := store.GetLesson(ctx, "L1"); l2.Score != 1 {
		t.Fatalf("score = %d after re-run, want 1", l2.Score)
	}
}

func TestAttributeLessons_Harmful(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	seedVerdictLesson(t, store, emb, "L2", "harmful")
	ix := builtLessonIndex(t, store, emb)
	deps := Deps{Store: store, LessonIndex: ix}

	n, err := AttributeLessons(ctx, deps, "r")
	must(t, err)
	if n != 1 {
		t.Fatalf("scored = %d, want 1", n)
	}
	l, err := store.GetLesson(ctx, "L2")
	must(t, err)
	if l.Score != -3 {
		t.Fatalf("score = %d, want -3 (harmful)", l.Score)
	}
}

func TestAttributeLessons_NoneLoaded(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	seedRun(t, store, "r", "task")
	must(t, store.CreateSession(ctx, db.SessionRecord{
		RunID: "r", SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionConcluded,
	}))
	// A phase with an empty lesson_verdicts list (nothing loaded).
	addPhaseWithVerdicts(t, store, 1, 1, "Phase", "did work", nil)
	ix := builtLessonIndex(t, store, emb)
	deps := Deps{Store: store, LessonIndex: ix}

	n, err := AttributeLessons(ctx, deps, "r")
	must(t, err)
	if n != 0 {
		t.Fatalf("scored = %d, want 0 (no lessons loaded)", n)
	}
}

// A lesson judged in two phases takes the MOST-RECENT phase's verdict (by
// end_step): an early "unhelpful" is overridden by a later "helpful".
func TestAttributeLessons_MostRecentPhaseWins(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	vecs, err := emb.Embed(ctx, []string{"a useful lesson"})
	must(t, err)
	must(t, store.InsertLesson(ctx, db.LessonRecord{
		LessonID: "L3", Title: "L", Description: "a useful lesson", Body: "do the thing",
		Embedding: vecs[0], EmbedderID: emb.ModelID(),
	}))
	seedRun(t, store, "r", "task")
	must(t, store.CreateSession(ctx, db.SessionRecord{
		RunID: "r", SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionConcluded,
	}))
	addPhaseWithVerdicts(t, store, 1, 2, "Early", "first try",
		[]map[string]any{{"handle": "lesson:L3", "verdict": "unhelpful"}})
	addPhaseWithVerdicts(t, store, 3, 4, "Later", "second try",
		[]map[string]any{{"handle": "lesson:L3", "verdict": "helpful"}})
	ix := builtLessonIndex(t, store, emb)
	deps := Deps{Store: store, LessonIndex: ix}

	n, err := AttributeLessons(ctx, deps, "r")
	must(t, err)
	if n != 1 {
		t.Fatalf("scored = %d, want 1", n)
	}
	l, err := store.GetLesson(ctx, "L3")
	must(t, err)
	if l.Score != 1 {
		t.Fatalf("score = %d, want +1 (later 'helpful' wins over earlier 'unhelpful')", l.Score)
	}
}

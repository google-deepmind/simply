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
	"encoding/json"
	"strings"
	"testing"

	"amplio/internal/db"
	"amplio/internal/embed"
	"amplio/internal/lessons"
	"amplio/internal/llm"
)

// mineMock answers the extraction and dedup calls (distinguished by the dedup
// system prompt mentioning "duplicate"). Both are plain (no-tool) JSON replies.
type mineMock struct {
	extraction string
	dedup      string
}

func (m mineMock) Call(_ context.Context, req llm.Request) (*llm.Response, error) {
	if strings.Contains(req.SystemPrompt, "duplicate") {
		return &llm.Response{Content: m.dedup}, nil
	}
	return &llm.Response{Content: m.extraction}, nil
}
func (mineMock) Stream(context.Context, llm.Request) (llm.Stream, error) { return nil, nil }
func (mineMock) ModelID() string                                         { return "mock-hq" }
func (mineMock) MaxTokens() int                                          { return 1000 }

func builtLessonIndex(t *testing.T, store db.Store, emb embed.Embedder) *lessons.Index {
	t.Helper()
	ix := lessons.NewIndex(store, emb)
	if err := ix.Build(context.Background()); err != nil {
		t.Fatal(err)
	}
	return ix
}

func miningReport() *RunReport {
	return &RunReport{Version: 1, Task: "do the thing", Phases: []ReportPhase{
		{SessionID: "main-agent", StartStep: 1, EndStep: 2, Title: "Work", Summary: "did work"},
	}}
}

func seedExistingLesson(t *testing.T, store db.Store, emb embed.Embedder, id, desc, body string) {
	t.Helper()
	ctx := context.Background()
	vecs, err := emb.Embed(ctx, []string{desc})
	must(t, err)
	must(t, store.InsertLesson(ctx, db.LessonRecord{
		LessonID: id, Title: "old title", Description: desc, Body: body,
		Embedding: vecs[0], EmbedderID: emb.ModelID(), SourceRunID: "old-run",
	}))
}

func TestMineLessons_Insert(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	ix := builtLessonIndex(t, store, emb)
	deps := Deps{Store: store, LessonIndex: ix, HQ: mineMock{
		extraction: `{"lessons":[{"title":"Retry flaky builds","description":"flaky build retry workaround","body":"rerun with --flaky_test_attempts"}]}`,
	}}

	n, err := MineLessons(ctx, deps, "run-1", miningReport())
	must(t, err)
	if n != 1 {
		t.Fatalf("inserted = %d, want 1", n)
	}
	all, err := store.ListAllLessons(ctx)
	must(t, err)
	if len(all) != 1 || all[0].Title != "Retry flaky builds" || all[0].SourceRunID != "run-1" {
		t.Fatalf("lessons = %+v", all)
	}
	if ix.Size() != 1 {
		t.Fatalf("index size = %d, want 1 (rebuilt after insert)", ix.Size())
	}
}

func TestMineLessons_Supersede(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	seedExistingLesson(t, store, emb, "sup", "flaky build retry workaround", "old retry body")
	ix := builtLessonIndex(t, store, emb)
	deps := Deps{Store: store, LessonIndex: ix, HQ: mineMock{
		extraction: `{"lessons":[{"title":"Better retry","description":"flaky build retry workaround","body":"new retry body"}]}`,
		dedup:      `{"verdict":"supersedes","reason":"adds detail"}`,
	}}

	n, err := MineLessons(ctx, deps, "run-2", miningReport())
	must(t, err)
	if n != 0 {
		t.Fatalf("inserted = %d, want 0 (superseded in place)", n)
	}
	all, err := store.ListAllLessons(ctx)
	must(t, err)
	if len(all) != 1 {
		t.Fatalf("want 1 lesson (replaced in place), got %d", len(all))
	}
	l, err := store.GetLesson(ctx, "sup")
	must(t, err)
	if l.Title != "Better retry" || l.Body != "new retry body" {
		t.Fatalf("lesson not superseded: %+v", l)
	}
}

func TestMineLessons_Duplicate(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	seedExistingLesson(t, store, emb, "dup", "spanner permission denied gotcha", "old spanner body")
	ix := builtLessonIndex(t, store, emb)
	deps := Deps{Store: store, LessonIndex: ix, HQ: mineMock{
		extraction: `{"lessons":[{"title":"Dup","description":"spanner permission denied gotcha","body":"same"}]}`,
		dedup:      `{"verdict":"duplicate","reason":"same knowledge"}`,
	}}

	n, err := MineLessons(ctx, deps, "run-3", miningReport())
	must(t, err)
	if n != 0 {
		t.Fatalf("inserted = %d, want 0 (duplicate dropped)", n)
	}
	l, err := store.GetLesson(ctx, "dup")
	must(t, err)
	if l.Title != "old title" || l.Body != "old spanner body" {
		t.Fatalf("duplicate should leave existing untouched: %+v", l)
	}
}

// combinedMock answers both the keen-critic agentic loop (request carries the
// investigation tools → end with the JSON report as content) and the mining
// calls (no tools → extraction/dedup JSON).
type combinedMock struct {
	summary    string
	extraction string
}

func (m combinedMock) Call(_ context.Context, req llm.Request) (*llm.Response, error) {
	if len(req.Tools) > 0 {
		// Keen-critic loop: terminate immediately with the JSON report (no tools).
		js, _ := json.Marshal(critique{Summary: m.summary})
		return &llm.Response{Content: string(js)}, nil
	}
	return &llm.Response{Content: m.extraction}, nil
}
func (combinedMock) Stream(context.Context, llm.Request) (llm.Stream, error) { return nil, nil }
func (combinedMock) ModelID() string                                         { return "mock-hq" }
func (combinedMock) MaxTokens() int                                          { return 1000 }

func TestFinalizer_MinesAfterReport(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "do the thing")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 2)
	addPhase(t, store, "r", "main-agent", 1, 2, "Work", "did work", nil)

	emb := embed.Mock{Dim: 4096}
	lessonIx := builtLessonIndex(t, store, emb)
	mock := combinedMock{
		summary:    "the run did stuff",
		extraction: `{"lessons":[{"title":"A lesson","description":"when X happens do Y","body":"do Y"}]}`,
	}
	fin := NewFinalizer(store, mock, nil, lessonIx, nil)

	fin.OnMainAgentConcluded(ctx, "r")

	reps, err := AllReports(ctx, store, "r")
	must(t, err)
	if len(reps) != 1 {
		t.Fatalf("reports = %d, want 1", len(reps))
	}
	all, err := store.ListAllLessons(ctx)
	must(t, err)
	if len(all) != 1 || all[0].Title != "A lesson" {
		t.Fatalf("mined lessons = %+v", all)
	}

	// Idempotent: a second finalize neither re-reports nor re-mines.
	fin.OnMainAgentConcluded(ctx, "r")
	if all2, _ := store.ListAllLessons(ctx); len(all2) != 1 {
		t.Fatalf("re-mined: %d lessons, want 1", len(all2))
	}
}

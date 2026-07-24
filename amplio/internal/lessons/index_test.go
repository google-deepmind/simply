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

package lessons

import (
	"context"
	"testing"

	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/embed"
)

func newStore(t *testing.T) db.Store {
	t.Helper()
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

// seedLesson inserts a lesson with an embedding computed from desc by emb, then
// applies the score counter.
func seedLesson(t *testing.T, store db.Store, emb embed.Embedder, id, desc string, score int) {
	t.Helper()
	ctx := context.Background()
	vecs, err := emb.Embed(ctx, []string{desc})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.InsertLesson(ctx, db.LessonRecord{
		LessonID: id, Title: id, Description: desc, Body: "body of " + id,
		Embedding: vecs[0], EmbedderID: emb.ModelID(),
	}); err != nil {
		t.Fatal(err)
	}
	if score != 0 {
		if err := store.AddToLessonScore(ctx, id, score); err != nil {
			t.Fatal(err)
		}
	}
}

func TestIndex_CosineRanking(t *testing.T) {
	store := newStore(t)
	emb := embed.Mock{Dim: 4096}
	seedLesson(t, store, emb, "apple", "apple apple apple", 0)
	seedLesson(t, store, emb, "banana", "banana banana banana", 0)

	ix := NewIndex(store, emb)
	if err := ix.Build(context.Background()); err != nil {
		t.Fatal(err)
	}
	if ix.Size() != 2 {
		t.Fatalf("size = %d, want 2", ix.Size())
	}

	hits, err := ix.Search(context.Background(), "apple", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].Lesson.LessonID != "apple" {
		t.Fatalf("hits = %+v, want apple first", hits)
	}
	if hits[0].Cosine <= hits[1].Cosine {
		t.Fatalf("apple cosine %v should exceed banana %v", hits[0].Cosine, hits[1].Cosine)
	}
}

func TestIndex_ConfidenceRanking(t *testing.T) {
	store := newStore(t)
	emb := embed.Mock{Dim: 4096}
	// Identical descriptions → identical cosine; the higher-score lesson wins on
	// the confidence multiplier.
	seedLesson(t, store, emb, "low", "alpha beta gamma", 0)
	seedLesson(t, store, emb, "high", "alpha beta gamma", 5)

	ix := NewIndex(store, emb)
	if err := ix.Build(context.Background()); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "alpha beta gamma", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].Lesson.LessonID != "high" {
		t.Fatalf("hits = %+v, want high-score first", hits)
	}
	if hits[0].Cosine != hits[1].Cosine {
		t.Fatalf("identical descriptions should share cosine: %v vs %v", hits[0].Cosine, hits[1].Cosine)
	}
	if hits[0].Score <= hits[1].Score {
		t.Fatalf("high-score lesson should rank higher: %v vs %v", hits[0].Score, hits[1].Score)
	}
}

func TestConfidenceMultiplier(t *testing.T) {
	if m := confidenceMultiplier(0, 0); m != 1.0 {
		t.Errorf("neutral lesson = %v, want 1.0", m)
	}
	// A positive score boosts; more loads damp the same score (1+4/2=3 vs 1+4/10=1.4).
	few, many := confidenceMultiplier(4, 0), confidenceMultiplier(4, 96)
	if !(few > many && many > 1) {
		t.Errorf("load damping: few-loads=%v should exceed many-loads=%v > 1", few, many)
	}
	// A very negative score floors at 0 (never flips a positive cosine).
	if m := confidenceMultiplier(-100, 0); m != 0 {
		t.Errorf("floor = %v, want 0", m)
	}
}

func TestIndex_LoadAndSkipMissingEmbedding(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	seedLesson(t, store, emb, "good", "hello world", 0)
	// A lesson with no embedding is skipped from the index.
	if err := store.InsertLesson(ctx, db.LessonRecord{LessonID: "noembed", Title: "x", Description: "y"}); err != nil {
		t.Fatal(err)
	}

	ix := NewIndex(store, emb)
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	if ix.Size() != 1 {
		t.Fatalf("size = %d, want 1 (noembed skipped)", ix.Size())
	}
	if _, ok := ix.Load("good"); !ok {
		t.Fatal("good not loadable")
	}
	if _, ok := ix.Load("noembed"); ok {
		t.Fatal("noembed should be absent")
	}
}

// TestIndex_SkipsEmptyEmbedderID guards Build's stricter "EmbedderID must
// exactly match the current embedder's ModelID" rule. A row that lost its
// embedder_id (older builds, cross-system import) is admitted by neither
// fresh-embedded queries nor dim-mismatched matrices; skipping it keeps the
// corpus clean until re-mining repopulates the field.
func TestIndex_SkipsEmptyEmbedderID(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	seedLesson(t, store, emb, "tagged", "alpha beta", 0)
	// Same vector dimensions, but blank embedder_id.
	vecs, err := emb.Embed(ctx, []string{"alpha beta"})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.InsertLesson(ctx, db.LessonRecord{
		LessonID: "untagged", Title: "x", Description: "y",
		Embedding: vecs[0], EmbedderID: "", // intentionally blank
	}); err != nil {
		t.Fatal(err)
	}
	ix := NewIndex(store, emb)
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	if ix.Size() != 1 {
		t.Fatalf("size = %d, want 1 (empty-EmbedderID row skipped)", ix.Size())
	}
	if _, ok := ix.Load("untagged"); ok {
		t.Fatal("untagged row should be skipped on empty EmbedderID")
	}
}

// TestIndex_DegenerateQueryReturnsEmpty guards the search-side bug the audit
// surfaced: a query that embeds to a zero vector (here: the empty string,
// which Mock.Embed deterministically maps to all-zeros) used to score every
// row at 0 and let an unstable sort pick arbitrary winners.
func TestIndex_DegenerateQueryReturnsEmpty(t *testing.T) {
	store := newStore(t)
	emb := embed.Mock{Dim: 4096}
	seedLesson(t, store, emb, "apple", "apple apple apple", 0)
	seedLesson(t, store, emb, "banana", "banana banana banana", 0)
	ix := NewIndex(store, emb)
	if err := ix.Build(context.Background()); err != nil {
		t.Fatal(err)
	}
	// Whitespace-only query: Mock.Embed → all-zero vector → SearchVec must bail.
	hits, err := ix.Search(context.Background(), "   ", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Errorf("degenerate query produced %d hits, want 0", len(hits))
	}
}

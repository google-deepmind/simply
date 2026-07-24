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
	"fmt"
	"log/slog"
	"math"
	"sort"
	"sync"

	"amplio/internal/db"
	"amplio/internal/embed"
	"amplio/internal/vec"
)

// confidenceK softens the usage normalization in the confidence multiplier, so a
// high score backed by few loads isn't over-trusted. Module constant, not a
// search parameter.
const confidenceK = 4.0

// Hit is one search result: the lesson, its raw cosine, and the final
// confidence-weighted score used for ranking.
type Hit struct {
	Lesson db.LessonRecord
	Cosine float64
	Score  float64
}

// Index is an in-memory cosine index over the Lesson table. Build (re)loads from
// the store; Search queries. Cheap to rebuild — lessons store their own
// embeddings, so Build does no embedding.
type Index struct {
	store    db.Store
	embedder embed.Embedder

	mu      sync.RWMutex
	lessons map[string]db.LessonRecord
	ids     []string    // row order
	matrix  [][]float32 // L2-normalized vectors, row per id
	built   bool
}

// NewIndex constructs an index; call Build to populate.
func NewIndex(store db.Store, embedder embed.Embedder) *Index {
	return &Index{store: store, embedder: embedder}
}

// Embedder returns the index's embedder (used by mining to embed candidate
// lessons with the same model the index searches against).
func (ix *Index) Embedder() embed.Embedder { return ix.embedder }

// Build loads all lessons and rebuilds the in-memory matrix. Rows with no
// embedding, or whose `embedder_id` does not exactly match the current
// embedder's ModelID, are skipped (logged) — they can't be compared against
// fresh query vectors. An empty `embedder_id` (rows from older builds or
// cross-system imports) is therefore also skipped: re-mining or re-embedding
// will populate the field on the next pass.
func (ix *Index) Build(ctx context.Context) error {
	recs, err := ix.store.ListAllLessons(ctx)
	if err != nil {
		return fmt.Errorf("list lessons: %w", err)
	}
	model := ix.embedder.ModelID()
	lessons := make(map[string]db.LessonRecord, len(recs))
	ids := make([]string, 0, len(recs))
	matrix := make([][]float32, 0, len(recs))
	skipped := 0
	for _, r := range recs {
		if len(r.Embedding) == 0 || r.EmbedderID != model {
			skipped++
			continue
		}
		lessons[r.LessonID] = r
		ids = append(ids, r.LessonID)
		matrix = append(matrix, vec.Normalize(r.Embedding))
	}

	ix.mu.Lock()
	ix.lessons, ix.ids, ix.matrix, ix.built = lessons, ids, matrix, true
	ix.mu.Unlock()

	if skipped > 0 {
		slog.Warn("lesson index: skipped rows with missing/mismatched embeddings", "skipped", skipped, "model", model)
	}
	slog.Info("lesson index built", "lessons", len(ids))
	return nil
}

// IsBuilt reports whether Build has completed.
func (ix *Index) IsBuilt() bool {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.built
}

// Size is the number of indexed lessons.
func (ix *Index) Size() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.ids)
}

// Load returns the lesson for id (in-memory; no DB read).
func (ix *Index) Load(id string) (db.LessonRecord, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	l, ok := ix.lessons[id]
	return l, ok
}

// RecordLoad persists a usage (loaded_count += 1) for a lesson. Best-effort: the
// caller (recall_load) returns the body regardless of a write failure. The
// in-memory count isn't bumped — it's only used by the next Build.
func (ix *Index) RecordLoad(ctx context.Context, id string) error {
	return ix.store.IncrementLessonLoadCount(ctx, id, 1)
}

// Search returns up to k lessons most relevant to query, ranked by cosine times
// a per-lesson confidence multiplier (descending). It embeds the query, then
// delegates to SearchVec.
func (ix *Index) Search(ctx context.Context, query string, k int) ([]Hit, error) {
	if k <= 0 || query == "" || ix.Size() == 0 {
		return nil, nil
	}
	vecs, err := ix.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	return ix.SearchVec(vecs[0], k), nil
}

// SearchVec ranks lessons against a precomputed query vector (no embedding) —
// used by mining for dedup against the existing corpus. A degenerate query
// vector (norm below vec.MinNorm — e.g. the embedder collapsed an empty or
// whitespace-only string) returns nil rather than every-row-tied-at-zero junk
// from an arbitrary unstable sort.
func (ix *Index) SearchVec(query []float32, k int) []Hit {
	ix.mu.RLock()
	ids, matrix, lessons := ix.ids, ix.matrix, ix.lessons
	ix.mu.RUnlock()

	if k <= 0 || len(ids) == 0 {
		return nil
	}
	qv := vec.NormalizeOrNil(query)
	if qv == nil {
		return nil
	}
	type scored struct {
		i      int
		cosine float64
		score  float64
	}
	scoredHits := make([]scored, len(ids))
	for i, row := range matrix {
		c := vec.Dot(row, qv)
		l := lessons[ids[i]]
		scoredHits[i] = scored{i, c, c * confidenceMultiplier(l.Score, l.LoadedCount)}
	}
	sort.Slice(scoredHits, func(a, b int) bool { return scoredHits[a].score > scoredHits[b].score })
	if k > len(scoredHits) {
		k = len(scoredHits)
	}
	out := make([]Hit, 0, k)
	for _, s := range scoredHits[:k] {
		out = append(out, Hit{Lesson: lessons[ids[s.i]], Cosine: s.cosine, Score: s.score})
	}
	return out
}

// confidenceMultiplier weights a lesson by its accumulated quality. A neutral
// lesson (score 0) is 1.0×; positive scores boost, negative scores demote toward
// (but never below) 0, so a bad lesson sinks in ranking but never flips a
// positive cosine negative.
func confidenceMultiplier(score, loadedCount int) float64 {
	m := 1 + float64(score)/math.Sqrt(float64(loadedCount)+confidenceK)
	if m < 0 {
		return 0
	}
	return m
}

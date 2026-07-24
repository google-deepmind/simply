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

package skills

import (
	"context"
	"log/slog"
	"sort"
	"sync"
	"time"

	"amplio/internal/embed"
	"amplio/internal/vec"
)

// Hit is one search result.
type Hit struct {
	Entry Entry
	Score float64 // cosine similarity in [-1, 1]
}

// Index is an in-memory cosine-similarity index over skill descriptions. Build
// once at startup (disk-cached embeddings); query repeatedly. Search is brute
// force over a few hundred rows — fast enough without a vector index.
type Index struct {
	sources  []Source
	embedder embed.Embedder
	cache    Cache

	mu      sync.RWMutex
	entries map[string]Entry
	names   []string    // row order
	matrix  [][]float32 // L2-normalized vectors, row per name
	built   bool
}

// NewIndex constructs an index; call Build to populate (does I/O + embeds).
func NewIndex(sources []Source, embedder embed.Embedder, cache Cache) *Index {
	return &Index{sources: sources, embedder: embedder, cache: cache}
}

func embedText(e Entry) string { return e.Name + ": " + e.Description }

// LoadCached hydrates the index purely from the persisted cache — no file scan,
// no embedding API. Fast: just a single DB read + matrix construction. After
// this returns successfully with cached entries, IsBuilt() reports true and
// Search/Load serve from the cached snapshot.
//
// Intended as Stage 1 of a two-stage startup: call this synchronously so the
// agent has recall immediately, then call Build in a goroutine to reconcile
// the in-memory state against the current on-disk corpus. The atomic swap at
// Build's end replaces this snapshot with the up-to-date one.
//
// Returns the number of entries hydrated (0 means the cache was empty — the
// caller should treat this as cold-start and Build synchronously). A failure
// to load the cache is logged-and-ignored; the index stays empty and Build
// will recover from disk.
func (ix *Index) LoadCached(ctx context.Context) int {
	cached, err := ix.cache.Load(ctx, ix.embedder.ModelID())
	if err != nil {
		slog.Warn("skill cache hydrate failed; starting empty", "error", err)
		return 0
	}
	if len(cached) == 0 {
		return 0
	}
	// Sort by name for stable ordering (the file-scan path uses a sort, too —
	// keeping both deterministic means Search results don't churn between
	// stages 1 and 2 just because of insertion order).
	names := make([]string, 0, len(cached))
	for n := range cached {
		names = append(names, n)
	}
	sort.Strings(names)
	entryMap := make(map[string]Entry, len(cached))
	matrix := make([][]float32, 0, len(cached))
	for _, n := range names {
		c := cached[n]
		entryMap[n] = Entry{
			Name:        n,
			Description: c.Description,
			Body:        c.Body,
			Path:        c.Path,
			ContentHash: c.Hash,
		}
		matrix = append(matrix, vec.Normalize(c.Vector))
	}
	ix.mu.Lock()
	ix.entries, ix.names, ix.matrix, ix.built = entryMap, names, matrix, true
	ix.mu.Unlock()
	slog.Info("skill index hydrated from cache (background reconcile to follow)", "skills", len(names))
	return len(names)
}

// Build scans the sources, embeds new/changed skills (reusing cached vectors for
// unchanged ones), persists the cache, and swaps in the in-memory state. It
// retries a transient embedding failure (skills matter) up to maxAttempts before
// returning the error; the caller decides whether to fail or degrade to no recall.
func (ix *Index) Build(ctx context.Context) error {
	const maxAttempts = 3
	var err error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err = ix.buildOnce(ctx); err == nil {
			return nil
		}
		slog.Warn("skill index build attempt failed", "attempt", attempt, "max", maxAttempts, "error", err)
		if attempt == maxAttempts {
			break
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt) * 2 * time.Second):
		}
	}
	return err
}

func (ix *Index) buildOnce(ctx context.Context) error {
	entries := scanSources(ix.sources)
	model := ix.embedder.ModelID()

	cached, err := ix.cache.Load(ctx, model)
	if err != nil {
		slog.Warn("skill cache load failed; re-embedding all", "error", err)
		cached = nil
	}

	var toEmbedNames, toEmbedTexts []string
	reuse := make(map[string][]float32)
	for _, e := range entries {
		if c, ok := cached[e.Name]; ok && c.Hash == e.ContentHash {
			reuse[e.Name] = c.Vector
			continue
		}
		toEmbedNames = append(toEmbedNames, e.Name)
		toEmbedTexts = append(toEmbedTexts, embedText(e))
	}

	fresh := make(map[string][]float32)
	if len(toEmbedTexts) > 0 {
		vecs, err := ix.embedder.Embed(ctx, toEmbedTexts)
		if err != nil {
			return err
		}
		for i, n := range toEmbedNames {
			fresh[n] = vecs[i]
		}
	}

	entryMap := make(map[string]Entry, len(entries))
	names := make([]string, 0, len(entries))
	matrix := make([][]float32, 0, len(entries))
	save := make(map[string]CacheEntry, len(entries))
	for _, e := range entries {
		embVec, ok := reuse[e.Name]
		if !ok {
			embVec, ok = fresh[e.Name]
		}
		if !ok {
			continue // unreachable: every entry was reused or embedded
		}
		entryMap[e.Name] = e
		names = append(names, e.Name)
		matrix = append(matrix, vec.Normalize(embVec))
		save[e.Name] = CacheEntry{
			Hash:        e.ContentHash,
			Vector:      embVec,
			Description: e.Description,
			Path:        e.Path,
			Body:        e.Body,
		}
	}

	if err := ix.cache.Save(ctx, model, save); err != nil {
		slog.Warn("skill cache save failed", "error", err)
	}

	ix.mu.Lock()
	ix.entries, ix.names, ix.matrix, ix.built = entryMap, names, matrix, true
	ix.mu.Unlock()
	slog.Info("skill index built", "skills", len(names), "embedded", len(fresh), "reused", len(reuse))
	return nil
}

// IsBuilt reports whether Build has completed.
func (ix *Index) IsBuilt() bool {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.built
}

// Size is the number of indexed skills.
func (ix *Index) Size() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.names)
}

// Load returns the parsed entry for name (in-memory; no file read).
func (ix *Index) Load(name string) (Entry, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	e, ok := ix.entries[name]
	return e, ok
}

// Search returns up to k skills most similar to query, by cosine descending.
// A degenerate query vector (norm below vec.MinNorm — e.g. the embedder
// collapsed a whitespace-only string) returns nil rather than every-row-
// tied-at-zero junk from an arbitrary unstable sort.
func (ix *Index) Search(ctx context.Context, query string, k int) ([]Hit, error) {
	ix.mu.RLock()
	names, matrix, entries := ix.names, ix.matrix, ix.entries
	ix.mu.RUnlock()

	if k <= 0 || len(names) == 0 || query == "" {
		return nil, nil
	}
	vecs, err := ix.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	qv := vec.NormalizeOrNil(vecs[0])
	if qv == nil {
		return nil, nil
	}
	type scored struct {
		i     int
		score float64
	}
	scoredHits := make([]scored, len(names))
	for i, row := range matrix {
		scoredHits[i] = scored{i, vec.Dot(row, qv)}
	}
	sort.Slice(scoredHits, func(a, b int) bool { return scoredHits[a].score > scoredHits[b].score })
	if k > len(scoredHits) {
		k = len(scoredHits)
	}
	out := make([]Hit, 0, k)
	for _, s := range scoredHits[:k] {
		out = append(out, Hit{Entry: entries[names[s.i]], Score: s.score})
	}
	return out, nil
}

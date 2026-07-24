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

	"amplio/internal/db"
)

// CacheEntry is a cached skill — embedding + the parsed SKILL.md fields the
// Index needs. ContentHash invalidates per-skill (mismatch → re-scan + re-embed
// that one). Description/Path/Body let the Index hydrate fully from this cache
// at startup, so recall_search and recall_load are useful before the
// background file-scan reconcile completes.
type CacheEntry struct {
	Hash        string
	Vector      []float32
	Description string
	Path        string
	Body        string
}

// Cache persists skill embeddings per embedder model so unchanged skills aren't
// re-embedded on every startup.
type Cache interface {
	Load(ctx context.Context, model string) (map[string]CacheEntry, error)
	Save(ctx context.Context, model string, entries map[string]CacheEntry) error
}

// dbCache stores skill vectors in the SQLite store (one table, keyed by
// model+name). Decoupled from the index via the Cache interface for testing.
type dbCache struct{ store db.Store }

// NewDBCache returns a Cache backed by the given store.
func NewDBCache(store db.Store) Cache { return dbCache{store: store} }

func (c dbCache) Load(ctx context.Context, model string) (map[string]CacheEntry, error) {
	vecs, err := c.store.GetSkillVectors(ctx, model)
	if err != nil {
		return nil, err
	}
	out := make(map[string]CacheEntry, len(vecs))
	for _, v := range vecs {
		out[v.Name] = CacheEntry{
			Hash:        v.ContentHash,
			Vector:      v.Vector,
			Description: v.Description,
			Path:        v.Path,
			Body:        v.Body,
		}
	}
	return out, nil
}

func (c dbCache) Save(ctx context.Context, model string, entries map[string]CacheEntry) error {
	vecs := make([]db.SkillVector, 0, len(entries))
	for name, e := range entries {
		vecs = append(vecs, db.SkillVector{
			Name:        name,
			ContentHash: e.Hash,
			Vector:      e.Vector,
			Description: e.Description,
			Path:        e.Path,
			Body:        e.Body,
		})
	}
	return c.store.PutSkillVectors(ctx, model, vecs)
}

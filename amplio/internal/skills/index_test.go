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
	"os"
	"path/filepath"
	"testing"

	"amplio/internal/db/sqlite"
	"amplio/internal/embed"
)

func writeSkill(t *testing.T, dir, name, desc, body string) {
	t.Helper()
	d := filepath.Join(dir, name)
	if err := os.MkdirAll(d, 0o700); err != nil {
		t.Fatal(err)
	}
	content := "---\nname: " + name + "\ndescription: " + desc + "\n---\n\n" + body
	if err := os.WriteFile(filepath.Join(d, "SKILL.md"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

// countingEmbedder wraps an embedder and counts how many texts it embeds.
type countingEmbedder struct {
	inner embed.Embedder
	calls int
}

func (c *countingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	c.calls += len(texts)
	return c.inner.Embed(ctx, texts)
}
func (c *countingEmbedder) ModelID() string { return c.inner.ModelID() }

func TestIndex_BuildSearchLoad(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "bazel", "build run and test code with bazel", "# Bazel\nbody")
	writeSkill(t, dir, "spanner", "query spanner sql databases", "# Spanner\nbody")
	writeSkill(t, dir, "gmail", "read and send email messages", "# Gmail\nbody")
	// A directory without a SKILL.md must be ignored, not fatal.
	if err := os.MkdirAll(filepath.Join(dir, "empty"), 0o700); err != nil {
		t.Fatal(err)
	}

	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()

	emb1 := &countingEmbedder{inner: embed.Mock{}}
	ix := NewIndex([]Source{{Name: "test", Path: dir}}, emb1, NewDBCache(store))
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	if !ix.IsBuilt() || ix.Size() != 3 {
		t.Fatalf("built=%v size=%d, want true/3", ix.IsBuilt(), ix.Size())
	}
	if emb1.calls != 3 {
		t.Fatalf("cold build embedded %d texts, want 3", emb1.calls)
	}

	hits, err := ix.Search(ctx, "query spanner sql", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Entry.Name != "spanner" {
		t.Fatalf("search hits=%+v, want spanner first", hits)
	}

	if e, ok := ix.Load("bazel"); !ok || e.Description == "" {
		t.Fatalf("load bazel: %+v ok=%v", e, ok)
	}

	// A fresh index over the SAME store reuses cached vectors (no re-embed).
	emb2 := &countingEmbedder{inner: embed.Mock{}}
	ix2 := NewIndex([]Source{{Name: "test", Path: dir}}, emb2, NewDBCache(store))
	if err := ix2.Build(ctx); err != nil {
		t.Fatal(err)
	}
	if emb2.calls != 0 {
		t.Fatalf("warm build embedded %d texts, want 0 (cache reuse)", emb2.calls)
	}
	if ix2.Size() != 3 {
		t.Fatalf("warm build size=%d, want 3", ix2.Size())
	}
}

// TestIndex_DegenerateQueryReturnsEmpty guards the search-side bug the audit
// surfaced: a query that embeds to a zero vector (here: whitespace-only, which
// Mock.Embed deterministically maps to all-zeros) used to score every row at 0
// and let an unstable sort pick arbitrary winners.
func TestIndex_DegenerateQueryReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "bazel", "build run and test code with bazel", "# Bazel\nbody")
	writeSkill(t, dir, "spanner", "query spanner sql databases", "# Spanner\nbody")

	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()

	ix := NewIndex([]Source{{Name: "test", Path: dir}}, embed.Mock{}, NewDBCache(store))
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	// Whitespace-only query: Mock.Embed → all-zero vector → Search must bail.
	hits, err := ix.Search(ctx, "   ", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Errorf("degenerate query produced %d hits, want 0", len(hits))
	}
}

func TestParseSkillMD(t *testing.T) {
	raw := []byte("---\nname: foo\ndescription: >-\n  multi line\n  description here\n---\n\n# Body\ncontent\n")
	e, ok := parseSkillMD("/x/SKILL.md", raw)
	if !ok {
		t.Fatal("parse failed")
	}
	if e.Name != "foo" || e.Description != "multi line description here" {
		t.Fatalf("entry = %+v", e)
	}
	if e.Body != "# Body\ncontent\n" {
		t.Fatalf("body = %q", e.Body)
	}
	// Malformed (no frontmatter) → not ok, no panic.
	if _, ok := parseSkillMD("/x", []byte("no frontmatter")); ok {
		t.Error("expected parse failure for missing frontmatter")
	}
}

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

package plain

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"amplio/internal/workspace"
)

func TestDescribe(t *testing.T) {
	ctx := context.Background()
	if got := New("/a/b/repo").Describe(ctx, ""); got != "Working directory: `/a/b/repo`" {
		t.Errorf("root describe = %q", got)
	}
	sub := New("/a/b/repo").Describe(ctx, "parent-1")
	if !strings.Contains(sub, "shared with the parent session `parent-1`") {
		t.Errorf("sub-agent describe missing shared provenance: %q", sub)
	}
}

func TestRoundTrip(t *testing.T) {
	w := New("/tmp")
	blob, err := workspace.Marshal(w)
	if err != nil {
		t.Fatal(err)
	}
	got, err := workspace.Unmarshal(blob)
	if err != nil {
		t.Fatal(err)
	}
	if got.Kind() != Kind || got.Root() != "/tmp" {
		t.Errorf("round-trip mismatch: kind=%q root=%q", got.Kind(), got.Root())
	}
}

func TestValidate(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	if err := New(dir).Validate(ctx); err != nil {
		t.Errorf("Validate(existing dir) = %v, want nil", err)
	}
	missing := filepath.Join(dir, "does-not-exist")
	if err := New(missing).Validate(ctx); err == nil {
		t.Error("Validate(missing dir) = nil, want error")
	}
	// A file (not a directory) must fail validation.
	f := filepath.Join(dir, "afile")
	if err := os.WriteFile(f, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := New(f).Validate(ctx); err == nil {
		t.Error("Validate(file) = nil, want error")
	}
}

func TestNameAndLinking(t *testing.T) {
	w := New("/a/b/myrepo")
	if w.Name() != "myrepo" {
		t.Errorf("Name() = %q, want myrepo", w.Name())
	}
	if _, err := w.CreateLinked(context.Background(), "child-1"); err == nil {
		t.Error("CreateLinked should be unsupported for plain workspaces")
	}
	if alias, err := w.ResolveAlias(context.Background()); err != nil || alias != "" {
		t.Errorf("ResolveAlias() = %q, %v; want \"\", nil", alias, err)
	}
}

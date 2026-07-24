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

package jj

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"amplio/internal/workspace"
)

func requireJJ(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("jj"); err != nil {
		t.Skip("jj not available")
	}
}

// newRepo creates an initialized jj repo at <tempdir>/<name> (a subdir of
// t.TempDir so the sibling "<name>-amplio" worktrees dir is cleaned up too).
func newRepo(t *testing.T, name string) string {
	t.Helper()
	ctx := context.Background()
	// Canonicalize the temp root: on macOS t.TempDir() lives under /var, a
	// symlink to /private/var, and jj/git report the resolved path from the
	// worktree. Resolving here keeps repo (and the paths derived from it) equal
	// to what the subprocess reports.
	tmp, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(tmp, name)
	if err := os.Mkdir(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := run(ctx, repo, "git", "init"); err != nil {
		t.Skipf("jj git init unsupported in this jj version: %v", err)
	}
	for _, kv := range [][2]string{{"user.email", "t@example.com"}, {"user.name", "tester"}} {
		_, _ = run(ctx, repo, "config", "set", "--repo", kv[0], kv[1])
	}
	return repo
}

func TestDetectAndRoundTrip(t *testing.T) {
	requireJJ(t)
	ctx := context.Background()

	if _, ok := Detect(ctx, t.TempDir()); ok {
		t.Error("non-jj dir detected as jj")
	}

	repo := newRepo(t, "myrepo")
	ws, ok := Detect(ctx, repo)
	if !ok {
		t.Fatal("jj repo not detected")
	}
	if ws.Kind() != Kind || ws.Root() != repo {
		t.Errorf("detected kind=%q root=%q, want jj %q", ws.Kind(), ws.Root(), repo)
	}

	blob, _ := workspace.Marshal(ws)
	got, err := workspace.Unmarshal(blob)
	if err != nil {
		t.Fatal(err)
	}
	if got.Root() != repo || got.Kind() != Kind {
		t.Errorf("round-trip kind=%q root=%q", got.Kind(), got.Root())
	}
}

func TestCreateLinked(t *testing.T) {
	requireJJ(t)
	ctx := context.Background()
	repo := newRepo(t, "myrepo")
	ws, _ := Detect(ctx, repo)

	child, err := ws.CreateLinked(ctx, "agent-1")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(filepath.Dir(repo), "myrepo-amplio", "worktrees", "agent-1")
	if child.Root() != want {
		t.Errorf("child root = %q, want %q", child.Root(), want)
	}
	if _, ok := Detect(ctx, child.Root()); !ok {
		t.Error("child is not a jj workspace")
	}
	if child.LinkedFrom() != repo {
		t.Errorf("LinkedFrom = %q, want %q", child.LinkedFrom(), repo)
	}
}

func TestDescribe(t *testing.T) {
	ctx := context.Background()
	root := (&jjWorkspace{Dir: "/r"}).Describe(ctx, "")
	if !strings.Contains(root, "VCS: Jujutsu (`jj`).") {
		t.Errorf("root describe = %q", root)
	}
	linked := (&jjWorkspace{Dir: "/r/wt", From: "/r"}).Describe(ctx, "p")
	if !strings.Contains(linked, "linked from the workspace of the parent session `p`") ||
		!strings.Contains(linked, "jj log") {
		t.Errorf("linked describe = %q", linked)
	}
}

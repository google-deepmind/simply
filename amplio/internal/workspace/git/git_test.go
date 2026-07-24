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

package git

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"amplio/internal/workspace"
)

func requireGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
}

// newRepo creates an initialized git repo (one commit) at <tempdir>/<name> and
// returns its path. The repo is a SUBDIR of t.TempDir so the sibling
// "<name>-amplio" worktrees dir created by CreateLinked is also cleaned up.
func newRepo(t *testing.T, name string) string {
	t.Helper()
	ctx := context.Background()
	// Canonicalize the temp root: on macOS t.TempDir() lives under /var, a
	// symlink to /private/var, so paths the git subprocess resolves could differ
	// from t.TempDir()-derived expectations. Resolving here keeps them equal.
	tmp, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(tmp, name)
	if err := os.Mkdir(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"init"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "tester"},
	} {
		if _, err := run(ctx, repo, args...); err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
	}
	if err := os.WriteFile(filepath.Join(repo, "f.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"add", "."}, {"commit", "-m", "init"}} {
		if _, err := run(ctx, repo, args...); err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
	}
	return repo
}

func TestDetectAndRoundTrip(t *testing.T) {
	requireGit(t)
	ctx := context.Background()

	// A non-git directory is not detected.
	if _, ok := Detect(ctx, t.TempDir()); ok {
		t.Error("non-git dir detected as git")
	}

	repo := newRepo(t, "myrepo")
	ws, ok := Detect(ctx, repo)
	if !ok {
		t.Fatal("git repo not detected")
	}
	if ws.Kind() != Kind || ws.Root() != repo {
		t.Errorf("detected ws kind=%q root=%q, want git %q", ws.Kind(), ws.Root(), repo)
	}

	blob, err := workspace.Marshal(ws)
	if err != nil {
		t.Fatal(err)
	}
	got, err := workspace.Unmarshal(blob)
	if err != nil {
		t.Fatal(err)
	}
	if got.Root() != repo || got.Kind() != Kind {
		t.Errorf("round-trip mismatch: kind=%q root=%q", got.Kind(), got.Root())
	}
}

func TestCreateLinked(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	repo := newRepo(t, "myrepo")
	ws, _ := Detect(ctx, repo)

	child, err := ws.CreateLinked(ctx, "agent-1")
	if err != nil {
		t.Fatal(err)
	}

	// Placement: <repoParent>/<repo>-amplio/worktrees/agent-1.
	want := filepath.Join(filepath.Dir(repo), "myrepo-amplio", "worktrees", "agent-1")
	if child.Root() != want {
		t.Errorf("child root = %q, want %q", child.Root(), want)
	}
	// The child is a real git worktree.
	if _, ok := Detect(ctx, child.Root()); !ok {
		t.Error("child is not a git worktree")
	}
	// It shares history with the parent (same HEAD commit).
	parentHead, _ := run(ctx, repo, "rev-parse", "HEAD")
	childHead, _ := run(ctx, child.Root(), "rev-parse", "HEAD")
	if parentHead != childHead {
		t.Errorf("child HEAD %q != parent HEAD %q", childHead, parentHead)
	}
	// Provenance: the child records the parent it was linked from; the parent
	// (a detected root) records nothing. Survives a JSON round-trip.
	if child.LinkedFrom() != repo {
		t.Errorf("child LinkedFrom = %q, want %q", child.LinkedFrom(), repo)
	}
	if ws.LinkedFrom() != "" {
		t.Errorf("root LinkedFrom = %q, want \"\"", ws.LinkedFrom())
	}
	blob, _ := workspace.Marshal(child)
	got, _ := workspace.Unmarshal(blob)
	if got.LinkedFrom() != repo {
		t.Errorf("round-tripped LinkedFrom = %q, want %q", got.LinkedFrom(), repo)
	}
	// The worktree lives OUTSIDE the repo (no .gitignore impact).
	if rel, err := filepath.Rel(repo, child.Root()); err == nil && !filepath.IsAbs(rel) && rel[0] != '.' {
		t.Errorf("worktree %q is inside the repo %q", child.Root(), repo)
	}
}

func TestCreateLinkedConcurrent(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	repo := newRepo(t, "myrepo")
	ws, _ := Detect(ctx, repo)

	const n = 8
	errs := make(chan error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := ws.CreateLinked(ctx, fmt.Sprintf("agent-%d", i))
			errs <- err
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("concurrent CreateLinked failed: %v", err)
		}
	}
}

func TestDescribe(t *testing.T) {
	ctx := context.Background()
	root := (&gitWorkspace{Dir: "/r"}).Describe(ctx, "")
	if !strings.Contains(root, "Working directory: `/r`") || !strings.Contains(root, "VCS: Git.") {
		t.Errorf("root describe = %q", root)
	}
	if strings.Contains(root, "parent session") {
		t.Errorf("root describe should have no provenance: %q", root)
	}
	linked := (&gitWorkspace{Dir: "/r/wt", From: "/r"}).Describe(ctx, "p")
	if !strings.Contains(linked, "linked from the workspace of the parent session `p`") ||
		!strings.Contains(linked, "git log --all") {
		t.Errorf("linked describe = %q", linked)
	}
}

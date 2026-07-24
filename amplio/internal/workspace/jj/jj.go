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
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"amplio/internal/workspace"
)

// Kind is the JSON discriminator for OSS jj workspaces.
const Kind = "jj"

func init() { workspace.RegisterKind(Kind, unmarshal) }

var _ workspace.Workspace = (*jjWorkspace)(nil)

// linkLocks serializes `jj workspace add` per shared repo (keyed by the repo
// store path). jj is the backend that actually NEEDS this: concurrent
// `workspace add`s on one repo can leave divergent operation heads.
var linkLocks = workspace.NewLinkLocks()

type jjWorkspace struct {
	Dir  string `json:"root"`
	From string `json:"linked_from,omitempty"` // parent Root for a linked workspace
}

// New wraps an existing jj working-copy path (resolved to absolute).
func New(dir string) workspace.Workspace {
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	return &jjWorkspace{Dir: abs}
}

// Detect returns a jj workspace for path if it is inside a jj repo, else
// ok=false. Checked BEFORE git so a jj repo colocated with git (the common OSS
// setup) is treated as jj. Returns false (not an error) when jj is absent or the
// path isn't a jj repo.
func Detect(ctx context.Context, path string) (workspace.Workspace, bool) {
	if _, err := run(ctx, path, "root"); err != nil {
		return nil, false
	}
	return New(path), true
}

func (w *jjWorkspace) Root() string       { return w.Dir }
func (w *jjWorkspace) Name() string       { return filepath.Base(w.Dir) }
func (w *jjWorkspace) Kind() string       { return Kind }
func (w *jjWorkspace) LinkedFrom() string { return w.From }

// Validate confirms the working-copy path still exists and is a directory (no
// VCS re-check, per the workspace contract).
func (w *jjWorkspace) Validate(context.Context) error {
	info, err := os.Stat(w.Dir)
	if err != nil {
		return fmt.Errorf("workspace %q: %w", w.Dir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("workspace %q is not a directory", w.Dir)
	}
	return nil
}

// ResolveAlias returns "" — jj workspaces have no alias concept.
func (w *jjWorkspace) ResolveAlias(context.Context) (string, error) { return "", nil }

// Describe reports the working directory, jj as the VCS, and (for a linked
// sub-agent) its shared-history provenance.
func (w *jjWorkspace) Describe(_ context.Context, parentSessionID string) string {
	lines := []string{
		fmt.Sprintf("Working directory: `%s`", w.Dir),
		"VCS: Jujutsu (`jj`).",
	}
	if p := workspace.ProvenanceLine(parentSessionID, w.From != "", "jj log"); p != "" {
		lines = append(lines, p)
	}
	return strings.Join(lines, "\n")
}

// CreateLinked adds an isolated jj workspace sharing the repo, named after
// childSessionID, at `<mainRepoParent>/<repo>-amplio/worktrees/<sessionID>`
// (same drive, outside the repo). `jj workspace add` is serialized per repo to
// avoid divergent operation heads, and snapshot.auto-update-stale is enabled so
// concurrent workspaces don't trip each other into stale errors.
func (w *jjWorkspace) CreateLinked(ctx context.Context, childSessionID string) (workspace.Workspace, error) {
	currentRoot, err := w.workspaceRoot(ctx)
	if err != nil {
		return nil, err
	}
	mainRoot, store := w.repoAnchor(ctx, currentRoot)

	subpath, err := filepath.Rel(currentRoot, w.Dir)
	if err != nil || strings.HasPrefix(subpath, "..") {
		subpath = "."
	}
	baseDir := filepath.Join(filepath.Dir(mainRoot), filepath.Base(mainRoot)+"-amplio", "worktrees")

	release := linkLocks.Acquire(store)
	defer release()

	placement := workspace.FreeSiblingPath(baseDir, childSessionID)
	if err := os.MkdirAll(filepath.Dir(placement), 0o755); err != nil {
		return nil, fmt.Errorf("jj link: create worktree parent: %w", err)
	}
	if _, err := run(ctx, w.Dir, "workspace", "add", placement); err != nil {
		return nil, fmt.Errorf("jj link: %w", err)
	}
	// Recommended for multi-workspace repos so a rewrite in one workspace doesn't
	// error others. Best-effort: a failure here doesn't invalidate the workspace.
	_, _ = run(ctx, w.Dir, "config", "set", "--repo", "snapshot.auto-update-stale", "true")

	childRoot := placement
	if subpath != "." {
		childRoot = filepath.Join(placement, subpath)
	}
	return &jjWorkspace{Dir: childRoot, From: w.Dir}, nil
}

// workspaceRoot returns the absolute root of the CURRENT jj workspace.
func (w *jjWorkspace) workspaceRoot(ctx context.Context) (string, error) {
	out, err := run(ctx, w.Dir, "root")
	if err != nil {
		return "", fmt.Errorf("jj link: resolve workspace root: %w", err)
	}
	return strings.TrimSpace(out), nil
}

// repoAnchor resolves the main workspace root (placement anchor) and the repo
// store path (lock key) from the current workspace root, by reading the jj
// `.jj/repo` marker: a directory in the main workspace, or a file pointing to it
// in a secondary workspace. Falls back to (currentRoot, currentRoot) if the
// layout can't be read — which still serializes the dominant fan-out (one parent
// spawning many children share currentRoot).
func (w *jjWorkspace) repoAnchor(_ context.Context, currentRoot string) (mainRoot, store string) {
	marker := filepath.Join(currentRoot, ".jj", "repo")
	info, err := os.Stat(marker)
	if err != nil {
		return currentRoot, currentRoot
	}
	if info.IsDir() {
		return currentRoot, marker
	}
	data, err := os.ReadFile(marker)
	if err != nil {
		return currentRoot, currentRoot
	}
	store = strings.TrimSpace(string(data))
	if !filepath.IsAbs(store) {
		store = filepath.Join(currentRoot, ".jj", store)
	}
	store = filepath.Clean(store)
	// store is <mainRoot>/.jj/repo → mainRoot is two levels up.
	return filepath.Dir(filepath.Dir(store)), store
}

func run(ctx context.Context, dir string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "jj", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("jj %s: %w: %s",
			strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func unmarshal(data []byte) (workspace.Workspace, error) {
	var w jjWorkspace
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, err
	}
	if w.Dir == "" {
		return nil, fmt.Errorf("jj workspace: missing root")
	}
	return &w, nil
}

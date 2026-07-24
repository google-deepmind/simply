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
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"amplio/internal/workspace"
)

// Kind is the JSON discriminator for git workspaces.
const Kind = "git"

func init() { workspace.RegisterKind(Kind, unmarshal) }

var _ workspace.Workspace = (*gitWorkspace)(nil)

// linkLocks serializes `git worktree add` per shared repo (keyed by the common
// git dir), so concurrent link-mode spawns off one repo can't race on repo
// locks. git tends to error-and-retry rather than corrupt, but serializing
// creation is cheap and uniform with the other backends.
var linkLocks = workspace.NewLinkLocks()

// gitWorkspace is the serialized form: the working-tree path the agent is rooted
// at (which may be a sub-folder of the worktree), plus the parent's root for a
// linked worktree (provenance). The repo root and common git dir are resolved
// live via git, so the struct stays minimal and stable across moves.
type gitWorkspace struct {
	Dir  string `json:"root"`
	From string `json:"linked_from,omitempty"` // parent Root for a linked worktree; "" otherwise
}

// New wraps an existing git working tree path (resolved to absolute).
func New(dir string) workspace.Workspace {
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	return &gitWorkspace{Dir: abs}
}

// Detect returns a git workspace for path if it is inside a (non-bare) git work
// tree, else ok=false. Cheap: one `git rev-parse`. Returns false (not an error)
// when git is absent or the path isn't a repo, so callers fall back to plain.
func Detect(ctx context.Context, path string) (workspace.Workspace, bool) {
	out, err := run(ctx, path, "rev-parse", "--is-inside-work-tree")
	if err != nil || strings.TrimSpace(out) != "true" {
		return nil, false
	}
	return New(path), true
}

func (w *gitWorkspace) Root() string       { return w.Dir }
func (w *gitWorkspace) Name() string       { return filepath.Base(w.Dir) }
func (w *gitWorkspace) Kind() string       { return Kind }
func (w *gitWorkspace) LinkedFrom() string { return w.From }

// Validate confirms the working tree path still exists and is a directory.
// Per the workspace contract it does NOT re-verify the VCS kind (a path that
// stopped being a git repo would surface a clear error at the next git op).
func (w *gitWorkspace) Validate(context.Context) error {
	info, err := os.Stat(w.Dir)
	if err != nil {
		return fmt.Errorf("workspace %q: %w", w.Dir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("workspace %q is not a directory", w.Dir)
	}
	return nil
}

// ResolveAlias returns "" — git workspaces have no alias concept.
func (w *gitWorkspace) ResolveAlias(context.Context) (string, error) { return "", nil }

// Describe reports the working directory, Git as the VCS, and (for a linked
// sub-agent) its shared-history provenance.
func (w *gitWorkspace) Describe(_ context.Context, parentSessionID string) string {
	lines := []string{
		fmt.Sprintf("Working directory: `%s`", w.Dir),
		"VCS: Git.",
	}
	if p := workspace.ProvenanceLine(parentSessionID, w.From != "", "git log --all"); p != "" {
		lines = append(lines, p)
	}
	return strings.Join(lines, "\n")
}

// CreateLinked creates an isolated git worktree (detached at the parent's
// current HEAD) sharing the repo's object store, named after childSessionID. It
// lives at `<mainRepoParent>/<repo>-amplio/worktrees/<childSessionID>` (on the
// repo's own drive, never inside the repo, so no .gitignore change is needed),
// and the child is rooted at the same sub-folder the parent was, if any.
func (w *gitWorkspace) CreateLinked(ctx context.Context, childSessionID string) (workspace.Workspace, error) {
	// Resolve the shared repo (the lock key) and the main worktree root (the
	// placement anchor) — both derived from the common git dir, so nested links
	// all group under the one main repo.
	commonDir, err := w.commonGitDir(ctx)
	if err != nil {
		return nil, err
	}
	mainRoot := filepath.Dir(commonDir)

	// Mirror the parent's sub-folder focus: the agent's cwd relative to its OWN
	// worktree root, replayed under the child's fresh checkout.
	topLevel, err := w.worktreeRoot(ctx)
	if err != nil {
		return nil, err
	}
	subpath, err := filepath.Rel(topLevel, w.Dir)
	if err != nil || strings.HasPrefix(subpath, "..") {
		subpath = "." // cwd outside its own worktree root shouldn't happen; be safe
	}

	baseDir := filepath.Join(filepath.Dir(mainRoot), filepath.Base(mainRoot)+"-amplio", "worktrees")

	// Serialize creation per shared repo. Probe + create happen INSIDE the lock
	// so concurrent same-repo links can't pick the same path.
	release := linkLocks.Acquire(commonDir)
	defer release()

	placement := workspace.FreeSiblingPath(baseDir, childSessionID)
	if err := os.MkdirAll(filepath.Dir(placement), 0o755); err != nil {
		return nil, fmt.Errorf("git link: create worktree parent: %w", err)
	}
	if _, err := run(ctx, w.Dir, "worktree", "add", "--detach", placement); err != nil {
		return nil, fmt.Errorf("git link: %w", err)
	}

	childRoot := placement
	if subpath != "." {
		childRoot = filepath.Join(placement, subpath)
	}
	// childRoot is already absolute (placement derives from the absolute main
	// root); record the parent's root as provenance.
	return &gitWorkspace{Dir: childRoot, From: w.Dir}, nil
}

// commonGitDir returns the absolute path of the repo's common git dir (shared by
// all worktrees) — the stable identity of the underlying repo.
func (w *gitWorkspace) commonGitDir(ctx context.Context) (string, error) {
	out, err := run(ctx, w.Dir, "rev-parse", "--git-common-dir")
	if err != nil {
		return "", fmt.Errorf("git link: resolve common dir: %w", err)
	}
	d := strings.TrimSpace(out)
	if !filepath.IsAbs(d) {
		d = filepath.Join(w.Dir, d)
	}
	return filepath.Clean(d), nil
}

// worktreeRoot returns the absolute root of the CURRENT worktree (not the main
// repo) — used to compute the agent's sub-folder offset.
func (w *gitWorkspace) worktreeRoot(ctx context.Context) (string, error) {
	out, err := run(ctx, w.Dir, "rev-parse", "--show-toplevel")
	if err != nil {
		return "", fmt.Errorf("git link: resolve worktree root: %w", err)
	}
	return strings.TrimSpace(out), nil
}

// run executes a git subcommand in dir, returning combined output. stderr is
// merged into stdout so error messages survive for diagnosis.
func run(ctx context.Context, dir string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("git %s: %w: %s",
			strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func unmarshal(data []byte) (workspace.Workspace, error) {
	var w gitWorkspace
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, err
	}
	if w.Dir == "" {
		return nil, fmt.Errorf("git workspace: missing root")
	}
	return &w, nil
}

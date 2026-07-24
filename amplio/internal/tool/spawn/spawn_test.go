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

package spawn

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"amplio/internal/agent"
	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/session"
	"amplio/internal/workspace"
	"amplio/internal/workspace/git"
)

type noopAgent struct{ id string }

func (a *noopAgent) Run(context.Context) error { return nil }
func (a *noopAgent) SessionID() string         { return a.id }

// linkCaptured records the workspace handed to the spawned child (set
// synchronously by the factory, before the child goroutine starts).
var linkCaptured workspace.Workspace

func init() {
	agent.Register("spawn_link_test_agent", func(env *agent.Env, cfg *agent.Config) (agent.Agent, error) {
		linkCaptured = env.Workspace
		return &noopAgent{id: cfg.SessionID}, nil
	})
}

func gitRepo(t *testing.T) string {
	t.Helper()
	repo := filepath.Join(t.TempDir(), "myrepo")
	if err := os.Mkdir(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init")
	run("config", "user.email", "t@example.com")
	run("config", "user.name", "tester")
	if err := os.WriteFile(filepath.Join(repo, "f"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	run("add", ".")
	run("commit", "-m", "init")
	return repo
}

func TestSpawnLinkMode(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	ctx := context.Background()
	repo := gitRepo(t)

	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	runID := db.NewRunID()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: runID}); err != nil {
		t.Fatal(err)
	}

	env := &agent.Env{
		Store:     store,
		RunID:     runID,
		Names:     session.NewNameAllocator(store, runID),
		Registry:  session.NewRegistry(),
		Workspace: git.New(repo),
	}
	ex := makeExecutor(env, "parent")
	args, _ := json.Marshal(Params{Task: "t", AgentType: "spawn_link_test_agent", WorkspaceMode: "link"})
	res, err := ex(ctx, args)
	if err != nil {
		t.Fatal(err)
	}
	if res.IsError {
		t.Fatalf("spawn errored: %s", res.Content)
	}

	if linkCaptured == nil {
		t.Fatal("child workspace not captured")
	}
	if linkCaptured.LinkedFrom() != repo {
		t.Errorf("child LinkedFrom = %q, want %q", linkCaptured.LinkedFrom(), repo)
	}
	if !strings.Contains(linkCaptured.Root(), "myrepo-amplio/worktrees/") {
		t.Errorf("child root = %q, want under myrepo-amplio/worktrees/", linkCaptured.Root())
	}
	if !workspace.PathExists(linkCaptured.Root()) {
		t.Errorf("linked worktree dir missing: %s", linkCaptured.Root())
	}
}

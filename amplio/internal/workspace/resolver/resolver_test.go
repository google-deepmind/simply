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

package resolver_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"amplio/internal/workspace/resolver"
)

func TestWrap_NonVCSIsPlain(t *testing.T) {
	if ws := resolver.Wrap(t.TempDir()); ws.Kind() != "plain" {
		t.Errorf("non-VCS path wrapped as %q, want plain", ws.Kind())
	}
}

func TestWrap_GitRepoIsGit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	repo := filepath.Join(t.TempDir(), "r")
	if err := os.Mkdir(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := exec.CommandContext(context.Background(), "git", "-C", repo, "init").Run(); err != nil {
		t.Fatalf("git init: %v", err)
	}
	if ws := resolver.Wrap(repo); ws.Kind() != "git" {
		t.Errorf("git repo wrapped as %q, want git", ws.Kind())
	}
}

func TestResolve_EmptyIsError(t *testing.T) {
	if _, err := resolver.Resolve("", "me"); err == nil {
		t.Error("Resolve(\"\") should error")
	}
}

func TestResolve_PathDelegatesToWrap(t *testing.T) {
	dir := t.TempDir()
	ws, err := resolver.Resolve(dir, "me")
	if err != nil {
		t.Fatal(err)
	}
	if ws.Kind() != "plain" || ws.Root() != dir {
		t.Errorf("Resolve(path) = kind %q root %q, want plain %q", ws.Kind(), ws.Root(), dir)
	}
}

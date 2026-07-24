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

package workspace

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// stubWS satisfies the Workspace interface enough for ReadWorkspaceAgentsMD,
// which only consumes Root().
type stubWS struct{ root string }

func (s *stubWS) Root() string                                            { return s.root }
func (s *stubWS) Name() string                                            { return filepath.Base(s.root) }
func (s *stubWS) Kind() string                                            { return "stub" }
func (s *stubWS) LinkedFrom() string                                      { return "" }
func (s *stubWS) Validate(context.Context) error                          { return nil }
func (s *stubWS) ResolveAlias(context.Context) (string, error)            { return "", nil }
func (s *stubWS) Describe(context.Context, string) string                 { return "" }
func (s *stubWS) CreateLinked(context.Context, string) (Workspace, error) { return nil, nil }
func (s *stubWS) MarshalJSON() ([]byte, error)                            { return []byte(`{}`), nil }

func TestReadWorkspaceAgentsMD_Missing(t *testing.T) {
	tmp := t.TempDir()
	ws := &stubWS{root: tmp}
	got, err := ReadWorkspaceAgentsMD(ws)
	if err != nil || got != "" {
		t.Errorf("got %q, err %v; want \"\", nil", got, err)
	}
}

func TestReadWorkspaceAgentsMD_Exists(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "AGENTS.md"),
		[]byte("\n  workspace rules  \n"), 0o600); err != nil { //nolint:gosec // test file
		t.Fatal(err)
	}
	ws := &stubWS{root: tmp}
	got, err := ReadWorkspaceAgentsMD(ws)
	if err != nil || got != "workspace rules" {
		t.Errorf("got %q, err %v; want \"workspace rules\", nil", got, err)
	}
}

func TestReadWorkspaceAgentsMD_NilWorkspace(t *testing.T) {
	// Defensive — eventloop bootstrap may be called with no workspace in
	// tests; nil should be the same as missing-file, not a panic.
	got, err := ReadWorkspaceAgentsMD(nil)
	if err != nil || got != "" {
		t.Errorf("got %q, err %v; want \"\", nil", got, err)
	}
}

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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"amplio/internal/workspace"
)

// Kind is the JSON discriminator for plain workspaces.
const Kind = "plain"

func init() { workspace.RegisterKind(Kind, unmarshal) }

var _ workspace.Workspace = (*plainWorkspace)(nil)

// plainWorkspace fields are exported (with json tags) so the struct itself is
// the serialized form (mirrors event types). The field is named Dir, not Root,
// because Root is the accessor method.
type plainWorkspace struct {
	Dir string `json:"root"`
}

// Factory creates a plain workspace from a path string. Suitable as a
// runtime.WorkspaceFactory.
func Factory(path string) workspace.Workspace { return New(path) }

// New creates a workspace from a plain directory path (resolved to absolute).
func New(root string) workspace.Workspace {
	abs, err := filepath.Abs(root)
	if err != nil {
		abs = root
	}
	return &plainWorkspace{Dir: abs}
}

func (w *plainWorkspace) Root() string { return w.Dir }
func (w *plainWorkspace) Name() string { return filepath.Base(w.Dir) }
func (w *plainWorkspace) Kind() string { return Kind }

func (w *plainWorkspace) CreateLinked(context.Context, string) (workspace.Workspace, error) {
	return nil, errors.New("plain workspace does not support linked workspaces")
}

// LinkedFrom returns "" — plain workspaces are never linked.
func (w *plainWorkspace) LinkedFrom() string { return "" }

// Describe reports the working directory (and, for a sub-agent, that it shares
// the parent's directory). Plain workspaces have no VCS.
func (w *plainWorkspace) Describe(_ context.Context, parentSessionID string) string {
	lines := []string{fmt.Sprintf("Working directory: `%s`", w.Dir)}
	if p := workspace.ProvenanceLine(parentSessionID, false, ""); p != "" {
		lines = append(lines, p)
	}
	return strings.Join(lines, "\n")
}

// Validate confirms the directory still exists and is a directory.
func (w *plainWorkspace) Validate(context.Context) error {
	info, err := os.Stat(w.Dir)
	if err != nil {
		return fmt.Errorf("workspace %q: %w", w.Dir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("workspace %q is not a directory", w.Dir)
	}
	return nil
}

// ResolveAlias returns "" — plain directories have no alias concept.
func (w *plainWorkspace) ResolveAlias(context.Context) (string, error) { return "", nil }

func unmarshal(data []byte) (workspace.Workspace, error) {
	var w plainWorkspace
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, err
	}
	if w.Dir == "" {
		return nil, fmt.Errorf("plain workspace: missing root")
	}
	return &w, nil
}

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
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// AgentsMDFilename is the per-workspace instruction file the eventloop reads
// at every fresh agent bootstrap (root or linked sub-agent). Convention name
// matching the global file under <data-dir>/.
const AgentsMDFilename = "AGENTS.md"

// ReadWorkspaceAgentsMD loads <workspace-root>/AGENTS.md and returns its
// trim-spaced content. Missing-file is NOT an error (the file is opt-in per
// workspace, just like the global one); permission / I/O errors propagate.
//
// Re-read on every fresh agent bootstrap, so a linked sub-agent in a
// different workspace picks up THAT workspace's instructions automatically.
// Mid-run edits don't affect already-running agents — the SystemEvent is
// step-0 immutable; only newly-spawned sub-agents see the change.
func ReadWorkspaceAgentsMD(ws Workspace) (string, error) {
	if ws == nil {
		return "", nil
	}
	path := filepath.Join(ws.Root(), AgentsMDFilename)
	data, err := os.ReadFile(path) //nolint:gosec // path derived from a Workspace root we already trust
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read workspace AGENTS.md at %s: %w", path, err)
	}
	return strings.TrimSpace(string(data)), nil
}

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

// Package runspec builds the inputs for creating a run: it resolves a raw
// workspace spec into a concrete path and snapshots the operator's AGENTS.md.
// Shared by the CLI (headless run) and the server (POST /api/runs) so both
// create runs identically.
package runspec

import (
	"fmt"
	"path/filepath"

	"amplio/internal/config"
	"amplio/internal/workspace"
	"amplio/internal/workspace/resolver"
)

// Prepare turns a raw workspace spec into the two workspace-derived fields of a
// RunConfig: the concrete workspace root path and the combined AGENTS.md
// snapshot. It is the single run-creation pre-step shared by the CLI and the
// server.
//
// The workspace spec is the ONLY run input whose handling differs between a
// fresh run and a resume: creation sentinels (e.g. `jj:new`) allocate a
// workspace with side effects and must run exactly once, here. Everything else
// in RunConfig is raw string data, identical across fresh and resume — which is
// why RunConfig stores only the concrete path and StartRun / RecoverRun both
// reconstruct the live workspace from it the same way.
//
// user is the workspace owner, needed to resolve CitC specs. The returned
// agentsMD is snapshotted now (global <data-dir>/AGENTS.md + the workspace's
// own AGENTS.md) so the run captures whatever's in effect at start time;
// respawns reuse the persisted snapshot rather than re-reading the files.
func Prepare(rawWorkspace, user string) (workspaceRoot, agentsMD string, err error) {
	ws, err := resolver.Resolve(rawWorkspace, user)
	if err != nil {
		return "", "", fmt.Errorf("workspace: %w", err)
	}
	globalMD, err := config.ReadGlobalAgentsMD()
	if err != nil {
		return "", "", err
	}
	wsMD, err := workspace.ReadWorkspaceAgentsMD(ws)
	if err != nil {
		return "", "", err
	}
	agentsMD = config.CombineAgentsMD(
		globalMD, config.GlobalAgentsMDPath(),
		wsMD, filepath.Join(ws.Root(), workspace.AgentsMDFilename),
	)
	return ws.Root(), agentsMD, nil
}

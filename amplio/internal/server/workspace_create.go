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

package server

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"

	"amplio/internal/workspace/resolver"
)

// handleCreateWorkspace creates a fresh anonymous workspace and returns its
// absolute path. ONLY accepts `new:` / `anon:` specs — opening an existing
// workspace (an alias, a path) is not a "create" operation and stays in
// /api/runs's own resolver call. This split lets the UI render an honest
// two-stage progress indicator: creation is the slow step (5-30s on the
// backends that materialize one); opening is fast and needs no separate stage.
//
// Callers that want one-shot start (CLI, automation) keep passing the
// spec directly to POST /api/runs and let the server resolve internally.
// The two-call flow is purely for UIs that want creation progress
// visibility.
//
// Auth: write-equivalent (materializes a workspace), behind requireAuth.
func (s *Server) handleCreateWorkspace(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Spec string `json:"spec"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	spec := strings.TrimSpace(req.Spec)
	if !isCreationSpec(spec) {
		writeErr(w, http.StatusBadRequest,
			"only `new:` / `anon:` specs are accepted here; pass other specs "+
				"(existing workspaces, paths) directly to POST /api/runs to open them")
		return
	}
	// The OS user is still needed locally to resolve a named workspace, but
	// it is no longer recorded on the run itself (single-user amplio).
	ws, err := resolver.Resolve(spec, os.Getenv("USER"))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "workspace: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"path": ws.Root(),
		"kind": ws.Kind(),
	})
}

// isCreationSpec returns true for the small set of specs that invoke a
// creation side-effect inside resolver.Resolve. Mirrors the cases at the
// top of resolver.Resolve so this gate doesn't silently drift if more
// creation specs are added later — keep the lists in sync.
func isCreationSpec(spec string) bool {
	return strings.HasPrefix(spec, "new:") || strings.HasPrefix(spec, "anon:")
}

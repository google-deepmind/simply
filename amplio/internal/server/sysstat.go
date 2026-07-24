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
	"net/http"

	"amplio/internal/sysstat"
)

// handleSysStat returns the current sysstat snapshot. Clients fetch this once
// on mount to seed their store, then receive updates via the kind=sysstat SSE
// event. The watcher behind this is owned by cmd/amplio/serve.go.
func (s *Server) handleSysStat(w http.ResponseWriter, _ *http.Request) {
	var snap sysstat.Snapshot
	if s.sysstat != nil {
		snap = s.sysstat.Latest()
	}
	writeJSON(w, http.StatusOK, snap)
}

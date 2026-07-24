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
	"strings"
)

// modelEntry is one option in the agent-model menu. Config entries are
// read-only; custom (DB) entries are removable.
type modelEntry struct {
	Spec      string `json:"spec"`
	Removable bool   `json:"removable"`
}

type modelMenu struct {
	Default string       `json:"default"`
	Models  []modelEntry `json:"models"`
}

// handleListModels returns the union of config.toml's [run].llms (read-only) and
// the user-added custom models (removable), deduped, config order first.
func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	custom, err := s.store.ListCustomModels(r.Context())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	seen := make(map[string]bool, len(s.defaults.LLMs)+len(custom))
	menu := modelMenu{Default: s.defaults.LLM, Models: []modelEntry{}}
	for _, spec := range s.defaults.LLMs {
		if spec == "" || seen[spec] {
			continue
		}
		seen[spec] = true
		menu.Models = append(menu.Models, modelEntry{Spec: spec, Removable: false})
	}
	for _, spec := range custom {
		if spec == "" || seen[spec] {
			continue // a custom that duplicates a config entry stays read-only
		}
		seen[spec] = true
		menu.Models = append(menu.Models, modelEntry{Spec: spec, Removable: true})
	}
	writeJSON(w, http.StatusOK, menu)
}

func (s *Server) handleAddModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Spec string `json:"spec"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	spec := strings.TrimSpace(req.Spec)
	if spec == "" {
		writeErr(w, http.StatusBadRequest, "spec is required")
		return
	}
	if err := s.store.AddCustomModel(r.Context(), spec); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"status": "added"})
}

func (s *Server) handleRemoveModel(w http.ResponseWriter, r *http.Request) {
	spec := strings.TrimSpace(r.URL.Query().Get("spec"))
	if spec == "" {
		writeErr(w, http.StatusBadRequest, "spec query param is required")
		return
	}
	if err := s.store.RemoveCustomModel(r.Context(), spec); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "removed"})
}

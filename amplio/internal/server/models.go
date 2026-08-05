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

	"amplio/internal/llm"
)

// modelEntry is one option in the agent-model menu. Config entries are
// read-only; custom (DB) entries are removable.
//
// Label is a derived display name (internal/llm.ShortLabel), sent so the menu
// and the run chips agree on one implementation. Spec remains the identity and
// the only value ever submitted back — the UI must keep it visible next to the
// label, not behind it.
type modelEntry struct {
	Spec      string `json:"spec"`
	Label     string `json:"label"`
	Removable bool   `json:"removable"`
	// Duplicate marks an entry whose provider spec is shared with another entry —
	// they differ only by the "#nickname" display label. Not an error (relabelling
	// an existing endpoint is done exactly this way), but worth flagging: the two
	// start identical runs that will then be LABELLED differently everywhere, so
	// it is usually a leftover rather than an intent.
	Duplicate bool `json:"duplicate"`
}

type modelMenu struct {
	Default string       `json:"default"`
	Models  []modelEntry `json:"models"`
}

// handleListModels returns the union of config.toml's [run].llms (read-only) and
// the user-added custom models (removable), deduped, config order first.
//
// Dedup is on the VERBATIM spec, so "x" and "x#nickname" are two entries. That
// is deliberate: adding a nickname for an already-listed endpoint is exactly how
// an operator relabels one (e.g. a reused test endpoint whose checkpoint
// changed), and silently collapsing it onto the unlabelled entry would discard
// the thing they just asked for.
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
		menu.Models = append(menu.Models, modelEntry{Spec: spec, Label: llm.ShortLabel(spec), Removable: false})
	}
	for _, spec := range custom {
		if spec == "" || seen[spec] {
			continue // a custom that duplicates a config entry stays read-only
		}
		seen[spec] = true
		menu.Models = append(menu.Models, modelEntry{Spec: spec, Label: llm.ShortLabel(spec), Removable: true})
	}
	markDuplicates(menu.Models)
	writeJSON(w, http.StatusOK, menu)
}

// markDuplicates flags every entry that shares a provider spec with another,
// i.e. entries that differ only by their "#nickname". Both stay in the menu —
// suppressing one would silently discard the relabel the operator just asked
// for, and the config-sourced entry (which can't be removed from the UI) is
// often the one they wanted to override.
func markDuplicates(entries []modelEntry) {
	count := make(map[string]int, len(entries))
	for _, e := range entries {
		count[llm.BaseSpec(e.Spec)]++
	}
	for i := range entries {
		entries[i].Duplicate = count[llm.BaseSpec(entries[i].Spec)] > 1
	}
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

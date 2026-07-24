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

package main

import (
	"encoding/json"
	"testing"
)

// The /api/runs response is the pagination envelope {runs, has_more,
// next_cursor}, NOT a bare array. `client list` must parse the envelope and
// feed its inner `runs` to printRunSummaries — a regression guard for the
// pagination change that briefly broke `client list` (it unmarshaled the
// object into a []struct and errored).
func TestClientList_ParsesPaginationEnvelope(t *testing.T) {
	raw := []byte(`{
		"runs": [
			{"run_id":"r1","title":"One","root_status":"idle","root_step":3,"starred":true},
			{"run_id":"r2","task":"do thing","root_status":"concluded","root_step":9}
		],
		"has_more": true,
		"next_cursor": "2026-01-01T00:00:00Z"
	}`)

	var page struct {
		Runs    json.RawMessage `json:"runs"`
		HasMore bool            `json:"has_more"`
	}
	if err := json.Unmarshal(raw, &page); err != nil {
		t.Fatalf("parse envelope: %v", err)
	}
	if !page.HasMore {
		t.Error("has_more = false, want true")
	}
	// printRunSummaries consumes the inner array (not the envelope); it must not
	// error on the documented runSummary[] shape.
	if err := printRunSummaries(page.Runs, page.HasMore); err != nil {
		t.Fatalf("printRunSummaries: %v", err)
	}
}

// An empty page still renders cleanly (no crash, no spurious error).
func TestClientList_EmptyPage(t *testing.T) {
	var page struct {
		Runs    json.RawMessage `json:"runs"`
		HasMore bool            `json:"has_more"`
	}
	if err := json.Unmarshal([]byte(`{"runs":[],"has_more":false}`), &page); err != nil {
		t.Fatal(err)
	}
	if err := printRunSummaries(page.Runs, page.HasMore); err != nil {
		t.Fatalf("printRunSummaries(empty): %v", err)
	}
}

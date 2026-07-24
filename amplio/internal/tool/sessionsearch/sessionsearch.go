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

// Package sessionsearch provides full-text search over session events.
package sessionsearch

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"amplio/internal/db"
	"amplio/internal/tool"
	"amplio/internal/util"
)

type Params struct {
	Query     string `json:"query" jsonschema:"required" jsonschema_description:"Search query (FTS5 syntax: words ANDed by default. Use quotes for phrases and - for negation)"`
	SessionID string `json:"session_id,omitempty" jsonschema_description:"Limit search to a specific session"`
	StepMin   *int   `json:"step_min,omitempty" jsonschema_description:"Inclusive lower bound on step number"`
	StepMax   *int   `json:"step_max,omitempty" jsonschema_description:"Inclusive upper bound on step number"`
	Limit     int    `json:"limit,omitempty" jsonschema_description:"Max results (default 20, max 100)"`
	RunID     string `json:"run_id,omitempty" jsonschema_description:"Optional: a prior run's id to search instead of the current run."`
}

func New(store db.Store, runID string) *tool.Tool {
	return &tool.Tool{
		Name:        "session_search",
		Description: "Full-text search over events in a run (the current run, or a prior run via run_id). Returns matching events ranked by relevance.",
		ParamType:   &Params{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[Params](args)
			if errResult != nil {
				return errResult, nil
			}
			rid := runID
			if params.RunID != "" {
				rid = params.RunID
			}
			limit := params.Limit
			if limit <= 0 {
				limit = 20
			}
			if limit > 100 {
				limit = 100
			}
			results, err := store.SearchEvents(ctx, rid, params.Query, db.SearchOpts{
				SessionID: params.SessionID,
				StepMin:   params.StepMin,
				StepMax:   params.StepMax,
				Limit:     limit,
			})
			if err != nil {
				return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
			}
			if len(results) == 0 {
				return &tool.Result{Content: fmt.Sprintf("No results for query %q.", params.Query)}, nil
			}

			var b strings.Builder
			fmt.Fprintf(&b, "%d result(s) for %q:\n", len(results), params.Query)
			for _, e := range results {
				// Rune-safe truncation: event content is frequently non-ASCII
				// (agent prose, file contents), so a byte slice could split a rune.
				content := util.TruncateRunes(e.Event.ToText(), 150)
				content = strings.ReplaceAll(content, "\n", " ")
				fmt.Fprintf(&b, "  session=%s step=%d type=%-12s %s\n",
					e.SessionID, e.Step, e.Event.EventType(), content)
			}
			return &tool.Result{Content: b.String()}, nil
		},
	}
}

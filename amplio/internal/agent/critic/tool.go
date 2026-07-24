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

package critic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"amplio/internal/db"
	"amplio/internal/tool"
)

type viewReportParams struct {
	RunID string `json:"run_id,omitempty" jsonschema_description:"Run to read reports from; defaults to the current run"`
}

// ViewRunReport returns the view_run_report tool. It dumps every iteration's
// report (newest first) — reports are short, so there's no list/get-one
// interface. currentRunID is used when the caller omits run_id (cross-run reads
// pass an explicit run_id).
func ViewRunReport(store db.Store, currentRunID string) *tool.Tool {
	return &tool.Tool{
		Name: "view_run_report",
		Description: "Read the run report(s): the independent end-of-iteration assessment(s) of a run " +
			"(summary, key achievements, failure modes, artifacts). Defaults to the current run; pass run_id for another run. " +
			"Returns every iteration, newest first.",
		ParamType: &viewReportParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			p, errResult := tool.ParseArgs[viewReportParams](args)
			if errResult != nil {
				return errResult, nil
			}
			runID := currentRunID
			if strings.TrimSpace(p.RunID) != "" {
				runID = strings.TrimSpace(p.RunID)
			}
			reports, err := AllReports(ctx, store, runID)
			if err != nil {
				return &tool.Result{Content: "Error: " + err.Error(), IsError: true}, nil
			}
			if len(reports) == 0 {
				return &tool.Result{Content: fmt.Sprintf("No run report yet for %s.", runID)}, nil
			}
			return &tool.Result{Content: formatReports(reports)}, nil
		},
	}
}

// formatReports renders all reports newest-first as plain text.
func formatReports(reports []*RunReport) string {
	var b strings.Builder
	for i := len(reports) - 1; i >= 0; i-- {
		writeReportText(&b, reports[i])
		if i > 0 {
			b.WriteString("\n")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func writeReportText(b *strings.Builder, r *RunReport) {
	fmt.Fprintf(b, "=== RUN REPORT — iteration %d (generated %s) ===\n", r.Version, r.CreatedAt.Format(time.RFC3339))
	if strings.TrimSpace(r.Task) != "" {
		fmt.Fprintf(b, "Task: %s\n", r.Task)
	}
	fmt.Fprintf(b, "\n%s\n", r.Summary)

	if len(r.KeyAchievements) > 0 {
		b.WriteString("\nKey achievements:\n")
		writeClaims(b, r.KeyAchievements)
	}
	if len(r.FailureModes) > 0 {
		b.WriteString("\nFailure modes:\n")
		writeClaims(b, r.FailureModes)
	}
	if len(r.ArtifactsByKind) > 0 {
		b.WriteString("\nArtifacts:\n")
		for kind, arts := range r.ArtifactsByKind {
			for _, a := range arts {
				fmt.Fprintf(b, "  * [%s] %s — %s\n", kind, a.Value, a.Context)
			}
		}
	}
	if len(r.Struggles) > 0 {
		fmt.Fprintf(b, "\nStruggle ranges (%d):\n", len(r.Struggles))
		for _, s := range r.Struggles {
			fmt.Fprintf(b, "  * session=%s %s\n", s.SessionID, stepRange(s.StartStep, s.EndStep))
			// Surface the first sample summary so the chatbot reading a
			// prior report sees WHAT the agent was struggling with, not just
			// where. Falls back silently for old reports that have no samples.
			if len(s.SampleSummaries) > 0 {
				fmt.Fprintf(b, "    e.g. %s\n", s.SampleSummaries[0])
			}
		}
	}
	b.WriteString("\n")
}

func writeClaims(b *strings.Builder, claims []CitedClaim) {
	for _, c := range claims {
		fmt.Fprintf(b, "  * %s\n", c.Statement)
		if len(c.Citations) > 0 {
			fmt.Fprintf(b, "    ↳ %s\n", strings.Join(c.Citations, ", "))
		}
	}
}

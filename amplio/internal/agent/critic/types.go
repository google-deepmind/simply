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

import "time"

// CitedClaim is one evaluative statement backed by concrete citations (the LLM
// half of the report).
type CitedClaim struct {
	Statement string   `json:"statement"`
	Citations []string `json:"citations"`
}

// Artifact is a concrete verifiable item produced during the run, lifted from a
// phase summary's artifacts and grouped by kind in the report.
type Artifact struct {
	Value     string `json:"value"`
	Context   string `json:"context"`
	SessionID string `json:"session_id"`
	StartStep int    `json:"start_step"`
	EndStep   int    `json:"end_step"`
}

// ReportPhase is a phase-summary projection included in the report.
type ReportPhase struct {
	SessionID string `json:"session_id"`
	StartStep int    `json:"start_step"`
	EndStep   int    `json:"end_step"`
	Title     string `json:"title"`
	Summary   string `json:"summary"`
}

// Struggle is a contiguous run of retrying/blocked steps in one session.
type Struggle struct {
	SessionID       string   `json:"session_id"`
	StartStep       int      `json:"start_step"`
	EndStep         int      `json:"end_step"`
	Length          int      `json:"length"`
	SampleSummaries []string `json:"sample_summaries"`
}

// SessionState is a per-session watermark snapshot at report time. The set of
// SessionStates is the report's coverage: the next iteration's delta is computed
// against the previous report's snapshot (per-session step ranges).
type SessionState struct {
	SessionID   string `json:"session_id"`
	AgentType   string `json:"agent_type"`
	Status      string `json:"status"`
	CurrentStep int    `json:"current_step"`
}

// RunReport is one iteration's structured report. The mechanical fields (Phases,
// ArtifactsByKind, Struggles, Sessions) are computed from observations; the
// evaluative fields (Summary, KeyAchievements, FailureModes, Grade) come from
// the keen-critic LLM. Stored as a versioned run_report observation.
type RunReport struct {
	Version   int       `json:"version"`
	CreatedAt time.Time `json:"created_at"`
	Task      string    `json:"task"`
	Summary   string    `json:"summary"`
	// Grade is the critic's overall verdict on the iteration
	// (1=garbage, 2=bad, 3=meh, 4=good, 5=excellent; 0=ungraded). It is
	// denormalized onto the run row as report_grade for the UI.
	Grade           int                   `json:"grade"`
	KeyAchievements []CitedClaim          `json:"key_achievements"`
	FailureModes    []CitedClaim          `json:"failure_modes"`
	ArtifactsByKind map[string][]Artifact `json:"artifacts_by_kind"`
	Phases          []ReportPhase         `json:"phases"`
	Struggles       []Struggle            `json:"struggles"`
	Sessions        []SessionState        `json:"sessions"`
}

// SessionStep returns the per-session watermark recorded in this report, or 0
// if the session wasn't in it (treated as "new since this report"). Exported
// so the server's coverage classifier can mirror the finalizer's delta rule.
func (r *RunReport) SessionStep(sessionID string) int {
	for _, s := range r.Sessions {
		if s.SessionID == sessionID {
			return s.CurrentStep
		}
	}
	return 0
}

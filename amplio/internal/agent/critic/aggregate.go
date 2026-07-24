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
	"fmt"
	"sort"

	"amplio/internal/config"
	"amplio/internal/db"
)

// These mirror the observer's observation kinds (intentionally duplicated as
// stable wire constants rather than importing the observer package).
const (
	phaseSummaryKind = "phase_summary"
	stepSummaryKind  = "step_summary"
)

// isAutonomous reports whether the run has an autonomous main-agent root.
func isAutonomous(sessions []db.SessionRecord) bool {
	for _, s := range sessions {
		if s.SessionID == config.RootAgentSessionID {
			return true
		}
	}
	return false
}

// subjectSessions returns the sessions a report is about. In an autonomous run
// that's the main-agent spine — the chatbot is a sidecar (the operator
// discussing the run) and is excluded. In an interactive run the chatbot IS the
// worker, so it's included.
func subjectSessions(sessions []db.SessionRecord) []db.SessionRecord {
	auto := isAutonomous(sessions)
	out := make([]db.SessionRecord, 0, len(sessions))
	for _, s := range sessions {
		if auto && (s.SessionID == config.ChatbotSessionID || s.AgentType == config.ChatbotAgentType) {
			continue
		}
		out = append(out, s)
	}
	return out
}

func snapshotSessions(subjects []db.SessionRecord) []SessionState {
	out := make([]SessionState, 0, len(subjects))
	for _, s := range subjects {
		out = append(out, SessionState{
			SessionID:   s.SessionID,
			AgentType:   s.AgentType,
			Status:      s.Status,
			CurrentStep: s.CurrentStep,
		})
	}
	return out
}

func findSession(sessions []db.SessionRecord, sessionID string) *db.SessionRecord {
	for i := range sessions {
		if sessions[i].SessionID == sessionID {
			return &sessions[i]
		}
	}
	return nil
}

// advancedByAtLeast reports whether any subject session progressed by at least
// `minSteps` past the previous report's per-session watermark. Used both as a
// "has anything happened?" gate (minSteps=1) and as the debounce threshold that
// suppresses trivial deltas below reportSkipMinSteps.
func advancedByAtLeast(prev *RunReport, subjects []db.SessionRecord, minSteps int) bool {
	for _, s := range subjects {
		if s.CurrentStep-prev.SessionStep(s.SessionID) >= minSteps {
			return true
		}
	}
	return false
}

// gatherPhasesAndArtifacts reads phase_summary observations for the subject
// sessions, keeping only phases whose end_step is past each session's previous
// watermark (delta scope), and collects their artifacts grouped by kind.
func gatherPhasesAndArtifacts(ctx context.Context, store db.Store, runID string, subjects []db.SessionRecord, prevStep func(string) int) ([]ReportPhase, map[string][]Artifact, error) {
	var phases []ReportPhase
	artifactsByKind := map[string][]Artifact{}
	for _, s := range subjects {
		recs, err := store.GetObservations(ctx, runID, db.ObsFilter{Kind: phaseSummaryKind, SessionID: s.SessionID})
		if err != nil {
			return nil, nil, err
		}
		floor := prevStep(s.SessionID)
		for _, rec := range recs {
			end := obsInt(rec.Data, "end_step")
			if end <= floor {
				continue // already covered by a prior report
			}
			start := obsInt(rec.Data, "start_step")
			phases = append(phases, ReportPhase{
				SessionID: s.SessionID,
				StartStep: start,
				EndStep:   end,
				Title:     obsStr(rec.Data, "title"),
				Summary:   obsStr(rec.Data, "summary"),
			})
			for _, a := range obsArtifacts(rec.Data) {
				artifactsByKind[a.kind] = append(artifactsByKind[a.kind], Artifact{
					Value:     a.value,
					Context:   a.context,
					SessionID: s.SessionID,
					StartStep: start,
					EndStep:   end,
				})
			}
		}
	}
	sort.Slice(phases, func(i, j int) bool {
		if phases[i].SessionID != phases[j].SessionID {
			return phases[i].SessionID < phases[j].SessionID
		}
		return phases[i].EndStep < phases[j].EndStep
	})
	return phases, artifactsByKind, nil
}

// stepRange renders a step-range tag for human/LLM display. Single-step
// ranges ("step 5") read more naturally than "steps 5-5"; for true ranges,
// the "(N steps)" suffix the old format carried was just (end - start + 1) —
// derivable on sight and visually noisy, so it's dropped.
func stepRange(start, end int) string {
	if start == end {
		return fmt.Sprintf("step %d", start)
	}
	return fmt.Sprintf("steps %d-%d", start, end)
}

// gatherStruggles reads step_summary observations for the subject sessions and
// detects contiguous runs of retrying/blocked steps past each session's previous
// watermark.
func gatherStruggles(ctx context.Context, store db.Store, runID string, subjects []db.SessionRecord, prevStep func(string) int) ([]Struggle, error) {
	var out []Struggle
	for _, s := range subjects {
		recs, err := store.GetObservations(ctx, runID, db.ObsFilter{Kind: stepSummaryKind, SessionID: s.SessionID})
		if err != nil {
			return nil, err
		}
		floor := prevStep(s.SessionID)
		type stepSum struct {
			step    int
			summary string
			tag     string
		}
		sums := make([]stepSum, 0, len(recs))
		for _, rec := range recs {
			if rec.Step == nil || *rec.Step <= floor {
				continue
			}
			sums = append(sums, stepSum{step: *rec.Step, summary: obsStr(rec.Data, "summary"), tag: obsStr(rec.Data, "status_tag")})
		}
		sort.Slice(sums, func(i, j int) bool { return sums[i].step < sums[j].step })
		for i := 0; i < len(sums); {
			if !struggling(sums[i].tag) {
				i++
				continue
			}
			// Grow the run over CONSECUTIVE summarized struggling steps. We group by
			// adjacency in `sums` (the sorted list of step_summary rows), not by
			// strict step-number adjacency: the observer writes exactly one
			// step_summary per finalized step (see observer.catchUpSteps, which even
			// writes a degraded row for a compacted step), so `sums` is dense and a
			// numeric +1 check would only add fragility. A non-struggling tag is the
			// real boundary that ends a streak.
			j := i
			var samples []string
			for j < len(sums) && struggling(sums[j].tag) {
				if len(samples) < 3 {
					samples = append(samples, sums[j].summary)
				}
				j++
			}
			out = append(out, Struggle{
				SessionID:       s.SessionID,
				StartStep:       sums[i].step,
				EndStep:         sums[j-1].step,
				Length:          j - i,
				SampleSummaries: samples,
			})
			i = j
		}
	}
	return out, nil
}

func struggling(tag string) bool { return tag == "retrying" || tag == "blocked" }

// --- observation Data (map[string]any from JSON) accessors ---

func obsInt(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	}
	return 0
}

func obsStr(m map[string]any, key string) string {
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

type rawArtifact struct{ kind, value, context string }

func obsArtifacts(m map[string]any) []rawArtifact {
	raw, ok := m["artifacts"].([]any)
	if !ok {
		return nil
	}
	out := make([]rawArtifact, 0, len(raw))
	for _, item := range raw {
		am, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, rawArtifact{kind: obsStr(am, "kind"), value: obsStr(am, "value"), context: obsStr(am, "context")})
	}
	return out
}

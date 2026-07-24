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
	"sort"
	"strings"

	"amplio/internal/db"
)

// trajStep is one step in the drill-down: its summary label. Raw events are
// fetched on demand (GET …/events?step=N) since they can be large.
type trajStep struct {
	Step      int    `json:"step"`
	Summary   string `json:"summary"`
	StatusTag string `json:"status_tag"`
}

// artifactDTO is one curated, verifiable item the phase summarizer flagged
// (a file path, command, metric, id, …).
type artifactDTO struct {
	Kind    string `json:"kind"`
	Value   string `json:"value"`
	Context string `json:"context"`
}

// lessonVerdictDTO is one lesson the agent loaded in this phase, with the phase
// summarizer's helpfulness verdict (the input to lesson-score attribution).
type lessonVerdictDTO struct {
	ID      string `json:"id"`               // bare lesson id (handle without the "lesson:" prefix)
	Title   string `json:"title"`            // resolved from the corpus; falls back to the id if deleted
	Verdict string `json:"verdict"`          // helpful | neutral | unhelpful | harmful
	Reason  string `json:"reason,omitempty"` // one-line evidence from the summarizer
}

// trajPhase is a summarized phase containing its steps.
type trajPhase struct {
	StartStep      int                `json:"start_step"`
	EndStep        int                `json:"end_step"`
	Title          string             `json:"title"`
	Summary        string             `json:"summary"`
	Artifacts      []artifactDTO      `json:"artifacts"`
	LessonVerdicts []lessonVerdictDTO `json:"lesson_verdicts"`
	Steps          []trajStep         `json:"steps"`
}

// trajectory is the session drill-down skeleton: phases → steps (+ loose, not-
// yet-phased steps). Cheap to load (summaries only); events come per-step later.
type trajectory struct {
	Phases      []trajPhase `json:"phases"`
	LooseSteps  []trajStep  `json:"loose_steps"`
	CurrentStep int         `json:"current_step"`
}

func (s *Server) handleTrajectory(w http.ResponseWriter, r *http.Request) {
	id, sid := r.PathValue("id"), r.PathValue("sid")
	sess, err := s.store.GetSession(r.Context(), id, sid)
	if err != nil {
		writeErr(w, http.StatusNotFound, "session not found")
		return
	}
	phaseObs, err := s.store.GetObservations(r.Context(), id, db.ObsFilter{Kind: "phase_summary", SessionID: sid})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	stepObs, err := s.store.GetObservations(r.Context(), id, db.ObsFilter{Kind: "step_summary", SessionID: sid})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	type ss struct{ summary, tag string }
	summaries := make(map[int]ss, len(stepObs))
	for _, o := range stepObs {
		if o.Step != nil {
			summaries[*o.Step] = ss{obsStr(o, "summary"), obsStr(o, "status_tag")}
		}
	}
	mkStep := func(n int) trajStep {
		s := summaries[n]
		return trajStep{Step: n, Summary: s.summary, StatusTag: s.tag}
	}

	sort.Slice(phaseObs, func(i, j int) bool {
		return obsInt(phaseObs[i], "end_step", 0) < obsInt(phaseObs[j], "end_step", 0)
	})
	phases := make([]trajPhase, 0, len(phaseObs))
	lastEnd := 0
	for _, p := range phaseObs {
		start := obsInt(p, "start_step", 0)
		end := obsInt(p, "end_step", 0)
		steps := make([]trajStep, 0, end-start+1)
		for n := start; n <= end; n++ {
			steps = append(steps, mkStep(n))
		}
		phases = append(phases, trajPhase{
			StartStep: start, EndStep: end,
			Title:          obsStr(p, "title"),
			Summary:        obsStr(p, "summary"),
			Artifacts:      obsArtifacts(p),
			LessonVerdicts: s.obsLessonVerdicts(r, p),
			Steps:          steps,
		})
		if end > lastEnd {
			lastEnd = end
		}
	}

	// At rest the session's current_step is empty — the concluding assistant
	// event lands in the prior step — so bound the tail by the last step that
	// actually has an event rather than rendering a trailing empty step.
	maxStep := 0
	if tail, err := s.store.GetTailEvent(r.Context(), id, sid); err == nil && tail != nil {
		maxStep = tail.Step
	}
	loose := make([]trajStep, 0)
	for n := lastEnd + 1; n <= maxStep; n++ {
		loose = append(loose, mkStep(n))
	}

	writeJSON(w, http.StatusOK, trajectory{
		Phases:      phases,
		LooseSteps:  loose,
		CurrentStep: sess.CurrentStep,
	})
}

// obsArtifacts extracts the phase summarizer's curated artifacts from an
// observation's Data bag (JSON round-trips them to []any of map[string]any).
func obsArtifacts(o db.ObservationRecord) []artifactDTO {
	raw, ok := o.Data["artifacts"].([]any)
	if !ok {
		return nil
	}
	out := make([]artifactDTO, 0, len(raw))
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		a := artifactDTO{}
		if s, ok := m["kind"].(string); ok {
			a.Kind = s
		}
		if s, ok := m["value"].(string); ok {
			a.Value = s
		}
		if s, ok := m["context"].(string); ok {
			a.Context = s
		}
		if a.Value != "" {
			out = append(out, a)
		}
	}
	return out
}

// obsLessonVerdicts extracts the phase summarizer's per-lesson verdicts from a
// phase observation's Data bag and resolves each handle to its lesson title (so
// the UI shows a name, not an opaque id). Skill handles are skipped (lesson
// scoring only); a deleted lesson falls back to its bare id.
func (s *Server) obsLessonVerdicts(r *http.Request, o db.ObservationRecord) []lessonVerdictDTO {
	// Always return a non-nil slice: nil marshals to JSON null, and the client
	// indexes .length on it (a phase with no loaded lessons is the common case).
	out := []lessonVerdictDTO{}
	raw, ok := o.Data["lesson_verdicts"].([]any)
	if !ok {
		return out
	}
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		handle, _ := m["handle"].(string)
		id, ok := strings.CutPrefix(strings.TrimSpace(handle), "lesson:")
		if !ok || id == "" {
			continue // skill or malformed handle; not scored here
		}
		verdict, _ := m["verdict"].(string)
		if verdict == "" {
			continue
		}
		reason, _ := m["reason"].(string)
		title := id // fallback when the lesson was deleted/superseded
		if l, err := s.store.GetLesson(r.Context(), id); err == nil {
			title = l.Title
		}
		out = append(out, lessonVerdictDTO{ID: id, Title: title, Verdict: verdict, Reason: reason})
	}
	return out
}

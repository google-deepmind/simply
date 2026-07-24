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
	"testing"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/llm"
)

// --- mock LLMs ---

// submitMock drives the keen-critic loop: it ends the loop immediately with a
// no-tool turn whose CONTENT is the JSON report (the new terminator contract).
type submitMock struct {
	summary      string
	grade        int
	achievements []CitedClaim
	failures     []CitedClaim
}

func (m submitMock) Call(_ context.Context, _ llm.Request) (*llm.Response, error) {
	js, _ := json.Marshal(critique{Summary: m.summary, Grade: m.grade, KeyAchievements: m.achievements, FailureModes: m.failures})
	return &llm.Response{Content: string(js)}, nil
}
func (submitMock) Stream(context.Context, llm.Request) (llm.Stream, error) { return nil, nil }
func (submitMock) ModelID() string                                         { return "mock-hq" }
func (submitMock) MaxTokens() int                                          { return 1000 }

// noSubmitMock ends with prose that is NOT a JSON report — exercises the
// fallback sentinel. (It returns the same non-JSON text to any repair call too,
// so repair also fails and the loop degrades.)
type noSubmitMock struct{}

func (noSubmitMock) Call(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{Content: "I refuse to produce a report"}, nil
}
func (noSubmitMock) Stream(context.Context, llm.Request) (llm.Stream, error) { return nil, nil }
func (noSubmitMock) ModelID() string                                         { return "mock-hq" }
func (noSubmitMock) MaxTokens() int                                          { return 1000 }

// --- seed helpers ---

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func newStore(t *testing.T) db.Store {
	t.Helper()
	store, err := sqlite.Open(":memory:")
	must(t, err)
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func seedRun(t *testing.T, store db.Store, runID, task string) {
	t.Helper()
	must(t, store.CreateRun(context.Background(), db.RunRecord{
		RunID:  runID,
		Config: config.RunConfig{Task: task},
	}))
}

func seedSession(t *testing.T, store db.Store, runID, sid, agentType, status string, steps int) {
	t.Helper()
	ctx := context.Background()
	must(t, store.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: sid, AgentType: agentType, Status: status,
	}))
	for range steps {
		_, err := store.AdvanceStep(ctx, runID, sid)
		must(t, err)
	}
}

func addPhase(t *testing.T, store db.Store, runID, sid string, start, end int, title, summary string, artifacts []map[string]any) {
	t.Helper()
	step := end
	must(t, store.AppendObservation(context.Background(), db.ObservationRecord{
		ObsID:     fmt.Sprintf("phase_summary-%s-%d", sid, end),
		RunID:     runID,
		Kind:      "phase_summary",
		SessionID: sid,
		Step:      &step,
		Data:      map[string]any{"title": title, "summary": summary, "start_step": start, "end_step": end, "artifacts": artifacts},
		CreatedAt: time.Now().UTC(),
	}))
}

func addStep(t *testing.T, store db.Store, runID, sid string, n int, summary, tag string) {
	t.Helper()
	step := n
	must(t, store.AppendObservation(context.Background(), db.ObservationRecord{
		ObsID:     fmt.Sprintf("step_summary-%s-%d", sid, n),
		RunID:     runID,
		Kind:      "step_summary",
		SessionID: sid,
		Step:      &step,
		Data:      map[string]any{"summary": summary, "status_tag": tag},
		CreatedAt: time.Now().UTC(),
	}))
}

// --- tests ---

func TestGenerateReport_MechanicalAndCapture(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "do the thing")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 3)
	addPhase(t, store, "r", "main-agent", 1, 2, "Setup", "set things up",
		[]map[string]any{{"kind": "path", "value": "/tmp/x", "context": "config"}})
	addPhase(t, store, "r", "main-agent", 3, 3, "Run", "ran the experiment",
		[]map[string]any{{"kind": "run", "value": "run/123", "context": "main exp"}})
	addStep(t, store, "r", "main-agent", 1, "did setup", "progressing")
	addStep(t, store, "r", "main-agent", 2, "retry build", "retrying")
	addStep(t, store, "r", "main-agent", 3, "blocked on perms", "blocked")

	deps := Deps{Store: store, HQ: submitMock{
		summary:      "good run",
		grade:        4,
		achievements: []CitedClaim{{Statement: "trained model", Citations: []string{"run/123"}}},
	}, CWD: "."}

	report, err := GenerateReport(ctx, deps, "r", nil)
	must(t, err)

	if report.Version != 1 || report.Task != "do the thing" {
		t.Fatalf("version=%d task=%q", report.Version, report.Task)
	}
	if report.Summary != "good run" || report.Grade != 4 || len(report.KeyAchievements) != 1 || report.KeyAchievements[0].Statement != "trained model" {
		t.Fatalf("captured LLM fields wrong: %+v", report)
	}
	if len(report.Phases) != 2 {
		t.Fatalf("phases = %d, want 2: %+v", len(report.Phases), report.Phases)
	}
	if got := report.ArtifactsByKind["run"]; len(got) != 1 || got[0].Value != "run/123" {
		t.Fatalf("run artifacts = %+v", got)
	}
	if len(report.Struggles) != 1 || report.Struggles[0].StartStep != 2 || report.Struggles[0].EndStep != 3 || report.Struggles[0].Length != 2 {
		t.Fatalf("struggles = %+v, want one [2-3] len 2", report.Struggles)
	}
	if s := findState(report.Sessions, "main-agent"); s == nil || s.CurrentStep != 3 {
		t.Fatalf("snapshot main-agent = %+v, want step 3", s)
	}
}

// TestGenerateReport_StruggleGap verifies that a struggling streak with a gap in
// the step_summary step numbers (e.g. a step whose summary was never written) is
// still grouped as ONE struggle. Grouping is by consecutive summarized entries,
// not strict step-number adjacency. See M9.
func TestGenerateReport_StruggleGap(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "task")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 5)
	// Steps 2, 4, 5 are all struggling; step 3's summary is absent (a numeric
	// gap). The old +1 adjacency check would split this into [2] and [4-5].
	addStep(t, store, "r", "main-agent", 2, "retry build", "retrying")
	addStep(t, store, "r", "main-agent", 4, "still retrying", "retrying")
	addStep(t, store, "r", "main-agent", 5, "blocked on perms", "blocked")

	report, err := GenerateReport(ctx, Deps{Store: store, HQ: submitMock{summary: "s"}, CWD: "."}, "r", nil)
	must(t, err)

	if len(report.Struggles) != 1 {
		t.Fatalf("struggles = %+v, want a single grouped struggle", report.Struggles)
	}
	if report.Struggles[0].StartStep != 2 || report.Struggles[0].EndStep != 5 || report.Struggles[0].Length != 3 {
		t.Fatalf("struggle = %+v, want [2-5] len 3", report.Struggles[0])
	}
}

func TestGenerateReport_DeltaScoping(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "task")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 5)
	addPhase(t, store, "r", "main-agent", 1, 2, "Setup", "old work", nil)
	addPhase(t, store, "r", "main-agent", 3, 3, "Run", "old work", nil)
	addPhase(t, store, "r", "main-agent", 4, 5, "More", "new work", nil)
	addStep(t, store, "r", "main-agent", 4, "progress", "progressing")
	addStep(t, store, "r", "main-agent", 5, "retry", "retrying")

	prev := &RunReport{Version: 1, Sessions: []SessionState{{SessionID: "main-agent", CurrentStep: 3}}}
	report, err := GenerateReport(ctx, Deps{Store: store, HQ: submitMock{summary: "s"}, CWD: "."}, "r", prev)
	must(t, err)

	if report.Version != 2 {
		t.Fatalf("version = %d, want 2", report.Version)
	}
	if len(report.Phases) != 1 || report.Phases[0].EndStep != 5 {
		t.Fatalf("delta phases = %+v, want only end_step 5", report.Phases)
	}
	if len(report.Struggles) != 1 || report.Struggles[0].StartStep != 5 {
		t.Fatalf("delta struggles = %+v, want only step 5", report.Struggles)
	}
}

func TestGenerateReport_FallbackOnNoSubmit(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "task")
	seedSession(t, store, "r", "main-agent", "standard_agent", db.SessionConcluded, 1)
	addPhase(t, store, "r", "main-agent", 1, 1, "Phase", "did work", nil)

	report, err := GenerateReport(ctx, Deps{Store: store, HQ: noSubmitMock{}, CWD: "."}, "r", nil)
	must(t, err)
	if !strings.HasPrefix(report.Summary, "[narrative generation failed]") {
		t.Fatalf("summary = %q, want fallback sentinel", report.Summary)
	}
	if len(report.Phases) != 1 {
		t.Fatalf("mechanical phases lost on fallback: %+v", report.Phases)
	}
}

func TestReportStore_Versioning(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()
	seedRun(t, store, "r", "task")

	must(t, writeReport(ctx, store, "r", &RunReport{Version: 1, Summary: "v1", CreatedAt: time.Now().UTC()}))
	must(t, writeReport(ctx, store, "r", &RunReport{Version: 2, Summary: "v2", CreatedAt: time.Now().UTC()}))

	all, err := AllReports(ctx, store, "r")
	must(t, err)
	if len(all) != 2 || all[0].Version != 1 || all[1].Version != 2 {
		t.Fatalf("AllReports = %+v, want [v1, v2]", all)
	}
	latest, err := LatestReport(ctx, store, "r")
	must(t, err)
	if latest == nil || latest.Version != 2 || latest.Summary != "v2" {
		t.Fatalf("LatestReport = %+v, want v2", latest)
	}
	none, err := LatestReport(ctx, store, "nonexistent")
	must(t, err)
	if none != nil {
		t.Fatalf("LatestReport(none) = %+v, want nil", none)
	}
}

func findState(states []SessionState, sid string) *SessionState {
	for i := range states {
		if states[i].SessionID == sid {
			return &states[i]
		}
	}
	return nil
}

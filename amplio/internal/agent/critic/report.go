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
	_ "embed"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"amplio/internal/agent/ephemeral"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	"amplio/internal/skills"
	"amplio/internal/tool"
	"amplio/internal/tool/bash"
	"amplio/internal/tool/inspect"
	"amplio/internal/tool/recall"
	"amplio/internal/tool/sessionsearch"
	"amplio/internal/tool/viewfile"
)

//go:embed keen_critic.md
var keenCriticTemplate string

// criticSystemPrompt returns the keen-critic system prompt with the build-split
// CITATION CONVENTIONS block substituted in (the __CITATION_CONVENTIONS__
// placeholder in keen_critic.md). Lazy (sync.OnceValue) so it reads
// criticCitationConventions AFTER prompts_internal.go's init() override runs on
// the internal build — a package-var initializer would capture the OSS default
// before init().
var criticSystemPrompt = sync.OnceValue(func() string {
	return strings.Replace(keenCriticTemplate, "__CITATION_CONVENTIONS__", criticCitationConventions, 1)
})

// criticMaxIterations bounds the keen-critic's investigation loop. The
// underlying soft guidance is 5-15 calls with a ~25 hard stop; 30 leaves
// headroom for edge cases.
const criticMaxIterations = 30

// Deps are the per-run dependencies the critic needs. The Finalizer builds these
// from a run's stored config (HQ provider, workspace) plus the shared skill index.
type Deps struct {
	Store       db.Store
	HQ          llm.Provider // SYSTEM_HQ tier (the keen-critic LLM)
	SkillIndex  *skills.Index
	LessonIndex *lessons.Index
	CWD         string // run workspace root (for bash / view_file)
}

// critique is the LLM-produced half of the report, parsed from the keen-critic
// loop's final no-tool turn (a JSON object matching these fields).
type critique struct {
	Summary         string       `json:"summary" jsonschema:"required" jsonschema_description:"2-3 paragraph honest narrative assessment of this iteration's work"`
	Grade           int          `json:"grade" jsonschema:"required" jsonschema_description:"Overall verdict on this iteration's work on a 5-level scale: 1=garbage, 2=bad, 3=meh, 4=good, 5=excellent"`
	KeyAchievements []CitedClaim `json:"key_achievements,omitempty" jsonschema_description:"Concrete results produced, each statement citing the artifacts that back it"`
	FailureModes    []CitedClaim `json:"failure_modes,omitempty" jsonschema_description:"Things that did not go well, each statement citing where it happened"`
}

// GenerateReport produces a run report for one iteration. It aggregates the
// mechanical fields (delta-scoped against prev) from observations, then runs the
// keen-critic EphemeralLoop for the evaluative fields. It does NOT persist —
// the Finalizer gates and writes. prev may be nil (first iteration).
func GenerateReport(ctx context.Context, deps Deps, runID string, prev *RunReport) (*RunReport, error) {
	run, err := deps.Store.GetRun(ctx, runID)
	if err != nil {
		return nil, fmt.Errorf("get run: %w", err)
	}
	sessions, err := deps.Store.ListSessions(ctx, runID)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	subjects := subjectSessions(sessions)
	prevStep := func(sid string) int {
		if prev == nil {
			return 0
		}
		return prev.SessionStep(sid)
	}

	phases, artifacts, err := gatherPhasesAndArtifacts(ctx, deps.Store, runID, subjects, prevStep)
	if err != nil {
		return nil, fmt.Errorf("gather phases: %w", err)
	}
	struggles, err := gatherStruggles(ctx, deps.Store, runID, subjects, prevStep)
	if err != nil {
		return nil, fmt.Errorf("gather struggles: %w", err)
	}

	version := 1
	if prev != nil {
		version = prev.Version + 1
	}
	report := &RunReport{
		Version:         version,
		CreatedAt:       time.Now().UTC(),
		Task:            run.Config.Task,
		ArtifactsByKind: artifacts,
		Phases:          phases,
		Struggles:       struggles,
		Sessions:        snapshotSessions(subjects),
	}

	crit := runKeenCritic(ctx, deps, runID, run.Config.Task, prev, report)
	report.Summary = crit.Summary
	report.Grade = crit.Grade
	report.KeyAchievements = crit.KeyAchievements
	report.FailureModes = crit.FailureModes
	return report, nil
}

// runKeenCritic runs the ephemeral investigation loop and returns the parsed
// critique. RunTyped injects the critique JSON schema into the system prompt and
// validates the loop's terminal no-tool turn against it, retrying IN CONTEXT (the
// model sees its own bad reply + the precise validation error) on a malformed or
// schema-invalid final message. On any failure (loop error, never produced a
// valid report, empty summary) it returns a sentinel critique so the mechanical
// report still lands.
func runKeenCritic(ctx context.Context, deps Deps, runID, task string, prev, report *RunReport) critique {
	briefing := buildBriefing(task, prev, report)

	crit, err := ephemeral.RunTyped[critique](ctx, ephemeral.Config{
		LLM:           deps.HQ,
		Tools:         criticTools(deps, runID),
		SystemPrompt:  criticSystemPrompt(),
		MaxIterations: criticMaxIterations,
	}, briefing, 0)
	if err != nil || crit == nil || strings.TrimSpace(crit.Summary) == "" {
		slog.Warn("keen-critic did not produce a valid report", "run_id", runID, "error", err)
		return critique{Summary: "[narrative generation failed] the keen-critic did not produce a valid report; the mechanical fields below are still valid."}
	}
	return *crit
}

// criticTools is the keen-critic's inspection/verification surface: everything
// useful that works in an ephemeral loop (no registry-bound tools, no edit_file —
// a reviewer never mutates the workspace).
func criticTools(deps Deps, runID string) []*tool.Tool {
	cwd := deps.CWD
	if cwd == "" {
		cwd = "."
	}
	inspectDeps := &inspect.Deps{Store: deps.Store, RunID: runID}
	tools := []*tool.Tool{
		inspect.SessionList(inspectDeps),
		inspect.SessionSteps(inspectDeps),
		inspect.SessionPeek(inspectDeps),
		inspect.SessionSummary(inspectDeps),
		sessionsearch.New(deps.Store, runID),
		viewfile.New(cwd, config.ArtifactDir(runID)),
		bash.New(cwd, "", ""), // ephemeral report loop: no notify context
		ViewRunReport(deps.Store, runID),
	}
	// Add the recall tools whenever the index OBJECTS exist (not just when
	// currently built): the tools gate per-corpus with IsBuilt at call time, so
	// an index still building when the critic loop is assembled starts
	// contributing as soon as it's ready. Mirrors standard.go.
	if deps.SkillIndex != nil || deps.LessonIndex != nil {
		tools = append(tools, recall.Search(deps.SkillIndex, deps.LessonIndex), recall.Load(deps.SkillIndex, deps.LessonIndex))
	}
	return tools
}

// buildBriefing renders the keen-critic's task message: iteration framing, the
// original task, the session snapshot (with per-session delta), and the
// delta-scoped phase summaries, artifacts, and struggle ranges.
func buildBriefing(task string, prev, report *RunReport) string {
	var b strings.Builder

	b.WriteString("=== ITERATION ===\n")
	fmt.Fprintf(&b, "This is report iteration %d.\n", report.Version)
	if prev != nil {
		fmt.Fprintf(&b, "The previous report (iteration %d) was generated at %s. Focus on what happened SINCE then; "+
			"call view_run_report to read earlier iterations in full.\n", prev.Version, prev.CreatedAt.Format(time.RFC3339))
	}

	b.WriteString("\n=== ORIGINAL TASK ===\n")
	if strings.TrimSpace(task) == "" {
		b.WriteString("(interactive run — no fixed task)\n")
	} else {
		b.WriteString(task + "\n")
	}

	b.WriteString("\n=== SESSION SNAPSHOT ===\n")
	for _, s := range report.Sessions {
		delta := ""
		if prev != nil {
			if adv := s.CurrentStep - prev.SessionStep(s.SessionID); adv > 0 {
				delta = fmt.Sprintf(", +%d steps since last report", adv)
			}
		}
		fmt.Fprintf(&b, "- %s (agent=%s, status=%s, step=%d%s)\n", s.SessionID, s.AgentType, s.Status, s.CurrentStep, delta)
	}

	b.WriteString("\n=== PHASE SUMMARIES (this iteration) ===\n")
	if len(report.Phases) == 0 {
		b.WriteString("(no closed phases — inspect raw steps with your tools if needed)\n")
	}
	for _, p := range report.Phases {
		fmt.Fprintf(&b, "--- session=%s steps %d-%d: %s ---\n%s\n\n", p.SessionID, p.StartStep, p.EndStep, p.Title, p.Summary)
	}

	if len(report.ArtifactsByKind) > 0 {
		b.WriteString("=== ARTIFACTS (this iteration) ===\n")
		for kind, arts := range report.ArtifactsByKind {
			fmt.Fprintf(&b, "- %s:\n", kind)
			for _, a := range arts {
				fmt.Fprintf(&b, "    - %s (session=%s steps %d-%d, context=%s)\n", a.Value, a.SessionID, a.StartStep, a.EndStep, a.Context)
			}
		}
		b.WriteString("\n")
	}

	if len(report.Struggles) > 0 {
		b.WriteString("=== STRUGGLE RANGES (this iteration) ===\n")
		for _, s := range report.Struggles {
			fmt.Fprintf(&b, "- session=%s %s\n", s.SessionID, stepRange(s.StartStep, s.EndStep))
			if len(s.SampleSummaries) > 0 {
				fmt.Fprintf(&b, "    e.g. %s\n", s.SampleSummaries[0])
			}
		}
		b.WriteString("\n")
	}

	b.WriteString("=== END BRIEFING ===\n")
	b.WriteString("Investigate with your tools as needed, then finish per the Final output contract in the system prompt.")
	return b.String()
}

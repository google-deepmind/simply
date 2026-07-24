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

// Package compaction summarizes a session's prior context into one cumulative
// memory when its conversation outgrows the model's context window.
//
// It is reactive: the event loop invokes Compact only after the provider rejects
// a call for exceeding the context window (confirmed by a fast-model judge).
// Compact reads the curated, bounded layered view of the session (so the
// summarizer itself fits in context) plus optional raw-step drill-down, and
// returns prose the caller commits via Store.CompactContext.
package compaction

import (
	"context"
	_ "embed"
	"fmt"
	"strings"
	"sync"

	"amplio/internal/agent/ephemeral"
	"amplio/internal/db"
	"amplio/internal/llm"
	"amplio/internal/tool"
	"amplio/internal/tool/inspect"
)

//go:embed compaction.md
var systemPromptTemplate string

// systemPrompt returns the compaction system prompt with the build-split
// artifact-identifier examples substituted in (the __ARTIFACT_ID_EXAMPLES__
// placeholder in compaction.md). Lazy (sync.OnceValue) so it reads
// compactionArtifactIDExamples AFTER prompts_internal.go's init() override runs
// on the internal build.
var systemPrompt = sync.OnceValue(func() string {
	return strings.Replace(systemPromptTemplate, "__ARTIFACT_ID_EXAMPLES__", compactionArtifactIDExamples, 1)
})

// maxIterations bounds the drill-down loop so a misbehaving summarizer can't
// spin forever on the agent's critical recovery path.
const maxIterations = 12

// Deps holds what Compact needs: the store to read session state, the HQ
// provider that writes the summary, and the target session coordinates.
type Deps struct {
	Store     db.Store
	HQ        llm.Provider
	RunID     string
	SessionID string
}

// Compact produces a cumulative first-person memory summary of the session's
// work through boundaryStep. It seeds an ephemeral HQ loop with the curated
// layered view and a session_steps tool for optional drill-down into specific
// raw steps, then returns the summary prose augmented with a deterministic
// structured trace of the most recent activity (see FormatOpenPhase). The
// caller commits it via Store.CompactContext.
func Compact(ctx context.Context, deps Deps, boundaryStep int) (string, error) {
	layered, err := inspect.LayeredSummary(ctx, deps.Store, deps.RunID, deps.SessionID, boundaryStep)
	if err != nil {
		return "", fmt.Errorf("build layered view: %w", err)
	}

	inspectDeps := &inspect.Deps{Store: deps.Store, RunID: deps.RunID}
	summary, err := ephemeral.Run(ctx, ephemeral.Config{
		LLM:           deps.HQ,
		Tools:         []*tool.Tool{inspect.SessionSteps(inspectDeps)},
		SystemPrompt:  systemPrompt(),
		MaxIterations: maxIterations,
	}, buildTask(deps.SessionID, boundaryStep, layered))
	if err != nil {
		return "", fmt.Errorf("compaction loop: %w", err)
	}
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return "", fmt.Errorf("compaction produced an empty summary")
	}

	// Augment the LLM prose with a deterministic, procedural trace of the most
	// recent activity (the always-open final phase). The prose reliably loses
	// low-level specifics — exact commands, file paths, tool order, external
	// inputs — that this grounds. Runs BEFORE CompactContext bumps the
	// generation, so the current-context read sees exactly the compacted range.
	// Best-effort: a formatting error must not fail the critical compaction path.
	if recent, _, err := inspect.FormatOpenPhase(ctx, deps.Store, deps.RunID, deps.SessionID, boundaryStep, 0); err == nil {
		if recent = strings.TrimSpace(recent); recent != "" {
			summary += "\n\n--- RECENT ACTIVITY (structured trace of the last phase) ---\n" + recent
		}
	}
	return summary, nil
}

func buildTask(sessionID string, boundaryStep int, layered string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Write a cumulative memory summary of session %q covering its work through step %d.\n\n",
		sessionID, boundaryStep)
	b.WriteString("Call session_steps on this session to drill into specific raw steps when you need exact " +
		"details (artifact ids, file paths, commands, error text) the curated view below omits. When you have " +
		"what you need, reply with the summary prose as your final message and no tool call.\n\n")
	b.WriteString("=== CURATED VIEW OF WORK SO FAR ===\n")
	b.WriteString(layered)
	return b.String()
}

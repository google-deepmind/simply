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

// LLM-backed hooks the serve mode injects into the HTTP server. These live in
// cmd/amplio (not internal/server) because they build on provider construction
// (createProvider, the system-tier providers) that is owned by the entrypoint;
// the server only holds the resulting func values. Wired in serve.go via
// server.SetLLMTester / server.SetFollowupSuggester.
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"amplio/internal/agent/critic"
	"amplio/internal/db"
	"amplio/internal/llm"
)

// testLLMProbePrompt is the trivial smoke-test message: cheap, deterministic,
// and unambiguous that a non-empty reply means the round-trip worked.
const testLLMProbePrompt = "Reply with the single word: OK"

// testLLM is the About-page pre-flight check, injected into the server
// (server.SetLLMTester). It separates the two failure classes a misconfigured
// model spec produces: a spec parse / unknown-provider error surfaces instantly
// and for free from createProvider; an auth / scope / availability error
// surfaces from one trivial, short-timeout Call. Returns the resolved model id,
// the (truncated) reply, and the call latency on success.
func testLLM(ctx context.Context, spec string) (modelID, reply string, latency time.Duration, err error) {
	provider, err := createProvider(spec)
	if err != nil {
		return "", "", 0, err
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	start := time.Now()
	resp, err := provider.Call(ctx, llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: testLLMProbePrompt}},
	})
	latency = time.Since(start)
	if err != nil {
		return "", "", 0, err
	}
	return provider.ModelID(), truncate(strings.TrimSpace(resp.Content), 200), latency, nil
}

// followupSystemPrompt steers the HQ model to draft ONE actionable follow-up
// instruction (not to perform the work). The operator edits the result before
// sending it as the next iteration's user message.
const followupSystemPrompt = "You are helping an operator plan the next iteration of an " +
	"autonomous coding/research run. You are given the ORIGINAL TASK and the critic's REPORT of " +
	"what happened. Propose ONE concrete, actionable follow-up instruction the operator could send " +
	"to continue or improve the work — e.g. address a specific failure mode, finish an unfinished " +
	"thread, harden/verify a result, or take the natural next step. Be specific and reference " +
	"concrete gaps from the report. Output ONLY the instruction prose addressed to the agent (no " +
	"preamble, no headings, no markdown fences, no quotes). Do NOT attempt the work yourself."

// makeFollowupSuggester returns the server's follow-up drafter: it reads the
// run's original task + latest report and asks the HQ model for a follow-up
// instruction the operator can edit before sending. Synchronous (the UI awaits
// it); errors propagate to the caller, which surfaces them inline.
func makeFollowupSuggester(store db.Store, hq llm.Provider) func(ctx context.Context, runID string) (string, error) {
	return func(ctx context.Context, runID string) (string, error) {
		run, err := store.GetRun(ctx, runID)
		if err != nil {
			return "", fmt.Errorf("load run: %w", err)
		}
		report, err := critic.LatestReport(ctx, store, runID)
		if err != nil {
			return "", fmt.Errorf("load report: %w", err)
		}
		if report == nil {
			return "", fmt.Errorf("no report available yet for this run")
		}
		ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
		defer cancel()
		resp, err := hq.Call(ctx, llm.Request{
			SystemPrompt: followupSystemPrompt,
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: "[ORIGINAL TASK]\n" + run.Config.Task + "\n\n[CRITIC REPORT]\n" + renderReportForFollowup(report),
			}},
		})
		if err != nil {
			return "", err
		}
		prompt := strings.TrimSpace(resp.Content)
		if prompt == "" {
			return "", fmt.Errorf("the model returned an empty suggestion")
		}
		return prompt, nil
	}
}

// renderReportForFollowup flattens a run report into a compact plain-text block
// for the follow-up prompt: the same signal the operator sees (summary, key
// achievements, failure modes, struggles), without citation noise.
func renderReportForFollowup(r *critic.RunReport) string {
	var b strings.Builder
	if s := strings.TrimSpace(r.Summary); s != "" {
		fmt.Fprintf(&b, "Summary:\n%s\n", s)
	}
	writeClaims := func(label string, claims []critic.CitedClaim) {
		if len(claims) == 0 {
			return
		}
		fmt.Fprintf(&b, "\n%s:\n", label)
		for _, c := range claims {
			if s := strings.TrimSpace(c.Statement); s != "" {
				fmt.Fprintf(&b, "- %s\n", s)
			}
		}
	}
	writeClaims("Key achievements", r.KeyAchievements)
	writeClaims("Failure modes", r.FailureModes)
	if len(r.Struggles) > 0 {
		b.WriteString("\nStruggles (repeated difficulty):\n")
		for _, s := range r.Struggles {
			for _, sample := range s.SampleSummaries {
				if t := strings.TrimSpace(sample); t != "" {
					fmt.Fprintf(&b, "- %s\n", t)
				}
			}
		}
	}
	return b.String()
}

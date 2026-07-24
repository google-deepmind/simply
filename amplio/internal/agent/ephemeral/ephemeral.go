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

// Package ephemeral provides a lightweight in-memory LLM-with-tools loop.
//
// Unlike EventLoopAgent, the ephemeral loop has no persistent session,
// no DB events, no registry, no step tracking, no park/wake lifecycle.
// It manages its own []llm.Message conversation in memory, runs tools,
// and returns the final text response.
//
// Used by:
//   - Compaction agent: inspect session events → produce summary
//   - Report generation (keen-critic): inspect run → produce report
//   - Any helper agent that runs as a sub-task and returns a result
package ephemeral

import (
	"context"
	"fmt"
	"log/slog"

	"amplio/internal/llm"
	"amplio/internal/tool"
)

// Config holds the configuration for an ephemeral loop.
type Config struct {
	LLM           llm.Provider
	Tools         []*tool.Tool
	SystemPrompt  string
	MaxIterations int // safety limit to prevent infinite loops (0 = default 50)
}

const defaultMaxIterations = 50

// Run executes a multi-turn LLM conversation with tools entirely in memory.
// Returns the final text response (the first no-tool LLM reply).
func Run(ctx context.Context, cfg Config, task string) (string, error) {
	return run(ctx, cfg, task, nil)
}

// run is the shared loop. onFinal, when non-nil, is invoked on each no-tool turn
// (the would-be terminal reply): it returns (followup, done). done=true ends the
// loop returning that content; done=false appends `followup` as a new USER turn
// and the loop continues (the model keeps full context, including its own bad
// reply — so we never re-paste it). nil onFinal = the first no-tool turn ends
// the loop (plain Run behavior).
func run(ctx context.Context, cfg Config, task string, onFinal func(content string) (string, bool)) (string, error) {
	toolMap := tool.ByName(cfg.Tools)
	toolDefs := tool.Defs(cfg.Tools)

	// Reject tools that require session infrastructure.
	for _, t := range cfg.Tools {
		if t.SessionRequired {
			return "", fmt.Errorf("ephemeral loop: tool %q requires session infrastructure and cannot be used in an ephemeral loop", t.Name)
		}
	}

	maxIter := cfg.MaxIterations
	if maxIter <= 0 {
		maxIter = defaultMaxIterations
	}

	// Build initial messages.
	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: cfg.SystemPrompt},
		{Role: llm.RoleUser, Content: task},
	}

	// finalizing is set once onFinal has REJECTED a terminal turn: from then on we
	// stop offering tools, so each repair turn must produce a final answer (it
	// can't wander back into tool calls). This both matches intent — fixing an
	// already-produced answer needs no investigation, the context has everything
	// — and makes the caller's repair budget a real bound (every retry is a
	// no-tool turn that re-enters onFinal).
	finalizing := false

	for iter := range maxIter {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}

		req := llm.Request{Messages: messages}
		if !finalizing {
			req.Tools = toolDefs
		}
		resp, err := cfg.LLM.Call(ctx, req)
		if err != nil {
			return "", fmt.Errorf("ephemeral loop iteration %d: %w", iter, err)
		}

		// Append assistant message. ProviderExtra MUST be carried forward: it holds
		// provider-namespaced cargo (e.g. Gemini per-tool-call thought_signatures)
		// that must be replayed on the next iteration's function_call parts, or the
		// provider rejects the follow-up turn (Gemini 400). The eventloop persists +
		// replays this too; the ephemeral loop keeps it in-memory across iterations.
		assistantMsg := llm.Message{
			Role:          llm.RoleAssistant,
			Content:       resp.Content,
			ToolCalls:     resp.ToolCalls,
			ProviderExtra: resp.ProviderExtra,
		}
		messages = append(messages, assistantMsg)

		// No tool calls — the would-be terminal turn.
		if len(resp.ToolCalls) == 0 {
			if onFinal == nil {
				return resp.Content, nil
			}
			followup, done := onFinal(resp.Content)
			if done {
				return resp.Content, nil
			}
			// Rejected: stop offering tools and nudge with a user turn. The bad
			// reply is already in `messages` as the prior assistant turn (full
			// context), so we never re-paste it.
			finalizing = true
			messages = append(messages, llm.Message{Role: llm.RoleUser, Content: followup})
			continue
		}

		// Execute tool calls and append results.
		callResults := tool.ExecuteAll(ctx, resp.ToolCalls, toolMap, nil)
		for _, r := range callResults {
			messages = append(messages, llm.Message{
				Role:       llm.RoleToolResult,
				ToolCallID: r.ToolCallID,
				Content:    r.Result.Content,
				IsError:    r.Result.IsError,
			})
		}

		slog.Debug("ephemeral loop iteration", "iter", iter, "tools", len(resp.ToolCalls))
	}

	return "", fmt.Errorf("ephemeral loop exceeded %d iterations", maxIter)
}

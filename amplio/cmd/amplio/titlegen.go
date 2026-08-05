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

package main

import (
	"context"
	"log/slog"
	"strings"
	"time"

	"amplio/internal/db"
	"amplio/internal/llm"
)

// Run-title generation: ask the fast system model for a short title and
// sanitize what comes back.

const titleSystemPrompt = "You are titling a task for a list UI. Output ONLY a short, " +
	"specific title of 3-8 words. Do NOT write code, explanations, quotes, or markdown " +
	"fences — respond with the bare title text only. Do NOT try to work on the task."

// makeTitleGenerator returns a fire-and-forget run-title generator: it asks the
// fast system model for a short title and writes it via UpdateRunTitle. Failure
// is logged and ignored — the title just stays empty and the UI falls back to a
// task prefix.
func makeTitleGenerator(store db.Store, fast llm.Provider) func(runID, task string) {
	return func(runID, task string) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		resp, err := fast.Call(ctx, llm.Request{
			SystemPrompt: titleSystemPrompt,
			// Frame the task as data (not an instruction to execute) and repeat
			// the ask in the user turn — weaker models otherwise try to do it.
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: "Write a short title for the following task. Do not perform it.\n\nTASK:\n" + task,
			}},
		})
		if err != nil {
			slog.Warn("title generation failed", "run_id", runID, "error", err)
			return
		}
		title := sanitizeTitle(resp.Content)
		if title == "" {
			// Model returned code/empty/unusable: leave the title empty (the UI
			// falls back to a task prefix) but log so it's visible.
			slog.Warn("title generation produced no usable title",
				"run_id", runID, "raw", truncate(resp.Content, 120))
			return
		}
		if err := store.UpdateRunTitle(ctx, runID, title); err != nil {
			slog.Warn("title update failed", "run_id", runID, "error", err)
			return
		}
		slog.Info("generated run title", "run_id", runID, "title", title)
	}
}

// sanitizeTitle extracts a clean single-line title from a model response, or ""
// if the response looks like code or yields nothing usable.
func sanitizeTitle(raw string) string {
	s := strings.TrimSpace(raw)
	// A leading fence means the model wrote a code block, not a title.
	if s == "" || strings.HasPrefix(s, "```") {
		return ""
	}
	line := s
	if i := strings.IndexByte(line, '\n'); i >= 0 {
		line = line[:i] // first line only
	}
	line = strings.TrimSpace(line)
	line = strings.TrimLeft(line, "#>*-+ \t") // markdown heading / list markers
	line = strings.Trim(line, "`\"'")         // surrounding fences / quotes
	line = strings.TrimSpace(line)
	if r := []rune(line); len(r) > 80 {
		line = strings.TrimSpace(string(r[:80]))
	}
	return line
}

// truncate shortens s to at most n runes for logging.
func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}

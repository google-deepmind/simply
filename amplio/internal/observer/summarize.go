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

package observer

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/llm"
	"amplio/internal/llm/jsonextract"

	"github.com/invopop/jsonschema"
)

// stepSystemPrompt is built lazily (not a package var) so the GOOD examples
// fragment reflects any build-tag override applied in an init() — package-var
// initializers run before init(), which would otherwise capture the OSS default.
var stepSystemPrompt = sync.OnceValue(func() string {
	return `You are an automated telemetry observer generating a fast, skimmable activity log for a single step of an autonomous agent's execution.
You have NO context of the past or future trajectory—evaluate ONLY the exact events in this step (the assistant reasoning, tool calls, and tool results).

Your job is to distill the step into a dense log line and assign a status tag.

STYLE RULES FOR THE SUMMARY:
* Be objective and observational. Avoid subjective interpretations or evaluations.
* Be concise but information-dense, no filler words ("in order to", "began to", etc.) or useless prefixes ("The agent...", "It...").
* Document actions, apparent goal, key inputs, and the direct outcome (success, failure, errors, new conclusions or immediate results).
* Be hyper-specific: include actual file paths, tool names, metric values, or error codes.

EXAMPLES:
` + stepGoodExamples + `
* BAD: "The agent began exploring the directory structure by listing..." (Too verbose, passive)
* BAD: "Made progress on the task by examining the codebase layout." (Subjective, lacks specific entities)

Your output MUST be a single JSON object matching the provided schema. Do not include markdown code fences, preamble, or any extra text.`
})

// stepSummaryOutput is the LLM's structured output. char_count is added by the
// observer before persisting; it is not part of the LLM contract.
type stepSummaryOutput struct {
	Summary   string `json:"summary" jsonschema_description:"Dense single-line activity log for this step (8-20 words)"`
	StatusTag string `json:"status_tag" jsonschema:"enum=progressing,enum=retrying,enum=blocked" jsonschema_description:"Step status: progressing | retrying | blocked"`
}

// stepSummarySchema renders the output JSON schema once for embedding in prompts.
var stepSummarySchema = sync.OnceValue(func() string {
	r := &jsonschema.Reflector{DoNotReference: true}
	b, _ := json.MarshalIndent(r.Reflect(&stepSummaryOutput{}), "", "  ") //nolint:errcheck
	return string(b)
})

// summarizeStep produces the step_summary data payload. On any LLM or parse
// failure it returns a degraded payload (with the reason) rather than failing,
// so the per-session pipeline keeps moving; the row can be re-run later by
// deleting it and letting the watermark refire.
func summarizeStep(ctx context.Context, llmFast llm.Provider, sessionID string, step int, events []event.Event) map[string]any {
	resp, err := llmFast.Call(ctx, llm.Request{
		SystemPrompt: stepSystemPrompt(),
		Messages:     []llm.Message{{Role: llm.RoleUser, Content: buildStepUserPrompt(sessionID, step, events)}},
	})
	if err != nil {
		return stepFailurePayload("LLM call failed: " + err.Error())
	}
	raw := strings.TrimSpace(resp.Content)
	if raw == "" {
		return stepFailurePayload("LLM returned empty content")
	}
	parsed, ok := parseStepSummary(ctx, llmFast, raw)
	if !ok {
		return stepFailurePayload("LLM did not return a valid step-summary JSON object")
	}
	return parsed
}

func buildStepUserPrompt(sessionID string, step int, events []event.Event) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Summarize step %d of session %q. Produce a single JSON object matching this schema:\n", step, sessionID)
	b.WriteString(stepSummarySchema())
	b.WriteString("\n\n")
	b.WriteString(renderStep(events, step, "this is what you are summarizing"))
	return b.String()
}

// renderStep renders a step's events under a header supplying the temporal
// context that Event.ToText() deliberately omits.
func renderStep(events []event.Event, step int, banner string) string {
	if len(events) == 0 {
		return fmt.Sprintf("=== STEP %d (%s) ===\n(no events)", step, banner)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "=== STEP %d (%s) ===\n", step, banner)
	for _, ev := range events {
		b.WriteString(ev.ToText())
		b.WriteString("\n\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// parseStepSummary extracts the typed step summary, asking llmFast to repair a
// formatting error once before giving up (the caller then degrades). The fast
// tier is fine for a pure JSON-syntax fix.
func parseStepSummary(ctx context.Context, llmFast llm.Provider, raw string) (map[string]any, bool) {
	out, err := jsonextract.Extract[stepSummaryOutput](ctx, raw, jsonextract.Options{
		Repair: llmFast,
		Hint:   "a single JSON object with string fields \"summary\" and \"status_tag\"",
	})
	if err != nil {
		return nil, false
	}
	if strings.TrimSpace(out.Summary) == "" {
		return nil, false
	}
	switch out.StatusTag {
	case "progressing", "retrying", "blocked":
	default:
		out.StatusTag = "progressing" // neutral fallback for missing/unknown tag
	}
	return map[string]any{"summary": out.Summary, "status_tag": out.StatusTag}, true
}

func stepFailurePayload(reason string) map[string]any {
	return map[string]any{
		"summary":    db.SummarizationFailedPrefix + " " + reason,
		"status_tag": "progressing",
	}
}

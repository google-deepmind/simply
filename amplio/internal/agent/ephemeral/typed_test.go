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

package ephemeral

import (
	"context"
	"strings"
	"testing"

	"amplio/internal/llm"
	"amplio/internal/tool"
)

type out struct {
	Summary string `json:"summary" jsonschema:"required"`
	Grade   int    `json:"grade" jsonschema:"required"`
}

func TestRunTyped_HappyPath(t *testing.T) {
	mock := &llm.MockProvider{
		Model:     "test",
		Responses: []llm.Response{{Content: `{"summary":"ok","grade":4}`}},
	}
	got, err := RunTyped[out](context.Background(), Config{LLM: mock, SystemPrompt: "sys"}, "do it", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.Summary != "ok" || got.Grade != 4 {
		t.Errorf("got %+v", got)
	}
	if mock.CallCount() != 1 {
		t.Errorf("calls = %d, want 1", mock.CallCount())
	}
	// The schema must have been appended to the system prompt.
	if !strings.Contains(mock.Recorded()[0].Messages[0].Content, "Final output") {
		t.Errorf("system prompt missing output contract: %s", mock.Recorded()[0].Messages[0].Content)
	}
}

func TestRunTyped_InContextContentRepair(t *testing.T) {
	// First terminal turn is schema-invalid (missing required grade); the model
	// is nudged and the second turn is valid. Verifies in-loop retry.
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			{Content: `{"summary":"ok"}`},           // invalid: missing grade
			{Content: `{"summary":"ok","grade":5}`}, // fixed after nudge
		},
	}
	got, err := RunTyped[out](context.Background(), Config{LLM: mock, SystemPrompt: "sys"}, "do it", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.Grade != 5 {
		t.Errorf("got %+v", got)
	}
	if mock.CallCount() != 2 {
		t.Errorf("calls = %d, want 2 (one nudge)", mock.CallCount())
	}
	// The nudge user turn should carry the validation error mentioning 'grade'.
	last := mock.Recorded()[1].Messages
	var nudge string
	for _, m := range last {
		if m.Role == llm.RoleUser {
			nudge = m.Content // last user message
		}
	}
	if !strings.Contains(nudge, "grade") {
		t.Errorf("nudge should cite the missing field: %q", nudge)
	}
	// The nudge should also re-state the schema (anchors the fix on a long loop).
	if !strings.Contains(nudge, "\"properties\"") {
		t.Errorf("nudge should re-include the schema: %q", nudge)
	}
}

func TestRunTyped_RepairTurnDropsTools(t *testing.T) {
	// After an invalid final turn, the repair call must NOT offer tools (forcing a
	// final answer + keeping the repair budget a real bound).
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			{Content: `{"summary":"ok"}`},           // invalid → triggers repair
			{Content: `{"summary":"ok","grade":1}`}, // valid
		},
	}
	tl := echoTool()
	_, err := RunTyped[out](context.Background(),
		Config{LLM: mock, Tools: []*tool.Tool{tl}, SystemPrompt: "sys"}, "do it", 0)
	if err != nil {
		t.Fatal(err)
	}
	rec := mock.Recorded()
	if len(rec) != 2 {
		t.Fatalf("calls = %d, want 2", len(rec))
	}
	if len(rec[0].Tools) == 0 {
		t.Error("first (investigation) call should offer tools")
	}
	if len(rec[1].Tools) != 0 {
		t.Errorf("repair call should NOT offer tools, got %d", len(rec[1].Tools))
	}
}

func TestRunTyped_ExhaustsRepairs(t *testing.T) {
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			{Content: `{"summary":"ok"}`}, // invalid
			{Content: `{"summary":"ok"}`}, // still invalid
		},
	}
	_, err := RunTyped[out](context.Background(), Config{LLM: mock, SystemPrompt: "sys"}, "do it", 2)
	if err == nil || !strings.Contains(err.Error(), "no schema-valid output") {
		t.Fatalf("expected exhaustion error, got: %v", err)
	}
	if mock.CallCount() != 2 {
		t.Errorf("calls = %d, want 2", mock.CallCount())
	}
}

func TestRunTyped_FenceTolerant(t *testing.T) {
	mock := &llm.MockProvider{
		Model:     "test",
		Responses: []llm.Response{{Content: "```json\n{\"summary\":\"ok\",\"grade\":2}\n```"}},
	}
	got, err := RunTyped[out](context.Background(), Config{LLM: mock, SystemPrompt: "sys"}, "do it", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.Grade != 2 || mock.CallCount() != 1 {
		t.Errorf("got %+v calls=%d", got, mock.CallCount())
	}
}

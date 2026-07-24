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
	"encoding/json"
	"strings"
	"testing"

	"amplio/internal/llm"
	"amplio/internal/tool"
)

type echoParams struct {
	Message string `json:"message" jsonschema:"required"`
}

func echoTool() *tool.Tool {
	return &tool.Tool{
		Name:        "echo",
		Description: "Echo a message",
		ParamType:   &echoParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[echoParams](args)
			if errResult != nil {
				return errResult, nil
			}
			return &tool.Result{Content: "Echo: " + params.Message}, nil
		},
	}
}

func TestRun_SimpleCompletion(t *testing.T) {
	mock := &llm.MockProvider{
		Model:     "test",
		Responses: []llm.Response{{Content: "The answer is 42."}},
	}

	result, err := Run(context.Background(), Config{
		LLM:          mock,
		SystemPrompt: "You are helpful.",
	}, "What is 42?")

	if err != nil {
		t.Fatal(err)
	}
	if result != "The answer is 42." {
		t.Errorf("result: %q", result)
	}
	if mock.CallCount() != 1 {
		t.Errorf("LLM called %d times", mock.CallCount())
	}
	if len(mock.Recorded()[0].Messages) != 2 {
		t.Errorf("messages: %d, want 2 (system + user)", len(mock.Recorded()[0].Messages))
	}
}

// A response's ProviderExtra (e.g. Gemini thought_signatures) must be carried
// onto the replayed assistant message, or the next iteration drops it and the
// provider rejects the follow-up turn. Regression test for the ephemeral loop
// omitting ProviderExtra (which the eventloop already carried).
func TestRun_ReplaysProviderExtra(t *testing.T) {
	extra := map[string]any{"gemini.fc_sigs_b64": []string{"SIG"}}
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			// First turn: tool call + provider cargo, forcing a second iteration.
			{Content: "calling", ProviderExtra: extra, ToolCalls: []llm.ToolCall{
				{ID: "tc1", Name: "echo", Arguments: `{"message":"hi"}`},
			}},
			{Content: "done"},
		},
	}

	if _, err := Run(context.Background(), Config{
		LLM:          mock,
		Tools:        []*tool.Tool{echoTool()},
		SystemPrompt: "You are helpful.",
	}, "Echo hi"); err != nil {
		t.Fatal(err)
	}
	if mock.CallCount() != 2 {
		t.Fatalf("LLM called %d times, want 2", mock.CallCount())
	}
	// The second call's replayed assistant turn must still carry ProviderExtra.
	var found *llm.Message
	for i := range mock.Recorded()[1].Messages {
		if m := &mock.Recorded()[1].Messages[i]; m.Role == llm.RoleAssistant {
			found = m
		}
	}
	if found == nil {
		t.Fatal("no assistant message in the second LLM call")
	}
	sigs, ok := found.ProviderExtra["gemini.fc_sigs_b64"].([]string)
	if !ok || len(sigs) != 1 || sigs[0] != "SIG" {
		t.Errorf("assistant ProviderExtra = %v, want the replayed thought signatures", found.ProviderExtra)
	}
}

func TestRun_WithToolCalls(t *testing.T) {
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			{Content: "Let me echo.", ToolCalls: []llm.ToolCall{
				{ID: "tc1", Name: "echo", Arguments: `{"message":"hello"}`},
			}},
			{Content: "The echo said: hello."},
		},
	}

	result, err := Run(context.Background(), Config{
		LLM:          mock,
		Tools:        []*tool.Tool{echoTool()},
		SystemPrompt: "You are helpful.",
	}, "Echo hello")

	if err != nil {
		t.Fatal(err)
	}
	if result != "The echo said: hello." {
		t.Errorf("result: %q", result)
	}
	if mock.CallCount() != 2 {
		t.Errorf("LLM called %d times, want 2", mock.CallCount())
	}
	lastCall := mock.Recorded()[1]
	var hasToolResult bool
	for _, m := range lastCall.Messages {
		if m.Role == llm.RoleToolResult && strings.Contains(m.Content, "Echo: hello") {
			hasToolResult = true
		}
	}
	if !hasToolResult {
		t.Error("expected tool result in second LLM call")
	}
}

func TestRun_UnknownTool(t *testing.T) {
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			{ToolCalls: []llm.ToolCall{{ID: "tc1", Name: "nonexistent", Arguments: `{}`}}},
			{Content: "OK, tool not found."},
		},
	}

	result, err := Run(context.Background(), Config{
		LLM:          mock,
		SystemPrompt: "You are helpful.",
	}, "Try a tool")

	if err != nil {
		t.Fatal(err)
	}
	if result != "OK, tool not found." {
		t.Errorf("result: %q", result)
	}
}

func TestRun_MaxIterations(t *testing.T) {
	var responses []llm.Response
	for range 10 {
		responses = append(responses, llm.Response{
			ToolCalls: []llm.ToolCall{{ID: "tc", Name: "echo", Arguments: `{"message":"loop"}`}},
		})
	}

	_, err := Run(context.Background(), Config{
		LLM:           &llm.MockProvider{Model: "test", Responses: responses},
		Tools:         []*tool.Tool{echoTool()},
		SystemPrompt:  "You are helpful.",
		MaxIterations: 3,
	}, "Loop forever")

	if err == nil || !strings.Contains(err.Error(), "exceeded 3 iterations") {
		t.Errorf("expected max iterations error, got: %v", err)
	}
}

func TestRun_Cancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := Run(ctx, Config{
		LLM:          &llm.MockProvider{Model: "test"},
		SystemPrompt: "You are helpful.",
	}, "Cancelled")

	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestRun_RejectsSessionRequiredTools(t *testing.T) {
	sessionTool := &tool.Tool{
		Name:            "spawn_agent",
		Description:     "Spawn a sub-agent",
		ParamType:       &struct{}{},
		SessionRequired: true,
		Execute: func(_ context.Context, _ json.RawMessage) (*tool.Result, error) {
			return &tool.Result{Content: "should not run"}, nil
		},
	}

	_, err := Run(context.Background(), Config{
		LLM:          &llm.MockProvider{Model: "test"},
		Tools:        []*tool.Tool{sessionTool},
		SystemPrompt: "test",
	}, "test")

	if err == nil || !strings.Contains(err.Error(), "requires session infrastructure") {
		t.Errorf("expected session-required rejection, got: %v", err)
	}
}

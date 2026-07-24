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

package llm

import (
	"context"
	"testing"
)

func TestMockProvider_Call(t *testing.T) {
	m := &MockProvider{
		Model: "test-model",
		Responses: []Response{
			{Content: "hello", StopReason: "end_turn"},
			{Content: "world", ToolCalls: []ToolCall{{ID: "tc1", Name: "bash", Arguments: `{"cmd":"ls"}`}}},
		},
	}

	resp, err := m.Call(context.Background(), Request{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hello" {
		t.Errorf("first call: %q", resp.Content)
	}

	resp, err = m.Call(context.Background(), Request{Messages: []Message{{Role: RoleUser, Content: "hi again"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "world" {
		t.Errorf("second call: %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "bash" {
		t.Errorf("tool calls: %v", resp.ToolCalls)
	}

	// Third call — past the response list, should return empty.
	resp, _ = m.Call(context.Background(), Request{})
	if resp.Content != "" {
		t.Errorf("exhausted: %q", resp.Content)
	}

	// Verify recorded calls.
	if m.CallCount() != 3 {
		t.Errorf("recorded %d calls", m.CallCount())
	}
}

func TestMockProvider_Stream(t *testing.T) {
	m := &MockProvider{
		Model:     "test-model",
		Responses: []Response{{Content: "streamed content"}},
	}

	stream, err := m.Stream(context.Background(), Request{})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	if !stream.Next() {
		t.Fatal("expected at least one chunk")
	}
	evt := stream.Event()
	if evt.DeltaText != "streamed content" {
		t.Errorf("delta: %q", evt.DeltaText)
	}

	if stream.Next() {
		t.Error("expected no more chunks")
	}

	resp := stream.Response()
	if resp.Content != "streamed content" {
		t.Errorf("final response: %q", resp.Content)
	}
}

func TestMockProvider_ModelID(t *testing.T) {
	m := &MockProvider{Model: "my-model"}
	if m.ModelID() != "my-model" {
		t.Errorf("ModelID: %q", m.ModelID())
	}
}

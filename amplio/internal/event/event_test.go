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

package event

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRoundTrip_AllTypes(t *testing.T) {
	events := []Event{
		&SystemEvent{Content: "system prompt here", Marker: "bootstrap"},
		&UserEvent{Content: "user message"},
		&AssistantEvent{
			Content: "response text",
			ToolCalls: []ToolCall{
				{ID: "tc1", Name: "bash", Arguments: `{"command":"ls"}`},
			},
			Thoughts: "thinking...",
			Usage:    &Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
		},
		&ToolResultEvent{
			Content:    "file listing output",
			ToolCallID: "tc1",
			Attachments: []Attachment{
				{MimeType: "image/png", BlobKey: "abc123", Size: 42, SourceHint: "/tmp/img.png"},
			},
		},
		&CompactionEvent{Content: "summary of prior context"},
		&MessageEvent{Content: "hello from peer", Sender: "swift-fox"},
		&ChildResultEvent{Content: "task done", ChildSessionID: "brave-owl", Verdict: "concluded"},
		&RecoverEvent{Content: "resumed"},
	}

	for _, e := range events {
		t.Run(e.EventType(), func(t *testing.T) {
			data, err := Marshal(e)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}

			// Verify type field is in JSON.
			var raw map[string]any
			if err := json.Unmarshal(data, &raw); err != nil {
				t.Fatalf("json.Unmarshal raw: %v", err)
			}
			if raw["type"] != e.EventType() {
				t.Errorf("type field: got %q, want %q", raw["type"], e.EventType())
			}

			// Round-trip.
			got, err := Unmarshal(data)
			if err != nil {
				t.Fatalf("Unmarshal: %v", err)
			}
			if got.EventType() != e.EventType() {
				t.Errorf("EventType: got %q, want %q", got.EventType(), e.EventType())
			}
		})
	}
}

func TestRoundTrip_AssistantEvent_Fields(t *testing.T) {
	orig := &AssistantEvent{
		Content: "here is the plan",
		ToolCalls: []ToolCall{
			{ID: "call_1", Name: "bash", Arguments: `{"command":"echo hi"}`},
			{ID: "call_2", Name: "view_file", Arguments: `{"path":"/tmp/x"}`},
		},
		Thoughts: "let me think",
		Usage:    &Usage{PromptTokens: 200, CompletionTokens: 100, TotalTokens: 300},
	}

	data, err := Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}

	e, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := e.(*AssistantEvent)
	if !ok {
		t.Fatalf("expected *AssistantEvent, got %T", e)
	}

	if got.Content != orig.Content {
		t.Errorf("Content: %q vs %q", got.Content, orig.Content)
	}
	if len(got.ToolCalls) != 2 {
		t.Fatalf("ToolCalls len: %d, want 2", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Name != "bash" {
		t.Errorf("ToolCalls[0].Name: %q", got.ToolCalls[0].Name)
	}
	if got.Thoughts != "let me think" {
		t.Errorf("Thoughts: %q", got.Thoughts)
	}
	if got.Usage == nil || got.Usage.TotalTokens != 300 {
		t.Errorf("Usage: %+v", got.Usage)
	}
}

func TestRoundTrip_MessageEvent_SenderType(t *testing.T) {
	orig := &MessageEvent{Content: "hi", Sender: "swift-fox", SenderType: SenderTypeAgent}
	data, err := Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}
	e, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := e.(*MessageEvent)
	if !ok {
		t.Fatalf("expected *MessageEvent, got %T", e)
	}
	if got.SenderType != SenderTypeAgent {
		t.Errorf("SenderType: got %q, want %q", got.SenderType, SenderTypeAgent)
	}
	if got.Sender != "swift-fox" {
		t.Errorf("Sender: got %q, want %q", got.Sender, "swift-fox")
	}
}

func TestRoundTrip_ToolResultEvent_WithAttachments(t *testing.T) {
	orig := &ToolResultEvent{
		Content:    "image content",
		ToolCallID: "tc_99",
		Attachments: []Attachment{
			// Data is transient: it must NOT survive marshaling (only the
			// blob reference is persisted).
			{MimeType: "image/webp", BlobKey: "deadbeef", Size: 4, SourceHint: "screenshot.png", Data: []byte("AAAA")},
		},
	}
	data, err := Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "AAAA") {
		t.Errorf("raw attachment bytes leaked into persisted event: %s", data)
	}
	e, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	got := e.(*ToolResultEvent)
	if len(got.Attachments) != 1 {
		t.Fatalf("Attachments len: %d", len(got.Attachments))
	}
	if got.Attachments[0].MimeType != "image/webp" {
		t.Errorf("MimeType: %q", got.Attachments[0].MimeType)
	}
	if got.Attachments[0].BlobKey != "deadbeef" {
		t.Errorf("BlobKey: %q", got.Attachments[0].BlobKey)
	}
	if len(got.Attachments[0].Data) != 0 {
		t.Errorf("Data should not be persisted, got %d bytes", len(got.Attachments[0].Data))
	}
}

// TestRoundTrip_ToolResultEvent_IsError locks the additive is_error field: it
// round-trips when set, and — critically for backward compatibility — a result
// WITHOUT the key (every pre-field row) deserializes to IsError=false and a
// successful result omits the key entirely (omitempty), so success rows stay
// byte-identical to historical ones.
func TestRoundTrip_ToolResultEvent_IsError(t *testing.T) {
	// Error result round-trips true.
	data, err := Marshal(&ToolResultEvent{Content: "boom", ToolCallID: "tc1", IsError: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"is_error":true`) {
		t.Errorf("error result should carry is_error:true; got %s", data)
	}
	e, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	if !e.(*ToolResultEvent).IsError {
		t.Error("IsError did not round-trip")
	}

	// Success result omits the key (omitempty), staying byte-identical to a
	// pre-field row.
	data, err = Marshal(&ToolResultEvent{Content: "ok", ToolCallID: "tc2"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "is_error") {
		t.Errorf("success result should omit is_error; got %s", data)
	}

	// Legacy row without the key deserializes to false (the correct default).
	e, err = Unmarshal([]byte(`{"type":"tool_result","content":"old","tool_call_id":"tc3"}`))
	if err != nil {
		t.Fatal(err)
	}
	if e.(*ToolResultEvent).IsError {
		t.Error("legacy row without is_error should deserialize to false")
	}
}

func TestUnmarshal_UnknownType(t *testing.T) {
	data := []byte(`{"type":"alien","content":"hello"}`)
	_, err := Unmarshal(data)
	if err == nil {
		t.Fatal("expected error for unknown type")
	}
	if !strings.Contains(err.Error(), "alien") {
		t.Errorf("error should mention unknown type: %v", err)
	}
}

func TestUnmarshal_MissingType(t *testing.T) {
	data := []byte(`{"content":"hello"}`)
	_, err := Unmarshal(data)
	if err == nil {
		t.Fatal("expected error for missing type")
	}
}

func TestUnmarshal_MalformedJSON(t *testing.T) {
	_, err := Unmarshal([]byte(`{not json`))
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

func TestColumnFields_NotInJSON(t *testing.T) {
	e := &SystemEvent{
		ColumnFields: ColumnFields{Step: 5, Generation: 2},
		Content:      "test",
	}
	data, err := Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	// Step and Generation should not be in the JSON (they come from DB columns).
	// But since we embed ColumnFields, they will appear. This test documents
	// the current behavior — the DB layer is responsible for ignoring them.
	// If we later want to strip them, add custom MarshalJSON.
}

func TestToText_AllTypes(t *testing.T) {
	tests := []struct {
		event     Event
		wantLabel string   // the section label in the "==== LABEL …" header
		wantIn    []string // substrings that must appear in the output
	}{
		{
			&SystemEvent{Content: "hello"},
			"system", []string{"hello"},
		},
		{
			&SystemEvent{Content: "session started", Marker: MarkerNewSession},
			"system", []string{"session started", "new_session"},
		},
		{
			&UserEvent{Content: "hi"},
			"user", []string{"hi"},
		},
		{
			&AssistantEvent{Content: "reply"},
			"assistant", []string{"reply"},
		},
		{
			&AssistantEvent{Content: "plan", ToolCalls: []ToolCall{
				{ID: "tc1", Name: "bash", Arguments: `{"cmd":"ls"}`},
			}},
			"assistant", []string{"plan", "tool_call", "name=bash", "id=tc1"},
		},
		{
			&ToolResultEvent{Content: "output", ToolCallID: "tc1"},
			"tool_result", []string{"output", "id=tc1"},
		},
		{
			&CompactionEvent{Content: "summary"},
			"compaction", []string{"summary"},
		},
		{
			&MessageEvent{Content: "msg", Sender: "a-b"},
			"message", []string{"msg", "from=a-b"},
		},
		{
			&ChildResultEvent{Content: "done", ChildSessionID: "c-d", Verdict: "concluded"},
			"child_result", []string{"done", "child=c-d", "verdict=concluded"},
		},
		{
			&RecoverEvent{Content: "bye"},
			"recover", []string{"bye"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.event.EventType(), func(t *testing.T) {
			got := tt.event.ToText()
			header := "==== " + tt.wantLabel
			if !strings.HasPrefix(got, header) {
				t.Errorf("missing header %q in:\n%s", header, got)
			}
			// The whole point of the format: no XML/tool-call markup leaks in.
			if strings.Contains(got, "<amplio:") || strings.Contains(got, "</") {
				t.Errorf("output still contains XML markup:\n%s", got)
			}
			for _, sub := range tt.wantIn {
				if !strings.Contains(got, sub) {
					t.Errorf("missing substring %q in:\n%s", sub, got)
				}
			}
		})
	}
}

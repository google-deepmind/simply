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

package subprocess

import (
	"encoding/json"

	"amplio/internal/llm"
)

// The wire protocol between amplio and a bridge subprocess. One streaming
// endpoint, POST /generate, takes a wireRequest and returns NDJSON: zero or more
// {"type":"delta"} lines, then exactly one {"type":"final"} (or {"type":"error"})
// line. Both Call and Stream consume this single stream — Call accumulates and
// returns the final; Stream yields the deltas and exposes the final via
// Response(). See bridges/README.md for the bridge-author contract.

type wireRequest struct {
	Model        string        `json:"model"`
	MaxTokens    int           `json:"max_tokens"`
	Temperature  *float64      `json:"temperature,omitempty"`
	SystemPrompt string        `json:"system_prompt,omitempty"`
	Messages     []wireMessage `json:"messages"`
	Tools        []wireTool    `json:"tools,omitempty"`
}

type wireMessage struct {
	Role          string           `json:"role"`
	Content       string           `json:"content,omitempty"`
	ToolCalls     []wireToolCall   `json:"tool_calls,omitempty"`
	ToolCallID    string           `json:"tool_call_id,omitempty"`
	IsError       bool             `json:"is_error,omitempty"`
	Attachments   []wireAttachment `json:"attachments,omitempty"`
	ProviderExtra map[string]any   `json:"provider_extra,omitempty"`
}

type wireToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireAttachment struct {
	MimeType   string `json:"mime_type"`
	Base64Data string `json:"base64_data"`
}

type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
}

type wireUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

type wireResponse struct {
	Content       string         `json:"content"`
	Thoughts      string         `json:"thoughts,omitempty"`
	ToolCalls     []wireToolCall `json:"tool_calls,omitempty"`
	Usage         wireUsage      `json:"usage"`
	StopReason    string         `json:"stop_reason,omitempty"`
	ProviderExtra map[string]any `json:"provider_extra,omitempty"`
}

// wireLine is one NDJSON line of the /generate response.
type wireLine struct {
	Type          string        `json:"type"` // "delta" | "final" | "error"
	Text          string        `json:"text,omitempty"`
	Thoughts      string        `json:"thoughts,omitempty"`
	ToolCallStart *wireTCStart  `json:"tool_call_start,omitempty"`
	ToolCallDelta *wireTCDelta  `json:"tool_call_delta,omitempty"`
	Response      *wireResponse `json:"response,omitempty"`
	Error         string        `json:"error,omitempty"`
}

type wireTCStart struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type wireTCDelta struct {
	ID             string `json:"id"`
	ArgumentsDelta string `json:"arguments_delta"`
}

// --- conversions ---

func (p *provider) toWire(req llm.Request) wireRequest {
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = p.maxTokens
	}
	w := wireRequest{
		Model:        p.model,
		MaxTokens:    maxTokens,
		Temperature:  req.Temperature,
		SystemPrompt: req.SystemPrompt,
	}
	for _, m := range req.Messages {
		wm := wireMessage{
			Role:          string(m.Role),
			Content:       m.Content,
			ToolCallID:    m.ToolCallID,
			IsError:       m.IsError,
			ProviderExtra: m.ProviderExtra,
		}
		for _, tc := range m.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, wireToolCall{ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments})
		}
		for _, att := range m.Attachments {
			wm.Attachments = append(wm.Attachments, wireAttachment{MimeType: att.MimeType, Base64Data: att.Base64Data})
		}
		w.Messages = append(w.Messages, wm)
	}
	for _, t := range req.Tools {
		w.Tools = append(w.Tools, wireTool{Name: t.Name, Description: t.Description, Schema: t.Schema})
	}
	return w
}

func (r *wireResponse) toLLM() *llm.Response {
	out := &llm.Response{
		Content:    r.Content,
		Thoughts:   r.Thoughts,
		StopReason: r.StopReason,
		Usage: llm.Usage{
			PromptTokens:     r.Usage.PromptTokens,
			CompletionTokens: r.Usage.CompletionTokens,
			TotalTokens:      r.Usage.TotalTokens,
			CacheReadTokens:  r.Usage.CacheReadTokens,
			CacheWriteTokens: r.Usage.CacheWriteTokens,
		},
		ProviderExtra: r.ProviderExtra,
	}
	if out.Usage.TotalTokens == 0 && (out.Usage.PromptTokens > 0 || out.Usage.CompletionTokens > 0) {
		out.Usage.TotalTokens = out.Usage.PromptTokens + out.Usage.CompletionTokens
	}
	for _, tc := range r.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, llm.ToolCall{ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments})
	}
	return out
}

func (l *wireLine) toStreamEvent() llm.StreamEvent {
	e := llm.StreamEvent{DeltaText: l.Text, DeltaThoughts: l.Thoughts}
	if l.ToolCallStart != nil {
		e.ToolCallStart = &llm.ToolCallStart{ID: l.ToolCallStart.ID, Name: l.ToolCallStart.Name}
	}
	if l.ToolCallDelta != nil {
		e.ToolCallDelta = &llm.ToolCallDelta{ID: l.ToolCallDelta.ID, ArgumentsDelta: l.ToolCallDelta.ArgumentsDelta}
	}
	return e
}

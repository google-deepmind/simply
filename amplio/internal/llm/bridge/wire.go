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

package bridge

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
	// SessionID is the harness's conversation id, forwarded so a backend can key
	// cache/routing affinity on it (the Claude provider sends it as
	// X-Vertex-Ai-Session-Id). Dropping it costs prompt-cache hits, which at
	// 100k-token prompts is a large fraction of both latency and spend.
	SessionID string `json:"session_id,omitempty"`
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
//
// Readers must IGNORE unknown types rather than fail: that is what lets the
// protocol grow (a "ping" keepalive, say) without a flag day between two
// separately built binaries.
type wireLine struct {
	Type          string        `json:"type"`           // "delta" | "final" | "error" | "ping"
	Code          string        `json:"code,omitempty"` // error class; see bridge.Error
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
		SessionID:    req.SessionID,
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

// --- conversions, serving side ---
//
// The inverses of the four above. They live next to their counterparts on
// purpose: a field added to one direction and forgotten in the other is exactly
// the bug this protocol exists to avoid, and the pair is far easier to eyeball
// than a symmetric struct in another file.

func (w *wireRequest) toLLM() llm.Request {
	req := llm.Request{
		SystemPrompt: w.SystemPrompt,
		MaxTokens:    w.MaxTokens,
		Temperature:  w.Temperature,
		SessionID:    w.SessionID,
	}
	for _, m := range w.Messages {
		msg := llm.Message{
			Role:          llm.Role(m.Role),
			Content:       m.Content,
			ToolCallID:    m.ToolCallID,
			IsError:       m.IsError,
			ProviderExtra: m.ProviderExtra,
		}
		for _, tc := range m.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, llm.ToolCall{ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments})
		}
		for _, att := range m.Attachments {
			msg.Attachments = append(msg.Attachments, llm.Attachment{MimeType: att.MimeType, Base64Data: att.Base64Data})
		}
		req.Messages = append(req.Messages, msg)
	}
	for _, t := range w.Tools {
		req.Tools = append(req.Tools, llm.ToolDef{Name: t.Name, Description: t.Description, Schema: t.Schema})
	}
	return req
}

func responseToWire(r *llm.Response) *wireResponse {
	if r == nil {
		return &wireResponse{}
	}
	out := &wireResponse{
		Content:    r.Content,
		Thoughts:   r.Thoughts,
		StopReason: r.StopReason,
		Usage: wireUsage{
			PromptTokens:     r.Usage.PromptTokens,
			CompletionTokens: r.Usage.CompletionTokens,
			TotalTokens:      r.Usage.TotalTokens,
			CacheReadTokens:  r.Usage.CacheReadTokens,
			CacheWriteTokens: r.Usage.CacheWriteTokens,
		},
		ProviderExtra: r.ProviderExtra,
	}
	for _, tc := range r.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, wireToolCall{ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments})
	}
	return out
}

func eventToWire(e llm.StreamEvent) wireLine {
	l := wireLine{Type: "delta", Text: e.DeltaText, Thoughts: e.DeltaThoughts}
	if e.ToolCallStart != nil {
		l.ToolCallStart = &wireTCStart{ID: e.ToolCallStart.ID, Name: e.ToolCallStart.Name}
	}
	if e.ToolCallDelta != nil {
		l.ToolCallDelta = &wireTCDelta{ID: e.ToolCallDelta.ID, ArgumentsDelta: e.ToolCallDelta.ArgumentsDelta}
	}
	return l
}

// --- embeddings ---
//
// Plain JSON in, plain JSON out: there is nothing to stream, and an embedding
// call either produces every vector or none. A container without model
// credentials needs this as much as it needs generation — without it, recall
// (skills and lessons) is simply off.

type wireEmbedRequest struct {
	// Model is advisory: a bridge serves the embedder it was configured with, and
	// says so in the response. Sending it lets a caller notice a mismatch rather
	// than silently mixing two embedding spaces in one index.
	Model string   `json:"model,omitempty"`
	Texts []string `json:"texts"`
}

type wireEmbedResponse struct {
	Model   string      `json:"model,omitempty"`
	Vectors [][]float32 `json:"vectors"`
	Error   string      `json:"error,omitempty"`
}

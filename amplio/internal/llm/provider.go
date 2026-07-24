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
	"encoding/json"
)

// Provider abstracts an LLM chat-completion endpoint.
type Provider interface {
	// Call sends a request and returns the complete response.
	Call(ctx context.Context, req Request) (*Response, error)

	// Stream sends a request and returns a stream of incremental events.
	// Streamed chunks are for live UI display only — never persisted.
	// Use stream.Response() after iteration to get the final accumulated message.
	Stream(ctx context.Context, req Request) (Stream, error)

	ModelID() string
	MaxTokens() int
}

// Stream delivers incremental LLM response chunks.
type Stream interface {
	// Next advances to the next chunk. Returns false when done or on error.
	Next() bool
	// Event returns the current chunk.
	Event() StreamEvent
	// Response returns the accumulated final response after iteration completes.
	Response() *Response
	// Err returns any error encountered during streaming.
	Err() error
	// Close releases resources. Must be called even if Next returns false.
	Close()
}

// StreamEvent is one incremental chunk from a streaming response.
type StreamEvent struct {
	DeltaText     string // incremental text content
	DeltaThoughts string // incremental thinking content
	ToolCallStart *ToolCallStart
	ToolCallDelta *ToolCallDelta
}

type ToolCallStart struct {
	ID   string
	Name string
}

type ToolCallDelta struct {
	ID             string
	ArgumentsDelta string // incremental JSON fragment
}

// --- Request / Response types ---

type Request struct {
	SystemPrompt string
	Messages     []Message
	Tools        []ToolDef
	MaxTokens    int
	Temperature  *float64 // nil = provider default
	// SessionID, when set, is used by the provider for cache/routing affinity
	// (Anthropic-on-Vertex sends it as X-Vertex-Ai-Session-Id). See
	// https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/prompt-caching#use_prompt_caching_with_the_global_endpoint
	SessionID string
}

// Message is one turn in the conversation.
type Message struct {
	Role        Role
	Content     string
	ToolCalls   []ToolCall   // only for Role=Assistant
	ToolCallID  string       // only for Role=ToolResult
	IsError     bool         // only for Role=ToolResult
	Attachments []Attachment // optional image attachments (tool results, user messages)
	// ProviderExtra is opaque, provider-namespaced cargo carried back to the
	// provider on replay (e.g. Gemini per-tool-call thought signatures under
	// "gemini.fc_sigs_b64"). Keys MUST be provider-namespaced. Only for
	// Role=Assistant. Persisted on the AssistantEvent.
	ProviderExtra map[string]any
}

// Attachment is an inline image sent to the LLM as part of a message.
type Attachment struct {
	MimeType   string // e.g. "image/png", "image/webp"
	Base64Data string
}

type Role string

const (
	RoleSystem     Role = "system"
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "tool_result"
)

type ToolCall struct {
	ID        string
	Name      string
	Arguments string // JSON string
}

type ToolDef struct {
	Name        string
	Description string
	Schema      json.RawMessage // JSON Schema for the tool's parameters
}

type Response struct {
	Content    string
	ToolCalls  []ToolCall
	Thoughts   string
	Usage      Usage
	StopReason string
	// ProviderExtra is opaque, provider-namespaced cargo that doesn't fit the
	// flat projection above (e.g. Gemini per-tool-call thought signatures). It is
	// persisted on the AssistantEvent and replayed via Message.ProviderExtra.
	ProviderExtra map[string]any
}

type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CacheReadTokens  int
	CacheWriteTokens int
}

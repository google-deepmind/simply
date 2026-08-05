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

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"amplio/internal/llm"
)

// defaultBaseURL is the hosted OpenAI endpoint; every other server is reached
// by overriding it (that override is the whole point of this provider).
const defaultBaseURL = "https://api.openai.com/v1"

// Generous: a CPU-hosted local model can take minutes for one turn, and the
// agent loop's own context governs cancellation anyway.
const requestTimeout = 30 * time.Minute

// sseMaxLine bounds one SSE line. The default bufio.Scanner cap (64 KiB) is not
// enough: reasoning signatures observed from real servers run to hundreds of
// bytes each and arrive many-per-chunk.
const sseMaxLine = 4 << 20

// ClientArgs are the arguments this provider interprets — the `{k=v}` block in
// a spec (see internal/llm/spec.go). They never reach the request body.
// Everything else is passed through verbatim (see buildBody), so a
// server-specific knob needs no code change here.
var ClientArgs = map[string]bool{
	"base_url":         true,
	"api_key_env":      true,
	"profile":          true,
	"max_tokens_field": true,
	"stream_usage":     true,
	"capture_extras":   true,
}

// profile is a named bundle of defaults for one flavour of server. They exist
// because the deviations cluster by implementation, and asking an operator to
// assemble knobs by hand is a worse interface than naming their server.
//
// Two knobs only, both about what we SEND — the parser is uniformly tolerant,
// so nothing here affects reading a response.
type profile struct {
	streamUsage    bool   // send stream_options.include_usage
	maxTokensField string // max_tokens (universal) vs max_completion_tokens (newer OpenAI)
}

var profiles = map[string]profile{
	// Newer OpenAI models reject max_tokens outright.
	"openai":  {streamUsage: true, maxTokensField: "max_completion_tokens"},
	"litellm": {streamUsage: true, maxTokensField: "max_tokens"},
	"vllm":    {streamUsage: true, maxTokensField: "max_tokens"},
	// Measured against ollama 0.x + qwen3.5: it takes either max_tokens field,
	// ignores unknown fields rather than 400ing, and both accepts AND honours
	// stream_options — so asking for usage is required, not optional. Without it
	// a streamed turn reports zero tokens.
	"ollama": {streamUsage: true, maxTokensField: "max_tokens"},
	// generic: send nothing optional. The safe default for an unknown server,
	// some of which reject unknown request fields with a 400.
	"generic": {streamUsage: false, maxTokensField: "max_tokens"},
}

type provider struct {
	model     string
	maxTokens int
	baseURL   string
	apiKey    string
	http      *http.Client
	prof      profile
	// extra holds spec-arg passthrough already expanded from dotted paths, merged
	// into every request body.
	extra map[string]any
	// captureExtras stores non-standard reasoning containers on the response's
	// ProviderExtra. Off by default: v1 never replays them, and the signatures are
	// large enough that persisting them on every turn is pure cost. The seam is
	// here so replay support is a read of data we already have.
	captureExtras bool
}

// New builds a provider for ANY OpenAI-compatible /v1/chat/completions endpoint.
// See docs/llm.md for the spec grammar; args not in clientOnlyArgs are injected
// into the request body via dotted paths (reasoning.effort=high →
// {"reasoning":{"effort":"high"}}), mirroring the vertex-claude convention.
func New(model string, maxTokens int, clientArgs, args url.Values) (llm.Provider, error) {
	base := clientArgs.Get("base_url")
	if base == "" {
		base = os.Getenv("OPENAI_BASE_URL")
	}
	if base == "" {
		base = defaultBaseURL
	}
	base = strings.TrimRight(base, "/")

	// Default the profile from the endpoint: the hosted OpenAI API is the only
	// one we can identify by URL, and everything else is safer treated as
	// unknown. An explicit profile= always wins.
	profName := clientArgs.Get("profile")
	if profName == "" {
		if base == defaultBaseURL {
			profName = "openai"
		} else {
			profName = "generic"
		}
	}
	prof, ok := profiles[profName]
	if !ok {
		return nil, fmt.Errorf("unknown profile %q; known: generic, litellm, ollama, openai, vllm", profName)
	}
	if v := clientArgs.Get("max_tokens_field"); v != "" {
		if v != "max_tokens" && v != "max_completion_tokens" {
			return nil, fmt.Errorf("max_tokens_field=%q; want max_tokens or max_completion_tokens", v)
		}
		prof.maxTokensField = v
	}
	if v := clientArgs.Get("stream_usage"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("stream_usage=%q: %w", v, err)
		}
		prof.streamUsage = b
	}
	capture := false
	if v := clientArgs.Get("capture_extras"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("capture_extras=%q: %w", v, err)
		}
		capture = b
	}

	// The key names an ENV VAR rather than carrying a secret, so several
	// endpoints can coexist in one config file without any of them holding
	// credentials. A missing key is not fatal: local servers ignore auth.
	keyEnv := clientArgs.Get("api_key_env")
	if keyEnv == "" {
		keyEnv = "OPENAI_API_KEY"
	}

	extra, err := expandArgs(args)
	if err != nil {
		return nil, err
	}

	return &provider{
		model:         model,
		maxTokens:     maxTokens,
		baseURL:       base,
		apiKey:        os.Getenv(keyEnv),
		http:          &http.Client{Timeout: requestTimeout},
		prof:          prof,
		extra:         extra,
		captureExtras: capture,
	}, nil
}

func (p *provider) ModelID() string { return p.model }
func (p *provider) MaxTokens() int  { return p.maxTokens }

// expandArgs turns model args into a nested body fragment, so dotted keys nest
// (reasoning.effort → {"reasoning":{"effort":…}}). Values are coerced to the
// most specific JSON type. We deliberately do NOT validate keys: the server is
// the authority on what it accepts, and an allowlist here would make every new
// server knob a code change. Client args never reach here — they are split off
// before construction (llm.ClientArgs).
func expandArgs(args url.Values) (map[string]any, error) {
	out := map[string]any{}
	for k := range args {
		if err := setPath(out, k, coerce(args.Get(k))); err != nil {
			return nil, fmt.Errorf("spec arg %q: %w", k, err)
		}
	}
	return out, nil
}

// setPath writes val at a dotted path, creating intermediate maps.
func setPath(m map[string]any, path string, val any) error {
	segs := strings.Split(path, ".")
	for i, s := range segs {
		if s == "" {
			return fmt.Errorf("empty path segment")
		}
		if i == len(segs)-1 {
			m[s] = val
			return nil
		}
		next, ok := m[s].(map[string]any)
		if !ok {
			next = map[string]any{}
			m[s] = next
		}
		m = next
	}
	return nil
}

// coerce parses a spec arg to the most specific JSON type: int, then bool, then
// float, else string (matching the anthropic provider's rule, so "1" stays
// numeric while "true" becomes a bool).
func coerce(s string) any {
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return i
	}
	if b, err := strconv.ParseBool(s); err == nil {
		return b
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}
	return s
}

// --- request ---------------------------------------------------------------

func (p *provider) buildBody(req llm.Request, stream bool) map[string]any {
	body := map[string]any{
		"model":    p.model,
		"messages": convertMessages(req),
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 {
		maxTok = p.maxTokens
	}
	if maxTok > 0 {
		body[p.prof.maxTokensField] = maxTok
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		body["tools"] = convertTools(req.Tools)
	}
	if stream {
		body["stream"] = true
		if p.prof.streamUsage {
			body["stream_options"] = map[string]any{"include_usage": true}
		}
	}
	// Model args last: an explicit temperature=0 overrides what we derived above,
	// which is the least surprising precedence. (max_tokens is a CLIENT arg — it
	// sets p.maxTokens, which lands in the profile's field above — so it cannot
	// arrive here and produce a second, conflicting cap.)
	for k, v := range p.extra {
		body[k] = v
	}
	return body
}

func convertTools(tools []llm.ToolDef) []any {
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		fn := map[string]any{"name": t.Name}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		if len(t.Schema) > 0 {
			fn["parameters"] = json.RawMessage(t.Schema)
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

// convertMessages projects amplio's conversation into OpenAI chat messages.
//
// Two shape mismatches are worth knowing about:
//   - Images on a TOOL RESULT have nowhere to live: the OpenAI schema allows
//     multi-part content on user messages only, and a `tool` message must carry
//     a plain string. We therefore emit the tool result as text, then a
//     follow-up user message carrying its images, which is the only portable
//     encoding and is what other clients settled on too.
//   - There is no is_error flag on a tool message; the error text in the result
//     content is the whole signal.
func convertMessages(req llm.Request) []any {
	out := make([]any, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		out = append(out, map[string]any{"role": "system", "content": req.SystemPrompt})
	}
	for _, m := range req.Messages {
		switch m.Role {
		case llm.RoleSystem:
			out = append(out, map[string]any{"role": "system", "content": m.Content})
		case llm.RoleUser:
			out = append(out, userMessage(m.Content, m.Attachments))
		case llm.RoleAssistant:
			msg := map[string]any{"role": "assistant"}
			// An assistant turn with tool calls may legitimately have no text, but
			// some servers reject a missing content field, so always send it.
			msg["content"] = m.Content
			if len(m.ToolCalls) > 0 {
				calls := make([]any, 0, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					args := tc.Arguments
					if strings.TrimSpace(args) == "" {
						args = "{}" // a null/empty argument string is invalid JSON to strict servers
					}
					calls = append(calls, map[string]any{
						"id": tc.ID, "type": "function",
						"function": map[string]any{"name": tc.Name, "arguments": args},
					})
				}
				msg["tool_calls"] = calls
			}
			out = append(out, msg)
		case llm.RoleToolResult:
			out = append(out, map[string]any{
				"role": "tool", "tool_call_id": m.ToolCallID, "content": m.Content,
			})
			if len(m.Attachments) > 0 {
				out = append(out, userMessage("", m.Attachments))
			}
		}
	}
	return out
}

// userMessage builds a user turn, using the multi-part form only when there are
// images (a plain string is what every server handles best).
func userMessage(text string, atts []llm.Attachment) map[string]any {
	if len(atts) == 0 {
		return map[string]any{"role": "user", "content": text}
	}
	parts := make([]any, 0, len(atts)+1)
	if text != "" {
		parts = append(parts, map[string]any{"type": "text", "text": text})
	}
	for _, a := range atts {
		parts = append(parts, map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url": "data:" + a.MimeType + ";base64," + a.Base64Data,
			},
		})
	}
	return map[string]any{"role": "user", "content": parts}
}

// post sends one request and returns the raw response, retrying transient
// failures (429 / 5xx / network) with bounded backoff.
func (p *provider) post(ctx context.Context, body map[string]any) (*http.Response, error) {
	blob, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	var lastErr error
	for attempt := range maxRetries {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(retryDelay(attempt)):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(blob))
		if err != nil {
			return nil, err
		}
		req.Header.Set("content-type", "application/json")
		if p.apiKey != "" {
			req.Header.Set("authorization", "Bearer "+p.apiKey)
		}
		resp, err := p.http.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = httpError(resp)
			resp.Body.Close()
			continue
		}
		if resp.StatusCode >= 400 {
			err := httpError(resp)
			resp.Body.Close()
			return nil, err // 4xx: a real error, retrying won't help
		}
		return resp, nil
	}
	return nil, fmt.Errorf("openai request failed after %d attempts: %w", maxRetries, lastErr)
}

const maxRetries = 3

func retryDelay(attempt int) time.Duration {
	return time.Duration(1<<attempt) * time.Second
}

// httpError renders a failed response, preferring the standard error envelope
// ({"error":{"message":…}}) but falling back to a body snippet — third-party
// servers invent their own shapes, and a bare status code is useless to debug.
func httpError(resp *http.Response) error {
	blob, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
	var env struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(blob, &env); err == nil && env.Error.Message != "" {
		return fmt.Errorf("openai %s: %s", resp.Status, env.Error.Message)
	}
	snippet := strings.TrimSpace(string(blob))
	if snippet == "" {
		return fmt.Errorf("openai %s", resp.Status)
	}
	return fmt.Errorf("openai %s: %s", resp.Status, snippet)
}

// --- response --------------------------------------------------------------

// content decodes a `content` field that may be a plain string OR an array of
// typed parts (servers differ, and the same server differs between the
// streaming and non-streaming shapes). Unknown shapes decode to empty rather
// than failing the turn.
type content struct{ text string }

func (c *content) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		c.text = s
		return nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(b, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			sb.WriteString(p.Text)
		}
		c.text = sb.String()
	}
	return nil
}

type apiUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *apiUsage) to() llm.Usage {
	if u == nil {
		return llm.Usage{}
	}
	out := llm.Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.PromptTokensDetails != nil {
		out.CacheReadTokens = u.PromptTokensDetails.CachedTokens
	}
	return out
}

type respToolCall struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// reasoning fields, in the order we prefer them. No server sends more than one
// with content, but several send an empty one alongside a populated other.
type reasoningFields struct {
	ReasoningContent string `json:"reasoning_content"`
	Reasoning        string `json:"reasoning"`
}

func (r reasoningFields) text() string {
	if r.ReasoningContent != "" {
		return r.ReasoningContent
	}
	return r.Reasoning
}

type chatResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content   content        `json:"content"`
			ToolCalls []respToolCall `json:"tool_calls"`
			reasoningFields
			Extra map[string]json.RawMessage `json:"provider_specific_fields"`
		} `json:"message"`
	} `json:"choices"`
	Usage *apiUsage `json:"usage"`
}

func (p *provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	resp, err := p.post(ctx, p.buildBody(req, false))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	blob, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai read response: %w", err)
	}
	var cr chatResponse
	if err := json.Unmarshal(blob, &cr); err != nil {
		return nil, fmt.Errorf("openai decode response: %w", err)
	}
	out := &llm.Response{Usage: cr.Usage.to()}
	// choices CAN be empty: a thinking model that spends its whole budget on
	// reasoning returns zero choices, so indexing [0] would panic. An empty
	// response is a legitimate (if useless) turn — the event loop handles it.
	if len(cr.Choices) == 0 {
		return out, nil
	}
	ch := cr.Choices[0]
	out.Content = ch.Message.Content.text
	out.Thoughts = ch.Message.text()
	out.StopReason = ch.FinishReason
	for i, tc := range ch.Message.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, llm.ToolCall{
			ID:        toolCallID(tc.ID, i),
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}
	if p.captureExtras && len(ch.Message.Extra) > 0 {
		blob, err := json.Marshal(ch.Message.Extra)
		if err == nil {
			out.ProviderExtra = map[string]any{"openai.provider_specific_fields": string(blob)}
		}
	}
	return out, nil
}

// toolCallID falls back to a synthesized id when the server omits one (some
// do). The id only has to be unique within the turn: the event loop matches
// results to calls by it.
func toolCallID(id string, index int) string {
	if id != "" {
		return id
	}
	return "call_" + strconv.Itoa(index)
}

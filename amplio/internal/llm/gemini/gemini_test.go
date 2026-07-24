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

package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"testing"
	"time"

	"amplio/internal/llm"

	"google.golang.org/genai"
)

func TestParseGeminiArgs(t *testing.T) {
	g, err := parseGeminiArgs(url.Values{
		"thinking_budget":  {"4096"},
		"include_thoughts": {"false"},
		"temperature":      {"0.5"},
	})
	if err != nil {
		t.Fatalf("parseGeminiArgs: %v", err)
	}
	if g.thinkingBudget == nil || *g.thinkingBudget != 4096 {
		t.Errorf("thinkingBudget = %v, want 4096", g.thinkingBudget)
	}
	if g.includeThoughts == nil || *g.includeThoughts != false {
		t.Errorf("includeThoughts = %v, want false", g.includeThoughts)
	}
	if g.temperature == nil || *g.temperature != 0.5 {
		t.Errorf("temperature = %v, want 0.5", g.temperature)
	}

	if _, err := parseGeminiArgs(url.Values{"foo": {"bar"}}); err == nil {
		t.Error("expected error for unknown gemini arg")
	}
	if _, err := parseGeminiArgs(url.Values{"thinking_budget": {"notanint"}}); err == nil {
		t.Error("expected error for bad thinking_budget")
	}
	if g, err := parseGeminiArgs(nil); err != nil || g.thinkingBudget != nil {
		t.Errorf("empty args: %+v err=%v", g, err)
	}
}

func TestRespAccAggregatesParts(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			FinishReason: genai.FinishReasonStop,
			Content: &genai.Content{Parts: []*genai.Part{
				{Text: "thinking hard", Thought: true},
				{Text: "hello "},
				{Text: "world"},
				{
					FunctionCall:     &genai.FunctionCall{Name: "ls", Args: map[string]any{"path": "/"}},
					ThoughtSignature: []byte("sigbytes"),
				},
			}},
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 10, CandidatesTokenCount: 5, TotalTokenCount: 15, CachedContentTokenCount: 3,
		},
	}
	var acc respAcc
	acc.add(resp)
	r := acc.response()

	if r.Content != "hello world" {
		t.Errorf("content = %q", r.Content)
	}
	if r.Thoughts != "thinking hard" {
		t.Errorf("thoughts = %q", r.Thoughts)
	}
	if r.StopReason != "STOP" {
		t.Errorf("stop = %q", r.StopReason)
	}
	if len(r.ToolCalls) != 1 || r.ToolCalls[0].Name != "ls" || r.ToolCalls[0].Arguments != `{"path":"/"}` {
		t.Fatalf("tool calls = %+v", r.ToolCalls)
	}
	if !strings.HasPrefix(r.ToolCalls[0].ID, "call_") {
		t.Errorf("expected synthesized id, got %q", r.ToolCalls[0].ID)
	}
	if r.Usage != (llm.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, CacheReadTokens: 3}) {
		t.Errorf("usage = %+v", r.Usage)
	}
	sigs := decodeSigs(r.ProviderExtra)
	if len(sigs) != 1 || sigs[0] != base64.StdEncoding.EncodeToString([]byte("sigbytes")) {
		t.Errorf("sigs = %v", sigs)
	}
}

func TestRespAccNoSigOmitsProviderExtra(t *testing.T) {
	resp := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "x", Name: "f", Args: map[string]any{}}},
		}},
	}}}
	var acc respAcc
	acc.add(resp)
	r := acc.response()
	if r.ProviderExtra != nil {
		t.Errorf("expected nil ProviderExtra with no signatures, got %v", r.ProviderExtra)
	}
	if r.ToolCalls[0].ID != "x" {
		t.Errorf("model-supplied id should be preserved, got %q", r.ToolCalls[0].ID)
	}
}

func TestConvertMessagesRolesAndSignatures(t *testing.T) {
	sig := base64.StdEncoding.EncodeToString([]byte("sigA"))
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "hi"},
		{
			Role:          llm.RoleAssistant,
			Content:       "ok",
			ToolCalls:     []llm.ToolCall{{ID: "c1", Name: "ls", Arguments: `{"path":"/"}`}},
			ProviderExtra: map[string]any{sigKey: []string{sig}},
		},
		{Role: llm.RoleToolResult, ToolCallID: "c1", Content: "file listing"},
	}
	contents, err := convertMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != 4 {
		t.Fatalf("contents len = %d", len(contents))
	}
	wantRoles := []string{"user", "user", "model", "user"}
	for i, want := range wantRoles {
		if contents[i].Role != want {
			t.Errorf("content[%d].Role = %q, want %q", i, contents[i].Role, want)
		}
	}

	// Assistant: text part + function call with the replayed thought signature.
	model := contents[2]
	var fc *genai.Part
	for _, p := range model.Parts {
		if p.FunctionCall != nil {
			fc = p
		}
	}
	if fc == nil {
		t.Fatal("no function call part on assistant turn")
	}
	if fc.FunctionCall.Name != "ls" || fc.FunctionCall.ID != "c1" {
		t.Errorf("function call = %+v", fc.FunctionCall)
	}
	if string(fc.ThoughtSignature) != "sigA" {
		t.Errorf("thought signature = %q, want sigA", fc.ThoughtSignature)
	}

	// Tool result: function response with the NAME looked up from the call.
	fr := contents[3].Parts[0].FunctionResponse
	if fr == nil || fr.Name != "ls" || fr.ID != "c1" {
		t.Fatalf("function response = %+v", fr)
	}
	if fr.Response["output"] != "file listing" {
		t.Errorf("response payload = %v", fr.Response)
	}
}

func TestConvertMessagesCoalescesToolResults(t *testing.T) {
	// An assistant turn with TWO tool calls must be answered by a single user
	// turn carrying both FunctionResponse parts (Gemini validates the response
	// part count against "the function call turn"). The harness records each
	// result as its own message; convertMessages must coalesce the run.
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: "go"},
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				{ID: "c1", Name: "bash", Arguments: `{}`},
				{ID: "c2", Name: "await_event", Arguments: `{}`},
			},
		},
		{Role: llm.RoleToolResult, ToolCallID: "c1", Content: "out1"},
		{Role: llm.RoleToolResult, ToolCallID: "c2", Content: "out2", IsError: true},
		// An interleaved user message (e.g. an environment notice) must start a
		// fresh content rather than being folded into the tool-result turn.
		{Role: llm.RoleUser, Content: "notice"},
	}
	contents, err := convertMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	// user, model, user(2 responses), user(notice).
	if len(contents) != 4 {
		t.Fatalf("contents len = %d, want 4: %+v", len(contents), contents)
	}
	wantRoles := []string{"user", "model", "user", "user"}
	for i, want := range wantRoles {
		if contents[i].Role != want {
			t.Errorf("content[%d].Role = %q, want %q", i, contents[i].Role, want)
		}
	}

	// The model turn made 2 calls; the following user turn must carry exactly 2
	// FunctionResponse parts (the invariant that the 400 error enforced).
	callParts := 0
	for _, p := range contents[1].Parts {
		if p.FunctionCall != nil {
			callParts++
		}
	}
	respParts := 0
	for _, p := range contents[2].Parts {
		if p.FunctionResponse != nil {
			respParts++
		}
	}
	if callParts != 2 || respParts != 2 {
		t.Fatalf("call parts = %d, response parts = %d, want 2 and 2", callParts, respParts)
	}

	// Names looked up from the calls; second carries the error payload.
	fr0 := contents[2].Parts[0].FunctionResponse
	fr1 := contents[2].Parts[1].FunctionResponse
	if fr0.ID != "c1" || fr0.Name != "bash" || fr0.Response["output"] != "out1" {
		t.Errorf("response[0] = %+v", fr0)
	}
	if fr1.ID != "c2" || fr1.Name != "await_event" || fr1.Response["error"] != "out2" {
		t.Errorf("response[1] = %+v", fr1)
	}

	// The interleaved notice survives as its own user turn.
	if contents[3].Parts[0].Text != "notice" {
		t.Errorf("notice content = %+v", contents[3].Parts)
	}
}

func TestConvertToolsUsesJSONSchema(t *testing.T) {
	tools := convertTools([]llm.ToolDef{{
		Name:        "t",
		Description: "d",
		Schema:      json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
	}})
	if len(tools) != 1 || len(tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("tools = %+v", tools)
	}
	fd := tools[0].FunctionDeclarations[0]
	if fd.Name != "t" || fd.Description != "d" {
		t.Errorf("decl = %+v", fd)
	}
	if fd.ParametersJsonSchema == nil {
		t.Error("expected ParametersJsonSchema to be set")
	}
	if fd.Parameters != nil {
		t.Error("should pass raw JSON schema, not the typed Parameters")
	}
}

func TestDecodeSigsTolueratesJSONRoundTrip(t *testing.T) {
	if got := decodeSigs(map[string]any{sigKey: []string{"a", "b"}}); len(got) != 2 || got[0] != "a" {
		t.Errorf("in-memory []string: %v", got)
	}
	// After a DB JSON round-trip the slice decodes as []any.
	if got := decodeSigs(map[string]any{sigKey: []any{"a", "b"}}); len(got) != 2 || got[1] != "b" {
		t.Errorf("json []any: %v", got)
	}
	if decodeSigs(nil) != nil || decodeSigs(map[string]any{}) != nil {
		t.Error("expected nil for missing key")
	}
}

func TestArgsToMap(t *testing.T) {
	if m, err := argsToMap(""); err != nil || len(m) != 0 {
		t.Errorf("empty: %v %v", m, err)
	}
	if m, err := argsToMap(`{"a":1}`); err != nil || m["a"] != float64(1) {
		t.Errorf("valid: %v %v", m, err)
	}
	if _, err := argsToMap(`not json`); err == nil {
		t.Error("expected error for invalid json")
	}
}

// TestBuildMapsSystemPromptToSystemInstruction locks the seam the eventloop
// relies on: req.SystemPrompt (the hoisted leading-system cluster) must land in
// Gemini's native SystemInstruction, NOT be demoted to a user turn. (Messages
// still carry the task + any post-task system events; a mid-conversation system
// message is separately demoted to user — see TestConvertMessagesRolesAndSignatures.)
func TestBuildMapsSystemPromptToSystemInstruction(t *testing.T) {
	p := &provider{maxTokens: 1024}
	_, config, err := p.build(llm.Request{
		SystemPrompt: "you are an agent",
		Messages:     []llm.Message{{Role: llm.RoleUser, Content: "the task"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if config.SystemInstruction == nil || len(config.SystemInstruction.Parts) == 0 {
		t.Fatal("SystemInstruction not set from req.SystemPrompt")
	}
	if got := config.SystemInstruction.Parts[0].Text; got != "you are an agent" {
		t.Errorf("SystemInstruction = %q, want %q", got, "you are an agent")
	}
}

// An empty SystemPrompt leaves SystemInstruction unset (no empty system block).
func TestBuildOmitsEmptySystemInstruction(t *testing.T) {
	p := &provider{maxTokens: 1024}
	_, config, err := p.build(llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if config.SystemInstruction != nil {
		t.Errorf("SystemInstruction should be nil for empty SystemPrompt, got %+v", config.SystemInstruction)
	}
}

func TestIsTransient(t *testing.T) {
	if isTransient(nil) {
		t.Error("nil error is never transient")
	}
	// Structured API errors: 429 + 5xx retryable, 4xx (other) not.
	transientCodes := []int{429, 500, 502, 503, 504}
	for _, code := range transientCodes {
		if !isTransient(genai.APIError{Code: code}) {
			t.Errorf("APIError code %d should be transient", code)
		}
	}
	for _, code := range []int{400, 401, 403, 404} {
		if isTransient(genai.APIError{Code: code}) {
			t.Errorf("APIError code %d should NOT be transient", code)
		}
	}
	// A structured 400 must win even if the message contains a transient-looking
	// word (the authoritative code should short-circuit the string fallback).
	if isTransient(genai.APIError{Code: 400, Message: "UNAVAILABLE-sounding but really a bad request"}) {
		t.Error("structured 400 must not be overridden by string match")
	}
	// The real-world PREFILL_QUEUE_OVERLOADED 429 (wrapped) is transient.
	prefill := fmt.Errorf("gemini call: %w", genai.APIError{
		Code:    429,
		Status:  "RESOURCE_EXHAUSTED",
		Message: "Resource exhausted ... PREFILL_QUEUE_OVERLOADED",
	})
	if !isTransient(prefill) {
		t.Error("wrapped 429 RESOURCE_EXHAUSTED should be transient")
	}
	// Bare connection errors (no HTTP status) fall back to string signatures.
	for _, s := range []string{"connection reset by peer", "unexpected EOF", "server is Overloaded"} {
		if !isTransient(errors.New(s)) {
			t.Errorf("connection-style error %q should be transient", s)
		}
	}
	if isTransient(errors.New("invalid argument: bad tool schema")) {
		t.Error("a plain non-transient error should fail fast")
	}
}

func TestRetryDelayBoundedJitter(t *testing.T) {
	orig := baseRetryDelay
	baseRetryDelay = time.Second
	t.Cleanup(func() { baseRetryDelay = orig })

	// attempt N targets base*2^(N-1); jitter keeps the result within [75%, 100%].
	for attempt := 1; attempt <= 4; attempt++ {
		target := time.Second << (attempt - 1)
		lo, hi := target*3/4, target
		for range 200 {
			d := retryDelay(attempt)
			if d < lo || d > hi {
				t.Fatalf("attempt %d: delay %v outside [%v, %v]", attempt, d, lo, hi)
			}
		}
	}
	// Beyond the cap, still bounded by maxRetryDelay.
	for range 200 {
		if d := retryDelay(20); d > maxRetryDelay || d < maxRetryDelay*3/4 {
			t.Fatalf("capped delay %v outside [%v, %v]", d, maxRetryDelay*3/4, maxRetryDelay)
		}
	}
	// A zero base yields zero delay (keeps tests fast).
	baseRetryDelay = 0
	if d := retryDelay(3); d != 0 {
		t.Errorf("zero base should give zero delay, got %v", d)
	}
}

func TestWithRetrySucceedsAfterTransient(t *testing.T) {
	orig := baseRetryDelay
	baseRetryDelay = 0
	t.Cleanup(func() { baseRetryDelay = orig })

	var n int
	out, err := withRetry(context.Background(), func() (string, error) {
		n++
		if n < 3 {
			return "", genai.APIError{Code: 503, Status: "UNAVAILABLE"}
		}
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
	if out != "ok" || n != 3 {
		t.Errorf("out=%q attempts=%d, want ok/3", out, n)
	}
}

func TestWithRetryExhausts(t *testing.T) {
	orig := baseRetryDelay
	baseRetryDelay = 0
	t.Cleanup(func() { baseRetryDelay = orig })

	var n int
	_, err := withRetry(context.Background(), func() (string, error) {
		n++
		return "", genai.APIError{Code: 429}
	})
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if n != maxAttempts {
		t.Errorf("attempts=%d, want %d", n, maxAttempts)
	}
}

func TestWithRetryNoRetryOnRealError(t *testing.T) {
	var n int
	_, err := withRetry(context.Background(), func() (string, error) {
		n++
		return "", genai.APIError{Code: 400, Message: "bad request"}
	})
	if err == nil {
		t.Fatal("expected the real error to propagate")
	}
	if n != 1 {
		t.Errorf("attempts=%d, want 1 (real errors fail fast)", n)
	}
}

func TestWithRetryHonorsContextCancel(t *testing.T) {
	orig := baseRetryDelay
	baseRetryDelay = time.Hour // force the backoff wait so cancel is what unblocks
	t.Cleanup(func() { baseRetryDelay = orig })

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled
	var n int
	_, err := withRetry(ctx, func() (string, error) {
		n++
		return "", genai.APIError{Code: 503}
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err=%v, want context.Canceled", err)
	}
	if n != 1 {
		t.Errorf("attempts=%d, want 1 (cancel before the second try)", n)
	}
}

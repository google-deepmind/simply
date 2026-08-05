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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"amplio/internal/llm"
)

// serveSSE stands up a fake endpoint replaying a canned SSE body, and captures
// the request it received so tests can assert on what we SEND as well as how we
// parse. This is the whole test strategy: the corpus under testdata/ holds real
// captures from real servers, so "compatible" is evidence rather than belief.
func serveSSE(t *testing.T, body string) (*httptest.Server, *map[string]any) {
	t.Helper()
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("content-type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

func serveJSON(t *testing.T, status int, body string) (*httptest.Server, *map[string]any) {
	t.Helper()
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("content-type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

// newProvider builds a provider the way createProvider does: one flat arg set,
// divided by llm.SplitArgs using this provider's declaration. Going through the
// real splitter (rather than hand-sorting the two maps here) means these tests
// also pin which side each arg lands on.
func newProvider(t *testing.T, base string, extra ...string) llm.Provider {
	t.Helper()
	args := url.Values{"base_url": {base}}
	for i := 0; i+1 < len(extra); i += 2 {
		args.Set(extra[i], extra[i+1])
	}
	p, err := newFromFlatArgs("test-model", 4096, args)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func drain(t *testing.T, s llm.Stream) *llm.Response {
	t.Helper()
	defer s.Close()
	for s.Next() {
	}
	if err := s.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	return s.Response()
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(b)
}

// --- egress audit ------------------------------------------------------------

// conformanceRequests are the ONLY requests the live suite sends to a
// third-party endpoint. They live here (untagged) rather than in the
// integration file so the audit test below can inspect them without the
// integration tag, and so there is exactly one definition of "what we egress".
//
// Every payload is deliberately synthetic. In particular NONE of them carries
// amplio's own system prompt: on the internal build that text describes the
// host environment and must not leave the machine.
func conformanceRequests() map[string]llm.Request {
	weather := llm.ToolDef{
		Name:        "get_weather",
		Description: "Current weather for a city.",
		Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
	return map[string]llm.Request{
		"call": {
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Reply with the single word: ok"}},
		},
		"stream_parallel_tools": {
			Messages: []llm.Message{{Role: llm.RoleUser,
				Content: "Call get_weather for BOTH Tokyo and Paris. Use the tool twice."}},
			Tools: []llm.ToolDef{weather},
		},
		"tool_result_round_trip": {
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "What is the weather in Tokyo?"},
				{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
					{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Tokyo"}`}}},
				{Role: llm.RoleToolResult, ToolCallID: "call_1", Content: "18C and raining"},
			},
			Tools: []llm.ToolDef{weather},
		},
	}
}

// forbiddenInEgress are markers of host/corp context that must never reach a
// third-party endpoint. The list is deliberately structural (paths, hostnames,
// tool names) rather than a semantic scrub — it exists to catch the realistic
// regression: someone wiring the live suite to a real agent prompt or a real
// workspace, both of which carry this vocabulary on the internal build.
var forbiddenInEgress = []string{
	"google3", "blaze", "citc", "/cns/", "borg", "xmanager",
	"corp.google", "googleplex", "googlers.com", "@google.com",
	"go/", "cl/", "//depot", "critique", "sponge",
}

// TestEgressAudit_ConformancePayloads prints, and vets, exactly what the live
// suite would send to a third party. Run it before pointing the suite at a new
// vendor: `go test ./internal/llm/openai/ -run TestEgressAudit -v`.
func TestEgressAudit_ConformancePayloads(t *testing.T) {
	for name, req := range conformanceRequests() {
		srv, got := serveJSON(t, 200, `{"choices":[]}`)
		if _, err := newProvider(t, srv.URL).Call(context.Background(), req); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		blob, err := json.Marshal(*got)
		if err != nil {
			t.Fatal(err)
		}
		body := string(blob)
		t.Logf("egress[%s]: %s", name, body)

		// No system prompt: amplio's own carries host-environment text.
		for _, m := range (*got)["messages"].([]any) {
			if role := m.(map[string]any)["role"]; role == "system" || role == "developer" {
				t.Errorf("%s: a %v message is being sent; the conformance payloads must carry no system prompt", name, role)
			}
		}
		lower := strings.ToLower(body)
		for _, marker := range forbiddenInEgress {
			if strings.Contains(lower, marker) {
				t.Errorf("%s: payload contains host/corp marker %q", name, marker)
			}
		}
	}
}

// --- recorded dialects ------------------------------------------------------

// Real capture: Claude via a LiteLLM proxy. The OpenAI-classic dialect — id and
// name only in the first delta per index, arguments split at arbitrary byte
// boundaries (here mid-string: `{"city": "P` + `aris"}`), empty-string argument
// deltas interleaved, finish_reason alone in its own frame, usage in a trailing
// frame whose delta is empty.
func TestStream_LiteLLMClaude_SplitToolArguments(t *testing.T) {
	srv, _ := serveSSE(t, fixture(t, "litellm-claude-tools.sse"))
	resp := drain(t, mustStream(t, newProvider(t, srv.URL)))

	if len(resp.ToolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2: %+v", len(resp.ToolCalls), resp.ToolCalls)
	}
	want := []string{`{"city": "Tokyo"}`, `{"city": "Paris"}`}
	for i, tc := range resp.ToolCalls {
		if tc.Name != "get_weather" {
			t.Errorf("call %d name = %q, want get_weather", i, tc.Name)
		}
		if tc.Arguments != want[i] {
			t.Errorf("call %d args = %q, want %q", i, tc.Arguments, want[i])
		}
		if !strings.HasPrefix(tc.ID, "toolu_") {
			t.Errorf("call %d id = %q, want the server's id carried through", i, tc.ID)
		}
	}
	if resp.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want tool_calls", resp.StopReason)
	}
	if resp.Content == "" {
		t.Errorf("content is empty; the capture has text before the calls")
	}
	if resp.Usage.TotalTokens != 498 {
		t.Errorf("usage total = %d, want 498 (from the trailing usage-only frame)", resp.Usage.TotalTokens)
	}
}

// Real capture: Gemini via the same proxy — the "whole call in one delta"
// dialect, with non-standard ids and no argument splitting. Same accumulator,
// no special-casing.
func TestStream_LiteLLMGemini_WholeToolCallPerDelta(t *testing.T) {
	srv, _ := serveSSE(t, fixture(t, "litellm-gemini-tools.sse"))
	resp := drain(t, mustStream(t, newProvider(t, srv.URL)))

	if len(resp.ToolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2: %+v", len(resp.ToolCalls), resp.ToolCalls)
	}
	got := []string{resp.ToolCalls[0].Arguments, resp.ToolCalls[1].Arguments}
	want := []string{`{"city": "Tokyo"}`, `{"city": "Paris"}`}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("call %d args = %q, want %q", i, got[i], want[i])
		}
	}
	if resp.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want tool_calls", resp.StopReason)
	}
	if resp.Usage.TotalTokens != 170 {
		t.Errorf("usage total = %d, want 170", resp.Usage.TotalTokens)
	}
}

// Real capture from the REFERENCE implementation (api.openai.com, gpt-5.4-nano).
// This is the dialect every other server claims compatibility with, and it is
// the most aggressive splitter of the three: arguments arrive in four fragments
// per call, broken MID-KEY (`{"ci` + `ty": ` + `"Tokyo` + `"}`), each call opens
// with an empty-string argument delta, finish_reason arrives alone, and usage
// rides a trailing frame whose `choices` array is EMPTY. Any per-frame JSON
// parsing dies here.
func TestStream_OpenAIReference_MidKeyArgumentSplits(t *testing.T) {
	srv, _ := serveSSE(t, fixture(t, "openai-nano-tools.sse"))
	resp := drain(t, mustStream(t, newProvider(t, srv.URL, "profile", "openai")))

	if len(resp.ToolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2: %+v", len(resp.ToolCalls), resp.ToolCalls)
	}
	want := []string{`{"city": "Tokyo"}`, `{"city": "Paris"}`}
	for i, tc := range resp.ToolCalls {
		if tc.Arguments != want[i] {
			t.Errorf("call %d args = %q, want %q", i, tc.Arguments, want[i])
		}
		if tc.Name != "get_weather" || !strings.HasPrefix(tc.ID, "call_") {
			t.Errorf("call %d = %+v, want the server's id and name carried through", i, tc)
		}
	}
	if resp.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want tool_calls", resp.StopReason)
	}
	// Usage arrives in the choices-less trailing frame.
	if resp.Usage.TotalTokens != 182 {
		t.Errorf("usage total = %d, want 182", resp.Usage.TotalTokens)
	}
}

// Real capture from a local ollama server (qwen3.5). A third shape again: the
// whole tool call arrives in ONE delta, `role: "assistant"` repeats on EVERY
// frame rather than just the first, reasoning comes back under `reasoning`
// (not `reasoning_content`), and no usage appears at all unless the request
// asked for it — which is why profile=ollama sends stream_options.
func TestStream_Ollama_WholeCallsAndRepeatedRole(t *testing.T) {
	srv, _ := serveSSE(t, fixture(t, "ollama-qwen35-tools.sse"))
	resp := drain(t, mustStream(t, newProvider(t, srv.URL, "profile", "ollama")))

	if len(resp.ToolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2: %+v", len(resp.ToolCalls), resp.ToolCalls)
	}
	want := []string{`{"city":"Tokyo"}`, `{"city":"Paris"}`}
	for i, tc := range resp.ToolCalls {
		if tc.Arguments != want[i] {
			t.Errorf("call %d args = %q, want %q", i, tc.Arguments, want[i])
		}
		if tc.Name != "get_weather" || tc.ID == "" {
			t.Errorf("call %d = %+v, want the server id and name", i, tc)
		}
	}
	if resp.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want tool_calls", resp.StopReason)
	}
	// The reasoning stream is the bulk of this capture; it must land in
	// Thoughts rather than being mistaken for content.
	if len(resp.Thoughts) == 0 {
		t.Error("no thoughts captured; the `reasoning` deltas were dropped")
	}
	if resp.Content != "" {
		t.Errorf("content = %q, want empty (this turn is reasoning + tool calls)", resp.Content)
	}
}

// --- hand-built edge cases --------------------------------------------------

func TestStream_Tolerances(t *testing.T) {
	tests := []struct {
		name  string
		sse   string
		check func(*testing.T, *llm.Response)
	}{{
		// Keep-alive comments, an unknown SSE field, a blank-line separator and a
		// CRLF line ending all appear in the wild; none is data.
		name: "noise lines are skipped",
		// ": OPENROUTER PROCESSING" is a real keep-alive some gateways emit to
		// hold the connection open; a parser that feeds comment lines to JSON
		// dies on it.
		sse: ": ping\n: OPENROUTER PROCESSING\n\nevent: message\r\n" +
			`data: {"choices":[{"delta":{"content":"hi"}}]}` + "\n\n" +
			": keep-alive\ndata: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if r.Content != "hi" {
				t.Errorf("content = %q, want hi", r.Content)
			}
		},
	}, {
		// No [DONE] sentinel: EOF ends the stream, and what we accumulated stands.
		name: "missing DONE sentinel",
		sse:  `data: {"choices":[{"delta":{"content":"done-less"},"finish_reason":"stop"}]}` + "\n",
		check: func(t *testing.T, r *llm.Response) {
			if r.Content != "done-less" || r.StopReason != "stop" {
				t.Errorf("got %q/%q, want done-less/stop", r.Content, r.StopReason)
			}
		},
	}, {
		// A tool call with neither id nor index: index defaults to 0 and the id is
		// synthesized, because the event loop matches results to calls by id.
		name: "tool call without id or index",
		sse: `data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"f","arguments":"{}"}}]}}]}` + "\n" +
			"data: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if len(r.ToolCalls) != 1 {
				t.Fatalf("tool calls = %d, want 1", len(r.ToolCalls))
			}
			if r.ToolCalls[0].ID != "call_0" || r.ToolCalls[0].Name != "f" {
				t.Errorf("got %+v, want synthesized id call_0 and name f", r.ToolCalls[0])
			}
		},
	}, {
		// Interleaved parallel calls: frames for index 1 arrive between frames for
		// index 0, so arrival order must not drive assembly.
		name: "interleaved parallel calls",
		sse: `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"f","arguments":"{\"x\":"}}]}}]}` + "\n" +
			`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"g","arguments":"{\"y\":"}}]}}]}` + "\n" +
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}` + "\n" +
			`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}` + "\n" +
			"data: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if len(r.ToolCalls) != 2 {
				t.Fatalf("tool calls = %d, want 2", len(r.ToolCalls))
			}
			if r.ToolCalls[0].ID != "a" || r.ToolCalls[0].Arguments != `{"x":1}` {
				t.Errorf("call 0 = %+v, want a/{\"x\":1}", r.ToolCalls[0])
			}
			if r.ToolCalls[1].ID != "b" || r.ToolCalls[1].Arguments != `{"y":2}` {
				t.Errorf("call 1 = %+v, want b/{\"y\":2}", r.ToolCalls[1])
			}
		},
	}, {
		// Reasoning under either field name lands in Thoughts.
		name: "reasoning_content and reasoning",
		sse: `data: {"choices":[{"delta":{"reasoning_content":"think "}}]}` + "\n" +
			`data: {"choices":[{"delta":{"reasoning":"more"}}]}` + "\n" +
			`data: {"choices":[{"delta":{"content":"answer"}}]}` + "\n" +
			"data: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if r.Thoughts != "think more" {
				t.Errorf("thoughts = %q, want %q", r.Thoughts, "think more")
			}
			if r.Content != "answer" {
				t.Errorf("content = %q, want answer", r.Content)
			}
		},
	}, {
		// A frame with no choices at all (the usage-only tail) and one with a
		// null delta must not panic or lose the usage.
		name: "choiceless and null-delta frames",
		sse: `data: {"choices":[]}` + "\n" +
			`data: {"choices":[{"delta":null}]}` + "\n" +
			`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}` + "\n" +
			"data: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if r.Usage.TotalTokens != 7 {
				t.Errorf("usage total = %d, want 7", r.Usage.TotalTokens)
			}
		},
	}, {
		// A malformed frame is skipped rather than killing a long turn.
		name: "malformed frame is skipped",
		sse: `data: {"choices":[{"delta":{"content":"a"}}]}` + "\n" +
			"data: {not json\n" +
			`data: {"choices":[{"delta":{"content":"b"}}]}` + "\n" +
			"data: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if r.Content != "ab" {
				t.Errorf("content = %q, want ab", r.Content)
			}
		},
	}, {
		// Reported in the wild: a server that finishes a tool-calling turn with
		// finish_reason "stop" instead of "tool_calls". Nothing may depend on the
		// label — the loop decides a turn had tools by the CALLS being present,
		// so this must still surface as a tool call.
		name: "tool calls with a stop finish_reason",
		sse: `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"stop"}]}` + "\n" +
			"data: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if len(r.ToolCalls) != 1 || r.ToolCalls[0].Name != "f" {
				t.Fatalf("tool calls = %+v, want one despite the stop label", r.ToolCalls)
			}
			if r.StopReason != "stop" {
				t.Errorf("stop reason = %q, want the server's label preserved verbatim", r.StopReason)
			}
		},
	}, {
		// Content as an array of typed parts (some servers do this even in deltas).
		name: "array-of-parts content",
		sse: `data: {"choices":[{"delta":{"content":[{"type":"text","text":"pa"},{"type":"text","text":"rts"}]}}]}` + "\n" +
			"data: [DONE]\n",
		check: func(t *testing.T, r *llm.Response) {
			if r.Content != "parts" {
				t.Errorf("content = %q, want parts", r.Content)
			}
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := serveSSE(t, tt.sse)
			tt.check(t, drain(t, mustStream(t, newProvider(t, srv.URL))))
		})
	}
}

func mustStream(t *testing.T, p llm.Provider) llm.Stream {
	t.Helper()
	s, err := p.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	return s
}

// --- non-streaming ----------------------------------------------------------

// The empty-choices case is real: a thinking model that spends its whole budget
// on reasoning returns zero choices, and indexing [0] would panic.
func TestCall_EmptyChoices(t *testing.T) {
	srv, _ := serveJSON(t, 200, `{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":12,"total_tokens":19,
		"completion_tokens_details":{"reasoning_tokens":12,"text_tokens":0}}}`)
	resp, err := newProvider(t, srv.URL).Call(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "" || len(resp.ToolCalls) != 0 {
		t.Errorf("want an empty response, got %+v", resp)
	}
	if resp.Usage.TotalTokens != 19 {
		t.Errorf("usage total = %d, want 19 (usage survives an empty turn)", resp.Usage.TotalTokens)
	}
}

func TestCall_ToolCallsAndCachedTokens(t *testing.T) {
	srv, _ := serveJSON(t, 200, `{"choices":[{"finish_reason":"tool_calls","message":{
		"content":"sure","reasoning_content":"pondering",
		"tool_calls":[{"id":"c1","function":{"name":"f","arguments":"{\"a\":1}"}}]}}],
		"usage":{"prompt_tokens":10,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":6}}}`)
	resp, err := newProvider(t, srv.URL).Call(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "sure" || resp.Thoughts != "pondering" || resp.StopReason != "tool_calls" {
		t.Errorf("got %+v", resp)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Arguments != `{"a":1}` {
		t.Errorf("tool calls = %+v", resp.ToolCalls)
	}
	if resp.Usage.CacheReadTokens != 6 {
		t.Errorf("cache read = %d, want 6", resp.Usage.CacheReadTokens)
	}
}

func TestCall_ErrorEnvelopes(t *testing.T) {
	t.Run("standard envelope", func(t *testing.T) {
		srv, _ := serveJSON(t, 400, `{"error":{"message":"model not found","type":"invalid_request_error"}}`)
		_, err := newProvider(t, srv.URL).Call(context.Background(), llm.Request{})
		if err == nil || !strings.Contains(err.Error(), "model not found") {
			t.Fatalf("err = %v, want the server's message", err)
		}
	})
	t.Run("non-standard body", func(t *testing.T) {
		srv, _ := serveJSON(t, 400, `not json at all`)
		_, err := newProvider(t, srv.URL).Call(context.Background(), llm.Request{})
		if err == nil || !strings.Contains(err.Error(), "not json at all") {
			t.Fatalf("err = %v, want the body snippet", err)
		}
	})
}

// --- what we SEND -----------------------------------------------------------

// The conservative-by-default rule: against an unknown server we send nothing
// optional, because several reject unknown request fields outright.
func TestBuildBody_ProfileDefaults(t *testing.T) {
	tests := []struct {
		name           string
		args           []string
		wantMaxField   string
		wantStreamOpts bool
	}{
		{"unknown server defaults to generic", nil, "max_tokens", false},
		{"litellm", []string{"profile", "litellm"}, "max_tokens", true},
		// Measured: ollama accepts and honours stream_options, so we must ask.
		{"ollama", []string{"profile", "ollama"}, "max_tokens", true},
		{"openai", []string{"profile", "openai"}, "max_completion_tokens", true},
		{"explicit override beats the profile", []string{"profile", "ollama", "max_tokens_field", "max_completion_tokens", "stream_usage", "true"}, "max_completion_tokens", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, got := serveSSE(t, "data: [DONE]\n")
			drain(t, mustStream(t, newProvider(t, srv.URL, tt.args...)))
			if _, ok := (*got)[tt.wantMaxField]; !ok {
				t.Errorf("body lacks %s; got keys %v", tt.wantMaxField, keysOf(*got))
			}
			if _, ok := (*got)["stream_options"]; ok != tt.wantStreamOpts {
				t.Errorf("stream_options present = %v, want %v", ok, tt.wantStreamOpts)
			}
		})
	}
}

// Spec args are passed through verbatim so a server-specific knob never needs a
// code change here; dotted keys nest, and values take their natural JSON type.
func TestBuildBody_SpecArgPassthrough(t *testing.T) {
	srv, got := serveJSON(t, 200, `{"choices":[]}`)
	p := newProvider(t, srv.URL,
		"reasoning.effort", "high",
		"provider.order", "cerebras",
		"temperature", "0.25",
		"logprobs", "true",
		"max_tokens", "128", // overrides the provider default
	)
	if _, err := p.Call(context.Background(), llm.Request{}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	body := *got
	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" {
		t.Errorf("reasoning = %#v, want nested {effort: high}", body["reasoning"])
	}
	if prov, ok := body["provider"].(map[string]any); !ok || prov["order"] != "cerebras" {
		t.Errorf("provider = %#v, want nested {order: cerebras}", body["provider"])
	}
	if body["temperature"] != 0.25 {
		t.Errorf("temperature = %#v, want float 0.25", body["temperature"])
	}
	if body["logprobs"] != true {
		t.Errorf("logprobs = %#v, want bool true", body["logprobs"])
	}
	// max_tokens is a CLIENT arg: it sets our cap, which buildBody then writes
	// into whichever field the profile selects — rather than being passed through
	// as a second, possibly conflicting, body field.
	if body["max_tokens"] != float64(128) {
		t.Errorf("max_tokens = %#v, want the client arg (128) to set the cap", body["max_tokens"])
	}
	if _, both := body["max_completion_tokens"]; both {
		t.Error("body carries both max_tokens and max_completion_tokens")
	}
	// Client-only args configure us and must never reach the server.
	for _, k := range []string{"base_url", "profile", "api_key_env", "stream_usage", "max_tokens_field", "capture_extras"} {
		if _, leaked := body[k]; leaked {
			t.Errorf("client-only arg %q leaked into the request body", k)
		}
	}
}

// A tool result carrying images has nowhere to put them (a `tool` message must
// be a plain string), so the images ride in a following user turn.
func TestConvertMessages_ToolResultImages(t *testing.T) {
	msgs := convertMessages(llm.Request{
		SystemPrompt: "sys",
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "view", Arguments: ""}}},
			{Role: llm.RoleToolResult, ToolCallID: "c1", Content: "here",
				Attachments: []llm.Attachment{{MimeType: "image/png", Base64Data: "AAAA"}}},
		},
	})
	if len(msgs) != 4 {
		t.Fatalf("messages = %d, want 4 (system, assistant, tool, image-carrier)", len(msgs))
	}
	assistant := msgs[1].(map[string]any)
	calls := assistant["tool_calls"].([]any)
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["arguments"] != "{}" {
		t.Errorf("empty arguments = %q, want {} (strict servers reject an empty string)", fn["arguments"])
	}
	tool := msgs[2].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "c1" || tool["content"] != "here" {
		t.Errorf("tool message = %#v", tool)
	}
	carrier := msgs[3].(map[string]any)
	parts, ok := carrier["content"].([]any)
	if carrier["role"] != "user" || !ok || len(parts) != 1 {
		t.Fatalf("image carrier = %#v, want a user turn with one image part", carrier)
	}
	img := parts[0].(map[string]any)["image_url"].(map[string]any)
	if img["url"] != "data:image/png;base64,AAAA" {
		t.Errorf("image url = %v", img["url"])
	}
}

// newFromFlatArgs mirrors createProvider's split for tests that express a spec
// as a single query.
func newFromFlatArgs(model string, maxTokens int, args url.Values) (llm.Provider, error) {
	// Sort the flat set the way a spec's block/query split would, so these tests
	// exercise the same classification production uses.
	block, query := url.Values{}, url.Values{}
	for k, v := range args {
		if ClientArgs[k] || k == "max_tokens" {
			block[k] = v
		} else {
			query[k] = v
		}
	}
	clientArgs, err := llm.ClientArgs(block, ClientArgs)
	if err != nil {
		return nil, err
	}
	modelArgs := query
	maxTokens, err = llm.MaxTokensArg(clientArgs, maxTokens)
	if err != nil {
		return nil, err
	}
	return New(model, maxTokens, clientArgs, modelArgs)
}

func TestNew_Validation(t *testing.T) {
	if _, err := newFromFlatArgs("m", 10, url.Values{"profile": {"nope"}}); err == nil {
		t.Error("unknown profile should fail fast at construction")
	}
	if _, err := newFromFlatArgs("m", 10, url.Values{"max_tokens_field": {"tokens"}}); err == nil {
		t.Error("bogus max_tokens_field should fail fast")
	}
	if _, err := newFromFlatArgs("m", 10, url.Values{"stream_usage": {"yes-please"}}); err == nil {
		t.Error("non-boolean stream_usage should fail fast")
	}
	// An undeclared key in the BLOCK is an error — the half we own is knowable,
	// so a typo is caught here instead of being shipped to the server.
	if _, err := llm.ClientArgs(url.Values{"bse_url": {"http://x"}}, ClientArgs); err == nil {
		t.Error("unknown client arg should fail fast")
	}
	// A model id containing a colon (ollama's "qwen3:30b-a3b") is normal: the
	// spec splits on the FIRST colon only, so it arrives here intact.
	p, err := newFromFlatArgs("qwen3:30b-a3b", 10, url.Values{"base_url": {"http://x/v1"}})
	if err != nil {
		t.Fatalf("New with a colon in the model: %v", err)
	}
	if p.ModelID() != "qwen3:30b-a3b" {
		t.Errorf("ModelID = %q, want the colon preserved", p.ModelID())
	}
}

func keysOf(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// A server that ignores stream:true and answers with plain JSON would otherwise
// look like an empty stream.
func TestStream_NonSSEResponseIsAnError(t *testing.T) {
	srv, _ := serveJSON(t, 200, `{"choices":[{"message":{"content":"not a stream"}}]}`)
	_, err := newProvider(t, srv.URL).Stream(context.Background(), llm.Request{})
	if err == nil || !strings.Contains(err.Error(), "not an SSE stream") {
		t.Fatalf("err = %v, want a clear non-SSE error", err)
	}
}

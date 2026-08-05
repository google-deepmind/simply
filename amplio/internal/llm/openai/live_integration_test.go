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

//go:build integration

// Live checks against a REAL OpenAI-compatible server. Unlike the fixture tests
// (which pin behaviour we've already observed), these catch the deviations we
// haven't met yet — run them against every new server before trusting it.
//
//	AMPLIO_OPENAI_TEST_BASE_URL=http://localhost:4000/v1 \
//	AMPLIO_OPENAI_TEST_MODEL=claude \
//	AMPLIO_OPENAI_TEST_PROFILE=litellm \
//	  make test-integration
//
// Skipped unless the base URL is set, so the tagged suite stays green anywhere.
package openai

import (
	"context"
	"encoding/json"
	"net/url"
	"os"
	"strings"
	"testing"

	"amplio/internal/llm"
)

func liveProvider(t *testing.T) llm.Provider {
	t.Helper()
	base := os.Getenv("AMPLIO_OPENAI_TEST_BASE_URL")
	if base == "" {
		t.Skip("set AMPLIO_OPENAI_TEST_BASE_URL to run the live OpenAI-compatible checks")
	}
	model := os.Getenv("AMPLIO_OPENAI_TEST_MODEL")
	if model == "" {
		model = "gpt-4o-mini"
	}
	args := url.Values{"base_url": {base}}
	if p := os.Getenv("AMPLIO_OPENAI_TEST_PROFILE"); p != "" {
		args.Set("profile", p)
	}
	p, err := newFromFlatArgs(model, 2048, args)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// Every live test draws its request from conformanceRequests() — the single
// audited definition of what may leave the machine (see
// TestEgressAudit_ConformancePayloads). Do not inline a request here: that
// would create a second, unaudited egress path.
func liveRequest(t *testing.T, name string) llm.Request {
	t.Helper()
	req, ok := conformanceRequests()[name]
	if !ok {
		t.Fatalf("no conformance request %q", name)
	}
	return req
}

func TestLive_Call(t *testing.T) {
	resp, err := liveProvider(t).Call(context.Background(), liveRequest(t, "call"))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Content), "ok") {
		t.Errorf("content = %q, want it to contain ok", resp.Content)
	}
	if resp.Usage.TotalTokens == 0 {
		t.Errorf("usage is zero; the server reported none on a non-streaming call")
	}
	t.Logf("content=%q thoughts=%d chars usage=%+v stop=%q",
		resp.Content, len(resp.Thoughts), resp.Usage, resp.StopReason)
}

// The one that matters: parallel tool calls over a live stream, which is where
// dialects diverge (split arguments, index semantics, id placement).
func TestLive_StreamParallelToolCalls(t *testing.T) {
	p := liveProvider(t)
	s, err := p.Stream(context.Background(), liveRequest(t, "stream_parallel_tools"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()
	var starts, argDeltas int
	for s.Next() {
		ev := s.Event()
		if ev.ToolCallStart != nil {
			starts++
		}
		if ev.ToolCallDelta != nil {
			argDeltas++
		}
	}
	if err := s.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	resp := s.Response()
	if len(resp.ToolCalls) < 2 {
		t.Fatalf("tool calls = %d, want 2: %+v", len(resp.ToolCalls), resp.ToolCalls)
	}
	if starts < 2 {
		t.Errorf("tool-call start events = %d, want one per call", starts)
	}
	// Every assembled argument string must be valid JSON on its own: this is the
	// property that fails if fragments are dropped or interleaved across calls.
	for i, tc := range resp.ToolCalls {
		var into map[string]any
		if err := json.Unmarshal([]byte(tc.Arguments), &into); err != nil {
			t.Errorf("call %d arguments are not valid JSON (%v): %q", i, err, tc.Arguments)
			continue
		}
		if into["city"] == nil {
			t.Errorf("call %d has no city: %q", i, tc.Arguments)
		}
		if tc.ID == "" || tc.Name != "get_weather" {
			t.Errorf("call %d = %+v, want an id and the tool name", i, tc)
		}
	}
	t.Logf("calls=%d starts=%d arg-deltas=%d stop=%q usage=%+v",
		len(resp.ToolCalls), starts, argDeltas, resp.StopReason, resp.Usage)
}

// A second turn that REPLAYS an assistant tool call plus its result: the shape
// most likely to be rejected by a strict server (empty arguments, missing
// content field, orphaned tool_call_id).
func TestLive_ToolResultRoundTrip(t *testing.T) {
	resp, err := liveProvider(t).Call(context.Background(), liveRequest(t, "tool_result_round_trip"))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	// PROTOCOL assertion: the server accepted a replayed assistant tool call +
	// its result and produced a turn. Whether the model then USES the result is
	// a quality question, and a thinking model can spend its whole budget on
	// reasoning and return empty content — a real observation, not a bug in the
	// transport. So that half is reported, not enforced, with enough detail to
	// tell the two apart next time.
	if resp.Content == "" && resp.Thoughts == "" && len(resp.ToolCalls) == 0 {
		t.Fatalf("empty turn: no content, no reasoning, no tool calls (stop=%q usage=%+v)",
			resp.StopReason, resp.Usage)
	}
	used := strings.Contains(resp.Content, "18") || strings.Contains(strings.ToLower(resp.Content), "rain")
	t.Logf("round trip: used-result=%v content=%d chars thoughts=%d chars stop=%q usage=%+v",
		used, len(resp.Content), len(resp.Thoughts), resp.StopReason, resp.Usage)
	if !used {
		t.Logf("NOTE: the model did not visibly use the tool result. Check stop=%q: "+
			"\"length\" means the budget went to reasoning; \"stop\" with empty content "+
			"means the model chose to say nothing.", resp.StopReason)
	}
}

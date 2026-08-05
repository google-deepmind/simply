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

package anthropic

import (
	"encoding/json"
	"net/url"
	"testing"

	"amplio/internal/llm"

	"github.com/anthropics/anthropic-sdk-go"
)

func TestParseCacheTTL(t *testing.T) {
	tests := []struct {
		arg  string
		want anthropic.CacheControlEphemeralTTL
	}{
		{"1h", anthropic.CacheControlEphemeralTTLTTL1h},
		{"5m", anthropic.CacheControlEphemeralTTLTTL5m},
		{"", anthropic.CacheControlEphemeralTTLTTL5m},      // default
		{"bogus", anthropic.CacheControlEphemeralTTLTTL5m}, // default
	}
	for _, tc := range tests {
		v := url.Values{}
		if tc.arg != "" {
			v.Set("cache_ttl", tc.arg)
		}
		if got := parseCacheTTL(v); got != tc.want {
			t.Errorf("parseCacheTTL(cache_ttl=%q) = %q, want %q", tc.arg, got, tc.want)
		}
	}
}

// TestClientArgsNeverReachTheBody guards the fix for the cache_ttl leak: a
// client arg (interpreted by parseCacheTTL, not by the API) must NOT be
// forwarded to the request body as a raw override, or Anthropic rejects the call
// with 400 "cache_ttl: Extra inputs are not permitted".
//
// The exclusion now happens in the split rather than inside specReqOpts, so the
// test asserts the pair: cache_ttl lands on the client side and reaches
// parseCacheTTL, while a genuine body override still becomes exactly one option.
func TestClientArgsNeverReachTheBody(t *testing.T) {
	for _, tc := range []struct {
		name         string
		block, query url.Values
	}{
		{"block spelling", url.Values{"cache_ttl": {"1h"}}, url.Values{"thinking.budget_tokens": {"2048"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientArgs, err := llm.ClientArgs(tc.block, ClientArgs)
			if err != nil {
				t.Fatalf("ClientArgs: %v", err)
			}
			modelArgs := tc.query
			if got := clientArgs.Get("cache_ttl"); got != "1h" {
				t.Errorf("cache_ttl reached the client side as %q, want %q", got, "1h")
			}
			if got := specReqOpts(modelArgs); len(got) != 1 {
				t.Errorf("specReqOpts produced %d opts, want 1 (only the real body override)", len(got))
			}
		})
	}
}
func TestCoerce(t *testing.T) {
	tests := []struct {
		in   string
		want any
	}{
		{"1024", int64(1024)},
		{"0", int64(0)},
		{"-5", int64(-5)},
		{"true", true},
		{"false", false},
		{"0.7", 0.7},
		{"high", "high"},
		{"adaptive", "adaptive"},
		{"", ""},
	}
	for _, tc := range tests {
		if got := coerce(tc.in); got != tc.want {
			t.Errorf("coerce(%q) = %v (%T), want %v (%T)", tc.in, got, got, tc.want, tc.want)
		}
	}
}

// Parallel tool calls whose results BOTH carry image attachments must serialize
// as a single user message whose content is the N tool_result blocks contiguous
// (each image nested INSIDE its own tool_result), not interleaved with sibling
// image blocks. The latter put an image between two tool_result blocks and made
// Vertex/Anthropic 400 with "tool_use ids were found without tool_result blocks
// immediately after".
func TestConvertMessages_ParallelToolResultsWithAttachments(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{ID: "call_a", Name: "view_file", Arguments: `{}`},
			{ID: "call_b", Name: "view_file", Arguments: `{}`},
		}},
		{Role: llm.RoleToolResult, ToolCallID: "call_a", Content: "a",
			Attachments: []llm.Attachment{{MimeType: "image/png", Base64Data: "AAAA"}}},
		{Role: llm.RoleToolResult, ToolCallID: "call_b", Content: "b",
			Attachments: []llm.Attachment{{MimeType: "image/png", Base64Data: "BBBB"}}},
	}

	out := convertMessages(msgs)
	// assistant turn + ONE coalesced user turn for both results.
	if len(out) != 2 {
		t.Fatalf("got %d messages, want 2 (assistant + one coalesced tool-result turn)", len(out))
	}
	turn := out[1]
	if len(turn.Content) != 2 {
		t.Fatalf("tool-result turn has %d content blocks, want 2 (one tool_result per call)", len(turn.Content))
	}
	for i, want := range []string{"call_a", "call_b"} {
		tr := turn.Content[i].OfToolResult
		if tr == nil {
			t.Fatalf("content[%d] is not a tool_result block", i)
		}
		if tr.ToolUseID != want {
			t.Errorf("content[%d] tool_use_id = %q, want %q", i, tr.ToolUseID, want)
		}
		// text + image carried INSIDE the tool_result, not as siblings.
		if len(tr.Content) != 2 {
			t.Fatalf("tool_result %q has %d inner blocks, want 2 (text + image)", want, len(tr.Content))
		}
		if tr.Content[0].OfText == nil {
			t.Errorf("tool_result %q inner[0] is not text", want)
		}
		if tr.Content[1].OfImage == nil {
			t.Errorf("tool_result %q inner[1] is not image", want)
		}
	}
}

func TestSpecReqOpts(t *testing.T) {
	// One option per arg; nil-safe for empty.
	if got := specReqOpts(nil); got != nil {
		t.Errorf("specReqOpts(nil) = %v, want nil", got)
	}
	args := url.Values{
		"thinking.type":          {"adaptive"},
		"output_config.effort":   {"high"},
		"thinking.budget_tokens": {"2048"},
	}
	if got := specReqOpts(args); len(got) != 3 {
		t.Errorf("specReqOpts: got %d opts, want 3", len(got))
	}
}

func TestBuildParams_AutomaticCaching(t *testing.T) {
	p := &provider{model: "claude-opus-4-8", maxTokens: 1024, cacheTTL: anthropic.CacheControlEphemeralTTLTTL1h}
	req := llm.Request{
		SystemPrompt: "you are a helpful agent",
		Tools: []llm.ToolDef{
			{Name: "a", Description: "tool a", Schema: []byte(`{"properties":{}}`)},
			{Name: "b", Description: "tool b", Schema: []byte(`{"properties":{}}`)},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "hello"},
		},
	}
	params := p.buildParams(req)

	// Automatic caching: a single top-level cache_control breakpoint at the
	// configured TTL. No per-block breakpoints.
	if params.CacheControl.Type != "ephemeral" {
		t.Errorf("missing top-level cache_control: %+v", params.CacheControl)
	}
	if params.CacheControl.TTL != "1h" {
		t.Errorf("top-level cache TTL = %q, want 1h", params.CacheControl.TTL)
	}
	for _, s := range params.System {
		if s.CacheControl.Type == "ephemeral" {
			t.Errorf("system block has an unexpected per-block breakpoint")
		}
	}
	for i, tu := range params.Tools {
		if tu.OfTool != nil && tu.OfTool.CacheControl.Type == "ephemeral" {
			t.Errorf("tool %d has an unexpected per-block breakpoint", i)
		}
	}
	for _, m := range params.Messages {
		for _, b := range m.Content {
			if b.OfText != nil && b.OfText.CacheControl.Type == "ephemeral" {
				t.Errorf("message block has an unexpected per-block breakpoint")
			}
		}
	}
}

// With prompt caching on, Anthropic reports input tokens split across three
// disjoint buckets (uncached input + cache read + cache write). PromptTokens
// must sum all three, so the status-bar total reflects the true prompt size
// rather than collapsing to the small uncached delta once the stable prefix is
// served from cache. Cache read/write are also surfaced individually.
func TestConvertResponse_CachedTokenAccounting(t *testing.T) {
	msg := &anthropic.Message{
		StopReason: anthropic.StopReasonEndTurn,
		Usage: anthropic.Usage{
			InputTokens:              120,   // newly-processed (uncached) input
			CacheReadInputTokens:     40000, // stable prefix served from cache
			CacheCreationInputTokens: 300,   // written to cache this turn
			OutputTokens:             500,
		},
	}
	resp := convertResponse(msg)
	if got, want := resp.Usage.PromptTokens, 120+40000+300; got != want {
		t.Errorf("PromptTokens = %d, want %d (sum of all three input buckets)", got, want)
	}
	if got, want := resp.Usage.CompletionTokens, 500; got != want {
		t.Errorf("CompletionTokens = %d, want %d", got, want)
	}
	if got, want := resp.Usage.CacheReadTokens, 40000; got != want {
		t.Errorf("CacheReadTokens = %d, want %d", got, want)
	}
	if got, want := resp.Usage.CacheWriteTokens, 300; got != want {
		t.Errorf("CacheWriteTokens = %d, want %d", got, want)
	}
	if got, want := resp.Usage.TotalTokens, 120+40000+300+500; got != want {
		t.Errorf("TotalTokens = %d, want %d (prompt + completion)", got, want)
	}
}

// convertResponse must PRESERVE thinking + redacted_thinking blocks (verbatim,
// in order, with signature) under preserveThinkingKey so they can be replayed on
// the next turn — Anthropic requires the tool-calling turn's thinking blocks to
// be passed back during tool use.
func TestConvertResponse_PreservesThinkingBlocks(t *testing.T) {
	msg := &anthropic.Message{
		StopReason: anthropic.StopReasonToolUse,
		Content: []anthropic.ContentBlockUnion{
			{Type: "thinking", Thinking: "let me reason", Signature: "sig-abc"},
			{Type: "redacted_thinking", Data: "enc-blob"},
			{Type: "text", Text: "the answer"},
			{Type: "tool_use", ID: "call_1", Name: "calc", Input: json.RawMessage(`{"x":1}`)},
		},
	}
	resp := convertResponse(msg)

	if resp.Content != "the answer" {
		t.Errorf("Content = %q, want %q", resp.Content, "the answer")
	}
	if resp.Thoughts != "let me reason" {
		t.Errorf("Thoughts = %q, want %q", resp.Thoughts, "let me reason")
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "call_1" {
		t.Fatalf("ToolCalls = %+v, want one call_1", resp.ToolCalls)
	}
	v, ok := resp.ProviderExtra[preserveThinkingKey]
	if !ok {
		t.Fatalf("ProviderExtra missing %q; got %+v", preserveThinkingKey, resp.ProviderExtra)
	}
	blocks, ok := v.([]map[string]string)
	if !ok {
		t.Fatalf("%q is %T, want []map[string]string", preserveThinkingKey, v)
	}
	if len(blocks) != 2 {
		t.Fatalf("preserved %d blocks, want 2 (thinking + redacted_thinking)", len(blocks))
	}
	if blocks[0]["type"] != "thinking" || blocks[0]["thinking"] != "let me reason" || blocks[0]["signature"] != "sig-abc" {
		t.Errorf("block[0] = %v, want verbatim thinking+signature", blocks[0])
	}
	if blocks[1]["type"] != "redacted_thinking" || blocks[1]["data"] != "enc-blob" {
		t.Errorf("block[1] = %v, want redacted_thinking+data", blocks[1])
	}
}

// No thinking blocks in the response => no ProviderExtra (the Vertex-adaptive
// case, where thinking is internal and never surfaced). Must stay nil so the
// replay path is a clean no-op.
func TestConvertResponse_NoThinkingNoProviderExtra(t *testing.T) {
	msg := &anthropic.Message{
		StopReason: anthropic.StopReasonEndTurn,
		Content:    []anthropic.ContentBlockUnion{{Type: "text", Text: "hi"}},
	}
	if resp := convertResponse(msg); resp.ProviderExtra != nil {
		t.Errorf("ProviderExtra = %v, want nil when no thinking blocks", resp.ProviderExtra)
	}
}

// convertMessages must REPLAY preserved thinking blocks as the LEADING content
// blocks of the assistant turn (before text and tool_use), verbatim. Exercised
// through a DB-style JSON round-trip of ProviderExtra ([]any / map[string]any),
// which is how the harness actually stores and reloads it.
func TestConvertMessages_ReplaysThinkingBlocksFirst(t *testing.T) {
	// Simulate the persisted+reloaded shape: marshal the native form to JSON and
	// unmarshal into map[string]any (what a DB round-trip yields).
	native := map[string]any{preserveThinkingKey: []map[string]string{
		{"type": "thinking", "thinking": "reasoning", "signature": "sig-xyz"},
		{"type": "redacted_thinking", "data": "enc"},
	}}
	b, err := json.Marshal(native)
	if err != nil {
		t.Fatal(err)
	}
	var extra map[string]any
	if err := json.Unmarshal(b, &extra); err != nil {
		t.Fatal(err)
	}

	msgs := []llm.Message{{
		Role:          llm.RoleAssistant,
		Content:       "visible text",
		ToolCalls:     []llm.ToolCall{{ID: "call_1", Name: "calc", Arguments: `{}`}},
		ProviderExtra: extra,
	}}
	out := convertMessages(msgs)
	if len(out) != 1 {
		t.Fatalf("got %d messages, want 1", len(out))
	}
	c := out[0].Content
	// Expected order: thinking, redacted_thinking, text, tool_use.
	if len(c) != 4 {
		t.Fatalf("assistant turn has %d blocks, want 4 (thinking, redacted_thinking, text, tool_use)", len(c))
	}
	if c[0].OfThinking == nil {
		t.Fatalf("block[0] is not a thinking block: %+v", c[0])
	}
	if c[0].OfThinking.Thinking != "reasoning" || c[0].OfThinking.Signature != "sig-xyz" {
		t.Errorf("thinking block = %+v, want verbatim reasoning+signature", c[0].OfThinking)
	}
	if c[1].OfRedactedThinking == nil || c[1].OfRedactedThinking.Data != "enc" {
		t.Errorf("block[1] = %+v, want redacted_thinking with data=enc", c[1])
	}
	if c[2].OfText == nil || c[2].OfText.Text != "visible text" {
		t.Errorf("block[2] = %+v, want text 'visible text'", c[2])
	}
	if c[3].OfToolUse == nil || c[3].OfToolUse.ID != "call_1" {
		t.Errorf("block[3] = %+v, want tool_use call_1", c[3])
	}
}

// An unsigned thinking block must NOT be replayed (the API rejects a thinking
// block without a signature); the rest of the turn still converts.
func TestConvertMessages_SkipsUnsignedThinking(t *testing.T) {
	msgs := []llm.Message{{
		Role:    llm.RoleAssistant,
		Content: "txt",
		ProviderExtra: map[string]any{preserveThinkingKey: []map[string]string{
			{"type": "thinking", "thinking": "no sig here", "signature": ""},
		}},
	}}
	c := convertMessages(msgs)[0].Content
	for i, b := range c {
		if b.OfThinking != nil {
			t.Errorf("block[%d] is an (unsigned) thinking block that should have been skipped", i)
		}
	}
	if len(c) != 1 || c[0].OfText == nil {
		t.Fatalf("want just the text block, got %d blocks", len(c))
	}
}

// No ProviderExtra => no thinking blocks replayed (backward-compatible with
// turns recorded before the fix, and with backends that never return thinking).
func TestConvertMessages_NoProviderExtraNoThinking(t *testing.T) {
	msgs := []llm.Message{{
		Role:      llm.RoleAssistant,
		Content:   "txt",
		ToolCalls: []llm.ToolCall{{ID: "c1", Name: "t", Arguments: `{}`}},
	}}
	c := convertMessages(msgs)[0].Content
	if len(c) != 2 || c[0].OfText == nil || c[1].OfToolUse == nil {
		t.Fatalf("want [text, tool_use], got %d blocks: %+v", len(c), c)
	}
}

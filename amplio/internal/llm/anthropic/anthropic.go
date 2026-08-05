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
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"strconv"

	"amplio/internal/llm"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/anthropics/anthropic-sdk-go/vertex"
)

// maxRetries is the per-call retry budget for transient backend failures
// (429 RESOURCE_EXHAUSTED / overloaded, 5xx, connection drops). The Anthropic
// SDK retries these with exponential backoff + Retry-After honoring; we raise
// its default (2) to ride out longer overload blips on the shared fleet.
const maxRetries = 5

// preserveThinkingKey is the provider-namespaced ProviderExtra key holding the
// assistant turn's thinking blocks, verbatim and IN ORDER, as a slice of
// {type,thinking,signature} / {type,data} maps. Anthropic requires the thinking
// blocks that produced a tool call to be replayed — unmodified, with their
// cryptographic signature — on the following request, and to appear FIRST in the
// assistant message's content (before text/tool_use). See convertResponse
// (capture) and convertMessages (replay).
//
// https://platform.claude.com/docs/en/build-with-claude/extended-thinking#preserving-thinking-blocks
const preserveThinkingKey = "anthropic.thinking_blocks"

type provider struct {
	client    anthropic.Client
	model     string
	maxTokens int
	// reqOpts are per-request raw body overrides parsed from the spec args
	// (e.g. ?thinking.type=adaptive&output_config.effort=high). Forwarded to
	// every call via option.WithJSONSet so model-specific knobs need no typed code.
	reqOpts []option.RequestOption
	// cacheTTL is the prompt-cache TTL ("5m" or "1h") for cache_control
	// breakpoints, set per model via the spec arg ?cache_ttl=5m|1h.
	cacheTTL anthropic.CacheControlEphemeralTTL
}

// NewVertex creates a Provider for Claude on the Vertex AI backend. Reads
// VERTEXAI_PROJECT and VERTEXAI_LOCATION from the environment (the same ADC auth
// as the Gemini provider and the embedder). args are spec `?k=v` overrides
// applied to the request body verbatim (see specReqOpts).
func NewVertex(model string, maxTokens int, clientArgs, args url.Values) (llm.Provider, error) {
	project := os.Getenv("VERTEXAI_PROJECT")
	location := os.Getenv("VERTEXAI_LOCATION")
	if project == "" {
		return nil, fmt.Errorf("VERTEXAI_PROJECT not set")
	}
	if location == "" {
		location = "us-east5"
	}
	// Pass the cloud-platform scope explicitly: WithGoogleAuth forwards scopes to
	// google.FindDefaultCredentials, and with none the minted ADC token is
	// scopeless — Vertex then rejects every call with "Scope required".
	client := anthropic.NewClient(
		vertex.WithGoogleAuth(context.Background(), location, project,
			"https://www.googleapis.com/auth/cloud-platform"),
		option.WithMaxRetries(maxRetries),
	)
	return &provider{client: client, model: model, maxTokens: maxTokens, reqOpts: specReqOpts(args), cacheTTL: parseCacheTTL(clientArgs)}, nil
}

// NewAPIKey creates a Provider on the direct Anthropic API backend using
// ANTHROPIC_API_KEY — the no-GCP path, mirroring gemini.NewAPIKey. args are spec
// `?k=v` overrides applied to the request body verbatim (see specReqOpts).
func NewAPIKey(model string, maxTokens int, clientArgs, args url.Values) (llm.Provider, error) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}
	client := anthropic.NewClient(option.WithAPIKey(key), option.WithMaxRetries(maxRetries))
	return &provider{client: client, model: model, maxTokens: maxTokens, reqOpts: specReqOpts(args), cacheTTL: parseCacheTTL(clientArgs)}, nil
}

// specReqOpts turns spec args into raw request-body overrides via sjson paths,
// so dotted keys nest (thinking.budget_tokens → {"thinking":{"budget_tokens":N}}).
// Values are coerced to the natural JSON type. The model itself validates them
// (e.g. opus-4-8 rejects thinking.type=enabled), so we don't second-guess keys.
//
// Client args never reach here: they control our behavior, not the request body
// (and the API rejects unknown body fields), so they are split off before
// construction — see llm.ClientArgs.
func specReqOpts(args url.Values) []option.RequestOption {
	var opts []option.RequestOption
	for k := range args {
		opts = append(opts, option.WithJSONSet(k, coerce(args.Get(k))))
	}
	return opts
}

// ClientArgs are the arguments this provider interprets — the `{k=v}` block in a
// spec (see internal/llm/spec.go). They are never forwarded to the request body
// (the Anthropic API rejects unknown fields with 400).
var ClientArgs = map[string]bool{
	"cache_ttl": true,
}

// coerce parses a spec arg string to the most specific JSON type: int, then
// bool, then float, else string. Int is tried before bool so "1"/"0" stay
// numeric (budget_tokens), while "true"/"false" become bool.
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

func (p *provider) ModelID() string { return p.model }
func (p *provider) MaxTokens() int  { return p.maxTokens }

func (p *provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	// Always stream + accumulate, even for this blocking path: Anthropic rejects
	// non-streaming requests whose max_tokens could exceed the 10-minute
	// non-streaming limit ("streaming is required for operations that may take
	// longer than 10 minutes"), and our default max_tokens is large (65536).
	params := p.buildParams(req)
	stream := p.client.Messages.NewStreaming(ctx, params, p.callOpts(req)...)
	defer stream.Close()
	var acc anthropic.Message
	for stream.Next() {
		if err := acc.Accumulate(stream.Current()); err != nil {
			return nil, fmt.Errorf("anthropic call accumulate: %w", err)
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("anthropic call: %w", err)
	}
	return convertResponse(&acc), nil
}

func (p *provider) Stream(ctx context.Context, req llm.Request) (llm.Stream, error) {
	params := p.buildParams(req)
	stream := p.client.Messages.NewStreaming(ctx, params, p.callOpts(req)...)
	return &anthropicStream{stream: stream}, nil
}

// callOpts adds the X-Vertex-Ai-Session-Id header (for cache/routing affinity)
// to the spec reqOpts when the request has a SessionID.
func (p *provider) callOpts(req llm.Request) []option.RequestOption {
	if req.SessionID == "" {
		return p.reqOpts
	}
	opts := make([]option.RequestOption, 0, len(p.reqOpts)+1)
	opts = append(opts, p.reqOpts...)
	opts = append(opts, option.WithHeader("X-Vertex-Ai-Session-Id", req.SessionID))
	return opts
}

// parseCacheTTL reads the cache_ttl client arg; "1h" selects the 1-hour tier,
// "5m"/empty the 5-minute tier. Any other value warns and defaults to 5m.
func parseCacheTTL(clientArgs url.Values) anthropic.CacheControlEphemeralTTL {
	switch v := clientArgs.Get("cache_ttl"); v {
	case "1h":
		return anthropic.CacheControlEphemeralTTLTTL1h
	case "", "5m":
		return anthropic.CacheControlEphemeralTTLTTL5m
	default:
		slog.Warn("anthropic: unsupported cache_ttl, defaulting to 5m", "cache_ttl", v, "supported", "5m|1h")
		return anthropic.CacheControlEphemeralTTLTTL5m
	}
}

// cacheBreakpoint returns an ephemeral cache_control marker at the configured TTL.
func (p *provider) cacheBreakpoint() anthropic.CacheControlEphemeralParam {
	cc := anthropic.NewCacheControlEphemeralParam()
	cc.TTL = p.cacheTTL
	return cc
}

func (p *provider) buildParams(req llm.Request) anthropic.MessageNewParams {
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = p.maxTokens
	}
	params := anthropic.MessageNewParams{
		Model:     p.model,
		MaxTokens: int64(maxTokens),
		Messages:  convertMessages(req.Messages),
	}
	if req.SystemPrompt != "" {
		params.System = []anthropic.TextBlockParam{{Text: req.SystemPrompt}}
	}
	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
	}
	// Automatic caching: a single top-level cache_control breakpoint that the
	// service auto-places on the last cacheable block and rolls forward as the
	// conversation grows (caches tools+system+messages incrementally).
	params.CacheControl = p.cacheBreakpoint()
	if req.Temperature != nil {
		params.Temperature = anthropic.Float(*req.Temperature)
	}
	return params
}

func convertMessages(msgs []llm.Message) []anthropic.MessageParam {
	var result []anthropic.MessageParam
	for i := 0; i < len(msgs); i++ {
		m := msgs[i]
		switch m.Role {
		case llm.RoleSystem:
			// Mid-conversation system messages are supported on the direct
			// Anthropic API (Opus 4.8+) but NOT on Vertex AI.
			// Convert to user message for Vertex AI compatibility. When Vertex
			// adds support, this can use MessageParamRoleSystem instead.
			result = append(result, anthropic.NewUserMessage(
				anthropic.NewTextBlock(m.Content),
			))
		case llm.RoleUser:
			blocks := []anthropic.ContentBlockParamUnion{
				anthropic.NewTextBlock(m.Content),
			}
			blocks = appendImageBlocks(blocks, m.Attachments)
			result = append(result, anthropic.NewUserMessage(blocks...))
		case llm.RoleAssistant:
			var blocks []anthropic.ContentBlockParamUnion
			// Thinking blocks FIRST, verbatim (see preserveThinkingKey). Anthropic
			// requires the tool-calling turn's thinking blocks — with their
			// signature intact — to lead the content array; a bare text/tool_use
			// assistant turn with thinking enabled is rejected otherwise. On
			// backends that don't return thinking blocks (e.g. Vertex adaptive)
			// there are none, so this is a no-op.
			blocks = append(blocks, replayThinkingBlocks(m.ProviderExtra)...)
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    tc.ID,
						Name:  tc.Name,
						Input: json.RawMessage(tc.Arguments),
					},
				})
			}
			// Anthropic rejects messages with empty `content` ("Field required"
			// on the offending index). An "empty assistant turn" (model
			// returned end_turn with no text AND no tool call — rare but
			// happens, e.g. on a safety-blocked completion or a truncated
			// stream) would otherwise serialize as MessageParam{Content: nil}.
			//
			// We can't just drop the message because the API also requires
			// user/assistant alternation — skipping would give us two
			// consecutive user messages and break the next call too. A
			// placeholder text block keeps the structure intact and is honest
			// about what happened ("turn produced nothing").
			if len(blocks) == 0 {
				blocks = append(blocks, anthropic.NewTextBlock("(empty turn)"))
			}
			result = append(result, anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleAssistant,
				Content: blocks,
			})
		case llm.RoleToolResult:
			// Anthropic requires the user turn answering an assistant turn with N
			// parallel tool_use blocks to carry ALL N tool_result blocks,
			// contiguous, in a SINGLE message. The harness records each tool result
			// as its own message, so coalesce the whole run of consecutive results
			// into one user turn (mirrors the Gemini provider). Each result's
			// images go INSIDE that result's content union — NOT as sibling
			// top-level blocks — so an image can't interleave between two
			// tool_result blocks and break the "tool_result immediately after"
			// rule (which 400s only on parallel calls that both carry attachments).
			var blocks []anthropic.ContentBlockParamUnion
			for ; i < len(msgs) && msgs[i].Role == llm.RoleToolResult; i++ {
				tr := msgs[i]
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: tr.ToolCallID,
						Content:   toolResultContent(tr.Content, tr.Attachments),
						IsError:   anthropic.Bool(tr.IsError),
					},
				})
			}
			i-- // the outer loop's i++ re-advances past the last result
			result = append(result, anthropic.NewUserMessage(blocks...))
		}
	}
	return result
}

// replayThinkingBlocks reconstructs the assistant turn's preserved thinking
// blocks (see preserveThinkingKey) as leading content blocks, verbatim. It
// tolerates both the in-memory []map[string]string and the []any /
// []map[string]any shapes that survive a DB JSON round-trip. A missing key, a
// wrong-typed value, or an empty signature yields no block (rather than sending
// a malformed one the API would reject). Redacted (encrypted) thinking is
// replayed from its opaque `data` blob.
func replayThinkingBlocks(extra map[string]any) []anthropic.ContentBlockParamUnion {
	if extra == nil {
		return nil
	}
	v, ok := extra[preserveThinkingKey]
	if !ok {
		return nil
	}
	// Normalize through JSON so both the native and DB-roundtripped shapes decode
	// into a uniform []map[string]string.
	raw, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var items []map[string]string
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil
	}
	var blocks []anthropic.ContentBlockParamUnion
	for _, it := range items {
		switch it["type"] {
		case "redacted_thinking":
			if d := it["data"]; d != "" {
				blocks = append(blocks, anthropic.NewRedactedThinkingBlock(d))
			}
		case "thinking", "":
			// A thinking block is only valid with a signature; skip an unsigned
			// one rather than let the API reject the whole request.
			if sig := it["signature"]; sig != "" {
				blocks = append(blocks, anthropic.NewThinkingBlock(sig, it["thinking"]))
			}
		}
	}
	return blocks
}

// toolResultContent builds a tool_result block's content: the text payload
// followed by any image attachments, all carried INSIDE the block (Anthropic's
// ToolResultBlockParamContentUnion supports images). The text block is kept
// unconditionally — matching the SDK's NewToolResultBlock, which tolerates an
// empty result string — so the block always has at least one content entry.
func toolResultContent(text string, attachments []llm.Attachment) []anthropic.ToolResultBlockParamContentUnion {
	content := []anthropic.ToolResultBlockParamContentUnion{
		{OfText: &anthropic.TextBlockParam{Text: text}},
	}
	for _, att := range attachments {
		content = append(content, anthropic.ToolResultBlockParamContentUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfBase64: &anthropic.Base64ImageSourceParam{
						Data:      att.Base64Data,
						MediaType: anthropic.Base64ImageSourceMediaType(att.MimeType),
					},
				},
			},
		})
	}
	return content
}

func appendImageBlocks(blocks []anthropic.ContentBlockParamUnion, attachments []llm.Attachment) []anthropic.ContentBlockParamUnion {
	for _, att := range attachments {
		blocks = append(blocks, anthropic.NewImageBlockBase64(att.MimeType, att.Base64Data))
	}
	return blocks
}

func convertTools(tools []llm.ToolDef) []anthropic.ToolUnionParam {
	var result []anthropic.ToolUnionParam
	for _, t := range tools {
		// Parse the JSON Schema into the SDK's expected structure.
		var props map[string]any
		var required []string
		if len(t.Schema) > 0 {
			var parsed map[string]any
			if err := json.Unmarshal(t.Schema, &parsed); err == nil {
				if p, ok := parsed["properties"]; ok {
					props = p.(map[string]any)
				}
				if r, ok := parsed["required"]; ok {
					if arr, ok := r.([]any); ok {
						for _, v := range arr {
							if s, ok := v.(string); ok {
								required = append(required, s)
							}
						}
					}
				}
			}
		}
		result = append(result, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: props,
					Required:   required,
				},
			},
		})
	}
	return result
}

func convertResponse(msg *anthropic.Message) *llm.Response {
	resp := &llm.Response{
		StopReason: string(msg.StopReason),
		Usage: llm.Usage{
			// With prompt caching on, Anthropic splits input tokens into three
			// disjoint buckets: InputTokens is only the NEWLY-processed (uncached)
			// input, while cache reads/writes are reported separately. PromptTokens
			// must sum all three to reflect the true total prompt size — otherwise it
			// collapses to the tiny per-turn delta once most of the (stable) prefix
			// is served from cache. Cache read/write are ALSO kept individually below.
			PromptTokens: int(msg.Usage.InputTokens) +
				int(msg.Usage.CacheReadInputTokens) +
				int(msg.Usage.CacheCreationInputTokens),
			CompletionTokens: int(msg.Usage.OutputTokens),
			CacheReadTokens:  int(msg.Usage.CacheReadInputTokens),
			CacheWriteTokens: int(msg.Usage.CacheCreationInputTokens),
		},
	}
	resp.Usage.TotalTokens = resp.Usage.PromptTokens + resp.Usage.CompletionTokens

	var thinkingBlocks []map[string]string
	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			resp.Content += block.Text
		case "thinking":
			resp.Thoughts += block.Thinking
			// Preserve the thinking block VERBATIM (text + cryptographic
			// signature) so it can be replayed on the next turn. Anthropic
			// requires the originating assistant turn's thinking blocks to be
			// passed back — unmodified, signature intact — when that turn made
			// tool calls (see convertMessages / preserveThinkingKey).
			thinkingBlocks = append(thinkingBlocks, map[string]string{
				"type":      "thinking",
				"thinking":  block.Thinking,
				"signature": block.Signature,
			})
		case "redacted_thinking":
			// Encrypted thinking: no readable text or signature, only an opaque
			// `data` blob that must also be echoed back verbatim during tool use.
			thinkingBlocks = append(thinkingBlocks, map[string]string{
				"type": "redacted_thinking",
				"data": block.Data,
			})
		case "tool_use":
			resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: string(block.Input),
			})
		}
	}
	if len(thinkingBlocks) > 0 {
		resp.ProviderExtra = map[string]any{preserveThinkingKey: thinkingBlocks}
	}
	return resp
}

// --- Streaming ---

type anthropicStream struct {
	stream  *ssestream.Stream[anthropic.MessageStreamEventUnion]
	current llm.StreamEvent
	acc     anthropic.Message // accumulates streamed events via acc.Accumulate()
}

func (s *anthropicStream) Next() bool {
	if !s.stream.Next() {
		return false
	}
	evt := s.stream.Current()
	_ = s.acc.Accumulate(evt)
	s.current = llm.StreamEvent{}

	switch evt.Type {
	case "content_block_delta":
		if evt.Delta.Type == "text_delta" {
			s.current.DeltaText = evt.Delta.Text
		} else if evt.Delta.Type == "thinking_delta" {
			s.current.DeltaThoughts = evt.Delta.Thinking
		} else if evt.Delta.Type == "input_json_delta" {
			s.current.ToolCallDelta = &llm.ToolCallDelta{
				ArgumentsDelta: evt.Delta.PartialJSON,
			}
		}
	case "content_block_start":
		if evt.ContentBlock.Type == "tool_use" {
			s.current.ToolCallStart = &llm.ToolCallStart{
				ID:   evt.ContentBlock.ID,
				Name: evt.ContentBlock.Name,
			}
		}
	}
	return true
}

func (s *anthropicStream) Event() llm.StreamEvent { return s.current }

func (s *anthropicStream) Response() *llm.Response {
	return convertResponse(&s.acc)
}

func (s *anthropicStream) Err() error { return s.stream.Err() }
func (s *anthropicStream) Close()     { s.stream.Close() }

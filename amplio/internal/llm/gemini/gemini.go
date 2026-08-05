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

// Package gemini implements llm.Provider for Gemini models via the unified
// google.golang.org/genai SDK, on either the Vertex AI backend (ADC +
// VERTEXAI_PROJECT/LOCATION) or the Gemini Developer API backend (GEMINI_API_KEY).
package gemini

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	mrand "math/rand/v2"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"amplio/internal/llm"

	"google.golang.org/genai"
)

// sigKey is the provider-namespaced ProviderExtra key holding per-tool-call
// thought signatures: a []string of base64 blobs, index-aligned to the assistant
// turn's tool calls. Replayed onto the FunctionCall parts so Gemini can resume
// its chain of thought across tool turns.
const sigKey = "gemini.fc_sigs_b64"

// maxAttempts is the total number of tries (1 initial + retries) for a
// transient-failing generate call. The genai SDK does NOT retry content
// generation (its only retry loop is the file-upload path), so we implement
// this ourselves. 5 attempts to ride out shared-fleet overload blips
// (429 RESOURCE_EXHAUSTED / PREFILL_QUEUE_OVERLOADED, 503 UNAVAILABLE).
const maxAttempts = 5

// baseRetryDelay is the first backoff; it doubles each attempt (capped) with
// jitter. A var so tests can zero it out.
var baseRetryDelay = 1 * time.Second

const maxRetryDelay = 20 * time.Second

// isTransient reports whether a generate error is a retryable backend overload /
// availability blip rather than a caller error. Matches on the SDK's structured
// APIError.Code (429 / 5xx) first, then falls back to string signatures for
// overload markers and bare connection errors that never carried an HTTP status.
// Deliberately does NOT retry 400/401/403/404 (bad request, auth, not found) or
// context-window overflow — those must fail fast (the eventloop's reactive
// compaction depends on the overflow error surfacing).
func isTransient(err error) bool {
	if err == nil {
		return false
	}
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		if apiErr.Code == 429 || (apiErr.Code >= 500 && apiErr.Code <= 599) {
			return true
		}
		// A structured non-transient status (400/401/403/404, etc.) is
		// authoritative — don't second-guess it via string matching.
		if apiErr.Code != 0 {
			return false
		}
	}
	s := err.Error()
	for _, sig := range []string{
		"RESOURCE_EXHAUSTED",
		"UNAVAILABLE",
		"PREFILL_QUEUE_OVERLOADED",
		"Overloaded",
		"overloaded",
		"connection reset",
		"connection refused",
		"EOF",
	} {
		if strings.Contains(s, sig) {
			return true
		}
	}
	return false
}

// retryDelay returns the backoff before the given (1-based) retry: exponential
// from baseRetryDelay, doubling, capped at maxRetryDelay, with BOUNDED jitter —
// the wait is 75%-100% of the target (matching the Anthropic SDK's strategy).
// Bounded (rather than full 0..delay) jitter still de-correlates concurrent
// clients but guarantees every retry actually waits a meaningful amount, so a
// retry never fires near-instantly back into an overloaded fleet.
func retryDelay(attempt int) time.Duration {
	d := baseRetryDelay << (attempt - 1)
	// Clamp to max on overflow (shift wrapped negative) or when exceeded. A
	// legitimately-zero base (tests) must stay zero, so only treat NEGATIVE as
	// overflow, not <= 0.
	if d > maxRetryDelay || d < 0 {
		d = maxRetryDelay
	}
	if d == 0 {
		return 0
	}
	// 75%-100% of d: fixed 75% floor + up to 25% jitter.
	quarter := int64(d) / 4
	//nolint:gosec // retry-backoff jitter, not a security context; math/rand is correct.
	return time.Duration(int64(d) - quarter + mrand.Int64N(quarter+1))
}

// withRetry runs fn, retrying transient failures (see isTransient) up to
// maxAttempts with jittered exponential backoff, honoring ctx cancellation.
// Non-transient failures return immediately so real errors aren't masked.
func withRetry[T any](ctx context.Context, fn func() (T, error)) (T, error) {
	var zero T
	var out T
	var err error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		out, err = fn()
		if err == nil || !isTransient(err) {
			return out, err
		}
		if attempt == maxAttempts {
			break
		}
		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		case <-time.After(retryDelay(attempt)):
		}
	}
	return out, err
}

type provider struct {
	client    *genai.Client
	model     string
	maxTokens int
	cfg       geminiArgs // thinking/temperature overrides from the spec args
}

// geminiArgs holds the genai-typed knobs parsed from spec `?k=v` args. genai is
// a typed struct (no raw passthrough like Anthropic), so we map a known key set.
type geminiArgs struct {
	thinkingBudget  *int32
	includeThoughts *bool
	temperature     *float32
}

// parseGeminiArgs maps spec args to typed knobs, erroring on unknown keys (so a
// typo fails fast rather than being silently ignored).
func parseGeminiArgs(args url.Values) (geminiArgs, error) {
	var g geminiArgs
	for k := range args {
		v := args.Get(k)
		switch k {
		case "thinking_budget":
			n, err := strconv.ParseInt(v, 10, 32)
			if err != nil {
				return g, fmt.Errorf("gemini arg thinking_budget=%q: %w", v, err)
			}
			b := int32(n)
			g.thinkingBudget = &b
		case "include_thoughts":
			b, err := strconv.ParseBool(v)
			if err != nil {
				return g, fmt.Errorf("gemini arg include_thoughts=%q: %w", v, err)
			}
			g.includeThoughts = &b
		case "temperature":
			f, err := strconv.ParseFloat(v, 32)
			if err != nil {
				return g, fmt.Errorf("gemini arg temperature=%q: %w", v, err)
			}
			t := float32(f)
			g.temperature = &t
		default:
			return g, fmt.Errorf("unknown gemini spec arg %q (supported: thinking_budget, include_thoughts, temperature)", k)
		}
	}
	return g, nil
}

// NewVertex builds a Gemini provider on the Vertex AI backend, reusing
// VERTEXAI_PROJECT / VERTEXAI_LOCATION + ADC (the same auth as the embedder and
// the Claude provider). args are spec `?k=v` thinking/temperature overrides.
func NewVertex(model string, maxTokens int, _, args url.Values) (llm.Provider, error) {
	g, err := parseGeminiArgs(args)
	if err != nil {
		return nil, err
	}
	project := os.Getenv("VERTEXAI_PROJECT")
	if project == "" {
		return nil, fmt.Errorf("VERTEXAI_PROJECT not set")
	}
	location := os.Getenv("VERTEXAI_LOCATION")
	if location == "" {
		location = "us-central1"
	}
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  project,
		Location: location,
	})
	if err != nil {
		return nil, fmt.Errorf("gemini vertex client: %w", err)
	}
	return &provider{client: client, model: model, maxTokens: maxTokens, cfg: g}, nil
}

// NewAPIKey builds a Gemini provider on the Developer API backend using
// GEMINI_API_KEY (falling back to GOOGLE_API_KEY) — the no-GCP, OSS-friendly path.
func NewAPIKey(model string, maxTokens int, _, args url.Values) (llm.Provider, error) {
	g, err := parseGeminiArgs(args)
	if err != nil {
		return nil, err
	}
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		key = os.Getenv("GOOGLE_API_KEY")
	}
	if key == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY not set")
	}
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  key,
	})
	if err != nil {
		return nil, fmt.Errorf("gemini api client: %w", err)
	}
	return &provider{client: client, model: model, maxTokens: maxTokens, cfg: g}, nil
}

func (p *provider) ModelID() string { return p.model }
func (p *provider) MaxTokens() int  { return p.maxTokens }

func (p *provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	contents, config, err := p.build(req)
	if err != nil {
		return nil, err
	}
	resp, err := withRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return p.client.Models.GenerateContent(ctx, p.model, contents, config)
	})
	if err != nil {
		return nil, fmt.Errorf("gemini call: %w", err)
	}
	var acc respAcc
	acc.add(resp)
	return acc.response(), nil
}

func (p *provider) Stream(ctx context.Context, req llm.Request) (llm.Stream, error) {
	contents, config, err := p.build(req)
	if err != nil {
		return nil, err
	}
	// Retry transient overloads at STREAM ESTABLISHMENT only. A genai stream
	// surfaces its errors lazily through the iterator, and the first pull is
	// where an overload/unavailable shows up (before any tokens). We pull that
	// first chunk inside the retry loop; on a transient error we discard the
	// dead iterator and re-establish. Once the first chunk is in hand, tokens
	// have (potentially) started flowing, so we never retry past it — that would
	// risk double-emitting text.
	type primed struct {
		next  func() (*genai.GenerateContentResponse, error, bool)
		stop  func()
		first *genai.GenerateContentResponse
		ok    bool
	}
	ps, err := withRetry(ctx, func() (primed, error) {
		next, stop := iter.Pull2(p.client.Models.GenerateContentStream(ctx, p.model, contents, config))
		resp, err, ok := next()
		if err != nil {
			stop() // drop the dead iterator before a possible retry
			return primed{}, err
		}
		return primed{next: next, stop: stop, first: resp, ok: ok}, nil
	})
	if err != nil {
		return nil, fmt.Errorf("gemini stream: %w", err)
	}
	return &geminiStream{next: ps.next, stop: ps.stop, first: ps.first, hasFirst: ps.ok}, nil
}

func (p *provider) build(req llm.Request) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	contents, err := convertMessages(req.Messages)
	if err != nil {
		return nil, nil, err
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = p.maxTokens
	}
	// Thinking defaults to on (IncludeThoughts) so thoughts surface; spec args
	// override the budget / inclusion per model.
	thinking := &genai.ThinkingConfig{IncludeThoughts: true}
	if p.cfg.includeThoughts != nil {
		thinking.IncludeThoughts = *p.cfg.includeThoughts
	}
	if p.cfg.thinkingBudget != nil {
		thinking.ThinkingBudget = p.cfg.thinkingBudget
	}
	config := &genai.GenerateContentConfig{
		//nolint:gosec // maxTokens is a small provider-controlled output cap.
		MaxOutputTokens: int32(maxTokens),
		ThinkingConfig:  thinking,
	}
	if req.SystemPrompt != "" {
		config.SystemInstruction = &genai.Content{Parts: []*genai.Part{{Text: req.SystemPrompt}}}
	}
	// Temperature: explicit per-request wins, else the spec arg default.
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		config.Temperature = &t
	} else if p.cfg.temperature != nil {
		config.Temperature = p.cfg.temperature
	}
	if len(req.Tools) > 0 {
		config.Tools = convertTools(req.Tools)
	}
	return contents, config, nil
}

func convertTools(tools []llm.ToolDef) []*genai.Tool {
	decls := make([]*genai.FunctionDeclaration, 0, len(tools))
	for _, t := range tools {
		decl := &genai.FunctionDeclaration{Name: t.Name, Description: t.Description}
		if len(t.Schema) > 0 {
			// Pass the raw JSON Schema straight through (mutually exclusive with
			// Parameters); genai forwards it as parametersJsonSchema.
			var schema any
			if err := json.Unmarshal(t.Schema, &schema); err == nil {
				decl.ParametersJsonSchema = schema
			}
		}
		decls = append(decls, decl)
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}
}

func convertMessages(msgs []llm.Message) ([]*genai.Content, error) {
	var out []*genai.Content
	// Gemini's FunctionResponse requires the function NAME, but tool-result
	// messages only carry the call ID; track id->name from assistant turns.
	names := map[string]string{}
	for i := 0; i < len(msgs); i++ {
		m := msgs[i]
		switch m.Role {
		case llm.RoleSystem:
			// Gemini has no mid-conversation system role → fold into a user turn.
			out = append(out, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: m.Content}}})
		case llm.RoleUser:
			var parts []*genai.Part
			if m.Content != "" {
				parts = append(parts, &genai.Part{Text: m.Content})
			}
			parts = appendImageParts(parts, m.Attachments)
			if len(parts) == 0 {
				parts = append(parts, &genai.Part{Text: ""})
			}
			out = append(out, &genai.Content{Role: "user", Parts: parts})
		case llm.RoleAssistant:
			sigs := decodeSigs(m.ProviderExtra)
			var parts []*genai.Part
			if m.Content != "" {
				parts = append(parts, &genai.Part{Text: m.Content})
			}
			for i, tc := range m.ToolCalls {
				names[tc.ID] = tc.Name
				args, err := argsToMap(tc.Arguments)
				if err != nil {
					return nil, fmt.Errorf("gemini: tool call %q args: %w", tc.Name, err)
				}
				part := &genai.Part{FunctionCall: &genai.FunctionCall{ID: tc.ID, Name: tc.Name, Args: args}}
				if i < len(sigs) && sigs[i] != "" {
					if raw, err := base64.StdEncoding.DecodeString(sigs[i]); err == nil {
						part.ThoughtSignature = raw
					}
				}
				parts = append(parts, part)
			}
			out = append(out, &genai.Content{Role: "model", Parts: parts})
		case llm.RoleToolResult:
			// Gemini requires a function-call turn with N FunctionCall parts to be
			// answered by a SINGLE following user turn carrying all N
			// FunctionResponse parts (it validates the count against "the function
			// call turn"). The harness records each tool result as its own
			// message, so coalesce the whole run of consecutive tool results into
			// one user Content rather than emitting one Content per result.
			var parts []*genai.Part
			for ; i < len(msgs) && msgs[i].Role == llm.RoleToolResult; i++ {
				tr := msgs[i]
				payload := map[string]any{"output": tr.Content}
				if tr.IsError {
					payload = map[string]any{"error": tr.Content}
				}
				parts = append(parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
					ID:       tr.ToolCallID,
					Name:     names[tr.ToolCallID],
					Response: payload,
				}})
				parts = appendImageParts(parts, tr.Attachments)
			}
			i-- // the for loop's i++ will re-advance past the last result
			out = append(out, &genai.Content{Role: "user", Parts: parts})
		}
	}
	return out, nil
}

func appendImageParts(parts []*genai.Part, atts []llm.Attachment) []*genai.Part {
	for _, att := range atts {
		data, err := base64.StdEncoding.DecodeString(att.Base64Data)
		if err != nil {
			continue
		}
		parts = append(parts, &genai.Part{InlineData: &genai.Blob{MIMEType: att.MimeType, Data: data}})
	}
	return parts
}

func argsToMap(s string) (map[string]any, error) {
	if strings.TrimSpace(s) == "" {
		return map[string]any{}, nil
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil, err
	}
	return m, nil
}

// decodeSigs reads per-tool-call thought signatures from ProviderExtra,
// tolerating both the in-memory []string and the []any that survives a DB JSON
// round-trip (json normalizes both).
func decodeSigs(extra map[string]any) []string {
	if extra == nil {
		return nil
	}
	v, ok := extra[sigKey]
	if !ok {
		return nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var sigs []string
	if err := json.Unmarshal(raw, &sigs); err != nil {
		return nil
	}
	return sigs
}

func synthID() string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return "call_" + hex.EncodeToString(b[:])
}

// --- response accumulation (shared by Call and streaming) ---

type respAcc struct {
	content  strings.Builder
	thoughts strings.Builder
	calls    []llm.ToolCall
	sigs     []string // base64 per call, index-aligned to calls
	usage    *genai.GenerateContentResponseUsageMetadata
	finish   genai.FinishReason
	anySig   bool
}

func (a *respAcc) add(resp *genai.GenerateContentResponse) {
	if resp == nil {
		return
	}
	if resp.UsageMetadata != nil {
		a.usage = resp.UsageMetadata
	}
	if len(resp.Candidates) == 0 {
		return
	}
	cand := resp.Candidates[0]
	if cand.FinishReason != "" {
		a.finish = cand.FinishReason
	}
	if cand.Content == nil {
		return
	}
	for _, part := range cand.Content.Parts {
		switch {
		case part.FunctionCall != nil:
			fc := part.FunctionCall
			id := fc.ID
			if id == "" {
				id = synthID()
			}
			args, _ := json.Marshal(fc.Args)
			a.calls = append(a.calls, llm.ToolCall{ID: id, Name: fc.Name, Arguments: string(args)})
			sig := ""
			if len(part.ThoughtSignature) > 0 {
				sig = base64.StdEncoding.EncodeToString(part.ThoughtSignature)
				a.anySig = true
			}
			a.sigs = append(a.sigs, sig)
		case part.Thought:
			a.thoughts.WriteString(part.Text)
		case part.Text != "":
			a.content.WriteString(part.Text)
		}
	}
}

func (a *respAcc) response() *llm.Response {
	r := &llm.Response{
		Content:    a.content.String(),
		Thoughts:   a.thoughts.String(),
		ToolCalls:  a.calls,
		StopReason: string(a.finish),
	}
	if a.usage != nil {
		r.Usage = llm.Usage{
			PromptTokens:     int(a.usage.PromptTokenCount),
			CompletionTokens: int(a.usage.CandidatesTokenCount),
			TotalTokens:      int(a.usage.TotalTokenCount),
			CacheReadTokens:  int(a.usage.CachedContentTokenCount),
		}
	}
	if a.anySig {
		r.ProviderExtra = map[string]any{sigKey: a.sigs}
	}
	return r
}

// --- streaming ---

type geminiStream struct {
	next func() (*genai.GenerateContentResponse, error, bool)
	stop func()
	cur  llm.StreamEvent
	acc  respAcc
	err  error
	// first is the chunk already pulled during stream establishment (so an
	// overload before the first token could be retried). The initial Next()
	// consumes it instead of pulling again; hasFirst gates that one-shot.
	first    *genai.GenerateContentResponse
	hasFirst bool
}

func (s *geminiStream) Next() bool {
	var resp *genai.GenerateContentResponse
	var err error
	var ok bool
	if s.hasFirst {
		resp, ok, s.hasFirst = s.first, true, false
		s.first = nil
	} else {
		resp, err, ok = s.next()
	}
	if !ok {
		return false
	}
	if err != nil {
		s.err = err
		return false
	}
	s.cur = llm.StreamEvent{}
	prevCalls := len(s.acc.calls)
	s.acc.add(resp)
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return true
	}
	var text, thoughts strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		switch {
		case part.FunctionCall != nil:
			// accumulated above; surfaced below as a preview start event
		case part.Thought:
			thoughts.WriteString(part.Text)
		case part.Text != "":
			text.WriteString(part.Text)
		}
	}
	s.cur.DeltaText = text.String()
	s.cur.DeltaThoughts = thoughts.String()
	// Preview the first newly-seen tool call (authoritative calls come from
	// Response()).
	if len(s.acc.calls) > prevCalls {
		tc := s.acc.calls[prevCalls]
		s.cur.ToolCallStart = &llm.ToolCallStart{ID: tc.ID, Name: tc.Name}
		s.cur.ToolCallDelta = &llm.ToolCallDelta{ID: tc.ID, ArgumentsDelta: tc.Arguments}
	}
	return true
}

func (s *geminiStream) Event() llm.StreamEvent  { return s.cur }
func (s *geminiStream) Response() *llm.Response { return s.acc.response() }
func (s *geminiStream) Err() error              { return s.err }
func (s *geminiStream) Close()                  { s.stop() }

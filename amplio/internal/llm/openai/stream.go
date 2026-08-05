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
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"amplio/internal/llm"
)

// chatChunk is one streamed frame. Everything is optional: real servers send
// frames with no choices (a usage-only tail), no delta, or no finish_reason.
type chatChunk struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Delta        struct {
			Content   content        `json:"content"`
			ToolCalls []respToolCall `json:"tool_calls"`
			reasoningFields
			Extra map[string]json.RawMessage `json:"provider_specific_fields"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *apiUsage `json:"usage"`
}

// accumulator folds streamed frames into a final response.
//
// The two dialects observed in the wild differ sharply here, and both must work:
//
//   - "OpenAI-style": id + function.name arrive ONLY in the first delta for a
//     given index, arguments are then split across frames at arbitrary byte
//     boundaries — mid-JSON-token, even mid-word ({"city": "P + aris"}) — and
//     empty-string argument deltas are interleaved.
//   - "whole-call": the entire call, arguments included, arrives in one delta.
//
// Hence: never parse a delta's arguments as JSON (only the concatenation is
// valid), and key state by `index` rather than by arrival order.
type accumulator struct {
	text     strings.Builder
	thoughts strings.Builder
	// byIndex maps a tool call's stream index to its slot in `calls`. `index` is
	// optional in the wire format; absent means 0.
	byIndex    map[int]int
	calls      []llm.ToolCall
	args       []*strings.Builder
	stop       string
	usage      llm.Usage
	hasUsage   bool
	extra      map[string]json.RawMessage
	captureExt bool
}

func newAccumulator(captureExtras bool) *accumulator {
	return &accumulator{byIndex: map[int]int{}, captureExt: captureExtras}
}

// add folds one frame in and reports the incremental events it produced (for
// the live UI stream).
func (a *accumulator) add(c *chatChunk) []llm.StreamEvent {
	var events []llm.StreamEvent
	if c.Usage != nil {
		a.usage = c.Usage.to()
		a.hasUsage = true
	}
	for _, ch := range c.Choices {
		if ch.FinishReason != "" {
			a.stop = ch.FinishReason
		}
		d := ch.Delta
		if d.Content.text != "" {
			a.text.WriteString(d.Content.text)
			events = append(events, llm.StreamEvent{DeltaText: d.Content.text})
		}
		if r := d.text(); r != "" {
			a.thoughts.WriteString(r)
			events = append(events, llm.StreamEvent{DeltaThoughts: r})
		}
		if a.captureExt && len(d.Extra) > 0 {
			if a.extra == nil {
				a.extra = map[string]json.RawMessage{}
			}
			for k, v := range d.Extra {
				a.extra[k] = v
			}
		}
		for _, tc := range d.ToolCalls {
			events = append(events, a.addToolCall(tc)...)
		}
	}
	return events
}

func (a *accumulator) addToolCall(tc respToolCall) []llm.StreamEvent {
	idx := 0
	if tc.Index != nil {
		idx = *tc.Index
	}
	slot, seen := a.byIndex[idx]
	if !seen {
		slot = len(a.calls)
		a.byIndex[idx] = slot
		a.calls = append(a.calls, llm.ToolCall{ID: toolCallID(tc.ID, idx), Name: tc.Function.Name})
		a.args = append(a.args, &strings.Builder{})
	}
	// A later frame may still be the one carrying id/name (servers vary on
	// whether the first frame for an index has them), so fill any gap.
	if a.calls[slot].Name == "" && tc.Function.Name != "" {
		a.calls[slot].Name = tc.Function.Name
	}
	if tc.ID != "" {
		a.calls[slot].ID = tc.ID
	}

	var events []llm.StreamEvent
	if !seen {
		events = append(events, llm.StreamEvent{
			ToolCallStart: &llm.ToolCallStart{ID: a.calls[slot].ID, Name: a.calls[slot].Name},
		})
	}
	if frag := tc.Function.Arguments; frag != "" {
		a.args[slot].WriteString(frag)
		events = append(events, llm.StreamEvent{
			ToolCallDelta: &llm.ToolCallDelta{ID: a.calls[slot].ID, ArgumentsDelta: frag},
		})
	}
	return events
}

func (a *accumulator) response() *llm.Response {
	out := &llm.Response{
		Content:    a.text.String(),
		Thoughts:   a.thoughts.String(),
		StopReason: a.stop,
		Usage:      a.usage,
	}
	for i := range a.calls {
		c := a.calls[i]
		c.Arguments = a.args[i].String()
		out.ToolCalls = append(out.ToolCalls, c)
	}
	if len(a.extra) > 0 {
		if blob, err := json.Marshal(a.extra); err == nil {
			out.ProviderExtra = map[string]any{"openai.provider_specific_fields": string(blob)}
		}
	}
	return out
}

// --- SSE -------------------------------------------------------------------

// sseFrames yields the payload of each `data:` line, skipping blanks, comment
// keep-alives (": ping") and any other field (event:, id:, retry:). It stops at
// the [DONE] sentinel — whose absence is also fine, since EOF ends the stream.
type sseReader struct {
	sc   *bufio.Scanner
	done bool
}

func newSSEReader(r io.Reader) *sseReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), sseMaxLine)
	return &sseReader{sc: sc}
}

// next returns the next data payload, or ok=false at end of stream.
func (s *sseReader) next() (payload string, ok bool) {
	if s.done {
		return "", false
	}
	for s.sc.Scan() {
		line := strings.TrimRight(s.sc.Text(), "\r")
		if line == "" || strings.HasPrefix(line, ":") {
			continue // blank separator or comment keep-alive
		}
		data, found := strings.CutPrefix(line, "data:")
		if !found {
			continue // event:/id:/retry: — nothing we need
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			s.done = true
			return "", false
		}
		if data == "" {
			continue
		}
		return data, true
	}
	s.done = true
	return "", false
}

func (s *sseReader) err() error { return s.sc.Err() }

// stream implements llm.Stream over an SSE body.
type stream struct {
	body    io.ReadCloser
	sse     *sseReader
	acc     *accumulator
	pending []llm.StreamEvent // events decoded from one frame, not yet handed out
	cur     llm.StreamEvent
	err     error
}

func (p *provider) Stream(ctx context.Context, req llm.Request) (llm.Stream, error) {
	resp, err := p.post(ctx, p.buildBody(req, true))
	if err != nil {
		return nil, err
	}
	// A server that ignores stream:true and answers with JSON would otherwise
	// look like an empty stream; fail loudly instead.
	if ct := resp.Header.Get("content-type"); ct != "" && !strings.Contains(ct, "event-stream") {
		defer resp.Body.Close()
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<10))
		return nil, fmt.Errorf("openai stream: server replied %s, not an SSE stream: %s",
			ct, strings.TrimSpace(string(snippet)))
	}
	return &stream{
		body: resp.Body,
		sse:  newSSEReader(resp.Body),
		acc:  newAccumulator(p.captureExtras),
	}, nil
}

func (s *stream) Next() bool {
	for {
		if len(s.pending) > 0 {
			s.cur, s.pending = s.pending[0], s.pending[1:]
			return true
		}
		payload, ok := s.sse.next()
		if !ok {
			s.err = s.sse.err()
			return false
		}
		var c chatChunk
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			// One malformed frame must not kill a long turn: skip it. (Seen from
			// proxies that inject their own status frames mid-stream.)
			continue
		}
		s.pending = s.acc.add(&c)
	}
}

func (s *stream) Event() llm.StreamEvent  { return s.cur }
func (s *stream) Response() *llm.Response { return s.acc.response() }
func (s *stream) Err() error              { return s.err }
func (s *stream) Close() {
	// Drain briefly so the connection can be reused, then close regardless.
	_, _ = io.Copy(io.Discard, io.LimitReader(s.body, 4<<10))
	_ = s.body.Close()
}

var _ llm.Stream = (*stream)(nil)

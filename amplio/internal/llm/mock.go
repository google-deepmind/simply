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
	"sync"
)

// MockProvider returns canned responses for testing. Configure the Responses
// slice before use; calls pop from the front. If Responses is empty, returns
// a default empty response.
//
// Safe for concurrent Call/Stream (agents share one provider, and tool loops
// can drive parallel calls): the request log and response cursor are guarded by
// mu. Read the recorded calls via Recorded() (also guarded) rather than the
// internal field, so assertions don't race a concurrent writer under -race.
type MockProvider struct {
	Model     string
	Responses []Response

	mu      sync.Mutex
	calls   []Request // records every Call/Stream request for assertions
	callIdx int
}

var _ Provider = (*MockProvider)(nil)

func (m *MockProvider) ModelID() string { return m.Model }
func (m *MockProvider) MaxTokens() int  { return 200000 }

func (m *MockProvider) Call(_ context.Context, req Request) (*Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls = append(m.calls, req)
	return m.nextResponseLocked(), nil
}

func (m *MockProvider) Stream(_ context.Context, req Request) (Stream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls = append(m.calls, req)
	return &mockStream{response: m.nextResponseLocked()}, nil
}

// nextResponseLocked pops the next canned response. Caller must hold m.mu.
func (m *MockProvider) nextResponseLocked() *Response {
	if m.callIdx < len(m.Responses) {
		r := m.Responses[m.callIdx]
		m.callIdx++
		return &r
	}
	return &Response{Content: "", StopReason: "end_turn"}
}

// Recorded returns a copy of the requests seen so far, safe to read while calls
// may still be in flight.
func (m *MockProvider) Recorded() []Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]Request(nil), m.calls...)
}

// CallCount returns how many Call/Stream invocations have been recorded.
func (m *MockProvider) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.calls)
}

// mockStream wraps a single Response as a one-shot stream.
type mockStream struct {
	response *Response
	done     bool
}

func (s *mockStream) Next() bool {
	if s.done {
		return false
	}
	s.done = true
	return true
}

func (s *mockStream) Event() StreamEvent {
	return StreamEvent{DeltaText: s.response.Content}
}

func (s *mockStream) Response() *Response { return s.response }
func (s *mockStream) Err() error          { return nil }
func (s *mockStream) Close()              {}

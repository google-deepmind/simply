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

package eventloop

import (
	"context"
	"testing"

	"amplio/internal/agent"
	"amplio/internal/llm"
)

type capturingBroadcaster struct{ texts []string }

func (c *capturingBroadcaster) Chunk(_, _ string, _ int, text, _ string) {
	if text != "" {
		c.texts = append(c.texts, text)
	}
}

func (c *capturingBroadcaster) WorkspaceAlias(_ string, _ int, _ string) {}
func (c *capturingBroadcaster) SysStat(_ map[string]any)                 {}

func TestCallLLM_StreamsWhenInteractive(t *testing.T) {
	b := &capturingBroadcaster{}
	a := &EventLoopAgent{
		env: &agent.Env{
			Broadcaster: b,
			RunID:       "r",
			LLM:         &llm.MockProvider{Responses: []llm.Response{{Content: "hello world"}}},
		},
		cfg: Config{Interactive: true, SessionID: "s"},
	}
	resp, err := a.callLLM(context.Background(), llm.Request{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hello world" {
		t.Errorf("content = %q", resp.Content)
	}
	if len(b.texts) == 0 {
		t.Error("interactive call did not emit any stream chunks")
	}
}

func TestCallLLM_BlockingWhenNotInteractive(t *testing.T) {
	b := &capturingBroadcaster{}
	a := &EventLoopAgent{
		env: &agent.Env{
			Broadcaster: b,
			RunID:       "r",
			LLM:         &llm.MockProvider{Responses: []llm.Response{{Content: "x"}}},
		},
		cfg: Config{Interactive: false, SessionID: "s"}, // autonomous → no streaming
	}
	resp, err := a.callLLM(context.Background(), llm.Request{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "x" {
		t.Errorf("content = %q", resp.Content)
	}
	if len(b.texts) != 0 {
		t.Errorf("non-interactive emitted %d chunks, want 0", len(b.texts))
	}
}

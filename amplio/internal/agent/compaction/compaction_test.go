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

package compaction

import (
	"context"
	"strings"
	"testing"
	"time"

	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/event"
	"amplio/internal/llm"
)

// hqMock returns a fixed summary on its first (no-tool) turn, ending the loop.
type hqMock struct {
	out      string
	gotTask  string
	captured bool
}

func (m *hqMock) Call(_ context.Context, req llm.Request) (*llm.Response, error) {
	if !m.captured {
		for _, msg := range req.Messages {
			if msg.Role == llm.RoleUser {
				m.gotTask = msg.Content
			}
		}
		m.captured = true
	}
	return &llm.Response{Content: m.out}, nil
}
func (*hqMock) Stream(context.Context, llm.Request) (llm.Stream, error) { return nil, nil }
func (*hqMock) ModelID() string                                         { return "mock-hq" }
func (*hqMock) MaxTokens() int                                          { return 1000 }

func seed(t *testing.T) (db.Store, string) {
	t.Helper()
	ctx := context.Background()
	s, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	runID := db.NewRunID()
	if err := s.CreateRun(ctx, db.RunRecord{RunID: runID, CreatedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing, CreatedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "SYS"})
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "TASK: investigate Y"})
	_, _ = s.AdvanceStep(ctx, runID, "s1")
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "did step 1"})
	_, _ = s.AdvanceStep(ctx, runID, "s1")
	_, _ = s.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "did step 2"})
	return s, runID
}

func TestCompact_ReturnsSummaryWithCuratedView(t *testing.T) {
	s, runID := seed(t)
	hq := &hqMock{out: "  I summarized the work.  "}

	out, err := Compact(context.Background(), Deps{Store: s, HQ: hq, RunID: runID, SessionID: "s1"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	// The LLM prose is trimmed and leads the output.
	if !strings.HasPrefix(out, "I summarized the work.") {
		t.Errorf("summary should start with the trimmed prose, got %q", out)
	}
	// The prose is augmented with the deterministic recent-activity trace: the
	// open phase (steps 1-2, unphased) renders under the header.
	if !strings.Contains(out, "RECENT ACTIVITY") || !strings.Contains(out, "did step 2") {
		t.Errorf("summary missing the structured recent-activity trace:\n%s", out)
	}
	// The summarizer must have been seeded with the curated view (the task).
	if !strings.Contains(hq.gotTask, "TASK: investigate Y") {
		t.Errorf("curated view not seeded into task prompt:\n%s", hq.gotTask)
	}
}

func TestCompact_EmptySummaryIsError(t *testing.T) {
	s, runID := seed(t)
	hq := &hqMock{out: "   "}
	if _, err := Compact(context.Background(), Deps{Store: s, HQ: hq, RunID: runID, SessionID: "s1"}, 2); err == nil {
		t.Fatal("expected an error for an empty summary")
	}
}

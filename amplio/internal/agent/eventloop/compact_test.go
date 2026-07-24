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
	"errors"
	"strings"
	"testing"

	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/llm"
)

// The compaction "treat this as your memory, continue" instruction is applied at
// LLM projection time (eventToMessage), NOT stored on the event. So it must
// appear in the projected LLM message but NOT in the event's canonical ToText
// (which the observer / inspect tools / session_search render).
func TestCompactionFraming_AtProjectionOnly(t *testing.T) {
	evt := &event.CompactionEvent{Content: "summary body"}

	text := evt.ToText()
	if strings.Contains(text, "ALREADY COMPLETED") {
		t.Errorf("ToText should not carry the LLM continuation framing:\n%s", text)
	}
	if !strings.Contains(text, "summary body") {
		t.Errorf("ToText should carry the summary content:\n%s", text)
	}

	var a EventLoopAgent
	msg := a.eventToMessage(evt)
	// Projected as a USER message (not system): a system-role projection would
	// be folded into buildMessages' leading system cluster, yielding zero
	// messages when the compaction boundary is at the conversation tip.
	if msg == nil || msg.Role != llm.RoleUser {
		t.Fatalf("compaction should project to a user message, got %+v", msg)
	}
	if !strings.Contains(msg.Content, "ALREADY COMPLETED") {
		t.Errorf("projected LLM message should carry the continuation framing:\n%s", msg.Content)
	}
	if !strings.Contains(msg.Content, "summary body") {
		t.Errorf("projected LLM message should still carry the summary content:\n%s", msg.Content)
	}
}

func TestIsYes(t *testing.T) {
	cases := map[string]bool{
		"YES":                      true,
		"yes":                      true,
		"  Yes, it is too long":    true,
		"NO":                       false,
		"no, this is a rate limit": false,
		"the context is too long":  false, // must lead with yes
		"":                         false,
	}
	for in, want := range cases {
		if got := isYes(in); got != want {
			t.Errorf("isYes(%q) = %v, want %v", in, got, want)
		}
	}
}

// seedForCompact builds a session at current_step 3 with bootstrap, two finished
// turns, and a fresh step-3 input (the boundary is callStep-1 = 2).
func seedForCompact(t *testing.T) (db.Store, string) {
	t.Helper()
	store := testStore(t)
	ctx := context.Background()
	runID := db.NewRunID()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: runID}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "s1", Status: db.SessionOngoing}); err != nil {
		t.Fatal(err)
	}
	_, _ = store.AppendEvent(ctx, runID, "s1", &event.SystemEvent{Content: "SYS"})
	_, _ = store.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "TASK"})
	_, _ = store.AdvanceStep(ctx, runID, "s1")
	_, _ = store.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "turn 1"})
	_, _ = store.AdvanceStep(ctx, runID, "s1")
	_, _ = store.AppendEvent(ctx, runID, "s1", &event.AssistantEvent{Content: "turn 2"})
	_, _ = store.AdvanceStep(ctx, runID, "s1")
	_, _ = store.AppendEvent(ctx, runID, "s1", &event.UserEvent{Content: "fresh input"})
	return store, runID
}

func gen(t *testing.T, store db.Store, runID string) int {
	t.Helper()
	s, err := store.GetSession(context.Background(), runID, "s1")
	if err != nil {
		t.Fatal(err)
	}
	return s.CurrentGeneration
}

func TestTryCompact_CompactsWhenJudgeYes(t *testing.T) {
	store, runID := seedForCompact(t)
	fast := &llm.MockProvider{Responses: []llm.Response{{Content: "YES"}}}
	hq := &llm.MockProvider{Responses: []llm.Response{{Content: "cumulative memory of the work"}}}
	a := newT(testCfg{RunID: runID, SessionID: "s1", Store: store, SystemFast: fast, SystemHQ: hq})

	if !a.tryCompact(context.Background(), errors.New("prompt is too long: 200000 tokens > 190000 maximum"), 3) {
		t.Fatal("tryCompact = false, want true")
	}
	if g := gen(t, store, runID); g != 1 {
		t.Fatalf("generation = %d, want 1 (compaction should have bumped it)", g)
	}
	// New context = bootstrap + compaction summary + carried fresh input.
	cur, _ := store.GetEvents(context.Background(), runID, "s1", db.EventFilter{CurrentContextOnly: true})
	var hasComp, hasFresh bool
	for _, e := range cur {
		switch ev := e.Event.(type) {
		case *event.CompactionEvent:
			hasComp = true
		case *event.UserEvent:
			if ev.Content == "fresh input" {
				hasFresh = true
			}
		}
	}
	if !hasComp || !hasFresh {
		t.Errorf("post-compaction context missing pieces: comp=%v fresh=%v", hasComp, hasFresh)
	}
}

func TestTryCompact_JudgeSaysNo(t *testing.T) {
	store, runID := seedForCompact(t)
	fast := &llm.MockProvider{Responses: []llm.Response{{Content: "NO"}}}
	hq := &llm.MockProvider{Responses: []llm.Response{{Content: "should not be used"}}}
	a := newT(testCfg{RunID: runID, SessionID: "s1", Store: store, SystemFast: fast, SystemHQ: hq})

	if a.tryCompact(context.Background(), errors.New("rate limit exceeded"), 3) {
		t.Fatal("tryCompact = true, want false (non-context error)")
	}
	if g := gen(t, store, runID); g != 0 {
		t.Fatalf("generation = %d, want 0 (no compaction)", g)
	}
	if hq.CallCount() != 0 {
		t.Errorf("HQ summarizer called %d times, want 0", hq.CallCount())
	}
}

func TestTryCompact_NoProvidersIsNoop(t *testing.T) {
	store, runID := seedForCompact(t)
	a := newT(testCfg{RunID: runID, SessionID: "s1", Store: store})
	if a.tryCompact(context.Background(), errors.New("prompt is too long"), 3) {
		t.Fatal("tryCompact = true with no tiers, want false")
	}
}

func TestTryCompact_BoundaryTooLowSkipsJudge(t *testing.T) {
	store, runID := seedForCompact(t)
	fast := &llm.MockProvider{Responses: []llm.Response{{Content: "YES"}}}
	hq := &llm.MockProvider{Responses: []llm.Response{{Content: "x"}}}
	a := newT(testCfg{RunID: runID, SessionID: "s1", Store: store, SystemFast: fast, SystemHQ: hq})

	// callStep 1 -> boundary 0: nothing but bootstrap precedes it; don't even judge.
	if a.tryCompact(context.Background(), errors.New("prompt is too long"), 1) {
		t.Fatal("tryCompact = true at boundary 0, want false")
	}
	if fast.CallCount() != 0 {
		t.Errorf("judge called %d times at boundary 0, want 0", fast.CallCount())
	}
}

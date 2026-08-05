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
	"strings"
	"testing"
	"time"

	"amplio/internal/agent"
	"amplio/internal/blob"
	"amplio/internal/cli"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/event"
	"amplio/internal/eventstream"
	"amplio/internal/llm"
	"amplio/internal/runtime"
	"amplio/internal/session"
	"amplio/internal/tool"
	"amplio/internal/tool/bash"
	"amplio/internal/workspace"
	"amplio/internal/workspace/plain"
)

// testCfg is a flat view over the per-run env and per-agent config, kept so
// tests can specify both in one literal (mirroring the pre-split Config). newT
// splits it into the *agent.Env + eventloop.Config that New now takes.
type testCfg struct {
	// env-derived (shared per run)
	RunID       string
	Store       db.Store
	LLM         llm.Provider
	Registry    *session.Registry
	Workspace   workspace.Workspace
	Broadcaster eventstream.Broadcaster
	SystemFast  llm.Provider
	SystemHQ    llm.Provider
	// per-agent-instance config
	SessionID     string
	Task          string
	ParentID      string
	FirstMessage  string
	Handle        *session.Handle
	AgentType     string
	Interactive   bool
	IdleTimeout   time.Duration
	CLITools      []cli.Tool
	SystemPrompt  string
	Tools         []*tool.Tool
	InitialRecall func(ctx context.Context, task string) string
	BlobStore     *blob.Store
}

func newT(c testCfg) *EventLoopAgent {
	env := &agent.Env{
		RunID:       c.RunID,
		Store:       c.Store,
		LLM:         c.LLM,
		Registry:    c.Registry,
		Workspace:   c.Workspace,
		Broadcaster: c.Broadcaster,
		SystemFast:  c.SystemFast,
		SystemHQ:    c.SystemHQ,
	}
	return New(env, Config{
		SessionID:     c.SessionID,
		Task:          c.Task,
		ParentID:      c.ParentID,
		FirstMessage:  c.FirstMessage,
		Handle:        c.Handle,
		AgentType:     c.AgentType,
		Interactive:   c.Interactive,
		IdleTimeout:   c.IdleTimeout,
		CLITools:      c.CLITools,
		SystemPrompt:  c.SystemPrompt,
		Tools:         c.Tools,
		InitialRecall: c.InitialRecall,
		BlobStore:     c.BlobStore,
	})
}

func testStore(t *testing.T) db.Store {
	t.Helper()
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() }) //nolint:errcheck
	return store
}

// testSetup creates a store, run registry, and per-run session registry
// with the notifier wired up. An optional RunConfig is persisted on the run
// record (e.g. to exercise AGENTS.md, which the eventloop reads from there).
func testSetup(t *testing.T, runCfg ...config.RunConfig) (db.Store, string, *session.Registry) {
	t.Helper()
	store := testStore(t)
	ctx := context.Background()
	runID := db.NewRunID()
	var cfg config.RunConfig
	if len(runCfg) > 0 {
		cfg = runCfg[0]
	}
	_ = store.CreateRun(ctx, db.RunRecord{RunID: runID, Config: cfg})

	runReg := runtime.NewRunRegistry()
	registry := runReg.GetOrCreate(runID)
	store.SetCommitListener(runtime.NewCommitNotifier(runReg, nil, nil))

	return store, runID, registry
}

// waitForIdle polls until the session reaches idle status or fails.
func waitForIdle(t *testing.T, store db.Store, runID, sessionID string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if s, _ := store.GetSession(context.Background(), runID, sessionID); s != nil && s.Status == db.SessionIdle {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	s, _ := store.GetSession(context.Background(), runID, sessionID)
	got := "<nil>"
	if s != nil {
		got = s.Status
	}
	t.Fatalf("session %s did not reach idle (got %q)", sessionID, got)
}

// An autonomous agent that finishes a task ends a bare no-tool turn → concluded.
func TestEventLoop_WorkThenConclude(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "The answer is 42.", StopReason: "end_turn"}},
	}

	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "main-agent",
		Task:         "What is the answer?",
		SystemPrompt: "You are a helpful agent.",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{},
		Workspace:    plain.New("/tmp"),
	})

	// Autonomous: Run returns when it concludes.
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	sess, err := store.GetSession(ctx, runID, "main-agent")
	if err != nil {
		t.Fatal(err)
	}
	if sess.Status != db.SessionConcluded {
		t.Errorf("status: %q, want concluded", sess.Status)
	}

	// Step model: bootstrap at step 0, AssistantEvent at step 1.
	events, _ := store.GetEvents(ctx, runID, "main-agent", db.EventFilter{})
	var bootstrapEvents, assistantEvents int
	for _, e := range events {
		if e.Step == 0 {
			bootstrapEvents++
		}
		if e.Event.EventType() == "assistant" {
			assistantEvents++
			if e.Step != 1 {
				t.Errorf("AssistantEvent at step %d, want step 1", e.Step)
			}
		}
	}
	if bootstrapEvents < 3 {
		t.Errorf("bootstrap events: %d, want >= 3", bootstrapEvents)
	}
	if assistantEvents != 1 {
		t.Errorf("assistant events: %d, want 1", assistantEvents)
	}
}

// An autonomous agent that ends a turn with NO tool calls AND empty content is
// nudged back into the loop (a likely-accidental conclusion) instead of
// concluding on nothing; a substantive next turn then concludes normally.
func TestEventLoop_EmptyConclusionNudged(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	mock := &llm.MockProvider{
		Model: "test-model",
		Responses: []llm.Response{
			{Content: "", StopReason: "end_turn"},                        // accidental: no tools, no text
			{Content: "Done: the answer is 42.", StopReason: "end_turn"}, // real conclusion
		},
	}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "q",
		SystemPrompt: "sp", LLM: mock, Store: store, Registry: registry,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if mock.CallCount() != 2 {
		t.Errorf("call count = %d, want 2 (one nudge, then conclude)", mock.CallCount())
	}
	if n := countNudgeEvents(t, store, runID); n != 1 {
		t.Errorf("nudge events = %d, want 1", n)
	}
	sess, _ := store.GetSession(ctx, runID, "main-agent")
	if sess.Status != db.SessionConcluded {
		t.Errorf("status = %q, want concluded", sess.Status)
	}
}

// The nudge is only useful if the model actually SEES it. Asserts the shape of
// the follow-up request: the empty assistant turn is replayed as a non-empty
// placeholder (providers reject empty assistant messages) and the nudge follows
// it as a user turn.
func TestEventLoop_NudgeReachesTheModel(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	mock := &llm.MockProvider{
		Model: "test-model",
		Responses: []llm.Response{
			{Content: "", StopReason: "end_turn"},
			{Content: "Done.", StopReason: "end_turn"},
		},
	}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "q",
		SystemPrompt: "sp", LLM: mock, Store: store, Registry: registry,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}
	recorded := mock.Recorded()
	if len(recorded) != 2 {
		t.Fatalf("recorded calls = %d, want 2", len(recorded))
	}
	second := recorded[1].Messages
	var sawPlaceholder, sawNudge bool
	for _, m := range second {
		if m.Role == llm.RoleAssistant && m.Content == emptyAssistantPlaceholder {
			sawPlaceholder = true
		}
		if m.Role == llm.RoleUser && m.Content == concludeNudgeText {
			sawNudge = true
		}
		if m.Role == llm.RoleAssistant && m.Content == "" {
			t.Errorf("an EMPTY assistant message reached the provider; many reject it")
		}
	}
	if !sawPlaceholder {
		t.Errorf("follow-up request lacks the %q placeholder for the empty turn", emptyAssistantPlaceholder)
	}
	if !sawNudge {
		t.Errorf("follow-up request lacks the nudge; the model can't act on advice it never receives")
	}
	// The nudge must be the LAST message: providers require the turn to end on a
	// user/tool-result message, and it reads as the instruction being answered.
	if last := second[len(second)-1]; last.Role != llm.RoleUser || last.Content != concludeNudgeText {
		t.Errorf("last message = %v/%q, want the nudge as a trailing user turn", last.Role, last.Content)
	}
}

// Regression, from a real run: after one empty turn the placeholder we project
// into the context becomes an exemplar, and the model answers with EXACTLY that
// string. It is not empty, so the naive guard accepted it and the agent
// "concluded" with the text "(empty response)". An echo must count as empty.
func TestEventLoop_PlaceholderEchoIsNudged(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	mock := &llm.MockProvider{
		Model: "test-model",
		Responses: []llm.Response{
			{Content: emptyAssistantPlaceholder, StopReason: "end_turn"}, // parroted, not authored
			{Content: "The real answer.", StopReason: "end_turn"},
		},
	}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "q",
		SystemPrompt: "sp", LLM: mock, Store: store, Registry: registry,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if n := countNudgeEvents(t, store, runID); n != 1 {
		t.Errorf("nudge events = %d, want 1 (the echo must not pass as a conclusion)", n)
	}
	events, err := store.GetEvents(ctx, runID, "main-agent", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	last := ""
	for _, rec := range events {
		if a, ok := rec.Event.(*event.AssistantEvent); ok {
			last = a.Content
		}
	}
	if last != "The real answer." {
		t.Errorf("final content = %q, want the run to end on real text", last)
	}
}

// A crash between the empty turn and its nudge leaves a no-tool AssistantEvent
// as the stream tail. On resume the at-rest path concludes without re-calling
// the LLM — which must NOT silently bypass the accidental-conclusion guard, or
// the guard is only as reliable as the process staying alive.
func TestEventLoop_ResumeAtRestEmptyIsNudged(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	// Simulate the crashed predecessor: a session whose tail is an EMPTY no-tool
	// assistant turn, still marked ongoing (status never flipped).
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionOngoing,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AdvanceStep(ctx, runID, "main-agent"); err != nil {
		t.Fatal(err)
	}
	if err := store.FinalizeStep(ctx, runID, "main-agent", 1, []event.Event{
		&event.UserEvent{Content: "do the thing"},
		&event.AssistantEvent{Content: "", StopReason: "end_turn"},
	}); err != nil {
		t.Fatal(err)
	}

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "Actually, here is the result.", StopReason: "end_turn"}},
	}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "q",
		SystemPrompt: "sp", LLM: mock, Store: store, Registry: registry,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if mock.CallCount() == 0 {
		t.Fatalf("the LLM was never called: the resume concluded on the empty turn, bypassing the nudge")
	}
	if n := countNudgeEvents(t, store, runID); n != 1 {
		t.Errorf("nudge events = %d, want 1 on the resume path", n)
	}
	sess, _ := store.GetSession(ctx, runID, "main-agent")
	if sess.Status != db.SessionConcluded {
		t.Fatalf("status = %q, want concluded", sess.Status)
	}
	// And the recovered run ends with real content rather than nothing.
	events, err := store.GetEvents(ctx, runID, "main-agent", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	last := ""
	for _, rec := range events {
		if a, ok := rec.Event.(*event.AssistantEvent); ok {
			last = a.Content
		}
	}
	if last != "Actually, here is the result." {
		t.Errorf("final assistant content = %q, want the post-nudge result", last)
	}
}

// The nudge is BOUNDED: a model that keeps emitting empty turns is nudged at
// most maxConcludeNudges times, then the empty conclusion is allowed through
// (proves no infinite loop).
func TestEventLoop_EmptyConclusionNudgeBounded(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	// No responses → MockProvider returns an empty turn on every call.
	mock := &llm.MockProvider{Model: "test-model"}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "q",
		SystemPrompt: "sp", LLM: mock, Store: store, Registry: registry,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if want := maxConcludeNudges + 1; mock.CallCount() != want {
		t.Errorf("call count = %d, want %d (maxConcludeNudges nudges then conclude)", mock.CallCount(), want)
	}
	if n := countNudgeEvents(t, store, runID); n != maxConcludeNudges {
		t.Errorf("nudge events = %d, want %d", n, maxConcludeNudges)
	}
	sess, _ := store.GetSession(ctx, runID, "main-agent")
	if sess.Status != db.SessionConcluded {
		t.Errorf("status = %q, want concluded", sess.Status)
	}
}

// A degenerate assistant turn (empty content, no tool calls) projects to a
// NON-EMPTY placeholder (Gemini rejects an empty model turn) while PRESERVING
// ProviderExtra, so thought signatures / thinking blocks still carry over. Turns
// with content or tool calls are projected unchanged.
func TestEventToMessage_EmptyAssistantPlaceholder(t *testing.T) {
	ag := newT(testCfg{SessionID: "main-agent", Workspace: plain.New("/tmp")})

	// Empty content + no tool calls → non-empty placeholder, ProviderExtra kept.
	msg := ag.eventToMessage(&event.AssistantEvent{
		Content:       "  ",
		Thoughts:      "pondered",
		ProviderExtra: map[string]any{"anthropic.thinking_blocks": []any{"sig"}},
	})
	if msg == nil {
		t.Fatal("empty assistant turn projected to nil (dropped); want a placeholder message")
	}
	if msg.Role != llm.RoleAssistant || strings.TrimSpace(msg.Content) == "" {
		t.Errorf("want non-empty assistant placeholder, got role=%q content=%q", msg.Role, msg.Content)
	}
	if msg.ProviderExtra["anthropic.thinking_blocks"] == nil {
		t.Errorf("ProviderExtra (thought signatures/thinking blocks) must be preserved on the placeholder turn, got %v", msg.ProviderExtra)
	}

	// A turn with real content is unchanged.
	if m := ag.eventToMessage(&event.AssistantEvent{Content: "hello"}); m == nil || m.Content != "hello" {
		t.Errorf("content turn should be preserved verbatim, got %+v", m)
	}
	// A turn with tool calls (empty content) is preserved, not placeholdered.
	m := ag.eventToMessage(&event.AssistantEvent{ToolCalls: []event.ToolCall{{ID: "1", Name: "bash", Arguments: "{}"}}})
	if m == nil || len(m.ToolCalls) != 1 || m.Content == emptyAssistantPlaceholder {
		t.Errorf("tool-call turn should be preserved, got %+v", m)
	}
}

func countNudgeEvents(t *testing.T, store db.Store, runID string) int {
	t.Helper()
	events, _ := store.GetEvents(context.Background(), runID, "main-agent", db.EventFilter{})
	n := 0
	for _, e := range events {
		if u, ok := e.Event.(*event.UserEvent); ok && u.Content == concludeNudgeText {
			n++
		}
	}
	return n
}

// bootstrap must persist the configured AgentType, not a hardcoded value, so the
// session can be respawned through the agent registry after a crash.
func TestEventLoop_BootstrapPersistsAgentType(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}},
	}

	ag := newT(testCfg{
		RunID:     runID,
		SessionID: "main-agent",
		AgentType: "standard_agent",
		Task:      "task",
		LLM:       mock,
		Store:     store,
		Registry:  registry,
		Tools:     []*tool.Tool{},
		Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	sess, err := store.GetSession(ctx, runID, "main-agent")
	if err != nil {
		t.Fatal(err)
	}
	if sess.AgentType != "standard_agent" {
		t.Errorf("session AgentType = %q, want %q", sess.AgentType, "standard_agent")
	}
}

// bootstrapSystemEvent fetches the step-0 SystemEvent with the given marker
// from the main-agent session, or nil if absent. Shared by the AGENTS.md
// tests below so the assertion pattern stays uniform.
func bootstrapSystemEvent(t *testing.T, store db.Store, runID, marker string) *event.SystemEvent {
	t.Helper()
	events, err := store.GetEvents(context.Background(), runID, "main-agent", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range events {
		if e.Step != 0 {
			continue
		}
		sysEv, ok := e.Event.(*event.SystemEvent)
		if !ok || sysEv.Marker != marker {
			continue
		}
		return sysEv
	}
	return nil
}

// runOneShot drives the agent through a single end_turn response so the
// bootstrap events get a chance to be persisted and the loop exits cleanly.
func runOneShot(t *testing.T, cfg testCfg) {
	t.Helper()
	cfg.LLM = &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}},
	}
	if cfg.Tools == nil {
		cfg.Tools = []*tool.Tool{}
	}
	if err := newT(cfg).Run(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// AgentsMD non-empty → exactly one step-0 SystemEvent with marker
// "agents_md" carries the content verbatim. The producer (server / CLI)
// is responsible for combining global + workspace sources; the eventloop
// just emits whatever it's given.
func TestEventLoop_Bootstrap_AgentsMDEmittedWhenSet(t *testing.T) {
	store, runID, registry := testSetup(t, config.RunConfig{
		AgentsMD: "## /path/AGENTS.md\n\nrule one\nrule two",
	})
	runOneShot(t, testCfg{
		RunID:     runID,
		SessionID: "main-agent",
		Task:      "task",
		Store:     store,
		Registry:  registry,
		Workspace: plain.New(t.TempDir()),
	})
	ev := bootstrapSystemEvent(t, store, runID, "agents_md")
	if ev == nil {
		t.Fatal("expected agents_md SystemEvent at step 0, got none")
	}
	if !strings.Contains(ev.Content, "rule one") || !strings.Contains(ev.Content, "rule two") {
		t.Errorf("agents_md event missing body: %q", ev.Content)
	}
}

// AgentsMD empty → no event emitted. Confirms the common path (no AGENTS.md
// files anywhere) pays no event-store cost.
func TestEventLoop_Bootstrap_AgentsMDSkippedWhenEmpty(t *testing.T) {
	store, runID, registry := testSetup(t)
	runOneShot(t, testCfg{
		RunID:     runID,
		SessionID: "main-agent",
		Task:      "task",
		Store:     store,
		Registry:  registry,
		Workspace: plain.New(t.TempDir()),
	})
	if got := bootstrapSystemEvent(t, store, runID, "agents_md"); got != nil {
		t.Errorf("unexpected agents_md event when AgentsMD is empty: %q", got.Content)
	}
}

// A chat run: empty task + FirstMessage → the agent answers the opening message
// at step 1 (no park), then parks idle for the next message.
func TestEventLoop_FirstMessageRunsImmediately(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "Hello! How can I help?", StopReason: "end_turn"}},
	}
	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "chatbot",
		AgentType:    "chatbot",
		Task:         "", // chat run: no task
		FirstMessage: "hi there",
		SystemPrompt: "You are a chatbot.",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{},
		Workspace:    plain.New("/tmp"),
		Interactive:  true,
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(ctx) }()

	waitForIdle(t, store, runID, "chatbot") // answered, then parked

	if mock.CallCount() != 1 {
		t.Errorf("LLM called %d times, want 1 (responded to opening message)", mock.CallCount())
	}
	events, _ := store.GetEvents(ctx, runID, "chatbot", db.EventFilter{})
	var userAt1, asstAt1 int
	for _, e := range events {
		if e.Step == 1 && e.Event.EventType() == "user" {
			userAt1++
		}
		if e.Step == 1 && e.Event.EventType() == "assistant" {
			asstAt1++
		}
	}
	if userAt1 != 1 {
		t.Errorf("user events at step 1: %d, want 1 (the opening message)", userAt1)
	}
	if asstAt1 != 1 {
		t.Errorf("assistant events at step 1: %d, want 1 (the reply)", asstAt1)
	}

	cancel()
	<-done
}

// An interactive (chatbot) agent parks idle on a no-tool turn, wakes on a user
// follow-up (a UserEvent — an Input), and parks idle again.
func TestEventLoop_ParkThenWake(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mock := &llm.MockProvider{
		Model: "test-model",
		Responses: []llm.Response{
			{Content: "Done with initial task.", StopReason: "end_turn"},
			{Content: "Done with follow-up.", StopReason: "end_turn"},
		},
	}

	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "main-agent",
		Task:         "Do something",
		SystemPrompt: "You are a helpful agent.",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{},
		Workspace:    plain.New("/tmp"),
		Interactive:  true,
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(ctx) }()

	waitForIdle(t, store, runID, "main-agent")

	_, _ = store.AppendEvent(ctx, runID, "main-agent",
		&event.UserEvent{Content: "Do something else"})

	time.Sleep(500 * time.Millisecond)
	sess, _ := store.GetSession(ctx, runID, "main-agent")
	if sess.Status != db.SessionIdle {
		t.Errorf("after follow-up: status %q, want idle", sess.Status)
	}
	if mock.CallCount() != 2 {
		t.Errorf("LLM called %d times, want 2", mock.CallCount())
	}

	cancel()
	<-done
}

// Autonomous agent: tool call, then a no-tool turn → concluded. Verifies the
// step model (AssistantEvent and its tool results share the call step).
func TestEventLoop_ToolCallThenConclude(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	mock := &llm.MockProvider{
		Model: "test-model",
		Responses: []llm.Response{
			{Content: "Let me check.", ToolCalls: []llm.ToolCall{
				{ID: "tc1", Name: "bash", Arguments: `{"command":"echo hello"}`},
			}},
			{Content: "Done.", StopReason: "end_turn"},
		},
	}

	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "main-agent",
		Task:         "Run echo hello",
		SystemPrompt: "You are a helpful agent.",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{bash.New("/tmp", "", "")},
		Workspace:    plain.New("/tmp"),
	})

	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	sess, _ := store.GetSession(ctx, runID, "main-agent")
	if sess.Status != db.SessionConcluded {
		t.Errorf("status: %q, want concluded", sess.Status)
	}

	events, _ := store.GetEvents(ctx, runID, "main-agent", db.EventFilter{})
	var assistantStep, toolResultStep int
	for _, e := range events {
		switch e.Event.EventType() {
		case "assistant":
			if e.Event.(*event.AssistantEvent).ToolCalls != nil {
				assistantStep = e.Step
			}
		case "tool_result":
			toolResultStep = e.Step
		}
	}
	if assistantStep != toolResultStep {
		t.Errorf("tool result at step %d, assistant at step %d — should be same", toolResultStep, assistantStep)
	}
	if mock.CallCount() != 2 {
		t.Errorf("LLM called %d times, want 2", mock.CallCount())
	}
}

// An interactive chatbot with no task parks idle and responds to the first user
// message (a UserEvent) at step 1, not step 0.
func TestEventLoop_ChatbotNoTask(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "Try cafe 40.", StopReason: "end_turn"}},
	}

	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "chatty-bot",
		Task:         "",
		SystemPrompt: "You are a chatbot.",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{},
		Workspace:    plain.New("/tmp"),
		Interactive:  true,
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(ctx) }()

	waitForIdle(t, store, runID, "chatty-bot")
	_, _ = store.AppendEvent(ctx, runID, "chatty-bot",
		&event.UserEvent{Content: "What's for lunch?"})

	time.Sleep(500 * time.Millisecond)
	sess, _ := store.GetSession(ctx, runID, "chatty-bot")
	if sess.Status != db.SessionIdle {
		t.Errorf("status: %q, want idle", sess.Status)
	}

	events, _ := store.GetEvents(ctx, runID, "chatty-bot", db.EventFilter{})
	for _, e := range events {
		if e.Event.EventType() == "user" && e.Step != 1 {
			t.Errorf("user message at step %d, want step 1", e.Step)
		}
	}

	cancel()
	<-done
}

// A canceller-driven cancel of a live, parked session stops it (status cancelled,
// goroutine returns). No cooperative cancel-event polling involved.
func TestEventLoop_CancelStopsLiveSession(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "Standing by.", StopReason: "end_turn"}},
	}

	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "main-agent",
		Task:         "do a thing",
		SystemPrompt: "agent",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{},
		Workspace:    plain.New("/tmp"),
		Interactive:  true, // park idle so we can cancel it while live
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(ctx) }()

	waitForIdle(t, store, runID, "main-agent")

	if err := session.CancelSession(store, registry, runID, "main-agent", "operator stop"); err != nil {
		t.Fatalf("CancelSession: %v", err)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return after cancel")
	}

	sess, _ := store.GetSession(ctx, runID, "main-agent")
	if sess.Status != db.SessionCancelled {
		t.Errorf("status: %q, want cancelled", sess.Status)
	}
}

// A cold idle session is restarted by an Input (UserEvent) already pending in
// the DB: the loop short-circuits the wait, resumes, and parks idle again.
func TestEventLoop_RestartIdleSession(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_ = store.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: "main-agent", Status: db.SessionIdle,
	})
	_, _ = store.AppendEvent(ctx, runID, "main-agent",
		&event.UserEvent{Content: "Follow up question"})

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "Here's your answer.", StopReason: "end_turn"}},
	}

	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "main-agent",
		SystemPrompt: "You are an agent.",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{},
		Workspace:    plain.New("/tmp"),
		Interactive:  true,
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(ctx) }()

	// The session starts idle, so we can't wait on "idle". Instead wait for the
	// agent to resume (short-circuit on the pending UserEvent), call the LLM
	// once, and re-park idle — observed via the assistant event in the DB
	// (race-free) plus the status returning to idle.
	deadline := time.Now().Add(2 * time.Second)
	var assistantCount int
	for time.Now().Before(deadline) {
		evts, _ := store.GetEvents(ctx, runID, "main-agent", db.EventFilter{})
		assistantCount = 0
		for _, e := range evts {
			if e.Event.EventType() == "assistant" {
				assistantCount++
			}
		}
		if s, _ := store.GetSession(ctx, runID, "main-agent"); assistantCount >= 1 && s != nil && s.Status == db.SessionIdle {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if assistantCount != 1 {
		t.Errorf("assistant events: %d, want 1 (LLM should be called exactly once)", assistantCount)
	}
	sess, _ := store.GetSession(ctx, runID, "main-agent")
	if sess.Status != db.SessionIdle {
		t.Errorf("status: %q, want idle", sess.Status)
	}

	cancel()
	<-done
}

// An autonomous child ends with a no-tool turn → concluded, posting its result
// to the parent's stream as a ChildResultEvent.
func TestEventLoop_Conclude(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	// Parent session that should receive the child's concluded result.
	_ = store.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: "main-agent", Status: db.SessionOngoing,
	})

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "all done", StopReason: "end_turn"}},
	}

	ag := newT(testCfg{
		RunID:        runID,
		SessionID:    "child",
		Task:         "do it",
		ParentID:     "main-agent",
		SystemPrompt: "You are an agent.",
		LLM:          mock,
		Store:        store,
		Registry:     registry,
		Tools:        []*tool.Tool{},
		Workspace:    plain.New("/tmp"),
	})

	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	child, _ := store.GetSession(ctx, runID, "child")
	if child.Status != db.SessionConcluded {
		t.Errorf("child status: %q, want concluded", child.Status)
	}

	evts, _ := store.GetEvents(ctx, runID, "main-agent", db.EventFilter{})
	var found *event.ChildResultEvent
	for _, e := range evts {
		if cr, ok := e.Event.(*event.ChildResultEvent); ok {
			found = cr
		}
	}
	if found == nil {
		t.Fatal("parent did not receive a ChildResultEvent")
	}
	if found.Verdict != db.SessionConcluded || found.Content != "all done" || found.ChildSessionID != "child" {
		t.Errorf("child result: %+v", found)
	}
}

// seedResumeStep advances a freshly-created session to current_step=2, mimicking
// the post-bump state of a session whose step-1 turn was interrupted by a crash.
func seedResumeStep(t *testing.T, store db.Store, runID, sid string) {
	t.Helper()
	for i := 0; i < 2; i++ {
		if _, err := store.AdvanceStep(context.Background(), runID, sid); err != nil {
			t.Fatal(err)
		}
	}
}

// Crash mid-tool-execution: an AssistantEvent with two tool_calls but only one
// result. Resume must synthesize a placeholder for the orphan before the LLM
// call, then advance and continue.
func TestEventLoop_ResumeOrphanToolCall(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "agent", Status: db.SessionOngoing})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.AssistantEvent{
		Content: "checking",
		ToolCalls: []event.ToolCall{
			{ID: "tc1", Name: "bash", Arguments: "{}"},
			{ID: "tc2", Name: "bash", Arguments: "{}"},
		},
	})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.ToolResultEvent{ToolCallID: "tc1", Content: "ok"})
	seedResumeStep(t, store, runID, "agent") // current_step=2, step 1 has orphan tc2

	mock := &llm.MockProvider{Model: "test", Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "agent", LLM: mock, Store: store,
		Registry: registry, Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	// Placeholder for the orphan tc2 was inserted.
	evts, _ := store.GetEvents(ctx, runID, "agent", db.EventFilter{})
	var placeholder *event.ToolResultEvent
	for _, e := range evts {
		if tr, ok := e.Event.(*event.ToolResultEvent); ok && tr.ToolCallID == "tc2" {
			placeholder = tr
		}
	}
	if placeholder == nil {
		t.Fatal("no placeholder ToolResult inserted for orphan tc2")
	}
	if placeholder.Content == "" {
		t.Error("placeholder has empty content")
	}
	if mock.CallCount() != 1 {
		t.Errorf("LLM called %d times, want 1 (resume turn after repair)", mock.CallCount())
	}
	if sess, _ := store.GetSession(ctx, runID, "agent"); sess.Status != db.SessionConcluded {
		t.Errorf("status: %q, want concluded", sess.Status)
	}
}

// Crash mid-LLM-call: the step was advanced but no AssistantEvent was written.
// Resume must NOT double-advance — it calls the LLM at the existing call step.
func TestEventLoop_ResumeMidLLM(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "agent", Status: db.SessionOngoing})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.UserEvent{Content: "do X"})
	seedResumeStep(t, store, runID, "agent") // current_step=2, step 1 has context but NO assistant

	mock := &llm.MockProvider{Model: "test", Responses: []llm.Response{{Content: "answer", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "agent", LLM: mock, Store: store,
		Registry: registry, Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	// The assistant must land at the predecessor's call step (1), not a new step,
	// and current_step must not have been double-bumped past 2.
	sess, _ := store.GetSession(ctx, runID, "agent")
	if sess.CurrentStep != 2 {
		t.Errorf("current_step: %d, want 2 (no double-advance on mid-LLM resume)", sess.CurrentStep)
	}
	evts, _ := store.GetEvents(ctx, runID, "agent", db.EventFilter{})
	var assistantStep = -1
	for _, e := range evts {
		if _, ok := e.Event.(*event.AssistantEvent); ok {
			assistantStep = e.Step
		}
	}
	if assistantStep != 1 {
		t.Errorf("assistant at step %d, want 1 (call step of the dead predecessor)", assistantStep)
	}
	if mock.CallCount() != 1 {
		t.Errorf("LLM called %d times, want 1", mock.CallCount())
	}
	if sess.Status != db.SessionConcluded {
		t.Errorf("status: %q, want concluded", sess.Status)
	}
}

// Crash after the final no-tool reply but before flipping status: an autonomous
// agent resumes "at rest" and concludes WITHOUT re-calling the LLM.
func TestEventLoop_ResumeAtRestAutonomous(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "agent", Status: db.SessionOngoing})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.AssistantEvent{Content: "final answer"})
	seedResumeStep(t, store, runID, "agent")

	mock := &llm.MockProvider{Model: "test", Responses: []llm.Response{{Content: "SHOULD NOT BE CALLED"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "agent", LLM: mock, Store: store,
		Registry: registry, Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	if mock.CallCount() != 0 {
		t.Errorf("LLM called %d times, want 0 (at-rest finalize)", mock.CallCount())
	}
	if sess, _ := store.GetSession(ctx, runID, "agent"); sess.Status != db.SessionConcluded {
		t.Errorf("status: %q, want concluded", sess.Status)
	}
}

// Same crash for an interactive agent: it resumes at rest, parks idle (no
// duplicate LLM call), and waits for the next message.
func TestEventLoop_ResumeAtRestInteractive(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "bot", Status: db.SessionOngoing})
	_ = store.AppendEventAtStep(ctx, runID, "bot", 1, &event.AssistantEvent{Content: "hi there"})
	seedResumeStep(t, store, runID, "bot")

	mock := &llm.MockProvider{Model: "test", Responses: []llm.Response{{Content: "SHOULD NOT BE CALLED"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "bot", LLM: mock, Store: store,
		Registry: registry, Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
		Interactive: true,
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(ctx) }()

	waitForIdle(t, store, runID, "bot")
	if mock.CallCount() != 0 {
		t.Errorf("LLM called %d times, want 0 (at-rest parks idle, no re-ask)", mock.CallCount())
	}

	cancel()
	<-done
}

// Bootstrap persists the session's own workspace into Session.Metadata so a
// later respawn reconstructs it from the row (not from run config).
func TestBootstrap_PersistsWorkspaceInMetadata(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()
	mock := &llm.MockProvider{Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "s1", Task: "do the thing",
		AgentType: "standard_agent", SystemPrompt: "sys",
		LLM: mock, Store: store, Registry: registry,
		Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	sess, err := store.GetSession(ctx, runID, "s1")
	if err != nil {
		t.Fatal(err)
	}
	raw, ok := sess.Metadata[workspace.SessionMetadataKey].(string)
	if !ok || raw == "" {
		t.Fatalf("workspace not persisted in metadata: %v", sess.Metadata)
	}
	ws, err := workspace.Unmarshal([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if ws.Root() != "/tmp" || ws.Kind() != "plain" {
		t.Errorf("reconstructed workspace root=%q kind=%q, want /tmp plain", ws.Root(), ws.Kind())
	}
}

// Bootstrap seeds an agent-visible workspace hint as a step-0 SystemEvent with
// marker "workspace" (the amplio marker mechanism, not a literal text wrapper).
func TestBootstrap_EmitsWorkspaceEvent(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()
	mock := &llm.MockProvider{Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "s1", Task: "do the thing",
		AgentType: "standard_agent", SystemPrompt: "sys",
		LLM: mock, Store: store, Registry: registry,
		Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	events, err := store.GetEvents(ctx, runID, "s1", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, rec := range events {
		se, ok := rec.Event.(*event.SystemEvent)
		if !ok || se.Marker != "workspace" {
			continue
		}
		found = true
		if rec.Step != 0 {
			t.Errorf("workspace event at step %d, want 0", rec.Step)
		}
		if !strings.Contains(se.Content, "Working directory:") {
			t.Errorf("workspace event body = %q", se.Content)
		}
	}
	if !found {
		t.Error("no workspace SystemEvent seeded at bootstrap")
	}
}

// Bootstrap describes the currently-available CLI tools as a step-0 SystemEvent
// (marker "cli_tools"); unavailable tools are omitted.
func TestBootstrap_InjectsAvailableCLITools(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()
	mock := &llm.MockProvider{Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "s1", Task: "do it",
		AgentType: "standard_agent", SystemPrompt: "sys",
		LLM: mock, Store: store, Registry: registry, Workspace: plain.New("/tmp"),
		CLITools: []cli.Tool{
			{Name: "sh", Snippet: "run a shell"},
			{Name: "amplio-not-a-real-binary-xyz", Snippet: "should be omitted"},
		},
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	events, err := store.GetEvents(ctx, runID, "s1", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, rec := range events {
		if se, ok := rec.Event.(*event.SystemEvent); ok && se.Marker == event.MarkerCLITools {
			body = se.Content
		}
	}
	if body == "" {
		t.Fatal("no tools SystemEvent seeded at bootstrap")
	}
	if !strings.Contains(body, "**sh**") {
		t.Errorf("available tool missing from body:\n%s", body)
	}
	if strings.Contains(body, "should be omitted") {
		t.Errorf("unavailable tool leaked into body:\n%s", body)
	}
}

// Scenario C — the crash window introduced by per-result appends: a step whose
// assistant and ALL tool results are durably written, but the process died
// before MarkStepFinalized. Orphan repair finds nothing to mend; recovery's
// finalize-after-mend must still advance last_finalized_step so the step is
// summarized deterministically (not left to a later turn's finalize).
func TestReconcileResume_FinalizesCompleteUnfinalizedStep(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "agent", Status: db.SessionOngoing})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.AssistantEvent{
		Content:   "running",
		ToolCalls: []event.ToolCall{{ID: "tc1", Name: "bash", Arguments: "{}"}},
	})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.ToolResultEvent{ToolCallID: "tc1", Content: "ok"})
	seedResumeStep(t, store, runID, "agent") // current_step=2; step 1 complete but unfinalized

	sess, _ := store.GetSession(ctx, runID, "agent")
	if sess.LastFinalizedStep != 0 {
		t.Fatalf("precondition: last_finalized_step=%d, want 0", sess.LastFinalizedStep)
	}

	ag := newT(testCfg{
		RunID: runID, SessionID: "agent", LLM: &llm.MockProvider{Model: "test"},
		Store: store, Registry: registry, Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if _, _, err := ag.reconcileResume(ctx, sess); err != nil {
		t.Fatal(err)
	}

	got, _ := store.GetSession(ctx, runID, "agent")
	if got.LastFinalizedStep != 1 {
		t.Errorf("last_finalized_step=%d, want 1 (recovery finalized the complete-but-unfinalized step)", got.LastFinalizedStep)
	}
	// No spurious orphan placeholder — the step was already complete.
	evts, _ := store.GetEvents(ctx, runID, "agent", db.EventFilter{})
	results := 0
	for _, e := range evts {
		if _, ok := e.Event.(*event.ToolResultEvent); ok {
			results++
		}
	}
	if results != 1 {
		t.Errorf("tool results=%d, want 1 (no spurious placeholder)", results)
	}
}

// Crash mid-execution with per-result appends: one tool result landed, another
// didn't. Recovery must synthesize the orphan AND finalize the now-complete step.
func TestReconcileResume_MendsOrphanThenFinalizes(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()

	_ = store.CreateSession(ctx, db.SessionRecord{RunID: runID, SessionID: "agent", Status: db.SessionOngoing})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.AssistantEvent{
		Content: "running",
		ToolCalls: []event.ToolCall{
			{ID: "tc1", Name: "bash", Arguments: "{}"},
			{ID: "tc2", Name: "bash", Arguments: "{}"},
		},
	})
	_ = store.AppendEventAtStep(ctx, runID, "agent", 1, &event.ToolResultEvent{ToolCallID: "tc1", Content: "ok"})
	seedResumeStep(t, store, runID, "agent") // step 1 has orphan tc2, unfinalized

	sess, _ := store.GetSession(ctx, runID, "agent")
	ag := newT(testCfg{
		RunID: runID, SessionID: "agent", LLM: &llm.MockProvider{Model: "test"},
		Store: store, Registry: registry, Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if _, _, err := ag.reconcileResume(ctx, sess); err != nil {
		t.Fatal(err)
	}

	got, _ := store.GetSession(ctx, runID, "agent")
	if got.LastFinalizedStep != 1 {
		t.Errorf("last_finalized_step=%d, want 1 (finalized after mending orphan)", got.LastFinalizedStep)
	}
	evts, _ := store.GetEvents(ctx, runID, "agent", db.EventFilter{})
	have := map[string]bool{}
	for _, e := range evts {
		if tr, ok := e.Event.(*event.ToolResultEvent); ok {
			have[tr.ToolCallID] = true
		}
	}
	if !have["tc1"] || !have["tc2"] {
		t.Errorf("results present: %v, want both tc1 and tc2 (orphan mended)", have)
	}
}

// TestBootstrap_TaskOrderedLastAmongFraming verifies the bootstrap ordering:
// all framing system events (new_session, system_prompt, cli_tools, workspace,
// agents_md) precede the Task, and task-derived initial_recall follows it. This
// is what lets the projection layer hoist the leading system cluster and leave
// the Task as the first user turn.
func TestBootstrap_TaskOrderedLastAmongFraming(t *testing.T) {
	store, runID, registry := testSetup(t, config.RunConfig{AgentsMD: "operator rules"})
	ctx := context.Background()
	mock := &llm.MockProvider{Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "the real task",
		AgentType: "standard_agent", SystemPrompt: "sys",
		LLM: mock, Store: store, Registry: registry, Workspace: plain.New("/tmp"),
		CLITools:      []cli.Tool{{Name: "sh", Snippet: "run a shell"}},
		InitialRecall: func(_ context.Context, _ string) string { return "recalled skills" },
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}

	events, err := store.GetEvents(ctx, runID, "main-agent", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	// Collect the step-0 bootstrap sequence as (marker | "TASK") labels.
	var seq []string
	for _, rec := range events {
		if rec.Step != 0 {
			continue
		}
		switch e := rec.Event.(type) {
		case *event.SystemEvent:
			seq = append(seq, e.Marker)
		case *event.UserEvent:
			seq = append(seq, "TASK")
		}
	}
	want := []string{
		event.MarkerNewSession,
		event.MarkerSystemPrompt,
		event.MarkerCLITools,
		event.MarkerWorkspace,
		event.MarkerAgentsMD,
		"TASK",
		event.MarkerInitialRecall,
	}
	if len(seq) != len(want) {
		t.Fatalf("bootstrap sequence = %v, want %v", seq, want)
	}
	for i := range want {
		if seq[i] != want[i] {
			t.Fatalf("bootstrap sequence = %v, want %v", seq, want)
		}
	}
}

// TestBuildMessages_HoistsLeadingSystemCluster verifies the projection layer:
// the contiguous leading run of system events is folded into systemPrompt, the
// Task becomes the first user message, and a post-task system event (recall)
// stays a message in place (not hoisted).
func TestBuildMessages_HoistsLeadingSystemCluster(t *testing.T) {
	a := &EventLoopAgent{}
	recs := []db.EventRecord{
		{Event: &event.SystemEvent{Content: "new session", Marker: event.MarkerNewSession}},
		{Event: &event.SystemEvent{Content: "you are an agent", Marker: event.MarkerSystemPrompt}},
		{Event: &event.SystemEvent{Content: "cli: sh", Marker: event.MarkerCLITools}},
		{Event: &event.UserEvent{Content: "the real task"}},
		{Event: &event.SystemEvent{Content: "recalled skills", Marker: event.MarkerInitialRecall}},
	}
	systemPrompt, messages := a.buildMessages(recs)

	// The three leading system events fold into systemPrompt (banners kept).
	for _, want := range []string{"new session", "you are an agent", "cli: sh", "system", "system_prompt"} {
		if !strings.Contains(systemPrompt, want) {
			t.Errorf("systemPrompt missing %q; got:\n%s", want, systemPrompt)
		}
	}
	// The post-task recall must NOT be in the system prompt.
	if strings.Contains(systemPrompt, "recalled skills") {
		t.Errorf("post-task recall was hoisted into systemPrompt; got:\n%s", systemPrompt)
	}
	// Messages: [user(task), system(recall)] — task is FIRST and is a user turn.
	if len(messages) != 2 {
		t.Fatalf("messages len = %d, want 2: %+v", len(messages), messages)
	}
	if messages[0].Role != llm.RoleUser || messages[0].Content != "the real task" {
		t.Errorf("first message = %+v, want user 'the real task'", messages[0])
	}
	if messages[1].Role != llm.RoleSystem || !strings.Contains(messages[1].Content, "recalled skills") {
		t.Errorf("second message = %+v, want system recall", messages[1])
	}
}

// TestBuildMessages_CompactionAtTipYieldsNonEmpty is a regression test for the
// crash where a compaction whose boundary sits at the tip of the conversation
// left the request with zero messages ("messages: Field required"). After such
// a compaction the new-generation context is only the step-0 bootstrap (all
// system events) followed by the CompactionEvent, with no post-boundary tail.
// The compaction must project to a user turn so buildMessages returns at least
// one message rather than folding everything into the system prompt.
func TestBuildMessages_CompactionAtTipYieldsNonEmpty(t *testing.T) {
	a := &EventLoopAgent{}
	recs := []db.EventRecord{
		{Event: &event.SystemEvent{Content: "new session", Marker: event.MarkerNewSession}},
		{Event: &event.SystemEvent{Content: "you are an agent", Marker: event.MarkerSystemPrompt}},
		{Event: &event.SystemEvent{Content: "cli: sh", Marker: event.MarkerCLITools}},
		{Event: &event.CompactionEvent{Content: "summary of prior work"}},
	}
	systemPrompt, messages := a.buildMessages(recs)

	if len(messages) == 0 {
		t.Fatalf("buildMessages returned zero messages; provider would reject with \"messages: Field required\"\nsystemPrompt:\n%s", systemPrompt)
	}
	// The compaction is the single (user) message; the bootstrap is the system prompt.
	if messages[len(messages)-1].Role != llm.RoleUser {
		t.Errorf("last message role = %q, want user", messages[len(messages)-1].Role)
	}
	if !strings.Contains(messages[len(messages)-1].Content, "summary of prior work") {
		t.Errorf("compaction message missing summary content; got:\n%s", messages[len(messages)-1].Content)
	}
}

// TestEventLoop_HandleFromLauncher_UsesPreRegisteredSlot verifies the launcher
// registration model: when Config.Handle is set (the launcher claimed the slot
// via RegisterAndContext), Run() adopts that handle rather than registering a
// fresh one, and does NOT unregister on exit (the launcher's release owns that).
func TestEventLoop_HandleFromLauncher_UsesPreRegisteredSlot(t *testing.T) {
	store, runID, registry := testSetup(t)

	// Launcher claims the slot up front.
	ctx, handle, release, ok := registry.RegisterAndContext(context.Background(), "main-agent")
	if !ok {
		t.Fatal("RegisterAndContext should win a fresh slot")
	}
	defer release()

	mock := &llm.MockProvider{
		Model:     "test-model",
		Responses: []llm.Response{{Content: "done", StopReason: "end_turn"}},
	}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "t", SystemPrompt: "sys",
		LLM: mock, Store: store, Registry: registry, Handle: handle,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})
	if err := ag.Run(ctx); err != nil {
		t.Fatal(err)
	}
	// Run() must have used the launcher's handle, not registered a rival one.
	if ag.Handle() != handle {
		t.Error("Run() should adopt Config.Handle, not create a new one")
	}
	// Run() must NOT have unregistered the slot (that's release's job); it's still
	// registered until the caller's deferred release fires.
	if !registry.IsRegistered("main-agent") {
		t.Error("Run() must not unregister a launcher-owned slot")
	}
}

// TestEventLoop_CancelBeforeRun_FreshSession is the "cancel before Run()" edge
// case: the launcher claims the slot, the ctx is cancelled (e.g. a session_cancel
// racing the goroutine start) BEFORE Run() executes, and a fresh session (no DB
// row yet) is launched. Run() must observe the latched cancellation and exit
// promptly rather than proceeding to bootstrap/loop.
func TestEventLoop_CancelBeforeRun_FreshSession(t *testing.T) {
	store, runID, registry := testSetup(t)

	ctx, handle, release, ok := registry.RegisterAndContext(context.Background(), "main-agent")
	if !ok {
		t.Fatal("claim should win")
	}
	defer release()
	// Cancel BEFORE Run() — the sticky ctx must be observed immediately.
	registry.Interrupt("main-agent")

	mock := &llm.MockProvider{Model: "m", Responses: []llm.Response{{Content: "x", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "main-agent", Task: "t", SystemPrompt: "sys",
		LLM: mock, Store: store, Registry: registry, Handle: handle,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(ctx) }()
	select {
	case <-done:
		// Exited promptly on the pre-cancelled ctx — good.
	case <-time.After(3 * time.Second):
		t.Fatal("Run() did not exit promptly on a pre-cancelled ctx")
	}
}

// TestEventLoop_CancelBeforeRun_ExistingTerminalSession is the same edge case for
// an EXISTING session whose DB row is already terminal (CancelSession writes the
// terminal status before it interrupts). Run() must exit without resurrecting it.
func TestEventLoop_CancelBeforeRun_ExistingTerminalSession(t *testing.T) {
	store, runID, registry := testSetup(t)
	ctx := context.Background()
	// Seed an existing, already-cancelled session row.
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: runID, SessionID: "sess", AgentType: "standard_agent", Status: db.SessionCancelled,
	}); err != nil {
		t.Fatal(err)
	}

	runCtx, handle, release, ok := registry.RegisterAndContext(context.Background(), "sess")
	if !ok {
		t.Fatal("claim should win")
	}
	defer release()
	registry.Interrupt("sess") // cancel before Run()

	mock := &llm.MockProvider{Model: "m", Responses: []llm.Response{{Content: "x", StopReason: "end_turn"}}}
	ag := newT(testCfg{
		RunID: runID, SessionID: "sess", SystemPrompt: "sys",
		LLM: mock, Store: store, Registry: registry, Handle: handle,
		Tools: []*tool.Tool{}, Workspace: plain.New("/tmp"),
	})

	done := make(chan error, 1)
	go func() { done <- ag.Run(runCtx) }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("Run() did not exit promptly on a pre-cancelled ctx for a terminal session")
	}
	// The terminal session must not have been flipped back to ongoing.
	sess, err := store.GetSession(ctx, runID, "sess")
	if err != nil {
		t.Fatal(err)
	}
	if sess.Status == db.SessionOngoing {
		t.Errorf("a cancelled session must not be resurrected to ongoing")
	}
}

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

package runtime

import (
	"context"
	"testing"
	"time"

	"amplio/internal/agent"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/event"
	"amplio/internal/llm"
	"amplio/internal/workspace"
	"amplio/internal/workspace/plain"
)

// recorderAgent records that it was (re)spawned, by session id, then exits. It
// stands in for a real agent so RecoverRun's orchestration can be tested without
// the LLM loop; the per-session resume mechanics are covered by eventloop tests.
type recorderAgent struct{ sid string }

func (a *recorderAgent) Run(context.Context) error { recoverSeen <- a.sid; return nil }
func (a *recorderAgent) SessionID() string         { return a.sid }

var recoverSeen = make(chan string, 64)

func init() {
	agent.Register("recover_test_agent", func(_ *agent.Env, cfg *agent.Config) (agent.Agent, error) {
		return &recorderAgent{sid: cfg.SessionID}, nil
	})
}

func TestRecoverRun(t *testing.T) {
	// Drain residue from any prior run (e.g. -count>1).
	for len(recoverSeen) > 0 {
		<-recoverSeen
	}

	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()
	runID := "run-recover"
	if err := store.CreateRun(ctx, db.RunRecord{RunID: runID}); err != nil {
		t.Fatal(err)
	}

	mk := func(sid, status, parent string, tail event.Event) {
		_ = store.CreateSession(ctx, db.SessionRecord{
			RunID: runID, SessionID: sid, AgentType: "recover_test_agent",
			Status: status, ParentID: parent,
		})
		if tail != nil {
			_ = store.AppendEventAtStep(ctx, runID, sid, 1, tail)
		}
	}
	ctxTail := func() event.Event { return &event.UserEvent{Content: "ctx"} }
	mk("ongoing", db.SessionOngoing, "", ctxTail())                             // spine, not at rest
	mk("awaiting", db.SessionAwaiting, "", ctxTail())                           // spine, not at rest
	mk("crashed-root", db.SessionCrashed, "", ctxTail())                        // spine, not at rest
	mk("crashed-child", db.SessionCrashed, "ongoing", ctxTail())                // NOT spine
	mk("concluded", db.SessionConcluded, "", ctxTail())                         // NOT spine
	mk("atrest", db.SessionOngoing, "", &event.AssistantEvent{Content: "done"}) // spine, AT REST

	runReg := NewRunRegistry()
	mgr := NewRunManager(store, func(string) (llm.Provider, error) { return &llm.MockProvider{Model: "m"}, nil }, runReg, plain.Factory)
	store.SetCommitListener(NewCommitNotifier(runReg, mgr.RespawnSession, mgr.SessionStatus))

	revived, err := mgr.RecoverRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}

	// Exactly the four spine sessions are respawned.
	want := map[string]bool{"ongoing": true, "awaiting": true, "crashed-root": true, "atrest": true}
	if revived != len(want) {
		t.Fatalf("RecoverRun revived %d sessions, want %d", revived, len(want))
	}
	got := map[string]bool{}
	deadline := time.After(2 * time.Second)
	for len(got) < len(want) {
		select {
		case sid := <-recoverSeen:
			got[sid] = true
		case <-deadline:
			t.Fatalf("timed out; respawned %v, want %v", got, want)
		}
	}
	// No non-spine session should be respawned.
	select {
	case sid := <-recoverSeen:
		t.Errorf("unexpected respawn of %q (non-spine or duplicate)", sid)
	case <-time.After(150 * time.Millisecond):
	}

	// RecoverEvent placement: non-at-rest spine got one; the at-rest one did not.
	for _, sid := range []string{"ongoing", "awaiting", "crashed-root"} {
		if !hasRecoverEvent(t, store, runID, sid) {
			t.Errorf("%q missing RecoverEvent", sid)
		}
	}
	if hasRecoverEvent(t, store, runID, "atrest") {
		t.Error("at-rest session should not get a RecoverEvent")
	}
}

// StartRun resolves the agent provider from each run's RunConfig.LLM via the
// injected factory, and caches one provider per distinct spec.
func TestStartRun_PerRunProviderCached(t *testing.T) {
	for len(recoverSeen) > 0 {
		<-recoverSeen
	}
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	// providerFor is called synchronously inside StartRun (before the agent
	// goroutine), so recording specs here needs no extra synchronization.
	var specs []string
	factory := func(spec string) (llm.Provider, error) {
		specs = append(specs, spec)
		return &llm.MockProvider{Model: spec}, nil
	}
	runReg := NewRunRegistry()
	mgr := NewRunManager(store, factory, runReg, plain.Factory)

	ctx := context.Background()
	start := func(spec string) {
		// recover_test_agent records + exits without touching the LLM, so the run
		// concludes immediately; we only care that the provider was resolved.
		if _, err := mgr.StartRun(ctx, StartRunConfig{
			RunConfig: config.RunConfig{
				Task: "t", Workspace: ".", LLM: spec, AgentType: "recover_test_agent",
			},
			RootSessionID: "root",
		}); err != nil {
			t.Fatalf("StartRun(%s): %v", spec, err)
		}
	}
	start("vertex:model-a")
	start("vertex:model-a") // same spec → served from cache
	start("vertex:model-b")

	counts := map[string]int{}
	for _, s := range specs {
		counts[s]++
	}
	if len(counts) != 2 || counts["vertex:model-a"] != 1 || counts["vertex:model-b"] != 1 {
		t.Errorf("factory calls = %v, want each distinct spec built once", counts)
	}

	// Drain the recorder sends so a later RecoverRun test starts clean.
	deadline := time.After(2 * time.Second)
	for range 3 {
		select {
		case <-recoverSeen:
		case <-deadline:
			t.Fatal("timed out draining recorder agent sends")
		}
	}
}

// A launchAgent failure AFTER it has claimed the registry slot (here: an unknown
// agent type, which fails at agent.Get) must release the slot AND GC the now-empty
// run registry + allocator — not leave a lingering empty entry behind.
func TestStartRun_LaunchFailureCleansUpRegistry(t *testing.T) {
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	runReg := NewRunRegistry()
	mgr := NewRunManager(store, func(string) (llm.Provider, error) { return &llm.MockProvider{Model: "m"}, nil }, runReg, plain.Factory)

	ctx := context.Background()
	_, err = mgr.StartRun(ctx, StartRunConfig{
		RunConfig:     config.RunConfig{Task: "t", Workspace: ".", LLM: "vertex:m", AgentType: "no_such_agent_type"},
		RootSessionID: "root",
	})
	if err == nil {
		t.Fatal("StartRun with an unknown agent type should fail")
	}

	// The run row was created before launchAgent failed; find its id.
	runs, _, err := store.ListRuns(ctx, db.ListRunsOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(runs))
	}
	runID := runs[0].RunID

	// launchAgent's GetOrCreate published a registry instance and RegisterAndContext
	// claimed a slot; the failed launch must have released the slot AND removed the
	// now-empty registry entry from the map (not just left it empty). Get returns
	// the raw map entry, so nil == fully cleaned up.
	if reg := runReg.Get(runID); reg != nil {
		t.Errorf("run registry entry for %q lingers after a failed launch (empty=%v)", runID, reg.IsEmpty())
	}
}

func TestExtractMarkdownTitle(t *testing.T) {
	cases := []struct{ in, want string }{
		{"# Hello world", "Hello world"},
		{"### Deep\nbody", "Deep"},
		{"\n\n## After blanks", "After blanks"},
		{"  ## indented", "indented"},
		{"no heading here", ""},
		{"#nospace", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := extractMarkdownTitle(c.in); got != c.want {
			t.Errorf("extractMarkdownTitle(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// StartRun resolves the title: explicit wins; else a leading markdown heading;
// else it fires the async generator with the task.
func TestStartRun_TitleResolution(t *testing.T) {
	for len(recoverSeen) > 0 {
		<-recoverSeen
	}
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	runReg := NewRunRegistry()
	mgr := NewRunManager(store, func(string) (llm.Provider, error) { return &llm.MockProvider{Model: "m"}, nil }, runReg, plain.Factory)
	gen := make(chan string, 4)
	mgr.SetTitleGenerator(func(_, task string) { gen <- task })

	ctx := context.Background()
	start := func(title, task string) string {
		id, err := mgr.StartRun(ctx, StartRunConfig{
			Title:         title,
			RunConfig:     config.RunConfig{Task: task, Workspace: ".", LLM: "x", AgentType: "recover_test_agent"},
			RootSessionID: "root",
		})
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	titleOf := func(id string) string {
		r, err := store.GetRun(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		return r.Title
	}

	if got := titleOf(start("Explicit", "# Heading\nbody")); got != "Explicit" {
		t.Errorf("explicit title = %q, want Explicit", got)
	}
	if got := titleOf(start("", "# Heading title\nbody")); got != "Heading title" {
		t.Errorf("markdown title = %q, want 'Heading title'", got)
	}
	if got := titleOf(start("", "just do a thing")); got != "" {
		t.Errorf("pre-gen title = %q, want empty", got)
	}

	// Generator fired exactly once (only the empty+non-markdown run).
	select {
	case task := <-gen:
		if task != "just do a thing" {
			t.Errorf("genTitle task = %q", task)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("genTitle was not fired")
	}
	select {
	case extra := <-gen:
		t.Errorf("genTitle fired again unexpectedly: %q", extra)
	case <-time.After(100 * time.Millisecond):
	}

	for range 3 { // drain recorder-agent sends
		select {
		case <-recoverSeen:
		case <-time.After(2 * time.Second):
			t.Fatal("timed out draining recorder sends")
		}
	}
}

func hasRecoverEvent(t *testing.T, store db.Store, runID, sid string) bool {
	t.Helper()
	evts, _ := store.GetEvents(context.Background(), runID, sid, db.EventFilter{})
	for _, e := range evts {
		if _, ok := e.Event.(*event.RecoverEvent); ok {
			return true
		}
	}
	return false
}

// A spine session whose agent type is no longer registered (e.g. a run from an
// older binary) must be skipped: not revived, and given no RecoverEvent. A
// registered spine session in the same run is still recovered, proving the skip
// is selective rather than aborting the whole recovery.
func TestRecoverRun_SkipsUnknownAgentType(t *testing.T) {
	for len(recoverSeen) > 0 {
		<-recoverSeen
	}

	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()
	runID := "run-recover-unknown"
	if err := store.CreateRun(ctx, db.RunRecord{RunID: runID}); err != nil {
		t.Fatal(err)
	}

	mk := func(sid, agentType string) {
		_ = store.CreateSession(ctx, db.SessionRecord{
			RunID: runID, SessionID: sid, AgentType: agentType,
			Status: db.SessionOngoing,
		})
		_ = store.AppendEventAtStep(ctx, runID, sid, 1, &event.UserEvent{Content: "ctx"})
	}
	mk("real", "recover_test_agent")     // spine, not at rest, registered
	mk("ghost", "event_loop_gone_stale") // spine, not at rest, UNREGISTERED

	runReg := NewRunRegistry()
	mgr := NewRunManager(store, func(string) (llm.Provider, error) { return &llm.MockProvider{Model: "m"}, nil }, runReg, plain.Factory)
	store.SetCommitListener(NewCommitNotifier(runReg, mgr.RespawnSession, mgr.SessionStatus))

	revived, err := mgr.RecoverRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if revived != 1 {
		t.Fatalf("RecoverRun revived %d sessions, want 1 (only the registered one)", revived)
	}

	select {
	case sid := <-recoverSeen:
		if sid != "real" {
			t.Errorf("respawned %q, want only %q", sid, "real")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("registered spine session was not respawned")
	}
	select {
	case sid := <-recoverSeen:
		t.Errorf("unexpected respawn of %q (ghost should be skipped)", sid)
	case <-time.After(150 * time.Millisecond):
	}

	if hasRecoverEvent(t, store, runID, "ghost") {
		t.Error("session with unknown agent type should not get a RecoverEvent")
	}
}

func TestWorkspaceForSession(t *testing.T) {
	mgr := NewRunManager(nil, nil, NewRunRegistry(), plain.Factory)
	fallback := config.RunConfig{Workspace: "/fallback"}

	// Persisted workspace metadata is reconstructed verbatim.
	blob, err := workspace.Marshal(plain.New("/some/dir"))
	if err != nil {
		t.Fatal(err)
	}
	persisted := &db.SessionRecord{
		RunID: "r", SessionID: "s",
		Metadata: map[string]any{workspace.SessionMetadataKey: string(blob)},
	}
	if got := mgr.workspaceForSession(persisted, fallback); got.Root() != "/some/dir" {
		t.Errorf("persisted: root = %q, want /some/dir", got.Root())
	}

	// No metadata (legacy session) falls back to the run-config workspace.
	legacy := &db.SessionRecord{RunID: "r", SessionID: "s2"}
	if got := mgr.workspaceForSession(legacy, fallback); got.Root() != "/fallback" {
		t.Errorf("legacy fallback: root = %q, want /fallback", got.Root())
	}

	// Corrupt metadata also falls back rather than failing.
	corrupt := &db.SessionRecord{
		RunID: "r", SessionID: "s3",
		Metadata: map[string]any{workspace.SessionMetadataKey: "not-json"},
	}
	if got := mgr.workspaceForSession(corrupt, fallback); got.Root() != "/fallback" {
		t.Errorf("corrupt fallback: root = %q, want /fallback", got.Root())
	}
}

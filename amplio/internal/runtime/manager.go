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
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"time"

	"amplio/internal/agent"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/eventstream"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	"amplio/internal/session"
	"amplio/internal/skills"
	"amplio/internal/util"
	"amplio/internal/workspace"
)

// WorkspaceFactory creates a Workspace from a path string (from RunConfig).
type WorkspaceFactory func(path string) workspace.Workspace

// ProviderFactory builds an llm.Provider from a spec string (e.g.
// "vertex-claude:claude-..."). RunManager calls it per run, so each run uses the model
// recorded in its own RunConfig.LLM rather than one server-wide provider.
type ProviderFactory func(spec string) (llm.Provider, error)

// RunManager launches and tracks agent runs.
type RunManager struct {
	store            db.Store
	newProvider      ProviderFactory
	runReg           *RunRegistry
	workspaceFactory WorkspaceFactory
	broadcaster      eventstream.Broadcaster  // live token-stream sink; nil = none
	genTitle         func(runID, task string) // async run-title generator; nil = none
	skillIndex       *skills.Index            // shared skill recall corpus; nil = no skill recall
	lessonIndex      *lessons.Index           // shared lesson recall corpus; nil = no lesson recall
	systemFast       llm.Provider             // shared fast tier (compaction judge); nil = none
	systemHQ         llm.Provider             // shared HQ tier (compaction summarizer); nil = none

	providerMu sync.Mutex
	providers  map[string]llm.Provider // spec -> provider; stateless, reused across runs

	allocMu    sync.Mutex
	allocators map[string]*session.NameAllocator // per-run session-id allocators

	chatbotMu sync.Mutex // serializes AddChatbot so concurrent attaches don't double-launch

	ephemeral *EphemeralAgentRegistry // per-run set of in-flight non-session workers (critic, …)
}

// SetBroadcaster installs the live token-stream sink injected into every agent
// Env. Call once at startup (the server does; the headless CLI leaves it nil).
func (m *RunManager) SetBroadcaster(b eventstream.Broadcaster) { m.broadcaster = b }

// EphemeralAgents returns the per-run registry of in-flight non-session
// workers (e.g. the critic's report generator). Producers (Finalizer,
// future title-gen) Register on start / Unregister on finish; the HTTP
// runDetail handler snapshots it for the UI.
func (m *RunManager) EphemeralAgents() *EphemeralAgentRegistry { return m.ephemeral }

// SetTitleGenerator installs an async run-title generator, called (in a new
// goroutine) by StartRun when a run is created with an empty title. The hook
// typically calls system_llm_fast and writes the result via UpdateRunTitle.
// Set once at startup.
func (m *RunManager) SetTitleGenerator(fn func(runID, task string)) { m.genTitle = fn }

// SetSkillIndex installs the shared skill-recall index injected into every agent
// Env. Set once at startup (the server builds it synchronously before binding,
// so it's always ready by the time runs launch); agents add recall tools only
// once it reports built.
func (m *RunManager) SetSkillIndex(ix *skills.Index) { m.skillIndex = ix }

// SetLessonIndex installs the shared lesson-recall index injected into every
// agent Env. Set once at startup; agents add lesson recall once it reports built.
func (m *RunManager) SetLessonIndex(ix *lessons.Index) { m.lessonIndex = ix }

// SetSystemProviders installs the shared, process-wide system-tier providers
// (fast judge, HQ summarizer) used for context compaction. These are a System
// property: one instance each, built once at startup and injected into every
// agent's Env. A nil provider makes the eventloop skip compaction (see
// tryCompact) rather than panic — but the standard wiring (cmd/amplio) builds
// both at startup and fails fast if either spec is invalid, so in practice they
// are always non-nil. Set once at startup.
func (m *RunManager) SetSystemProviders(fast, hq llm.Provider) {
	m.systemFast, m.systemHQ = fast, hq
}

// markdownHeadingRe matches a leading markdown heading line (the title).
var markdownHeadingRe = regexp.MustCompile(`^\s*#{1,6}\s+(.+?)\s*$`)

// extractMarkdownTitle returns the heading text if the task's first non-empty
// line is a markdown heading, else "".
func extractMarkdownTitle(task string) string {
	for _, line := range strings.Split(task, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		if m := markdownHeadingRe.FindStringSubmatch(line); m != nil {
			return strings.TrimSpace(m[1])
		}
		return "" // first non-empty line isn't a heading
	}
	return ""
}

func NewRunManager(store db.Store, newProvider ProviderFactory, runReg *RunRegistry, wsFactory WorkspaceFactory) *RunManager {
	return &RunManager{
		store:            store,
		newProvider:      newProvider,
		runReg:           runReg,
		workspaceFactory: wsFactory,
		providers:        make(map[string]llm.Provider),
		allocators:       make(map[string]*session.NameAllocator),
		ephemeral:        NewEphemeralAgentRegistry(),
	}
}

// providerFor returns the (cached) provider for an LLM spec, building it on first
// use. Providers are stateless and safe to share across runs and sessions.
func (m *RunManager) providerFor(spec string) (llm.Provider, error) {
	m.providerMu.Lock()
	defer m.providerMu.Unlock()
	if p, ok := m.providers[spec]; ok {
		return p, nil
	}
	p, err := m.newProvider(spec)
	if err != nil {
		return nil, err
	}
	m.providers[spec] = p
	return p, nil
}

// allocatorFor returns the run's shared session-id allocator, creating it on
// first use. All sessions in a run share one allocator, so concurrent spawns
// serialize on it and receive distinct names.
func (m *RunManager) allocatorFor(runID string) *session.NameAllocator {
	m.allocMu.Lock()
	defer m.allocMu.Unlock()
	a, ok := m.allocators[runID]
	if !ok {
		a = session.NewNameAllocator(m.store, runID)
		m.allocators[runID] = a
	}
	return a
}

// removeAllocator drops a run's allocator once the run goes cold; a later resume
// re-creates it and re-seeds from the DB.
func (m *RunManager) removeAllocator(runID string) {
	m.allocMu.Lock()
	delete(m.allocators, runID)
	m.allocMu.Unlock()
}

// StartRun creates a new run in the DB and launches the root agent goroutine.
func (m *RunManager) StartRun(ctx context.Context, cfg StartRunConfig) (string, error) {
	// Resolve the run's provider before creating anything, so a bad/empty spec
	// fails fast instead of leaving an orphaned run with no agent.
	provider, err := m.providerFor(cfg.RunConfig.LLM)
	if err != nil {
		return "", fmt.Errorf("resolve agent llm %q: %w", cfg.RunConfig.LLM, err)
	}

	// Resolve title: explicit → leading markdown heading → (async LLM gen below).
	// For chat runs the task is empty, so the opening message is the basis.
	titleBasis := cfg.RunConfig.Task
	if titleBasis == "" {
		titleBasis = cfg.FirstMessage
	}
	title := cfg.Title
	if title == "" {
		title = extractMarkdownTitle(titleBasis)
	}

	runID, err := m.createRun(ctx, db.RunRecord{
		Config:    cfg.RunConfig,
		Title:     title,
		CreatedAt: time.Now().UTC(),
	})
	if err != nil {
		return "", err
	}

	// No title yet → generate one in the background (best-effort).
	if title == "" && m.genTitle != nil {
		go m.genTitle(runID, titleBasis)
	}

	agentCfg := &agent.Config{
		SessionID:    cfg.RootSessionID,
		Task:         cfg.RunConfig.Task,
		FirstMessage: cfg.FirstMessage,
	}
	// Reconstruct the live workspace from the persisted path — the same wrap
	// RecoverRun does — so fresh and resume are symmetric. The path is already
	// concrete (creation sentinels were resolved by the caller's pre-step).
	ws := m.workspaceFactory(cfg.RunConfig.Workspace)
	if err := m.launchAgent(ctx, runID, cfg.RunConfig.AgentType, provider, m.systemFast, m.systemHQ, ws, agentCfg); err != nil {
		return "", err
	}
	return runID, nil
}

// createRun inserts a new run with a freshly-generated id, retrying on the
// (effectively impossible) id collision: a UNIQUE violation regenerates the id.
func (m *RunManager) createRun(ctx context.Context, rec db.RunRecord) (string, error) {
	const attempts = 5
	for range attempts {
		rec.RunID = db.NewRunID()
		err := m.store.CreateRun(ctx, rec)
		if err == nil {
			return rec.RunID, nil
		}
		if !db.IsUniqueViolation(err) {
			return "", fmt.Errorf("create run: %w", err)
		}
	}
	return "", fmt.Errorf("create run: could not allocate a unique id after %d attempts", attempts)
}

// launchAgent wires an agent goroutine for a session in an (already-created) run
// and starts it. Shared by StartRun (the root agent) and AddChatbot (a sidecar).
// fast and hq are the shared system-tier providers (for compaction), the same
// instances for every run (see SetSystemProviders).
// The operator's AGENTS.md snapshot is NOT passed here — the agent reads it from
// the persisted RunConfig at bootstrap (see eventloop.bootstrap).
// runSessionGoroutine launches ag.Run(ctx) in a background goroutine with the
// standard manager cleanup, deferred in LIFO order: log a panic, release the
// registry slot (cancel + Unregister), then GC the run registry if it went empty
// (mirroring the goroutine-exit cleanup that RemoveIfEmpty is designed for).
// ctx is the session's cancelable context (from RegisterAndContext) and release
// is its paired teardown.
func (m *RunManager) runSessionGoroutine(ctx context.Context, runID, sessionID string, ag agent.Agent, release func()) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("session goroutine panicked", "run_id", runID, "session_id", sessionID, "panic", r)
			}
			release() // cancel ctx + Unregister the slot
			// Clean up run registry if no sessions remain. RemoveIfEmpty checks and
			// deletes atomically so a concurrent launch can't orphan a live session
			// (see RunRegistry.RemoveIfEmpty).
			if m.runReg.RemoveIfEmpty(runID) {
				m.removeAllocator(runID)
			}
		}()
		if err := ag.Run(ctx); err != nil {
			slog.Error("session run error", "run_id", runID, "session_id", sessionID, "error", err)
		}
	}()
}

func (m *RunManager) launchAgent(ctx context.Context, runID, agentType string, provider, fast, hq llm.Provider, ws workspace.Workspace, agentCfg *agent.Config) error {
	registry := m.runReg.GetOrCreate(runID)
	// Claim the slot SYNCHRONOUSLY (before spawning the goroutine) so the run
	// registry instance can't be orphaned by a concurrent RemoveIfEmpty, and hand
	// the resulting handle to the agent via Config. A brand-new session must win
	// the slot; already-registered here means a double-launch bug.
	runCtx, handle, release, ok := registry.RegisterAndContext(ctx, agentCfg.SessionID)
	if !ok {
		return fmt.Errorf("session %q already registered", agentCfg.SessionID)
	}
	agentCfg.Handle = handle
	// Release the claimed slot on any pre-launch failure below, unless the
	// goroutine is launched (it then owns teardown via its deferred release). This
	// mirrors RespawnSession: releasing may empty the run registry, so drop it if
	// so (a StartRun failure would otherwise leak the run's empty registry +
	// allocator). AddChatbot on an already-active run leaves it non-empty, so
	// RemoveIfEmpty is a harmless no-op there.
	launched := false
	defer func() {
		if launched {
			return
		}
		release()
		if m.runReg.RemoveIfEmpty(runID) {
			m.removeAllocator(runID)
		}
	}()
	env := &agent.Env{
		Store:       m.store,
		RunID:       runID,
		LLM:         provider,
		Registry:    registry,
		Names:       m.allocatorFor(runID),
		Workspace:   ws,
		Broadcaster: m.broadcaster,
		Inflight:    m.ephemeral,
		SkillIndex:  m.skillIndex,
		LessonIndex: m.lessonIndex,
		SystemFast:  fast,
		SystemHQ:    hq,
	}
	factory, err := agent.Get(agentType)
	if err != nil {
		return err
	}
	ag, err := factory(env, agentCfg)
	if err != nil {
		return fmt.Errorf("create agent: %w", err)
	}
	launched = true
	m.runSessionGoroutine(runCtx, runID, agentCfg.SessionID, ag, release)
	return nil
}

// AddChatbot attaches an interactive chatbot session to an existing run — a
// sidecar co-pilot for an autonomous run. Idempotent: if the chatbot session
// already exists it returns its id without launching a second one. The chatbot
// is created with no task/opening message, so it parks immediately and wakes on
// the operator's first message (via the usual respawn-on-input path).
func (m *RunManager) AddChatbot(ctx context.Context, runID string) (string, error) {
	m.chatbotMu.Lock()
	defer m.chatbotMu.Unlock()

	run, err := m.store.GetRun(ctx, runID)
	if err != nil {
		return "", fmt.Errorf("get run: %w", err)
	}
	sid := config.ChatbotSessionID
	// Idempotent: already live (a turn in flight) or already persisted (parked,
	// revived on the next message). The fixed session id + PK is the ultimate
	// guard; these checks just avoid a wasted second launch on the common paths.
	if m.runReg.GetOrCreate(runID).IsRegistered(sid) {
		return sid, nil
	}
	if _, err := m.store.GetSession(ctx, runID, sid); err == nil {
		return sid, nil
	}

	provider, err := m.providerFor(run.Config.LLM)
	if err != nil {
		return "", fmt.Errorf("resolve llm %q: %w", run.Config.LLM, err)
	}
	ws := m.workspaceFactory(run.Config.Workspace)
	agentCfg := &agent.Config{SessionID: sid} // no task/first message → parks
	if err := m.launchAgent(ctx, runID, config.ChatbotAgentType, provider, m.systemFast, m.systemHQ, ws, agentCfg); err != nil {
		return "", err
	}
	return sid, nil
}

// RunRegistry returns the run registry (for HTTP API, notifier).
func (m *RunManager) RunRegistry() *RunRegistry {
	return m.runReg
}

// workspaceForSession reconstructs a session's OWN persisted workspace from its
// metadata (written at bootstrap), so a respawn restores the exact workspace the
// session ran in — including a sub-agent's linked workspace. Falls back to the
// run-config-derived workspace for legacy sessions (created before per-session
// workspace persistence) or if the stored blob is unreadable.
func (m *RunManager) workspaceForSession(sess *db.SessionRecord, runCfg config.RunConfig) workspace.Workspace {
	if raw, ok := sess.Metadata[workspace.SessionMetadataKey].(string); ok && raw != "" {
		if ws, err := workspace.Unmarshal([]byte(raw)); err != nil {
			slog.Warn("respawn: workspace unmarshal failed; falling back to run config",
				"run_id", sess.RunID, "session_id", sess.SessionID, "error", err)
		} else {
			return ws
		}
	}
	return m.workspaceFactory(runCfg.Workspace)
}

// SessionStatus returns a session's current status (ok=false if it can't be
// read). Passed to NewCommitNotifier as its StatusFunc to gate environment
// revival of terminal sessions.
func (m *RunManager) SessionStatus(runID, sessionID string) (string, bool) {
	sess, err := m.store.GetSession(context.Background(), runID, sessionID)
	if err != nil || sess == nil {
		return "", false
	}
	return sess.Status, true
}

// RespawnSession re-creates an agent goroutine for a session that has no
// active goroutine (after idle timeout or server restart).
func (m *RunManager) RespawnSession(runID, sessionID string) {
	ctx := context.Background()

	sess, err := m.store.GetSession(ctx, runID, sessionID)
	if err != nil {
		slog.Error("respawn: session not found", "run_id", runID, "session_id", sessionID, "error", err)
		return
	}
	// No status gate: the notifier only calls this on an Input-class event
	// (db.IsInput), and every non-ongoing status is Input-restartable.
	run, err := m.store.GetRun(ctx, runID)
	if err != nil {
		slog.Error("respawn: run not found", "run_id", runID, "error", err)
		return
	}

	// Claim the registry slot SYNCHRONOUSLY right after GetOrCreate, BEFORE any
	// slow pre-launch work (workspace.Validate does I/O). This closes the
	// run-registry INSTANCE ORPHANING race: GetOrCreate publishes a *session.
	// Registry instance, but if it stayed EMPTY across the slow work, a concurrent
	// goroutine-exit RemoveIfEmpty would delete it from the RunRegistry map,
	// orphaning the session we're about to launch (Notify would miss it and a
	// rival respawn would spawn a second goroutine -> duplicate tool_result ->
	// 400). Registering now keeps the instance non-empty. ok=false means the slot
	// is already taken (a live goroutine or a rival respawn) -> nothing to do.
	//
	// Parent ctx is context.Background(), not the notifier's ctx: the respawned
	// session must outlive the wake that triggered it.
	registry := m.runReg.GetOrCreate(runID)
	runCtx, handle, release, ok := registry.RegisterAndContext(context.Background(), sessionID)
	if !ok {
		return // already alive (live goroutine) or claimed by a rival respawn
	}
	// Release the claimed slot on any early return below, UNLESS the goroutine is
	// launched (launched=true), in which case the goroutine's deferred release
	// owns teardown. Forgetting this would leak the slot and wedge the session.
	launched := false
	defer func() {
		if launched {
			return
		}
		release()
		// The slot may have been the run's only session; drop the now-empty run
		// registry so it doesn't leak (mirrors the goroutine-exit cleanup).
		if m.runReg.RemoveIfEmpty(runID) {
			m.removeAllocator(runID)
		}
	}()

	factory, err := agent.Get(sess.AgentType)
	if err != nil {
		slog.Error("respawn: unknown agent type", "agent_type", sess.AgentType, "error", err)
		return
	}

	provider, err := m.providerFor(run.Config.LLM)
	if err != nil {
		slog.Error("respawn: resolve llm failed", "run_id", runID, "llm", run.Config.LLM, "error", err)
		return
	}

	ws := m.workspaceForSession(sess, run.Config)
	if err := ws.Validate(ctx); err != nil {
		// A vanished/invalid workspace can't host an agent; refuse to respawn
		// (an Input would re-trigger and re-check) rather than letting the agent
		// fail confusingly deep in its first tool call.
		slog.Error("respawn: workspace invalid; not respawning",
			"run_id", runID, "session_id", sessionID, "root", ws.Root(), "error", err)
		return
	}
	env := &agent.Env{
		Store:       m.store,
		RunID:       runID,
		LLM:         provider,
		Registry:    registry,
		Names:       m.allocatorFor(runID),
		Workspace:   ws,
		Broadcaster: m.broadcaster,
		Inflight:    m.ephemeral,
		SkillIndex:  m.skillIndex,
		LessonIndex: m.lessonIndex,
		SystemFast:  m.systemFast,
		SystemHQ:    m.systemHQ,
	}
	ag, err := factory(env, &agent.Config{
		SessionID: sessionID,
		Task:      sess.Task,
		ParentID:  sess.ParentID,
		Handle:    handle, // use the slot claimed above instead of re-registering
	})
	if err != nil {
		slog.Error("respawn: create agent failed", "error", err)
		return
	}

	// Debug, not Info: a chatbot wakes on every operator message, and any
	// session bump triggers a respawn attempt; at Info this would dominate
	// the default log stream. Operators diagnosing wakeup behavior bump to
	// debug to see them.
	slog.Debug("respawning session", "run_id", runID, "session_id", sessionID)
	// The goroutine now owns the slot (its deferred release Unregisters on exit);
	// stop the bail-out release above from firing.
	launched = true
	m.runSessionGoroutine(runCtx, runID, sessionID, ag, release)
}

// RecoverRun re-spawns a run's active spine after a process restart or an
// operator resume: ongoing/awaiting sessions with no live goroutine, plus a
// crashed root. For each, it appends a RecoverEvent (an Input-class "resumed"
// marker) so the commit notifier respawns it through the normal machinery —
// EXCEPT a session whose stream is already at rest (a no-tool AssistantEvent
// tail), which is respawned directly without a RecoverEvent (the marker would
// force a redundant LLM round-trip; the goroutine's resume path finalizes it).
//
// Per-session crash artifacts (orphan tool calls, mid-LLM resume) are repaired
// by the respawned goroutine's resume path, not here. Recover does not aim at a
// single dormant session — those revive reactively via an Input.
// It returns the number of sessions it revived (0 means the run was already at
// rest, so there was nothing to resume).
func (m *RunManager) RecoverRun(ctx context.Context, runID string) (int, error) {
	sessions, err := m.store.ListSessions(ctx, runID)
	if err != nil {
		return 0, fmt.Errorf("recover: list sessions: %w", err)
	}
	reg := m.runReg.GetOrCreate(runID)
	revived := 0
	for i := range sessions {
		s := sessions[i]
		if reg.IsRegistered(s.SessionID) || !db.IsSpine(s) {
			continue
		}
		// Skip sessions whose agent type is no longer registered (e.g. runs from
		// an older binary). They can't be respawned; appending a RecoverEvent
		// would just be junk on a dead session.
		if _, err := agent.Get(s.AgentType); err != nil {
			slog.Warn("recover: skipping session with unknown agent type",
				"run_id", runID, "session_id", s.SessionID, "agent_type", s.AgentType)
			continue
		}
		atRest, err := m.streamAtRest(ctx, runID, s.SessionID)
		if err != nil {
			return revived, fmt.Errorf("recover %s: %w", s.SessionID, err)
		}
		if atRest {
			// Skip the RecoverEvent; respawn directly so the resume path
			// finalizes the at-rest stream without a redundant LLM call.
			m.RespawnSession(runID, s.SessionID)
			revived++
			continue
		}
		// The RecoverEvent is Input-class, so the commit notifier respawns this
		// cold session through the normal machinery (no direct RespawnSession
		// here — that would double-respawn).
		if _, err := m.store.AppendEvent(ctx, runID, s.SessionID, &event.RecoverEvent{
			Content: fmt.Sprintf("[RESUMED at %s]", util.FormatLocalISO(time.Now())),
		}); err != nil {
			return revived, fmt.Errorf("recover %s: append RecoverEvent: %w", s.SessionID, err)
		}
		revived++
	}
	return revived, nil
}

// streamAtRest reports whether the session's current-context tail is a no-tool
// AssistantEvent (the prior life produced its final reply but never flipped
// status).
func (m *RunManager) streamAtRest(ctx context.Context, runID, sessionID string) (bool, error) {
	events, err := m.store.GetEvents(ctx, runID, sessionID, db.EventFilter{CurrentContextOnly: true})
	if err != nil {
		return false, err
	}
	if len(events) == 0 {
		return false, nil
	}
	return event.IsNoToolAssistant(events[len(events)-1].Event), nil
}

type StartRunConfig struct {
	Title         string // explicit title; empty → derived (markdown heading) or async-generated
	FirstMessage  string // opening chat message (chat runs); seeded at step 1
	RunConfig     config.RunConfig
	RootSessionID string
}

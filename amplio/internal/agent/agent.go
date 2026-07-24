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

package agent

import (
	"context"
	"fmt"
	"sync"

	"amplio/internal/db"
	"amplio/internal/eventstream"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	"amplio/internal/session"
	"amplio/internal/skills"
	"amplio/internal/workspace"
)

// Agent is the interface that all agent types implement.
type Agent interface {
	// Run executes the agent's main loop. Blocks until the agent completes,
	// crashes, or is cancelled via context.
	Run(ctx context.Context) error

	// SessionID returns the agent's session identifier.
	SessionID() string
}

// Env holds the shared dependencies injected into every agent: the per-run
// object graph constructed from the raw RunConfig at run start, shared by
// pointer across every agent in the run. It carries only live objects, never
// raw run config — the operator's persisted AGENTS.md, for example, is read
// from the run record at bootstrap (see eventloop.bootstrap), not cached here.
// Each agent receives its own run's SessionRegistry — agents only coordinate
// with sessions in the same run.
type Env struct {
	Store       db.Store
	RunID       string
	LLM         llm.Provider
	Registry    *session.Registry      // per-run session registry
	Names       *session.NameAllocator // per-run unique session-id allocator
	Workspace   workspace.Workspace
	Broadcaster eventstream.Broadcaster // live token-stream sink (nil = none)
	SkillIndex  *skills.Index           // skill recall corpus (nil / not-built = no skill recall)
	LessonIndex *lessons.Index          // lesson recall corpus (nil / not-built = no lesson recall)
	// SystemFast and SystemHQ are the shared system-tier providers
	SystemFast llm.Provider
	SystemHQ   llm.Provider
	// Inflight advertises in-flight ephemeral background work (e.g. context
	// compaction) to the UI. Satisfied by runtime.EphemeralAgentRegistry; nil in
	// headless/tests (no UI to notify). See InflightTracker.
	Inflight InflightTracker
}

// InflightTracker advertises in-flight ephemeral background work (work with no
// session row of its own — e.g. context compaction, the run-report critic) so a
// live UI can show a progress indicator. Register returns an id; pair it with a
// deferred Unregister. subject is what the work targets (a session id for
// compaction; "" for run-level work like the report) — NOT the worker's identity.
// runtime.EphemeralAgentRegistry implements this; the interface lives here so
// the eventloop can use it without importing runtime (which imports agent).
type InflightTracker interface {
	Register(runID, kind, subject string) uint64
	Unregister(id uint64) string
}

// Config holds per-agent configuration for creating a new agent.
type Config struct {
	SessionID string
	Task      string
	ParentID  string
	// FirstMessage seeds an opening user message at step 1 (chat runs). Only set
	// on initial start, never on respawn.
	FirstMessage string
	// Handle, if non-nil, is the registry slot a launcher already claimed for this
	// session via Registry.RegisterAndContext (synchronously, before spawning the
	// agent goroutine, so the run-registry instance can't be orphaned by a
	// concurrent RemoveIfEmpty). The agent's Run() uses it directly and the
	// launcher's release owns Unregister. Nil for a direct caller / test, where
	// Run() registers its own handle and owns the lifecycle.
	Handle *session.Handle
}

// Factory creates an Agent instance from an Env and Config.
type Factory func(env *Env, cfg *Config) (Agent, error)

// --- Registry ---

var (
	registryMu sync.RWMutex
	registry   = make(map[string]Factory)
)

// Register adds an agent factory under the given name.
func Register(name string, factory Factory) {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := registry[name]; exists {
		panic(fmt.Sprintf("agent type %q already registered", name))
	}
	registry[name] = factory
}

// Get returns the factory for the given agent type name.
func Get(name string) (Factory, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	f, ok := registry[name]
	if !ok {
		// Build the available-names list inline rather than calling Names() (which
		// re-takes registryMu.RLock): RWMutex RLock is not reentrant, so a second
		// RLock while a writer is waiting would deadlock.
		names := make([]string, 0, len(registry))
		for n := range registry {
			names = append(names, n)
		}
		return nil, fmt.Errorf("unknown agent type %q; available: %v", name, names)
	}
	return f, nil
}

// Names returns all registered agent type names.
func Names() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	return names
}

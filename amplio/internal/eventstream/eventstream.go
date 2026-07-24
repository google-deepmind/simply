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

// Package eventstream is the in-process liveness backbone: one process-global
// fan-out (Bus) of RunEvents to N subscribers (the SSE handlers). Each
// subscriber filters to a single run, or to all runs (the dashboard).
//
// RunEvents come in two flavors (mirrored client-side):
//   - Invalidation (every kind except StreamChunk): "persisted state may have
//     changed; refetch via the normal read APIs." Carries what changed.
//   - Ephemeral (StreamChunk only): the token-preview payload itself, never
//     persisted; only the live chat UI renders it. DB-mirroring clients ignore
//     it and instead re-read on the SessionBump that follows the LLM call.
//
// On overflow or (re)subscribe, a RefetchAll marker tells the client to resync,
// so reads are always stream-triggered — never an independent poll racing SSE.
package eventstream

// RunEvent kinds.
const (
	KindRefetchAll      = "refetch_all"
	KindSessionBump     = "session_bump"     // a session got new events → refetch them
	KindStatusChange    = "status_change"    // session status changed
	KindStepAdvanced    = "step_advanced"    // session current_step advanced
	KindSessionCreated  = "session_created"  // a new session appeared
	KindObservation     = "observation"      // a step/phase summary (etc.) was written
	KindRunUpdated      = "run_updated"      // a run's overlay (title/star/archive) changed
	KindStreamChunk     = "stream_chunk"     // ephemeral live token delta (not persisted)
	KindWorkspaceAlias  = "workspace_alias"  // a CitC workspace's resolved alias changed
	KindSysStat         = "sysstat"          // server-host system status snapshot (gcert, etc.)
	KindEphemeralAgents = "ephemeral_agents" // an in-flight non-session worker (report, compaction) started/ended
)

// RunEvent is one liveness signal. Fields are populated per kind; JSON omits
// empties so the wire stays small.
type RunEvent struct {
	Kind          string `json:"kind"`
	RunID         string `json:"run_id"`
	SessionID     string `json:"session_id,omitempty"`
	ParentID      string `json:"parent_id,omitempty"`
	AgentType     string `json:"agent_type,omitempty"`
	Step          int    `json:"step,omitempty"`
	NewStatus     string `json:"new_status,omitempty"`
	ObsKind       string `json:"obs_kind,omitempty"`
	TextDelta     string `json:"text_delta,omitempty"`
	ThoughtsDelta string `json:"thoughts_delta,omitempty"`
	Reason        string `json:"reason,omitempty"`
	// In-flight ephemeral worker signal (KindEphemeralAgents). EphemeralKind is
	// the work kind ("report" | "compaction"); Active is true on start, false on
	// end; SessionID carries the subject session ("" = run-level, e.g. report).
	// The UI updates indicators directly from these without a refetch.
	EphemeralKind string `json:"ephemeral_kind,omitempty"`
	Active        bool   `json:"active,omitempty"`
	// CitC workspace alias resolution (KindWorkspaceAlias). RunID is empty —
	// a single workspace can underlie multiple runs, so every global subscriber
	// receives it and patches any view whose workspace numeric id matches.
	User      string `json:"user,omitempty"`
	NumericID int    `json:"numeric_id,omitempty"`
	Alias     string `json:"alias,omitempty"`
	// Server system-status snapshot (KindSysStat). Opaque map so the wire and
	// the sysstat package's Snapshot struct can evolve independently — the
	// frontend just stores the latest. RunID is empty (global signal).
	SysStat map[string]any `json:"sysstat,omitempty"`
}

// Broadcaster publishes ephemeral live UI signals. The eventloop calls Chunk
// during streaming (interactive agents) and Compaction around a context compact;
// off-path uses NoOp.
type Broadcaster interface {
	Chunk(runID, sessionID string, step int, textDelta, thoughtsDelta string)
	// WorkspaceAlias announces that a CitC workspace's resolved alias changed
	// (initial resolve OR drift). Sent globally (no run id); UI patches every
	// view whose workspace numeric id matches.
	WorkspaceAlias(user string, numericID int, alias string)
	// SysStat publishes the latest server system-status snapshot (gcert today,
	// future: cpu/mem). Sent globally on change.
	SysStat(snapshot map[string]any)
}

// NoOpBroadcaster discards every signal. Default for the headless CLI and tests.
type NoOpBroadcaster struct{}

func (NoOpBroadcaster) Chunk(string, string, int, string, string) {}
func (NoOpBroadcaster) WorkspaceAlias(string, int, string)        {}
func (NoOpBroadcaster) SysStat(map[string]any)                    {}

// BusBroadcaster wraps each non-empty delta in a StreamChunk RunEvent and
// publishes it to a Bus.
type BusBroadcaster struct{ bus *Bus }

func NewBusBroadcaster(bus *Bus) *BusBroadcaster { return &BusBroadcaster{bus: bus} }

func (b *BusBroadcaster) Chunk(runID, sessionID string, step int, textDelta, thoughtsDelta string) {
	if textDelta == "" && thoughtsDelta == "" {
		return
	}
	b.bus.Publish(RunEvent{
		Kind:          KindStreamChunk,
		RunID:         runID,
		SessionID:     sessionID,
		Step:          step,
		TextDelta:     textDelta,
		ThoughtsDelta: thoughtsDelta,
	})
}

// WorkspaceAlias publishes a CitC alias-resolution signal globally. UI patches
// any view whose workspace numeric id matches.
func (b *BusBroadcaster) WorkspaceAlias(user string, numericID int, alias string) {
	b.bus.Publish(RunEvent{
		Kind:      KindWorkspaceAlias,
		User:      user,
		NumericID: numericID,
		Alias:     alias,
	})
}

// SysStat publishes a system-status snapshot globally.
func (b *BusBroadcaster) SysStat(snapshot map[string]any) {
	b.bus.Publish(RunEvent{Kind: KindSysStat, SysStat: snapshot})
}

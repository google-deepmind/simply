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
	"amplio/internal/db"
	"amplio/internal/event"
)

// RespawnFunc is called when a respawn-worthy event arrives for a session that
// has no active goroutine.
type RespawnFunc func(runID, sessionID string)

// StatusFunc reports a session's current status (ok=false if it can't be read).
// The commit notifier uses it to gate environment revival of finished sessions.
// It may be nil (then no env-revival gating is applied).
type StatusFunc func(runID, sessionID string) (status string, ok bool)

// envUnrevivableStatuses is the set of session statuses that an ENVIRONMENT
// notification ($AMPLIO_NOTIFY) must NOT revive — the *finished* states, where
// the session has stopped for good. Deliberately scoped to THIS policy. In
// particular `crashed` is the debatable member — a crash isn't a deliberate
// stop — so drop it from this set (only) if env should be allowed to nudge a
// crashed agent back.
//
// The PARKED states (idle/awaiting) are intentionally ABSENT: an idle
// interactive agent and an awaiting agent are waiting, not finished, so a
// background job SHOULD still wake them.
var envUnrevivableStatuses = map[string]bool{
	db.SessionConcluded: true,
	db.SessionCancelled: true,
	db.SessionCrashed:   true,
}

// NewCommitNotifier returns a db.CommitListener that bridges committed events
// to the per-run session registries. It is the synchronous, lossless wake path
// (installed via Store.SetCommitListener):
//
//   - If the target session is live, every event wakes it (bumps its waiter).
//   - If the target session is cold, only an Input-class event (db.IsInput)
//     revives it; other events just persist.
//   - EXCEPTION: an environment notification ($AMPLIO_NOTIFY) does NOT revive a
//     cold TERMINAL session (concluded/crashed/cancelled). See the gate below.
func NewCommitNotifier(runReg *RunRegistry, respawn RespawnFunc, statusOf StatusFunc) db.CommitListener {
	return func(runID, sessionID string, evt event.Event) {
		// A ToolResultEvent is the session's OWN turn output, never an external
		// wake. Results are now appended one-by-one as each tool finishes, so
		// they commit MID-STEP while an await_event sibling is parked — waking on
		// them would make await return immediately on its own step's results
		// (the bug this guards). await waits for FUTURE external events, which
		// arrive as UserEvent / ChildResultEvent, not ToolResultEvent. (The UI
		// still updates: that rides the separate EventAppended bus, not this
		// wake path.)
		if _, ok := evt.(*event.ToolResultEvent); ok {
			return
		}
		reg := runReg.Get(runID)
		if reg != nil && reg.Notify(sessionID) {
			return // woke a live session
		}
		if respawn == nil || !db.IsInput(evt) {
			return
		}
		// Environment notifications ($AMPLIO_NOTIFY) must not resurrect a cold
		// TERMINAL state that is a deliberate stop.
		if isEnvNotification(evt) && statusOf != nil {
			if status, ok := statusOf(runID, sessionID); ok && envUnrevivableStatuses[status] {
				return // persist as Notice; leave the finished session at rest
			}
		}
		respawn(runID, sessionID)
	}
}

// isEnvNotification reports whether evt is an $AMPLIO_NOTIFY environment message
// (SenderType==environment), as opposed to an agent-to-agent send_message.
func isEnvNotification(evt event.Event) bool {
	m, ok := evt.(*event.MessageEvent)
	return ok && m.SenderType == event.SenderTypeEnvironment
}

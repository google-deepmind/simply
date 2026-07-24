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

// NewCommitNotifier returns a db.CommitListener that bridges committed events
// to the per-run session registries. It is the synchronous, lossless wake path
// (installed via Store.SetCommitListener):
//
//   - If the target session is live, every event wakes it (bumps its waiter).
//   - If the target session is cold, only an Input-class event (db.IsInput)
//     revives it; other events just persist.
func NewCommitNotifier(runReg *RunRegistry, respawn RespawnFunc) db.CommitListener {
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
		if respawn != nil && db.IsInput(evt) {
			respawn(runID, sessionID)
		}
	}
}

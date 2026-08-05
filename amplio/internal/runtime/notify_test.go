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
	"testing"

	"amplio/internal/db"
	"amplio/internal/event"
)

// TestCommitNotifier_EnvRevivalGate verifies that an environment notification
// ($AMPLIO_NOTIFY) does not respawn a cold TERMINAL session (concluded/crashed/
// cancelled), while parked sessions and non-env inputs still respawn. An empty
// RunRegistry makes every target "cold" (Get returns nil), isolating the gate.
func TestCommitNotifier_EnvRevivalGate(t *testing.T) {
	env := &event.MessageEvent{Content: "job done", SenderType: event.SenderTypeEnvironment}
	agentMsg := &event.MessageEvent{Content: "hi", SenderType: event.SenderTypeAgent}
	userMsg := &event.UserEvent{Content: "keep going"}

	tests := []struct {
		name        string
		status      string // "" = leave unset (unreadable → fail-open)
		evt         event.Event
		wantRespawn bool
	}{
		{"env→concluded suppressed", db.SessionConcluded, env, false},
		{"env→cancelled suppressed", db.SessionCancelled, env, false},
		{"env→crashed suppressed", db.SessionCrashed, env, false},
		{"env→idle revives", db.SessionIdle, env, true},
		{"env→awaiting revives", db.SessionAwaiting, env, true},
		{"env→unreadable revives (fail-open)", "", env, true},
		{"agent msg→concluded revives", db.SessionConcluded, agentMsg, true},
		{"user input→concluded revives", db.SessionConcluded, userMsg, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var respawned []string
			respawn := func(_, sid string) { respawned = append(respawned, sid) }
			statusOf := func(_, _ string) (string, bool) {
				if tc.status == "" {
					return "", false
				}
				return tc.status, true
			}
			notify := NewCommitNotifier(NewRunRegistry(), respawn, statusOf)
			notify("run", "sess", tc.evt)

			got := len(respawned) == 1
			if got != tc.wantRespawn {
				t.Errorf("respawn=%v, want %v (respawned=%v)", got, tc.wantRespawn, respawned)
			}
		})
	}
}

// A nil StatusFunc disables env-terminal gating (backward-compatible: every
// Input-class event revives a cold session, as before).
func TestCommitNotifier_NilStatusFuncNoGating(t *testing.T) {
	var respawned []string
	notify := NewCommitNotifier(NewRunRegistry(),
		func(_, sid string) { respawned = append(respawned, sid) }, nil)
	notify("run", "sess", &event.MessageEvent{Content: "x", SenderType: event.SenderTypeEnvironment})
	if len(respawned) != 1 {
		t.Errorf("nil statusOf must not gate; respawned=%v", respawned)
	}
}

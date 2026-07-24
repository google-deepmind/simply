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

package server

import (
	"context"

	"amplio/internal/db"
	"amplio/internal/eventstream"
)

// Bridge republishes store commit events onto the bus as invalidation
// RunEvents until ctx is done. Run in a goroutine for the server's lifetime.
func (s *Server) Bridge(ctx context.Context) {
	events := s.store.Events()
	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-events:
			if !ok {
				return
			}
			s.bus.Publish(toRunEvent(ev))
		}
	}
}

func toRunEvent(e db.StoreEvent) eventstream.RunEvent {
	switch x := e.(type) {
	case db.EventAppended:
		return eventstream.RunEvent{Kind: eventstream.KindSessionBump, RunID: x.RunID, SessionID: x.SessionID}
	case db.SessionStatusChanged:
		return eventstream.RunEvent{Kind: eventstream.KindStatusChange, RunID: x.RunID, SessionID: x.SessionID, NewStatus: x.NewStatus}
	case db.StepAdvanced:
		return eventstream.RunEvent{Kind: eventstream.KindStepAdvanced, RunID: x.RunID, SessionID: x.SessionID, Step: x.NewStep}
	case db.SessionCreated:
		return eventstream.RunEvent{Kind: eventstream.KindSessionCreated, RunID: x.RunID, SessionID: x.SessionID, ParentID: x.ParentID, AgentType: x.AgentType}
	case db.ObservationAppended:
		ev := eventstream.RunEvent{Kind: eventstream.KindObservation, RunID: x.RunID, SessionID: x.SessionID, ObsKind: x.Kind}
		if x.Step != nil {
			ev.Step = *x.Step
		}
		return ev
	case db.RunUpdated:
		return eventstream.RunEvent{Kind: eventstream.KindRunUpdated, RunID: x.RunID}
	default:
		// Unknown event type → tell everyone to resync rather than miss a change.
		return eventstream.RunEvent{Kind: eventstream.KindRefetchAll, Reason: "unknown store event"}
	}
}

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

package db

// StoreEvent is emitted on the Store's event channel after each successful
// write commit. Consumers (SessionRegistry, EventBus) receive these via a
// channel — no locks, no callbacks, pure goroutine communication.
type StoreEvent interface {
	storeEventTag()
}

type SessionCreated struct {
	RunID     string
	SessionID string
	ParentID  string
	AgentType string
}

func (SessionCreated) storeEventTag() {}

type EventAppended struct {
	RunID     string
	SessionID string
}

func (EventAppended) storeEventTag() {}

type StepAdvanced struct {
	RunID     string
	SessionID string
	NewStep   int
}

func (StepAdvanced) storeEventTag() {}

type SessionStatusChanged struct {
	RunID     string
	SessionID string
	NewStatus string
}

func (SessionStatusChanged) storeEventTag() {}

type ObservationAppended struct {
	RunID     string
	Kind      string
	SessionID string
	Step      *int
}

func (ObservationAppended) storeEventTag() {}

// RunUpdated signals a change to a run's editable overlay (title/note/starred/
// grade/archived) or its cached critic report_grade. Run-level (no session);
// consumers refetch run summaries.
type RunUpdated struct {
	RunID string
}

func (RunUpdated) storeEventTag() {}

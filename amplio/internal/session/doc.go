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

// Package session provides a per-run in-memory registry for session coordination.
//
// Each run has its own Registry mapping sessionID to Handle. Agents only hold
// a reference to their own run's registry — no composite keys, no cross-run
// concerns.
//
// Notification uses a counter + condition pattern (see Waiter): capture the
// counter with Handle.Counter, do whatever might produce events, then block in
// WaitAfter(counter). A notify landing between the capture and the wait has
// already advanced the counter, so WaitAfter returns immediately — this closes
// the classic missed-notify race without relying on channel buffering.
package session

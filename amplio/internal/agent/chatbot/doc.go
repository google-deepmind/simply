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

// Package chatbot provides an interactive per-run chatbot agent: the root of a
// chat-driven run, or a co-pilot sidecar attached to an autonomous run. It is
// an event-loop agent with Interactive=true — a bare no-tool turn parks idle
// waiting for the next operator message instead of concluding.
//
// The system prompt adapts to the chatbot's role in the run: as the run's only
// worker (root) it reads as the primary collaborator; alongside an autonomous
// `main-agent` (sidecar) it gains shared-workspace caution. Role is derived from
// the run's sessions, so it's correct on fresh start, on attach, and on respawn.
package chatbot

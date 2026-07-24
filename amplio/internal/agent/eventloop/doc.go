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

// Package eventloop implements the canonical LLM-in-a-loop agent.
//
// Step model invariant: the AssistantEvent is the LAST event of a step.
// The step counter bumps BEFORE LLM generation. The AssistantEvent and its
// tool results are written at the previous step (the "call step"). Events
// arriving during generation land at the bumped step (next turn).
// See docs/step_model.md for the full design.
package eventloop

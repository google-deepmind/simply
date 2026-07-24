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

// Package critic generates a run's keen-critic report: an independent,
// evaluative end-of-iteration assessment of an autonomous run.
//
// The critic is a pure function (GenerateReport) driven by an in-memory
// EphemeralLoop — no session, no registry, no event readback. It combines
// mechanically-aggregated facts
// (phases, artifacts, struggles, read from observations) with the LLM's judgment
// (summary, achievements, failures, parsed from the loop's final JSON turn) and
// stores each iteration as a versioned run_report observation.
//
// Triggering and idempotency live in Finalizer: the observer fires it when the
// main-agent concludes; the operator endpoint calls it on demand. A new report
// is produced only when the main-agent advanced past the latest report's
// watermark, so it is safe to call from both the live trigger and crash
// recovery.
package critic

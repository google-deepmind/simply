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

package observer

// Build-split citation vocabulary for the observer prompts.
//
// The step and phase summarizers are the LOW-LEVEL producers: the step
// summarizer is the first thing to read raw events and distill a concrete
// identifier out of them, and the phase summarizer aggregates from those. When
// their prompt examples demonstrate the environment's real identifier shapes,
// the summaries carry link-ready tokens that everything downstream (compaction,
// the critic, and — on the internal build — the UI's shortlink linkifier) copies
// verbatim. That lets the higher-level prompts stay generic ("collect
// experiment IDs") while the concrete tokens still flow through.
//
// These vars hold the neutral OSS defaults; the internal build overrides them in
// prompts_internal.go via init() with the concrete corp forms, which the
// Copybara mirror excludes so no corp identifiers ship in the OSS binary.
var (
	// stepGoodExamples are the GOOD example lines in the step-summary prompt,
	// demonstrating the identifier/path/tool vocabulary to surface.
	stepGoodExamples = `* GOOD: "Compared run 12345678 (accuracy=0.92) to baseline 87654321 (accuracy=0.75); new best result."
* GOOD: "Ran the build/test target train; passed in 12s."
* GOOD: "Retried listing /data/foo (3rd attempt, still PERMISSION_DENIED)."`

	// phaseArtifactIdentifiers names the identifier kinds the phase summarizer
	// should extract, embedded mid-sentence in the EXTRACT ARTIFACTS objective.
	phaseArtifactIdentifiers = "experiment IDs/issue IDs, commit hashes"

	// artifactKindDesc and artifactValueDesc are the phaseArtifact schema field
	// descriptions: the kind-tag vocabulary and the canonical short form the
	// phase summarizer should emit for each artifact. They are injected into the
	// reflected schema by phaseSummarySchema (a static struct tag can't hold a
	// build-split value). These neutral OSS defaults keep artifacts generic;
	// prompts_internal.go swaps in the concrete corp forms so the internal build
	// restores the canonical short forms.
	artifactKindDesc  = "Kind tag: experiment | issue | path | url | metric | commit | etc."
	artifactValueDesc = "The artifact in canonical short form (key=value)"
)

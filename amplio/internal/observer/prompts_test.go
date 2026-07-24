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

import (
	"strings"
	"testing"
)

// TestPromptsBuildInvariants asserts the build-split citation vocabulary is
// wired into both observer prompts on every build: the step prompt embeds the
// GOOD examples fragment and the phase prompt embeds the artifact-identifier
// fragment. The concrete-vs-neutral vocabulary is asserted per build in
// prompts_oss_test.go / prompts_internal_test.go.
func TestPromptsBuildInvariants(t *testing.T) {
	step := stepSystemPrompt()
	if !strings.Contains(step, stepGoodExamples) {
		t.Errorf("stepSystemPrompt does not embed stepGoodExamples fragment")
	}
	if !strings.Contains(step, "EXAMPLES:") || !strings.Contains(step, "BAD:") {
		t.Errorf("stepSystemPrompt lost its EXAMPLES/BAD scaffolding")
	}
	phase := phaseSystemPrompt()
	if !strings.Contains(phase, phaseArtifactIdentifiers) {
		t.Errorf("phaseSystemPrompt does not embed phaseArtifactIdentifiers fragment")
	}
	if !strings.Contains(phase, "EXTRACT ARTIFACTS") {
		t.Errorf("phaseSystemPrompt lost its EXTRACT ARTIFACTS objective")
	}
}

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

//go:build !internal

package critic

import (
	"strings"
	"testing"
)

// TestCriticPromptNoCorpIdentifiersOSS guards the OSS build: the run-report
// system prompt must not carry 1P identifier shapes. The concrete forms live
// only behind the internal build tag (prompts_internal.go), asserted in
// prompts_internal_test.go.
func TestCriticPromptNoCorpIdentifiersOSS(t *testing.T) {
	prompt := criticSystemPrompt()
	for _, id := range []string{"xid/", "cl/", "/cns/"} {
		if strings.Contains(prompt, id) {
			t.Errorf("critic system prompt leaks corp identifier %q on OSS build", id)
		}
	}
	// The placeholder must have been substituted (not shipped literally).
	if strings.Contains(prompt, "__CITATION_CONVENTIONS__") {
		t.Error("critic system prompt still contains the __CITATION_CONVENTIONS__ placeholder")
	}
}

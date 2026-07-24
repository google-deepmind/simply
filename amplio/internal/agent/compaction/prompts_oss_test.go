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

package compaction

import (
	"strings"
	"testing"
)

// TestCompactionPromptNoCorpIdentifiersOSS guards the OSS build: the compaction
// system prompt must not carry 1P identifier shapes. The concrete forms live
// only behind the internal build tag (prompts_internal.go).
func TestCompactionPromptNoCorpIdentifiersOSS(t *testing.T) {
	prompt := systemPrompt()
	if strings.Contains(prompt, "xid") || strings.Contains(prompt, "cl/") {
		t.Errorf("compaction system prompt leaks corp identifier on OSS build:\n%s", prompt)
	}
	if strings.Contains(prompt, "__ARTIFACT_ID_EXAMPLES__") {
		t.Error("compaction system prompt still contains the __ARTIFACT_ID_EXAMPLES__ placeholder")
	}
}

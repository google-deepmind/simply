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

package observer

import (
	"strings"
	"testing"
)

// TestPromptsNoCorpIdentifiersOSS guards the OSS build: the observer prompts
// must not carry 1P identifier shapes. The concrete forms live only behind the
// internal build tag (prompts_internal.go), asserted in prompts_internal_test.go.
func TestPromptsNoCorpIdentifiersOSS(t *testing.T) {
	corp := []string{"xid/", "/cns/", "blaze", "fileutil"}
	for _, prompt := range []struct{ name, text string }{
		{"step", stepSystemPrompt()},
		{"phase", phaseSystemPrompt()},
		{"phase-schema", phaseSummarySchema()},
	} {
		for _, id := range corp {
			if strings.Contains(prompt.text, id) {
				t.Errorf("%s prompt leaks corp identifier %q on OSS build", prompt.name, id)
			}
		}
	}
}

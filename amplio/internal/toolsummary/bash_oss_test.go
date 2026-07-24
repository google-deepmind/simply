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

package toolsummary

import "testing"

// On the OSS build the corp tool tables aren't registered, so 1P CLI names fall
// back to the generic "run <prog>" verb (target extraction still applies where
// keyed on the tool name). Mirrors TestBashSummary_CorpTools on the internal build.
func TestBashSummary_CorpToolsFallBackOnOSS(t *testing.T) {
	cases := []struct {
		name, cmd, wantVerb, wantTarget string
	}{
		{"web_search", `web_search --query='Muon optimizer learning rate' --num_results=10`, "run web_search", "Muon optimizer learning rate"},
		{"blaze test", `blaze test //foo/bar/baz:widget_test --test_arg=x`, "run blaze", "test widget_test"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, tg := BashSummary(tc.cmd)
			if v != tc.wantVerb || tg != tc.wantTarget {
				t.Errorf("BashSummary(%q)\n  got  verb=%q target=%q\n  want verb=%q target=%q",
					tc.cmd, v, tg, tc.wantVerb, tc.wantTarget)
			}
		})
	}
}

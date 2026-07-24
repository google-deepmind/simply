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

package env

import "testing"

func TestDetect(t *testing.T) {
	existing := t.TempDir()
	missing := existing + "/nope"

	tests := []struct {
		name     string
		override string
		probe    string
		want     bool
	}{
		{"override yes wins over missing probe", "yes", missing, true},
		{"override true", "TRUE", missing, true},
		{"override 1", "1", missing, true},
		{"override no wins over existing probe", "no", existing, false},
		{"override false", "false", existing, false},
		{"override 0", "0", existing, false},
		{"auto: probe exists", "", existing, true},
		{"auto: probe missing", "", missing, false},
		{"unknown override falls through to probe", "maybe", existing, true},
		{"whitespace trimmed", "  yes  ", missing, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := detect(tc.override, tc.probe); got != tc.want {
				t.Errorf("detect(%q, %q) = %v, want %v", tc.override, tc.probe, got, tc.want)
			}
		})
	}
}

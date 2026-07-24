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

package main

import "testing"

func TestSanitizeTitle(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"plain", "Recursive LoC histogram report", "Recursive LoC histogram report"},
		{"code fence python", "```python\nimport os\nprint('hi')\n```", ""},
		{"code fence bare", "```\ncode\n```", ""},
		{"quoted", `"Quoted Title"`, "Quoted Title"},
		{"heading", "# Heading Title", "Heading Title"},
		{"bullet", "- bullet title", "bullet title"},
		{"backticked", "`backticked`", "backticked"},
		{"multiline takes first", "First line title\nsecond line", "First line title"},
		{"empty", "", ""},
		{"whitespace", "   \n  ", ""},
	}
	for _, c := range cases {
		if got := sanitizeTitle(c.in); got != c.want {
			t.Errorf("%s: sanitizeTitle(%q) = %q, want %q", c.name, c.in, got, c.want)
		}
	}
}

func TestSanitizeTitle_CapsLength(t *testing.T) {
	long := ""
	for range 100 {
		long += "x"
	}
	if got := sanitizeTitle(long); len([]rune(got)) != 80 {
		t.Errorf("len = %d, want capped at 80", len([]rune(got)))
	}
}

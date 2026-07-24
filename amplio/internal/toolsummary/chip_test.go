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

package toolsummary

import "testing"

func TestToolDetail(t *testing.T) {
	cases := []struct{ name, args, want string }{
		{"await_event", `{"timeout":60}`, "60s"},
		{"await_event", `{}`, "5m"},
		{"await_event", `{"timeout":90}`, "90s"},
		{"await_event", `{"timeout":120}`, "2m"},
		{"send_message", `{"session_id":"worker-3","content":"hi"}`, "→ worker-3"},
		{"session_peek", `{"session_id":"worker-3"}`, "worker-3"},
		{"view_file", `{"path":"/ws/internal/server/chat.go"}`, "chat.go"},
		{"edit_file", `{"path":"foo/bar.go","edits":[]}`, "bar.go"},
		{"recall_load", `{"handle":"skill:blaze"}`, "skill:blaze"},
		{"spawn_agent", `{"task":"do a big thing","agent_type":"critic"}`, "critic"},
		{"session_search", `{"query":"deadline exceeded"}`, `"deadline exceeded"`},
		{"bash", `{"command":"rg -n foo | sort"}`, ""},
		{"view_file", `not json`, ""},
		{"view_file", ``, ""},
	}
	for _, c := range cases {
		if got := ToolDetail(c.name, c.args); got != c.want {
			t.Errorf("ToolDetail(%q, %q) = %q, want %q", c.name, c.args, got, c.want)
		}
	}
}

func TestBrief(t *testing.T) {
	cases := []struct{ name, args, want string }{
		// bash → verb + target from BashSummary.
		{"bash", `{"command":"grep -n needle file.go"}`, "search needle"},
		{"bash", `{"command":"cd /x && ls foo"}`, "list foo"},
		// non-bash with a concise detail → "name detail".
		{"view_file", `{"path":"foo/bar.go"}`, "view_file bar.go"},
		{"send_message", `{"session_id":"worker-3"}`, "send_message → worker-3"},
		// non-bash without a concise detail → just the tool name.
		{"await_event", `{"timeout":0}`, "await_event 5m"},
		{"recall_load", `{}`, "recall_load"},
		{"spawn_agent", `not json`, "spawn_agent"},
	}
	for _, c := range cases {
		if got := Brief(c.name, c.args); got != c.want {
			t.Errorf("Brief(%q, %q) = %q, want %q", c.name, c.args, got, c.want)
		}
	}
}

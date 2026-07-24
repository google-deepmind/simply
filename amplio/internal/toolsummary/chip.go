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

// Package toolsummary derives short, human-meaningful {verb, target} briefs
// from a tool call's name and JSON arguments. It is the shared heuristic behind
// the chat-UI tool-call chips and the session-summary open-phase trace, kept in
// a low-level package so both internal/server and internal/tool/inspect can use
// it without an import cycle.
package toolsummary

import (
	"encoding/json"
	"fmt"
	"path"
)

// ToolChip returns the {verb, detail} for a tool-call chip. bash commands are
// too varied for a single fixed field, so we derive a {verb, target} from the
// command itself (see BashSummary). Other tools keep their tool name as the
// label (verb stays "") and use ToolDetail for the object.
func ToolChip(name, args string) (verb, detail string) {
	if name == "bash" {
		var a struct {
			Command string `json:"command"`
		}
		if json.Unmarshal([]byte(args), &a) == nil {
			return BashSummary(a.Command)
		}
		return "", ""
	}
	return "", ToolDetail(name, args)
}

// ToolDetail extracts a short, human-meaningful summary from a tool call's JSON
// arguments, for the chat chip (e.g. "view_file foo.go"). It returns "" for
// tools without a concise field (notably bash, whose commands are too varied).
// Parsing is best-effort: malformed args yield "".
func ToolDetail(name, args string) string {
	if args == "" {
		return ""
	}
	parse := func(dst any) bool { return json.Unmarshal([]byte(args), dst) == nil }
	switch name {
	case "await_event":
		var a struct {
			Timeout float64 `json:"timeout"`
		}
		if !parse(&a) {
			return ""
		}
		if a.Timeout <= 0 {
			a.Timeout = 300 // default
		}
		return fmtDurSecs(a.Timeout)
	case "send_message":
		var a struct {
			SessionID string `json:"session_id"`
		}
		if parse(&a) && a.SessionID != "" {
			return "→ " + a.SessionID
		}
	case "session_cancel", "session_summary", "session_steps", "session_peek":
		var a struct {
			SessionID string `json:"session_id"`
		}
		if parse(&a) && a.SessionID != "" {
			return a.SessionID
		}
	case "view_file", "edit_file":
		var a struct {
			Path string `json:"path"`
		}
		if parse(&a) && a.Path != "" {
			return path.Base(a.Path)
		}
	case "recall_load":
		var a struct {
			Handle string `json:"handle"`
		}
		if parse(&a) && a.Handle != "" {
			return a.Handle
		}
	case "spawn_agent":
		var a struct {
			AgentType string `json:"agent_type"`
		}
		if parse(&a) && a.AgentType != "" {
			return a.AgentType
		}
	case "session_search", "recall_search":
		var a struct {
			Query string `json:"query"`
		}
		if parse(&a) && a.Query != "" {
			return `"` + truncRunes(a.Query, 40) + `"`
		}
	}
	return ""
}

// Brief renders a tool call as a single-line verb/target string, e.g.
// "search foo", "view_file chat.go", or just the tool name when no concise
// detail is available. It composes ToolChip's {verb, detail} into the one-line
// form used by the session-summary open-phase trace.
func Brief(name, args string) string {
	verb, detail := ToolChip(name, args)
	label := verb
	if label == "" {
		label = name
	}
	if detail == "" {
		return label
	}
	return label + " " + detail
}

// fmtDurSecs renders a whole-second duration compactly. It keeps the agent's
// own unit (seconds) for small/odd values and only collapses larger round values
// to minutes, so 60→"60s", 90→"90s", 120→"2m", 300→"5m".
func fmtDurSecs(s float64) string {
	sec := int(s)
	if sec >= 120 && sec%60 == 0 {
		return fmt.Sprintf("%dm", sec/60)
	}
	return fmt.Sprintf("%ds", sec)
}

// truncRunes shortens s to at most n runes, appending an ellipsis if cut.
func truncRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

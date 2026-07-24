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

package util

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// --- JSON extraction ---

// ExtractJSONObject returns the outermost {...} JSON object found in LLM output
// text, or "" if none. It handles ```json fences (trying each fenced block as
// well as the whole text) and slices from the first "{" to the last "}". It does
// NOT validate that the slice parses — callers unmarshal into their own type.
func ExtractJSONObject(text string) string {
	candidates := []string{text}

	if strings.Contains(text, "```") {
		chunks := strings.Split(text, "```")
		for i, chunk := range chunks {
			if i%2 == 1 { // inside a fence
				cleaned := strings.TrimSpace(chunk)
				lower := strings.ToLower(cleaned)
				if strings.HasPrefix(lower, "json") {
					cleaned = strings.TrimSpace(cleaned[4:])
				}
				candidates = append(candidates, cleaned)
			}
		}
	}

	for _, candidate := range candidates {
		s := strings.TrimSpace(candidate)
		if s == "" {
			continue
		}
		start := strings.Index(s, "{")
		end := strings.LastIndex(s, "}")
		if start < 0 || end <= start {
			continue
		}
		// Prefer a candidate that actually parses; fall back to the first
		// syntactically-bracketed slice so callers still get something to try.
		slice := s[start : end+1]
		if json.Valid([]byte(slice)) {
			return slice
		}
	}
	// No candidate parsed cleanly; return the first bracketed slice (if any) so a
	// caller's typed Unmarshal can surface a precise error.
	for _, candidate := range candidates {
		s := strings.TrimSpace(candidate)
		start := strings.Index(s, "{")
		end := strings.LastIndex(s, "}")
		if start >= 0 && end > start {
			return s[start : end+1]
		}
	}
	return ""
}

// ParseJSONObject extracts a JSON object from LLM output text and unmarshals it
// into a map. Handles ```json fences and locates the outermost {...} block.
// Returns nil if no valid JSON object is found.
func ParseJSONObject(text string) map[string]any {
	js := ExtractJSONObject(text)
	if js == "" {
		return nil
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(js), &obj); err != nil {
		return nil
	}
	return obj
}

// --- Time formatting ---

// FormatLocalISO renders a time in the local timezone as ISO 8601 with offset.
// If t is zero, uses time.Now().
func FormatLocalISO(t time.Time) string {
	if t.IsZero() {
		t = time.Now()
	}
	return t.Local().Format(time.RFC3339)
}

// TruncateRunes shortens s to at most n runes, appending an ellipsis if cut.
func TruncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// FormatRelativeTime renders a past time as a compact "Xs/Xm/Xh/Xd ago" string.
// Returns "never" for zero time.
func FormatRelativeTime(when time.Time, now time.Time) string {
	if when.IsZero() {
		return "never"
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	delta := now.Sub(when)
	if delta < 0 {
		delta = 0
	}
	secs := int(delta.Seconds())
	if secs < 60 {
		return fmt.Sprintf("%ds ago", secs)
	}
	mins := secs / 60
	if mins < 60 {
		return fmt.Sprintf("%dm ago", mins)
	}
	hours := mins / 60
	if hours < 24 {
		return fmt.Sprintf("%dh ago", hours)
	}
	days := hours / 24
	return fmt.Sprintf("%dd ago", days)
}

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
	"testing"
	"time"
)

func TestParseJSONObject(t *testing.T) {
	tests := []struct {
		name string
		text string
		want map[string]any
	}{
		{
			name: "plain json",
			text: `{"key": "value"}`,
			want: map[string]any{"key": "value"},
		},
		{
			name: "fenced json",
			text: "```json\n{\"key\": \"value\"}\n```",
			want: map[string]any{"key": "value"},
		},
		{
			name: "fenced json with surrounding text",
			text: "Here is the result:\n```json\n{\"a\": 1}\n```\nDone.",
			want: map[string]any{"a": float64(1)},
		},
		{
			name: "json with leading text",
			text: "The answer is {\"x\": true}",
			want: map[string]any{"x": true},
		},
		{
			name: "empty string",
			text: "",
			want: nil,
		},
		{
			name: "no json",
			text: "just some text",
			want: nil,
		},
		{
			name: "malformed json",
			text: "{key: value}",
			want: nil,
		},
		{
			name: "json array not object",
			text: `[1, 2, 3]`,
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseJSONObject(tt.text)
			if tt.want == nil {
				if got != nil {
					t.Errorf("want nil, got %v", got)
				}
				return
			}
			if got == nil {
				t.Fatal("want non-nil, got nil")
			}
			for k, v := range tt.want {
				if got[k] != v {
					t.Errorf("key %q: want %v, got %v", k, v, got[k])
				}
			}
		})
	}
}

func TestFormatRelativeTime(t *testing.T) {
	now := time.Date(2026, 6, 4, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name string
		when time.Time
		want string
	}{
		{"zero", time.Time{}, "never"},
		{"10 seconds", now.Add(-10 * time.Second), "10s ago"},
		{"59 seconds", now.Add(-59 * time.Second), "59s ago"},
		{"1 minute", now.Add(-60 * time.Second), "1m ago"},
		{"45 minutes", now.Add(-45 * time.Minute), "45m ago"},
		{"1 hour", now.Add(-60 * time.Minute), "1h ago"},
		{"23 hours", now.Add(-23 * time.Hour), "23h ago"},
		{"1 day", now.Add(-24 * time.Hour), "1d ago"},
		{"3 days", now.Add(-72 * time.Hour), "3d ago"},
		{"future clamps to 0s", now.Add(10 * time.Second), "0s ago"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FormatRelativeTime(tt.when, now)
			if got != tt.want {
				t.Errorf("want %q, got %q", tt.want, got)
			}
		})
	}
}

func TestFormatLocalISO(t *testing.T) {
	ts := time.Date(2026, 6, 4, 12, 30, 45, 0, time.UTC)
	got := FormatLocalISO(ts)
	if got == "" {
		t.Fatal("empty result")
	}
	// Should contain the date portion.
	if !contains(got, "2026") {
		t.Errorf("expected year 2026 in %q", got)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsImpl(s, sub))
}

func containsImpl(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

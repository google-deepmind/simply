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

package skills

import "log/slog"

// Source is a directory tree of skills to scan, with an optional per-source
// blocklist of skill names to exclude.
type Source struct {
	Name    string
	Path    string
	Blocked []string
}

// scanSources scans every source in order and merges into a flat entry list.
// Same-named skills resolve last-wins (a later source overrides an earlier one),
// so callers can layer an override dir after a base dir.
func scanSources(sources []Source) []Entry {
	merged := make(map[string]Entry)
	order := make([]string, 0) // first-seen order, stable across overrides
	for _, src := range sources {
		blocked := make(map[string]bool, len(src.Blocked))
		for _, b := range src.Blocked {
			blocked[b] = true
		}
		for _, e := range scanSkills(src.Path) {
			if blocked[e.Name] {
				slog.Info("skipping blocked skill", "name", e.Name, "source", src.Name)
				continue
			}
			if _, exists := merged[e.Name]; exists {
				slog.Info("skill overrides earlier definition", "name", e.Name, "source", src.Name)
			} else {
				order = append(order, e.Name)
			}
			merged[e.Name] = e
		}
	}
	out := make([]Entry, 0, len(order))
	for _, n := range order {
		out = append(out, merged[n])
	}
	return out
}

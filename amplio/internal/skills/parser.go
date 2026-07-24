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

// Package skills builds an in-memory similarity index over a corpus of
// SKILL.md guides so agents can recall relevant ones by description.
package skills

import (
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// Entry is a parsed SKILL.md.
type Entry struct {
	Name        string // frontmatter `name`: the canonical id + cache/handle key
	Description string // frontmatter `description`: embedded + shown as the search preview
	Body        string // markdown after the closing `---`, verbatim
	Path        string // absolute path to the SKILL.md
	ContentHash string // sha256 of the raw file bytes; invalidates one cached vector
}

const frontmatterDelim = "---"

// parseSkillMD parses one SKILL.md. Returns ok=false (with a WARN log) on any
// malformation — one bad skill must never stop the index from building.
func parseSkillMD(path string, raw []byte) (Entry, bool) {
	text := string(raw)
	if !strings.HasPrefix(text, frontmatterDelim+"\n") && !strings.HasPrefix(text, frontmatterDelim+"\r\n") {
		slog.Warn("skill: no YAML frontmatter at top", "path", path)
		return Entry{}, false
	}
	// Drop the opening delimiter line, then find the closing "\n---".
	_, rest, _ := strings.Cut(text, "\n")
	endIdx := strings.Index(rest, "\n"+frontmatterDelim)
	if endIdx < 0 {
		slog.Warn("skill: unterminated YAML frontmatter", "path", path)
		return Entry{}, false
	}
	yamlText := rest[:endIdx]
	// Body starts after the closing delimiter line; trim leading blank lines.
	afterDelim := rest[endIdx+len("\n"+frontmatterDelim):]
	if nl := strings.IndexByte(afterDelim, '\n'); nl >= 0 {
		afterDelim = afterDelim[nl+1:]
	}
	body := strings.TrimLeft(afterDelim, "\r\n")

	var meta struct {
		Name        string `yaml:"name"`
		Description string `yaml:"description"`
	}
	if err := yaml.Unmarshal([]byte(yamlText), &meta); err != nil {
		slog.Warn("skill: YAML parse error", "path", path, "error", err)
		return Entry{}, false
	}
	name := strings.TrimSpace(meta.Name)
	desc := strings.TrimSpace(meta.Description)
	if name == "" {
		slog.Warn("skill: missing name", "path", path)
		return Entry{}, false
	}
	if desc == "" {
		slog.Warn("skill: missing description", "path", path)
		return Entry{}, false
	}
	sum := sha256.Sum256(raw)
	return Entry{
		Name: name, Description: desc, Body: body, Path: path,
		ContentHash: hex.EncodeToString(sum[:]),
	}, true
}

// scanParallelism caps the SKILL.md read fan-out. The skill tree often lives on
// a network FS (depot/CitC) where per-file latency dominates and srcfs can lag
// for tens of seconds per file on cold reads, so a wide pool saturates the
// fleet of round-trips without burning CPU.
const scanParallelism = 64

// scanSkills parses every top-level <dir>/SKILL.md under root (no recursion),
// deduplicated by name (first wins, by sorted dir path). A missing/!dir root
// yields nil with a WARN (graceful on hosts without depot access).
func scanSkills(root string) []Entry {
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		slog.Warn("skill source missing or not a directory; treating as empty", "path", root)
		return nil
	}
	dirEntries, err := os.ReadDir(root)
	if err != nil {
		slog.Warn("skill source unreadable; treating as empty", "path", root, "error", err)
		return nil
	}
	var dirs []string
	for _, de := range dirEntries {
		if de.IsDir() {
			dirs = append(dirs, filepath.Join(root, de.Name()))
		}
	}
	sort.Strings(dirs) // deterministic first-wins tiebreak

	parsed := make([]*Entry, len(dirs))
	sem := make(chan struct{}, scanParallelism)
	var wg sync.WaitGroup
	for i, dir := range dirs {
		wg.Add(1)
		go func(i int, dir string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			raw, err := os.ReadFile(filepath.Join(dir, "SKILL.md"))
			if err != nil {
				return // no SKILL.md here (or unreadable) — skip
			}
			if e, ok := parseSkillMD(filepath.Join(dir, "SKILL.md"), raw); ok {
				parsed[i] = &e
			}
		}(i, dir)
	}
	wg.Wait()

	seen := make(map[string]string) // name -> path of first occurrence
	out := make([]Entry, 0, len(dirs))
	for _, e := range parsed {
		if e == nil {
			continue
		}
		if prev, dup := seen[e.Name]; dup {
			slog.Warn("duplicate skill name; keeping first", "name", e.Name, "dropped", e.Path, "kept", prev)
			continue
		}
		seen[e.Name] = e.Path
		out = append(out, *e)
	}
	return out
}

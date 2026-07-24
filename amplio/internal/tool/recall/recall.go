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

// Package recall exposes the skill + lesson knowledge corpus to agents:
// recall_search finds relevant guides, recall_load fetches one's full body.
// Hits carry typed handles (skill:<name> / lesson:<id>) so the kind is explicit
// in the trajectory and recall_load can dispatch on it. Either corpus may be
// absent; the tools surface whichever indexes are built.
package recall

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"amplio/internal/db"
	"amplio/internal/lessons"
	"amplio/internal/skills"
	"amplio/internal/tool"
	"amplio/internal/util"
)

const (
	skillPrefix  = "skill:"
	lessonPrefix = "lesson:"

	// descPreviewMax bounds each search hit's description so a wide result set
	// stays readable in the tool output.
	descPreviewMax = 300

	// initialRecallHits is how many hits per corpus the run-start seed shows —
	// intentionally terser than the recall_search default (10) since it's
	// unsolicited bootstrap context, not an explicit query.
	initialRecallHits = 5
)

func skillReady(ix *skills.Index) bool   { return ix != nil && ix.IsBuilt() }
func lessonReady(ix *lessons.Index) bool { return ix != nil && ix.IsBuilt() }

type searchParams struct {
	Query string `json:"query" jsonschema:"required" jsonschema_description:"Natural-language description of what you need to do"`
	Limit int    `json:"limit,omitempty" jsonschema_description:"Max results per corpus (default 10)"`
}

// Search returns the recall_search tool over the skill and lesson indexes
// (either may be nil/unbuilt — that corpus is skipped).
func Search(skillIx *skills.Index, lessonIx *lessons.Index) *tool.Tool {
	return &tool.Tool{
		Name: "recall_search",
		Description: "Search the skill library and the lessons mined from past runs for guides relevant to a task. " +
			"Returns handles + previews; pass a handle to recall_load to read the full guide. Use this before attempting " +
			"unfamiliar tools, internal systems, CLIs, or resource paths.",
		ParamType: &searchParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			p, errResult := tool.ParseArgs[searchParams](args)
			if errResult != nil {
				return errResult, nil
			}
			limit := p.Limit
			if limit <= 0 {
				limit = 10
			}
			var b strings.Builder
			hitCount := 0

			if skillReady(skillIx) {
				hits, err := skillIx.Search(ctx, p.Query, limit)
				if err != nil {
					return &tool.Result{Content: "Error: " + err.Error(), IsError: true}, nil
				}
				if len(hits) > 0 {
					b.WriteString("Skills:\n")
					for _, h := range hits {
						fmt.Fprintf(&b, "  %s%s — %s\n", skillPrefix, h.Entry.Name, preview(h.Entry.Description))
					}
					hitCount += len(hits)
				}
			}
			if lessonReady(lessonIx) {
				hits, err := lessonIx.Search(ctx, p.Query, limit)
				if err != nil {
					return &tool.Result{Content: "Error: " + err.Error(), IsError: true}, nil
				}
				if len(hits) > 0 {
					if b.Len() > 0 {
						b.WriteString("\n")
					}
					b.WriteString("Lessons (mined from past runs):\n")
					for _, h := range hits {
						fmt.Fprintf(&b, "  %s%s — %s: %s\n", lessonPrefix, h.Lesson.LessonID, h.Lesson.Title, preview(h.Lesson.Description))
					}
					hitCount += len(hits)
				}
			}

			if hitCount == 0 {
				return &tool.Result{Content: fmt.Sprintf("No matching skills or lessons for %q.", p.Query)}, nil
			}
			b.WriteString("\nLoad one with recall_load(handle=\"skill:<name>\" or \"lesson:<id>\").")
			return &tool.Result{Content: b.String()}, nil
		},
	}
}

type loadParams struct {
	Handle string `json:"handle" jsonschema:"required" jsonschema_description:"A handle from recall_search, e.g. \"skill:blaze\" or \"lesson:abc123\""`
}

// Load returns the recall_load tool, dispatching on the handle prefix.
func Load(skillIx *skills.Index, lessonIx *lessons.Index) *tool.Tool {
	return &tool.Tool{
		Name:        "recall_load",
		Description: "Load the full content of a skill or lesson by its handle from recall_search.",
		ParamType:   &loadParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			p, errResult := tool.ParseArgs[loadParams](args)
			if errResult != nil {
				return errResult, nil
			}
			handle := strings.TrimSpace(p.Handle)

			if name, ok := strings.CutPrefix(handle, skillPrefix); ok {
				if !skillReady(skillIx) {
					return &tool.Result{Content: "Skill recall is unavailable.", IsError: true}, nil
				}
				e, ok := skillIx.Load(name)
				if !ok {
					return &tool.Result{Content: fmt.Sprintf("No skill named %q.", name), IsError: true}, nil
				}
				return &tool.Result{Content: fmt.Sprintf("# Skill: %s\nPath: %s\n\n%s", e.Name, e.Path, e.Body)}, nil
			}
			if id, ok := strings.CutPrefix(handle, lessonPrefix); ok {
				if !lessonReady(lessonIx) {
					return &tool.Result{Content: "Lesson recall is unavailable.", IsError: true}, nil
				}
				l, ok := lessonIx.Load(id)
				if !ok {
					return &tool.Result{Content: fmt.Sprintf("No lesson with id %q.", id), IsError: true}, nil
				}
				// Usage tracking is best-effort and must not block returning the body.
				if err := lessonIx.RecordLoad(ctx, id); err != nil {
					slog.Warn("recall_load: record lesson load failed", "lesson_id", id, "error", err)
				}
				return &tool.Result{Content: formatLesson(l)}, nil
			}
			return &tool.Result{
				Content: fmt.Sprintf("Unknown handle %q; expected \"skill:<name>\" or \"lesson:<id>\".", p.Handle),
				IsError: true,
			}, nil
		},
	}
}

func formatLesson(l db.LessonRecord) string {
	src := l.SourceRunID
	if src == "" {
		src = "unknown"
	}
	return fmt.Sprintf("# Lesson: %s\n# Id: %s\n# Source run: %s\n# Score: %d (retrieved %d time(s))\n\n%s",
		l.Title, l.LessonID, src, l.Score, l.LoadedCount, l.Body)
}

// InitialContent returns a short "relevant to this task" block to seed at run
// start (for non-empty tasks) so the agent sees applicable skills/lessons
// without searching first. Returns "" on no hits / empty task.
func InitialContent(ctx context.Context, skillIx *skills.Index, lessonIx *lessons.Index, task string) string {
	if task == "" {
		return ""
	}
	var b strings.Builder
	if skillReady(skillIx) {
		if hits, err := skillIx.Search(ctx, task, initialRecallHits); err == nil && len(hits) > 0 {
			b.WriteString("Skills that may be relevant — read one with recall_load, or recall_search for more:\n")
			for _, h := range hits {
				fmt.Fprintf(&b, "  %s%s — %s\n", skillPrefix, h.Entry.Name, preview(h.Entry.Description))
			}
		}
	}
	if lessonReady(lessonIx) {
		if hits, err := lessonIx.Search(ctx, task, initialRecallHits); err == nil && len(hits) > 0 {
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString("Lessons from past runs that may be relevant:\n")
			for _, h := range hits {
				fmt.Fprintf(&b, "  %s%s — %s: %s\n", lessonPrefix, h.Lesson.LessonID, h.Lesson.Title, preview(h.Lesson.Description))
			}
		}
	}
	return b.String()
}

func preview(s string) string {
	s = strings.Join(strings.Fields(s), " ") // collapse whitespace/newlines
	// Rune-safe truncation so a multibyte description isn't cut mid-rune.
	return util.TruncateRunes(s, descPreviewMax)
}

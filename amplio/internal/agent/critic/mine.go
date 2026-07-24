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

package critic

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"amplio/internal/db"
	"amplio/internal/llm"
	"amplio/internal/llm/jsonextract"

	"github.com/invopop/jsonschema"
)

// lessonsMinedKind marks an iteration's lessons as mined (idempotency sentinel).
const lessonsMinedKind = "lessons_mined"

const (
	// maxCandidatesPerRun caps how many lessons one iteration can propose.
	maxCandidatesPerRun = 10
	// dedupSimilarityThreshold is the raw-cosine floor above which a candidate is
	// close enough to an existing lesson to warrant an LLM dedup judgment. Below
	// it, the candidate is simply inserted as new.
	dedupSimilarityThreshold = 0.9
	// maxBriefingArtifacts bounds the artifacts listed in the extraction prompt.
	maxBriefingArtifacts = 50
)

const extractionSystemPrompt = `You are mining reusable knowledge from a completed autonomous agent run. Output a single JSON object matching the schema in the user prompt — no other text.

Look for things this run discovered that future agents would want to know: workarounds for broken tools, non-obvious flags, gotchas in APIs, tips for navigating internal systems. Skip generic advice and anything specific to this run's exact task. Quality matters far more than quantity — zero candidate lessons is a fine answer if the run didn't surface anything reusable.`

const dedupSystemPrompt = `You are deciding whether a newly-extracted lesson is a duplicate of an existing one. Output a single JSON object matching the schema in the user prompt — no other text.

Be strict about "duplicate": only call it a duplicate if the existing lesson already conveys the same actionable knowledge. If the candidate adds new information, choose "supersedes" (replaces the existing lesson) or "complementary" (insert as new) instead.`

type candidateLesson struct {
	Title       string `json:"title" jsonschema_description:"Short 2-5 word noun-phrase name, e.g. 'Sub-agent task batching'"`
	Description string `json:"description" jsonschema_description:"One sentence: when is this useful — phrased as a search query a future agent might type"`
	Body        string `json:"body" jsonschema_description:"The full reusable guidance, in markdown"`
}

type extractionOutput struct {
	Lessons []candidateLesson `json:"lessons" jsonschema_description:"Reusable lessons mined from the run (0-10; empty is fine)"`
}

type dedupVerdict struct {
	Verdict string `json:"verdict" jsonschema_description:"One of: duplicate | supersedes | complementary"`
	Reason  string `json:"reason" jsonschema_description:"One sentence explaining the verdict"`
}

var extractionSchema = sync.OnceValue(func() string { return schemaOf(&extractionOutput{}) })
var dedupSchema = sync.OnceValue(func() string { return schemaOf(&dedupVerdict{}) })

func schemaOf(v any) string {
	r := &jsonschema.Reflector{DoNotReference: true}
	b, _ := json.MarshalIndent(r.Reflect(v), "", "  ") //nolint:errcheck
	return string(b)
}

// MineLessons extracts reusable lessons from an iteration's work (the report's
// delta-scoped phases + artifacts), dedups each against the existing corpus, and
// persists new (complementary) or replacement (supersedes) lessons. Returns the
// number of newly inserted lessons. Best-effort: per-candidate failures are
// logged and skipped. Requires a built lesson index (for dedup + the embedder).
func MineLessons(ctx context.Context, deps Deps, runID string, report *RunReport) (int, error) {
	if deps.LessonIndex == nil || !deps.LessonIndex.IsBuilt() {
		return 0, nil // no corpus to dedup against / store into
	}
	if len(report.Phases) == 0 {
		return 0, nil // nothing happened this iteration worth mining
	}
	candidates := proposeCandidates(ctx, deps.HQ, report)
	if len(candidates) == 0 {
		return 0, nil
	}

	embedder := deps.LessonIndex.Embedder()
	descs := make([]string, len(candidates))
	for i, c := range candidates {
		descs[i] = c.Description
	}
	vecs, err := embedder.Embed(ctx, descs)
	if err != nil {
		return 0, fmt.Errorf("embed candidates: %w", err)
	}

	inserted, changed := 0, false
	for i, c := range candidates {
		verdict, existing := "complementary", (*db.LessonRecord)(nil)
		if hits := deps.LessonIndex.SearchVec(vecs[i], 1); len(hits) > 0 && hits[0].Cosine >= dedupSimilarityThreshold {
			existing = &hits[0].Lesson
			verdict = decideDedup(ctx, deps.HQ, c, *existing)
		}
		switch verdict {
		case "duplicate":
			continue
		case "supersedes":
			if err := deps.Store.UpdateLesson(ctx, db.LessonRecord{
				LessonID: existing.LessonID, Title: c.Title, Description: c.Description, Body: c.Body,
				Embedding: vecs[i], EmbedderID: embedder.ModelID(),
			}); err != nil {
				slog.Warn("mine: update lesson failed", "lesson_id", existing.LessonID, "error", err)
				continue
			}
			changed = true
		default: // complementary (and any unexpected verdict)
			if err := deps.Store.InsertLesson(ctx, db.LessonRecord{
				LessonID: db.NewLessonID(), Title: c.Title, Description: c.Description, Body: c.Body,
				Embedding: vecs[i], EmbedderID: embedder.ModelID(), SourceRunID: runID,
			}); err != nil {
				slog.Warn("mine: insert lesson failed", "error", err)
				continue
			}
			inserted++
			changed = true
		}
	}

	if changed {
		if err := deps.LessonIndex.Build(ctx); err != nil {
			slog.Warn("mine: lesson index rebuild failed", "run_id", runID, "error", err)
		}
	}
	return inserted, nil
}

// proposeCandidates runs the extraction LLM call and returns up to
// maxCandidatesPerRun non-empty candidates ([] on any failure).
func proposeCandidates(ctx context.Context, hq llm.Provider, report *RunReport) []candidateLesson {
	resp, err := hq.Call(ctx, llm.Request{
		SystemPrompt: extractionSystemPrompt,
		Messages:     []llm.Message{{Role: llm.RoleUser, Content: buildExtractionPrompt(report)}},
	})
	if err != nil {
		slog.Warn("mine: extraction call failed", "error", err)
		return nil
	}
	out, err := jsonextract.Extract[extractionOutput](ctx, resp.Content, jsonextract.Options{
		Repair: hq,
		Hint:   "a single JSON object with a \"lessons\" array",
	})
	if err != nil {
		slog.Warn("mine: extraction parse failed", "error", err)
		return nil
	}
	valid := out.Lessons[:0]
	for _, c := range out.Lessons {
		if strings.TrimSpace(c.Title) != "" && strings.TrimSpace(c.Description) != "" && strings.TrimSpace(c.Body) != "" {
			valid = append(valid, c)
		}
	}
	if len(valid) > maxCandidatesPerRun {
		valid = valid[:maxCandidatesPerRun]
	}
	return valid
}

func buildExtractionPrompt(report *RunReport) string {
	var b strings.Builder
	b.WriteString("Mine candidate lessons from the run below. Output a single JSON object matching this schema:\n")
	b.WriteString(extractionSchema())
	b.WriteString("\n\n=== ORIGINAL TASK ===\n")
	if strings.TrimSpace(report.Task) == "" {
		b.WriteString("(interactive run — no fixed task)\n")
	} else {
		b.WriteString(report.Task + "\n")
	}
	b.WriteString("\n=== PHASES ===\n")
	for _, p := range report.Phases {
		fmt.Fprintf(&b, "- %s (session=%s steps %d-%d): %s\n", p.Title, p.SessionID, p.StartStep, p.EndStep, p.Summary)
	}
	b.WriteString("\n=== KEY ARTIFACTS ===\n")
	n := 0
	for kind, arts := range report.ArtifactsByKind {
		for _, a := range arts {
			if n >= maxBriefingArtifacts {
				break
			}
			fmt.Fprintf(&b, "- [%s] %s (%s)\n", kind, a.Value, a.Context)
			n++
		}
	}
	return b.String()
}

// decideDedup asks the LLM whether the candidate duplicates / supersedes /
// complements the existing lesson. Defaults to "complementary" on any failure.
func decideDedup(ctx context.Context, hq llm.Provider, candidate candidateLesson, existing db.LessonRecord) string {
	resp, err := hq.Call(ctx, llm.Request{
		SystemPrompt: dedupSystemPrompt,
		Messages:     []llm.Message{{Role: llm.RoleUser, Content: buildDedupPrompt(candidate, existing)}},
	})
	if err != nil {
		slog.Warn("mine: dedup call failed", "error", err)
		return "complementary"
	}
	v, err := jsonextract.Extract[dedupVerdict](ctx, resp.Content, jsonextract.Options{
		Repair: hq,
		Hint:   "a single JSON object with a string \"verdict\" field",
	})
	if err != nil {
		return "complementary"
	}
	switch v.Verdict {
	case "duplicate", "supersedes", "complementary":
		return v.Verdict
	default:
		return "complementary"
	}
}

func buildDedupPrompt(candidate candidateLesson, existing db.LessonRecord) string {
	var b strings.Builder
	b.WriteString("Decide whether the CANDIDATE lesson duplicates, supersedes, or complements the EXISTING lesson. ")
	b.WriteString("Output a single JSON object matching this schema:\n")
	b.WriteString(dedupSchema())
	fmt.Fprintf(&b, "\n\n=== EXISTING ===\nTitle: %s\nDescription: %s\nBody:\n%s\n", existing.Title, existing.Description, existing.Body)
	fmt.Fprintf(&b, "\n=== CANDIDATE ===\nTitle: %s\nDescription: %s\nBody:\n%s\n", candidate.Title, candidate.Description, candidate.Body)
	return b.String()
}

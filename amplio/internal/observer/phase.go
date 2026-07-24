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

package observer

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/llm"
	"amplio/internal/llm/jsonextract"

	"github.com/invopop/jsonschema"
)

const (
	phaseSummaryKind = "phase_summary"

	// phaseCharThreshold triggers a phase close once this many chars of raw
	// step events accumulate un-phased.
	phaseCharThreshold = 200_000

	// phaseCharHardCap bounds the phase summarizer's prompt: if carryover lets a
	// chunk grow past this, it's truncated at a step boundary.
	phaseCharHardCap = 800_000
)

// phaseSystemPrompt is built lazily (not a package var) so the artifact-identifier
// vocabulary reflects any build-tag override applied in an init() — package-var
// initializers run before init(), which would otherwise capture the OSS default.
var phaseSystemPrompt = sync.OnceValue(func() string {
	return `You are an expert AI behavior analyst producing a phase-level evaluation of an autonomous research agent's trajectory. You receive a chunk of raw execution events and the previous phase's summary for narrative context.

Your output MUST be a valid JSON object matching the schema in the user prompt. Output no other text, markdown, or explanation outside the JSON object.

Your objectives:

1. FIND THE BOUNDARY. Identify the FIRST point in the chunk where the current coherent unit of work ends — a distinct semantic transition: an activity shift (exploration → implementation → experiment → debugging), a milestone (a test passes, a script completes), or a hard pivot (abandoning a failing approach). If the chunk is one continuous phase with no clear transition, set end_step to the final step in the chunk. end_step MUST be a step that exists in the events.

2. EVALUATE PRODUCTIVITY & BEHAVIOR. Write a judgmental, analytical summary. Don't just narrate — assess how well the agent did: was it systematic and hypothesis-driven, or thrashing (blind guessing, repetitive debugging loops, ignoring errors, abandoning task prematurely, acknowleding gaps but not closing them)? What progress was made, what stalled? Ground every critique by citing specific step numbers, tool calls, error messages, or metric values from the events.

3. EXTRACT ARTIFACTS. Curate up to 5 high-signal, verifiable items a reviewer would need to check your summary (file paths, commands, metrics, ` + phaseArtifactIdentifiers + `), each in canonical short form.

4. ATTRIBUTE LOADED LESSONS. The user prompt lists the lessons the agent loaded via recall_load in this chunk (under "LESSONS LOADED IN THIS CHUNK"), each with the step it was loaded at. For each listed lesson, judge from the agent's SUBSEQUENT actions in this chunk whether it helped: "helpful" (the agent applied it and it advanced the work), "neutral" (no clear effect, or loaded but not yet acted on), "unhelpful" (applied but it didn't move things forward), or "harmful" (the agent followed it and got worse off). Be conservative about "harmful" — only when there's clear evidence the agent followed it and was worse off; "tried it and was still stuck" is "unhelpful". Emit one lesson_verdicts entry per listed lesson, echoing its handle verbatim. Empty list if none were listed.`
})

const forceCloseInstruction = `This is the FINAL phase of this iteration — there will be no further phases. Set end_step to the LAST step in the chunk. Do not pick an earlier boundary; no carryover is allowed.`

type phaseLessonVerdict struct {
	Handle  string `json:"handle" jsonschema_description:"The lesson handle exactly as listed under 'LESSONS LOADED IN THIS CHUNK', verbatim including the 'lesson:' prefix."`
	Verdict string `json:"verdict" jsonschema:"enum=helpful,enum=neutral,enum=unhelpful,enum=harmful" jsonschema_description:"Did this lesson help the agent in this chunk: helpful | neutral | unhelpful | harmful"`
	Reason  string `json:"reason" jsonschema_description:"One sentence of evidence from the agent's actions for the verdict"`
}

type phaseArtifact struct {
	// The kind/value descriptions are build-split (concrete 1P identifier shapes
	// internally, generic in OSS), so the struct tags carry placeholders that
	// phaseSummarySchema substitutes from artifactKindDesc / artifactValueDesc.
	// This keeps the vocabulary in one place (prompts.go / prompts_internal.go)
	// and out of a static tag that would ship verbatim to OSS.
	Kind    string `json:"kind" jsonschema_description:"__ARTIFACT_KIND_DESC__"`
	Value   string `json:"value" jsonschema_description:"__ARTIFACT_VALUE_DESC__"`
	Context string `json:"context" jsonschema_description:"One short phrase on why this artifact matters in this phase"`
}

type phaseSummaryOutput struct {
	Title          string               `json:"title" jsonschema_description:"Short 3-8 word noun phrase naming the phase"`
	Summary        string               `json:"summary" jsonschema_description:"Judgmental 3-6 sentence paragraph citing specific step numbers, tool calls, and errors"`
	EndStep        int                  `json:"end_step" jsonschema_description:"Step where this phase ends (inclusive): the first semantic boundary in the chunk, or the last step if none"`
	Artifacts      []phaseArtifact      `json:"artifacts,omitempty" jsonschema_description:"Up to 5 concrete verifiable items supporting the summary"`
	LessonVerdicts []phaseLessonVerdict `json:"lesson_verdicts,omitempty" jsonschema_description:"One entry per lesson listed under 'LESSONS LOADED IN THIS CHUNK', judging whether it helped the agent's subsequent actions"`
}

var phaseSummarySchema = sync.OnceValue(func() string {
	r := &jsonschema.Reflector{DoNotReference: true}
	b, _ := json.MarshalIndent(r.Reflect(&phaseSummaryOutput{}), "", "  ") //nolint:errcheck
	// Substitute the build-split artifact kind/value descriptions (placeholders
	// in the phaseArtifact struct tags). Done here, at schema-string build time,
	// so the internal init() override in prompts_internal.go is already applied
	// and no corp token lives in a static tag.
	s := string(b)
	s = strings.ReplaceAll(s, "__ARTIFACT_KIND_DESC__", artifactKindDesc)
	s = strings.ReplaceAll(s, "__ARTIFACT_VALUE_DESC__", artifactValueDesc)
	return s
})

// summarizePhase produces the phase_summary payload and the resolved end_step.
// On any LLM/parse failure it returns a degraded payload covering the full chunk
// so the caller still advances its cursor. end_step resolution: out-of-range LLM
// values are clamped to the last step; force-close overrides to the last step
// (no carryover).
func summarizePhase(ctx context.Context, llmHQ llm.Provider, sessionID string, startStep int, prevSummary string, records []db.EventRecord, forceClose bool) (map[string]any, int) {
	maxStep := startStep
	for _, r := range records {
		if r.Step > maxStep {
			maxStep = r.Step
		}
	}
	resp, err := llmHQ.Call(ctx, llm.Request{
		SystemPrompt: phaseSystemPrompt(),
		Messages:     []llm.Message{{Role: llm.RoleUser, Content: buildPhaseUserPrompt(sessionID, startStep, prevSummary, records, forceClose)}},
	})
	if err != nil {
		return phaseFailurePayload(startStep, maxStep, "LLM call failed: "+err.Error())
	}
	parsed, ok := parsePhaseSummary(ctx, llmHQ, strings.TrimSpace(resp.Content))
	if !ok {
		return phaseFailurePayload(startStep, maxStep, "LLM did not return a valid phase-summary JSON object")
	}
	endStep := parsed.EndStep
	if forceClose || endStep < startStep || endStep > maxStep {
		endStep = maxStep
	}
	return map[string]any{
		"title":           parsed.Title,
		"summary":         parsed.Summary,
		"start_step":      startStep,
		"end_step":        endStep,
		"artifacts":       parsed.Artifacts,
		"lesson_verdicts": parsed.LessonVerdicts,
	}, endStep
}

func buildPhaseUserPrompt(sessionID string, startStep int, prevSummary string, records []db.EventRecord, forceClose bool) string {
	maxStep := startStep
	for _, r := range records {
		if r.Step > maxStep {
			maxStep = r.Step
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Produce a phase summary for session %q covering a chunk that begins at step %d. The events span steps %d through %d. Pick an end_step within that range. Output a single JSON object matching this schema:\n", sessionID, startStep, startStep, maxStep)
	b.WriteString(phaseSummarySchema())
	b.WriteString("\n\n")
	if forceClose {
		b.WriteString(forceCloseInstruction)
		b.WriteString("\n\n")
	}
	b.WriteString("=== PREVIOUS PHASE SUMMARY (narrative anchor) ===\n")
	if prevSummary != "" {
		b.WriteString(prevSummary)
	} else {
		b.WriteString("(none — this is the first phase of the session)")
	}
	if loaded := renderLoadedLessons(records); loaded != "" {
		b.WriteString("\n\n=== LESSONS LOADED IN THIS CHUNK ===\n")
		b.WriteString("Judge each of these (lesson_verdicts), echoing its handle verbatim:\n")
		b.WriteString(loaded)
	}
	b.WriteString("\n\n=== EVENTS IN THIS CHUNK ===\n")
	b.WriteString(renderPhaseChunk(records))
	return b.String()
}

// loadedLessonHandle returns the bare lesson handle ("lesson:<id>") for a
// recall_load tool call's args, or "" for a skill handle or non-lesson call.
func loadedLessonHandle(argsJSON string) string {
	var p struct {
		Handle string `json:"handle"`
	}
	if json.Unmarshal([]byte(argsJSON), &p) != nil {
		return ""
	}
	h := strings.TrimSpace(p.Handle)
	if strings.HasPrefix(h, "lesson:") {
		return h
	}
	return "" // skills excluded; bare/other handles ignored at the prompt level
}

// renderLoadedLessons lists the lesson handles loaded via recall_load in the
// chunk, each with the step it was loaded at, deterministically (so the
// summarizer judges a known set rather than spotting loads itself). "" if none.
func renderLoadedLessons(records []db.EventRecord) string {
	var b strings.Builder
	seen := map[string]bool{}
	for _, r := range records {
		ae, ok := r.Event.(*event.AssistantEvent)
		if !ok {
			continue
		}
		for _, tc := range ae.ToolCalls {
			if tc.Name != "recall_load" {
				continue
			}
			h := loadedLessonHandle(tc.Arguments)
			if h == "" || seen[h] {
				continue
			}
			seen[h] = true
			fmt.Fprintf(&b, "- %s (loaded at step %d)\n", h, r.Step)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// renderPhaseChunk renders records grouped by step (records are step-ordered).
func renderPhaseChunk(records []db.EventRecord) string {
	var b strings.Builder
	for i := 0; i < len(records); {
		step := records[i].Step
		var evs []event.Event
		j := i
		for j < len(records) && records[j].Step == step {
			evs = append(evs, records[j].Event)
			j++
		}
		b.WriteString(renderStep(evs, step, "in chunk"))
		b.WriteString("\n\n")
		i = j
	}
	return strings.TrimRight(b.String(), "\n")
}

// parsePhaseSummary extracts the typed phase summary, asking llmHQ to repair a
// formatting error once before giving up (the caller then degrades).
func parsePhaseSummary(ctx context.Context, llmHQ llm.Provider, raw string) (*phaseSummaryOutput, bool) {
	out, err := jsonextract.Extract[phaseSummaryOutput](ctx, raw, jsonextract.Options{
		Repair: llmHQ,
		Hint:   "a single JSON object with string fields \"title\" and \"summary\"",
	})
	if err != nil {
		return nil, false
	}
	if strings.TrimSpace(out.Title) == "" || strings.TrimSpace(out.Summary) == "" {
		return nil, false
	}
	return out, true
}

func phaseFailurePayload(startStep, endStep int, reason string) (map[string]any, int) {
	return map[string]any{
		"title":      "[phase summarization failed]",
		"summary":    reason,
		"start_step": startStep,
		"end_step":   endStep,
		"artifacts":  []phaseArtifact{},
	}, endStep
}

func phaseSummaryObsID(sessionID string, endStep int) string {
	return fmt.Sprintf("%s-%s-%d", phaseSummaryKind, sessionID, endStep)
}

// closePhases closes as many phases as are due for a session: while the un-phased
// char_sum crosses the threshold (or force-close applies for a settled session),
// it summarizes the chunk, writes the phase row, and advances last_phased_step.
// The carryover from an LLM-picked mid-chunk boundary is re-checked by the loop.
func (o *Observer) closePhases(ctx context.Context, key sessionKey) {
	sess, err := o.store.GetSession(ctx, key.runID, key.sessionID)
	if err != nil {
		o.warn(ctx, "phase: get session", key, 0, err)
		return
	}
	// Force-close the trailing phase once a session settles: the complement of
	// the crash-recovery spine, minus idle (interactive parks there constantly).
	forceClose := !db.IsSpine(*sess) && sess.Status != db.SessionIdle
	watermark := sess.LastSummarizedStep
	lastPhased := sess.LastPhasedStep

	for watermark > lastPhased {
		if ctx.Err() != nil {
			return
		}
		charSum, err := o.store.SumStepSummaryChars(ctx, key.runID, key.sessionID, lastPhased, watermark)
		if err != nil {
			o.warn(ctx, "phase: sum chars", key, lastPhased, err)
			return
		}
		if charSum < phaseCharThreshold && !forceClose {
			return // below threshold and not settled — wait for more steps
		}
		lo, hi := lastPhased+1, watermark
		recs, err := o.store.GetEvents(ctx, key.runID, key.sessionID, db.EventFilter{StartStep: &lo, EndStep: &hi})
		if err != nil {
			o.warn(ctx, "phase: get events", key, lastPhased, err)
			return
		}
		if len(recs) == 0 {
			return // defensive: char_sum>0 but no events; avoid an infinite loop
		}
		recs = truncateChunkToCap(recs, phaseCharHardCap)
		prev := o.readPrevPhaseSummary(ctx, key, lastPhased)
		payload, endStep := summarizePhase(ctx, o.llmHQ, key.sessionID, lastPhased+1, prev, recs, forceClose)
		es := endStep
		if err := o.store.AppendObservation(ctx, db.ObservationRecord{
			ObsID:     phaseSummaryObsID(key.sessionID, endStep),
			RunID:     key.runID,
			Kind:      phaseSummaryKind,
			SessionID: key.sessionID,
			Step:      &es,
			Data:      payload,
			CreatedAt: time.Now().UTC(),
		}); err != nil {
			o.warn(ctx, "phase: write summary", key, endStep, err)
			return
		}
		if err := o.store.SetLastPhasedStep(ctx, key.runID, key.sessionID, endStep); err != nil {
			o.warn(ctx, "phase: bump cursor", key, endStep, err)
			return
		}
		lastPhased = endStep // carryover: loop re-checks (endStep, watermark]
	}
}

// readPrevPhaseSummary returns the prior phase's summary prose (narrative anchor),
// or "" for the first phase.
func (o *Observer) readPrevPhaseSummary(ctx context.Context, key sessionKey, lastPhased int) string {
	if lastPhased <= 0 {
		return ""
	}
	step := lastPhased
	recs, err := o.store.GetObservations(ctx, key.runID, db.ObsFilter{
		Kind: phaseSummaryKind, SessionID: key.sessionID, StartStep: &step, EndStep: &step,
	})
	if err != nil || len(recs) == 0 {
		return ""
	}
	if s, ok := recs[0].Data["summary"].(string); ok {
		return s
	}
	return ""
}

// truncateChunkToCap keeps a whole-step prefix of records under cap chars (always
// at least the first step), so a runaway carryover can't blow the prompt size.
func truncateChunkToCap(records []db.EventRecord, capChars int) []db.EventRecord {
	if len(records) == 0 {
		return records
	}
	running, keptUpto := 0, 0
	for i := 0; i < len(records); {
		step := records[i].Step
		j, stepChars := i, 0
		for j < len(records) && records[j].Step == step {
			stepChars += len(records[j].Event.ToText())
			j++
		}
		if keptUpto > 0 && running+stepChars > capChars {
			break
		}
		running += stepChars
		keptUpto = j
		i = j
	}
	return records[:keptUpto]
}

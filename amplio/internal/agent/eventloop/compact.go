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

package eventloop

import (
	"context"
	"strings"

	"amplio/internal/agent/compaction"
	"amplio/internal/llm"
)

// maxConsecutiveCompactions bounds back-to-back compactions before the loop
// gives up and crashes (so a too-small context limit can't spin forever).
const maxConsecutiveCompactions = 3

// inflightKindCompaction is the EphemeralAgent.Kind for an in-flight context
// compaction, advertised via Env.Inflight so the UI can show a "compacting…"
// indicator on the affected session.
const inflightKindCompaction = "compaction"

// contextJudgePrompt asks the fast model to classify a provider error as a
// context-window overflow or not. Message-based detection is provider-specific
// and brittle; a semantic judge is robust across providers and wordings.
const contextJudgePrompt = `You are classifying an error returned by an LLM API call.
Answer with exactly one word — YES or NO — and nothing else.
Answer YES if the error indicates the request exceeded the model's maximum context window or token limit (e.g. "prompt is too long", "maximum context length", "too many tokens", "input is too large").
Answer NO for any other error (network/timeout, authentication, permission, rate limit, server/5xx, content filter, malformed request unrelated to length, etc.).`

// tryCompact judges whether llmErr is a context-window overflow and, if so,
// compacts the session's prior context (summary at callStep-1) and commits it.
// Returns true only when the context was actually compacted, so the caller may
// retry the same turn. All internal failures degrade to false (the caller then
// crashes via recordFailure, and recovery re-judges later).
func (a *EventLoopAgent) tryCompact(ctx context.Context, llmErr error, callStep int) bool {
	if a.env.SystemFast == nil || a.env.SystemHQ == nil {
		return false // tiers not configured — can't judge or summarize
	}
	boundaryStep := callStep - 1
	if boundaryStep < 1 {
		// Only bootstrap (system prompt + task) precedes the call: there is
		// nothing to summarize, so compaction can't shrink the context.
		return false
	}
	if !a.judgeContextExceeded(ctx, llmErr) {
		return false
	}

	a.logger.Info("context window exceeded; compacting", "boundary_step", boundaryStep)
	// Advertise the in-flight compaction to the UI (subject = this session, so a
	// session view shows the indicator in the right place). Deferred Unregister
	// survives a panic in the summarize call below.
	if a.env.Inflight != nil {
		id := a.env.Inflight.Register(a.env.RunID, inflightKindCompaction, a.cfg.SessionID)
		defer a.env.Inflight.Unregister(id)
	}

	summary, err := compaction.Compact(ctx, compaction.Deps{
		Store:     a.env.Store,
		HQ:        a.env.SystemHQ,
		RunID:     a.env.RunID,
		SessionID: a.cfg.SessionID,
	}, boundaryStep)
	if err != nil {
		a.logger.Error("compaction failed", "error", err)
		return false
	}
	if _, err := a.env.Store.CompactContext(ctx, a.env.RunID, a.cfg.SessionID, boundaryStep, summary); err != nil {
		a.logger.Error("compaction commit failed", "error", err)
		return false
	}
	a.logger.Info("compaction complete", "boundary_step", boundaryStep)
	return true
}

// judgeContextExceeded asks the fast model whether llmErr is a context-window
// overflow. A failed judge call returns false (treat as a non-context error):
// better to crash and recover (which re-judges) than to compact on a transient
// network blip.
func (a *EventLoopAgent) judgeContextExceeded(ctx context.Context, llmErr error) bool {
	resp, err := a.env.SystemFast.Call(ctx, llm.Request{
		SystemPrompt: contextJudgePrompt,
		Messages:     []llm.Message{{Role: llm.RoleUser, Content: "ERROR:\n" + llmErr.Error()}},
	})
	if err != nil {
		a.logger.Warn("context-window judge call failed; treating as non-context error", "error", err)
		return false
	}
	return isYes(resp.Content)
}

// isYes reports whether the model's reply begins with "yes".
func isYes(s string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(s)), "yes")
}

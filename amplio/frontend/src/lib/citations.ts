/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Citation parser for report claims (Key achievements / Failure modes).
//
// The keen-critic prompt (internal/agent/critic/keen_critic.md) instructs the
// LLM to format step references as:
//   - "step N"
//   - "step N-M"  (or "steps N-M")
//   - "session SID step N" / "session SID steps N-M"   (with explicit session)
//
// We parse those patterns and turn them into links to the session page with a
// ?step=N (or N-M) query — which the session page reads to expand and scroll
// to the matching step(s). Everything else (ids, file paths, metric "k=v"
// values) is returned as plain text — those don't have stable internal
// targets right now.

export type ParsedCitation =
	| { type: 'text'; raw: string }
	| { type: 'step'; raw: string; session: string; step: number }
	| { type: 'range'; raw: string; session: string; startStep: number; endStep: number };

// Tolerant whitespace, ASCII hyphen and en-dash, both `step` and `steps`,
// optional leading "session SID" prefix with EITHER ` ` or `=` as the
// separator (the LLM often picks up the `session=…` form from the briefing
// template's "(agent=…, status=…, step=…)" syntax). Anchored: the whole
// citation must match (so we don't accidentally linkify substrings of a
// bigger phrase).
const STEP_REF =
	/^(?:session\s*=?\s*(\S+?)\s+)?steps?\s*=?\s*(\d+)(?:\s*[-–]\s*(\d+))?$/i;

export function parseCitation(raw: string, defaultSession: string): ParsedCitation {
	const m = raw.trim().match(STEP_REF);
	if (!m) return { type: 'text', raw };
	const session = m[1] || defaultSession;
	const start = parseInt(m[2], 10);
	const end = m[3] ? parseInt(m[3], 10) : start;
	if (start === end) return { type: 'step', raw, session, step: start };
	return { type: 'range', raw, session, startStep: start, endStep: end };
}

// Builds the session-page URL with the ?step= param the session page reads.
// Returns null for unparseable citations so callers can fall back to plain text.
export function stepHref(runId: string, c: ParsedCitation): string | null {
	if (c.type === 'text') return null;
	const v = c.type === 'step' ? `${c.step}` : `${c.startStep}-${c.endStep}`;
	return `/runs/${runId}/sessions/${encodeURIComponent(c.session)}?step=${v}`;
}

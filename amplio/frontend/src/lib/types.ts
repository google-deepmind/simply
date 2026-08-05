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

// Mirrors the server DTOs (internal/server/dto.go) and RunEvent
// (internal/eventstream).

export interface ModelEntry {
	spec: string;
	// Server-derived short display name (internal/llm.ShortLabel). Display only:
	// `spec` is the identity and the value submitted back, and must stay visible
	// beside the label rather than be replaced by it.
	label: string;
	removable: boolean;
	// Another entry has the same provider spec, differing only by #nickname.
	// Legal (that is how you relabel an endpoint) but usually a leftover.
	duplicate: boolean;
}

export interface ModelMenu {
	default: string;
	models: ModelEntry[];
}

export interface RootInfo {
	session_id: string;
	agent_type: string;
	status: string;
	step: number;
	status_changed_at: string; // ISO; bumps only on status transitions
}

export interface RunSummary {
	run_id: string;
	task: string;
	title: string;
	starred: boolean;
	// Human grade ("garbage".. "excellent"), or null when ungraded. report_grade
	// is the cached keen-critic grade the human grade overrides; null when the
	// critic hasn't graded the run. The effective grade is grade ?? report_grade.
	grade: string | null;
	report_grade: string | null;
	archived: boolean;
	created_at: string;
	workspace: string;
	workspace_name: string;
	// Optional workspace fields, set only by backends that support naming and
	// editor links; empty otherwise.
	workspace_kind: 'citc' | 'plain' | 'jj' | 'external' | '';
	workspace_alias: string; // "" when anonymous
	workspace_numeric_id: number; // 0 when the backend has no numeric id
	cider_url: string; // "" when the workspace has no editor URL
	llm: string; // full spec; the tooltip value, and the only form worth copying
	llm_name: string; // short display label derived from llm (display only)
	roots: RootInfo[];
	// Primary-root convenience (autonomous agent if present, else chatbot root).
	root_session_id: string;
	root_status: string;
	root_step: number;
	root_status_changed_at: string; // ISO; primary root's last status transition
	root_agent_type: string; // chatbot vs standard agent — picks the row's identity icon
	session_count: number;
	// Server-computed dashboard badge: a relevant root status changed since the
	// run was last seen. Replaces the former client-side localStorage comparison.
	has_updates: boolean;
}

// RunCounts is the GET /api/runs/counts response: the dashboard banner's exact,
// global tallies over non-archived runs, independent of list pagination.
export interface RunCounts {
	active: number;
	updates: number;
}

// RunsPage is one page of the paginated GET /api/runs response. next_cursor is
// fed back as ?before to fetch the following page; empty when has_more is false.
export interface RunsPage {
	runs: RunSummary[];
	has_more: boolean;
	next_cursor: string;
}

export interface ChatToolCall {
	id: string; // tool-call id; used to fetch the call+result detail on demand
	name: string;
	verb?: string; // action label for bash (e.g. "search"); empty for other tools
	completed: boolean;
	errored?: boolean; // tool reported an error; chip renders red
	detail?: string; // short arg summary / target for the chip (e.g. a filename)
}

export interface ChatBubble {
	event_id: string;
	// operator/chatbot are the conversation; agent/environment are inbound
	// messages (peer send_message / amplio notify) projected into the chat;
	// child_result is a spawned sub-agent's terminal result posted back to the
	// parent; compaction is a context-summary seam (expandable divider).
	kind: 'operator' | 'chatbot' | 'agent' | 'environment' | 'child_result' | 'compaction';
	content: string;
	thoughts?: string;
	from?: string;
	// verdict is set only for kind==='child_result': the sub-agent's terminal
	// status (concluded | crashed | cancelled).
	verdict?: string;
	step: number;
	created_at: string;
	tool_calls: ChatToolCall[];
}

export interface PhaseCard {
	start_step: number;
	end_step: number;
	title: string;
	summary: string;
}

export interface TrajStep {
	step: number;
	summary: string;
	status_tag: string;
}

export interface TrajArtifact {
	kind: string;
	value: string;
	context: string;
}

export interface TrajLessonVerdict {
	id: string;
	title: string;
	verdict: 'helpful' | 'neutral' | 'unhelpful' | 'harmful';
	reason?: string;
}

export interface TrajPhase {
	start_step: number;
	end_step: number;
	title: string;
	summary: string;
	artifacts: TrajArtifact[];
	lesson_verdicts: TrajLessonVerdict[];
	steps: TrajStep[];
}

export interface Trajectory {
	phases: TrajPhase[];
	loose_steps: TrajStep[];
	current_step: number;
}

export interface ChatUsage {
	prompt_tokens: number;
	completion_tokens: number;
	total_tokens: number;
}

export interface ChatFeed {
	messages: ChatBubble[];
	phase_cards: PhaseCard[];
	usage: ChatUsage | null;
}

export interface SessionDTO {
	session_id: string;
	parent_id: string;
	agent_type: string;
	task: string;
	status: string;
	current_step: number;
	created_at: string;
	status_changed_at: string; // ISO; mirrors RootInfo for run-page header synthesis
	workspace?: string; // full path (tooltip); may differ per session (linked sub-agents)
	workspace_name?: string; // cheap display name
}

export interface RunDetail {
	run_id: string;
	task: string;
	title: string;
	note: string;
	starred: boolean;
	// Human grade ("garbage".. "excellent"), or null when ungraded. report_grade
	// is the cached keen-critic grade the human grade overrides; null when the
	// critic hasn't graded the run. The effective grade is grade ?? report_grade.
	grade: string | null;
	report_grade: string | null;
	archived: boolean;
	workspace: string;
	workspace_name: string;
	// Optional workspace fields; mirrors RunSummary so the run header (which
	// uses the same RunCard component) renders the pill the same way.
	workspace_kind: 'citc' | 'plain' | 'jj' | 'external' | '';
	workspace_alias: string;
	workspace_numeric_id: number;
	cider_url: string;
	llm: string; // full spec; Overview shows it verbatim
	llm_name: string; // short display label derived from llm (display only)
	agent_type: string;
	system_llm_hq: string;
	system_llm_fast: string;
	agents_md: string; // raw markdown supplied at run creation; can be long
	created_at: string;
	updated_at: string; // last metadata edit (title/note/star/archive); equals created_at when untouched
	sessions: SessionDTO[];
	// In-flight non-session workers for this run (today: the critic generating
	// a report). Empty array when nothing's running. Backend fires the
	// 'ephemeral_agents' SSE event on register/unregister so the array stays
	// in sync without polling.
	ephemeral_agents: EphemeralAgent[];
	// Server-computed dashboard badge (a relevant root status changed since last
	// seen). Mirrors RunSummary.has_updates so the run-page tab can keep its
	// favicon dot correct from detail refreshes.
	has_updates: boolean;
	// Coverage of the latest report vs. the main-agent's current step; used by
	// the run overview to decide whether the report is up-to-date, whether a
	// (small) delta was deliberately skipped by the debounce threshold, or
	// whether a substantive new iteration is pending.
	//
	//   - undefined / '': coverage is not applicable (chat run without a
	//     main-agent, or no prior report yet).
	//   - 'covered':      the latest report is at or past the current step.
	//   - 'trivial_gap':  the delta is below the finalizer's debounce
	//     threshold; no new iteration is being generated. UI shows an honest
	//     "below threshold" note instead of a spinner.
	//   - 'substantive_gap': the delta is at or above threshold; a new
	//     iteration is being (or should be) generated.
	report_coverage?: '' | 'covered' | 'trivial_gap' | 'substantive_gap';
	// Actual main-agent step delta the coverage was computed from. Only
	// meaningful for 'trivial_gap' / 'substantive_gap'.
	report_gap_steps?: number;
}

// EphemeralAgent: one in-flight background worker (no DB session, no event
// log). Today the only kind is "report" (critic generation). Future kinds
// would join this without schema change.
export interface EphemeralAgent {
	kind: string; // "report" today
	started_at: string; // ISO; UI renders elapsed-time relative to now
}

export interface EventDTO {
	step: number;
	generation: number;
	created_at: string;
	event: AgentEvent;
}

// Attachment is a tool-result binary artifact (today, view_file images). The
// bytes live in the run's blob store; the UI fetches them via blobUrl(blob_key).
export interface Attachment {
	mime_type: string;
	blob_key?: string;
	size?: number;
	source_hint?: string;
}

// AgentEvent is the marshaled typed event: a `type` discriminator plus fields.
export interface AgentEvent {
	type: string;
	content?: string;
	thoughts?: string;
	tool_calls?: { id: string; name: string; arguments: string }[];
	tool_call_id?: string;
	is_error?: boolean; // tool_result only: the tool reported a failure
	attachments?: Attachment[];
	marker?: string;
	sender?: string;
	sender_type?: string;
	child_session_id?: string;
	verdict?: string;
}

// Artifact browser (run Files tab).
export interface ArtifactEntry {
	name: string;
	is_dir: boolean;
	size: number;
	mtime: string;
}
export interface ArtifactListing {
	root: string;
	path: string;
	entries: ArtifactEntry[];
}
// A single file in a recursive (flat) artifact listing: full subpath under the
// artifact root, forward-slashed.
export interface ArtifactFile {
	path: string;
	size: number;
	mtime: string;
}
export interface ArtifactAllListing {
	root: string;
	files: ArtifactFile[];
}

// WorkspaceInfo feeds the New-Run workspace control.
export interface WorkspaceInfo {
	// Workspace sources this build offers beyond a plain path; empty when none.
	workspace_modes: string[];
	server_root: string;
	recent: string[];
}

export interface ObservationDTO {
	kind: string;
	session_id: string;
	step: number | null;
	char_count: number;
	data: Record<string, unknown>;
	created_at: string;
}

export interface CitedClaim {
	statement: string;
	citations: string[] | null;
}

export interface ReportArtifact {
	value: string;
	context: string;
	session_id: string;
	start_step: number;
	end_step: number;
}

export interface ReportStruggle {
	session_id: string;
	start_step: number;
	end_step: number;
	length: number;
	sample_summaries: string[] | null;
}

// RunReport mirrors critic.RunReport. Nil Go slices/maps marshal as null.
export interface RunReport {
	version: number;
	created_at: string;
	task: string;
	// The keen-critic's grade for THIS iteration, as an integer RANK
	// (0=ungraded, 1=garbage.. 5=excellent). Unlike the run DTO's string grade,
	// the report endpoint serializes critic.RunReport directly, so this is the
	// raw int. It is per-iteration, unlike the run-level report_grade which
	// always tracks the latest report. 0 covers reports that predate grading.
	grade: number;
	summary: string;
	key_achievements: CitedClaim[] | null;
	failure_modes: CitedClaim[] | null;
	artifacts_by_kind: Record<string, ReportArtifact[]> | null;
	struggles: ReportStruggle[] | null;
	// Per-session watermark snapshot at the time the report was generated.
	// Used by the overview to detect when the latest report is stale (main
	// agent has advanced past the recorded watermark), matching the
	// finalizer's own idempotency rule. Empty / missing on legacy reports
	// generated before watermarks were added; the staleness check treats
	// missing as "trust the report" rather than false-flagging old runs.
	sessions: RunReportSessionState[] | null;
}

export interface RunReportSessionState {
	session_id: string;
	agent_type: string;
	status: string;
	current_step: number;
}

export interface RecallSkillHit {
	handle: string;
	name: string;
	description: string;
}

export interface RecallLessonHit {
	handle: string;
	id: string;
	title: string;
	description: string;
	score: number;
	loaded_count: number;
}

export interface RecallResults {
	skills: RecallSkillHit[];
	lessons: RecallLessonHit[];
}

export interface RecallItem {
	kind: 'skill' | 'lesson';
	name?: string;
	path?: string;
	id?: string;
	title?: string;
	description?: string;
	body: string;
	score?: number;
	loaded_count?: number;
	source_run?: string;
}

export interface LessonSummary {
	id: string;
	title: string;
	description: string;
	score: number;
	loaded_count: number;
	source_run: string;
	created_at: string;
	updated_at: string;
}

export type RunEventKind =
	| 'refetch_all'
	| 'session_bump'
	| 'status_change'
	| 'step_advanced'
	| 'session_created'
	| 'observation'
	| 'run_updated'
	| 'stream_chunk'
	| 'workspace_alias'
	| 'ephemeral_agents'
	| 'sysstat';

export interface RunEvent {
	kind: RunEventKind;
	run_id: string;
	session_id?: string;
	parent_id?: string;
	agent_type?: string;
	step?: number;
	new_status?: string;
	obs_kind?: string;
	text_delta?: string;
	thoughts_delta?: string;
	reason?: string;
	// In-flight ephemeral worker signal (kind === 'ephemeral_agents'):
	// ephemeral_kind is 'report' | 'compaction'; active is true on start, false
	// on end; session_id is the subject session ('' / undefined = run-level).
	ephemeral_kind?: string;
	active?: boolean;
	// Global (empty run_id) events:
	user?: string;
	numeric_id?: number;
	alias?: string;
	sysstat?: {
		credential_seconds?: number | null;
		load_avg_1m?: number | null;
		cpu_pct?: number | null;
		mem_pct?: number | null;
		swap_pct?: number | null;
	};
}

export const TERMINAL_STATUSES = new Set(['concluded', 'crashed', 'cancelled']);

// AboutInfo is the read-only server-introspection payload for the About page.
export interface AboutInfo {
	channel: string;
	commit?: string;
	modified: boolean;
	build_time?: string;
	go_version: string;
	data_dir: string;
	config_path: string;
	logs_dir: string;
	default_llm: string;
	models: string[];
	system_llm_hq: string;
	system_llm_fast: string;
	owner: string;
	auth_on: boolean;
	caller_authed: boolean;
}

// TestLLMResult is the outcome of the About page's LLM pre-flight test.
export interface TestLLMResult {
	ok: boolean;
	model_id?: string;
	reply?: string;
	latency_ms?: number;
	error?: string;
}

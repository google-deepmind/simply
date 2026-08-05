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

import { getContext, setContext } from 'svelte';
import { page } from '$app/state';
import { api, errorText } from './api';
import type { Trajectory, TrajStep, TrajArtifact, TrajLessonVerdict } from './types';

const KEY = Symbol('logview');

// The log viewer is "pick a step range, render it two ways". A GROUP is one
// selectable range in the left column; the right pane renders it either as the
// trajectory (step rows + raw events) or as the chat transcript. Three kinds:
//
//   bootstrap — step 0 (system prompt, workspace, AGENTS.md …). Always present:
//               every session has it, and it's where the agent's setup is read.
//   phase     — one summarized phase from the observer.
//   current   — the not-yet-phased tail (a live session's newest steps).
//
// Uniform ranges are what let one selection drive both renderers and the
// ranged fetches (`from_step`/`to_step`) behind them.
export type LogGroupKind = 'bootstrap' | 'phase' | 'current';

export interface LogGroup {
	key: string; // "<start>-<end>" — also the ?phase= value
	kind: LogGroupKind;
	title: string;
	start: number;
	end: number;
	summary: string;
	artifacts: TrajArtifact[];
	lesson_verdicts: TrajLessonVerdict[];
	steps: TrajStep[];
}

export type LogMode = 'trajectory' | 'chat';

// Last session viewed per run, so the nav rail's Trajectory / Chat-log entries
// return you where you were instead of resetting to the root session. In-memory
// only — a page reload legitimately starts from the default session.
const lastSession = new Map<string, string>();
export function rememberSession(runId: string, sid: string) {
	if (runId && sid) lastSession.set(runId, sid);
}
export function recallSession(runId: string): string {
	return lastSession.get(runId) ?? '';
}

function parseRange(v: string | null): { start: number; end: number } | null {
	const m = v?.match(/^(\d+)(?:-(\d+))?$/);
	if (!m) return null;
	const start = parseInt(m[1], 10);
	const end = m[2] ? parseInt(m[2], 10) : start;
	return start <= end ? { start, end } : null;
}

// LogViewStore is owned by the session-log layout and shared with its two mode
// pages (trajectory / chat) via context. It holds the phase INDEX only — the
// per-group content is fetched by whichever page is mounted, so switching modes
// never re-fetches the index and never loses the selection.
export class LogViewStore {
	runId = $state('');
	sid = $state('');
	traj = $state<Trajectory | null>(null);
	error = $state('');

	// Which renderer is mounted, derived from the route (…/sessions/<sid>/chat).
	// Kept here so link builders in every consumer agree on it.
	mode = $derived<LogMode>(page.url.pathname.endsWith('/chat') ? 'chat' : 'trajectory');

	groups = $derived.by((): LogGroup[] => {
		const t = this.traj;
		if (!t) return [];
		const out: LogGroup[] = [
			{
				key: '0-0',
				kind: 'bootstrap',
				title: 'Bootstrap',
				start: 0,
				end: 0,
				summary: '',
				artifacts: [],
				lesson_verdicts: [],
				steps: [{ step: 0, summary: 'session bootstrap', status_tag: '' }]
			}
		];
		for (const p of t.phases) {
			out.push({
				key: `${p.start_step}-${p.end_step}`,
				kind: 'phase',
				title: p.title,
				start: p.start_step,
				end: p.end_step,
				summary: p.summary,
				artifacts: p.artifacts ?? [],
				lesson_verdicts: p.lesson_verdicts ?? [],
				steps: p.steps ?? []
			});
		}
		const loose = t.loose_steps ?? [];
		if (loose.length) {
			out.push({
				key: `${loose[0].step}-${loose[loose.length - 1].step}`,
				kind: 'current',
				title: 'Current',
				start: loose[0].step,
				end: loose[loose.length - 1].step,
				summary: '',
				artifacts: [],
				lesson_verdicts: [],
				steps: loose
			});
		}
		return out;
	});

	// Selection lives in the URL (?phase=a-b), so it survives a mode switch, a
	// reload and a paste. Resolution order:
	//   1. exact ?phase= key, else the group OVERLAPPING that range (the phase
	//      index can shift under a live session as the observer closes a phase);
	//   2. the group containing ?step=N (report citations deep-link to a step);
	//   3. the newest group — "show me where things stand".
	selected = $derived.by((): LogGroup | null => {
		const gs = this.groups;
		if (!gs.length) return null;
		const want = parseRange(page.url.searchParams.get('phase'));
		if (want) {
			const key = `${want.start}-${want.end}`;
			return (
				gs.find((g) => g.key === key) ??
				gs.find((g) => g.start <= want.end && g.end >= want.start) ??
				gs[gs.length - 1]
			);
		}
		const step = parseRange(page.url.searchParams.get('step'));
		if (step) {
			const hit = gs.find((g) => g.start <= step.start && g.end >= step.start);
			if (hit) return hit;
		}
		return gs[gs.length - 1];
	});

	// href for one group in the CURRENT mode (left-column entries).
	hrefFor(g: LogGroup): string {
		return logHref(this.runId, this.sid, this.mode, g.key);
	}

	// href for the other mode, keeping the selected group (the mode tabs).
	hrefForMode(mode: LogMode): string {
		return logHref(this.runId, this.sid, mode, this.selected?.key);
	}

	async load(runId: string, sid: string) {
		this.runId = runId;
		this.sid = sid;
		if (!runId || !sid) return;
		try {
			const t = await api.getTrajectory(runId, sid);
			// A slow response for a session we've navigated away from must not land.
			if (runId !== this.runId || sid !== this.sid) return;
			this.traj = t;
			this.error = '';
		} catch (e) {
			if (runId === this.runId && sid === this.sid) this.error = errorText(e);
		}
	}

	// reset clears the index when switching sessions, so the left column can't
	// briefly show the previous session's phases against the new session's id.
	reset(runId: string, sid: string) {
		this.runId = runId;
		this.sid = sid;
		this.traj = null;
		this.error = '';
	}
}

// logHref is the one place that knows the viewer's URL shape: the mode is a
// path segment (two distinct destinations for the nav rail) and the selected
// group a query param (survives the mode switch).
export function logHref(runId: string, sid: string, mode: LogMode, phaseKey?: string): string {
	if (!sid) return `/runs/${runId}/sessions${mode === 'chat' ? '?view=chat' : ''}`;
	const base = `/runs/${runId}/sessions/${encodeURIComponent(sid)}${mode === 'chat' ? '/chat' : ''}`;
	return phaseKey ? `${base}?phase=${phaseKey}` : base;
}

export function setLogView(store: LogViewStore) {
	setContext(KEY, store);
}

export function getLogView(): LogViewStore {
	const s = getContext<LogViewStore>(KEY);
	if (!s) throw new Error('getLogView() called outside the session-log layout');
	return s;
}

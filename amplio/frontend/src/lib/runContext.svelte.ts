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
import { api, errorText } from './api';
import { openStream } from './stream';
import type { RunDetail, RunEvent } from './types';

const KEY = Symbol('run');

// Structural events that change the shared run detail (status/step/tree/overlay,
// in-flight ephemeral workers, run-scoped observation writes).
//
// ephemeral_agents: register/unregister of in-flight non-session workers
//   (the critic generating a report). refetch picks up the new
//   `ephemeral_agents` array on RunDetail so "Generating report…" shows up
//   the moment it starts and disappears the moment it ends.
// observation: report writes land as observations (kind="run_report").
//   Refreshing detail isn't enough to pick up a new iteration — but the
//   listeners (the overview page) re-fetch /report on this signal too.
//   Keeping it in STRUCTURAL is a defense-in-depth refetch even though
//   ephemeral_agents (which fires on unregister) is the primary path.
const STRUCTURAL = new Set([
	'session_created',
	'status_change',
	'step_advanced',
	'run_updated',
	'ephemeral_agents',
	'observation',
	// A workspace alias resolving changes RunDetail (alias / editor url); the
	// run-detail page must refetch to upgrade the workspace pill to a link (the
	// dashboard already refetches on any non-ephemeral event).
	'workspace_alias',
	'refetch_all'
]);

// RunStore is the per-tab run context owned by the run-shell layout: it holds
// ONE per-run SSE subscription, keeps `detail` fresh, and fans raw events out to
// sub-views (the trajectory view subscribes for its session's bumps + token
// stream). Sub-views read `detail` instead of opening their own streams.
export class RunStore {
	runId = $state('');
	detail = $state<RunDetail | null>(null);
	error = $state('');

	#listeners = new Set<(ev: RunEvent) => void>();
	#close: (() => void) | null = null;

	// switchTo (re)points the store at a run, resubscribing its single stream.
	// The run-shell layout persists across [id] changes, so this is how we move
	// between runs without churning extra connections.
	switchTo(runId: string) {
		if (runId === this.runId && this.#close) return;
		this.#stopStream();
		this.runId = runId;
		this.detail = null;
		this.error = '';
		if (!runId) return;
		void this.refresh();
		this.#close = openStream(runId, (ev) => this.#onEvent(ev));
	}

	async refresh() {
		const id = this.runId;
		try {
			const d = await api.getRun(id);
			if (id === this.runId) this.detail = d; // ignore a response from a prior run
			this.error = '';
		} catch (e) {
			// errorText => '' when the server is unreachable (the global
			// ServerStatusBanner already covers that case; avoid double-surfacing).
			if (id === this.runId) this.error = errorText(e);
		}
	}

	// on registers a raw-event listener; returns an unsubscribe.
	on(fn: (ev: RunEvent) => void): () => void {
		this.#listeners.add(fn);
		return () => this.#listeners.delete(fn);
	}

	destroy() {
		this.#stopStream();
		this.#listeners.clear();
	}

	#onEvent(ev: RunEvent) {
		if (STRUCTURAL.has(ev.kind)) void this.refresh();
		for (const fn of this.#listeners) fn(ev);
	}

	#stopStream() {
		this.#close?.();
		this.#close = null;
	}
}

export function setRunStore(store: RunStore) {
	setContext(KEY, store);
}

export function getRunStore(): RunStore {
	return getContext(KEY);
}

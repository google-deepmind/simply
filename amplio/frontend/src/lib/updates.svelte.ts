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

// "Unread updates" tracking, per run — now SERVER-AUTHORITATIVE.
//
// Seen-state lives in the DB (Run.last_seen_at); amplio is single-operator, so
// it's a global fact, not per-device. This store is a thin client cache:
//
//   - count: the exact, global "N runs have updates" tally, from
//     GET /api/runs/counts (independent of list pagination).
//   - per-run has_updates: seeded from whatever run payloads the UI has loaded
//     (dashboard list rows carry `has_updates`; the run page seeds its own run).
//     Used by the favicon dot and the run-list badges.
//   - markSeen(runId): PATCHes the server (clears the badge globally) and
//     optimistically updates the local cache so the dot clears instantly.
//
// No localStorage, no cross-tab storage hack, no client-side timestamp compare:
// the server computes "has updates" (a relevant root status changed since
// last_seen_at) and we just render it.

import { api } from './api';
import type { RunSummary } from './types';

class UpdatesStore {
	// Exact global count of runs with updates, from /api/runs/counts.
	count = $state(0);
	// Exact global count of active (ongoing/awaiting) runs, same source.
	active = $state(0);
	// runId → has_updates, seeded from loaded run payloads. Powers the favicon
	// dot and per-row badges. Not necessarily exhaustive (only loaded runs), so
	// the global `count` above — not this map's size — is the authoritative total.
	private byRun = $state<Record<string, boolean>>({});

	/** Refresh the global counts from the server. Call on app mount and on every
	 *  (debounced) run-state SSE signal. Best-effort: a failure leaves the prior
	 *  counts (the server-unreachable banner explains the staleness). */
	async refreshCounts(): Promise<void> {
		try {
			const c = await api.getRunCounts();
			this.count = c.updates;
			this.active = c.active;
		} catch {
			// silent — serverHealth flips via the api wrapper
		}
	}

	/** Seed per-run has_updates from a batch of loaded run summaries (dashboard
	 *  list / runsStore). Merges, so paging in more runs accumulates. */
	seedFromRuns(runs: RunSummary[]) {
		const next = { ...this.byRun };
		for (const r of runs) next[r.run_id] = r.has_updates;
		this.byRun = next;
	}

	/** Seed a single run's has_updates (e.g. from a run-detail refresh), so the
	 *  favicon on a backgrounded run-page tab reflects the server's badge even
	 *  when the dashboard list never loaded that run. */
	seedOne(runId: string, hasUpdates: boolean) {
		if (this.byRun[runId] !== hasUpdates) {
			this.byRun = { ...this.byRun, [runId]: hasUpdates };
		}
	}

	/** Mark a run seen NOW: clear the badge globally (server) and locally. The
	 *  optimistic local clear makes the dot vanish without waiting for the SSE
	 *  round-trip; refreshCounts on the resulting run_updated keeps the count
	 *  honest. Idempotent. */
	markSeen(runId: string) {
		if (!runId) return;
		if (this.byRun[runId]) {
			this.byRun = { ...this.byRun, [runId]: false };
			// Optimistically decrement so the banner doesn't lag a refetch.
			if (this.count > 0) this.count -= 1;
		}
		void api.markRunSeen(runId).catch(() => {
			// On failure, a later refreshCounts/seed re-derives the truth.
		});
	}

	/** Put a run's badge back. The run page clears a badge the moment it appears
	 *  (see routes/runs/[id]/+layout.svelte), so the caller must LEAVE the run for
	 *  this to stick — RunCard navigates to the dashboard when it is the run
	 *  page's own card. */
	markUnseen(runId: string) {
		if (!runId) return;
		if (!this.byRun[runId]) {
			this.byRun = { ...this.byRun, [runId]: true };
			this.count += 1;
		}
		void api.markRunUnseen(runId).catch(() => {
			// On failure, a later refreshCounts/seed re-derives the truth.
		});
	}

	hasUpdatesById(runId: string): boolean {
		return !!this.byRun[runId];
	}

	get anyUnread(): boolean {
		return this.count > 0;
	}
}

export const updates = new UpdatesStore();

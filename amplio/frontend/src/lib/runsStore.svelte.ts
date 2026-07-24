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

// Global banner-counts driver. One process-wide subscription that keeps the
// TopBanner's "N active / N updates" counters fresh from the server's exact,
// pagination-independent counts endpoint (GET /api/runs/counts).
//
// Previously this fetched the full (capped) run list and counted client-side;
// now the server counts, so this store no longer holds any run rows — it just
// pokes `updates.refreshCounts()` on every relevant SSE signal. The dashboard
// page owns its own paginated list fetch; the run-detail page owns its detail.

import { openStream } from './stream';
import { updates } from './updates.svelte';

const REFRESH_DEBOUNCE_MS = 250;

class RunsStore {
	private initialized = false;
	private debounceTimer: ReturnType<typeof setTimeout> | undefined;
	private closeStream: (() => void) | null = null;

	/** Idempotent. Call once at app mount (root layout). */
	async init(): Promise<void> {
		if (this.initialized) return;
		this.initialized = true;
		await updates.refreshCounts();
		// SSE primes with refetch_all and emits every non-ephemeral run-state
		// signal we care about (session_bump, status_change, session_created,
		// run_updated). We ignore stream_chunk (token preview) and sysstat (its own
		// store); ephemeral_agents (report/compaction) is rare enough to refresh on.
		this.closeStream = openStream(null, (ev) => {
			if (ev.kind === 'stream_chunk' || ev.kind === 'sysstat') {
				return;
			}
			this.scheduleRefresh();
		});
	}

	destroy(): void {
		this.closeStream?.();
		if (this.debounceTimer) clearTimeout(this.debounceTimer);
	}

	private scheduleRefresh() {
		clearTimeout(this.debounceTimer);
		this.debounceTimer = setTimeout(() => void updates.refreshCounts(), REFRESH_DEBOUNCE_MS);
	}

	/** Active (ongoing/awaiting) run count — exact + global, from the server. */
	get activeCount(): number {
		return updates.active;
	}

	/** Runs with unseen updates — exact + global, from the server. */
	get unreadCount(): number {
		return updates.count;
	}
}

export const runsStore = new RunsStore();

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

import { serverHealth } from './serverHealth.svelte';
import { sysstat } from './sysstat.svelte';
import type { RunEvent } from './types';

// openStream subscribes to the SSE liveness channel for one run (or all runs
// when runID is null, for the dashboard). Returns a disposer.
//
// The stream is an open read; EventSource sends the same-origin auth cookie
// automatically, so no token is needed in the URL. Invalidation events tell the
// caller WHAT to refetch; stream_chunk carries ephemeral token deltas. The
// browser auto-reconnects, and the server primes each (re)connection with a
// refetch_all.
export function openStream(runID: string | null, onEvent: (ev: RunEvent) => void): () => void {
	const url = runID ? `/api/runs/${runID}/stream` : '/api/stream';
	const es = new EventSource(url);
	es.onopen = () => serverHealth.markOk();
	es.onerror = () => {
		// EventSource auto-reconnects; treat CONNECTING as transient (no banner
		// yet, serverHealth promotes after a brief grace period) and CLOSED as a
		// hard failure.
		if (es.readyState === EventSource.CLOSED) serverHealth.markDown();
		else serverHealth.markReconnecting();
	};
	es.onmessage = (e: MessageEvent) => {
		try {
			const ev = JSON.parse(e.data) as RunEvent;
			// Global signals are dispatched directly to their stores so every open
			// SSE delivers them uniformly (and per-route subscribers don't each
			// need to know about them). Run-scoped events still flow to onEvent.
			if (ev.kind === 'sysstat' && ev.sysstat) {
				sysstat.update(ev.sysstat);
				return;
			}
			onEvent(ev);
		} catch {
			// ignore malformed frame
		}
	};
	return () => es.close();
}

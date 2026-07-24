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

// Global server-side system-status store, kept fresh by the SSE bus.
//
// Seeded once on app mount via GET /api/sysstat, then updated by kind=sysstat
// SSE events (stream.ts dispatches directly to update() — no per-component
// polling). Future signals (cpu/mem) ride the same channel: add the field to
// SysStat below, populate it in the server-side watcher, render where wanted.

export interface SysStat {
	credential_seconds?: number | null; // auth credential remaining seconds; null/undefined = unknown or no probe (OSS)
	load_avg_1m?: number | null; // 1-minute load average
	cpu_pct?: number | null; // aggregate CPU busy %, 0-100
	mem_pct?: number | null; // (MemTotal - MemAvailable) / MemTotal × 100
	swap_pct?: number | null; // omitted/null when swap is disabled
}

class SysStatStore {
	credentialSeconds = $state<number | null>(null); // null = unknown
	credentialUpdatedAt = $state(0); // wall-clock at last credential update, for local decrement

	loadAvg1m = $state<number | null>(null);
	cpuPct = $state<number | null>(null);
	memPct = $state<number | null>(null);
	swapPct = $state<number | null>(null);

	update(snap: SysStat) {
		// credential: only bump credentialUpdatedAt when the value actually
		// changes, so the chip's local decrement keeps ticking from the last
		// server-reported value rather than resetting on every load-probe
		// snapshot.
		const c = snap.credential_seconds ?? null;
		if (c !== this.credentialSeconds) {
			this.credentialSeconds = c;
			this.credentialUpdatedAt = Date.now();
		}
		this.loadAvg1m = snap.load_avg_1m ?? null;
		this.cpuPct = snap.cpu_pct ?? null;
		this.memPct = snap.mem_pct ?? null;
		this.swapPct = snap.swap_pct ?? null;
	}
}

export const sysstat = new SysStatStore();

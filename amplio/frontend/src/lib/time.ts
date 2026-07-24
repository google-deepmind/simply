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

// Local-timezone ISO-ish absolute format (YYYY-MM-DD HH:MM:SS). Used in
// metadata cards and tooltips where users want a sortable, unambiguous
// timestamp without locale guesswork. Date.toISOString() returns UTC so
// we hand-assemble from the local components.
export function formatLocalIso(iso: string): string {
	const d = new Date(iso);
	if (isNaN(d.getTime())) return '';
	const pad = (n: number) => String(n).padStart(2, '0');
	const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
	const time = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
	return `${date} ${time}`;
}

// Compact relative time for list rows ("just now", "3m", "5h", "2d", then a
// locale date). Shared by the run card and the session rows.
export function timeAgo(iso: string): string {
	const t = new Date(iso).getTime();
	if (!t) return '';
	const s = Math.floor((Date.now() - t) / 1000);
	if (s < 60) return 'just now';
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m`;
	const h = Math.floor(m / 60);
	if (h < 24) return `${h}h`;
	const d = Math.floor(h / 24);
	if (d < 7) return `${d}d`;
	return new Date(t).toLocaleDateString();
}

// Compact elapsed duration since a timestamp, for live "running for…" tickers
// (the chat's working indicator, the overview's report-generation banner). Caller
// passes `nowMs` (usually a 1s-ticking $state) so the value updates reactively.
// Format: "Ns" under a minute, "Nm Ns" under an hour, "Nh Nm" beyond.
export function formatElapsed(sinceIso: string, nowMs: number): string {
	const t = new Date(sinceIso).getTime();
	if (!t) return '';
	const s = Math.max(0, Math.floor((nowMs - t) / 1000));
	if (s < 60) return `${s}s`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m ${s % 60}s`;
	return `${Math.floor(m / 60)}h ${m % 60}m`;
}

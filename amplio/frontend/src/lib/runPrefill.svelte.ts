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

// "New run like this" carrier. The run overflow menu (dashboard rows only) sets
// a source run's config here; the dashboard's StartRunForm reacts and pre-fills
// its fields once. This is a LIVE in-memory signal (not sessionStorage): the
// menu and the form always live on the SAME page (the item is restricted to the
// dashboard), so no cross-navigation persistence is needed — and using memory
// means a stale prefill can never resurface later. It's consumed exactly once
// (take() clears it), and any navigation clears it too (see clearRunPrefill).

// The subset of a run's config carried into a new run. Mirrors the StartRunForm
// fields (all prefilled so the user edits on top). Excludes only workspace:
// WorkspaceField derives its value from an internal mode/path and can't be
// seeded from a resolved path, and a variation almost always wants a FRESH
// workspace anyway, so the new run defaults to one.
export interface RunPrefill {
	task: string;
	title: string;
	llm: string;
	interactive: boolean;
}

const store = $state<{ pending: RunPrefill | null }>({ pending: null });

// Stash a prefill for the dashboard composer to pick up on its next reactive
// tick. Overwrites any prior pending value.
export function setRunPrefill(p: RunPrefill): void {
	store.pending = p;
}

// Reactive read of the pending prefill (null when none). StartRunForm watches
// this via $effect and consumes it with take().
export function peekRunPrefill(): RunPrefill | null {
	return store.pending;
}

// Consume-once: return the pending prefill and clear it.
export function takeRunPrefill(): RunPrefill | null {
	const p = store.pending;
	store.pending = null;
	return p;
}

// Drop any pending prefill without consuming it — called on navigation so an
// un-consumed prefill can't linger and surface on an unrelated later visit.
export function clearRunPrefill(): void {
	store.pending = null;
}

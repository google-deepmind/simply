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

// Browser tab/title helper. Every page builds its title from this so the
// app name lives in ONE place and the separator/order stays consistent
// across routes.
//
// Convention: most-specific-first, ending in the app name. This way the
// useful identifier survives when the OS truncates the tab title:
//   "swift-fox · Amplio" → readable after truncation as "swift-fox · …"
//   "Amplio · swift-fox" → truncates to "Amplio · …" (useless).
//
// Empty / null / undefined parts are skipped so callers can pass optional
// state without guarding each one.
const APP_NAME = 'Amplio';

// Browsers/OSes generally truncate tabs around 30–60 chars; a run's task
// can be 500+ chars when used as the fallback identifier. Cap the
// caller-supplied bits so we don't bloat the head + leave room for the
// "· Amplio" suffix to stay visible.
const MAX_PART_LEN = 60;

function clip(s: string): string {
	const trimmed = s.trim();
	if (trimmed.length <= MAX_PART_LEN) return trimmed;
	return trimmed.slice(0, MAX_PART_LEN - 1).trimEnd() + '…';
}

export function pageTitle(...parts: (string | null | undefined)[]): string {
	const cleaned = parts
		.filter((p): p is string => typeof p === 'string' && p.trim() !== '')
		.map(clip);
	return [...cleaned, APP_NAME].join(' · ');
}

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

// formatToolArgs renders a tool call's JSON argument string in a readable
// YAML-ish form: `key: value` per line, multiline strings as an indented block
// on the following lines (no JSON quoting/escaping noise), other values via
// compact JSON. Falls back to the raw string when it isn't valid JSON (e.g.
// a mid-completion LLM produced bad JSON).
export function formatToolArgs(raw: string): string {
	if (!raw) return '';
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch {
		return raw;
	}
	if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
		return Object.entries(parsed as Record<string, unknown>)
			.map(([k, v]) => `${k}: ${formatValue(v)}`)
			.join('\n');
	}
	return JSON.stringify(parsed, null, 2);
}

function formatValue(v: unknown): string {
	if (typeof v === 'string') {
		if (v.includes('\n')) {
			// Block scalar: each line indented two spaces under the key.
			return '\n' + v.replace(/\n$/, '').split('\n').map((line) => '  ' + line).join('\n');
		}
		return v;
	}
	return JSON.stringify(v);
}

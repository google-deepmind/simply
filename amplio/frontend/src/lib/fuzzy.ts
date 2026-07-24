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

// Lightweight filename fuzzy matcher for the artifact browser's search box.
// Not fzf-grade — just two tiers, which covers filename search well:
//   1. exact (case-insensitive) SUBSTRING match — always ranked above fuzzy;
//   2. subsequence match (query chars appear in order) — the "fuzzy" tier.
// Within each tier, better matches score higher (see scoreSubstring/scoreSub).
// The query is matched against the full subpath; a match in the BASENAME (after
// the last '/') is boosted, since that's usually what the user is aiming at.

export interface FuzzyMatch<T> {
	item: T;
	score: number;
}

// scoreSubstring rates an exact-substring hit. Higher is better. Rewards: an
// earlier hit, a hit in the basename, and a shorter haystack (tighter match).
function scoreSubstring(hayLower: string, needleLower: string, baseStart: number): number {
	const idx = hayLower.indexOf(needleLower);
	if (idx < 0) return -1;
	let s = 1000; // base: substring tier always beats the fuzzy tier
	s -= idx; // earlier is better
	if (idx >= baseStart) s += 300; // hit lands in the basename
	if (idx === baseStart) s += 200; // basename prefix
	if (idx === 0) s += 100; // full-path prefix
	s -= hayLower.length * 0.5; // prefer shorter paths on ties
	return s;
}

// scoreSubsequence rates a subsequence match (needle chars in order, gaps
// allowed). Returns -1 when the needle isn't a subsequence. Rewards contiguous
// runs and basename hits; penalizes gaps — a cheap approximation of fzf.
function scoreSubsequence(hayLower: string, needleLower: string, baseStart: number): number {
	if (needleLower.length === 0) return 0;
	let hi = 0;
	let ni = 0;
	let score = 0;
	let prevMatch = -2;
	let firstIdx = -1;
	while (hi < hayLower.length && ni < needleLower.length) {
		if (hayLower[hi] === needleLower[ni]) {
			if (firstIdx < 0) firstIdx = hi;
			if (hi === prevMatch + 1) score += 8; // contiguous run bonus
			if (hi >= baseStart) score += 3; // char lands in the basename
			prevMatch = hi;
			ni++;
		}
		hi++;
	}
	if (ni < needleLower.length) return -1; // not a subsequence
	score -= (hayLower.length - needleLower.length) * 0.1; // mild length penalty
	if (firstIdx >= 0) score -= firstIdx * 0.1; // earlier start is better
	return score;
}

// fuzzyMatch scores one haystack against a query. Returns -1 for no match.
// Exported for unit reasoning; fuzzyFilter is the usual entry point.
export function fuzzyScore(haystack: string, query: string): number {
	const q = query.trim().toLowerCase();
	if (q === '') return 0;
	const hay = haystack.toLowerCase();
	const baseStart = hay.lastIndexOf('/') + 1; // 0 when no slash
	const sub = scoreSubstring(hay, q, baseStart);
	if (sub >= 0) return sub;
	return scoreSubsequence(hay, q, baseStart);
}

// fuzzyFilter ranks items by how well `key(item)` matches `query`. An empty
// query returns every item unchanged (score 0), preserving input order. Ties
// are broken by shorter key then lexicographic, for stable, sensible ordering.
export function fuzzyFilter<T>(items: T[], query: string, key: (item: T) => string): FuzzyMatch<T>[] {
	const q = query.trim();
	if (q === '') return items.map((item) => ({ item, score: 0 }));
	const out: FuzzyMatch<T>[] = [];
	for (const item of items) {
		const score = fuzzyScore(key(item), q);
		if (score >= 0) out.push({ item, score });
	}
	out.sort((a, b) => {
		if (b.score !== a.score) return b.score - a.score;
		const ka = key(a.item);
		const kb = key(b.item);
		if (ka.length !== kb.length) return ka.length - kb.length;
		return ka < kb ? -1 : ka > kb ? 1 : 0;
	});
	return out;
}

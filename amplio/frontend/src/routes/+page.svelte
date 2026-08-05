<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { api, errorText } from '$lib/api';
	import { openStream } from '$lib/stream';
	import { updates } from '$lib/updates.svelte';
	import type { RunSummary } from '$lib/types';
	import RunList from '$lib/components/RunList.svelte';
	import StartRunForm from '$lib/components/StartRunForm.svelte';
	import { auth } from '$lib/auth.svelte';
	import { prefetchWorkspaceInfo } from '$lib/workspaceInfo';
	import { prefetchModelMenu } from '$lib/modelMenu';
	import { clearRunPrefill } from '$lib/runPrefill.svelte';
	import { pageTitle } from '$lib/title';

	// Page size for the runs list. Modest so the front page stays responsive;
	// "Load more" pages backwards in time via the server cursor. 25 is the
	// sweet spot — a full first screen without an over-long initial paint.
	const PAGE_SIZE = 25;
	// Upper bound for a single refetch, mirroring the server's maxRunsPage cap. A
	// live (SSE) refresh re-requests up to this many rows to PRESERVE an expanded
	// ("Load more") view instead of snapping back to page 1; past it the list
	// stabilizes at the cap (still far better than resetting to one page).
	const MAX_RUNS_PAGE = 200;

	let runs = $state<RunSummary[]>([]);
	let hasMore = $state(false);
	let nextCursor = $state('');
	let loadingMore = $state(false);
	let error = $state('');
	let timer: ReturnType<typeof setTimeout> | undefined;

	// Filters are ALL server-side now (search/starred/grade join the existing
	// status/archived filters), so they compose (AND) and paginate over the true
	// combined match set — not just the loaded page. They're URL-driven (?q=,
	// ?starred=1, ?grade=, ?filter=) so a view is deep-linkable, shareable, and
	// survives reload; writes use replaceState (see updateURL) so live typing
	// doesn't spam browser history. `showArchived` stays a transient toggle.
	const statusFilter = $derived(page.url.searchParams.get('filter') ?? 'all');
	const search = $derived(page.url.searchParams.get('q') ?? '');
	const starredOnly = $derived(page.url.searchParams.get('starred') === '1');
	// Exact-match effective-grade filter. 'all' = no filter; 'ungraded' = no
	// effective grade (neither human nor critic); else a grade name matched against
	// the effective grade (human `grade` if set, else `report_grade`).
	const gradeFilter = $derived(page.url.searchParams.get('grade') ?? 'all');
	let showArchived = $state(false);

	// The search box binds to this local for instant typing feedback; the URL (and
	// thus the server refetch) is updated debounced (see onSearchInput). Seeded
	// from the URL and kept in sync when the URL changes by other means (back/fwd,
	// a deep link).
	let searchInput = $state('');
	$effect(() => {
		const q = search; // track the URL value
		untrack(() => {
			if (q !== searchInput) searchInput = q;
		});
	});

	// The status group (active/done/failed/updates) is applied SERVER-SIDE — see
	// listRuns({filter}) — so the dashboard no longer maps statuses to groups
	// client-side; the server's filter and /counts agree by construction.
	// Filter labels shown in the page heading. Order matches the dropdown.
	const HEADINGS: Record<string, string> = {
		all: 'Runs',
		active: 'Active Runs',
		done: 'Done',
		failed: 'Failed',
		updates: 'Recent Updates'
	};
	const heading = $derived(HEADINGS[statusFilter] ?? 'Runs');

	// All filters are server-side, so the loaded `runs` ARE the matching set for
	// the current query/page — no client-side narrowing. Any active filter still
	// paginates correctly, so "Load more" is always valid.
	const anyFilterActive = $derived(
		search.trim() !== '' || starredOnly || gradeFilter !== 'all' || statusFilter !== 'all'
	);

	// updateURL writes a filter param (empty/'all'/false clears it) via replaceState:
	// it OVERWRITES the current history entry rather than pushing, so live typing /
	// toggling never grows history depth and the back button stays anchored to the
	// page the user arrived from. Changing the URL re-derives the filters above and
	// triggers the refetch effect below.
	function updateURL(key: string, val: string | null) {
		const url = new URL(page.url);
		if (val === null || val === '' || val === 'all') url.searchParams.delete(key);
		else url.searchParams.set(key, val);
		goto(url.pathname + url.search, { replaceState: true, noScroll: true, keepFocus: true });
	}
	function setFilter(val: string) {
		updateURL('filter', val);
	}

	// Debounced search: bind the input to searchInput for instant feedback, then
	// push it to ?q= after a pause (one timer, ~250ms). The URL change drives the
	// server refetch; replaceState keeps history flat regardless of typing speed.
	let searchTimer: ReturnType<typeof setTimeout> | undefined;
	function onSearchInput() {
		clearTimeout(searchTimer);
		searchTimer = setTimeout(() => updateURL('q', searchInput.trim()), 250);
	}

	// refresh reloads from the TOP. A mount/filter refresh loads page 1; a live
	// (SSE) refresh passes preserveExpanded=true to refetch as many rows as are
	// currently shown (capped at MAX_RUNS_PAGE), so a background update doesn't snap
	// a "Load more"-expanded list back to page 1. Either way it rebuilds from the
	// top so the newest runs and latest statuses always show.
	async function refresh(preserveExpanded = false) {
		const limit = preserveExpanded
			? Math.min(Math.max(PAGE_SIZE, runs.length), MAX_RUNS_PAGE)
			: PAGE_SIZE;
		try {
			const pageData = await api.listRuns({
				showArchived,
				limit,
				filter: statusFilter,
				search,
				starred: starredOnly,
				grade: gradeFilter
			});
			runs = pageData.runs;
			hasMore = pageData.has_more;
			nextCursor = pageData.next_cursor;
			// Re-seed the updates store from the fresh list so per-run badges/favicon
			// reflect the server's has_updates for the loaded runs.
			updates.seedFromRuns(runs);
			error = '';
		} catch (e) {
			error = errorText(e); // '' when unreachable (global banner covers it)
		}
	}

	// loadMore appends the next page using the server cursor. Guarded against
	// concurrent clicks. A subsequent live (SSE) refresh PRESERVES this expanded
	// count (see refresh's preserveExpanded), so it no longer snaps back to page 1.
	async function loadMore() {
		if (loadingMore || !hasMore) return;
		loadingMore = true;
		try {
			const pageData = await api.listRuns({
				showArchived,
				limit: PAGE_SIZE,
				before: nextCursor,
				filter: statusFilter,
				search,
				starred: starredOnly,
				grade: gradeFilter
			});
			runs = [...runs, ...pageData.runs];
			hasMore = pageData.has_more;
			nextCursor = pageData.next_cursor;
			updates.seedFromRuns(runs);
			error = '';
		} catch (e) {
			error = errorText(e); // '' when unreachable (global banner covers it)
		} finally {
			loadingMore = false;
		}
	}
	function schedule() {
		clearTimeout(timer);
		// Live update: preserve the expanded ("Load more") view rather than resetting.
		timer = setTimeout(() => refresh(true), 250);
	}

	onMount(() => {
		refresh();
		prefetchWorkspaceInfo(); // warm caches so the composer opens without a flicker
		prefetchModelMenu();
		const close = openStream(null, (ev) => {
			// Ignore signals that change no persisted state the dashboard renders:
			// token previews and sysstat (handled by its own store). Keep this in
			// sync with runsStore's ignore set. (ephemeral_agents — report/compaction
			// — is rare enough to just refresh on.)
			if (ev.kind !== 'stream_chunk' && ev.kind !== 'sysstat') schedule();
		});
		return () => {
			close();
			clearTimeout(timer);
			// Drop any un-consumed "new run like" prefill on leaving the dashboard, so
			// it can't resurface on a later visit (belt-and-suspenders: the composer
			// normally consumes it immediately, but it may be absent when signed out).
			clearRunPrefill();
		};
	});

	// Every filter is SERVER-side, so any change must refetch page 1 (resetting
	// the cursor) rather than re-derive a client view. All four are URL-driven, so
	// this fires on banner deep-links, the dropdowns, the star toggle, and the
	// (debounced) search URL write alike. Keyed on a combined signature; guarded
	// so it doesn't double-fetch alongside onMount's initial refresh.
	const filterKey = $derived(`${statusFilter}\u0000${search}\u0000${starredOnly}\u0000${gradeFilter}`);
	let lastFilterKey = untrack(() => filterKey);
	$effect(() => {
		const k = filterKey; // track
		if (k !== lastFilterKey) {
			lastFilterKey = k;
			void refresh();
		}
	});
</script>

<svelte:head>
	<title>{pageTitle(heading)}</title>
</svelte:head>

<div class="dash">
	<div class="toolbar">
		<h2>{heading} <span class="count dim">{runs.length}{hasMore ? '+' : ''}</span></h2>
		<div class="filters">
			<input
				class="search"
				type="search"
				placeholder="Search title, task, run id, or workspace…"
				aria-label="Search runs by title, task, run id, or workspace"
				bind:value={searchInput}
				oninput={onSearchInput}
			/>
			<select
				value={statusFilter}
				onchange={(e) => setFilter(e.currentTarget.value)}
				aria-label="Filter by status"
			>
				<option value="all">All</option>
				<option value="active">Active</option>
				<option value="done">Done</option>
				<option value="failed">Failed</option>
				<option value="updates">Updates</option>
			</select>
			<!-- Exact-match grade filter, kept alongside the star toggle. Effective
			     grade = human grade ?? critic grade; "Ungraded" = neither set. -->
			<select
				value={gradeFilter}
				onchange={(e) => updateURL('grade', e.currentTarget.value)}
				aria-label="Filter by grade"
			>
				<option value="all">All grades</option>
				<option value="excellent">Excellent</option>
				<option value="good">Good</option>
				<option value="meh">Meh</option>
				<option value="bad">Bad</option>
				<option value="garbage">Garbage</option>
				<option value="ungraded">Ungraded</option>
			</select>
			<button
				class="chip"
				class:on={starredOnly}
				aria-pressed={starredOnly}
				onclick={() => updateURL('starred', starredOnly ? null : '1')}>★ Starred</button
			>
			<label class="chip">
				<input type="checkbox" bind:checked={showArchived} onchange={() => refresh()} /> Archived
			</label>
		</div>
	</div>
	{#if error}<p class="err">{error}</p>{/if}
	<section class="runs">
		<RunList {runs} onmutated={refresh} />
		{#if runs.length === 0 && anyFilterActive}
			<p class="more-hint dim small">No runs match the current filters.</p>
		{/if}
		<!-- Pagination affordance at the scroll-tail. Every filter is server-side, so
		     "Load more" paginates the true (filtered) match set — always valid. -->
		{#if hasMore}
			<button class="load-more" onclick={loadMore} disabled={loadingMore}>
				{loadingMore ? 'Loading…' : 'Load more runs'}
			</button>
		{/if}
	</section>
	{#if auth.authed}
		<div class="compose">
			<StartRunForm />
		</div>
	{/if}
</div>

<style>
	/* App-shell dashboard: fills main exactly. Toolbar pinned at top, the run
	   list scrolls in the middle, and the composer is anchored at the bottom
	   (chat/search-bar style). */
	.dash {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
		gap: 0.8rem;
	}
	.runs {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	/* Pagination affordance at the scroll-tail of the list. Centered so it reads
	   as a list control, not a stray button bumping the composer. */
	.load-more {
		display: block;
		margin: 0.8rem auto 0.4rem;
		padding: 0.4rem 1.1rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		background: var(--bg);
		color: var(--text);
		cursor: pointer;
	}
	.load-more:hover:not(:disabled) {
		background: var(--bg-hover, var(--border));
	}
	.load-more:disabled {
		opacity: 0.6;
		cursor: default;
	}
	.more-hint {
		text-align: center;
		margin: 0.8rem auto 0.4rem;
	}
	/* Centered composer, like a search/chat input. Grows upward on focus. */
	.compose {
		width: 100%;
		max-width: 760px;
		margin: 0.2rem auto 0;
	}
	.toolbar {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 1rem;
		flex-wrap: wrap;
	}
	h2 {
		margin: 0;
	}
	.count {
		font-size: var(--fs-md);
		font-weight: 400;
	}
	.filters {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	/* The global input/select rule (app.css) sets width: 100% so text fields
	   stretch in plain forms; in this horizontal toolbar that makes each one
	   demand the full row, forcing flex-wrap to stack them. Override to
	   natural widths so the filter group lays out on a single line. */
	.filters .search {
		font: inherit;
		width: 16rem;
		min-width: 10rem;
	}
	.filters select {
		width: auto;
	}
	.chip {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		font-size: var(--fs-md);
		padding: 0.3rem 0.6rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		background: var(--bg-elev);
		color: var(--text-dim);
		cursor: pointer;
	}
	.chip.on {
		color: var(--accent);
		border-color: var(--accent-dim);
	}
</style>

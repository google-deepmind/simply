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
	// Session-log shell: the read-only viewer over ONE session's history.
	//
	// It owns everything both modes share — which session, the phase index, which
	// phase is selected — and renders the mode page (trajectory / chat) in its
	// right pane. Because the two modes are CHILD routes of this layout, flipping
	// between them keeps this component mounted: no index refetch, no lost
	// selection, no scroll reset.
	import { page } from '$app/state';
	import { getRunStore } from '$lib/runContext.svelte';
	import { LogViewStore, setLogView, logHref, rememberSession } from '$lib/logView.svelte';
	import SessionTree from '$lib/components/SessionTree.svelte';
	import { pageTitle } from '$lib/title';
	import type { SessionDTO } from '$lib/types';

	let { children } = $props();

	const runStore = getRunStore();
	const runId = $derived(page.params.id ?? '');
	const sid = $derived(page.params.sid ?? '');
	const sessions = $derived(runStore.detail?.sessions ?? []);

	const log = new LogViewStore();
	setLogView(log);

	// (Re)load the phase index when the session changes, and keep it fresh while
	// the session is live. Only structural signals for THIS session matter: a
	// step finalizing (new loose step), an observation landing (a phase closing).
	$effect(() => {
		const id = runId;
		const s = sid;
		log.reset(id, s);
		if (!id || !s) return;
		rememberSession(id, s);
		void log.load(id, s);
		const off = runStore.on((ev) => {
			if (ev.session_id !== s) return;
			if (ev.kind === 'session_bump' || ev.kind === 'observation' || ev.kind === 'step_advanced') {
				void log.load(id, s);
			}
		});
		return () => off();
	});

	// Switching sessions keeps the mode but drops ?phase= — a phase key from one
	// session means nothing in another (the store then falls back to "newest").
	const sessionHref = (s: SessionDTO) => logHref(runId, s.session_id, log.mode);

	const runLabel = $derived(runStore.detail?.title || runStore.detail?.task || runId);
	const modeLabel = $derived(log.mode === 'chat' ? 'chat log' : 'trajectory');
</script>

<svelte:head>
	<title>{pageTitle(`${sid} · ${modeLabel}`, runLabel)}</title>
</svelte:head>

<div class="logview">
	<header class="lv-head">
		<span class="lv-label dim small">Session</span>
		<!-- Mode tabs are links (not buttons) so a mode is bookmarkable and
		     middle-clickable, and they carry the current phase across the switch. -->
		<nav class="modes">
			<a class="mode" class:on={log.mode === 'trajectory'} href={log.hrefForMode('trajectory')}
				>Trajectory</a
			>
			<a class="mode" class:on={log.mode === 'chat'} href={log.hrefForMode('chat')}>Chat log</a>
		</nav>
	</header>

	<!-- Selector: the same session tree as the Overview, at picker density. Height
	     is capped so a run with many sub-agents scrolls here instead of pushing
	     the content out of view. -->
	<div class="selector">
		{#if sessions.length}
			<SessionTree {runId} {sessions} variant="selector" selectedId={sid} hrefFor={sessionHref} />
		{:else}
			<p class="dim small nopad">Loading sessions…</p>
		{/if}
	</div>

	{#if log.error}<p class="err">{log.error}</p>{/if}

	<div class="lv-body">
		<!-- Phase index: the selectable step ranges. One column, always the same
		     entries in both modes — only the right pane's renderer differs. -->
		<aside class="groups">
			{#if !log.traj}
				<p class="dim small">Loading…</p>
			{:else}
				{#each log.groups as g (g.key)}
					<a class="group" class:on={g.key === log.selected?.key} href={log.hrefFor(g)}>
						<span class="grange mono dim">
							{g.kind === 'bootstrap' ? 'step 0' : `steps ${g.start}–${g.end}`}
						</span>
						<span class="gtitle">{g.title}</span>
						{#if g.summary}<span class="gsum dim small">{g.summary}</span>{/if}
					</a>
				{/each}
			{/if}
		</aside>
		<div class="pane">{@render children()}</div>
	</div>
</div>

<style>
	/* Fills the run shell's scroll pane; the selector and phase index are fixed
	   chrome and each column scrolls internally, so the page itself never grows. */
	.logview {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.lv-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.6rem;
	}
	.lv-label {
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.modes {
		display: flex;
		gap: 0.2rem;
	}
	.mode {
		padding: 0.2rem 0.7rem;
		border: 1px solid transparent;
		border-radius: var(--radius-pill);
		color: var(--text-dim);
		font-size: var(--fs-sm);
	}
	.mode:hover {
		text-decoration: none;
		color: var(--text);
		background: var(--bg-elev2);
	}
	.mode.on {
		color: var(--accent);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border-color: color-mix(in srgb, var(--accent) 40%, transparent);
	}
	/* Deliberately NOT a card: a bordered, filled box here would nest a box (the
	   selector) inside a box (the page) holding more boxes (the rows), and the
	   nesting is what made the chrome read as busy. A single hairline rule
	   separates it from the panes below; the rows carry their own selection wash. */
	.selector {
		max-height: 9rem;
		overflow-y: auto;
		padding-bottom: 0.4rem;
		border-bottom: 1px solid var(--border);
	}
	.nopad {
		margin: 0.2rem 0.4rem;
	}
	.lv-body {
		flex: 1;
		min-height: 0;
		display: flex;
		gap: 0.8rem;
	}
	/* Phase index. Fixed width (not a ratio) so it keeps a readable width as the
	   window grows — the content pane takes the remainder. */
	.groups {
		flex: 0 0 20rem;
		min-height: 0;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		padding-right: 0.2rem;
	}
	/* The phase index is the ONE place that keeps the accent left-bar: it's the
	   viewer's primary selection, and with the selector above now unmarked the
	   marker is unambiguous again. Entries are flat (no per-card border) — the
	   fill and the bar carry the state; a border on every row was pure noise. */
	.group {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.45rem 0.6rem;
		border-left: 2px solid transparent;
		border-radius: var(--radius-sm);
		color: var(--text);
	}
	.group:hover {
		text-decoration: none;
		background: var(--bg-elev);
	}
	.group.on {
		background: var(--bg-elev2);
		border-left-color: var(--accent);
	}
	.group.on .gtitle {
		color: var(--accent);
	}
	.grange {
		font-size: var(--fs-xs);
	}
	.gtitle {
		font-size: var(--fs-md);
		font-weight: 500;
	}
	/* Summary preview: a few lines of plain text (not rendered markdown — this is
	   an index entry, not the content). */
	.gsum {
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
	.pane {
		flex: 1;
		min-width: 0;
		min-height: 0;
		display: flex;
		flex-direction: column;
	}
</style>

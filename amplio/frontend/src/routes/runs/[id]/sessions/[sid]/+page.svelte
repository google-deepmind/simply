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
	import { tick, untrack } from 'svelte';
	import { page } from '$app/state';
	import { api, errorText } from '$lib/api';
	import { getRunStore } from '$lib/runContext.svelte';
	import type { Trajectory, TrajStep, EventDTO, AgentEvent } from '$lib/types';
	import EventCard from '$lib/components/EventCard.svelte';
	import { renderMarkdown } from '$lib/markdown';
	import { pageTitle } from '$lib/title';
	import { CaretRightIcon } from 'phosphor-svelte';

	const store = getRunStore();
	const runId = $derived(page.params.id ?? '');
	const sid = $derived(page.params.sid ?? '');

	let traj = $state<Trajectory | null>(null);
	// Raw events per step, fetched on demand when a step is expanded.
	let stepEvents = $state<Record<number, EventDTO[]>>({});
	let error = $state('');

	// Deep-link target: `?step=N` or `?step=N-M`. Drives the deep-link effect
	// below — opens the matching <details> for each step in the range, opens
	// any ancestor phase <details> too, and scrolls the first into view. Used
	// by citation links from the run report (overview page) and by the
	// struggle-range links.
	const stepParam = $derived(page.url.searchParams.get('step'));
	const targetRange = $derived.by((): { start: number; end: number } | null => {
		if (!stepParam) return null;
		const m = stepParam.match(/^(\d+)(?:-(\d+))?$/);
		if (!m) return null;
		const start = parseInt(m[1], 10);
		const end = m[2] ? parseInt(m[2], 10) : start;
		return start <= end ? { start, end } : null;
	});
	// Tracks the last (target, session) we've already opened+scrolled for so
	// the effect doesn't yank the page on every live `traj` refresh — only on
	// initial mount and on URL changes. Encoded as a string for cheap compare.
	let lastDeepLinkKey: string | null = null;

	// pairEvents groups a step's tool_result events with the tool_calls they
	// answer (matched by tool_call_id) so the trajectory renders each call+result
	// together via <ToolCall>. Returns the events to DISPLAY (paired results
	// dropped, since they render under their call) and a call_id→result map.
	// An orphan tool_result (no matching call in this step) is kept in display so
	// nothing silently disappears.
	function pairEvents(events: EventDTO[]): { display: EventDTO[]; results: Record<string, AgentEvent> } {
		const results: Record<string, AgentEvent> = {};
		const callIds = new Set<string>();
		for (const e of events) {
			for (const tc of e.event.tool_calls ?? []) callIds.add(tc.id);
		}
		for (const e of events) {
			const id = e.event.tool_call_id;
			if (e.event.type === 'tool_result' && id && callIds.has(id)) {
				results[id] = e.event;
			}
		}
		const display = events.filter(
			(e) => !(e.event.type === 'tool_result' && e.event.tool_call_id && results[e.event.tool_call_id])
		);
		return { display, results };
	}

	async function loadStep(step: number, force = false) {
		if (!force && stepEvents[step]) return;
		try {
			// Capture into a local BEFORE the spread/assign: spreading across the
			// await would let N concurrent loadStep calls each snapshot the old
			// stepEvents and last-write-wins. Splitting it out makes the spread read
			// the freshest state so concurrent writes accumulate.
			const events = await api.getEvents(runId, sid, step);
			stepEvents = { ...stepEvents, [step]: events };
		} catch (e) {
			error = errorText(e); // '' when unreachable (global banner covers it)
		}
	}

	function onToggle(e: Event, step: number) {
		if ((e.currentTarget as HTMLDetailsElement).open) loadStep(step);
	}

	async function refresh(id: string, s: string) {
		let t: typeof traj;
		try {
			t = await api.getTrajectory(id, s);
		} catch (e) {
			// Ignore an error from a request for a session we've since navigated away from.
			if (id === runId && s === sid) error = errorText(e); // '' when unreachable
			return;
		}
		// A slower request for a prior session must not clobber the current one.
		if (id !== runId || s !== sid) return;
		traj = t;
		error = ''; // a successful refresh clears any stale transient-error banner
		// Keep the live (current) step fresh if it's already expanded.
		if (traj && stepEvents[traj.current_step] !== undefined) loadStep(traj.current_step, true);
	}

	$effect(() => {
		const id = runId;
		const s = sid;
		traj = null;
		stepEvents = {};
		error = '';
		// Reset the deep-link guard so navigating between sessions re-applies
		// the URL's step target instead of treating it as "already opened".
		lastDeepLinkKey = null;
		if (!id || !s) return;
		refresh(id, s);
		const off = store.on((ev) => {
			if (ev.session_id !== s) return;
			if (ev.kind === 'session_bump' || ev.kind === 'observation' || ev.kind === 'step_advanced') {
				refresh(id, s);
			}
		});
		return () => off();
	});

	// Deep-link expander: when the page lands with ?step=N (or N-M), open
	// each matching step's <details> plus any ancestor phase, then scroll the
	// first into view. Imperative DOM mutation (vs reactive bind:open) because
	// each step is rendered inside an {#each} that doesn't expose handles, and
	// we want user-toggles to win after the initial expansion.
	$effect(() => {
		const target = targetRange;
		const t = traj;
		const session = sid;
		if (!target || !t) return;
		const key = `${session}|${target.start}-${target.end}`;
		// Use untrack so reading/writing the guard doesn't loop the effect.
		const skip = untrack(() => lastDeepLinkKey === key);
		if (skip) return;
		untrack(() => {
			lastDeepLinkKey = key;
		});
		// Wait for the DOM to render the latest traj before querying IDs.
		void tick().then(() => {
			for (let n = target.start; n <= target.end; n++) {
				const el = document.getElementById(`step-${n}`);
				if (!(el instanceof HTMLDetailsElement)) continue;
				el.open = true;
				// Open every ancestor <details> (the enclosing phase, if any).
				let p: HTMLElement | null = el.parentElement;
				while (p) {
					if (p instanceof HTMLDetailsElement) p.open = true;
					p = p.parentElement;
				}
				// The toggle handler we'd normally rely on (onToggle) doesn't
				// fire for programmatic .open=true, so kick the event fetch
				// ourselves — same effect, no waiting for the user to click.
				void loadStep(n);
			}
			document
				.getElementById(`step-${target.start}`)
				?.scrollIntoView({ behavior: 'smooth', block: 'center' });
		});
	});

	// Tab title: session id first (the bit that differentiates one open
	// session from another for the same run), then run identity. Same
	// run-label fallback chain as the other run sub-pages.
	const runLabel = $derived(store.detail?.title || store.detail?.task || runId);
</script>

<svelte:head>
	<title>{pageTitle(sid, runLabel)}</title>
</svelte:head>

{#snippet stepRow(st: TrajStep, label: string)}
	<details class="step" id="step-{st.step}" ontoggle={(e) => onToggle(e, st.step)}>
		<summary>
			<span class="caret"><CaretRightIcon size={14} /></span>
			<span class="stepno mono">{label}</span>
			{#if st.status_tag}<span class="tag tag-{st.status_tag}">{st.status_tag}</span>{/if}
			<span class="stepsum dim">{st.summary || '—'}</span>
		</summary>
		<div class="events">
			{#if stepEvents[st.step]}
				{#if stepEvents[st.step].length === 0}
					<p class="dim small">No events.</p>
				{:else}
					{@const paired = pairEvents(stepEvents[st.step])}
					{#each paired.display as e, i (i)}
						<EventCard ev={e.event} step={e.step} {runId} results={paired.results} />
					{/each}
				{/if}
			{:else}
				<p class="dim small">Loading…</p>
			{/if}
		</div>
	</details>
{/snippet}

<div class="crumb dim small">
	<a href="/runs/{runId}">Overview</a><span>›</span><span class="mono">{sid}</span>
</div>
{#if error}<p class="err">{error}</p>{/if}

{#if !traj}
	{#if !error}<p class="dim">Loading…</p>{/if}
{:else}
	<div class="traj">
		<!-- Every session has step-0 bootstrap events (system prompt etc.), so the
		     row is always shown. -->
		{@render stepRow({ step: 0, summary: 'session bootstrap', status_tag: '' }, 'bootstrap')}
		{#each traj.phases as ph (ph.end_step)}
			<details class="phase">
				<summary>
					<span class="caret"><CaretRightIcon size={14} /></span>
					<span class="range mono dim">steps {ph.start_step}–{ph.end_step}</span>
					<span class="ptitle">{ph.title}</span>
				</summary>
				<div class="md psummary dim">{@html renderMarkdown(ph.summary)}</div>
				{#if ph.artifacts?.length}
					<div class="artifacts">
						{#each ph.artifacts as a, i (i)}
							<div class="artifact">
								<span class="akind">{a.kind}</span>
								<code class="aval">{a.value}</code>
								{#if a.context}<span class="actx dim">{a.context}</span>{/if}
							</div>
						{/each}
					</div>
				{/if}
				{#if ph.lesson_verdicts?.length}
					<div class="lessons">
						{#each ph.lesson_verdicts as lv, i (i)}
							<div class="lesson" title={lv.reason ?? ''}>
								<span class="lverdict" class:helpful={lv.verdict === 'helpful'} class:unhelpful={lv.verdict === 'unhelpful'} class:harmful={lv.verdict === 'harmful'}>{lv.verdict}</span>
								<span class="ltitle">{lv.title}</span>
								{#if lv.reason}<span class="lreason dim">{lv.reason}</span>{/if}
							</div>
						{/each}
					</div>
				{/if}
				<div class="psteps">
					{#each ph.steps as st (st.step)}
						{@render stepRow(st, `step ${st.step}`)}
					{/each}
				</div>
			</details>
		{/each}
		{#each traj.loose_steps as st (st.step)}
			{@render stepRow(st, `step ${st.step}`)}
		{/each}

	</div>
{/if}

<style>
	.crumb {
		display: flex;
		gap: 0.4rem;
		align-items: center;
		margin-bottom: 0.6rem;
	}
	.traj {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}
	details {
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		background: var(--bg-elev);
	}
	summary {
		cursor: pointer;
		padding: 0.5rem 0.75rem;
		display: flex;
		align-items: center;
		gap: 0.6rem;
		list-style: none;
	}
	summary::-webkit-details-marker {
		display: none;
	}
	/* Disclosure caret: a Phosphor icon (matching the app's icon language) that
	   rotates from ▶ to ▼ when its <details> opens. */
	.caret {
		display: inline-flex;
		flex-shrink: 0;
		color: var(--text-dim);
		transition: transform 0.12s ease;
	}
	details[open] > summary > .caret {
		transform: rotate(90deg);
	}
	summary:hover .caret {
		color: var(--text);
	}
	/* Phase: a touch elevated; its steps nest inside. */
	.phase > summary {
		font-weight: 500;
	}
	.range {
		flex-shrink: 0;
		font-size: var(--fs-md);
	}
	.ptitle {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.psummary {
		padding: 0 0.85rem 0.5rem 1.6rem;
		font-size: var(--fs-md);
	}
	.artifacts {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		padding: 0 0.85rem 0.6rem 1.6rem;
	}
	.artifact {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
		font-size: var(--fs-md);
	}
	.akind {
		flex-shrink: 0;
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--text-dim);
		border: 1px solid var(--border);
		border-radius: var(--radius-xs);
		padding: 0.02rem 0.35rem;
	}
	.aval {
		font-family: var(--mono);
		color: var(--accent);
		word-break: break-all;
	}
	.actx {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.lessons {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		padding: 0 0.85rem 0.6rem 1.6rem;
	}
	.lesson {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
		font-size: var(--fs-md);
	}
	.lverdict {
		flex-shrink: 0;
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--text-dim);
		border: 1px solid var(--border);
		border-radius: var(--radius-xs);
		padding: 0.02rem 0.35rem;
	}
	/* Verdict polarity: green helps, amber wastes, red harms. neutral keeps the
	   default muted border. */
	.lverdict.helpful {
		color: var(--ok, #3fb950);
		border-color: var(--ok, #3fb950);
	}
	.lverdict.unhelpful {
		color: var(--warn, #d29922);
		border-color: var(--warn, #d29922);
	}
	.lverdict.harmful {
		color: var(--err, #f85149);
		border-color: var(--err, #f85149);
	}
	.ltitle {
		flex-shrink: 0;
	}
	.lreason {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.psteps {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 0 0.6rem 0.6rem 1.4rem;
	}
	.step {
		background: var(--bg);
	}
	.stepno {
		flex-shrink: 0;
		font-size: var(--fs-md);
		color: var(--text-dim);
	}
	.stepsum {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.tag {
		flex-shrink: 0;
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.03em;
		padding: 0.05rem 0.4rem;
		border-radius: var(--radius-pill);
		border: 1px solid var(--border);
	}
	.tag-progressing {
		color: var(--ok);
		border-color: color-mix(in srgb, var(--ok) 40%, transparent);
	}
	.tag-retrying {
		color: var(--warn);
		border-color: color-mix(in srgb, var(--warn) 40%, transparent);
	}
	.tag-blocked {
		color: var(--err);
		border-color: color-mix(in srgb, var(--err) 40%, transparent);
	}
	.events {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 0 0.75rem 0.6rem 1.6rem;
	}
</style>

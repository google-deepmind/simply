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
	// Trajectory mode: the selected phase rendered as step rows over raw events.
	// The session, the phase index and the selection all live in the enclosing
	// layout's LogViewStore — this page only renders ONE group and fetches its
	// events (per step on demand, or the whole range via "Expand all").
	import { tick, untrack } from 'svelte';
	import { page } from '$app/state';
	import { api, errorText } from '$lib/api';
	import { getLogView } from '$lib/logView.svelte';
	import type { TrajStep, EventDTO, AgentEvent } from '$lib/types';
	import EventCard from '$lib/components/EventCard.svelte';
	import { renderMarkdown } from '$lib/markdown';
	import { CaretRightIcon } from 'phosphor-svelte';

	const log = getLogView();
	const runId = $derived(log.runId);
	const sid = $derived(log.sid);
	const group = $derived(log.selected);

	// Raw events per step, fetched on demand when a step is expanded (or in one
	// ranged request by "Expand all"). Cleared when the session changes.
	let stepEvents = $state<Record<number, EventDTO[]>>({});
	let error = $state('');
	let expanding = $state(false);

	$effect(() => {
		sid; // re-run on session change
		untrack(() => {
			stepEvents = {};
			error = '';
			lastDeepLinkKey = null;
		});
	});

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

	// Expand all: ONE ranged request for the whole phase (per-step fetching would
	// be a round-trip per row), then open every row. This is the "find something
	// in this phase" affordance — with everything expanded, ⌘F works.
	async function expandAll() {
		const g = group;
		if (!g || expanding) return;
		expanding = true;
		try {
			const events = await api.getEventsRange(runId, sid, g.start, g.end);
			const byStep: Record<number, EventDTO[]> = {};
			// Seed every step so one with no events reads as "loaded, empty"
			// rather than sitting on a permanent "Loading…".
			for (const st of g.steps) byStep[st.step] = [];
			for (const e of events) (byStep[e.step] ??= []).push(e);
			stepEvents = { ...stepEvents, ...byStep };
			await tick();
			setOpen(true);
		} catch (e) {
			error = errorText(e);
		} finally {
			expanding = false;
		}
	}

	// The rows are <details> inside an {#each} (no per-row handles), so opening /
	// closing them is a DOM walk — the same imperative approach the deep-link
	// expander below uses, and it leaves later user toggles in charge.
	function setOpen(open: boolean) {
		for (const st of group?.steps ?? []) {
			const el = document.getElementById(`step-${st.step}`);
			if (el instanceof HTMLDetailsElement) el.open = open;
		}
	}

	// Keep an expanded LIVE step fresh: while the newest (unphased) group is
	// selected, its last step is still being written.
	$effect(() => {
		const cur = log.traj?.current_step;
		if (cur === undefined) return;
		untrack(() => {
			if (stepEvents[cur] !== undefined) void loadStep(cur, true);
		});
	});

	// Deep-link target: `?step=N` or `?step=N-M`. Opens the matching rows and
	// scrolls the first into view. Used by report citations and struggle-range
	// links; the layout's selection resolver has already switched to the group
	// containing the target, so the rows exist by the time this runs.
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
	// the effect doesn't yank the page on every live refresh — only on initial
	// mount and on URL changes. Encoded as a string for cheap compare.
	let lastDeepLinkKey: string | null = null;

	$effect(() => {
		const target = targetRange;
		const g = group;
		const session = sid;
		if (!target || !g) return;
		const key = `${session}|${target.start}-${target.end}`;
		// Use untrack so reading/writing the guard doesn't loop the effect.
		const skip = untrack(() => lastDeepLinkKey === key);
		if (skip) return;
		untrack(() => {
			lastDeepLinkKey = key;
		});
		// Wait for the DOM to render the selected group before querying IDs.
		void tick().then(() => {
			for (let n = target.start; n <= target.end; n++) {
				const el = document.getElementById(`step-${n}`);
				if (!(el instanceof HTMLDetailsElement)) continue;
				el.open = true;
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
</script>

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

{#if error}<p class="err">{error}</p>{/if}

{#if !group}
	{#if !error && !log.error}<p class="dim">Loading…</p>{/if}
{:else}
	<div class="phead">
		<span class="range mono dim">
			{group.kind === 'bootstrap' ? 'step 0' : `steps ${group.start}–${group.end}`}
		</span>
		<span class="ptitle">{group.title}</span>
		<span class="spacer"></span>
		<button class="tbtn" onclick={expandAll} disabled={expanding}>
			{expanding ? 'Expanding…' : 'Expand all'}
		</button>
		<button class="tbtn" onclick={() => setOpen(false)}>Collapse</button>
	</div>

	<div class="scroll">
		{#if group.summary}
			<div class="md psummary dim">{@html renderMarkdown(group.summary)}</div>
		{/if}
		{#if group.artifacts?.length}
			<div class="artifacts">
				{#each group.artifacts as a, i (i)}
					<div class="artifact">
						<span class="akind">{a.kind}</span>
						<code class="aval">{a.value}</code>
						{#if a.context}<span class="actx dim">{a.context}</span>{/if}
					</div>
				{/each}
			</div>
		{/if}
		{#if group.lesson_verdicts?.length}
			<div class="lessons">
				{#each group.lesson_verdicts as lv, i (i)}
					<div class="lesson" title={lv.reason ?? ''}>
						<span
							class="lverdict"
							class:helpful={lv.verdict === 'helpful'}
							class:unhelpful={lv.verdict === 'unhelpful'}
							class:harmful={lv.verdict === 'harmful'}>{lv.verdict}</span
						>
						<span class="ltitle">{lv.title}</span>
						{#if lv.reason}<span class="lreason dim">{lv.reason}</span>{/if}
					</div>
				{/each}
			</div>
		{/if}
		<div class="psteps">
			{#each group.steps as st (st.step)}
				{@render stepRow(st, st.step === 0 ? 'bootstrap' : `step ${st.step}`)}
			{/each}
		</div>
	</div>
{/if}

<style>
	/* Pane header: what's selected + the bulk expand controls. Fixed chrome; the
	   content below scrolls. */
	.phead {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--border);
	}
	.spacer {
		flex: 1;
	}
	.tbtn {
		padding: 0.2rem 0.6rem;
		font-size: var(--fs-sm);
		background: none;
		color: var(--text-dim);
	}
	.tbtn:hover:not(:disabled) {
		color: var(--text);
		background: var(--bg-elev2);
	}
	.scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		padding: 0.6rem 0.2rem 0 0;
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
	.range {
		flex-shrink: 0;
		font-size: var(--fs-md);
	}
	.ptitle {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-weight: 500;
	}
	.psummary {
		padding: 0 0.2rem 0.6rem;
		font-size: var(--fs-md);
	}
	.artifacts {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		padding: 0 0.2rem 0.6rem;
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
		padding: 0 0.2rem 0.6rem;
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

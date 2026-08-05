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
	// Chat-log mode: the selected phase rendered as the chat transcript — the same
	// <ChatMessages> the live chat page uses, minus everything live (no composer,
	// no streaming preview, no optimistic echoes). The server projects any
	// session's events into bubbles, so this works for an autonomous agent too
	// (its turns and tool calls read as a conversation with itself).
	import { untrack } from 'svelte';
	import { api, errorText } from '$lib/api';
	import { getLogView } from '$lib/logView.svelte';
	import ChatMessages from '$lib/components/ChatMessages.svelte';
	import type { ChatBubble } from '$lib/types';

	const log = getLogView();
	const runId = $derived(log.runId);
	const sid = $derived(log.sid);
	const group = $derived(log.selected);

	let messages = $state<ChatBubble[]>([]);
	let scrollEl = $state<HTMLElement>();
	let loading = $state(false);
	let error = $state('');

	// What's currently rendered, as a (session, range) token. The phase index is
	// re-derived on every live refresh, handing this effect a fresh `group` object
	// each time; without the token we'd refetch a SETTLED phase on every step
	// bump. Only a genuine change of range — a new selection, or the live tail
	// growing — gets past it.
	let loaded = '';

	$effect(() => {
		const id = runId;
		const s = sid;
		const g = group;
		const upTo = log.traj?.current_step ?? 0; // grows while the tail is live
		if (!id || !s || !g) return;
		// The newest group is still being written; a settled one never changes, so
		// only the live tail re-reads on a step bump.
		const end = g.kind === 'current' ? Math.max(g.end, upTo) : g.end;
		const token = `${id}|${s}|${g.start}-${end}`;
		if (untrack(() => loaded) === token) return;
		untrack(() => {
			loaded = token;
			void loadRange(id, s, g.start, end);
		});
	});

	async function loadRange(id: string, s: string, from: number, to: number) {
		loading = true;
		try {
			const feed = await api.getChatRange(id, s, from, to);
			// Ignore a response for a session/phase we've since navigated away from.
			if (id !== runId || s !== sid) return;
			messages = feed.messages;
			error = '';
		} catch (e) {
			if (id === runId && s === sid) error = errorText(e);
		} finally {
			loading = false;
		}
	}
</script>

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
		<span class="dim small">read-only</span>
	</div>

	<div class="scroll" bind:this={scrollEl}>
		<div class="column">
			{#if messages.length}
				<ChatMessages {runId} {sid} {messages} {scrollEl} />
			{:else if loading}
				<p class="dim">Loading…</p>
			{:else}
				<!-- A phase can legitimately hold no conversational turns (e.g. the
				     bootstrap step, or a stretch of pure tool work with empty replies).
				     Point at the trajectory, which renders the raw events. -->
				<p class="dim">
					Nothing conversational in this range —
					<a href={log.hrefForMode('trajectory')}>see the trajectory</a>.
				</p>
			{/if}
		</div>
	</div>
{/if}

<style>
	/* Mirrors the trajectory pane's header so the two modes line up exactly when
	   you flip between them. */
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
	/* The transcript keeps a reading width (as the live chat's 54rem column does)
	   but CENTERS in the pane: left-aligning it stranded a wide pane's right half
	   as dead space. The scroller itself spans the pane so the scrollbar stays at
	   the pane edge; the width cap lives on the inner column. */
	.scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		/* No scrollbar: the segment navigator in the right gutter replaces it.
		   Scrolling itself is untouched (wheel, trackpad, keyboard, drag). */
		scrollbar-width: none;
		padding: 0.7rem 0.2rem 0 0;
	}
	.column {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		width: 100%;
		max-width: 54rem;
		margin: 0 auto;
	}
	.scroll::-webkit-scrollbar {
		display: none;
	}
</style>

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
	import { untrack } from 'svelte';
	import { page } from '$app/state';
	import { RunStore, setRunStore } from '$lib/runContext.svelte';
	import { updates } from '$lib/updates.svelte';
	import RunNav from '$lib/components/RunNav.svelte';
	import RunHeader from '$lib/components/RunHeader.svelte';

	let { children } = $props();
	const runId = $derived(page.params.id ?? '');

	// One store for the whole run shell: holds the single per-run SSE and the
	// shared run detail, provided to all sub-views via context.
	const store = new RunStore();
	setRunStore(store);
	$effect(() => {
		store.switchTo(runId); // re-points (and re-subscribes) when the [id] changes
	});
	$effect(() => () => store.destroy());

	// Seen-state integration (server-authoritative), driven by detail.has_updates
	// (the server's badge, refetched on each run-state signal) + tab visibility:
	//   - tab VISIBLE: the operator is watching, so clear the badge (markSeen) the
	//     moment it would show — new activity never flashes unread under their eyes.
	//   - tab HIDDEN: don't mark seen; mirror the server's has_updates into the
	//     favicon map so a backgrounded run-page tab grows its dot when the run
	//     finishes (the bug this fixes).
	// visibilityState isn't reactive, so a small $state flag (updated by a
	// listener) feeds it into the reaction below.
	let tabVisible = $state(typeof document === 'undefined' || document.visibilityState === 'visible');
	$effect(() => {
		if (typeof document === 'undefined') return;
		const sync = () => (tabVisible = document.visibilityState === 'visible');
		document.addEventListener('visibilitychange', sync);
		return () => document.removeEventListener('visibilitychange', sync);
	});
	$effect(() => {
		const id = runId;
		const hasUpdates = store.detail?.has_updates ?? false;
		const visible = tabVisible;
		if (!id) return;
		// untrack: these write updates state; tracking them would self-trigger.
		untrack(() => {
			if (visible) {
				// Only PATCH when there's actually a badge to clear (avoids a write
				// on every step_advanced while idly watching).
				if (hasUpdates) updates.markSeen(id);
				else updates.seedOne(id, false);
			} else {
				updates.seedOne(id, hasUpdates);
			}
		});
	});
</script>

<div class="shell">
	<RunNav {runId} sessions={store.detail?.sessions ?? []} />
	<div class="main">
		<RunHeader {store} />
		{#if store.error}<p class="err">{store.error}</p>{/if}
		<div class="content">{@render children()}</div>
	</div>
</div>

<style>
	/* App-shell: the run nav + header stay put; .content is the scroll pane, so
	   sub-views (overview/trajectory/chat) scroll within a persistent frame. */
	.shell {
		flex: 1;
		min-height: 0;
		display: flex;
		gap: 1.1rem;
		align-items: stretch;
	}
	.main {
		flex: 1;
		min-width: 0;
		min-height: 0;
		display: flex;
		flex-direction: column;
		gap: 0.9rem;
	}
	.content {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
	}
</style>

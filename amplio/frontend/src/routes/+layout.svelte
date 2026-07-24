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
	import '../app.css';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { auth } from '$lib/auth.svelte';
	import { api } from '$lib/api';
	import { sysstat } from '$lib/sysstat.svelte';
	import { updates } from '$lib/updates.svelte';
	import { runsStore } from '$lib/runsStore.svelte';
	import { setFavicon } from '$lib/favicon';
	import TopBanner from '$lib/components/TopBanner.svelte';
	import ServerStatusBanner from '$lib/components/ServerStatusBanner.svelte';

	let { children } = $props();

	// Seed once on app mount. auth.init() first exchanges any ?token= for the
	// auth cookie and strips it from the URL, then resolves read/write status
	// (drives the readonly UI). sysstat seeds the status chips' first paint;
	// runsStore seeds the top-banner counters. All best-effort — a network fail
	// surfaces via the server-unreachable banner (api.ts wrapper).
	onMount(() => {
		void auth.init();
		api.getSysStat()
			.then((s) => sysstat.update(s))
			.catch(() => {});
		void runsStore.init();
	});

	// Favicon dot reflects what THIS tab is showing:
	//   - On /runs/{id}/*: dot iff that specific run has unread updates.
	//     Opening a run also markSeens it, so a fresh tab on a run with no
	//     updates of its own shows no dot, even if other runs are unread.
	//   - On any other route (dashboard, /recall): dot iff ANY run is unread.
	//     The dashboard tab is the global "stuff happened" surface.
	// This is what makes per-tab semantics correct: each browser tab's
	// favicon mirrors its own content, not a global state-of-the-world.
	$effect(() => {
		const path = page.url.pathname;
		const m = path.match(/^\/runs\/([^/]+)/);
		const dotted = m ? updates.hasUpdatesById(m[1]) : updates.anyUnread;
		// Chat-page tabs get the speech-bubble variant so they're recognizable at a
		// glance among many open run tabs. Match the /runs/{id}/chat route.
		const chat = /^\/runs\/[^/]+\/chat/.test(path);
		setFavicon({ dotted, chat });
	});
</script>

<TopBanner />
<ServerStatusBanner />
<main>
	{@render children()}
</main>

<style>
	main {
		flex: 1;
		min-height: 0;
		width: 100%;
		max-width: 1720px;
		margin: 0 auto;
		padding: 1.5rem clamp(1.2rem, 3vw, 2.5rem) 3rem;
		display: flex;
		flex-direction: column;
		/* Page-scroll views (dashboard) scroll here; the run app-shell fills main
		   exactly, so this never shows a scrollbar there. */
		overflow-y: auto;
	}
</style>

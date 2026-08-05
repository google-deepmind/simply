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
	// Full-page artifact browser. The browser/preview logic lives in the shared
	// ArtifactBrowser component (also used compact in the chat side panel); this
	// route is a thin wrapper that supplies the run id and owns the URL.
	//
	// ?file=<relpath> is the single source of truth for what's open: it seeds the
	// browser (deep link / "Expand" from the chat panel) AND is rewritten as the
	// operator navigates, so Back/Forward walk the files they visited, reload keeps
	// their place, and "copy link" shares what they're actually reading. The URL
	// lives HERE, not in the component: the chat side panel renders the same
	// component and must not touch the address bar.
	import { untrack } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import ArtifactBrowser from '$lib/components/ArtifactBrowser.svelte';
	import { pageTitle } from '$lib/title';
	import { getRunStore } from '$lib/runContext.svelte';

	const runId = $derived(page.params.id ?? '');
	const store = getRunStore();
	const runLabel = $derived(store.detail?.title || store.detail?.task || runId);
	const urlFile = $derived(page.url.searchParams.get('file') ?? '');

	let viewer = $state<ArtifactBrowser>();
	let selectedFile = $state('');

	// Mirror the browser's selection into ?file=. `via` decides push vs replace:
	// deliberate jumps (a file click, a relative link inside a preview) earn a
	// history entry; arrow-key scanning and URL/host-driven restores don't, or
	// Back would have to unwind every file glanced at on the way.
	//
	// The equality check is what keeps this from looping: our own write flows back
	// in as `initialFile`, and the echo it produces matches the URL and stops here.
	function onSelect(file: string, via: 'click' | 'keyboard' | 'link' | 'browse' | 'restore') {
		if (file === urlFile) return;
		const url = new URL(page.url);
		if (file) url.searchParams.set('file', file);
		else url.searchParams.delete('file');
		// noScroll/keepFocus: this is an in-place state update, not a page change —
		// it must not jump the viewport or steal focus from the file list.
		void goto(url, {
			replaceState: via !== 'click' && via !== 'link',
			noScroll: true,
			keepFocus: true,
		});
	}

	// The other direction, for the one case the initialFile prop can't express:
	// history landing on a URL with NO ?file= (e.g. Back to the page you arrived
	// on). An empty prop means "nothing to seed", not "close what's open", so the
	// route says so explicitly — without disturbing the directory being browsed.
	$effect(() => {
		const f = urlFile;
		untrack(() => {
			if (!f && selectedFile) viewer?.clearPreview();
		});
	});
</script>

<svelte:head>
	<title>{pageTitle(runLabel)}</title>
</svelte:head>

<ArtifactBrowser bind:this={viewer} bind:selectedFile {runId} initialFile={urlFile} {onSelect} />

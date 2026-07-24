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
	// route is a thin wrapper that supplies the run id + an optional ?file deep
	// link (used by the chat panel's "Expand" bridge).
	import { page } from '$app/state';
	import ArtifactBrowser from '$lib/components/ArtifactBrowser.svelte';
	import { pageTitle } from '$lib/title';
	import { getRunStore } from '$lib/runContext.svelte';

	const runId = $derived(page.params.id ?? '');
	const store = getRunStore();
	const runLabel = $derived(store.detail?.title || store.detail?.task || runId);
	// ?file=<relpath> deep-selects a file (the chat side-panel "Expand" target).
	const initialFile = $derived(page.url.searchParams.get('file') ?? '');
</script>

<svelte:head>
	<title>{pageTitle(runLabel)}</title>
</svelte:head>

<ArtifactBrowser {runId} {initialFile} />

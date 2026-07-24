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
	import type { RunSummary } from '$lib/types';
	import RunCard from './RunCard.svelte';

	// Thin list wrapper around <RunCard>: same visual / actions as the run-page
	// header, but with linkable=true so each title is a navigation <a>. All
	// per-row state, styling, and mutation handlers live in <RunCard>.

	let { runs, onmutated }: { runs: RunSummary[]; onmutated?: () => void } = $props();
</script>

{#if runs.length === 0}
	<p class="dim">No runs.</p>
{:else}
	<div class="list">
		{#each runs as r (r.run_id)}
			<RunCard run={r} linkable {onmutated} />
		{/each}
	</div>
{/if}

<style>
	.list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
</style>

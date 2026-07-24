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
	import { WarningIcon } from 'phosphor-svelte';
	import { serverHealth } from '$lib/serverHealth.svelte';

	// Loud, top-of-page banner shown only when the server is confirmed
	// unreachable (after a short grace period — see serverHealth.svelte.ts).
	// Critical for an SPA: without it, an unreachable server presents as random
	// components going stale or broken with no explanation.
</script>

{#if serverHealth.state === 'down'}
	<div class="banner" role="alert">
		<WarningIcon size={18} weight="fill" />
		<span><strong>Server unreachable.</strong> Retrying… check that <code>amplio serve</code> is running.</span>
	</div>
{/if}

<style>
	.banner {
		display: flex;
		align-items: center;
		gap: 0.55rem;
		padding: 0.55rem 1rem;
		background: color-mix(in srgb, var(--err) 18%, var(--bg-elev));
		border-bottom: 1px solid color-mix(in srgb, var(--err) 50%, transparent);
		color: color-mix(in srgb, var(--err) 55%, var(--text));
		font-size: var(--fs-md);
	}
	code {
		font-family: var(--mono);
		background: color-mix(in srgb, var(--err) 25%, transparent);
		padding: 0.05rem 0.35rem;
		border-radius: var(--radius-xs);
	}
</style>

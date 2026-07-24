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
	import { CheckIcon, CopyIcon } from 'phosphor-svelte';

	let { text, size = 13 }: { text: string; size?: number } = $props();
	let copied = $state(false);

	async function copy() {
		try {
			await navigator.clipboard.writeText(text);
			copied = true;
			setTimeout(() => (copied = false), 1200);
		} catch {
			// clipboard unavailable (e.g. insecure context) — silently ignore
		}
	}
</script>

<button class="copy" class:copied title="Copy" aria-label="Copy" onclick={copy}>
	{#if copied}<CheckIcon {size} weight="bold" />{:else}<CopyIcon {size} />{/if}
</button>

<style>
	.copy {
		display: inline-flex;
		align-items: center;
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		padding: 0.15rem;
		line-height: 1;
		border-radius: var(--radius-xs);
	}
	.copy:hover {
		color: var(--text);
	}
	.copy.copied {
		color: var(--ok);
	}
</style>

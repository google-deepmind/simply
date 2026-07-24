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
	import { CaretRightIcon } from 'phosphor-svelte';
	import { renderMarkdown } from '$lib/markdown';

	let { thoughts, streaming = false }: { thoughts: string; streaming?: boolean } = $props();

	// We measure "how much it thought" in characters — the one signal available
	// identically while streaming and after commit (no persisted duration), so
	// the live label freezes into the committed one with no reflow.
	function fmtChars(n: number): string {
		if (n < 1000) return `${n} chars`;
		return `~${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k chars`;
	}
	const label = $derived(`💭 ${streaming ? 'Thinking…' : 'Thought'} · ${fmtChars(thoughts.length)}`);

	// Keep the live box pinned to the tail as deltas arrive, so its height stays
	// fixed instead of growing the bubble.
	let box = $state<HTMLPreElement>();
	$effect(() => {
		void thoughts; // re-run on each delta
		if (box) box.scrollTop = box.scrollHeight;
	});
</script>

{#if streaming}
	<div class="think">
		<span class="chip dim small">{label}</span>
		<pre class="box" bind:this={box}>{thoughts}</pre>
	</div>
{:else}
	<details class="think">
		<summary class="chip dim small">
			<span class="caret"><CaretRightIcon size={11} weight="bold" /></span>
			{label}
		</summary>
		<div class="md dim body">{@html renderMarkdown(thoughts)}</div>
	</details>
{/if}

<style>
	.think {
		margin-bottom: 0.35rem;
	}
	.chip {
		display: inline-flex;
		align-items: baseline;
		gap: 0.35rem;
		list-style: none;
		cursor: default;
	}
	details.think summary.chip {
		cursor: pointer;
	}
	details.think summary::-webkit-details-marker {
		display: none;
	}
	/* Disclosure affordance: caret rotates 90° when the thought is open, so the
	   open/closed state is visually unambiguous even with the 💭 emoji unchanged. */
	.caret {
		display: inline-flex;
		align-items: center;
		transition: transform 0.15s ease;
	}
	details[open] > summary .caret {
		transform: rotate(90deg);
	}
	.box {
		margin: 0.25rem 0 0;
		max-height: 6.5em;
		overflow-y: auto;
		white-space: pre-wrap;
		word-break: break-word;
		font-family: var(--mono);
		font-size: var(--fs-sm);
		line-height: 1.45;
		opacity: 0.8;
	}
	.body {
		margin-top: 0.25rem;
		/* Breathing room between the unfolded thought and the assistant text
		   below it; the folded state stays compact (no .body rendered). */
		margin-bottom: 0.6rem;
	}
</style>

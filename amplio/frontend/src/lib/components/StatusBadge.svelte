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
	// Icons are referenced statically (not via a dynamic `<Icon/>` from a
	// variable) so Rollup tree-shakes phosphor down to just these — a dynamic
	// component defeats that and pulls in the entire ~8MB icon dataset.
	import {
		CheckCircleIcon,
		CircleNotchIcon,
		HourglassMediumIcon,
		MoonIcon,
		WarningCircleIcon,
		ProhibitIcon,
		QuestionIcon
	} from 'phosphor-svelte';

	let { status }: { status: string } = $props();
	const color = $derived(
		status === 'concluded'
			? 'var(--ok)'
			: status === 'ongoing'
				? 'var(--accent)'
				: status === 'awaiting' || status === 'idle'
					? 'var(--warn)'
					: status === 'crashed' || status === 'cancelled'
						? 'var(--err)'
						: 'var(--text-dim)'
	);
</script>

<span class="badge" style="--c: {color}">
	{#if status === 'ongoing'}
		<CircleNotchIcon size={13} class="spin" />
	{:else if status === 'awaiting'}
		<HourglassMediumIcon size={13} weight="fill" />
	{:else if status === 'idle'}
		<MoonIcon size={13} weight="fill" />
	{:else if status === 'concluded'}
		<CheckCircleIcon size={13} weight="fill" />
	{:else if status === 'crashed'}
		<WarningCircleIcon size={13} weight="fill" />
	{:else if status === 'cancelled'}
		<ProhibitIcon size={13} weight="fill" />
	{:else}
		<QuestionIcon size={13} weight="fill" />
	{/if}
	<span>{status || 'unknown'}</span>
</span>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.05rem 0.5rem;
		border-radius: var(--radius-pill);
		font-size: var(--fs-xs);
		white-space: nowrap;
		color: var(--c);
		border: 1px solid var(--c);
		background: color-mix(in srgb, var(--c) 12%, transparent);
	}
</style>

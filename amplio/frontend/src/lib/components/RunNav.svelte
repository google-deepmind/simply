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
	import { page } from '$app/state';
	import { TreeStructureIcon, ChatCircleIcon, FolderIcon } from 'phosphor-svelte';

	let { runId }: { runId: string } = $props();

	const base = $derived(`/runs/${runId}`);
	const path = $derived(page.url.pathname);
	const items = $derived([
		// Trajectory (/sessions/...) is part of the Overview domain → keep it lit.
		{
			label: 'Overview',
			href: base,
			icon: TreeStructureIcon,
			active: path === base || path.startsWith(`${base}/sessions`)
		},
		{ label: 'Chat', href: `${base}/chat`, icon: ChatCircleIcon, active: path.startsWith(`${base}/chat`) },
		{
			label: 'Artifacts',
			href: `${base}/artifacts`,
			icon: FolderIcon,
			active: path.startsWith(`${base}/artifacts`)
		}
	]);
</script>

<nav class="rail">
	{#each items as it (it.href)}
		{@const Icon = it.icon}
		<a
			href={it.href}
			class="item"
			class:active={it.active}
			aria-current={it.active ? 'page' : undefined}
		>
			<span class="ic"><Icon size={28} weight={it.active ? 'fill' : 'regular'} /></span>
			<span class="label">{it.label}</span>
		</a>
	{/each}
</nav>

<style>
	.rail {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		width: 84px;
		flex-shrink: 0;
	}
	.item {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.3rem;
		padding: 0.55rem 0;
		border-radius: var(--radius-lg);
		color: var(--text-dim);
		font-size: var(--fs-sm);
	}
	.item:hover {
		text-decoration: none;
		color: var(--text);
	}
	.item.active {
		color: var(--accent);
	}
	.ic {
		display: inline-flex;
		padding: 0.35rem 1rem;
		border-radius: var(--radius-pill);
	}
	.item:hover .ic {
		background: var(--bg-elev2);
	}
	.item.active .ic {
		background: color-mix(in srgb, var(--accent) 16%, transparent);
	}
</style>

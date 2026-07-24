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
	import { onMount } from 'svelte';
	import { api } from '$lib/api';
	import type { ModelEntry, ModelMenu } from '$lib/types';
	import { cachedModelMenu, loadModelMenu } from '$lib/modelMenu';
	import { CubeIcon, CaretDownIcon, PlusIcon, MinusIcon } from 'phosphor-svelte';

	let { value = $bindable(''), onpick }: { value?: string; onpick?: () => void } = $props();

	// Seed synchronously from the prefetched cache (common case) so the pill shows
	// its model immediately rather than populating after the fetch resolves.
	const cached = cachedModelMenu();
	let models = $state<ModelEntry[]>(cached?.models ?? []);
	let open = $state(false);
	let adding = $state(false);
	let newSpec = $state('');
	let error = $state('');

	if (cached) applyMenu(cached);

	function applyMenu(menu: ModelMenu, preferred?: string) {
		models = menu.models;
		const specs = models.map((m) => m.spec);
		if (preferred && specs.includes(preferred)) value = preferred;
		else if (!value || !specs.includes(value)) value = menu.default || specs[0] || '';
	}

	function focusEl(node: HTMLInputElement) {
		node.focus();
	}

	// Close the popover on any click outside it.
	function clickOutside(node: HTMLElement, cb: () => void) {
		function handle(e: MouseEvent) {
			if (!node.contains(e.target as Node)) cb();
		}
		document.addEventListener('click', handle, true);
		return {
			destroy() {
				document.removeEventListener('click', handle, true);
			}
		};
	}

	// load fills the menu from the cache; force refetches (after add/remove).
	async function load(preferred?: string, force = false) {
		try {
			applyMenu(await loadModelMenu(force), preferred);
			error = '';
		} catch (e) {
			error = String(e);
		}
	}

	// Cache miss (expanded before the prefetch resolved): fetch on mount.
	onMount(() => {
		if (!cached) load();
	});

	function select(spec: string) {
		value = spec;
		open = false;
		onpick?.(); // let the parent restore focus (e.g. back to the composer)
	}

	async function add() {
		const spec = newSpec.trim();
		if (!spec) return;
		try {
			await api.addModel(spec);
			newSpec = '';
			adding = false;
			await load(spec, true); // force: the menu changed
		} catch (e) {
			error = String(e);
		}
	}

	async function remove(e: Event, spec: string) {
		e.stopPropagation();
		try {
			await api.removeModel(spec);
			await load(undefined, true); // force: the menu changed
		} catch (err) {
			error = String(err);
		}
	}
</script>

<div class="wrap" use:clickOutside={() => (open = false)}>
	<button type="button" class="trigger" title="Agent model" onclick={() => (open = !open)}>
		<CubeIcon size={16} />
		<span class="label">{value || 'select model'}</span>
		<span class="caret"><CaretDownIcon size={12} /></span>
	</button>
	{#if open}
		<div class="menu">
			{#each models as m (m.spec)}
				<div class="row" class:selected={m.spec === value}>
					<button type="button" class="pick" onclick={() => select(m.spec)}>{m.spec}</button>
					{#if m.removable}
						<button
							type="button"
							class="rm"
							title="Remove"
							aria-label="Remove model"
							onclick={(e) => remove(e, m.spec)}
						>
							<MinusIcon size={14} />
						</button>
					{/if}
				</div>
			{/each}
			{#if adding}
				<div class="row">
					<input
						class="newspec"
						bind:value={newSpec}
						use:focusEl
						placeholder="vertex:model-id"
						onkeydown={(e) => {
							if (e.key === 'Enter') add();
							else if (e.key === 'Escape') adding = false;
						}}
					/>
					<button type="button" class="ok" onclick={add}>add</button>
				</div>
			{:else}
				<button type="button" class="addrow" onclick={() => (adding = true)}>
					<PlusIcon size={13} /> Add model…
				</button>
			{/if}
			{#if error}<p class="err">{error}</p>{/if}
		</div>
	{/if}
</div>

<style>
	.wrap {
		position: relative;
	}
	.trigger {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		background: var(--bg-elev);
		border: 1px solid var(--border);
		color: var(--text);
		border-radius: var(--radius-sm);
		padding: 0.35rem 0.6rem;
		cursor: pointer;
		font: inherit;
	}
	.label {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 220px;
	}
	.caret {
		display: inline-flex;
		color: var(--text-dim);
	}
	.menu {
		position: absolute;
		bottom: calc(100% + 4px);
		left: 0;
		min-width: 240px;
		/* Cap horizontal growth so a long custom spec (e.g. with several query
		   params) wraps inside the menu rather than pushing the trailing
		   remove button past the viewport edge (where it can't be clicked).
		   28rem fits ~50 chars of typical model spec on one line. */
		max-width: 28rem;
		background: var(--bg-elev2);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 0.25rem;
		z-index: 10;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
	}
	.row {
		display: flex;
		/* start (not center) so the remove button sits beside the FIRST line
		   of a wrapped spec, rather than mid-block of a 3-line entry. */
		align-items: start;
		gap: 0.25rem;
	}
	.row.selected {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
		border-radius: var(--radius-xs);
	}
	.pick {
		flex: 1;
		min-width: 0; /* allow shrink-to-fit so the row can constrain the .pick width */
		text-align: left;
		background: none;
		border: none;
		color: var(--text);
		padding: 0.4rem 0.5rem;
		cursor: pointer;
		font: inherit;
		/* Plain fill-each-line wrap. Model specs are URL-shaped
		   (vertex:claude-…?thinking=true&max_tokens=8192), and with
		   overflow-wrap: anywhere the browser still tries to find "word"
		   boundaries first (after & or =) and ends up with ragged short
		   lines and orphans. word-break: break-all just fills each line to
		   the edge and continues — much cleaner for opaque identifier-like
		   strings with no real prose structure. */
		word-break: break-all;
	}
	.pick:hover {
		color: var(--accent);
	}
	.rm,
	.ok {
		display: inline-flex;
		align-items: center;
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		padding: 0.4rem 0.5rem;
		font: inherit;
	}
	.rm:hover {
		color: var(--err);
	}
	.addrow {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		width: 100%;
		background: none;
		border: none;
		border-top: 1px solid var(--border);
		margin-top: 0.25rem;
		color: var(--text-dim);
		padding: 0.45rem 0.5rem;
		cursor: pointer;
		font: inherit;
	}
	.addrow:hover {
		color: var(--text);
	}
	.newspec {
		flex: 1;
		font: inherit;
	}
	.err {
		font-size: var(--fs-sm);
		margin: 0.25rem 0.5rem 0;
	}
</style>

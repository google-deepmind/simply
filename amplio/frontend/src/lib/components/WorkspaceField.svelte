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
	import { cachedWorkspaceInfo, loadWorkspaceInfo, refreshWorkspaceInfo } from '$lib/workspaceInfo';
	import { INTERNAL, ExtraWorkspacePicker, resolveExtraWorkspace, NO_SELECTION } from './internal';
	import { FolderIcon, CaretDownIcon } from 'phosphor-svelte';
	import type { Component } from 'svelte';

	// value is the resolved workspace spec sent to the server. ondirty fires on
	// any user edit so the parent form can stay expanded. interactive (run mode)
	// picks the default source when the user hasn't chosen one: an interactive
	// run stays in the server's working directory, an autonomous one prefers a
	// fresh workspace where the build offers one.
	let {
		value = $bindable('.'),
		summary = $bindable(''),
		ondirty,
		interactive = false
	}: { value?: string; summary?: string; ondirty?: () => void; interactive?: boolean } = $props();

	// Initialize synchronously from the prefetched cache when available (the
	// common case — the dashboard warms it at page load), so the pill renders the
	// abs path immediately with no "working dir" → path flicker.
	const cached = cachedWorkspaceInfo();

	let path = $state(cached?.server_root ?? ''); // abs cwd; never a bare "."
	let recent = $state<string[]>(cached?.recent ?? []);
	let loaded = $state(!!cached);
	let open = $state(false);

	// Extra workspace sources, if this build has any. Two independent gates:
	// INTERNAL is a build-time constant, so the block below is dropped from
	// builds without those components; workspace_modes is what the SERVER says
	// it can actually resolve. Offering a source the server would reject is
	// worse than not offering it.
	let modes = $state<string[]>(cached?.workspace_modes ?? []);
	const extras = $derived(INTERNAL && modes.length > 0);

	// The picker's controls live inside the popover and unmount with it, so the
	// SELECTION lives here (the host is always mounted) and is resolved on every
	// render. That keeps the choice correct — and the run-mode default applied —
	// while the popover is closed, which is most of the time.
	let extraState = $state<Record<string, unknown>>({});
	const extraSel = $derived(
		extras ? resolveExtraWorkspace(extraState, !interactive) : NO_SELECTION,
	);

	let userPicked = $state(false);

	// Refresh in the background; a cache miss fills the fields on arrival.
	onMount(() => {
		loadWorkspaceInfo()
			.then((info) => {
				if (!info) return;
				modes = info.workspace_modes ?? [];
				recent = info.recent ?? [];
				if (!userPicked && !path) path = info.server_root ?? '';
				loaded = true;
			})
			.catch(() => {});
	});

	// Re-read recents when the popover opens, so a workspace created since page
	// load shows up. Only the candidate list is refreshed — never the user's
	// in-progress choices.
	$effect(() => {
		if (!open) return;
		refreshWorkspaceInfo()
			.then((info) => {
				if (info) recent = info.recent ?? [];
			})
			.catch(() => {});
	});

	// Derive the spec to submit. Empty path → "." so a submit before load still
	// resolves to the server cwd (== server_root).
	$effect(() => {
		value = extraSel.active ? extraSel.spec : path.trim() || '.';
	});

	function basename(p: string): string {
		const parts = p.split('/').filter(Boolean);
		return parts.length ? parts[parts.length - 1] : p;
	}

	// Pill label: short and stable; full detail on hover (title). Pushed into
	// the bindable `summary` prop via $effect so the parent (e.g. StartRunForm)
	// can use the same short label in its own UI — notably the Start button
	// shows it during the resolving stage.
	$effect(() => {
		summary = extraSel.active
			? extraSel.summary
			: loaded
				? basename(path) || 'working dir'
				: 'working dir';
	});
	const title = $derived(extraSel.active ? extraSel.title : path || 'server working directory');
	const TriggerIcon = $derived(extraSel.icon ?? FolderIcon);

	function onExtraPick() {
		userPicked = true;
		ondirty?.();
	}
	function touch() {
		ondirty?.();
	}

	// Close the popover on any click outside it (mirrors ModelSelect).
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
	// Autofocus the active text input when the popover (re)renders it.
	function focusEl(node: HTMLInputElement) {
		node.focus();
	}
</script>

<div class="wrap" use:clickOutside={() => (open = false)}>
	<button type="button" class="trigger" title={title} onclick={() => (open = !open)}>
		<TriggerIcon size={15} />
		<span class="label">{summary}</span>
		<span class="caret"><CaretDownIcon size={12} /></span>
	</button>

	{#if open}
		<div class="menu">
			{#if extras}
				<ExtraWorkspacePicker
					bind:persisted={extraState}
					{recent}
					{loaded}
					prefer={!interactive}
					onpick={onExtraPick}
				/>
			{/if}

			{#if !extraSel.active}
				<input
					class="field"
					bind:value={path}
					oninput={touch}
					use:focusEl
					placeholder="working directory"
					spellcheck="false"
					autocapitalize="off"
				/>
			{/if}
		</div>
	{/if}
</div>

<style>
	.wrap {
		position: relative;
		display: inline-block;
	}
	/* Pill trigger — styled to match ModelSelect's pill. */
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
	/* Fixed-length label so the pill width stays stable; full path on hover. */
	.label {
		width: 8rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.caret {
		display: inline-flex;
		color: var(--text-dim);
	}
	.menu {
		position: absolute;
		bottom: calc(100% + 4px);
		left: 0;
		min-width: 260px;
		background: var(--bg-elev2);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 0.5rem;
		z-index: 10;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
	}
	.field {
		width: 100%;
		font: inherit;
	}
</style>

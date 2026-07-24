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
	import { FolderIcon, FolderPlusIcon, GitBranchIcon, CaretDownIcon } from 'phosphor-svelte';

	// value is the resolved workspace spec sent to the server (abs path,
	// `new:citc/jj`, `new:citc/fig`, or `citc:<alias>`). ondirty fires on any
	// user edit so the parent form can stay expanded. interactive (run mode)
	// drives the default workspace mode when the user hasn't manually picked
	// one: autonomous → "new" (fresh CitC worktree, the common case for
	// fire-and-forget tasks), interactive → "path" (the server cwd, since the
	// operator usually wants to chat in whatever existing workspace they're
	// in). Switching back and forth with no manual pick toggles the default.
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

	type Mode = 'path' | 'new' | 'citc';
	// Default to "new anonymous citc/jj" when citc is available (the most
	// common new-run intent on corp dev hosts). Cache-miss case falls back to
	// 'path' and switches in onMount once we've confirmed citc availability,
	// unless the user has manually picked a mode by then (userPicked). Both
	// initializers read from the same cached snapshot (NOT from each other)
	// so they reflect a consistent first-render state.
	let citcAvailable = $state(cached?.citc_available ?? false);
	let mode = $state<Mode>(cached?.citc_available ? 'new' : 'path');
	let path = $state(cached?.server_root ?? ''); // abs cwd; never a bare "."
	let vcs = $state<'jj' | 'fig'>('jj');
	let alias = $state('');
	let recent = $state<string[]>(cached?.recent ?? []);
	let loaded = $state(cached != null);
	let open = $state(false);
	let userPicked = $state(false);

	// Cache miss (user expanded before the prefetch resolved): fetch on mount.
	// Degrades silently to a plain path input if it fails. If the async load
	// reveals citc is available AND the user hasn't manually picked a mode yet,
	// switch the default to 'new' to match the warm-cache experience.
	onMount(async () => {
		if (!loaded) {
			try {
				const info = await loadWorkspaceInfo();
				citcAvailable = info.citc_available;
				recent = info.recent ?? [];
				if (info.server_root) path = info.server_root;
				if (citcAvailable && !userPicked && mode === 'path') {
					mode = 'new';
				}
			} catch {
				/* keep defaults; the picker degrades to free-form path entry */
			} finally {
				loaded = true;
			}
		}
		// Fire-and-forget: the cached recents are frozen at page load, so a CitC
		// workspace created/used since then is missing. Each form expansion mounts
		// this component, so we refresh in the background and land ONLY the fresh
		// candidate list — never the user's in-progress mode/path/alias choices. If
		// it lands while they're picking, more/fresher options simply appear; if it
		// fails or never returns, the cached list stays usable. Non-blocking.
		refreshWorkspaceInfo()
			.then((info) => {
				recent = info.recent ?? [];
			})
			.catch(() => {});
	});

	// Run-mode → workspace-mode auto-switch. Activates only AFTER the initial
	// load resolves (so cache-miss flicker doesn't fight us) and only when
	// the user hasn't explicitly picked a workspace mode. Setting any mode
	// via the popover flips userPicked, freezing this behavior so subsequent
	// run-mode toggles don't undo the user's choice.
	$effect(() => {
		if (!loaded) return;
		if (userPicked) return;
		mode = interactive ? 'path' : citcAvailable ? 'new' : 'path';
	});

	// Derive the spec to submit from the active mode. Empty path → "." so a
	// submit before load still resolves to the server cwd (== server_root).
	$effect(() => {
		if (mode === 'new') value = `new:citc/${vcs}`;
		else if (mode === 'citc') value = alias.trim() ? `citc:${alias.trim()}` : '';
		else value = path.trim() || '.';
	});

	const filtered = $derived(
		alias.trim()
			? recent.filter((a) => a.toLowerCase().includes(alias.trim().toLowerCase()))
			: recent
	);

	function basename(p: string): string {
		const parts = p.split('/').filter(Boolean);
		return parts.length ? parts[parts.length - 1] : p;
	}

	// Pill label: short and stable; full detail on hover (title). Pushed into
	// the bindable `summary` prop via $effect so the parent (e.g.
	// StartRunForm) can use the same short label in its own UI — notably
	// the Start button shows it during the resolving stage so the user sees
	// what's being created without inflating the button text.
	$effect(() => {
		summary =
			mode === 'new'
				? `new · ${vcs}`
				: mode === 'citc'
					? alias.trim() || 'citc workspace'
					: loaded
						? basename(path) || 'working dir'
						: 'working dir';
	});
	const title = $derived(
		mode === 'new'
			? `New anonymous CitC workspace (${vcs})`
			: mode === 'citc'
				? alias.trim()
					? `citc:${alias.trim()}`
					: 'Open an existing named CitC workspace'
				: path || 'server working directory'
	);

	// Icon mirrors the active mode so the trigger pill telegraphs which kind
	// of workspace is selected without having to read the label:
	//   path → 📁 folder; new → 📁+ folder-plus; citc → 🌿 git-branch.
	const TriggerIcon = $derived(
		mode === 'new' ? FolderPlusIcon : mode === 'citc' ? GitBranchIcon : FolderIcon
	);

	function setMode(m: Mode) {
		mode = m;
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
			{#if citcAvailable}
				<div class="modes" role="group" aria-label="Workspace source">
					<button type="button" class:on={mode === 'new'} onclick={() => setMode('new')}
						>New CitC</button
					>
					<button type="button" class:on={mode === 'citc'} onclick={() => setMode('citc')}
						>Open CitC</button
					>
					<button type="button" class:on={mode === 'path'} onclick={() => setMode('path')}
						>Path</button
					>
				</div>
			{/if}

			{#if mode === 'path'}
				<input
					class="field"
					bind:value={path}
					oninput={touch}
					use:focusEl
					placeholder="working directory"
					spellcheck="false"
					autocapitalize="off"
				/>
			{:else if mode === 'new'}
				<div class="newrow">
					<div class="vcs" role="group" aria-label="VCS">
						<button type="button" class:on={vcs === 'jj'} onclick={() => { vcs = 'jj'; touch(); }}
							>jj</button
						>
						<button type="button" class:on={vcs === 'fig'} onclick={() => { vcs = 'fig'; touch(); }}
							>fig</button
						>
					</div>
					<span class="dim small">unnamed workspace</span>
				</div>
			{:else if mode === 'citc'}
				<input
					class="field"
					bind:value={alias}
					oninput={touch}
					use:focusEl
					placeholder="workspace alias…"
					spellcheck="false"
					autocapitalize="off"
				/>
				{#if filtered.length}
					<ul class="recents">
						{#each filtered as a (a)}
							<li>
								<button type="button" onclick={() => { alias = a; open = false; touch(); }}>{a}</button>
							</li>
						{/each}
					</ul>
				{:else if loaded}
					<p class="hint dim small">No recent workspaces.</p>
				{/if}
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
	.modes {
		display: inline-flex;
		gap: 0.15rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		padding: 0.12rem;
		margin-bottom: 0.5rem;
	}
	.modes button {
		background: none;
		border: none;
		border-radius: var(--radius-pill);
		padding: 0.2rem 0.7rem;
		font-size: var(--fs-md);
		color: var(--text-dim);
		cursor: pointer;
	}
	.modes button.on {
		background: var(--bg-elev2);
		color: var(--text);
	}
	.field {
		width: 100%;
		font: inherit;
	}
	.newrow {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.vcs {
		display: inline-flex;
		gap: 0.15rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		padding: 0.12rem;
	}
	.vcs button {
		background: none;
		border: none;
		border-radius: var(--radius-pill);
		padding: 0.15rem 0.7rem;
		font-size: var(--fs-md);
		color: var(--text-dim);
		cursor: pointer;
	}
	.vcs button.on {
		background: var(--bg-elev2);
		color: var(--text);
	}
	.hint {
		margin: 0.4rem 0 0;
	}
	.recents {
		list-style: none;
		margin: 0.4rem 0 0;
		padding: 0;
		max-height: 12rem;
		overflow-y: auto;
	}
	.recents button {
		display: block;
		width: 100%;
		text-align: left;
		background: none;
		border: none;
		border-radius: var(--radius-xs);
		padding: 0.3rem 0.5rem;
		font-size: var(--fs-md);
		color: var(--text);
		cursor: pointer;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.recents button:hover {
		background: var(--bg-elev2);
	}
</style>

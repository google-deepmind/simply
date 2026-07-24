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
	import type { SessionDTO } from '$lib/types';
	import { timeAgo } from '$lib/time';
	import StatusBadge from './StatusBadge.svelte';
	import { AndroidLogoIcon, ChatTeardropIcon, LinkIcon } from 'phosphor-svelte';
	import { iconForName } from '$lib/sessionIcon';

	// One row per session, echoing the run card's visual language (identity icon
	// · name · task fade · workspace chip · status · step+time). The name is the
	// session_id (a human nickname). Rows are links to each session's trajectory
	// so they're bookmarkable / middle-clickable; the parent/child tree is shown
	// by indentation. No overflow menu — a session row is a pure link.
	let { runId, sessions }: { runId: string; sessions: SessionDTO[] } = $props();

	const roots = $derived(sessions.filter((s) => !s.parent_id));
	function childrenOf(id: string) {
		return sessions.filter((s) => s.parent_id === id);
	}

	// Show a session's workspace chip only when it DIVERGES from where it would
	// inherit — i.e. its parent session's workspace (compared on the full,
	// already-resolved path). This is per-parent, not per-root, so it handles
	// chains: a linked sub-agent A shows a chip; A's inherit-mode child B does
	// not (B's workspace equals A's), even though both differ from the root. A
	// root session inherits the run's workspace (shown in the run header), so it
	// never shows a chip.
	const wsById = $derived(new Map(sessions.map((s) => [s.session_id, s.workspace ?? ''])));
	function ownsWorkspace(s: SessionDTO): boolean {
		const ws = s.workspace ?? '';
		if (!ws || !s.parent_id) return false;
		return ws !== (wsById.get(s.parent_id) ?? '');
	}
</script>

{#snippet node(s: SessionDTO)}
	{@const AnimalIcon = iconForName(s.session_id)}
	<div class="node">
		<a class="sess card" href="/runs/{runId}/sessions/{s.session_id}">
			<span class="titleblock">
				<span class="identity" class:animal={AnimalIcon} title={s.agent_type}>
					{#if AnimalIcon}
						<AnimalIcon size={14} weight="bold" />
					{:else if s.agent_type === 'chatbot'}
						<ChatTeardropIcon size={14} weight="bold" />
					{:else}
						<AndroidLogoIcon size={14} weight="bold" />
					{/if}
				</span>
				<span class="name mono">{s.session_id}</span>
				{#if s.task}<span class="task" title={s.task}>{s.task}</span>{/if}
			</span>
			{#if ownsWorkspace(s)}
				<!-- LinkIcon (chain) instead of FolderIcon: the chip only shows
				     when this sub-agent's workspace DIVERGES from its parent's,
				     and the dominant cause is `WorkspaceMode = "link"` (linked
				     CitC worktree on a shared repo). The chain conveys that
				     parent–child relationship; folder would just say "a
				     directory" without the linkage cue. -->
				<span class="chip" title={s.workspace}>
					<LinkIcon size={12} weight="bold" />
					<span class="chip-text">{s.workspace_name}</span>
				</span>
			{/if}
			<StatusBadge status={s.status} />
			<span class="meta dim" title={new Date(s.status_changed_at).toLocaleString()}>
				step {s.current_step} · {timeAgo(s.status_changed_at)}
			</span>
		</a>
		{#each childrenOf(s.session_id) as c (c.session_id)}
			<div class="child">{@render node(c)}</div>
		{/each}
	</div>
{/snippet}

{#each roots as r (r.session_id)}
	{@render node(r)}
{/each}

<style>
	/* Row-density override of the global .card padding (bg/border/radius come
	   from .card), matching RunCard. */
	.sess {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		padding: 0.5rem 0.9rem;
		margin-bottom: 0.35rem;
		color: var(--text);
	}
	.sess:hover {
		text-decoration: none;
		border-color: var(--accent-dim);
	}
	.sess:hover .name {
		color: var(--accent);
	}
	/* identity icon + name + task fade; consumes the space left of the chips. */
	.titleblock {
		flex: 1;
		min-width: 0;
		display: flex;
		align-items: center;
		gap: 0.6rem;
		overflow: hidden;
	}
	.identity {
		flex-shrink: 0;
		color: var(--text-dim);
		display: inline-flex;
		align-items: center;
	}
	/* When the noun maps to an animal (e.g. "swift-fox" → Dog), the animal
	   replaces the generic agent-type icon. Brighter than the fallback
	   chat/android marker so the row reads as a specific session at a
	   glance; type info still surfaces via the wrapper's title= on hover. */
	.identity.animal {
		color: var(--text);
	}
	.name {
		flex-shrink: 0;
		font-size: var(--fs-md);
		font-weight: 500;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* Task snippet — muted, single line, fading into transparency on the right
	   so it flows into the chips instead of slamming into them (mirrors RunCard). */
	.task {
		flex: 1 1 0;
		min-width: 0;
		font-size: var(--fs-sm);
		color: var(--text-dim);
		overflow: hidden;
		white-space: nowrap;
		mask-image: linear-gradient(to right, black calc(100% - 2rem), transparent);
		-webkit-mask-image: linear-gradient(to right, black calc(100% - 2rem), transparent);
	}
	.chip {
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-family: var(--mono);
		font-size: var(--fs-xs);
		padding: 0.15rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		color: var(--text-dim);
		white-space: nowrap;
		max-width: 14rem;
	}
	.chip > :global(svg) {
		flex-shrink: 0;
	}
	.chip-text {
		overflow: hidden;
		text-overflow: ellipsis;
		min-width: 0;
	}
	.meta {
		flex-shrink: 0;
		font-size: var(--fs-sm);
		font-family: var(--mono);
		white-space: nowrap;
	}
	/* Sub-agent nesting: indent + a connector rail, as before. */
	.child {
		margin-left: 1rem;
		padding-left: 0.5rem;
		border-left: 1px solid var(--border);
	}
</style>

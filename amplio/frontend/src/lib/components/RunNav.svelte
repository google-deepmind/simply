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
	import { TreeStructureIcon, ChatCircleIcon, FolderIcon, PathIcon, ScrollIcon } from 'phosphor-svelte';
	import { logHref, recallSession } from '$lib/logView.svelte';
	import type { SessionDTO } from '$lib/types';

	let { runId, sessions = [] }: { runId: string; sessions?: SessionDTO[] } = $props();

	const base = $derived(`/runs/${runId}`);
	const path = $derived(page.url.pathname);

	// The two read-only viewers are per-session, so the rail needs a session to
	// point at: whichever you looked at last in this run, else a per-mode default.
	// The defaults differ because the modes suit different agents — the trajectory
	// is for the run's worker (the autonomous root), while a chat LOG is really
	// about an interactive session, today only the chatbot. Any session can still
	// be opened in either mode from the viewer's own selector; this is just where
	// the rail lands you first. Empty until the run detail arrives — logHref then
	// targets the index route, which redirects with the same rule.
	function defaultSid(mode: 'trajectory' | 'chat'): string {
		const remembered = recallSession(runId);
		if (remembered && sessions.some((s) => s.session_id === remembered)) return remembered;
		const roots = sessions.filter((s) => !s.parent_id);
		const autonomous = roots.find((s) => s.agent_type !== 'chatbot') ?? roots[0];
		if (mode === 'chat') {
			return (sessions.find((s) => s.agent_type === 'chatbot') ?? autonomous)?.session_id ?? '';
		}
		return autonomous?.session_id ?? '';
	}
	// A session-log route: /runs/<id>/sessions[/<sid>[/chat]]. The chat-log page
	// is the one ending in /chat — note the LIVE chat lives at /runs/<id>/chat and
	// never matches this prefix.
	const inLog = $derived(path.startsWith(`${base}/sessions`));
	const inLogChat = $derived(inLog && path.endsWith('/chat'));

	const items = $derived([
		{
			label: 'Overview',
			href: base,
			icon: TreeStructureIcon,
			active: path === base
		},
		{ label: 'Chat', href: `${base}/chat`, icon: ChatCircleIcon, active: path.startsWith(`${base}/chat`) },
		{
			label: 'Trajectory',
			href: logHref(runId, defaultSid('trajectory'), 'trajectory'),
			icon: PathIcon,
			active: inLog && !inLogChat
		},
		{
			label: 'Chat log',
			href: logHref(runId, defaultSid('chat'), 'chat'),
			icon: ScrollIcon,
			active: inLogChat
		},
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

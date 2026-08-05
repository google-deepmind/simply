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
	import { browser } from '$app/environment';
	import { tick } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { api, errorText } from '$lib/api';
	import { getRunStore } from '$lib/runContext.svelte';
	import { pageTitle } from '$lib/title';
	import type { ChatBubble, PhaseCard, ChatUsage } from '$lib/types';
	import MessageBox from '$lib/components/MessageBox.svelte';
	import { auth } from '$lib/auth.svelte';
	import ChatMessages from '$lib/components/ChatMessages.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import ArtifactBrowser from '$lib/components/ArtifactBrowser.svelte';
	import { goto } from '$app/navigation';
	import { renderMarkdown } from '$lib/markdown';
	import { logHref } from '$lib/logView.svelte';
	import { AndroidLogoIcon, FolderIcon, ScrollIcon } from 'phosphor-svelte';
	import { iconForName } from '$lib/sessionIcon';
	import { toolIcon } from '$lib/toolIcon';

	const store = getRunStore();
	const chatbotSession = $derived(store.detail?.sessions.find((s) => s.agent_type === 'chatbot'));
	const chatbotSid = $derived(chatbotSession?.session_id ?? '');

	// Status-bar state, derived live from store.detail (refreshed on
	// status_change / step_advanced).
	const working = $derived(chatbotSession?.status === 'ongoing');
	const unhealthy = $derived(chatbotSession?.status === 'crashed');
	const step = $derived(chatbotSession?.current_step ?? 0);
	const ACTIVE = new Set(['ongoing', 'awaiting']);
	const activeSubagents = $derived(
		(store.detail?.sessions ?? []).filter((s) => s.parent_id === chatbotSid && ACTIVE.has(s.status))
	);

	let messages = $state<ChatBubble[]>([]);
	let phaseCards = $state<PhaseCard[]>([]);
	let usage = $state<ChatUsage | null>(null);
	// The live streaming assistant bubble. `step` is the call step the chunk
	// carries (the assistant's own slot, T) so the preview renders in its
	// chronological position — peers that arrive DURING generation land at T+1 and
	// must stay below it, matching the settled order once the stream finalizes.
	let preview = $state<{ text: string; thoughts: string; step: number } | null>(null);
	let compacting = $state(false); // true while the agent summarizes an overlong context
	// Operator messages shown instantly, before the server round-trip + SSE land.
	let optimistic = $state<{ key: string; content: string; at: number }[]>([]);
	let error = $state('');
	let starting = $state(false);

	let scrollEl = $state<HTMLDivElement>();
	let pinned = true; // follow the bottom unless the operator scrolls up

	// Intra-turn timestamp clustering: a left-aligned timestamp separates clusters

	async function startChatbot() {
		starting = true;
		error = '';
		try {
			await api.startChatbot(store.runId);
			await store.refresh();
		} catch (e) {
			// errorText suppresses the redundant local banner for an unreachable
			// server (the global ServerStatusBanner already covers that case).
			error = errorText(e);
		} finally {
			starting = false;
		}
	}

	async function load(sid: string) {
		try {
			const feed = await api.getChat(store.runId, sid);
			// A slower load for a prior session must not clobber the current one
			// (fast session switch, or a session_bump racing the initial load).
			if (sid !== chatbotSid) return;
			// Clear the streaming preview ONLY when the assistant's own message has
			// finalized — detected by a new chatbot message appearing in the feed —
			// so the live bubble swaps seamlessly into the committed one. A refetch
			// triggered by a PEER message (operator / sub-agent) arriving mid-stream
			// must NOT wipe the preview: that peer message renders above (in
			// `messages`) while the stream keeps accumulating below. Comparing
			// chatbot-message counts (not just the tail) also handles a peer + a
			// finalize landing in the same refetch.
			const countChatbot = (ms: typeof messages) => ms.filter((m) => m.kind === 'chatbot').length;
			const finalized = countChatbot(feed.messages) > countChatbot(messages);
			messages = feed.messages;
			phaseCards = feed.phase_cards;
			usage = feed.usage;
			// A successful refetch clears any stale transient-error banner (e.g. a
			// brief UberProxy/network blip on a prior poll) — the chat is demonstrably
			// healthy again. Without this the red banner sticks over a working chat.
			error = '';
			if (finalized) {
				preview = null;
			}
			// Drop optimistic bubbles once the canonical message lands (content
			// match) or after 30s (safety, e.g. a send that silently differed).
			const present = new Set(messages.filter((m) => m.kind === 'operator').map((m) => m.content));
			const cutoff = Date.now() - 30_000;
			optimistic = optimistic.filter((o) => !present.has(o.content) && o.at > cutoff);
			await tick(); // let the bubbles render before measuring scrollHeight
			maybeScroll();
		} catch (e) {
			error = errorText(e); // '' when unreachable (global banner covers it)
		}
	}

	// Scroll to the bottom on new content, but only while pinned (the operator
	// hasn't scrolled up to read history).
	function maybeScroll() {
		if (!pinned || !scrollEl) return;
		const el = scrollEl;
		requestAnimationFrame(() => (el.scrollTop = el.scrollHeight));
	}

	// Wrap send with an optimistic echo; rollback + rethrow on failure so the
	// MessageBox keeps the draft for retry.
	async function sendMessage(content: string) {
		const opt = { key: crypto.randomUUID(), content, at: Date.now() };
		optimistic = [...optimistic, opt];
		pinned = true; // sending jumps back to the bottom
		maybeScroll();
		try {
			await api.sendMessage(store.runId, chatbotSid, content);
			error = ''; // a successful send clears any stale transient-error banner
		} catch (e) {
			optimistic = optimistic.filter((o) => o.key !== opt.key);
			error = errorText(e); // '' when unreachable (global banner covers it)
			throw e;
		}
	}

	// --- Tool-call detail popup ---
	// The chat feed deliberately omits tool-result bodies (they're large), so on
	// demand we fetch the call's step via the existing /events?step=N endpoint
	// (the step is already known — no DB-wide id search), find the call + its
	// matching tool_result by id, and render the pair in a modal via <ToolCall>.
	// ALWAYS a modal (native <dialog>), regardless of viewport: a tool call is
	// tied to one specific past event, not standing reference content, so it

	// --- Right-hand side panel: AMBIENT content (data-driven, no open/close
	// state of its own) + ARTIFACTS (the one remaining user-toggled, "transient"
	// slot). Ambient content (today: active sub-agents) is shown whenever
	// present, full detail by default. Opening artifacts never REPLACES it —
	// artifacts gets the bulk of the panel and ambient content demotes to a
	// compact footer, so a forgotten-open artifacts browser can never hide live
	// status.
	let artifactsOpen = $state(false);
	// Ref to the mounted ArtifactBrowser, so a pill click can imperatively
	// (re)open a file even when the browser is already showing a different one
	// (its initialFile prop only reacts to a CHANGED value — see openArtifactFile).
	let artifactBrowser = $state<ArtifactBrowser>();
	// Deep-selects a file when artifacts opens from a $AMPLIO_ARTIFACT_DIR/ pill
	// (ArtifactBrowser's `initialFile` prop); '' = open at the root.
	let artifactInitialFile = $state('');
	// The artifact browser reports its currently-previewed file here (bind), so
	// the header's Expand can open the full page at exactly that file.
	let artifactSelectedFile = $state('');
	// Persist the operator's open/closed choice AND currently-selected file per
	// run (not globally — a preference on one run shouldn't leak into another),
	// so both survive a reload. Read once when the run resolves; every change
	// writes back.
	const artifactsOpenKey = $derived(store.runId ? `amplio-artifacts-open:${store.runId}` : '');
	const artifactFileKey = $derived(store.runId ? `amplio-artifacts-file:${store.runId}` : '');
	// Guards the write-back effects below against a startup race: restoring
	// artifactsOpen=true also sets artifactInitialFile, but ArtifactBrowser only
	// mirrors that into artifactSelectedFile (bind:selectedFile) a render cycle
	// later, once it mounts and deep-selects it. Without this gate, the write
	// effect fires the INSTANT artifactFileKey resolves — with the still-stale
	// default '' — and clobbers the just-restored value in localStorage before
	// the child ever reports the real selection back. Closed immediately on every
	// restore (incl. a run switch, so the OLD run's file can't leak into the NEW
	// run's key), reopened one tick later once the mount has settled.
	let artifactsHydrated = $state(false);
	$effect(() => {
		if (!browser || !artifactsOpenKey) return;
		artifactsHydrated = false;
		// Compute the restored value into a LOCAL first, not `artifactsOpen ? ... :
		// ...` directly — branching on the live state would make $effect track
		// `artifactsOpen` as a dependency too (it auto-tracks every reactive read,
		// not just artifactsOpenKey), turning every later toggleArtifacts() click
		// into a re-trigger of THIS restore effect, which immediately overwrites
		// the click with whatever was last persisted. Reading from the local instead
		// keeps this effect's only dependency the run-scoped keys, as intended.
		const restoredOpen = localStorage.getItem(artifactsOpenKey) === '1';
		artifactsOpen = restoredOpen;
		// Only restore the file selection alongside a restored-open state, so it
		// can't linger and resurface on a LATER manual open (mirrors toggleArtifacts,
		// which always clears artifactInitialFile on close).
		artifactInitialFile = restoredOpen ? (localStorage.getItem(artifactFileKey) ?? '') : '';
		tick().then(() => {
			artifactsHydrated = true;
		});
	});
	$effect(() => {
		if (browser && artifactsOpenKey) localStorage.setItem(artifactsOpenKey, artifactsOpen ? '1' : '0');
	});
	$effect(() => {
		if (browser && artifactFileKey && artifactsHydrated) {
			localStorage.setItem(artifactFileKey, artifactSelectedFile);
		}
	});

	const SIDE_PANEL_MIN = 1100; // px; below this there's no room for the aside at all
	let viewportW = $state(typeof window !== 'undefined' ? window.innerWidth : 1400);
	const panelMode = $derived(viewportW >= SIDE_PANEL_MIN);
	// The aside renders when there's room AND something to show — either ambient
	// content or the artifacts browser. Never shown just because it COULD be.
	const asideVisible = $derived(panelMode && (activeSubagents.length > 0 || artifactsOpen));

	// Open an artifact file (from a $AMPLIO_ARTIFACT_DIR/ pill). Only makes sense
	// with panel room; on a narrow viewport we send the operator to the full
	// Artifacts page instead (deep-linked to the file).
	function openArtifactFile(file: string) {
		if (!panelMode) {
			// Empty file = root chip: open the full Artifacts page at the root (no ?file).
			const q = file ? `?file=${encodeURIComponent(file)}` : '';
			goto(`/runs/${store.runId}/artifacts${q}`);
			return;
		}
		if (artifactsOpen && artifactBrowser) {
			// Already open: the initialFile prop won't re-trigger a deep-select when
			// `file` is unchanged (e.g. re-clicking the same pill, or clicking a pill
			// for the file that seeded the panel while the user has since browsed
			// elsewhere). Drive the (re)select imperatively so it always lands. An
			// empty file is the root chip → navigate to the artifact root.
			if (file) {
				artifactBrowser.openFile(file);
			} else {
				artifactBrowser.openRoot();
			}
		} else {
			// Fresh open: seed via the prop; the browser deep-selects on mount (or
			// opens at the root when file is empty).
			artifactInitialFile = file;
			artifactsOpen = true;
		}
	}

	// The shell header's Artifacts toggle: it means exactly one thing (open/close
	// the artifacts browser) regardless of what ambient content is also showing.
	// On narrow viewports there's no aside at all, so this routes to the full
	// Artifacts page instead.
	function toggleArtifacts() {
		if (!panelMode) {
			goto(`/runs/${store.runId}/artifacts`);
			return;
		}
		artifactsOpen = !artifactsOpen;
		if (!artifactsOpen) artifactInitialFile = '';
	}

	// The panel's "Expand" bridge: leave the chat and open the full Artifacts page
	// at the given file (or the browser root).
	function expandArtifact(file: string) {
		const q = file ? `?file=${encodeURIComponent(file)}` : '';
		goto(`/runs/${store.runId}/artifacts${q}`);
	}

	// Event delegation for artifact pills inside {@html} markdown: a click on a
	// [data-artifact-path] element opens that file in the panel. One listener on
	// the scroll container covers every rendered message. Implemented as an ACTION
	// (imperative listener) rather than an inline onclick: the delegation target is
	// the non-interactive scroll div, and an inline handler there trips the a11y
	// lint — the real interactive elements are the <button.artifact-pill> pills
	// (keyboard-activatable), whose clicks bubble here.
	function artifactPills(node: HTMLElement) {
		const handler = (e: MouseEvent) => {
			const el = (e.target as HTMLElement)?.closest?.('[data-artifact-path]');
			if (!el) return;
			e.preventDefault();
			// The attribute is guaranteed present (closest matched on it); an empty
			// value is the ROOT chip (a bare folder mention), which openArtifactFile
			// routes to the artifact root.
			openArtifactFile(el.getAttribute('data-artifact-path') ?? '');
		};
		node.addEventListener('click', handler);
		return { destroy: () => node.removeEventListener('click', handler) };
	}

	function formatTokens(n: number): string {
		return n >= 1000 ? `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k` : String(n);
	}

	$effect(() => {
		const sid = chatbotSid;
		messages = [];
		phaseCards = [];
		preview = null;
		compacting = false;
		error = '';
		pinned = true;
		if (!sid) return;
		load(sid);
		const off = store.on((ev) => {
			// refetch_all is the (re)connection prime (global, no session_id). Use
			// it to recover after a dropped SSE / server restart — an idle chatbot
			// emits no session_bump, so this is the only reload signal then.
			if (ev.kind === 'refetch_all') {
				load(sid);
				return;
			}
			if (ev.session_id !== sid) return;
			if (ev.kind === 'session_bump') {
				load(sid); // load() clears the preview once the real message is in
			} else if (ev.kind === 'stream_chunk') {
				const base = preview ?? { text: '', thoughts: '', step: ev.step ?? 0 };
				preview = {
					text: base.text + (ev.text_delta ?? ''),
					thoughts: base.thoughts + (ev.thoughts_delta ?? ''),
					step: ev.step ?? base.step
				};
				maybeScroll();
			} else if (ev.kind === 'ephemeral_agents' && ev.ephemeral_kind === 'compaction') {
				// Context compaction for THIS session (session_id == sid, gated above)
				// started/ended — toggle the "compacting…" indicator directly off the
				// event (no refetch).
				compacting = ev.active === true;
				maybeScroll();
			}
		});
		return () => off();
	});

	function onScroll() {
		if (!scrollEl) return;
		pinned = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 80;
	}

	// (Chat-tab title "●" indicator removed — superseded by the run-level
	// unread system: a settled-state transition while the tab is hidden flips
	// the favicon dot AND surfaces a row badge on the dashboard.
	// See src/lib/updates.svelte.ts.)

	// Type-to-activate: a visible character typed anywhere on the page (no
	// modifier, focus not already in a field) jumps to the composer so you can
	// just start typing. We don't preventDefault, so the character itself lands in
	// the now-focused textarea.
	let composer = $state<HTMLTextAreaElement>();
	$effect(() => {
		if (!browser || !chatbotSid) return;
		const onKey = (e: KeyboardEvent) => {
			if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
			if (e.key.length !== 1 || e.key === ' ') return; // visible char only
			const el = document.activeElement as HTMLElement | null;
			if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) {
				return; // already typing somewhere
			}
			composer?.focus();
		};
		window.addEventListener('keydown', onKey);
		return () => window.removeEventListener('keydown', onKey);
	});

	// Tab title: "chat · <run> · Amplio". Most-specific bit (page kind) goes
	// first so it survives OS truncation; the run identifier carries enough
	// context to distinguish multiple open chats.
	const runLabel = $derived(store.detail?.title || store.detail?.task || store.runId);
</script>

<svelte:head>
	<title>{pageTitle(runLabel)}</title>
</svelte:head>

<!-- Track viewport width so the aside can appear/disappear with room; ESC
     closes the tool-call modal (native dialog handles this itself too). -->
<svelte:window bind:innerWidth={viewportW} />

{#if !store.detail}
	<p class="dim">Loading…</p>
{:else if !chatbotSid}
	<!-- Empty-state for autonomous runs with no chatbot session yet. Plain
	     left-aligned text + a single button — no card chrome, so the page
	     doesn't render a half-width box on wide viewports. Workspace
	     sharing / read-only semantics are NOT mentioned: those are
	     constraints the chatbot itself handles; the operator doesn't need
	     to know them up front. "Co-pilot" avoided to dodge the Microsoft
	     branding collision. -->
	<div class="startcard">
		<p>
			Autonomous run — start a companion chatbot to inspect results or steer ongoing run.
		</p>
		{#if auth.authed}
			<button class="primary" onclick={startChatbot} disabled={starting}>
				{starting ? 'Starting…' : 'Start Chat'}
			</button>
		{:else}
			<p class="dim small">Read-only view — the run owner can start a chat.</p>
		{/if}
		{#if error}<p class="err">{error}</p>{/if}
	</div>
{:else}
	<div class="chat-layout">
		<!-- Artifacts toggle: rendered floating over chat when the aside isn't shown
		     (nothing ambient, artifacts closed), or inside the aside's header once it
		     is. Always means the same thing — open/close the artifacts browser. -->
		{#snippet artifactsToggle(float: boolean)}
			<button
				class="panel-toggle"
				class:panel-toggle--float={float}
				class:on={artifactsOpen}
				title={artifactsOpen ? 'Close artifacts' : 'Open artifacts'}
				aria-pressed={artifactsOpen}
				onclick={toggleArtifacts}
			>
				<FolderIcon size={16} weight="bold" />
				<span>Artifacts</span>
			</button>
		{/snippet}
		<!-- 0-arg wrapper so the toggle can be passed as ArtifactBrowser's
		     `toolbarEnd` Snippet prop, riding at the end of ITS toolbar row when
		     artifacts is open (so the button stays in the same header spot whether
		     that row is showing "Status" or the browser's own breadcrumbs). -->
		{#snippet artifactsHeaderToggle()}
			{@render artifactsToggle(false)}
		{/snippet}
		{#if panelMode && !asideVisible}
			<div transition:fade={{ duration: 150 }}>
				{@render artifactsToggle(true)}
			</div>
		{/if}
		<!-- Sub-agent status, as snippets sharing one data source so the aside and
		     its compact footer / the status bar never duplicate the rendering logic:
		       - subagentsRows: one line per sub-agent (icon, name, task fade, status
		         badge) — used in the aside where there's room, one row per line.
		       - subagentsCompact: a one-line count — used in the status bar
		         (always: whether the aside is showing the full rows elsewhere, or
		         there's no room for an aside at all) and in the aside's own footer
		         once artifacts is open. -->
		{#snippet subagentsRows()}
			{#each activeSubagents as s (s.session_id)}
				{@const SubIcon = iconForName(s.session_id) ?? AndroidLogoIcon}
				<a class="sa-row" href={`/runs/${store.runId}/sessions/${s.session_id}`}>
					<SubIcon size={14} />
					<span class="sa-name mono">{s.session_id}</span>
					{#if s.task}<span class="sa-task dim small" title={s.task}>{s.task}</span>{/if}
					<StatusBadge status={s.status} />
				</a>
			{/each}
		{/snippet}
		<!-- `clickable`: only the aside's OWN footer (rendered while artifacts is
		     already open) makes this act like "expand" — clicking toggles artifacts
		     closed, revealing the full subagentsRows underneath. The status-bar use
		     has no such row to reveal (there's no aside at all, or artifacts isn't
		     open there), so it stays plain, unclickable text. -->
		{#snippet subagentsCompact(clickable: boolean = false)}
			{#if clickable}
				<button class="sa-compact-btn dim small" onclick={toggleArtifacts}>
					<span class="dot working"></span>
					<span>{activeSubagents.length} sub-agent{activeSubagents.length === 1 ? '' : 's'} active</span>
				</button>
			{:else}
				<span class="sa-compact dim small">
					<span class="dot working"></span>
					<span>{activeSubagents.length} sub-agent{activeSubagents.length === 1 ? '' : 's'} active</span>
				</span>
			{/if}
		{/snippet}
	<div class="chat">
		{#if error}<p class="err">{error}</p>{/if}
		<!-- use:artifactPills delegates artifact-pill (data-artifact-path) clicks from
		     the {@html} markdown to open the file in the side panel. -->
		<div class="scroll" bind:this={scrollEl} onscroll={onScroll} use:artifactPills>
			{#each phaseCards as c (c.end_step)}
				<div class="phase">
					<!-- The WHOLE card opens this rolled-up phase in the read-only chat
					     log, but the anchor wraps only the head: the summary below is
					     rendered markdown that may contain its own anchors, and nesting
					     those inside an <a> is invalid HTML. The head's ::after overlay
					     (see .phase-head::after) stretches the hit area over the card
					     instead — one real link, card-sized target, markdown links inside
					     the summary still reachable (they sit above the overlay). -->
					<a
						class="phase-head"
						href={logHref(store.runId, chatbotSid, 'chat', `${c.start_step}-${c.end_step}`)}
						title="Read this phase in the chat log"
					>
						<span class="phase-range mono dim">steps {c.start_step}–{c.end_step}</span>
						<span class="phase-title">{c.title}</span>
						<!-- Explicit affordance: the head looked like plain text, so the
						     link went unnoticed. The icon is the nav rail's Chat-log icon,
						     naming the destination rather than just "this is clickable". -->
						<span class="phase-open dim small"><ScrollIcon size={13} />read</span>
					</a>
					<div class="md dim small">{@html renderMarkdown(c.summary)}</div>
				</div>
			{/each}

			<!-- The transcript itself (bubbles + tool pills + the tool-detail modal)
			     lives in <ChatMessages>, shared verbatim with the read-only session-log
			     viewer. This page keeps only what is live: the scroll container and its
			     pinning, the streaming preview and the optimistic echoes it passes in. -->
			<ChatMessages
				runId={store.runId}
				sid={chatbotSid}
				{messages}
				{scrollEl}
				{preview}
				{optimistic}
				{working}
			/>

			{#if messages.length === 0 && optimistic.length === 0 && !preview && !compacting && phaseCards.length === 0}
				<p class="dim">No messages yet.</p>
			{/if}
		</div>


		<!-- The status bar always shows just a count, never full per-agent detail:
		     on wide viewports the aside already shows the full rows (no need to
		     duplicate them here); on narrow viewports there's no aside at all, so
		     full detail would wrap the bar across multiple lines past a few
		     sub-agents — a rare enough case that a bare count is an acceptable
		     trade for staying on one line. -->
		<div class="statusbar">
			<div class="sb-lhs">
				{#if compacting}
					<span class="dot working"></span>
					<span class="dim small">compacting context…</span>
				{:else if working || activeSubagents.length > 0}
					<span class="dot working"></span>
					<span class="dim small">working</span>
					<!-- Only when the aside ISN'T showing sub-agent info (i.e. narrow
					     viewports: on wide ones asideVisible is true whenever sub-agents are
					     active, and the aside shows them as full rows or a compact footer).
					     Prevents showing the count here AND in the aside simultaneously. -->
					{#if activeSubagents.length > 0 && !asideVisible}
						{@render subagentsCompact()}
					{/if}
				{:else if unhealthy}
					<span class="dot warn"></span>
					<span class="dim small">last turn crashed — sending will retry</span>
				{:else}
					<span class="dot ok"></span>
					<span class="dim small">chatbot online</span>
				{/if}
			</div>
			<div class="sb-rhs dim small">
				<span>step {step}</span>
				{#if usage}
					<span class="sep">·</span>
					<span
						title={`prompt ${usage.prompt_tokens} · completion ${usage.completion_tokens} · total ${usage.total_tokens}`}
						>~{formatTokens(usage.total_tokens)} tokens</span
					>
				{/if}
				<span class="sep">·</span>
				<a href={`/runs/${store.runId}/sessions/${chatbotSid}`}>full stream &rarr;</a>
			</div>
		</div>

		{#if auth.authed}
			<MessageBox
				bind:element={composer}
				onSend={sendMessage}
				draftKey={`amplio-chat-draft:${store.runId}:${chatbotSid}`}
				placeholder="Message the chatbot — Enter to send, Shift+Enter for newline"
			/>
		{/if}
	</div>

	<!-- Right-hand aside: visible whenever there's ambient content (active
	     sub-agents) or the artifacts browser is open. Ambient content is never
	     replaced by opening artifacts — it demotes to a compact footer instead,
	     so an open artifacts browser can't hide live status updates. -->
	{#if asideVisible}
		<aside class="side-panel" transition:fly={{ x: 16, duration: 150 }}>
			<!-- Shell header: fixed position always at the top of the aside, but its
			     CONTENT swaps — "Status" + the toggle when showing ambient content, or
			     ArtifactBrowser's own breadcrumb toolbar (with the toggle riding in via
			     toolbarEnd) once artifacts is open. This keeps the toggle button in the
			     same visual spot in both states, and avoids a duplicate "Artifacts"
			     title stacked above the browser's own breadcrumbs. -->
			{#if !artifactsOpen}
				<header class="sp-head">
					<span class="sp-title">Status</span>
					{@render artifactsToggle(false)}
				</header>
				<div class="sp-body">
					{#if activeSubagents.length > 0}
						<div class="sp-ambient-head dim small">Sub-agents ({activeSubagents.length})</div>
						<div class="sp-ambient-list">
						{@render subagentsRows()}
						</div>
					{/if}
				</div>
			{:else}
				<div class="sp-body sp-body--flush sp-body--grow">
					<ArtifactBrowser
						bind:this={artifactBrowser}
						runId={store.runId}
						initialFile={artifactInitialFile}
						bind:selectedFile={artifactSelectedFile}
						compact
						onExpand={expandArtifact}
						expandBase={`/runs/${store.runId}/artifacts`}
						toolbarEnd={artifactsHeaderToggle}
					/>
				</div>
				{#if activeSubagents.length > 0}
					<footer class="sp-footer">
						{@render subagentsCompact(true)}
					</footer>
				{/if}
			{/if}
		</aside>
	{/if}
	</div>
{/if}

<style>
	/* No card chrome (no border / background / padding) — the empty-state
	   message should read as inline page text, not a boxed panel. Modest
	   max-width keeps the line length comfortable on wide viewports
	   without imposing a visible boundary. */
	.startcard {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.6rem;
		max-width: 40rem;
	}
	.startcard p {
		margin: 0;
	}
	/* Layout row: the conversation column + an optional right aside. Fills the
	   run layout's scroll pane; each child scrolls internally. Centered as a group
	   so, with no aside, .chat sits centered exactly as before; with an aside, the
	   pair is centered and the aside takes the freed right-hand width. */
	.chat-layout {
		flex: 1;
		min-height: 0;
		display: flex;
		gap: 1rem;
		justify-content: center;
		/* Anchor for the top-right floating Artifacts toggle. */
		position: relative;
	}
	/* Artifacts toggle. Base form is an in-flow labeled pill (used inside the
	   aside's shell header). The --float modifier lifts it to the top-right of the
	   shell interior as the OPEN affordance when the aside isn't rendered at all.
	   Dim by default, accent when selected (artifacts open). */
	.panel-toggle {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.25rem 0.6rem;
		flex-shrink: 0;
		border: 1px solid transparent;
		border-radius: var(--radius-pill);
		background: none;
		color: var(--text-dim);
		font-size: var(--fs-sm);
		cursor: pointer;
	}
	/* Floating variant (aside not rendered): pinned to the top-right of the shell
	   interior. */
	.panel-toggle--float {
		position: absolute;
		top: calc(0.4rem + 1px);
		right: calc(0.5rem + 1px);
		z-index: 5;
		border-color: var(--border);
		background: var(--bg-elev);
	}
	.panel-toggle:hover {
		color: var(--text);
		background: var(--bg-elev2);
	}
	.panel-toggle.on {
		color: var(--accent);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border-color: color-mix(in srgb, var(--accent) 40%, transparent);
	}
	/* Fill the run layout's scroll pane; the message list scrolls internally and
	   the status bar + input stay pinned below it. */
	.chat {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		min-height: 0;
		/* FIXED reading width: the conversation column is a stable 54rem — centered
		   when solo (via the row's justify-content), and it does NOT reflow when the
		   side panel opens (the panel takes the remaining width, below). max-width
		   caps it to the pane if the pane is ever narrower than 54rem (safety). */
		flex: 0 0 54rem;
		max-width: 100%;
		min-width: 0;
	}
	/* The aside takes ALL remaining width. It can't grow unbounded: the whole app
	   is capped at 1600px (root <main>), so this only ever fills the bounded pane
	   remainder. On narrow viewports the aside isn't rendered at all (asideVisible
	   is gated at SIDE_PANEL_MIN in the script) — the tool detail is always a
	   modal regardless, and artifact clicks route to the full Artifacts page. */
	.side-panel {
		flex: 1 1 0;
		min-width: 0;
		display: flex;
		flex-direction: column;
		min-height: 0;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		background: var(--bg-elev);
	}
	/* Shell header: fixed chrome, always present whenever the aside is. Title
	   names what's below it ("Status" for ambient content, "Artifacts" once the
	   browser is open). */
	.sp-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.4rem 0.5rem 0.4rem 0.7rem;
		border-bottom: 1px solid var(--border);
	}
	.sp-title {
		color: var(--text-dim);
		font-size: var(--fs-sm);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.sp-body {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		padding: 0.7rem;
	}
	/* Ambient (Status) widgets: a small dim label above each widget's rows. */
	.sp-ambient-head {
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-bottom: 0.35rem;
	}
	.sp-ambient-list {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}
	/* One sub-agent row in the aside: one line (icon · name · task fade · status
	   badge), mirroring SessionTree's row layout so a session reads identically
	   in both places. Task truncates via mask-fade (not clipped text), so
	   hovering the row still surfaces the full string via title=. */
	.sa-row {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.5rem;
		border-radius: var(--radius-sm);
		color: var(--text);
	}
	.sa-row:hover {
		text-decoration: none;
		background: var(--bg-elev2);
	}
	.sa-row > :global(svg) {
		flex-shrink: 0;
		color: var(--text-dim);
	}
	.sa-name {
		flex-shrink: 0;
		font-size: var(--fs-sm);
		font-weight: 500;
	}
	.sa-task {
		flex: 1 1 0;
		min-width: 0;
		overflow: hidden;
		white-space: nowrap;
		mask-image: linear-gradient(to right, black calc(100% - 1.5rem), transparent);
		-webkit-mask-image: linear-gradient(to right, black calc(100% - 1.5rem), transparent);
	}
	/* The artifact browser manages its own scroll/padding, so its host body is
	   flush (no double padding / nested scrollbars). --grow additionally forces it
	   to fill all space ABOVE the ambient footer (rather than sizing to content),
	   so the footer stays pinned to the bottom of the aside. */
	.sp-body--flush {
		overflow: hidden;
		padding: 0;
		display: flex;
		flex-direction: column;
	}
	.sp-body--grow {
		flex: 1 1 auto;
	}
	/* Ambient footer: shown UNDER the artifact browser (never covering it), so
	   an open browser can't hide live status. Fixed height (compact rows only)
	   and visually distinct from the browser content above it. */
	.sp-footer {
		flex: 0 0 auto;
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		padding: 0.4rem 0.7rem;
		border-top: 1px solid var(--border);
		background: var(--bg);
	}
	/* The aside-footer's compact sub-agent count acts like an "expand" affordance
	   (click closes artifacts, revealing the full subagentsRows underneath) —
	   styled to still read as plain dim text, not an obvious button, since its
	   main job is informational and the click is a bonus shortcut. */
	.sa-compact-btn {
		background: none;
		border: none;
		padding: 0;
		font: inherit;
		color: inherit;
		text-align: left;
		cursor: pointer;
	}
	/* Both compact variants (clickable button + plain span) align a small icon
	   with the label on one baseline. */
	.sa-compact,
	.sa-compact-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
	}
	.sa-compact > :global(svg),
	.sa-compact-btn > :global(svg) {
		flex-shrink: 0;
	}
	.sa-compact-btn:hover {
		color: var(--text);
	}
	.scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		/* No scrollbar: the segment navigator in the right gutter replaces it.
		   Scrolling itself is untouched (wheel, trackpad, keyboard, drag). */
		scrollbar-width: none;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		/* Horizontal gutter inside .scroll, so the hover-revealed copy buttons
		   that sit outside each message (left of chatbot, right of bubble) land
		   in this padding area — visible, not clipped. (overflow-y:auto implies
		   overflow-x:auto per CSS spec, so the copy MUST be inside .scroll's box
		   to render. Padding does exactly that.) */
		padding-inline: 1.8rem;
	}
	.statusbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: 0.25rem 0.1rem;
		border-top: 1px solid var(--border);
	}
	.sb-lhs {
		display: flex;
		align-items: center;
		gap: 0.45rem;
		flex-wrap: wrap;
		min-width: 0;
	}
	.sb-rhs {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		white-space: nowrap;
	}
	.sb-rhs .sep {
		color: var(--text-dim);
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}
	.dot.ok {
		background: var(--ok);
	}
	.dot.warn {
		background: var(--warn);
	}
	.dot.working {
		background: var(--chat);
		animation: pulse 1.1s ease-in-out infinite;
	}
	@keyframes pulse {
		50% {
			opacity: 0.3;
		}
	}
	.phase {
		position: relative; /* containing block for the head's stretched hit area */
		border: 1px dashed var(--border);
		border-radius: var(--radius-md);
		padding: 0.5rem 0.75rem;
		background: var(--bg);
		cursor: pointer;
	}
	/* Hover is driven by the CARD, not the head: the whole thing is the target,
	   so the whole thing must react (the head-only hover cue was why the link
	   went unnoticed). */
	.phase:hover {
		border-color: var(--accent-dim);
		background: var(--bg-elev);
	}
	.phase:hover .phase-title,
	.phase:hover .phase-open {
		color: var(--accent);
	}
	/* Keyboard parity: the ring lands on the card, not just the head text. */
	.phase:has(.phase-head:focus-visible) {
		outline: 2px solid var(--accent-dim);
		outline-offset: 2px;
	}
	.phase-head {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
		margin-bottom: 0.2rem;
		color: var(--text);
	}
	.phase-head:hover {
		text-decoration: none;
	}
	/* Stretched link: an invisible overlay that grows the head anchor's hit area
	   to the full card without nesting anchors. */
	.phase-head::after {
		content: '';
		position: absolute;
		inset: 0;
		border-radius: inherit;
	}
	/* … with links inside the summary lifted above it so they stay clickable. */
	.phase .md :global(a) {
		position: relative;
		z-index: 1;
	}
	.phase-title {
		font-weight: 600;
	}
	/* Right-aligned "read" affordance; visible but quiet until hover. */
	.phase-open {
		margin-left: auto;
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
		align-self: center;
	}
	.scroll::-webkit-scrollbar {
		display: none;
	}
</style>

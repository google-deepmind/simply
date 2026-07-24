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
	import type { ChatBubble, ChatToolCall, PhaseCard, ChatUsage, AgentEvent } from '$lib/types';
	import MessageBox from '$lib/components/MessageBox.svelte';
	import { auth } from '$lib/auth.svelte';
	import CopyButton from '$lib/components/CopyButton.svelte';
	import ToolCall from '$lib/components/ToolCall.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import ArtifactBrowser from '$lib/components/ArtifactBrowser.svelte';
	import { goto } from '$app/navigation';
	import Thoughts from '$lib/components/Thoughts.svelte';
	import { renderMarkdown } from '$lib/markdown';
	import {
		CircleNotchIcon,
		TerminalIcon,
		BrainIcon,
		AndroidLogoIcon,
		CaretRightIcon,
		FolderIcon,
		CheckCircleIcon,
		WarningCircleIcon,
		ProhibitIcon
	} from 'phosphor-svelte';
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
	let now = $state(Date.now()); // ticks in-flight so pending gap timestamps update live

	let scrollEl = $state<HTMLDivElement>();
	let pinned = true; // follow the bottom unless the operator scrolls up

	// Intra-turn timestamp clustering: a left-aligned timestamp separates clusters
	// of assistant output. It's shown after a non-operator bubble when the bubble
	// ends a turn (terminal: a chatbot reply with no tool calls), or a non-trivial
	// gap precedes the next output. For the in-flight tail (a tool running, no next
	// output yet) the gap is measured against `now`, so a timestamp surfaces once
	// the wait gets long; `reserved` pre-allocates that line so its appearance
	// doesn't shift layout, and collapses if the next output lands within the gap.
	// Operator bubbles are untouched (they keep their own in-bubble timestamp).
	const GAP_MS = 45_000;
	type Row = { m: ChatBubble; showSep: boolean; reserved: boolean };
	const rows = $derived.by((): Row[] => {
		const out: Row[] = [];
		for (let i = 0; i < messages.length; i++) {
			const m = messages[i];
			// Only the chatbot's own inline output is cluster-separated. Operator,
			// agent, and environment are discrete bubbles that carry their own
			// timestamp, so they never get a separator.
			if (m.kind !== 'chatbot') {
				out.push({ m, showSep: false, reserved: false });
				continue;
			}
			const isLast = i === messages.length - 1;
			const at = new Date(m.created_at).getTime();
			let nextTime: number | null = null;
			if (i + 1 < messages.length) nextTime = new Date(messages[i + 1].created_at).getTime();
			else if (preview || working) nextTime = now; // in-flight: measure against now
			if (nextTime === null) {
				// Conversation currently rests on this output → end-of-turn timestamp.
				out.push({ m, showSep: true, reserved: false });
				continue;
			}
			// A timestamp only when a non-trivial gap precedes the next output, so
			// back-to-back turns (e.g. an inbound message waking a new turn) blend.
			const gapped = nextTime - at > GAP_MS;
			// Reserve the line only for the awaiting tail (no preview yet); once it
			// streams or the gap resolves, showSep takes over (same height).
			out.push({ m, showSep: gapped, reserved: isLast && working && !preview && !gapped });
		}
		return out;
	});

	// The index in `rows` AFTER which the streaming preview belongs: it occupies
	// the assistant's call step (preview.step = T), so it renders after every
	// message with step <= T and BEFORE any peer that arrived during generation
	// (which lands at T+1). This keeps the during-stream order identical to the
	// settled order, so a peer never jumps from above to below the assistant
	// bubble when the stream finalizes. -1 = before all rows; rows.length = after all.
	const previewAt = $derived.by((): number => {
		if (!preview) return -1;
		let idx = rows.length; // default: after everything (no later peer)
		for (let i = 0; i < rows.length; i++) {
			if (rows[i].m.step > preview.step) {
				idx = i;
				break;
			}
		}
		return idx;
	});

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
	// never competes with the ambient/artifacts side panel below.
	type ToolDetail = {
		name: string;
		args: string;
		result?: string;
		isError: boolean;
		attachments: AgentEvent['attachments'];
	};
	let toolDetail = $state<ToolDetail | null>(null);
	let toolDetailLoading = $state(false);
	let toolDetailErr = $state('');
	let toolDialog: HTMLDialogElement | undefined = $state();

	// Sync the tool-detail <dialog> to `toolDetail` presence. We can't use
	// bind:open to *open* a modal dialog (the spec forbids setting it true), so
	// drive showModal()/close() imperatively; the dialog's own close event (ESC,
	// backdrop dismiss) clears toolDetail via onclose below.
	$effect(() => {
		if (!toolDialog) return;
		if (toolDetail && !toolDialog.open) toolDialog.showModal();
		else if (!toolDetail && toolDialog.open) toolDialog.close();
	});

	async function openToolDetail(step: number, callId: string, name: string, errored: boolean) {
		toolDetail = { name, args: '', isError: errored, attachments: [] };
		toolDetailLoading = true;
		toolDetailErr = '';
		try {
			const events = await api.getEvents(store.runId, chatbotSid, step);
			let args = '';
			for (const e of events) {
				const tc = e.event.tool_calls?.find((c) => c.id === callId);
				if (tc) args = tc.arguments;
			}
			const res = events.find((e) => e.event.type === 'tool_result' && e.event.tool_call_id === callId);
			toolDetail = {
				name,
				args,
				result: res?.event.content,
				isError: res?.event.is_error ?? errored,
				attachments: res?.event.attachments ?? []
			};
		} catch (e) {
			toolDetailErr = String(e);
		} finally {
			toolDetailLoading = false;
		}
	}

	function closeToolDetail() {
		toolDetail = null;
	}

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

	// Tick `now` once a second while the turn is in-flight (a tool may be running)
	// or sub-agents are active, so elapsed + pending gap timestamps update live.
	$effect(() => {
		if (!working && activeSubagents.length === 0) return;
		const t = setInterval(() => (now = Date.now()), 1000);
		return () => clearInterval(t);
	});

	function onScroll() {
		if (!scrollEl) return;
		pinned = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 80;
	}

	// Full local timestamp with timezone, e.g. "2026-06-05 22:24:44 PDT".
	function formatTime(t: string | number): string {
		const d = new Date(t);
		const p = (n: number) => String(n).padStart(2, '0');
		const date = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
		const time = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
		const tz =
			new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' })
				.formatToParts(d)
				.find((x) => x.type === 'timeZoneName')?.value ?? '';
		return `${date} ${time}${tz ? ' ' + tz : ''}`;
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
					<div class="phase-head">
						<span class="phase-range mono dim">steps {c.start_step}–{c.end_step}</span>
						<span class="phase-title">{c.title}</span>
					</div>
					<div class="md dim small">{@html renderMarkdown(c.summary)}</div>
				</div>
			{/each}

			<!-- Chatbot output, rendered inline (no bubble) so its width is stable;
			     shared by the committed message and the live streaming preview. -->
			{#snippet chatbotInline(
				content: string,
				thoughts: string,
				toolCalls: ChatToolCall[],
				streaming: boolean,
				step: number
			)}
				<div class="asst-inline">
					{#if content && !streaming}
						<span class="asst-copy"><CopyButton text={content} /></span>
					{/if}
					{#if thoughts}<Thoughts {thoughts} {streaming} />{/if}
					{#if content}<div class="md">{@html renderMarkdown(content, { linkifyArtifacts: true })}</div>{/if}
					{#if streaming}<span class="cursor"></span>{/if}
					{#if toolCalls.length}
						<div class="tools">
						{#each toolCalls as tc}
							{#if tc.completed}
								{@const Icon = toolIcon(tc.name)}
								<!-- Completed call: a button opening the call+result detail. -->
								<button
									class="pill done"
									class:error={tc.errored}
									title="Show tool call detail"
									onclick={() => openToolDetail(step, tc.id, tc.name, tc.errored ?? false)}
								>
									<Icon size={12} />
									{tc.verb || tc.name}{#if tc.detail}<span class="pill-detail">{tc.detail}</span>{/if}
								</button>
							{:else}
								<span class="pill running">
									<CircleNotchIcon size={12} class="spin" />
									{tc.verb || tc.name}{#if tc.detail}<span class="pill-detail">{tc.detail}</span>{/if}
								</span>
							{/if}
						{/each}
						</div>
					{/if}
				</div>
			{/snippet}

			<!-- The live streaming bubble, rendered at its chronological step slot
			     (preview.step = T) so a peer arriving mid-stream (step T+1) stays
			     below it both during AND after the stream finalizes — no reorder jump. -->
			{#snippet previewBubble()}
				{#if preview}
					{@render chatbotInline(preview.text, preview.thoughts, [], true, preview.step)}
				{/if}
			{/snippet}

			{#if previewAt === 0}{@render previewBubble()}{/if}
			{#each rows as row, i (row.m.event_id)}
				{@const m = row.m}
				{#if m.kind === 'operator'}
					<div class="row user">
						<div class="bubble">
							<span class="bubble-copy"><CopyButton text={m.content} /></span>
							<!-- pre-wrap lives on .content (not the bubble) so the
							     literal whitespace between <span> and the content
							     in this template source doesn't render as a
							     leading space/newline inside the user message. -->
							<span class="content">{m.content}</span>
							<div class="time dim">{formatTime(m.created_at)}</div>
						</div>
					</div>
				{:else if m.kind === 'environment'}
					<div class="row inbound">
						<div class="bubble env">
							<span class="bubble-copy"><CopyButton text={m.content} /></span>
							<div class="ihead dim small"><TerminalIcon size={13} />{m.from || 'environment'}</div>
							<pre class="pre">{m.content}</pre>
							<div class="time dim">{formatTime(m.created_at)}</div>
						</div>
					</div>
				{:else if m.kind === 'agent'}
					{@const FromIcon = iconForName(m.from) ?? AndroidLogoIcon}
					<div class="row inbound">
						<div class="bubble agent">
							<span class="bubble-copy"><CopyButton text={m.content} /></span>
							<div class="ihead dim small">
								<FromIcon size={13} />
								{#if m.from}
									<a href={`/runs/${store.runId}/sessions/${m.from}`}>{m.from}</a>
								{:else}
									agent
								{/if}
							</div>
							<div class="md">{@html renderMarkdown(m.content)}</div>
							<div class="time dim">{formatTime(m.created_at)}</div>
						</div>
					</div>
				{:else if m.kind === 'child_result'}
					<!-- A spawned sub-agent's terminal result, posted back to this session.
					     Rendered like an inbound agent bubble, badged with the verdict. -->
					{@const FromIcon = iconForName(m.from) ?? AndroidLogoIcon}
					<div class="row inbound">
						<div class="bubble agent child-result" class:crashed={m.verdict === 'crashed'} class:cancelled={m.verdict === 'cancelled'}>
							<span class="bubble-copy"><CopyButton text={m.content} /></span>
							<div class="ihead dim small">
								<FromIcon size={13} />
								{#if m.from}
									<a href={`/runs/${store.runId}/sessions/${m.from}`}>{m.from}</a>
								{:else}
									sub-agent
								{/if}
								<span class="verdict verdict-{m.verdict}">
									{#if m.verdict === 'concluded'}<CheckCircleIcon size={13} weight="fill" />concluded
									{:else if m.verdict === 'crashed'}<WarningCircleIcon size={13} weight="fill" />crashed
									{:else if m.verdict === 'cancelled'}<ProhibitIcon size={13} weight="fill" />cancelled
									{:else}{m.verdict}{/if}
								</span>
							</div>
							<div class="md">{@html renderMarkdown(m.content)}</div>
							<div class="time dim">{formatTime(m.created_at)}</div>
						</div>
					</div>
				{:else if m.kind === 'compaction'}
					<!-- Context-compaction marker: a full-width dashed card (like the phase
					     cards in the trajectory view). Collapsed it's just the header;
					     click to fold/unfold the summary. It's meta, not a turn. -->
					<details class="compaction">
						<summary class="compaction-head">
							<span class="caret"><CaretRightIcon size={12} weight="bold" /></span>
							<BrainIcon size={14} />
							<span class="compaction-title">Context compacted</span>
							<span class="compaction-range mono dim">step {m.step}</span>
						</summary>
						<div class="md dim small compaction-body">{@html renderMarkdown(m.content)}</div>
					</details>
				{:else}
					{@render chatbotInline(m.content, m.thoughts ?? '', m.tool_calls, false, m.step)}
				{/if}

				{#if row.showSep || row.reserved}
					<div class="sep dim" class:empty={!row.showSep}>
						{#if row.showSep}{formatTime(m.created_at)}{/if}
					</div>
				{/if}
				{#if previewAt === i + 1}{@render previewBubble()}{/if}
			{/each}

			{#each optimistic as o (o.key)}
				<div class="row user">
					<div class="bubble pending">
						<!-- Same .content wrapper as the persisted bubble below
						     (line ~382) so `.user .bubble .content`'s
						     `white-space: pre-wrap` applies and multi-paragraph
						     pastes don't collapse to one line — otherwise the
						     bubble would visibly grow taller on the swap to the
						     server-echoed canonical message. -->
						<span class="content">{o.content}</span>
						<!-- "sending…" occupies the same slot as the committed timestamp, so
						     the swap doesn't shift height or flash a changing time. -->
						<div class="time dim">sending…</div>
					</div>
				</div>
			{/each}

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

<!-- Tool-call detail modal: ALWAYS how a tool call's detail is shown, regardless
     of viewport (see the ToolDetail section above) — it's tied to one specific
     past event, not standing reference content, so it never competes with the
     ambient/artifacts aside. Native <dialog> gives ESC-to-close, a focus trap,
     and a styleable ::backdrop; a click on the dialog itself (the backdrop)
     dismisses. onclose clears toolDetail, which the $effect above syncs from. -->
<dialog
	bind:this={toolDialog}
	onclose={closeToolDetail}
	onclick={(e) => {
		if (e.target === toolDialog) closeToolDetail();
	}}
	class="tool-modal"
>
	{#if toolDetail}
		<div class="tool-modal-inner">
			<div class="modal-head">
				<span class="mono">{toolDetail.name}</span>
				<button class="modal-x" title="Close" onclick={closeToolDetail}>×</button>
			</div>
			{#if toolDetailLoading}
				<p class="dim small">Loading…</p>
			{:else if toolDetailErr}
				<p class="err small">{toolDetailErr}</p>
			{:else}
				<ToolCall
					name={toolDetail.name}
					args={toolDetail.args}
					result={toolDetail.result}
					isError={toolDetail.isError}
					attachments={toolDetail.attachments ?? []}
					runId={store.runId}
				/>
			{/if}
		</div>
	{/if}
</dialog>
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
	.pending {
		opacity: 0.55;
	}
	.row {
		display: flex;
	}
	.row.user {
		justify-content: flex-end;
	}
	.bubble {
		position: relative;
		max-width: 80%;
		border-radius: var(--radius-lg);
		padding: 0.5rem 0.75rem;
		border: 1px solid var(--border);
	}
	/* Copy button sits FLUSH at the bubble's outer-right edge (left:100%, top:0)
	   so it lives in the layout's right gutter — same x for every operator/agent/
	   env bubble (no jitter). DOM-child + flush = no hover gap. */
	.bubble-copy {
		position: absolute;
		top: 0;
		left: 100%;
		padding-left: 0.35rem;
		opacity: 0;
		transition: opacity 0.1s ease;
	}
	.bubble:hover .bubble-copy {
		opacity: 1;
	}
	.time {
		margin-top: 0.3rem;
		font-size: var(--fs-xs);
	}
	.user .bubble {
		background: color-mix(in srgb, var(--chat) 14%, transparent);
		border-color: color-mix(in srgb, var(--chat) 40%, transparent);
		/* min-width:0 lets the bubble shrink inside the flex row so a long
		   unbreakable run (a bare //path or URL in the operator's message) can
		   actually wrap instead of forcing the bubble past its max-width. The
		   character-level break itself lives on .content (overflow-wrap:anywhere). */
		min-width: 0;
	}
	/* User message content: pre-wrap preserves the operator's typed newlines
	   + multi-space alignment. Scoped to .content (not .bubble) so the
	   bubble's source-code whitespace between <span class="bubble-copy">
	   and the content node doesn't leak in as a leading visible space. */
	.user .bubble .content {
		white-space: pre-wrap;
		/* Force mid-token breaking of long unbreakable runs (paths, URLs) that have
		   no space to wrap at; ordinary prose still breaks at spaces. Kept here with
		   pre-wrap so typed newlines/alignment are preserved. */
		overflow-wrap: anywhere;
	}
	/* Chatbot output: inline (no bubble) and full-width within the (already
	   width-controlled) conversation column. The container caps the measure, so we
	   don't ALSO clamp the assistant to the left 80% — that left an awkward empty
	   band on the right (most visible beside the tool-call side panel). User
	   messages stay capped + flush-right; assistant prose fills the column. */
	.asst-inline {
		position: relative;
		width: 100%;
		align-self: stretch;
		padding: 0.1rem 0.2rem;
	}
	/* Copy button sits FLUSH at the inline's outer-left edge (right:100%, top:0)
	   so it lives in the layout's left gutter — same x for every chatbot turn (no
	   jitter). It's a DOM child of .asst-inline, so hovering it counts as hovering
	   the parent; flush positioning means zero gap to cross, so the reveal stays
	   stable when the cursor moves onto it. */
	.asst-copy {
		position: absolute;
		top: 0;
		right: 100%;
		padding-right: 0.35rem;
		opacity: 0;
		transition: opacity 0.1s ease;
	}
	.asst-inline:hover .asst-copy {
		opacity: 1;
	}
	/* Inbound messages (peer agent / environment notify): right-flushed bubbles
	   (like the operator) in distinct colors. The chatbot's own output stays left
	   inline, so the left column is "the chatbot talking" and the right column is
	   everything else (operator input + inbound messages). */
	.row.inbound {
		justify-content: flex-end;
	}
	.bubble.agent {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
		border-color: color-mix(in srgb, var(--accent) 40%, transparent);
	}
	.bubble.env {
		background: color-mix(in srgb, var(--warn) 12%, transparent);
		border-color: color-mix(in srgb, var(--warn) 40%, transparent);
	}
	/* Sub-agent terminal result: same base as an agent bubble, tinted by verdict
	   (concluded keeps the agent/accent tint; crashed/cancelled shift to error). */
	.bubble.child-result.crashed,
	.bubble.child-result.cancelled {
		background: color-mix(in srgb, var(--err) 12%, transparent);
		border-color: color-mix(in srgb, var(--err) 40%, transparent);
	}
	/* Verdict badge in the child-result header. */
	.verdict {
		display: inline-flex;
		align-items: center;
		gap: 0.2rem;
		margin-left: 0.35rem;
		padding: 0.02rem 0.35rem;
		border-radius: var(--radius-sm);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}
	.verdict-concluded {
		color: var(--ok);
		background: color-mix(in srgb, var(--ok) 15%, transparent);
	}
	.verdict-crashed,
	.verdict-cancelled {
		color: var(--err);
		background: color-mix(in srgb, var(--err) 15%, transparent);
	}
	.ihead {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		margin-bottom: 0.25rem;
	}
	/* Intra-turn cluster timestamp. `.empty` reserves the same height while the
	   tail is awaiting output, so the timestamp appears without a layout shift. */
	.sep {
		font-size: var(--fs-xs);
		/* Align the timestamp under the chatbot text (matches .asst-inline padding). */
		margin: 0.15rem 0 0.35rem 0.2rem;
		min-height: 1.1em;
	}
	.tools {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		margin-top: 0.4rem;
	}
	.pill {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
		font-size: var(--fs-sm);
		padding: 0.12rem 0.45rem;
		border-radius: var(--radius-pill);
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--text-dim);
	}
	.pill.running {
		color: var(--accent);
		border-color: color-mix(in srgb, var(--accent) 45%, transparent);
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
	.pill.done {
		color: var(--ok);
		border-color: color-mix(in srgb, var(--ok) 45%, transparent);
		background: color-mix(in srgb, var(--ok) 12%, transparent);
	}
	/* Errored call overrides the done-green with red. Declared after .pill.done so
	   it wins (the button carries both classes: `pill done` + class:error). */
	.pill.error {
		color: var(--err);
		border-color: color-mix(in srgb, var(--err) 45%, transparent);
		background: color-mix(in srgb, var(--err) 12%, transparent);
	}
	/* Target after the verb — dimmer, monospace, and truncated with ellipsis so
	   long paths/queries/regexes don't blow out the pill. */
	.pill-detail {
		opacity: 0.72;
		font-family: var(--mono);
		max-width: 24ch;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.cursor {
		display: inline-block;
		width: 7px;
		height: 1em;
		vertical-align: text-bottom;
		background: var(--chat);
		animation: blink 1s steps(2, start) infinite;
	}
	@keyframes blink {
		to {
			visibility: hidden;
		}
	}
	.phase {
		border: 1px dashed var(--border);
		border-radius: var(--radius-md);
		padding: 0.5rem 0.75rem;
		background: var(--bg);
	}
	.phase-head {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
		margin-bottom: 0.2rem;
	}
	.phase-title {
		font-weight: 600;
	}
	/* Context-compaction marker: a full-width dashed card mirroring the phase
	   cards in the trajectory view. A native <details> folds it; collapsed it's
	   just the header, click to reveal the summary. It reads as meta (a structural
	   marker) rather than an assistant turn. */
	.compaction {
		border: 1px dashed var(--border);
		border-radius: var(--radius-md);
		background: var(--bg);
		margin: 0.4rem 0;
	}
	.compaction-head {
		display: flex;
		align-items: center;
		gap: 0.45rem;
		padding: 0.4rem 0.6rem;
		cursor: pointer;
		list-style: none;
		color: var(--text-dim);
		font-size: var(--fs-sm);
	}
	.compaction-head::-webkit-details-marker {
		display: none;
	}
	.compaction-title {
		font-weight: 600;
		letter-spacing: 0.02em;
	}
	.compaction-range {
		font-size: var(--fs-xs);
	}
	.compaction-head .caret {
		display: inline-flex;
		align-items: center;
		transition: transform 0.15s ease;
	}
	details[open] > .compaction-head .caret {
		transform: rotate(90deg);
	}
	/* Body sits below the header, separated by the same dashed rule as the card
	   border so the open state reads as a titled panel. */
	.compaction-body {
		padding: 0.5rem 0.75rem;
		border-top: 1px dashed var(--border);
	}
	/* A completed tool pill is a <button>; reset the global button chrome and make
	   it read as a clickable chip (the .pill rule above supplies the visuals). */
	button.pill {
		/* Inherit the family only (buttons default to the system font); the .pill
		   rule's font-size: var(--fs-sm) must survive, so don't use the `font`
		   shorthand here — it would reset font-size back to the larger body value. */
		font-family: inherit;
		cursor: pointer;
	}
	/* Hover affordance: brighten the border only. We deliberately do NOT override
	   color here — doing so flattened the semantic done-green / error-red text into
	   a neutral gray on hover (an accidental regression when pills became buttons). */
	button.pill:hover {
		border-color: var(--accent-dim);
	}
	/* Tool-call detail modal (native <dialog>). Padding lives on the inner wrapper
	   (not the dialog) so a click landing on the dialog element itself means the
	   backdrop — the onclick handler uses that to dismiss. */
	.tool-modal {
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 0;
		max-width: 56rem;
		width: calc(100vw - 3rem);
		max-height: 80vh;
		overflow-y: auto;
		box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
		color: var(--text);
	}
	.tool-modal-inner {
		padding: 0.9rem 1rem;
	}
	.tool-modal::backdrop {
		background: rgba(0, 0, 0, 0.5);
	}
	.modal-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.6rem;
	}
	.modal-x {
		background: none;
		border: none;
		color: var(--text-dim);
		font-size: var(--fs-xl);
		line-height: 1;
		cursor: pointer;
		padding: 0 0.3rem;
	}
	.modal-x:hover {
		color: var(--text);
	}
</style>

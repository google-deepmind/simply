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
	// The chat TRANSCRIPT: a projected message list rendered as bubbles, plus the
	// tool-call detail modal its pills open. Extracted from the chat page so the
	// read-only session-log viewer renders history with the exact same markup
	// instead of a look-alike.
	//
	// It owns only what is derived from the messages (turn clustering, the
	// preview's chronological slot, tool-detail fetching). Everything around the
	// transcript — the scroll container and its pinning, phase cards, the status
	// bar, the composer — stays with the caller, as does ownership of `preview`
	// and `optimistic` (the live page produces them; the log viewer passes none).
	import { tick } from 'svelte';
	import { api } from '$lib/api';
	import type { ChatBubble, ChatToolCall, AgentEvent } from '$lib/types';
	import CopyButton from './CopyButton.svelte';
	import ChatNavigator from './ChatNavigator.svelte';
	import ToolCall from './ToolCall.svelte';
	import Thoughts from './Thoughts.svelte';
	import { renderMarkdown } from '$lib/markdown';
	import {
		CircleNotchIcon,
		TerminalIcon,
		BrainIcon,
		AndroidLogoIcon,
		CaretRightIcon,
		CheckCircleIcon,
		WarningCircleIcon,
		ProhibitIcon
	} from 'phosphor-svelte';
	import { iconForName } from '$lib/sessionIcon';
	import { toolIcon } from '$lib/toolIcon';

	let {
		runId,
		sid,
		messages,
		scrollEl = null,
		preview = null,
		optimistic = [],
		working = false
	}: {
		runId: string;
		// The session these messages belong to — the tool-detail modal fetches the
		// call's step from it.
		sid: string;
		messages: ChatBubble[];
		// The scrollport this transcript lives in. Used ONLY by the segment
		// navigator, to follow scroll position and size itself; its placement is
		// structural (a sticky anchor in this column's flow).
		scrollEl?: HTMLElement | null;
		// Live streaming turn (chat page only); null in the read-only viewer.
		preview?: { text: string; thoughts: string; step: number } | null;
		// Operator messages echoed before the server round-trip (chat page only).
		optimistic?: { key: string; content: string; at: number }[];
		// A turn is in flight: keeps the in-flight gap timestamp ticking.
		working?: boolean;
	} = $props();

	let now = $state(Date.now()); // ticks in-flight so pending gap timestamps update live

	// Tick `now` once a second while a turn is in flight (a tool may be running),
	// so the pending gap timestamp appears as the wait gets long. A settled
	// transcript (the log viewer) never starts the interval.
	$effect(() => {
		if (!working && !preview) return;
		const t = setInterval(() => (now = Date.now()), 1000);
		return () => clearInterval(t);
	});

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

	// --- Tool-call detail popup ---
	// The chat feed deliberately omits tool-result bodies (they're large), so on
	// demand we fetch the call's step via the existing /events?step=N endpoint
	// (the step is already known — no DB-wide id search), find the call + its
	// matching tool_result by id, and render the pair in a modal via <ToolCall>.
	// ALWAYS a modal: a tool call is tied to one specific past event, not
	// standing reference content.
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
			const events = await api.getEvents(runId, sid, step);
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

</script>

			<!-- Chatbot output, rendered inline (no bubble) so its width is stable;
			     shared by the committed message and the live streaming preview. -->
			{#snippet chatbotInline(
				content: string,
				thoughts: string,
				toolCalls: ChatToolCall[],
				streaming: boolean,
				step: number,
				eid: string = ''
			)}
				<!-- data-eid is the navigator's scroll target: it resolves a segment to a
				     DOM node without the navigator knowing anything about this markup. -->
				<div class="asst-inline" data-eid={eid}>
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

			<ChatNavigator {messages} {scrollEl} {working} />

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
					<div class="row user" data-eid={m.event_id}>
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
					<div class="row inbound" data-eid={m.event_id}>
						<div class="bubble env">
							<span class="bubble-copy"><CopyButton text={m.content} /></span>
							<div class="ihead dim small"><TerminalIcon size={13} />{m.from || 'environment'}</div>
							<pre class="pre">{m.content}</pre>
							<div class="time dim">{formatTime(m.created_at)}</div>
						</div>
					</div>
				{:else if m.kind === 'agent'}
					{@const FromIcon = iconForName(m.from) ?? AndroidLogoIcon}
					<div class="row inbound" data-eid={m.event_id}>
						<div class="bubble agent">
							<span class="bubble-copy"><CopyButton text={m.content} /></span>
							<div class="ihead dim small">
								<FromIcon size={13} />
								{#if m.from}
									<a href={`/runs/${runId}/sessions/${m.from}`}>{m.from}</a>
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
					<div class="row inbound" data-eid={m.event_id}>
						<div class="bubble agent child-result" class:crashed={m.verdict === 'crashed'} class:cancelled={m.verdict === 'cancelled'}>
							<span class="bubble-copy"><CopyButton text={m.content} /></span>
							<div class="ihead dim small">
								<FromIcon size={13} />
								{#if m.from}
									<a href={`/runs/${runId}/sessions/${m.from}`}>{m.from}</a>
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
					<details class="compaction" data-eid={m.event_id}>
						<summary class="compaction-head">
							<span class="caret"><CaretRightIcon size={12} weight="bold" /></span>
							<BrainIcon size={14} />
							<span class="compaction-title">Context compacted</span>
							<span class="compaction-range mono dim">step {m.step}</span>
						</summary>
						<div class="md dim small compaction-body">{@html renderMarkdown(m.content)}</div>
					</details>
				{:else}
					{@render chatbotInline(m.content, m.thoughts ?? '', m.tool_calls, false, m.step, m.event_id)}
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
					{runId}
				/>
			{/if}
		</div>
	{/if}
</dialog>
<style>
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
	/* Copy button sits FLUSH at the bubble's outer-LEFT edge (right:100%, top:0) —
	   beside the bubble, NOT out in the far gutter. Bubbles are right-flushed and
	   vary in width, so this x varies with the bubble; that is the accepted trade
	   for keeping the right gutter clear for the segment navigator, and it keeps
	   the button adjacent to the content it copies. */
	.bubble-copy {
		position: absolute;
		top: 0;
		right: 100%;
		padding-right: 0.35rem;
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
	/* Copy sits FLUSH at the inline's outer-LEFT edge (right:100%, top:0), in the
	   left gutter: the RIGHT gutter now belongs to the segment navigator, and a
	   hover-revealed button there would collide with the rail.
	   DOM child + flush = no hover gap when the cursor moves onto it. */
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

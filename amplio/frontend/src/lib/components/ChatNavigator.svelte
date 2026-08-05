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
	// Segment navigator: one dot per SEGMENT, in the chat's left gutter.
	//
	// A segment is an operator message and everything up to the next one — the
	// unit an operator actually navigates by ("where did I ask about X, and what
	// came back"). Everything inside it (tool turns, the closing summary, a child
	// result) is detail, surfaced in the hover card rather than on the rail.
	//
	// Three rules keep it from accreting:
	//   1. One element (a dot), identical for every segment. Kind, counts and
	//      previews live in the card, never on the rail.
	//   2. Position encodes ORDER, not document offset — this is a navigator, not
	//      a scrollbar. Dots stack UPWARD from the bottom, so the newest segment
	//      keeps the same position as the conversation grows.
	//   3. Exactly one dot is selected at a time, and selection is binary. A
	//      gradient (the fisheye pattern) is what makes it ambiguous which one
	//      you are about to get.
	import type { ChatBubble } from '$lib/types';
	import { SparkleIcon, GraphIcon, ChatCircleIcon, BellIcon } from 'phosphor-svelte';

	let {
		messages,
		scrollEl,
		working = false
	}: {
		messages: ChatBubble[];
		scrollEl?: HTMLElement | null;
		// True while a turn is in flight. The ONLY thing that distinguishes "the
		// agent hasn't finished" from "the operator cut in before it did" — the
		// two cases a missing summary otherwise conflates.
		working?: boolean;
	} = $props();

	type Row = {
		kind: 'summary' | 'child_result' | 'agent' | 'environment';
		eid: string;
		text: string;
		who?: string;
		verdict?: string;
		n?: number; // environment rows aggregate
		big?: boolean; // the first closing summary gets the larger quota
	};
	type Segment = {
		eid: string;
		text: string;
		at: string;
		steps: number;
		tools: number;
		rows: Row[];
		// Last assistant turn that carried text but also made tool calls; only used
		// when the segment never produced a closing summary.
		lastSaid?: { eid: string; text: string };
	};

	// Beyond this the row list is truncated; measured segments almost never reach
	// it (1 row is the median, >4 happens about once per session).
	const ROW_CAP = 4;
	// Total preview lines in the card body, redistributed to the two previews so
	// the common one-row card is not needlessly clipped (see quotaFor).
	const BUDGET = 9;
	const TITLE_MIN = 2, TITLE_MAX = 5, SUM_MIN = 2, SUM_MAX = 6;

	const segments = $derived.by<Segment[]>(() => {
		const out: Segment[] = [];
		let cur: Segment | null = null;
		let steps = new Set<number>();
		let sawSummary = false;
		for (const m of messages) {
			if (m.kind === 'operator') {
				cur = {
					eid: m.event_id,
					text: m.content ?? '',
					at: m.created_at,
					steps: 0,
					tools: 0,
					rows: [],
					lastSaid: undefined
				};
				steps = new Set();
				sawSummary = false;
				out.push(cur);
				continue;
			}
			if (!cur) continue; // anything before the first operator message
			if (m.kind === 'chatbot') {
				steps.add(m.step);
				cur.tools += m.tool_calls?.length ?? 0;
				if (!m.tool_calls?.length && (m.content ?? '').trim()) {
					cur.rows.push({ kind: 'summary', eid: m.event_id, text: m.content, big: !sawSummary });
					sawSummary = true;
				} else if ((m.content ?? '').trim()) {
					// Spoke while still working. Remembered so an interrupted segment can
					// show the last thing the agent actually said instead of nothing.
					cur.lastSaid = { eid: m.event_id, text: m.content };
				}
			} else if (m.kind === 'environment') {
				// Notifications aggregate: they reach 26 in a single live window, and
				// listing them individually buries everything that matters.
				const last = cur.rows.find((r) => r.kind === 'environment');
				if (last) last.n = (last.n ?? 1) + 1;
				else cur.rows.push({ kind: 'environment', eid: m.event_id, text: '', n: 1 });
			} else if (m.kind === 'child_result' || m.kind === 'agent') {
				cur.rows.push({
					kind: m.kind,
					eid: m.event_id,
					text: m.content ?? '',
					who: m.from,
					verdict: m.verdict
				});
			}
			cur.steps = steps.size;
		}
		return out;
	});

	let hovered = $state<number | null>(null);
	// Centres of the dots, cached on entry: hovering must select the NEAREST dot
	// from anywhere in the pill, or a 5px mark at a 7px pitch is a coin toss. Read
	// once per entry rather than per mousemove, so it costs one layout, not sixty.
	let dotEls: HTMLElement[] = [];
	let centres: number[] = [];
	let hereIdx = $state(0);
	let cardY = $state(0);
	let railEl = $state<HTMLElement>();
	let hideTimer: ReturnType<typeof setTimeout> | null = null;

	const active = $derived(hovered ?? null);
	const shown = $derived(active === null ? null : segments[active]);
	const rows = $derived(shown ? shown.rows.slice(0, ROW_CAP) : []);
	const extra = $derived(shown ? shown.rows.length - rows.length : 0);
	const quota = $derived(quotaFor(rows));
	// "Still working" is only ever true of the LAST segment, and only while a turn
	// is actually in flight. Every other summary-less segment is finished work.
	const inFlight = $derived(working && active !== null && active === segments.length - 1);

	// Most segments hold exactly one anchor, so a fixed 2+2 split leaves the card
	// half empty. Spend the whole budget: extra rows take lines back from the two
	// previews, and with no extra rows the previews get everything.
	function quotaFor(rs: Row[]) {
		const extras = rs.filter((r) => !r.big).length;
		const hasSum = rs.some((r) => r.big);
		let spare = BUDGET - extras - TITLE_MIN - (hasSum ? SUM_MIN : 0);
		let title = TITLE_MIN;
		let sum = hasSum ? SUM_MIN : 0;
		while (spare > 0) {
			let moved = false;
			if (hasSum && sum < SUM_MAX) { sum++; spare--; moved = true; }
			if (title < TITLE_MAX && spare > 0) { title++; spare--; moved = true; }
			if (!moved) break;
		}
		return { title, sum };
	}

	function nodeFor(eid: string): HTMLElement | null {
		return scrollEl?.querySelector(`[data-eid="${CSS.escape(eid)}"]`) ?? null;
	}

	// POSITIONING IS STRUCTURAL, NOT MEASURED. The rail renders inside the message
	// column and is `position: sticky`, so it inherits the column's horizontal
	// position for free and floats at the middle of the scrollport vertically.
	//
	// The previous attempt measured the column with getBoundingClientRect plus a
	// ResizeObserver, which cannot work: opening the artifact panel MOVES the 54rem
	// column without resizing it, and nothing fires an event for a move.
	const RAIL_PAD = 24;
	const MIN_PITCH = 7,
		MAX_PITCH = 17;
	let viewportH = $state(0);

	// Fit inside the scrollport instead of scrolling: a navigator you have to
	// navigate is a failed navigator.
	const pitch = $derived(
		segments.length
			? Math.max(MIN_PITCH, Math.min(MAX_PITCH, (viewportH - RAIL_PAD) / segments.length))
			: MAX_PITCH
	);

	function jump(eid: string) {
		nodeFor(eid)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		close();
	}

	// The card is centred on the pointer but kept inside the viewport: near the
	// top or bottom of a tall rail it would otherwise hang off-screen.
	function clampY(y: number) {
		const half = 150;
		return Math.min(Math.max(y, half), Math.max(half, window.innerHeight - half));
	}
	function cacheCentres() {
		centres = dotEls.map((el) => {
			const r = el?.getBoundingClientRect();
			return r ? r.top + r.height / 2 : 0;
		});
	}
	function nearest(clientY: number) {
		let best = 0;
		let bd = Infinity;
		for (let i = 0; i < centres.length; i++) {
			const d = Math.abs(centres[i] - clientY);
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		return best;
	}
	function onRailMove(ev: MouseEvent) {
		if (!centres.length) cacheCentres();
		open(nearest(ev.clientY), ev);
	}

	function open(i: number, ev: MouseEvent) {
		if (hideTimer) clearTimeout(hideTimer);
		hovered = i;
		cardY = clampY(ev.clientY);
	}
	// Leaving the rail toward the CARD needs a grace period, or the card (and its
	// buttons) would vanish mid-travel. Leaving in any other direction does not —
	// and paying the delay there just leaves a dot lit after the pointer is gone.
	// So the corridor is directional: only exits on the card's side wait.
	function scheduleClose(ev?: MouseEvent) {
		if (hideTimer) clearTimeout(hideTimer);
		const railRight = railEl?.getBoundingClientRect().right ?? 0;
		const towardCard = !!ev && ev.clientX >= railRight - 1;
		if (!towardCard) {
			hovered = null;
			return;
		}
		hideTimer = setTimeout(() => (hovered = null), 220);
	}
	function close() {
		if (hideTimer) clearTimeout(hideTimer);
		hovered = null;
	}

	// "Where you are" = the bottom-most segment whose user message is FULLY
	// visible; if none is (you are deep inside one segment), the last one that
	// starts above the fold — i.e. the segment you are reading.
	//
	// Deliberately rect-based. The obvious alternative, comparing cached content
	// offsets against scrollTop, needs a cache and an invalidation rule (markdown,
	// images and streaming all change heights after first paint) — and cached
	// offsets are what made an earlier version depend on offsetParent, which the
	// host's positioning could silently change. Rects are viewport-relative and
	// therefore never stale.
	function syncHere() {
		const sc = scrollEl;
		if (!sc || !segments.length) return;
		const box = sc.getBoundingClientRect();
		// One query per scroll rather than one per segment; the map is local, so
		// there is nothing to keep in sync.
		const byEid = new Map<string, Element>();
		for (const n of sc.querySelectorAll('[data-eid]')) {
			const id = n.getAttribute('data-eid');
			if (id) byEid.set(id, n);
		}
		let above = 0;
		let visible = -1;
		segments.forEach((s, i) => {
			const n = byEid.get(s.eid);
			if (!n) return;
			const r = n.getBoundingClientRect();
			if (r.top >= box.top && r.bottom <= box.bottom) visible = i;
			else if (r.top < box.top) above = i;
		});
		hereIdx = visible >= 0 ? visible : above;
	}

	$effect(() => {
		const el = scrollEl;
		if (!el) return;
		syncHere();
		// Height only — it decides how many dots fit, nothing about placement.
		viewportH = el.clientHeight;
		el.addEventListener('scroll', syncHere, { passive: true });
		const ro = new ResizeObserver(() => {
			viewportH = el.clientHeight;
			centres = [];
		});
		ro.observe(el);
		return () => {
			el.removeEventListener('scroll', syncHere);
			ro.disconnect();
		};
	});

	// A new turn (or a switch between views) moves every dot.
	$effect(() => {
		void messages.length;
		centres = [];
		queueMicrotask(syncHere);
	});

	function firstLine(s: string, max = 480) {
		return (s ?? '').replace(/\s+/g, ' ').trim().slice(0, max);
	}
	function timeLabel(iso: string) {
		const d = new Date(iso);
		if (Number.isNaN(+d)) return '';
		const p = (n: number) => String(n).padStart(2, '0');
		return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
	}
</script>

{#if segments.length > 1}
	<!-- Zero-size sticky anchor: it sits at the column's left edge IN THE FLOW, so
	     the rail follows the column wherever the layout puts it, and sticks to the
	     middle of the scrollport as the conversation scrolls past. -->
	<div class="anchor">
		<div
			class="rail"
			bind:this={railEl}
			style:--pitch="{pitch}px"
			onmouseenter={cacheCentres}
			onmousemove={onRailMove}
			onmouseleave={(e) => scheduleClose(e)}
			role="presentation"
		>
			<!-- No `title`: the native tooltip would render on top of our own card. -->
			{#each segments as s, i (s.eid)}
				<button
					class="dot"
					class:here={i === hereIdx && active !== i}
					class:sel={active === i}
					bind:this={dotEls[i]}
					aria-label="Segment {i + 1}: {firstLine(s.text, 60)}"
					onfocus={(e) => open(i, e as unknown as MouseEvent)}
					onclick={() => jump(s.eid)}
				></button>
			{/each}
		</div>
	</div>
{/if}

{#if shown}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="card"
		style:top="{cardY}px"
		style:left="{(railEl?.getBoundingClientRect().right ?? 0) + 12}px"
		onmouseenter={() => hideTimer && clearTimeout(hideTimer)}
		onmouseleave={close}
	>
		<div class="hdr">
			<span>{timeLabel(shown.at)}</span>
			<span>
				{#if shown.steps}{shown.steps} step{shown.steps > 1 ? 's' : ''}{/if}
				{#if shown.steps && shown.tools} · {/if}
				{#if shown.tools}{shown.tools} tool{shown.tools > 1 ? 's' : ''}{/if}
			</span>
		</div>
		<button class="q" style:-webkit-line-clamp={quota.title} onclick={() => jump(shown.eid)}>
			{firstLine(shown.text)}
		</button>
		<div class="rows">
			{#each rows as r (r.eid)}
				<button class="row {r.kind}" class:big={r.big} onclick={() => jump(r.eid)}>
					<span class="ic">
						{#if r.kind === 'summary'}<SparkleIcon size={12} />
						{:else if r.kind === 'child_result'}<GraphIcon size={12} />
						{:else if r.kind === 'agent'}<ChatCircleIcon size={12} />
						{:else}<BellIcon size={12} />{/if}
					</span>
					<span class="tx" style:-webkit-line-clamp={r.big ? quota.sum : 1}>
						{#if r.kind === 'environment'}
							{r.n} notification{(r.n ?? 1) > 1 ? 's' : ''}
						{:else}
							<!-- No separator glyph: the name is already set apart by colour and
							     weight. Spacing comes from the span's margin, not from template
							     whitespace, which HTML collapses asymmetrically around inline
							     elements (the source of the lopsided " ·" this replaces). -->
							{#if r.who}<span class="who">{r.who}</span>{/if}{#if r.verdict && r.verdict !== 'concluded'}<span
									class="bad">{r.verdict}</span
								>{/if}{firstLine(r.text, r.big ? 600 : 240)}
						{/if}
					</span>
				</button>
			{/each}
			<!-- The fallbacks below describe an EMPTY segment, so they are mutually
			     exclusive with having rows. Nesting rather than chaining keeps that
			     true: a flat chain let `lastSaid` render alongside a real summary,
			     which showed the same exchange twice, newest first. -->
			{#if rows.length}
				{#if extra > 0}
					<div class="row muted"><span class="ic"></span><span class="tx">+{extra} more</span></div>
				{/if}
			{:else if inFlight}
				<div class="row muted"><span class="ic"></span><span class="tx">still working…</span></div>
			{:else if shown.lastSaid}
				<!-- Interrupted: no closing summary because the next message arrived
				     first (7-16% of segments). Show the last thing the agent said —
				     it exists, it just is not a conclusion. -->
				<button class="row" onclick={() => jump(shown.lastSaid!.eid)}>
					<span class="ic"><SparkleIcon size={12} /></span>
					<span class="tx" style:-webkit-line-clamp={quota.sum || 2}>
						{firstLine(shown.lastSaid.text, 600)}
					</span>
				</button>
			{:else}
				<div class="row muted"><span class="ic"></span><span class="tx">no reply</span></div>
			{/if}
		</div>
	</div>
{/if}

<style>
	/* Sticky + zero-size + in the column's flow: no ancestor has to be positioned,
	   and anything that moves the column (the artifact panel opening) moves the
	   rail with it, which measurement could not achieve. */
	.anchor {
		position: sticky;
		top: 50%;
		height: 0;
		/* full width so its RIGHT edge is the column's right edge — the rail hangs
		   off that, replacing the scrollbar rather than sitting opposite it */
		width: 100%;
		z-index: 4;
	}
	.rail {
		position: absolute;
		/* Sits in the host's right gutter. The offset is a variable rather than a
		   constant because the gutter belongs to the HOST (.scroll's padding-inline,
		   1.8rem today): a host with a different gutter overrides --nav-offset
		   instead of this component guessing. It must stay under that padding —
		   .scroll is overflow-y:auto, so overflow-x computes to auto too and
		   anything past the padding box is clipped, z-index or not. */
		right: calc(-1 * var(--nav-offset, 30px));
		top: 0;
		transform: translateY(-50%);
		display: flex;
		flex-direction: column; /* oldest at the top, newest at the bottom */
		align-items: center;
	}

	.dot {
		width: 16px;
		/* pitch shrinks to fit the viewport; the 5px mark stays put inside it */
		height: var(--pitch, 17px);
		display: grid;
		place-items: center;
		background: none;
		border: 0;
		padding: 0;
		cursor: pointer;
		flex: 0 0 auto;
	}
	/* The visible mark is the ::before, so the 14px button stays a comfortable hit
	   target while the dot itself is small, and a selected dot grows OUT of its
	   box without reflowing its neighbours. */
	.dot::before {
		content: '';
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--text-dim);
		opacity: 0.34;
		transition:
			width 0.1s ease,
			height 0.1s ease,
			opacity 0.1s ease;
	}
	.dot:hover::before,
	.dot:focus-visible::before {
		opacity: 0.85;
	}
	.dot.here::before {
		background: var(--text-strong);
		opacity: 0.75;
	}
	.dot.sel::before {
		width: 11px;
		height: 11px;
		background: var(--accent);
		opacity: 1;
	}

	.card {
		position: fixed;
		z-index: 40;
		width: 27rem;
		max-width: 42vw;
		transform: translateY(-50%);
		background: var(--bg-elev2);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 0.6rem 0.7rem;
		box-shadow: 0 12px 34px rgba(0, 0, 0, 0.55);
		font-size: var(--fs-sm);
	}
	.hdr {
		display: flex;
		justify-content: space-between;
		gap: 0.5rem;
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--text-dim);
		margin-bottom: 0.35rem;
	}
	/* Brightness, not weight: bold is a heading signal and stops meaning "title"
	   once it runs to five lines of prose. */
	.q {
		display: -webkit-box;
		-webkit-box-orient: vertical;
		overflow: hidden;
		width: 100%;
		text-align: left;
		background: none;
		border: 0;
		padding: 0;
		font: inherit;
		color: var(--text-strong);
		cursor: pointer;
	}
	.q:hover {
		color: var(--accent);
	}
	.rows {
		margin-top: 0.45rem;
		border-top: 1px solid var(--border);
		padding-top: 0.35rem;
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
	}
	.row {
		display: grid;
		grid-template-columns: 15px 1fr;
		gap: 0.4rem;
		align-items: start;
		width: 100%;
		background: none;
		border: 0;
		color: var(--text-dim);
		font: inherit;
		font-size: var(--fs-xs);
		text-align: left;
		padding: 0.16rem 0.2rem;
		border-radius: var(--radius-xs);
		cursor: pointer;
	}
	.row:hover {
		background: var(--bg-elev);
		color: var(--text);
	}
	.row .ic {
		display: inline-flex;
		margin-top: 2px;
		opacity: 0.8;
	}
	.row .tx {
		display: -webkit-box;
		-webkit-box-orient: vertical;
		overflow: hidden;
		word-break: break-word;
	}
	.row.big {
		color: var(--text);
	}
	.row .who {
		color: var(--text-strong);
		font-weight: 600;
		margin-right: 0.4em;
	}
	.row.child_result .who {
		color: var(--child);
	}
	.row.agent .who {
		color: var(--chat);
	}
	.row .bad {
		color: var(--err);
		font-weight: 600;
		margin-right: 0.4em;
	}
	.row.muted {
		opacity: 0.75;
		cursor: default;
	}
	.row.muted:hover {
		background: none;
	}
</style>

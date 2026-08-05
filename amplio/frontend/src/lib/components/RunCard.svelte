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
	import { api } from '$lib/api';
	import { goto } from '$app/navigation';
	import { updates } from '$lib/updates.svelte';
	import { auth } from '$lib/auth.svelte';
	import { timeAgo } from '$lib/time';
	import type { RunSummary } from '$lib/types';
	import StatusBadge from './StatusBadge.svelte';
	import {
		StarIcon,
		CpuIcon,
		ChatCircleTextIcon,
		DotsThreeVerticalIcon,
		PencilSimpleIcon,
		ArchiveIcon,
		EnvelopeIcon,
		FolderIcon,
		CubeIcon,
		XIcon,
		ArrowClockwiseIcon,
		ArrowSquareOutIcon,
		TagIcon,
		CopyIcon,
		TrashIcon
	} from 'phosphor-svelte';
	import { iconForName } from '$lib/sessionIcon';
	import { setRunPrefill } from '$lib/runPrefill.svelte';
	import { EditorIcon, WorkspaceNameModal } from './internal';
	import GradePicker from './GradePicker.svelte';

	// Unified run card. The dashboard renders a list of these (linkable=true,
	// title is a navigation <a>); the run page renders one (linkable=false,
	// title is plain selectable text). Visuals are identical; only the title's
	// interaction differs.
	//
	// Card row holds the always-visible bits: grade picker, identity icon (which
	// doubles as the star toggle), title, chips, status pill, step + last-update
	// time, ⋮ menu. The leftmost slot is the grade picker (a smiley/dashed
	// circle); the identity icon to its right toggles starred (gold when
	// starred). Star also lives in the ⋮ menu as a secondary affordance. Other
	// mutations and
	// operational actions live in the overflow menu — Rename, Archive,
	// Restart (when crashed), Cancel (when ongoing/awaiting) — keeping the
	// scanned row visually quiet and hiding destructive actions behind one
	// extra click. The Cancel item has a two-step confirm (one click arms,
	// second click commits) because it cascades to all sessions.
	//
	// After any mutation the parent is informed via `onmutated` so it can
	// refetch.

	let {
		run,
		linkable = false,
		onmutated
	}: { run: RunSummary; linkable?: boolean; onmutated?: () => void } = $props();

	const isChat = $derived(run.root_agent_type === 'chatbot');
	const hasUpdates = $derived(updates.hasUpdatesById(run.run_id));
	const AnimalIcon = $derived(iconForName(run.run_id));
	// Chat runs land on the chat tab directly; autonomous land on the overview.
	const linkPath = $derived(isChat ? `/runs/${run.run_id}/chat` : `/runs/${run.run_id}`);
	const displayTitle = $derived(
		run.title || run.task || (isChat ? '(untitled chat)' : '(no task)')
	);
	const lastChange = $derived(run.root_status_changed_at || run.created_at);

	// --- Workspace pill / "Name workspace" editor integration ---
	//
	// Three pill shapes branch off the run's workspace metadata:
	//   * cider_url present  → pill is an <a> opening the workspace in its
	//     editor, with the editor brand mark + trailing ↗
	//     external-link glyph as a click affordance cue.
	//   * nameable, unnamed → pill stays a plain non-link chip; the overflow
	//     menu surfaces a "Name workspace…" item that opens the modal.
	//   * other backends (plain/jj/external) → plain chip, no menu item.
	// Unnamed and named are the only two states where the menu
	// item appears/disappears — we never offer to rename an already-named
	// workspace from this UI (would risk fighting external mutations).
	const canOpenInEditor = $derived(!!run.cider_url);
	const canNameWorkspace = $derived(
		run.workspace_kind === 'citc' && !run.workspace_alias && run.workspace_numeric_id > 0
	);
	let nameWorkspaceOpen = $state(false);
	function openNameWorkspace() {
		closeMenu();
		nameWorkspaceOpen = true;
	}

	// --- Star ---
	async function toggleStar() {
		await api.updateRun(run.run_id, { starred: !run.starred });
		onmutated?.();
	}

	// --- New run like this ---
	// Pre-fill the dashboard composer with this run's task/title/model/mode — a
	// fast way to relaunch a variation (e.g. same task, different model). Only
	// offered on the dashboard (linkable rows), where the composer is present, so
	// this just hands the prefill to the live composer on the same page; no
	// navigation. Workspace is intentionally dropped (the copy gets a fresh
	// workspace; see runPrefill).
	function newRunLike() {
		setRunPrefill({
			task: run.task,
			title: run.title,
			llm: run.llm,
			interactive: run.root_agent_type === 'chatbot'
		});
		closeMenu();
	}

	// --- Operational actions: Cancel (two-step confirm) / Restart (one-shot) ---
	let cancelling = $state(false);
	let confirmCancel = $state(false);
	let confirmTimer: ReturnType<typeof setTimeout> | undefined;
	let restarting = $state(false);
	const canCancel = $derived(['ongoing', 'awaiting'].includes(run.root_status));
	const canRestart = $derived(run.root_status === 'crashed');

	const CONFIRM_WINDOW_MS = 4000;
	function clearConfirmTimer() {
		if (confirmTimer) {
			clearTimeout(confirmTimer);
			confirmTimer = undefined;
		}
	}
	async function cancelClicked() {
		if (!confirmCancel) {
			// First click: arm. Menu STAYS OPEN so the user can see the
			// "Confirm cancel?" label and click it again. The auto-disarm
			// timer resets the armed state if they walk away.
			confirmCancel = true;
			clearConfirmTimer();
			confirmTimer = setTimeout(() => (confirmCancel = false), CONFIRM_WINDOW_MS);
			return;
		}
		// Second click: commit. Close the menu before the await so the row
		// stops looking "menu open" while the cascade runs.
		closeMenu();
		cancelling = true;
		try {
			await api.cancelRun(run.run_id);
			onmutated?.();
		} finally {
			cancelling = false;
		}
	}
	async function restart() {
		closeMenu();
		restarting = true;
		try {
			await api.restartRun(run.run_id);
			onmutated?.();
		} finally {
			restarting = false;
		}
	}

	// --- Delete (two-step confirm, mirrors Cancel) ---
	// Permanently removes the run + all its data. Same arm-then-commit gesture as
	// Cancel so a stray click can't nuke a run; the armed state auto-disarms and
	// is cleared whenever the menu closes (see closeMenu).
	let deleting = $state(false);
	let confirmDelete = $state(false);
	let deleteTimer: ReturnType<typeof setTimeout> | undefined;
	function clearDeleteTimer() {
		if (deleteTimer) {
			clearTimeout(deleteTimer);
			deleteTimer = undefined;
		}
	}
	async function deleteClicked() {
		if (!confirmDelete) {
			confirmDelete = true;
			clearDeleteTimer();
			deleteTimer = setTimeout(() => (confirmDelete = false), CONFIRM_WINDOW_MS);
			return;
		}
		closeMenu();
		deleting = true;
		try {
			await api.deleteRun(run.run_id);
			onmutated?.();
		} finally {
			deleting = false;
		}
	}

	// --- Overflow menu ---
	//
	// closeMenu is the single sink for "menu goes away" so the armed-Cancel
	// invariant ("closing the menu always disarms") holds for every path:
	// click-outside, action items (Rename/Archive/Restart), and the trigger's
	// toggle-to-close. Without this, the user could arm Cancel, click outside,
	// re-open the menu, and find Cancel still armed — confusing.
	let menuOpen = $state(false);
	function openMenu() {
		menuOpen = true;
	}
	function closeMenu() {
		menuOpen = false;
		clearConfirmTimer();
		confirmCancel = false;
		clearDeleteTimer();
		confirmDelete = false;
	}
	function toggleMenu(e: MouseEvent) {
		e.stopPropagation();
		if (menuOpen) closeMenu();
		else openMenu();
	}
	function clickOutside(node: HTMLElement, callback: () => void) {
		const handle = (e: MouseEvent) => {
			if (!node.contains(e.target as Node)) callback();
		};
		document.addEventListener('click', handle);
		return { destroy: () => document.removeEventListener('click', handle) };
	}

	// Rename is a per-card inline edit overlay (title-row → input). Triggered
	// from the menu, not from title click.
	let editing = $state(false);
	let titleVal = $state('');
	function focusEl(n: HTMLInputElement) {
		n.focus();
		n.select();
	}
	function startEdit() {
		titleVal = run.title;
		editing = true;
		closeMenu();
	}
	function cancelEdit() {
		editing = false;
	}
	async function saveEdit() {
		if (!editing) return;
		editing = false;
		const t = titleVal.trim();
		if (t !== run.title) {
			await api.updateRun(run.run_id, { title: t });
			onmutated?.();
		}
	}

	// Mark as unread: put the dashboard badge back for a run the operator looked
	// at but isn't done with. On the RUN PAGE this must also leave the page — the
	// run-page layout clears a badge the moment it appears while the tab is
	// visible, so staying would undo it within a refetch. Leaving is the whole
	// mechanism, which is why it isn't optional.
	async function markUnread() {
		closeMenu();
		// LEAVE FIRST, then mark. The PATCH makes the server emit run_updated; a
		// still-mounted run page refetches, sees has_updates while visible, and
		// clears the badge again — measured, not theoretical. Unmounting the page
		// before the write removes the only thing that would undo it.
		if (!linkable) await goto('/');
		updates.markUnseen(run.run_id);
	}

	async function toggleArchive() {
		closeMenu();
		await api.updateRun(run.run_id, { archived: !run.archived });
		onmutated?.();
	}
</script>

<!--
	Inner content of the title block — shared between the linkable <a> and the
	plain <div> variants so the only difference between them is the wrapping
	element + its interaction semantics.
-->
{#snippet titleblockBody()}
	<span class="title">
		{#if hasUpdates}<span class="unread-dot" title="Unread updates"></span>{/if}{displayTitle}
	</span>
	{#if !isChat && run.task}
		<!-- Autonomous-run task snippet: muted, single line, fades into the
		     chips on the right so it never visually slams into the metadata
		     strip. Hidden for chat runs (no task) and while editing. -->
		<span class="task" title={run.task}>{run.task}</span>
	{/if}
{/snippet}

<div class="card row">
	<!--
		Leading slot: the identity/star glyph only. The grade picker used to sit
		here too, but a smiley next to the object-style identity icon read as a
		crowded, cross-design-language pair — and grade is logically an OUTCOME
		control, not an identity one. It now lives at the row's trailing end (before
		the ⋮ menu). Status and grade are deliberately kept apart: they're
		orthogonal (a graded run can be restarted back to ongoing).
	-->
	<div class="lead">
		<!--
			Identity icon doubles as the star toggle: the animal/chat/android glyph
			turns gold + bold when starred. It's a sibling of the titleblock <a>
			(not inside it) and stops propagation, so clicking it toggles the star
			rather than navigating into the run. Star is also surfaced in the ⋮ menu.
		-->
		{#if auth.authed}
			<button
				class="identity-btn"
				class:on={run.starred}
				class:animal={AnimalIcon}
				title={run.starred ? 'Unstar' : `Star (${isChat ? 'chat' : 'autonomous'} run)`}
				aria-label={run.starred ? 'Unstar' : 'Star'}
				onclick={(e) => {
					e.stopPropagation();
					toggleStar();
				}}
			>
				{#if AnimalIcon}
					<AnimalIcon size={18} weight={run.starred ? 'fill' : 'bold'} />
				{:else if isChat}
					<ChatCircleTextIcon size={18} weight={run.starred ? 'fill' : 'bold'} />
				{:else}
					<CpuIcon size={18} weight={run.starred ? 'fill' : 'bold'} />
				{/if}
			</button>
		{:else}
			<!-- Read-only identity glyph (no auth → no toggling). -->
			<span
				class="identity"
				class:animal={AnimalIcon}
				title={isChat ? 'Chat run' : 'Autonomous run'}
			>
				{#if AnimalIcon}
					<AnimalIcon size={18} weight="bold" />
				{:else if isChat}
					<ChatCircleTextIcon size={18} weight="bold" />
				{:else}
					<CpuIcon size={18} weight="bold" />
				{/if}
			</span>
		{/if}
	</div>

	<!--
		Title block: the full left region between the star and the chips is one
		clickable target when linkable (dashboard row), or one flat selectable
		block when not (run header). Identity icon + title + task snippet + any
		trailing empty space all live inside this single element, so a click
		anywhere in the gap navigates into the run (no "must hit the title text
		exactly" papercuts), and hover lights up the title color as the
		clickability signal.
	-->
	<div class="title-wrap">
		{#if editing}
			<input
				class="title-edit"
				bind:value={titleVal}
				use:focusEl
				onkeydown={(e) => {
					if (e.key === 'Enter') saveEdit();
					else if (e.key === 'Escape') cancelEdit();
				}}
				onblur={saveEdit}
			/>
		{:else if linkable}
			<a class="titleblock" href={linkPath}>
				{@render titleblockBody()}
			</a>
		{:else}
			<div class="titleblock plain">
				{@render titleblockBody()}
			</div>
		{/if}
	</div>

	{#if canOpenInEditor}
		<a
			class="chip editor"
			href={run.cider_url}
			target="_blank"
			rel="noopener"
			title={`Open ${run.workspace_alias} · ${run.workspace}`}
		>
			<EditorIcon size={12} />
			<span class="chip-text">{run.workspace_name || run.workspace}</span>
			<ArrowSquareOutIcon size={10} weight="bold" />
		</a>
	{:else if run.workspace_name || run.workspace}
		<span class="chip" title={run.workspace}>
			<FolderIcon size={12} weight="bold" />
			<span class="chip-text">{run.workspace_name || run.workspace}</span>
		</span>
	{/if}
	{#if run.llm}
		<!-- Short label in the chip, full spec in the tooltip: specs run to 120+
		     chars and the chip is a glance target, not a reference. -->
		<span class="chip" title="Model: {run.llm}">
			<CubeIcon size={12} weight="bold" />
			<span class="chip-text">{run.llm_name || run.llm}</span>
		</span>
	{/if}

	<StatusBadge status={run.root_status} />

	<span class="meta dim" title={new Date(lastChange).toLocaleString()}>
		step {run.root_step} · {timeAgo(lastChange)}
	</span>

	<!--
		Grade picker at the trailing end (just before the ⋮ menu): grade is an
		outcome control, so it lives with the other interactive controls on the
		right — but separated from the StatusBadge by the meta strip, since status
		and grade are orthogonal (a graded run can restart back to ongoing). Authed-
		only, like the ⋮ menu (read-only viewers don't grade).
	-->
	{#if auth.authed}
		<GradePicker {run} onmutated={() => onmutated?.()} />
		<div class="menu-wrap" use:clickOutside={closeMenu}>
			<button class="iconbtn" aria-label="More actions" title="More actions" onclick={toggleMenu}>
				<DotsThreeVerticalIcon size={16} weight="bold" />
			</button>
			{#if menuOpen}
				<div class="menu" role="menu">
					<button class="item" onclick={startEdit}>
						<PencilSimpleIcon size={14} />
						Rename
					</button>
					<!-- Relaunch a variation: pre-fills the dashboard composer with this
					     run's task/title/model. Dashboard-only (linkable), where the
					     composer is present to receive it. -->
					{#if linkable}
						<button
							class="item"
							onclick={newRunLike}
							title="Pre-fill the composer with this run's task, title, and model"
						>
							<CopyIcon size={14} />
							New run like this…
						</button>
					{/if}
					<!-- Star here is the secondary affordance; the identity icon is the
					     primary one. Closing the menu afterwards keeps it tidy. -->
					<button
						class="item"
						onclick={() => {
							closeMenu();
							toggleStar();
						}}
					>
						<StarIcon size={14} weight={run.starred ? 'fill' : 'regular'} />
						{run.starred ? 'Unstar' : 'Star'}
					</button>
					{#if canNameWorkspace}
						<button
							class="item"
							onclick={openNameWorkspace}
							title="Name this workspace so it can be opened in an editor"
						>
							<TagIcon size={14} weight="bold" />
							Name workspace…
						</button>
					{/if}
					<button
						class="item"
						onclick={markUnread}
						title="Put this run's badge back and return to the dashboard"
					>
						<EnvelopeIcon size={14} weight="regular" />
						Mark as unread
					</button>
					<button class="item" onclick={toggleArchive}>
						<ArchiveIcon size={14} weight={run.archived ? 'fill' : 'regular'} />
						{run.archived ? 'Unarchive' : 'Archive'}
					</button>
					<!-- Operational/destructive cluster. Delete is ALWAYS available (it's
					     the escape hatch for corrupted/dead runs), so the divider always
					     shows. -->
					<div class="divider" role="separator"></div>
					{#if canRestart}
						<button
							class="item recover"
							onclick={restart}
							disabled={restarting}
							title="Revive this run's active spine (re-runs the crashed root)"
						>
							<ArrowClockwiseIcon size={14} weight="bold" />
							{restarting ? 'Restarting…' : 'Restart run'}
						</button>
					{/if}
					{#if canCancel}
						<button
							class="item danger"
							class:armed={confirmCancel}
							onclick={cancelClicked}
							disabled={cancelling}
							title={confirmCancel
								? 'Click again to confirm cancellation'
								: 'Cancel this run (cascades to all sessions)'}
						>
							<XIcon size={14} weight="bold" />
							{cancelling ? 'Cancelling…' : confirmCancel ? 'Confirm cancel?' : 'Cancel run'}
						</button>
					{/if}
					<button
						class="item danger"
						class:armed={confirmDelete}
						onclick={deleteClicked}
						disabled={deleting}
						title={confirmDelete
							? 'Click again to permanently delete this run and all its data'
							: 'Delete this run permanently (DB rows + artifacts/blobs; mined lessons are kept)'}
					>
						<TrashIcon size={14} weight="bold" />
						{deleting ? 'Deleting…' : confirmDelete ? 'Confirm delete?' : 'Delete run'}
					</button>
				</div>
			{/if}
		</div>
	{/if}
</div>

{#if canNameWorkspace}
	<!--
		Modal lives OUTSIDE the .card flex row so its sizing doesn't influence
		the row layout. Rendered only when the action is applicable so a row
		on a backend without naming pays no DOM cost. onattached refreshes the
		parent's run list so the pill upgrades to a link immediately,
		in addition to whatever SSE workspace_alias signal arrives later.
	-->
	<WorkspaceNameModal
		bind:open={nameWorkspaceOpen}
		runId={run.run_id}
		numericId={run.workspace_numeric_id}
		onattached={() => onmutated?.()}
	/>
{/if}

<style>
	/* Local override of the global .card padding for row-density. The global
	   provides bg/border/radius; we tighten the padding for list use. */
	.card {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		padding: 0.55rem 0.9rem;
	}
	/* Leading slot: just the identity/star glyph now (the grade picker moved to
	   the trailing end). flex-shrink:0 + centered so it holds its size and the
	   .card gap spaces it from the title. */
	.lead {
		display: flex;
		align-items: center;
		flex-shrink: 0;
	}
	/* Outer flex slot in the row — hosts either the inline edit input or the
	   .titleblock click target. flex:1 + min-width:0 lets it consume whatever
	   horizontal space is left between the star button and the chip strip. */
	.title-wrap {
		flex: 1;
		min-width: 0;
		display: flex;
	}
	/* Inner click target: one element holds identity icon + title + task
	   snippet + trailing empty space. As an <a>, every pixel inside (icon, gap,
	   task fade, trailing whitespace) navigates into the run — no "must hit
	   the title text exactly" papercut. Hover lights up the title text color
	   as the clickability signal, kept subtle so a dense list isn't noisy. */
	.titleblock {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		width: 100%;
		min-width: 0;
		overflow: hidden;
		color: inherit;
		text-decoration: none;
		/* Floor so a single-short-word title still has a comfortable click area. */
		min-height: 1.5rem;
	}
	.titleblock:not(.plain) {
		cursor: pointer;
	}
	.titleblock:not(.plain):hover .title {
		color: var(--accent);
		text-decoration: none;
	}
	.titleblock.plain {
		cursor: default;
	}
	.title {
		flex-shrink: 0;
		font-weight: 500;
		color: var(--text);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.title-edit {
		width: 100%;
		font: inherit;
	}
	/* Task snippet — muted, single line, with a soft right-edge mask that
	   fades the text into transparency so it flows naturally into the chips
	   and doesn't compete with the title for visual prominence. */
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
	/* Unread dot before the title — same convention as before. */
	.unread-dot {
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--accent);
		margin-right: 0.4rem;
		vertical-align: middle;
		flex-shrink: 0;
	}
	.identity {
		flex-shrink: 0;
		color: var(--text-dim);
		display: inline-flex;
		align-items: center;
	}
	/* Identity icon as the star toggle: a bare icon button that sits where the
	   identity glyph used to live inside the titleblock. Dim by default (matches
	   the read-only .identity look), brightens on hover, and turns gold when the
	   run is starred so the star state is legible at a glance. */
	.identity-btn {
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		padding: 0.2rem;
		line-height: 1;
	}
	.identity-btn:hover {
		color: var(--text);
	}
	/* Animal-named runs read brighter (a specific run, not a generic type), same
	   convention as the read-only .identity.animal. */
	.identity-btn.animal {
		color: var(--text);
	}
	/* Starred: gold, overriding the dim/animal colors. */
	.identity-btn.on,
	.identity-btn.on.animal {
		color: var(--accent);
	}
	/* When the noun maps to an animal (e.g. "swift-fox" → Dog), the animal
	   replaces the generic agent-type icon. Brighter than the fallback
	   chat/android marker so the row reads as a specific run at a glance;
	   type info still surfaces via the wrapper's title= on hover. */
	.identity.animal {
		color: var(--text);
	}
	/* Workspace + LLM chips: compact mono pills with a leading icon. Both
	   share the same shape so the row has a uniform "metadata" strip; the
	   workspace chip carries the full path in title= for hover detail. The
	   inner .chip-text handles the truncation so the icon stays visible. */
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
		max-width: 16rem;
	}
	/* Phosphor icons render as <svg> with intrinsic width/height attributes,
	   but flex still treats them as shrinkable children — when the chip hits
	   its max-width with long text, flex would otherwise squash the icon
	   proportionally with the text, leaving a dot-sized smear instead of a
	   recognizable shape. Pin it. :global() because the <svg> is emitted by
	   a child component (phosphor-svelte) so the default scoped selector
	   wouldn't match it. */
	.chip > :global(svg) {
		flex-shrink: 0;
	}
	.chip-text {
		overflow: hidden;
		text-overflow: ellipsis;
		min-width: 0;
	}
	/* Editor variant: a clickable workspace pill (<a>) that opens the workspace
	   in its editor. The brand mark already carries the amber identity,
	   so the chip chrome stays neutral (matches other chips); hover lights up
	   the border + text to telegraph clickability. The trailing ↗ glyph is the
	   click affordance cue — without it, the chip looks identical to the
	   plain workspace chip until hover. Suppress the global a:hover underline
	   since this is a metadata pill, not prose. */
	a.chip.editor {
		text-decoration: none;
		color: var(--text-dim);
		cursor: pointer;
	}
	a.chip.editor:hover {
		border-color: var(--accent-dim);
		color: var(--text);
		text-decoration: none;
	}
	.meta {
		flex-shrink: 0;
		font-size: var(--fs-sm);
		font-family: var(--mono);
		white-space: nowrap;
	}
	.iconbtn {
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		padding: 0.2rem;
		line-height: 1;
	}
	.iconbtn:hover {
		color: var(--text);
	}
	/* Overflow menu — small popover anchored to the ⋮ trigger. clickOutside
	   closes when the user clicks anywhere else. */
	.menu-wrap {
		position: relative;
		flex-shrink: 0;
	}
	.menu {
		position: absolute;
		right: 0;
		top: calc(100% + 0.3rem);
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.25rem;
		z-index: 20;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
		display: flex;
		flex-direction: column;
		min-width: 11rem;
	}
	.menu .item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.4rem 0.6rem;
		background: none;
		border: none;
		color: var(--text);
		text-align: left;
		cursor: pointer;
		border-radius: calc(var(--radius) - 2px);
		font: inherit;
	}
	.menu .item:hover:not(:disabled) {
		background: var(--bg-elev2);
	}
	.menu .item:disabled {
		opacity: 0.6;
		cursor: default;
	}
	/* Horizontal rule separating housekeeping (Rename/Archive) from the
	   operational/destructive cluster (Restart/Cancel) below. Same border
	   color as the menu's own border for a quiet, consistent treatment. */
	.menu .divider {
		height: 1px;
		background: var(--border);
		margin: 0.25rem 0.1rem;
	}
	/* Recovery-positive (Restart) and destructive (Cancel) items, both with a
	   tinted text color to telegraph the kind of action before reading the
	   label. Hover gets a matching tinted background. */
	.menu .item.recover {
		color: var(--ok);
	}
	.menu .item.recover:hover:not(:disabled) {
		background: color-mix(in srgb, var(--ok) 14%, transparent);
	}
	.menu .item.danger {
		color: var(--err);
	}
	.menu .item.danger:hover:not(:disabled) {
		background: color-mix(in srgb, var(--err) 14%, transparent);
	}
	/* Armed Cancel: full red fill so the second click is visually obvious as a
	   destructive commit, not a casual hover. */
	.menu .item.danger.armed {
		background: color-mix(in srgb, var(--err) 22%, transparent);
		color: var(--err);
	}
</style>

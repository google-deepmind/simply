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
	// Reusable artifact browser: a directory tree + inline file preview, confined
	// to a run's artifact dir (served via /artifacts/raw with an os.Root sandbox +
	// CSP). Used full-page by the Artifacts route and compact in the chat aside.
	// `initialFile` deep-selects a file on open (the chat pill / tool "view"
	// path); `compact` tightens spacing for the narrow panel; `onExpand`, when
	// set, renders an "Expand" affordance (the panel → full-page bridge).
	import { untrack } from 'svelte';
	import { browser } from '$app/environment';
	import { api, artifactRawUrl, errorText } from '$lib/api';
	import { renderMarkdown, highlightFile } from '$lib/markdown';
	import { fuzzyFilter } from '$lib/fuzzy';
	import type { ArtifactEntry, ArtifactFile } from '$lib/types';
	import {
		FolderIcon,
		FileIcon,
		ArrowClockwiseIcon,
		CopyIcon,
		CheckIcon,
		ArrowSquareOutIcon,
		ArrowsOutSimpleIcon,
		MagnifyingGlassIcon,
		XIcon
	} from 'phosphor-svelte';

	let {
		runId,
		initialFile = '',
		compact = false,
		// Bindable: the currently-previewed file's subpath ("" = none). Lets a host
		// (the chat aside) offer a panel-level "Expand" that knows what's open.
		selectedFile = $bindable(''),
		// When set, an Expand control appears in the toolbar (compact/side-panel use);
		// called with the currently-selected file ("" if none) to bridge to full page.
		onExpand,
		// Base URL of the full Artifacts page (e.g. /runs/<id>/artifacts). When set,
		// the Expand control renders as a real <a href> (so it's middle-clickable /
		// cmd-clickable to open in a new tab); a plain left-click is intercepted and
		// routed through onExpand for in-app SPA navigation.
		expandBase = '',
		// Optional extra control appended to the toolbar row (the chat aside uses this
		// to embed its Artifacts close-toggle directly in the breadcrumb row, so the
		// same button occupies the same spot whether it's showing "Status" or the
		// browser).
		toolbarEnd,
		// Called whenever the previewed file changes ("" = selection cleared), with
		// HOW it changed. The full-page route uses this to keep ?file= in sync so the
		// Back button works; `via` lets it choose push vs replace, which is the
		// difference between usable history and 30 entries of arrow-key scanning:
		//   click    a file row or search result   → deliberate, push
		//   link     a relative link in a preview  → deliberate, push
		//   keyboard arrow-stepping the list       → scanning, replace
		//   browse   changed directory, no file    → not a document, replace
		//   restore  driven BY the URL / the host  → already reflected, replace
		// The component itself stays URL-agnostic — the chat side panel passes no
		// handler and must never touch the address bar.
		onSelect
	}: {
		runId: string;
		initialFile?: string;
		compact?: boolean;
		selectedFile?: string;
		onExpand?: (file: string) => void;
		expandBase?: string;
		toolbarEnd?: import('svelte').Snippet;
		onSelect?: (file: string, via: 'click' | 'keyboard' | 'link' | 'browse' | 'restore') => void;
	} = $props();

	// How a selection change came about; drives the host's history policy.
	type NavVia = 'click' | 'keyboard' | 'link' | 'browse' | 'restore';

	let path = $state(''); // subpath under the artifact root ("" = root)
	let root = $state(''); // absolute artifact dir (for copy-path)
	let entries = $state<ArtifactEntry[]>([]);
	let error = $state('');
	let loading = $state(false);

	// `selected` mirrors the bindable `selectedFile` (kept as a plain local for the
	// existing race-guard comparisons; synced to the prop below).
	let selected = $state('');
	$effect(() => {
		selectedFile = selected;
	});

	let previewKind = $state<'none' | 'image' | 'text' | 'markdown' | 'binary' | 'missing'>('none');
	let previewText = $state('');
	let copied = $state('');

	// --- filename search -----------------------------------------------------
	// When `query` is non-empty the list shows fuzzy matches over ALL files in the
	// run's artifact tree (recursive), instead of the current directory listing.
	let query = $state('');
	let allFiles = $state<ArtifactFile[]>([]);
	let allFilesLoaded = $state(false); // cache is populated
	let allFilesLoading = $state(false); // a refresh is in flight
	let searchEl = $state<HTMLInputElement>();
	let listEl = $state<HTMLElement>(); // the file-list container (for scroll-into-view)

	// Refresh the recursive filename cache. Cheap (one call); invoked lazily on
	// first search-box focus and eagerly by the toolbar refresh button.
	async function loadAllFiles() {
		if (allFilesLoading) return;
		allFilesLoading = true;
		try {
			const res = await api.listArtifactsAll(runId);
			allFiles = res.files;
			allFilesLoaded = true;
		} catch {
			// Non-fatal: search just stays empty; the dir listing still works.
		} finally {
			allFilesLoading = false;
		}
	}

	function onSearchFocus() {
		// Refresh the cache on focus so a just-written file is findable. Skip if a
		// load is already in flight; the first focus always triggers one.
		void loadAllFiles();
	}

	function clearSearch() {
		query = '';
		searchEl?.focus();
	}

	// Ranked matches for the current query (empty when not searching).
	const searchResults = $derived(
		query.trim() === '' ? [] : fuzzyFilter(allFiles, query, (f) => f.path).map((m) => m.item)
	);
	const searching = $derived(query.trim() !== '');
	const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'ico'];
	const MD_EXT = ['md', 'markdown'];
	const TEXT_PREVIEW_MAX = 256 * 1024;

	function ext(name: string): string {
		const i = name.lastIndexOf('.');
		return i < 0 ? '' : name.slice(i + 1).toLowerCase();
	}
	function join(p: string, name: string): string {
		return p ? `${p}/${name}` : name;
	}
	function fmtSize(n: number): string {
		if (n < 1024) return `${n} B`;
		if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
		return `${(n / (1024 * 1024)).toFixed(1)} MB`;
	}
	function fmtTime(s: string): string {
		const d = new Date(s);
		return isNaN(d.getTime()) ? s : d.toLocaleString();
	}

	const crumbs = $derived(path ? path.split('/').filter(Boolean) : []);
	// reloadTick cache-busts the raw URL on an explicit preview refresh (so an
	// updated file / image re-fetches instead of serving the browser cache).
	let reloadTick = $state(0);
	const rawUrl = $derived(
		selected
			? artifactRawUrl(runId, selected) + (reloadTick ? `&_=${reloadTick}` : '')
			: ''
	);

	async function load() {
		loading = true;
		error = '';
		try {
			const listing = await api.listArtifacts(runId, path);
			entries = listing.entries;
			root = listing.root;
		} catch (e) {
			error = errorText(e);
			entries = [];
		} finally {
			loading = false;
		}
	}

	// Toolbar refresh: reload the current dir listing AND (if it's been populated)
	// the recursive filename cache, so search results reflect newly-written files.
	function refresh() {
		void load();
		if (allFilesLoaded) void loadAllFiles();
	}

	// Preview a file by its FULL subpath (dir-independent, so deep-linking works).
	// `via` is passed straight through to onSelect (see the prop docs).
	async function previewPath(full: string, size: number | null, via: NavVia = 'click') {
		selected = full;
		// Treat every selection as "the initial file is now this", so a host echoing
		// it back through the initialFile prop (the route writing ?file= after our
		// own onSelect) is recognised as already-applied and doesn't re-fetch.
		lastInitial = full;
		onSelect?.(full, via);
		previewText = '';
		copied = '';
		const name = full.slice(full.lastIndexOf('/') + 1);
		if (IMAGE_EXT.includes(ext(name))) {
			previewKind = 'image';
			return;
		}
		if (size != null && size > TEXT_PREVIEW_MAX) {
			previewKind = 'binary';
			return;
		}
		try {
			// no-store so an explicit refresh re-fetches rather than serving cache.
			const res = await fetch(artifactRawUrl(runId, full), { cache: 'no-store' });
			const text = await res.text();
			if (selected !== full) return; // a newer selection won the race
			// A 404/500 body is the API's JSON error, not file content — rendering it
			// as the "file" is worse than saying nothing. Reachable now that markdown
			// links can point at a stale path (agents rename and delete files).
			if (!res.ok) {
				previewKind = 'missing';
				return;
			}
			previewText = text;
			previewKind = MD_EXT.includes(ext(name)) ? 'markdown' : 'text';
		} catch {
			if (selected !== full) return;
			previewKind = 'binary';
		}
	}

	// Reload the currently-previewed file's content (toolbar refresh). Re-fetches
	// text/markdown; for images bumps reloadTick so the cache-busted rawUrl forces
	// the <img> to re-request. Passing size=null skips the size-cap re-check
	// (already passed on first open).
	function reloadPreview() {
		if (!selected) return;
		reloadTick = Date.now();
		if (previewKind !== 'image') {
			void previewPath(selected, null);
		}
	}

	function openEntry(e: ArtifactEntry) {
		if (e.is_dir) {
			path = join(path, e.name);
			clearSelection('browse');
			return;
		}
		previewPath(join(path, e.name), e.size, 'click');
	}

	function goCrumb(i: number) {
		path = crumbs.slice(0, i + 1).join('/');
		clearSelection('browse');
	}

	// Drop the preview (browsing a directory selects no file). Reported like any
	// other selection change so the route can drop ?file= from the URL.
	function clearSelection(via: NavVia) {
		selected = '';
		lastInitial = '';
		previewKind = 'none';
		onSelect?.('', via);
	}

	// Deep-select a file by full subpath: browse to its parent dir and preview it.
	// Shared by the initialFile effect (mount-time seed), the exported openFile
	// (imperative re-open from the host, e.g. a repeat artifact-pill click) and
	// relative links inside a markdown preview.
	function deepSelect(f: string, via: NavVia = 'click') {
		if (!f) return;
		const slash = f.lastIndexOf('/');
		path = slash < 0 ? '' : f.slice(0, slash);
		void previewPath(f, null, via);
	}

	// Click a search result: deep-select the file (browse to its dir + preview).
	// Keeps the search open so the user can pick another match.
	function openSearchResult(f: string, via: NavVia = 'click') {
		deepSelect(f, via);
	}

	// The navigable FILE subpaths in the current list mode (dirs excluded), in the
	// order shown. Up/down arrow keys step through these. In search mode it's the
	// ranked results; otherwise the current directory's files.
	const navFiles = $derived(
		searching
			? searchResults.map((f) => f.path)
			: entries.filter((e) => !e.is_dir).map((e) => join(path, e.name))
	);

	// Move the selection by delta (+1 next, -1 previous) among navFiles and preview
	// it. With nothing selected yet, the first step picks the first (down) or last
	// (up) file. Clamps at the ends (no wraparound). Returns whether it moved.
	function navigate(delta: number): boolean {
		if (navFiles.length === 0) return false;
		const cur = navFiles.indexOf(selected);
		let next: number;
		if (cur < 0) {
			next = delta > 0 ? 0 : navFiles.length - 1;
		} else {
			next = Math.max(0, Math.min(navFiles.length - 1, cur + delta));
		}
		if (next === cur) return false;
		// 'keyboard': stepping the list is scanning, not navigating — the route
		// replaces the history entry instead of stacking one per file.
		openSearchResult(navFiles[next], 'keyboard'); // deep-selects + previews (both modes)
		return true;
	}

	// Keyboard nav for the browser: Up/Down step through the file list. Attached to
	// the .files container (focusable) and left active even when the search input
	// has focus, so you can type a query then arrow into the results without
	// leaving the box. A held modifier is ignored (leave OS/text shortcuts alone).
	function onKeydown(e: KeyboardEvent) {
		if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
		if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
		if (navigate(e.key === 'ArrowDown' ? 1 : -1)) {
			e.preventDefault(); // don't also scroll the list / move the text caret
			focusSelectedRow();
		}
	}

	// Move DOM focus onto the newly-selected row so the browser's focus ring and
	// our .sel highlight land on the SAME row — otherwise arrow-nav leaves a stray
	// focus outline on whatever row was last clicked, showing two "selected" rows.
	// Also scrolls it into view as nav walks past the viewport edge. rAF waits for
	// the reactive .sel class to paint before we query for the row.
	function focusSelectedRow() {
		if (!browser) return;
		// If the user is arrow-navigating FROM the search box, keep focus in the box
		// (so they can keep typing / arrowing) and don't move the DOM focus ring onto
		// a row — the .sel highlight alone marks the selection there. Only unify
		// focus+selection when focus is already inside the list.
		const keepInSearch = document.activeElement === searchEl;
		requestAnimationFrame(() => {
			const row = listEl?.querySelector<HTMLElement>('.row.sel');
			if (!keepInSearch) row?.focus({ preventScroll: true });
			row?.scrollIntoView({ block: 'nearest' });
		});
	}

	// Tracks the last file we deep-selected, so the initialFile effect below only
	// fires on a genuinely NEW prop value (not a re-open of the same file).
	let lastInitial = $state('');

	// Relative links inside a previewed markdown file (`[notes](sub/notes.md)`)
	// open IN the viewer instead of navigating away to the raw bytes. The rendered
	// HTML reaches the DOM through {@html}, so per-link Svelte handlers aren't
	// possible: renderMarkdown tags each resolved link with data-artifact-path (see
	// rewriteRelativeUrls) and ONE delegated listener on the preview body covers
	// them all.
	//
	// Modified clicks (⌘/ctrl/shift/alt, middle button) fall through to the
	// browser so "open in a new tab" keeps working — the href points at the
	// artifacts route, so that tab lands in the viewer too.
	function artifactLinks(node: HTMLElement) {
		const onClick = (e: MouseEvent) => {
			if (e.defaultPrevented || e.button !== 0) return;
			if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
			const a = (e.target as HTMLElement | null)?.closest?.('a[data-artifact-path]');
			if (!a) return;
			e.preventDefault();
			const target = a.getAttribute('data-artifact-path') ?? '';
			if (!target) {
				path = '';
				clearSelection('browse');
			} else if (target.endsWith('/')) {
				// Directory link: browse into it; there's nothing to preview.
				path = target.slice(0, -1);
				clearSelection('browse');
			} else {
				deepSelect(target, 'link'); // switches the listing to its dir + previews it
			}
		};
		node.addEventListener('click', onClick);
		return { destroy: () => node.removeEventListener('click', onClick) };
	}

	// Imperative open, callable by the host via bind:this. Unlike the initialFile
	// prop (which only reacts to a CHANGED value), this always (re)selects — so a
	// host can re-open the SAME file even when the browser has since navigated to a
	// different one (the repeat-pill-click case). Exported = part of the component
	// API when the parent binds a ref.
	export function openFile(f: string) {
		// 'restore': the host asked for this file, so it already knows — no new
		// history entry (the chat panel passes no onSelect at all).
		deepSelect(f, 'restore');
	}

	// Imperative open at the artifact ROOT (no file selected). Used by the chat
	// view's root chip (a bare $AMPLIO_ARTIFACT_DIR folder mention). lastInitial is
	// cleared so a later initialFile='' prop can't be mistaken for a new deep-select.
	export function openRoot() {
		path = '';
		clearSelection('restore');
	}

	// Drop the preview but STAY in the current directory. The full-page route calls
	// this when history moves back to a URL with no ?file= — openRoot() would also
	// yank the listing to the artifact root, which isn't what Back meant.
	export function clearPreview() {
		clearSelection('restore');
	}

	// Deep-select initialFile on mount / when it changes to a new value (a fresh
	// panel open seeded with a pill's file). Repeat opens of the same file go
	// through openFile() above, since an unchanged prop won't re-trigger this.
	//
	// lastInitial is READ and WRITTEN inside untrack: it's internal bookkeeping,
	// not a dependency. Tracking it would let openFile() (which sets lastInitial
	// to drive an imperative re-open) re-fire this effect, which would then see
	// the still-stale initialFile prop and clobber the just-opened file back to
	// the panel's original seed. The effect must depend ONLY on initialFile.
	$effect(() => {
		const f = initialFile;
		untrack(() => {
			// 'restore': this selection came FROM the URL / the host, so reporting it
			// back must not stack a history entry (the route no-ops on a match).
			if (f && f !== lastInitial) deepSelect(f, 'restore');
		});
	});

	// (Re)load the listing whenever run or dir changes.
	$effect(() => {
		void runId;
		void path;
		load();
	});

	const selectedAbs = $derived(root && selected ? `${root}/${selected}` : selected);
	// The selected file's directory (subpath under the artifact root, "" = root),
	// used as the base for resolving that file's relative image/link URLs.
	const selectedDir = $derived(
		selected.includes('/') ? selected.slice(0, selected.lastIndexOf('/')) : ''
	);
	const canCopyContent = $derived(previewKind === 'text' || previewKind === 'markdown');

	// Full-page URL for the Expand affordance, deep-linked to the current file so a
	// native middle-/cmd-click opens exactly what's showing in a new tab.
	const expandHref = $derived(
		expandBase ? expandBase + (selected ? `?file=${encodeURIComponent(selected)}` : '') : ''
	);

	// Left-click on the Expand anchor: keep it in-app (SPA nav via onExpand) unless
	// the user is asking for a new tab/window (modifier or middle-click), in which
	// case we let the browser handle the real href natively.
	function onExpandClick(e: MouseEvent) {
		if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
			return;
		}
		e.preventDefault();
		onExpand?.(selected);
	}

	async function copy(what: 'content' | 'path') {
		try {
			await navigator.clipboard.writeText(what === 'content' ? previewText : selectedAbs);
			copied = what;
			setTimeout(() => (copied = copied === what ? '' : copied), 1200);
		} catch {
			/* clipboard blocked; ignore */
		}
	}

	// --- Full-page left-right split: a draggable gutter resizes the file-list
	// column. Width is a browser-wide layout preference (localStorage), clamped to
	// a sane range. Only takes visual effect in the non-compact, wide layout (see
	// the split media query in the style block); in the chat aside / narrow window
	// the list and preview stack top-down and listW is simply ignored.
	const LIST_W_KEY = 'amplio-artifacts-list-w';
	const LIST_W_MIN = 180;
	const LIST_W_MAX = 560;
	const LIST_W_DEFAULT = 300;
	let listW = $state(LIST_W_DEFAULT);
	// Precomputed inline style carrying the list width as a CSS custom property
	// (the split media query consumes --list-w). Kept as a derived string rather
	// than inline `style="--list-w: {..}"` for parser friendliness.
	const bodyStyle = $derived(`--list-w: ${Math.round(listW)}px`);
	$effect(() => {
		if (!browser) return;
		const saved = Number(localStorage.getItem(LIST_W_KEY));
		if (saved >= LIST_W_MIN && saved <= LIST_W_MAX) listW = saved;
	});

	let dragging = $state(false);
	function startResize(e: PointerEvent) {
		e.preventDefault();
		dragging = true;
		const startX = e.clientX;
		const startW = listW;
		const onMove = (ev: PointerEvent) => {
			const w = Math.min(LIST_W_MAX, Math.max(LIST_W_MIN, startW + (ev.clientX - startX)));
			listW = w;
		};
		const onUp = () => {
			dragging = false;
			window.removeEventListener('pointermove', onMove);
			window.removeEventListener('pointerup', onUp);
			if (browser) localStorage.setItem(LIST_W_KEY, String(Math.round(listW)));
		};
		window.addEventListener('pointermove', onMove);
		window.addEventListener('pointerup', onUp);
	}
	// Keyboard resize for a11y (the gutter is a focusable separator).
	function keyResize(e: KeyboardEvent) {
		const step = e.shiftKey ? 40 : 16;
		if (e.key === 'ArrowLeft') listW = Math.max(LIST_W_MIN, listW - step);
		else if (e.key === 'ArrowRight') listW = Math.min(LIST_W_MAX, listW + step);
		else return;
		e.preventDefault();
		if (browser) localStorage.setItem(LIST_W_KEY, String(Math.round(listW)));
	}
</script>

<!-- The browser is keyboard-navigable (Up/Down step through the file list), so it
     takes focus and a keydown handler. It's a container region, not a single
     control, which the a11y linter doesn't model — hence the ignores; the rows
     themselves are real <button>s. tabindex=-1 keeps it out of the Tab order
     (focus lands here on click / when a child is focused) while still focusable. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div class="files" class:compact tabindex="-1" role="group" onkeydown={onKeydown}>
	<div class="bar">
		<nav class="crumbs">
			<button class="crumb" class:on={path === ''} onclick={() => goCrumb(-1)}>Artifacts</button>
			{#each crumbs as c, i (i)}
				<span class="sep">/</span>
				<button class="crumb" class:on={i === crumbs.length - 1} onclick={() => goCrumb(i)}>{c}</button>
			{/each}
		</nav>

		<!-- Filename search: fuzzy-matches over the whole (recursive) artifact tree.
		     Focus refreshes the cached file list; a non-empty query swaps the dir
		     listing for ranked matches. -->
		<div class="search">
			<span class="search-ic"><MagnifyingGlassIcon size={15} /></span>
			<input
				bind:this={searchEl}
				bind:value={query}
				type="text"
				class="search-input"
				placeholder="Search filenames…"
				spellcheck="false"
				autocapitalize="off"
				autocomplete="off"
				onfocus={onSearchFocus}
				onkeydown={(e) => e.key === 'Escape' && clearSearch()}
			/>
			{#if query}
				<button class="x-clear" title="Clear search" aria-label="Clear search" onclick={clearSearch}>
					<XIcon size={14} weight="bold" />
				</button>
			{/if}
		</div>

		<div class="bar-actions">
			<button class="tbtn" title="Refresh" onclick={refresh} aria-label="Refresh">
				<ArrowClockwiseIcon size={16} weight="bold" />
			</button>
			{#if onExpand}
				<!-- A real <a href> so middle-/cmd-click opens the full page in a new
				     tab natively; plain left-click is intercepted for in-app SPA nav. -->
				<a
					class="tbtn"
					href={expandHref}
					title="Open in the full Artifacts page"
					aria-label="Open in the full Artifacts page"
					onclick={onExpandClick}
				>
					<ArrowsOutSimpleIcon size={16} weight="bold" />
				</a>
			{/if}
			{@render toolbarEnd?.()}
		</div>
	</div>

	{#if error}<p class="err">{error}</p>{/if}

	<!-- list + gutter + preview. In top-down mode `.body` is display:contents, so
	     these flow in the parent column exactly as before; in the wide full-page
	     layout `.body` becomes a flex row (list left, draggable gutter, preview
	     right) — see the split media query in the style block. --list-w drives the
	     list width. -->
	<div class="body" style={bodyStyle}>
	<div class="list card" bind:this={listEl}>
		{#if searching}
			<!-- Search mode: ranked fuzzy matches over the whole artifact tree. -->
			{#if allFilesLoading && allFiles.length === 0}
				<p class="dim small pad">Loading…</p>
			{:else if searchResults.length === 0}
				<p class="dim small pad">No matching files.</p>
			{:else}
				{#each searchResults as f (f.path)}
					<button
						class="row"
						class:sel={f.path === selected}
						onclick={() => openSearchResult(f.path)}
						title={f.path}
					>
						<span class="ic"><FileIcon size={16} /></span>
						<span class="name path">
							{#if f.path.includes('/')}<span class="dir">{f.path.slice(0, f.path.lastIndexOf('/') + 1)}</span>{/if}<span class="base">{f.path.slice(f.path.lastIndexOf('/') + 1)}</span>
						</span>
						<span class="meta dim small">{fmtSize(f.size)}</span>
					</button>
				{/each}
			{/if}
		{:else if loading && entries.length === 0}
			<p class="dim small pad">Loading…</p>
		{:else if entries.length === 0}
			<p class="dim small pad">Empty.</p>
		{:else}
			{#each entries as e (e.name)}
				<button
					class="row"
					class:sel={!e.is_dir && join(path, e.name) === selected}
					onclick={() => openEntry(e)}
				>
					<span class="ic">
						{#if e.is_dir}<FolderIcon size={16} weight="fill" />{:else}<FileIcon size={16} />{/if}
					</span>
					<span class="name">{e.name}{e.is_dir ? '/' : ''}</span>
					<span class="meta dim small">{e.is_dir ? '' : fmtSize(e.size)}</span>
					{#if !compact}<span class="meta dim small">{fmtTime(e.mtime)}</span>{/if}
				</button>
			{/each}
		{/if}
	</div>

	<!-- Resize handle (visible only in the wide split layout): a window-splitter
	     between the file list and preview. Arrow keys resize it; aria-valuenow
	     exposes the current width. A focusable separator IS interactive here, which
	     the a11y linter doesn't model — hence the two ignores. -->
	<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div
		class="gutter"
		class:dragging
		role="separator"
		aria-orientation="vertical"
		aria-label="Resize file list"
		aria-valuenow={Math.round(listW)}
		aria-valuemin={LIST_W_MIN}
		aria-valuemax={LIST_W_MAX}
		tabindex="0"
		onpointerdown={startResize}
		onkeydown={keyResize}
	></div>

	{#if selected}
		<div class="preview card">
			<div class="phead">
				<span class="mono small fname">{selected}</span>
				<div class="actions">
					<button type="button" title="Reload file content" onclick={reloadPreview}>
						<ArrowClockwiseIcon size={14} /><span>Reload</span>
					</button>
					{#if canCopyContent}
						<button type="button" title="Copy file content" onclick={() => copy('content')}>
							{#if copied === 'content'}<CheckIcon size={14} />{:else}<CopyIcon size={14} />{/if}
							<span>Content</span>
						</button>
					{/if}
					<button type="button" title="Copy absolute file path" onclick={() => copy('path')}>
						{#if copied === 'path'}<CheckIcon size={14} />{:else}<CopyIcon size={14} />{/if}
						<span>Path</span>
					</button>
					<a class="btn" href={rawUrl} target="_blank" rel="noopener" title="Open raw file in a new tab">
						<ArrowSquareOutIcon size={14} /><span>Raw</span>
					</a>
				</div>
			</div>
			<div class="pbody">
				{#if previewKind === 'image'}
					<img src={rawUrl} alt={selected} />
				{:else if previewKind === 'markdown'}
					<!-- Authored .md files soft-wrap at ~80 cols; render with standard
					     paragraph reflow (breaks:false) so source wraps aren't hard <br>s.
					     resolveArtifacts resolves the file's RELATIVE URLs against the
					     artifact dir (not the page URL): images point at the raw endpoint,
					     while inter-file links are tagged for use:artifactLinks, which
					     opens them right here in the preview. -->
					<div class="md" use:artifactLinks>
						{@html renderMarkdown(previewText, {
							breaks: false,
							resolveArtifacts: { runId, baseDir: selectedDir }
						})}
					</div>
				{:else if previewKind === 'text'}
					<!-- Source files get syntax highlighting by extension (hljs, same lib +
					     Ayu token colors as markdown code fences); unknown types render as
					     escaped plain text. -->
					<pre class="pre hljs">{@html highlightFile(previewText, selected).html}</pre>
				{:else if previewKind === 'missing'}
					<p class="dim small">
						Not found — <code>{selected}</code> may have been moved, renamed or deleted.
					</p>
				{:else}
					<p class="dim small">No inline preview — use Open raw to download.</p>
				{/if}
			</div>
		</div>
	{:else}
		<!-- Empty right column in the split layout (hidden in top-down mode). -->
		<div class="placeholder dim small">Select a file to preview.</div>
	{/if}
	</div>
</div>

<style>
	.files {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
		min-height: 0;
	}
	.files.compact {
		gap: 0.5rem;
	}
	/* Default (top-down): the body is transparent to layout, so list + preview
	   flow directly in .files' column exactly as before. The gutter and the
	   split-only placeholder are hidden here; they light up in the media query. */
	.body {
		display: contents;
	}
	.gutter {
		display: none;
	}
	.placeholder {
		display: none;
	}
	.bar {
		display: flex;
		align-items: center;
		gap: 0.6rem;
	}
	/* Compact (chat aside): this row doubles as the aside's shell header when
	   artifacts is open, so it pixel-matches .sp-head there (same padding,
	   height, and bottom border) — the toggle button riding in via `toolbarEnd`
	   sits in the exact same spot whether the header shows "Status" or this
	   breadcrumb row. */
	.compact .bar {
		padding: 0.4rem 0.5rem 0.4rem 0.7rem;
		border-bottom: 1px solid var(--border);
	}
	.crumbs {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		flex-wrap: wrap;
		/* Crumbs take their natural width but yield to the search box, which grows
		   to fill the middle of the bar; they can shrink rather than crowd it out. */
		flex: 0 1 auto;
		min-width: 0;
	}
	.crumb {
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		padding: 0.1rem 0.2rem;
		font: inherit;
	}
	.crumb:hover {
		color: var(--text);
	}
	.crumb.on {
		color: var(--text);
		font-weight: 500;
	}
	.sep {
		color: var(--text-dim);
	}
	.bar-actions {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		flex-shrink: 0;
	}
	.tbtn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1.75rem;
		height: 1.75rem;
		/* Reset the global `button` padding (0.4rem 0.8rem): with box-sizing:
		   border-box the fixed 1.75rem square would otherwise shrink its content
		   box to ~0 and crush the centered icon to invisible. */
		padding: 0;
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
	}
	.tbtn:hover {
		color: var(--text);
	}
	/* The Expand affordance is an <a> (for native middle-/cmd-click) styled as a
	   tbtn: strip anchor defaults so it matches the sibling icon buttons. */
	a.tbtn {
		text-decoration: none;
	}
	/* Filename search: an inline pill in the toolbar that grows to fill the space
	   between the breadcrumbs and the action buttons. */
	.search {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		flex: 1 1 auto;
		min-width: 6rem;
		height: 1.75rem;
		padding: 0 0.4rem;
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}
	.search:focus-within {
		border-color: color-mix(in srgb, var(--accent) 55%, var(--border));
	}
	.search-ic {
		display: inline-flex;
		color: var(--text-dim);
		flex-shrink: 0;
	}
	/* The extra .files ancestor raises specificity above app.css's global
	   `input:not([type=checkbox]):not([type=radio])` (0,2,1) and `button` rules,
	   which would otherwise re-impose their own border/background/padding and make
	   the field render as a box INSIDE this pill. */
	.files .search input.search-input {
		flex: 1;
		min-width: 0;
		width: auto;
		background: none;
		border: none;
		color: var(--text);
		font: inherit;
		font-size: var(--fs-sm);
		padding: 0;
	}
	.files .search input.search-input:focus {
		outline: none;
	}
	.search-input::placeholder {
		color: var(--text-dim);
	}
	/* Clear (✕) sits inside the pill, so it's a bare icon rather than a bordered
	   button. The .files ancestor beats app.css's global `button` rule (0,0,1). */
	.files .search button.x-clear {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		width: 1.1rem;
		height: 1.1rem;
		padding: 0;
		background: none;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
	}
	.files .search button.x-clear:hover {
		color: var(--text);
	}
	/* Search results show the full subpath as a dimmed directory prefix + a
	   prominent basename; the prefix ellipsises so the basename stays visible. */
	.name.path {
		display: flex;
		min-width: 0;
	}
	.name.path .dir {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		color: var(--text-dim);
		flex: 0 1 auto;
	}
	.name.path .base {
		white-space: nowrap;
		flex: 0 0 auto;
	}
	.list {
		padding: 0.25rem;
		overflow-y: auto;
		flex: 0 1 auto;
		/* Cap the directory list so a folder with many entries scrolls internally
		   instead of pushing the preview off-screen. */
		max-height: 250px;
	}
	/* Compact (side-panel) mode: the host panel is already an elevated bordered
	   card, so the inner list/preview drop their own card chrome (no nested boxes)
	   and separate with a single divider instead. */
	.compact .list {
		background: none;
		border: none;
		border-radius: 0;
		border-bottom: 1px solid var(--border);
		padding: 0.15rem 0.25rem 0.4rem;
	}
	.pad {
		padding: 0.6rem;
	}
	.row {
		display: grid;
		grid-template-columns: auto 1fr auto auto;
		align-items: center;
		gap: 0.6rem;
		width: 100%;
		text-align: left;
		background: none;
		border: none;
		border-radius: var(--radius-sm);
		padding: 0.4rem 0.55rem;
		color: var(--text);
		cursor: pointer;
		font: inherit;
	}
	.compact .row {
		grid-template-columns: auto 1fr auto;
		padding: 0.3rem 0.45rem;
		font-size: var(--fs-sm);
	}
	.row:hover {
		background: var(--bg-elev2);
	}
	.row.sel {
		background: color-mix(in srgb, var(--accent) 14%, transparent);
	}
	.ic {
		display: inline-flex;
		color: var(--text-dim);
	}
	.name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.meta {
		white-space: nowrap;
	}
	.preview {
		padding: 0;
		flex: 1 1 0;
		min-height: 0;
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}
	/* Compact: no card chrome (the host panel provides it); the file header below
	   still sets it apart via its own subtle fill. */
	.compact .preview {
		background: none;
		border: none;
		border-radius: 0;
	}
	.phead {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 1rem;
		padding: 0.45rem 0.7rem;
		background: var(--bg-elev2);
		border-bottom: 1px solid var(--border);
		border-radius: var(--radius-md) var(--radius-md) 0 0;
	}
	/* Compact: the file header is flush with the panel edges (no rounded top,
	   no side inset) so it reads as a divider band, not a nested card header. */
	.compact .phead {
		border-radius: 0;
		padding: 0.45rem 0.7rem;
	}
	.fname {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		flex-shrink: 0;
	}
	.actions button,
	.actions .btn {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		background: var(--bg-elev);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		padding: 0.22rem 0.5rem;
		font: inherit;
		font-size: var(--fs-sm);
		cursor: pointer;
	}
	.actions button:hover,
	.actions .btn:hover {
		color: var(--text);
		text-decoration: none;
	}
	.pbody {
		padding: 0.6rem 0.7rem;
		overflow: auto;
		min-height: 0;
	}
	.preview img {
		max-width: 100%;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}
	.preview .pre {
		margin: 0;
		max-height: 60vh;
		overflow: auto;
	}
	.err {
		font-size: var(--fs-md);
	}

	/* --- Full-page left-right split ---
	   Only the wide, NON-compact instance (the /artifacts route) turns into a
	   side-by-side layout: a resizable file-list column on the left, the preview
	   filling the right. The chat aside (.compact) and narrow windows keep the
	   top-down stack above. 900px is the breakpoint below which side-by-side gets
	   too cramped and we fall back to stacking. */
	@media (min-width: 900px) {
		.files:not(.compact) .body {
			display: flex;
			flex-direction: row;
			align-items: stretch;
			gap: 0;
			flex: 1 1 0;
			min-height: 0;
		}
		.files:not(.compact) .list {
			flex: 0 0 var(--list-w);
			width: var(--list-w);
			/* Fill the column height and scroll internally (drop the top-down cap). */
			max-height: none;
			height: 100%;
		}
		.files:not(.compact) .preview {
			flex: 1 1 0;
			min-width: 0;
			height: 100%;
		}
		.files:not(.compact) .placeholder {
			display: flex;
			align-items: center;
			justify-content: center;
			flex: 1 1 0;
			min-width: 0;
			border: 1px dashed var(--border);
			border-radius: var(--radius-md);
		}
		/* In the split, the preview fills the pane, so let its <pre> use the full
		   column height rather than the top-down 60vh cap. */
		.files:not(.compact) .preview .pre {
			max-height: none;
		}
		/* Draggable divider between the two columns. */
		.files:not(.compact) .gutter {
			display: block;
			flex: 0 0 auto;
			width: 9px;
			margin: 0 -1px;
			cursor: col-resize;
			background: transparent;
			position: relative;
			z-index: 1;
		}
		/* A thin center line that brightens on hover / drag / focus. */
		/* Invisible at rest — the two bordered .card columns already show the seam;
		   the line only appears (accent) on hover / drag / focus to confirm it's a
		   grab target. Purely decorative: the drag/keyboard handlers live on the
		   9px-wide .gutter itself, so this being transparent doesn't affect resize. */
		.files:not(.compact) .gutter::before {
			content: '';
			position: absolute;
			top: 0;
			bottom: 0;
			left: 50%;
			width: 2px;
			transform: translateX(-50%);
			background: transparent;
			transition: background 0.1s ease;
		}
		.files:not(.compact) .gutter:hover::before,
		.files:not(.compact) .gutter.dragging::before,
		.files:not(.compact) .gutter:focus-visible::before {
			background: var(--accent-dim);
		}
		.files:not(.compact) .gutter:focus-visible {
			outline: none;
		}
	}
</style>

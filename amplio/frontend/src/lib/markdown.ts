/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { artifactRawUrl } from '$lib/api';
import { linkifyShortlinks } from '$lib/internal';
import { markedHighlight } from 'marked-highlight';
import hljs from 'highlight.js/lib/core';
import bash from 'highlight.js/lib/languages/bash';
import go from 'highlight.js/lib/languages/go';
import python from 'highlight.js/lib/languages/python';
import typescript from 'highlight.js/lib/languages/typescript';
import javascript from 'highlight.js/lib/languages/javascript';
import json from 'highlight.js/lib/languages/json';
import yaml from 'highlight.js/lib/languages/yaml';
import diff from 'highlight.js/lib/languages/diff';
import sql from 'highlight.js/lib/languages/sql';
import rust from 'highlight.js/lib/languages/rust';
import markdown from 'highlight.js/lib/languages/markdown';
import css from 'highlight.js/lib/languages/css';
import xml from 'highlight.js/lib/languages/xml';

// Register only the languages a coding-agent UI realistically renders, to keep
// the bundle lean (the full highlight.js build is large). Common aliases (sh,
// py, ts, js, yml) are registered by these language modules themselves.
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('go', go);
hljs.registerLanguage('python', python);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('json', json);
hljs.registerLanguage('yaml', yaml);
hljs.registerLanguage('diff', diff);
hljs.registerLanguage('sql', sql);
hljs.registerLanguage('rust', rust);
hljs.registerLanguage('markdown', markdown);
hljs.registerLanguage('css', css);
hljs.registerLanguage('xml', xml);

// Highlight ONLY language-tagged fences (```go, ```python, …) for a known
// language. Untagged or unknown-language blocks are left as plain text — no
// auto-detection, which is occasionally wrong and not worth the risk here.
// markedHighlight wraps output in <code class="hljs language-…">; the emitted
// hljs-* spans survive DOMPurify's default allowlist (span + class).
marked.use(
	markedHighlight({
		highlight(code, lang) {
			if (lang && hljs.getLanguage(lang)) {
				return hljs.highlight(code, { language: lang }).value;
			}
			// Untagged or unknown language: return the input UNCHANGED so
			// marked-highlight's updateToken no-ops (it only overwrites when the
			// returned string differs) and marked escapes the code normally.
			// Returning '' here would instead blank the block.
			return code;
		},
	}),
);

// GFM, gold-standard. `breaks` defaults to true for AGENT PROSE (chat / phase
// summaries / reports): the model emits single newlines mid-paragraph and means
// them as line breaks. Authored markdown FILES (the artifact viewer) are the
// opposite — they soft-wrap source lines at ~80 cols and rely on standard
// markdown paragraph reflow, so those callers pass breaks:false to avoid turning
// every wrap point into a hard <br>. Output is always sanitized before it
// reaches {@html}, since it originates from the model / tools / files (untrusted).
marked.setOptions({ gfm: true, breaks: true });

// Require DOUBLE tildes (~~text~~) for strikethrough. marked's GFM `del`
// tokenizer also treats a SINGLE ~ as a delimiter, but agents overwhelmingly use
// ~ to mean "approximately" (~3%, ~5k) — and two such approximations in one line
// (e.g. "~3x~5x") would then render as spurious strikethrough. Override `del` to
// match only ~~...~~; a lone ~ falls through to literal text. (The double-tilde
// regex mirrors GFM: no space just inside the delimiters, non-greedy body.)
const doubleTilde = /^~~(?=\S)([\s\S]*?\S)~~/;
marked.use({
	tokenizer: {
		del(src: string) {
			const m = doubleTilde.exec(src);
			if (!m) return undefined; // not ~~...~~ → let a single ~ stay as text
			return {
				type: 'del',
				raw: m[0],
				text: m[1],
				tokens: this.lexer.inlineTokens(m[1]),
			};
		},
	},
});

// Allow the artifact-pill markup (a <span> carrying data-artifact-path) through
// DOMPurify. The default allowlist keeps span+class but strips data-* attrs, so
// we opt that one attribute in. (We linkify AFTER sanitizing, so this only
// matters if a raw source ever contained the attribute — belt-and-suspenders.)
const PURIFY_OPTS = { ADD_ATTR: ['data-artifact-path'] };

// linkifyArtifactPaths rewrites `$AMPLIO_ARTIFACT_DIR/<relpath>` mentions into a
// clickable pill. The agent is prompted to reference artifact files with this
// exact prefix; it often wraps them in inline code (`$AMPLIO_ARTIFACT_DIR/x`).
// Both plain-text and inside-<code> occurrences are handled. The pill carries
// the RELATIVE path in a data attribute; the chat view opens it in the side
// panel via event delegation (the markdown reaches the DOM through {@html}, so
// per-pill Svelte handlers aren't possible). Runs on the sanitized HTML string.
const ARTIFACT_PREFIX = '$AMPLIO_ARTIFACT_DIR';
// Match the prefix, then OPTIONALLY `/<relpath>`. The path char class stops at
// whitespace, closing tags, quotes, and brackets so we don't swallow the
// surrounding sentence. A bare `$AMPLIO_ARTIFACT_DIR` or `$AMPLIO_ARTIFACT_DIR/`
// (folder mention, capture empty) becomes a root chip rather than an empty pill.
const ARTIFACT_RE = /\$AMPLIO_ARTIFACT_DIR(?:\/([^\s<>"'`)\]]*))?/g;
// Label shown on the root chip (a bare folder mention).
const ARTIFACT_ROOT_LABEL = 'Artifacts';

// Quote-escape for a value going into a double-quoted attribute. The input is a
// slice of ALREADY-sanitized HTML (so &/</> are already entities); we only need
// to neutralize any " so it can't break out of the attribute. Kept minimal to
// avoid double-escaping the existing entities.
function attrEsc(s: string): string {
	return s.replace(/"/g, '&quot;');
}

// pill builds the pill markup for one already-sanitized relative path (prefix
// already stripped by the caller's capture). rel is the capture, which may be
// undefined (bare `$AMPLIO_ARTIFACT_DIR`) or empty (`$AMPLIO_ARTIFACT_DIR/`) for
// a folder mention.
function pill(rel: string | undefined): string {
	// Strip trailing sentence punctuation the greedy path class may have caught
	// (e.g. "see $AMPLIO_ARTIFACT_DIR/x.md.") so it stays outside the pill.
	const trail = rel ? (rel.match(/[.,;:/]+$/)?.[0] ?? '') : '';
	const clean = rel && trail ? rel.slice(0, -trail.length) : (rel ?? '');
	// No path after the prefix → a folder mention. Emit a ROOT chip (empty
	// data-artifact-path opens the browser at the artifact root) labeled
	// "Artifacts", rather than an empty/degenerate pill.
	if (clean === '') {
		return `<button type="button" class="artifact-pill" data-artifact-path="" title="Artifacts">${ARTIFACT_ROOT_LABEL}</button>${trail}`;
	}
	// `clean` is already-sanitized HTML text; safe to reinsert as element text. A
	// real <button> makes the pill natively keyboard-focusable/activatable; the
	// chat view's delegated click handler catches both mouse and keyboard clicks.
	return `<button type="button" class="artifact-pill" data-artifact-path="${attrEsc(clean)}" title="${attrEsc(clean)}">${clean}</button>${trail}`;
}

export function linkifyArtifactPaths(html: string): string {
	if (!html.includes(ARTIFACT_PREFIX)) return html;
	// Pass 1: UNWRAP an inline <code> that contains an artifact path, so the pill
	// isn't nested inside inline-code styling (which produced a double-pill look).
	// Agents almost always put just the path in backticks; the whole code span is
	// replaced by pills for any $AMPLIO_ARTIFACT_DIR mentions it holds.
	html = html.replace(/<code>([^<]*\$AMPLIO_ARTIFACT_DIR[^<]*)<\/code>/g, (_m, inner: string) =>
		inner.replace(ARTIFACT_RE, (_x, rel: string | undefined) => pill(rel)),
	);
	// Pass 2: bare occurrences in ordinary text. (Pills from pass 1 carry no
	// literal prefix, so they aren't re-matched.)
	return html.replace(ARTIFACT_RE, (_m, rel: string | undefined) => pill(rel));
}

// A URL is "absolute" (leave it alone) if it has a scheme (http:, https:, data:,
// mailto:, …), is protocol-relative (//host), is root-relative (/path), or is a
// pure fragment (#anchor). Everything else is a relative path we resolve against
// the previewed file's directory.
const ABSOLUTE_URL_RE = /^([a-z][a-z0-9+.-]*:|\/\/|\/|#)/i;

// resolveArtifactPath resolves a relative URL (from a previewed markdown file)
// against that file's directory, normalizing . and .. segments, into a path
// relative to the ARTIFACT ROOT. Any #fragment / ?query is split off and
// returned separately (callers re-append it where it makes sense). A path that
// climbs above the artifact root is clamped at the root (the server's os.Root
// also rejects escapes, so this is belt-and-suspenders).
function resolveArtifactPath(
	baseDir: string,
	url: string,
): { path: string; suffix: string; isDir: boolean } {
	const hashIdx = url.search(/[?#]/);
	const suffix = hashIdx >= 0 ? url.slice(hashIdx) : '';
	const rawPath = hashIdx >= 0 ? url.slice(0, hashIdx) : url;
	const decoded = decodeURIComponent(rawPath);
	// A trailing slash is the author saying "directory" — it survives neither the
	// segment walk below nor a join, so capture it first.
	const isDir = decoded.endsWith('/');
	const segs = baseDir ? baseDir.split('/') : [];
	for (const seg of decoded.split('/')) {
		if (seg === '' || seg === '.') continue;
		if (seg === '..') segs.pop();
		else segs.push(seg);
	}
	return { path: segs.join('/'), suffix, isDir };
}

// resolveArtifactRel is the raw-endpoint form, for embedded media (<img>) that
// the browser must fetch directly.
function resolveArtifactRel(runId: string, baseDir: string, url: string): string {
	const { path, suffix } = resolveArtifactPath(baseDir, url);
	return artifactRawUrl(runId, path) + suffix;
}

// rewriteRelativeUrls points a previewed markdown file's RELATIVE img/a URLs at
// the artifact-raw endpoint, so embedded images/plots and inter-file links
// resolve against the artifact dir instead of the (wrong) page URL. Runs on the
// sanitized HTML; only touches relative URLs (absolute/data/anchor pass through).
// baseDir is the previewed file's directory ("" = artifact root).
function rewriteRelativeUrls(html: string, runId: string, baseDir: string): string {
	const doc = new DOMParser().parseFromString(html, 'text/html');
	for (const img of doc.querySelectorAll('img[src]')) {
		const src = img.getAttribute('src') ?? '';
		if (src && !ABSOLUTE_URL_RE.test(src)) {
			img.setAttribute('src', resolveArtifactRel(runId, baseDir, src));
		}
	}
	for (const a of doc.querySelectorAll('a[href]')) {
		const href = a.getAttribute('href') ?? '';
		if (!href || ABSOLUTE_URL_RE.test(href)) continue;
		const { path, isDir } = resolveArtifactPath(baseDir, href);
		// An inter-file link is a link between DOCUMENTS, so it should land in the
		// viewer, not dump the raw bytes: data-artifact-path is the in-app hook the
		// artifact browser intercepts (see its artifactLinks action) to open the
		// target in its own preview pane. A trailing slash marks a directory link.
		a.setAttribute('data-artifact-path', isDir ? `${path}/` : path);
		// The href stays a REAL link so ⌘/middle-click still opens a new tab — but
		// pointed at the artifacts ROUTE, so that tab lands in the viewer too
		// (previously this was the raw endpoint, i.e. an unrendered file dump).
		// Directories have no route-level deep link yet, so they open at the root.
		// The #fragment is dropped: it addresses a heading inside the target
		// document, which the viewer doesn't (yet) anchor to.
		a.setAttribute(
			'href',
			path && !isDir
				? `/runs/${runId}/artifacts?file=${encodeURIComponent(path)}`
				: `/runs/${runId}/artifacts`,
		);
		if (!a.getAttribute('title')) {
			a.setAttribute('title', path ? `${path} — open in the artifact viewer` : 'artifact root');
		}
	}
	return doc.body.innerHTML;
}

export function renderMarkdown(
	src: string | undefined | null,
	opts: {
		breaks?: boolean;
		linkifyArtifacts?: boolean;
		// Shortlinks in prose are linkified by default; set to opt a surface out.
		noShortlinks?: boolean;
		// When set, relative img/a URLs are resolved against the artifact dir: runId
		// scopes the /artifacts/raw endpoint; baseDir is the previewed file's own
		// directory ("" = root). Used by the artifact markdown preview so embedded
		// images and inter-file links work.
		resolveArtifacts?: { runId: string; baseDir: string };
	} = {},
): string {
	if (!src) return '';
	const html = marked.parse(src, { async: false, breaks: opts.breaks ?? true }) as string;
	let clean = DOMPurify.sanitize(html, PURIFY_OPTS);
	if (opts.resolveArtifacts) {
		clean = rewriteRelativeUrls(clean, opts.resolveArtifacts.runId, opts.resolveArtifacts.baseDir);
	}
	if (!opts.noShortlinks) {
		clean = linkifyShortlinks(clean);
	}
	return opts.linkifyArtifacts ? linkifyArtifactPaths(clean) : clean;
}

// Map a filename to a registered hljs language for the artifact source-file
// preview. Only the languages registered above are usable; anything else falls
// back to plain (escaped) text. Keys are lowercased file extensions (and a few
// exact basenames for extensionless config files).
const EXT_LANG: Record<string, string> = {
	go: 'go',
	py: 'python',
	ts: 'typescript',
	tsx: 'typescript',
	js: 'javascript',
	jsx: 'javascript',
	mjs: 'javascript',
	cjs: 'javascript',
	json: 'json',
	yaml: 'yaml',
	yml: 'yaml',
	sh: 'bash',
	bash: 'bash',
	sql: 'sql',
	rs: 'rust',
	css: 'css',
	html: 'xml',
	xml: 'xml',
	svg: 'xml',
	diff: 'diff',
	patch: 'diff',
};
const BASENAME_LANG: Record<string, string> = {
	dockerfile: 'bash',
	makefile: 'bash',
	'.bashrc': 'bash',
};

// escapeHtml escapes text for safe insertion via {@html} (the plain-text
// fallback when a file's language isn't highlighted).
function escapeHtml(s: string): string {
	return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!);
}

// highlightFile returns syntax-highlighted HTML for a source file's raw text,
// choosing the language from its filename. Unknown types return escaped plain
// text. The emitted markup is hljs <span class="hljs-…"> tokens (safe, no
// arbitrary HTML from the source) so it's used directly via {@html}. `hasLang`
// reports whether highlighting actually applied (lets the caller style/label).
export function highlightFile(text: string, filename: string): { html: string; hasLang: boolean } {
	const base = filename.slice(filename.lastIndexOf('/') + 1).toLowerCase();
	const dot = base.lastIndexOf('.');
	const ext = dot > 0 ? base.slice(dot + 1) : '';
	const lang = EXT_LANG[ext] ?? BASENAME_LANG[base];
	if (lang && hljs.getLanguage(lang)) {
		try {
			return { html: hljs.highlight(text, { language: lang }).value, hasLang: true };
		} catch {
			/* fall through to plain */
		}
	}
	return { html: escapeHtml(text), hasLang: false };
}

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

// Dynamic favicon swap, composed from two independent signals:
//   - dotted: a red top-right badge when there are unread updates (regardless
//     of tab visibility; the actively-viewed run is exempted by the updates
//     store, which keeps the badge meaningful).
//   - chat: a speech-bubble TAIL on the lower-left of the disc when the tab is
//     on a chat page, so a chat tab is recognizable at a glance among many.
// The two are orthogonal (badge = top-right corner, tail = bottom-left corner),
// so all four combinations render cleanly.
//
// SVG (not canvas) because:
//   - The source favicon is an inline SVG already (static/favicon.svg).
//   - A modified-SVG data: URL is one line; canvas would have to wait for an
//     image to load, draw, export — more code, more failure modes.
//   - Chrome / Firefox both honor mid-session <link rel="icon"> swaps to SVG.
//
// Keep this in sync with static/favicon.svg if the mark ever changes.

const ORIGINAL_HREF = '/favicon.svg';

// All composed variants render in a 256×256 viewBox so the chevron/badge share
// one coordinate system.
//
// The disc is IDENTICAL in both modes — the chat variant is the exact same
// circle PLUS a lower-left corner fill that squares off that one quadrant into a
// speech-bubble "tail" (à la Phosphor's chat-teardrop, but composed so the
// circle stays pixel-identical between modes and the chevron never shifts). The
// squared-off corner reads clearly even at favicon scale, unlike a thin tail.
const DISC = `<circle cx="128" cy="128" r="128" fill="#56B6C6"/>`;

// Lower-left corner fill: left edge tangent at (0,128), bottom edge tangent at
// (128,256), outer (bottom-left) corner rounded by TAIL_R. Its top/right edges
// sit at the disc center, fully inside the circle, so the union is just "the
// circle with a squared lower-left corner."
const TAIL_R = 60;
const TAIL = `<path d="M0,128 L0,${256 - TAIL_R} Q0,256 ${TAIL_R},256 L128,256 L128,128 Z" fill="#56B6C6"/>`;

// The amplio chevron (same glyph as static/favicon.svg), scaled from its native
// 24-space onto the 256-space disc and centered on (128,128). Shared by both
// modes — the disc center is identical, so the chevron never shifts when the
// favicon swaps between plain and chat on navigation.
const CHEVRON =
	`<g fill="none" stroke="#ffffff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" ` +
	`transform="translate(21.3 21.3) scale(9.4) translate(0 -0.6) rotate(-90 12 12) translate(4.2 3.6) scale(0.7)">` +
	`<path d="m9 18 6-6-6-6"/></g>`;

// Unread badge: top-right notification dot. Sits in the corner the disc doesn't
// reach and straddles its edge; the dark ring gives contrast on teal and on any
// (transparent) tab background. Warm red on cool teal = high-attention.
const BADGE = `<circle cx="196" cy="54" r="53" fill="#ff5d5d" stroke="#0b0e14" stroke-width="16"/>`;

// buildSVG composes a variant in 256-space.
function buildSVG(dotted: boolean, chat: boolean): string {
	return (
		`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">` +
		DISC +
		(chat ? TAIL : '') +
		CHEVRON +
		(dotted ? BADGE : '') +
		`</svg>`
	);
}

function href(dotted: boolean, chat: boolean): string {
	// The plain, no-signal variant is the static file (lets the browser cache it).
	if (!dotted && !chat) return ORIGINAL_HREF;
	return 'data:image/svg+xml;utf8,' + encodeURIComponent(buildSVG(dotted, chat));
}

export function setFavicon(opts: { dotted: boolean; chat: boolean }) {
	if (typeof document === 'undefined') return;
	const link = document.querySelector('link[rel="icon"]') as HTMLLinkElement | null;
	if (!link) return;
	const want = href(opts.dotted, opts.chat);
	if (link.href !== want) link.href = want;
}

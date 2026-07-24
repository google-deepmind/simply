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

// Web auth state. The server gates writes on a token presented as an HttpOnly
// cookie (set once via the magic-link exchange) or a Bearer header (CLI). Reads
// are open, so a token-less shared URL resolves to a readonly view.
//
// init() runs once on app mount: if the URL carries ?token= (the link the owner
// opened from their terminal), it POSTs it to /api/auth/login so the server sets
// the cookie, then strips the token from the URL — after which the cookie alone
// authorizes this browser, and a copied (token-less) URL shared with a teammate
// lands them in readonly. `authed` drives the UI (hide write controls); the
// server enforces it regardless.
class Auth {
	authed = $state(false);
	user = $state(''); // server owner's username, for the readonly banner
	ready = $state(false); // false until /api/auth has answered (avoid UI flash)

	async init() {
		if (typeof window === 'undefined') return;
		const url = new URL(window.location.href);
		const token = url.searchParams.get('token');
		if (token) {
			try {
				await fetch(`/api/auth/login?token=${encodeURIComponent(token)}`, { method: 'POST' });
			} catch {
				// network error — fall through; we'll just resolve to readonly
			}
			url.searchParams.delete('token');
			history.replaceState({}, '', url.toString());
		}
		await this.refresh();
	}

	async refresh() {
		try {
			const res = await fetch('/api/auth');
			if (res.ok) {
				const d = (await res.json()) as { authed?: boolean; user?: string };
				this.authed = !!d.authed;
				this.user = d.user ?? '';
			}
		} catch {
			// leave defaults (readonly)
		}
		this.ready = true;
	}
}

export const auth = new Auth();

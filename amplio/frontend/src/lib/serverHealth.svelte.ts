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

// Global server-reachability state, observable from any component.
//
// Updated by the SSE handler (open/error) and the api.ts fetch wrapper
// (success/network-fail). The top banner watches this to surface a loud
// "server unreachable" message — important because amplio is an SPA and an
// unreachable server otherwise just makes individual components silently
// stale or broken, which is confusing.
//
// States:
//   ok            normal
//   reconnecting  one signal failed; we're waiting briefly to see if it
//                 recovers before raising the alarm
//   down          confirmed unreachable; banner is shown

export type ServerHealthState = 'ok' | 'reconnecting' | 'down';

// Promote 'reconnecting' to 'down' after this much wall time without a fresh
// 'ok' signal. Short enough to be responsive; long enough to ride out the
// browser's EventSource reconnect dance on a transient blip.
const RECONNECTING_TO_DOWN_MS = 3_000;

class ServerHealth {
	state = $state<ServerHealthState>('ok');
	private demoteHandle: ReturnType<typeof setTimeout> | undefined;

	/** Successful round-trip to the server — clear any alarm. */
	markOk() {
		if (this.demoteHandle) {
			clearTimeout(this.demoteHandle);
			this.demoteHandle = undefined;
		}
		this.state = 'ok';
	}

	/** A signal failed but might still recover (EventSource auto-reconnect,
	 *  brief network blip). Shows nothing visible yet; if not cleared by an
	 *  `ok` within RECONNECTING_TO_DOWN_MS, transitions to `down`. */
	markReconnecting() {
		if (this.state === 'down') return; // already loud; stay loud
		if (this.state !== 'reconnecting') this.state = 'reconnecting';
		if (this.demoteHandle) return;
		this.demoteHandle = setTimeout(() => {
			this.demoteHandle = undefined;
			if (this.state === 'reconnecting') this.state = 'down';
		}, RECONNECTING_TO_DOWN_MS);
	}

	/** Definitive failure (fetch threw a network error). Raise the alarm now. */
	markDown() {
		if (this.demoteHandle) {
			clearTimeout(this.demoteHandle);
			this.demoteHandle = undefined;
		}
		this.state = 'down';
	}
}

export const serverHealth = new ServerHealth();

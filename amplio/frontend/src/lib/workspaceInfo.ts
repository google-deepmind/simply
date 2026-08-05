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

// Process-lifetime cache for the New-Run workspace info. The abs path can only
// come from the server, so we prefetch it at page load (alongside the run list)
// and read it synchronously when the composer expands — avoiding a
// "working dir" → abs-path flicker on first open.
import { api } from './api';
import type { WorkspaceInfo } from './types';

let cache: WorkspaceInfo | null = null;
let inflight: Promise<WorkspaceInfo> | null = null;

// cachedWorkspaceInfo returns the prefetched info if available, else null.
export function cachedWorkspaceInfo(): WorkspaceInfo | null {
	return cache;
}

// loadWorkspaceInfo resolves the info, fetching once and memoizing. Concurrent
// callers share the in-flight request; a failure clears it so a later call retries.
export function loadWorkspaceInfo(): Promise<WorkspaceInfo> {
	if (cache) return Promise.resolve(cache);
	if (!inflight) {
		inflight = api
			.getWorkspaceInfo()
			.then((info) => {
				cache = info;
				return info;
			})
			.catch((e) => {
				inflight = null; // allow a retry
				throw e;
			});
	}
	return inflight;
}

// prefetchWorkspaceInfo warms the cache; safe to call early and to ignore errors.
export function prefetchWorkspaceInfo(): void {
	loadWorkspaceInfo().catch(() => {});
}

let refreshInflight: Promise<WorkspaceInfo> | null = null;

// refreshWorkspaceInfo forces a fresh fetch (bypassing the cache) and updates the
// cache on success. Unlike loadWorkspaceInfo (cache-first), this always hits the
// server — used for a fire-and-forget background refresh when the New-Run form is
// expanded, so a workspace created or used since page load shows up in the
// recents without a reload. Concurrent calls coalesce; once settled, the next
// call fetches anew. Failures are non-fatal (the cached list stays usable).
export function refreshWorkspaceInfo(): Promise<WorkspaceInfo> {
	if (refreshInflight) return refreshInflight;
	refreshInflight = api
		.getWorkspaceInfo()
		.then((info) => {
			cache = info;
			return info;
		})
		.finally(() => {
			refreshInflight = null;
		});
	return refreshInflight;
}

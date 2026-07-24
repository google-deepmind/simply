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

// Process-lifetime cache for the agent model menu, prefetched at page load so
// the model selector renders its value immediately instead of populating
// on-the-fly when the composer first expands. force=true refetches after a
// mutation (add/remove custom model).
import { api } from './api';
import type { ModelMenu } from './types';

let cache: ModelMenu | null = null;
let inflight: Promise<ModelMenu> | null = null;

// cachedModelMenu returns the prefetched menu if available, else null.
export function cachedModelMenu(): ModelMenu | null {
	return cache;
}

// loadModelMenu resolves the menu, memoizing. force discards the cache and
// refetches (use after add/remove). A failure clears the in-flight so a later
// call retries.
export function loadModelMenu(force = false): Promise<ModelMenu> {
	if (force) {
		cache = null;
		inflight = null;
	}
	if (cache) return Promise.resolve(cache);
	if (!inflight) {
		inflight = api
			.listModels()
			.then((m) => {
				cache = m;
				return m;
			})
			.catch((e) => {
				inflight = null;
				throw e;
			});
	}
	return inflight;
}

// prefetchModelMenu warms the cache; safe to call early and to ignore errors.
export function prefetchModelMenu(): void {
	loadModelMenu().catch(() => {});
}

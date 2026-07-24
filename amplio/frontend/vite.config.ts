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

import { sveltekit } from '@sveltejs/kit/vite';
import { sveltePhosphorOptimize } from 'phosphor-svelte/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	// sveltePhosphorOptimize rewrites barrel `import { X } from 'phosphor-svelte'`
	// to per-icon subpaths so only used icons are bundled (otherwise the whole
	// ~8MB icon dataset is pulled in).
	plugins: [sveltekit(), sveltePhosphorOptimize()],
	server: {
		proxy: {
			// Dev: forward API + SSE to a locally running `amplio serve`
			// (default listen 0.0.0.0:26759). Keep in sync with config.toml `listen`.
			'/api': {
				target: 'http://localhost:26759',
				changeOrigin: true,
			},
		},
	},
});

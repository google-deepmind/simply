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

import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	compilerOptions: {
		// Force runes mode for the project, except for libraries.
		runes: ({ filename }) => (filename.split(/[/\\]/).includes('node_modules') ? undefined : true)
	},
	kit: {
		// SPA: a single index.html fallback + hashed assets, embedded into the Go
		// server via go:embed. Output goes to the git-ignored webdist/client
		// subdir (the adapter wipes its output dir on each build, so the embed
		// anchor webdist/README.md lives one level up and survives).
		adapter: adapter({
			pages: '../internal/server/webdist/client',
			assets: '../internal/server/webdist/client',
			fallback: 'index.html',
			precompress: false,
			strict: false
		}),
		paths: { relative: true }
	}
};

export default config;

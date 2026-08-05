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
	// Session-log index: no session in the path, so resolve one and redirect to
	// the real viewer. Reached from a hand-typed URL, or from the nav rail before
	// the run detail has loaded (it can't know a session id yet).
	//
	// ?view=chat selects the chat-log mode. It exists ONLY here — the mode is a
	// path segment everywhere else; this route can't use one because
	// /sessions/chat would be ambiguous with a session literally named "chat".
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { getRunStore } from '$lib/runContext.svelte';
	import { logHref, recallSession } from '$lib/logView.svelte';

	const store = getRunStore();
	const runId = $derived(page.params.id ?? '');
	const mode = $derived(page.url.searchParams.get('view') === 'chat' ? 'chat' : 'trajectory');

	$effect(() => {
		const sessions = store.detail?.sessions ?? [];
		if (!runId || !sessions.length) return;
		const remembered = recallSession(runId);
		const roots = sessions.filter((s) => !s.parent_id);
		// Chat log defaults to the conversational session when there is one;
		// trajectory defaults to the autonomous root (the run's main worker).
		const preferred =
			mode === 'chat'
				? (sessions.find((s) => s.agent_type === 'chatbot') ??
					roots.find((s) => s.agent_type !== 'chatbot') ??
					roots[0])
				: (roots.find((s) => s.agent_type !== 'chatbot') ?? roots[0]);
		const sid =
			remembered && sessions.some((s) => s.session_id === remembered)
				? remembered
				: (preferred?.session_id ?? '');
		if (!sid) return;
		// replaceState: this route is a resolver, not a place — Back should return
		// to wherever the operator came from, not bounce through here again.
		void goto(logHref(runId, sid, mode), { replaceState: true });
	});
</script>

{#if store.detail && !store.detail.sessions.length}
	<p class="dim">This run has no sessions yet.</p>
{:else}
	<p class="dim">Loading…</p>
{/if}

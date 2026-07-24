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
	import { browser } from '$app/environment';
	import { PaperPlaneTiltIcon } from 'phosphor-svelte';

	let {
		onSend,
		placeholder = 'Send a message — Enter to send, Shift+Enter for newline',
		draftKey,
		element = $bindable()
	}: {
		onSend: (content: string) => Promise<void>;
		placeholder?: string;
		draftKey?: string; // when set, persist the unsent draft in localStorage
		element?: HTMLTextAreaElement; // bound out so callers can focus the input
	} = $props();

	let content = $state('');
	let busy = $state(false);

	// Restore a saved draft when the key (re)appears; survives reloads / outages.
	$effect(() => {
		if (browser && draftKey) content = localStorage.getItem(draftKey) ?? '';
	});

	function saveDraft() {
		if (!browser || !draftKey) return;
		if (content) localStorage.setItem(draftKey, content);
		else localStorage.removeItem(draftKey);
	}

	function autogrow() {
		if (!element) return;
		element.style.height = 'auto';
		element.style.height = Math.min(element.scrollHeight, 200) + 'px';
	}

	function onInput() {
		autogrow();
		saveDraft();
	}

	async function send() {
		const c = content.trim();
		if (!c || busy) return;
		const sent = content;
		// Clear immediately so focus stays and you can keep typing; restore the
		// draft on failure for retry.
		content = '';
		if (element) element.style.height = 'auto';
		saveDraft();
		busy = true;
		try {
			await onSend(sent);
		} catch {
			content = sent;
			saveDraft();
		} finally {
			busy = false;
			element?.focus();
		}
	}

	function onKey(e: KeyboardEvent) {
		// Enter sends; Shift+Enter is a newline. IME-safe.
		if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
			e.preventDefault();
			void send();
		}
	}
</script>

<form
	class="box"
	onsubmit={(e) => {
		e.preventDefault();
		void send();
	}}
>
	<textarea
		class="input"
		bind:this={element}
		bind:value={content}
		{placeholder}
		rows="1"
		oninput={onInput}
		onkeydown={onKey}
	></textarea>
	<button class="primary send" disabled={busy || !content.trim()}>
		Send
		<PaperPlaneTiltIcon size={14} weight="fill" />
	</button>
</form>

<style>
	.box {
		display: flex;
		align-items: flex-end;
		gap: 0.5rem;
	}
	.input {
		flex: 1;
		resize: none;
		max-height: 200px;
		overflow-y: auto;
		line-height: 1.4;
	}
	.send {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
	}
</style>

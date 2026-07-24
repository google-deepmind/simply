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
	import { api, errorText } from '$lib/api';
	import { auth } from '$lib/auth.svelte';
	import { pageTitle } from '$lib/title';
	import { formatLocalIso } from '$lib/time';
	import CopyButton from '$lib/components/CopyButton.svelte';
	import type { AboutInfo, TestLLMResult } from '$lib/types';

	let info = $state<AboutInfo | null>(null);
	let loadErr = $state('');

	$effect(() => {
		void load();
	});
	async function load() {
		try {
			info = await api.getAbout();
			loadErr = '';
		} catch (e) {
			loadErr = errorText(e); // '' when unreachable (global banner covers it)
		}
	}

	const versionLine = $derived(
		info
			? [
					info.channel + (info.modified ? ' (dirty)' : ''),
					info.commit ? info.commit.slice(0, 12) : null,
					info.build_time ? formatLocalIso(info.build_time) : null,
					info.go_version
				]
					.filter(Boolean)
					.join(' · ')
			: ''
	);

	// --- LLM tester ---
	let spec = $state('');
	let testing = $state(false);
	let result = $state<TestLLMResult | null>(null);
	let testErr = $state('');

	async function runTest() {
		const s = spec.trim();
		if (!s || testing) return;
		testing = true;
		result = null;
		testErr = '';
		try {
			result = await api.testLLM(s);
		} catch (e) {
			testErr = String(e);
		} finally {
			testing = false;
		}
	}
</script>

<svelte:head>
	<title>{pageTitle('About')}</title>
</svelte:head>

<div class="about">
	<h2>About this server</h2>

	{#if loadErr}
		<p class="err">{loadErr}</p>
	{:else if !info}
		<p class="dim">Loading…</p>
	{:else}
		<section class="card">
			<h3>Server</h3>
			<dl class="kv">
				<dt>version</dt>
				<dd class="mono">{versionLine}</dd>
				<dt>owner</dt>
				<dd class="mono">{info.owner || '—'}</dd>
				<dt>auth</dt>
				<dd class="mono">{info.auth_on ? 'token required for writes' : 'open (no token)'}</dd>
			</dl>
		</section>

		<section class="card">
			<h3>On-disk layout</h3>
			<dl class="kv">
				<dt>data dir</dt>
				<dd class="mono path">{info.data_dir} <CopyButton text={info.data_dir} /></dd>
				<dt>config</dt>
				<dd class="mono path">{info.config_path} <CopyButton text={info.config_path} /></dd>
				<dt>logs</dt>
				<dd class="mono path">{info.logs_dir} <CopyButton text={info.logs_dir} /></dd>
			</dl>
		</section>

		<section class="card">
			<h3>Models</h3>
			<dl class="kv">
				<dt>default</dt>
				<dd class="mono">{info.default_llm || '—'}</dd>
				<dt>menu</dt>
				<dd class="mono">{info.models.length ? info.models.join(', ') : '—'}</dd>
				<dt>system (hq)</dt>
				<dd class="mono">{info.system_llm_hq || '—'}</dd>
				<dt>system (fast)</dt>
				<dd class="mono">{info.system_llm_fast || '—'}</dd>
			</dl>
		</section>

		<section class="card">
			<h3>LLM tester</h3>
			<p class="dim small">
				Validate a model spec (<code>provider:model[?args]</code>) before starting a run —
				builds the provider and makes one trivial call. Catches typos, unknown providers, and
				auth/scope failures cheaply.
			</p>
			{#if auth.ready && !auth.authed}
				<p class="dim small">Sign in to run the tester (it makes a real LLM call).</p>
			{:else}
				<div class="tester">
					<input
						class="mono"
						placeholder="vertex-claude:claude-opus-4-8"
						bind:value={spec}
						onkeydown={(e) => {
							if (e.key === 'Enter') runTest();
						}}
					/>
					<button class="primary" onclick={runTest} disabled={testing || !spec.trim()}>
						{testing ? 'Testing…' : 'Test'}
					</button>
				</div>
				{#if testErr}
					<p class="err small">{testErr}</p>
				{:else if result}
					{#if result.ok}
						<p class="result ok small">
							✓ OK — model <span class="mono">{result.model_id}</span>, {result.latency_ms}ms
							{#if result.reply}· reply: <span class="mono">{result.reply}</span>{/if}
						</p>
					{:else}
						<p class="result bad small">✗ {result.error}</p>
					{/if}
				{/if}
			{/if}
		</section>
	{/if}
</div>

<style>
	.about {
		max-width: 52rem;
		margin: 0 auto;
		padding: 1.2rem;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	h2 {
		margin: 0;
		font-size: var(--fs-xl);
	}
	h3 {
		margin: 0 0 0.6rem;
		font-size: var(--fs-lg);
	}
	.card {
		display: flex;
		flex-direction: column;
	}
	/* Label/value grid shared by the info panels (mirrors the run-overview kv). */
	.kv {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: 0.3rem 0.9rem;
		margin: 0;
	}
	.kv dt {
		color: var(--text-dim);
		font-size: var(--fs-sm);
	}
	.kv dd {
		margin: 0;
		min-width: 0;
		overflow-wrap: anywhere;
	}
	.path {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
	}
	.tester {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.3rem;
	}
	.tester input {
		flex: 1;
	}
	.tester button {
		flex-shrink: 0;
	}
	.result {
		margin: 0.6rem 0 0;
	}
	.result.ok {
		color: var(--ok);
	}
	.result.bad {
		color: var(--err);
	}
</style>

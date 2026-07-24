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
	import type { AgentEvent } from '$lib/types';
	import { renderMarkdown } from '$lib/markdown';
	import { blobUrl } from '$lib/api';
	import { CaretRightIcon } from 'phosphor-svelte';
	import ToolCall from './ToolCall.svelte';
	// results maps a tool_call_id to its ToolResultEvent. When provided (the
	// trajectory view), this event's tool_calls render as paired call+result
	// blocks via <ToolCall>; absent, tool_calls render call-only (legacy).
	let {
		ev,
		step,
		runId = '',
		results = {}
	}: { ev: AgentEvent; step: number; runId?: string; results?: Record<string, AgentEvent> } =
		$props();
</script>

<div class="evt {ev.type}">
	<div class="hd">
		<span class="type">{ev.type}</span>
		<span class="dim">step {step}</span>
	</div>
	{#if ev.thoughts}
		<details>
			<summary>
				<span class="caret"><CaretRightIcon size={12} /></span>
				<span class="dim">thinking</span>
			</summary>
			<div class="md dim">{@html renderMarkdown(ev.thoughts)}</div>
		</details>
	{/if}
	{#if ev.content}
		{#if ev.type === 'assistant'}
			<!-- Agent prose → markdown; tool results / inputs / system stay raw. -->
			<div class="md">{@html renderMarkdown(ev.content)}</div>
		{:else}
			<pre class="pre">{ev.content}</pre>
		{/if}
	{/if}
	{#if ev.tool_calls?.length}
		{#each ev.tool_calls as tc (tc.id)}
			{@const res = results[tc.id]}
			<div class="toolwrap">
				<ToolCall
					name={tc.name}
					args={tc.arguments}
					result={res?.content}
					isError={res?.is_error ?? false}
					attachments={res?.attachments ?? []}
					{runId}
				/>
			</div>
		{/each}
	{/if}
	{#if ev.attachments?.length}
		<div class="atts">
			{#each ev.attachments as att, i (att.blob_key ?? i)}
				{#if att.blob_key && att.mime_type.startsWith('image/')}
					<a
						class="att-img"
						href={blobUrl(runId, att.blob_key)}
						target="_blank"
						rel="noopener"
						title={att.source_hint}
					>
						<img src={blobUrl(runId, att.blob_key)} alt={att.source_hint ?? 'attachment'} />
					</a>
				{:else if att.blob_key}
					<a class="dim small" href={blobUrl(runId, att.blob_key)} target="_blank" rel="noopener">
						{att.source_hint ?? att.mime_type} ({att.size ?? 0} bytes)
					</a>
				{/if}
			{/each}
		</div>
	{/if}
	{#if ev.type === 'child_result'}
		<div class="dim small">child {ev.child_session_id} → {ev.verdict}</div>
	{/if}
</div>

<style>
	.evt {
		border: 1px solid var(--border);
		border-left: 3px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0.5rem 0.7rem;
		background: var(--bg-elev);
	}
	.evt.assistant {
		border-left-color: var(--accent);
	}
	.evt.tool_result {
		border-left-color: var(--ok);
	}
	.evt.user {
		border-left-color: var(--warn);
	}
	.evt.child_result {
		border-left-color: var(--child);
	}
	.hd {
		display: flex;
		justify-content: space-between;
		font-size: var(--fs-sm);
		margin-bottom: 0.3rem;
	}
	.type {
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--text-dim);
	}
	.accent {
		color: var(--accent);
	}
	.toolwrap {
		margin-top: 0.4rem;
	}
	.atts {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		margin-top: 0.4rem;
	}
	.att-img img {
		max-width: 16rem;
		max-height: 16rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		display: block;
	}
	.args {
		margin-top: 0.2rem;
		max-height: 22rem;
		overflow: auto;
	}
	details {
		margin-bottom: 0.3rem;
	}
	summary {
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 0.35rem;
		font-size: var(--fs-sm);
		list-style: none;
	}
	summary::-webkit-details-marker {
		display: none;
	}
	/* Disclosure caret (matches the trajectory view): rotates ▶ → ▼ on open. */
	.caret {
		display: inline-flex;
		flex-shrink: 0;
		color: var(--text-dim);
		transition: transform 0.12s ease;
	}
	details[open] > summary > .caret {
		transform: rotate(90deg);
	}
	summary:hover .caret {
		color: var(--text);
	}
</style>

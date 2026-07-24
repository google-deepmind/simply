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
	// Shared renderer for a tool-call / tool-result PAIR. Used by the trajectory
	// view (call + result rendered together) and the chat tool-call popup. Generic
	// by default (YAML-ish args + raw result text); a few high-value tools
	// (edit_file, bash) get a specialized layout. It renders ONLY what's in the
	// call args + result — never re-reads files, so it can't drift from the
	// captured state. The edit_file diff is parsed from the RESULT (the authoritative
	// outcome, which resolves anchors and knows the removed text), NOT the args.
	import { formatToolArgs } from '$lib/toolargs';
	import { blobUrl } from '$lib/api';
	import { toolIcon } from '$lib/toolIcon';
	import type { Attachment } from '$lib/types';

	let {
		name,
		args = '',
		// Result is optional: a call may still be in-flight (no result yet).
		result = undefined,
		// isError: the tool reported a failure (event.is_error). On error we show the
		// attempted args (there's no diff/output to show) and tint everything red.
		isError = false,
		attachments = [],
		runId = ''
	}: {
		name: string;
		args?: string;
		result?: string;
		isError?: boolean;
		attachments?: Attachment[];
		runId?: string;
	} = $props();

	const done = $derived(result !== undefined);

	// Parsed args, when valid JSON — used by the specialized renderers. Generic
	// path uses formatToolArgs (which has its own raw-string fallback).
	const parsed = $derived.by((): Record<string, unknown> | null => {
		if (!args) return null;
		try {
			const v = JSON.parse(args);
			return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
		} catch {
			return null;
		}
	});

	// --- edit_file specialization: colorize the unified diff the tool emitted in
	//     its RESULT. The tool prefixes every removed line "- ", added line "+ "
	//     and context line "  "; header lines ("Applied …", "Edit #N …") have no
	//     prefix and render plain. We classify by prefix only — no re-derivation. ---
	type ResultLine = { cls: 'add' | 'del' | 'ctx'; text: string };
	const diffResult = $derived.by((): ResultLine[] | null => {
		// On error the result is a diagnostic message, not a diff — don't colorize it.
		if (name !== 'edit_file' || isError || result === undefined) return null;
		return result.replace(/\n+$/, '').split('\n').map((line): ResultLine => {
			if (line.startsWith('+ ')) return { cls: 'add', text: line };
			if (line.startsWith('- ')) return { cls: 'del', text: line };
			return { cls: 'ctx', text: line };
		});
	});

	// edit_file target path, surfaced from the args as a header.
	const editPath = $derived(name === 'edit_file' && parsed ? String(parsed.path ?? '') : '');

	// Tool identity icon, shared with the chat pills via $lib/toolIcon.
	const Icon = $derived(toolIcon(name));

	// --- bash specialization: command from args, output from result. ---
	const bashCommand = $derived(name === 'bash' && parsed ? String(parsed.command ?? '') : '');
</script>

<div class="tc" class:running={!done} class:error={isError}>
	<div class="tc-head">
		<Icon size={14} weight="bold" class="tc-icon" />
		<span class="name mono">{name}</span>
		{#if !done}
			<span class="status running">running…</span>
		{:else if isError}
			<span class="status errored">error</span>
		{/if}
	</div>

	{#if isError}
		<!-- Error: show what was attempted (args) — there's no diff/output — plus the
		     tool's diagnostic message. -->
		{#if args}<pre class="pre small args">{formatToolArgs(args)}</pre>{/if}
		{#if result}
			<div class="result-label err small">error</div>
			<pre class="pre block result err-body">{result}</pre>
		{/if}
	{:else if diffResult}
		<!-- edit_file: colorized unified diff, parsed from the tool's result. -->
		{#if editPath}<div class="epath mono dim">{editPath}</div>{/if}
		<pre class="pre block diff">{#each diffResult as dl}<span class="dl {dl.cls}">{dl.text || ' '}</span>{/each}</pre>
	{:else if bashCommand}
		<!-- bash: command block + raw output. -->
		<pre class="pre block cmd">{bashCommand}</pre>
		{#if done && result}
			<div class="result-label dim small">output</div>
			<pre class="pre block result">{result}</pre>
		{/if}
	{:else}
		<!-- generic call: YAML-ish args + raw result. -->
		<pre class="pre small args">{formatToolArgs(args)}</pre>
		{#if done && result}
			<div class="result-label dim small">result</div>
			<pre class="pre block result">{result}</pre>
		{/if}
	{/if}

	{#if attachments.length}
		<div class="atts">
			{#each attachments as att, i (att.blob_key ?? i)}
				{#if att.blob_key && att.mime_type.startsWith('image/')}
					<a class="att-img" href={blobUrl(runId, att.blob_key)} target="_blank" rel="noopener" title={att.source_hint}>
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
</div>

<style>
	.tc {
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0.4rem 0.6rem;
		background: var(--bg);
	}
	.tc-head {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.3rem;
	}
	.name {
		color: var(--accent);
		font-size: var(--fs-md);
	}
	/* Error: tint the whole header red so a failed call is unmistakable. */
	.tc.error {
		border-color: color-mix(in srgb, var(--err) 45%, var(--border));
	}
	.tc.error .name {
		color: var(--err);
	}
	/* Tool identity icon in the head (same icon as the chat pill) — tinted to
	   match the tool name; dims while the call is still running, reddens on error. */
	:global(.tc .tc-icon) {
		color: var(--accent);
		flex-shrink: 0;
	}
	:global(.tc.running .tc-icon) {
		color: var(--text-dim);
	}
	:global(.tc.error .tc-icon) {
		color: var(--err);
	}
	.status {
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}
	.status.running {
		color: var(--text-dim);
	}
	.status.errored {
		color: var(--err);
	}
	.block {
		border: 1px solid var(--border);
		border-radius: var(--radius-xs);
		padding: 0.3rem 0.5rem;
		margin: 0.2rem 0 0;
		max-height: 22rem;
		overflow: auto;
		/* Secondary content: the detail panel/modal is supporting material, so its
		   code blocks read at the smaller "code" scale (matches chat inline code),
		   never larger than the main conversation text. */
		font-size: var(--fs-sm);
	}
	.args {
		margin-top: 0.2rem;
		max-height: 22rem;
		overflow: auto;
	}
	.result {
		background: var(--bg-elev);
	}
	.result-label {
		margin-top: 0.4rem;
	}
	.result-label.err {
		color: var(--err);
	}
	/* Error message body: a subtle red wash so it reads as a failure, not output. */
	.err-body {
		background: color-mix(in srgb, var(--err) 8%, transparent);
		border-color: color-mix(in srgb, var(--err) 35%, var(--border));
		color: var(--err);
	}
	.cmd {
		background: var(--bg-elev);
	}
	.epath {
		font-size: var(--fs-sm);
		margin: 0.2rem 0;
	}
	/* Unified diff (parsed from the tool result): context lines untinted, removed
	   red, added green. The -/+ markers are part of each line's text (the tool
	   emits them) so a copy yields a valid diff. */
	.diff {
		padding: 0.25rem 0;
	}
	.diff .dl {
		display: block;
		padding: 0 0.5rem;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.diff .del {
		color: var(--err);
		background: color-mix(in srgb, var(--err) 14%, transparent);
	}
	.diff .add {
		color: var(--ok);
		background: color-mix(in srgb, var(--ok) 14%, transparent);
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
</style>

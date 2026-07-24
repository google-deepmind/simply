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
	import { getRunStore } from '$lib/runContext.svelte';
	import SessionTree from '$lib/components/SessionTree.svelte';
	import Spinner from '$lib/components/Spinner.svelte';
	import { auth } from '$lib/auth.svelte';
	import { api, errorText } from '$lib/api';
	import { renderMarkdown } from '$lib/markdown';
	import { parseCitation, stepHref } from '$lib/citations';
	import { formatLocalIso, formatElapsed } from '$lib/time';
	import { pageTitle } from '$lib/title';
	import type { RunReport } from '$lib/types';
	import { gradeSpecByRank } from '$lib/components/GradePicker.svelte';
	import {
		CheckCircleIcon,
		XCircleIcon,
		WarningIcon,
		PaperclipIcon
	} from 'phosphor-svelte';

	const store = getRunStore();
	const detail = $derived(store.detail);

	// Run topology drives all the report-card affordances. The decisive question
	// is "is there a main (autonomous) agent root?":
	//   - yes → reports auto-generate when it concludes, so there's NO Generate
	//          Report button; instead, when it's concluded, we offer a follow-up
	//          textarea that sends a new user message to that main-agent
	//          session, which respawns it for the next iteration.
	//   - no  → chat-only run; reports never auto-generate, so we show the
	//          Generate Report button (the explicit, manual trigger). No
	//          follow-up — sending messages to a chatbot is the chat tab's job.
	const mainAgentSession = $derived(
		detail?.sessions.find((s) => !s.parent_id && s.agent_type !== 'chatbot')
	);
	const hasAutonomousRoot = $derived(!!mainAgentSession);
	const mainAgentConcluded = $derived(mainAgentSession?.status === 'concluded');
	// Default session ID for citation step refs that omit the session prefix.
	// Prefer the autonomous main agent (where reports typically come from);
	// fall back to the first root for chat-only runs whose manually-generated
	// reports would otherwise have nowhere to point.
	const primarySessionId = $derived(
		mainAgentSession?.session_id ??
			detail?.sessions.find((s) => !s.parent_id)?.session_id ??
			''
	);

	// Run report(s) — auto-loaded on the Overview. `latest` is the most recent
	// iteration (always present when reports.length > 0). `displayed` reflects
	// the user's tab selection; null means "view latest", letting it track new
	// iterations as they arrive without the user needing to re-click.
	let reports = $state<RunReport[]>([]);
	let selectedVersion = $state<number | null>(null);
	let generating = $state(false);
	let reportErr = $state('');
	// deferredNotice: transient copy shown after a manual Generate that the
	// backend deferred (delta below threshold; previous report unchanged).
	// Chat runs get this since their Generate button is always visible;
	// autonomous runs mostly never reach it (offerGenerate hides the button on
	// trivial_gap), but the notice still applies if a race lets a click land.
	let deferredNotice = $state('');
	const latest = $derived(reports.length ? reports[reports.length - 1] : null);
	const hasReport = $derived(latest !== null);
	const displayed = $derived.by(() => {
		if (!reports.length) return null;
		if (selectedVersion === null) return latest;
		return reports.find((r) => r.version === selectedVersion) ?? latest;
	});
	const isViewingLatest = $derived(
		selectedVersion === null || (latest != null && selectedVersion === latest.version)
	);

	// Ephemeral (non-session) report generator — registered by the critic
	// finalizer for both auto and manual triggers. Drives the "Generating
	// report…" indicator. Re-derives reactively as ephemeral_agents flows
	// through SSE → store.refresh → detail mutation.
	const ephemeralReport = $derived(
		detail?.ephemeral_agents?.find((a) => a.kind === 'report') ?? null
	);
	const ephemeralReportRunning = $derived(ephemeralReport !== null);

	// Report coverage from the server (mirrors critic.ReportSkipMinSteps). The
	// three states let the UI distinguish "a new iteration is coming"
	// (substantive_gap) from "the framework saw a small delta and deliberately
	// declined to regenerate" (trivial_gap) — without the second, a silent
	// finalizer skip leaves the spinner running forever.
	//
	// Coverage is empty for chat runs (no main-agent) and pre-first-report
	// runs; falling back to the legacy per-report watermark check keeps those
	// paths working unchanged.
	const reportCoverage = $derived(detail?.report_coverage ?? '');
	const reportGapSteps = $derived(detail?.report_gap_steps ?? 0);

	// Staleness (visible "regenerate to bring it up to date" banner). For
	// autonomous runs the server-side coverage is authoritative; a
	// trivial_gap suppresses the banner (we're deliberately not regenerating)
	// so only substantive_gap shows as stale. For chat runs the server does
	// not compute coverage, so fall back to the legacy per-report watermark
	// check (trust-by-default when watermarks are missing).
	const mainAgentReportStep = $derived.by(() => {
		if (!latest) return null;
		const sessionState = latest.sessions?.find((s) => s.session_id === 'main-agent');
		return sessionState?.current_step ?? null;
	});
	const reportStale = $derived(
		hasAutonomousRoot
			? mainAgentConcluded && hasReport && reportCoverage === 'substantive_gap'
			: mainAgentConcluded &&
					hasReport &&
					mainAgentReportStep !== null &&
					(mainAgentSession?.current_step ?? 0) > mainAgentReportStep
	);

	// "Pending generation": autonomous run, main agent concluded, the latest
	// report doesn't cover the latest conclusion, but the critic hasn't
	// registered its ephemeral yet. This is the 15-50s (sometimes minutes)
	// window during which the observer runs catchUpSteps + closePhases —
	// each fires an LLM call (fast-tier for step summaries, HQ-tier for the
	// force-closed tail phase) BEFORE maybeFinalize spawns the critic
	// goroutine.
	//
	// Gated on report_coverage === 'substantive_gap' so a trivial_gap does
	// NOT enter the pending state — the finalizer will skip such deltas
	// (below the debounce threshold), and staying in "pending" indefinitely
	// would leave the spinner running forever.
	//
	// No time-based escape from this state. The ephemeral lifecycle is
	// authoritative: serve.go wraps the critic goroutine with panic
	// recovery, and the Finalizer's defer Unregister runs on panic, so the
	// ephemeral always disappears when work ends. On full server crash the
	// in-memory registry restarts empty and serve.go's backfillReports
	// re-triggers the critic. So an "eternal spinner" only happens if the
	// observer pipeline itself never reaches maybeFinalize, which is rare
	// — and a manual page refresh + [Generate] from a fresh state recovers.
	const reportPending = $derived(
		hasAutonomousRoot &&
			mainAgentConcluded &&
			!ephemeralReportRunning &&
			(!hasReport || reportCoverage === 'substantive_gap')
	);
	// One source-of-truth "report work in progress" — covers both the
	// pre-ephemeral pending window and the actual ephemeral-running window.
	// Drives the spinner banner, the now-ticker, and the Generate-button
	// suppression.
	const reportInProgress = $derived(ephemeralReportRunning || reportPending);

	// `now` ticks every second WHILE the spinner is showing so the elapsed
	// time updates live. Stopped otherwise (no time-based state to
	// re-evaluate now that the grace window is gone).
	let now = $state(Date.now());
	$effect(() => {
		if (!reportInProgress) return;
		const t = setInterval(() => (now = Date.now()), 1000);
		return () => clearInterval(t);
	});

	// Elapsed-time anchor: LATCHED at the start of reportInProgress so the
	// spinner shows continuous elapsed time across the pending→ephemeral
	// transition. Without this, the timer visibly resets when the
	// ephemeral registers (because its started_at is later than the
	// conclude). Capture once on the rising edge, reset on the falling
	// edge. Next iteration re-latches naturally (back to null after the
	// run, captured again when main agent concludes the next iteration).
	let workStartedAt = $state<string | null>(null);
	$effect(() => {
		if (reportInProgress) {
			if (workStartedAt === null) {
				// Prefer the ephemeral's start when it's already running (manual
				// Regenerate after stale failure: the wait clock starts at
				// "they clicked"). Fall back to the conclude time when we're
				// in the pending window (the wait clock starts at conclude).
				workStartedAt =
					ephemeralReport?.started_at ?? mainAgentSession?.status_changed_at ?? null;
			}
		} else {
			workStartedAt = null;
		}
	});
	const reportElapsed = $derived.by(() => (workStartedAt ? formatElapsed(workStartedAt, now) : ''));

	// Combined "is there ANY active generation right now?" — the manual-click
	// path sets `generating` synchronously around the HTTP call (which can be
	// slow even before the ephemeral notification arrives), the auto path
	// only sets ephemeralReportRunning, and reportPending covers the
	// pre-ephemeral observer-summarization window. UI buttons disable on any.
	const reportGenerating = $derived(generating || reportInProgress);

	// Whether to offer a generate / regenerate button. Chat-only runs always
	// show one (no auto-trigger anyway). Autonomous runs hide it whenever
	// report work is in progress (including the pending window — clicking
	// Generate during pending would just serialize behind the auto-trigger
	// via the per-run lock, so the button is misleading at best) and also
	// when coverage is a trivial_gap: a manual Regenerate on such a delta
	// would just defer server-side too, so hiding the button matches the
	// truth. (chat runs don't get report_coverage, so this term is a no-op
	// there and the button stays always-visible.)
	const offerGenerate = $derived(
		!reportGenerating &&
			(!hasAutonomousRoot || (mainAgentConcluded && (!hasReport || reportStale)))
	);

	// Follow-up: send a new user message to the main-agent session. The
	// existing /sessions/{sid}/message endpoint handles the respawn; main
	// agent re-enters ongoing and emits a new report when it next concludes.
	let followupText = $state('');
	let sending = $state(false);
	let sendErr = $state('');
	// Auto-draft a follow-up prompt from the run's task + latest report.
	let suggesting = $state(false);

	$effect(() => {
		const id = store.runId;
		if (id) loadReports(id);
	});

	// Re-fetch reports on the FALLING edge of ephemeralReportRunning — a
	// generation just finished (successfully or not), so the report list may
	// have a new iteration. We use ephemeralReportRunning specifically (not
	// reportInProgress) because we DON'T want to re-fetch when transitioning
	// out of the pending window into ephemeral-running (no report yet);
	// only the ephemeral end signals "report written or attempt complete".
	let wasRunning = $state(false);
	$effect(() => {
		const id = store.runId;
		const running = ephemeralReportRunning;
		if (wasRunning && !running && id) {
			void loadReports(id);
		}
		wasRunning = running;
	});

	async function loadReports(id: string) {
		try {
			reports = await api.getReports(id);
		} catch {
			reports = []; // non-fatal: just offer the generate button (chat) or wait (autonomous)
		}
	}

	async function genReport() {
		generating = true;
		reportErr = '';
		deferredNotice = '';
		try {
			const { deferred } = await api.startReport(store.runId);
			reports = await api.getReports(store.runId);
			if (deferred) {
				deferredNotice = 'No new report iteration — the delta since the previous report is below the new-report threshold.';
			}
		} catch (e) {
			reportErr = String(e);
		} finally {
			generating = false;
		}
	}

	// Generate a follow-up draft and fill the textarea. Warns before clobbering a
	// dirty draft (the operator's own typing wins unless they confirm).
	async function suggestFollowup() {
		if (suggesting) return;
		if (followupText.trim() && !confirm('Replace your current draft with a generated suggestion?')) {
			return;
		}
		suggesting = true;
		sendErr = '';
		try {
			const { prompt } = await api.suggestFollowup(store.runId);
			followupText = prompt;
		} catch (e) {
			sendErr = errorText(e); // '' when unreachable (global banner covers it)
		} finally {
			suggesting = false;
		}
	}

	async function sendFollowup() {
		const sess = mainAgentSession;
		const text = followupText.trim();
		if (!sess || !text || sending) return;
		sending = true;
		sendErr = '';
		try {
			await api.sendMessage(store.runId, sess.session_id, text);
			followupText = '';
			// Snap selection back to "latest" so the next auto-generated report
			// shows up as the visible iteration without the user having to click.
			selectedVersion = null;
			// Refresh: main agent's status will flip back to ongoing, the
			// follow-up section will hide accordingly.
			await store.refresh();
			await loadReports(store.runId);
		} catch (e) {
			sendErr = String(e);
		} finally {
			sending = false;
		}
	}

	// Tab title: the user's chosen title wins (most-meaningful identifier);
	// fall back to the task, then the immutable run id. detail is reactive
	// so renames and the late-loading initial fetch both update the tab live.
	const tabLabel = $derived(detail?.title || detail?.task || detail?.run_id);
</script>

<svelte:head>
	<title>{pageTitle(tabLabel)}</title>
</svelte:head>

<!-- Renders a citation list: each entry is either a plain text token (an id,
     file path, etc.) OR a parseable step-reference, in which case
     it becomes a link to the session page with ?step=N (or N-M), which the
     session page reads to expand+scroll to the matching step(s). The visible
     link text is the LLM's original citation string (preserves wording). -->
{#snippet renderCitations(citations: string[])}
	<span class="cite mono">
		{#each citations as cit, i}
			{#if i > 0}, {/if}
			{@const parsed = parseCitation(cit, primarySessionId)}
			{@const href = stepHref(store.runId, parsed)}
			{#if href}
				<a href={href}>{cit}</a>
			{:else}
				{cit}
			{/if}
		{/each}
	</span>
{/snippet}

{#if detail}
<div class="overview-stack">
	<!-- Persisted run config: shown in full (no truncation) — long values like
	     workspace paths and subprocess: LLM specs wrap rather than ellipsizing.
	     The header card above this surfaces live status/step/time; this card
	     is the immutable-at-creation knobs the user might want to copy or
	     reference (Run ID especially). -->
	<section class="card meta">
		<dl class="kv">
			<dt>run id</dt><dd class="mono">{detail.run_id}</dd>
			<dt>workspace</dt><dd class="mono">{detail.workspace || '—'}</dd>
			<dt>model</dt><dd class="mono">{detail.llm || '—'}</dd>
			<dt>agent type</dt><dd class="mono">{detail.agent_type || '—'}</dd>
			<dt>system (hq)</dt><dd class="mono">{detail.system_llm_hq || '—'}</dd>
			<dt>system (fast)</dt><dd class="mono">{detail.system_llm_fast || '—'}</dd>
			<dt>created</dt><dd class="mono">{formatLocalIso(detail.created_at)}</dd>
			{#if detail.updated_at && detail.updated_at !== detail.created_at}
				<dt title="Last metadata edit (rename / archive / star / note)">last edit</dt>
				<dd class="mono">{formatLocalIso(detail.updated_at)}</dd>
			{/if}
		</dl>

		{#if detail.agents_md}
			<!-- Operator-supplied AGENTS.md context, persisted with the run.
			     Collapsible — most viewers don't need it unless they're auditing
			     the run's prompt environment. -->
			<details class="agents-md-wrap">
				<summary>
					AGENTS.md context <span class="dim">— {detail.agents_md.length} chars</span>
				</summary>
				<pre class="agents-md mono">{detail.agents_md}</pre>
			</details>
		{/if}

		{#if detail.note}<p class="note dim">{detail.note}</p>{/if}
	</section>

	<!-- Two-column body: task on the left, latest report on the right. The
	     grid collapses to a single column on narrow viewports OR when the run
	     has no task (interactive chat runs); the report takes the full width
	     in that case rather than half-empty next to a placeholder. -->
	<div class="overview-grid" class:single={!detail.task}>
		{#if detail.task}
			<section class="card task-block">
				<h3>Task</h3>
				<div class="md">{@html renderMarkdown(detail.task)}</div>
			</section>
		{/if}

		<section class="card report-block">
			<div class="report-head">
				<h3>Run report</h3>
				<div class="head-actions">
					<!-- Iteration tabs: only when there's more than one. Clicking a
					     non-latest tab freezes the view; the latest tab restores the
					     "track newest" behavior (null selection). -->
					{#if reports.length > 1}
						<div class="iteration-tabs" role="tablist">
							{#each reports as r (r.version)}
								<button
									type="button"
									class="tab"
									class:on={displayed?.version === r.version}
									title="Iteration {r.version} — {new Date(r.created_at).toLocaleString()}"
									onclick={() => (selectedVersion = r === latest ? null : r.version)}
								>
									v{r.version}{r === latest ? ' · latest' : ''}
								</button>
							{/each}
						</div>
					{/if}
					<!-- Generate / Regenerate: shown when the run is in a state where
					     a NEW report would be meaningful and no generation is in
					     flight. Chat-only: always. Autonomous: only when the main
					     agent is concluded AND (no report yet OR the latest is
					     stale relative to the current main-agent step). -->
					{#if offerGenerate && auth.authed}
						<button class="primary" onclick={genReport} disabled={reportGenerating}>
							{hasReport && reportStale ? 'Regenerate report' : 'Generate report'}
						</button>
					{/if}
				</div>
			</div>

			{#if reportErr}
				<p class="err">{reportErr}</p>
			{/if}
			{#if deferredNotice}
				<p class="status-banner deferred dim">{deferredNotice}</p>
			{/if}

			<!-- Status banner above the report body. At most one applies at a
			     time. Precedence (most-actionable first):
			       1. reportInProgress: catches both the pending window (after
			          conclude, before critic registers — the observer is
			          still summarizing) AND the ephemeral-running window.
			          Same spinner, same text — from the user's perspective
			          "summarizing" and "actually generating" are indistinct
			          background work, both gate the report's arrival.
			       2. stale: report exists but doesn't cover the latest
			          conclusion AND nothing is running to fix it (post-failure).
			       3. followup: agent revived for a follow-up; previous report
			          on display is honestly labelled.
			       4. trivial_gap (autonomous, at rest): the main-agent added a
			          handful of steps since the last report but not enough to
			          cross the debounce threshold; no new iteration will be
			          generated automatically. Shown as a quiet informational
			          line rather than an action-required banner. -->
			{#if reportInProgress}
				<p class="status-banner generating">
					<Spinner />
					{hasReport
						? `Generating new iteration… (${reportElapsed})`
						: `Generating report… (${reportElapsed})`}
				</p>
			{:else if reportStale}
				<p class="status-banner stale">
					Report below covers an earlier conclusion (main agent step {mainAgentReportStep}, now at step {mainAgentSession?.current_step}).
					Regenerate to bring it up to date.
				</p>
			{:else if hasAutonomousRoot && !mainAgentConcluded && hasReport}
				<p class="status-banner followup">
					Agent is working on a follow-up. The report below covers an earlier iteration.
				</p>
			{:else if hasAutonomousRoot && mainAgentConcluded && hasReport && reportCoverage === 'trivial_gap'}
				<p class="status-banner trivial-gap dim">
					Only {reportGapSteps} step{reportGapSteps === 1 ? '' : 's'} since the last report, below the new-report threshold.
				</p>
			{/if}

			{#if displayed}
				<!-- The critic's grade for THIS iteration. Read-only here (the editable
				     verdict lives on the run card's GradePicker); shown with duotone
				     weight to match the "critic" rendering elsewhere. undefined (rank 0)
				     when the report predates grading. {@const} must be the immediate
				     child of a block, so it sits here at the top of the {#if}. -->
				{@const gs = gradeSpecByRank(displayed.grade)}
				<div class="report">
					<p class="dim ver">
						iteration {displayed.version} · {new Date(displayed.created_at).toLocaleString()}
						{#if gs}
							<span class="report-grade" style="color:{gs.color}" title="Critic grade: {gs.label}">
								· <gs.icon size={15} weight="duotone" />
								<span class="grade-label">{gs.label}</span>
							</span>
						{/if}
					</p>
					<div class="md summary">{@html renderMarkdown(displayed.summary)}</div>

					<!-- Each subsection is a <details> collapsed by default. The count
					     in the summary lets you triage what's worth opening at a glance
					     (e.g. "Key achievements 12" vs "Failure modes 0") without
					     having to expand each one. -->
					{#if displayed.key_achievements?.length}
						<details class="report-section achievements">
							<summary>
								<CheckCircleIcon size={15} weight="bold" />
								<span class="label">Key achievements</span>
								<span class="count">{displayed.key_achievements.length}</span>
							</summary>
							<ul class="claims">
								{#each displayed.key_achievements as c}
									<li>
										<span class="claim-statement">{c.statement}</span>
										{#if c.citations?.length}
											{@render renderCitations(c.citations)}
										{/if}
									</li>
								{/each}
							</ul>
						</details>
					{/if}

					{#if displayed.failure_modes?.length}
						<details class="report-section failures">
							<summary>
								<XCircleIcon size={15} weight="bold" />
								<span class="label">Failure modes</span>
								<span class="count">{displayed.failure_modes.length}</span>
							</summary>
							<ul class="claims">
								{#each displayed.failure_modes as c}
									<li>
										<span class="claim-statement">{c.statement}</span>
										{#if c.citations?.length}
											{@render renderCitations(c.citations)}
										{/if}
									</li>
								{/each}
							</ul>
						</details>
					{/if}

					{#if displayed.artifacts_by_kind && Object.keys(displayed.artifacts_by_kind).length}
						{@const artifactCount = Object.values(displayed.artifacts_by_kind).reduce((n, a) => n + a.length, 0)}
						<details class="report-section artifacts">
							<summary>
								<PaperclipIcon size={15} weight="bold" />
								<span class="label">Artifacts</span>
								<span class="count">{artifactCount}</span>
							</summary>
							{#each Object.entries(displayed.artifacts_by_kind) as [kind, arts]}
								<div class="artifact-group">
									<span class="artifact-kind mono">{kind}</span>
									<ul class="claims">
										{#each arts as a}
											{@const stepSuffix = a.start_step === a.end_step
												? `${a.start_step}`
												: `${a.start_step}-${a.end_step}`}
											{@const rangeLabel = a.start_step === a.end_step
												? `step ${a.start_step}`
												: `steps ${a.start_step}-${a.end_step}`}
											<li>
												<span class="claim-statement mono">{a.value}</span>
												{#if a.context}
													<span class="cite">{a.context}</span>
												{/if}
												<!-- Provenance link: where in the trajectory this
												     artifact came from. Same href shape as citation
												     links — the session page deep-links by ?step=. -->
												<a
													class="cite artifact-loc"
													href="/runs/{store.runId}/sessions/{encodeURIComponent(
														a.session_id
													)}?step={stepSuffix}"
												>
													{a.session_id} {rangeLabel}
												</a>
											</li>
										{/each}
									</ul>
								</div>
							{/each}
						</details>
					{/if}

					{#if displayed.struggles?.length}
						<details class="report-section struggles">
							<summary>
								<WarningIcon size={15} weight="bold" />
								<span class="label">Struggle ranges</span>
								<span class="count">{displayed.struggles.length}</span>
							</summary>
							<ul class="claims">
								{#each displayed.struggles as s}
									{@const rangeSuffix = s.start_step === s.end_step
										? `${s.start_step}`
										: `${s.start_step}-${s.end_step}`}
									<li>
										<a
											class="claim-statement mono"
											href="/runs/{store.runId}/sessions/{encodeURIComponent(
												s.session_id
											)}?step={rangeSuffix}"
										>
											{s.session_id}
											{s.start_step === s.end_step
												? `step ${s.start_step}`
												: `steps ${s.start_step}-${s.end_step}`}
										</a>
										{#if s.sample_summaries?.length}
											<!-- One bullet per sample summary (up to 3 captured at
											     observation time). These ARE the most useful bit —
											     without them the entry is just step numbers with no
											     context about what went wrong. -->
											<ul class="samples">
												{#each s.sample_summaries as sample}
													<li class="cite">{sample}</li>
												{/each}
											</ul>
										{/if}
									</li>
								{/each}
							</ul>
						</details>
					{/if}
				</div>
			{:else if !reportGenerating}
				{#if hasAutonomousRoot && !mainAgentConcluded}
					<p class="dim">
						Main agent is working — a report will be generated automatically when it concludes.
					</p>
				{:else if hasAutonomousRoot && mainAgentConcluded}
					<!-- Should be unreachable in normal autonomous flow: after
					     conclude, reportPending → reportInProgress is true, so
					     the spinner banner above takes the slot. Reaching here
					     means we're explicitly "not generating" AND have no
					     report — i.e. an auto-trigger that didn't fire (rare,
					     e.g. main agent concluded before the registry's
					     onChange hook was wired). The [Generate report] button
					     in the head-actions strip handles the recovery. -->
					<p class="dim">No report yet.</p>
				{:else}
					<p class="dim">No report yet — click Generate report to create one.</p>
				{/if}
			{/if}

			<!-- Follow-up: autonomous + main agent concluded + viewing latest. Hidden
			     mid-flight (sending would queue and confuse "is the agent acting on
			     this now or later?") and hidden when an older iteration is being
			     viewed (the actionable surface is the LATEST report). -->
			{#if hasAutonomousRoot && mainAgentConcluded && isViewingLatest && auth.authed}
				<div class="followup">
					<h4>Send follow-up to main agent</h4>
					<p class="dim followup-hint">
						Starts a new iteration. The main agent receives this as a user message and
						respawns to act on it.
					</p>
					<textarea
						bind:value={followupText}
						placeholder="What should the main agent do next?"
						rows="3"
						onkeydown={(e) => {
							if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
								e.preventDefault();
								sendFollowup();
							}
						}}
					></textarea>
					{#if sendErr}<p class="err">{sendErr}</p>{/if}
					<div class="followup-actions">
						<button class="gen-btn" onclick={suggestFollowup} disabled={suggesting || sending}>
							{#if suggesting}<Spinner />{/if}
							{suggesting ? 'Generating…' : 'Generate follow-up prompt'}
						</button>
						<button
							class="primary"
							onclick={sendFollowup}
							disabled={sending || suggesting || !followupText.trim()}
						>
							{sending ? 'Sending…' : 'Send follow-up'}
						</button>
					</div>
				</div>
			{/if}
		</section>
	</div>

	<section>
		<h3>Sessions</h3>
		<SessionTree runId={store.runId} sessions={detail.sessions} />
	</section>
</div>
{:else if !store.error}
	<p class="dim">Loading…</p>
{/if}

<style>
	/* Outer stack: gap between the top-level sections (meta card, overview
	   grid, sessions). Replaces ad-hoc top-margins. */
	.overview-stack {
		display: flex;
		flex-direction: column;
		gap: 0.9rem;
	}
	/* Metadata key/value table: two PAIR columns on wide screens (label1
	   value1 | label2 value2), single column on narrow. Each value renders
	   in FULL (no truncation) — long paths and subprocess: LLM specs wrap
	   inside their cell instead of getting cut off. */
	.kv {
		display: grid;
		grid-template-columns: auto 1fr auto 1fr;
		gap: 0.5rem 1.4rem;
		margin: 0;
		font-size: var(--fs-md);
	}
	@media (max-width: 900px) {
		.kv {
			grid-template-columns: auto 1fr;
		}
	}
	.kv dt {
		color: var(--text-dim);
	}
	.kv dd {
		margin: 0;
		min-width: 0;
		overflow-wrap: anywhere;
	}
	/* Two-column body grid: task | run report. 1:2 ratio since the task is
	   usually a brief brief while the report is dense (summary + lists).
	   Collapses to one column under 900px OR when there's no task (chat
	   runs); the report takes full width in those cases.
	   
	   minmax(0, Nfr) — NOT bare Nfr — because bare fr has an implicit
	   min-width of "auto" (= the cell's content min-width). When the task or
	   report contains unbreakable text (long URLs, code blocks, file paths),
	   bare fr refuses to shrink below that content width and the proportions
	   collapse toward 1:1 instead of 1:2. minmax(0, ...) lets the cell shrink
	   to 0 and wrap content inside rather than stealing column width. */
	.overview-grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
		gap: 0.8rem;
	}
	.overview-grid.single {
		grid-template-columns: minmax(0, 1fr);
	}
	@media (max-width: 900px) {
		.overview-grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.task-block h3,
	.report-block h3 {
		margin: 0 0 0.6rem;
	}
	/* AGENTS.md collapsible. Kept monospace + scrollable to handle long
	   bodies without breaking the rest of the page. */
	.agents-md-wrap {
		margin-top: 0.9rem;
		font-size: var(--fs-md);
	}
	.agents-md-wrap summary {
		cursor: pointer;
		color: var(--text-dim);
		padding: 0.3rem 0;
	}
	.agents-md-wrap summary:hover {
		color: var(--text);
	}
	.agents-md {
		margin: 0.5rem 0 0;
		padding: 0.7rem 0.9rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		font-size: var(--fs-sm);
		max-height: 24rem;
		overflow: auto;
		white-space: pre-wrap;
		word-break: break-word;
	}
	.note {
		margin: 0.9rem 0 0;
		white-space: pre-wrap;
	}
	h3 {
		margin: 0 0 0.6rem;
	}
	/* Header lays out as: [h3]  ………  [tabs + button cluster].
	   margin-left:auto on the cluster pushes it right while h3 stays anchored
	   to the left, and gap keeps tabs + button visually separated when both
	   are present. flex-wrap allows the cluster to drop below the h3 on
	   narrow viewports rather than overflowing. */
	.report-head {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
		/* Space before whatever comes next (status banner, report body, or
		   empty placeholder). Without this the banner/body sits flush
		   against the h3 baseline and the section reads as cramped. */
		margin-bottom: 0.9rem;
	}
	.report-head h3 {
		margin: 0;
	}
	.report-head .head-actions {
		margin-left: auto;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	/* Iteration tabs — compact pills matching the workspace/llm chip aesthetic.
	   The "on" tab gets accent-tinted bg + text so the current selection is
	   obvious; others stay quiet. */
	.iteration-tabs {
		display: flex;
		gap: 0.2rem;
	}
	.iteration-tabs .tab {
		font: inherit;
		font-size: var(--fs-sm);
		padding: 0.18rem 0.55rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
		white-space: nowrap;
	}
	.iteration-tabs .tab:hover:not(.on) {
		color: var(--text);
	}
	.iteration-tabs .tab.on {
		color: var(--text);
		border-color: var(--accent-dim);
		background: color-mix(in srgb, var(--accent) 12%, var(--bg-elev));
	}
	/* Status banners above the report body. One banner at a time (mutually
	   exclusive by the template's branches). All share the same layout
	   chrome — soft tinted background + accent border — and differ only in
	   tint so the eye reads them as a single semantic slot.
	   Padding is intentionally larger than ".55rem" because <p>-as-flex
	   with multi-line text has visibly uneven top/bottom half-leading from
	   font metrics; .75rem padding + 1.55 line-height makes that asymmetry
	   imperceptible and gives the banner a calm "card" feel that matches
	   the surrounding chips. */
	.status-banner {
		margin: 0 0 0.9rem 0;
		padding: 0.75rem 0.9rem;
		border: 1px solid var(--border);
		border-left-width: 3px;
		border-radius: var(--radius-sm);
		font-size: var(--fs-sm);
		line-height: 1.55;
		display: flex;
		align-items: center;
		gap: 0.55rem;
	}
	.status-banner.generating {
		border-left-color: var(--accent);
		background: color-mix(in srgb, var(--accent) 6%, var(--bg-elev));
	}
	.status-banner.stale {
		border-left-color: var(--warn);
		background: color-mix(in srgb, var(--warn) 6%, var(--bg-elev));
	}
	.status-banner.followup {
		border-left-color: var(--text-dim);
		background: var(--bg-elev);
		color: var(--text-dim);
	}
	/* Follow-up section: visually distinct from the report body via a top
	   border, and the textarea + send button are clearly grouped. */
	.followup {
		margin-top: 1.2rem;
		padding-top: 1rem;
		border-top: 1px solid var(--border);
	}
	.followup h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.followup-hint {
		margin: 0.2rem 0 0.5rem;
		font-size: var(--fs-sm);
	}
	.followup textarea {
		min-height: 4.5rem;
		font: inherit;
	}
	.followup-actions {
		margin-top: 0.6rem;
		display: flex;
		justify-content: flex-end;
		gap: 0.5rem;
	}
	/* Keep the spinner and label on one baseline-centered row. */
	.gen-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
	}
	.report .ver {
		margin: 0.4rem 0 0.7rem;
		font-size: var(--fs-sm);
	}
	/* The critic's per-iteration grade, inlined after the timestamp. The icon
	   color carries the rank tint (set inline); the label inherits it. Aligned
	   to the text baseline so the smiley sits inline with "iteration N · …". */
	.report .ver .report-grade {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
		margin-left: 0.15rem;
		vertical-align: text-bottom;
	}
	.report .ver .report-grade .grade-label {
		font-weight: 500;
	}
	.report .summary {
		margin-bottom: 0.5rem;
	}
	/* Each report subsection is a <details> collapsed by default. The summary
	   line is the heading (icon + label + count); clicking it expands the
	   list. Top border separates each section visually so they read as
	   distinct categories even when all collapsed. */
	.report-section {
		margin-top: 1rem;
		padding-top: 0.7rem;
		border-top: 1px solid var(--border);
	}
	.report-section summary {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		cursor: pointer;
		font-size: var(--fs-md);
		font-weight: 600;
		list-style: none; /* hide the default disclosure triangle (we get cleaner spacing without it) */
		user-select: none;
	}
	.report-section summary::-webkit-details-marker {
		display: none; /* Safari counterpart of list-style: none */
	}
	.report-section summary .label {
		flex: 0 0 auto;
	}
	/* Count badge: small mono pill against the section's own color. Lets
	   you triage "what's worth opening" at a glance without expanding. */
	.report-section summary .count {
		font-family: var(--mono);
		font-size: var(--fs-sm);
		font-weight: 400;
		padding: 0.05rem 0.45rem;
		border-radius: var(--radius-pill);
		background: color-mix(in srgb, currentColor 12%, transparent);
		border: 1px solid color-mix(in srgb, currentColor 30%, transparent);
	}
	/* Semantic tinting per section: positive (green), negative (red), info
	   (accent), cautionary (orange). The icon AND label share the color so
	   the cluster reads as one "this is good / bad / etc." chip. The count
	   badge picks up currentColor automatically (mix with transparent). */
	.report-section.achievements summary {
		color: var(--ok);
	}
	.report-section.failures summary {
		color: var(--err);
	}
	.report-section.artifacts summary {
		color: var(--accent);
	}
	.report-section.struggles summary {
		color: var(--warn);
	}
	/* Body content (the lists) shouldn't inherit the colored heading —
	   reset to normal text color once expanded. */
	.report-section[open] > :not(summary) {
		color: var(--text);
		margin-top: 0.4rem;
	}
	/* Nested samples list under each struggle: indented, dim — these are
	   the actual context (what the agent was struggling with) and their
	   visual weight should be secondary to the step-range line. */
	.samples {
		margin: 0.25rem 0 0;
		padding-left: 1.2rem;
		list-style: disc;
	}
	.samples li {
		margin: 0.15rem 0;
	}
	.claims {
		margin: 0;
		padding-left: 1.4rem;
		font-size: var(--fs-md);
	}
	.claims li {
		margin: 0.3rem 0;
	}
	.claim-statement {
		display: inline;
	}
	/* Citations live on a NEW line, dim, mono — keeps the statement readable
	   while citation noise stays available but visually deprioritized. */
	.cite {
		display: block;
		margin-top: 0.15rem;
		font-size: var(--fs-sm);
		color: var(--text-dim);
	}
	/* Citation links: inherit the dim color so the link doesn't visually shout
	   over its claim, but gain accent on hover so it's discoverable as
	   navigable. Suppress the global a:hover underline (citation chunks are
	   short identifiers, not prose links — keep them clean).
	   
	   Two forms exist:
	     `.cite a`   — anchor INSIDE a .cite span (citation refs in claims).
	     `a.cite`    — the anchor itself IS the .cite (artifact provenance,
	                   where we don't want the wrapping span). Both need the
	                   same color/hover behavior; without `a.cite` the global
	                   a:hover underline rule wins, which is the inconsistency
	                   you noticed on artifact links. */
	.cite a,
	a.cite {
		color: inherit;
		text-decoration: none;
	}
	.cite a:hover,
	a.cite:hover {
		color: var(--accent);
		text-decoration: none;
	}
	/* Struggle-range link uses the claim-statement class; the same hover
	   override applies (.claim-statement is the element). */
	a.claim-statement {
		color: inherit;
		text-decoration: none;
	}
	a.claim-statement:hover {
		color: var(--accent);
		text-decoration: none;
	}
	/* Artifact provenance link sits inline (.cite is normally display:block);
	   restore inline so it sits to the right of the value/context on the same
	   line. font-family mono matches the chip aesthetic; the existing .cite
	   selectors above already handle color + hover. */
	.cite.artifact-loc {
		display: inline;
		margin-left: 0.5rem;
		font-family: var(--mono);
	}
	/* Artifacts are grouped by kind; the kind label sits as a small mono
	   chip above its list, so kinds with many entries read as cohesive
	   buckets rather than one long bulleted soup. */
	.artifact-group {
		margin-top: 0.5rem;
	}
	.artifact-group:first-child {
		margin-top: 0;
	}
	.artifact-kind {
		display: inline-block;
		font-size: var(--fs-xs);
		color: var(--text-dim);
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		padding: 0.05rem 0.45rem;
		margin-bottom: 0.2rem;
	}
	.err {
		font-size: var(--fs-md);
		white-space: pre-wrap;
	}
</style>

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

import { serverHealth } from './serverHealth.svelte';
import type { SysStat } from './sysstat.svelte';
import type {
	RunSummary,
	RunsPage,
	RunCounts,
	RunDetail,
	EventDTO,
	ObservationDTO,
	ModelMenu,
	ChatFeed,
	Trajectory,
	RunReport,
	RecallResults,
	RecallItem,
	LessonSummary,
	WorkspaceInfo,
	ArtifactListing,
	ArtifactAllListing,
	AboutInfo,
	TestLLMResult
} from './types';

// blobUrl builds a same-origin URL for a run's tool-result blob (e.g. an image),
// usable directly in <img src>. These are open reads, and the browser sends the
// auth cookie automatically anyway — no token in the URL.
export function blobUrl(runId: string, key: string): string {
	return `/api/runs/${runId}/blobs/${key}`;
}

// artifactRawUrl builds a same-origin URL for a file under a run's artifact dir
// (works in <img>/<a>/fetch). path is relative to the artifact root.
export function artifactRawUrl(runId: string, path: string): string {
	return `/api/runs/${runId}/artifacts/raw?path=${encodeURIComponent(path)}`;
}

// Transient conditions worth retrying on an idempotent read: a fetch that threw
// (no HTTP response reached us), or a gateway/proxy status. These are infra/
// network blips (e.g. an UberProxy AuthController timeout) rather than an app
// error, so a brief retry usually succeeds.
const GATEWAY_STATUSES = new Set([502, 503, 504]);

// HttpError preserves the status so callers/retry logic can branch on it; its
// message is the friendly text shown to the operator.
class HttpError extends Error {
	status: number;
	constructor(status: number, message: string) {
		super(message);
		this.name = 'HttpError';
		this.status = status;
	}
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// reqOnce performs a single fetch+parse. On a non-2xx it throws an HttpError; on
// a transport failure it rethrows the native TypeError. Two orthogonal flags are
// stamped on thrown errors:
//   - unreachable: the server returned NO response (fetch threw). This is the
//     only case the global "Server unreachable" banner owns, so callers suppress
//     their own local error banner for it (see isUnreachable).
//   - isProxyError: the failure looks like a transient infra/proxy blip worth
//     retrying on an idempotent read (transport throw, gateway status, or a 5xx
//     whose body isn't our JSON {error}). A proxy 5xx is reachable-but-failing,
//     so it is isProxyError WITHOUT unreachable — it must still surface locally.
async function reqOnce<T>(
	path: string,
	opts: RequestInit,
	headers: Record<string, string>
): Promise<{ value: T }> {
	let res: Response;
	try {
		res = await fetch('/api' + path, { ...opts, headers });
	} catch (e) {
		// fetch only throws on network / abort failures — no response reached us.
		// Use the SOFT signal (reconnecting, 3s grace) not markDown: a blip that a
		// retry heals within the grace window never shows the global banner. req()
		// escalates to markDown() only when it finally gives up.
		serverHealth.markReconnecting();
		(e as { isProxyError?: boolean; unreachable?: boolean }).isProxyError = true;
		(e as { unreachable?: boolean }).unreachable = true;
		throw e;
	}
	// Any HTTP response — including non-2xx — proves the server is reachable.
	serverHealth.markOk();
	if (!res.ok) {
		// The amplio server ALWAYS sends JSON {error} bodies (see server.writeErr).
		// A non-JSON body on an error therefore came from the proxy/infra layer
		// (e.g. UberProxy's HTML 500 page), not the app — collapse it to one line
		// instead of dumping the whole HTML page into the UI.
		const body = await res.text();
		let msg = '';
		let appJSON = false;
		try {
			msg = (JSON.parse(body) as { error?: string }).error ?? '';
			appJSON = true;
		} catch {
			// non-JSON body — an upstream proxy error page, not an amplio response.
		}
		const proxyish = GATEWAY_STATUSES.has(res.status) || (!appJSON && res.status >= 500);
		if (!msg) {
			msg = proxyish
				? `Upstream proxy error (HTTP ${res.status}) — likely a transient network or auth-infra blip, not the amplio server. Retry or reload.`
				: `${res.status} ${res.statusText}`;
		}
		const err = new HttpError(res.status, msg);
		// A 5xx with a response is reachable-but-failing: retryable (isProxyError)
		// but NOT unreachable, so the local banner still surfaces it.
		(err as { isProxyError?: boolean }).isProxyError = proxyish;
		throw err;
	}
	if (res.status === 204) return { value: undefined as T };
	return { value: (await res.json()) as T };
}

// isUnreachable reports whether an error came from the server returning no
// response at all (a transport failure that req() ultimately gave up on). Such
// errors are already surfaced by the global ServerStatusBanner, so a component's
// catch block should NOT also set its own local error banner for them — that
// would double-surface the same root cause. App errors and proxy 5xx (which the
// global banner does NOT cover) return false and should be shown locally.
export function isUnreachable(e: unknown): boolean {
	return !!(e as { unreachable?: boolean })?.unreachable;
}

// errorText is the standard way a catch block turns an error into local-banner
// text: it returns '' for an unreachable-server failure (already shown by the
// global ServerStatusBanner — suppress the redundant local copy) and the error
// message otherwise (app errors, proxy 5xx, validation — which the global banner
// does NOT cover). Assign it directly: `error = errorText(e)`.
export function errorText(e: unknown): string {
	return isUnreachable(e) ? '' : String(e);
}

async function req<T>(path: string, opts: RequestInit = {}): Promise<T> {
	// Auth rides on the same-origin cookie (fetch sends it by default); writes
	// without it get a 403 the caller surfaces. No explicit token header.
	const headers: Record<string, string> = {
		...(opts.headers as Record<string, string>)
	};
	if (opts.body) headers['Content-Type'] = 'application/json';

	// Retry ONLY idempotent reads (GET/HEAD), and only on transient proxy/network
	// conditions. Never auto-retry writes (POST/PATCH/DELETE) — without idempotency
	// keys a retried mutation risks a double-post/double-create; those callers keep
	// their own draft and let the operator retry manually.
	const method = (opts.method ?? 'GET').toUpperCase();
	const idempotent = method === 'GET' || method === 'HEAD';
	const maxAttempts = idempotent ? 3 : 1;

	let lastErr: unknown;
	for (let attempt = 1; attempt <= maxAttempts; attempt++) {
		try {
			const { value } = await reqOnce<T>(path, opts, headers);
			return value;
		} catch (e) {
			lastErr = e;
			const transient = (e as { isProxyError?: boolean }).isProxyError === true;
			if (idempotent && transient && attempt < maxAttempts) {
				// Backoff: ~250ms, 500ms before the 2nd/3rd tries (~0.75s total).
				await sleep(250 * attempt);
				continue;
			}
			// Giving up. If the server never responded (transport failure), escalate
			// the soft 'reconnecting' signal to a hard 'down' now so the global banner
			// shows immediately rather than waiting out the 3s grace timer. A proxy
			// 5xx (reachable) is left to surface via the caller's local banner.
			if ((e as { unreachable?: boolean }).unreachable) serverHealth.markDown();
			throw e;
		}
	}
	throw lastErr; // unreachable (loop either returns or throws), satisfies the type checker
	throw lastErr; // unreachable (loop either returns or throws), satisfies the type checker
}

export interface StartRunRequest {
	task?: string; // autonomous run
	message?: string; // interactive run: opening message (with interactive: true)
	interactive?: boolean;
	title?: string;
	workspace?: string;
	agent?: string;
	llm?: string;
}

export interface RunUpdate {
	title?: string;
	note?: string;
	starred?: boolean;
	// Grade as a string ("garbage".. "excellent") to set the human grade, or
	// null to clear it (fall back to the critic grade). Omit to leave unchanged.
	grade?: string | null;
	archived?: boolean;
}

export const api = {
	// listRuns fetches one page of runs, newest first. Pass `before` (a prior
	// page's next_cursor) to page backwards in time; `limit` overrides the
	// server's default page size.
	listRuns: (
		opts: {
			showArchived?: boolean;
			before?: string;
			limit?: number;
			filter?: string;
			// Search + starred + grade are server-side too, so they compose with the
			// status/archived filters and paginate over the true combined match set.
			search?: string;
			starred?: boolean;
			grade?: string;
		} = {}
	) => {
		const q = new URLSearchParams();
		if (opts.showArchived) q.set('archived', '1');
		if (opts.before) q.set('before', opts.before);
		if (opts.limit != null) q.set('limit', String(opts.limit));
		// Server-side status group (active/done/failed/updates) so pagination is
		// over the matching set. Omitted/'all' => no constraint.
		if (opts.filter && opts.filter !== 'all') q.set('filter', opts.filter);
		if (opts.search && opts.search.trim() !== '') q.set('q', opts.search.trim());
		if (opts.starred) q.set('starred', '1');
		if (opts.grade && opts.grade !== 'all') q.set('grade', opts.grade);
		const qs = q.toString();
		return req<RunsPage>(`/runs${qs ? `?${qs}` : ''}`);
	},
	getRun: (id: string) => req<RunDetail>(`/runs/${id}`),
	// getRunCounts returns the exact, global banner tallies (active + updates)
	// over non-archived runs, independent of list pagination.
	getRunCounts: () => req<RunCounts>('/runs/counts'),
	// markRunSeen clears a run's dashboard "has updates" badge (records that the
	// operator viewed it). Single-operator, so the effect is global.
	markRunSeen: (id: string) =>
		req<void>(`/runs/${id}`, { method: 'PATCH', body: JSON.stringify({ seen: true }) }),
	// markRunUnseen puts the badge back, for a run the operator looked at but is
	// not done with.
	markRunUnseen: (id: string) =>
		req<void>(`/runs/${id}`, { method: 'PATCH', body: JSON.stringify({ seen: false }) }),
	updateRun: (id: string, body: RunUpdate) =>
		req<void>(`/runs/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
	getSysStat: () => req<SysStat>('/sysstat'),
	listModels: () => req<ModelMenu>('/models'),
	addModel: (spec: string) =>
		req<void>('/models', { method: 'POST', body: JSON.stringify({ spec }) }),
	removeModel: (spec: string) =>
		req<void>(`/models?spec=${encodeURIComponent(spec)}`, { method: 'DELETE' }),
	getEvents: (id: string, sid: string, step?: number) =>
		req<EventDTO[]>(
			`/runs/${id}/sessions/${sid}/events${step !== undefined ? `?step=${step}` : ''}`
		),
	// Every event in an inclusive step range, in ONE request — the log viewer's
	// "expand all" over a phase (per-step fetching would be N round-trips).
	getEventsRange: (id: string, sid: string, from: number, to: number) =>
		req<EventDTO[]>(`/runs/${id}/sessions/${sid}/events?from_step=${from}&to_step=${to}`),
	getChat: (id: string, sid: string) => req<ChatFeed>(`/runs/${id}/sessions/${sid}/chat`),
	// Chat projection for an inclusive step range: the read-only log viewer
	// rendering one phase. Unlike getChat, nothing is rolled up into cards — the
	// requested range comes back verbatim (and usage is null).
	getChatRange: (id: string, sid: string, from: number, to: number) =>
		req<ChatFeed>(`/runs/${id}/sessions/${sid}/chat?from_step=${from}&to_step=${to}`),
	getTrajectory: (id: string, sid: string) =>
		req<Trajectory>(`/runs/${id}/sessions/${sid}/trajectory`),
	getObservations: (id: string, sid: string) =>
		req<ObservationDTO[]>(`/runs/${id}/sessions/${sid}/observations`),
	startRun: (body: StartRunRequest) =>
		req<{ run_id: string }>('/runs', { method: 'POST', body: JSON.stringify(body) }),
	// Attach a chatbot co-pilot to an existing (autonomous) run; idempotent.
	startChatbot: (id: string) =>
		req<{ session_id: string }>(`/runs/${id}/chatbot`, { method: 'POST' }),
	sendMessage: (id: string, sid: string, content: string) =>
		req<void>(`/runs/${id}/sessions/${sid}/message`, {
			method: 'POST',
			body: JSON.stringify({ content })
		}),
	cancelRun: (id: string) => req<void>(`/runs/${id}/cancel`, { method: 'POST' }),
	// Permanently delete a run and all its data (DB rows + on-disk artifact/blob
	// dirs). Irreversible; cancels any live sessions first. Mined lessons survive.
	deleteRun: (id: string) => req<void>(`/runs/${id}`, { method: 'DELETE' }),
	// Revive a single run's active spine (same path the server takes at boot
	// for crashed runs). Idempotent: a run already at rest returns
	// {status: "restarted", revived: 0}.
	restartRun: (id: string) =>
		req<{ status: string; revived: number }>(`/runs/${id}/restart`, { method: 'POST' }),
	// All keen-critic report iterations for a run (ascending by version; [] if none).
	getReports: (id: string) => req<RunReport[]>(`/runs/${id}/report`),
	// startReport posts a manual generation request. The backend answers 201
	// Created for a brand-new iteration and 200 OK when it deferred because the
	// delta since the previous report was below critic.ReportSkipMinSteps (in
	// which case the returned report body IS the previous iteration, unchanged).
	// The `deferred` flag lets callers show an honest "no new iteration"
	// affordance rather than looking like a fresh success. Uses a bespoke
	// fetch (rather than the shared req helper) because it needs the HTTP
	// status code, which req discards. Reports have no idempotency key so
	// this is not auto-retried — same as the previous non-status variant.
	startReport: async (id: string): Promise<{ report: RunReport; deferred: boolean }> => {
		const res = await fetch('/api' + `/runs/${id}/report`, { method: 'POST' });
		if (!res.ok) {
			const body = await res.text();
			let msg = '';
			try {
				msg = (JSON.parse(body) as { error?: string }).error ?? '';
			} catch {
				msg = `${res.status} ${res.statusText}`;
			}
			throw new Error(msg || `${res.status} ${res.statusText}`);
		}
		const report = (await res.json()) as RunReport;
		return { report, deferred: res.status === 200 };
	},
	// Trigger keen-critic report generation (synchronous; can take ~a minute).

	// Draft a follow-up instruction from the run's task + latest report (system HQ
	// model). Synchronous; the operator edits the returned prompt before sending.
	suggestFollowup: (id: string) =>
		req<{ prompt: string }>(`/runs/${id}/followup-suggest`, { method: 'POST' }),
	// Recall corpus browse (the same skill+lesson search agents use).
	searchRecall: (q: string, k?: number) =>
		req<RecallResults>(`/recall?q=${encodeURIComponent(q)}${k ? `&k=${k}` : ''}`),
	getRecallItem: (handle: string) =>
		req<RecallItem>(`/recall/item?handle=${encodeURIComponent(handle)}`),
	listLessons: () => req<LessonSummary[]>('/lessons'),
	// Server introspection for the About page (build identity + on-disk layout +
	// configured tiers). Open read.
	getAbout: () => req<AboutInfo>('/about'),
	// Pre-flight test of an agent-LLM spec (build provider + one trivial Call)
	// WITHOUT starting a run. Returns ok=false with a diagnostic on failure.
	testLLM: (spec: string) =>
		req<TestLLMResult>('/about/test-llm', { method: 'POST', body: JSON.stringify({ spec }) }),
	// New-Run workspace control: the workspace sources this build offers, the
	// path to prefill, and recent workspaces (recency-ranked).
	getWorkspaceInfo: () => req<WorkspaceInfo>('/workspaces'),
	// Create a fresh anonymous workspace from a `new:` / `anon:` spec
	// (the slow path: 5-30s on backends that materialize one). Returns the resolved
	// absolute path; the UI then passes that path to startRun. Rejects
	// non-creation specs (existing workspaces, paths) with 400 — for those the UI
	// should call startRun directly with the spec and let the server
	// resolve internally (fast open, no UX value in a separate stage).
	createWorkspace: (spec: string) =>
		req<{ path: string; kind: string }>('/workspaces/new', {
			method: 'POST',
			body: JSON.stringify({ spec })
		}),
	// Live-check a run's workspace alias by bypassing the cache and reading the
	// backend's alias record directly; updates the cache as a side effect
	// so other tabs converge via the workspace_alias SSE event. Used by the
	// Name-workspace modal on open to catch out-of-band attaches.
	checkWorkspaceAlias: (id: string) =>
		req<{ alias: string; cider_url: string }>(`/runs/${id}/workspace/check-alias`, {
			method: 'POST'
		}),
	// Attach an operator-chosen alias to a run's unnamed workspace and return
	// its editor URL. 400 on invalid alias; 409 when the workspace is already
	// named, unsupported by the backend, or the backend rejects the attach.
	attachWorkspaceAlias: (id: string, alias: string) =>
		req<{ alias: string; cider_url: string }>(`/runs/${id}/workspace/alias`, {
			method: 'POST',
			body: JSON.stringify({ alias })
		}),
	// List a directory under a run's artifact dir (path relative to the root).
	listArtifacts: (id: string, path = '') =>
		req<ArtifactListing>(`/runs/${id}/artifacts${path ? `?path=${encodeURIComponent(path)}` : ''}`),
	// Flat, recursive listing of every file under a run's artifact dir (for the
	// browser's filename search).
	listArtifactsAll: (id: string) => req<ArtifactAllListing>(`/runs/${id}/artifacts/all`)
};

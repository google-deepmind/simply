// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"context"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"amplio/internal/agent/critic"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/llm"
	"amplio/internal/workspace"
)

// defaultRunsPage is the server-side page size when the client omits ?limit.
// Kept modest so the front page stays responsive; clients page via ?before.
const defaultRunsPage = 10

// maxRunsPage caps a client-supplied ?limit so a single request can't ask for
// the whole table (that's what the unbounded sweeps are for, server-side only).
const maxRunsPage = 200

func (s *Server) handleListRuns(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	opts := db.ListRunsOpts{
		Limit:        defaultRunsPage,
		ShowArchived: q.Get("archived") == "1",
	}
	// `filter` restricts to a status group (active/done/failed/updates) server-
	// side, so pagination is over the matching set, not just the loaded page.
	opts.StatusFilter = q.Get("filter")
	// Search + starred + grade are ALSO server-side, so they compose (AND) with
	// the status/archived filters and paginate over the true combined match set.
	opts.Search = q.Get("q")
	opts.StarredOnly = q.Get("starred") == "1"
	opts.GradeFilter = parseGradeFilter(q.Get("grade"))
	if v := q.Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			opts.Limit = min(n, maxRunsPage)
		}
	}
	// `before` is the keyset cursor from a prior page's next_cursor: an opaque
	// "<created_at RFC3339Nano>|<run_id>" token. Keyset (not offset) pagination is
	// stable under concurrent run creation — no rows skipped or duplicated at a page
	// boundary. A malformed cursor is ignored (falls back to the first page).
	if v := q.Get("before"); v != "" {
		if ts, rid, ok := strings.Cut(v, "|"); ok {
			if t, err := time.Parse(time.RFC3339Nano, ts); err == nil {
				opts.Before, opts.BeforeRunID = t, rid
			}
		}
	}
	runs, hasMore, err := s.store.ListRunsWithSessions(r.Context(), opts)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := make([]runSummary, 0, len(runs))
	for _, rw := range runs {
		out = append(out, toRunSummary(rw))
	}
	// next_cursor encodes the LAST returned run's (created_at, run_id) as the
	// keyset anchor for the next page, fed back as ?before. Empty when there's no
	// more. Opaque to the client (it round-trips the token unchanged).
	var nextCursor string
	if hasMore && len(runs) > 0 {
		last := runs[len(runs)-1].Run
		nextCursor = last.CreatedAt.Format(time.RFC3339Nano) + "|" + last.RunID
	}
	writeJSON(w, http.StatusOK, runsPage{Runs: out, HasMore: hasMore, NextCursor: nextCursor})
}

// handleRunCounts returns the dashboard banner's exact, global tally over
// non-archived runs (active + has-updates), independent of list pagination.
func (s *Server) handleRunCounts(w http.ResponseWriter, r *http.Request) {
	c, err := s.store.RunCounts(r.Context())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, runCounts{Active: c.Active, Updates: c.Updates})
}

func (s *Server) handleGetRun(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	run, err := s.store.GetRun(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusNotFound, "run not found")
		return
	}
	sessions, err := s.store.ListSessions(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	wm := runWorkspaceMeta(sessions, run.Config.Workspace)
	d := runDetail{
		RunID: run.RunID, Task: run.Config.Task, Title: run.Title, Note: run.Note,
		Starred: run.Starred, Archived: run.Archived,
		Grade: gradeToString(run.Grade), ReportGrade: gradeToString(run.ReportGrade),
		Workspace:          run.Config.Workspace,
		WorkspaceName:      wm.Name,
		WorkspaceKind:      wm.Kind,
		WorkspaceAlias:     wm.Alias,
		WorkspaceNumericID: wm.NumericID,
		CiderURL:           wm.CiderURL,
		LLM:                run.Config.LLM,
		LLMName:            llm.ShortLabel(run.Config.LLM),
		AgentType:          run.Config.AgentType,
		// System tiers are a process-wide property (not per-run); surface the
		// server's current defaults so the Overview still shows what the run's
		// observer/report/compaction actually use.
		SystemLLMHQ:   s.defaults.SystemLLMHQ,
		SystemLLMFast: s.defaults.SystemLLMFast,
		AgentsMD:      run.Config.AgentsMD,
		CreatedAt:     run.CreatedAt,
		UpdatedAt:     run.UpdatedAt,
	}
	for _, sess := range sessions {
		d.Sessions = append(d.Sessions, toSessionDTO(sess))
	}
	// has_updates from the run's ROOT sessions, same predicate as the list view,
	// so the run-page tab's favicon dot stays correct off detail refreshes.
	var roots []db.SessionRecord
	for _, sess := range sessions {
		if sess.ParentID == "" {
			roots = append(roots, sess)
		}
	}
	d.HasUpdates = runHasUpdates(db.RunWithSessions{Run: *run, RootSessions: roots})
	// Report coverage: describes the gap between the latest report's main-agent
	// watermark and the main-agent's current step, using the SAME rule the
	// critic finalizer applies (critic.ReportSkipMinSteps). Lets the UI
	// distinguish "a new iteration is coming" (substantive_gap) from "we saw
	// the delta but chose not to regenerate" (trivial_gap) — the second
	// preventing the eternal-spinner bug where the finalizer's silent skip
	// leaves the UI thinking generation is in flight. Chat runs (no
	// main-agent) and pre-first-report runs leave both fields at their zero
	// value, omitted from the JSON.
	d.ReportCoverage, d.ReportGapSteps = s.computeReportCoverage(r.Context(), id, sessions)
	// Snapshot in-flight ephemeral agents (e.g. the critic generating a
	// report) from the runtime registry. Empty when nothing's running.
	if reg := s.mgr.EphemeralAgents(); reg != nil {
		for _, ag := range reg.ForRun(id) {
			d.EphemeralAgents = append(d.EphemeralAgents, ephemeralAgentDTO{
				Kind:      ag.Kind,
				StartedAt: ag.StartedAt,
			})
		}
	}
	writeJSON(w, http.StatusOK, d)
}

// computeReportCoverage classifies the pending regeneration state of a run,
// mirroring the finalizer's own delta rule. Errors are treated as "no info":
// coverage is a UI hint, not a hard invariant, so a failed observation read
// (rare) shouldn't fail the whole run-detail request.
func (s *Server) computeReportCoverage(ctx context.Context, runID string, sessions []db.SessionRecord) (coverage string, gap int) {
	var main *db.SessionRecord
	for i := range sessions {
		if sessions[i].SessionID == config.RootAgentSessionID {
			main = &sessions[i]
			break
		}
	}
	if main == nil {
		return "", 0 // chat run — no main-agent watermark to compare against
	}
	prev, err := critic.LatestReport(ctx, s.store, runID)
	if err != nil || prev == nil {
		return "", 0
	}
	gap = main.CurrentStep - prev.SessionStep(config.RootAgentSessionID)
	switch {
	case gap <= 0:
		return "covered", 0
	case gap < critic.ReportSkipMinSteps:
		return "trivial_gap", gap
	default:
		return "substantive_gap", gap
	}
}

// workspaceMeta holds the render-ready info the dashboard rows and run-detail
// card need about a run's workspace: a cheap display name plus optional
// fields that enable the Open-in-Cider link (named) and the Name-workspace
// menu item (anonymous). Empty fields mean "not applicable" — non-CitC
// backends carry only Kind+Name.
type workspaceMeta struct {
	Name      string // alias (CitC named) / basename (plain) / numeric id (CitC anon)
	Kind      string // workspace.Workspace.Kind(): "citc"/"plain"/"jj"/"external"/""
	Alias     string // CitC named only ("" for anonymous or non-CitC)
	NumericID int    // CitC only (0 for non-CitC)
	CiderURL  string // populated only for CitC named (kind="citc" && alias!="")
}

// runWorkspaceMeta reconstructs workspace info from a run's top-level session
// metadata. CitC alias is read through the non-blocking cache: cache hit returns
// the alias immediately, cache miss returns "" and kicks a background refresh
// whose eventual value fans out via the workspace-alias SSE event. Never blocks
// on FUSE. Falls back to the run's recorded workspace path basename when no
// session metadata is parseable (legacy runs predating workspace persistence).
//
// Internal-only fields (Alias, NumericID, CiderURL for CitC workspaces) are
// populated by attachCorpWorkspaceMeta — a stub in the OSS build, real in the
// internal build (see read_extras{,_internal}.go).
func runWorkspaceMeta(sessions []db.SessionRecord, fallbackPath string) workspaceMeta {
	for _, s := range sessions {
		if s.ParentID != "" {
			continue // top-level sessions carry the run's own workspace
		}
		raw, ok := s.Metadata[workspace.SessionMetadataKey].(string)
		if !ok || raw == "" {
			continue
		}
		ws, err := workspace.Unmarshal([]byte(raw))
		if err != nil {
			continue
		}
		m := workspaceMeta{Kind: ws.Kind()}
		attachCorpWorkspaceMeta(&m, ws)
		if m.Name == "" {
			m.Name = ws.Name() // fallback: basename for plain/jj, numeric id for anonymous CitC
		}
		return m
	}
	if fallbackPath == "" {
		return workspaceMeta{}
	}
	return workspaceMeta{Name: filepath.Base(fallbackPath)}
}

// handleGetReport returns all keen-critic report iterations for a run (ascending
// by version; empty array if none yet). The Overview page auto-loads this to
// display the latest report without re-generating.
func (s *Server) handleGetReport(w http.ResponseWriter, r *http.Request) {
	reports, err := critic.AllReports(r.Context(), s.store, r.PathValue("id"))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, reports)
}

// stepRange parses the optional inclusive from_step/to_step query pair shared
// by the events and chat endpoints (the session-log viewer fetches one phase at
// a time). Returns nil for an absent or unparsable bound, so a caller can treat
// "neither present" as "no range requested".
func stepRange(r *http.Request) (from, to *int) {
	parse := func(key string) *int {
		v := r.URL.Query().Get(key)
		if v == "" {
			return nil
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil
		}
		return &n
	}
	return parse("from_step"), parse("to_step")
}

func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	id, sid := r.PathValue("id"), r.PathValue("sid")
	// All generations: this is a human-facing trajectory view, not the LLM's
	// current context — show the full history including pre-compaction events.
	filter := db.EventFilter{}
	if v := r.URL.Query().Get("since_step"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			start := n + 1
			filter.StartStep = &start
		}
	}
	// step=N fetches exactly one step (the trajectory drill-down's on-demand load).
	if v := r.URL.Query().Get("step"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			filter.StartStep = &n
			filter.EndStep = &n
		}
	}
	// from_step/to_step fetch an inclusive RANGE in one request — the session-log
	// viewer's "expand all" over a whole phase, which would otherwise be one
	// request per step. Either bound may be given alone (open-ended on the other
	// side). Applied after step=N so an explicit single step still wins.
	if from, to := stepRange(r); from != nil || to != nil {
		if filter.StartStep == nil {
			filter.StartStep = from
		}
		if filter.EndStep == nil {
			filter.EndStep = to
		}
	}
	recs, err := s.store.GetEvents(r.Context(), id, sid, filter)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := make([]eventDTO, 0, len(recs))
	for _, rec := range recs {
		dto, err := toEventDTO(rec)
		if err != nil {
			continue // skip an unmarshalable event rather than failing the page
		}
		out = append(out, dto)
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleObservations(w http.ResponseWriter, r *http.Request) {
	id, sid := r.PathValue("id"), r.PathValue("sid")
	recs, err := s.store.GetObservations(r.Context(), id, db.ObsFilter{SessionID: sid})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := make([]observationDTO, 0, len(recs))
	for _, o := range recs {
		out = append(out, toObservationDTO(o))
	}
	writeJSON(w, http.StatusOK, out)
}

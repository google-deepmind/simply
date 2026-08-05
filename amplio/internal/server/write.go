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
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"amplio/internal/config"
	"amplio/internal/event"
	"amplio/internal/runspec"
	"amplio/internal/runtime"
	"amplio/internal/session"
)

func (s *Server) handleStartRun(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Task        string `json:"task"`
		Title       string `json:"title"`
		Workspace   string `json:"workspace"`
		Agent       string `json:"agent"`
		LLM         string `json:"llm"`
		Interactive bool   `json:"interactive"` // interactive run (chatbot root, empty task)
		Message     string `json:"message"`     // opening message (interactive mode)
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Workspace == "" {
		req.Workspace = "."
	}
	// Per-run model override; falls back to the server's configured default.
	llmSpec := req.LLM
	if llmSpec == "" {
		llmSpec = s.defaults.LLM
	}

	// Interactive mode: empty task, chatbot root, the message seeded as step-1
	// input. Autonomous mode: a task drives the standard (or requested) agent.
	task, agentType, rootSessionID, firstMessage := req.Task, req.Agent, config.RootAgentSessionID, ""
	if req.Interactive {
		if req.Message == "" {
			writeErr(w, http.StatusBadRequest, "message is required for an interactive run")
			return
		}
		task, agentType, rootSessionID, firstMessage = "", config.ChatbotAgentType, config.ChatbotSessionID, req.Message
	} else {
		if req.Task == "" {
			writeErr(w, http.StatusBadRequest, "task is required")
			return
		}
		if agentType == "" {
			agentType = s.defaults.AgentType
		}
	}
	// Resolve the workspace spec to a concrete path (creation sentinels run
	// their side effects here) and snapshot the operator's AGENTS.md. Storing
	// the RESOLVED path in RunConfig (not the raw sentinel) means re-deriving it
	// later for a chatbot sidecar / respawn re-detects the same workspace
	// instead of creating a fresh one. The snapshot captures whatever's in
	// effect at start time; respawns reuse it rather than re-reading the files.
	// The OS user is needed locally to resolve a citc workspace, but is no
	// longer recorded on the run itself (single-user amplio).
	wsRoot, agentsMD, err := runspec.Prepare(req.Workspace, os.Getenv("USER"))
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	// runCtx (server lifetime), not r.Context(): the run must outlive the request.
	runID, err := s.mgr.StartRun(s.runCtx, runtime.StartRunConfig{
		Title:        req.Title,
		FirstMessage: firstMessage,
		RunConfig: config.RunConfig{
			Task:      task,
			Workspace: wsRoot,
			LLM:       llmSpec,
			AgentType: agentType,
			AgentsMD:  agentsMD,
		},
		RootSessionID: rootSessionID,
	})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"run_id": runID})
}

// maxEnvNoticesPerStep bounds how many environment notifications
// ($AMPLIO_NOTIFY) may pile up at a SINGLE step of one session, after which
// handleNotify refuses them.
const maxEnvNoticesPerStep = 50

// errEnvNoticeCapped is the stable machine-readable prefix of the 429 body when
// maxEnvNoticesPerStep is hit. Kept constant and free of variable text so a
// background script can match it exactly (`grep -q env_notice_capped`) and stop
// itself; the status code alone can't distinguish this from any other refusal.
const errEnvNoticeCapped = "env_notice_capped"

// handleNotify delivers an environment message to a session, used by the
// `amplio notify` CLI that background scripts (spawned via the bash tool) call.
// The message is an Input-class MessageEvent (SenderType=environment): it wakes a
// live session and revives a cold PARKED one (idle/awaiting) — so a background
// job's completion reaches an agent that merely forgot to await_event — but it
// does NOT revive a FINISHED session (concluded/crashed/cancelled): the
// environment can't resurrect a deliberately-finished agent (the wake-path gate
// in runtime.NewCommitNotifier). See docs/session_lifecycle.md.
//
// Because a notification to a finished session still PERSISTS (it is simply not
// acted on), an abandoned notifier can append for as long as it runs; hence the
// per-step cap below, which applies uniformly rather than only on that path.
func (s *Server) handleNotify(w http.ResponseWriter, r *http.Request) {
	id, sid := r.PathValue("id"), r.PathValue("sid")
	var req struct {
		Content string `json:"content"`
		Sender  string `json:"sender"` // optional source label; SenderType is always environment
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Content == "" {
		writeErr(w, http.StatusBadRequest, "content is required")
		return
	}
	// A mistyped --session is the obvious failure mode now that the target is
	// named, and an FK violation surfacing as a 500 tells the caller nothing.
	if _, err := s.store.GetSession(r.Context(), id, sid); err != nil {
		writeErr(w, http.StatusNotFound,
			fmt.Sprintf("no session %q in run %q", sid, id))
		return
	}

	// Flood guard: refuse once this step has absorbed its budget. 429 rather
	// than a silent drop, because the caller is a script — `amplio notify` maps
	// non-2xx to exit code 3 and prints the body on stderr, so a runaway loop
	// gets a signal it can act on instead of spinning forever. The body is a
	// fixed, greppable token.
	n, err := s.store.CountEnvNotices(r.Context(), id, sid)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if n >= maxEnvNoticesPerStep {
		writeErr(w, http.StatusTooManyRequests,
			fmt.Sprintf("%s: per-step cap %d", errEnvNoticeCapped, maxEnvNoticesPerStep))
		return
	}
	sender := req.Sender
	if sender == "" {
		sender = event.EnvironmentSenderID
	}
	if _, err := s.store.AppendEvent(r.Context(), id, sid,
		&event.MessageEvent{Content: req.Content, Sender: sender, SenderType: event.SenderTypeEnvironment}); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"status": "delivered"})
}

// handleStartChatbot attaches an interactive chatbot sidecar to an existing run
// (idempotent) and returns its session id. Used by the chat tab of an autonomous
// run to lazily spin up a co-pilot on first visit.
func (s *Server) handleStartChatbot(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	// s.runCtx, not r.Context(): the chatbot must outlive this request.
	sid, err := s.mgr.AddChatbot(s.runCtx, id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"session_id": sid})
}

// handleGenerateReport produces (or refreshes) the run's keen-critic report on
// operator demand. Returns 201 (Created) with the new report body when a new
// iteration was generated, and 200 (OK) with the previous report body when the
// finalizer deferred because the delta since the previous report is below
// critic.ReportSkipMinSteps. s.runCtx, not r.Context(): generation outlives the
// request.
func (s *Server) handleGenerateReport(w http.ResponseWriter, r *http.Request) {
	if s.genReport == nil {
		writeErr(w, http.StatusNotImplemented, "report generation is not configured")
		return
	}
	report, deferred, err := s.genReport(s.runCtx, r.PathValue("id"))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	status := http.StatusCreated
	if deferred {
		status = http.StatusOK
	}
	writeJSON(w, status, report)
}

// handleSuggestFollowup drafts a follow-up instruction for a concluded
// autonomous run by asking the system HQ model to read the run's original task +
// latest report. The result is a SUGGESTION the operator edits before sending —
// nothing is started here. Behind requireAuth: it makes a real billable LLM call.
func (s *Server) handleSuggestFollowup(w http.ResponseWriter, r *http.Request) {
	if s.suggestFollowup == nil {
		writeErr(w, http.StatusNotImplemented, "follow-up suggestion is not configured")
		return
	}
	prompt, err := s.suggestFollowup(r.Context(), r.PathValue("id"))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"prompt": prompt})
}

// handleUpdateRun applies a partial update to a run's editable overlay. Fields
// are pointers so absent ones are left untouched (vs. set-to-empty/false).
func (s *Server) handleUpdateRun(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req struct {
		Title   *string `json:"title"`
		Note    *string `json:"note"`
		Starred *bool   `json:"starred"`
		// Grade uses RawMessage so we can tell three cases apart: field absent
		// (nil, leave unchanged), explicit JSON null (clear → grade 0), and a
		// grade string ("garbage".. "excellent" → rank 1..5). A plain *string
		// can't distinguish "absent" from "null".
		Grade    json.RawMessage `json:"grade"`
		Archived *bool           `json:"archived"`
		// Seen records that the operator viewed this run now (true, clearing the
		// dashboard badge) or that they are not done with it (false, putting the
		// badge back — see MarkRunUnseen).
		Seen *bool `json:"seen"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	apply := func(err error) bool {
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
		}
		return err == nil
	}
	if req.Title != nil && !apply(s.store.UpdateRunTitle(r.Context(), id, *req.Title)) {
		return
	}
	if req.Note != nil && !apply(s.store.UpdateRunNote(r.Context(), id, *req.Note)) {
		return
	}
	if req.Starred != nil && !apply(s.store.SetRunStarred(r.Context(), id, *req.Starred)) {
		return
	}
	// Grade: explicit null clears (rank 0); a known grade string sets the rank;
	// anything else is a 400. Absent (nil RawMessage) leaves it untouched.
	if req.Grade != nil {
		grade, ok := parseGradePatch(req.Grade)
		if !ok {
			writeErr(w, http.StatusBadRequest, `grade must be null or one of "garbage","bad","meh","good","excellent"`)
			return
		}
		if !apply(s.store.SetRunGrade(r.Context(), id, grade)) {
			return
		}
	}
	if req.Archived != nil && !apply(s.store.SetRunArchived(r.Context(), id, *req.Archived)) {
		return
	}
	if req.Seen != nil {
		mark := s.store.MarkRunSeen
		if !*req.Seen {
			mark = s.store.MarkRunUnseen
		}
		if !apply(mark(r.Context(), id)) {
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "updated"})
}

// parseGradePatch decodes a PATCH `grade` value (a non-nil RawMessage) into a
// stored rank: JSON null → 0 (clear), a known grade string → 1..5. ok is false
// for malformed JSON or an unknown grade string, which the caller maps to 400.
func parseGradePatch(raw json.RawMessage) (grade int, ok bool) {
	if string(raw) == "null" {
		return 0, true
	}
	var name string
	if err := json.Unmarshal(raw, &name); err != nil {
		return 0, false
	}
	rank := gradeFromString(name)
	if rank == 0 {
		return 0, false
	}
	return rank, true
}

func (s *Server) handleSendMessage(w http.ResponseWriter, r *http.Request) {
	id, sid := r.PathValue("id"), r.PathValue("sid")
	var req struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Content == "" {
		writeErr(w, http.StatusBadRequest, "content is required")
		return
	}
	// A UserEvent is Input-class: the commit notifier wakes/respawns the target
	// session (the iteration / follow-up mechanism).
	eventID, err := s.store.AppendEvent(r.Context(), id, sid, &event.UserEvent{Content: req.Content})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"status": "delivered", "event_id": eventID})
}

func (s *Server) handleCancelRun(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	sessions, err := s.store.ListSessions(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	reg := s.mgr.RunRegistry().GetOrCreate(id)
	// Cancelling each root cascades to its children (CancelSession is recursive).
	for _, sess := range sessions {
		if sess.ParentID == "" {
			if err := session.CancelSession(s.store, reg, id, sess.SessionID, "cancelled via UI"); err != nil {
				slog.Warn("cancel run: session cancel failed", "run", id, "session", sess.SessionID, "error", err)
			}
		}
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"status": "cancelling"})
}

// handleRestartRun revives a single run's active spine the same way the server
// would on a fresh boot. Useful mainly for a crashed run: clicking Restart in
// the UI re-spawns the dead root via RecoverRun (which is idempotent — a run
// already at rest revives zero sessions and returns cleanly).
func (s *Server) handleRestartRun(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	// Use runCtx (server lifetime), not r.Context(): the respawned goroutines
	// must outlive the HTTP request that triggered them.
	revived, err := s.mgr.RecoverRun(s.runCtx, id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{
		"status":  "restarted",
		"revived": revived,
	})
}

// handleDeleteRun permanently deletes a run: it cancels any still-live sessions
// first (so no goroutine writes rows we're about to delete), removes all the
// run's DB rows in one transaction, then best-effort removes its on-disk
// artifact + blob dirs. This is the operator's escape hatch for clearing out
// corrupted / dead runs that can't be recovered. Destructive and irreversible;
// the UI gates it behind a two-step confirm. Mined lessons are kept (they carry
// a soft source_run_id, not an FK).
func (s *Server) handleDeleteRun(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	// Cancel live sessions first. Best-effort: a run with no live goroutines
	// (the common case for a corrupted/dead run) cancels nothing and proceeds.
	if sessions, err := s.store.ListSessions(r.Context(), id); err == nil {
		reg := s.mgr.RunRegistry().GetOrCreate(id)
		for _, sess := range sessions {
			if sess.ParentID == "" {
				if err := session.CancelSession(s.store, reg, id, sess.SessionID, "run deleted via UI"); err != nil {
					slog.Warn("delete run: session cancel failed", "run", id, "session", sess.SessionID, "error", err)
				}
			}
		}
	}
	if err := s.store.DeleteRun(r.Context(), id); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	// On-disk data lives outside the DB. Best-effort: a failure here leaves an
	// orphan dir (harmless, just disk) but the run is already gone from the UI.
	if err := os.RemoveAll(config.ArtifactDir(id)); err != nil {
		slog.Warn("delete run: remove artifact dir failed", "run", id, "error", err)
	}
	if err := os.RemoveAll(config.BlobDir(id)); err != nil {
		slog.Warn("delete run: remove blob dir failed", "run", id, "error", err)
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

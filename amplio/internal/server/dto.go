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
	"net/http"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/llm"
	"amplio/internal/workspace"
)

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v) //nolint:errcheck // response already committed
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

// gradeNames maps the integer grade ranks to their JSON/UI string form. Index 0
// (ungraded) has no string: it is represented as a null grade in JSON.
var gradeNames = [...]string{"", "garbage", "bad", "meh", "good", "excellent"}

// gradeToString converts a stored integer grade (0=ungraded, 1=garbage..
// 5=excellent) to its boundary string form, returning nil for ungraded (0) and
// for any out-of-range value so the JSON field renders as null.
func gradeToString(g int) *string {
	if g < 1 || g >= len(gradeNames) {
		return nil
	}
	s := gradeNames[g]
	return &s
}

// gradeFromString converts a boundary grade string to its integer rank, or 0 if
// the string isn't one of the five known names (the caller validates).
func gradeFromString(s string) int {
	for i, name := range gradeNames {
		if i > 0 && name == s {
			return i
		}
	}
	return 0
}

// parseGradeFilter maps the dashboard's ?grade= value to the store's grade
// filter enum: "" / "all" = no filter, "ungraded" = neither grade set, a known
// grade name = that exact rank. Unknown names yield GradeUngraded's sibling
// "match nothing" only for truly invalid ranks (handled in the store); here an
// unrecognized name falls through to no filter so a stray param doesn't blank
// the list.
func parseGradeFilter(s string) db.GradeFilter {
	switch s {
	case "", "all":
		return db.GradeNone
	case "ungraded":
		return db.GradeUngraded
	default:
		if rank := gradeFromString(s); rank > 0 {
			return db.GradeFilter(rank)
		}
		return db.GradeNone
	}
}

// rootInfo is one top-level (parentless) agent of a run. A run usually has one
// (the autonomous root, or the chatbot root of a chat run), but an autonomous
// run with a sidecar chatbot attached has two — hence the list on runSummary.
type rootInfo struct {
	SessionID       string    `json:"session_id"`
	AgentType       string    `json:"agent_type"`
	Status          string    `json:"status"`
	Step            int       `json:"step"`
	StatusChangedAt time.Time `json:"status_changed_at"` // bumps only on transitions; powers the "unread updates" badge
}

// runSummary is one row of the dashboard: enough to show liveness AND identity
// (workspace, llm, agent-type) without a per-session fetch.
// runCounts is the GET /api/runs/counts response: the dashboard banner's exact,
// global tallies over non-archived runs, independent of list pagination.
type runCounts struct {
	Active  int `json:"active"`
	Updates int `json:"updates"`
}

// runsPage is the paginated response from GET /api/runs. Runs is one page
// (newest first); HasMore signals another page exists; NextCursor is the
// created_at to pass back as ?before to fetch it (empty when exhausted).
type runsPage struct {
	Runs       []runSummary `json:"runs"`
	HasMore    bool         `json:"has_more"`
	NextCursor string       `json:"next_cursor"`
}

type runSummary struct {
	RunID   string `json:"run_id"`
	Task    string `json:"task"`
	Title   string `json:"title"`
	Starred bool   `json:"starred"`
	// Grade is the human grade as a string ("garbage".. "excellent"), or null
	// when ungraded. ReportGrade is the cached keen-critic grade the human grade
	// overrides; null when the critic hasn't graded the run.
	Grade              *string    `json:"grade"`
	ReportGrade        *string    `json:"report_grade"`
	Archived           bool       `json:"archived"`
	CreatedAt          time.Time  `json:"created_at"`
	Workspace          string     `json:"workspace"`                      // full resolved path (tooltip)
	WorkspaceName      string     `json:"workspace_name"`                 // cheap display (basename / workspace alias)
	WorkspaceKind      string     `json:"workspace_kind"`                 // "citc"/"plain"/"jj"/"external"/""
	WorkspaceAlias     string     `json:"workspace_alias,omitempty"`      // set only by backends that name workspaces
	WorkspaceNumericID int        `json:"workspace_numeric_id,omitempty"` // set only by backends that have one
	CiderURL           string     `json:"cider_url,omitempty"`            // editor URL; the row pill becomes a link when non-empty
	LLM                string     `json:"llm"`                            // full spec (tooltip); the only form that may be copied or re-used
	LLMName            string     `json:"llm_name"`                       // cheap display label, derived — never show it without llm reachable
	Roots              []rootInfo `json:"roots"`
	// Primary-root convenience fields (the autonomous agent if present, else the
	// chatbot root). The UI can show all roots via Roots; these keep the simple
	// single-status row working.
	RootSessionID       string    `json:"root_session_id"`
	RootStatus          string    `json:"root_status"`
	RootStep            int       `json:"root_step"`
	RootStatusChangedAt time.Time `json:"root_status_changed_at"` // primary root's last status transition; drives the "last update" display
	RootAgentType       string    `json:"root_agent_type"`        // chatbot vs standard agent — picks the row's identity icon
	SessionCount        int       `json:"session_count"`
	// HasUpdates is the server-computed dashboard badge: a relevant root status
	// changed since the run was last seen. Replaces the former client-side
	// localStorage comparison so the badge is global and pagination-independent.
	HasUpdates bool `json:"has_updates"`
}

func toRunSummary(rw db.RunWithSessions) runSummary {
	wm := runWorkspaceMeta(rw.RootSessions, rw.Run.Config.Workspace)
	s := runSummary{
		RunID:              rw.Run.RunID,
		Task:               rw.Run.Config.Task,
		Title:              rw.Run.Title,
		Starred:            rw.Run.Starred,
		Grade:              gradeToString(rw.Run.Grade),
		ReportGrade:        gradeToString(rw.Run.ReportGrade),
		Archived:           rw.Run.Archived,
		CreatedAt:          rw.Run.CreatedAt,
		Workspace:          rw.Run.Config.Workspace,
		WorkspaceName:      wm.Name,
		WorkspaceKind:      wm.Kind,
		WorkspaceAlias:     wm.Alias,
		WorkspaceNumericID: wm.NumericID,
		CiderURL:           wm.CiderURL,
		LLM:                rw.Run.Config.LLM,
		LLMName:            llm.ShortLabel(rw.Run.Config.LLM),
		SessionCount:       len(rw.RootSessions),
		Roots:              make([]rootInfo, 0, len(rw.RootSessions)),
	}
	for _, r := range rw.RootSessions {
		s.Roots = append(s.Roots, rootInfo{
			SessionID:       r.SessionID,
			AgentType:       r.AgentType,
			Status:          r.Status,
			Step:            r.CurrentStep,
			StatusChangedAt: r.StatusChangedAt,
		})
	}
	if root := primaryRoot(rw.RootSessions); root != nil {
		s.RootSessionID = root.SessionID
		s.RootStatus = root.Status
		s.RootStep = root.CurrentStep
		s.RootStatusChangedAt = root.StatusChangedAt
		s.RootAgentType = root.AgentType
	}
	s.HasUpdates = runHasUpdates(rw)
	return s
}

// relevantUpdateStatuses are the root statuses that count as "the agent is done
// and wants you to look" — the same set the dashboard badge used client-side.
var relevantUpdateStatuses = map[string]bool{
	db.SessionIdle:      true,
	db.SessionConcluded: true,
	db.SessionCrashed:   true,
	db.SessionCancelled: true,
}

// runHasUpdates mirrors the store's runUpdatesPredicate for a single in-memory
// run: a relevant root status changed after the run was last seen. A zero
// LastSeenAt means SEEN (NULL last_seen_at — legacy rows or dismissed runs), not
// never-seen: new runs are stamped at creation, so a zero value here is the
// "no badge" case. Kept in sync with sqlite.runUpdatesPredicate.
func runHasUpdates(rw db.RunWithSessions) bool {
	if rw.Run.LastSeenAt.IsZero() {
		return false
	}
	for _, r := range rw.RootSessions {
		if relevantUpdateStatuses[r.Status] && r.StatusChangedAt.After(rw.Run.LastSeenAt) {
			return true
		}
	}
	return false
}

// primaryRoot picks the root whose status best represents the run: the first
// non-chatbot (autonomous) root if any, else the first root (a chat-only run,
// whose chatbot is the run). Returns nil if there are no roots.
func primaryRoot(roots []db.SessionRecord) *db.SessionRecord {
	for i := range roots {
		if roots[i].AgentType != config.ChatbotAgentType {
			return &roots[i]
		}
	}
	if len(roots) > 0 {
		return &roots[0]
	}
	return nil
}

type sessionDTO struct {
	SessionID       string    `json:"session_id"`
	ParentID        string    `json:"parent_id"`
	AgentType       string    `json:"agent_type"`
	Task            string    `json:"task"`
	Status          string    `json:"status"`
	CurrentStep     int       `json:"current_step"`
	CreatedAt       time.Time `json:"created_at"`
	StatusChangedAt time.Time `json:"status_changed_at"` // mirrors rootInfo so the run-page header can synthesize the same view as a dashboard row
	// Per-session workspace from the session's own metadata: sub-agents spawned
	// with WorkspaceMode "link" run in their own linked worktree, so this can
	// differ from the run's (and across siblings). Empty if not yet persisted.
	Workspace     string `json:"workspace,omitempty"`      // full path (tooltip)
	WorkspaceName string `json:"workspace_name,omitempty"` // cheap display (basename / link name / numeric id)
}

func toSessionDTO(s db.SessionRecord) sessionDTO {
	path, name := sessionWorkspace(s)
	return sessionDTO{
		SessionID:       s.SessionID,
		ParentID:        s.ParentID,
		AgentType:       s.AgentType,
		Task:            s.Task,
		Status:          s.Status,
		CurrentStep:     s.CurrentStep,
		CreatedAt:       s.CreatedAt,
		StatusChangedAt: s.StatusChangedAt,
		Workspace:       path,
		WorkspaceName:   name,
	}
}

// sessionWorkspace reads a session's own workspace (persisted at bootstrap under
// Session.Metadata) and returns its full path and a cheap display name. Uses
// ws.Name() rather than ResolveAlias to stay I/O-free — this runs per session
// when building the run detail. Returns empty strings when absent/unparseable.
func sessionWorkspace(s db.SessionRecord) (path, name string) {
	raw, ok := s.Metadata[workspace.SessionMetadataKey].(string)
	if !ok || raw == "" {
		return "", ""
	}
	ws, err := workspace.Unmarshal([]byte(raw))
	if err != nil {
		return "", ""
	}
	return ws.Root(), ws.Name()
}

type runDetail struct {
	RunID   string `json:"run_id"`
	Task    string `json:"task"`
	Title   string `json:"title"`
	Note    string `json:"note"`
	Starred bool   `json:"starred"`
	// Grade is the human grade as a string ("garbage".. "excellent"), or null
	// when ungraded. ReportGrade is the cached keen-critic grade the human grade
	// overrides; null when the critic hasn't graded the run.
	Grade              *string `json:"grade"`
	ReportGrade        *string `json:"report_grade"`
	Archived           bool    `json:"archived"`
	Workspace          string  `json:"workspace"`                      // full resolved path (tooltip / detail)
	WorkspaceName      string  `json:"workspace_name"`                 // cheap display name (basename / numeric id)
	WorkspaceKind      string  `json:"workspace_kind"`                 // "citc"/"plain"/"jj"/"external"/""
	WorkspaceAlias     string  `json:"workspace_alias,omitempty"`      // set only by backends that name workspaces
	WorkspaceNumericID int     `json:"workspace_numeric_id,omitempty"` // set only by backends that have one
	CiderURL           string  `json:"cider_url,omitempty"`            // editor URL, when the backend provides one
	LLM                string  `json:"llm"`                            // full spec (Overview shows it verbatim)
	LLMName            string  `json:"llm_name"`                       // cheap display label, derived — never show it without llm reachable
	// Configured at run creation; full picture of the run's persisted RunConfig
	// so the Overview page can show "everything we know about this run" without
	// needing extra round-trips.
	AgentType     string       `json:"agent_type"`
	SystemLLMHQ   string       `json:"system_llm_hq"`
	SystemLLMFast string       `json:"system_llm_fast"`
	AgentsMD      string       `json:"agents_md"` // raw markdown the operator supplied; can be long, UI collapses
	CreatedAt     time.Time    `json:"created_at"`
	UpdatedAt     time.Time    `json:"updated_at"` // last metadata edit (title/note/star/archive); == created_at if never edited
	Sessions      []sessionDTO `json:"sessions"`
	// EphemeralAgents lists in-flight non-session workers for the run (today:
	// the keen-critic report generator). Empty when nothing's running. The
	// UI uses it to render "Generating report… (Nm Ns)" with elapsed time,
	// and to distinguish "no report yet because we're working on one" from
	// "no report yet because none was triggered" (the wording is the same
	// either way, but the spinner differs).
	EphemeralAgents []ephemeralAgentDTO `json:"ephemeral_agents"`
	// HasUpdates is the server-computed dashboard badge (a relevant root status
	// changed since last seen). Mirrors runSummary.has_updates so the run-page tab
	// can keep its favicon dot accurate from detail refreshes alone.
	HasUpdates bool `json:"has_updates"`

	// ReportCoverage classifies the gap between the latest report's main-agent
	// watermark and the main-agent's current step:
	//   - "":                no autonomous main-agent (chat run), or no prior
	//                        report yet — coverage is not meaningful.
	//   - "covered":          the latest report is at or past the current step.
	//   - "trivial_gap":      gap in (0, critic.ReportSkipMinSteps) — the
	//                        finalizer deferred; no auto-report will fire and a
	//                        manual Generate would also defer. UI shows an
	//                        honest "no new iteration" note.
	//   - "substantive_gap": gap ≥ critic.ReportSkipMinSteps — a real pending
	//                        regeneration (post-crash, or the auto-trigger has
	//                        not fired yet).
	ReportCoverage string `json:"report_coverage,omitempty"`
	// ReportGapSteps is the actual main-agent step delta the coverage was
	// computed from. Only meaningful when ReportCoverage is "trivial_gap" or
	// "substantive_gap"; the UI uses it in operator-facing copy.
	ReportGapSteps int `json:"report_gap_steps,omitempty"`
}

// ephemeralAgentDTO is the wire shape of a runtime.EphemeralAgent for the
// run detail view. The internal ID is dropped — the UI only needs the kind
// and an elapsed-time anchor.
type ephemeralAgentDTO struct {
	Kind      string    `json:"kind"`       // "report" today; future: "title", …
	StartedAt time.Time `json:"started_at"` // UTC; UI renders relative-to-now
}

// eventDTO carries the marshaled typed event so the client renders per kind.
type eventDTO struct {
	Step       int             `json:"step"`
	Generation int             `json:"generation"`
	CreatedAt  time.Time       `json:"created_at"`
	Event      json.RawMessage `json:"event"`
}

func toEventDTO(r db.EventRecord) (eventDTO, error) {
	data, err := event.Marshal(r.Event)
	if err != nil {
		return eventDTO{}, err
	}
	return eventDTO{Step: r.Step, Generation: r.Generation, CreatedAt: r.CreatedAt, Event: data}, nil
}

type observationDTO struct {
	Kind      string         `json:"kind"`
	SessionID string         `json:"session_id"`
	Step      *int           `json:"step"`
	CharCount int            `json:"char_count"`
	Data      map[string]any `json:"data"`
	CreatedAt time.Time      `json:"created_at"`
}

func toObservationDTO(o db.ObservationRecord) observationDTO {
	return observationDTO{
		Kind:      o.Kind,
		SessionID: o.SessionID,
		Step:      o.Step,
		CharCount: o.CharCount,
		Data:      o.Data,
		CreatedAt: o.CreatedAt,
	}
}

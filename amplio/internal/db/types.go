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

package db

import (
	"strings"
	"time"

	"amplio/internal/config"
	"amplio/internal/event"
)

// --- Session status ---
//
// These represent the last-known state of a session, not real-time status.
// Every non-ongoing status is restartable; what differs is the trigger
// (see docs/session_lifecycle.md).
//
//   - ongoing:   has unfinished work (not necessarily executing right now).
//   - awaiting:  parked via await_event (an explicit wake predicate).
//   - idle:      parked after a bare no-tool-call turn (no declared predicate).
//   - concluded: finished via the explicit conclude tool.
//   - crashed:   stopped by an uncaught error (durably recorded).
//   - cancelled: explicitly stopped.

const (
	SessionOngoing   = "ongoing"
	SessionAwaiting  = "awaiting"
	SessionIdle      = "idle"
	SessionConcluded = "concluded"
	SessionCrashed   = "crashed"
	SessionCancelled = "cancelled"
)

// SessionTerminalStatuses are the "finished" states. A session in one of these
// has stopped and already posted any parent notification (concluded/crashed/
// cancelled all go through TerminateAndNotifyParent). Used to skip
// cascade-cancel of already-finished children.
var SessionTerminalStatuses = map[string]bool{
	SessionConcluded: true,
	SessionCrashed:   true,
	SessionCancelled: true,
}

// EventClass sorts a stream event by whether it should revive a dormant session.
type EventClass int

const (
	// ClassNotice events persist but never revive a dormant session; only a live
	// awaiting session reacts to them.
	ClassNotice EventClass = iota
	// ClassInput events revive any non-ongoing session (and advance an idle one).
	ClassInput
)

// Classify sorts a stream event into its class (see docs/session_lifecycle.md):
//
//   - UserEvent (operator/user input) → Input.
//   - MessageEvent (agent send_message OR environment $AMPLIO_NOTIFY) → Input.
//     Both are Input at the event level. NOTE: the wake path
//     (runtime.NewCommitNotifier) additionally refuses to revive a *finished*
//     session (concluded/crashed/cancelled) from an *environment* notification
//     — the environment can't resurrect a deliberately-finished/stopped agent
//     (see docs/session_lifecycle.md). Agent send_messages still revive them.
//     Classify stays a pure function of the event; that state-dependent gate
//     (and its own status set) lives at the wake path, not here.
//   - ChildResultEvent verdict concluded → Input; crashed/cancelled → Notice
//     (a failing child must not hammer a dormant parent back to life).
//   - RecoverEvent (produced only by Recover) → Input.
//   - all other (self/system writes) → Notice.
func Classify(evt event.Event) EventClass {
	switch e := evt.(type) {
	case *event.UserEvent:
		return ClassInput
	case *event.MessageEvent:
		// Input at the event level for both agent (send_message) and environment
		// ($AMPLIO_NOTIFY) messages. The env-notification-vs-TERMINAL-session
		// exception is enforced at the wake path (runtime.NewCommitNotifier), which
		// has the target's status; Classify stays pure on the event.
		return ClassInput
	case *event.ChildResultEvent:
		if e.Verdict == SessionConcluded {
			return ClassInput
		}
		return ClassNotice
	case *event.RecoverEvent:
		return ClassInput
	default:
		return ClassNotice
	}
}

// IsInput reports whether an event revives a dormant (non-ongoing) session.
func IsInput(evt event.Event) bool { return Classify(evt) == ClassInput }

// --- Record types ---

// RunRecord holds the persistent identity and configuration of a run.
// Run liveness (is the server actively managing this run?) is tracked
// in-memory by the server, not in the DB. Resume-after-crash is derived
// from session states (any non-terminal root session = needs resume).
type RunRecord struct {
	RunID  string
	Config config.RunConfig
	// User-editable overlay (single-user; folded in from the old RunAnnotation).
	Title   string
	Note    string
	Starred bool
	// Grade is the human grade (0=ungraded, 1=garbage..5=excellent). When
	// nonzero it overrides ReportGrade, the grade denormalized from the
	// keen-critic run report (also 0=ungraded, 1..5).
	Grade       int
	ReportGrade int
	Archived    bool
	CreatedAt   time.Time
	UpdatedAt   time.Time
	// LastSeenAt is when the operator last viewed this run (zero = never seen).
	// Single-operator, so this is global, not per-viewer. Drives the dashboard
	// "has updates" badge.
	LastSeenAt time.Time
}

// RunWithSessions is the view returned by ListRunsWithSessions.
// Contains the run plus its root-level sessions (parent_id = ”).
type RunWithSessions struct {
	Run          RunRecord
	RootSessions []SessionRecord // sessions with no parent
}

type SessionRecord struct {
	SessionID         string
	RunID             string
	AgentType         string
	Task              string
	Status            string
	CurrentStep       int
	CurrentGeneration int
	Metadata          map[string]any
	ParentID          string
	CreatedAt         time.Time
	StatusChangedAt   time.Time // bumped only on status transitions

	// Observer cursors (monotonic, non-decreasing): LastFinalizedStep is the
	// highest step whose events are fully written (producer, set by
	// FinalizeStep); LastSummarizedStep is the highest step-summarized step;
	// LastPhasedStep is the highest phase-closed step. Invariant:
	// LastFinalizedStep >= LastSummarizedStep >= LastPhasedStep. Step work is
	// pending iff finalized > summarized; phase work iff summarized > phased.
	LastFinalizedStep  int
	LastSummarizedStep int
	LastPhasedStep     int
}

// PendingSummary identifies a session with pending observer work (step and/or
// phase), with its current cursors.
type PendingSummary struct {
	RunID              string
	SessionID          string
	LastFinalizedStep  int
	LastSummarizedStep int
	LastPhasedStep     int
}

// IsSpine reports whether crash recovery should auto-resume this session: it was
// actively working (ongoing), waiting (awaiting), or is a crashed root with no
// parent to retry it. The observer's force-close set is the complement of this
// (minus idle) — keeping both rules here prevents drift.
func IsSpine(s SessionRecord) bool {
	switch s.Status {
	case SessionOngoing, SessionAwaiting:
		return true
	case SessionCrashed:
		return s.ParentID == ""
	default:
		return false
	}
}

type EventRecord struct {
	EventID    string
	SessionID  string
	RunID      string
	Step       int
	Generation int
	Event      event.Event
	CreatedAt  time.Time
}

// SummarizationFailedPrefix marks a step_summary whose LLM summarization failed:
// the observer writes a degraded payload whose "summary" starts with this prefix
// (followed by the reason) instead of a real summary. Shared by the writer
// (observer) and readers (session summary, critic) so the sentinel stays
// consistent. Readers should treat such a summary as "no summary" and fall back
// to showing the raw step.
const SummarizationFailedPrefix = "[summarization failed]"

// IsSummarizationFailure reports whether a step_summary string is the observer's
// degraded failure payload (see SummarizationFailedPrefix).
func IsSummarizationFailure(summary string) bool {
	return strings.HasPrefix(strings.TrimSpace(summary), SummarizationFailedPrefix)
}

type ObservationRecord struct {
	ObsID     string
	RunID     string
	Kind      string
	SessionID string
	Step      *int
	CharCount int // raw-event char count of the summarized step (step_summary only)
	Data      map[string]any
	CreatedAt time.Time
}

type LessonRecord struct {
	LessonID    string
	Title       string
	Description string
	Body        string
	Embedding   []float32
	EmbedderID  string
	SourceRunID string
	Score       int
	LoadedCount int
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// --- Filter / option types ---

type ListRunsOpts struct {
	// Before + BeforeRunID are the KEYSET pagination cursor: ListRuns returns rows
	// strictly older than (Before, BeforeRunID) in the newest-first ordering
	// (created_at DESC, run_id DESC). A zero Before means the first page. run_id (a
	// unique random id) is the tiebreaker so runs sharing a millisecond created_at
	// can't be skipped or duplicated at a page boundary — the failure mode of the
	// prior OFFSET pagination under concurrent run creation. run_id is never
	// time-ordered (it's a random hash), so created_at remains the sort key; run_id
	// only disambiguates ties.
	Before      time.Time
	BeforeRunID string
	// Limit caps the page size. 0 (unset) uses a default page; a NEGATIVE limit
	// means UNBOUNDED — return every matching run, used by full-table sweeps
	// (recovery, report backfill) that must see all runs, not a page.
	Limit        int
	ShowArchived bool // when false (default), archived runs are excluded
	// StatusFilter, when non-empty, restricts to a primary-root status GROUP so
	// pagination is over the matching set (matches the dashboard's deep-link
	// filters): "active" (ongoing/awaiting), "done" (concluded), "failed"
	// (crashed/cancelled), or "updates" (a relevant root changed since last seen).
	// Empty = no status filter. Unknown values match nothing.
	StatusFilter string
	// Search, when non-empty, restricts to runs matching a case-insensitive query
	// over the run's title, run_id, task, and workspace path. Whitespace splits it
	// into terms that are ANDed (each term must appear in at least one field).
	Search string
	// StarredOnly, when true, restricts to starred runs.
	StarredOnly bool
	// GradeFilter restricts by EFFECTIVE grade (the human grade if set, else the
	// critic report grade). GradeNone = no filter; GradeUngraded matches runs with
	// neither grade set; a rank 1..5 matches that exact effective grade. The
	// grade-name vocabulary lives in the server layer, which resolves it to this
	// enum before calling the store.
	GradeFilter GradeFilter
	// All of the above compose (AND) with each other and with ShowArchived.
}

// GradeFilter is the resolved effective-grade filter for ListRunsOpts.
// 0 (GradeNone) = no filter; -1 (GradeUngraded) = ungraded; 1..5 = that rank.
type GradeFilter int

const (
	GradeNone     GradeFilter = 0  // no grade filter
	GradeUngraded GradeFilter = -1 // neither human nor report grade set
)

// Status-filter group values for ListRunsOpts.StatusFilter. Mirror the
// dashboard's URL ?filter= groups so server counts and list pagination agree.
const (
	RunFilterActive  = "active"
	RunFilterDone    = "done"
	RunFilterFailed  = "failed"
	RunFilterUpdates = "updates"
)

// RunCounts is the dashboard banner's exact, global tally over NON-archived
// runs (independent of pagination). Active = primary root ongoing/awaiting;
// Updates = a relevant root status changed since the run was last seen.
type RunCounts struct {
	Active  int
	Updates int
}

// SkillVector is a cached skill embedding + the parsed SKILL.md fields the
// in-memory Index needs at search/load time. Caching Description/Path/Body
// lets the Index hydrate fully from the DB at startup without touching the
// (slow, cloud-backed) skill source FS; a background reconcile updates them
// when the file's ContentHash changes.
type SkillVector struct {
	Name        string
	ContentHash string
	Vector      []float32
	Description string // SKILL.md frontmatter — recall_search preview text
	Path        string // absolute path to the SKILL.md — recall_load reference
	Body        string // SKILL.md body — recall_load content (kept in cache so a load doesn't pay srcfs latency)
}

type EventFilter struct {
	CurrentContextOnly bool
	StartStep          *int
	EndStep            *int
}

type SearchOpts struct {
	SessionID string
	StepMin   *int
	StepMax   *int
	Limit     int
}

type ObsFilter struct {
	Kind      string
	SessionID string
	StartStep *int
	EndStep   *int
}

type LessonBrowseOpts struct {
	SortBy string
	Order  string
	Limit  int
	Offset int
}

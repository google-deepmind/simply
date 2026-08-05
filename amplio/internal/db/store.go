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
	"context"
	"crypto/rand"
	"strings"

	"amplio/internal/event"

	"github.com/google/uuid"
)

// Store is the persistence interface for all amplio state. A single Store
// instance is shared across the entire process — web server, agent goroutines,
// observer goroutines all use the same Store.
//
// Implementations must be safe for concurrent use by multiple goroutines.
// The SQLite implementation achieves this via WAL mode + a serializing mutex
// on writes, or a dedicated owner goroutine.
//
// Write methods emit StoreEvents on the channel returned by Events().
// Consumers (SessionRegistry, EventBus) select on this channel for
// notification wakeups — no callbacks, no locks across boundaries.
type Store interface {
	// --- Run operations ---

	CreateRun(ctx context.Context, run RunRecord) error
	GetRun(ctx context.Context, runID string) (*RunRecord, error)
	ListRuns(ctx context.Context, opts ListRunsOpts) (runs []RunRecord, hasMore bool, err error)

	// Run annotation: targeted single-field updates of the user-editable overlay
	// (title/note/starred/grade/archived). Each bumps Run.updated_at and emits
	// RunUpdated.
	UpdateRunTitle(ctx context.Context, runID, title string) error
	UpdateRunNote(ctx context.Context, runID, note string) error
	SetRunStarred(ctx context.Context, runID string, starred bool) error
	// SetRunGrade sets the human grade (0=ungraded, 1=garbage..5=excellent),
	// which overrides the cached report grade when nonzero.
	SetRunGrade(ctx context.Context, runID string, grade int) error
	// SetRunReportGrade caches the keen-critic report grade onto the run row; the
	// human grade overrides it when set. Called by the critic finalizer.
	SetRunReportGrade(ctx context.Context, runID string, grade int) error
	SetRunArchived(ctx context.Context, runID string, archived bool) error
	// MarkRunSeen records that the operator viewed the run now, clearing its
	// dashboard "has updates" badge. Single-operator: global, not per-viewer.
	MarkRunSeen(ctx context.Context, runID string) error
	// MarkRunUnseen puts the run's badge back: an operator who looked at a run
	// but hasn't finished with it says so, and finds it again later the same way
	// they found it the first time. Rewinds last_seen_at to the run's creation,
	// so "unseen" means what it means for a run nobody has opened.
	MarkRunUnseen(ctx context.Context, runID string) error
	// DeleteRun permanently removes a run and ALL its DB rows (the Run row plus
	// its Sessions, Events, and Observations) in one transaction. Mined Lessons
	// (which carry source_run_id as a soft reference) are intentionally kept —
	// they outlive the run they came from. On-disk artifact/blob dirs are NOT
	// touched here; the caller removes those. Idempotent: deleting an unknown run
	// is a no-op, not an error.
	DeleteRun(ctx context.Context, runID string) error
	// RunCounts returns the exact, global dashboard tally over non-archived runs
	// (active + has-updates), independent of list pagination.
	RunCounts(ctx context.Context) (RunCounts, error)

	// --- Custom model menu (user-added models; the app-owned half of the menu) ---
	ListCustomModels(ctx context.Context) ([]string, error)
	AddCustomModel(ctx context.Context, spec string) error
	RemoveCustomModel(ctx context.Context, spec string) error
	// ListRunsWithSessions returns runs joined with their root-level sessions
	// (parent_id = ''). Used by the dashboard to show per-run status derived
	// from session states + notification dots via status_changed_at.
	ListRunsWithSessions(ctx context.Context, opts ListRunsOpts) ([]RunWithSessions, bool, error)

	// --- Session operations ---

	CreateSession(ctx context.Context, sess SessionRecord) error
	GetSession(ctx context.Context, runID, sessionID string) (*SessionRecord, error)
	ListSessions(ctx context.Context, runID string) ([]SessionRecord, error)
	GetChildSessions(ctx context.Context, runID, parentID string) ([]SessionRecord, error)
	UpdateSessionStatus(ctx context.Context, runID, sessionID, status string) error

	// TerminateAndNotifyParent atomically transitions a session to a terminal
	// status (concluded/crashed/cancelled), optionally writes selfEvent to the
	// session's own stream (e.g. a cancel/error SystemEvent marker), and — if
	// parentID is non-empty — appends a ChildResultEvent (verdict=status,
	// content) to the parent's stream, all in one transaction.
	//
	// selfEventStep pins the step for selfEvent: a negative value uses the
	// session's current step (the natural place for an async cancel marker); a
	// non-negative value places it at that step (e.g. an agent records a turn's
	// crash at the turn's call step, not the bumped current step).
	//
	// The transition is conditional: if the session is ALREADY terminal the call
	// is a no-op (no writes, no notify). Status is thus the dedupe token. The
	// commit handler fires for the PARENT only — never for the terminating
	// session itself, so it never self-wakes the session being stopped.
	TerminateAndNotifyParent(ctx context.Context, runID, sessionID, parentID, status, content string, selfEvent event.Event, selfEventStep int) error

	MergeSessionMetadata(ctx context.Context, runID, sessionID string, updates map[string]any) error

	// --- Event operations ---

	// AppendEvent writes an event at the session's current step and returns the
	// generated event id (so create-style callers can identify what they wrote
	// without re-reading the stream).
	AppendEvent(ctx context.Context, runID, sessionID string, evt event.Event) (string, error)

	// AppendEventAtStep writes an event at a specific step, regardless of
	// current_step. Used by the event loop to write the AssistantEvent of a
	// tool-calling step (before tools run); the results land via FinalizeStep.
	AppendEventAtStep(ctx context.Context, runID, sessionID string, step int, evt event.Event) error

	// FinalizeStep atomically appends the step's remaining events (tool results,
	// or the lone assistant of a no-tool turn), bumps last_finalized_step to the
	// max of its current value and step, and — after commit — fires the
	// StepFinalized hook. This is the single point that marks a step "complete":
	// all of its events are durably present and it is ready to summarize.
	FinalizeStep(ctx context.Context, runID, sessionID string, step int, events []event.Event) error

	// MarkStepFinalized finalizes a step whose events were already written
	// individually — tool results appended one-by-one as each tool finished, or
	// results synthesized during crash recovery. It bumps last_finalized_step to
	// max(current, step) and fires the StepFinalized hook, without writing any
	// events. Idempotent. (FinalizeStep is the same cursor bump bundled with an
	// atomic event write; use it when the events and the finalize must be one
	// transaction, e.g. the lone assistant of a no-tool turn.)
	MarkStepFinalized(ctx context.Context, runID, sessionID string, step int) error

	// AdvanceStep atomically increments the session's current_step.
	// Returns the new step number.
	AdvanceStep(ctx context.Context, runID, sessionID string) (int, error)

	// GetEvents returns events for a session, filtered by opts.
	GetEvents(ctx context.Context, runID, sessionID string, opts EventFilter) ([]EventRecord, error)

	// GetEventCount returns the number of events matching the filter.
	GetEventCount(ctx context.Context, runID, sessionID string, opts EventFilter) (int, error)

	// CountEnvNotices returns how many environment notifications ($AMPLIO_NOTIFY
	// MessageEvents) are already recorded at the session's CURRENT step. It backs
	// the notify flood guard (server.handleNotify); the step is resolved inside
	// the same statement so the count cannot be read against a step the session
	// has since advanced past.
	CountEnvNotices(ctx context.Context, runID, sessionID string) (int, error)

	// GetTailEvent returns the most recent event in a session, or nil.
	GetTailEvent(ctx context.Context, runID, sessionID string) (*EventRecord, error)

	// SearchEvents performs full-text search over events in a run.
	SearchEvents(ctx context.Context, runID, query string, opts SearchOpts) ([]EventRecord, error)

	// CompactContext bumps the session's generation, inserts a CompactionEvent
	// carrying the summary at boundaryStep, and carries every current-generation
	// event after boundaryStep forward into the new generation (so fresh inputs
	// that arrived after the boundary survive verbatim). With the current-context
	// query (step 0 OR current generation), this keeps exactly: bootstrap, the
	// latest summary, and the post-boundary tail. Returns the new generation.
	CompactContext(ctx context.Context, runID, sessionID string, boundaryStep int, summary string) (int, error)

	// --- Skill embedding cache ---

	// GetSkillVectors returns all cached skill embeddings for an embedder model.
	GetSkillVectors(ctx context.Context, model string) ([]SkillVector, error)
	// PutSkillVectors upserts skill embeddings for a model (the full current set).
	PutSkillVectors(ctx context.Context, model string, vectors []SkillVector) error

	// --- Observation operations ---

	AppendObservation(ctx context.Context, obs ObservationRecord) error
	GetObservations(ctx context.Context, runID string, opts ObsFilter) ([]ObservationRecord, error)
	HasObservation(ctx context.Context, runID, kind string) (bool, error)
	MaxObservationStep(ctx context.Context, runID, sessionID, kind string) (*int, error)

	// SetLastSummarizedStep advances the session's step-summary consumer cursor
	// to max(current, step). Bumped by the observer after summarizing a step.
	SetLastSummarizedStep(ctx context.Context, runID, sessionID string, step int) error

	// SetLastPhasedStep advances the session's phase consumer cursor to
	// max(current, step). Bumped by the observer after closing a phase.
	SetLastPhasedStep(ctx context.Context, runID, sessionID string, step int) error

	// SumStepSummaryChars returns the total char_count of this session's
	// step_summary rows in (stepLowerExclusive, stepUpperInclusive] — the phase
	// trigger's accumulator.
	SumStepSummaryChars(ctx context.Context, runID, sessionID string, stepLowerExclusive, stepUpperInclusive int) (int, error)

	// ListPendingSessions returns every session (across all runs) with pending
	// observer work: last_finalized_step > last_summarized_step (step work) OR
	// last_summarized_step > last_phased_step (phase work). Backed by partial
	// indexes, so the cost is O(dirty), not O(history). Used for observer
	// startup catch-up and final drain.
	ListPendingSessions(ctx context.Context) ([]PendingSummary, error)

	// --- Lesson operations ---

	InsertLesson(ctx context.Context, lesson LessonRecord) error
	GetLesson(ctx context.Context, lessonID string) (*LessonRecord, error)
	ListAllLessons(ctx context.Context) ([]LessonRecord, error)
	// UpdateLesson replaces a lesson's content (title/description/body/embedding/
	// embedder_id) and bumps updated_at, keeping its id, counters (score,
	// loaded_count), and created_at. Used by the supersede dedup verdict.
	UpdateLesson(ctx context.Context, lesson LessonRecord) error
	IncrementLessonLoadCount(ctx context.Context, lessonID string, delta int) error
	AddToLessonScore(ctx context.Context, lessonID string, delta int) error

	// --- Lifecycle ---

	// SetCommitListener installs a synchronous, lossless hook invoked after
	// each event append commits (outside the store lock). This is the wake
	// path: the runtime wires it to bump a live session's waiter or respawn a
	// cold one. Set once at startup before any runs are launched.
	SetCommitListener(fn CommitListener)

	// SetStepFinalizedListener installs a hook fired (outside the store lock)
	// after a step is finalized via FinalizeStep. The global observer wires it
	// to summarize the step on demand. Best-effort: a dropped signal is caught
	// by the durable last_finalized_step cursor on the next sweep. Set once at
	// startup.
	SetStepFinalizedListener(fn StepFinalizedListener)

	// SetSessionStatusChangedListener installs a hook fired (outside the store
	// lock) after a session's status changes. The observer wires it to
	// force-close the trailing phase when a session settles (concluded /
	// cancelled / crashed sub-agent). Set once at startup.
	SetSessionStatusChangedListener(fn SessionStatusListener)

	// Events returns the channel on which StoreEvents are published after each
	// successful write. This is a best-effort, lossy feed for non-critical
	// observers (UI, logging) — NOT the wake path (see SetCommitListener).
	Events() <-chan StoreEvent

	Close() error
}

// CommitListener is invoked synchronously after an event append commits, with
// the appended event. It must not block and must not call back into the store
// in a way that re-enters the append path on the same goroutine.
type CommitListener func(runID, sessionID string, evt event.Event)

// StepFinalizedListener is invoked after a step is finalized. It must not block
// (the observer enqueues and returns).
type StepFinalizedListener func(runID, sessionID string, step int)

// SessionStatusListener is invoked after a session's status changes, with the
// new status. It must not block.
type SessionStatusListener func(runID, sessionID, newStatus string)

// --- ID generators ---

// runIDAlphabet is lowercase Crockford base32 (no i/l/o/u): 32 symbols → 5 bits
// each. Case-insensitive so run ids are safe as directory names on
// case-insensitive filesystems (artifacts/<id>, blobs/<id>) and easy to dictate.
const runIDAlphabet = "0123456789abcdefghjkmnpqrstvwxyz"

// runIDLen is 8 → 40 bits of entropy: ample for a single user's runs, with a
// CreateRun retry guarding the astronomically-rare collision. Ordering is by
// created_at, so the id needn't encode time.
const runIDLen = 8

func NewRunID() string {
	b := make([]byte, runIDLen)
	_, _ = rand.Read(b) // crypto/rand.Read never returns a short read / error in practice
	for i, c := range b {
		b[i] = runIDAlphabet[c&31] // 256 % 32 == 0 → unbiased
	}
	return string(b)
}

// IsUniqueViolation reports whether err is a SQL UNIQUE/PRIMARY-KEY conflict,
// used to retry on the (effectively impossible) run-id collision.
func IsUniqueViolation(err error) bool {
	return err != nil && strings.Contains(err.Error(), "UNIQUE constraint")
}

func NewEventID() string {
	return uuid.New().String()
}

// NewLessonID returns a 12-hex-char id.
func NewLessonID() string {
	u := uuid.New().String() // 8-4-4-4-12 hex with dashes
	return u[:8] + u[9:13]   // 12 hex chars, dash at index 8 skipped
}

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
	"errors"
	"fmt"

	"amplio/internal/event"
)

// ErrStore marks an error as originating from the Store layer (a DB operation),
// as opposed to the work the caller was doing (an LLM call, a tool, etc.).
//
// Callers test for it with errors.Is(err, db.ErrStore). The canonical use is
// the agent loop's failure path: when a turn fails because the DB itself is
// broken, writing a crash record (another DB write) would just fail again — so
// the loop leaves the session ongoing for crash recovery instead of attempting
// the doomed write. This replaces brittle error-message substring matching:
// every error returned through the Store interface wraps ErrStore (see Tag), so
// classification can never miss a DB error or mislabel a non-DB one.
var ErrStore = errors.New("db store")

// Tag wraps a Store so that every error it returns wraps ErrStore. Install it
// once at the boundary (the concrete store's constructor returns Tag(impl)), so
// the rest of the program sees a Store whose errors are uniformly classifiable
// with errors.Is(err, ErrStore). Non-error return values pass through unchanged.
func Tag(s Store) Store { return &taggedStore{s} }

// tag wraps a non-nil error so errors.Is(err, ErrStore) is true while
// preserving the original error (and its own wrapped chain) for messages and
// further unwrapping. nil passes through.
func tag(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w: %w", ErrStore, err)
}

// taggedStore is a transparent decorator: it delegates every method to the
// underlying Store and runs returned errors through tag(). The delegation is
// mechanical and exhaustive — the compiler enforces (via the Store interface)
// that a newly added method is wired here too.
type taggedStore struct{ s Store }

// --- Run operations ---

func (t *taggedStore) CreateRun(ctx context.Context, run RunRecord) error {
	return tag(t.s.CreateRun(ctx, run))
}

func (t *taggedStore) GetRun(ctx context.Context, runID string) (*RunRecord, error) {
	r, err := t.s.GetRun(ctx, runID)
	return r, tag(err)
}

func (t *taggedStore) ListRuns(ctx context.Context, opts ListRunsOpts) ([]RunRecord, bool, error) {
	runs, hasMore, err := t.s.ListRuns(ctx, opts)
	return runs, hasMore, tag(err)
}

func (t *taggedStore) UpdateRunTitle(ctx context.Context, runID, title string) error {
	return tag(t.s.UpdateRunTitle(ctx, runID, title))
}

func (t *taggedStore) UpdateRunNote(ctx context.Context, runID, note string) error {
	return tag(t.s.UpdateRunNote(ctx, runID, note))
}

func (t *taggedStore) SetRunStarred(ctx context.Context, runID string, starred bool) error {
	return tag(t.s.SetRunStarred(ctx, runID, starred))
}

func (t *taggedStore) SetRunGrade(ctx context.Context, runID string, grade int) error {
	return tag(t.s.SetRunGrade(ctx, runID, grade))
}

func (t *taggedStore) SetRunReportGrade(ctx context.Context, runID string, grade int) error {
	return tag(t.s.SetRunReportGrade(ctx, runID, grade))
}

func (t *taggedStore) SetRunArchived(ctx context.Context, runID string, archived bool) error {
	return tag(t.s.SetRunArchived(ctx, runID, archived))
}

func (t *taggedStore) MarkRunSeen(ctx context.Context, runID string) error {
	return tag(t.s.MarkRunSeen(ctx, runID))
}

func (t *taggedStore) MarkRunUnseen(ctx context.Context, runID string) error {
	return tag(t.s.MarkRunUnseen(ctx, runID))
}

func (t *taggedStore) DeleteRun(ctx context.Context, runID string) error {
	return tag(t.s.DeleteRun(ctx, runID))
}

func (t *taggedStore) RunCounts(ctx context.Context) (RunCounts, error) {
	c, err := t.s.RunCounts(ctx)
	return c, tag(err)
}

// --- Custom model menu ---

func (t *taggedStore) ListCustomModels(ctx context.Context) ([]string, error) {
	m, err := t.s.ListCustomModels(ctx)
	return m, tag(err)
}

func (t *taggedStore) AddCustomModel(ctx context.Context, spec string) error {
	return tag(t.s.AddCustomModel(ctx, spec))
}

func (t *taggedStore) RemoveCustomModel(ctx context.Context, spec string) error {
	return tag(t.s.RemoveCustomModel(ctx, spec))
}

func (t *taggedStore) ListRunsWithSessions(ctx context.Context, opts ListRunsOpts) ([]RunWithSessions, bool, error) {
	r, hasMore, err := t.s.ListRunsWithSessions(ctx, opts)
	return r, hasMore, tag(err)
}

// --- Session operations ---

func (t *taggedStore) CreateSession(ctx context.Context, sess SessionRecord) error {
	return tag(t.s.CreateSession(ctx, sess))
}

func (t *taggedStore) GetSession(ctx context.Context, runID, sessionID string) (*SessionRecord, error) {
	r, err := t.s.GetSession(ctx, runID, sessionID)
	return r, tag(err)
}

func (t *taggedStore) ListSessions(ctx context.Context, runID string) ([]SessionRecord, error) {
	r, err := t.s.ListSessions(ctx, runID)
	return r, tag(err)
}

func (t *taggedStore) GetChildSessions(ctx context.Context, runID, parentID string) ([]SessionRecord, error) {
	r, err := t.s.GetChildSessions(ctx, runID, parentID)
	return r, tag(err)
}

func (t *taggedStore) UpdateSessionStatus(ctx context.Context, runID, sessionID, status string) error {
	return tag(t.s.UpdateSessionStatus(ctx, runID, sessionID, status))
}

func (t *taggedStore) TerminateAndNotifyParent(ctx context.Context, runID, sessionID, parentID, status, content string, selfEvent event.Event, selfEventStep int) error {
	return tag(t.s.TerminateAndNotifyParent(ctx, runID, sessionID, parentID, status, content, selfEvent, selfEventStep))
}

func (t *taggedStore) MergeSessionMetadata(ctx context.Context, runID, sessionID string, updates map[string]any) error {
	return tag(t.s.MergeSessionMetadata(ctx, runID, sessionID, updates))
}

// --- Event operations ---

func (t *taggedStore) AppendEvent(ctx context.Context, runID, sessionID string, evt event.Event) (string, error) {
	id, err := t.s.AppendEvent(ctx, runID, sessionID, evt)
	return id, tag(err)
}

func (t *taggedStore) AppendEventAtStep(ctx context.Context, runID, sessionID string, step int, evt event.Event) error {
	return tag(t.s.AppendEventAtStep(ctx, runID, sessionID, step, evt))
}

func (t *taggedStore) FinalizeStep(ctx context.Context, runID, sessionID string, step int, events []event.Event) error {
	return tag(t.s.FinalizeStep(ctx, runID, sessionID, step, events))
}

func (t *taggedStore) MarkStepFinalized(ctx context.Context, runID, sessionID string, step int) error {
	return tag(t.s.MarkStepFinalized(ctx, runID, sessionID, step))
}

func (t *taggedStore) AdvanceStep(ctx context.Context, runID, sessionID string) (int, error) {
	n, err := t.s.AdvanceStep(ctx, runID, sessionID)
	return n, tag(err)
}

func (t *taggedStore) GetEvents(ctx context.Context, runID, sessionID string, opts EventFilter) ([]EventRecord, error) {
	r, err := t.s.GetEvents(ctx, runID, sessionID, opts)
	return r, tag(err)
}

func (t *taggedStore) GetEventCount(ctx context.Context, runID, sessionID string, opts EventFilter) (int, error) {
	n, err := t.s.GetEventCount(ctx, runID, sessionID, opts)
	return n, tag(err)
}

func (t *taggedStore) CountEnvNotices(ctx context.Context, runID, sessionID string) (int, error) {
	n, err := t.s.CountEnvNotices(ctx, runID, sessionID)
	return n, tag(err)
}

func (t *taggedStore) GetTailEvent(ctx context.Context, runID, sessionID string) (*EventRecord, error) {
	r, err := t.s.GetTailEvent(ctx, runID, sessionID)
	return r, tag(err)
}

func (t *taggedStore) SearchEvents(ctx context.Context, runID, query string, opts SearchOpts) ([]EventRecord, error) {
	r, err := t.s.SearchEvents(ctx, runID, query, opts)
	return r, tag(err)
}

func (t *taggedStore) CompactContext(ctx context.Context, runID, sessionID string, boundaryStep int, summary string) (int, error) {
	n, err := t.s.CompactContext(ctx, runID, sessionID, boundaryStep, summary)
	return n, tag(err)
}

// --- Skill embedding cache ---

func (t *taggedStore) GetSkillVectors(ctx context.Context, model string) ([]SkillVector, error) {
	r, err := t.s.GetSkillVectors(ctx, model)
	return r, tag(err)
}

func (t *taggedStore) PutSkillVectors(ctx context.Context, model string, vectors []SkillVector) error {
	return tag(t.s.PutSkillVectors(ctx, model, vectors))
}

// --- Observation operations ---

func (t *taggedStore) AppendObservation(ctx context.Context, obs ObservationRecord) error {
	return tag(t.s.AppendObservation(ctx, obs))
}

func (t *taggedStore) GetObservations(ctx context.Context, runID string, opts ObsFilter) ([]ObservationRecord, error) {
	r, err := t.s.GetObservations(ctx, runID, opts)
	return r, tag(err)
}

func (t *taggedStore) HasObservation(ctx context.Context, runID, kind string) (bool, error) {
	b, err := t.s.HasObservation(ctx, runID, kind)
	return b, tag(err)
}

func (t *taggedStore) MaxObservationStep(ctx context.Context, runID, sessionID, kind string) (*int, error) {
	n, err := t.s.MaxObservationStep(ctx, runID, sessionID, kind)
	return n, tag(err)
}

func (t *taggedStore) SetLastSummarizedStep(ctx context.Context, runID, sessionID string, step int) error {
	return tag(t.s.SetLastSummarizedStep(ctx, runID, sessionID, step))
}

func (t *taggedStore) SetLastPhasedStep(ctx context.Context, runID, sessionID string, step int) error {
	return tag(t.s.SetLastPhasedStep(ctx, runID, sessionID, step))
}

func (t *taggedStore) SumStepSummaryChars(ctx context.Context, runID, sessionID string, stepLowerExclusive, stepUpperInclusive int) (int, error) {
	n, err := t.s.SumStepSummaryChars(ctx, runID, sessionID, stepLowerExclusive, stepUpperInclusive)
	return n, tag(err)
}

func (t *taggedStore) ListPendingSessions(ctx context.Context) ([]PendingSummary, error) {
	r, err := t.s.ListPendingSessions(ctx)
	return r, tag(err)
}

// --- Lesson operations ---

func (t *taggedStore) InsertLesson(ctx context.Context, lesson LessonRecord) error {
	return tag(t.s.InsertLesson(ctx, lesson))
}

func (t *taggedStore) GetLesson(ctx context.Context, lessonID string) (*LessonRecord, error) {
	r, err := t.s.GetLesson(ctx, lessonID)
	return r, tag(err)
}

func (t *taggedStore) ListAllLessons(ctx context.Context) ([]LessonRecord, error) {
	r, err := t.s.ListAllLessons(ctx)
	return r, tag(err)
}

func (t *taggedStore) UpdateLesson(ctx context.Context, lesson LessonRecord) error {
	return tag(t.s.UpdateLesson(ctx, lesson))
}

func (t *taggedStore) IncrementLessonLoadCount(ctx context.Context, lessonID string, delta int) error {
	return tag(t.s.IncrementLessonLoadCount(ctx, lessonID, delta))
}

func (t *taggedStore) AddToLessonScore(ctx context.Context, lessonID string, delta int) error {
	return tag(t.s.AddToLessonScore(ctx, lessonID, delta))
}

// --- Lifecycle (no error returns; pure pass-through) ---

func (t *taggedStore) SetCommitListener(fn CommitListener) { t.s.SetCommitListener(fn) }

func (t *taggedStore) SetStepFinalizedListener(fn StepFinalizedListener) {
	t.s.SetStepFinalizedListener(fn)
}

func (t *taggedStore) SetSessionStatusChangedListener(fn SessionStatusListener) {
	t.s.SetSessionStatusChangedListener(fn)
}

func (t *taggedStore) Events() <-chan StoreEvent { return t.s.Events() }

func (t *taggedStore) Close() error { return tag(t.s.Close()) }

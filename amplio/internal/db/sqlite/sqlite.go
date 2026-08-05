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

package sqlite

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/event"

	_ "modernc.org/sqlite" // register sqlite driver
)

// commitListenerHolder boxes a db.CommitListener so it can be stored in an
// atomic.Pointer (func values aren't comparable / atomically storable directly).
type commitListenerHolder struct {
	fn db.CommitListener
}

type stepFinalizedHolder struct {
	fn db.StepFinalizedListener
}

type sessionStatusHolder struct {
	fn db.SessionStatusListener
}

// sqliteStore implements db.Store for a single SQLite database file.
type sqliteStore struct {
	sqlDB          *sql.DB
	mu             sync.Mutex // serializes all writes
	events         chan db.StoreEvent
	eventsMu       sync.RWMutex // guards events send vs close (emit/Close)
	eventsClosed   bool
	closeOnce      sync.Once
	listener       atomic.Pointer[commitListenerHolder]
	stepListener   atomic.Pointer[stepFinalizedHolder]
	statusListener atomic.Pointer[sessionStatusHolder]
}

func (s *sqliteStore) SetCommitListener(fn db.CommitListener) {
	s.listener.Store(&commitListenerHolder{fn: fn})
}

// fireCommit invokes the commit listener (if installed) synchronously. It MUST
// be called after the store mutex is released so the listener can re-enter the
// store (e.g. respawn reading a session) without deadlocking.
func (s *sqliteStore) fireCommit(runID, sessionID string, evt event.Event) {
	if h := s.listener.Load(); h != nil && h.fn != nil {
		h.fn(runID, sessionID, evt)
	}
}

func (s *sqliteStore) SetStepFinalizedListener(fn db.StepFinalizedListener) {
	s.stepListener.Store(&stepFinalizedHolder{fn: fn})
}

func (s *sqliteStore) fireStepFinalized(runID, sessionID string, step int) {
	if h := s.stepListener.Load(); h != nil && h.fn != nil {
		h.fn(runID, sessionID, step)
	}
}

func (s *sqliteStore) SetSessionStatusChangedListener(fn db.SessionStatusListener) {
	s.statusListener.Store(&sessionStatusHolder{fn: fn})
}

func (s *sqliteStore) fireSessionStatusChanged(runID, sessionID, newStatus string) {
	if h := s.statusListener.Load(); h != nil && h.fn != nil {
		h.fn(runID, sessionID, newStatus)
	}
}

var _ db.Store = (*sqliteStore)(nil)

// Open creates or opens a SQLite database at the given path and returns
// a db.Store. Use ":memory:" for tests.
func Open(path string) (db.Store, error) {
	sqlDB, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("sqlite open %s: %w", path, err)
	}
	// An in-memory database lives inside a single connection: database/sql pools
	// multiple connections, and each ":memory:" connection is a SEPARATE empty
	// database, so a concurrent read could hit an empty one. Pin to one
	// connection so all reads/writes share the same in-memory DB. (File-backed
	// DBs share the file across connections and keep WAL's read concurrency.)
	if path == ":memory:" {
		sqlDB.SetMaxOpenConns(1)
	}
	// WAL mode for concurrent reads during writes.
	for _, pragma := range []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA busy_timeout=5000",
		"PRAGMA foreign_keys=ON",
	} {
		if _, err := sqlDB.Exec(pragma); err != nil {
			sqlDB.Close()
			return nil, fmt.Errorf("sqlite pragma %q: %w", pragma, err)
		}
	}
	if _, err := sqlDB.Exec(schema); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("sqlite schema: %w", err)
	}
	if err := migrate(sqlDB); err != nil {
		sqlDB.Close()
		return nil, err
	}
	// Wrap at the boundary so every error returned through the Store interface
	// wraps db.ErrStore — callers classify DB failures via errors.Is, not by
	// matching error-message substrings.
	return db.Tag(&sqliteStore{
		sqlDB:  sqlDB,
		events: make(chan db.StoreEvent, 64),
	}), nil
}

// migrate applies additive schema changes that CREATE TABLE IF NOT EXISTS can't
// make to a pre-existing table. Idempotent: safe to run on every Open.
func migrate(sqlDB *sql.DB) error {
	// Add the observer step cursors to older Session tables. SQLite lacks ADD
	// COLUMN IF NOT EXISTS, so tolerate the duplicate-column error on fresh DBs
	// that already have them via CREATE TABLE.
	for _, stmt := range []string{
		`ALTER TABLE Session ADD COLUMN last_finalized_step INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE Session ADD COLUMN last_summarized_step INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE Session ADD COLUMN last_phased_step INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE Observation ADD COLUMN char_count INTEGER NOT NULL DEFAULT 0`,
		// Run annotation columns (folded in from the old RunAnnotation table).
		// ALTER ADD COLUMN forbids non-constant defaults, so updated_at is
		// nullable and written explicitly on insert/update.
		`ALTER TABLE Run ADD COLUMN title TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE Run ADD COLUMN note TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE Run ADD COLUMN starred INTEGER NOT NULL DEFAULT 0`,
		// Run grade: a human grade (0=ungraded, 1=garbage..5=excellent) and the
		// denormalized keen-critic report grade it overrides when nonzero.
		`ALTER TABLE Run ADD COLUMN grade INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE Run ADD COLUMN report_grade INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE Run ADD COLUMN archived INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE Run ADD COLUMN updated_at TEXT`,
		// Seen-state cursor for the dashboard "has updates" badge (single-operator,
		// so global). NULL/empty means SEEN (the `> NULL` predicate is falsy), so old
		// runs don't all flash the badge; the backfill below stamps legacy NULLs as
		// seen-at-upgrade so they can still badge on FUTURE transitions.
		`ALTER TABLE Run ADD COLUMN last_seen_at TEXT`,
		// Lesson recency cursor. Like Run.updated_at, the ALTER can't carry the
		// schema's strftime default (non-constant), so it lands nullable and is
		// backfilled below.
		`ALTER TABLE Lesson ADD COLUMN updated_at TEXT`,
		// SkillEmbedding gains description/path/body so the Index can hydrate
		// fully from cache without re-scanning the (cloud-backed, slow) skill
		// source FS at every startup. Empty defaults are backfilled by the
		// first post-migration successful PutSkillVectors (a normal reconcile).
		`ALTER TABLE SkillEmbedding ADD COLUMN description TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE SkillEmbedding ADD COLUMN path TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE SkillEmbedding ADD COLUMN body TEXT NOT NULL DEFAULT ''`,
	} {
		if _, err := sqlDB.Exec(stmt); err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("sqlite migrate: %w", err)
		}
	}
	// Backfill Lesson.updated_at for rows that predate the column (the lesson
	// scan reads it as a non-null string, so a bare NULL would break listing).
	// Idempotent: a no-op once every row has a value.
	if _, err := sqlDB.Exec(`UPDATE Lesson SET updated_at = created_at WHERE updated_at IS NULL`); err != nil {
		return fmt.Errorf("sqlite migrate: backfill lesson updated_at: %w", err)
	}
	// Backfill Run.last_seen_at for rows that predate the column. The semantics
	// is "NULL = seen" (so old runs don't all flash the has-updates badge), but a
	// bare NULL ALSO deadlocks the UI: the badge predicate is falsy for NULL, and
	// the client only calls markSeen when has_updates is already true — so a NULL
	// row can never self-heal, and a genuinely-new terminal transition on an old
	// run is suppressed forever. Stamp these as seen AS OF NOW (the upgrade): the
	// historical backlog stays quiet (all past transitions predate now), while any
	// FUTURE transition (status_changed_at > now) correctly raises the badge.
	// Deliberately now(), not created_at — created_at would mark every finished old
	// run as updated at once (the flood we want to avoid). Idempotent: only NULL/
	// empty rows are touched, and stamping makes them non-NULL.
	if _, err := sqlDB.Exec(
		`UPDATE Run SET last_seen_at = ? WHERE last_seen_at IS NULL OR last_seen_at = ''`,
		formatTime(time.Now()),
	); err != nil {
		return fmt.Errorf("sqlite migrate: backfill run last_seen_at: %w", err)
	}
	// Drop the vestigial Run.user column (an earlier multi-user schema; always
	// written "" and never read). Separate from the ADD loop because the
	// tolerated error differs: a fresh DB created from the current schema never
	// had the column, so DROP fails with "no such column" — which we ignore.
	// (DROP COLUMN needs SQLite >= 3.35; the bundled modernc.org/sqlite is far
	// newer.)
	if _, err := sqlDB.Exec(`ALTER TABLE Run DROP COLUMN user`); err != nil && !strings.Contains(err.Error(), "no such column") {
		return fmt.Errorf("sqlite migrate: drop Run.user: %w", err)
	}
	// Partial indexes over the dirty sets; created after the columns exist. Each
	// holds only in-flight sessions, so observer catch-up is O(dirty).
	for _, stmt := range []string{
		`CREATE INDEX IF NOT EXISTS idx_session_pending_summary ON Session(run_id, session_id)
		 WHERE last_finalized_step > last_summarized_step`,
		`CREATE INDEX IF NOT EXISTS idx_session_pending_phase ON Session(run_id, session_id)
		 WHERE last_summarized_step > last_phased_step`,
		// Serves the default dashboard listing (non-archived, newest first).
		`CREATE INDEX IF NOT EXISTS idx_run_active ON Run(created_at DESC) WHERE archived = 0`,
	} {
		if _, err := sqlDB.Exec(stmt); err != nil {
			return fmt.Errorf("sqlite migrate index: %w", err)
		}
	}
	return nil
}

func (s *sqliteStore) Events() <-chan db.StoreEvent { return s.events }

func (s *sqliteStore) Close() error {
	var err error
	s.closeOnce.Do(func() {
		// Mark closed and close the channel under the write lock so an in-flight
		// emit (which holds the read lock) can't send on a closing channel —
		// otherwise a late agent goroutine racing teardown panics / data-races.
		s.eventsMu.Lock()
		s.eventsClosed = true
		close(s.events)
		s.eventsMu.Unlock()
		err = s.sqlDB.Close()
	})
	return err
}

// emit sends a StoreEvent non-blocking. If the channel is full, the event is
// dropped (consumers must keep up). Safe against a concurrent Close: the read
// lock excludes the close, and the closed flag makes a post-close emit a no-op.
func (s *sqliteStore) emit(evt db.StoreEvent) {
	s.eventsMu.RLock()
	defer s.eventsMu.RUnlock()
	if s.eventsClosed {
		return
	}
	select {
	case s.events <- evt:
	default:
	}
}

// --- time helpers ---

const timeFormat = "2006-01-02T15:04:05.000Z"

func formatTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(timeFormat)
}

func parseTime(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	t, err := time.Parse(timeFormat, s)
	if err != nil {
		// A malformed timestamp shouldn't crash a scan, but it signals data
		// corruption (or a format drift), so surface it rather than silently
		// returning the zero time (which sorts first / renders as 0001-01-01).
		slog.Warn("sqlite: unparseable timestamp; using zero time", "value", s, "error", err)
		return time.Time{}
	}
	return t
}

func parseNullableTime(s sql.NullString) time.Time {
	if !s.Valid {
		return time.Time{}
	}
	return parseTime(s.String)
}

// --- Run operations ---

func (s *sqliteStore) CreateRun(ctx context.Context, run db.RunRecord) error {
	configJSON, err := json.Marshal(run.Config)
	if err != nil {
		return fmt.Errorf("marshal run config: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	// Invariant: a new run ALWAYS has a non-empty created_at AND last_seen_at, so
	// last_seen_at is NULL only for legacy (pre-column) or explicitly-dismissed
	// rows. Default a zero CreatedAt to now so a caller that omits it can't write
	// an empty timestamp (which NULLIF would collapse to NULL, breaking the
	// "new run starts seen" rule).
	createdAt := run.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}
	created := formatTime(createdAt)
	_, err = s.sqlDB.ExecContext(ctx,
		`INSERT INTO Run (run_id, config_json, title, note, starred, grade, report_grade, archived, created_at, updated_at, last_seen_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		run.RunID, string(configJSON), run.Title, run.Note,
		// last_seen_at = created_at: a fresh run starts SEEN (no badge). Its later
		// terminal transition (status_changed_at > created) is what raises the
		// badge. Stamping (vs leaving NULL) is what distinguishes "new, tracked"
		// from "legacy/dismissed" (NULL = seen).
		b2i(run.Starred), run.Grade, run.ReportGrade, b2i(run.Archived), created, created, created,
	)
	if err != nil {
		return fmt.Errorf("insert run: %w", err)
	}
	return nil
}

// b2i maps a bool to sqlite's 0/1 integer representation.
func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (s *sqliteStore) GetRun(ctx context.Context, runID string) (*db.RunRecord, error) {
	row := s.sqlDB.QueryRowContext(ctx,
		`SELECT run_id, config_json, title, note, starred, grade, report_grade, archived, created_at, updated_at, last_seen_at
		 FROM Run WHERE run_id = ?`, runID,
	)
	r, err := scanRunRow(row)
	if err != nil {
		return nil, fmt.Errorf("scan run: %w", err)
	}
	return &r, nil
}

// scanRunRow scans a Run row from any *sql.Row or *sql.Rows (column order must
// match the SELECT in GetRun/ListRuns).
func scanRunRow(sc interface{ Scan(...any) error }) (db.RunRecord, error) {
	var r db.RunRecord
	var configJSON, createdAt, updatedAt, lastSeenAt sql.NullString
	var starred, archived int
	if err := sc.Scan(&r.RunID, &configJSON, &r.Title, &r.Note,
		&starred, &r.Grade, &r.ReportGrade, &archived, &createdAt, &updatedAt, &lastSeenAt); err != nil {
		return r, err
	}
	r.Starred = starred != 0
	r.Archived = archived != 0
	r.CreatedAt = parseNullableTime(createdAt)
	r.UpdatedAt = parseNullableTime(updatedAt)
	r.LastSeenAt = parseNullableTime(lastSeenAt)
	if configJSON.Valid {
		_ = json.Unmarshal([]byte(configJSON.String), &r.Config)
	}
	return r, nil
}

// rootSessionFilter matches a run's ROOT sessions (no parent). Used inside the
// correlated subqueries below; `Run.run_id` is the outer reference.
const rootSessionFilter = `s.run_id = Run.run_id AND (s.parent_id = '' OR s.parent_id IS NULL)`

// primaryRootStatusExpr is a correlated scalar subquery yielding the run's
// primary-root status: the first non-chatbot root by created_at, else the
// earliest root. Mirrors server.primaryRoot so the dashboard's status groups
// agree between the per-row view and the server-side filter/count.
var primaryRootStatusExpr = fmt.Sprintf(`(
	SELECT s.status FROM Session s
	WHERE %s
	ORDER BY (s.agent_type = %q) ASC, s.created_at ASC
	LIMIT 1
)`, rootSessionFilter, config.ChatbotAgentType)

// runUpdatesPredicate is true when a run "has updates": some root session in a
// relevant (agent-done) state changed AFTER the run was last seen.
//
// last_seen_at NULL/empty means SEEN, not never-seen: `x > NULL` is NULL (falsy)
// in SQL, so legacy rows (pre-column, all NULL) and any run the operator has
// dismissed read as no-updates. New runs are stamped with last_seen_at =
// created_at at CreateRun, so their later terminal transition correctly counts.
// (A future "mark unseen" would move last_seen_at BEFORE the last update.)
var runUpdatesPredicate = fmt.Sprintf(`EXISTS (
	SELECT 1 FROM Session s
	WHERE %s
	  AND s.status IN (%q, %q, %q, %q)
	  AND s.status_changed_at > NULLIF(Run.last_seen_at, '')
)`, rootSessionFilter, db.SessionIdle, db.SessionConcluded, db.SessionCrashed, db.SessionCancelled)

// statusFilterClause returns a SQL WHERE fragment for a dashboard status group,
// and false if the filter is empty (no constraint). An UNKNOWN non-empty filter
// returns "1=0" (matches nothing) so a bad value can't silently widen results.
func statusFilterClause(filter string) (string, bool) {
	switch filter {
	case "":
		return "", false
	case db.RunFilterActive:
		return fmt.Sprintf("%s IN (%q, %q)", primaryRootStatusExpr, db.SessionOngoing, db.SessionAwaiting), true
	case db.RunFilterDone:
		return fmt.Sprintf("%s = %q", primaryRootStatusExpr, db.SessionConcluded), true
	case db.RunFilterFailed:
		return fmt.Sprintf("%s IN (%q, %q)", primaryRootStatusExpr, db.SessionCrashed, db.SessionCancelled), true
	case db.RunFilterUpdates:
		return runUpdatesPredicate, true
	default:
		return "1=0", true
	}
}

// effectiveGradeExpr is the run's effective grade: the human `grade` when set
// (>0), else the critic `report_grade`. Matches the dashboard's effective-grade
// logic (human overrides critic).
const effectiveGradeExpr = "(CASE WHEN grade > 0 THEN grade ELSE report_grade END)"

// gradeFilterClause returns a WHERE fragment for an effective-grade filter, and
// ok=false when there's no filter (GradeNone). GradeUngraded matches runs with
// no effective grade; a rank 1..5 matches that exact effective grade. The rank
// is a validated enum (not user text), so interpolating it is injection-safe.
func gradeFilterClause(g db.GradeFilter) (string, bool) {
	switch {
	case g == db.GradeNone:
		return "", false
	case g == db.GradeUngraded:
		return effectiveGradeExpr + " = 0", true
	case g >= 1 && g <= 5:
		return fmt.Sprintf("%s = %d", effectiveGradeExpr, int(g)), true
	default:
		return "1=0", true // unknown rank matches nothing
	}
}

// escapeLike escapes the LIKE wildcards (% and _) and the escape char itself in
// a user search term, so a literal '%' in the query doesn't act as a wildcard.
// Pairs with `ESCAPE '\'` on the LIKE. Backslash first so we don't double-escape.
func escapeLike(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, "%", `\%`)
	s = strings.ReplaceAll(s, "_", `\_`)
	return s
}
func (s *sqliteStore) ListRuns(ctx context.Context, opts db.ListRunsOpts) ([]db.RunRecord, bool, error) {
	limit := opts.Limit
	switch {
	case limit < 0:
		limit = 0 // unbounded sweep (recover/backfill): no LIMIT clause
	case limit == 0:
		limit = 20 // unset: a sane default page
	}
	var args []any
	query := `SELECT run_id, config_json, title, note, starred, grade, report_grade, archived, created_at, updated_at, last_seen_at FROM Run`
	var wheres []string
	if !opts.ShowArchived {
		wheres = append(wheres, "archived = 0")
	}
	if clause, ok := statusFilterClause(opts.StatusFilter); ok {
		wheres = append(wheres, clause)
	}
	if opts.StarredOnly {
		wheres = append(wheres, "starred = 1")
	}
	// Effective grade = human `grade` if set (>0), else `report_grade`. The filter
	// clause is a static SQL fragment; no user value is interpolated.
	if clause, ok := gradeFilterClause(opts.GradeFilter); ok {
		wheres = append(wheres, clause)
	}
	// Search: each whitespace-split term must appear (case-insensitively) in the
	// title, run_id, task, or workspace. Terms are ANDed; fields within a term are
	// ORed. Bound as %term% params (LIKE), so no injection. task/workspace are
	// pulled from config_json via json_extract for precision (vs. matching the
	// whole blob, which would hit agent_type / agents_md / JSON keys).
	for _, term := range strings.Fields(opts.Search) {
		wheres = append(wheres, `(
			LOWER(title) LIKE ? ESCAPE '\'
			OR LOWER(run_id) LIKE ? ESCAPE '\'
			OR LOWER(COALESCE(json_extract(config_json, '$.task'), '')) LIKE ? ESCAPE '\'
			OR LOWER(COALESCE(json_extract(config_json, '$.workspace'), '')) LIKE ? ESCAPE '\'
		)`)
		like := "%" + escapeLike(strings.ToLower(term)) + "%"
		args = append(args, like, like, like, like)
	}
	// Keyset cursor: rows strictly older than (Before, BeforeRunID) in the
	// (created_at DESC, run_id DESC) ordering. run_id breaks created_at ties so a
	// page boundary landing on equal timestamps can't skip or duplicate a row
	// (the OFFSET pagination this replaced did, under concurrent run creation).
	// Appended last so its placeholders line up with these trailing args.
	if !opts.Before.IsZero() {
		wheres = append(wheres, "(created_at < ? OR (created_at = ? AND run_id < ?))")
		bt := formatTime(opts.Before)
		args = append(args, bt, bt, opts.BeforeRunID)
	}
	if len(wheres) > 0 {
		query += " WHERE " + strings.Join(wheres, " AND ")
	}
	// Newest-first, with run_id as a stable tiebreaker so the ordering is TOTAL
	// (required for the keyset cursor above to be gap/dup-free). Starred runs are
	// surfaced via the "show starred" UI filter, not a pinned sort order, so they
	// stay in their natural chronological position here.
	query += " ORDER BY created_at DESC, run_id DESC"
	if limit > 0 {
		query += " LIMIT ?"
		args = append(args, limit+1) // +1 probe for has_more
	}

	rows, err := s.sqlDB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, false, fmt.Errorf("list runs: %w", err)
	}
	defer rows.Close()

	var runs []db.RunRecord
	for rows.Next() {
		r, err := scanRunRow(rows)
		if err != nil {
			return nil, false, fmt.Errorf("scan run row: %w", err)
		}
		runs = append(runs, r)
	}
	// limit == 0 means unbounded (no LIMIT was applied): never report has_more.
	hasMore := limit > 0 && len(runs) > limit
	if hasMore {
		runs = runs[:limit]
	}
	return runs, hasMore, rows.Err()
}

func (s *sqliteStore) ListRunsWithSessions(ctx context.Context, opts db.ListRunsOpts) ([]db.RunWithSessions, bool, error) {
	// First get the runs.
	runs, hasMore, err := s.ListRuns(ctx, opts)
	if err != nil {
		return nil, false, err
	}
	if len(runs) == 0 {
		return nil, false, nil
	}

	// Batch-fetch root sessions for all returned runs in one query. Bind the run
	// IDs as a single JSON array and expand it with json_each, so the SQL is a
	// constant string (no per-row placeholder concatenation) and the IDs travel
	// purely as a bound parameter.
	runIDs := make([]string, len(runs))
	for i, r := range runs {
		runIDs[i] = r.RunID
	}
	idsJSON, err := json.Marshal(runIDs)
	if err != nil {
		return nil, false, fmt.Errorf("marshal run ids: %w", err)
	}
	query := `SELECT ` + sessionCols + ` FROM Session
		WHERE run_id IN (SELECT value FROM json_each(?))
		  AND (parent_id = '' OR parent_id IS NULL)
		ORDER BY created_at ASC`

	rows, err := s.sqlDB.QueryContext(ctx, query, string(idsJSON))
	if err != nil {
		return nil, false, fmt.Errorf("list root sessions: %w", err)
	}
	defer rows.Close()

	sessions, err := scanSessions(rows)
	if err != nil {
		return nil, false, err
	}

	// Group sessions by run_id.
	sessionsByRun := make(map[string][]db.SessionRecord)
	for _, sess := range sessions {
		sessionsByRun[sess.RunID] = append(sessionsByRun[sess.RunID], sess)
	}

	result := make([]db.RunWithSessions, len(runs))
	for i, r := range runs {
		result[i] = db.RunWithSessions{
			Run:          r,
			RootSessions: sessionsByRun[r.RunID],
		}
	}
	return result, hasMore, nil
}

// --- Session operations ---

func (s *sqliteStore) CreateSession(ctx context.Context, sess db.SessionRecord) error {
	metaJSON, _ := json.Marshal(sess.Metadata)
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.sqlDB.ExecContext(ctx,
		`INSERT INTO Session (run_id, session_id, agent_type, task, status, metadata, parent_id, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		sess.RunID, sess.SessionID, sess.AgentType, sess.Task, sess.Status,
		string(metaJSON), sess.ParentID, formatTime(sess.CreatedAt),
	)
	if err != nil {
		return fmt.Errorf("insert session: %w", err)
	}
	s.emit(db.SessionCreated{RunID: sess.RunID, SessionID: sess.SessionID, ParentID: sess.ParentID, AgentType: sess.AgentType})
	return nil
}

const sessionCols = `run_id, session_id, agent_type, task, status, current_step, current_generation,
	metadata, parent_id, created_at, status_changed_at, last_finalized_step, last_summarized_step, last_phased_step`

func (s *sqliteStore) GetSession(ctx context.Context, runID, sessionID string) (*db.SessionRecord, error) {
	row := s.sqlDB.QueryRowContext(ctx,
		`SELECT `+sessionCols+` FROM Session WHERE run_id = ? AND session_id = ?`, runID, sessionID,
	)
	return scanSession(row)
}

func scanSession(row *sql.Row) (*db.SessionRecord, error) {
	var r db.SessionRecord
	var metaJSON, parentID, createdAt, statusChangedAt sql.NullString
	if err := row.Scan(
		&r.RunID, &r.SessionID, &r.AgentType, &r.Task, &r.Status,
		&r.CurrentStep, &r.CurrentGeneration,
		&metaJSON, &parentID, &createdAt, &statusChangedAt,
		&r.LastFinalizedStep, &r.LastSummarizedStep, &r.LastPhasedStep,
	); err != nil {
		return nil, fmt.Errorf("scan session: %w", err)
	}
	r.CreatedAt = parseNullableTime(createdAt)
	r.StatusChangedAt = parseNullableTime(statusChangedAt)
	r.ParentID = parentID.String
	if metaJSON.Valid && metaJSON.String != "" {
		_ = json.Unmarshal([]byte(metaJSON.String), &r.Metadata)
	}
	return &r, nil
}

func (s *sqliteStore) ListSessions(ctx context.Context, runID string) ([]db.SessionRecord, error) {
	rows, err := s.sqlDB.QueryContext(ctx,
		`SELECT `+sessionCols+` FROM Session WHERE run_id = ? ORDER BY created_at ASC`, runID,
	)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()
	return scanSessions(rows)
}

func (s *sqliteStore) GetChildSessions(ctx context.Context, runID, parentID string) ([]db.SessionRecord, error) {
	rows, err := s.sqlDB.QueryContext(ctx,
		`SELECT `+sessionCols+` FROM Session WHERE run_id = ? AND parent_id = ? ORDER BY created_at ASC`,
		runID, parentID,
	)
	if err != nil {
		return nil, fmt.Errorf("get child sessions: %w", err)
	}
	defer rows.Close()
	return scanSessions(rows)
}

func scanSessions(rows *sql.Rows) ([]db.SessionRecord, error) {
	var result []db.SessionRecord
	for rows.Next() {
		var r db.SessionRecord
		var metaJSON, parentID, createdAt, statusChangedAt sql.NullString
		if err := rows.Scan(
			&r.RunID, &r.SessionID, &r.AgentType, &r.Task, &r.Status,
			&r.CurrentStep, &r.CurrentGeneration,
			&metaJSON, &parentID, &createdAt, &statusChangedAt,
			&r.LastFinalizedStep, &r.LastSummarizedStep, &r.LastPhasedStep,
		); err != nil {
			return nil, fmt.Errorf("scan session row: %w", err)
		}
		r.CreatedAt = parseNullableTime(createdAt)
		r.StatusChangedAt = parseNullableTime(statusChangedAt)
		r.ParentID = parentID.String
		if metaJSON.Valid && metaJSON.String != "" {
			_ = json.Unmarshal([]byte(metaJSON.String), &r.Metadata)
		}
		result = append(result, r)
	}
	return result, rows.Err()
}

func (s *sqliteStore) UpdateSessionStatus(ctx context.Context, runID, sessionID, status string) error {
	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		_, err := s.sqlDB.ExecContext(ctx,
			`UPDATE Session SET status = ?, status_changed_at = ? WHERE run_id = ? AND session_id = ?`,
			status, formatTime(time.Now()), runID, sessionID,
		)
		if err != nil {
			return fmt.Errorf("update session status: %w", err)
		}
		s.emit(db.SessionStatusChanged{RunID: runID, SessionID: sessionID, NewStatus: status})
		return nil
	}(); err != nil {
		return err
	}
	s.fireSessionStatusChanged(runID, sessionID, status)
	return nil
}

func (s *sqliteStore) TerminateAndNotifyParent(ctx context.Context, runID, sessionID, parentID, status, content string, selfEvent event.Event, selfEventStep int) error {
	notify := parentID != ""
	var notifyEvt event.Event
	transitioned := false

	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		tx, err := s.sqlDB.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin terminate tx: %w", err)
		}
		defer tx.Rollback() //nolint:errcheck

		// Conditional terminal transition: a no-op if the session is already
		// terminal. Status is the dedupe token — the first terminator wins and
		// any later one (a racing conclude, the spawn fallback runner) does
		// nothing.
		res, err := tx.ExecContext(ctx,
			`UPDATE Session SET status = ?, status_changed_at = ?
			 WHERE run_id = ? AND session_id = ?
			   AND status NOT IN ('concluded', 'crashed', 'cancelled')`,
			status, formatTime(time.Now()), runID, sessionID,
		)
		if err != nil {
			return fmt.Errorf("update session status: %w", err)
		}
		if n, _ := res.RowsAffected(); n == 0 {
			return nil // already terminal — no writes, no notify
		}
		transitioned = true

		// Optional self-marker (cancel/error) on the terminating session's own
		// stream. Defaults to the current step; a non-negative selfEventStep pins
		// it (e.g. a crash recorded at the failed turn's call step).
		if selfEvent != nil {
			var step, gen int
			if err := tx.QueryRowContext(ctx,
				`SELECT current_step, current_generation FROM Session WHERE run_id = ? AND session_id = ?`,
				runID, sessionID,
			).Scan(&step, &gen); err != nil {
				return fmt.Errorf("read self step: %w", err)
			}
			if selfEventStep >= 0 {
				step = selfEventStep
			}
			data, err := event.Marshal(selfEvent)
			if err != nil {
				return fmt.Errorf("marshal self event: %w", err)
			}
			if _, err := tx.ExecContext(ctx,
				`INSERT INTO Event (run_id, session_id, event_id, step, generation, data)
				 VALUES (?, ?, ?, ?, ?, ?)`,
				runID, sessionID, db.NewEventID(), step, gen, string(data),
			); err != nil {
				return fmt.Errorf("insert self event: %w", err)
			}
		}

		if notify {
			// Append the child result at the parent's current step, in the same
			// transaction so status + notification commit atomically.
			var pStep, pGen int
			if err := tx.QueryRowContext(ctx,
				`SELECT current_step, current_generation FROM Session WHERE run_id = ? AND session_id = ?`,
				runID, parentID,
			).Scan(&pStep, &pGen); err != nil {
				return fmt.Errorf("read parent step: %w", err)
			}
			evt := &event.ChildResultEvent{ChildSessionID: sessionID, Verdict: status, Content: content}
			data, err := event.Marshal(evt)
			if err != nil {
				return fmt.Errorf("marshal child result: %w", err)
			}
			if _, err := tx.ExecContext(ctx,
				`INSERT INTO Event (run_id, session_id, event_id, step, generation, data)
				 VALUES (?, ?, ?, ?, ?, ?)`,
				runID, parentID, db.NewEventID(), pStep, pGen, string(data),
			); err != nil {
				return fmt.Errorf("insert child result: %w", err)
			}
			notifyEvt = evt
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit terminate: %w", err)
		}
		s.emit(db.SessionStatusChanged{RunID: runID, SessionID: sessionID, NewStatus: status})
		if notify {
			s.emit(db.EventAppended{RunID: runID, SessionID: parentID})
		}
		return nil
	}(); err != nil {
		return err
	}

	if !transitioned {
		return nil // already terminal — the first terminator fired the hooks
	}
	// Fire the status hook for the just-settled session (outside the lock) so
	// the observer can force-close its trailing phase. Then wake/respawn the
	// PARENT only — the terminating session is stopped via ctx, not its stream.
	s.fireSessionStatusChanged(runID, sessionID, status)
	if notify {
		s.fireCommit(runID, parentID, notifyEvt)
	}
	return nil
}

func (s *sqliteStore) MergeSessionMetadata(ctx context.Context, runID, sessionID string, updates map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Read-modify-write under the write lock.
	var existing map[string]any
	var metaJSON sql.NullString
	err := s.sqlDB.QueryRowContext(ctx,
		`SELECT metadata FROM Session WHERE run_id = ? AND session_id = ?`,
		runID, sessionID,
	).Scan(&metaJSON)
	if err != nil {
		return fmt.Errorf("read session metadata: %w", err)
	}
	if metaJSON.Valid && metaJSON.String != "" {
		if err := json.Unmarshal([]byte(metaJSON.String), &existing); err != nil {
			// Don't silently drop the existing metadata (which would clobber e.g.
			// the persisted workspace blob on the next write). Refuse the merge so
			// the corruption is visible and recoverable rather than compounded.
			return fmt.Errorf("merge metadata: existing blob is corrupt: %w", err)
		}
	}
	if existing == nil {
		existing = make(map[string]any)
	}
	for k, v := range updates {
		existing[k] = v
	}
	merged, _ := json.Marshal(existing)
	_, err = s.sqlDB.ExecContext(ctx,
		`UPDATE Session SET metadata = ? WHERE run_id = ? AND session_id = ?`,
		string(merged), runID, sessionID,
	)
	return err
}

// --- Event operations ---

func (s *sqliteStore) AppendEvent(ctx context.Context, runID, sessionID string, evt event.Event) (string, error) {
	data, err := event.Marshal(evt)
	if err != nil {
		return "", fmt.Errorf("marshal event: %w", err)
	}
	eventID := db.NewEventID()

	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		// Read-then-insert in ONE transaction so the event's step/generation
		// can't drift from a concurrent AdvanceStep/CompactContext between the
		// read and the write. (s.mu already serializes writers, so this is
		// belt-and-suspenders, but it keeps the atomicity in the DB rather than
		// relying solely on the mutex — matching FinalizeStep/CompactContext.)
		tx, err := s.sqlDB.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin append tx: %w", err)
		}
		defer tx.Rollback() //nolint:errcheck

		var step, gen int
		if err := tx.QueryRowContext(ctx,
			`SELECT current_step, current_generation FROM Session WHERE run_id = ? AND session_id = ?`,
			runID, sessionID,
		).Scan(&step, &gen); err != nil {
			return fmt.Errorf("read session step: %w", err)
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO Event (run_id, session_id, event_id, step, generation, data)
			 VALUES (?, ?, ?, ?, ?, ?)`,
			runID, sessionID, eventID, step, gen, string(data),
		); err != nil {
			return fmt.Errorf("insert event: %w", err)
		}
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit append: %w", err)
		}
		s.emit(db.EventAppended{RunID: runID, SessionID: sessionID})
		return nil
	}(); err != nil {
		return "", err
	}
	s.fireCommit(runID, sessionID, evt)
	return eventID, nil
}

func (s *sqliteStore) AppendEventAtStep(ctx context.Context, runID, sessionID string, step int, evt event.Event) error {
	data, err := event.Marshal(evt)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}

	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		// Read-then-insert in one transaction (see AppendEvent). The step is
		// explicitly provided; only the generation is read from the session.
		tx, err := s.sqlDB.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin append tx: %w", err)
		}
		defer tx.Rollback() //nolint:errcheck

		var gen int
		if err := tx.QueryRowContext(ctx,
			`SELECT current_generation FROM Session WHERE run_id = ? AND session_id = ?`,
			runID, sessionID,
		).Scan(&gen); err != nil {
			return fmt.Errorf("read session generation: %w", err)
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO Event (run_id, session_id, event_id, step, generation, data)
			 VALUES (?, ?, ?, ?, ?, ?)`,
			runID, sessionID, db.NewEventID(), step, gen, string(data),
		); err != nil {
			return fmt.Errorf("insert event at step: %w", err)
		}
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit append: %w", err)
		}
		s.emit(db.EventAppended{RunID: runID, SessionID: sessionID})
		return nil
	}(); err != nil {
		return err
	}
	s.fireCommit(runID, sessionID, evt)
	return nil
}

func (s *sqliteStore) FinalizeStep(ctx context.Context, runID, sessionID string, step int, events []event.Event) error {
	// Marshal and assign IDs outside the lock.
	type pending struct {
		id, data string
		evt      event.Event
	}
	rows := make([]pending, 0, len(events))
	for _, evt := range events {
		data, err := event.Marshal(evt)
		if err != nil {
			return fmt.Errorf("marshal event: %w", err)
		}
		rows = append(rows, pending{id: db.NewEventID(), data: string(data), evt: evt})
	}

	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		tx, err := s.sqlDB.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin tx: %w", err)
		}
		defer tx.Rollback() //nolint:errcheck

		// Read the generation inside the tx so it can't drift from the inserts.
		var gen int
		if err := tx.QueryRowContext(ctx,
			`SELECT current_generation FROM Session WHERE run_id = ? AND session_id = ?`,
			runID, sessionID,
		).Scan(&gen); err != nil {
			return fmt.Errorf("read session generation: %w", err)
		}
		for _, r := range rows {
			if _, err := tx.ExecContext(ctx,
				`INSERT INTO Event (run_id, session_id, event_id, step, generation, data)
				 VALUES (?, ?, ?, ?, ?, ?)`,
				runID, sessionID, r.id, step, gen, r.data,
			); err != nil {
				return fmt.Errorf("insert event: %w", err)
			}
		}
		if _, err := tx.ExecContext(ctx,
			`UPDATE Session SET last_finalized_step = max(last_finalized_step, ?)
			 WHERE run_id = ? AND session_id = ?`,
			step, runID, sessionID,
		); err != nil {
			return fmt.Errorf("update last_finalized_step: %w", err)
		}
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit: %w", err)
		}
		s.emit(db.EventAppended{RunID: runID, SessionID: sessionID})
		return nil
	}(); err != nil {
		return err
	}

	for _, r := range rows {
		s.fireCommit(runID, sessionID, r.evt)
	}
	s.fireStepFinalized(runID, sessionID, step)
	return nil
}

// MarkStepFinalized bumps last_finalized_step (the durable summarizer cursor)
// and fires the StepFinalized hook, without writing events — for steps whose
// events were already appended individually. No EventAppended emit: the events
// (and their UI bumps) were emitted by their own AppendEventAtStep calls.
func (s *sqliteStore) MarkStepFinalized(ctx context.Context, runID, sessionID string, step int) error {
	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		if _, err := s.sqlDB.ExecContext(ctx,
			`UPDATE Session SET last_finalized_step = max(last_finalized_step, ?)
			 WHERE run_id = ? AND session_id = ?`,
			step, runID, sessionID,
		); err != nil {
			return fmt.Errorf("update last_finalized_step: %w", err)
		}
		return nil
	}(); err != nil {
		return err
	}
	s.fireStepFinalized(runID, sessionID, step)
	return nil
}

func (s *sqliteStore) SetLastSummarizedStep(ctx context.Context, runID, sessionID string, step int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.sqlDB.ExecContext(ctx,
		`UPDATE Session SET last_summarized_step = max(last_summarized_step, ?)
		 WHERE run_id = ? AND session_id = ?`,
		step, runID, sessionID,
	); err != nil {
		return fmt.Errorf("update last_summarized_step: %w", err)
	}
	return nil
}

func (s *sqliteStore) SetLastPhasedStep(ctx context.Context, runID, sessionID string, step int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.sqlDB.ExecContext(ctx,
		`UPDATE Session SET last_phased_step = max(last_phased_step, ?)
		 WHERE run_id = ? AND session_id = ?`,
		step, runID, sessionID,
	); err != nil {
		return fmt.Errorf("update last_phased_step: %w", err)
	}
	return nil
}

func (s *sqliteStore) SumStepSummaryChars(ctx context.Context, runID, sessionID string, lo, hi int) (int, error) {
	var sum sql.NullInt64
	err := s.sqlDB.QueryRowContext(ctx,
		`SELECT SUM(char_count) FROM Observation
		 WHERE run_id = ? AND kind = 'step_summary' AND session_id = ? AND step > ? AND step <= ?`,
		runID, sessionID, lo, hi,
	).Scan(&sum)
	if err != nil {
		return 0, fmt.Errorf("sum step summary chars: %w", err)
	}
	return int(sum.Int64), nil
}

// ListPendingSessions unions the two partial-index dirty sets (step-pending and
// phase-pending). Two separate queries so each hits its partial index — a single
// OR query would force a full Session scan.
func (s *sqliteStore) ListPendingSessions(ctx context.Context) ([]db.PendingSummary, error) {
	seen := make(map[[2]string]struct{})
	var out []db.PendingSummary
	// Two full literal queries (not a concatenated predicate) so each hits its
	// partial index — and so there is no SQL string building to worry about.
	for _, q := range []string{
		`SELECT run_id, session_id, last_finalized_step, last_summarized_step, last_phased_step
		 FROM Session WHERE last_finalized_step > last_summarized_step`,
		`SELECT run_id, session_id, last_finalized_step, last_summarized_step, last_phased_step
		 FROM Session WHERE last_summarized_step > last_phased_step`,
	} {
		rows, err := s.sqlDB.QueryContext(ctx, q)
		if err != nil {
			return nil, fmt.Errorf("list pending sessions: %w", err)
		}
		for rows.Next() {
			var p db.PendingSummary
			if err := rows.Scan(&p.RunID, &p.SessionID, &p.LastFinalizedStep, &p.LastSummarizedStep, &p.LastPhasedStep); err != nil {
				_ = rows.Close()
				return nil, fmt.Errorf("scan pending session: %w", err)
			}
			key := [2]string{p.RunID, p.SessionID}
			if _, dup := seen[key]; dup {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, p)
		}
		cerr := rows.Err()
		_ = rows.Close()
		if cerr != nil {
			return nil, cerr
		}
	}
	return out, nil
}

func (s *sqliteStore) AdvanceStep(ctx context.Context, runID, sessionID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var newStep int
	err := s.sqlDB.QueryRowContext(ctx,
		`UPDATE Session SET current_step = current_step + 1
		 WHERE run_id = ? AND session_id = ?
		 RETURNING current_step`,
		runID, sessionID,
	).Scan(&newStep)
	if err != nil {
		return 0, fmt.Errorf("advance step: %w", err)
	}
	s.emit(db.StepAdvanced{RunID: runID, SessionID: sessionID, NewStep: newStep})
	return newStep, nil
}

// eventFilterClause builds the shared WHERE tail (and its args) for the event
// filter, so GetEvents and GetEventCount apply IDENTICAL filtering — a drift
// between them (e.g. a fix to the current-context predicate in only one) would
// make a count disagree with its own listing. runID/sessionID are the leading
// args the base query already binds; this appends only the optional clauses.
func eventFilterClause(runID, sessionID string, opts db.EventFilter) (string, []any) {
	var clause string
	var args []any
	if opts.CurrentContextOnly {
		clause += ` AND (step = 0 OR generation = (
			SELECT current_generation FROM Session WHERE run_id = ? AND session_id = ?
		))`
		args = append(args, runID, sessionID)
	}
	if opts.StartStep != nil {
		clause += " AND step >= ?"
		args = append(args, *opts.StartStep)
	}
	if opts.EndStep != nil {
		clause += " AND step <= ?"
		args = append(args, *opts.EndStep)
	}
	return clause, args
}

const (
	getEventsQueryHead = `SELECT run_id, session_id, event_id, step, generation, data, created_at
	          FROM Event WHERE run_id = ? AND session_id = ?`
	getEventsQueryTail = " ORDER BY step ASC, created_at ASC"
)

func (s *sqliteStore) GetEvents(ctx context.Context, runID, sessionID string, opts db.EventFilter) ([]db.EventRecord, error) {
	clause, filterArgs := eventFilterClause(runID, sessionID, opts)
	// clause is built from static fragments only (no user input — all values are
	// bound via ? placeholders in filterArgs), so the concatenation is injection-safe.
	query := getEventsQueryHead + clause + getEventsQueryTail //nolint:gosec // static fragments; values are bound params
	args := append([]any{runID, sessionID}, filterArgs...)

	rows, err := s.sqlDB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("get events: %w", err)
	}
	defer rows.Close()
	return scanEvents(rows)
}

func (s *sqliteStore) GetEventCount(ctx context.Context, runID, sessionID string, opts db.EventFilter) (int, error) {
	clause, filterArgs := eventFilterClause(runID, sessionID, opts)
	query := `SELECT COUNT(*) FROM Event WHERE run_id = ? AND session_id = ?` + clause
	args := append([]any{runID, sessionID}, filterArgs...)
	var count int
	err := s.sqlDB.QueryRowContext(ctx, query, args...).Scan(&count)
	return count, err
}

// CountEnvNotices counts the environment notifications already sitting at the
// session's current step. The step is resolved by subselect inside the same
// statement rather than read separately, so a concurrent AdvanceStep can't make
// the count refer to a step that is no longer current. A missing session yields
// a NULL subselect, hence 0 — the subsequent append reports that properly.
//
// Served by idx_event_step (run_id, session_id, step), so this is a ranged
// count over one step, not a scan of the session.
func (s *sqliteStore) CountEnvNotices(ctx context.Context, runID, sessionID string) (int, error) {
	var n int
	err := s.sqlDB.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM Event
		  WHERE run_id = ? AND session_id = ?
		    AND step = (SELECT current_step FROM Session WHERE run_id = ? AND session_id = ?)
		    AND json_extract(data, '$.type') = 'message'
		    AND json_extract(data, '$.sender_type') = ?`,
		runID, sessionID, runID, sessionID, event.SenderTypeEnvironment,
	).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("count env notices: %w", err)
	}
	return n, nil
}

func (s *sqliteStore) GetTailEvent(ctx context.Context, runID, sessionID string) (*db.EventRecord, error) {
	rows, err := s.sqlDB.QueryContext(ctx,
		`SELECT run_id, session_id, event_id, step, generation, data, created_at
		 FROM Event WHERE run_id = ? AND session_id = ?
		 ORDER BY step DESC, created_at DESC LIMIT 1`,
		runID, sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("get tail event: %w", err)
	}
	defer rows.Close()
	records, err := scanEvents(rows)
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, nil
	}
	return &records[0], nil
}

func (s *sqliteStore) SearchEvents(ctx context.Context, runID, query string, opts db.SearchOpts) ([]db.EventRecord, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	sqlQuery := `SELECT e.run_id, e.session_id, e.event_id, e.step, e.generation, e.data, e.created_at
	             FROM Event e
	             JOIN EventFTS f ON e.rowid = f.rowid
	             WHERE e.run_id = ? AND f.data MATCH ?`
	args := []any{runID, query}
	if opts.SessionID != "" {
		sqlQuery += " AND e.session_id = ?"
		args = append(args, opts.SessionID)
	}
	if opts.StepMin != nil {
		sqlQuery += " AND e.step >= ?"
		args = append(args, *opts.StepMin)
	}
	if opts.StepMax != nil {
		sqlQuery += " AND e.step <= ?"
		args = append(args, *opts.StepMax)
	}
	sqlQuery += " ORDER BY rank LIMIT ?"
	args = append(args, limit)

	rows, err := s.sqlDB.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("search events: %w", err)
	}
	defer rows.Close()
	return scanEvents(rows)
}

func (s *sqliteStore) CompactContext(ctx context.Context, runID, sessionID string, boundaryStep int, summary string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.sqlDB.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin compact tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	// Read current generation + step (step bounds the boundary validation).
	var curGen, curStep int
	if err := tx.QueryRowContext(ctx,
		`SELECT current_generation, current_step FROM Session WHERE run_id = ? AND session_id = ?`,
		runID, sessionID,
	).Scan(&curGen, &curStep); err != nil {
		return 0, fmt.Errorf("read generation: %w", err)
	}

	// Guard the boundary at the store edge (callers should already, but a bad
	// boundary corrupts the context silently): it must leave at least bootstrap
	// (step 0) behind the summary and not exceed the current step. Outside
	// [1, curStep] the summary lands with nothing meaningful to compact.
	if boundaryStep < 1 || boundaryStep > curStep {
		return 0, fmt.Errorf("compact: boundaryStep %d out of range [1, %d]", boundaryStep, curStep)
	}

	newGen := curGen + 1

	// Bump generation.
	if _, err := tx.ExecContext(ctx,
		`UPDATE Session SET current_generation = ? WHERE run_id = ? AND session_id = ?`,
		newGen, runID, sessionID,
	); err != nil {
		return 0, fmt.Errorf("update generation: %w", err)
	}

	// Carry every current-generation event AFTER the boundary into the new
	// generation. These are the fresh inputs (operator message, child result,
	// resume marker) that arrived after the boundary and have no response yet —
	// they must survive compaction verbatim. Step 0 (bootstrap) is never carried:
	// it stays included by the current-context query's step=0 clause.
	if _, err := tx.ExecContext(ctx,
		`UPDATE Event SET generation = ?
		 WHERE run_id = ? AND session_id = ? AND generation = ? AND step > ?`,
		newGen, runID, sessionID, curGen, boundaryStep,
	); err != nil {
		return 0, fmt.Errorf("carry post-boundary events: %w", err)
	}

	// Insert the CompactionEvent at the boundary step of the new generation, so
	// it sorts (step ASC) before the carried tail and after step-0 bootstrap.
	compactionEvt := &event.CompactionEvent{Content: summary}
	data, err := event.Marshal(compactionEvt)
	if err != nil {
		return 0, fmt.Errorf("marshal compaction event: %w", err)
	}
	eventID := db.NewEventID()
	if _, err := tx.ExecContext(ctx,
		`INSERT INTO Event (run_id, session_id, event_id, step, generation, data)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		runID, sessionID, eventID, boundaryStep, newGen, string(data),
	); err != nil {
		return 0, fmt.Errorf("insert compaction event: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit compact: %w", err)
	}
	s.emit(db.EventAppended{RunID: runID, SessionID: sessionID})
	return newGen, nil
}

func scanEvents(rows *sql.Rows) ([]db.EventRecord, error) {
	var result []db.EventRecord
	for rows.Next() {
		var r db.EventRecord
		var data, createdAt string
		if err := rows.Scan(
			&r.RunID, &r.SessionID, &r.EventID,
			&r.Step, &r.Generation, &data, &createdAt,
		); err != nil {
			return nil, fmt.Errorf("scan event row: %w", err)
		}
		r.CreatedAt = parseTime(createdAt)
		evt, err := event.Unmarshal([]byte(data))
		if err != nil {
			return nil, fmt.Errorf("unmarshal event %s: %w", r.EventID, err)
		}
		// Inject column fields into the event.
		setColumnFields(evt, r.Step, r.Generation, r.CreatedAt)
		r.Event = evt
		result = append(result, r)
	}
	return result, rows.Err()
}

func setColumnFields(evt event.Event, step, gen int, createdAt time.Time) {
	cf := event.ColumnFields{Step: step, Generation: gen, CreatedAt: createdAt}
	switch e := evt.(type) {
	case *event.SystemEvent:
		e.ColumnFields = cf
	case *event.UserEvent:
		e.ColumnFields = cf
	case *event.AssistantEvent:
		e.ColumnFields = cf
	case *event.ToolResultEvent:
		e.ColumnFields = cf
	case *event.CompactionEvent:
		e.ColumnFields = cf
	case *event.MessageEvent:
		e.ColumnFields = cf
	case *event.ChildResultEvent:
		e.ColumnFields = cf
	case *event.RecoverEvent:
		e.ColumnFields = cf
	}
}

// --- Observation operations ---

func (s *sqliteStore) AppendObservation(ctx context.Context, obs db.ObservationRecord) error {
	dataJSON, _ := json.Marshal(obs.Data)
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.sqlDB.ExecContext(ctx,
		`INSERT OR REPLACE INTO Observation (run_id, obs_id, kind, session_id, step, char_count, data)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		obs.RunID, obs.ObsID, obs.Kind, obs.SessionID, obs.Step, obs.CharCount, string(dataJSON),
	)
	if err != nil {
		return fmt.Errorf("insert observation: %w", err)
	}
	s.emit(db.ObservationAppended{RunID: obs.RunID, Kind: obs.Kind, SessionID: obs.SessionID, Step: obs.Step})
	return nil
}

func (s *sqliteStore) GetObservations(ctx context.Context, runID string, opts db.ObsFilter) ([]db.ObservationRecord, error) {
	query := `SELECT run_id, obs_id, kind, session_id, step, char_count, data, created_at
	          FROM Observation WHERE run_id = ?`
	args := []any{runID}
	if opts.Kind != "" {
		query += " AND kind = ?"
		args = append(args, opts.Kind)
	}
	if opts.SessionID != "" {
		query += " AND session_id = ?"
		args = append(args, opts.SessionID)
	}
	if opts.StartStep != nil {
		query += " AND step >= ?"
		args = append(args, *opts.StartStep)
	}
	if opts.EndStep != nil {
		query += " AND step <= ?"
		args = append(args, *opts.EndStep)
	}
	query += " ORDER BY created_at ASC"

	rows, err := s.sqlDB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("get observations: %w", err)
	}
	defer rows.Close()

	var result []db.ObservationRecord
	for rows.Next() {
		var r db.ObservationRecord
		var sessionID sql.NullString
		var step sql.NullInt64
		var dataJSON, createdAt string
		if err := rows.Scan(&r.RunID, &r.ObsID, &r.Kind, &sessionID, &step, &r.CharCount, &dataJSON, &createdAt); err != nil {
			return nil, fmt.Errorf("scan observation: %w", err)
		}
		r.SessionID = sessionID.String
		if step.Valid {
			v := int(step.Int64)
			r.Step = &v
		}
		r.CreatedAt = parseTime(createdAt)
		if dataJSON != "" {
			_ = json.Unmarshal([]byte(dataJSON), &r.Data)
		}
		result = append(result, r)
	}
	return result, rows.Err()
}

func (s *sqliteStore) HasObservation(ctx context.Context, runID, kind string) (bool, error) {
	var exists int
	err := s.sqlDB.QueryRowContext(ctx,
		`SELECT 1 FROM Observation WHERE run_id = ? AND kind = ? LIMIT 1`,
		runID, kind,
	).Scan(&exists)
	if err == sql.ErrNoRows {
		return false, nil
	}
	return err == nil, err
}

func (s *sqliteStore) MaxObservationStep(ctx context.Context, runID, sessionID, kind string) (*int, error) {
	var step sql.NullInt64
	err := s.sqlDB.QueryRowContext(ctx,
		`SELECT MAX(step) FROM Observation WHERE run_id = ? AND session_id = ? AND kind = ?`,
		runID, sessionID, kind,
	).Scan(&step)
	if err != nil || !step.Valid {
		return nil, err
	}
	v := int(step.Int64)
	return &v, nil
}

// --- Annotation operations ---

// Run annotation updates. Each is a fully-static, parameterized single-field
// UPDATE that also bumps updated_at — no read-modify-write, no per-viewer row.

func (s *sqliteStore) UpdateRunTitle(ctx context.Context, runID, title string) error {
	return s.updateRunField(ctx, `UPDATE Run SET title = ?, updated_at = ? WHERE run_id = ?`, title, runID)
}

func (s *sqliteStore) UpdateRunNote(ctx context.Context, runID, note string) error {
	return s.updateRunField(ctx, `UPDATE Run SET note = ?, updated_at = ? WHERE run_id = ?`, note, runID)
}

func (s *sqliteStore) SetRunStarred(ctx context.Context, runID string, starred bool) error {
	return s.updateRunField(ctx, `UPDATE Run SET starred = ?, updated_at = ? WHERE run_id = ?`, b2i(starred), runID)
}

// SetRunGrade sets the human grade (0=ungraded, 1=garbage..5=excellent), which
// overrides report_grade when nonzero.
func (s *sqliteStore) SetRunGrade(ctx context.Context, runID string, grade int) error {
	return s.updateRunField(ctx, `UPDATE Run SET grade = ?, updated_at = ? WHERE run_id = ?`, grade, runID)
}

// SetRunReportGrade caches the keen-critic run report's grade onto the run row
// (denormalization), so the card can show a critic grade without loading the
// full report. Called by the critic finalizer after the report is written.
func (s *sqliteStore) SetRunReportGrade(ctx context.Context, runID string, grade int) error {
	return s.updateRunField(ctx, `UPDATE Run SET report_grade = ?, updated_at = ? WHERE run_id = ?`, grade, runID)
}

func (s *sqliteStore) SetRunArchived(ctx context.Context, runID string, archived bool) error {
	return s.updateRunField(ctx, `UPDATE Run SET archived = ?, updated_at = ? WHERE run_id = ?`, b2i(archived), runID)
}

// DeleteRun permanently removes a run and all its DB rows in one transaction,
// child-to-parent so the foreign keys (foreign_keys=ON) are never violated
// mid-delete: Observation + Event + Session, then the Run row. Mined Lessons
// (source_run_id is a soft reference with no FK) are deliberately left alone —
// knowledge outlives the run that produced it. Idempotent: an unknown run
// deletes zero rows and still commits. Emits RunUpdated so live dashboards
// refetch and drop the row.
func (s *sqliteStore) DeleteRun(ctx context.Context, runID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.sqlDB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("delete run: begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck
	for _, stmt := range []string{
		`DELETE FROM Observation WHERE run_id = ?`,
		`DELETE FROM Event WHERE run_id = ?`,
		`DELETE FROM Session WHERE run_id = ?`,
		`DELETE FROM Run WHERE run_id = ?`,
	} {
		if _, err := tx.ExecContext(ctx, stmt, runID); err != nil {
			return fmt.Errorf("delete run: %w", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("delete run: commit: %w", err)
	}
	s.emit(db.RunUpdated{RunID: runID})
	return nil
}

// updateRunField runs a fixed single-field Run UPDATE (query is a constant from
// the caller) with value + the bumped updated_at + runID, then emits RunUpdated
// so live dashboards refetch.
func (s *sqliteStore) updateRunField(ctx context.Context, query string, value any, runID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.sqlDB.ExecContext(ctx, query, value, formatTime(time.Now()), runID); err != nil {
		return fmt.Errorf("update run: %w", err)
	}
	s.emit(db.RunUpdated{RunID: runID})
	return nil
}

// MarkRunSeen records that the operator viewed runID now, clearing its "has
// updates" badge. Deliberately does NOT touch updated_at (seeing a run is not a
// modification of it) but DOES emit RunUpdated so live dashboards refetch and
// drop the dot. Single-operator, so last_seen_at is a global fact.
// MarkRunUnseen rewinds last_seen_at to created_at, which is the state a run
// nobody has opened is in — so the badge reappears exactly when it would have
// for a fresh run: on any relevant root status change since the run started.
//
// created_at rather than NULL: a NULL last_seen_at reads as SEEN here (legacy
// rows), so it would clear the badge instead of restoring it.
func (s *sqliteStore) MarkRunUnseen(ctx context.Context, runID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.sqlDB.ExecContext(ctx,
		`UPDATE Run SET last_seen_at = created_at WHERE run_id = ?`, runID,
	); err != nil {
		return fmt.Errorf("mark run unseen: %w", err)
	}
	s.emit(db.RunUpdated{RunID: runID})
	return nil
}

func (s *sqliteStore) MarkRunSeen(ctx context.Context, runID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.sqlDB.ExecContext(ctx,
		`UPDATE Run SET last_seen_at = ? WHERE run_id = ?`, formatTime(time.Now()), runID,
	); err != nil {
		return fmt.Errorf("mark run seen: %w", err)
	}
	s.emit(db.RunUpdated{RunID: runID})
	return nil
}

// RunCounts returns the dashboard banner's exact, global tally over NON-archived
// runs (independent of pagination): Active = primary root ongoing/awaiting;
// Updates = a relevant root status changed since the run was last seen.
const runCountsQueryFmt = `SELECT
	COUNT(*) FILTER (WHERE %s),
	COUNT(*) FILTER (WHERE %s)
	FROM Run WHERE archived = 0`

func (s *sqliteStore) RunCounts(ctx context.Context) (db.RunCounts, error) {
	var c db.RunCounts
	activeClause, _ := statusFilterClause(db.RunFilterActive)
	// Both predicates are built from package constants (see statusFilterClause /
	// runUpdatesPredicate), never user input, so this format is injection-safe.
	query := fmt.Sprintf(runCountsQueryFmt, activeClause, runUpdatesPredicate) //nolint:gosec // predicates are static package constants
	if err := s.sqlDB.QueryRowContext(ctx, query).Scan(&c.Active, &c.Updates); err != nil {
		return c, fmt.Errorf("run counts: %w", err)
	}
	return c, nil
}

func (s *sqliteStore) ListCustomModels(ctx context.Context) ([]string, error) {
	rows, err := s.sqlDB.QueryContext(ctx, `SELECT spec FROM CustomModel ORDER BY created_at, spec`)
	if err != nil {
		return nil, fmt.Errorf("list custom models: %w", err)
	}
	defer rows.Close()
	var specs []string
	for rows.Next() {
		var spec string
		if err := rows.Scan(&spec); err != nil {
			return nil, fmt.Errorf("scan custom model: %w", err)
		}
		specs = append(specs, spec)
	}
	return specs, rows.Err()
}

func (s *sqliteStore) AddCustomModel(ctx context.Context, spec string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.sqlDB.ExecContext(ctx,
		`INSERT INTO CustomModel (spec, created_at) VALUES (?, ?) ON CONFLICT(spec) DO NOTHING`,
		spec, formatTime(time.Now()),
	)
	if err != nil {
		return fmt.Errorf("add custom model: %w", err)
	}
	return nil
}

func (s *sqliteStore) RemoveCustomModel(ctx context.Context, spec string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.sqlDB.ExecContext(ctx, `DELETE FROM CustomModel WHERE spec = ?`, spec); err != nil {
		return fmt.Errorf("remove custom model: %w", err)
	}
	return nil
}

// --- Lesson operations ---

func (s *sqliteStore) InsertLesson(ctx context.Context, lesson db.LessonRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Stamp updated_at explicitly: the column is nullable on migrated DBs (the
	// ALTER ADD COLUMN couldn't carry the schema's non-constant strftime default),
	// so an INSERT that omits it would leave NULL and break the later string scan.
	_, err := s.sqlDB.ExecContext(ctx,
		`INSERT INTO Lesson (lesson_id, title, description, body, embedding, embedder_id, source_run_id, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))`,
		lesson.LessonID, lesson.Title, lesson.Description, lesson.Body,
		encodeEmbedding(lesson.Embedding), lesson.EmbedderID, lesson.SourceRunID,
	)
	return err
}

func (s *sqliteStore) GetLesson(ctx context.Context, lessonID string) (*db.LessonRecord, error) {
	var r db.LessonRecord
	var embBlob []byte
	var sourceRunID sql.NullString
	var createdAt string
	var updatedAt sql.NullString // nullable on migrated DBs (see InsertLesson)
	err := s.sqlDB.QueryRowContext(ctx,
		`SELECT lesson_id, title, description, body, embedding, embedder_id,
		        source_run_id, score, loaded_count, created_at, updated_at
		 FROM Lesson WHERE lesson_id = ?`, lessonID,
	).Scan(&r.LessonID, &r.Title, &r.Description, &r.Body, &embBlob,
		&r.EmbedderID, &sourceRunID, &r.Score, &r.LoadedCount, &createdAt, &updatedAt)
	if err != nil {
		return nil, fmt.Errorf("get lesson: %w", err)
	}
	r.Embedding = decodeEmbedding(embBlob)
	r.SourceRunID = sourceRunID.String
	r.CreatedAt = parseTime(createdAt)
	r.UpdatedAt = parseTime(updatedAt.String)
	return &r, nil
}

func (s *sqliteStore) ListAllLessons(ctx context.Context) ([]db.LessonRecord, error) {
	rows, err := s.sqlDB.QueryContext(ctx,
		`SELECT lesson_id, title, description, body, embedding, embedder_id,
		        source_run_id, score, loaded_count, created_at, updated_at
		 FROM Lesson ORDER BY created_at ASC`,
	)
	if err != nil {
		return nil, fmt.Errorf("list lessons: %w", err)
	}
	defer rows.Close()

	var result []db.LessonRecord
	for rows.Next() {
		var r db.LessonRecord
		var embBlob []byte
		var sourceRunID sql.NullString
		var createdAt string
		var updatedAt sql.NullString // nullable on migrated DBs (see InsertLesson)
		if err := rows.Scan(&r.LessonID, &r.Title, &r.Description, &r.Body, &embBlob,
			&r.EmbedderID, &sourceRunID, &r.Score, &r.LoadedCount, &createdAt, &updatedAt); err != nil {
			return nil, fmt.Errorf("scan lesson: %w", err)
		}
		r.Embedding = decodeEmbedding(embBlob)
		r.SourceRunID = sourceRunID.String
		r.CreatedAt = parseTime(createdAt)
		r.UpdatedAt = parseTime(updatedAt.String)
		result = append(result, r)
	}
	return result, rows.Err()
}

func (s *sqliteStore) UpdateLesson(ctx context.Context, lesson db.LessonRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.sqlDB.ExecContext(ctx,
		`UPDATE Lesson SET title = ?, description = ?, body = ?, embedding = ?, embedder_id = ?,
		        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		 WHERE lesson_id = ?`,
		lesson.Title, lesson.Description, lesson.Body,
		encodeEmbedding(lesson.Embedding), lesson.EmbedderID, lesson.LessonID,
	)
	return err
}

func (s *sqliteStore) IncrementLessonLoadCount(ctx context.Context, lessonID string, delta int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.sqlDB.ExecContext(ctx,
		`UPDATE Lesson SET loaded_count = loaded_count + ?,
		        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		 WHERE lesson_id = ?`,
		delta, lessonID,
	)
	return err
}

func (s *sqliteStore) AddToLessonScore(ctx context.Context, lessonID string, delta int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.sqlDB.ExecContext(ctx,
		`UPDATE Lesson SET score = score + ?,
		        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		 WHERE lesson_id = ?`,
		delta, lessonID,
	)
	return err
}

// --- embedding encoding (little-endian float32 bytes) ---

// --- Skill embedding cache ---

func (s *sqliteStore) GetSkillVectors(ctx context.Context, model string) ([]db.SkillVector, error) {
	rows, err := s.sqlDB.QueryContext(ctx,
		`SELECT name, content_hash, vector, description, path, body FROM SkillEmbedding WHERE model = ?`, model)
	if err != nil {
		return nil, fmt.Errorf("get skill vectors: %w", err)
	}
	defer rows.Close()
	var out []db.SkillVector
	for rows.Next() {
		var v db.SkillVector
		var blob []byte
		if err := rows.Scan(&v.Name, &v.ContentHash, &blob, &v.Description, &v.Path, &v.Body); err != nil {
			return nil, fmt.Errorf("scan skill vector: %w", err)
		}
		v.Vector = decodeEmbedding(blob)
		out = append(out, v)
	}
	return out, rows.Err()
}

// PutSkillVectors replaces the full set of cached embeddings for a model: it
// upserts every vector in `vectors` and, in the same transaction, deletes any
// existing row for the model whose name is NOT in the new set. The caller passes
// the complete current scan (see skills.Index.buildOnce), so pruning here stops
// a skill that was removed from the source tree from lingering in the cache and
// being resurrected into recall by LoadCached on the next startup.
func (s *sqliteStore) PutSkillVectors(ctx context.Context, model string, vectors []db.SkillVector) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.sqlDB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin skill vectors tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck
	keep := make(map[string]struct{}, len(vectors))
	for _, v := range vectors {
		keep[v.Name] = struct{}{}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO SkillEmbedding (model, name, content_hash, vector, description, path, body)
			 VALUES (?, ?, ?, ?, ?, ?, ?)
			 ON CONFLICT(model, name) DO UPDATE SET
			   content_hash=excluded.content_hash,
			   vector=excluded.vector,
			   description=excluded.description,
			   path=excluded.path,
			   body=excluded.body`,
			model, v.Name, v.ContentHash, encodeEmbedding(v.Vector),
			v.Description, v.Path, v.Body); err != nil {
			return fmt.Errorf("upsert skill vector %q: %w", v.Name, err)
		}
	}
	// Prune rows for this model that are no longer in the current set.
	rows, err := tx.QueryContext(ctx, `SELECT name FROM SkillEmbedding WHERE model = ?`, model)
	if err != nil {
		return fmt.Errorf("list skill vectors for prune: %w", err)
	}
	var stale []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			rows.Close()
			return fmt.Errorf("scan skill vector name: %w", err)
		}
		if _, ok := keep[name]; !ok {
			stale = append(stale, name)
		}
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("iterate skill vectors for prune: %w", err)
	}
	rows.Close()
	for _, name := range stale {
		if _, err := tx.ExecContext(ctx,
			`DELETE FROM SkillEmbedding WHERE model = ? AND name = ?`, model, name); err != nil {
			return fmt.Errorf("prune stale skill vector %q: %w", name, err)
		}
	}
	return tx.Commit()
}

func encodeEmbedding(vec []float32) []byte {
	if len(vec) == 0 {
		return nil
	}
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func decodeEmbedding(blob []byte) []float32 {
	if len(blob) == 0 {
		return nil
	}
	n := len(blob) / 4
	vec := make([]float32, n)
	for i := range n {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	return vec
}

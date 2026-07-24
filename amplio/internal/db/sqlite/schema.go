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

const schema = `
CREATE TABLE IF NOT EXISTS Run (
    run_id      TEXT PRIMARY KEY,
    config_json TEXT,
    -- User-editable overlay (single-user: no per-viewer table needed).
    title       TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    starred     INTEGER NOT NULL DEFAULT 0,
    -- Human grade (0=ungraded, 1=garbage..5=excellent). When nonzero it
    -- overrides report_grade, the grade denormalized from the keen-critic run
    -- report (also 0=ungraded, 1..5).
    grade        INTEGER NOT NULL DEFAULT 0,
    report_grade INTEGER NOT NULL DEFAULT 0,
    archived    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at  TEXT,
    -- When the operator last viewed this run. The dashboard "has updates" badge
    -- is: a relevant root status_changed_at > last_seen_at. NULL/empty means
    -- SEEN (the "> NULL" comparison is falsy) -- legacy pre-column rows and
    -- explicitly-dismissed runs read as no-updates. New runs are stamped with
    -- last_seen_at = created_at at insert, so their later terminal transition
    -- correctly raises the badge.
    last_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_created ON Run(created_at DESC);

CREATE TABLE IF NOT EXISTS Session (
    run_id             TEXT NOT NULL REFERENCES Run(run_id),
    session_id         TEXT NOT NULL,
    agent_type         TEXT NOT NULL DEFAULT '',
    task               TEXT NOT NULL DEFAULT '',
    status             TEXT NOT NULL,
    current_step       INTEGER NOT NULL DEFAULT 0,
    current_generation INTEGER NOT NULL DEFAULT 0,
    metadata           TEXT,
    parent_id          TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    status_changed_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    last_finalized_step  INTEGER NOT NULL DEFAULT 0,
    last_summarized_step INTEGER NOT NULL DEFAULT 0,
    last_phased_step     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_parent ON Session(run_id, parent_id);

CREATE TABLE IF NOT EXISTS Event (
    run_id     TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_id   TEXT NOT NULL,
    step       INTEGER NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (run_id, session_id, event_id),
    FOREIGN KEY (run_id, session_id) REFERENCES Session(run_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_event_step ON Event(run_id, session_id, step, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS EventFTS USING fts5(
    data,
    content='Event',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync with Event table.
CREATE TRIGGER IF NOT EXISTS event_ai AFTER INSERT ON Event BEGIN
    INSERT INTO EventFTS(rowid, data) VALUES (new.rowid, new.data);
END;
CREATE TRIGGER IF NOT EXISTS event_ad AFTER DELETE ON Event BEGIN
    INSERT INTO EventFTS(EventFTS, rowid, data) VALUES('delete', old.rowid, old.data);
END;
CREATE TRIGGER IF NOT EXISTS event_au AFTER UPDATE ON Event BEGIN
    INSERT INTO EventFTS(EventFTS, rowid, data) VALUES('delete', old.rowid, old.data);
    INSERT INTO EventFTS(rowid, data) VALUES (new.rowid, new.data);
END;

CREATE TABLE IF NOT EXISTS Observation (
    run_id     TEXT NOT NULL REFERENCES Run(run_id),
    obs_id     TEXT NOT NULL,
    kind       TEXT NOT NULL,
    session_id TEXT,
    step       INTEGER,
    generation INTEGER,
    char_count INTEGER NOT NULL DEFAULT 0,
    data       TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (run_id, obs_id)
);
CREATE INDEX IF NOT EXISTS idx_obs_kind ON Observation(run_id, kind, session_id, step);

-- User-added agent models (the app-owned half of the model menu; the other half
-- is config.toml's [run].llms, which we never rewrite).
CREATE TABLE IF NOT EXISTS CustomModel (
    spec       TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Cached skill-description embeddings, keyed by embedder model + skill name, so
-- unchanged skills aren't re-embedded on startup (content_hash invalidates one).
-- description/path/body let the in-memory Index hydrate FULLY from cache without
-- touching the (possibly cloud-backed, slow) skill source FS. recall_search
-- reads description; recall_load reads body+path. The background reconcile
-- against the source tree updates them when SKILL.md content changes.
CREATE TABLE IF NOT EXISTS SkillEmbedding (
    model        TEXT NOT NULL,
    name         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    vector       BLOB NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    path         TEXT NOT NULL DEFAULT '',
    body         TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (model, name)
);

CREATE TABLE IF NOT EXISTS Lesson (
    lesson_id    TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    body         TEXT NOT NULL DEFAULT '',
    embedding    BLOB,
    embedder_id  TEXT NOT NULL DEFAULT '',
    source_run_id TEXT,
    score        INTEGER NOT NULL DEFAULT 0,
    loaded_count INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
`

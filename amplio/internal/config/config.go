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

package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"amplio/internal/nickname"
)

// Reserved session IDs for per-run singletons.
const (
	RootAgentSessionID = nickname.RootAgent
	ChatbotSessionID   = nickname.Chatbot
)

// ChatbotAgentType is the registered agent type for the per-run chatbot (root of
// a chat-driven run; steering sidekick of an autonomous run). Shared so the
// server can request it without importing the agent implementation.
const ChatbotAgentType = "chatbot"

// Environment variable names.
const (
	EnvDataDir       = "AMPLIO_DATA_DIR"
	EnvArtifactDir   = "AMPLIO_ARTIFACT_DIR"
	EnvRunID         = "AMPLIO_RUN_ID"
	EnvListen        = "AMPLIO_LISTEN" // serve bind address override (host:port)
	EnvSystemLLMHQ   = "AMPLIO_SYSTEM_LLM_HQ"
	EnvSystemLLMFast = "AMPLIO_SYSTEM_LLM_FAST"
	EnvEmbedModel    = "AMPLIO_EMBED_MODEL"
	EnvSkillDirs     = "AMPLIO_SKILL_DIRS" // OS path-list separated (filepath.SplitList)
	EnvSessionID     = "AMPLIO_SESSION_ID" // bash subprocess: the agent's own session (notify default target)
	EnvNotify        = "AMPLIO_NOTIFY"     // bash subprocess: path to the amplio binary for `notify`
)

// --- Data directories ---

// dataDirOverride pins the data dir for this process (set once at startup from
// the --data-dir flag). Empty means fall back to $AMPLIO_DATA_DIR / the default.
var dataDirOverride string

// SetDataDir pins the data directory, taking precedence over $AMPLIO_DATA_DIR
// and the default. Call once at startup (from the --data-dir flag) before any
// DataDir consumer runs.
func SetDataDir(dir string) {
	dataDirOverride = dir
	_ = os.MkdirAll(dir, 0o755)
}

// DataDir returns the per-user data directory. Precedence: --data-dir (via
// SetDataDir) > $AMPLIO_DATA_DIR > ~/.amplio.
func DataDir() string {
	if dataDirOverride != "" {
		return dataDirOverride
	}
	if override := os.Getenv(EnvDataDir); override != "" {
		_ = os.MkdirAll(override, 0o755)
		return override
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".amplio")
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

func SkillCacheDir() string {
	dir := filepath.Join(DataDir(), "skill_cache")
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

func LogsDir() string {
	dir := filepath.Join(DataDir(), "logs")
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

func ArtifactDir(runID string) string {
	dir := filepath.Join(DataDir(), "artifacts", runID)
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

// BlobDir is the on-disk directory for a run's content-addressed blobs (e.g.
// tool-result images), kept out of the event log so SQLite rows stay small.
// Pure path helper: the blob store creates the directory lazily on first write.
func BlobDir(runID string) string {
	return filepath.Join(DataDir(), "blobs", runID)
}

// --- Per-run configuration ---

// RunConfig holds user-supplied configuration for one run.
// Persisted to Run.config_json at run creation, re-loaded on resume.
type RunConfig struct {
	Task      string `json:"task,omitempty"`
	Workspace string `json:"workspace,omitempty"`
	LLM       string `json:"llm,omitempty"`
	AgentType string `json:"agent_type,omitempty"`
	AgentsMD  string `json:"agents_md,omitempty"`
}

// --- Required config resolution ---

// GlobalAgentsMDPath returns the canonical path of the user's global
// AGENTS.md (data-dir-rooted). Pure path helper — the file may not exist.
func GlobalAgentsMDPath() string {
	return filepath.Join(DataDir(), "AGENTS.md")
}

// ReadGlobalAgentsMD loads the global AGENTS.md content from the data
// directory. Missing-file is NOT an error (users opt in by creating the
// file); other read errors are propagated. The result is trim-spaced so
// trailing newlines don't pollute the agent's context.
//
// Called at run-start to snapshot the file into RunConfig — re-reads
// later in the run lifecycle would create inconsistent state (the
// system prompt is a step-0 immutable event), so callers should treat
// this as a one-shot capture.
func ReadGlobalAgentsMD() (string, error) {
	data, err := os.ReadFile(GlobalAgentsMDPath())
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read global AGENTS.md at %s: %w", GlobalAgentsMDPath(), err)
	}
	return strings.TrimSpace(string(data)), nil
}

// CombineAgentsMD merges already-read AGENTS.md contents into one block
// suitable for storage in RunConfig.AgentsMD. Sections are labeled with
// their absolute source path so the agent (and human readers) can tell
// what came from where, and the operator can copy/paste the path to open
// the right file. Empty inputs are skipped; returns "" when both are
// empty (no event will be emitted downstream).
//
// The expected source-of-truth is two AGENTS.md files: the user's global
// one (`<data-dir>/AGENTS.md`) and the workspace-rooted one
// (`<workspace>/AGENTS.md`). Sub-agent workspaces are framework-created
// in link mode and don't typically have their own — so it's safe to
// snapshot once at run-start and inherit through every sub-agent.
func CombineAgentsMD(globalContent, globalPath, workspaceContent, workspacePath string) string {
	var sections []string
	if globalContent != "" {
		sections = append(sections, "## "+globalPath+"\n\n"+globalContent)
	}
	if workspaceContent != "" {
		sections = append(sections, "## "+workspacePath+"\n\n"+workspaceContent)
	}
	return strings.Join(sections, "\n\n")
}

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
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/pelletier/go-toml/v2"
)

// Config is the persistent server/runtime configuration, loaded from
// <data-dir>/config.toml. Go is the single source of truth: DefaultConfig holds
// the fallback values and the TOML file overrides only the keys it sets.
type Config struct {
	Listen        string       `toml:"listen"`          // HTTP bind address for `serve`
	DB            string       `toml:"db"`              // sqlite path; defaults to <data-dir>/amplio.db
	Token         string       `toml:"token"`           // web auth token; random per `serve` when empty
	SystemLLMHQ   string       `toml:"system_llm_hq"`   // observer phase summaries / reports (server-level)
	SystemLLMFast string       `toml:"system_llm_fast"` // observer step summaries (server-level)
	EmbedModel    string       `toml:"embed_model"`     // Vertex embedding model (skills + lessons recall)
	Run           RunDefaults  `toml:"run"`             // defaults applied to new runs
	Skills        SkillsConfig `toml:"skills"`          // skill corpus sources
	// AmplioBinPaths are directories prepended to $PATH at startup so amplio's
	// shipped 1p CLI tools (e.g. web_search) resolve by bare name for both our
	// probes and the agent's bash subprocesses. Omitted → the built-in default;
	// explicit empty list → none.
	AmplioBinPaths []string `toml:"amplio_bin_paths"`
}

// SkillsConfig is the [skills] section: where to scan for SKILL.md files.
type SkillsConfig struct {
	// Dirs are skill source directories, layered in order (last-wins on
	// same-named skills). Omitted → the built-in default; an explicit empty list
	// → skills disabled. See Config.SkillDirs.
	Dirs    []string `toml:"dirs"`
	Blocked []string `toml:"blocked"` // skill names to exclude from all sources
}

// DefaultSkillsDir is the fallback when [skills].dirs is omitted from
// config.toml. Empty by default; a 1P init in load_internal.go overrides it
// to the canonical 1P skill tree. Set [skills].dirs explicitly to point at
// your own directory of SKILL.md files.
var DefaultSkillsDir = ""

// DefaultEmbedModel is the embedding model used for recall when embed_model is
// unset. Default empty in OSS (recall is then disabled — no embedder is built).
var DefaultEmbedModel = ""

// DefaultListen is the serve bind address when [listen] is unset. localhost in
// OSS (safe default — not exposed to the network).
var DefaultListen = "localhost:26759"

// DefaultAmplioBinPath is the fallback for amplio_bin_paths in config.toml
// (prepended to $PATH at startup). Empty in OSS; a 1P init in
// load_internal.go overrides it to the corp release dir holding amplio's
// shipped 1P CLI tools.
var DefaultAmplioBinPath = ""

// BinPaths returns the directories to prepend to $PATH at startup, falling back
// to the built-in default only when amplio_bin_paths is omitted entirely (nil).
// An explicit empty list disables the prepend.
func (c Config) BinPaths() []string {
	if c.AmplioBinPaths == nil {
		return []string{DefaultAmplioBinPath}
	}
	return c.AmplioBinPaths
}

// SkillDirs returns the configured skill source directories, falling back to the
// built-in default only when [skills].dirs is omitted entirely (nil). An
// explicit empty list disables skills.
func (c Config) SkillDirs() []string {
	if c.Skills.Dirs == nil {
		return []string{DefaultSkillsDir}
	}
	return c.Skills.Dirs
}

// EmbedModelOrDefault returns the configured embedding model, or the built-in
// default when unset. The result may be "" (OSS default) — callers treat an
// empty model as "no embedder configured" and skip recall.
func (c Config) EmbedModelOrDefault() string {
	if c.EmbedModel != "" {
		return c.EmbedModel
	}
	return DefaultEmbedModel
}

// DefaultAgentType and DefaultWorkspace are the GLOBAL fallbacks for a run's
// agent type and working directory.
const (
	DefaultAgentType = "standard_agent"
	DefaultWorkspace = "."
)

// RunDefaults is the [run] section. Currently it carries only the agent model menu.
type RunDefaults struct {
	LLMs []string `toml:"llms"` // agent model menu; the first is the default
}

// DefaultLLM is the default agent model (the first configured), or "" if none.
func (c Config) DefaultLLM() string {
	if len(c.Run.LLMs) > 0 {
		return c.Run.LLMs[0]
	}
	return ""
}

// DefaultConfig is the authoritative fallback. A user's config.toml overrides
// these per-key; anything it omits keeps these values.
func DefaultConfig() Config {
	return Config{
		Listen: DefaultListen,
	}
}

// ConfigPath is the config file location for a data directory.
func ConfigPath(dataDir string) string {
	return filepath.Join(dataDir, "config.toml")
}

// Load reads <dataDir>/config.toml over DefaultConfig. A missing file is fine
// (pure defaults). DB defaults to <dataDir>/amplio.db when unset.
func Load(dataDir string) (Config, error) {
	cfg := DefaultConfig()
	path := ConfigPath(dataDir)
	data, err := os.ReadFile(path)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		// No file: defaults only.
	case err != nil:
		return Config{}, fmt.Errorf("read config %s: %w", path, err)
	default:
		if err := toml.Unmarshal(data, &cfg); err != nil {
			return Config{}, fmt.Errorf("parse config %s: %w", path, err)
		}
	}
	if cfg.DB == "" {
		cfg.DB = filepath.Join(dataDir, "amplio.db")
	}
	return cfg, nil
}

// Overrides carries the command-line flag values for the layered knobs, so
// Resolve can apply the FLAG > ENV > CONFIG > DEFAULT precedence uniformly.
// Each field is the raw flag value (""/nil = flag not given). SkillDirsSet
// distinguishes "--skill-dir not passed" from "--skill-dir explicitly cleared",
// since an empty skill-dir list is meaningful (disables skills).
type Overrides struct {
	SystemLLMHQ   string
	SystemLLMFast string
	EmbedModel    string
	SkillDirs     []string
	SkillDirsSet  bool
}

// Resolve loads <dataDir>/config.toml and overlays the flag/env layers on top,
// yielding the effective Config. Precedence per knob: flag > env > config-file >
// built-in default. The required system tiers (hq + fast) must resolve to a
// non-empty value or Resolve returns an error naming the flag/env to set.
//
// This is the single resolution point for serve / headless run / headless
// resume, replacing the per-command inline checks.
func Resolve(dataDir string, o Overrides) (Config, error) {
	cfg, err := Load(dataDir)
	if err != nil {
		return Config{}, err
	}

	// Scalars: flag > env > (already-loaded config/default).
	cfg.SystemLLMHQ = firstNonEmpty(o.SystemLLMHQ, os.Getenv(EnvSystemLLMHQ), cfg.SystemLLMHQ)
	cfg.SystemLLMFast = firstNonEmpty(o.SystemLLMFast, os.Getenv(EnvSystemLLMFast), cfg.SystemLLMFast)
	cfg.EmbedModel = firstNonEmpty(o.EmbedModel, os.Getenv(EnvEmbedModel), cfg.EmbedModel)

	// Skill dirs (list, REPLACE semantics): the highest layer that is SET wins
	// wholesale. flag-set (SkillDirsSet) > env-set (LookupEnv) > config (handled
	// by SkillDirs()/the [skills].dirs nil-vs-empty distinction).
	if o.SkillDirsSet {
		cfg.Skills.Dirs = o.SkillDirs
	} else if env, ok := os.LookupEnv(EnvSkillDirs); ok {
		cfg.Skills.Dirs = filepath.SplitList(env)
	}

	// Required: the system tiers drive the process-global observer/finalizer and
	// reactive compaction; without them the run-hosting modes can't function.
	if cfg.SystemLLMHQ == "" {
		return Config{}, requiredErr("--system-llm-hq", EnvSystemLLMHQ, "system_llm_hq")
	}
	if cfg.SystemLLMFast == "" {
		return Config{}, requiredErr("--system-llm-fast", EnvSystemLLMFast, "system_llm_fast")
	}
	return cfg, nil
}

// firstNonEmpty returns the first non-empty string, or "" if all are empty.
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func requiredErr(flagName, envVar, tomlKey string) error {
	return fmt.Errorf(
		"missing required config %s: pass %s=<provider:model>, set $%s, or add %s to %s",
		tomlKey, flagName, envVar, tomlKey, ConfigPath(DataDir()),
	)
}

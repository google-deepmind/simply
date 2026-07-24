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
	"os"
	"path/filepath"
	"testing"
)

func TestDataDir_Default(t *testing.T) {
	t.Setenv(EnvDataDir, "")
	dir := DataDir()
	home, _ := os.UserHomeDir()
	want := filepath.Join(home, ".amplio")
	if dir != want {
		t.Errorf("DataDir() = %q, want %q", dir, want)
	}
}

func TestDataDir_Override(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv(EnvDataDir, tmp)
	dir := DataDir()
	if dir != tmp {
		t.Errorf("DataDir() = %q, want %q", dir, tmp)
	}
}

// clearLayerEnv blanks every env var Resolve consults so a test starts from a
// clean slate regardless of the ambient environment.
func clearLayerEnv(t *testing.T) {
	t.Helper()
	for _, e := range []string{EnvSystemLLMHQ, EnvSystemLLMFast, EnvEmbedModel, EnvSkillDirs} {
		t.Setenv(e, "")
	}
}

func TestResolve_RequiredTiersMissing(t *testing.T) {
	clearLayerEnv(t)
	if _, err := Resolve(t.TempDir(), Overrides{}); err == nil {
		t.Fatal("expected error when system tiers are unset")
	}
}

func TestResolve_FlagBeatsEnv(t *testing.T) {
	clearLayerEnv(t)
	t.Setenv(EnvSystemLLMHQ, "env:hq")
	t.Setenv(EnvSystemLLMFast, "env:fast")
	cfg, err := Resolve(t.TempDir(), Overrides{SystemLLMHQ: "flag:hq", SystemLLMFast: "flag:fast"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SystemLLMHQ != "flag:hq" || cfg.SystemLLMFast != "flag:fast" {
		t.Errorf("flag should win: hq=%q fast=%q", cfg.SystemLLMHQ, cfg.SystemLLMFast)
	}
}

func TestResolve_EnvBeatsConfig(t *testing.T) {
	clearLayerEnv(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.toml"),
		[]byte("system_llm_hq = \"cfg:hq\"\nsystem_llm_fast = \"cfg:fast\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(EnvSystemLLMHQ, "env:hq")
	cfg, err := Resolve(dir, Overrides{})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SystemLLMHQ != "env:hq" { // env overrides file
		t.Errorf("env should beat config: hq=%q", cfg.SystemLLMHQ)
	}
	if cfg.SystemLLMFast != "cfg:fast" { // file used where env absent
		t.Errorf("config should apply where env absent: fast=%q", cfg.SystemLLMFast)
	}
}

func TestResolve_SkillDirsEnvPathList(t *testing.T) {
	clearLayerEnv(t)
	t.Setenv(EnvSystemLLMHQ, "x:hq")
	t.Setenv(EnvSystemLLMFast, "x:fast")
	t.Setenv(EnvSkillDirs, "/a"+string(os.PathListSeparator)+"/b")
	cfg, err := Resolve(t.TempDir(), Overrides{})
	if err != nil {
		t.Fatal(err)
	}
	got := cfg.SkillDirs()
	if len(got) != 2 || got[0] != "/a" || got[1] != "/b" {
		t.Errorf("skill dirs from env path-list = %v, want [/a /b]", got)
	}
}

func TestResolve_SkillDirsFlagReplacesEnv(t *testing.T) {
	clearLayerEnv(t)
	t.Setenv(EnvSystemLLMHQ, "x:hq")
	t.Setenv(EnvSystemLLMFast, "x:fast")
	t.Setenv(EnvSkillDirs, "/from/env")
	cfg, err := Resolve(t.TempDir(), Overrides{SkillDirs: []string{"/from/flag"}, SkillDirsSet: true})
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.SkillDirs(); len(got) != 1 || got[0] != "/from/flag" {
		t.Errorf("flag should replace env wholesale: %v", got)
	}
}

func TestReadGlobalAgentsMD_Missing(t *testing.T) {
	// Opt-in by file presence: a data dir without AGENTS.md is the common
	// case and must not error.
	tmp := t.TempDir()
	t.Setenv(EnvDataDir, tmp)
	got, err := ReadGlobalAgentsMD()
	if err != nil || got != "" {
		t.Errorf("got %q, err %v", got, err)
	}
}

func TestReadGlobalAgentsMD_Exists(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv(EnvDataDir, tmp)
	if err := os.WriteFile(filepath.Join(tmp, "AGENTS.md"),
		[]byte("  instructions here  \n\n"), 0o600); err != nil { //nolint:gosec // test file
		t.Fatal(err)
	}
	// Verifies surrounding whitespace + trailing blank lines are trimmed so
	// they don't pollute the agent's context.
	got, err := ReadGlobalAgentsMD()
	if err != nil || got != "instructions here" {
		t.Errorf("got %q, err %v", got, err)
	}
}

func TestGlobalAgentsMDPath_UsesDataDir(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv(EnvDataDir, tmp)
	want := filepath.Join(tmp, "AGENTS.md")
	if got := GlobalAgentsMDPath(); got != want {
		t.Errorf("got %q; want %q", got, want)
	}
}

func TestCombineAgentsMD(t *testing.T) {
	cases := []struct {
		name              string
		gContent, gPath   string
		wsContent, wsPath string
		want              string
	}{
		{
			name:     "both",
			gContent: "global rule", gPath: "/g.md",
			wsContent: "ws rule", wsPath: "/ws.md",
			want: "## /g.md\n\nglobal rule\n\n## /ws.md\n\nws rule",
		},
		{
			name:     "global only",
			gContent: "global rule", gPath: "/g.md",
			want: "## /g.md\n\nglobal rule",
		},
		{
			name:      "workspace only",
			wsContent: "ws rule", wsPath: "/ws.md",
			want: "## /ws.md\n\nws rule",
		},
		{
			name: "both empty",
			want: "",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := CombineAgentsMD(c.gContent, c.gPath, c.wsContent, c.wsPath)
			if got != c.want {
				t.Errorf("got %q; want %q", got, c.want)
			}
		})
	}
}

func TestSetDataDir_TakesPrecedence(t *testing.T) {
	t.Cleanup(func() { dataDirOverride = "" })
	pinned := t.TempDir()
	t.Setenv(EnvDataDir, t.TempDir()) // env is set but override must win
	SetDataDir(pinned)
	if got := DataDir(); got != pinned {
		t.Errorf("DataDir() = %q, want pinned %q", got, pinned)
	}
}

func TestLoad_NoFile_Defaults(t *testing.T) {
	dir := t.TempDir()
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	// Default is build-tag-dependent (localhost in OSS, 0.0.0.0 internal), so
	// assert against the same var the default is sourced from.
	if cfg.Listen != DefaultListen {
		t.Errorf("Listen = %q, want default %q", cfg.Listen, DefaultListen)
	}
	if want := filepath.Join(dir, "amplio.db"); cfg.DB != want {
		t.Errorf("DB = %q, want %q", cfg.DB, want)
	}
}

func TestLoad_PartialOverridePreservesDefaults(t *testing.T) {
	dir := t.TempDir()
	content := "listen = \"127.0.0.1:9000\"\n[run]\nllms = [\"vertex:model-x\", \"vertex:model-y\"]\n"
	if err := os.WriteFile(ConfigPath(dir), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Listen != "127.0.0.1:9000" {
		t.Errorf("Listen = %q, want override", cfg.Listen)
	}
	if cfg.DefaultLLM() != "vertex:model-x" {
		t.Errorf("DefaultLLM() = %q, want first of llms", cfg.DefaultLLM())
	}
	if len(cfg.Run.LLMs) != 2 {
		t.Errorf("LLMs = %v, want 2 entries", cfg.Run.LLMs)
	}
	// Keys the file omits keep their defaults.
	if want := filepath.Join(dir, "amplio.db"); cfg.DB != want {
		t.Errorf("DB = %q, want default %q", cfg.DB, want)
	}
}

func TestLoad_SystemLLMsAreTopLevel(t *testing.T) {
	dir := t.TempDir()
	content := "system_llm_hq = \"vertex:hq\"\nsystem_llm_fast = \"vertex:fast\"\n[run]\nllms = [\"vertex:agent\"]\n"
	if err := os.WriteFile(ConfigPath(dir), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SystemLLMHQ != "vertex:hq" {
		t.Errorf("SystemLLMHQ = %q, want top-level value", cfg.SystemLLMHQ)
	}
	if cfg.SystemLLMFast != "vertex:fast" {
		t.Errorf("SystemLLMFast = %q, want top-level value", cfg.SystemLLMFast)
	}
	if cfg.DefaultLLM() != "vertex:agent" {
		t.Errorf("DefaultLLM() = %q, want [run] value", cfg.DefaultLLM())
	}
}

func TestLoad_ExplicitDB(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(ConfigPath(dir), []byte("db = \"/custom/x.db\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DB != "/custom/x.db" {
		t.Errorf("DB = %q, want explicit", cfg.DB)
	}
}

func TestLoad_BadTOML(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(ConfigPath(dir), []byte("listen = = nope"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir); err == nil {
		t.Error("expected error for malformed TOML")
	}
}

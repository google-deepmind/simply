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

package bash

import (
	"amplio/internal/config"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestBash_Echo(t *testing.T) {
	tool := New("/tmp", "", "")
	result := tool.ParseAndExecute(context.Background(), `{"command":"echo hello world"}`)
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "hello world") {
		t.Errorf("expected 'hello world' in output: %s", result.Content)
	}
	if !strings.Contains(result.Content, "Finished normally") {
		t.Errorf("expected normal-finish status line: %s", result.Content)
	}
}

func TestBash_ExitCode(t *testing.T) {
	tool := New("/tmp", "", "")
	result := tool.ParseAndExecute(context.Background(), `{"command":"exit 42"}`)
	if !strings.Contains(result.Content, "return code 42") {
		t.Errorf("expected return code 42: %s", result.Content)
	}
}

func TestBash_Stderr(t *testing.T) {
	tool := New("/tmp", "", "")
	result := tool.ParseAndExecute(context.Background(), `{"command":"echo err >&2"}`)
	if !strings.Contains(result.Content, "STDERR:") {
		t.Errorf("expected STDERR section: %s", result.Content)
	}
	if !strings.Contains(result.Content, "err") {
		t.Errorf("expected 'err' in stderr: %s", result.Content)
	}
}

func TestBash_Timeout(t *testing.T) {
	tool := New("/tmp", "", "")
	result := tool.ParseAndExecute(context.Background(), `{"command":"sleep 10","timeout":0.5}`)
	if !strings.Contains(result.Content, "Timed out") {
		t.Errorf("expected timeout message: %s", result.Content)
	}
}

func TestBash_Truncation(t *testing.T) {
	tool := New("/tmp", "", "")
	result := tool.ParseAndExecute(context.Background(), `{"command":"seq 1 10000","max_output_length":200}`)
	if len(result.Content) > 300 { // some margin for the truncation marker
		t.Errorf("output too long: %d chars", len(result.Content))
	}
	if !strings.Contains(result.Content, "truncated") {
		t.Errorf("expected truncation marker: %s", result.Content)
	}
}

func TestBash_InvalidJSON(t *testing.T) {
	tool := New("/tmp", "", "")
	result := tool.ParseAndExecute(context.Background(), `{bad json}`)
	if !result.IsError {
		t.Error("expected error for invalid JSON")
	}
}

func TestBash_Schema(t *testing.T) {
	tool := New("/tmp", "", "")
	def := tool.Def()
	var schema map[string]any
	if err := json.Unmarshal(def.Schema, &schema); err != nil {
		t.Fatal(err)
	}
	props := schema["properties"].(map[string]any)
	if _, ok := props["command"]; !ok {
		t.Error("missing 'command' in schema")
	}
}

func TestAgentEnv(t *testing.T) {
	if got := agentEnv("", "sid"); got != nil {
		t.Errorf("no run id should yield nil env, got %v", got)
	}
	env := agentEnv("run1", "sid1")
	has := func(prefix string) bool {
		for _, e := range env {
			if strings.HasPrefix(e, prefix) {
				return true
			}
		}
		return false
	}
	if !has("AMPLIO_RUN_ID=run1") {
		t.Error("missing AMPLIO_RUN_ID")
	}
	if !has("AMPLIO_SESSION_ID=sid1") {
		t.Error("missing AMPLIO_SESSION_ID")
	}
	if !has("AMPLIO_NOTIFY=") {
		t.Error("missing AMPLIO_NOTIFY")
	}
	if !has("AMPLIO_ARTIFACT_DIR=") {
		t.Error("missing AMPLIO_ARTIFACT_DIR")
	}
	// Empty session id: run id present, no session id key.
	env = agentEnv("run1", "")
	for _, e := range env {
		if strings.HasPrefix(e, "AMPLIO_SESSION_ID=") {
			t.Errorf("unexpected AMPLIO_SESSION_ID with empty session: %q", e)
		}
	}
}

// TestTruncateText_RuneSafe verifies truncation never splits a multi-byte rune
// and never exceeds maxLength (including the marker). See the bash UTF-8 finding.
func TestTruncateText_RuneSafe(t *testing.T) {
	// 400 multi-byte runes (each "世" is 3 bytes) → 1200 bytes.
	text := strings.Repeat("世", 400)
	for _, max := range []int{50, 100, 201, 1000} {
		out := truncateText(text, max)
		if len(out) > max {
			t.Errorf("max=%d: len(out)=%d exceeds max", max, len(out))
		}
		if !utf8.ValidString(out) {
			t.Errorf("max=%d: output is not valid UTF-8 (split a rune)", max)
		}
	}
}

// TestTruncateText_MarkerLargerThanMax: when the marker alone exceeds the
// budget, the result is the (rune-safe) trimmed marker, still within max.
func TestTruncateText_MarkerLargerThanMax(t *testing.T) {
	out := truncateText(strings.Repeat("x", 100), 5)
	if len(out) > 5 {
		t.Errorf("len(out)=%d exceeds max 5: %q", len(out), out)
	}
	if !utf8.ValidString(out) {
		t.Errorf("output not valid UTF-8: %q", out)
	}
}

// TestBash_BoundedRead verifies a command emitting more than max_output_length
// bytes is read with a cap (P15) — the result stays bounded and carries the
// read-boundary truncation note, rather than slurping the whole stream first.
func TestBash_BoundedRead(t *testing.T) {
	tool := New("/tmp", "", "")
	// Emit 100 KB to stdout with a small output cap.
	result := tool.ParseAndExecute(context.Background(),
		`{"command":"head -c 100000 /dev/zero | tr '\\0' 'x'","max_output_length":2000}`)
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content)
	}
	// The capped read appends a note naming the true byte count.
	if !strings.Contains(result.Content, "output truncated at") {
		t.Errorf("expected read-boundary truncation note: %s", result.Content)
	}
	// And the whole result is still display-truncated to the cap (plus the
	// marker), not 100 KB.
	if len(result.Content) > 4000 {
		t.Errorf("result not bounded: %d bytes", len(result.Content))
	}
}

// A tool that prompts must fail fast, not hang. Stdin is /dev/null, but the
// conventional way to prompt when stdin is redirected is to open /dev/tty — and
// before the child was given its own session it inherited the terminal running
// `amplio serve`, so prompts appeared in the operator's window and blocked
// forever ("Valid responses: y, yes, n, no", repeatedly, for days).
func TestBash_NoControllingTerminal(t *testing.T) {
	// Bounded by the test itself, not by execute's timeout: if /dev/tty is
	// reachable the child blocks on it, and a bash reading the terminal from a
	// background process group gets SIGTTIN and STOPS rather than dying — so the
	// call can outlive its own timeout and hang the whole suite. Fail instead.
	done := make(chan string, 1)
	go func() {
		done <- execute(context.Background(), `read -p "Continue? [y/n] " x < /dev/tty; echo "read-exit=$?"`,
			t.TempDir(), nil, 5*time.Second, 4096)
	}()
	select {
	case out := <-done:
		if strings.Contains(out, "timed out") {
			t.Errorf("the prompt blocked until the command timeout; /dev/tty is still reachable:\n%s", out)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("prompt never returned: the child still has a controlling terminal")
	}
}

// A timeout must reap the whole process group. Killing only the shell leaves its
// children running: that is how one hung command becomes a permanent orphan.
func TestBash_TimeoutKillsChildren(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "alive")
	// The backgrounded child outlives the shell and would keep touching the file.
	execute(context.Background(),
		`( while true; do touch `+marker+`; sleep 0.2; done ) & sleep 30`,
		t.TempDir(), nil, 1*time.Second, 4096)
	time.Sleep(300 * time.Millisecond)
	_ = os.Remove(marker)
	time.Sleep(700 * time.Millisecond) // long enough for two more touches
	if _, err := os.Stat(marker); err == nil {
		t.Error("a child survived the timeout: the process group was not killed")
	}
}

// The environment an agent's subprocess sees encodes who is told about what:
// a narrow notifier on PATH for everyone, the CLI behind a variable the system
// prompt never mentions.
func TestAgentEnv_EntryPoints(t *testing.T) {
	env := agentEnv("run-1", "sess-1")
	get := func(k string) string {
		for _, kv := range env {
			if strings.HasPrefix(kv, k+"=") {
				return strings.TrimPrefix(kv, k+"=")
			}
		}
		return ""
	}
	exe, _ := os.Executable()
	// Kept, deprecated: scripts and agents already written against it must keep
	// working, which is what lets `amplio-notify` have a single clean interface.
	if got := get(config.EnvNotify); got != exe {
		t.Errorf("%s = %q, want the binary %q (still needed by in-flight scripts)", config.EnvNotify, got, exe)
	}
}

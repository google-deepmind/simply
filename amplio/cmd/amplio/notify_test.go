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

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"amplio/internal/config"
)

func TestExecuteNotify(t *testing.T) {
	// Capture what the subcommand POSTs.
	var gotPath, gotAuth string
	var gotBody map[string]string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer ts.Close()

	dir := t.TempDir()
	config.SetDataDir(dir)
	t.Cleanup(func() { config.SetDataDir("") })
	if err := writeServerInfo(dir, serverInfo{PID: 1, URL: ts.URL, Addr: ts.URL, Token: "tok"}); err != nil {
		t.Fatal(err)
	}
	t.Setenv(config.EnvRunID, "run-xyz")
	t.Setenv(config.EnvSessionID, "swift-fox")

	// 1 positional → self (uses $AMPLIO_SESSION_ID); --from sets sender.
	if err := executeNotify(context.Background(), "hello there", "", "watcher"); err != nil {
		t.Fatalf("executeNotify: %v", err)
	}
	if gotPath != "/api/runs/run-xyz/sessions/swift-fox/notify" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer tok" {
		t.Errorf("auth = %q", gotAuth)
	}
	// sender is the --from label plus the caller's pid stamp (so a revived agent
	// can identify/kill a runaway notifier): "watcher (pid=<ppid>)".
	if gotBody["content"] != "hello there" ||
		!strings.HasPrefix(gotBody["sender"], "watcher (pid=") ||
		!strings.HasSuffix(gotBody["sender"], ")") {
		t.Errorf("body = %v", gotBody)
	}

	// --session overrides the env default. It used to be an optional first
	// positional, which made the meaning of an argument depend on how many there
	// were — so an unquoted message silently addressed a session named after its
	// first word.
	if err := executeNotify(context.Background(), "ping", "brave-owl", ""); err != nil {
		t.Fatalf("executeNotify (--session): %v", err)
	}
	if gotPath != "/api/runs/run-xyz/sessions/brave-owl/notify" {
		t.Errorf("--session path = %q", gotPath)
	}
}

func TestExecuteNotifyExitCodes(t *testing.T) {
	codeOf := func(err error) int {
		var ce *codedError
		if errors.As(err, &ce) {
			return ce.code
		}
		return 0
	}

	// Missing $AMPLIO_RUN_ID → usage error (exit 1).
	t.Setenv(config.EnvRunID, "")
	if got := codeOf(executeNotify(context.Background(), "x", "", "")); got != notifyExitUsage {
		t.Errorf("missing run id: exit %d, want %d", got, notifyExitUsage)
	}

	// No target session (no arg, no env) → usage.
	t.Setenv(config.EnvRunID, "r")
	t.Setenv(config.EnvSessionID, "")
	if got := codeOf(executeNotify(context.Background(), "x", "", "")); got != notifyExitUsage {
		t.Errorf("no session: exit %d, want %d", got, notifyExitUsage)
	}

	// Valid args but no running server (no server.json) → unreachable (exit 2).
	config.SetDataDir(t.TempDir())
	t.Cleanup(func() { config.SetDataDir("") })
	t.Setenv(config.EnvSessionID, "sid")
	if got := codeOf(executeNotify(context.Background(), "x", "", "")); got != notifyExitUnreach {
		t.Errorf("no server: exit %d, want %d", got, notifyExitUnreach)
	}
}

// The shim is a second, NARROWER interface — not a replacement. $AMPLIO_NOTIFY
// still points at the binary, so anything already written against it keeps
// working; this name simply cannot reach the rest of the CLI.
func TestDispatchShim(t *testing.T) {
	orig := os.Args
	t.Cleanup(func() { os.Args = orig })

	// Not the shim name: main's normal command tree handles it.
	os.Args = []string{"/usr/local/bin/amplio", "notify", "hi"}
	if handled, _ := dispatchShim(); handled {
		t.Error("dispatchShim claimed a plain `amplio` invocation")
	}

	t.Setenv(config.EnvRunID, "run-xyz")
	t.Setenv(config.EnvSessionID, "sess-1")
	t.Setenv(config.EnvDataDir, t.TempDir())

	// A message is just a message — including one that is the word "notify",
	// which an optional-leading-word compatibility hack would have swallowed.
	for _, msg := range []string{"a message", "notify"} {
		os.Args = []string{"/data/bin/amplio-notify", msg}
		handled, err := dispatchShim()
		if !handled {
			t.Fatal("shim did not handle an amplio-notify invocation")
		}
		// No server configured, so it must fail at the SEND step: that proves the
		// message parsed and only delivery failed.
		var ce *codedError
		if !errors.As(err, &ce) || ce.code != notifyExitUnreach {
			t.Fatalf("msg %q: err = %v; want the unreachable-server code (%d)", msg, err, notifyExitUnreach)
		}
	}

	// The rest of the CLI is unreachable through this name.
	os.Args = []string{"/data/bin/amplio-notify", "client", "submit", "--task=x"}
	if _, err := dispatchShim(); err == nil {
		t.Error("`client submit` went through the notify shim")
	}
}

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
	if err := executeNotify(context.Background(), []string{"hello there"}, "watcher"); err != nil {
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

	// 2 positionals → explicit session id overrides the env default.
	if err := executeNotify(context.Background(), []string{"brave-owl", "ping"}, ""); err != nil {
		t.Fatalf("executeNotify (2-arg): %v", err)
	}
	if gotPath != "/api/runs/run-xyz/sessions/brave-owl/notify" {
		t.Errorf("2-arg path = %q", gotPath)
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
	if got := codeOf(executeNotify(context.Background(), []string{"x"}, "")); got != notifyExitUsage {
		t.Errorf("missing run id: exit %d, want %d", got, notifyExitUsage)
	}

	// No target session (no arg, no env) → usage.
	t.Setenv(config.EnvRunID, "r")
	t.Setenv(config.EnvSessionID, "")
	if got := codeOf(executeNotify(context.Background(), []string{"x"}, "")); got != notifyExitUsage {
		t.Errorf("no session: exit %d, want %d", got, notifyExitUsage)
	}

	// Valid args but no running server (no server.json) → unreachable (exit 2).
	config.SetDataDir(t.TempDir())
	t.Cleanup(func() { config.SetDataDir("") })
	t.Setenv(config.EnvSessionID, "sid")
	if got := codeOf(executeNotify(context.Background(), []string{"x"}, "")); got != notifyExitUnreach {
		t.Errorf("no server: exit %d, want %d", got, notifyExitUnreach)
	}
}

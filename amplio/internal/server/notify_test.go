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

package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"amplio/internal/db"
	"amplio/internal/event"
)

func TestServer_Notify(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 1)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	url := ts.URL + "/api/runs/" + testRun + "/sessions/main-agent/notify?token=secret"
	code, _ := doReq(t, "POST", url, `{"content":"build done","sender":"xm-watcher"}`)
	if code != 202 {
		t.Fatalf("notify status = %d, want 202", code)
	}

	evts, err := store.GetEvents(context.Background(), testRun, "main-agent", db.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	var msg *event.MessageEvent
	for _, e := range evts {
		if m, ok := e.Event.(*event.MessageEvent); ok {
			msg = m
		}
	}
	if msg == nil {
		t.Fatal("no MessageEvent written")
	}
	if msg.Content != "build done" {
		t.Errorf("content = %q", msg.Content)
	}
	if msg.SenderType != event.SenderTypeEnvironment {
		t.Errorf("sender_type = %q, want environment", msg.SenderType)
	}
	if msg.Sender != "xm-watcher" {
		t.Errorf("sender = %q, want xm-watcher (the --from label)", msg.Sender)
	}

	// Empty content is rejected.
	if code, _ := doReq(t, "POST", url, `{"content":""}`); code != 400 {
		t.Errorf("empty content status = %d, want 400", code)
	}
}

// A runaway notifier (one heartbeating out of an unbounded loop, or one still
// firing at a session that has finished) must not append without bound: the events cost
// DB space, and every one of them is replayed into the model's context on the
// next turn. Real floods reached 785 notices at a single step.
func TestServer_NotifyFloodCap(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 1)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	ctx := context.Background()

	url := ts.URL + "/api/runs/" + testRun + "/sessions/main-agent/notify?token=secret"
	for i := range maxEnvNoticesPerStep {
		if code, _ := doReq(t, "POST", url, `{"content":"poll"}`); code != 202 {
			t.Fatalf("notice %d: status = %d, want 202 (still within the budget)", i, code)
		}
	}

	code, body := doReq(t, "POST", url, `{"content":"poll"}`)
	if code != http.StatusTooManyRequests {
		t.Fatalf("over-cap status = %d, want 429", code)
	}
	// The body is the only feedback a background script gets (via notify's exit
	// code 3 + stderr), and its reader is a shell, not a model: assert the stable
	// token a script can grep for, and that no variable count leaks into it (that
	// would defeat an exact match).
	if !strings.Contains(string(body), errEnvNoticeCapped) {
		t.Errorf("429 body = %s; want the %q token", body, errEnvNoticeCapped)
	}
	if strings.Contains(string(body), strconv.Itoa(maxEnvNoticesPerStep+1)) {
		t.Errorf("429 body = %s; must not carry a live notice count", body)
	}

	n, err := store.CountEnvNotices(ctx, testRun, "main-agent")
	if err != nil {
		t.Fatal(err)
	}
	if n != maxEnvNoticesPerStep {
		t.Errorf("persisted notices = %d, want exactly the cap %d", n, maxEnvNoticesPerStep)
	}

	// The budget is per step, so a session that is actually working gets a fresh
	// allowance every turn — the cap only bites a runaway, and a stuck/finished
	// session (whose step never advances) stays capped for good.
	if _, err := store.AdvanceStep(ctx, testRun, "main-agent"); err != nil {
		t.Fatal(err)
	}
	if code, _ := doReq(t, "POST", url, `{"content":"next turn"}`); code != 202 {
		t.Errorf("after AdvanceStep: status = %d, want 202 (budget resets per step)", code)
	}
}

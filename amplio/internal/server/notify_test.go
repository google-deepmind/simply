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
	"net/http/httptest"
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

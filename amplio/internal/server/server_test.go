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
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"amplio/internal/agent/critic"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/embed"
	"amplio/internal/event"
	"amplio/internal/eventstream"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	"amplio/internal/runtime"
	"amplio/internal/workspace/plain"

	// Register agent types so StartRun can resolve them.
	_ "amplio/internal/agent/chatbot"
	_ "amplio/internal/agent/standard"
)

const testRun = "r1"

func newTestServer(t *testing.T) (*Server, *eventstream.Bus, db.Store) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() }) // runs last
	t.Cleanup(cancel)                       // runs first: stop any run goroutines before close
	mgr := runtime.NewRunManager(store, func(string) (llm.Provider, error) { return &llm.MockProvider{Model: "m"}, nil }, runtime.NewRunRegistry(), plain.Factory)
	bus := eventstream.NewBus()
	srv := New(ctx, store, mgr, bus, "secret", RunDefaults{LLM: "vertex:x", LLMs: []string{"vertex:x", "vertex:y"}})
	return srv, bus, store
}

func seedRun(t *testing.T, store db.Store, status string, step int) {
	t.Helper()
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "main-agent", AgentType: "standard_agent", Status: status,
	}); err != nil {
		t.Fatal(err)
	}
	for range step {
		if _, err := store.AdvanceStep(ctx, testRun, "main-agent"); err != nil {
			t.Fatal(err)
		}
	}
}

// doReq issues a request, reads+closes the body, and returns status and body.
func doReq(t *testing.T, method, url, body string) (int, []byte) {
	t.Helper()
	var r io.Reader
	if body != "" {
		r = strings.NewReader(body)
	}
	req, err := http.NewRequestWithContext(context.Background(), method, url, r)
	if err != nil {
		t.Fatal(err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close() //nolint:errcheck
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, b
}

func TestServer_Auth(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 0)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Reads are open — the readonly share view needs no token.
	if code, _ := doReq(t, http.MethodGet, ts.URL+"/api/runs", ""); code != http.StatusOK {
		t.Errorf("read without token: %d, want 200 (reads open)", code)
	}

	// Writes require the token.
	patch := ts.URL + "/api/runs/" + testRun
	if code, _ := doReq(t, http.MethodPatch, patch, `{"starred":true}`); code != http.StatusUnauthorized {
		t.Errorf("write no token: %d, want 401", code)
	}
	if code, _ := doReq(t, http.MethodPatch, patch+"?token=wrong", `{"starred":true}`); code != http.StatusUnauthorized {
		t.Errorf("write wrong token: %d, want 401", code)
	}
	if code, _ := doReq(t, http.MethodPatch, patch+"?token=secret", `{"starred":true}`); code == http.StatusUnauthorized {
		t.Errorf("write good token: %d, want non-401 (auth passed)", code)
	}

	// /api/auth reports write capability (drives the readonly UI).
	authed := func(u string) bool {
		_, body := doReq(t, http.MethodGet, u, "")
		var a struct {
			Authed bool `json:"authed"`
		}
		_ = json.Unmarshal(body, &a)
		return a.Authed
	}
	if authed(ts.URL + "/api/auth") {
		t.Error("/api/auth without token: authed=true, want false")
	}
	if !authed(ts.URL + "/api/auth?token=secret") {
		t.Error("/api/auth with token: authed=false, want true")
	}
}

// The magic-link exchange: POST /api/auth/login with the token sets an HttpOnly
// cookie that then authorizes writes on its own (no token in the URL) — the
// basis for stripping the token and sharing token-less readonly links.
func TestServer_AuthCookieLogin(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 0)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	jar, _ := cookiejar.New(nil)
	client := &http.Client{Jar: jar}

	loginReq, err := http.NewRequestWithContext(t.Context(), http.MethodPost, ts.URL+"/api/auth/login?token=secret", nil)
	if err != nil {
		t.Fatal(err)
	}
	loginReq.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(loginReq)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("login: %d, want 200", resp.StatusCode)
	}
	u, _ := url.Parse(ts.URL)
	gotCookie := false
	for _, c := range jar.Cookies(u) {
		if c.Name == authCookie {
			gotCookie = true
		}
	}
	if !gotCookie {
		t.Fatal("login set no auth cookie")
	}

	// A write authorizes via the cookie alone (no ?token=).
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPatch, ts.URL+"/api/runs/"+testRun, strings.NewReader(`{"starred":true}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err = client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		t.Errorf("PATCH with cookie: 401, want authorized")
	}

	// A bad token is rejected and sets no cookie.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/auth/login?token=wrong", ""); code != http.StatusUnauthorized {
		t.Errorf("login wrong token: %d, want 401", code)
	}
}

func TestServer_ListAndGetRun(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 3)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs?token=secret", "")
	var page runsPage
	if err := json.Unmarshal(body, &page); err != nil {
		t.Fatal(err)
	}
	list := page.Runs
	if len(list) != 1 || list[0].RunID != testRun || list[0].RootStatus != db.SessionOngoing || list[0].RootStep != 3 {
		t.Fatalf("list = %+v", list)
	}
	if page.HasMore || page.NextCursor != "" {
		t.Errorf("single run should not report has_more: %+v", page)
	}

	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"?token=secret", "")
	var detail runDetail
	if err := json.Unmarshal(body, &detail); err != nil {
		t.Fatal(err)
	}
	if detail.RunID != testRun || len(detail.Sessions) != 1 || detail.Sessions[0].SessionID != "main-agent" {
		t.Fatalf("detail = %+v", detail)
	}
}

func TestServer_ListRuns_Pagination(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	// Three runs with strictly increasing created_at, so the newest-first keyset
	// walk is deterministic: run-3, run-2, run-1.
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i, id := range []string{"run-1", "run-2", "run-3"} {
		if err := store.CreateRun(ctx, db.RunRecord{RunID: id, CreatedAt: base.Add(time.Duration(i) * time.Minute)}); err != nil {
			t.Fatal(err)
		}
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	getPage := func(before string) runsPage {
		t.Helper()
		url := ts.URL + "/api/runs?token=secret&limit=2"
		if before != "" {
			url += "&before=" + before
		}
		_, body := doReq(t, http.MethodGet, url, "")
		var p runsPage
		if err := json.Unmarshal(body, &p); err != nil {
			t.Fatal(err)
		}
		return p
	}

	// Page 1: newest two (run-3, run-2), has_more=true.
	p1 := getPage("")
	if len(p1.Runs) != 2 || p1.Runs[0].RunID != "run-3" || p1.Runs[1].RunID != "run-2" {
		t.Fatalf("page 1 = %+v", p1.Runs)
	}
	if !p1.HasMore || p1.NextCursor == "" {
		t.Fatalf("page 1 should have more: %+v", p1)
	}

	// Page 2: the remaining run-1, has_more=false.
	p2 := getPage(p1.NextCursor)
	if len(p2.Runs) != 1 || p2.Runs[0].RunID != "run-1" {
		t.Fatalf("page 2 = %+v", p2.Runs)
	}
	if p2.HasMore || p2.NextCursor != "" {
		t.Fatalf("page 2 should be the last: %+v", p2)
	}
}

// The server threads the ?q= / ?starred= / ?grade= query params into the store
// filter, so they compose (AND) and paginate server-side.
func TestServer_ListRuns_SearchAndFilters(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	mk := func(id string, i int, r db.RunRecord) {
		r.RunID = id
		r.CreatedAt = base.Add(time.Duration(i) * time.Minute)
		if err := store.CreateRun(ctx, r); err != nil {
			t.Fatal(err)
		}
	}
	mk("run-alpha", 0, db.RunRecord{Title: "Refactor parser", Config: config.RunConfig{Task: "rewrite", Workspace: "/w/proj-a"}, Starred: true, Grade: 5})
	mk("run-beta", 1, db.RunRecord{Title: "Fix bug", Config: config.RunConfig{Task: "parser crash", Workspace: "/w/proj-b"}})

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	ids := func(query string) []string {
		t.Helper()
		_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs?token=secret&"+query, "")
		var p runsPage
		if err := json.Unmarshal(body, &p); err != nil {
			t.Fatal(err)
		}
		out := make([]string, len(p.Runs))
		for i, r := range p.Runs {
			out[i] = r.RunID
		}
		return out
	}

	// Search over title (alpha) + task (beta): "parser" hits both, newest-first.
	if got := ids("q=parser"); len(got) != 2 || got[0] != "run-beta" || got[1] != "run-alpha" {
		t.Errorf("q=parser = %v, want [run-beta run-alpha]", got)
	}
	// Search over workspace (in config_json).
	if got := ids("q=proj-a"); len(got) != 1 || got[0] != "run-alpha" {
		t.Errorf("q=proj-a = %v, want [run-alpha]", got)
	}
	// starred=1.
	if got := ids("starred=1"); len(got) != 1 || got[0] != "run-alpha" {
		t.Errorf("starred=1 = %v, want [run-alpha]", got)
	}
	// grade=excellent (rank 5).
	if got := ids("grade=excellent"); len(got) != 1 || got[0] != "run-alpha" {
		t.Errorf("grade=excellent = %v, want [run-alpha]", got)
	}
	// Compose: q=parser AND starred=1 => alpha only.
	if got := ids("q=parser&starred=1"); len(got) != 1 || got[0] != "run-alpha" {
		t.Errorf("q=parser&starred=1 = %v, want [run-alpha]", got)
	}
}

func TestServer_RunCountsAndFilterAndSeen(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	now := time.Now().UTC()
	// run-a: ongoing (active, no update). run-b: concluded (done + update).
	mk := func(id, final string, off time.Duration) {
		if err := store.CreateRun(ctx, db.RunRecord{RunID: id, CreatedAt: now.Add(off)}); err != nil {
			t.Fatal(err)
		}
		if err := store.CreateSession(ctx, db.SessionRecord{
			RunID: id, SessionID: "main-agent", AgentType: "standard_agent",
			Status: db.SessionOngoing, CreatedAt: now.Add(off),
		}); err != nil {
			t.Fatal(err)
		}
		if final != db.SessionOngoing {
			if err := store.UpdateSessionStatus(ctx, id, "main-agent", final); err != nil {
				t.Fatal(err)
			}
		}
	}
	mk("run-a", db.SessionOngoing, -1*time.Minute)
	mk("run-b", db.SessionConcluded, -2*time.Minute)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Counts endpoint.
	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/counts?token=secret", "")
	var c runCounts
	if err := json.Unmarshal(body, &c); err != nil {
		t.Fatal(err)
	}
	if c.Active != 1 || c.Updates != 1 {
		t.Fatalf("counts = %+v, want {active:1 updates:1}", c)
	}

	// Server-side ?filter=updates returns only run-b, with has_updates=true.
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs?filter=updates&token=secret", "")
	var p runsPage
	if err := json.Unmarshal(body, &p); err != nil {
		t.Fatal(err)
	}
	if len(p.Runs) != 1 || p.Runs[0].RunID != "run-b" || !p.Runs[0].HasUpdates {
		t.Fatalf("filter=updates = %+v", p.Runs)
	}

	// PATCH seen:true clears run-b from the updates count.
	code, _ := doReq(t, http.MethodPatch, ts.URL+"/api/runs/run-b?token=secret", `{"seen":true}`)
	if code != http.StatusOK {
		t.Fatalf("patch seen: %d", code)
	}
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs/counts?token=secret", "")
	if err := json.Unmarshal(body, &c); err != nil {
		t.Fatal(err)
	}
	if c.Updates != 0 {
		t.Errorf("updates after seen = %d, want 0", c.Updates)
	}
}
func TestServer_UpdateRun(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 0)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, _ := doReq(t, http.MethodPatch, ts.URL+"/api/runs/"+testRun+"?token=secret",
		`{"title":"Renamed","starred":true,"archived":true}`)
	if code != http.StatusOK {
		t.Fatalf("PATCH: %d, want 200", code)
	}

	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"?token=secret", "")
	var detail runDetail
	if err := json.Unmarshal(body, &detail); err != nil {
		t.Fatal(err)
	}
	if detail.Title != "Renamed" || !detail.Starred || !detail.Archived {
		t.Fatalf("after PATCH: %+v", detail)
	}
	// Grade is absent in the body above, so it stays ungraded (null).
	if detail.Grade != nil {
		t.Fatalf("grade should be untouched (null), got %v", *detail.Grade)
	}

	// Set a grade by string; it round-trips as the same string.
	code, _ = doReq(t, http.MethodPatch, ts.URL+"/api/runs/"+testRun+"?token=secret",
		`{"grade":"good"}`)
	if code != http.StatusOK {
		t.Fatalf("PATCH grade: %d, want 200", code)
	}
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"?token=secret", "")
	detail = runDetail{}
	if err := json.Unmarshal(body, &detail); err != nil {
		t.Fatal(err)
	}
	if detail.Grade == nil || *detail.Grade != "good" {
		t.Fatalf("after grade PATCH: %+v", detail.Grade)
	}

	// An unknown grade string is rejected with 400.
	if code, _ := doReq(t, http.MethodPatch, ts.URL+"/api/runs/"+testRun+"?token=secret",
		`{"grade":"superb"}`); code != http.StatusBadRequest {
		t.Fatalf("bad grade PATCH: %d, want 400", code)
	}

	// Explicit null clears the grade back to ungraded.
	code, _ = doReq(t, http.MethodPatch, ts.URL+"/api/runs/"+testRun+"?token=secret",
		`{"grade":null}`)
	if code != http.StatusOK {
		t.Fatalf("PATCH clear grade: %d, want 200", code)
	}
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"?token=secret", "")
	detail = runDetail{}
	if err := json.Unmarshal(body, &detail); err != nil {
		t.Fatal(err)
	}
	if detail.Grade != nil {
		t.Fatalf("after clear grade: %v, want null", *detail.Grade)
	}

	// Archived run is hidden by default, visible with ?archived=1.
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs?token=secret", "")
	var def runsPage
	if err := json.Unmarshal(body, &def); err != nil {
		t.Fatal(err)
	}
	if len(def.Runs) != 0 {
		t.Errorf("default list should hide archived, got %+v", def.Runs)
	}
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs?archived=1&token=secret", "")
	var all runsPage
	if err := json.Unmarshal(body, &all); err != nil {
		t.Fatal(err)
	}
	if len(all.Runs) != 1 || !all.Runs[0].Archived || all.Runs[0].Title != "Renamed" {
		t.Fatalf("archived list = %+v", all.Runs)
	}
}

func TestServer_Models(t *testing.T) {
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	get := func() modelMenu {
		_, body := doReq(t, http.MethodGet, ts.URL+"/api/models?token=secret", "")
		var m modelMenu
		if err := json.Unmarshal(body, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}

	m := get()
	if m.Default != "vertex:x" || len(m.Models) != 2 || m.Models[0].Removable || m.Models[1].Removable {
		t.Fatalf("initial menu = %+v, want 2 read-only config models", m)
	}

	code, _ := doReq(t, http.MethodPost, ts.URL+"/api/models?token=secret", `{"spec":"vertex:custom"}`)
	if code != http.StatusCreated {
		t.Fatalf("POST model: %d, want 201", code)
	}
	m = get()
	if len(m.Models) != 3 {
		t.Fatalf("after add: %+v, want 3", m)
	}
	last := m.Models[2]
	if last.Spec != "vertex:custom" || !last.Removable {
		t.Errorf("custom entry = %+v, want vertex:custom removable", last)
	}

	code, _ = doReq(t, http.MethodDelete,
		ts.URL+"/api/models?spec="+url.QueryEscape("vertex:custom")+"&token=secret", "")
	if code != http.StatusOK {
		t.Fatalf("DELETE model: %d, want 200", code)
	}
	if m = get(); len(m.Models) != 2 {
		t.Fatalf("after remove: %+v, want 2", m)
	}

	// A "#nickname" relabels an endpoint that is already in the menu. Both entries
	// stay: dropping either would discard something the operator asked for (the
	// config entry can't be removed from the UI; the nickname is the whole point
	// of the custom one). Both are flagged instead, since two rows that start
	// IDENTICAL runs under different labels is usually a leftover.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/models?token=secret",
		`{"spec":"vertex:x#house-model"}`); code != http.StatusCreated {
		t.Fatalf("POST nicknamed model: %d, want 201", code)
	}
	m = get()
	if len(m.Models) != 3 {
		t.Fatalf("after nickname add: %+v, want 3 (nickname must NOT collapse onto the plain spec)", m)
	}
	for _, e := range m.Models {
		switch e.Spec {
		case "vertex:x", "vertex:x#house-model":
			if !e.Duplicate {
				t.Errorf("entry %q: Duplicate = false, want true (shares a provider spec)", e.Spec)
			}
		default:
			if e.Duplicate {
				t.Errorf("entry %q: Duplicate = true, want false", e.Spec)
			}
		}
		if e.Label == "" {
			t.Errorf("entry %q has no label", e.Spec)
		}
	}
	// The nickname is what the UI shows for that row — the point of the override.
	for _, e := range m.Models {
		if e.Spec == "vertex:x#house-model" && e.Label != "house-model" {
			t.Errorf("nicknamed entry label = %q, want %q", e.Label, "house-model")
		}
	}
}

func TestServer_StartInteractiveRun(t *testing.T) {
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Interactive mode requires a message.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs?token=secret", `{"interactive":true}`); code != http.StatusBadRequest {
		t.Errorf("interactive without message: %d, want 400", code)
	}
	// Autonomous mode still requires a task.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs?token=secret", `{}`); code != http.StatusBadRequest {
		t.Errorf("autonomous without task: %d, want 400", code)
	}

	// A valid interactive run is created with an empty task (the canonical signal).
	code, body := doReq(t, http.MethodPost, ts.URL+"/api/runs?token=secret", `{"interactive":true,"message":"hi there"}`)
	if code != http.StatusCreated {
		t.Fatalf("interactive start: %d, want 201", code)
	}
	var created struct {
		RunID string `json:"run_id"`
	}
	if err := json.Unmarshal(body, &created); err != nil || created.RunID == "" {
		t.Fatalf("bad create response: %s", body)
	}
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs/"+created.RunID+"?token=secret", "")
	var detail runDetail
	if err := json.Unmarshal(body, &detail); err != nil {
		t.Fatal(err)
	}
	if detail.Task != "" {
		t.Errorf("interactive run task = %q, want empty", detail.Task)
	}
}

func TestServer_StartChatbotSidecar(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 1) // autonomous run with an active main-agent
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, body := doReq(t, http.MethodPost, ts.URL+"/api/runs/"+testRun+"/chatbot?token=secret", "")
	if code != http.StatusCreated {
		t.Fatalf("attach chatbot: %d, want 201 (%s)", code, body)
	}
	var got struct {
		SessionID string `json:"session_id"`
	}
	if err := json.Unmarshal(body, &got); err != nil || got.SessionID != "chatty-bot" {
		t.Fatalf("bad response: %s", body)
	}

	// The sidecar bootstraps into a second root session.
	waitForSession(t, store, "chatty-bot")

	// Idempotent: a second attach returns the same id without erroring.
	if code, body = doReq(t, http.MethodPost, ts.URL+"/api/runs/"+testRun+"/chatbot?token=secret", ""); code != http.StatusCreated {
		t.Fatalf("re-attach: %d (%s)", code, body)
	}

	// The run now has two roots, but primary status still tracks the autonomous
	// main-agent (not the chatbot).
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs?token=secret", "")
	var page runsPage
	if err := json.Unmarshal(body, &page); err != nil || len(page.Runs) != 1 {
		t.Fatalf("list runs: %s", body)
	}
	runs := page.Runs
	if len(runs[0].Roots) != 2 {
		t.Fatalf("roots = %d, want 2: %+v", len(runs[0].Roots), runs[0].Roots)
	}
	if runs[0].RootSessionID != "main-agent" || runs[0].RootStatus != db.SessionOngoing {
		t.Errorf("primary root = %s/%s, want main-agent/ongoing", runs[0].RootSessionID, runs[0].RootStatus)
	}
}

func TestServer_Trajectory(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionOngoing,
	}); err != nil {
		t.Fatal(err)
	}
	for range 4 {
		if _, err := store.AdvanceStep(ctx, testRun, "main-agent"); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.FinalizeStep(ctx, testRun, "main-agent", 4, []event.Event{
		&event.AssistantEvent{Content: "work at step 4"},
	}); err != nil {
		t.Fatal(err)
	}
	// Advance once more so the session is "at rest": current_step (5) has no
	// events; the last event is at step 4.
	if _, err := store.AdvanceStep(ctx, testRun, "main-agent"); err != nil {
		t.Fatal(err)
	}
	stepObs := func(id string, step int, summary, tag string) {
		t.Helper()
		s := step
		if err := store.AppendObservation(ctx, db.ObservationRecord{
			ObsID: id, RunID: testRun, Kind: "step_summary", SessionID: "main-agent", Step: &s,
			Data: map[string]any{"summary": summary, "status_tag": tag},
		}); err != nil {
			t.Fatal(err)
		}
	}
	stepObs("s1", 1, "did one", "progressing")
	stepObs("s2", 2, "did two", "blocked")
	// A lesson the agent loaded, so the trajectory can resolve its title.
	if err := store.InsertLesson(ctx, db.LessonRecord{
		LessonID: "lz1", Title: "Retry flaky builds", Description: "d", Body: "b", EmbedderID: "e",
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "ph1", RunID: testRun, Kind: "phase_summary", SessionID: "main-agent",
		Data: map[string]any{
			"title": "Phase A", "summary": "phase sum", "start_step": 1, "end_step": 2,
			"artifacts": []map[string]any{
				{"kind": "path", "value": "/foo.go", "context": "edited main"},
			},
			"lesson_verdicts": []map[string]any{
				{"handle": "lesson:lz1", "verdict": "helpful", "reason": "applied the retry flag"},
				{"handle": "skill:deploy", "verdict": "cited"}, // skill: excluded
			},
		},
	}); err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/main-agent/trajectory?token=secret", "")
	var tr trajectory
	if err := json.Unmarshal(body, &tr); err != nil {
		t.Fatal(err)
	}
	if tr.CurrentStep != 5 {
		t.Fatalf("current_step=%d, want 5", tr.CurrentStep)
	}
	if len(tr.Phases) != 1 || len(tr.Phases[0].Steps) != 2 {
		t.Fatalf("phases=%+v, want 1 phase with 2 steps", tr.Phases)
	}
	if tr.Phases[0].Steps[0].Summary != "did one" || tr.Phases[0].Steps[1].StatusTag != "blocked" {
		t.Errorf("phase steps = %+v", tr.Phases[0].Steps)
	}
	if len(tr.Phases[0].Artifacts) != 1 || tr.Phases[0].Artifacts[0].Value != "/foo.go" {
		t.Fatalf("phase artifacts = %+v, want 1 with value /foo.go", tr.Phases[0].Artifacts)
	}
	// lesson_verdicts: skill handle excluded, lesson title resolved from the corpus.
	lv := tr.Phases[0].LessonVerdicts
	if len(lv) != 1 {
		t.Fatalf("lesson_verdicts = %+v, want 1 (skill excluded)", lv)
	}
	if lv[0].ID != "lz1" || lv[0].Title != "Retry flaky builds" || lv[0].Verdict != "helpful" || lv[0].Reason != "applied the retry flag" {
		t.Errorf("lesson verdict = %+v", lv[0])
	}
	// Loose tail is bounded by the last event (step 4); the empty trailing
	// step 5 is skipped.
	if len(tr.LooseSteps) != 2 || tr.LooseSteps[0].Step != 3 || tr.LooseSteps[1].Step != 4 {
		t.Fatalf("loose_steps = %+v, want steps 3,4 (empty step 5 skipped)", tr.LooseSteps)
	}

	// Per-step event fetch returns only that step's events.
	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/main-agent/events?step=4&token=secret", "")
	var evs []eventDTO
	if err := json.Unmarshal(body, &evs); err != nil {
		t.Fatal(err)
	}
	if len(evs) != 1 || !strings.Contains(string(evs[0].Event), "work at step 4") {
		t.Fatalf("step 4 events = %+v, want the single step-4 event", evs)
	}
}

func waitForSession(t *testing.T, store db.Store, sid string) {
	t.Helper()
	ctx := context.Background()
	for range 200 {
		if _, err := store.GetSession(ctx, testRun, sid); err == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("session %q did not appear", sid)
}

func TestServer_ChatProjection(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "chatty-bot", AgentType: "chatbot", Status: db.SessionIdle,
	}); err != nil {
		t.Fatal(err)
	}
	finalize := func(step int, evs ...event.Event) {
		t.Helper()
		for range step {
			if _, err := store.AdvanceStep(ctx, testRun, "chatty-bot"); err != nil {
				t.Fatal(err)
			}
		}
		if err := store.FinalizeStep(ctx, testRun, "chatty-bot", step, evs); err != nil {
			t.Fatal(err)
		}
	}
	// Step 1 is rolled into a phase card (boundary=2), so it must NOT appear.
	finalize(1, &event.AssistantEvent{Content: "ROLLED UP PHASE TEXT"})
	// Steps 2→3, then the live turn at step 3.
	finalize(2, &event.UserEvent{Content: "hello"},
		&event.AssistantEvent{
			Content:  "hi there",
			Thoughts: "pondering",
			ToolCalls: []event.ToolCall{
				{ID: "tc1", Name: "view_file"},
				{ID: "tc2", Name: "bash"},
			},
		},
		&event.ToolResultEvent{ToolCallID: "tc1", Content: "TOOL RESULT BODY SECRET", IsError: true},
	)
	mkObs := func(id, title, summary string, start, end int) {
		t.Helper()
		if err := store.AppendObservation(ctx, db.ObservationRecord{
			ObsID: id, RunID: testRun, Kind: "phase_summary", SessionID: "chatty-bot",
			Data: map[string]any{"title": title, "summary": summary, "start_step": start, "end_step": end},
		}); err != nil {
			t.Fatal(err)
		}
	}
	mkObs("p1", "Phase A", "did A", 1, 1) // rolled into a card
	mkObs("p2", "Phase B", "did B", 2, 3) // newest closed phase stays inline

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/chatty-bot/chat?token=secret", "")

	if strings.Contains(string(body), "TOOL RESULT BODY SECRET") {
		t.Errorf("tool_result body leaked into chat feed: %s", body)
	}
	var feed chatFeed
	if err := json.Unmarshal(body, &feed); err != nil {
		t.Fatal(err)
	}
	if len(feed.PhaseCards) != 1 || feed.PhaseCards[0].Title != "Phase A" {
		t.Fatalf("phase_cards = %+v, want 1 card 'Phase A'", feed.PhaseCards)
	}
	if len(feed.Messages) != 2 {
		t.Fatalf("messages = %d, want 2 (rolled-up step hidden): %+v", len(feed.Messages), feed.Messages)
	}
	op, bot := feed.Messages[0], feed.Messages[1]
	if op.Kind != "operator" || op.Content != "hello" {
		t.Errorf("first bubble = %+v, want operator 'hello'", op)
	}
	if bot.Kind != "chatbot" || bot.Content != "hi there" || bot.Thoughts != "pondering" {
		t.Errorf("second bubble = %+v, want chatbot 'hi there' thoughts 'pondering'", bot)
	}
	if len(bot.ToolCalls) != 2 || !bot.ToolCalls[0].Completed || bot.ToolCalls[1].Completed {
		t.Errorf("tool_calls = %+v, want view_file completed, bash pending", bot.ToolCalls)
	}
	// tc1's result carried IsError, so the projected chip is flagged errored;
	// tc2 (pending, no result) is not.
	if !bot.ToolCalls[0].Errored || bot.ToolCalls[1].Errored {
		t.Errorf("tool_calls errored = [%v %v], want [true false]", bot.ToolCalls[0].Errored, bot.ToolCalls[1].Errored)
	}
}

// A RANGED chat request (from_step/to_step) is the read-only session-log viewer
// browsing one closed phase: the phase's own turns come back as bubbles (no
// boundary rollup), with no cards and no usage. It also works on a non-chatbot
// session — an autonomous agent's turns project identically.
func TestServer_ChatProjection_Ranged(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionConcluded,
	}); err != nil {
		t.Fatal(err)
	}
	finalize := func(step int, evs ...event.Event) {
		t.Helper()
		for range step {
			if _, err := store.AdvanceStep(ctx, testRun, "main-agent"); err != nil {
				t.Fatal(err)
			}
		}
		if err := store.FinalizeStep(ctx, testRun, "main-agent", step, evs); err != nil {
			t.Fatal(err)
		}
	}
	finalize(1, &event.AssistantEvent{
		Content:   "EARLY PHASE TEXT",
		ToolCalls: []event.ToolCall{{ID: "tc1", Name: "bash"}},
		Usage:     &event.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
	})
	// The call's result is pinned to the SAME step, so a one-step range still
	// sees it and marks the chip completed.
	finalize(2, &event.AssistantEvent{Content: "LATER PHASE TEXT"})
	if err := store.AppendEventAtStep(ctx, testRun, "main-agent", 1,
		&event.ToolResultEvent{ToolCallID: "tc1", Content: "RESULT BODY"}); err != nil {
		t.Fatal(err)
	}
	for _, o := range []struct {
		id, title  string
		start, end int
	}{{"p1", "Phase A", 1, 1}, {"p2", "Phase B", 2, 2}} {
		if err := store.AppendObservation(ctx, db.ObservationRecord{
			ObsID: o.id, RunID: testRun, Kind: "phase_summary", SessionID: "main-agent",
			Data: map[string]any{"title": o.title, "start_step": o.start, "end_step": o.end},
		}); err != nil {
			t.Fatal(err)
		}
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	base := ts.URL + "/api/runs/" + testRun + "/sessions/main-agent/chat"

	// Live mode: step 1 is rolled into a card and hidden.
	_, body := doReq(t, http.MethodGet, base+"?token=secret", "")
	var live chatFeed
	if err := json.Unmarshal(body, &live); err != nil {
		t.Fatal(err)
	}
	if len(live.PhaseCards) != 1 || len(live.Messages) != 1 || live.Messages[0].Content != "LATER PHASE TEXT" {
		t.Fatalf("live feed = %d cards / %+v messages, want 1 card + only the inline phase", len(live.PhaseCards), live.Messages)
	}

	// Ranged mode: exactly the requested phase, no cards, no usage.
	_, body = doReq(t, http.MethodGet, base+"?from_step=1&to_step=1&token=secret", "")
	var ranged chatFeed
	if err := json.Unmarshal(body, &ranged); err != nil {
		t.Fatal(err)
	}
	if len(ranged.Messages) != 1 || ranged.Messages[0].Content != "EARLY PHASE TEXT" {
		t.Fatalf("ranged messages = %+v, want only the step-1 turn", ranged.Messages)
	}
	if len(ranged.PhaseCards) != 0 {
		t.Errorf("ranged phase_cards = %+v, want none (client has the phase index)", ranged.PhaseCards)
	}
	if ranged.Usage != nil {
		t.Errorf("ranged usage = %+v, want nil (a historical slice has no latest turn)", ranged.Usage)
	}
	if tcs := ranged.Messages[0].ToolCalls; len(tcs) != 1 || !tcs[0].Completed {
		t.Errorf("ranged tool_calls = %+v, want the same-step result to mark it completed", tcs)
	}
	if strings.Contains(string(body), "RESULT BODY") {
		t.Errorf("tool_result body leaked into the ranged feed: %s", body)
	}
}

func TestServer_ChatProjection_InboundMessages(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "chatty-bot", AgentType: "chatbot", Status: db.SessionIdle,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AdvanceStep(ctx, testRun, "chatty-bot"); err != nil {
		t.Fatal(err)
	}
	// A user turn plus two inbound MessageEvents: one from a peer agent
	// (send_message) and one from the environment (amplio notify, shell output).
	if err := store.FinalizeStep(ctx, testRun, "chatty-bot", 1, []event.Event{
		&event.UserEvent{Content: "status?"},
		&event.MessageEvent{Content: "peer update", Sender: "worker-7", SenderType: event.SenderTypeAgent},
		&event.MessageEvent{Content: "+ exit 0\n", Sender: "build.sh", SenderType: event.SenderTypeEnvironment},
	}); err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/chatty-bot/chat?token=secret", "")
	var feed chatFeed
	if err := json.Unmarshal(body, &feed); err != nil {
		t.Fatal(err)
	}
	if len(feed.Messages) != 3 {
		t.Fatalf("messages = %d, want 3: %+v", len(feed.Messages), feed.Messages)
	}
	agent, env := feed.Messages[1], feed.Messages[2]
	if agent.Kind != "agent" || agent.Content != "peer update" || agent.From != "worker-7" {
		t.Errorf("agent bubble = %+v, want agent 'peer update' from worker-7", agent)
	}
	if env.Kind != "environment" || env.Content != "+ exit 0\n" || env.From != "build.sh" {
		t.Errorf("env bubble = %+v, want environment '+ exit 0' from build.sh", env)
	}
}

// A sub-agent's terminal result (ChildResultEvent, posted back to the parent
// chatbot's stream) surfaces as a child_result bubble carrying the verdict.
func TestServer_ChatProjection_ChildResult(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "chatty-bot", AgentType: "chatbot", Status: db.SessionIdle,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AdvanceStep(ctx, testRun, "chatty-bot"); err != nil {
		t.Fatal(err)
	}
	if err := store.FinalizeStep(ctx, testRun, "chatty-bot", 1, []event.Event{
		&event.ChildResultEvent{ChildSessionID: "worker-3", Verdict: db.SessionConcluded, Content: "found the bug"},
		&event.ChildResultEvent{ChildSessionID: "worker-9", Verdict: db.SessionCrashed, Content: "panic: boom"},
	}); err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/chatty-bot/chat?token=secret", "")
	var feed chatFeed
	if err := json.Unmarshal(body, &feed); err != nil {
		t.Fatal(err)
	}
	if len(feed.Messages) != 2 {
		t.Fatalf("messages = %d, want 2: %+v", len(feed.Messages), feed.Messages)
	}
	concluded, crashed := feed.Messages[0], feed.Messages[1]
	if concluded.Kind != "child_result" || concluded.Verdict != db.SessionConcluded ||
		concluded.From != "worker-3" || concluded.Content != "found the bug" {
		t.Errorf("concluded bubble = %+v, want child_result concluded from worker-3", concluded)
	}
	if crashed.Kind != "child_result" || crashed.Verdict != db.SessionCrashed || crashed.From != "worker-9" {
		t.Errorf("crashed bubble = %+v, want child_result crashed from worker-9", crashed)
	}
}

func TestServer_ChatProjection_Compaction(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "chatty-bot", AgentType: "chatbot", Status: db.SessionIdle,
	}); err != nil {
		t.Fatal(err)
	}
	// Two real turns, then compact the context at the boundary. CompactContext
	// writes a CompactionEvent into the stream at the boundary step.
	if _, err := store.AdvanceStep(ctx, testRun, "chatty-bot"); err != nil {
		t.Fatal(err)
	}
	if err := store.FinalizeStep(ctx, testRun, "chatty-bot", 1, []event.Event{
		&event.UserEvent{Content: "first"},
		&event.AssistantEvent{Content: "reply one"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AdvanceStep(ctx, testRun, "chatty-bot"); err != nil {
		t.Fatal(err)
	}
	if err := store.FinalizeStep(ctx, testRun, "chatty-bot", 2, []event.Event{
		&event.UserEvent{Content: "second"},
		&event.AssistantEvent{Content: "reply two"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CompactContext(ctx, testRun, "chatty-bot", 1, "SUMMARY OF EARLIER WORK"); err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/chatty-bot/chat?token=secret", "")
	var feed chatFeed
	if err := json.Unmarshal(body, &feed); err != nil {
		t.Fatal(err)
	}
	var comp *chatBubble
	for i := range feed.Messages {
		if feed.Messages[i].Kind == "compaction" {
			comp = &feed.Messages[i]
			break
		}
	}
	if comp == nil {
		t.Fatalf("no compaction bubble in feed: %+v", feed.Messages)
	}
	if comp.Content != "SUMMARY OF EARLIER WORK" {
		t.Errorf("compaction content = %q, want the summary", comp.Content)
	}
}
func TestServer_EventsAndObservations(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	seedRun(t, store, db.SessionConcluded, 1)
	if err := store.FinalizeStep(ctx, testRun, "main-agent", 1, []event.Event{&event.AssistantEvent{Content: "hi"}}); err != nil {
		t.Fatal(err)
	}
	step := 1
	if err := store.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "step_summary-main-agent-1", RunID: testRun, Kind: "step_summary",
		SessionID: "main-agent", Step: &step, CharCount: 5, Data: map[string]any{"summary": "did hi"},
	}); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/main-agent/events?token=secret", "")
	var evs []eventDTO
	if err := json.Unmarshal(body, &evs); err != nil {
		t.Fatal(err)
	}
	if len(evs) != 1 || !strings.Contains(string(evs[0].Event), "hi") {
		t.Fatalf("events = %+v", evs)
	}

	_, body = doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/sessions/main-agent/observations?token=secret", "")
	var obs []observationDTO
	if err := json.Unmarshal(body, &obs); err != nil {
		t.Fatal(err)
	}
	if len(obs) != 1 || obs[0].Kind != "step_summary" || obs[0].Data["summary"] != "did hi" {
		t.Fatalf("observations = %+v", obs)
	}
}

// from_step/to_step fetch an inclusive step RANGE in one request (the log
// viewer's "expand all" over a phase). An explicit step=N still wins.
func TestServer_EventsStepRange(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateSession(ctx, db.SessionRecord{
		RunID: testRun, SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionOngoing,
	}); err != nil {
		t.Fatal(err)
	}
	for step := 1; step <= 4; step++ {
		if _, err := store.AdvanceStep(ctx, testRun, "main-agent"); err != nil {
			t.Fatal(err)
		}
		if err := store.FinalizeStep(ctx, testRun, "main-agent", step, []event.Event{
			&event.AssistantEvent{Content: fmt.Sprintf("turn %d", step)},
		}); err != nil {
			t.Fatal(err)
		}
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	base := ts.URL + "/api/runs/" + testRun + "/sessions/main-agent/events"
	steps := func(query string) []int {
		t.Helper()
		_, body := doReq(t, http.MethodGet, base+query+"&token=secret", "")
		var evs []eventDTO
		if err := json.Unmarshal(body, &evs); err != nil {
			t.Fatal(err)
		}
		out := make([]int, 0, len(evs))
		for _, e := range evs {
			out = append(out, e.Step)
		}
		return out
	}

	if got, want := steps("?from_step=2&to_step=3"), []int{2, 3}; !slices.Equal(got, want) {
		t.Errorf("from_step=2&to_step=3 = %v, want %v", got, want)
	}
	if got, want := steps("?from_step=3"), []int{3, 4}; !slices.Equal(got, want) {
		t.Errorf("from_step=3 (open end) = %v, want %v", got, want)
	}
	if got, want := steps("?to_step=2"), []int{1, 2}; !slices.Equal(got, want) {
		t.Errorf("to_step=2 (open start) = %v, want %v", got, want)
	}
	if got, want := steps("?step=2&from_step=1&to_step=4"), []int{2}; !slices.Equal(got, want) {
		t.Errorf("step=2 with a range = %v, want %v (explicit step wins)", got, want)
	}
	if got, want := steps("?from_step=bogus"), []int{1, 2, 3, 4}; !slices.Equal(got, want) {
		t.Errorf("unparsable bound = %v, want the full stream %v", got, want)
	}
}

func TestServer_StartRunValidation(t *testing.T) {
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs?token=secret", `{}`); code != http.StatusBadRequest {
		t.Errorf("start without task: %d, want 400", code)
	}
}

func TestServer_SendMessageAndCancel(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionIdle, 1)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, body := doReq(t, http.MethodPost, ts.URL+"/api/runs/"+testRun+"/sessions/main-agent/message?token=secret", `{"content":"hello"}`)
	if code != http.StatusAccepted {
		t.Fatalf("send message: %d, want 202", code)
	}
	// The response carries the created event's id so a writer can identify its
	// own injected message in the stream without re-matching by content.
	var sendResp struct {
		Status  string `json:"status"`
		EventID string `json:"event_id"`
	}
	if err := json.Unmarshal(body, &sendResp); err != nil {
		t.Fatal(err)
	}
	if sendResp.EventID == "" {
		t.Error("send message response missing event_id")
	}
	evs, _ := store.GetEvents(context.Background(), testRun, "main-agent", db.EventFilter{})
	foundUser := false
	for _, e := range evs {
		if _, ok := e.Event.(*event.UserEvent); ok {
			foundUser = true
			if e.EventID != sendResp.EventID {
				t.Errorf("returned event_id %q != stored %q", sendResp.EventID, e.EventID)
			}
		}
	}
	if !foundUser {
		t.Error("message did not append a UserEvent")
	}

	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs/"+testRun+"/cancel?token=secret", ""); code != http.StatusAccepted {
		t.Fatalf("cancel: %d, want 202", code)
	}
	sess, _ := store.GetSession(context.Background(), testRun, "main-agent")
	if sess.Status != db.SessionCancelled {
		t.Errorf("session status = %q, want cancelled", sess.Status)
	}
}

func TestServer_DeleteRun(t *testing.T) {
	srv, _, store := newTestServer(t)
	seedRun(t, store, db.SessionConcluded, 2)
	ctx := context.Background()
	// Give the run an event + an observation, and a lesson sourced from it, so we
	// can assert the delete removes run-owned rows but KEEPS the mined lesson.
	if err := store.FinalizeStep(ctx, testRun, "main-agent", 1,
		[]event.Event{&event.AssistantEvent{Content: "work"}}); err != nil {
		t.Fatal(err)
	}
	step := 1
	if err := store.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "o1", RunID: testRun, Kind: "step_summary", SessionID: "main-agent", Step: &step,
		Data: map[string]any{"summary": "s"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.InsertLesson(ctx, db.LessonRecord{
		LessonID: "L1", Title: "keep me", Description: "d", Body: "b", EmbedderID: "e", SourceRunID: testRun,
	}); err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Unauth'd delete is rejected (write endpoint).
	if code, _ := doReq(t, http.MethodDelete, ts.URL+"/api/runs/"+testRun, ""); code != http.StatusUnauthorized {
		t.Fatalf("unauth delete: %d, want 401", code)
	}

	if code, _ := doReq(t, http.MethodDelete, ts.URL+"/api/runs/"+testRun+"?token=secret", ""); code != http.StatusOK {
		t.Fatalf("delete: %d, want 200", code)
	}

	// Run + all its child rows are gone.
	if _, err := store.GetRun(ctx, testRun); err == nil {
		t.Error("run still exists after delete")
	}
	if sessions, _ := store.ListSessions(ctx, testRun); len(sessions) != 0 {
		t.Errorf("sessions = %d after delete, want 0", len(sessions))
	}
	if evs, _ := store.GetEvents(ctx, testRun, "main-agent", db.EventFilter{}); len(evs) != 0 {
		t.Errorf("events = %d after delete, want 0", len(evs))
	}
	if obs, _ := store.GetObservations(ctx, testRun, db.ObsFilter{}); len(obs) != 0 {
		t.Errorf("observations = %d after delete, want 0", len(obs))
	}
	// Mined lessons survive (soft source_run_id reference, not an FK).
	if _, err := store.GetLesson(ctx, "L1"); err != nil {
		t.Errorf("lesson L1 should survive run delete: %v", err)
	}

	// Idempotent: deleting an already-gone run is a clean no-op (200).
	if code, _ := doReq(t, http.MethodDelete, ts.URL+"/api/runs/"+testRun+"?token=secret", ""); code != http.StatusOK {
		t.Fatalf("re-delete: %d, want 200 (idempotent)", code)
	}
}

func TestServer_About(t *testing.T) {
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// About is an open read: surfaces version + on-disk layout + configured tiers.
	code, body := doReq(t, http.MethodGet, ts.URL+"/api/about", "")
	if code != http.StatusOK {
		t.Fatalf("about: %d, want 200", code)
	}
	var info aboutInfo
	if err := json.Unmarshal(body, &info); err != nil {
		t.Fatal(err)
	}
	if info.GoVersion == "" || info.DataDir == "" {
		t.Errorf("about missing core fields: %+v", info)
	}
	if info.DefaultLLM != "vertex:x" || len(info.Models) != 2 {
		t.Errorf("about model fields = %q / %v", info.DefaultLLM, info.Models)
	}
}

func TestServer_TestLLM(t *testing.T) {
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Unconfigured (no tester hook) → 501.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/about/test-llm?token=secret", `{"spec":"x:y"}`); code != http.StatusNotImplemented {
		t.Fatalf("test-llm unconfigured: %d, want 501", code)
	}

	// Wire a stub tester: echoes success for a known spec, errors otherwise.
	srv.SetLLMTester(func(_ context.Context, spec string) (string, string, time.Duration, error) {
		if spec == "good:model" {
			return "good/model", "OK", 42 * time.Millisecond, nil
		}
		return "", "", 0, fmt.Errorf("unknown LLM provider %q", spec)
	})

	// Auth required (real billable call) → 401 without token.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/about/test-llm", `{"spec":"good:model"}`); code != http.StatusUnauthorized {
		t.Fatalf("test-llm unauth: %d, want 401", code)
	}

	// Success path: ok=true with model id + latency.
	code, body := doReq(t, http.MethodPost, ts.URL+"/api/about/test-llm?token=secret", `{"spec":"good:model"}`)
	if code != http.StatusOK {
		t.Fatalf("test-llm ok: %d, want 200", code)
	}
	var res testLLMResult
	if err := json.Unmarshal(body, &res); err != nil {
		t.Fatal(err)
	}
	if !res.OK || res.ModelID != "good/model" || res.LatencyMs != 42 {
		t.Errorf("test-llm success = %+v", res)
	}

	// Failure path: the test RAN (200) but ok=false carries the diagnostic.
	_, body = doReq(t, http.MethodPost, ts.URL+"/api/about/test-llm?token=secret", `{"spec":"bad:model"}`)
	_ = json.Unmarshal(body, &res)
	if res.OK || res.Error == "" {
		t.Errorf("test-llm failure should be ok=false with error: %+v", res)
	}
}

func TestServer_SuggestFollowup(t *testing.T) {
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Unconfigured (no suggester hook) → 501.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs/r1/followup-suggest?token=secret", ""); code != http.StatusNotImplemented {
		t.Fatalf("followup-suggest unconfigured: %d, want 501", code)
	}

	// Wire a stub: echoes a draft for a known run, errors otherwise.
	srv.SetFollowupSuggester(func(_ context.Context, runID string) (string, error) {
		if runID == "r1" {
			return "Harden the parser against malformed input.", nil
		}
		return "", fmt.Errorf("no report available yet for this run")
	})

	// Auth required (real billable call) → 401 without token.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs/r1/followup-suggest", ""); code != http.StatusUnauthorized {
		t.Fatalf("followup-suggest unauth: %d, want 401", code)
	}

	// Success path: 200 with the drafted prompt.
	code, body := doReq(t, http.MethodPost, ts.URL+"/api/runs/r1/followup-suggest?token=secret", "")
	if code != http.StatusOK {
		t.Fatalf("followup-suggest ok: %d, want 200", code)
	}
	var res struct {
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(body, &res); err != nil {
		t.Fatal(err)
	}
	if res.Prompt != "Harden the parser against malformed input." {
		t.Errorf("followup-suggest prompt = %q", res.Prompt)
	}

	// Failure path: the suggester errored → 500 with the diagnostic.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs/r2/followup-suggest?token=secret", ""); code != http.StatusInternalServerError {
		t.Fatalf("followup-suggest error: %d, want 500", code)
	}
}

func TestServer_SSE(t *testing.T) {
	srv, bus, store := newTestServer(t)
	seedRun(t, store, db.SessionOngoing, 0)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/runs/"+testRun+"/stream?token=secret", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close() //nolint:errcheck
	sc := bufio.NewScanner(resp.Body)

	if ev := nextSSE(t, sc); ev.Kind != eventstream.KindRefetchAll {
		t.Fatalf("first event = %q, want refetch_all prime", ev.Kind)
	}
	bus.Publish(eventstream.RunEvent{Kind: eventstream.KindSessionBump, RunID: testRun, SessionID: "main-agent"})
	if ev := nextSSE(t, sc); ev.Kind != eventstream.KindSessionBump || ev.SessionID != "main-agent" {
		t.Fatalf("bump event = %+v", ev)
	}
}

func TestServer_StaticSPA(t *testing.T) {
	if _, built := clientFS(); !built {
		t.Skip("frontend not built (run: make frontend-build)")
	}
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Index is served unauthenticated (the page then reads the token from URL).
	code, body := doReq(t, http.MethodGet, ts.URL+"/", "")
	if code != http.StatusOK {
		t.Fatalf("GET /: %d, want 200", code)
	}
	html := strings.ToLower(string(body))
	if !strings.Contains(html, "<!doctype html") || !strings.Contains(html, "_app") {
		t.Fatalf("GET / did not serve the SPA shell (got %d bytes)", len(body))
	}
	// Unknown client route → SPA fallback (same shell, client routes it).
	code, body = doReq(t, http.MethodGet, ts.URL+"/runs/anything", "")
	if code != http.StatusOK || !strings.Contains(strings.ToLower(string(body)), "_app") {
		t.Fatalf("SPA fallback for /runs/anything failed: %d", code)
	}
}

func TestRecallEndpoints(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	emb := embed.Mock{Dim: 4096}
	vecs, err := emb.Embed(ctx, []string{"flaky build retry workaround"})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.InsertLesson(ctx, db.LessonRecord{
		LessonID: "L1", Title: "Retry flaky", Description: "flaky build retry workaround",
		Body: "rerun with flag", Embedding: vecs[0], EmbedderID: emb.ModelID(),
	}); err != nil {
		t.Fatal(err)
	}
	lessonIx := lessons.NewIndex(store, emb)
	if err := lessonIx.Build(ctx); err != nil {
		t.Fatal(err)
	}
	srv.SetRecall(nil, lessonIx) // skills omitted; exercise the lesson path

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// List all lessons.
	code, body := doReq(t, http.MethodGet, ts.URL+"/api/lessons?token=secret", "")
	if code != http.StatusOK {
		t.Fatalf("lessons: %d: %s", code, body)
	}
	var list []struct {
		ID    string `json:"id"`
		Title string `json:"title"`
	}
	if err := json.Unmarshal(body, &list); err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 || list[0].ID != "L1" {
		t.Fatalf("lessons list = %+v", list)
	}

	// Search ranks the lesson.
	code, body = doReq(t, http.MethodGet, ts.URL+"/api/recall?q=flaky+build+retry&token=secret", "")
	if code != http.StatusOK {
		t.Fatalf("recall search: %d: %s", code, body)
	}
	var res struct {
		Lessons []struct {
			Handle string `json:"handle"`
		} `json:"lessons"`
	}
	if err := json.Unmarshal(body, &res); err != nil {
		t.Fatal(err)
	}
	if len(res.Lessons) != 1 || res.Lessons[0].Handle != "lesson:L1" {
		t.Fatalf("recall lessons = %+v", res.Lessons)
	}

	// Load the full body.
	code, body = doReq(t, http.MethodGet, ts.URL+"/api/recall/item?handle=lesson:L1&token=secret", "")
	if code != http.StatusOK {
		t.Fatalf("recall item: %d: %s", code, body)
	}
	var it struct {
		Kind string `json:"kind"`
		Body string `json:"body"`
	}
	if err := json.Unmarshal(body, &it); err != nil {
		t.Fatal(err)
	}
	if it.Kind != "lesson" || !strings.Contains(it.Body, "rerun with flag") {
		t.Fatalf("recall item = %+v", it)
	}
}

func TestGetReportEndpoint(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	if err := store.CreateRun(ctx, db.RunRecord{RunID: "r1"}); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendObservation(ctx, db.ObservationRecord{
		ObsID: "run_report-1", RunID: "r1", Kind: "run_report",
		Data: map[string]any{"version": 1, "summary": "seeded report"},
	}); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/r1/report?token=secret", "")
	if code != http.StatusOK {
		t.Fatalf("get report: %d, want 200: %s", code, body)
	}
	var reports []critic.RunReport
	if err := json.Unmarshal(body, &reports); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(reports) != 1 || reports[0].Version != 1 || reports[0].Summary != "seeded report" {
		t.Fatalf("reports = %+v", reports)
	}
}

func TestGenerateReportEndpoint(t *testing.T) {
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// No generator configured → 501.
	if code, _ := doReq(t, http.MethodPost, ts.URL+"/api/runs/r1/report?token=secret", ""); code != http.StatusNotImplemented {
		t.Fatalf("unconfigured: %d, want 501", code)
	}

	// With a generator → 201 + the report JSON.
	srv.SetReportGenerator(func(_ context.Context, runID string) (*critic.RunReport, bool, error) {
		return &critic.RunReport{Version: 1, Summary: "stub for " + runID}, false, nil
	})
	code, body := doReq(t, http.MethodPost, ts.URL+"/api/runs/r1/report?token=secret", "")
	if code != http.StatusCreated {
		t.Fatalf("configured: %d, want 201: %s", code, body)
	}
	var rep critic.RunReport
	if err := json.Unmarshal(body, &rep); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if rep.Version != 1 || rep.Summary != "stub for r1" {
		t.Fatalf("report = %+v", rep)
	}

	// deferred=true → 200 (OK) with the (previous) report body, so the UI can
	// distinguish "generated a new iteration" (201) from "kept the existing
	// one because the delta was below the debounce threshold" (200) without
	// inspecting the body.
	srv.SetReportGenerator(func(_ context.Context, runID string) (*critic.RunReport, bool, error) {
		return &critic.RunReport{Version: 1, Summary: "unchanged for " + runID}, true, nil
	})
	code, body = doReq(t, http.MethodPost, ts.URL+"/api/runs/r1/report?token=secret", "")
	if code != http.StatusOK {
		t.Fatalf("deferred: %d, want 200: %s", code, body)
	}
	if err := json.Unmarshal(body, &rep); err != nil {
		t.Fatalf("unmarshal deferred report: %v", err)
	}
	if rep.Version != 1 || rep.Summary != "unchanged for r1" {
		t.Fatalf("deferred report body = %+v", rep)
	}
}

// The run-detail DTO carries report_coverage + report_gap_steps so the UI can
// distinguish "a new iteration is coming soon" (substantive_gap) from "the
// framework saw the small delta and deliberately declined to regenerate"
// (trivial_gap) — without the second, a silent finalizer skip would leave the
// frontend showing an eternal "Generating…" spinner.
func TestGetRun_ReportCoverage(t *testing.T) {
	ctx := context.Background()

	// Helper: seed the given autonomous run at `currentStep` and, when >0, a
	// report observation whose main-agent watermark is `reportStep`. Uses the
	// same JSON shape critic.writeReport / dataToReport round-trip through.
	seed := func(t *testing.T, store db.Store, currentStep, reportStep int) {
		t.Helper()
		if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
			t.Fatal(err)
		}
		if err := store.CreateSession(ctx, db.SessionRecord{
			RunID: testRun, SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionConcluded,
		}); err != nil {
			t.Fatal(err)
		}
		for range currentStep {
			if _, err := store.AdvanceStep(ctx, testRun, "main-agent"); err != nil {
				t.Fatal(err)
			}
		}
		if reportStep <= 0 {
			return
		}
		rep := &critic.RunReport{
			Version: 1,
			Sessions: []critic.SessionState{{
				SessionID: "main-agent", AgentType: "standard_agent", Status: db.SessionConcluded, CurrentStep: reportStep,
			}},
		}
		blob, _ := json.Marshal(rep)
		var data map[string]any
		_ = json.Unmarshal(blob, &data)
		if err := store.AppendObservation(ctx, db.ObservationRecord{
			ObsID: "run_report-1", RunID: testRun, Kind: "run_report", Data: data,
		}); err != nil {
			t.Fatal(err)
		}
	}

	type want struct {
		coverage string
		gap      int
	}
	cases := []struct {
		name                    string
		currentStep, reportStep int
		want                    want
	}{
		{"no_report_yet", 5, 0, want{"", 0}},
		{"covered", 10, 10, want{"covered", 0}},
		{"trivial_gap", 12, 10, want{"trivial_gap", 2}},
		{"just_below_threshold", 10 + critic.ReportSkipMinSteps - 1, 10, want{"trivial_gap", critic.ReportSkipMinSteps - 1}},
		{"at_threshold_is_substantive", 10 + critic.ReportSkipMinSteps, 10, want{"substantive_gap", critic.ReportSkipMinSteps}},
		{"large_gap", 200, 10, want{"substantive_gap", 190}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _, store := newTestServer(t)
			seed(t, store, tc.currentStep, tc.reportStep)
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			code, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"?token=secret", "")
			if code != http.StatusOK {
				t.Fatalf("status=%d: %s", code, body)
			}
			var d runDetail
			if err := json.Unmarshal(body, &d); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if d.ReportCoverage != tc.want.coverage || d.ReportGapSteps != tc.want.gap {
				t.Fatalf("coverage=%q gap=%d, want coverage=%q gap=%d",
					d.ReportCoverage, d.ReportGapSteps, tc.want.coverage, tc.want.gap)
			}
		})
	}

	// Chat run (no main-agent session): coverage is empty — UI treats it as
	// not-applicable, and Generate stays always-available.
	t.Run("chat_run_no_main_agent", func(t *testing.T) {
		srv, _, store := newTestServer(t)
		if err := store.CreateRun(ctx, db.RunRecord{RunID: testRun}); err != nil {
			t.Fatal(err)
		}
		if err := store.CreateSession(ctx, db.SessionRecord{
			RunID: testRun, SessionID: "chatty-bot", AgentType: "chatbot", Status: db.SessionIdle,
		}); err != nil {
			t.Fatal(err)
		}
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()
		_, body := doReq(t, http.MethodGet, ts.URL+"/api/runs/"+testRun+"?token=secret", "")
		var d runDetail
		if err := json.Unmarshal(body, &d); err != nil {
			t.Fatal(err)
		}
		if d.ReportCoverage != "" || d.ReportGapSteps != 0 {
			t.Fatalf("chat: coverage=%q gap=%d, want empty", d.ReportCoverage, d.ReportGapSteps)
		}
	})
}

func nextSSE(t *testing.T, sc *bufio.Scanner) eventstream.RunEvent {
	t.Helper()
	for sc.Scan() {
		if data, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
			var ev eventstream.RunEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				t.Fatal(err)
			}
			return ev
		}
	}
	t.Fatal("stream ended before a data line")
	return eventstream.RunEvent{}
}

// TestPatchRun_SeenBothDirections: the dashboard badge is server-authoritative,
// so "mark as unread" is a PATCH like any other — and it has to actually restore
// the badge, which is the half a "don't clear it" implementation would miss.
func TestPatchRun_SeenBothDirections(t *testing.T) {
	srv, _, store := newTestServer(t)
	ctx := context.Background()
	seedRun(t, store, db.SessionOngoing, 3)
	// The badge is "a root's status changed AFTER last_seen_at", and a fresh run
	// is stamped seen at creation — so the transition has to happen after that,
	// as it does in life.
	time.Sleep(5 * time.Millisecond) // timestamps are millisecond-resolution
	if err := store.UpdateSessionStatus(ctx, testRun, "main-agent", db.SessionConcluded); err != nil {
		t.Fatal(err)
	}

	hasUpdates := func() bool {
		runs, _, err := store.ListRunsWithSessions(ctx, db.ListRunsOpts{})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range runs {
			if r.Run.RunID == testRun {
				return runHasUpdates(r)
			}
		}
		t.Fatalf("run %s not found", testRun)
		return false
	}

	patch := func(body string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPatch, "/api/runs/"+testRun, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer secret")
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("PATCH %s = %d (%s)", body, w.Code, w.Body.String())
		}
	}

	if !hasUpdates() {
		t.Fatal("a concluded run should start with a badge")
	}
	patch(`{"seen":true}`)
	if hasUpdates() {
		t.Error("seen:true should clear the badge")
	}
	patch(`{"seen":false}`)
	if !hasUpdates() {
		t.Error("seen:false should put the badge back")
	}
}

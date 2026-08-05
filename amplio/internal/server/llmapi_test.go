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
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"amplio/internal/llm"
	"amplio/internal/llm/bridge"
)

const lendToken = "lend-secret"

func lending(t *testing.T, srv *Server, embedder bridge.Embedder) (http.Handler, *[]string) {
	t.Helper()
	var built []string
	h := srv.LendingHandler(lendToken, func(spec string) (llm.Provider, error) {
		built = append(built, spec)
		return &llm.MockProvider{Model: spec, Responses: []llm.Response{{Content: "from the lender"}}}, nil
	}, embedder)
	return h, &built
}

func post(t *testing.T, h http.Handler, path, token, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

// TestLending_SurfaceIsOnlyLLM is the property the separate listener exists for:
// every GET on the MAIN server is unauthenticated (the read-only share view), so
// the port forwarded to a container must expose generations and nothing else.
func TestLending_SurfaceIsOnlyLLM(t *testing.T) {
	srv, _, _ := newTestServer(t)
	h, _ := lending(t, srv, nil)

	for _, path := range []string{
		"/api/runs", "/api/runs/x/sessions/main-agent/chat", "/api/runs/x/artifacts/raw",
		"/api/recall", "/api/about", "/api/stream", "/",
	} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		req.Header.Set("Authorization", "Bearer "+lendToken)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != http.StatusNotFound {
			t.Errorf("GET %s on the lending listener = %d, want 404", path, w.Code)
		}
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/health", nil))
	if w.Code != http.StatusOK {
		t.Errorf("GET /health = %d, want 200", w.Code)
	}
}

func TestLending_RequiresItsOwnToken(t *testing.T) {
	srv, _, _ := newTestServer(t)
	h, _ := lending(t, srv, nil)
	// "secret" is the SERVER token in newTestServer: it must not open this door,
	// or the container would need a credential that also starts runs.
	for _, tok := range []string{"", "secret", "wrong"} {
		if got := post(t, h, "/generate", tok, `{"model":"vertex:x"}`).Code; got != http.StatusUnauthorized {
			t.Errorf("token %q = %d, want 401", tok, got)
		}
	}
	if got := post(t, h, "/generate", lendToken, `{"model":"vertex:x"}`).Code; got != http.StatusOK {
		t.Errorf("lending token = %d, want 200", got)
	}
}

// TestLending_MenuIsTheAllowlist: a caller may ask for what the model picker
// shows — config plus models added through the new-run form — and nothing else.
func TestLending_MenuIsTheAllowlist(t *testing.T) {
	srv, _, store := newTestServer(t)
	if err := store.AddCustomModel(context.Background(), "openai{base_url=http://localhost:4000/v1}:claude#proxy"); err != nil {
		t.Fatal(err)
	}
	h, built := lending(t, srv, nil)

	tests := []struct {
		name       string
		handle     string
		wantStatus int
		wantSpec   string
		wantErr    string
	}{
		{name: "a config model", handle: "vertex:x", wantStatus: 200, wantSpec: "vertex:x"},
		{name: "a DB model by nickname", handle: "proxy", wantStatus: 200,
			wantSpec: "openai{base_url=http://localhost:4000/v1}:claude"},
		{name: "a model nobody offered", handle: "vertex:secret-model",
			wantStatus: http.StatusForbidden, wantErr: "not in this server's menu"},
		{
			// The dangerous one: a caller-supplied bin= would be arbitrary
			// execution on the machine holding the credentials.
			name: "a caller-supplied client block", handle: "subprocess{bin=/bin/sh}:m",
			wantStatus: http.StatusForbidden, wantErr: "client args are not accepted",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			*built = nil
			w := post(t, h, "/generate", lendToken, `{"model":"`+tt.handle+`"}`)
			if w.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d (body %q)", w.Code, tt.wantStatus, w.Body.String())
			}
			if tt.wantErr != "" {
				if !strings.Contains(w.Body.String(), tt.wantErr) {
					t.Errorf("body = %q, want %q", w.Body.String(), tt.wantErr)
				}
				if len(*built) != 0 {
					t.Errorf("a refused handle still built a provider: %v", *built)
				}
				return
			}
			if len(*built) != 1 || (*built)[0] != tt.wantSpec {
				t.Errorf("built %v, want [%q]", *built, tt.wantSpec)
			}
		})
	}
}

func TestLending_ListsModels(t *testing.T) {
	srv, _, _ := newTestServer(t)
	h, _ := lending(t, srv, nil)

	req := httptest.NewRequest(http.MethodGet, "/models", nil)
	req.Header.Set("Authorization", "Bearer "+lendToken)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /models = %d", w.Code)
	}
	var menu struct {
		Models []struct{ Spec string }
	}
	if err := json.Unmarshal(w.Body.Bytes(), &menu); err != nil {
		t.Fatalf("decode: %v (%s)", err, w.Body)
	}
	if len(menu.Models) != 2 {
		t.Errorf("models = %+v, want the two configured", menu.Models)
	}
	// It reveals what this machine will spend money on.
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/models", nil))
	if w.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated GET /models = %d, want 401", w.Code)
	}
}

func TestLending_Embed(t *testing.T) {
	srv, _, _ := newTestServer(t)

	// No embedder configured → the route is absent, rather than returning
	// vectors from nowhere.
	h, _ := lending(t, srv, nil)
	if got := post(t, h, "/embed", lendToken, `{"texts":["a"]}`).Code; got != http.StatusNotFound {
		t.Errorf("status = %d, want 404 with no embedder", got)
	}

	h, _ = lending(t, srv, stubEmbedder{})
	w := post(t, h, "/embed", lendToken, `{"texts":["a","b"]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d (%s)", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"vectors"`) {
		t.Errorf("body = %q", w.Body.String())
	}
}

// TestLending_EndToEnd drives a real bridge provider against the lending
// listener: the unit tests either side can both pass while the halves disagree
// about a path or a header.
func TestLending_EndToEnd(t *testing.T) {
	srv, _, _ := newTestServer(t)
	h, _ := lending(t, srv, nil)
	ts := httptest.NewServer(h)
	defer ts.Close()

	t.Setenv(bridge.DefaultTokenEnv, lendToken)
	p, err := bridge.NewBridge("vertex:x", 4096, url.Values{"url": {ts.URL}}, nil)
	if err != nil {
		t.Fatalf("NewBridge: %v", err)
	}
	resp, err := p.Call(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "from the lender" {
		t.Errorf("content = %q", resp.Content)
	}

	t.Setenv(bridge.DefaultTokenEnv, "wrong")
	bad, err := bridge.NewBridge("vertex:x", 4096, url.Values{"url": {ts.URL}}, nil)
	if err != nil {
		t.Fatalf("NewBridge: %v", err)
	}
	if _, err := bad.Call(context.Background(), llm.Request{}); err == nil ||
		!strings.Contains(err.Error(), "401") {
		t.Errorf("error with a bad token = %v, want a 401", err)
	}
}

type stubEmbedder struct{}

func (stubEmbedder) ModelID() string { return "stub" }
func (stubEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 2, 3}
	}
	return out, nil
}

// TestLending_AnnouncesEachModelOnce: the operator should learn that something
// started spending their credentials, and on what — but not once per request.
// A line per generation scales with the caller's traffic and buries the
// server's own output; a local run logs nothing per call at all.
func TestLending_AnnouncesEachModelOnce(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
	defer slog.SetDefault(prev)

	srv, _, _ := newTestServer(t)
	h, _ := lending(t, srv, nil)
	for range 3 {
		if got := post(t, h, "/generate", lendToken, `{"model":"vertex:x"}`).Code; got != http.StatusOK {
			t.Fatalf("status = %d", got)
		}
	}
	if n := strings.Count(buf.String(), "lending: first generation"); n != 1 {
		t.Errorf("announced %d times across 3 requests, want 1:\n%s", n, buf.String())
	}
	// A second model announces itself too — the point is per-model, not once ever.
	if got := post(t, h, "/generate", lendToken, `{"model":"vertex:y"}`).Code; got != http.StatusOK {
		t.Fatalf("status = %d", got)
	}
	if n := strings.Count(buf.String(), "lending: first generation"); n != 2 {
		t.Errorf("announced %d times for two models, want 2", n)
	}
}

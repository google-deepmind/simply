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
	"sync"
	"time"

	"amplio/internal/agent/critic"
	"amplio/internal/db"
	"amplio/internal/eventstream"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	"amplio/internal/runtime"
	"amplio/internal/skills"
	"amplio/internal/sysstat"
)

// RunDefaults are applied to runs started via the API (the server has one
// configured LLM tier set; the request only supplies task/workspace/agent).
type RunDefaults struct {
	LLM           string   // default agent model (first of LLMs)
	LLMs          []string // configured model menu (read-only; from config.toml)
	SystemLLMHQ   string
	SystemLLMFast string
	AgentType     string
}

// Server exposes the run store + manager over HTTP: authenticated JSON reads,
// write actions, and an SSE liveness stream, plus the embedded SPA.
type Server struct {
	store        db.Store
	mgr          *runtime.RunManager
	bus          *eventstream.Bus
	token        string
	owner        string // username running the server; shown in the readonly banner
	cookieName   string // per-instance auth cookie name (empty → default authCookie)
	secureCookie bool   // set the Secure flag on the auth cookie (HTTPS-only)
	defaults     RunDefaults
	runCtx       context.Context // lifetime of started runs (NOT a request ctx)

	// genReport produces a run report on operator demand (nil = unconfigured →
	// the endpoint returns 501). Wired by the entrypoint to the Finalizer. The
	// deferred return signals "delta below threshold, previous report unchanged"
	// so the handler can respond 200 (deferred) vs 201 (created).
	genReport func(ctx context.Context, runID string) (report *critic.RunReport, deferred bool, err error)

	// lentSeen remembers which specs have already been logged at INFO, so lending
	// announces a model once instead of once per request.
	lentSeen sync.Map

	// lendProvider builds a provider for the lending listener; nil unless
	// LendingHandler was called (see llmapi.go).
	lendProvider func(spec string) (llm.Provider, error)

	// testLLM validates an agent-LLM spec (build the provider + one trivial Call),
	// for the About page's pre-flight tester. nil = unconfigured (endpoint 501).
	// Wired by the entrypoint since provider construction lives in cmd/amplio.
	// Returns the resolved model id, the probe reply, the call latency, or an error.
	testLLM func(ctx context.Context, spec string) (modelID, reply string, latency time.Duration, err error)

	// suggestFollowup asks the system HQ model to draft a follow-up instruction
	// for a concluded autonomous run, from its original task + latest report. nil
	// = unconfigured (endpoint 501). Wired by the entrypoint (provider lives in
	// cmd/amplio). Makes a real billable LLM call — behind requireAuth.
	suggestFollowup func(ctx context.Context, runID string) (string, error)

	// Recall corpus indexes, for the /recall browse page (nil = unavailable).
	skillIndex  *skills.Index
	lessonIndex *lessons.Index

	// sysstat (nil → handleSysStat returns the zero snapshot) backs both the
	// GET /api/sysstat seed and (via the broadcaster) the kind=sysstat SSE
	// updates. Wired by cmd/amplio/serve.go.
	sysstat *sysstat.Watcher
}

// SetSysStat installs the server-host status watcher behind GET /api/sysstat.
func (s *Server) SetSysStat(w *sysstat.Watcher) { s.sysstat = w }

// SetOwner records the username running the server, surfaced by GET /api/auth
// for the readonly banner ("Readonly view of <owner>'s runs").
func (s *Server) SetOwner(user string) { s.owner = user }

// SetCookieName overrides the auth cookie name. Cookies ignore the port, so
// multiple servers on the same host would otherwise share one cookie slot and
// clobber each other's session — give each a per-instance name (e.g. derived
// from its data dir).
func (s *Server) SetCookieName(name string) { s.cookieName = name }

// SetSecureCookie controls the Secure flag on the auth cookie. Must be true
// when serving over HTTPS (browsers refuse to send Secure cookies over plain
// HTTP, and modern Chrome will start blocking non-Secure cookies on HTTPS
// connections too). Default false (serving plain HTTP).
func (s *Server) SetSecureCookie(secure bool) { s.secureCookie = secure }

// SetReportGenerator installs the operator-triggered report generator. Set once
// at startup.
func (s *Server) SetReportGenerator(fn func(ctx context.Context, runID string) (*critic.RunReport, bool, error)) {
	s.genReport = fn
}

// SetLLMTester installs the About-page LLM pre-flight tester. Set once at
// startup (provider construction lives in cmd/amplio, hence the injection).
func (s *Server) SetLLMTester(fn func(ctx context.Context, spec string) (modelID, reply string, latency time.Duration, err error)) {
	s.testLLM = fn
}

// SetFollowupSuggester installs the report follow-up drafter (system HQ model,
// from the run's task + latest report). Set once at startup; provider lives in
// cmd/amplio.
func (s *Server) SetFollowupSuggester(fn func(ctx context.Context, runID string) (string, error)) {
	s.suggestFollowup = fn
}

// SetRecall installs the skill + lesson indexes powering the /recall browse page
// (the same indexes agents search). Set once at startup.
func (s *Server) SetRecall(skillIx *skills.Index, lessonIx *lessons.Index) {
	s.skillIndex = skillIx
	s.lessonIndex = lessonIx
}

func New(runCtx context.Context, store db.Store, mgr *runtime.RunManager, bus *eventstream.Bus, token string, defaults RunDefaults) *Server {
	if defaults.AgentType == "" {
		defaults.AgentType = "standard_agent"
	}
	return &Server{store: store, mgr: mgr, bus: bus, token: token, defaults: defaults, runCtx: runCtx}
}

// Handler wires the routes. Reads (GET) and the SPA are open — that's the
// readonly share view anyone on the corp network gets. Mutations require the
// token (cookie/Bearer) via requireAuth. The two /api/auth endpoints are open:
// login exchanges a token for the cookie; the status drives the readonly UI.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Auth.
	mux.HandleFunc("GET /api/auth", s.handleAuth)
	mux.HandleFunc("POST /api/auth/login", s.handleAuthLogin)

	// Reads — open (readonly view).
	mux.HandleFunc("GET /api/runs", s.handleListRuns)
	// Registered before the {id} route is irrelevant for net/http's mux (it
	// matches by specificity), but keep counts adjacent to the list it summarizes.
	mux.HandleFunc("GET /api/runs/counts", s.handleRunCounts)
	mux.HandleFunc("GET /api/runs/{id}", s.handleGetRun)
	mux.HandleFunc("GET /api/sysstat", s.handleSysStat)
	mux.HandleFunc("GET /api/models", s.handleListModels)
	mux.HandleFunc("GET /api/runs/{id}/sessions/{sid}/chat", s.handleChat)
	mux.HandleFunc("GET /api/runs/{id}/sessions/{sid}/trajectory", s.handleTrajectory)
	mux.HandleFunc("GET /api/runs/{id}/sessions/{sid}/events", s.handleEvents)
	mux.HandleFunc("GET /api/runs/{id}/sessions/{sid}/observations", s.handleObservations)
	mux.HandleFunc("GET /api/runs/{id}/blobs/{key}", s.handleBlob)
	mux.HandleFunc("GET /api/runs/{id}/artifacts", s.handleArtifacts)
	mux.HandleFunc("GET /api/runs/{id}/artifacts/all", s.handleArtifactsAll)
	mux.HandleFunc("GET /api/runs/{id}/artifacts/raw", s.handleArtifactRaw)
	mux.HandleFunc("GET /api/workspaces", s.handleWorkspaces)
	mux.HandleFunc("GET /api/recall", s.handleRecallSearch)
	mux.HandleFunc("GET /api/recall/item", s.handleRecallItem)
	mux.HandleFunc("GET /api/lessons", s.handleListLessons)
	mux.HandleFunc("GET /api/about", s.handleAbout)
	mux.HandleFunc("GET /api/runs/{id}/report", s.handleGetReport)
	mux.HandleFunc("GET /api/runs/{id}/stream", s.handleRunStream)
	mux.HandleFunc("GET /api/stream", s.handleGlobalStream)

	// Writes — require the token.
	mux.HandleFunc("POST /api/runs", s.requireAuth(s.handleStartRun))
	mux.HandleFunc("PATCH /api/runs/{id}", s.requireAuth(s.handleUpdateRun))
	mux.HandleFunc("POST /api/models", s.requireAuth(s.handleAddModel))
	mux.HandleFunc("DELETE /api/models", s.requireAuth(s.handleRemoveModel))
	mux.HandleFunc("POST /api/runs/{id}/sessions/{sid}/message", s.requireAuth(s.handleSendMessage))
	mux.HandleFunc("POST /api/runs/{id}/sessions/{sid}/notify", s.requireAuth(s.handleNotify))
	mux.HandleFunc("POST /api/runs/{id}/chatbot", s.requireAuth(s.handleStartChatbot))
	mux.HandleFunc("POST /api/runs/{id}/report", s.requireAuth(s.handleGenerateReport))
	mux.HandleFunc("POST /api/runs/{id}/followup-suggest", s.requireAuth(s.handleSuggestFollowup))
	mux.HandleFunc("POST /api/runs/{id}/cancel", s.requireAuth(s.handleCancelRun))
	mux.HandleFunc("POST /api/runs/{id}/restart", s.requireAuth(s.handleRestartRun))
	mux.HandleFunc("DELETE /api/runs/{id}", s.requireAuth(s.handleDeleteRun))
	// LLM pre-flight tester: makes a real (billable) call, so write-gated.
	mux.HandleFunc("POST /api/about/test-llm", s.requireAuth(s.handleTestLLM))
	mux.HandleFunc("POST /api/workspaces/new", s.requireAuth(s.handleCreateWorkspace))

	// Routes that depend on optional backends (e.g. workspace alias
	// management). Stub in the OSS build; tagged file installs the real
	// handlers — see routes_extras{,_internal}.go.
	s.registerInternalRoutes(mux)

	mux.Handle("/", s.staticHandler())
	return mux
}

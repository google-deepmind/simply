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
	"crypto/subtle"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"amplio/internal/llm"
	"amplio/internal/llm/bridge"
)

// Lending: this server running generations for an authenticated caller, on its
// own credentials. The motivating case is an amplio in a container with no
// credentials and one outbound channel, reaching a workstation that has them.
//
// It lives on a SEPARATE LISTENER (config lend_llm), and that is the important
// part. The lending listener serves generations, embeddings and the model list.
//
// The protocol lives in internal/llm/bridge, and these are the paths any bridge
// serves (bridges/README.md), so a caller need not know whether it is talking to
// amplio or to a 200-line Python script.

// LendingHandler builds the lending listener's mux. token is its own bearer
// secret, separate from the server token — which also authorises starting runs,
// i.e. shell access on this machine, and therefore should never be given to a
// container.
func (s *Server) LendingHandler(token string, newProvider func(spec string) (llm.Provider, error), embedder bridge.Embedder) http.Handler {
	s.lendProvider = newProvider

	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", lendAuth(token, bridge.GenerateHandler(s.resolveLentModel)))
	if embedder != nil {
		// Absent an embedder the route is absent too: recall needs one, and a
		// caller is better told 404 than handed vectors from nowhere.
		mux.HandleFunc("POST /embed", lendAuth(token, bridge.EmbedHandler(embedder)))
	}
	// What this server will run. Same handler the UI uses, so the list and the
	// allowlist cannot disagree.
	mux.HandleFunc("GET /models", lendAuth(token, s.handleListModels))
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	return mux
}

// lendAuth gates a lending route on the shared secret. Constant-time compare for
// the same reason the main server's check uses one.
func lendAuth(token string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
			http.Error(w, "invalid or missing token", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

// resolveLentModel turns a caller's handle into a provider, or refuses.
//
// The menu is the union the model picker already shows — config.toml's
// [run].llms plus the models added through the new-run form — so "what this
// server offers" and "what it will run for you" cannot drift apart. See llm.Menu
// for the resolution rule and why the caller may not supply a client block.
func (s *Server) resolveLentModel(ctx context.Context, handle string) (llm.Provider, error) {
	custom, err := s.store.ListCustomModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("read the model menu: %w", err)
	}
	menu := llm.Menu{Specs: append(append([]string{}, s.defaults.LLMs...), custom...)}
	spec, err := menu.Resolve(handle)
	if err != nil {
		return nil, err
	}
	provider, err := s.lendProvider(spec)
	if err != nil {
		return nil, fmt.Errorf("build %s: %w", spec, err)
	}
	// INFO once per model, DEBUG for the rest. The operator should learn that
	// something started spending their credentials, and on what — but a line per
	// generation scales with the caller's traffic and buries the server's own
	// output, which is the argument logPipe already makes for bridge chatter. A
	// local run logs nothing at all per call.
	if _, seen := s.lentSeen.LoadOrStore(spec, struct{}{}); !seen {
		slog.Info("lending: first generation for this model", "handle", handle, "spec", spec)
	} else {
		slog.Debug("lending: serving generation", "handle", handle, "spec", spec)
	}
	return provider, nil
}

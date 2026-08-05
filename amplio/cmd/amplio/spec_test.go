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
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"amplio/internal/config"
	"amplio/internal/llm"
)

// TestCreateProvider_ClientBlockReachesProvider proves the `{k=v}` block is
// wired all the way through: the only way the request can arrive at the test
// server is if base_url made it from the block into the provider.
func TestCreateProvider_ClientBlockReachesProvider(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": "ok"}}},
		})
	}))
	defer srv.Close()

	for _, spec := range []string{
		"openai{base_url=" + srv.URL + "}:test-model",
		"openai{base_url=" + srv.URL + "}:test-model#n", // with a nickname
	} {
		t.Run(spec, func(t *testing.T) {
			before := hits
			p, err := createProvider(spec)
			if err != nil {
				t.Fatalf("createProvider(%q) = %v", spec, err)
			}
			if got := p.ModelID(); got != "test-model" {
				t.Errorf("ModelID() = %q, want %q", got, "test-model")
			}
			resp, err := p.Call(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("Call: %v", err)
			}
			if resp.Content != "ok" {
				t.Errorf("content = %q, want %q", resp.Content, "ok")
			}
			if hits != before+1 {
				t.Errorf("server saw %d requests, want 1 — base_url did not reach the provider", hits-before)
			}
		})
	}
}

// TestCreateProvider_QueryGoesToTheModel: the query is the model's namespace,
// including a key that shares a name with a client arg. Nothing inspects it, so
// nothing can steal it — that is what makes a nested spec unambiguous.
func TestCreateProvider_QueryGoesToTheModel(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": "ok"}}},
		})
	}))
	defer srv.Close()

	p, err := createProvider("openai{base_url=" + srv.URL + "}:m?profile=server-side&top_p=0.9")
	if err != nil {
		t.Fatalf("createProvider: %v", err)
	}
	if _, err := p.Call(context.Background(), llm.Request{}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	// profile is one of OUR client args by name, but written in the query it is
	// the server's: forwarded verbatim, not claimed.
	if body["profile"] != "server-side" {
		t.Errorf("body profile = %#v, want it forwarded", body["profile"])
	}
	if body["top_p"] != 0.9 {
		t.Errorf("body top_p = %#v", body["top_p"])
	}
	// …and the block's base_url still configured the client, or the request would
	// not have arrived here at all.
}

func TestCreateProvider_Errors(t *testing.T) {
	tests := []struct {
		name string
		spec string
		want string
	}{
		{"unknown provider", "nope:model", "unknown LLM provider"},
		{"no colon", "gpt-5", "missing"},
		{"empty model", "openai:?a=b", "empty model"},
		{"unterminated block", "openai{base_url=x:gpt-5", "unterminated"},
		{"block on an unknown provider still reports the provider", "nope{a=b}:m", "unknown LLM provider"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := createProvider(tt.spec)
			if err == nil {
				t.Fatalf("createProvider(%q) = nil error, want one", tt.spec)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want it to contain %q", err, tt.want)
			}
			if !strings.Contains(err.Error(), strings.TrimSpace(tt.spec)) {
				t.Errorf("error = %q, want it to quote the spec %q", err, tt.spec)
			}
		})
	}
}

// TestCreateEmbedder_SpecShapes covers the shapes that don't need credentials:
// the bare-name back-compat path is exercised for its parse only (constructing a
// vertex embedder needs ADC), so we assert on the error it produces rather than
// on success.
func TestCreateEmbedder_SpecShapes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	spec := "openai{base_url=" + srv.URL + "}:nomic-embed-text:latest"
	e, err := createEmbedder(context.Background(), spec)
	if err != nil {
		t.Fatalf("createEmbedder(%q) = %v", spec, err)
	}
	if got := e.ModelID(); !strings.HasPrefix(got, "openai_nomic-embed-text:latest@") {
		t.Errorf("ModelID() = %q, want openai_nomic-embed-text:latest@…", got)
	}
	// An undeclared key in the block is still refused, naming what is accepted.
	if _, err := createEmbedder(context.Background(), "openai{bse_url=http://x}:m"); err == nil ||
		!strings.Contains(err.Error(), "base_url") {
		t.Errorf("typo in the block: err = %v, want it to name the accepted args", err)
	}

	if _, err := createEmbedder(context.Background(), "nope:model"); err == nil ||
		!strings.Contains(err.Error(), "unknown embed backend") {
		t.Errorf("unknown backend error = %v, want 'unknown embed backend'", err)
	}
}

// TestCreateProvider_ClientArgValidation pins the payoff of declaring client
// args: a typo in the half amplio owns is caught at construction, while the
// model's half stays deliberately unvalidated (the server is the authority).
func TestCreateProvider_ClientArgValidation(t *testing.T) {
	if _, err := createProvider("openai{bse_url=http://x}:m"); err == nil {
		t.Error("a misspelled client arg should fail fast")
	} else if !strings.Contains(err.Error(), "base_url") {
		t.Errorf("error %q should name the accepted args", err)
	}
	// Same key in the query is a MODEL arg as far as we are concerned: not ours
	// to validate, so it is forwarded and the server decides.
	if _, err := createProvider("openai:m?bse_url=http://x"); err != nil {
		t.Errorf("an unknown model arg must pass through, got %v", err)
	}
	// Gemini declares no client args at all.
	if _, err := createProvider("vertex-gemini{base_url=http://x}:m"); err == nil {
		t.Error("a client arg on a provider that declares none should fail")
	}
	// max_tokens is universal.
	if _, err := createProvider("openai{max_tokens=128}:m"); err != nil {
		t.Errorf("max_tokens should be accepted by every provider, got %v", err)
	}
	if _, err := createProvider("openai{max_tokens=nope}:m"); err == nil {
		t.Error("a non-numeric max_tokens should fail fast")
	}
}

// TestBridgeEndpointTable: a spec can name a link instead of repeating where it
// is and which variable holds its token. Resolved before the provider is built,
// so internal/llm/bridge never sees the name — it is application configuration.
func TestBridgeEndpointTable(t *testing.T) {
	prev := bridgeEndpoints
	defer func() { bridgeEndpoints = prev }()
	bridgeEndpoints = map[string]config.BridgeEndpoint{
		"corp": {URL: "unix:///tmp/corp-bridge.sock", TokenEnv: "CORP_TOKEN", IdleTimeout: "10m"},
		"ws":   {URL: "https://workstation:26760"},
		"bad":  {TokenEnv: "X"}, // no url
	}

	t.Run("the name expands to the link", func(t *testing.T) {
		args := url.Values{"endpoint": {"corp"}}
		if err := expandBridgeEndpoint(args); err != nil {
			t.Fatalf("expand: %v", err)
		}
		for k, want := range map[string]string{
			"url": "unix:///tmp/corp-bridge.sock", "token_env": "CORP_TOKEN", "idle_timeout": "10m",
		} {
			if got := args.Get(k); got != want {
				t.Errorf("%s = %q, want %q", k, got, want)
			}
		}
		if args.Has("endpoint") {
			t.Error("endpoint= should be consumed, not forwarded")
		}
	})

	t.Run("the spec overrides the link's defaults", func(t *testing.T) {
		args := url.Values{"endpoint": {"corp"}, "idle_timeout": {"30s"}}
		if err := expandBridgeEndpoint(args); err != nil {
			t.Fatalf("expand: %v", err)
		}
		if got := args.Get("idle_timeout"); got != "30s" {
			t.Errorf("idle_timeout = %q, want the spec's value", got)
		}
	})

	for _, tc := range []struct {
		name, want string
		args       url.Values
	}{
		{"unknown name lists the known ones", "bad, corp, ws",
			url.Values{"endpoint": {"nope"}}},
		{"endpoint and url together", "mutually exclusive",
			url.Values{"endpoint": {"corp"}, "url": {"http://x"}}},
		{"a link with no url", "has no url",
			url.Values{"endpoint": {"bad"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := expandBridgeEndpoint(tc.args)
			if err == nil {
				t.Fatal("want an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to contain %q", err, tc.want)
			}
		})
	}

	t.Run("end to end through createProvider", func(t *testing.T) {
		t.Setenv("CORP_TOKEN", "s3cret")
		p, err := createProvider("bridge{endpoint=corp}:some-model")
		if err != nil {
			t.Fatalf("createProvider: %v", err)
		}
		if got := p.ModelID(); got != "some-model" {
			t.Errorf("ModelID = %q", got)
		}
		// A name nobody configured fails at construction, not at first use.
		if _, err := createProvider("bridge{endpoint=nope}:m"); err == nil ||
			!strings.Contains(err.Error(), "unknown bridge endpoint") {
			t.Errorf("err = %v, want an unknown-endpoint failure", err)
		}
	})
}

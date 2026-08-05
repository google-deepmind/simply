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
	"fmt"
	"net/url"
	"sort"
	"strings"

	"amplio/internal/config"
	"amplio/internal/embed"
	"amplio/internal/llm"
	anthropicprovider "amplio/internal/llm/anthropic"
	bridgeprovider "amplio/internal/llm/bridge"
	geminiprovider "amplio/internal/llm/gemini"
	openaiprovider "amplio/internal/llm/openai"
)

// Construction of LLM providers and embedders from spec strings, plus the
// named-bridge-endpoint table those specs may refer to. The spec grammar
// itself lives in internal/llm; this file is the application half: which
// provider classes exist, and what config.toml lets a spec name.

// defaultMaxOutputTokens caps generation for every provider. Generous because
// coding agents emit long replies/edits; thinking budgets are separate.
const defaultMaxOutputTokens = 65536

// providerFactory builds a provider from a model and the two halves of its spec
// args: clientArgs are the ones this provider interprets (the `{k=v}` block),
// args are passed through to the model untouched (the `?k=v` query). See
// internal/llm/spec.go for the rule that decides which is which.
type providerFactory func(model string, maxTokens int, clientArgs, args url.Values) (llm.Provider, error)

// providerEntry is a provider class: how to build it, and which client args it
// declares. The declaration is what lets an unknown key in the block be an
// error instead of a silent pass-through to the server — the client owns that
// namespace, so a typo in it is knowable.
type providerEntry struct {
	new        providerFactory
	clientArgs map[string]bool
}

// providerRegistry maps an LLM spec prefix (the provider "class") to its entry.
// A spec is "<prefix>[{k=v&…}]:<model>[?k=v&…]".
var providerRegistry = map[string]providerEntry{
	// Claude on Vertex AI (ADC) and the direct Anthropic API (ANTHROPIC_API_KEY).
	"vertex-claude": {anthropicprovider.NewVertex, anthropicprovider.ClientArgs},
	"claude":        {anthropicprovider.NewAPIKey, anthropicprovider.ClientArgs},
	// Gemini on Vertex AI (ADC) and the Developer API (API key). Its spec args
	// are a closed, typed set of model knobs; it has no client args of its own.
	"vertex-gemini": {geminiprovider.NewVertex, nil},
	"gemini":        {geminiprovider.NewAPIKey, nil},
	// Any OpenAI-compatible /v1/chat/completions server — the hosted API by
	// default, or whatever base_url= points at (vLLM, ollama, LiteLLM,
	// OpenRouter, a corp gateway). One provider, most of the ecosystem.
	"openai": {openaiprovider.New, openaiprovider.ClientArgs},
	// Bridges: any process speaking amplio's own NDJSON protocol. We spawn
	// and own the process for subprocess:, and dial existing server for bridge:.
	// See bridges/README.md.
	"subprocess": {bridgeprovider.NewSubprocess, bridgeprovider.ClientArgsSubprocess},
	"bridge":     {bridgeprovider.NewBridge, bridgeprovider.ClientArgsBridge},
}

// embedClientArgs are the client args an embed spec accepts. Only the openai
// backend has any; the genai-backed ones take the model and nothing else.
var embedClientArgs = map[string]bool{
	"base_url":    true,
	"api_key_env": true,
	"url":         true, // bridge: endpoint
	"token_env":   true, // bridge: which variable holds the bearer token
	"endpoint":    true, // bridge: a named link from config.toml
}

// bridgeEndpoints is the [bridge.<name>] table, stashed by resolveConfig so
// createProvider can expand endpoint=<name> without every command having to
// thread the config through. Read once at startup: see config.BridgeEndpoint.
var bridgeEndpoints map[string]config.BridgeEndpoint

// expandBridgeEndpoint turns endpoint=<name> into the url (and token_env,
// idle_timeout) that name stands for, so the rest of the system — including
// internal/llm/bridge — only ever deals in explicit endpoints.
//
// Keeping the lookup here rather than in the bridge package is deliberate: what
// the name "corp" means is application configuration, and a provider library
// that reads config.toml is one that cannot be used from anywhere else.
//
// The spec wins over the table for the keys both can set: the table supplies
// defaults for a link, not rules about it.
func expandBridgeEndpoint(clientArgs url.Values) error {
	name := clientArgs.Get("endpoint")
	if name == "" {
		return nil
	}
	if clientArgs.Get("url") != "" {
		return fmt.Errorf("endpoint=%s and url= are mutually exclusive; pick one", name)
	}
	ep, ok := bridgeEndpoints[name]
	if !ok {
		return fmt.Errorf("unknown bridge endpoint %q; config.toml defines %s", name, knownEndpoints())
	}
	if ep.URL == "" {
		return fmt.Errorf("bridge endpoint %q has no url in config.toml", name)
	}
	clientArgs.Del("endpoint")
	clientArgs.Set("url", ep.URL)
	for k, v := range map[string]string{"token_env": ep.TokenEnv, "idle_timeout": ep.IdleTimeout} {
		if v != "" && clientArgs.Get(k) == "" {
			clientArgs.Set(k, v)
		}
	}
	return nil
}

func knownEndpoints() string {
	if len(bridgeEndpoints) == 0 {
		return "no [bridge.<name>] sections"
	}
	names := make([]string, 0, len(bridgeEndpoints))
	for n := range bridgeEndpoints {
		names = append(names, n)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// createEmbedder builds an Embedder from a "<backend>:<model>[?k=v&…]" spec,
// mirroring createProvider. Backends: "vertex" (ADC, project-based), "gemini"
// (Gemini Developer API, key-based) and "openai" (any OpenAI-compatible
// /v1/embeddings endpoint — the hosted API, or ?base_url= for a local server,
// which is what lets a self-hosted deployment use recall at all). A bare model
// name (no ":") defaults to the vertex backend for back-compat with older
// embed_model config values. Note model availability is backend-specific (e.g.
// text-embedding-005 is Vertex-only; gemini-embedding-001 works on both).
func createEmbedder(ctx context.Context, spec string) (embed.Embedder, error) {
	// A bare model name (no backend) predates the spec grammar and still means
	// vertex, so it is resolved before parsing.
	if base := llm.BaseSpec(spec); !strings.Contains(base, ":") {
		if base == "" {
			return nil, fmt.Errorf("invalid embed model spec %q; want <backend>:<model>", spec)
		}
		return embed.NewVertex(ctx, base)
	}
	sp, err := llm.ParseSpec(spec)
	if err != nil {
		return nil, err
	}
	model, _, err := sp.Model() // an embed backend takes no model args today
	if err != nil {
		return nil, fmt.Errorf("invalid embed model spec %q: %w", spec, err)
	}
	clientArgs, err := llm.ClientArgs(sp.Client, embedClientArgs)
	if err != nil {
		return nil, fmt.Errorf("embed model spec %q: %w", spec, err)
	}
	if err := expandBridgeEndpoint(clientArgs); err != nil {
		return nil, fmt.Errorf("embed model spec %q: %w", spec, err)
	}
	backend := sp.Provider
	switch backend {
	case "vertex":
		return embed.NewVertex(ctx, model)
	case "gemini":
		return embed.NewAPIKey(ctx, model)
	case "openai":
		return embed.NewOpenAI(model, clientArgs.Get("base_url"), clientArgs.Get("api_key_env"))
	case "bridge":
		return bridgeprovider.NewEmbedder(model, clientArgs)
	default:
		return nil, fmt.Errorf("unknown embed backend %q in %q; known: bridge, gemini, openai, vertex", backend, spec)
	}
}

func createProvider(spec string) (llm.Provider, error) {
	// ParseSpec drops any "#nickname" display override: it is a harness-side
	// label (see internal/llm/label.go) and no provider ever sees it. Its errors
	// quote the ORIGINAL spec, so what the operator reads matches what they
	// configured.
	sp, err := llm.ParseSpec(spec)
	if err != nil {
		return nil, err
	}
	model, args, err := sp.Model()
	if err != nil {
		return nil, fmt.Errorf("invalid LLM spec %q: %w", spec, err)
	}
	entry, ok := providerRegistry[sp.Provider]
	if !ok {
		return nil, fmt.Errorf("unknown LLM provider %q in %q; known: %s", sp.Provider, spec, knownProviders())
	}
	clientArgs, err := llm.ClientArgs(sp.Client, entry.clientArgs)
	if err != nil {
		return nil, fmt.Errorf("LLM spec %q: %w", spec, err)
	}
	if err := expandBridgeEndpoint(clientArgs); err != nil {
		return nil, fmt.Errorf("LLM spec %q: %w", spec, err)
	}
	maxTokens, err := llm.MaxTokensArg(clientArgs, defaultMaxOutputTokens)
	if err != nil {
		return nil, fmt.Errorf("LLM spec %q: %w", spec, err)
	}
	return entry.new(model, maxTokens, clientArgs, args)
}

func knownProviders() string {
	keys := make([]string, 0, len(providerRegistry))
	for k := range providerRegistry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

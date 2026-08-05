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

// Package embed computes text embeddings used for skill and lesson recall.
package embed

import (
	"context"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
)

// Embedder turns text into vectors. Implementations handle batching internally,
// so callers may pass arbitrarily large slices.
type Embedder interface {
	// Embed returns one vector per input text, aligned 1:1. All vectors share
	// the embedder's native dimensionality.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// ModelID identifies the embedding space a stored vector belongs to, so that
	// switching models (or endpoints) naturally invalidates cached vectors. It is
	// persisted as a DB column value, never a path, so any characters a model
	// name legitimately contains (':' on ollama, '/' on gateway-hosted models)
	// are fine — it only has to be stable and distinct.
	ModelID() string
}

const (
	// Vertex text-embedding requests are capped by BOTH the instance count
	// (hard limit 250) and the total input tokens (~20000). We batch under both,
	// estimating tokens conservatively from characters (real English is ~4
	// chars/token; using ~3 leaves headroom against the 20000 cap).
	vertexMaxInstances = 250
	vertexTokenBudget  = 16000
	// vertexMaxCharsPerText clips any single over-long input so one skill can't
	// blow the per-request budget on its own.
	vertexMaxCharsPerText = 8000
)

// estTokens conservatively estimates the token count of s (overestimates, so
// batches stay safely under the real limit).
func estTokens(s string) int { return (len(s) + 2) / 3 }

// embedBatches splits texts into [start,end) ranges, each within the instance
// and token budgets. A single oversized text gets its own (already-clipped)
// batch.
func embedBatches(texts []string) [][2]int {
	var batches [][2]int
	for i := 0; i < len(texts); {
		j, tokens := i, 0
		for j < len(texts) {
			tk := estTokens(texts[j])
			if j > i && (j-i >= vertexMaxInstances || tokens+tk > vertexTokenBudget) {
				break
			}
			tokens += tk
			j++
		}
		batches = append(batches, [2]int{i, j})
		i = j
	}
	return batches
}

// genaiEmbedder is the shared google.golang.org/genai-backed embedder. Both the
// Vertex (ADC) and Gemini Developer API (key) constructors build the same struct
// — only the genai client's backend differs — so the batching/clipping/Embed
// logic is written once. backend tags the ModelID so vectors from different
// backends never collide in the cache.
type genaiEmbedder struct {
	client  *genai.Client
	model   string
	backend string
}

// NewVertex builds a Vertex-backed embedder for the given model (e.g.
// "text-embedding-005"), reusing VERTEXAI_PROJECT / VERTEXAI_LOCATION + ADC (the
// same auth the Claude/Gemini Vertex providers use).
func NewVertex(ctx context.Context, model string) (Embedder, error) {
	project := os.Getenv("VERTEXAI_PROJECT")
	if project == "" {
		return nil, fmt.Errorf("VERTEXAI_PROJECT not set")
	}
	location := os.Getenv("VERTEXAI_LOCATION")
	if location == "" {
		location = "us-central1"
	}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  project,
		Location: location,
	})
	if err != nil {
		return nil, fmt.Errorf("create genai client: %w", err)
	}
	return &genaiEmbedder{client: client, model: model, backend: "vertex"}, nil
}

// NewAPIKey builds a Gemini Developer API embedder using GEMINI_API_KEY
// (falling back to GOOGLE_API_KEY) — the no-GCP path, mirroring
// gemini.NewAPIKey. Note model availability differs by backend: e.g.
// gemini-embedding-001 is available here, but text-embedding-005 is Vertex-only.
func NewAPIKey(ctx context.Context, model string) (Embedder, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		key = os.Getenv("GOOGLE_API_KEY")
	}
	if key == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY not set")
	}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  key,
	})
	if err != nil {
		return nil, fmt.Errorf("create genai client: %w", err)
	}
	return &genaiEmbedder{client: client, model: model, backend: "gemini"}, nil
}

func (e *genaiEmbedder) ModelID() string { return e.backend + "_" + e.model }

func (e *genaiEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	// Clip over-long inputs (UTF-8-safe) so no single text blows the per-request
	// token budget, then batch under the instance + token caps.
	clipped := make([]string, len(texts))
	for i, t := range texts {
		if len(t) > vertexMaxCharsPerText {
			t = strings.ToValidUTF8(t[:vertexMaxCharsPerText], "")
		}
		clipped[i] = t
	}

	out := make([][]float32, 0, len(texts))
	for _, b := range embedBatches(clipped) {
		batch := clipped[b[0]:b[1]]
		contents := make([]*genai.Content, 0, len(batch))
		for _, t := range batch {
			contents = append(contents, genai.NewContentFromText(t, genai.RoleUser))
		}
		resp, err := e.client.Models.EmbedContent(ctx, e.model, contents, nil)
		if err != nil {
			return nil, fmt.Errorf("embed batch [%d,%d): %w", b[0], b[1], err)
		}
		if len(resp.Embeddings) != len(batch) {
			return nil, fmt.Errorf("embed: got %d vectors for %d inputs", len(resp.Embeddings), len(batch))
		}
		for _, emb := range resp.Embeddings {
			out = append(out, emb.Values)
		}
	}
	return out, nil
}

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

package embed

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

const openAIDefaultBaseURL = "https://api.openai.com/v1"

// openAIEmbedder talks to any OpenAI-compatible /v1/embeddings endpoint: the
// hosted API, a LiteLLM proxy, vLLM, ollama, LM Studio. It is what makes recall
// (skills + lessons) available on a self-hosted deployment, which otherwise had
// no embedder at all.
type openAIEmbedder struct {
	model   string
	baseURL string
	apiKey  string
	http    *http.Client
	// cacheTag disambiguates the stored-vector cache when the endpoint is not the
	// hosted OpenAI API — see ModelID.
	cacheTag string
}

// NewOpenAI builds an embedder against an OpenAI-compatible endpoint. baseURL
// empty means the hosted API; keyEnv empty means OPENAI_API_KEY. A missing key
// is not an error: local servers ignore auth.
func NewOpenAI(model, baseURL, keyEnv string) (Embedder, error) {
	if model == "" {
		return nil, fmt.Errorf("openai embedder: empty model")
	}
	if baseURL == "" {
		baseURL = os.Getenv("OPENAI_BASE_URL")
	}
	if baseURL == "" {
		baseURL = openAIDefaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if keyEnv == "" {
		keyEnv = "OPENAI_API_KEY"
	}
	// The SAME model name at a different endpoint is a different embedding space
	// (e.g. a local server's "text-embedding-3-small" is not OpenAI's), and the
	// vector cache is keyed by ModelID. Without this tag the two would collide
	// silently and recall would rank against mismatched vectors.
	tag := ""
	if baseURL != openAIDefaultBaseURL {
		sum := sha256.Sum256([]byte(baseURL))
		tag = "@" + hex.EncodeToString(sum[:3])
	}
	return &openAIEmbedder{
		model:    model,
		baseURL:  baseURL,
		apiKey:   os.Getenv(keyEnv),
		http:     &http.Client{Timeout: 5 * time.Minute},
		cacheTag: tag,
	}, nil
}

// ModelID is the vector-cache key: backend, model, and (for a non-default
// endpoint) a short hash of the base URL.
func (e *openAIEmbedder) ModelID() string { return "openai_" + e.model + e.cacheTag }

type openAIEmbedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (e *openAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	// Reuse the same clipping + batching budgets as the Vertex embedder: they're
	// about model context and request size, which are the same concerns here.
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
		vecs, err := e.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("embed batch [%d,%d): %w", b[0], b[1], err)
		}
		if len(vecs) != len(batch) {
			return nil, fmt.Errorf("embed: got %d vectors for %d inputs", len(vecs), len(batch))
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (e *openAIEmbedder) embedBatch(ctx context.Context, batch []string) ([][]float32, error) {
	blob, err := json.Marshal(map[string]any{"model": e.model, "input": batch})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embeddings", bytes.NewReader(blob))
	if err != nil {
		return nil, err
	}
	req.Header.Set("content-type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("authorization", "Bearer "+e.apiKey)
	}
	resp, err := e.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var er openAIEmbedResponse
	if err := json.Unmarshal(body, &er); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	// The spec says data is index-ordered, but not every server honours that, and
	// a silently permuted batch would poison the cache with mismatched vectors.
	sort.Slice(er.Data, func(i, j int) bool { return er.Data[i].Index < er.Data[j].Index })
	vecs := make([][]float32, 0, len(er.Data))
	for _, d := range er.Data {
		vecs = append(vecs, d.Embedding)
	}
	return vecs, nil
}

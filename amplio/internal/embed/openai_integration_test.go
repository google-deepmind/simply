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

//go:build integration

package embed

import (
	"context"
	"os"
	"testing"
)

// Live check against a real OpenAI-compatible /v1/embeddings endpoint:
//
//	AMPLIO_OPENAI_TEST_BASE_URL=http://localhost:4000/v1 \
//	AMPLIO_EMBED_TEST_MODEL=embed make test-integration
func TestLive_OpenAIEmbedder(t *testing.T) {
	base := os.Getenv("AMPLIO_OPENAI_TEST_BASE_URL")
	if base == "" {
		t.Skip("set AMPLIO_OPENAI_TEST_BASE_URL to run the live embedder check")
	}
	model := os.Getenv("AMPLIO_EMBED_TEST_MODEL")
	if model == "" {
		model = "text-embedding-3-small"
	}
	e, err := NewOpenAI(model, base, "")
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	vecs, err := e.Embed(context.Background(), []string{"hello world", "a second string"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("vectors = %d, want 2 (aligned 1:1 with the inputs)", len(vecs))
	}
	if len(vecs[0]) == 0 || len(vecs[0]) != len(vecs[1]) {
		t.Fatalf("dims = %d and %d, want equal and non-zero", len(vecs[0]), len(vecs[1]))
	}
	// Cache-key rule, in both directions: a NON-default endpoint must contribute
	// a base-URL tag (two servers' vectors for the same model name are different
	// embedding spaces and must not collide), while the hosted API — the default
	// — must NOT, or every existing cached vector would be orphaned.
	id := e.ModelID()
	if base == "https://api.openai.com/v1" {
		if id != "openai_"+model {
			t.Errorf("ModelID = %q, want the untagged %q for the default endpoint", id, "openai_"+model)
		}
	} else if id == "openai_"+model {
		t.Errorf("ModelID = %q, want a base-URL tag for a non-default endpoint", id)
	}
	t.Logf("model=%s dims=%d cache-key=%s", model, len(vecs[0]), e.ModelID())
}

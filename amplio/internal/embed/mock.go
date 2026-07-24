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
	"context"
	"hash/fnv"
	"strings"
)

// Mock is a deterministic, network-free embedder for tests and offline dev. It
// is a hashed bag-of-words: each lowercase token bumps one dimension, so texts
// that share words land closer in cosine space (enough to exercise recall
// ranking) while staying fully reproducible.
type Mock struct {
	Dim int // vector size; defaults to 64
}

func (m Mock) ModelID() string { return "mock" }

func (m Mock) dim() int {
	if m.Dim > 0 {
		return m.Dim
	}
	return 64
}

func (m Mock) Embed(_ context.Context, texts []string) ([][]float32, error) {
	d := m.dim()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, d)
		for _, w := range strings.Fields(strings.ToLower(t)) {
			h := fnv.New32a()
			_, _ = h.Write([]byte(w))
			v[h.Sum32()%uint32(d)]++
		}
		out[i] = v
	}
	return out, nil
}

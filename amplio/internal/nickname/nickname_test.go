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

package nickname

import (
	"math/rand/v2"
	"strings"
	"testing"
)

func TestPickUnique_Basic(t *testing.T) {
	name := PickUnique(nil, nil)
	parts := strings.SplitN(name, "-", 2)
	if len(parts) != 2 {
		t.Fatalf("expected adj-noun, got %q", name)
	}
	if parts[0] == "" || parts[1] == "" {
		t.Fatalf("empty component in %q", name)
	}
}

func TestPickUnique_NeverReturnsReserved(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0)) //nolint:gosec // test determinism
	for range 200 {
		name := PickUnique(nil, rng)
		if IsReserved(name) {
			t.Fatalf("got reserved name %q", name)
		}
	}
}

func TestPickUnique_RespectsUsed(t *testing.T) {
	used := map[string]bool{"swift-fox": true}
	rng := rand.New(rand.NewPCG(42, 0)) //nolint:gosec // test determinism
	for range 500 {
		name := PickUnique(used, rng)
		if name == "swift-fox" {
			t.Fatal("returned a used name")
		}
	}
}

func TestPickUnique_Deterministic(t *testing.T) {
	a := PickUnique(nil, rand.New(rand.NewPCG(99, 0))) //nolint:gosec // test determinism
	b := PickUnique(nil, rand.New(rand.NewPCG(99, 0))) //nolint:gosec // test determinism
	if a != b {
		t.Fatalf("same seed produced different names: %q vs %q", a, b)
	}
}

func TestPickUnique_FallbackOnExhaustion(t *testing.T) {
	// Fill the entire dynamic pool.
	used := make(map[string]bool)
	for _, a := range adjectives {
		if reservedAdj[a] {
			continue
		}
		for _, n := range nouns {
			if reservedNoun[n] {
				continue
			}
			used[a+"-"+n] = true
		}
	}
	rng := rand.New(rand.NewPCG(42, 0)) //nolint:gosec // test determinism
	name := PickUnique(used, rng)
	// Fallback names have 3 segments.
	parts := strings.Split(name, "-")
	if len(parts) != 3 {
		t.Fatalf("expected adj-noun-suffix, got %q", name)
	}
	if len(parts[2]) != fallbackSuffixLen {
		t.Fatalf("suffix length %d, want %d, in %q", len(parts[2]), fallbackSuffixLen, name)
	}
}

func TestPickUnique_NoReservedComponents(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0)) //nolint:gosec // test determinism
	for range 500 {
		name := PickUnique(nil, rng)
		parts := strings.SplitN(name, "-", 2)
		if reservedAdj[parts[0]] {
			t.Fatalf("returned reserved adjective %q in %q", parts[0], name)
		}
		if reservedNoun[parts[1]] {
			t.Fatalf("returned reserved noun %q in %q", parts[1], name)
		}
	}
}

func TestIsReserved(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{RootAgent, true},
		{Chatbot, true},
		{"keen-critic", false}, // no longer reserved
		{"swift-fox", false},
		{"main-fox", false}, // "main" is reserved adj but "main-fox" is not a reserved name
		{"", false},
	}
	for _, tt := range tests {
		if got := IsReserved(tt.name); got != tt.want {
			t.Errorf("IsReserved(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}

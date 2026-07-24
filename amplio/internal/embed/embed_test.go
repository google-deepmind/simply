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
	"strings"
	"testing"
)

func TestEmbedBatches(t *testing.T) {
	// Small texts well within both budgets → a single batch.
	if got := embedBatches([]string{"a", "b", "c"}); len(got) != 1 || got[0] != [2]int{0, 3} {
		t.Fatalf("small: %v, want one [0,3]", got)
	}

	// Tiny texts past the instance cap (token budget irrelevant) → split at 250.
	many := make([]string, 600)
	for i := range many {
		many[i] = "x"
	}
	got := embedBatches(many)
	want := [][2]int{{0, 250}, {250, 500}, {500, 600}}
	if len(got) != len(want) {
		t.Fatalf("instance cap: %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("instance cap[%d]=%v, want %v", i, got[i], want[i])
		}
	}

	// Token budget forces a split: two ~10k-token texts can't share the budget.
	big := strings.Repeat("x", 30000) // estTokens ≈ 10000, > half the budget
	if got := embedBatches([]string{big, big}); len(got) != 2 {
		t.Fatalf("token budget: %v, want 2 batches", got)
	}

	// A single text over the budget still gets its own batch (Embed clips it to
	// stay under the hard cap).
	huge := strings.Repeat("y", 60000) // estTokens ≈ 20000 > budget
	got = embedBatches([]string{huge, "small"})
	if len(got) != 2 || got[0] != [2]int{0, 1} || got[1] != [2]int{1, 2} {
		t.Fatalf("oversized solo: %v, want [0,1],[1,2]", got)
	}
}

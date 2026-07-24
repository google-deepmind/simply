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

package vec

import (
	"math"
	"testing"
)

func TestNormalizeUnitLength(t *testing.T) {
	out := Normalize([]float32{3, 4}) // norm 5
	var sum float64
	for _, f := range out {
		sum += float64(f) * float64(f)
	}
	if math.Abs(math.Sqrt(sum)-1) > 1e-6 {
		t.Errorf("normalized norm = %v, want 1", math.Sqrt(sum))
	}
}

func TestNormalizeDegenerateIsZeros(t *testing.T) {
	out := Normalize([]float32{0, 0, 0})
	for i, f := range out {
		if f != 0 {
			t.Errorf("out[%d] = %v, want 0 for a degenerate vector", i, f)
		}
	}
}

// TestNormalizeOrNilDegenerateReturnsNil is the whole reason the helper exists:
// a query whose vector is below MinNorm must return nil so callers can bail
// before sort.Slice picks an arbitrary top-k off all-tied-at-zero scores.
func TestNormalizeOrNilDegenerateReturnsNil(t *testing.T) {
	if out := NormalizeOrNil([]float32{0, 0, 0}); out != nil {
		t.Errorf("NormalizeOrNil(zero) = %v, want nil", out)
	}
	// Just under the floor.
	if out := NormalizeOrNil([]float32{MinNorm / 2, 0, 0}); out != nil {
		t.Errorf("NormalizeOrNil(sub-MinNorm) = %v, want nil", out)
	}
}

func TestNormalizeOrNilNonDegenerateIsUnit(t *testing.T) {
	out := NormalizeOrNil([]float32{3, 4}) // norm 5
	if out == nil {
		t.Fatal("NormalizeOrNil(non-degenerate) returned nil")
	}
	var sum float64
	for _, f := range out {
		sum += float64(f) * float64(f)
	}
	if math.Abs(math.Sqrt(sum)-1) > 1e-6 {
		t.Errorf("normalized norm = %v, want 1", math.Sqrt(sum))
	}
}

func TestDotCosineOfNormalized(t *testing.T) {
	a := Normalize([]float32{1, 0})
	b := Normalize([]float32{1, 0})
	if got := Dot(a, b); math.Abs(got-1) > 1e-6 {
		t.Errorf("Dot(identical) = %v, want 1", got)
	}
	o := Normalize([]float32{0, 1})
	if got := Dot(a, o); math.Abs(got) > 1e-6 {
		t.Errorf("Dot(orthogonal) = %v, want 0", got)
	}
}

// TestDotLengthMismatch is the point of the shared helper: differing dimensions
// (different embedding models) score 0, not a misleading shared-prefix dot.
func TestDotLengthMismatch(t *testing.T) {
	if got := Dot([]float32{1, 2, 3}, []float32{1, 2}); got != 0 {
		t.Errorf("Dot(mismatched lengths) = %v, want 0", got)
	}
}

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

// Package vec holds the small embedding-vector helpers shared by the skill and
// lesson recall indices (cosine similarity over L2-normalized vectors).
package vec

import "math"

// MinNorm is the floor below which a vector is treated as degenerate (its
// Normalize result is all zeros, yielding cosine 0 against anything).
const MinNorm = 1e-7

// Normalize returns an L2-normalized copy of v. A vector whose norm is below
// MinNorm is returned as zeros (so Dot against it is 0 rather than NaN/Inf).
func Normalize(v []float32) []float32 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	n := math.Sqrt(sum)
	out := make([]float32, len(v))
	if n < MinNorm {
		return out // zeros
	}
	for i, f := range v {
		out[i] = float32(float64(f) / n)
	}
	return out
}

// NormalizeOrNil is the query-side variant of Normalize: it returns nil for a
// degenerate input (norm below MinNorm) instead of an all-zero vector. Callers
// ranking against a corpus should bail on nil — a zero query vector dots to 0
// against every row, leaving the (unstable) sort to pick arbitrary winners,
// which is worse than returning no results. Storage-side callers that just want
// "score 0 means buries itself" can keep using Normalize.
func NormalizeOrNil(v []float32) []float32 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	n := math.Sqrt(sum)
	if n < MinNorm {
		return nil
	}
	out := make([]float32, len(v))
	for i, f := range v {
		out[i] = float32(float64(f) / n)
	}
	return out
}

// Dot is the dot product of two equal-length vectors (cosine similarity when
// both are Normalize'd). Vectors of differing length are treated as
// incomparable and return 0 rather than silently scoring on the shared prefix:
// a mismatch means the query and the index were embedded by different models
// (dimension is model-determined), so any partial score would be meaningless.
func Dot(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

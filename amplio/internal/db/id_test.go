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

package db

import (
	"errors"
	"strings"
	"testing"
)

func TestNewRunID(t *testing.T) {
	seen := make(map[string]bool, 2000)
	for i := 0; i < 2000; i++ {
		id := NewRunID()
		if len(id) != runIDLen {
			t.Fatalf("len = %d, want %d (%q)", len(id), runIDLen, id)
		}
		for _, c := range id {
			if !strings.ContainsRune(runIDAlphabet, c) {
				t.Fatalf("char %q not in alphabet (%q)", c, id)
			}
		}
		if seen[id] {
			t.Fatalf("duplicate id %q within 2000 draws", id)
		}
		seen[id] = true
	}
}

func TestIsUniqueViolation(t *testing.T) {
	if IsUniqueViolation(nil) {
		t.Error("nil is not a violation")
	}
	if IsUniqueViolation(errors.New("disk full")) {
		t.Error("unrelated error misclassified")
	}
	if !IsUniqueViolation(errors.New("constraint failed: UNIQUE constraint failed: Run.run_id (2067)")) {
		t.Error("real UNIQUE violation not detected")
	}
}

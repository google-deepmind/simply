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

package blob

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestPutRoundTrip(t *testing.T) {
	s := NewStore(t.TempDir())
	data := []byte("hello image bytes")

	key, err := s.Put(data)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if !ValidKey(key) {
		t.Errorf("Put returned invalid key %q", key)
	}
	if key != Key(data) {
		t.Errorf("Put key %q != Key(data) %q", key, Key(data))
	}

	got, err := s.ReadAll(key)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Errorf("ReadAll = %q, want %q", got, data)
	}
}

func TestPutDedup(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	data := []byte("dup")
	k1, _ := s.Put(data)
	k2, err := s.Put(data)
	if err != nil || k1 != k2 {
		t.Fatalf("Put not idempotent: %q/%q err=%v", k1, k2, err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Errorf("expected 1 file after dedup, got %d", len(entries))
	}
}

func TestOpenRejectsTraversal(t *testing.T) {
	s := NewStore(t.TempDir())
	for _, bad := range []string{
		"../etc/passwd",
		"..",
		"short",
		"NOTHEX!!" + string(make([]byte, 56)),
		filepath.Join("a", "b"),
	} {
		if _, err := s.Open(bad); err == nil {
			t.Errorf("Open(%q) should reject invalid key", bad)
		}
	}
}

func TestValidKey(t *testing.T) {
	good := Key([]byte("x")) // 64 hex chars
	if !ValidKey(good) {
		t.Errorf("ValidKey(%q) = false", good)
	}
	if ValidKey(good+"a") || ValidKey("ABCDEF") || ValidKey("") {
		t.Error("ValidKey accepted a malformed key")
	}
}

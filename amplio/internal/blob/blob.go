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

// Package blob provides a content-addressed, run-scoped store for binary
// artifacts (today, tool-result images) kept on disk instead of inline in the
// event log, so SQLite event/FTS rows stay small.
package blob

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Store is a flat content-addressed blob directory. Keys are the lowercase hex
// SHA-256 of the contents, so writes dedup and keys are path-safe by
// construction. The zero value is unusable; use NewStore. Safe for concurrent
// use.
type Store struct{ root string }

// NewStore returns a Store rooted at dir. The directory is created lazily on the
// first Put.
func NewStore(dir string) *Store { return &Store{root: dir} }

// Key computes the content key for data without writing it.
func Key(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// Put writes data under its content key and returns the key. It is idempotent: a
// no-op when the blob already exists. The write is atomic (temp file + rename)
// so a crash can't leave a partial blob under a valid key.
func (s *Store) Put(data []byte) (string, error) {
	key := Key(data)
	path := filepath.Join(s.root, key)
	if _, err := os.Stat(path); err == nil {
		return key, nil
	}
	if err := os.MkdirAll(s.root, 0o755); err != nil {
		return "", fmt.Errorf("blob: mkdir: %w", err)
	}
	tmp, err := os.CreateTemp(s.root, ".tmp-*")
	if err != nil {
		return "", fmt.Errorf("blob: temp: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once renamed
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return "", fmt.Errorf("blob: write: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", fmt.Errorf("blob: close: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return "", fmt.Errorf("blob: rename: %w", err)
	}
	return key, nil
}

// Open opens the blob for key. The key must be a valid content key, which guards
// path-derived lookups (e.g. the HTTP route) against traversal.
func (s *Store) Open(key string) (*os.File, error) {
	if !ValidKey(key) {
		return nil, fmt.Errorf("blob: invalid key %q", key)
	}
	//nolint:gosec // key is validated as 64-char hex (ValidKey); no traversal possible.
	return os.Open(filepath.Join(s.root, key))
}

// ReadAll returns the full contents of the blob for key.
func (s *Store) ReadAll(key string) ([]byte, error) {
	f, err := s.Open(key)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}

// ValidKey reports whether key is a well-formed content key (64 lowercase hex
// chars), i.e. safe to use as a filename with no path traversal.
func ValidKey(key string) bool {
	if len(key) != 64 {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

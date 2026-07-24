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

package workspace

import (
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
)

// PathExists reports whether a filesystem path exists.
func PathExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// RandSuffix returns 4 random hex chars for disambiguating a name on a clash.
// Not security-sensitive (callers also probe + let the create command be the
// final guard); crypto/rand just keeps it lint-clean.
func RandSuffix() string {
	var b [2]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "0000"
	}
	return fmt.Sprintf("%02x%02x", b[0], b[1])
}

// FreeSiblingPath returns baseDir/name, appending -<4hex> on the (rare) clash
// with an existing path. The caller's subsequent create (worktree add, etc.) is
// the final collision guard.
func FreeSiblingPath(baseDir, name string) string {
	if p := filepath.Join(baseDir, name); !PathExists(p) {
		return p
	}
	for range 10 {
		if cand := filepath.Join(baseDir, name+"-"+RandSuffix()); !PathExists(cand) {
			return cand
		}
	}
	return filepath.Join(baseDir, name+"-"+RandSuffix())
}

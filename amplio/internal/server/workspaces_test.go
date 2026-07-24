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

package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestListRecentWorkspaces(t *testing.T) {
	root := t.TempDir()
	// Create dirs with explicit, increasing mtimes; expect MRU-first order.
	base := time.Now()
	for i, name := range []string{"oldest", "middle", "newest"} {
		p := filepath.Join(root, name)
		if err := os.Mkdir(p, 0o755); err != nil {
			t.Fatal(err)
		}
		mt := base.Add(time.Duration(i) * time.Hour)
		if err := os.Chtimes(p, mt, mt); err != nil {
			t.Fatal(err)
		}
	}
	// A regular file must be ignored.
	if err := os.WriteFile(filepath.Join(root, "afile"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	got := listRecentWorkspaces(root)
	want := []string{"newest", "middle", "oldest"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("position %d: got %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

func TestListRecentWorkspacesMissingDir(t *testing.T) {
	if got := listRecentWorkspaces(filepath.Join(t.TempDir(), "nope")); got != nil {
		t.Errorf("expected nil for missing dir, got %v", got)
	}
}

func TestListRecentWorkspacesCap(t *testing.T) {
	root := t.TempDir()
	for i := 0; i < recentWorkspacesLimit+10; i++ {
		if err := os.Mkdir(filepath.Join(root, string(rune('a'+i%26))+"-"+itoa(i)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if got := listRecentWorkspaces(root); len(got) != recentWorkspacesLimit {
		t.Errorf("expected cap %d, got %d", recentWorkspacesLimit, len(got))
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

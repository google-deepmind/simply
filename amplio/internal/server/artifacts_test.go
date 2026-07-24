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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"amplio/internal/config"
)

func TestServer_Artifacts(t *testing.T) {
	config.SetDataDir(t.TempDir())
	t.Cleanup(func() { config.SetDataDir("") })
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Populate the run's artifact dir: a file + a subdir with a file.
	base := config.ArtifactDir(testRun)
	if err := os.WriteFile(filepath.Join(base, "plan.md"), []byte("# Plan\nstep 1"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(base, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "sub", "note.txt"), []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}

	get := func(path string) (int, string, []byte) {
		t.Helper()
		req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+path, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close() //nolint:errcheck
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header.Get("Content-Security-Policy"), b
	}

	// Root listing: dirs first (sub), then files (plan.md).
	status, _, body := get("/api/runs/" + testRun + "/artifacts?token=secret")
	if status != 200 {
		t.Fatalf("list status = %d", status)
	}
	var listing artifactListing
	if err := json.Unmarshal(body, &listing); err != nil {
		t.Fatal(err)
	}
	if len(listing.Entries) != 2 || !listing.Entries[0].IsDir || listing.Entries[0].Name != "sub" || listing.Entries[1].Name != "plan.md" {
		t.Fatalf("entries = %+v", listing.Entries)
	}

	// Subdir listing.
	_, _, body = get("/api/runs/" + testRun + "/artifacts?path=sub&token=secret")
	_ = json.Unmarshal(body, &listing)
	if len(listing.Entries) != 1 || listing.Entries[0].Name != "note.txt" {
		t.Errorf("sub entries = %+v", listing.Entries)
	}

	// File serve: content + CSP sandbox.
	status, csp, body := get("/api/runs/" + testRun + "/artifacts/raw?path=plan.md&token=secret")
	if status != 200 || string(body) != "# Plan\nstep 1" {
		t.Errorf("raw status=%d body=%q", status, body)
	}
	if !strings.Contains(csp, "sandbox") {
		t.Errorf("missing CSP sandbox header: %q", csp)
	}

	// Traversal attempt is confined by os.Root → not found.
	if status, _, _ := get("/api/runs/" + testRun + "/artifacts/raw?path=../../config.toml&token=secret"); status != http.StatusNotFound {
		t.Errorf("traversal status = %d, want 404", status)
	}

	// Listing a file (not a dir) → 400.
	if status, _, _ := get("/api/runs/" + testRun + "/artifacts?path=plan.md&token=secret"); status != http.StatusBadRequest {
		t.Errorf("list-a-file status = %d, want 400", status)
	}

	// Recursive listing: every FILE (not dir), forward-slashed subpaths, sorted.
	status, _, body = get("/api/runs/" + testRun + "/artifacts/all?token=secret")
	if status != 200 {
		t.Fatalf("all status = %d", status)
	}
	var all struct {
		Root  string         `json:"root"`
		Files []artifactFile `json:"files"`
	}
	if err := json.Unmarshal(body, &all); err != nil {
		t.Fatal(err)
	}
	gotPaths := make([]string, len(all.Files))
	for i, f := range all.Files {
		gotPaths[i] = f.Path
	}
	want := []string{"plan.md", "sub/note.txt"}
	if len(gotPaths) != len(want) || gotPaths[0] != want[0] || gotPaths[1] != want[1] {
		t.Errorf("recursive files = %v, want %v", gotPaths, want)
	}
}

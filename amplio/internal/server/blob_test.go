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
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"amplio/internal/blob"
	"amplio/internal/config"
)

func TestServer_Blob(t *testing.T) {
	config.SetDataDir(t.TempDir())
	srv, _, _ := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	png := []byte("\x89PNG\r\n\x1a\nfake png payload")
	key, err := blob.NewStore(config.BlobDir(testRun)).Put(png)
	if err != nil {
		t.Fatal(err)
	}

	get := func(path string) (status int, contentType string, body []byte) {
		t.Helper()
		req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+path, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close() //nolint:errcheck
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header.Get("Content-Type"), b
	}

	// Valid blob: 200, image content-type, exact bytes.
	status, ct, body := get("/api/runs/" + testRun + "/blobs/" + key + "?token=secret")
	if status != http.StatusOK {
		t.Fatalf("valid blob status = %d", status)
	}
	if !strings.HasPrefix(ct, "image/") {
		t.Errorf("content-type = %q, want image/*", ct)
	}
	if !bytes.Equal(body, png) {
		t.Error("served bytes differ from stored blob")
	}

	// Malformed key: 400.
	if status, _, _ := get("/api/runs/" + testRun + "/blobs/not-a-key?token=secret"); status != http.StatusBadRequest {
		t.Errorf("bad key status = %d, want 400", status)
	}

	// Well-formed but absent key: 404.
	absent := blob.Key([]byte("nothing stored under this"))
	if status, _, _ := get("/api/runs/" + testRun + "/blobs/" + absent + "?token=secret"); status != http.StatusNotFound {
		t.Errorf("absent blob status = %d, want 404", status)
	}
}

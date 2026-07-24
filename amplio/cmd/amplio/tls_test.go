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

package main

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveTLS_NoCertNoMkcert_ReturnsEmpty(t *testing.T) {
	// Isolated PATH so exec.LookPath("mkcert") fails deterministically.
	t.Setenv("PATH", "")
	dataDir := t.TempDir()

	certFile, keyFile, err := resolveTLS(context.Background(), dataDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if certFile != "" || keyFile != "" {
		t.Errorf("expected empty paths when no cert and no mkcert; got %q %q", certFile, keyFile)
	}
}

func TestResolveTLS_ExistingCert_ReturnsPaths(t *testing.T) {
	dataDir := t.TempDir()
	certFile, keyFile := tlsPaths(dataDir)
	if err := os.WriteFile(certFile, []byte("placeholder"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyFile, []byte("placeholder"), 0o600); err != nil {
		t.Fatal(err)
	}

	gotCert, gotKey, err := resolveTLS(context.Background(), dataDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotCert != certFile || gotKey != keyFile {
		t.Errorf("paths = %q %q, want %q %q", gotCert, gotKey, certFile, keyFile)
	}
}

func TestResolveTLS_OnlyCert_NoMkcert_FallsBack(t *testing.T) {
	// Only cert present (no key) should NOT be treated as configured: both
	// files must exist or ServeTLS fails at runtime.
	t.Setenv("PATH", "")
	dataDir := t.TempDir()
	certFile, _ := tlsPaths(dataDir)
	if err := os.WriteFile(certFile, []byte("placeholder"), 0o600); err != nil {
		t.Fatal(err)
	}

	gotCert, gotKey, err := resolveTLS(context.Background(), dataDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotCert != "" || gotKey != "" {
		t.Errorf("expected empty paths when only one of cert/key present; got %q %q", gotCert, gotKey)
	}
}

func TestFileExists(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "exists")
	if err := os.WriteFile(f, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if !fileExists(f) {
		t.Error("regular file should exist")
	}
	if fileExists(filepath.Join(dir, "missing")) {
		t.Error("missing file should not exist")
	}
	if fileExists(dir) {
		t.Error("directory should not count as a file")
	}
}

func TestIsLoopback(t *testing.T) {
	cases := map[string]bool{
		"localhost":   true,
		"127.0.0.1":   true,
		"127.0.0.42":  true, // any 127.0.0.0/8
		"::1":         true,
		"0.0.0.0":     false,
		"example.com": false,
		"192.168.1.1": false,
		"":            false,
	}
	for host, want := range cases {
		if got := isLoopback(host); got != want {
			t.Errorf("isLoopback(%q) = %v, want %v", host, got, want)
		}
	}
}

func TestLoopbackHTTPClient(t *testing.T) {
	cases := []struct {
		name       string
		url        string
		skipVerify bool
	}{
		{"http loopback uses default", "http://127.0.0.1:8080", false},
		{"https loopback skips verify", "https://127.0.0.1:8080", true},
		{"https localhost skips verify", "https://localhost:8080", true},
		{"https public verifies", "https://example.com", false},
		{"http public uses default", "http://example.com", false},
		{"malformed uses default", "::not a url::", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			client := loopbackHTTPClient(c.url)
			tr, ok := client.Transport.(*http.Transport)
			if !ok && c.skipVerify {
				t.Fatalf("expected custom transport for skip-verify case")
			}
			if c.skipVerify {
				if tr.TLSClientConfig == nil || !tr.TLSClientConfig.InsecureSkipVerify {
					t.Errorf("expected InsecureSkipVerify=true for %q", c.url)
				}
			} else if client != http.DefaultClient {
				t.Errorf("expected http.DefaultClient for %q", c.url)
			}
		})
	}
}

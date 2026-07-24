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
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
)

// tlsCertName / tlsKeyName are the conventional filenames amplio looks for
// inside the data dir. When both exist the server starts in HTTPS mode (which
// auto-enables HTTP/2 in Go's net/http), fixing the per-origin 6-connection
// cap that otherwise throttles multi-tab SSE on HTTP/1.1.
const (
	tlsCertName = "cert.pem"
	tlsKeyName  = "key.pem"
)

// tlsPaths returns the conventional cert/key paths inside the data dir.
func tlsPaths(dataDir string) (certFile, keyFile string) {
	return filepath.Join(dataDir, tlsCertName), filepath.Join(dataDir, tlsKeyName)
}

// resolveTLS returns the cert/key paths to use, generating them via mkcert if
// they don't exist and mkcert is on PATH. Returns ("", "", nil) when no cert is
// present and mkcert isn't available — caller falls back to plain HTTP.
//
// The auto-generation step requires the user to have run `mkcert -install`
// previously (one-time per machine) so the local CA is trusted by browsers;
// we DON'T run it automatically because it needs sudo and modifies the system
// trust store. A warning is logged on first cert generation pointing at the
// docs.
func resolveTLS(ctx context.Context, dataDir string) (certFile, keyFile string, err error) {
	certFile, keyFile = tlsPaths(dataDir)
	if fileExists(certFile) && fileExists(keyFile) {
		return certFile, keyFile, nil
	}
	// No cert yet. Try mkcert auto-gen.
	mkcert, lookErr := exec.LookPath("mkcert")
	if lookErr != nil {
		slog.Debug("mkcert not on PATH; serving plain HTTP",
			"hint", "install mkcert and run `mkcert -install` once to enable HTTPS+HTTP/2",
		)
		return "", "", nil
	}
	if err := generateMkcertCert(ctx, mkcert, certFile, keyFile); err != nil {
		return "", "", fmt.Errorf("mkcert: %w", err)
	}
	slog.Info("generated localhost TLS cert via mkcert",
		"cert", certFile, "key", keyFile,
		"hint", "if browsers complain about an untrusted cert, run `mkcert -install` once on THIS machine; "+
			"if connecting via SSH tunnel from a laptop, generate the cert on the laptop instead "+
			"(where the browser's trust store lives) and copy cert.pem/key.pem to this data dir",
	)
	return certFile, keyFile, nil
}

// generateMkcertCert shells out to mkcert to generate a cert+key for
// localhost. mkcert produces files relative to its working directory unless
// -cert-file / -key-file are explicit, so we always pass them. Includes
// 127.0.0.1 + ::1 so the local CLI clients (loopback) also validate cleanly
// against the cert's SANs.
func generateMkcertCert(ctx context.Context, mkcertBin, certFile, keyFile string) error {
	if err := os.MkdirAll(filepath.Dir(certFile), 0o755); err != nil {
		return fmt.Errorf("mkdir cert dir: %w", err)
	}
	// mkcertBin is from exec.LookPath; the SANs are hardcoded literals from
	// this file. No untrusted input.
	cmd := exec.CommandContext(ctx, mkcertBin, //nolint:gosec
		"-cert-file", certFile,
		"-key-file", keyFile,
		"localhost", "127.0.0.1", "::1",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, string(out))
	}
	return nil
}

// fileExists reports whether path resolves to a regular file (not a directory
// or missing). Helper for TLS cert detection.
func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

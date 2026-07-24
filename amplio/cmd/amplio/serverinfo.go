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
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gofrs/flock"
)

// serverInfo is the discovery metadata a running `serve` writes to
// <data-dir>/server.json so clients (e.g. `submit`) can find it. It is NOT a
// lock — the data-dir flock is. A stale file (after a crash) is harmless: the
// URL simply refuses the connection.
type serverInfo struct {
	PID   int    `json:"pid"`
	URL   string `json:"url"`   // banner base URL for humans/frontend, e.g. http://host:port
	Addr  string `json:"addr"`  // loopback base URL for CLI clients, e.g. http://127.0.0.1:port
	Token string `json:"token"` // web auth token
}

func serverInfoPath(dataDir string) string { return filepath.Join(dataDir, "server.json") }

func tokenFilePath(dataDir string) string { return filepath.Join(dataDir, "auth.token") }

// authCookieName derives a per-data-dir auth cookie name. Cookies ignore the
// port, so two amplio servers on the same host would otherwise share one cookie
// and clobber each other's session; keying the name to the data dir (the
// instance's stable identity) keeps them independent — and stable across
// restarts and port changes.
func authCookieName(dataDir string) string {
	if abs, err := filepath.Abs(dataDir); err == nil {
		dataDir = abs
	}
	sum := sha256.Sum256([]byte(dataDir))
	return "amplio_auth_" + hex.EncodeToString(sum[:4]) // 8 hex chars
}

// loadOrCreateToken returns the durable web auth token for a data dir, creating
// it (32 bytes of crypto-random, base64url) on first use. Unlike server.json
// (rewritten each start and deleted on clean exit), this file persists across
// restarts, so the magic-link cookie a user already holds stays valid — they
// don't have to re-open the tokened URL after every restart. Rotating auth is
// just deleting this file. 0600: it's a credential.
func loadOrCreateToken(dataDir string) (string, error) {
	path := tokenFilePath(dataDir)
	if data, err := os.ReadFile(path); err == nil {
		if tok := string(data); tok != "" {
			return tok, nil
		}
	}
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generate token: %w", err)
	}
	tok := base64.RawURLEncoding.EncodeToString(b)
	if err := os.WriteFile(path, []byte(tok), 0o600); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}
	return tok, nil
}

func writeServerInfo(dataDir string, info serverInfo) error {
	data, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return err
	}
	// 0600: contains the auth token.
	if err := os.WriteFile(serverInfoPath(dataDir), data, 0o600); err != nil {
		return fmt.Errorf("write %s: %w", serverInfoPath(dataDir), err)
	}
	return nil
}

func readServerInfo(dataDir string) (serverInfo, error) {
	data, err := os.ReadFile(serverInfoPath(dataDir))
	if err != nil {
		return serverInfo{}, err
	}
	var info serverInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return serverInfo{}, fmt.Errorf("parse %s: %w", serverInfoPath(dataDir), err)
	}
	return info, nil
}

// lockDataDir takes the exclusive owner lock for a data dir. Only one process
// (a serve / run / resume) may own a data dir at a time, since they share one
// sqlite DB and would otherwise double-execute the same runs. The lock is an
// advisory flock the kernel releases automatically on process exit — including
// a crash — so there is never a stale lock to clean up by hand.
func lockDataDir(dataDir string) (*flock.Flock, error) {
	lk := flock.New(filepath.Join(dataDir, "lock"))
	ok, err := lk.TryLock()
	if err != nil {
		return nil, fmt.Errorf("lock data dir %s: %w", dataDir, err)
	}
	if !ok {
		hint := ""
		if info, e := readServerInfo(dataDir); e == nil {
			hint = fmt.Sprintf(" (pid %d, %s)", info.PID, info.URL)
		}
		return nil, fmt.Errorf("another amplio process already owns %s%s", dataDir, hint)
	}
	return lk, nil
}

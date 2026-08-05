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
	"os"
	"path/filepath"
	"testing"

	"amplio/internal/config"
)

func TestServerInfoRoundTrip(t *testing.T) {
	dir := t.TempDir()
	in := serverInfo{PID: 1234, URL: "http://host:26759", Token: "deadbeef"}
	if err := writeServerInfo(dir, in); err != nil {
		t.Fatal(err)
	}
	out, err := readServerInfo(dir)
	if err != nil {
		t.Fatal(err)
	}
	if out != in {
		t.Errorf("round trip = %+v, want %+v", out, in)
	}
}

func TestReadServerInfo_Missing(t *testing.T) {
	if _, err := readServerInfo(t.TempDir()); err == nil {
		t.Error("expected error when server.json is absent")
	}
}

func TestLockDataDir_Exclusive(t *testing.T) {
	dir := t.TempDir()
	lk, err := lockDataDir(dir)
	if err != nil {
		t.Fatalf("first lock: %v", err)
	}
	if _, err := lockDataDir(dir); err == nil {
		t.Error("second lock should fail while the first is held")
	}
	if err := lk.Unlock(); err != nil {
		t.Fatalf("unlock: %v", err)
	}
	lk2, err := lockDataDir(dir)
	if err != nil {
		t.Fatalf("lock after unlock should succeed: %v", err)
	}
	_ = lk2.Unlock()
}

// Claiming the data dir must also furnish it. This lives with the lock (rather
// than in serve) precisely so headless runs get it too: the first version
// installed the shim in serve only, and headless agents were told by their
// prompt to use a command that did not exist for them.
func TestLockDataDirInstallsNotifyShim(t *testing.T) {
	dir := t.TempDir()
	lk, err := lockDataDir(dir)
	if err != nil {
		t.Fatalf("lockDataDir: %v", err)
	}
	defer lk.Unlock() //nolint:errcheck

	exe, _ := os.Executable()
	// Both names: the narrow one the prompt teaches, and the CLI the skill does.
	for _, name := range []string{config.NotifyShimName, config.CLIShimName} {
		target, err := os.Readlink(filepath.Join(dir, "bin", name))
		if err != nil {
			t.Fatalf("no %s shim: %v", name, err)
		}
		if target != exe {
			t.Errorf("%s points at %q, want the running binary %q", name, target, exe)
		}
	}
	// Rerunning must be idempotent — every boot re-claims the same data dir.
	if err := installNotifyShim(dir); err != nil {
		t.Errorf("second install: %v", err)
	}
}

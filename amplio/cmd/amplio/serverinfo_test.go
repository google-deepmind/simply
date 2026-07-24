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

import "testing"

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

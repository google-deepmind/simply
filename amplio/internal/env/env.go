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

// Package env reports facts about the runtime environment that gate 1P-only
// features.
package env

import (
	"os"
	"strings"
	"sync"
)

// EnvInternal overrides internal-mode detection: "yes"/"true"/"1" force on,
// "no"/"false"/"0" force off; anything else (including unset) auto-detects.
const EnvInternal = "AMPLIO_INTERNAL"

// internalProbe is the filesystem marker for a 1P dev environment: the
// /google mount.
const internalProbe = "/google"

var (
	internalOnce sync.Once
	internalVal  bool
)

// Internal reports whether amplio is running inside a 1P Google developer
// environment — characterized by the /google mount, LOAS auth, and 1P CLI
// tools. It gates 1P-only features (e.g. CitC workspaces). Detected once:
// AMPLIO_INTERNAL wins; otherwise we probe for the /google mount.
func Internal() bool {
	internalOnce.Do(func() { internalVal = detect(os.Getenv(EnvInternal), internalProbe) })
	return internalVal
}

// detect is the pure decision: an explicit override short-circuits the probe.
func detect(override, probePath string) bool {
	switch strings.ToLower(strings.TrimSpace(override)) {
	case "yes", "true", "1":
		return true
	case "no", "false", "0":
		return false
	}
	fi, err := os.Stat(probePath)
	return err == nil && fi.IsDir()
}

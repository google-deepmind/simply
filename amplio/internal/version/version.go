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

// Package version exposes the binary's build identity for the `amplio
// --version` / `amplio version` outputs.
//
// The build identity is read at runtime from debug.ReadBuildInfo() — the
// stdlib mechanism Go 1.18+ uses to embed the VCS revision, commit time, and
// dirty flag automatically when `go build` runs inside a git checkout
// (controlled by `-buildvcs=true`, the default). No -ldflags surgery is
// required for the common case.
//
// The `Channel` var IS overridable via -ldflags so release pipelines can
// label nightly builds, tagged releases, etc.:
//
//	go build -ldflags "-X amplio/internal/version.Channel=nightly" ./cmd/amplio
//
// Local `go build` / `go run` / tests don't set Channel, so it stays at its
// default "dev" — which is the truthful label for an untagged local build.
package version

import (
	"runtime"
	"runtime/debug"
	"strings"
	"time"
)

// Channel labels the build provenance shown in --version output. Default
// "dev" covers local builds and tests; release pipelines override via
// -ldflags (e.g. "nightly" or "v0.1.0").
var Channel = "dev"

// Info captures the build identity of the running binary. Empty / zero
// fields mean the data wasn't available (e.g. tests don't populate VCS
// settings because they aren't built with `go build`).
type Info struct {
	Channel   string    // "dev" / "nightly" / "v0.1.0"
	Commit    string    // full git commit SHA, "" when unavailable
	Modified  bool      // true when built from a dirty worktree
	Time      time.Time // commit time (NOT build time), zero when unavailable
	GoVersion string    // e.g. "go1.26.2"
}

// Build returns the current binary's build identity by reading
// runtime/debug.ReadBuildInfo(). Cheap; safe to call repeatedly.
func Build() Info {
	out := Info{Channel: Channel, GoVersion: runtime.Version()}
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return out
	}
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			out.Commit = s.Value
		case "vcs.modified":
			out.Modified = s.Value == "true"
		case "vcs.time":
			out.Time, _ = time.Parse(time.RFC3339, s.Value)
		}
	}
	return out
}

// String formats the build identity as a single human-readable line for
// --version output. Components are dot-separated and omitted when empty,
// so a binary built without VCS info still produces a sensible
// "dev · go1.26.2" line instead of trailing separators.
//
// Example output (full):
//
//	dev · 8a3f2c1 · 2026-06-09T14:32:00Z · go1.26.2
//
// With a dirty worktree:
//
//	dev · 8a3f2c1-dirty · 2026-06-09T14:32:00Z · go1.26.2
func (i Info) String() string {
	parts := make([]string, 0, 4)
	if i.Channel != "" {
		parts = append(parts, i.Channel)
	}
	if i.Commit != "" {
		c := i.Commit
		if len(c) > 7 {
			c = c[:7]
		}
		if i.Modified {
			c += "-dirty"
		}
		parts = append(parts, c)
	}
	if !i.Time.IsZero() {
		parts = append(parts, i.Time.UTC().Format(time.RFC3339))
	}
	if i.GoVersion != "" {
		parts = append(parts, i.GoVersion)
	}
	return strings.Join(parts, " · ")
}

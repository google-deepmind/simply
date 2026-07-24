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

package version

import (
	"strings"
	"testing"
	"time"
)

func TestInfoString_AllFieldsPresent(t *testing.T) {
	i := Info{
		Channel:   "nightly",
		Commit:    "8a3f2c1d4e5f6a7b8c9d",
		Modified:  false,
		Time:      time.Date(2026, 6, 9, 14, 32, 0, 0, time.UTC),
		GoVersion: "go1.26.2",
	}
	got := i.String()
	want := "nightly · 8a3f2c1 · 2026-06-09T14:32:00Z · go1.26.2"
	if got != want {
		t.Errorf("String() = %q; want %q", got, want)
	}
}

func TestInfoString_DirtyMarker(t *testing.T) {
	i := Info{Channel: "dev", Commit: "8a3f2c1d", Modified: true, GoVersion: "go1.26.2"}
	got := i.String()
	if !strings.Contains(got, "8a3f2c1-dirty") {
		t.Errorf("String() = %q; want -dirty suffix on commit", got)
	}
}

func TestInfoString_MissingFieldsOmitted(t *testing.T) {
	// A binary built without VCS info (no -buildvcs) — verify we don't
	// produce empty fields or trailing separators.
	i := Info{Channel: "dev", GoVersion: "go1.26.2"}
	got := i.String()
	want := "dev · go1.26.2"
	if got != want {
		t.Errorf("String() = %q; want %q", got, want)
	}
}

func TestInfoString_EmptyInfo(t *testing.T) {
	// Defensive: a zero Info doesn't blow up or produce dangling separators.
	got := Info{}.String()
	if got != "" {
		t.Errorf("String() = %q; want \"\"", got)
	}
}

func TestBuild_ReturnsGoVersion(t *testing.T) {
	// Even without VCS info (tests don't have it), GoVersion is always set
	// from runtime.Version().
	if got := Build().GoVersion; got == "" {
		t.Error("Build().GoVersion = \"\"; want runtime.Version()")
	}
}

func TestBuild_DefaultsChannelToDev(t *testing.T) {
	// Channel defaults to "dev" until -ldflags overrides it. Test the
	// package-level default via Build().
	if got := Build().Channel; got != "dev" {
		t.Errorf("Build().Channel = %q; want \"dev\" (default; tests don't set -ldflags)", got)
	}
}

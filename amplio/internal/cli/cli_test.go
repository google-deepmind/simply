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

package cli

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

const absent = "amplio-definitely-not-a-real-binary-xyz"

func TestAvailable(t *testing.T) {
	if !(Tool{Name: "sh"}).Available() {
		t.Error("expected sh to be available")
	}
	if (Tool{Name: absent}).Available() {
		t.Error("expected a bogus command to be unavailable")
	}
	// An absolute path is checked directly (LookPath honors slash).
	if !(Tool{Name: "/bin/sh"}).Available() {
		t.Error("expected /bin/sh to be available via direct path check")
	}
}

func TestBindPaths(t *testing.T) {
	orig := os.Getenv("PATH")
	t.Cleanup(func() { _ = os.Setenv("PATH", orig) })

	BindPaths(nil)
	if os.Getenv("PATH") != orig {
		t.Error("BindPaths(nil) should be a no-op")
	}
	BindPaths([]string{""})
	if os.Getenv("PATH") != orig {
		t.Error("BindPaths with only empty entries should be a no-op")
	}

	dir := "/zzz-amplio-test-bin"
	BindPaths([]string{dir})
	got := os.Getenv("PATH")
	if !strings.HasPrefix(got, dir+string(os.PathListSeparator)) {
		t.Errorf("BindPaths should prepend %q; PATH=%q", dir, got)
	}
	if !strings.Contains(got, orig) {
		t.Error("BindPaths should preserve the existing PATH")
	}
}

func TestBootstrapSnippet(t *testing.T) {
	tools := []Tool{
		{Name: "sh", Snippet: "run a shell\n  sh -c 'echo hi'"},
		{Name: absent, Snippet: "should not appear"},
	}
	out := BootstrapSnippet(tools)
	if !strings.Contains(out, "**sh**") || !strings.Contains(out, "run a shell") {
		t.Errorf("expected available tool block; got:\n%s", out)
	}
	if strings.Contains(out, absent) || strings.Contains(out, "should not appear") {
		t.Errorf("unavailable tool leaked into snippet:\n%s", out)
	}

	if BootstrapSnippet([]Tool{{Name: absent, Snippet: "x"}}) != "" {
		t.Error("expected empty snippet when no tool is available")
	}
	if BootstrapSnippet(nil) != "" {
		t.Error("expected empty snippet for nil tools")
	}
}

func TestPrintStatus(t *testing.T) {
	var buf bytes.Buffer
	PrintStatus(&buf, []Tool{
		{Name: "sh"},
		{Name: absent, InstallHint: "install the thing"},
	})
	out := buf.String()

	if !strings.Contains(out, "✓ sh") || !strings.Contains(out, "available") {
		t.Errorf("expected available tool line; got:\n%s", out)
	}
	if !strings.Contains(out, "✗ "+absent) || !strings.Contains(out, "not found — install the thing") {
		t.Errorf("expected missing tool line with hint; got:\n%s", out)
	}
	// A bytes.Buffer is not a TTY, so no ANSI color should be emitted.
	if strings.Contains(out, "\x1b[") {
		t.Errorf("did not expect ANSI color for non-TTY writer; got:\n%q", out)
	}
}

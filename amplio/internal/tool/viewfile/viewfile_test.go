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

package viewfile

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func tmpFile(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil { //nolint:gosec
		t.Fatal(err)
	}
	return path
}

func TestViewFile_FullFile(t *testing.T) {
	path := tmpFile(t, "test.txt", "line1\nline2\nline3\n")
	tool := New(filepath.Dir(path), "")
	// show_anchors:false to exercise the plain line-number rendering (a small
	// file would otherwise auto-show anchors).
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`","show_anchors":false}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if !strings.Contains(result.Content, "line1") || !strings.Contains(result.Content, "line3") {
		t.Errorf("missing content: %s", result.Content)
	}
	if !strings.Contains(result.Content, "   1:") {
		t.Error("missing line numbers")
	}
}

func TestViewFile_LineRange(t *testing.T) {
	path := tmpFile(t, "test.txt", "a\nb\nc\nd\ne\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`","start_line":2,"end_line":4,"show_anchors":false}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if !strings.Contains(result.Content, "b") || !strings.Contains(result.Content, "d") {
		t.Errorf("missing lines 2-4: %s", result.Content)
	}
	if strings.Contains(result.Content, "   1: a") {
		t.Error("should not contain line 1")
	}
	if !strings.Contains(result.Content, "showing lines") {
		t.Error("expected range note in output")
	}
}

func TestViewFile_NotFound(t *testing.T) {
	tool := New("/tmp", "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"/nonexistent/file.txt"}`)
	if !result.IsError {
		t.Error("expected error for missing file")
	}
}

func TestViewFile_Directory(t *testing.T) {
	tool := New("/tmp", "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"/tmp"}`)
	if !result.IsError || !strings.Contains(result.Content, "directory") {
		t.Errorf("expected directory error: %s", result.Content)
	}
}

func TestViewFile_Binary(t *testing.T) {
	path := tmpFile(t, "test.bin", "hello\x00world")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`"}`)
	if !result.IsError || !strings.Contains(result.Content, "binary") {
		t.Errorf("expected binary error: %s", result.Content)
	}
}

func TestViewFile_Image(t *testing.T) {
	// Write a minimal PNG header.
	png := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
		0, 0, 0, 0, 0, 0, 0, 0}
	dir := t.TempDir()
	path := filepath.Join(dir, "test.png")
	os.WriteFile(path, png, 0o644) //nolint:gosec,errcheck
	tool := New(dir, "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"test.png"}`)
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content)
	}
	if len(result.Attachments) != 1 {
		t.Fatalf("expected 1 attachment, got %d", len(result.Attachments))
	}
	if !strings.HasPrefix(result.Attachments[0].MimeType, "image/") {
		t.Errorf("mime type: %s", result.Attachments[0].MimeType)
	}
	if len(result.Attachments[0].Data) == 0 {
		t.Error("expected raw attachment bytes in Data")
	}
}

func TestViewFile_AbsolutePath(t *testing.T) {
	path := tmpFile(t, "abs.txt", "absolute content\n")
	tool := New("/some/other/dir", "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+path+`"}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if !strings.Contains(result.Content, "absolute content") {
		t.Error("should read absolute path regardless of cwd")
	}
}

func TestViewFile_EmptyFile(t *testing.T) {
	path := tmpFile(t, "empty.txt", "")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`"}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
}

func TestViewFile_StartLineBeyondEnd(t *testing.T) {
	path := tmpFile(t, "short.txt", "one\ntwo\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`","start_line":100}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if !strings.Contains(result.Content, "only 2 line") {
		t.Errorf("expected note about line count: %s", result.Content)
	}
}

func TestViewFile_TooLarge(t *testing.T) {
	// A file over the read ceiling must be refused up front (not slurped into
	// memory), with a hint pointing at the range params. See M14.
	big := strings.Repeat("x\n", 5*1024*1024) // ~10MB, over maxReadBytes (8MB)
	path := tmpFile(t, "big.txt", big)
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`"}`)
	if !result.IsError {
		t.Fatal("expected error for oversized file")
	}
	if !strings.Contains(result.Content, "too large") || !strings.Contains(result.Content, "start_line") {
		t.Errorf("expected too-large error with range hint, got: %q", result.Content)
	}
}

// --- Auto anchor default (tri-state show_anchors) ---

// hasAnchors reports whether the rendered output is in anchor format (a
// file_fingerprint header + [L..  #..] line prefixes) rather than plain "  N:".
func hasAnchors(s string) bool {
	return strings.Contains(s, "file_fingerprint=") && strings.Contains(s, "] ")
}

func TestViewFile_AutoAnchorsOnSmallView(t *testing.T) {
	// A small file (<= autoAnchorMaxLines) with show_anchors omitted should
	// auto-show anchors.
	path := tmpFile(t, "small.go", "package x\n\nfunc A() {}\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{"path":"`+filepath.Base(path)+`"}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if !hasAnchors(result.Content) {
		t.Errorf("small view should auto-show anchors:\n%s", result.Content)
	}
}

func TestViewFile_AutoNoAnchorsOnLargeView(t *testing.T) {
	// A whole-file view larger than the threshold, show_anchors omitted, must
	// stay in plain line-number format.
	var sb strings.Builder
	for i := 0; i < autoAnchorMaxLines+10; i++ {
		sb.WriteString("line\n")
	}
	path := tmpFile(t, "big.txt", sb.String())
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{"path":"`+filepath.Base(path)+`"}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if hasAnchors(result.Content) {
		t.Errorf("large view should NOT auto-show anchors:\n%s", result.Content[:200])
	}
	if !strings.Contains(result.Content, "   1: ") {
		t.Errorf("large view should use plain line numbers:\n%s", result.Content[:200])
	}
}

func TestViewFile_AutoAnchorsOnSmallRangeOfLargeFile(t *testing.T) {
	// A small RANGE of a large file: the auto decision keys on rendered lines,
	// not total file size, so anchors should appear.
	var sb strings.Builder
	for i := 0; i < 500; i++ {
		sb.WriteString("line\n")
	}
	path := tmpFile(t, "huge.txt", sb.String())
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`","start_line":10,"end_line":20}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if !hasAnchors(result.Content) {
		t.Errorf("small range of large file should auto-show anchors:\n%s", result.Content)
	}
}

func TestViewFile_ExplicitFalseSuppressesOnSmallView(t *testing.T) {
	path := tmpFile(t, "small.go", "package x\n\nfunc A() {}\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`","show_anchors":false}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if hasAnchors(result.Content) {
		t.Errorf("explicit show_anchors=false must suppress anchors on a small view:\n%s", result.Content)
	}
}

func TestViewFile_ExplicitTrueForcesOnLargeView(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < autoAnchorMaxLines+10; i++ {
		sb.WriteString("line\n")
	}
	path := tmpFile(t, "big.txt", sb.String())
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(),
		`{"path":"`+filepath.Base(path)+`","show_anchors":true}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	if !hasAnchors(result.Content) {
		t.Errorf("explicit show_anchors=true must force anchors on a large view:\n%s", result.Content[:200])
	}
}

func TestViewFile_ImageAutoAnchorsNoError(t *testing.T) {
	// A 1x1 PNG. With show_anchors omitted (auto), an image must render normally
	// (no anchor error — that's only for an EXPLICIT true).
	png := []byte{
		0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 'I', 'H', 'D', 'R',
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89,
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "px.png")
	if err := os.WriteFile(path, png, 0o644); err != nil { //nolint:gosec
		t.Fatal(err)
	}
	tool := New(dir, "")
	result := tool.ParseAndExecute(context.Background(), `{"path":"px.png"}`)
	if result.IsError {
		t.Fatalf("image with auto anchors should not error: %s", result.Content)
	}
}

// resolvePath: OSS build (citc stub returns ok=false), so `//` paths are NOT
// specially handled — they fall through as already-absolute and are cleaned by
// filepath. Relative paths join onto cwd; absolute paths pass through; a leading
// $AMPLIO_ARTIFACT_DIR expands to the run's artifact dir.
func TestResolvePath(t *testing.T) {
	const art = "/data/artifacts/run7"
	cases := []struct {
		cwd, artifactDir, in, want string
	}{
		{"/work/dir", art, "sub/f.txt", "/work/dir/sub/f.txt"},
		{"/work/dir", art, "/abs/f.txt", "/abs/f.txt"},
		// `//foo` is IsAbs on Linux; with the OSS citc stub it isn't rewritten, so
		// it passes through unchanged (not joined onto cwd).
		{"/work/dir", art, "//foo/bar", "//foo/bar"},
		// $AMPLIO_ARTIFACT_DIR prefix expands (both bare and braced forms).
		{"/work/dir", art, "$AMPLIO_ARTIFACT_DIR/plan.md", art + "/plan.md"},
		{"/work/dir", art, "${AMPLIO_ARTIFACT_DIR}/plan.md", art + "/plan.md"},
		{"/work/dir", art, "$AMPLIO_ARTIFACT_DIR", art},
		// Only a PREFIX match counts — mid-path or unrelated vars are untouched.
		{"/work/dir", art, "sub/$AMPLIO_ARTIFACT_DIR/x", "/work/dir/sub/$AMPLIO_ARTIFACT_DIR/x"},
		// No run context (artifactDir=="") leaves the token literal (then joined).
		{"/work/dir", "", "$AMPLIO_ARTIFACT_DIR/x", "/work/dir/$AMPLIO_ARTIFACT_DIR/x"},
	}
	for _, tc := range cases {
		if got := resolvePath(tc.cwd, tc.artifactDir, tc.in); got != tc.want {
			t.Errorf("resolvePath(%q, %q, %q) = %q, want %q", tc.cwd, tc.artifactDir, tc.in, got, tc.want)
		}
	}
}

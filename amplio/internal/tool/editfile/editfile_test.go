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

package editfile

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

func readFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestEditFile_SingleReplace(t *testing.T) {
	path := tmpFile(t, "test.go", "func main() {\n\tfmt.Println(\"hello\")\n}\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "hello", "new_text": "world"}]
	}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	got := readFile(t, path)
	if !strings.Contains(got, "world") || strings.Contains(got, "hello") {
		t.Errorf("content: %s", got)
	}
}

func TestEditFile_MultipleEdits(t *testing.T) {
	path := tmpFile(t, "test.txt", "aaa\nbbb\nccc\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [
			{"old_text": "aaa", "new_text": "AAA"},
			{"old_text": "ccc", "new_text": "CCC"}
		]
	}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	got := readFile(t, path)
	if got != "AAA\nbbb\nCCC\n" {
		t.Errorf("content: %q", got)
	}
}

func TestEditFile_CreateFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "new.txt")
	tool := New(dir, "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "new.txt",
		"edits": [{"old_text": "", "new_text": "new content\n"}]
	}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	got := readFile(t, path)
	if got != "new content\n" {
		t.Errorf("content: %q", got)
	}
}

func TestEditFile_CreateExistingFails(t *testing.T) {
	path := tmpFile(t, "exists.txt", "already here")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "", "new_text": "overwrite"}]
	}`)
	if !result.IsError || !strings.Contains(result.Content, "already exists") {
		t.Errorf("expected 'already exists' error: %s", result.Content)
	}
}

func TestEditFile_NotFound(t *testing.T) {
	tool := New(t.TempDir(), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "nonexistent.txt",
		"edits": [{"old_text": "foo", "new_text": "bar"}]
	}`)
	if !result.IsError {
		t.Error("expected error for missing file")
	}
}

func TestEditFile_OldTextNotFound(t *testing.T) {
	path := tmpFile(t, "test.txt", "hello world")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "not here", "new_text": "bar"}]
	}`)
	if !result.IsError || !strings.Contains(result.Content, "not found") {
		t.Errorf("expected 'not found' error: %s", result.Content)
	}
}

func TestEditFile_MultipleMatches(t *testing.T) {
	path := tmpFile(t, "test.txt", "aaa\naaa\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "aaa", "new_text": "bbb"}]
	}`)
	if !result.IsError || !strings.Contains(result.Content, "multiple_matches") {
		t.Errorf("expected 'multiple_matches' error: %s", result.Content)
	}
}

// TestEditFile_SelfOverlappingCount guards the non-overlapping match count:
// "aa" in "aaaa" is 2 occurrences, not 3. The find loop must advance past the
// whole match (offset += len(old_text)), or expected_match_count is inflated and
// the overlap check spuriously trips. With expected_match_count=2 both should be
// replaced cleanly.
func TestEditFile_SelfOverlappingCount(t *testing.T) {
	path := tmpFile(t, "test.txt", "aaaa")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "aa", "new_text": "X", "expected_match_count": 2}]
	}`)
	if result.IsError {
		t.Fatalf("unexpected error (overlap count regression?): %s", result.Content)
	}
	if got := readFile(t, path); got != "XX" {
		t.Errorf("content = %q, want %q", got, "XX")
	}
}

func TestEditFile_NoOp(t *testing.T) {
	path := tmpFile(t, "test.txt", "same")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "same", "new_text": "same"}]
	}`)
	if !result.IsError || !strings.Contains(result.Content, "no_op") {
		t.Errorf("expected no_op error: %s", result.Content)
	}
}

func TestEditFile_Overlap(t *testing.T) {
	path := tmpFile(t, "test.txt", "abcdef")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [
			{"old_text": "abcd", "new_text": "ABCD"},
			{"old_text": "cdef", "new_text": "CDEF"}
		]
	}`)
	if !result.IsError || !strings.Contains(result.Content, "overlap") {
		t.Errorf("expected overlap error: %s", result.Content)
	}
}

func TestEditFile_CreateSubdirectory(t *testing.T) {
	dir := t.TempDir()
	tool := New(dir, "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "sub/dir/new.txt",
		"edits": [{"old_text": "", "new_text": "deep file\n"}]
	}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	got := readFile(t, filepath.Join(dir, "sub", "dir", "new.txt"))
	if got != "deep file\n" {
		t.Errorf("content: %q", got)
	}
}

func TestEditFile_EndBeforeStart(t *testing.T) {
	// end_line < start_line must not panic the byte-range slicing.
	path := tmpFile(t, "test.txt", "aaa\nbbb\nccc\n")
	tool := New(filepath.Dir(path), "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "bbb", "new_text": "BBB", "start_line": 3, "end_line": 1}]
	}`)
	if !result.IsError {
		t.Fatal("expected error for end_line < start_line")
	}
	if !strings.Contains(result.Content, "before start_line") {
		t.Errorf("content: %q", result.Content)
	}
}

func TestEditFile_PreservesMode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "script.sh")
	if err := os.WriteFile(path, []byte("echo hello\n"), 0o700); err != nil { //nolint:gosec
		t.Fatal(err)
	}
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	tool := New(dir, "")
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "script.sh",
		"edits": [{"old_text": "hello", "new_text": "world"}]
	}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Mode().Perm() != before.Mode().Perm() {
		t.Errorf("edit changed file mode: before=%o after=%o", before.Mode().Perm(), after.Mode().Perm())
	}
}

// TestEditFile_CRLFLineRange exercises the line-range → byte-range math on a
// CRLF file (P3): the window code splits on "\n" and sums len(line)+1, where the
// trailing "\r" stays inside len(line). A start_line/end_line edit on such a
// file must still target the right bytes and preserve the surrounding CRLFs.
func TestEditFile_CRLFLineRange(t *testing.T) {
	path := tmpFile(t, "crlf.txt", "alpha\r\nbeta\r\ngamma\r\ndelta\r\n")
	tool := New(filepath.Dir(path), "")
	// Replace "beta" but only within lines 2..2, so the window math is in play.
	result := tool.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": [{"old_text": "beta", "new_text": "BETA", "start_line": 2, "end_line": 2}]
	}`)
	if result.IsError {
		t.Fatal(result.Content)
	}
	got := readFile(t, path)
	want := "alpha\r\nBETA\r\ngamma\r\ndelta\r\n"
	if got != want {
		t.Errorf("CRLF range edit:\n got %q\nwant %q", got, want)
	}
}

// --- unifiedDiff helper ---

func TestUnifiedDiff_AlignsContext(t *testing.T) {
	// A multi-line change where one line is unchanged: it must appear as context
	// ("  "), not as a remove+add pair.
	got := unifiedDiff("line two\nline three", "line two CHANGED\nbrand new line\nline three", 60)
	want := "- line two\n+ line two CHANGED\n+ brand new line\n  line three\n"
	if got != want {
		t.Errorf("unifiedDiff mismatch:\n got %q\nwant %q", got, want)
	}
}

func TestUnifiedDiff_PureAddAndDelete(t *testing.T) {
	if got := unifiedDiff("", "a\nb", 60); got != "+ a\n+ b\n" {
		t.Errorf("pure add: got %q", got)
	}
	if got := unifiedDiff("a\nb", "", 60); got != "- a\n- b\n" {
		t.Errorf("pure delete: got %q", got)
	}
}

func TestUnifiedDiff_TrailingNewlineNoSpuriousLine(t *testing.T) {
	// "a\n" is one line, not [a, ""]; the diff must not emit an empty added line.
	if got := unifiedDiff("a\n", "a\n", 60); got != "  a\n" {
		t.Errorf("trailing newline: got %q", got)
	}
}

func TestUnifiedDiff_LargeEditSkipsAlignment(t *testing.T) {
	removed := strings.Repeat("x\n", 5)
	added := strings.Repeat("y\n", 5)
	got := unifiedDiff(removed, added, 3)
	if !strings.Contains(got, "diff alignment skipped") {
		t.Errorf("large edit should note skipped alignment; got:\n%s", got)
	}
	if !strings.Contains(got, "- x") || !strings.Contains(got, "+ y") {
		t.Errorf("large edit should still emit prefixed lines; got:\n%s", got)
	}
}

func TestUnifiedDiff_LongLineTruncated(t *testing.T) {
	long := strings.Repeat("z", 500)
	got := unifiedDiff("", long, 60)
	if !strings.Contains(got, "…") {
		t.Errorf("a >300-rune line should be truncated with an ellipsis; got len=%d", len(got))
	}
}

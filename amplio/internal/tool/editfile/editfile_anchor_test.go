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
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"amplio/internal/tool/anchor"
)

// anchorsOf returns the per-line anchor hashes for src.
func anchorsOf(src string) []string {
	lines, _ := anchor.SplitTextForAnchors(src)
	return anchor.ComputeAnchors(lines)
}

// runEdits executes a raw JSON edits payload against a temp file holding src and
// returns (result, newContent).
func runEdits(t *testing.T, src, editsJSON string) (string, string) {
	t.Helper()
	path := tmpFile(t, "f.txt", src)
	tl := New(filepath.Dir(path), "")
	res := tl.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"edits": `+editsJSON+`
	}`)
	return res.Content, readFile(t, path)
}

func TestAnchor_ReplaceSingleLine(t *testing.T) {
	src := "alpha\nbeta\ngamma\n"
	a := anchorsOf(src)
	edits := fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","new_text":"BETA"}]`, a[1])
	_, got := runEdits(t, src, edits)
	want := "alpha\nBETA\ngamma\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_ReplaceRange(t *testing.T) {
	src := "alpha\nbeta\ngamma\ndelta\n"
	a := anchorsOf(src)
	edits := fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","at_to":"#%s","new_text":"X\nY"}]`, a[1], a[2])
	_, got := runEdits(t, src, edits)
	want := "alpha\nX\nY\ndelta\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_DeleteSingleLine(t *testing.T) {
	src := "alpha\nbeta\ngamma\n"
	a := anchorsOf(src)
	edits := fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","new_text":""}]`, a[1])
	_, got := runEdits(t, src, edits)
	want := "alpha\ngamma\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_InsertBefore(t *testing.T) {
	src := "first\nsecond\n"
	a := anchorsOf(src)
	edits := fmt.Sprintf(`[{"kind":"insert_before","at_from":"#%s","new_text":"middle"}]`, a[1])
	_, got := runEdits(t, src, edits)
	want := "first\nmiddle\nsecond\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_InsertAfter(t *testing.T) {
	src := "first\nsecond\n"
	a := anchorsOf(src)
	edits := fmt.Sprintf(`[{"kind":"insert_after","at_from":"#%s","new_text":"middle"}]`, a[0])
	_, got := runEdits(t, src, edits)
	want := "first\nmiddle\nsecond\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_InsertBeforeBOFPrepends(t *testing.T) {
	src := "a\nb\n"
	edits := `[{"kind":"insert_before","at_from":"#BOF","new_text":"top"}]`
	_, got := runEdits(t, src, edits)
	want := "top\na\nb\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_InsertEOFAppends(t *testing.T) {
	src := "a\nb\n"
	edits := `[{"kind":"insert_before","at_from":"#EOF","new_text":"bottom"}]`
	_, got := runEdits(t, src, edits)
	want := "a\nb\nbottom\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_InsertEOFNoTrailingNewlinePreserved(t *testing.T) {
	// File without a final newline: appending at EOF should not add one.
	src := "a\nb"
	edits := `[{"kind":"insert_after","at_from":"#EOF","new_text":"c"}]`
	_, got := runEdits(t, src, edits)
	want := "a\nbc"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_MultiEditLineShiftSafe(t *testing.T) {
	src := "L1\nL2\nL3\nL4\nL5\n"
	a := anchorsOf(src)
	// Delete L2, replace L4 with two lines, insert after L5.
	edits := fmt.Sprintf(`[
		{"kind":"replace","at_from":"#%s","new_text":""},
		{"kind":"replace","at_from":"#%s","new_text":"L4a\nL4b"},
		{"kind":"insert_after","at_from":"#%s","new_text":"L6"}
	]`, a[1], a[3], a[4])
	_, got := runEdits(t, src, edits)
	want := "L1\nL3\nL4a\nL4b\nL5\nL6\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_DisambiguationWithAtN(t *testing.T) {
	// Seven identical interior lines give colliding 5-line windows at indices
	// 3,4,5 (each [m,m,m,m,m]); @N disambiguates.
	src := "a\nm\nm\nm\nm\nm\nm\nm\nb\n"
	a := anchorsOf(src)
	if a[3] != a[4] || a[4] != a[5] {
		t.Fatalf("expected colliding anchors for identical interior lines: %v", a)
	}
	// @4 (1-based) selects index 3.
	edits := fmt.Sprintf(`[{"kind":"replace","at_from":"#%s@4","new_text":"M"}]`, a[3])
	res, got := runEdits(t, src, edits)
	if strings.HasPrefix(res, "Error") {
		t.Fatalf("unexpected error: %s", res)
	}
	want := "a\nm\nm\nM\nm\nm\nm\nm\nb\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestAnchor_MixedStrAndAnchor(t *testing.T) {
	src := "func add(a, b int) int {\n    return a + b\n}\n"
	a := anchorsOf(src)
	edits := fmt.Sprintf(`[
		{"old_text":"add","new_text":"sum"},
		{"kind":"insert_after","at_from":"#%s","new_text":"    // sum"}
	]`, a[1])
	_, got := runEdits(t, src, edits)
	want := "func sum(a, b int) int {\n    return a + b\n    // sum\n}\n"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

// --- Failure-mode parity ---

func wantErrPrefix(t *testing.T, res, code string) {
	t.Helper()
	if !strings.HasPrefix(res, "Error: "+code+":") {
		t.Errorf("expected error code %q, got: %s", code, res)
	}
}

func TestAnchor_TargetNotFoundUnknownAnchor(t *testing.T) {
	res, _ := runEdits(t, "a\nb\n", `[{"kind":"replace","at_from":"#000000","new_text":"X"}]`)
	wantErrPrefix(t, res, "target_not_found")
}

func TestAnchor_StaleAnchorAmbiguous(t *testing.T) {
	src := "a\nm\nm\nm\nm\nm\nm\nm\nb\n"
	a := anchorsOf(src)
	if a[3] != a[4] {
		t.Fatalf("need colliding interior anchors: %v", a)
	}
	// Ambiguous anchor without @N -> stale_anchor.
	res, _ := runEdits(t, src, fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","new_text":"X"}]`, a[3]))
	wantErrPrefix(t, res, "stale_anchor")
}

func TestAnchor_StaleAnchorAtNMisalignment(t *testing.T) {
	src := "a\nb\nc\n"
	a := anchorsOf(src)
	res, _ := runEdits(t, src, fmt.Sprintf(`[{"kind":"replace","at_from":"#%s@99","new_text":"X"}]`, a[1]))
	wantErrPrefix(t, res, "stale_anchor")
}

func TestAnchor_MalformedAddress(t *testing.T) {
	res, _ := runEdits(t, "x\n", `[{"kind":"replace","at_from":"#xyz","new_text":"Y"}]`)
	wantErrPrefix(t, res, "malformed_args")
}

func TestAnchor_SentinelInReplaceRejected(t *testing.T) {
	res, _ := runEdits(t, "x\n", `[{"kind":"replace","at_from":"#BOF","new_text":"Y"}]`)
	wantErrPrefix(t, res, "malformed_args")
}

func TestAnchor_FromAfterToRejected(t *testing.T) {
	src := "a\nb\nc\n"
	a := anchorsOf(src)
	res, _ := runEdits(t, src, fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","at_to":"#%s","new_text":"X"}]`, a[2], a[0]))
	wantErrPrefix(t, res, "malformed_args")
}

func TestAnchor_TwoInsertsSamePositionOverlap(t *testing.T) {
	src := "a\nb\n"
	a := anchorsOf(src)
	res, _ := runEdits(t, src, fmt.Sprintf(`[
		{"kind":"insert_before","at_from":"#%s","new_text":"X"},
		{"kind":"insert_before","at_from":"#%s","new_text":"Y"}
	]`, a[1], a[1]))
	wantErrPrefix(t, res, "edit_overlap")
	// A batch-abort failure must spell out that NOTHING was applied, so a caller
	// reading only the error doesn't assume the other edits in the batch landed.
	if !strings.Contains(res, "file is unchanged") {
		t.Errorf("overlap error should note the batch was abandoned: %q", res)
	}
}

func TestAnchor_EditAgainstEmptyFileTargetNotFound(t *testing.T) {
	res, _ := runEdits(t, "", `[{"kind":"replace","at_from":"#000000","new_text":"X"}]`)
	wantErrPrefix(t, res, "target_not_found")
}

func TestAnchor_ExpectedFileHashEnforced(t *testing.T) {
	src := "hello\n"
	a := anchorsOf(src)
	path := tmpFile(t, "f.txt", src)
	tl := New(filepath.Dir(path), "")
	res := tl.ParseAndExecute(context.Background(), `{
		"path": "`+filepath.Base(path)+`",
		"expected_file_hash": "000000000000",
		"edits": [{"kind":"replace","at_from":"#`+a[0]+`","new_text":"goodbye"}]
	}`)
	wantErrPrefix(t, res.Content, "expected_hash_mismatch")
}

// --- Success echo (#1) and failure hint (#2) ---

func TestAnchor_SuccessEchoReportsLinesAndRemovedContent(t *testing.T) {
	src := "alpha\nbeta\ngamma\ndelta\n"
	a := anchorsOf(src)
	// replace lines 2-3 (beta..gamma); the echo should name the span and show
	// the removed content even though the address was a #hash.
	res, _ := runEdits(t, src, fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","at_to":"#%s","new_text":"B"}]`, a[1], a[2]))
	if strings.HasPrefix(res, "Error") {
		t.Fatalf("unexpected error: %s", res)
	}
	if !strings.Contains(res, "replace lines 2-3") {
		t.Errorf("echo should report the resolved line span; got:\n%s", res)
	}
	// Both removed lines must carry the "- " prefix (the old format marked only
	// the first line of a multi-line change).
	if !strings.Contains(res, "- beta") || !strings.Contains(res, "- gamma") {
		t.Errorf("echo should show every removed line with a - prefix; got:\n%s", res)
	}
}

func TestAnchor_SuccessEchoSingleLineReplace(t *testing.T) {
	src := "alpha\nbeta\ngamma\n"
	a := anchorsOf(src)
	res, _ := runEdits(t, src, fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","new_text":"BETA"}]`, a[1]))
	if !strings.Contains(res, "replace line 2") {
		t.Errorf("single-line replace should report 'line 2'; got:\n%s", res)
	}
}

func TestAnchor_SuccessEchoInsertReportsLine(t *testing.T) {
	src := "first\nsecond\n"
	a := anchorsOf(src)
	res, _ := runEdits(t, src, fmt.Sprintf(`[{"kind":"insert_after","at_from":"#%s","new_text":"mid"}]`, a[0]))
	if !strings.Contains(res, "insert_after at line 2") {
		t.Errorf("insert echo should report the insertion line; got:\n%s", res)
	}
}

func TestStrEcho_ReportsLine(t *testing.T) {
	src := "alpha\nbeta\ngamma\n"
	res, _ := runEdits(t, src, `[{"old_text":"beta","new_text":"BETA"}]`)
	if !strings.Contains(res, "Edit #1 line 2") {
		t.Errorf("str edit echo should report the matched line; got:\n%s", res)
	}
}

func TestAnchor_FailureHintSuggestsReview(t *testing.T) {
	res, _ := runEdits(t, "a\nb\n", `[{"kind":"replace","at_from":"#000000","new_text":"X"}]`)
	if !strings.Contains(res, "view_file(show_anchors=true)") {
		t.Errorf("resolve failure should hint to re-view; got: %s", res)
	}
}

// --- expect_at_from / expect_at_to intent guards ---

// The headline case: a valid anchor that resolves to ONE line, but not the
// intended one. expect_at_from catches it where @N never could.
func TestExpect_AtFromMismatchRejects(t *testing.T) {
	src := "alpha\nbeta\ngamma\n"
	a := anchorsOf(src)
	// Address line 2 (beta) but assert it contains "gamma" — a mismatch.
	res, got := runEdits(t, src,
		fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","expect_at_from":"gamma","new_text":"X"}]`, a[1]))
	wantErrPrefix(t, res, "expect_mismatch")
	if !strings.Contains(res, `"beta"`) {
		t.Errorf("error should echo the actual resolved line; got: %s", res)
	}
	if got != src {
		t.Errorf("file must be unchanged on mismatch; got %q", got)
	}
}

func TestExpect_AtFromMatchApplies(t *testing.T) {
	src := "alpha\nbeta\ngamma\n"
	a := anchorsOf(src)
	_, got := runEdits(t, src,
		fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","expect_at_from":"beta","new_text":"BETA"}]`, a[1]))
	if want := "alpha\nBETA\ngamma\n"; got != want {
		t.Errorf("matching expect should apply; got %q want %q", got, want)
	}
}

// The range-clip case (my mistake #2): at_to landed on the wrong line.
func TestExpect_AtToMismatchRejects(t *testing.T) {
	src := "alpha\nbeta\ngamma\ndelta\n"
	a := anchorsOf(src)
	// Replace lines 2-3 but assert the end line is "delta" (it's gamma).
	res, got := runEdits(t, src,
		fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","at_to":"#%s","expect_at_to":"delta","new_text":"X"}]`, a[1], a[2]))
	wantErrPrefix(t, res, "expect_mismatch")
	if got != src {
		t.Errorf("file must be unchanged on mismatch; got %q", got)
	}
}

func TestExpect_AtToMatchApplies(t *testing.T) {
	src := "alpha\nbeta\ngamma\ndelta\n"
	a := anchorsOf(src)
	_, got := runEdits(t, src,
		fmt.Sprintf(`[{"kind":"replace","at_from":"#%s","at_to":"#%s","expect_at_from":"beta","expect_at_to":"gamma","new_text":"X"}]`, a[1], a[2]))
	if want := "alpha\nX\ndelta\n"; got != want {
		t.Errorf("matching expects should apply; got %q want %q", got, want)
	}
}

func TestExpect_InsertGuard(t *testing.T) {
	src := "first\nsecond\n"
	a := anchorsOf(src)
	// insert_after the line containing "first" — guard matches, so it applies.
	_, got := runEdits(t, src,
		fmt.Sprintf(`[{"kind":"insert_after","at_from":"#%s","expect_at_from":"first","new_text":"mid"}]`, a[0]))
	if want := "first\nmid\nsecond\n"; got != want {
		t.Errorf("matching insert guard should apply; got %q want %q", got, want)
	}
	// Mismatch rejects.
	res, _ := runEdits(t, src,
		fmt.Sprintf(`[{"kind":"insert_after","at_from":"#%s","expect_at_from":"second","new_text":"mid"}]`, a[0]))
	wantErrPrefix(t, res, "expect_mismatch")
}

// expect_at_to is meaningless for inserts (no at_to) — reject as malformed.
func TestExpect_AtToOnInsertRejected(t *testing.T) {
	src := "first\nsecond\n"
	a := anchorsOf(src)
	res, _ := runEdits(t, src,
		fmt.Sprintf(`[{"kind":"insert_after","at_from":"#%s","expect_at_to":"x","new_text":"mid"}]`, a[0]))
	wantErrPrefix(t, res, "malformed_args")
}

// The guards are anchor-only: a str_replace already self-verifies via old_text.
func TestExpect_OnStrReplaceRejected(t *testing.T) {
	res, _ := runEdits(t, "alpha\nbeta\n",
		`[{"old_text":"beta","new_text":"BETA","expect_at_from":"beta"}]`)
	wantErrPrefix(t, res, "malformed_args")
}

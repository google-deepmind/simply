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

// Package editfile provides the file editing tool.
//
// Supports three edit kinds, mirroring the upstream edit_file tool:
//
//   - str_replace (kind "" / "str_replace"): find a literal old_text in the
//     search window and swap it for new_text. old_text="" creates a new file.
//   - replace (kind "replace"): replace the inclusive line range addressed by
//     content-hash anchors [at_from..at_to] with new_text (new_text="" deletes).
//   - insert_before / insert_after (kind "insert_before"|"insert_after"): splice
//     new_text before/after the anchored line. #BOF/#EOF sentinels prepend/append.
//
// Anchors come from view_file(show_anchors=true). Edits are atomic (temp file +
// rename). Multiple edits per call are resolved against the pre-call snapshot,
// overlap-checked, then applied bottom-up so earlier byte offsets stay valid.
package editfile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"amplio/internal/tool"
	"amplio/internal/tool/anchor"
	"amplio/internal/workspace/citc"
)

type Params struct {
	Path             string `json:"path" jsonschema:"required" jsonschema_description:"File to edit (relative to workspace or absolute)"`
	Edits            []Edit `json:"edits" jsonschema:"required" jsonschema_description:"List of edits to apply atomically (max 64)"`
	ExpectedFileHash string `json:"expected_file_hash,omitempty" jsonschema_description:"Expected file_fingerprint from view_file(show_anchors=true). Edit is rejected if the file has changed."`
	ValidateOnly     bool   `json:"validate_only,omitempty" jsonschema_description:"Dry run: resolve all edits and report success/error without writing."`
}

// Edit is a single edit operation. The Kind field discriminates between the
// str-replace path (default) and the anchor-addressed paths. Go's tool schema
// is a flat struct rather than a JSON discriminated union, so fields not used by
// a given kind are simply omitted from the call.
type Edit struct {
	Kind string `json:"kind,omitempty" jsonschema_description:"Edit kind: '' or 'str_replace' (find/replace old_text); 'replace' (replace anchor line range at_from..at_to); 'insert_before'/'insert_after' (splice new_text around anchored line)."`

	// str_replace fields.
	OldText            string `json:"old_text,omitempty" jsonschema_description:"str_replace: exact text to find. Empty string creates a new file with new_text."`
	NewText            string `json:"new_text" jsonschema:"required" jsonschema_description:"Replacement/inserted text. Empty string deletes the match (str_replace/replace)."`
	ExpectedMatchCount int    `json:"expected_match_count,omitempty" jsonschema_description:"str_replace: required number of occurrences (default 1). Set N>1 for multi-site rename."`
	StartLine          *int   `json:"start_line,omitempty" jsonschema_description:"str_replace: 1-indexed inclusive lower bound to narrow the search window."`
	EndLine            *int   `json:"end_line,omitempty" jsonschema_description:"str_replace: 1-indexed inclusive upper bound to narrow the search window."`

	// Anchor fields (replace / insert_before / insert_after).
	AtFrom string `json:"at_from,omitempty" jsonschema_description:"anchor edits: content-hash anchor of the target line from view_file(show_anchors=true), e.g. #a3f1c0. Ambiguous anchors matching several lines are detected and reported with a clear error — use line number @N (#a3f1c0@42) to break the collision (prefer w/o @N unless collision happens). #BOF/#EOF sentinels for inserts only."`
	AtTo   string `json:"at_to,omitempty" jsonschema_description:"replace: content-hash anchor of the LAST line to replace (inclusive). Defaults to at_from (single-line replace)."`

	// Optional intent guards for anchor edits: a verbatim substring the resolved
	// line must contain. They catch a valid-but-misremembered anchor (the hash
	// resolves to ONE line, just not the one you meant) — the failure mode @N
	// can't catch. expect_at_from guards the at_from line; expect_at_to the at_to
	// line. Substring match against the trailing-whitespace-normalized line.
	ExpectAtFrom string `json:"expect_at_from,omitempty" jsonschema_description:"anchor edits: optional verbatim substring the at_from line must contain. Rejects the edit if the resolved line doesn't contain it — guards against a stale/misremembered #hash that resolves to the wrong line."`
	ExpectAtTo   string `json:"expect_at_to,omitempty" jsonschema_description:"replace: optional verbatim substring the at_to line must contain (same intent guard as expect_at_from, for the range end)."`
}

// envArtifactDir is the env-var name (literal, not imported from config to keep
// this tool config-free) that a path may lead with; resolvePath expands it to
// the run's artifact dir.
const envArtifactDir = "AMPLIO_ARTIFACT_DIR"

// New builds the edit_file tool. cwd anchors relative paths; artifactDir is the
// run's scratch dir, used to expand a leading $AMPLIO_ARTIFACT_DIR in paths.
func New(cwd, artifactDir string) *tool.Tool {
	return &tool.Tool{
		Name: "edit_file",
		Description: fmt.Sprintf("Edit a file. On success the result echoes a unified diff of the applied change. "+
			"Two ways to address an edit:\n"+
			"  str_replace: {old_text,new_text} — find literal old_text, swap in new_text "+
			"(old_text='' creates a file; new_text='' deletes). Best when the target text is unique.\n"+
			"  anchor edits: address lines by #hash from view_file(show_anchors=true). Each #hash "+
			fmt.Sprintf("is a CONTENT fingerprint of the line plus its %d neighbor line(s) on each side. "+
				"It changes only when that line or one of those neighbors changes — it is stable ", anchor.WindowRadius)+
			"across line-number shifts from earlier edits, so a batch "+
			"of anchor edits in one call stays valid even as they move each other around. "+
			"Kinds: {kind:'replace',at_from,at_to,new_text} replaces the inclusive line "+
			"range (new_text='' deletes); {kind:'insert_before'|'insert_after',at_from,new_text} splices "+
			"around the anchored line (#BOF/#EOF prepend/append). Use view_file(show_anchors=true) "+
			"first to read current #hashes. To self-verify an anchor edit (recommended when "+
			"editing from #hashes you didn't just view), add expect_at_from (and expect_at_to for a "+
			"range): a substring the resolved line must contain — the edit is rejected if the anchor "+
			"lands on the wrong line.\nCWD=%q.", cwd),
		ParamType: &Params{},
		Execute:   makeExecutor(cwd, artifactDir),
	}
}

// isStrKind reports whether an edit uses the literal find-and-replace path.
// The empty kind is accepted for backward compatibility with callers that omit
// the discriminator entirely.
func isStrKind(kind string) bool {
	return kind == "" || kind == "str_replace" || kind == "str"
}

func makeExecutor(cwd, artifactDir string) tool.Executor {
	return func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
		params, errResult := tool.ParseArgs[Params](args)
		if errResult != nil {
			return errResult, nil
		}
		return execute(cwd, artifactDir, params)
	}
}

// resolvePath turns a tool-supplied path into an absolute filesystem path.
// Precedence: expand a leading $AMPLIO_ARTIFACT_DIR (the one env var we document
// to agents) → resolve a `//` repo-root-relative path (CitC) → join a relative
// path onto cwd → absolute paths pass through. Only that single variable is
// expanded (not a general ExpandEnv), so no other server env leaks into paths.
func resolvePath(cwd, artifactDir, path string) string {
	path = expandArtifactDir(artifactDir, path)
	if abs, ok := citc.ResolveRootPath(cwd, path); ok {
		return abs
	}
	if !filepath.IsAbs(path) {
		return filepath.Join(cwd, path)
	}
	return path
}

// expandArtifactDir replaces a leading $AMPLIO_ARTIFACT_DIR or
// ${AMPLIO_ARTIFACT_DIR} token with the run's artifact dir. Only a prefix match
// counts (the variable names a directory); artifactDir=="" leaves path untouched.
func expandArtifactDir(artifactDir, path string) string {
	if artifactDir == "" {
		return path
	}
	for _, tok := range []string{"${" + envArtifactDir + "}", "$" + envArtifactDir} {
		if rest, ok := strings.CutPrefix(path, tok); ok {
			return artifactDir + rest
		}
	}
	return path
}

func execute(cwd, artifactDir string, params *Params) (*tool.Result, error) {
	if len(params.Edits) == 0 {
		return errResult("malformed_args", "no edits provided"), nil
	}
	if len(params.Edits) > 64 {
		return errResult("malformed_args", "too many edits (max 64)"), nil
	}

	path := resolvePath(cwd, artifactDir, params.Path)

	// Handle file creation (str_replace with old_text == "" for the single
	// edit). Anchor edits also have an empty old_text, so gate on the kind.
	if len(params.Edits) == 1 && isStrKind(params.Edits[0].Kind) && params.Edits[0].OldText == "" {
		if params.ValidateOnly {
			return &tool.Result{Content: "Validation passed: would create file."}, nil
		}
		return createFile(path, params.Edits[0].NewText)
	}

	// Read existing file.
	data, err := os.ReadFile(path)
	if err != nil {
		return errResult("file_not_found", err.Error()), nil
	}

	// Binary detection.
	probe := data
	if len(probe) > 8192 {
		probe = probe[:8192]
	}
	for _, b := range probe {
		if b == 0 {
			return errResult("binary_file", fmt.Sprintf("%s appears to be a binary file", path)), nil
		}
	}

	content := string(data)

	// Optional ETag check.
	if params.ExpectedFileHash != "" {
		lines, hadTrailing := anchor.SplitTextForAnchors(content)
		fp := anchor.FileFingerprint(lines, hadTrailing)
		if fp != params.ExpectedFileHash {
			return errResult("expected_hash_mismatch",
				fmt.Sprintf("expected %s, got %s; file changed since last view",
					params.ExpectedFileHash, fp)), nil
		}
	}

	// Resolve all edits before mutating. Anchor edits resolve against the
	// snapshot's per-line anchors, computed lazily on first use so the common
	// str-replace-only path doesn't pay for the hashing.
	var resolved []resolvedEdit
	var snapshotLines []string
	var snapshotAnchors []string
	anchorsReady := false
	ensureAnchors := func() {
		if !anchorsReady {
			snapshotLines, _ = anchor.SplitTextForAnchors(content)
			snapshotAnchors = anchor.ComputeAnchors(snapshotLines)
			anchorsReady = true
		}
	}

	for i, edit := range params.Edits {
		if isStrKind(edit.Kind) {
			recs, errRes := resolveStrEdit(content, i, edit)
			if errRes != nil {
				return withBatchAbortNote(errRes), nil
			}
			resolved = append(resolved, recs...)
			continue
		}
		ensureAnchors()
		rec, errRes := resolveAnchorEdit(content, i, edit, snapshotLines, snapshotAnchors)
		if errRes != nil {
			return withBatchAbortNote(errRes), nil
		}
		resolved = append(resolved, rec)
	}

	// Overlap detection (byte coordinates of the original file). Two ranges
	// overlap iff they intersect; additionally, two ZERO-WIDTH inserts from
	// different edits at the same offset are ambiguous in apply order and are
	// rejected too.
	for i := 0; i < len(resolved); i++ {
		for j := i + 1; j < len(resolved); j++ {
			a, b := resolved[i], resolved[j]
			aInsert := a.start == a.end
			bInsert := b.start == b.end
			if aInsert && bInsert && a.start == b.start && a.editIndex != b.editIndex {
				return withBatchAbortNote(errResult("edit_overlap",
					fmt.Sprintf("edits #%d and #%d both insert at the same position (byte %d); their ordering is ambiguous. Merge them into a single insert.",
						a.editIndex+1, b.editIndex+1, a.start))), nil
			}
			if a.start < b.end && b.start < a.end {
				return withBatchAbortNote(errResult("edit_overlap",
					fmt.Sprintf("edits #%d and #%d overlap at byte range [%d, %d) and [%d, %d)",
						a.editIndex+1, b.editIndex+1, a.start, a.end, b.start, b.end))), nil
			}
		}
	}

	// Apply edits bottom-up.
	sortByStartDesc(resolved)
	result := content
	for _, r := range resolved {
		result = result[:r.start] + r.newText + result[r.end:]
	}

	// No-op check on final result.
	if result == content {
		return errResult("no_op", "edits produce identical text"), nil
	}

	if params.ValidateOnly {
		return &tool.Result{Content: "Validation passed: all edits resolve correctly."}, nil
	}

	// Atomic write.
	if err := atomicWrite(path, []byte(result)); err != nil {
		return &tool.Result{Content: fmt.Sprintf("Error writing file: %s", err), IsError: true}, nil
	}

	// Build response with diff and fingerprints.
	return formatSuccess(path, content, result, params.Edits, resolved), nil
}

func errResult(code, detail string) *tool.Result {
	return &tool.Result{
		Content: fmt.Sprintf("Error: %s: %s", code, detail),
		IsError: true,
	}
}

// batchAbortNote is appended to a failure that aborts the WHOLE call after
// resolution begins (a per-edit resolve failure, or an overlap between edits).
// edit_file is all-or-nothing: on any such failure NONE of the edits are
// applied and the file is left unchanged. Spelled out so a caller reading only
// the error doesn't assume the other edits in the batch went through.
const batchAbortNote = " — the file is unchanged (edit_file is atomic). Fix and resend the edits."

// withBatchAbortNote appends batchAbortNote to a pre-write failure result.
func withBatchAbortNote(r *tool.Result) *tool.Result {
	r.Content += batchAbortNote
	return r
}

func formatSuccess(path, oldContent, newContent string, edits []Edit, resolved []resolvedEdit) *tool.Result {
	oldLines, oldTrailing := anchor.SplitTextForAnchors(oldContent)
	newLines, newTrailing := anchor.SplitTextForAnchors(newContent)
	oldFP := anchor.FileFingerprint(oldLines, oldTrailing)
	newFP := anchor.FileFingerprint(newLines, newTrailing)

	lineDelta := len(newLines) - len(oldLines)
	deltaStr := ""
	if lineDelta > 0 {
		deltaStr = fmt.Sprintf(" (+%d lines)", lineDelta)
	} else if lineDelta < 0 {
		deltaStr = fmt.Sprintf(" (%d lines)", lineDelta)
	}

	// Map each edit index to its first resolved record so the per-edit snippet
	// can report the line(s) actually touched in the ORIGINAL file. (str edits
	// with expected_match_count>1 produce several records; reporting the first
	// is enough to orient the reader, and the count is implicit in the diff.)
	byIndex := make(map[int]resolvedEdit, len(resolved))
	for _, r := range resolved {
		if _, seen := byIndex[r.editIndex]; !seen {
			byIndex[r.editIndex] = r
		}
	}
	offsets := lineByteOffsets(oldContent)

	var b strings.Builder
	fmt.Fprintf(&b, "Applied %d edit(s) to %s%s.\n", len(edits), path, deltaStr)
	fmt.Fprintf(&b, "old_fingerprint=%s new_fingerprint=%s\n", oldFP, newFP)

	// Per-edit unified diff (capped). Every removed line is prefixed "- ", every
	// added line "+ ", unchanged context "  " — the familiar git style, which is
	// unambiguous for multi-line edits and trivially machine-parseable (the old
	// format marked only the FIRST line of a multi-line change).
	for i, edit := range edits {
		if b.Len() > 6000 {
			fmt.Fprintf(&b, "\n[...remaining edits omitted for brevity...]\n")
			break
		}
		added := edit.NewText
		removed := ""
		if isStrKind(edit.Kind) {
			removed = edit.OldText
			loc := ""
			if r, ok := byIndex[i]; ok {
				loc = " " + lineSpan(offsets, r.start, r.end)
			}
			fmt.Fprintf(&b, "\nEdit #%d%s:\n", i+1, loc)
		} else {
			// Anchor edit: report the kind + resolved line span, and (for a replace
			// that removed content) the removed text — the #hash alone doesn't tell
			// the reader what it hit, so the diff stays self-verifying.
			r, ok := byIndex[i]
			loc := edit.AtFrom
			if ok {
				loc = lineSpan(offsets, r.start, r.end)
			}
			if edit.Kind == "replace" && ok && r.end > r.start {
				removed = strings.TrimSuffix(oldContent[r.start:r.end], "\n")
			}
			fmt.Fprintf(&b, "\nEdit #%d (%s %s):\n", i+1, edit.Kind, loc)
		}
		b.WriteString(unifiedDiff(removed, added, 60))
		b.WriteString("\n")
	}
	return &tool.Result{Content: b.String()}
}

// unifiedDiff renders a git-style line diff of removed→added: each line of the
// LCS-aligned result is prefixed "- " (removed), "+ " (added) or "  " (context).
// Inputs are split on '\n'; a trailing newline does not produce a spurious empty
// final line. If either side exceeds maxLines, the diff is not computed (the
// O(n*m) table would be wasteful for the LLM's token budget); instead the lines
// are emitted as a plain all-removed-then-all-added block with an explicit
// truncation note so the reader knows the alignment was skipped.
func unifiedDiff(removed, added string, maxLines int) string {
	delLines := splitNonEmpty(removed)
	addLines := splitNonEmpty(added)

	var b strings.Builder
	if len(delLines) > maxLines || len(addLines) > maxLines {
		for _, ln := range delLines {
			fmt.Fprintf(&b, "- %s\n", truncLine(ln))
		}
		for _, ln := range addLines {
			fmt.Fprintf(&b, "+ %s\n", truncLine(ln))
		}
		fmt.Fprintf(&b, "[diff alignment skipped: edit larger than %d lines]\n", maxLines)
		return b.String()
	}

	for _, d := range diffLines(delLines, addLines) {
		b.WriteString(d.prefix)
		b.WriteString(truncLine(d.text))
		b.WriteByte('\n')
	}
	return b.String()
}

// truncLine bounds a single rendered diff line so one pathological long line
// can't blow past the overall result cap. The appended "…" makes the truncation
// visible to both the model and the operator.
func truncLine(s string) string {
	const maxRunes = 300
	r := []rune(s)
	if len(r) <= maxRunes {
		return s
	}
	return string(r[:maxRunes]) + "…"
}

// splitNonEmpty splits on '\n' but treats "" as zero lines (not one empty line)
// and ignores a single trailing newline, so "a\nb\n" yields ["a","b"].
func splitNonEmpty(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(strings.TrimSuffix(s, "\n"), "\n")
}

type diffLine struct {
	prefix string // "- ", "+ " or "  "
	text   string
}

// diffLines computes a line-level LCS diff of a→b, emitting removals, additions
// and shared context lines in source order.
func diffLines(a, b []string) []diffLine {
	n, m := len(a), len(b)
	// LCS length table: dp[i][j] = LCS(a[i:], b[j:]).
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if a[i] == b[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else if dp[i+1][j] >= dp[i][j+1] {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1]
			}
		}
	}
	var out []diffLine
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case a[i] == b[j]:
			out = append(out, diffLine{"  ", a[i]})
			i++
			j++
		case dp[i+1][j] >= dp[i][j+1]:
			out = append(out, diffLine{"- ", a[i]})
			i++
		default:
			out = append(out, diffLine{"+ ", b[j]})
			j++
		}
	}
	for ; i < n; i++ {
		out = append(out, diffLine{"- ", a[i]})
	}
	for ; j < m; j++ {
		out = append(out, diffLine{"+ ", b[j]})
	}
	return out
}

// lineSpan renders the 1-based line location of a byte range [start,end) within
// the file whose line-start offsets are given. A zero-width range (insert) is
// rendered as a single insertion point; a multi-line range as "lines A-B".
func lineSpan(offsets []int, start, end int) string {
	lineOf := func(pos int) int {
		// Largest line index whose start offset is <= pos (1-based result).
		lo, hi := 0, len(offsets)-1
		for lo < hi {
			mid := (lo + hi + 1) / 2
			if offsets[mid] <= pos {
				lo = mid
			} else {
				hi = mid - 1
			}
		}
		return lo + 1
	}
	if start == end {
		return fmt.Sprintf("at line %d", lineOf(start))
	}
	// A whole-line range ends at the start of the line AFTER the last replaced
	// line, so subtract one byte to land on the last replaced line itself.
	from := lineOf(start)
	to := lineOf(end - 1)
	if from == to {
		return fmt.Sprintf("line %d", from)
	}
	return fmt.Sprintf("lines %d-%d", from, to)
}

func createFile(path, content string) (*tool.Result, error) {
	if _, err := os.Stat(path); err == nil {
		// File exists — allow writing into an empty file.
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return errResult("file_not_found", readErr.Error()), nil
		}
		if len(data) > 0 {
			return errResult("malformed_args",
				fmt.Sprintf("file already exists and is non-empty: %s. Use old_text to edit existing files.", path)), nil
		}
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return &tool.Result{Content: fmt.Sprintf("Error creating directory: %s", err), IsError: true}, nil
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		return &tool.Result{Content: fmt.Sprintf("Error writing file: %s", err), IsError: true}, nil
	}
	// Count lines, not newline separators: a file with content but no trailing
	// newline still has one (or more) line(s).
	lines := strings.Count(content, "\n")
	if content != "" && !strings.HasSuffix(content, "\n") {
		lines++
	}
	// Show the created content as an all-added diff block (every line prefixed
	// "+ "), so the result is self-describing for both the LLM and the UI — same
	// format as an edit's diff, with the same truncation rules.
	var b strings.Builder
	fmt.Fprintf(&b, "Created %s (%d lines).\n", path, lines)
	if content != "" {
		b.WriteString(unifiedDiff("", content, 60))
	}
	return &tool.Result{Content: b.String()}, nil
}

func atomicWrite(path string, data []byte) error {
	// Preserve the existing file's permissions: rename would otherwise reset
	// them to the temp file's mode (e.g. clobbering an executable bit).
	mode := os.FileMode(0o600)
	if info, err := os.Stat(path); err == nil {
		mode = info.Mode().Perm()
	}
	tmp := path + ".amplio.tmp"
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return err
	}
	// WriteFile is subject to umask; chmod to guarantee the intended mode.
	if err := os.Chmod(tmp, mode); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

type resolvedEdit struct {
	start, end int
	newText    string
	editIndex  int // 0-based position in the original edits list, for error attribution
}

// resolveStrEdit resolves a literal find-and-replace edit into one or more
// byte-splice records against the original content. On failure it returns a
// tool.Result error and a nil slice.
func resolveStrEdit(content string, i int, edit Edit) ([]resolvedEdit, *tool.Result) {
	if edit.ExpectAtFrom != "" || edit.ExpectAtTo != "" {
		return nil, errResult("malformed_args",
			fmt.Sprintf("edit #%d: expect_at_from/expect_at_to apply to anchor edits only; a str_replace's old_text already verifies the target", i+1))
	}
	if edit.OldText == "" {
		return nil, errResult("malformed_args",
			fmt.Sprintf("edit #%d: empty old_text only works as a single edit for file creation", i+1))
	}
	if edit.OldText == edit.NewText {
		return nil, errResult("no_op", fmt.Sprintf("edit #%d: old_text == new_text", i+1))
	}

	expectedCount := edit.ExpectedMatchCount
	if expectedCount <= 0 {
		expectedCount = 1
	}

	// Determine search window.
	searchContent := content
	searchOffset := 0
	if edit.StartLine != nil || edit.EndLine != nil {
		contentLines := strings.Split(content, "\n")
		start := 1
		end := len(contentLines)
		if edit.StartLine != nil && *edit.StartLine > 1 {
			start = *edit.StartLine
		}
		if edit.EndLine != nil && *edit.EndLine < end {
			end = *edit.EndLine
		}
		if start > len(contentLines) {
			return nil, errResult("target_not_found",
				fmt.Sprintf("edit #%d: start_line %d exceeds file length %d", i+1, start, len(contentLines)))
		}
		if end < start {
			return nil, errResult("malformed_args",
				fmt.Sprintf("edit #%d: end_line %d is before start_line %d", i+1, end, start))
		}
		// Convert line range to byte range.
		byteStart := 0
		for j := 0; j < start-1 && j < len(contentLines); j++ {
			byteStart += len(contentLines[j]) + 1 // +1 for \n
		}
		byteEnd := 0
		for j := 0; j < end && j < len(contentLines); j++ {
			byteEnd += len(contentLines[j]) + 1
		}
		if byteEnd > len(content) {
			byteEnd = len(content)
		}
		searchContent = content[byteStart:byteEnd]
		searchOffset = byteStart
	}

	// Find all NON-overlapping occurrences in the search window. Advance past
	// the whole match (not just one byte) so e.g. old_text "aa" in "aaaa"
	// counts as 2, not 3 — otherwise expected_match_count is inflated and the
	// overlap check below trips on self-overlapping matches. OldText is
	// guaranteed non-empty here, so len(edit.OldText) >= 1 and the loop
	// always advances.
	var matches []int
	offset := 0
	for {
		idx := strings.Index(searchContent[offset:], edit.OldText)
		if idx < 0 {
			break
		}
		matches = append(matches, searchOffset+offset+idx)
		offset += idx + len(edit.OldText)
	}

	if len(matches) == 0 {
		hint := closestMatch(content, edit.OldText)
		return nil, errResult("target_not_found",
			fmt.Sprintf("edit #%d: old_text not found in file.%s", i+1, hint))
	}
	if len(matches) != expectedCount {
		return nil, errResult("multiple_matches",
			fmt.Sprintf("edit #%d: found %d occurrences, expected %d. Provide more context or set expected_match_count=%d.",
				i+1, len(matches), expectedCount, len(matches)))
	}

	recs := make([]resolvedEdit, 0, len(matches))
	for _, matchStart := range matches {
		recs = append(recs, resolvedEdit{
			start:     matchStart,
			end:       matchStart + len(edit.OldText),
			newText:   edit.NewText,
			editIndex: i,
		})
	}
	return recs, nil
}

// lineByteOffsets returns the byte offset of the START of each line in text.
// An empty file returns [0]; a file ending in "\n" has a final entry equal to
// len(text) (the EOF position), so insert-after-last-line and insert-before-EOF
// both map cleanly.
func lineByteOffsets(text string) []int {
	offsets := []int{0}
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			offsets = append(offsets, i+1)
		}
	}
	return offsets
}

// viewHint nudges the caller toward the produce-side step when an anchor can't
// be resolved — the most common cause is editing from stale or guessed hashes.
const viewHint = " (call view_file(show_anchors=true) on the current file to get fresh #hashes — anchors track line content, so they only change where the nearby lines changed)"

// anchorErrResult maps an anchor.ResolveAnchor error to the right failure code.
// Parse errors -> malformed_args; "not found in current snapshot" ->
// target_not_found; ambiguous / "@N did not match" -> stale_anchor. The resolve
// failures (not the parse failure, which is a malformed address) carry a hint to
// re-view, since editing from stale/guessed hashes is the usual cause.
func anchorErrResult(i int, err error) *tool.Result {
	msg := err.Error()
	switch {
	case errors.Is(err, anchor.ErrAnchorParse):
		return errResult("malformed_args", fmt.Sprintf("edit #%d: bad anchor address: %s", i+1, msg))
	case strings.Contains(msg, "not found"):
		return errResult("target_not_found", fmt.Sprintf("edit #%d: %s%s", i+1, msg, viewHint))
	default:
		return errResult("stale_anchor", fmt.Sprintf("edit #%d: %s%s", i+1, msg, viewHint))
	}
}

// checkExpect verifies that the line at the given 0-based index contains the
// expected substring (trailing-whitespace normalized, matching anchor hashing).
// which names the guard field for the error message. A zero-value expect or an
// out-of-range index is a no-op (the caller validates the index separately).
func checkExpect(lines []string, idx int, expect, which string, i int) *tool.Result {
	if expect == "" {
		return nil
	}
	if idx < 0 || idx >= len(lines) {
		return nil
	}
	line := strings.TrimRight(lines[idx], " \t\r\n")
	if strings.Contains(line, expect) {
		return nil
	}
	return errResult("expect_mismatch",
		fmt.Sprintf("edit #%d: %s %q not found on resolved line %d: %q%s",
			i+1, which, expect, idx+1, line, viewHint))
}

// resolveAnchorEdit resolves an anchor-addressed replace/insert edit into one
// byte-splice record. nLines is len(lines).
func resolveAnchorEdit(content string, i int, edit Edit, lines, anchors []string) (resolvedEdit, *tool.Result) {
	nLines := len(lines)
	offsets := lineByteOffsets(content)

	switch edit.Kind {
	case "insert_before", "insert_after":
		if edit.ExpectAtTo != "" {
			return resolvedEdit{}, errResult("malformed_args",
				fmt.Sprintf("edit #%d: expect_at_to is only valid for replace (inserts have no at_to)", i+1))
		}
		pos, err := anchor.ResolveAnchor(edit.AtFrom, anchors, nLines)
		if err != nil {
			return resolvedEdit{}, anchorErrResult(i, err)
		}
		if errRes := checkExpect(lines, pos, edit.ExpectAtFrom, "expect_at_from", i); errRes != nil {
			return resolvedEdit{}, errRes
		}
		// insert_before targets the anchored line; insert_after the next line.
		// BOF resolves to -1 and EOF to nLines; both reduce to a file extreme.
		targetLine := pos
		if edit.Kind == "insert_after" {
			targetLine = pos + 1
		}
		if targetLine < 0 {
			targetLine = 0
		}
		if targetLine > nLines {
			targetLine = nLines
		}
		byteStart := len(content)
		if targetLine < len(offsets) {
			byteStart = offsets[targetLine]
		}
		newText := edit.NewText
		// Ensure the inserted block ends with a newline so surrounding lines
		// stay on their own lines — except when inserting at EOF on a file with
		// no trailing newline (so the agent can author no-final-newline files).
		if newText != "" && !strings.HasSuffix(newText, "\n") {
			insertingAtEOF := targetLine == nLines
			fileHasTrailingNL := strings.HasSuffix(content, "\n")
			if !(insertingAtEOF && !fileHasTrailingNL) {
				newText += "\n"
			}
		}
		return resolvedEdit{start: byteStart, end: byteStart, newText: newText, editIndex: i}, nil

	case "replace":
		posFrom, err := anchor.ResolveAnchor(edit.AtFrom, anchors, nLines)
		if err != nil {
			return resolvedEdit{}, anchorErrResult(i, err)
		}
		if posFrom < 0 || posFrom >= nLines {
			return resolvedEdit{}, badSentinelResult(i, edit.Kind)
		}
		atTo := edit.AtTo
		if atTo == "" {
			atTo = edit.AtFrom
		}
		posTo, err := anchor.ResolveAnchor(atTo, anchors, nLines)
		if err != nil {
			return resolvedEdit{}, anchorErrResult(i, err)
		}
		if posTo < 0 || posTo >= nLines {
			return resolvedEdit{}, badSentinelResult(i, edit.Kind)
		}
		if posFrom > posTo {
			return resolvedEdit{}, errResult("malformed_args",
				fmt.Sprintf("edit #%d: at_from resolves to line %d which is AFTER at_to at line %d", i+1, posFrom+1, posTo+1))
		}
		if errRes := checkExpect(lines, posFrom, edit.ExpectAtFrom, "expect_at_from", i); errRes != nil {
			return resolvedEdit{}, errRes
		}
		if errRes := checkExpect(lines, posTo, edit.ExpectAtTo, "expect_at_to", i); errRes != nil {
			return resolvedEdit{}, errRes
		}
		// Byte range = [start of line posFrom .. start of line posTo+1).
		byteStart := offsets[posFrom]
		byteEnd := len(content)
		if posTo+1 < len(offsets) {
			byteEnd = offsets[posTo+1]
		}
		newText := edit.NewText
		// If the matched region ended with "\n", keep the replacement
		// newline-terminated so we don't collapse the next line into ours.
		// Empty new_text (delete) is exempt.
		matched := content[byteStart:byteEnd]
		if strings.HasSuffix(matched, "\n") && newText != "" && !strings.HasSuffix(newText, "\n") {
			newText += "\n"
		}
		return resolvedEdit{start: byteStart, end: byteEnd, newText: newText, editIndex: i}, nil

	default:
		return resolvedEdit{}, errResult("malformed_args",
			fmt.Sprintf("edit #%d: unknown kind %q", i+1, edit.Kind))
	}
}

// badSentinelResult is returned when a #BOF/#EOF sentinel is used in a replace,
// where only concrete in-range lines are legal.
func badSentinelResult(i int, kind string) *tool.Result {
	return errResult("malformed_args",
		fmt.Sprintf("edit #%d: %s with sentinel anchor #BOF/#EOF — sentinels are only legal in insert_before/insert_after.", i+1, kind))
}

func sortByStartDesc(edits []resolvedEdit) {
	for i := 1; i < len(edits); i++ {
		for j := i; j > 0 && edits[j].start > edits[j-1].start; j-- {
			edits[j], edits[j-1] = edits[j-1], edits[j]
		}
	}
}

// closestMatch finds the most similar line to old_text's first line.
func closestMatch(content, target string) string {
	targetLines := strings.Split(strings.TrimSpace(target), "\n")
	if len(targetLines) == 0 {
		return ""
	}
	firstLine := strings.TrimSpace(targetLines[0])
	if len(firstLine) < 5 {
		return ""
	}

	contentLines := strings.Split(content, "\n")
	bestScore := 0
	bestLine := ""
	bestLineNum := 0

	for i, line := range contentLines {
		trimmed := strings.TrimSpace(line)
		score := commonPrefixLen(firstLine, trimmed)
		if score > bestScore && score > len(firstLine)/3 {
			bestScore = score
			bestLine = trimmed
			bestLineNum = i + 1
		}
	}
	if bestLine == "" {
		return ""
	}
	if len(bestLine) > 80 {
		bestLine = bestLine[:80] + "..."
	}
	return fmt.Sprintf("\n  Closest match at line %d: %q", bestLineNum, bestLine)
}

func commonPrefixLen(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

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

// Package viewfile provides the file viewing tool.
package viewfile

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"amplio/internal/event"
	"amplio/internal/tool"
	"amplio/internal/tool/anchor"
	"amplio/internal/workspace/citc"
)

const (
	maxOutputBytes = 200 * 1024       // cap on the RENDERED text returned to the model
	maxReadBytes   = 8 * 1024 * 1024  // hard ceiling on a text file we'll read into memory
	maxImageSize   = 10 * 1024 * 1024 // 10MB image cap

	// autoAnchorMaxLines is the rendered-line threshold for the AUTO anchor
	// default: when show_anchors is unset, a view of at most this many lines gets
	// content-hash anchors (they make a following anchor edit possible without a
	// second view), while larger views stay clean. Small views are ~half of all
	// views and where anchors already cluster; the per-line anchor overhead is
	// cheap there and wasteful (and rarely anchor-edited) on big dumps. Start
	// conservative; safe to raise once we confirm it doesn't hurt comprehension.
	autoAnchorMaxLines = 90

	// envArtifactDir is the env-var name (literal, not imported from config to
	// keep this tool config-free) that a path may lead with; resolvePath expands
	// it to the run's artifact dir.
	envArtifactDir = "AMPLIO_ARTIFACT_DIR"
)

type Params struct {
	Path      string `json:"path" jsonschema:"required" jsonschema_description:"File path (relative to workspace or absolute)"`
	StartLine *int   `json:"start_line,omitempty" jsonschema_description:"1-indexed inclusive start line"`
	EndLine   *int   `json:"end_line,omitempty" jsonschema_description:"1-indexed inclusive end line"`
	// ShowAnchors is tri-state: nil (omitted) = AUTO (anchors shown iff the view
	// is small, see autoAnchorMaxLines); true/false = force on/off. AUTO lets a
	// small read double as the produce-side of an anchor edit without a second
	// view, while keeping large/whole-file reads clean.
	ShowAnchors *bool `json:"show_anchors,omitempty" jsonschema_description:"Show content-hash anchors per line (for edit_file anchor edits). Omit for auto (on for small views); set true/false to force."`
}

// New builds the view_file tool. cwd anchors relative paths; artifactDir is the
// run's scratch dir, used to expand a leading $AMPLIO_ARTIFACT_DIR in paths (the
// agent learns that variable from the bash tool and naturally reuses it here).
func New(cwd, artifactDir string) *tool.Tool {
	return &tool.Tool{
		Name: "view_file",
		Description: fmt.Sprintf("Read a file's contents. Returns text with line numbers (or renders images inline). "+
			"Small views also show per-line #hash anchors by default so you can follow up with an edit_file anchor edit "+
			"without re-viewing; pass show_anchors=false to suppress, or true to force on a large view. CWD=%q.", cwd),
		ParamType: &Params{},
		Execute:   makeExecutor(cwd, artifactDir),
	}
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
// counts (the variable names a directory); artifactDir=="" (no run context)
// leaves the path untouched.
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
	path := resolvePath(cwd, artifactDir, params.Path)

	info, err := os.Stat(path)
	if err != nil {
		return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
	}
	if info.IsDir() {
		return &tool.Result{Content: fmt.Sprintf("Error: %s is a directory, not a file.", path), IsError: true}, nil
	}

	// Check if it's an image by reading the first 512 bytes.
	if isImage(path) {
		// Only an EXPLICIT show_anchors=true is an error; the AUTO default (nil)
		// silently means "no anchors" for images.
		if params.ShowAnchors != nil && *params.ShowAnchors {
			return &tool.Result{Content: "Error: show_anchors is not supported for image files.", IsError: true}, nil
		}
		return viewImage(path, info.Size())
	}

	return viewText(path, info.Size(), params.StartLine, params.EndLine, params.ShowAnchors)
}

func isImage(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	buf := make([]byte, 512)
	n, _ := f.Read(buf)
	mime := http.DetectContentType(buf[:n])
	return strings.HasPrefix(mime, "image/")
}

func viewImage(path string, size int64) (*tool.Result, error) {
	if size > maxImageSize {
		return &tool.Result{
			Content: fmt.Sprintf("Error: image too large (%d bytes, max %d).", size, maxImageSize),
			IsError: true,
		}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return &tool.Result{Content: fmt.Sprintf("Error reading image: %s", err), IsError: true}, nil
	}
	mime := http.DetectContentType(data[:min(512, len(data))])
	return &tool.Result{
		Content: fmt.Sprintf("Image: %s (%d bytes)", filepath.Base(path), len(data)),
		Attachments: []event.Attachment{{
			MimeType:   mime,
			Data:       data,
			SourceHint: path,
		}},
	}, nil
}

func viewText(path string, size int64, startLine, endLine *int, showAnchors *bool) (*tool.Result, error) {
	// Refuse to read an arbitrarily large file into memory. The output is capped
	// (maxOutputBytes) but only AFTER reading + line-splitting the whole file, so
	// without this guard a multi-hundred-MB file would be slurped and split
	// before anything is truncated. Anchoring also requires the full content, so
	// there's no partial-read shortcut — refuse and point at the range params.
	if size > maxReadBytes {
		return &tool.Result{
			Content: fmt.Sprintf("Error: file too large to view (%d bytes, max %d). Use start_line/end_line to view a slice, or read it with a bash tool.", size, maxReadBytes),
			IsError: true,
		}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return &tool.Result{Content: fmt.Sprintf("Error: %s", err), IsError: true}, nil
	}

	// Binary detection: NUL byte in first 8KB.
	probe := data
	if len(probe) > 8192 {
		probe = probe[:8192]
	}
	for _, b := range probe {
		if b == 0 {
			return &tool.Result{
				Content: fmt.Sprintf("Error: %s appears to be a binary file.", path),
				IsError: true,
			}, nil
		}
	}

	lines := strings.Split(string(data), "\n")
	// Remove trailing empty line from split (file ending with \n).
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	totalLines := len(lines)

	// Validate and apply line range.
	start := 1
	end := totalLines
	if startLine != nil {
		start = *startLine
	}
	if endLine != nil {
		end = *endLine
	}
	if start < 1 {
		start = 1
	}
	if end > totalLines {
		end = totalLines
	}
	if start > totalLines {
		return &tool.Result{
			Content: fmt.Sprintf("Note: requested start_line=%d but file has only %d line(s).", start, totalLines),
		}, nil
	}
	if start > end {
		return &tool.Result{
			Content: fmt.Sprintf("Error: start_line=%d > end_line=%d.", start, end),
			IsError: true,
		}, nil
	}

	// Resolve the tri-state anchor flag against the RENDERED line count (after
	// clamping), so the AUTO default keys on what's actually shown: a small range
	// of a big file gets anchors; a whole-file view of a large file doesn't.
	anchorsOn := false
	if showAnchors != nil {
		anchorsOn = *showAnchors // explicit override wins
	} else {
		anchorsOn = (end - start + 1) <= autoAnchorMaxLines
	}

	// Build output.
	var b strings.Builder
	rangeNote := ""
	if start != 1 || end != totalLines {
		rangeNote = fmt.Sprintf(" (showing lines %d-%d of %d)", start, end, totalLines)
	}

	if anchorsOn {
		// Compute anchors for ALL lines (needed for correct window hashing).
		anchors := anchor.ComputeAnchors(lines)
		_, hadTrailingNewline := anchor.SplitTextForAnchors(string(data))
		fp := anchor.FileFingerprint(lines, hadTrailingNewline)
		fmt.Fprintf(&b, "file_fingerprint=%s\n", fp)
		fmt.Fprintf(&b, "File: %s%s\n", path, rangeNote)
		for i := start - 1; i < end; i++ {
			fmt.Fprintf(&b, "[L%d  #%s] %s\n", i+1, anchors[i], lines[i])
		}
	} else {
		fmt.Fprintf(&b, "File: %s%s\n", path, rangeNote)
		for i := start - 1; i < end; i++ {
			fmt.Fprintf(&b, "%4d: %s\n", i+1, lines[i])
		}
	}

	content := b.String()
	if len(content) > maxOutputBytes {
		// Back the cut off to a UTF-8 boundary so we never emit a split rune
		// (the rendered text is frequently non-ASCII source code/prose).
		cut := maxOutputBytes
		for cut > 0 && !utf8.RuneStart(content[cut]) {
			cut--
		}
		content = content[:cut] + "\n[...truncated...]"
	}
	return &tool.Result{Content: content}, nil
}

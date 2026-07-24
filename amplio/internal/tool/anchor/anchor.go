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

// Package anchor provides content-hash line anchors for view_file + edit_file.
//
// Anchors are 6-hex-char (24-bit) BLAKE2b digests of a 5-line window
// (±2 lines: the 2 above + current + 2 below) centered on each line. They give the LLM a
// content-addressed handle that:
//
//   - Is stable across CRLF/LF and trailing-whitespace changes.
//   - Survives line-number drift from earlier edits.
//   - Almost never collides on real code (24-bit = ~2.93% collision
//     in 1000-line file, well below true structural duplicate rate).
//
// The whole-file fingerprint is a 12-hex (48-bit) digest used as an ETag.
package anchor

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"golang.org/x/crypto/blake2b"
)

const (
	HashHexLen        = 6  // 24-bit per-line anchor
	WindowRadius      = 2  // 5-line window
	FingerprintHexLen = 12 // 48-bit whole-file ETag

	SentinelBOF = "BOF"
	SentinelEOF = "EOF"
)

var (
	ErrAnchorParse   = errors.New("anchor parse error")
	ErrAnchorResolve = errors.New("anchor resolve error")
)

// normalizeLine strips trailing whitespace and line terminators, preserving
// leading indentation (load-bearing in Python, YAML, etc.).
func normalizeLine(line string) string {
	return strings.TrimRight(line, " \t\r\n")
}

// WindowHash returns the anchor hash for lines[idx] over a ±WindowRadius window.
func WindowHash(lines []string, idx int) string {
	var parts []string
	for j := idx - WindowRadius; j <= idx+WindowRadius; j++ {
		if j >= 0 && j < len(lines) {
			parts = append(parts, normalizeLine(lines[j]))
		} else {
			parts = append(parts, "")
		}
	}
	blob := []byte(strings.Join(parts, "\x00"))
	h, _ := blake2b.New(HashHexLen/2, nil) // digest_size = 3 bytes → 6 hex chars
	h.Write(blob)
	return fmt.Sprintf("%x", h.Sum(nil))[:HashHexLen]
}

// ComputeAnchors returns the per-line anchor hash for every line.
func ComputeAnchors(lines []string) []string {
	anchors := make([]string, len(lines))
	for i := range lines {
		anchors[i] = WindowHash(lines, i)
	}
	return anchors
}

// FileFingerprint computes the ETag-style fingerprint of the whole file.
func FileFingerprint(lines []string, hadTrailingNewline bool) string {
	var parts []string
	for _, l := range lines {
		parts = append(parts, normalizeLine(l))
	}
	normalized := strings.Join(parts, "\n")
	if hadTrailingNewline && len(lines) > 0 {
		normalized += "\n"
	}
	h, _ := blake2b.New(FingerprintHexLen/2, nil)
	h.Write([]byte(normalized))
	return fmt.Sprintf("%x", h.Sum(nil))[:FingerprintHexLen]
}

// SplitTextForAnchors splits file text into lines and captures trailing newline.
func SplitTextForAnchors(text string) (lines []string, hadTrailingNewline bool) {
	if text == "" {
		return nil, false
	}
	hadTrailingNewline = strings.HasSuffix(text, "\n") || strings.HasSuffix(text, "\r\n")
	lines = strings.Split(text, "\n")
	// Remove trailing empty string from split.
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return
}

// --- Anchor address parser + resolver ---

// ParseAnchor parses an anchor address into (canonical, lineHint).
// Accepted shapes: #a3f1c0, #a3f1c0@42, #BOF, #EOF.
// lineHint is -1 for bare/sentinel anchors, 1-based for @N form.
func ParseAnchor(text string) (canonical string, lineHint int, err error) {
	lineHint = -1
	if text == "" {
		return "", -1, fmt.Errorf("%w: empty anchor address", ErrAnchorParse)
	}
	stripped := strings.TrimLeft(text, "#")
	stripped = strings.TrimSpace(stripped)
	if stripped == "" {
		return "", -1, fmt.Errorf("%w: empty anchor address: %q", ErrAnchorParse, text)
	}

	if stripped == SentinelBOF {
		return SentinelBOF, -1, nil
	}
	if stripped == SentinelEOF {
		return SentinelEOF, -1, nil
	}

	if at := strings.Index(stripped, "@"); at >= 0 {
		lineStr := stripped[at+1:]
		if lineStr == "" {
			return "", -1, fmt.Errorf("%w: anchor %q has empty line-number suffix", ErrAnchorParse, text)
		}
		n, err := strconv.Atoi(lineStr)
		if err != nil {
			return "", -1, fmt.Errorf("%w: anchor %q has non-integer line suffix: %v", ErrAnchorParse, text, err)
		}
		if n < 1 {
			return "", -1, fmt.Errorf("%w: anchor %q: line numbers are 1-based", ErrAnchorParse, text)
		}
		lineHint = n
		stripped = stripped[:at]
	}

	if len(stripped) != HashHexLen {
		return "", -1, fmt.Errorf("%w: anchor %q: hash must be %d hex chars (got %d)",
			ErrAnchorParse, text, HashHexLen, len(stripped))
	}
	stripped = strings.ToLower(stripped)
	for _, c := range stripped {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return "", -1, fmt.Errorf("%w: anchor %q: hash must be hexadecimal", ErrAnchorParse, text)
		}
	}
	return stripped, lineHint, nil
}

// ResolveAnchor resolves an anchor address against a snapshot.
// Returns 0-based line index. BOF → -1, EOF → nLines.
func ResolveAnchor(text string, anchors []string, nLines int) (int, error) {
	canonical, lineHint, err := ParseAnchor(text)
	if err != nil {
		return 0, err
	}
	if canonical == SentinelBOF {
		return -1, nil
	}
	if canonical == SentinelEOF {
		return nLines, nil
	}

	var hits []int
	for i, a := range anchors {
		if a == canonical {
			hits = append(hits, i)
		}
	}
	if len(hits) == 0 {
		return 0, fmt.Errorf("%w: anchor #%s not found in current snapshot; re-view the file",
			ErrAnchorResolve, canonical)
	}
	if lineHint >= 1 {
		idx := lineHint - 1
		found := false
		for _, h := range hits {
			if h == idx {
				found = true
				break
			}
		}
		if !found {
			lineNums := make([]int, len(hits))
			for i, h := range hits {
				lineNums[i] = h + 1
			}
			return 0, fmt.Errorf("%w: anchor #%s@%d did not match; anchor is on line(s) %v now",
				ErrAnchorResolve, canonical, lineHint, lineNums)
		}
		return idx, nil
	}
	if len(hits) > 1 {
		lineNums := make([]int, len(hits))
		for i, h := range hits {
			lineNums[i] = h + 1
		}
		return 0, fmt.Errorf("%w: anchor #%s is ambiguous: matches lines %v; disambiguate as #%s@<lineno>",
			ErrAnchorResolve, canonical, lineNums, canonical)
	}
	return hits[0], nil
}

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

package anchor

import (
	"errors"
	"testing"
)

func TestWindowHash_Deterministic(t *testing.T) {
	lines := []string{"func main() {", "\tfmt.Println(\"hello\")", "}"}
	h1 := WindowHash(lines, 1)
	h2 := WindowHash(lines, 1)
	if h1 != h2 {
		t.Errorf("non-deterministic: %q vs %q", h1, h2)
	}
	if len(h1) != HashHexLen {
		t.Errorf("hash length: %d, want %d", len(h1), HashHexLen)
	}
}

func TestWindowHash_DifferentLines(t *testing.T) {
	lines1 := []string{"aaa", "bbb", "ccc"}
	lines2 := []string{"aaa", "xxx", "ccc"}
	h1 := WindowHash(lines1, 1)
	h2 := WindowHash(lines2, 1)
	if h1 == h2 {
		t.Error("different content should produce different hashes")
	}
}

func TestWindowHash_EdgeLines(t *testing.T) {
	lines := []string{"only"}
	h := WindowHash(lines, 0)
	if len(h) != HashHexLen {
		t.Errorf("hash length: %d", len(h))
	}
}

func TestComputeAnchors(t *testing.T) {
	lines := []string{"a", "b", "c", "d"}
	anchors := ComputeAnchors(lines)
	if len(anchors) != 4 {
		t.Fatalf("anchors: %d, want 4", len(anchors))
	}
	for i, a := range anchors {
		if len(a) != HashHexLen {
			t.Errorf("anchor[%d] length: %d", i, len(a))
		}
	}
}

func TestFileFingerprint(t *testing.T) {
	lines := []string{"hello", "world"}
	fp1 := FileFingerprint(lines, true)
	fp2 := FileFingerprint(lines, true)
	if fp1 != fp2 {
		t.Error("non-deterministic fingerprint")
	}
	if len(fp1) != FingerprintHexLen {
		t.Errorf("fingerprint length: %d", len(fp1))
	}

	// Different trailing newline → different fingerprint.
	fp3 := FileFingerprint(lines, false)
	if fp1 == fp3 {
		t.Error("trailing newline should affect fingerprint")
	}
}

func TestSplitTextForAnchors(t *testing.T) {
	tests := []struct {
		text     string
		wantN    int
		wantTail bool
	}{
		{"", 0, false},
		{"hello\n", 1, true},
		{"a\nb\n", 2, true},
		{"a\nb", 2, false},
	}
	for _, tt := range tests {
		lines, tail := SplitTextForAnchors(tt.text)
		if len(lines) != tt.wantN || tail != tt.wantTail {
			t.Errorf("SplitTextForAnchors(%q) = (%d lines, tail=%v), want (%d, %v)",
				tt.text, len(lines), tail, tt.wantN, tt.wantTail)
		}
	}
}

func TestParseAnchor(t *testing.T) {
	tests := []struct {
		text      string
		wantCanon string
		wantLine  int
		wantErr   bool
	}{
		{"#a3f1c0", "a3f1c0", -1, false},
		{"a3f1c0", "a3f1c0", -1, false},
		{"#A3F1C0", "a3f1c0", -1, false},
		{"#a3f1c0@42", "a3f1c0", 42, false},
		{"#BOF", "BOF", -1, false},
		{"#EOF", "EOF", -1, false},
		{"", "", -1, true},
		{"#", "", -1, true},
		{"#abc", "", -1, true},        // too short
		{"#abcdefgh", "", -1, true},   // too long
		{"#a3f1c0@", "", -1, true},    // empty line suffix
		{"#a3f1c0@0", "", -1, true},   // line < 1
		{"#a3f1c0@abc", "", -1, true}, // non-integer
		{"#zzzzzz", "", -1, true},     // non-hex
	}
	for _, tt := range tests {
		canon, line, err := ParseAnchor(tt.text)
		if tt.wantErr {
			if err == nil {
				t.Errorf("ParseAnchor(%q): expected error", tt.text)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseAnchor(%q): %v", tt.text, err)
			continue
		}
		if canon != tt.wantCanon || line != tt.wantLine {
			t.Errorf("ParseAnchor(%q) = (%q, %d), want (%q, %d)",
				tt.text, canon, line, tt.wantCanon, tt.wantLine)
		}
	}
}

func TestResolveAnchor(t *testing.T) {
	lines := []string{"aaa", "bbb", "ccc", "bbb"}
	anchors := ComputeAnchors(lines)

	// Unique anchor.
	idx, err := ResolveAnchor("#"+anchors[0], anchors, len(lines))
	if err != nil || idx != 0 {
		t.Errorf("resolve line 0: idx=%d, err=%v", idx, err)
	}

	// BOF/EOF sentinels.
	idx, err = ResolveAnchor("#BOF", anchors, len(lines))
	if err != nil || idx != -1 {
		t.Errorf("BOF: idx=%d, err=%v", idx, err)
	}
	idx, err = ResolveAnchor("#EOF", anchors, len(lines))
	if err != nil || idx != len(lines) {
		t.Errorf("EOF: idx=%d, err=%v", idx, err)
	}

	// Not found.
	_, err = ResolveAnchor("#000000", anchors, len(lines))
	if !errors.Is(err, ErrAnchorResolve) {
		t.Errorf("expected resolve error, got: %v", err)
	}
}

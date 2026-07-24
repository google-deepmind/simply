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

package server

import (
	"io/fs"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"amplio/internal/config"
)

// artifactEntry is one item in a run's artifact directory listing.
type artifactEntry struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir"`
	Size  int64  `json:"size"`
	MTime string `json:"mtime"` // RFC3339
}

// artifactListing is the JSON for a directory under a run's artifact dir.
type artifactListing struct {
	Root    string          `json:"root"` // absolute artifact dir (for "copy path")
	Path    string          `json:"path"` // subpath relative to the root ("" = root)
	Entries []artifactEntry `json:"entries"`
}

// artifactFile is one file in a recursive (flat) artifact listing: its full
// subpath relative to the artifact root, plus size/mtime.
type artifactFile struct {
	Path  string `json:"path"` // subpath relative to root, forward-slashed
	Size  int64  `json:"size"`
	MTime string `json:"mtime"` // RFC3339
}

// maxArtifactWalk caps the recursive listing so a pathological artifact tree
// can't produce an unbounded response (the client uses it for filename search).
const maxArtifactWalk = 20000

// handleArtifacts lists a directory under a run's artifact dir
// (<data-dir>/artifacts/<run-id>). The subpath comes from ?path= and is opened
// through os.Root, which confines access to the artifact dir (rejecting ".."
// and symlink escapes).
func (s *Server) handleArtifacts(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	sub := cleanArtifactSub(r.URL.Query().Get("path"))

	base := config.ArtifactDir(id) // creates the base
	root, err := os.OpenRoot(base)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "open artifact dir: "+err.Error())
		return
	}
	defer root.Close() //nolint:errcheck

	name := sub
	if name == "" {
		name = "."
	}
	f, err := root.Open(name)
	if err != nil {
		writeErr(w, http.StatusNotFound, "not found")
		return
	}
	defer f.Close() //nolint:errcheck
	fi, err := f.Stat()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if !fi.IsDir() {
		writeErr(w, http.StatusBadRequest, "not a directory (use the raw endpoint for files)")
		return
	}
	des, err := f.ReadDir(-1)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	entries := make([]artifactEntry, 0, len(des))
	for _, de := range des {
		info, err := de.Info()
		if err != nil {
			continue // skip unstattable entries (broken symlink, perms)
		}
		entries = append(entries, artifactEntry{
			Name:  de.Name(),
			IsDir: de.IsDir(),
			Size:  info.Size(),
			MTime: info.ModTime().UTC().Format(time.RFC3339),
		})
	}
	// Directories first, then files; each name-sorted.
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].IsDir != entries[j].IsDir {
			return entries[i].IsDir
		}
		return entries[i].Name < entries[j].Name
	})
	writeJSON(w, http.StatusOK, artifactListing{Root: base, Path: sub, Entries: entries})
}

// handleArtifactsAll returns a flat, recursive listing of every FILE under a
// run's artifact dir (subpaths relative to the root). It backs the browser's
// filename fuzzy-search; directories are omitted. Confined to the artifact dir
// via os.Root, and capped at maxArtifactWalk entries.
func (s *Server) handleArtifactsAll(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	base := config.ArtifactDir(id) // creates the base
	root, err := os.OpenRoot(base)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "open artifact dir: "+err.Error())
		return
	}
	defer root.Close() //nolint:errcheck

	files := make([]artifactFile, 0, 64)
	fsys := root.FS()
	err = fs.WalkDir(fsys, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable subtrees rather than failing the whole walk
		}
		if p == "." || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil // skip unstattable entries (broken symlink, perms)
		}
		files = append(files, artifactFile{
			Path:  p, // io/fs always uses forward slashes
			Size:  info.Size(),
			MTime: info.ModTime().UTC().Format(time.RFC3339),
		})
		if len(files) >= maxArtifactWalk {
			return fs.SkipAll
		}
		return nil
	})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	writeJSON(w, http.StatusOK, map[string]any{"root": base, "files": files})
}

// handleArtifactRaw serves a single file from a run's artifact dir. The file is
// sandboxed (CSP + nosniff) so an agent-written HTML/SVG can't execute JS in the
// dashboard origin; non-displayable types are sent as a download.
func (s *Server) handleArtifactRaw(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	sub := cleanArtifactSub(r.URL.Query().Get("path"))
	if sub == "" {
		writeErr(w, http.StatusBadRequest, "path is required")
		return
	}
	root, err := os.OpenRoot(config.ArtifactDir(id))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "open artifact dir: "+err.Error())
		return
	}
	defer root.Close() //nolint:errcheck
	f, err := root.Open(sub)
	if err != nil {
		writeErr(w, http.StatusNotFound, "not found")
		return
	}
	defer f.Close() //nolint:errcheck
	fi, err := f.Stat()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if fi.IsDir() {
		writeErr(w, http.StatusBadRequest, "is a directory")
		return
	}

	// Content-type detection: prefer the file extension, fall back to byte
	// sniffing. http.DetectContentType implements the WhatWG MIME-sniff spec,
	// which has signatures for PNG/GIF/JPEG/WebP/etc. but NOT for SVG —
	// an SVG file would otherwise be detected as text/xml (XML prologue) or
	// text/html (naked <svg>), and the browser would render the markup tree
	// instead of the image. The extension path uses Go's built-in mime table
	// (.svg → image/svg+xml, .json → application/json, .css → text/css; …)
	// which is correct for the common cases; sniffing covers the long tail
	// (extensionless files, generic binary blobs).
	buf := make([]byte, 512)
	n, _ := f.Read(buf)
	if _, err := f.Seek(0, 0); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	ct := mime.TypeByExtension(filepath.Ext(sub))
	if ct == "" {
		ct = http.DetectContentType(buf[:n])
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Security-Policy", "sandbox; default-src 'none'")
	// Inline-preview only safe types; everything else downloads.
	if !strings.HasPrefix(ct, "image/") && !strings.HasPrefix(ct, "text/") {
		w.Header().Set("Content-Disposition", "attachment")
	}
	http.ServeContent(w, r, fi.Name(), fi.ModTime(), f)
}

// cleanArtifactSub trims a leading slash so callers may pass "/foo" or "foo";
// os.Root rejects any escape regardless.
func cleanArtifactSub(p string) string {
	return strings.TrimPrefix(p, "/")
}

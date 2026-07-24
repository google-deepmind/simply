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
	"bytes"
	"net/http"
	"strings"
	"time"

	"amplio/internal/blob"
	"amplio/internal/config"
)

// handleBlob serves a run's content-addressed tool-result blob (e.g. a view_file
// image). Keys are validated hex (no traversal) and content is immutable, so it
// is cached aggressively. Non-image content is forced to a download with nosniff
// to avoid serving agent-influenced bytes as inline HTML.
func (s *Server) handleBlob(w http.ResponseWriter, r *http.Request) {
	id, key := r.PathValue("id"), r.PathValue("key")
	if strings.ContainsAny(id, `/\`) || strings.Contains(id, "..") {
		writeErr(w, http.StatusBadRequest, "invalid run id")
		return
	}
	if !blob.ValidKey(key) {
		writeErr(w, http.StatusBadRequest, "invalid blob key")
		return
	}
	data, err := blob.NewStore(config.BlobDir(id)).ReadAll(key)
	if err != nil {
		writeErr(w, http.StatusNotFound, "blob not found")
		return
	}
	ct := http.DetectContentType(data)
	if !strings.HasPrefix(ct, "image/") {
		ct = "application/octet-stream"
		w.Header().Set("Content-Disposition", "attachment")
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	http.ServeContent(w, r, key, time.Time{}, bytes.NewReader(data))
}

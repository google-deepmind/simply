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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"amplio/internal/eventstream"
)

// sseTick drives keepalive comments and the overflow → RefetchAll check.
const sseTick = 15 * time.Second

// sseWriteTimeout bounds a single SSE write+flush. Without it a TCP-stalled
// client (slow reader / half-open connection) blocks the stream goroutine in
// Flush() until the OS connection times out, leaking a goroutine per wedged
// client. On a missed deadline the write fails and the handler returns, running
// the deferred sub.Close(). Generous relative to sseTick so a healthy but slow
// link is never tripped.
const sseWriteTimeout = 30 * time.Second

func (s *Server) handleRunStream(w http.ResponseWriter, r *http.Request) {
	s.stream(w, r, r.PathValue("id"))
}

func (s *Server) handleGlobalStream(w http.ResponseWriter, r *http.Request) {
	s.stream(w, r, "") // dashboard: all runs
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request, runID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Disable response buffering on reverse proxies that honor it (nginx /
	// GFE-family, e.g. UberProxy) so each flushed event reaches the client
	// immediately instead of being held in a proxy buffer. A harmless no-op on
	// proxies that ignore it.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	sub := s.bus.Subscribe(runID)
	defer sub.Close()
	ticker := time.NewTicker(sseTick)
	defer ticker.Stop()

	// Per-write deadlines via the ResponseController guard against a stalled
	// client wedging this goroutine in Flush() (see sseWriteTimeout).
	rc := http.NewResponseController(w)

	writeAndFlush := func(emit func()) bool {
		_ = rc.SetWriteDeadline(time.Now().Add(sseWriteTimeout)) // best-effort; nil if unsupported
		emit()
		// Prefer the ResponseController's Flush (returns an error, so a stalled or
		// closed client surfaces here instead of silently wedging the goroutine);
		// fall back to the plain Flusher if unsupported.
		if err := rc.Flush(); err != nil {
			if errors.Is(err, http.ErrNotSupported) {
				flusher.Flush()
				return true
			}
			return false // stalled/closed client: stop and let defer sub.Close() run
		}
		return true
	}

	for {
		select {
		case <-r.Context().Done():
			return
		case ev := <-sub.C():
			if !writeAndFlush(func() { writeSSE(w, ev) }) {
				return
			}
		case <-ticker.C:
			ok := writeAndFlush(func() {
				if sub.TakeOverflow() {
					writeSSE(w, eventstream.RunEvent{Kind: eventstream.KindRefetchAll, RunID: runID, Reason: "overflow"})
				} else {
					fmt.Fprint(w, ": keepalive\n\n") // SSE comment line
				}
			})
			if !ok {
				return
			}
		}
	}
}

func writeSSE(w http.ResponseWriter, ev eventstream.RunEvent) {
	data, err := json.Marshal(ev)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
}

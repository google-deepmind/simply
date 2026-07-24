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

package log

import (
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// HTTPMiddleware wraps a handler to log every request as it finishes. Level
// is mapped from the response status class so the default Info stream stays
// quiet (no 2xx noise) while every 4xx/5xx surfaces automatically:
//
//	2xx → Debug   (visible at --log-level=debug; hidden by default)
//	3xx → Info
//	4xx → Warn
//	5xx → Error
//
// SSE streams (Content-Type: text/event-stream) get one line on close with
// total duration — not one per pushed event. A long-lived SSE that's still
// open never logs at all, by design: the connect happened, the close hasn't,
// and there's nothing useful to say in between. The handler chain's Flush
// passthrough is required so SSE keeps working transparently.
func HTTPMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		logRequest(rec, r, time.Since(start))
	})
}

// statusRecorder captures the response status so the middleware can pick its
// log level without parsing handlers. WriteHeader is recorded once (the
// stdlib drops subsequent calls anyway). Flush passthrough keeps SSE / NDJSON
// handlers working; without it, http.Flusher's type assertion in the inner
// handler would fail.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (s *statusRecorder) WriteHeader(code int) {
	if !s.wroteHeader {
		s.status = code
		s.wroteHeader = true
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Write(b []byte) (int, error) {
	s.wroteHeader = true // implicit 200
	return s.ResponseWriter.Write(b)
}

func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func logRequest(rec *statusRecorder, r *http.Request, dur time.Duration) {
	level := slog.LevelDebug
	switch {
	case rec.status >= 500:
		level = slog.LevelError
	case rec.status >= 400:
		level = slog.LevelWarn
	case rec.status >= 300:
		level = slog.LevelInfo
	}
	msg := "http"
	if strings.HasPrefix(rec.Header().Get("Content-Type"), "text/event-stream") {
		// Only logged after the stream closes (the handler returned), so the
		// duration is the full session length — a meaningful signal.
		msg = "http sse"
	}
	slog.Log(r.Context(), level, msg,
		"method", r.Method,
		"path", r.URL.Path,
		"status", rec.status,
		"ms", dur.Milliseconds(),
	)
}

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

// Package log owns Amplio's process-wide slog setup and the small set of
// helpers that everyone else relies on (level parsing + HTTP middleware).
//
// Why this exists:
//
//   - Without Init, every package's slog.* call falls through to the stdlib
//     default handler — text format, Info level, stderr — which is fine for
//     a hello-world but means our existing slog.Debug calls are silently
//     dropped and operators have no knob to turn for verbosity.
//   - Init replaces the default with an explicit handler whose level lives
//     in a single LevelVar (atomic, safe to mutate from any goroutine), so
//     a future --log-level toggle, a per-test override, or a SIGUSR1-style
//     dynamic bump all share one source of truth.
//
// Callers should invoke Init exactly once, early — typically from cobra's
// PersistentPreRunE in main.go. Subcommands that want to add a destination
// (e.g. serve adding a file alongside stderr) call Init a second time with
// a MultiWriter; later calls fully replace the prior handler.
package log

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
)

// Options describes the handler installed by Init. Zero values default to a
// Info-level text handler on stderr (the previous stdlib behavior, just
// explicit).
type Options struct {
	Level  slog.Level
	Format string    // "text" (default) or "json"
	Writer io.Writer // defaults to os.Stderr
}

// levelVar is the single atomic level shared by every handler Init has ever
// installed. Exposed via SetLevel so dynamic adjustments (a future runtime
// toggle, test helpers) don't need to rebuild the handler.
var levelVar = new(slog.LevelVar)

// Init installs slog's default handler with the given options. Idempotent:
// later calls fully replace the handler — that's how serve adds a file
// destination on top of stderr (Options.Writer = io.MultiWriter(stderr, file)).
func Init(opts Options) {
	if opts.Writer == nil {
		opts.Writer = os.Stderr
	}
	levelVar.Set(opts.Level)
	hopts := &slog.HandlerOptions{Level: levelVar}
	var h slog.Handler
	switch strings.ToLower(opts.Format) {
	case "json":
		h = slog.NewJSONHandler(opts.Writer, hopts)
	default:
		h = slog.NewTextHandler(opts.Writer, hopts)
	}
	slog.SetDefault(slog.New(h))
}

// ParseLevel maps a flag/env string to a slog.Level. Empty string returns
// LevelInfo with no error so callers can pass through unset flags directly.
func ParseLevel(s string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "info":
		return slog.LevelInfo, nil
	case "debug":
		return slog.LevelDebug, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error", "err":
		return slog.LevelError, nil
	}
	return slog.LevelInfo, fmt.Errorf("unknown log level %q (want debug|info|warn|error)", s)
}

// SetLevel updates the active level at runtime. Safe to call from any
// goroutine; visible to every slog.* call thereafter.
func SetLevel(l slog.Level) { levelVar.Set(l) }

// Level returns the currently active level.
func Level() slog.Level { return levelVar.Level() }

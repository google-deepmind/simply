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

package main

import (
	"context"
	"log/slog"

	"amplio/internal/agent/critic"
	"amplio/internal/db"
)

// Run reports: the startup backfill that fills in reports for runs that
// concluded while no critic was watching, and the panic guard both it and the
// live post-run trigger (system.go) call the critic through.

// backfillReports generates any missing run reports for already-concluded runs,
// one at a time. Run in a background goroutine (off the startup path) so it never
// blocks serving; the per-run watermark makes already-reported runs no-ops.
func backfillReports(ctx context.Context, store db.Store, fin *critic.Finalizer) {
	// Limit:-1 = unbounded: backfill must visit every concluded run, not a page.
	runs, _, err := store.ListRuns(ctx, db.ListRunsOpts{Limit: -1})
	if err != nil {
		slog.Warn("report backfill: list runs failed", "error", err)
		return
	}
	for _, run := range runs {
		if ctx.Err() != nil {
			return
		}
		safeFinalize(fin, ctx, run.RunID)
	}
}

// safeFinalize runs the critic for one run with panic recovery, so a single
// bad run (LLM provider crash, panicking deserialization, etc.) doesn't kill
// either the live-trigger goroutine (would crash the server) or the backfill
// loop (would prevent subsequent runs from being processed). The Finalizer's
// own `defer tracker.Unregister(id)` runs ahead of the panic propagation, so
// the ephemeral registry is left clean — recover here only protects the
// caller's goroutine identity.
func safeFinalize(fin *critic.Finalizer, ctx context.Context, runID string) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("critic panicked", "run_id", runID, "panic", r)
		}
	}()
	fin.OnMainAgentConcluded(ctx, runID)
}

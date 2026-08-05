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
	"fmt"

	"amplio/internal/agent/critic"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/db/sqlite"
	"amplio/internal/embed"
	"amplio/internal/eventstream"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	bridgeprovider "amplio/internal/llm/bridge"
	"amplio/internal/observer"
	"amplio/internal/runtime"
	"amplio/internal/skills"
)

// system is the initialized, run-INDEPENDENT object graph shared by every mode
// that hosts runs (serve / headless run / headless resume). It is the living
// counterpart of config.Config: setupSystem performs the one-time bootstrap
// (open DB, wire the manager, build recall, start the observer + finalizer) so
// each mode reduces to "setupSystem(...) + its own thin tail". The RunManager
// then maps a raw RunConfig into a living run on top of this system.
type system struct {
	cfg         config.Config
	store       db.Store
	mgr         *runtime.RunManager
	obs         *observer.Observer
	fin         *critic.Finalizer
	skillIndex  *skills.Index
	embedder    embed.Embedder
	lessonIndex *lessons.Index
	systemHQ    llm.Provider // shared HQ provider (finalizer/observer); reused by serve for the follow-up suggester
	// cleanup tears down everything setupSystem started, in reverse order:
	// drain the observer, reap bridge subprocesses, close the DB. The data-dir
	// lock and signal context are owned by the caller (each mode's entrypoint),
	// not here, so setupSystem stays a pure object-builder.
	cleanup func(ctx context.Context)
}

// systemOpts carries the only variation between modes:
//   - broadcaster: live UI signal sink. serve wires a bus-backed broadcaster;
//     headless leaves it nil (nobody is watching).
//   - liveReports: whether a concluded main-agent auto-triggers report
//     generation via the observer. serve sets true (the finalizer runs in its
//     own panic-recovered goroutine); headless leaves it false and finalizes
//     explicitly after waitForRun, since the observer may exit before
//     processing the conclude. The trigger is built here from the system's own
//     finalizer, so callers don't need a handle to it before it exists.
type systemOpts struct {
	broadcaster eventstream.Broadcaster
	liveReports bool
}

// setupSystem performs the shared, run-independent bootstrap and returns the
// assembled system. The caller is responsible for taking the data-dir lock and
// establishing the signal context BEFORE calling this, and for invoking
// system.cleanup on exit.
func setupSystem(ctx context.Context, cfg config.Config, opts systemOpts) (*system, error) {
	store, err := sqlite.Open(cfg.DB)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}

	// System-tier providers: a process-wide property (not per-run), so build each
	// ONCE here and share the single instance across every consumer below
	// (compaction via the manager, observer summaries, the report finalizer, and
	// the title generator). A bad spec hard-fails startup — consistent for all
	// consumers — rather than silently degrading some.
	systemFast, err := createProvider(cfg.SystemLLMFast)
	if err != nil {
		_ = store.Close()
		return nil, fmt.Errorf("create system_llm_fast: %w", err)
	}
	systemHQ, err := createProvider(cfg.SystemLLMHQ)
	if err != nil {
		_ = store.Close()
		return nil, fmt.Errorf("create system_llm_hq: %w", err)
	}

	mgr := buildManager(store)
	mgr.SetSystemProviders(systemFast, systemHQ)
	bindCLITools(cfg)
	if opts.broadcaster != nil {
		mgr.SetBroadcaster(opts.broadcaster)
	}

	// Recall (skills + lessons) is built before the finalizer/observer so a
	// freshly-launched or recovered run gets recall immediately.
	skillIndex, lessonIndex, embedder := setupRecall(ctx, mgr, store, cfg)

	// Report finalizer: shared by the observer (auto trigger on main-agent
	// conclude) and the operator endpoint. Runs on the shared system-tier HQ
	// provider — the same instance as the observer. Built before the observer so
	// its hook is installed before the observer's workers start.
	fin := critic.NewFinalizer(store, systemHQ, skillIndex, lessonIndex, mgr.EphemeralAgents())

	// Live report trigger runs generation in its own panic-recovered goroutine so
	// it never occupies a summarizer worker and a critic bug can't take the
	// process down. Single-flight + watermark in the finalizer prevent dups.
	var reportTrigger func(ctx context.Context, runID string)
	if opts.liveReports {
		reportTrigger = func(c context.Context, runID string) { go safeFinalize(fin, c, runID) }
	}
	obs := startObserver(ctx, store, systemFast, systemHQ, reportTrigger)

	// Title generator reuses the shared fast provider.
	mgr.SetTitleGenerator(makeTitleGenerator(store, systemFast))

	cleanup := func(ctx context.Context) {
		obs.Stop(ctx)             // drain final summaries before exit
		bridgeprovider.Shutdown() // reap any bridge subprocesses
		_ = store.Close()
	}

	return &system{
		cfg:         cfg,
		store:       store,
		mgr:         mgr,
		obs:         obs,
		fin:         fin,
		skillIndex:  skillIndex,
		embedder:    embedder,
		lessonIndex: lessonIndex,
		systemHQ:    systemHQ,
		cleanup:     cleanup,
	}, nil
}

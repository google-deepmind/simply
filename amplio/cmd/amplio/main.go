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

// Amplio is an agentic framework for long-horizon autonomous research runs.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"os/signal"
	"sort"
	"strings"
	"time"

	"amplio/internal/cli"
	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/embed"
	"amplio/internal/lessons"
	"amplio/internal/llm"
	anthropicprovider "amplio/internal/llm/anthropic"
	geminiprovider "amplio/internal/llm/gemini"
	subprocessprovider "amplio/internal/llm/subprocess"
	amlog "amplio/internal/log"
	"amplio/internal/observer"
	"amplio/internal/runspec"
	"amplio/internal/runtime"
	"amplio/internal/skills"
	"amplio/internal/version"
	"amplio/internal/workspace/resolver"

	// Register agent types.
	_ "amplio/internal/agent/chatbot"
	_ "amplio/internal/agent/standard"

	"github.com/spf13/cobra"
)

// Layered-knob flag vars (root-persistent). Bound in main, read by the
// run-hosting subcommands via resolveConfig, which applies the
// flag > env > config > default precedence through config.Resolve.
var (
	flagSystemLLMHQ   string
	flagSystemLLMFast string
	flagEmbedModel    string
	flagSkillDirs     []string
)

// resolveConfig loads + layers the effective Config for a run-hosting command.
// cmd is needed to distinguish an explicitly-passed --skill-dir (even empty)
// from an absent one, which controls REPLACE-vs-default for the skill list.
func resolveConfig(cmd *cobra.Command) (config.Config, error) {
	return config.Resolve(config.DataDir(), config.Overrides{
		SystemLLMHQ:   flagSystemLLMHQ,
		SystemLLMFast: flagSystemLLMFast,
		EmbedModel:    flagEmbedModel,
		SkillDirs:     flagSkillDirs,
		SkillDirsSet:  cmd.Flags().Changed("skill-dir"),
	})
}

func main() {
	var (
		dataDir   string
		logLevel  string
		logFormat string
	)
	root := &cobra.Command{
		Use:     "amplio",
		Short:   "Agentic framework for long-horizon autonomous research runs",
		Version: version.Build().String(),
		// Resolve the data dir once, before any subcommand reads config or the DB,
		// and install the process-wide logger so EVERY package's slog.* call goes
		// through one handler with one level. Subcommands may re-Init later (e.g.
		// serve adds a file destination) but the level set here is the floor.
		PersistentPreRunE: func(_ *cobra.Command, _ []string) error {
			if dataDir != "" {
				config.SetDataDir(dataDir)
			}
			// Canonicalize $AMPLIO_DATA_DIR to the RESOLVED value (flag > env >
			// default) so child processes — e.g. a background script calling
			// `amplio notify` — target this instance's data dir, even when the
			// inherited env disagreed with our --data-dir or we took the default.
			_ = os.Setenv(config.EnvDataDir, config.DataDir())

			levelStr := firstNonEmpty(logLevel, os.Getenv("AMPLIO_LOG_LEVEL"))
			lvl, err := amlog.ParseLevel(levelStr)
			if err != nil {
				return err
			}
			amlog.Init(amlog.Options{
				Level:  lvl,
				Format: firstNonEmpty(logFormat, os.Getenv("AMPLIO_LOG_FORMAT"), "text"),
				Writer: os.Stderr,
			})
			return nil
		},
		SilenceUsage:  true,
		SilenceErrors: true, // we print + map exit codes ourselves below
	}
	// Cobra's default version template prints "appname version X", which puts
	// the word "version" awkwardly between the name and the actual identity
	// (which itself contains channel + commit + time). Replace with a tighter
	// "amplio <identity>" so the line reads naturally end-to-end.
	root.SetVersionTemplate("{{.Name}} {{.Version}}\n")

	root.PersistentFlags().StringVar(&dataDir, "data-dir", "",
		"Data directory holding config.toml + DB (default $AMPLIO_DATA_DIR or ~/.amplio)")
	root.PersistentFlags().StringVar(&logLevel, "log-level", "",
		"Log level: debug|info|warn|error (env AMPLIO_LOG_LEVEL; default info)")
	root.PersistentFlags().StringVar(&logFormat, "log-format", "",
		"Log format: text|json (env AMPLIO_LOG_FORMAT; default text)")
	root.PersistentFlags().StringVar(&flagSystemLLMHQ, "system-llm-hq", "",
		"System HQ LLM spec for observer summaries/reports (required; env AMPLIO_SYSTEM_LLM_HQ or config system_llm_hq)")
	root.PersistentFlags().StringVar(&flagSystemLLMFast, "system-llm-fast", "",
		"System fast LLM spec for step summaries/compaction (required; env AMPLIO_SYSTEM_LLM_FAST or config system_llm_fast)")
	root.PersistentFlags().StringVar(&flagEmbedModel, "embed-model", "",
		"Embedding model for recall (env AMPLIO_EMBED_MODEL or config embed_model; empty disables recall)")
	root.PersistentFlags().StringArrayVar(&flagSkillDirs, "skill-dir", nil,
		"Skill source directory (repeatable; env AMPLIO_SKILL_DIRS path-list; or config [skills].dirs)")

	root.AddCommand(serveCmd(), notifyCmd(), headlessCmd(), clientCmd())
	if err := root.Execute(); err != nil {
		var ce *codedError
		if errors.As(err, &ce) {
			fmt.Fprintln(os.Stderr, "Error:", ce.msg)
			os.Exit(ce.code)
		}
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func runCmd() *cobra.Command {
	var (
		task      string
		workspace string
		llmSpec   string
		agentType string
	)
	cmd := &cobra.Command{
		Use:   "run [task]",
		Short: "Start a run in this process and wait for it to finish (headless)",
		Long: "Run a task to completion in the foreground. This process owns the data" +
			" directory for its lifetime, so it cannot share a data dir with a running" +
			" `serve` — use `amplio client submit` to hand a task to a running server instead.",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				task = args[0]
			}
			cfg, err := resolveConfig(cmd)
			if err != nil {
				return err
			}
			return executeRun(cfg, task, workspace, llmSpec, agentType)
		},
	}
	cmd.Flags().StringVar(&task, "task", "", "Task description (or pass as a positional arg)")
	cmd.Flags().StringVar(&workspace, "workspace", "", "Working directory (default run.workspace)")
	cmd.Flags().StringVar(&llmSpec, "llm", "", "Agent LLM spec (default run.llm)")
	cmd.Flags().StringVar(&agentType, "agent", "", "Agent type (default run.agent_type)")
	return cmd
}

func executeRun(cfg config.Config, task, workspace, llmSpec, agentType string) error {
	if task == "" {
		return fmt.Errorf("a task is required (positional arg or --task)")
	}
	dataDir := config.DataDir()
	llmSpec = firstNonEmpty(llmSpec, cfg.DefaultLLM())
	if llmSpec == "" {
		return fmt.Errorf("no agent LLM: pass --llm or set run.llms in %s", config.ConfigPath(dataDir))
	}
	workspace = firstNonEmpty(workspace, config.DefaultWorkspace)
	agentType = firstNonEmpty(agentType, config.DefaultAgentType)

	lock, err := lockDataDir(dataDir)
	if err != nil {
		return err
	}
	defer func() { _ = lock.Unlock() }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Shared, run-independent system. Headless: no broadcaster (no UI) and no
	// live report trigger — the run concludes once and we finalize the report
	// explicitly after waitForRun, below.
	sysEnv, err := setupSystem(ctx, cfg, systemOpts{})
	if err != nil {
		return err
	}
	defer sysEnv.cleanup(ctx)
	mgr, fin := sysEnv.mgr, sysEnv.fin

	// Resolve the workspace spec to a concrete path (creation sentinels run
	// their side effects here — the sole fresh-vs-resume difference) and snapshot
	// the operator's AGENTS.md. The OS user is needed locally to resolve a citc
	// workspace.
	wsRoot, agentsMD, err := runspec.Prepare(workspace, os.Getenv("USER"))
	if err != nil {
		return err
	}
	runID, err := mgr.StartRun(ctx, runtime.StartRunConfig{
		RunConfig: config.RunConfig{
			Task:      task,
			Workspace: wsRoot,
			LLM:       llmSpec,
			AgentType: agentType,
			AgentsMD:  agentsMD,
		},
		RootSessionID: config.RootAgentSessionID,
	})
	if err != nil {
		return err
	}
	slog.Info("run started", "run_id", runID, "agent", agentType, "model", llmSpec)
	waitForRun(ctx, mgr, runID)
	// Deterministically produce the run report for this headless run: the observer
	// may exit before processing the conclude. No-op if not autonomous / already
	// reported. Best-effort summaries (the critic falls back to raw events).
	fin.OnMainAgentConcluded(ctx, runID)
	return nil
}

func resumeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "resume <run-id>",
		Short: "Resume a previously interrupted run (headless)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := resolveConfig(cmd)
			if err != nil {
				return err
			}
			return executeResume(cfg, args[0])
		},
	}
	return cmd
}

func executeResume(cfg config.Config, runID string) error {
	dataDir := config.DataDir()

	lock, err := lockDataDir(dataDir)
	if err != nil {
		return err
	}
	defer func() { _ = lock.Unlock() }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Shared, run-independent system. The run's provider/workspace/agent are
	// reconstructed from its stored config, so resume needs no --llm/--workspace
	// flags. System tiers come from current config (the observer + finalizer are
	// process-global, not per-run); the manager resolves the agent provider
	// per-run from run.Config.LLM. Headless: no broadcaster, no live report
	// trigger — we finalize explicitly after waitForRun.
	sysEnv, err := setupSystem(ctx, cfg, systemOpts{})
	if err != nil {
		return err
	}
	defer sysEnv.cleanup(ctx)
	store, mgr, fin := sysEnv.store, sysEnv.mgr, sysEnv.fin

	run, err := store.GetRun(ctx, runID)
	if err != nil {
		return fmt.Errorf("run %q not found: %w", runID, err)
	}

	revived, err := mgr.RecoverRun(ctx, runID)
	if err != nil {
		return fmt.Errorf("recover run: %w", err)
	}
	if revived == 0 {
		slog.Info("run is already at rest; nothing to resume", "run_id", runID)
		return nil
	}
	slog.Info("run resumed", "run_id", runID, "sessions", revived, "model", run.Config.LLM)
	waitForRun(ctx, mgr, runID)
	fin.OnMainAgentConcluded(ctx, runID) // deterministic report for this headless resume
	return nil
}

// firstNonEmpty returns the first non-empty string, or "" if all are empty.
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// buildManager wires a run manager and its commit notifier over a store. The
// manager resolves each run's agent provider from its RunConfig.LLM via
// createProvider, so different runs can use different models.
// setupRecall builds the skill and lesson recall indexes from config, installs
// them on the manager, and returns them (nil when unavailable). Lessons build
// SYNCHRONOUSLY (DB-only — fast: ~ms).
//
// Skills use a two-stage build to keep startup snappy without sacrificing
// recall availability:
//
//  1. LoadCached (sync, ~ms): hydrate the in-memory Index from the DB cache.
//     IsBuilt() returns true immediately and Search/Load serve from the cached
//     snapshot, so freshly-spawned agents have full recall right away.
//  2. Reconcile (background, slow): full file scan against the skill source
//     tree, re-embed any changed/new skills, atomic-swap in the fresh index.
//     The skill tree often lives on a network FS where per-file reads are
//     tens of seconds cold; doing this in background means startup is
//     responsive even on a slow-srcfs day.
//
// Cold-start exception: if LoadCached found nothing (empty cache), we run
// Build synchronously instead — backgrounding would leave Search returning
// empty results for the entire scan duration, which is a much worse UX than
// the one-time wait. Subsequent restarts hit the warm cache and are instant.
//
// Degrades to no recall (logged) when the embedder is unavailable (e.g. no
// Vertex creds).
// recallSubsystem captures whether one recall subsystem (skills or knowledge)
// came up, with a short human-readable detail (a count, or why it's disabled).
type recallSubsystem struct {
	enabled bool
	detail  string
}

func recallEnabled(detail string) recallSubsystem {
	return recallSubsystem{enabled: true, detail: detail}
}
func recallDisabled(reason string) recallSubsystem {
	return recallSubsystem{enabled: false, detail: reason}
}

// recallStatus is the startup outcome of the recall subsystem, rendered once by
// printRecallStatus so the operator can see at a glance whether skills +
// knowledge are live.
type recallStatus struct {
	embedModel string
	skills     recallSubsystem
	knowledge  recallSubsystem
}

// printRecallStatus writes an operator-facing block mirroring cli.PrintStatus:
// a green ✓ for a live subsystem, a yellow ✗ + reason for a disabled one. Color
// is used only when w is a TTY and NO_COLOR is unset.
func printRecallStatus(w io.Writer, st recallStatus) {
	const (
		reset  = "\x1b[0m"
		green  = "\x1b[32m"
		yellow = "\x1b[33m"
	)
	color := false
	if f, ok := w.(*os.File); ok && os.Getenv("NO_COLOR") == "" {
		if fi, err := f.Stat(); err == nil && fi.Mode()&os.ModeCharDevice != 0 {
			color = true
		}
	}
	paint := func(s, code string) string {
		if !color {
			return s
		}
		return code + s + reset
	}
	line := func(name string, sub recallSubsystem) {
		if sub.enabled {
			msg := "available"
			if sub.detail != "" {
				msg += " (" + sub.detail + ")"
			}
			fmt.Fprintf(w, "  %s %-9s %s\n", paint("✓", green), name, msg)
		} else {
			fmt.Fprintf(w, "  %s %-9s %s\n", paint("✗", yellow), name, paint("disabled — "+sub.detail, yellow))
		}
	}
	header := "Recall subsystem (agent skill + knowledge search):"
	if st.embedModel != "" {
		header = fmt.Sprintf("Recall subsystem (embed model %s):", st.embedModel)
	}
	fmt.Fprintln(w, header)
	line("skills", st.skills)
	line("knowledge", st.knowledge)
}

func setupRecall(ctx context.Context, mgr *runtime.RunManager, store db.Store, cfg config.Config) (*skills.Index, *lessons.Index) {
	var st recallStatus
	defer func() { printRecallStatus(os.Stderr, st) }()

	embedModel := cfg.EmbedModelOrDefault()
	if embedModel == "" {
		// No embedder configured: recall needs embeddings, so it's disabled.
		// Set --embed-model / $AMPLIO_EMBED_MODEL / embed_model.
		st.skills = recallDisabled("no embed model configured")
		st.knowledge = recallDisabled("no embed model configured")
		return nil, nil
	}
	st.embedModel = embedModel
	embedder, err := createEmbedder(ctx, embedModel)
	if err != nil {
		reason := fmt.Sprintf("embedder unavailable: %s", err)
		st.skills = recallDisabled(reason)
		st.knowledge = recallDisabled(reason)
		return nil, nil
	}

	// Lessons ("knowledge"): synchronous; reads from the DB, populated by
	// end-of-run mining.
	lessonIx := lessons.NewIndex(store, embedder)
	if err := lessonIx.Build(ctx); err != nil {
		st.knowledge = recallDisabled(fmt.Sprintf("index build failed: %s", err))
	} else {
		st.knowledge = recallEnabled("")
	}
	mgr.SetLessonIndex(lessonIx)

	// Skills: only when dirs are configured.
	dirs := cfg.SkillDirs()
	var skillIx *skills.Index
	if len(dirs) == 0 {
		st.skills = recallDisabled("no skill dirs configured")
		return skillIx, lessonIx
	}
	sources := make([]skills.Source, 0, len(dirs))
	for _, d := range dirs {
		sources = append(sources, skills.Source{Name: d, Path: d, Blocked: cfg.Skills.Blocked})
	}
	skillIx = skills.NewIndex(sources, embedder, skills.NewDBCache(store))
	mgr.SetSkillIndex(skillIx)

	// Stage 1: hydrate from cache. Fast even with hundreds of skills.
	hydrated := skillIx.LoadCached(ctx)
	if hydrated == 0 {
		// Cold start: no cache to lean on. Block — empty Search results would
		// be worse than the one-time wait, and this only happens on the very
		// first run (or after a manual DB wipe).
		slog.Info("skill cache empty; indexing synchronously (cold start)", "dirs", dirs)
		if err := skillIx.Build(ctx); err != nil {
			st.skills = recallDisabled(fmt.Sprintf("index build failed: %s", err))
		} else {
			st.skills = recallEnabled(fmt.Sprintf("%d skills", skillIx.Size()))
		}
		return skillIx, lessonIx
	}

	// Stage 2: background reconcile against the on-disk corpus. During this
	// window the index serves cached results (possibly slightly stale for
	// skills whose SKILL.md changed since the last run); the atomic swap at
	// Build's end refreshes everything.
	st.skills = recallEnabled(fmt.Sprintf("%d skills, reconciling in background", hydrated))
	go func() {
		start := time.Now()
		if err := skillIx.Build(ctx); err != nil {
			slog.Error("skill reconcile failed; serving stale cached index", "error", err)
			return
		}
		slog.Info("skill index reconciled", "elapsed", time.Since(start).Round(time.Second))
	}()
	return skillIx, lessonIx
}

// bindCLITools prepends the configured amplio bin dirs to $PATH (so shipped 1p
// CLI tools resolve by bare name for our probes and the agent's bash subprocess)
// and warns the operator once about any optional tool that isn't installed.
func bindCLITools(cfg config.Config) {
	cli.BindPaths(cfg.BinPaths())
	cli.PrintStatus(os.Stderr, cli.All())
}

func buildManager(store db.Store) *runtime.RunManager {
	mgr := runtime.NewRunManager(store, createProvider, runtime.NewRunRegistry(), resolver.Wrap)
	store.SetCommitListener(runtime.NewCommitNotifier(mgr.RunRegistry(), mgr.RespawnSession))
	return mgr
}

// makeTitleGenerator returns a fire-and-forget run-title generator: it asks the
// fast system model for a short title and writes it via UpdateRunTitle. Failure
// is logged and ignored — the title just stays empty and the UI falls back to a
// task prefix.
const titleSystemPrompt = "You are titling a task for a list UI. Output ONLY a short, " +
	"specific title of 3-8 words. Do NOT write code, explanations, quotes, or markdown " +
	"fences — respond with the bare title text only. Do NOT try to work on the task."

func makeTitleGenerator(store db.Store, fast llm.Provider) func(runID, task string) {
	return func(runID, task string) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		resp, err := fast.Call(ctx, llm.Request{
			SystemPrompt: titleSystemPrompt,
			// Frame the task as data (not an instruction to execute) and repeat
			// the ask in the user turn — weaker models otherwise try to do it.
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: "Write a short title for the following task. Do not perform it.\n\nTASK:\n" + task,
			}},
		})
		if err != nil {
			slog.Warn("title generation failed", "run_id", runID, "error", err)
			return
		}
		title := sanitizeTitle(resp.Content)
		if title == "" {
			// Model returned code/empty/unusable: leave the title empty (the UI
			// falls back to a task prefix) but log so it's visible.
			slog.Warn("title generation produced no usable title",
				"run_id", runID, "raw", truncate(resp.Content, 120))
			return
		}
		if err := store.UpdateRunTitle(ctx, runID, title); err != nil {
			slog.Warn("title update failed", "run_id", runID, "error", err)
			return
		}
		slog.Info("generated run title", "run_id", runID, "title", title)
	}
}

// sanitizeTitle extracts a clean single-line title from a model response, or ""
// if the response looks like code or yields nothing usable.
func sanitizeTitle(raw string) string {
	s := strings.TrimSpace(raw)
	// A leading fence means the model wrote a code block, not a title.
	if s == "" || strings.HasPrefix(s, "```") {
		return ""
	}
	line := s
	if i := strings.IndexByte(line, '\n'); i >= 0 {
		line = line[:i] // first line only
	}
	line = strings.TrimSpace(line)
	line = strings.TrimLeft(line, "#>*-+ \t") // markdown heading / list markers
	line = strings.Trim(line, "`\"'")         // surrounding fences / quotes
	line = strings.TrimSpace(line)
	if r := []rune(line); len(r) > 80 {
		line = strings.TrimSpace(string(r[:80]))
	}
	return line
}

// truncate shortens s to at most n runes for logging.
func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}

// startObserver installs the run-report finalizer (before starting, so worker
// goroutines never race the field) and launches the process-global observer on
// the shared system-tier providers. The caller Stops it (drains pending
// summaries) before exit. finalizer may be nil (no report generation).
func startObserver(ctx context.Context, store db.Store, fast, hq llm.Provider, finalizer func(context.Context, string)) *observer.Observer {
	obs := observer.New(store, fast, hq, observer.DefaultWorkers)
	obs.SetFinalizer(finalizer)
	obs.Start(ctx)
	return obs
}

// waitForRun blocks until the run goes inactive, cancelling it on interrupt.
func waitForRun(ctx context.Context, mgr *runtime.RunManager, runID string) {
	reg := mgr.RunRegistry()
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			slog.Info("interrupted, cancelling run")
			reg.CancelAll()
			time.Sleep(time.Second)
			return
		case <-ticker.C:
			if !reg.IsRunActive(runID) {
				return
			}
		}
	}
}

// defaultMaxOutputTokens caps generation for every provider. Generous because
// coding agents emit long replies/edits; thinking budgets are separate.
const defaultMaxOutputTokens = 65536

// providerFactory builds a provider from a model and optional spec args (the
// `?k=v` query, passed through to the model — e.g. thinking controls).
type providerFactory func(model string, maxTokens int, args url.Values) (llm.Provider, error)

// providerRegistry maps an LLM spec prefix (the provider "class") to its
// constructor. A spec is "<prefix>:<model>[?k=v&...]"; the provider parses the
// model and forwards the args (each family interprets them, e.g. thinking).
var providerRegistry = map[string]providerFactory{
	"vertex-claude": anthropicprovider.NewVertex, // Claude on Vertex AI (ADC)
	"claude":        anthropicprovider.NewAPIKey, // Claude direct Anthropic API (ANTHROPIC_API_KEY)
	"vertex-gemini": geminiprovider.NewVertex,    // Gemini on Vertex AI (ADC)
	"gemini":        geminiprovider.NewAPIKey,    // Gemini Developer API (API key)
	"subprocess":    subprocessprovider.New,      // out-of-process bridge (e.g. Beyond); see bridges/README.md
}

// createEmbedder builds an Embedder from a "<backend>:<model>" spec, mirroring
// createProvider. Backends: "vertex" (ADC, project-based) and "gemini" (Gemini
// Developer API, key-based). A bare model name (no ":") defaults to the vertex
// backend for back-compat with older embed_model config values. Note model
// availability is backend-specific (e.g. text-embedding-005 is Vertex-only;
// gemini-embedding-001 works on both).
func createEmbedder(ctx context.Context, spec string) (embed.Embedder, error) {
	backend, model, ok := strings.Cut(spec, ":")
	if !ok {
		backend, model = "vertex", spec // bare name => vertex (back-compat)
	}
	if model == "" {
		return nil, fmt.Errorf("invalid embed model spec %q; want <backend>:<model>", spec)
	}
	switch backend {
	case "vertex":
		return embed.NewVertex(ctx, model)
	case "gemini":
		return embed.NewAPIKey(ctx, model)
	default:
		return nil, fmt.Errorf("unknown embed backend %q in %q; known: gemini, vertex", backend, spec)
	}
}

func createProvider(spec string) (llm.Provider, error) {
	prefix, rest, ok := strings.Cut(spec, ":")
	if !ok || rest == "" {
		return nil, fmt.Errorf("invalid LLM spec %q; want <provider>:<model>[?k=v&...]", spec)
	}
	model, rawArgs, _ := strings.Cut(rest, "?")
	if model == "" {
		return nil, fmt.Errorf("invalid LLM spec %q; empty model", spec)
	}
	args, err := url.ParseQuery(rawArgs)
	if err != nil {
		return nil, fmt.Errorf("parse LLM spec args in %q: %w", spec, err)
	}
	factory, ok := providerRegistry[prefix]
	if !ok {
		return nil, fmt.Errorf("unknown LLM provider %q in %q; known: %s", prefix, spec, knownProviders())
	}
	return factory(model, defaultMaxOutputTokens, args)
}

func knownProviders() string {
	keys := make([]string, 0, len(providerRegistry))
	for k := range providerRegistry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

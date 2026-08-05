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
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/embed"
	"amplio/internal/eventstream"
	"amplio/internal/llm"
	"amplio/internal/llm/bridge"
	amlog "amplio/internal/log"
	"amplio/internal/runtime"
	"amplio/internal/server"
	"amplio/internal/sysstat"

	"github.com/spf13/cobra"
)

func serveCmd() *cobra.Command {
	var listen, lendLLM string
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start the amplio web server (hosts runs and the UI)",
		Long: "Start the long-lived server: it owns the data directory (DB, observer)," +
			" recovers interrupted runs, and serves the API + UI. Settings come from" +
			" <data-dir>/config.toml (see --data-dir); the bind address may also be set" +
			" via --listen or $AMPLIO_LISTEN.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := resolveConfig(cmd)
			if err != nil {
				return err
			}
			return executeServe(cfg, listen, lendLLM)
		},
	}
	cmd.Flags().StringVar(&listen, "listen", "",
		"HTTP bind address host:port; overrides $AMPLIO_LISTEN and config.toml [listen]")
	cmd.Flags().StringVar(&lendLLM, "lend-llm", "",
		"bind address host:port for the LLM lending listener (empty disables); "+
			"overrides $AMPLIO_LEND_LLM and config.toml [lend_llm]")
	return cmd
}

// listenAddr applies the serve bind-address precedence: --listen flag >
// $AMPLIO_LISTEN > config.toml [listen] (cfgListen already holds the file value
// or the built-in default). The lending listener uses the same rule, via
// addrFrom.
func listenAddr(flagVal, envVal, cfgListen string) string {
	return addrFrom(flagVal, envVal, cfgListen)
}

// addrFrom is flag > env > config, the precedence every address setting uses.
func addrFrom(flagVal, envVal, cfgVal string) string {
	switch {
	case flagVal != "":
		return flagVal
	case envVal != "":
		return envVal
	default:
		return cfgVal
	}
}

func executeServe(cfg config.Config, listenOverride, lendOverride string) error {
	dataDir := config.DataDir()
	// System tiers are already validated by resolveConfig. --listen has its own
	// serve-only precedence (flag > env > resolved config). serve additionally
	// needs the agent model menu for the manager provider + UI default.
	cfg.Listen = listenAddr(listenOverride, os.Getenv(config.EnvListen), cfg.Listen)
	if cfg.DefaultLLM() == "" {
		return fmt.Errorf("no agent model: set run.llms in %s", config.ConfigPath(dataDir))
	}

	// Take the data-dir owner lock before touching the DB so we never run two
	// servers (or a server + headless run) against one DB.
	lock, err := lockDataDir(dataDir)
	if err != nil {
		return err
	}
	defer func() { _ = lock.Unlock() }()

	// Mirror logs to a timestamped file on disk so operators can grep for
	// failures after a serve session ends without needing to have tailed the
	// terminal. The level set by --log-level / AMPLIO_LOG_LEVEL still
	// governs both writers. Re-Init is intentional: it replaces the
	// stderr-only handler the root PreRunE installed.
	logFile, err := openLogFile()
	if err != nil {
		// Non-fatal: stderr still works. Log via the existing stderr handler
		// so the operator notices.
		slog.Warn("could not open log file; logging to stderr only", "error", err)
	} else {
		defer func() { _ = logFile.Close() }()
		amlog.Init(amlog.Options{
			Level:  amlog.Level(),
			Format: os.Getenv("AMPLIO_LOG_FORMAT"),
			Writer: io.MultiWriter(os.Stderr, logFile),
		})
		slog.Info("logging to file", "path", logFile.Name())
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Live UI signal fabric: built before setupSystem so the manager broadcasts
	// token/status deltas from the moment runs are recovered.
	bus := eventstream.NewBus()
	bc := eventstream.NewBusBroadcaster(bus)

	// Shared, run-independent system: DB, manager, recall, observer (with a live
	// report trigger), finalizer, title generator. serve is the only mode with a
	// broadcaster and live reports.
	sysEnv, err := setupSystem(ctx, cfg, systemOpts{broadcaster: bc, liveReports: true})
	if err != nil {
		return err
	}
	defer sysEnv.cleanup(ctx)
	store, mgr, fin := sysEnv.store, sysEnv.mgr, sysEnv.fin
	skillIndex, lessonIndex := sysEnv.skillIndex, sysEnv.lessonIndex

	// Publish an ephemeral_agents event whenever a non-session worker (the
	// critic's report generator, context compaction) starts or ends. The event
	// carries enough to update the UI directly (kind + subject session + whether
	// it just became active), so a page can toggle e.g. a per-session
	// "compacting…" indicator without a refetch; it also doubles as a structural
	// refetch hint for the run-overview "Generating report…" state.
	mgr.EphemeralAgents().SetOnChange(func(ag runtime.EphemeralAgent, active bool) {
		bus.Publish(eventstream.RunEvent{
			Kind:          eventstream.KindEphemeralAgents,
			RunID:         ag.RunID,
			SessionID:     ag.Subject, // the session the work targets ("" = run-level)
			EphemeralKind: ag.Kind,    // "report" | "compaction"
			Active:        active,     // true = started, false = ended
		})
	})

	// Wire any build-specific extras. Stub in the OSS build.
	setupCorpExtras(bc)

	// System status watcher: one goroutine probes server-host signals
	// (cpu/mem/swap/load; credential probe in the internal build). Snapshot
	// updates publish on the SSE bus so all connected tabs share one probe —
	// no per-tab polling.
	sys := sysstat.New(func(s sysstat.Snapshot) {
		bc.SysStat(map[string]any{
			"credential_seconds": s.CredentialSeconds,
			"load_avg_1m":        s.LoadAvg1m,
			"cpu_pct":            s.CPUPercent,
			"mem_pct":            s.MemPercent,
			"swap_pct":           s.SwapPercent,
		})
	})
	go sys.Run(ctx)

	// Bind first: a port conflict should fail before we recover runs or print a URL.
	ln, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return fmt.Errorf("listen %s: %w", cfg.Listen, err)
	}

	// Revive runs interrupted by a prior process. Skills are already built (above,
	// before bind), so recovered runs get recall just like new ones.
	//
	// recoverRuns SCHEDULES respawns (each RespawnSession launches a goroutine and
	// returns); the memory ramp happens asynchronously AFTER this returns. We log
	// the scheduled count here as a coarse signal for diagnosing the suspected
	// recovery-driven OOM (see oom_deepdive); detailed memory sampling can be added
	// later if the count implicates recovery.
	recovered := recoverRuns(ctx, store, mgr)
	slog.Info("recovery scheduled", "sessions", recovered)

	// Backfill any missing reports for already-concluded runs, in the background
	// (off the startup path) and sequentially (one at a time, to bound HQ-LLM
	// load). Idempotent: runs already covered by a report are cheap no-ops.
	go backfillReports(ctx, store, fin)

	// Web auth: reads are open (the readonly share view); writes need this token,
	// presented as an HttpOnly cookie (browser, after the magic-link exchange) or
	// a Bearer header (CLI). An explicit config token wins; otherwise we use a
	// durable per-data-dir token so the cookie survives restarts.
	token := cfg.Token
	if token == "" {
		var terr error
		if token, terr = loadOrCreateToken(dataDir); terr != nil {
			return terr
		}
	}
	// TLS auto-detection: <data-dir>/cert.pem + key.pem if present; otherwise
	// try `mkcert` to generate them; otherwise fall back to plain HTTP.
	// HTTPS unlocks HTTP/2 (Go's net/http auto-negotiates on ServeTLS), which
	// removes the browser's per-origin 6-connection cap that throttles SSE
	// across multi-tab sessions. See docs/tls.md.
	certFile, keyFile, terr := resolveTLS(ctx, dataDir)
	if terr != nil {
		slog.Warn("TLS setup failed; serving plain HTTP", "error", terr)
	}
	useTLS := certFile != "" && keyFile != ""
	scheme := "http"
	if useTLS {
		scheme = "https"
	}

	// Banner URLs: one or two depending on bind (see bannerHosts). The first
	// is the "primary" — what gets recorded for the CLI clients and shown in
	// readonly share links. Subsequent entries (only for wildcard binds) are
	// local-access alternatives for the same machine / SSH-tunnel users.
	hosts := bannerHosts(ln.Addr())
	baseURLs := make([]string, len(hosts))
	for i, h := range hosts {
		baseURLs[i] = scheme + "://" + h
	}

	// localURL: a loopback base for same-machine CLI clients (submit, notify),
	// avoiding the banner host (which may be a non-loopback FQDN behind a
	// reverse proxy and unreachable from this process).
	localURL := baseURLs[0]
	if tcp, ok := ln.Addr().(*net.TCPAddr); ok {
		localURL = fmt.Sprintf("%s://127.0.0.1:%d", scheme, tcp.Port)
	}
	if err := writeServerInfo(dataDir, serverInfo{PID: os.Getpid(), URL: baseURLs[0], Addr: localURL, Token: token}); err != nil {
		return err
	}
	defer func() { _ = os.Remove(serverInfoPath(dataDir)) }() // best-effort; stale is harmless

	bannerURLs := make([]string, len(baseURLs))
	for i, base := range baseURLs {
		if token != "" {
			bannerURLs[i] = fmt.Sprintf("%s/?token=%s", base, token)
		} else {
			bannerURLs[i] = base + "/"
		}
	}

	srv := server.New(ctx, store, mgr, bus, token, server.RunDefaults{
		LLM:           cfg.DefaultLLM(),
		LLMs:          cfg.Run.LLMs,
		SystemLLMHQ:   cfg.SystemLLMHQ,
		SystemLLMFast: cfg.SystemLLMFast,
	})
	srv.SetReportGenerator(fin.Generate)
	srv.SetRecall(skillIndex, lessonIndex)
	srv.SetSysStat(sys)
	srv.SetOwner(os.Getenv("USER"))
	srv.SetCookieName(authCookieName(dataDir))
	srv.SetSecureCookie(useTLS)
	srv.SetLLMTester(testLLM)
	srv.SetFollowupSuggester(makeFollowupSuggester(store, sysEnv.systemHQ))
	if lendAddr := addrFrom(lendOverride, os.Getenv(config.EnvLendLLM), cfg.LendLLM); lendAddr != "" {
		stopLending, err := serveLending(lendAddr, cfg, srv, sysEnv.embedder)
		if err != nil {
			return err
		}
		defer stopLending()
	}
	go srv.Bridge(ctx)

	// HTTP log middleware: quiet by default (2xx → Debug, hidden at info level),
	// chatty on failures (4xx Warn, 5xx Error), one-line-on-close for SSE. Cost
	// is ~10 µs/request — negligible at our scale.
	handler := amlog.HTTPMiddleware(srv.Handler())
	httpServer := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shCtx)
	}()

	fmt.Println()
	for i, u := range bannerURLs {
		if i == 0 {
			fmt.Printf("  amplio → %s\n", u)
		} else {
			// Indent so the arrows align: "  amplio →" is 10 display columns
			// (2 spaces + 6 letters + space + arrow), so 9 spaces + arrow
			// lines up the second-line arrow under the first.
			fmt.Printf("         → %s  (local / SSH tunnel)\n", u)
		}
	}
	fmt.Println()
	slog.Info("amplio serving", "addr", ln.Addr().String(), "tls", useTLS, "auth", token != "")
	var serveErr error
	if useTLS {
		// Go's net/http auto-enables HTTP/2 on ServeTLS (NextProtos includes
		// "h2" by default). Browsers connect over HTTP/2, multiplexing all
		// SSE + XHR streams onto one TCP connection — no per-origin
		// 6-connection cap in play.
		serveErr = httpServer.ServeTLS(ln, certFile, keyFile)
	} else {
		serveErr = httpServer.Serve(ln)
	}
	if serveErr != nil && serveErr != http.ErrServerClosed {
		return fmt.Errorf("http server: %w", serveErr)
	}
	return nil
}

// bannerHosts returns the host:port strings to print on the startup banner.
// Three cases:
//   - Wildcard bind (0.0.0.0 or ::): two entries — the machine hostname
//     (shareable with teammates on the same network) AND "localhost" (for
//     the same machine or an SSH tunnel). The user picks whichever fits
//     their access path. Collapses to one entry if the hostname lookup
//     fails or already returns "localhost".
//   - Loopback bind (127.x.x.x or ::1): one entry — "localhost", which is
//     friendlier than the raw IP and matches what a local browser or
//     SSH-tunnel browser would type.
//   - Specific external IP: one entry — the IP as-is.
func bannerHosts(addr net.Addr) []string {
	tcp, ok := addr.(*net.TCPAddr)
	if !ok {
		return []string{addr.String()}
	}
	port := strconv.Itoa(tcp.Port)
	switch {
	case tcp.IP == nil || tcp.IP.IsUnspecified():
		hostname := "localhost"
		if hn, err := os.Hostname(); err == nil && hn != "" {
			hostname = hn
		}
		if hostname == "localhost" {
			return []string{net.JoinHostPort("localhost", port)}
		}
		return []string{
			net.JoinHostPort(hostname, port),
			net.JoinHostPort("localhost", port),
		}
	case tcp.IP.IsLoopback():
		return []string{net.JoinHostPort("localhost", port)}
	default:
		return []string{net.JoinHostPort(tcp.IP.String(), port)}
	}
}

// recoverRuns revives runs interrupted by a prior process and returns the total
// number of sessions it SCHEDULED for respawn (each respawn runs asynchronously,
// so this count is the driver of the post-return memory ramp, not a measure of
// peak memory itself).
func recoverRuns(ctx context.Context, store db.Store, mgr *runtime.RunManager) int {
	// Limit:-1 = unbounded: recovery must consider EVERY interrupted run, not a
	// page (pagination is a UI concern, not an operational one).
	runs, _, err := store.ListRuns(ctx, db.ListRunsOpts{Limit: -1})
	if err != nil {
		slog.Warn("recover: list runs failed", "error", err)
		return 0
	}
	total := 0
	for _, run := range runs {
		if n, err := mgr.RecoverRun(ctx, run.RunID); err != nil {
			slog.Warn("recover run failed", "run_id", run.RunID, "error", err)
		} else if n > 0 {
			slog.Info("recovered run", "run_id", run.RunID, "sessions", n)
			total += n
		}
	}
	return total
}

// openLogFile creates a fresh per-serve log file under config.LogsDir() with
// a timestamped name (one per serve invocation, so restarts don't append to a
// stale tail). Append mode is harmless given the timestamp grain, but kept
// for the edge case where two serves on different data-dirs race to the same
// path — rare, but cheap to handle. Caller is responsible for Close().
func openLogFile() (*os.File, error) {
	dir := config.LogsDir()
	name := fmt.Sprintf("amplio-%s.log", time.Now().Format("20060102-150405"))
	return os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
}

// serveLending starts the LLM lending listener and returns a stop function.
//
// A second listener rather than routes on the main one. This port serves generations, embeddings
// and the model list, and nothing else exists on it to reach.
func serveLending(addr string, cfg config.Config, srv *server.Server, embedder embed.Embedder) (func(), error) {
	tokenEnv := cfg.LendLLMTokenEnv
	if tokenEnv == "" {
		// The same default the bridge provider reads, so both ends of a tunnel
		// are configured alike.
		tokenEnv = bridge.DefaultTokenEnv
	}
	token := os.Getenv(tokenEnv)
	if token == "" {
		// Refusing beats defaulting: an open lending port is an invitation to
		// spend someone else's model budget. Its own secret, too — the server
		// token also authorises starting runs, i.e. a shell here.
		return nil, fmt.Errorf("lend_llm is set but %s is empty; set it to a shared secret "+
			"that callers will present as a bearer token", tokenEnv)
	}
	// Cached: a served API builds a provider per request, and construction mints
	// credentials and builds the connection pool that makes the second request
	// fast. A run builds one provider and uses it for hours.
	handler := srv.LendingHandler(token, llm.CacheProviders(createProvider), lentEmbedder(embedder))

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("listen for lending on %s: %w", addr, err)
	}
	lendSrv := &http.Server{
		Handler: handler,
		// No write timeout: a generation streams for minutes. Reading headers is
		// the part an idle connection can abuse.
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		if err := lendSrv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("lending listener stopped", "error", err)
		}
	}()
	slog.Warn("lending LLM: authenticated callers on this port may spend this machine's model credentials",
		"addr", ln.Addr().String(), "models", len(cfg.Run.LLMs), "token_env", tokenEnv)

	return func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = lendSrv.Shutdown(shutdownCtx)
	}, nil
}

// lentEmbedder maps a nil embed.Embedder to a nil interface: an interface value
// holding a nil pointer passes a != nil check and then panics on first use.
func lentEmbedder(e embed.Embedder) bridge.Embedder {
	if e == nil {
		return nil
	}
	return e
}

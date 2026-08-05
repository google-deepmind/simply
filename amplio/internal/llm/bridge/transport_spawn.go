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

package bridge

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	healthTimeout = 30 * time.Second      // max wait for a fresh bridge to bind + answer /health (corp PARs can be slow to cold-start)
	healthPoll    = 50 * time.Millisecond // gap between /health probes during startup
)

// Both cooldowns are vars so tests can lower them.
var (
	// restartCooldown is the minimum gap between a successful start and a
	// subsequent restart, to absorb the worst of a hot death-loop when a
	// bridge crashes immediately after every request.
	restartCooldown = 2 * time.Second
	// startCooldown is the minimum gap between a *failed* start and the next
	// start attempt. Without it, every Input-class event that wakes a session
	// whose bridge is permanently broken (stale blaze-bin, missing deps,
	// port collision, …) would burn another ~healthTimeout's worth of
	// polling before failing. With it, repeated pokes get a fast, cached
	// error pointing at the original cause until the user fixes the bridge.
	startCooldown = 10 * time.Second
)

// extraSpawnEnv is appended to every bridge subprocess's environment. Empty in
// production; the test suite sets it (e.g. AMPLIO_TEST_BRIDGE=1) so the test
// binary, when re-spawned as a bridge, can take over its init() and serve the
// protocol — letting tests exercise the real exec path without a separate cmd/.
var extraSpawnEnv []string

// manager owns bridge subprocesses, one per spec key, reused across runs and
// goroutines for the process lifetime. A spec key is the bridge path plus its
// canonical spec args, so distinct models/params get distinct processes while
// identical specs share one (the reuse the design calls for).
type manager struct {
	mu    sync.Mutex
	procs map[string]*proc
}

var defaultManager = &manager{procs: make(map[string]*proc)}

// spawnTransport is the transport for a bridge THIS process owns: it spawns the
// binary on demand, talks to it over a private unix socket, and restarts it once
// if the connection fails — which is the normal way a crashed bridge presents,
// since the socket goes away with the process.
type spawnTransport struct {
	key       string // reuse key: binary + canonical args
	binary    string
	specQuery string // canonical args, forwarded to the bridge as AMPLIO_BRIDGE_SPEC
}

func (t *spawnTransport) describe() string { return t.binary }

func (t *spawnTransport) post(ctx context.Context, path string, body []byte) (*http.Response, error) {
	client, err := defaultManager.client(t.key, t.binary, t.specQuery)
	if err != nil {
		return nil, err
	}
	resp, err := postTo(ctx, client, "http://unix"+path, body)
	if err != nil {
		if ctx.Err() != nil {
			return nil, err // caller cancelled; don't churn the subprocess
		}
		client, rerr := defaultManager.restartClient(t.key, client)
		if rerr != nil {
			return nil, fmt.Errorf("subprocess: bridge restart failed: %w (original: %v)", rerr, err)
		}
		resp, err = postTo(ctx, client, "http://unix"+path, body)
		if err != nil {
			return nil, fmt.Errorf("subprocess: request failed after restart: %w", err)
		}
	}
	return resp, nil
}

// postTo issues one protocol request. Shared by every transport: only the client
// and the URL differ.
func postTo(ctx context.Context, client *http.Client, url string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(protocolVersionHeader, protocolVersion)
	// Ownership of the response body transfers to the caller (closed in Call's
	// defer or subStream.Close).
	return client.Do(req) //nolint:bodyclose
}

// Shutdown kills every running bridge and removes its socket dir. Call on
// graceful amplio exit; Pdeathsig backstops the crash case.
func Shutdown() { defaultManager.shutdown() }

func (m *manager) shutdown() {
	m.mu.Lock()
	all := make([]*proc, 0, len(m.procs))
	for _, p := range m.procs {
		all = append(all, p)
	}
	m.mu.Unlock()
	for _, p := range all {
		p.mu.Lock()
		p.stopLocked()
		p.mu.Unlock()
	}
}

type proc struct {
	key       string
	binary    string
	specQuery string

	// stderrTail keeps the bridge's last few stderr lines so a startup failure or
	// an unexpected exit can SHOW them: everything logPipe writes goes to Debug,
	// which the default level drops.
	stderrTail *ringBuffer

	mu        sync.Mutex // serializes start/stop/restart and guards the fields below
	cmd       *exec.Cmd
	client    *http.Client
	tmpDir    string
	alive     bool
	startedAt time.Time // when alive last flipped to true

	// Sticky last-failure state. lastFailErr is cleared on every successful
	// start; it gates re-attempts inside `startCooldown` so a broken bridge
	// can't be hot-looped by repeated session inputs.
	lastFailAt  time.Time
	lastFailErr error
}

// client returns a live HTTP client for the bridge, starting it on first use or
// if it has since died.
func (m *manager) client(key, binary, specQuery string) (*http.Client, error) {
	p := m.procFor(key, binary, specQuery)
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.ensureAliveLocked(); err != nil {
		return nil, err
	}
	return p.client, nil
}

// restartClient replaces a dead bridge after a connection failure and returns a
// fresh client. If another goroutine already restarted it (the live client
// differs from the dead one), that live client is returned without restarting.
func (m *manager) restartClient(key string, dead *http.Client) (*http.Client, error) {
	m.mu.Lock()
	p := m.procs[key]
	m.mu.Unlock()
	if p == nil {
		return nil, fmt.Errorf("subprocess: no bridge registered for %q", key)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.alive && p.client != dead {
		return p.client, nil // already restarted by a concurrent request
	}
	p.stopLocked()
	// Deliberately under p.mu: this serializes restarts of THIS bridge (other
	// bridges have their own proc/mu, so they're unaffected). A concurrent
	// same-bridge caller blocks here but then takes the early-return above and
	// gets the freshly-restarted client — it waits, but never starts a parallel
	// restart.
	if d := restartCooldown - time.Since(p.startedAt); d > 0 {
		time.Sleep(d) // anti-churn after a quick death; rare in practice
	}
	if err := p.ensureAliveLocked(); err != nil {
		return nil, err
	}
	return p.client, nil
}

// ensureAliveLocked starts the bridge if it isn't alive, honoring the
// start-fail cooldown so a permanently broken bridge can't be hot-looped by
// repeated session inputs. Caller must hold p.mu.
func (p *proc) ensureAliveLocked() error {
	if p.alive {
		return nil
	}
	if p.lastFailErr != nil {
		if d := time.Until(p.lastFailAt.Add(startCooldown)); d > 0 {
			return fmt.Errorf(
				"subprocess: bridge %q recently failed to start (retry in ~%s): %w",
				p.binary, d.Round(time.Second), p.lastFailErr)
		}
	}
	return p.startLocked()
}

func (m *manager) procFor(key, binary, specQuery string) *proc {
	m.mu.Lock()
	defer m.mu.Unlock()
	if p := m.procs[key]; p != nil {
		return p
	}
	p := &proc{key: key, binary: binary, specQuery: specQuery, stderrTail: newRingBuffer(stderrTailLines)}
	m.procs[key] = p
	return p
}

// startLocked spawns a fresh bridge and blocks until it answers /health. Caller
// must hold p.mu. Records lastFailErr/lastFailAt on failure (cleared on
// success) so ensureAliveLocked can gate repeated start attempts.
func (p *proc) startLocked() (err error) {
	defer func() {
		if err != nil {
			p.lastFailAt = time.Now()
			p.lastFailErr = err
		} else {
			p.lastFailErr = nil
		}
	}()

	tmpDir, err := os.MkdirTemp("", "amplio-bridge-")
	if err != nil {
		return fmt.Errorf("subprocess: temp dir: %w", err)
	}
	socket := filepath.Join(tmpDir, "bridge.sock")

	// G204: the bridge path is operator-supplied via the LLM spec, by design.
	cmd := exec.Command(p.binary, "--socket", socket) //nolint:gosec
	cmd.Env = append(os.Environ(), "AMPLIO_BRIDGE_SPEC="+p.specQuery)
	cmd.Env = append(cmd.Env, extraSpawnEnv...)
	setDeathSig(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = os.RemoveAll(tmpDir)
		return fmt.Errorf("subprocess: stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		_ = os.RemoveAll(tmpDir)
		return fmt.Errorf("subprocess: stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		_ = os.RemoveAll(tmpDir)
		return fmt.Errorf("subprocess: start %q: %w", p.binary, err)
	}
	go logPipe(p.binary, "stdout", stdout, nil)
	go logPipe(p.binary, "stderr", stderr, p.stderrTail)

	// died is closed by the wait goroutine when the process exits. waitHealthy
	// selects on it so an early-exit (e.g. ImportError before binding the
	// socket) fails the start in milliseconds instead of waiting out
	// healthTimeout. cmd.Wait completes before the close, so cmd.ProcessState
	// is safely populated by the time waitHealthy reads it.
	died := make(chan struct{})
	client := unixClient(socket)
	go func() {
		_ = cmd.Wait()
		close(died)
		p.markDead(cmd)
	}()

	if err := waitHealthy(client, cmd, died, p.stderrTail); err != nil {
		_ = cmd.Process.Kill() // no-op if the process already exited
		_ = os.RemoveAll(tmpDir)
		return fmt.Errorf("subprocess: bridge %q never became healthy: %w", p.binary, err)
	}

	p.cmd = cmd
	p.client = client
	p.tmpDir = tmpDir
	p.alive = true
	p.startedAt = time.Now()
	slog.Info("bridge subprocess started", "binary", p.binary, "pid", cmd.Process.Pid, "socket", socket)
	return nil
}

// stopLocked kills the current bridge and cleans up. Caller must hold p.mu.
func (p *proc) stopLocked() {
	if p.cmd != nil && p.cmd.Process != nil {
		_ = p.cmd.Process.Kill() // the wait goroutine reaps it
	}
	if p.tmpDir != "" {
		_ = os.RemoveAll(p.tmpDir)
	}
	p.alive = false
	p.cmd = nil
	p.client = nil
	p.tmpDir = ""
}

// markDead flips the proc to not-alive when its process exits, but only if it's
// still the current cmd (a restart's new process is untouched).
func (p *proc) markDead(cmd *exec.Cmd) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cmd == cmd {
		p.alive = false
		attrs := []any{"binary", p.binary, "pid", cmd.Process.Pid}
		if tail := p.stderrTail.String(); tail != "" {
			attrs = append(attrs, "stderr_tail", tail)
		}
		slog.Warn("bridge subprocess exited", attrs...)
	}
}

func unixClient(socket string) *http.Client {
	return &http.Client{
		// No client timeout: a generation may stream for minutes; cancellation
		// rides on the per-request context instead.
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", socket)
			},
			MaxIdleConns:        32,
			MaxIdleConnsPerHost: 32,
			IdleConnTimeout:     90 * time.Second,
		},
	}
}

// waitHealthy polls the bridge's /health until OK, the process exits, or
// healthTimeout elapses. On early process exit it returns immediately with a
// diagnostic that points at the bridge's exit code (stderr is already in the
// amplio log via logPipe), so a broken bridge fails fast instead of burning the
// full timeout against a socket that will never appear.
func waitHealthy(client *http.Client, cmd *exec.Cmd, died <-chan struct{}, tail *ringBuffer) error {
	deadline := time.Now().Add(healthTimeout)
	var lastErr error
	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "http://unix/health", nil)
		resp, err := client.Do(req)
		cancel()
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
			lastErr = fmt.Errorf("health returned %d", resp.StatusCode)
		} else {
			lastErr = err
		}
		select {
		case <-died:
			return fmt.Errorf("bridge exited during startup with code %d%s",
				cmd.ProcessState.ExitCode(), withStderr(tail))
		case <-time.After(healthPoll):
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("timed out after %s", healthTimeout)
	}
	return lastErr
}

// stderrTailLines is how much of a bridge's stderr to keep for the failure
// paths. Enough for a Python traceback, short enough that the log line stays
// readable when nothing is wrong with it.
const stderrTailLines = 50

// withStderr appends captured stderr to an error message, or explains that
// there wasn't any — which is itself diagnostic (a bridge that dies silently is
// usually the wrong binary or a missing exec bit, not a crash).
func withStderr(tail *ringBuffer) string {
	if tail == nil {
		return ""
	}
	if s := tail.String(); s != "" {
		return "; bridge stderr:\n" + s
	}
	return " (the bridge printed nothing to stderr)"
}

// logPipe forwards a bridge's stdout/stderr to the amplio log at Debug level
// so bridge diagnostics are AVAILABLE (--log-level=debug surfaces every line)
// without polluting the default Info stream. Bridges are talkative — they
// typically log per-RPC HTTP traces, model loading, and retries, often dozens
// of lines per agent step — and almost all of it is
// useful only when something's going wrong. Operators who care can flip
// AMPLIO_LOG_LEVEL=debug for the symptomatic session.
//
// tail, when non-nil, also keeps the last few lines for the failure paths: at
// the default log level everything here is dropped, so a bridge that dies during
// startup used to fail with "see stderr above for diagnostics" pointing at
// output nobody printed.
func logPipe(binary, stream string, r io.Reader, tail *ringBuffer) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		slog.Debug("bridge", "binary", filepath.Base(binary), "stream", stream, "line", line)
		if tail != nil {
			tail.add(line)
		}
	}
}

// ringBuffer keeps the most recent n lines. Small on purpose: it exists to make
// a startup failure diagnosable, not to be a log.
type ringBuffer struct {
	mu    sync.Mutex
	lines []string
	n     int
}

func newRingBuffer(n int) *ringBuffer { return &ringBuffer{n: n} }

func (r *ringBuffer) add(line string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lines = append(r.lines, line)
	if len(r.lines) > r.n {
		r.lines = r.lines[len(r.lines)-r.n:]
	}
}

func (r *ringBuffer) String() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return strings.Join(r.lines, "\n")
}

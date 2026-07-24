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

package subprocess

import (
	"bufio"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"amplio/internal/llm"
)

func mustArgs(q string) url.Values {
	v, _ := url.ParseQuery(q)
	return v
}

// bridgeBin is this test binary itself: when re-invoked with AMPLIO_TEST_BRIDGE=1
// it serves as the bridge (see init+serveTestBridge). This keeps tests pure Go
// with no extra `go build` step or separate cmd/ package — exercising the real
// exec + unix socket + HTTP path against a binary the test framework already has.
var bridgeBin string

func TestMain(m *testing.M) {
	restartCooldown = 0 // keep restart tests fast
	startCooldown = 0   // overridden per-test where the cooldown is exercised
	extraSpawnEnv = []string{"AMPLIO_TEST_BRIDGE=1"}
	exe, err := os.Executable()
	if err != nil {
		panic("os.Executable: " + err.Error())
	}
	bridgeBin = exe

	code := m.Run()
	Shutdown()
	os.Exit(code)
}

func newProvider(t *testing.T, model string) *provider {
	t.Helper()
	p, err := New(bridgeBin, 1000, mustArgs("model="+model))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p.(*provider)
}

func userReq(content string) llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: content}}}
}

func TestNew_RequiresModel(t *testing.T) {
	if _, err := New(bridgeBin, 1000, mustArgs("")); err == nil {
		t.Error("expected error when ?model= is missing")
	}
}

func TestCall_Echo(t *testing.T) {
	p := newProvider(t, "call-echo")
	resp, err := p.Call(context.Background(), userReq("hello"))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "echo: hello" {
		t.Errorf("content = %q, want %q", resp.Content, "echo: hello")
	}
	if resp.Usage.TotalTokens == 0 {
		t.Errorf("usage not populated: %+v", resp.Usage)
	}
}

func TestStream_Echo(t *testing.T) {
	p := newProvider(t, "stream-echo")
	s, err := p.Stream(context.Background(), userReq("hello"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()
	var got strings.Builder
	for s.Next() {
		got.WriteString(s.Event().DeltaText)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if got.String() != "echo: hello" {
		t.Errorf("streamed text = %q, want %q", got.String(), "echo: hello")
	}
	if s.Response().Content != "echo: hello" {
		t.Errorf("final content = %q, want %q", s.Response().Content, "echo: hello")
	}
}

func TestCall_ToolCall(t *testing.T) {
	p := newProvider(t, "tool")
	resp, err := p.Call(context.Background(), userReq("__TOOL__ go"))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "echo_tool" {
		t.Errorf("tool calls = %+v, want one echo_tool", resp.ToolCalls)
	}
}

func TestConcurrent_ReuseOneSubprocess(t *testing.T) {
	p := newProvider(t, "concurrent")
	const n = 20
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := p.Call(context.Background(), userReq("hi")); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent call: %v", err)
	}
	// All requests were served by a single reused subprocess for this spec key.
	defaultManager.mu.Lock()
	pr := defaultManager.procs[p.key]
	defaultManager.mu.Unlock()
	if pr == nil || !pr.alive {
		t.Fatalf("expected one live reused subprocess for key %q", p.key)
	}
}

func TestCrash_AutoRestart(t *testing.T) {
	p := newProvider(t, "restart")
	if _, err := p.Call(context.Background(), userReq("warmup")); err != nil {
		t.Fatalf("warmup Call: %v", err)
	}
	// Crash the bridge mid-request: the call fails...
	if _, err := p.Call(context.Background(), userReq("__CRASH__")); err == nil {
		t.Fatal("expected an error when the bridge crashes mid-request")
	}
	// ...but the next call transparently restarts it and succeeds.
	var resp *llm.Response
	var err error
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp, err = p.Call(context.Background(), userReq("after"))
		if err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("expected restart+success, last error: %v", err)
	}
	if resp.Content != "echo: after" {
		t.Errorf("content = %q, want %q", resp.Content, "echo: after")
	}
}

// A bridge that exits during startup (no socket ever bound) must fail in
// milliseconds via the died-channel, not after the full healthTimeout.
func TestStartup_EarlyExitFailsFast(t *testing.T) {
	p := newProvider(t, "__DIE_STARTUP_FAST__")
	start := time.Now()
	_, err := p.Call(context.Background(), userReq("hi"))
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected error for a bridge that exits during startup")
	}
	if !strings.Contains(err.Error(), "exited during startup") {
		t.Errorf("error %q should mention 'exited during startup'", err)
	}
	if !strings.Contains(err.Error(), "code 7") {
		t.Errorf("error %q should surface the bridge exit code", err)
	}
	// Generous bound: should be well under a second, certainly nowhere near
	// the 30s healthTimeout. 2s avoids flakes on a loaded test machine.
	if elapsed > 2*time.Second {
		t.Errorf("startup-fail took %s; expected fast exit detection (< 2s)", elapsed)
	}
}

// After a failed start, subsequent calls within startCooldown should be gated
// with a fast cached-error return. After the cooldown elapses, a fresh start
// attempt is allowed (and also fails, since the binary is still broken — but
// via a fresh attempt, not the cache).
func TestStartCooldown_GatesAndExpires(t *testing.T) {
	old := startCooldown
	startCooldown = 250 * time.Millisecond
	t.Cleanup(func() { startCooldown = old })

	p := newProvider(t, "__DIE_STARTUP_COOLDOWN__")
	if _, err := p.Call(context.Background(), userReq("a")); err == nil {
		t.Fatal("expected first call to fail")
	}

	// Immediate retry: gated, fast, cached error.
	t1 := time.Now()
	_, err := p.Call(context.Background(), userReq("b"))
	elapsed := time.Since(t1)
	if err == nil {
		t.Fatal("expected gated retry to fail with cached error")
	}
	if !strings.Contains(err.Error(), "recently failed") {
		t.Errorf("gated error %q should mention 'recently failed'", err)
	}
	if elapsed > 100*time.Millisecond {
		t.Errorf("gated retry took %s; should be near-instant from the cache", elapsed)
	}

	// After cooldown elapses: a fresh start is attempted (still fails, but the
	// error shape is the live failure, not the cached one).
	time.Sleep(startCooldown + 50*time.Millisecond)
	_, err = p.Call(context.Background(), userReq("c"))
	if err == nil {
		t.Fatal("expected third call to fail (binary is still broken)")
	}
	if strings.Contains(err.Error(), "recently failed") {
		t.Errorf("post-cooldown error %q should be a fresh failure, not cached", err)
	}
	if !strings.Contains(err.Error(), "exited during startup") {
		t.Errorf("post-cooldown error %q should report a fresh startup failure", err)
	}
}

func TestContextCancel(t *testing.T) {
	p := newProvider(t, "cancel")
	// Warm up so the subprocess exists, then cancel before the call.
	if _, err := p.Call(context.Background(), userReq("warmup")); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Call(ctx, userReq("hello")); err == nil {
		t.Error("expected error for a cancelled context")
	}
}

// --- test bridge (self-exec) ---
//
// When the test binary is re-spawned with AMPLIO_TEST_BRIDGE=1, init() takes
// over before any tests run and serves the bridge protocol on the socket given
// by the second command-line arg (matching the `--socket <path>` shape amplio's
// manager invokes bridges with). The "model" message hooks "__CRASH__" and
// "__TOOL__" mirror what cmd/bridgemock used to provide.

func init() {
	if os.Getenv("AMPLIO_TEST_BRIDGE") != "1" {
		return
	}
	// Any "__DIE_*" model exits before binding the socket, simulating a
	// bridge that crashes during startup (e.g. a Python ImportError).
	// Distinct suffixes give distinct proc-cache keys so cooldown-state
	// doesn't leak between tests.
	if spec := os.Getenv("AMPLIO_BRIDGE_SPEC"); strings.Contains(spec, "model=__DIE_") {
		os.Stderr.WriteString("test-bridge: dying without binding (simulated startup failure)\n")
		os.Exit(7)
	}
	var socket string
	for i, a := range os.Args {
		if a == "--socket" && i+1 < len(os.Args) {
			socket = os.Args[i+1]
		}
	}
	if socket == "" {
		os.Stderr.WriteString("test-bridge: --socket is required\n")
		os.Exit(2)
	}
	serveTestBridge(socket)
}

func serveTestBridge(socket string) {
	_ = os.Remove(socket)
	ln, err := net.Listen("unix", socket)
	if err != nil {
		os.Stderr.WriteString("test-bridge: listen: " + err.Error() + "\n")
		os.Exit(1)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(200) })
	mux.HandleFunc("/generate", testBridgeGenerate)
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	_ = srv.Serve(ln)
	os.Exit(0)
}

func testBridgeGenerate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Messages []struct{ Role, Content string } `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	last := ""
	for _, m := range req.Messages {
		if m.Role == "user" || m.Role == "tool_result" {
			last = m.Content
		}
	}
	if strings.Contains(last, "__CRASH__") {
		os.Exit(1)
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(200)
	bw := bufio.NewWriter(w)
	flusher, _ := w.(http.Flusher)
	emit := func(v any) {
		b, _ := json.Marshal(v)
		_, _ = bw.Write(b)
		_ = bw.WriteByte('\n')
		_ = bw.Flush()
		if flusher != nil {
			flusher.Flush()
		}
	}
	if strings.Contains(last, "__TOOL__") {
		emit(map[string]any{"type": "delta", "tool_call_start": map[string]string{"id": "call_t1", "name": "echo_tool"}})
		emit(map[string]any{"type": "final", "response": map[string]any{
			"content":     "",
			"tool_calls":  []map[string]string{{"id": "call_t1", "name": "echo_tool", "arguments": `{"text":"hi"}`}},
			"usage":       map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
			"stop_reason": "tool_use",
		}})
		return
	}
	reply := "echo: " + last
	mid := len(reply) / 2
	emit(map[string]any{"type": "delta", "text": reply[:mid]})
	emit(map[string]any{"type": "delta", "text": reply[mid:]})
	emit(map[string]any{"type": "final", "response": map[string]any{
		"content":     reply,
		"usage":       map[string]int{"prompt_tokens": len(last), "completion_tokens": len(reply), "total_tokens": len(last) + len(reply)},
		"stop_reason": "end_turn",
	}})
}

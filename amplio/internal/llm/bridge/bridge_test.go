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
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
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
	p, err := NewSubprocess(model, 1000, mustArgs("bin="+bridgeBin), nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p.(*provider)
}

func userReq(content string) llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: content}}}
}

func TestNew_RequiresModel(t *testing.T) {
	if _, err := NewSubprocess("", 1000, mustArgs("bin="+bridgeBin), nil); err == nil {
		t.Error("expected error when the model is missing")
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
	key := p.tr.(*spawnTransport).key
	defaultManager.mu.Lock()
	pr := defaultManager.procs[key]
	defaultManager.mu.Unlock()
	if pr == nil || !pr.alive {
		t.Fatalf("expected one live reused subprocess for key %q", key)
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
	// The bridge's own stderr must be IN the error. Everything logPipe writes
	// goes to Debug, which the default level drops, so an error that merely
	// points at "stderr above" points at nothing.
	if !strings.Contains(err.Error(), "dying without binding") {
		t.Errorf("error %q should quote the bridge's stderr", err)
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

// TestPostToSendsProtocolHeaders pins what every transport puts on the wire.
// The version header is what lets two separately built ends fail loudly on a
// mismatch instead of as a puzzling decode error.
func TestPostToSendsProtocolHeaders(t *testing.T) {
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
	}))
	defer srv.Close()

	resp, err := postTo(context.Background(), srv.Client(), srv.URL+"/generate", []byte(`{}`))
	if err != nil {
		t.Fatalf("postTo: %v", err)
	}
	_ = resp.Body.Close()
	if v := got.Get("Content-Type"); v != "application/json" {
		t.Errorf("Content-Type = %q", v)
	}
	if v := got.Get(protocolVersionHeader); v != protocolVersion {
		t.Errorf("%s = %q, want %q", protocolVersionHeader, v, protocolVersion)
	}
}

// TestToWireCarriesSessionID: the session id drives cache/routing affinity on
// the far side (the Claude provider turns it into X-Vertex-Ai-Session-Id).
// Dropping it silently costs prompt-cache hits, which is invisible in tests and
// expensive in production — hence a test for one field.
func TestToWireCarriesSessionID(t *testing.T) {
	p := &provider{model: "m", maxTokens: 10}
	w := p.toWire(llm.Request{SessionID: "sess-42"})
	if w.SessionID != "sess-42" {
		t.Errorf("wire session_id = %q, want %q", w.SessionID, "sess-42")
	}
	if w := (&provider{model: "m"}).toWire(llm.Request{}); w.SessionID != "" {
		t.Errorf("absent session id should stay absent, got %q", w.SessionID)
	}
}

// TestUnknownLineTypesAreIgnored is the forward-compatibility guarantee that
// lets the protocol grow: a reader from an older build must skip a line kind it
// has never heard of (a keepalive "ping", say) rather than fail the request.
func TestUnknownLineTypesAreIgnored(t *testing.T) {
	stream := `{"type":"ping"}` + "\n" +
		`{"type":"delta","text":"hel"}` + "\n" +
		`{"type":"ping"}` + "\n" +
		`{"type":"delta","text":"lo"}` + "\n" +
		`{"type":"something-from-the-future","payload":{"a":1}}` + "\n" +
		`{"type":"final","response":{"content":"hello"}}` + "\n"

	resp, err := readFinal(strings.NewReader(stream))
	if err != nil {
		t.Fatalf("readFinal: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("content = %q, want %q", resp.Content, "hello")
	}
}

// TestSubprocessSpecShapes: both spellings must reach the same bridge with the
// same instructions, because the DB holds specs written in the old one and a
// silent difference between them would be a support nightmare.
func TestSubprocessSpecShapes(t *testing.T) {
	current, err := NewSubprocess("some-model", 100,
		mustArgs("bin=/opt/bridges/corp"), mustArgs("temperature=0.2"))
	if err != nil {
		t.Fatalf("current form: %v", err)
	}
	for _, tc := range []struct {
		name string
		p    llm.Provider
	}{{"current", current}} {
		tr := tc.p.(*provider).tr.(*spawnTransport)
		if tr.binary != "/opt/bridges/corp" {
			t.Errorf("%s: binary = %q", tc.name, tr.binary)
		}
		if got := tc.p.ModelID(); got != "some-model" {
			t.Errorf("%s: model = %q", tc.name, got)
		}
		// model= is part of the bridge contract, so it is sent whichever way the
		// operator spelled the spec; bin= is ours and means nothing to a bridge.
		if !strings.Contains(tr.specQuery, "model=some-model") {
			t.Errorf("%s: AMPLIO_BRIDGE_SPEC = %q, want model= in it", tc.name, tr.specQuery)
		}
		if strings.Contains(tr.specQuery, "bin=") {
			t.Errorf("%s: AMPLIO_BRIDGE_SPEC = %q, must not leak bin=", tc.name, tr.specQuery)
		}
		if !strings.Contains(tr.specQuery, "temperature=0.2") {
			t.Errorf("%s: model args should ride along: %q", tc.name, tr.specQuery)
		}
	}
	// Neither half may be missing.
	if _, err := NewSubprocess("", 100, mustArgs(""), nil); err == nil {
		t.Error("want an error with no bridge path")
	}
	if _, err := NewSubprocess("", 100, mustArgs("bin=/opt/bridges/corp"), nil); err == nil {
		t.Error("want an error with no model")
	}
}

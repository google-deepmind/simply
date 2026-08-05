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
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"amplio/internal/llm"
)

// thoughtSignature stands in for the opaque blobs providers attach to reasoning
// — Gemini's runs to ~523 characters and rides inside the tool-call ID. Anything
// that shortens, validates or regenerates an ID silently destroys it, so the
// round trip is tested with a realistic one rather than "id-1".
var thoughtSignature = "y9a8ksen__thought__" + strings.Repeat("AY89a18zqHlEaX0FCdyMSMiwuHzWvlf0xG1NJJRIkxwC7sJidYU", 10)

// fakeProvider records what it was asked and replays a scripted stream.
type fakeProvider struct {
	got    llm.Request
	events []llm.StreamEvent
	final  *llm.Response
	err    error
	gate   chan struct{} // when set, Stream blocks on it: a model that thinks
}

func (f *fakeProvider) ModelID() string { return "fake" }
func (f *fakeProvider) MaxTokens() int  { return 1 }
func (f *fakeProvider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	return f.final, f.err
}

func (f *fakeProvider) Stream(ctx context.Context, req llm.Request) (llm.Stream, error) {
	if f.gate != nil {
		<-f.gate
	}
	f.got = req
	if f.err != nil {
		return nil, f.err
	}
	return &fakeStream{events: f.events, final: f.final}, nil
}

type fakeStream struct {
	events []llm.StreamEvent
	i      int
	final  *llm.Response
}

func (s *fakeStream) Next() bool {
	if s.i >= len(s.events) {
		return false
	}
	s.i++
	return true
}
func (s *fakeStream) Event() llm.StreamEvent { return s.events[s.i-1] }
func (s *fakeStream) Response() *llm.Response {
	return s.final
}
func (s *fakeStream) Err() error { return nil }
func (s *fakeStream) Close()     {}

// serveBridge stands up the serving half and returns a dialling provider aimed
// at it — both ends of the protocol, in-process.
func serveBridge(t *testing.T, fake *fakeProvider, clientArgs url.Values) llm.Provider {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", GenerateHandler(func(ctx context.Context, handle string) (llm.Provider, error) {
		if handle == "refused" {
			return nil, fmt.Errorf("model %q is not in this server's menu", handle)
		}
		return fake, nil
	}))
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	args := url.Values{"url": {srv.URL}}
	for k, v := range clientArgs {
		args[k] = v
	}
	p, err := NewBridge("some-model", 4096, args, nil)
	if err != nil {
		t.Fatalf("NewBridge: %v", err)
	}
	return p
}

// TestRoundTrip_Lossless is the point of the whole design: a generation crossing
// the bridge must arrive on the far side, and come back, byte-identical. An
// OpenAI-shaped hop cannot do this — thinking signatures and provider-specific
// fields have nowhere to live in that schema — which is why the bridge speaks
// amplio's own types instead.
func TestRoundTrip_Lossless(t *testing.T) {
	fake := &fakeProvider{
		events: []llm.StreamEvent{
			{DeltaThoughts: "let me think"},
			{DeltaText: "par"},
			{DeltaText: "tial"},
			{ToolCallStart: &llm.ToolCallStart{ID: thoughtSignature, Name: "bash"}},
			{ToolCallDelta: &llm.ToolCallDelta{ID: thoughtSignature, ArgumentsDelta: `{"cmd":`}},
			{ToolCallDelta: &llm.ToolCallDelta{ID: thoughtSignature, ArgumentsDelta: `"ls"}`}},
		},
		final: &llm.Response{
			Content:    "partial",
			Thoughts:   "let me think",
			StopReason: "tool_use",
			ToolCalls:  []llm.ToolCall{{ID: thoughtSignature, Name: "bash", Arguments: `{"cmd":"ls"}`}},
			Usage: llm.Usage{
				PromptTokens: 113_000, CompletionTokens: 42, TotalTokens: 113_042,
				CacheReadTokens: 100_000, CacheWriteTokens: 13_000,
			},
			ProviderExtra: map[string]any{
				"beyond.fc_sigs_b64": []any{"c2lnbmF0dXJl", "YW5vdGhlcg=="},
				"nested":             map[string]any{"depth": "two"},
			},
		},
	}
	p := serveBridge(t, fake, nil)

	req := llm.Request{
		SystemPrompt: "you are a bridge test",
		SessionID:    "sess-42",
		Temperature:  ptr(0.25),
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "hello", Attachments: []llm.Attachment{{MimeType: "image/png", Base64Data: "iVBOR"}}},
			{
				Role:          llm.RoleAssistant,
				Content:       "thinking",
				ToolCalls:     []llm.ToolCall{{ID: thoughtSignature, Name: "bash", Arguments: `{"cmd":"ls"}`}},
				ProviderExtra: map[string]any{"beyond.fc_sigs_b64": []any{"cmVwbGF5"}},
			},
			{Role: llm.RoleToolResult, ToolCallID: thoughtSignature, Content: "a.txt", IsError: true},
		},
		Tools: []llm.ToolDef{{Name: "bash", Description: "run a command", Schema: []byte(`{"type":"object"}`)}},
	}

	st, err := p.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var got []llm.StreamEvent
	for st.Next() {
		got = append(got, st.Event())
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	final := st.Response()
	st.Close()

	// --- request direction ---
	sent := fake.got
	if sent.SystemPrompt != req.SystemPrompt {
		t.Errorf("system prompt = %q", sent.SystemPrompt)
	}
	if sent.SessionID != "sess-42" {
		t.Errorf("session id = %q, want it forwarded (cache affinity)", sent.SessionID)
	}
	if sent.Temperature == nil || *sent.Temperature != 0.25 {
		t.Errorf("temperature = %v", sent.Temperature)
	}
	if !reflect.DeepEqual(sent.Messages, req.Messages) {
		t.Errorf("messages differ:\n got %#v\nwant %#v", sent.Messages, req.Messages)
	}
	if !reflect.DeepEqual(sent.Tools, req.Tools) {
		t.Errorf("tools differ:\n got %#v\nwant %#v", sent.Tools, req.Tools)
	}

	// --- response direction ---
	if !reflect.DeepEqual(got, fake.events) {
		t.Errorf("stream events differ:\n got %#v\nwant %#v", got, fake.events)
	}
	if !reflect.DeepEqual(final, fake.final) {
		t.Errorf("final response differs:\n got %#v\nwant %#v", final, fake.final)
	}
	// Called out separately: these are the fields a foreign schema loses, and a
	// DeepEqual failure above would not say WHICH.
	if len(final.ToolCalls) != 1 || final.ToolCalls[0].ID != thoughtSignature {
		t.Errorf("tool-call id (%d chars) did not survive: %#v", len(thoughtSignature), final.ToolCalls)
	}
	if !reflect.DeepEqual(final.ProviderExtra, fake.final.ProviderExtra) {
		t.Errorf("provider_extra did not survive: %#v", final.ProviderExtra)
	}
}

// TestRoundTrip_ErrorTextIsVerbatim guards a non-obvious dependency: amplio
// decides whether to compact a session by asking a model to READ the provider's
// error text. Rewriting an upstream error on the way through would silently
// disable compaction for every bridged run.
func TestRoundTrip_ErrorTextIsVerbatim(t *testing.T) {
	const upstream = "input is too long: 250000 tokens > 200000 maximum context length"
	p := serveBridge(t, &fakeProvider{err: fmt.Errorf("%s", upstream)}, nil)

	// The refusal to generate is not an HTTP failure: the far side answered 200
	// and reported the model's error on the stream, so it surfaces via Err().
	st, err := p.Stream(context.Background(), llm.Request{})
	if err == nil {
		for st.Next() {
		}
		err = st.Err()
		st.Close()
	}
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), upstream) {
		t.Errorf("error = %q, want it to contain the upstream text verbatim", err)
	}

	// Same for the blocking path, which reads the same stream.
	if _, err := p.Call(context.Background(), llm.Request{}); err == nil ||
		!strings.Contains(err.Error(), upstream) {
		t.Errorf("Call error = %v, want the upstream text verbatim", err)
	}
}

func TestRoundTrip_RefusalIsDistinct(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", GenerateHandler(func(ctx context.Context, handle string) (llm.Provider, error) {
		return nil, fmt.Errorf("model %q is not in this server's menu; try: opus-xhigh", handle)
	}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	p, err := NewBridge("nope", 10, url.Values{"url": {srv.URL}}, nil)
	if err != nil {
		t.Fatalf("NewBridge: %v", err)
	}
	_, err = p.Call(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("expected a refusal")
	}
	// The caller needs to tell "you may not ask for that" from "the model failed":
	// one is fixed by editing a spec, the other by retrying.
	if !strings.Contains(err.Error(), "403") || !strings.Contains(err.Error(), "not in this server's menu") {
		t.Errorf("error = %q, want an HTTP 403 carrying the reason", err)
	}
}

func TestNewBridge_Validation(t *testing.T) {
	t.Setenv(urlEnv, "")
	if _, err := NewBridge("m", 10, nil, nil); err == nil {
		t.Error("want an error with no endpoint configured")
	}
	if _, err := NewBridge("m", 10, url.Values{"url": {"ftp://x"}}, nil); err == nil {
		t.Error("want an error for an unsupported scheme")
	}
	if _, err := NewBridge("", 10, url.Values{"url": {"http://x"}}, nil); err == nil {
		t.Error("want an error for an empty handle")
	}
	if _, err := NewBridge("m", 10, url.Values{"url": {"http://x"}, "idle_timeout": {"soon"}}, nil); err == nil {
		t.Error("want an error for an unparseable idle_timeout")
	}
	// The endpoint may come from the environment, so a spec is portable between
	// a container and a workstation.
	t.Setenv(urlEnv, "http://127.0.0.1:1/api/llm")
	if _, err := NewBridge("m", 10, nil, nil); err != nil {
		t.Errorf("endpoint from %s should be accepted: %v", urlEnv, err)
	}
}

// TestBridgeHandleCarriesModelArgs: model args ride along with the handle, so
// `bridge{url=…}:vertex-claude:opus?effort=high` reaches the far end as one
// spec string rather than losing its query.
func TestBridgeHandleCarriesModelArgs(t *testing.T) {
	var handle string
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", GenerateHandler(func(ctx context.Context, h string) (llm.Provider, error) {
		handle = h
		return &fakeProvider{final: &llm.Response{}}, nil
	}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	p, err := NewBridge("vertex-claude:claude-opus-5", 10,
		url.Values{"url": {srv.URL}},
		url.Values{"output_config.effort": {"xhigh"}, "cache_ttl": {"1h"}})
	if err != nil {
		t.Fatalf("NewBridge: %v", err)
	}
	if _, err := p.Call(context.Background(), llm.Request{}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	want := "vertex-claude:claude-opus-5?cache_ttl=1h&output_config.effort=xhigh"
	if handle != want {
		t.Errorf("handle = %q, want %q", handle, want)
	}
}

func ptr[T any](v T) *T { return &v }

// TestServerPingsWhileWaiting: the far side emits keepalives while the model is
// still thinking, which is what lets the client set an idle timeout at all —
// minutes of silence are normal, and TCP keepalive defaults to hours.
func TestServerPingsWhileWaiting(t *testing.T) {
	release := make(chan struct{})
	fake := &fakeProvider{final: &llm.Response{Content: "done"}, gate: release}
	mux := http.NewServeMux()
	// Resolution happens BEFORE any bytes are written (a refusal still needs its
	// status code), so the wait that matters — and the one this exercises — is
	// the model's, inside Stream.
	mux.HandleFunc("POST /generate", generateHandler(func(ctx context.Context, h string) (llm.Provider, error) {
		return fake, nil
	}, 10*time.Millisecond))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	go func() {
		time.Sleep(80 * time.Millisecond)
		close(release)
	}()

	hreq, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		srv.URL+"/generate", strings.NewReader(`{"model":"m"}`))
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	hreq.Header.Set("Content-Type", "application/json")
	resp, err := srv.Client().Do(hreq)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	if n := strings.Count(string(body), `"type":"ping"`); n == 0 {
		t.Errorf("no pings in %q; the client cannot distinguish a slow model from a dead link", body)
	}
	if !strings.Contains(string(body), `"type":"final"`) {
		t.Errorf("stream did not end with a final line: %q", body)
	}
}

// TestPingerCannotOutliveHandler pins a crash that a single-request test misses:
// a ResponseWriter is only valid until its handler returns, so a keepalive
// goroutine still inside Write at that moment panics inside net/http with a
// nil-pointer dereference. Signalling it to stop is not enough — it has to be
// joined. Ping fast, respond instantly, repeat: unjoined, this panics within a
// few iterations.
func TestPingerCannotOutliveHandler(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", generateHandler(func(ctx context.Context, h string) (llm.Provider, error) {
		return &fakeProvider{final: &llm.Response{Content: "done"}}, nil
	}, time.Microsecond))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	for i := 0; i < 200; i++ {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
			srv.URL+"/generate", strings.NewReader(`{"model":"m"}`))
		if err != nil {
			t.Fatalf("request: %v", err)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatalf("iteration %d: %v", i, err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}
}

// TestEmbedRoundTrip: vectors cross the bridge, batching is transparent, and the
// cache key encodes the endpoint.
func TestEmbedRoundTrip(t *testing.T) {
	var batches []int
	mux := http.NewServeMux()
	mux.HandleFunc("POST /embed", EmbedHandler(&fakeEmbedder{model: "vertex_text-embedding-005", onBatch: func(n int) {
		batches = append(batches, n)
	}}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	e, err := NewEmbedder("", url.Values{"url": {srv.URL}})
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	texts := make([]string, 150) // > 2 batches
	for i := range texts {
		texts[i] = fmt.Sprintf("doc %d", i)
	}
	vectors, err := e.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != len(texts) {
		t.Fatalf("got %d vectors for %d texts", len(vectors), len(texts))
	}
	// 1:1 alignment is the contract callers rely on; a batching bug shows up as
	// a silent off-by-one in a recall index, months later.
	for i, v := range vectors {
		if len(v) != 1 || v[0] != float32(i) {
			t.Fatalf("vector %d = %v, want [%d] (batching lost alignment)", i, v, i)
		}
	}
	if len(batches) != 3 || batches[0] != embedBatch {
		t.Errorf("batches = %v, want three requests of at most %d", batches, embedBatch)
	}

	// The cache key must distinguish endpoints: vectors from two bridges are not
	// interchangeable even when both call the model the same thing.
	other, _ := NewEmbedder("", url.Values{"url": {"http://elsewhere:9"}})
	if e.ModelID() == other.ModelID() {
		t.Errorf("two endpoints share the cache key %q", e.ModelID())
	}
}

func TestEmbedRefusesAnotherModel(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /embed", EmbedHandler(&fakeEmbedder{model: "vertex_text-embedding-005"}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	e, err := NewEmbedder("some-other-model", url.Values{"url": {srv.URL}})
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	_, err = e.Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "corrupt your index") {
		t.Errorf("error = %v, want a refusal to serve a different embedding space", err)
	}
}

type fakeEmbedder struct {
	model   string
	n       int
	onBatch func(int)
}

func (f *fakeEmbedder) ModelID() string { return f.model }
func (f *fakeEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if f.onBatch != nil {
		f.onBatch(len(texts))
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{float32(f.n)}
		f.n++
	}
	return out, nil
}

// TestErrorCodes: a caller has to be able to tell "fix your spec" from "the
// model failed" without reading English. The message stays verbatim either way —
// that is what compaction reads.
func TestErrorCodes(t *testing.T) {
	tests := []struct {
		name     string
		handle   string
		resolve  Resolver
		wantCode string
		wantText string
	}{
		{
			name:   "a refusal is not retryable and says so",
			handle: "nope",
			resolve: func(ctx context.Context, h string) (llm.Provider, error) {
				return nil, fmt.Errorf("%q is not in this server's menu", h)
			},
			wantCode: CodeNotAllowed,
			wantText: "not in this server's menu",
		},
		{
			name:   "a model failure carries the upstream text",
			handle: "ok",
			resolve: func(ctx context.Context, h string) (llm.Provider, error) {
				return &fakeProvider{err: fmt.Errorf("input is too long: 250000 tokens")}, nil
			},
			wantCode: CodeProvider,
			wantText: "input is too long",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mux := http.NewServeMux()
			mux.HandleFunc("POST /generate", GenerateHandler(tt.resolve))
			srv := httptest.NewServer(mux)
			defer srv.Close()

			p, err := NewBridge(tt.handle, 10, url.Values{"url": {srv.URL}}, nil)
			if err != nil {
				t.Fatalf("NewBridge: %v", err)
			}
			_, err = p.Call(context.Background(), llm.Request{})
			if err == nil {
				t.Fatal("want an error")
			}
			if got := CodeOf(err); got != tt.wantCode {
				t.Errorf("CodeOf = %q, want %q (err: %v)", got, tt.wantCode, err)
			}
			if !strings.Contains(err.Error(), tt.wantText) {
				t.Errorf("error = %q, want it to contain %q", err, tt.wantText)
			}
		})
	}

	// A bad token is its own class: retrying cannot help, and it is not the
	// model's fault.
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "invalid or missing token", http.StatusUnauthorized)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	p, err := NewBridge("m", 10, url.Values{"url": {srv.URL}}, nil)
	if err != nil {
		t.Fatalf("NewBridge: %v", err)
	}
	if _, err := p.Call(context.Background(), llm.Request{}); CodeOf(err) != CodeUnauthorized {
		t.Errorf("CodeOf = %q, want %q (err: %v)", CodeOf(err), CodeUnauthorized, err)
	}

	// A non-bridge error must not claim a class.
	if got := CodeOf(fmt.Errorf("something else")); got != "" {
		t.Errorf("CodeOf(non-bridge) = %q, want empty", got)
	}
}

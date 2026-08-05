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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"amplio/internal/llm"
)

// The transport for a bridge somebody ELSE is running: amplio dials an endpoint
// instead of spawning a process.
//
// Spec form: bridge{url=…&token_env=…}:<handle>, where <handle> is whatever the
// far end understands — for an amplio, a model spec or a nickname from its menu.

const (
	// DefaultTokenEnv is where the bearer token comes from when a spec does not
	// name a variable — and, by the same default on the serving side, what a
	// lending amplio requires. One constant so both ends of a tunnel are
	// configured alike without anyone having to remember the string.
	//
	// The spec names the VARIABLE, never the value: specs are persisted in the
	// DB, rendered in the UI, and quoted verbatim in errors.
	DefaultTokenEnv = "AMPLIO_BRIDGE_TOKEN" //nolint:gosec // the NAME of a variable, which is the point: never the value

	// urlEnv supplies url= when the spec omits it, so a container can be pointed
	// at its bridge purely by environment — which is how containers are usually
	// configured — and the same spec then works unmodified on a workstation.
	urlEnv = "AMPLIO_BRIDGE_URL"

	// defaultIdleTimeout fails a stream that has produced no bytes for this long.
	// Generous, because silence is NORMAL: a reasoning model can think for
	// minutes before the first token. An amplio bridge pings every 15s, which is
	// what makes any timeout safe to set; a bridge that never pings should raise
	// this (or set idle_timeout=0) rather than be cut off mid-thought.
	defaultIdleTimeout = 5 * time.Minute
)

// ClientArgsBridge are the client args a dialled bridge accepts.
var ClientArgsBridge = map[string]bool{
	"url":          true,
	"token_env":    true,
	"idle_timeout": true,
	// endpoint names a link configured in config.toml. It is resolved into the
	// keys above BEFORE a provider is built (cmd/amplio), so this package never
	// sees it and keeps knowing nothing about application configuration.
	"endpoint": true,
}

type httpTransport struct {
	client   *http.Client
	base     string // e.g. https://ws:26759/api/llm, or http://unix for a socket
	endpoint string // what the operator wrote, for error messages
	token    string
	idle     time.Duration
}

func (t *httpTransport) describe() string { return t.endpoint }

func (t *httpTransport) post(ctx context.Context, path string, body []byte) (*http.Response, error) {
	// A stream that stalls forever is indistinguishable from a model that is
	// thinking, so the deadline is per-read (see idleReader), not overall: a
	// generation legitimately runs for many minutes.
	ctx, cancel := context.WithCancel(ctx)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.base+path, strings.NewReader(string(body)))
	if err != nil {
		cancel()
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(protocolVersionHeader, protocolVersion)
	if t.token != "" {
		req.Header.Set("Authorization", "Bearer "+t.token)
	}
	resp, err := t.client.Do(req) //nolint:bodyclose // ownership transfers to the caller
	if err != nil {
		cancel()
		return nil, fmt.Errorf("bridge %s: %w", t.endpoint, err)
	}
	resp.Body = newIdleReader(resp.Body, t.idle, cancel, t.endpoint)
	return resp, nil
}

// NewBridge builds a provider that dials a bridge. The model position is the
// HANDLE: it is forwarded verbatim (plus any model args) and deliberately not
// interpreted here, because the far end may not be an amplio and is in any case
// the authority on what its handles mean.
func NewBridge(handle string, maxTokens int, clientArgs, args url.Values) (llm.Provider, error) {
	if handle == "" {
		return nil, fmt.Errorf("bridge provider: empty handle; want bridge{url=…}:<model spec or nickname>")
	}
	raw := clientArgs.Get("url")
	if raw == "" {
		raw = os.Getenv(urlEnv)
	}
	if raw == "" {
		return nil, fmt.Errorf("bridge provider: no endpoint; set url= in the spec or %s", urlEnv)
	}
	client, base, err := dialer(raw)
	if err != nil {
		return nil, fmt.Errorf("bridge provider: %w", err)
	}
	tokenEnv := clientArgs.Get("token_env")
	if tokenEnv == "" {
		tokenEnv = DefaultTokenEnv
	}
	idle := defaultIdleTimeout
	if v := clientArgs.Get("idle_timeout"); v != "" {
		d, err := parseIdle(v)
		if err != nil {
			return nil, fmt.Errorf("bridge provider: idle_timeout=%q: %w", v, err)
		}
		idle = d
	}
	// The handle keeps its model args: they are the far end's business, and
	// re-encoding them canonically means the same request produces the same
	// string, which is what the far end matches against its menu.
	if len(args) > 0 {
		handle += "?" + llm.EncodeArgs(args)
	}
	return &provider{
		tr: &httpTransport{
			client:   client,
			base:     base,
			endpoint: raw,
			token:    os.Getenv(tokenEnv),
			idle:     idle,
		},
		model:     handle,
		maxTokens: maxTokens,
	}, nil
}

func parseIdle(v string) (time.Duration, error) {
	if n, err := strconv.Atoi(v); err == nil { // bare seconds, or 0 to disable
		if n < 0 {
			return 0, fmt.Errorf("want a duration or a non-negative number of seconds")
		}
		return time.Duration(n) * time.Second, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil || d < 0 {
		return 0, fmt.Errorf("want a duration like 90s, or 0 to disable")
	}
	return d, nil
}

// dialer builds the HTTP client for an endpoint and the base URL to use with it.
// unix:// gets a socket dialer and a placeholder host, which is how the same
// code path serves both a remote workstation and a bridge running next door.
func dialer(raw string) (*http.Client, string, error) {
	if socket, ok := strings.CutPrefix(raw, "unix://"); ok {
		if socket == "" {
			return nil, "", fmt.Errorf("unix:// endpoint with no path")
		}
		return &http.Client{
			// No client timeout: a generation may stream for minutes, and
			// liveness is the idle reader's job.
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
					var d net.Dialer
					return d.DialContext(ctx, "unix", socket)
				},
			},
		}, "http://unix", nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, "", fmt.Errorf("endpoint %q: %w", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, "", fmt.Errorf("endpoint %q: want http://, https:// or unix://", raw)
	}
	return &http.Client{}, strings.TrimRight(raw, "/"), nil
}

// idleReader aborts a stream that goes quiet for too long. The timer is reset by
// every read that returns data — including a keepalive ping — so a healthy but
// slow generation is never cut off, while a connection that died silently
// (middlebox reaped it, peer power-cycled) fails in minutes instead of hanging
// until the OS notices, which by default takes hours.
type idleReader struct {
	rc       io.ReadCloser
	timer    *time.Timer
	cancel   context.CancelFunc
	endpoint string
	fired    chan struct{}
}

func newIdleReader(rc io.ReadCloser, d time.Duration, cancel context.CancelFunc, endpoint string) io.ReadCloser {
	if d <= 0 {
		return &cancelCloser{ReadCloser: rc, cancel: cancel}
	}
	r := &idleReader{rc: rc, cancel: cancel, endpoint: endpoint, fired: make(chan struct{})}
	r.timer = time.AfterFunc(d, func() {
		close(r.fired)
		cancel() // unblocks the read below with context.Canceled
	})
	return r
}

func (r *idleReader) Read(p []byte) (int, error) {
	n, err := r.rc.Read(p)
	if n > 0 {
		r.timer.Reset(defaultIdleTimeout)
	}
	if err != nil {
		select {
		case <-r.fired:
			// Translate: the caller would otherwise see "context canceled" and
			// conclude that amplio cancelled its own request.
			return n, fmt.Errorf("bridge %s: no data for %s; connection presumed dead", r.endpoint, defaultIdleTimeout)
		default:
		}
	}
	return n, err
}

func (r *idleReader) Close() error {
	r.timer.Stop()
	err := r.rc.Close()
	r.cancel()
	return err
}

// cancelCloser releases the request context when the body is closed, for the
// idle_timeout=0 case.
type cancelCloser struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (c *cancelCloser) Close() error {
	err := c.ReadCloser.Close()
	c.cancel()
	return err
}

// embedBatch bounds one /embed request. Recall indexes thousands of documents at
// a time, and one request carrying all of them would be a multi-megabyte body
// with a single point of failure; the far side batches again for its own API.
const embedBatch = 64

type bridgeEmbedder struct {
	tr    *httpTransport
	model string // the handle, which is also the cache key's stable part
	tag   string
}

// NewEmbedder builds an embedder that runs on a bridge. The returned ModelID
// encodes the ENDPOINT as well as the model: vectors from two different bridges
// are not interchangeable even if both call the model "text-embedding-005", and
// a cache keyed on the name alone would silently mix them.
func NewEmbedder(handle string, clientArgs url.Values) (Embedder, error) {
	raw := clientArgs.Get("url")
	if raw == "" {
		raw = os.Getenv(urlEnv)
	}
	if raw == "" {
		return nil, fmt.Errorf("bridge embedder: no endpoint; set url= in the spec or %s", urlEnv)
	}
	client, base, err := dialer(raw)
	if err != nil {
		return nil, fmt.Errorf("bridge embedder: %w", err)
	}
	tokenEnv := clientArgs.Get("token_env")
	if tokenEnv == "" {
		tokenEnv = DefaultTokenEnv
	}
	sum := sha256.Sum256([]byte(raw))
	return &bridgeEmbedder{
		tr: &httpTransport{
			client:   client,
			base:     base,
			endpoint: raw,
			token:    os.Getenv(tokenEnv),
			idle:     defaultIdleTimeout,
		},
		model: handle,
		tag:   hex.EncodeToString(sum[:3]),
	}, nil
}

func (e *bridgeEmbedder) ModelID() string {
	if e.model == "" {
		return "bridge@" + e.tag
	}
	return "bridge_" + e.model + "@" + e.tag
}

func (e *bridgeEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += embedBatch {
		end := min(start+embedBatch, len(texts))
		vectors, err := e.embedOne(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		if len(vectors) != end-start {
			return nil, fmt.Errorf("bridge %s: asked for %d embeddings, got %d", e.tr.endpoint, end-start, len(vectors))
		}
		out = append(out, vectors...)
	}
	return out, nil
}

func (e *bridgeEmbedder) embedOne(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(wireEmbedRequest{Model: e.model, Texts: texts})
	if err != nil {
		return nil, err
	}
	resp, err := e.tr.post(ctx, "/embed", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("bridge %s: HTTP %d: %s", e.tr.endpoint, resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var out wireEmbedResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxLine)).Decode(&out); err != nil {
		return nil, fmt.Errorf("bridge %s: decode embeddings: %w", e.tr.endpoint, err)
	}
	if out.Error != "" {
		// Verbatim, for the same reason generation errors are.
		return nil, fmt.Errorf("bridge %s: %s", e.tr.endpoint, out.Error)
	}
	return out.Vectors, nil
}

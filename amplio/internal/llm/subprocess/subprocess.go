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

// Package subprocess implements llm.Provider by bridging to an out-of-process
// "bridge" binary over HTTP on a Unix domain socket. It lets amplio reach
// model backends whose SDKs can't (or shouldn't) be linked into the main binary
// — e.g. corp-only APIs like Beyond — by shelling out to a small bridge that
// speaks the wire protocol in wire.go.
//
// Spec form: subprocess:/path/to/bridge?model=NAME[&k=v...]. The path is the
// bridge binary; the query args configure it (forwarded via the
// AMPLIO_BRIDGE_SPEC env var) and form part of the reuse key, so identical specs
// share one long-lived subprocess while distinct ones get their own. See
// bridges/README.md for the bridge contract.
package subprocess

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"amplio/internal/llm"
)

// maxLine bounds a single NDJSON line (a final response can be large: full
// content plus tool-call arguments).
const maxLine = 16 * 1024 * 1024

type provider struct {
	key       string // reuse key: binary + canonical args
	binary    string
	specQuery string // canonical args, forwarded to the bridge as AMPLIO_BRIDGE_SPEC
	model     string
	maxTokens int
}

// New builds a subprocess-backed provider. binary is the bridge path (the spec's
// positional model field); args are the spec query. ?model= is required (it is
// the real model id the bridge serves); ?max_tokens= overrides the default cap.
func New(binary string, maxTokens int, args url.Values) (llm.Provider, error) {
	if binary == "" {
		return nil, fmt.Errorf("subprocess provider: empty bridge path; want subprocess:/path/to/bridge?model=NAME")
	}
	model := args.Get("model")
	if model == "" {
		return nil, fmt.Errorf("subprocess provider: spec must include ?model=NAME")
	}
	mt := maxTokens
	if v := args.Get("max_tokens"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			mt = n
		}
	}
	q := args.Encode() // sorted, canonical
	return &provider{
		key:       binary + "?" + q,
		binary:    binary,
		specQuery: q,
		model:     model,
		maxTokens: mt,
	}, nil
}

func (p *provider) ModelID() string { return p.model }
func (p *provider) MaxTokens() int  { return p.maxTokens }

func (p *provider) Call(ctx context.Context, req llm.Request) (*llm.Response, error) {
	resp, err := p.post(ctx, req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	return readFinal(resp.Body)
}

func (p *provider) Stream(ctx context.Context, req llm.Request) (llm.Stream, error) {
	resp, err := p.post(ctx, req) //nolint:bodyclose // body ownership transfers to subStream.Close
	if err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), maxLine)
	return &subStream{resp: resp, sc: sc}, nil
}

// post sends the request to the bridge's /generate, restarting the subprocess
// and retrying once if the connection fails (the bridge crashed/was never up).
// The returned response body is owned by the caller and must be closed.
func (p *provider) post(ctx context.Context, req llm.Request) (*http.Response, error) {
	body, err := json.Marshal(p.toWire(req))
	if err != nil {
		return nil, fmt.Errorf("subprocess: marshal request: %w", err)
	}
	client, err := defaultManager.client(p.key, p.binary, p.specQuery)
	if err != nil {
		return nil, err
	}
	resp, err := doPost(ctx, client, body)
	if err != nil {
		if ctx.Err() != nil {
			return nil, err // caller cancelled; don't churn the subprocess
		}
		client, rerr := defaultManager.restartClient(p.key, client)
		if rerr != nil {
			return nil, fmt.Errorf("subprocess: bridge restart failed: %w (original: %v)", rerr, err)
		}
		resp, err = doPost(ctx, client, body)
		if err != nil {
			return nil, fmt.Errorf("subprocess: request failed after restart: %w", err)
		}
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
		return nil, fmt.Errorf("subprocess: bridge returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	return resp, nil
}

func doPost(ctx context.Context, client *http.Client, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://unix/generate", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	// Ownership of the response body transfers to the caller (closed in Call's
	// defer or subStream.Close).
	return client.Do(req) //nolint:bodyclose
}

// readFinal scans an NDJSON /generate response, ignoring deltas, and returns the
// final accumulated response (the form used by Call).
func readFinal(r io.Reader) (*llm.Response, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), maxLine)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var wl wireLine
		if err := json.Unmarshal(line, &wl); err != nil {
			return nil, fmt.Errorf("subprocess: bad protocol line: %w", err)
		}
		switch wl.Type {
		case "final":
			if wl.Response == nil {
				return nil, fmt.Errorf("subprocess: final line without a response")
			}
			return wl.Response.toLLM(), nil
		case "error":
			return nil, fmt.Errorf("bridge error: %s", wl.Error)
		}
		// deltas and unknown line types are ignored for the blocking path
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("subprocess: read stream: %w", err)
	}
	return nil, fmt.Errorf("subprocess: stream ended without a final response")
}

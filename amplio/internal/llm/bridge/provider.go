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

// Package bridge implements llm.Provider by talking to a BRIDGE: any process
// that speaks the NDJSON protocol in wire.go over HTTP. It lets amplio reach
// model backends whose SDKs can't (or shouldn't) be linked into the main binary
// — e.g. a corp-only API — and, because the protocol serialises amplio's own
// request and response types (ProviderExtra included), it loses nothing on the
// way, unlike a hop through a foreign schema.
//
// The transports differ only in who owns the bridge process:
//
//   - subprocess:/path/to/bridge?model=NAME — WE own it. This package spawns it,
//     watches it, restarts it, and talks over a private unix socket. The spec
//     args configure it (forwarded via AMPLIO_BRIDGE_SPEC) and form part of the
//     reuse key, so identical specs share one long-lived subprocess while
//     distinct ones get their own.
//
// See bridges/README.md for the contract a bridge must implement.
package bridge

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"amplio/internal/llm"
)

// maxLine bounds a single NDJSON line (a final response can be large: full
// content plus tool-call arguments).
const maxLine = 16 * 1024 * 1024

// protocolVersionHeader announces which revision of wire.go we speak. The two
// ends are separate builds — a bridge is by definition not compiled with us —
// so a mismatch has to be able to fail loudly rather than as a puzzling decode
// error. Absent means 1, which is what every bridge written before this sends.
const (
	protocolVersionHeader = "X-Amplio-Bridge-Protocol"
	protocolVersion       = "1"
)

// transport carries one request to a bridge and hands back its streaming
// response. It is the only thing that differs between a bridge we spawn and a
// bridge someone else is running: everything above it — the wire types, request
// building, NDJSON decoding — is transport-agnostic.
type transport interface {
	// post sends body to path (e.g. "/generate") and returns the response, whose
	// body becomes the caller's to close.
	post(ctx context.Context, path string, body []byte) (*http.Response, error)
	// describe names the far end for error messages.
	describe() string
}

type provider struct {
	tr        transport
	model     string
	maxTokens int
}

// ClientArgsSubprocess are the arguments this transport interprets — the `{k=v}`
// block in a spec (see internal/llm/spec.go): subprocess{bin=/path/to/bridge}:MODEL.
//
// bin= rather than the model position, so that a spec says which MODEL it is
// wherever it is displayed; specs stored before this put the binary there, and
// every model behind one bridge rendered identically.
var ClientArgsSubprocess = map[string]bool{
	"bin": true,
}

// NewSubprocess builds a provider for a bridge THIS process spawns and owns. binary is the bridge path (the spec's
// positional model field); model= is required (it is the real model id the
// bridge serves).
//
// The bridge is configured by the WHOLE spec query — client and model args alike
// — forwarded verbatim as AMPLIO_BRIDGE_SPEC, because from amplio's side a
// bridge is opaque: we cannot know which of its knobs configure the client and
// which configure the model. That union is also the reuse key, so two specs
// differing in any arg get their own subprocess.
func NewSubprocess(pos string, maxTokens int, clientArgs, args url.Values) (llm.Provider, error) {
	binary, model := clientArgs.Get("bin"), pos
	if binary == "" {
		return nil, fmt.Errorf("subprocess provider: no bridge path; want subprocess{bin=/path/to/bridge}:MODEL")
	}
	if model == "" {
		return nil, fmt.Errorf("subprocess provider: no model; want subprocess{bin=%s}:MODEL", binary)
	}

	// What the bridge is told. model= is part of the contract (bridges/README.md)
	// so it is always present regardless of which spelling the operator used;
	// bin= is ours and would be meaningless to the bridge. Everything else rides
	// along, because from amplio's side a bridge is opaque — we cannot know which
	// of its knobs configure the client and which configure the model.
	all := url.Values{}
	for k, v := range args {
		all[k] = v
	}
	for k, v := range clientArgs {
		all[k] = v
	}
	delete(all, "bin")
	all.Set("model", model)
	q := all.Encode() // sorted, canonical

	// The reuse key is the binary plus everything the bridge was told, so two
	// specs that differ in any way it can observe get their own process — and two
	// spellings of the SAME endpoint share one, which is why the key is built
	// from the resolved values rather than from the spec text.
	return &provider{
		tr:        &spawnTransport{key: binary + "?" + q, binary: binary, specQuery: q},
		model:     model,
		maxTokens: maxTokens,
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

// post sends the request to the bridge's /generate. The returned response body
// is owned by the caller and must be closed.
func (p *provider) post(ctx context.Context, req llm.Request) (*http.Response, error) {
	body, err := json.Marshal(p.toWire(req))
	if err != nil {
		return nil, fmt.Errorf("bridge: marshal request: %w", err)
	}
	resp, err := p.tr.post(ctx, "/generate", body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
		return nil, &Error{
			Code: codeForStatus(resp.StatusCode),
			Message: fmt.Sprintf("bridge %s: HTTP %d: %s",
				p.tr.describe(), resp.StatusCode, strings.TrimSpace(string(b))),
		}
	}
	return resp, nil
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
			return nil, &Error{Code: CodeProtocol, Message: fmt.Sprintf("bridge: bad protocol line: %s", err)}
		}
		switch wl.Type {
		case "final":
			if wl.Response == nil {
				return nil, fmt.Errorf("bridge: final line without a response")
			}
			return wl.Response.toLLM(), nil
		case "error":
			return nil, &Error{Code: cmp.Or(wl.Code, CodeProvider), Message: "bridge error: " + wl.Error}
		}
		// deltas and unknown line types are ignored for the blocking path
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("bridge: read stream: %w", err)
	}
	return nil, fmt.Errorf("bridge: stream ended without a final response")
}

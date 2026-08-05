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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"amplio/internal/llm"
)

// The serving side of the protocol: this is what makes an amplio itself a bridge.
// It lives beside the client in one package so the two halves of
// the wire format cannot drift, and so the round-trip test that guards
// losslessness is a plain in-package test.
//
// POLICY IS NOT HERE. The handler takes a Resolver and asks it to turn a handle
// into a provider; the mount decides what a handle means, which handles are
// allowed, and who may ask. A standalone bridge resolves against its config; the
// amplio server resolves against its model menu and refuses anything else. This
// package supplies the protocol and nothing more.

const (
	// pingInterval keeps an idle-looking connection alive and, more importantly,
	// tells the client the difference between "the model is still thinking" and
	// "the connection died". Minutes of silence are normal during reasoning, and
	// TCP keepalive defaults to hours, so without this a client cannot set an
	// idle timeout at all. Matches the SSE stream's cadence.
	pingInterval = 15 * time.Second

	// writeTimeout bounds a single line write+flush. Without it a stalled reader
	// wedges the handler goroutine until the OS gives up on the connection.
	writeTimeout = 30 * time.Second
)

// Resolver turns a model handle from the wire into a provider. Returning an
// error refuses the request; the error text reaches the caller verbatim, so it
// should say what was asked for and what would have been acceptable.
type Resolver func(ctx context.Context, handle string) (llm.Provider, error)

// GenerateHandler serves POST /generate: a wireRequest in, an NDJSON stream of
// wireLines out.
func GenerateHandler(resolve Resolver) http.HandlerFunc {
	return generateHandler(resolve, pingInterval)
}

// generateHandler takes the keepalive cadence as a parameter so a test can use a
// short one. A mutable package var would be simpler and would also be a data
// race against every in-flight handler.
func generateHandler(resolve Resolver, pingEvery time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req wireRequest
		if err := json.NewDecoder(io.LimitReader(r.Body, maxLine)).Decode(&req); err != nil {
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if v := r.Header.Get(protocolVersionHeader); v != "" && v != protocolVersion {
			http.Error(w, fmt.Sprintf("unsupported bridge protocol %q; this server speaks %s", v, protocolVersion),
				http.StatusBadRequest)
			return
		}
		provider, err := resolve(r.Context(), req.Model)
		if err != nil {
			// Refusals are HTTP-level so a caller can tell "you may not ask for
			// that" from "the model failed", which are differently actionable.
			http.Error(w, err.Error(), http.StatusForbidden)
			return
		}
		stream(w, r, provider, req, pingEvery)
	}
}

func stream(w http.ResponseWriter, r *http.Request, provider llm.Provider, req wireRequest, pingEvery time.Duration) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	// Proxies that honour it stop buffering our lines; a harmless no-op on those
	// that don't. Without this a reverse proxy can hold every delta and release
	// them in one batch at the end, which looks exactly like a hung model.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	rc := http.NewResponseController(w)
	enc := &lineWriter{w: w, rc: rc}
	_ = enc.flush() // headers out now, so the client sees an established stream

	// Ping while the model thinks: Stream blocks until its first token, which for
	// a reasoning model can be minutes.
	//
	// The pinger is JOINED, not merely signalled. A ResponseWriter is only valid
	// until its handler returns, so a ping goroutine still inside Write when
	// stream() returns panics in net/http with a nil-pointer dereference — which
	// is exactly what happened before this defer waited for it.
	stop, pinged := make(chan struct{}), make(chan struct{})
	defer func() {
		close(stop)
		<-pinged
	}()
	go func() {
		defer close(pinged)
		t := time.NewTicker(pingEvery)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				enc.write(wireLine{Type: "ping"})
			}
		}
	}()

	st, err := provider.Stream(r.Context(), req.toLLM())
	if err != nil {
		enc.write(errorLine(err))
		return
	}
	defer st.Close()

	for st.Next() {
		if !enc.write(eventToWire(st.Event())) {
			return // client gone or stalled; nothing useful left to do
		}
	}
	if err := st.Err(); err != nil {
		enc.write(errorLine(err))
		return
	}
	enc.write(wireLine{Type: "final", Response: responseToWire(st.Response())})
}

// errorLine carries a failure to the caller VERBATIM. That matters more than it
// looks: amplio classifies a context-window overflow by asking a model to read
// the provider's error text, so wrapping "prompt is too long" in our own words
// silently disables compaction on the far side. Provenance is added by the
// client, which knows which bridge it was talking to.
func errorLine(err error) wireLine {
	return wireLine{Type: "error", Code: CodeProvider, Error: err.Error()}
}

// lineWriter serialises NDJSON lines with a per-write deadline, flushing each
// one so deltas arrive as they are produced. Writes are serialised because the
// ping goroutine shares the connection with the response.
type lineWriter struct {
	w  http.ResponseWriter
	rc *http.ResponseController
	mu sync.Mutex
}

func (lw *lineWriter) write(line wireLine) bool {
	lw.mu.Lock()
	defer lw.mu.Unlock()
	b, err := json.Marshal(line)
	if err != nil {
		slog.Error("bridge server: marshal line", "err", err)
		return false
	}
	_ = lw.rc.SetWriteDeadline(time.Now().Add(writeTimeout)) // best effort
	if _, err := lw.w.Write(append(b, '\n')); err != nil {
		return false
	}
	return lw.flush() == nil
}

func (lw *lineWriter) flush() error {
	if err := lw.rc.Flush(); err != nil && !errors.Is(err, http.ErrNotSupported) {
		return err
	}
	return nil
}

// Embedder is the far side's embedding model. Declared here rather than imported
// so this package keeps depending only on internal/llm; embed.Embedder satisfies
// it structurally.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	ModelID() string
}

// EmbedHandler serves POST /embed.
//
// Unlike generation there is no menu to resolve against: embedders are
// configured singly (one embed_model per server), and a second embedding space
// would silently corrupt an index rather than fail. So this serves exactly the
// embedder it was given, and a caller asking for a different one is refused
// rather than quietly served the wrong vectors.
func EmbedHandler(e Embedder) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req wireEmbedRequest
		if err := json.NewDecoder(io.LimitReader(r.Body, maxLine)).Decode(&req); err != nil {
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if req.Model != "" && req.Model != e.ModelID() {
			http.Error(w, fmt.Sprintf("this bridge serves %q, not %q; mixing embedding spaces would corrupt your index",
				e.ModelID(), req.Model), http.StatusForbidden)
			return
		}
		vectors, err := e.Embed(r.Context(), req.Texts)
		if err != nil {
			writeJSON(w, http.StatusOK, wireEmbedResponse{Model: e.ModelID(), Error: err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, wireEmbedResponse{Model: e.ModelID(), Vectors: vectors})
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("bridge server: write response", "err", err)
	}
}

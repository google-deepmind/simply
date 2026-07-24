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
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"amplio/internal/llm"
)

// subStream adapts the bridge's NDJSON /generate response to llm.Stream. It owns
// the HTTP response body and releases it in Close.
type subStream struct {
	resp  *http.Response
	sc    *bufio.Scanner
	cur   llm.StreamEvent
	final *llm.Response
	err   error
}

func (s *subStream) Next() bool {
	for s.sc.Scan() {
		line := bytes.TrimSpace(s.sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var wl wireLine
		if err := json.Unmarshal(line, &wl); err != nil {
			s.err = fmt.Errorf("subprocess: bad protocol line: %w", err)
			return false
		}
		switch wl.Type {
		case "delta":
			s.cur = wl.toStreamEvent()
			return true
		case "final":
			if wl.Response != nil {
				s.final = wl.Response.toLLM()
			}
			return false
		case "error":
			s.err = fmt.Errorf("bridge error: %s", wl.Error)
			return false
		}
		// unknown line types are skipped
	}
	if err := s.sc.Err(); err != nil {
		s.err = fmt.Errorf("subprocess: read stream: %w", err)
	}
	return false
}

func (s *subStream) Event() llm.StreamEvent { return s.cur }

func (s *subStream) Response() *llm.Response {
	if s.final != nil {
		return s.final
	}
	return &llm.Response{}
}

func (s *subStream) Err() error { return s.err }

func (s *subStream) Close() { _ = s.resp.Body.Close() }

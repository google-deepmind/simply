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
	"errors"
	"net/http"
)

// Error classes. A bridge failure is one of four things, and they call for
// different responses: fix the spec, fix the token, wait and retry, or file a
// bug. Without a class, a caller can only pattern-match error text — which is
// how the harness ends up asking a model to read English to decide what to do.
const (
	// CodeProvider: the far side reached the model and the model (or its API)
	// failed. Retryable in general; the MESSAGE is the interesting part, and it
	// is carried verbatim because amplio classifies context-window overflow by
	// reading it.
	CodeProvider = "provider"
	// CodeNotAllowed: the far side refused to run this handle — not on its menu,
	// ambiguous, or carrying client args a caller may not supply. Retrying is
	// pointless; the spec has to change.
	CodeNotAllowed = "not_allowed"
	// CodeUnauthorized: bad or missing token. Also pointless to retry.
	CodeUnauthorized = "unauthorized"
	// CodeProtocol: the two ends disagree — an unparseable line, an unexpected
	// status, a version mismatch. Usually a version skew between the amplio that
	// dials and the one that answers.
	CodeProtocol = "protocol"
)

// Error is a bridge failure with a machine-readable class.
//
// Error() returns the message UNWRAPPED beyond the provenance prefix the client
// adds, because the text is load-bearing: rewriting an upstream "input is too
// long" would silently disable compaction on this side of the hop.
type Error struct {
	Code    string
	Message string
}

func (e *Error) Error() string { return e.Message }

// CodeOf reports the class of a bridge failure, or "" if err did not come from
// a bridge.
func CodeOf(err error) string {
	var be *Error
	if errors.As(err, &be) {
		return be.Code
	}
	return ""
}

// codeForStatus maps an HTTP failure onto a class. Refusals arrive as statuses
// rather than as stream lines because they happen before any generation starts.
func codeForStatus(status int) string {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		if status == http.StatusUnauthorized {
			return CodeUnauthorized
		}
		return CodeNotAllowed
	case http.StatusNotFound:
		// Lending is off, or the endpoint URL is wrong — either way the caller's
		// configuration is what has to change.
		return CodeNotAllowed
	default:
		return CodeProtocol
	}
}

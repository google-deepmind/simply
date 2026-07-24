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

package citc

import (
	"context"
	"errors"

	"amplio/internal/workspace"
)

// Kind is the JSON discriminator for citc workspaces.
const Kind = "citc"

// VCS values backing a citc workspace.
const (
	VcsJJ  = "jj"
	VcsFig = "fig"
)

// ErrNotAvailable is returned by the OSS stubs when a caller invokes a citc
// operation that requires the corp backend (only present in the internal
// build). Stubs return this rather than panicking so off-corp tooling can
// gracefully degrade.
var ErrNotAvailable = errors.New("citc: not available in this build")

// AliasCache is the minimal surface citc needs from a cache implementation.
// Production wires aliascache; tests can pass any AliasCache impl. Defined
// here (not in the cache package) so the dependency arrow points cache → citc
// rather than the other way around.
type AliasCache interface {
	Get(user string, id int) string
	Set(user string, id int, alias string)
}

// The public package surface is provided as function variables so the
// internal-build init() can swap in real implementations. The OSS build
// keeps these stubs and the corp behavior is simply absent at runtime;
// callers (resolver, server, serve.go) compile against the stubs and
// receive sensible zero values / errors.

// Info reports the CitC identity (user, numericID, vcs) of a workspace. OK is
// false for non-CitC workspaces. The OSS stub always returns false.
var Info = func(w workspace.Workspace) (user string, numericID int, vcs string, ok bool) {
	return "", 0, "", false
}

// LookupAlias reads the workspace's current primary alias directly from disk
// (bypassing any cache). The OSS stub returns "".
var LookupAlias = func(user string, numericID int) string { return "" }

// RefreshAlias forces a synchronous re-read of the alias and writes it back
// through the installed cache. The OSS stub returns "".
var RefreshAlias = func(user string, numericID int) string { return "" }

// SetAliasCache installs the process-global alias cache. The OSS stub is a
// no-op (no cache is ever consulted because no citc workspaces exist).
var SetAliasCache = func(c AliasCache) {}

// Detect returns a citc workspace for a CitC path. The OSS stub always
// returns ok=false; the resolver falls through to jj/git/plain.
var Detect = func(path string) (workspace.Workspace, bool) { return nil, false }

// CreateAnonymous creates an anonymous CitC workspace via the corp CLI. The
// OSS stub returns ErrNotAvailable.
var CreateAnonymous = func(ctx context.Context, vcs, user string) (workspace.Workspace, error) {
	return nil, ErrNotAvailable
}

// Undelete re-attaches an alias to an anonymous CitC workspace. The OSS stub
// returns ErrNotAvailable.
var Undelete = func(ctx context.Context, user string, numericID int, newAlias, vcs string) error {
	return ErrNotAvailable
}

// ResolveRootPath resolves a `//`-prefixed, repo-root-relative path to an
// absolute filesystem path, given the current working directory. The OSS stub
// always returns ok=false.
var ResolveRootPath = func(cwd, path string) (abs string, ok bool) { return "", false }

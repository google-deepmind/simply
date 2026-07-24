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

// Package resolver maps an operator-supplied workspace string to a concrete
// workspace backend. Resolve handles the full grammar (creation sentinels +
// path auto-detection) at run start; Wrap is the path-only detector installed
// as the manager's WorkspaceFactory.
//
// Public surface (Resolve, Wrap) is dual-built: the OSS variant
// (resolver_oss.go, !internal) supports only path-based wrapping; the
// internal variant (resolver_internal.go, internal) layers CitC sentinels
// and citc.Detect on top. The detectChain helper below is shared by both.
package resolver

import (
	"context"
	"time"

	"amplio/internal/workspace"
	"amplio/internal/workspace/git"
	"amplio/internal/workspace/jj"
	"amplio/internal/workspace/plain"
)

// detectTimeout bounds the VCS-detection probes (a couple of `git`/`jj`
// invocations) so a slow/hung filesystem can't stall run start or resume.
const detectTimeout = 10 * time.Second

// detectChain runs the path-only backend detection used by both build
// flavors: jj (checked before git so a colocated jj/git repo is treated as
// jj), then git, otherwise plain. Internal Wrap layers citc.Detect on top
// (see resolver_internal.go); OSS Wrap calls this directly.
func detectChain(path string) workspace.Workspace {
	ctx, cancel := context.WithTimeout(context.Background(), detectTimeout)
	defer cancel()
	if ws, ok := jj.Detect(ctx, path); ok {
		return ws
	}
	if ws, ok := git.Detect(ctx, path); ok {
		return ws
	}
	return plain.New(path)
}

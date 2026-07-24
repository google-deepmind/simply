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

//go:build !internal

package resolver

import (
	"fmt"
	"strings"

	"amplio/internal/workspace"
)

// Resolve in the OSS build supports only path-based workspaces. The
// new:/anon:/citc: sentinels require the CitC backend (internal build) and
// are rejected with a clear error here, rather than silently degrading to a
// plain workspace at a bogus path.
func Resolve(raw, _ string) (workspace.Workspace, error) {
	if raw == "" {
		return nil, fmt.Errorf("resolve: workspace must be non-empty")
	}
	if strings.HasPrefix(raw, "new:") || strings.HasPrefix(raw, "anon:") || strings.HasPrefix(raw, "citc:") {
		return nil, fmt.Errorf("resolve: workspace sentinel %q is only supported in the internal build", raw)
	}
	return Wrap(raw), nil
}

// Wrap auto-detects the backend for an existing path: a jj repo → jj
// (checked before git so a colocated jj/git repo is treated as jj), a git
// work tree → git, otherwise plain. It matches runtime.WorkspaceFactory.
func Wrap(path string) workspace.Workspace {
	return detectChain(path)
}

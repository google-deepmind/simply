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

package server

import (
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// extraWorkspaceModes lists the workspace sources this build offers beyond a
// plain path, in the order the picker should show them. A build that cannot
// resolve a mode must not advertise it: the picker would offer a choice that
// fails at run start. Empty here; builds that add sources install their own.
var extraWorkspaceModes = func() []string { return nil }

// recentWorkspacesLimit caps the New-Run workspace picker. It's a recency
// quick-select, not an exhaustive browser — the operator can always type an
// alias not in the list.
const recentWorkspacesLimit = 50

// workspaceRecentRoot is the per-user directory whose children are offered as
// recent workspaces. Empty here (nothing to list); builds with a managed
// workspace root install their own.
var workspaceRecentRoot = ""

// workspaceInfo feeds the New-Run workspace control: the extra workspace
// sources this build offers, the path to prefill (the server's cwd — the
// "inherit" default), and recent workspaces to quick-select.
type workspaceInfo struct {
	Modes      []string `json:"workspace_modes"`
	ServerRoot string   `json:"server_root"`
	Recent     []string `json:"recent"`
}

func (s *Server) handleWorkspaces(w http.ResponseWriter, r *http.Request) {
	user := os.Getenv("USER")
	writeJSON(w, http.StatusOK, workspaceInfo{
		Modes:      extraWorkspaceModes(),
		ServerRoot: serverCwd(),
		Recent:     listRecentWorkspaces(filepath.Join(workspaceRecentRoot, user)),
	})
}

func serverCwd() string {
	if wd, err := os.Getwd(); err == nil {
		return wd
	}
	return "."
}

// listRecentWorkspaces returns the names of root's immediate subdirectories,
// most-recently-modified first, capped at recentWorkspacesLimit. On a networked
// FS each attached alias is a directory whose mtime tracks last use, so a plain
// mtime sort yields recency with no shell-out. Best-effort: any FS error yields
// an empty list so the picker degrades to free-form entry.
func listRecentWorkspaces(root string) []string {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	type item struct {
		name string
		mod  time.Time
	}
	items := make([]item, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		fi, err := e.Info()
		if err != nil {
			continue
		}
		items = append(items, item{e.Name(), fi.ModTime()})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].mod.After(items[j].mod) })
	if len(items) > recentWorkspacesLimit {
		items = items[:recentWorkspacesLimit]
	}
	names := make([]string, len(items))
	for i, it := range items {
		names[i] = it.name
	}
	return names
}

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

package workspace

import (
	"context"
	"encoding/json"
	"fmt"
)

// SessionMetadataKey is the Session.Metadata key under which a session's
// serialized Workspace is persisted (so resume reconstructs the session's OWN
// workspace, not one re-derived from run config — essential once sub-agents get
// their own linked workspaces).
const SessionMetadataKey = "workspace"

// Workspace abstracts a VCS-managed working directory. Agents interact with
// files and VCS via tools (bash, view_file, edit_file), not through this
// interface; the workspace provides identity, sub-agent isolation (linked
// workspaces), and a JSON round-trip for persistence.
//
// Implementations are pure data + methods (no non-serializable dependencies):
// the struct that satisfies this interface IS what gets serialized via Marshal,
// so each backend lives in its own package, registers an Unmarshaler in init(),
// and the entrypoint blank-imports the backends it wants (OSS builds can omit
// corp-only backends).
type Workspace interface {
	// Root returns the absolute path to the working directory.
	Root() string

	// Name returns a cheap, human-readable identifier (directory basename,
	// worktree name, or numeric id). Pure — never does I/O. For a live,
	// possibly-mutable display name (e.g. a CitC alias) use ResolveAlias.
	Name() string

	// Kind returns the backend discriminator stamped into the JSON form
	// (e.g. "plain", "git", "jj", "citc") and used by Unmarshal to dispatch.
	Kind() string

	// CreateLinked creates an isolated workspace that shares commit history but
	// has an independent working copy, named after childSessionID. Used for
	// sub-agent parallelism (each sub-agent edits without conflicts). Returns an
	// error for backends that don't support linking.
	CreateLinked(ctx context.Context, childSessionID string) (Workspace, error)

	// LinkedFrom returns the Root of the workspace this one was linked from, or
	// "" if it wasn't created via CreateLinked (a run root or a directly-wrapped
	// path). Provenance for display ("linked from …"), lineage, and operator
	// promotion flows — the parent isn't otherwise recoverable, since a linked
	// worktree only records the shared repo, not which sibling spawned it.
	LinkedFrom() string

	// Describe renders this workspace's bootstrap description body from the
	// agent's own perspective: working directory, VCS, and (for a sub-agent) its
	// link/share provenance to parentSessionID ("" for a run root). It is the one
	// place a live alias is resolved (once per fresh session). The eventloop
	// seeds the result as a step-0 SystemEvent with marker "workspace".
	// Best-effort: returns "" when there is nothing useful to say.
	Describe(ctx context.Context, parentSessionID string) string

	// Validate checks the workspace is usable RIGHT NOW (does I/O): the path
	// exists and is the right shape. Called at run start and before resume so a
	// vanished workspace fails clearly instead of surfacing deep in a later tool
	// call. Deserialization (Unmarshal) deliberately does NOT call this.
	Validate(ctx context.Context) error

	// ResolveAlias reads the workspace's current, possibly-mutable display alias
	// (does I/O). "" when the backend has no alias concept or none is attached.
	// Not folded into Name() precisely because it does I/O; callers that show it
	// in a UI should memoize per request (an out-of-band rename must surface on
	// the next read, so never cache it process-wide).
	ResolveAlias(ctx context.Context) (string, error)
}

// ProvenanceLine renders a sub-agent's relationship to its spawning parent for
// the workspace bootstrap description: "" for a run root (parentSessionID == ""),
// otherwise a linked-vs-shared sentence naming the parent session. A linked
// workspace cites the VCS log command (shareGraphHint, e.g. "jj log") the agent
// can run to see sibling commits. Shared by the backends' Describe methods.
func ProvenanceLine(parentSessionID string, linked bool, shareGraphHint string) string {
	if parentSessionID == "" {
		return ""
	}
	if linked {
		return fmt.Sprintf(
			"This workspace is linked from the workspace of the parent session `%s`; "+
				"your file edits and commits are isolated, but you share the commit graph "+
				"(`%s` shows sibling commits).", parentSessionID, shareGraphHint)
	}
	return fmt.Sprintf("This workspace is shared with the parent session `%s`.", parentSessionID)
}

// Unmarshaler reconstructs a Workspace of one Kind from its JSON form.
type Unmarshaler func(data []byte) (Workspace, error)

var registry = map[string]Unmarshaler{}

// RegisterKind registers a backend's Unmarshaler under its Kind discriminator.
// Backends call this from init(); panics on a duplicate kind (a wiring bug).
func RegisterKind(kind string, fn Unmarshaler) {
	if _, dup := registry[kind]; dup {
		panic(fmt.Sprintf("workspace: kind %q already registered", kind))
	}
	registry[kind] = fn
}

// Marshal serializes a Workspace to JSON, stamping its Kind discriminator
// (mirrors event.Marshal). The result round-trips through Unmarshal.
func Marshal(w Workspace) ([]byte, error) {
	data, err := json.Marshal(w)
	if err != nil {
		return nil, err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	kindBytes, _ := json.Marshal(w.Kind())
	m["kind"] = kindBytes
	return json.Marshal(m)
}

// Unmarshal reconstructs a Workspace by dispatching on the "kind" discriminator
// to the registered backend. PURE: structural only — it does NO filesystem I/O
// and never checks that the path still exists (call Workspace.Validate for
// that), so read paths (e.g. building DTOs for a run whose dir was later
// deleted) never hard-fail. Errors on a missing/unknown kind.
func Unmarshal(data []byte) (Workspace, error) {
	var probe struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("workspace unmarshal probe: %w", err)
	}
	if probe.Kind == "" {
		return nil, fmt.Errorf("workspace unmarshal: missing 'kind' field")
	}
	fn, ok := registry[probe.Kind]
	if !ok {
		return nil, fmt.Errorf("workspace unmarshal: unknown kind %q (backend not imported?)", probe.Kind)
	}
	return fn(data)
}

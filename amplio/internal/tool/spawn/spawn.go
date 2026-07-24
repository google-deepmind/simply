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

// Package spawn provides the tool for spawning sub-agent sessions.
package spawn

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"amplio/internal/agent"
	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/tool"
	"amplio/internal/util"
)

type Params struct {
	Task          string `json:"task" jsonschema:"required" jsonschema_description:"Task description for the sub-agent"`
	AgentType     string `json:"agent_type,omitempty" jsonschema_description:"Agent type to spawn (default: standard_agent)"`
	WorkspaceMode string `json:"workspace_mode,omitempty" jsonschema_description:"share (default): work in the same workspace as the parent. link: an isolated linked workspace (git/jj worktree or CitC link) sharing history but with independent file edits — use for parallel sub-agents that must not clobber each other."`
}

func New(env *agent.Env, parentSessionID string) *tool.Tool {
	return &tool.Tool{
		Name: "spawn_agent",
		Description: "Spawn an autonomous sub-agent to work on a task in parallel. It runs independently and " +
			"notifies you when it completes. A concluded sub-agent can be re-engaged via `send_message`. " +
			"if it is not running, it will be revived with full prior context. If there is follow-up " +
			"work that benefits from prior context, re-engage an existing sub-agent instead of spawning a fresh one.",
		ParamType:       &Params{},
		SessionRequired: true,
		Execute:         makeExecutor(env, parentSessionID),
	}
}

func makeExecutor(env *agent.Env, parentSessionID string) tool.Executor {
	return func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
		params, errResult := tool.ParseArgs[Params](args)
		if errResult != nil {
			return errResult, nil
		}

		agentType := params.AgentType
		if agentType == "" {
			agentType = "standard_agent"
		}

		factory, err := agent.Get(agentType)
		if err != nil {
			return &tool.Result{
				Content: fmt.Sprintf("Error: %s", err),
				IsError: true,
			}, nil
		}

		// Allocate a unique session ID from the run's shared allocator (seeded
		// from the DB, reserved in-memory under a lock — so it excludes cold
		// sessions and never collides with a concurrent spawn).
		sessionID, err := env.Names.Next(ctx)
		if err != nil {
			return &tool.Result{
				Content: fmt.Sprintf("Error allocating session id: %s", err),
				IsError: true,
			}, nil
		}

		// Workspace: share the parent's by default, or create an isolated linked
		// workspace (worktree / CitC link) named after the child session so
		// parallel sub-agents don't clobber each other's files.
		childEnv := env
		linkedInfo := ""
		if params.WorkspaceMode == "link" {
			linked, err := env.Workspace.CreateLinked(ctx, sessionID)
			if err != nil {
				return &tool.Result{
					Content: fmt.Sprintf("Error creating linked workspace: %s", err),
					IsError: true,
				}, nil
			}
			e := *env
			e.Workspace = linked
			childEnv = &e
			linkedInfo = fmt.Sprintf(" in linked workspace %s", linked.Root())
		}

		// Claim the registry slot SYNCHRONOUSLY (before spawning the goroutine) so
		// the run-registry instance can't be orphaned by a concurrent RemoveIfEmpty,
		// and hand the resulting handle to the child via Config. Parent ctx is
		// context.Background() (NOT the parent's tool ctx) so the child keeps running
		// after the parent finishes/idle-exits (keep-alive). A freshly-allocated
		// session id can't already be registered, but guard defensively.
		childCtx, handle, release, ok := childEnv.Registry.RegisterAndContext(context.Background(), sessionID)
		if !ok {
			return &tool.Result{
				Content: fmt.Sprintf("Error: session %q already running", sessionID),
				IsError: true,
			}, nil
		}

		child, err := factory(childEnv, &agent.Config{
			SessionID: sessionID,
			Task:      params.Task,
			ParentID:  parentSessionID,
			Handle:    handle,
		})
		if err != nil {
			release() // release the claimed slot on a pre-launch failure
			return &tool.Result{
				Content: fmt.Sprintf("Error creating sub-agent: %s", err),
				IsError: true,
			}, nil
		}

		// Launch in a background goroutine on childCtx, a context independent of the
		// parent's tool ctx, so the parent finishing/idle-exiting does not cancel the
		// child (keep-alive). The child's loop posts its own result to the parent
		// atomically (TerminateAndNotifyParent) on conclude/crash/cancel; this runner
		// only backstops a panic that bypasses that path. release (cancel +
		// Unregister) is deferred FIRST so it runs LAST — after the panic backstop.
		go func() {
			defer release()
			defer func() {
				if r := recover(); r != nil {
					slog.Error("sub-agent panicked", "session", sessionID, "panic", r)
					msg := fmt.Sprintf("panic: %v", r)
					// -1: a panic backstop has no specific turn step; use current.
					// childEnv (not env) for consistency — they share Store/RunID today,
					// but only childEnv is guaranteed to describe the child.
					_ = childEnv.Store.TerminateAndNotifyParent(context.Background(),
						childEnv.RunID, sessionID, parentSessionID, db.SessionCrashed, msg,
						&event.SystemEvent{Content: msg, Marker: event.MarkerError}, -1)
				}
			}()
			if err := child.Run(childCtx); err != nil {
				slog.Error("sub-agent run error", "session", sessionID, "error", err)
			}
		}()

		return &tool.Result{
			Content: fmt.Sprintf("Spawned sub-agent %q (type=%s)%s at %s for task: %s",
				sessionID, agentType, linkedInfo, util.FormatLocalISO(time.Now()), util.TruncateRunes(params.Task, 80)),
		}, nil
	}
}

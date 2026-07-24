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

package standard

import (
	"context"
	_ "embed"

	"amplio/internal/agent"
	"amplio/internal/agent/critic"
	"amplio/internal/agent/eventloop"
	"amplio/internal/blob"
	"amplio/internal/cli"
	"amplio/internal/config"
	"amplio/internal/session"
	"amplio/internal/tool"
	"amplio/internal/tool/bash"
	"amplio/internal/tool/coordination"
	"amplio/internal/tool/editfile"
	"amplio/internal/tool/inspect"
	"amplio/internal/tool/recall"
	"amplio/internal/tool/sessionsearch"
	"amplio/internal/tool/spawn"
	"amplio/internal/tool/viewfile"
)

//go:embed standard_agent.md
var systemPrompt string

// completionSnippet ends the assembled prompt: how an autonomous agent concludes.
const completionSnippet = "\n\n## Completion\n\n" +
	"When the overarching task is completely finished, give your final summary in " +
	"plain text with NO tool calls — the absence of tool calls signals completion. " +
	"Be concise and direct about the final state."

const AgentType = "standard_agent"

func init() {
	agent.Register(AgentType, factory)
}

func factory(env *agent.Env, cfg *agent.Config) (agent.Agent, error) {
	cwd := env.Workspace.Root()

	coordDeps := &coordination.Deps{
		Store:    env.Store,
		RunID:    env.RunID,
		Registry: env.Registry,
	}
	inspectDeps := &inspect.Deps{
		Store: env.Store,
		RunID: env.RunID,
	}

	// Use a pointer so the closure can reference it after assignment.
	var ag *eventloop.EventLoopAgent

	tools := []*tool.Tool{
		bash.New(cwd, env.RunID, cfg.SessionID),
		viewfile.New(cwd, config.ArtifactDir(env.RunID)),
		editfile.New(cwd, config.ArtifactDir(env.RunID)),
		coordination.SendMessage(coordDeps, cfg.SessionID),
		coordination.SessionCancel(coordDeps),
		coordination.AwaitEvent(coordDeps, cfg.SessionID, func() *session.Handle {
			return ag.Handle()
		}),
		inspect.SessionList(inspectDeps),
		inspect.SessionSteps(inspectDeps),
		inspect.SessionPeek(inspectDeps),
		inspect.SessionSummary(inspectDeps),
		critic.ViewRunReport(env.Store, env.RunID),
		sessionsearch.New(env.Store, env.RunID),
		spawn.New(env, cfg.SessionID),
	}
	// Recall tools + task-relevant seeding over whichever corpora are built (skills
	// and/or mined lessons). Add the tools whenever the index OBJECTS exist —
	// not just when they're currently built — because the skill index is built
	// in a background goroutine and may finish AFTER this agent is constructed.
	// The tools (recall.Search, recall.Load, recall.InitialContent) all gate
	// per-corpus with IsBuilt at call time, so an unbuilt index just skips that
	// corpus until it's ready, and a freshly-built index starts contributing on
	// the very next call without needing the agent to be respawned.
	var initialRecall func(ctx context.Context, task string) string
	if env.SkillIndex != nil || env.LessonIndex != nil {
		tools = append(tools, recall.Search(env.SkillIndex, env.LessonIndex), recall.Load(env.SkillIndex, env.LessonIndex))
		sIx, lIx := env.SkillIndex, env.LessonIndex
		initialRecall = func(ctx context.Context, task string) string {
			return recall.InitialContent(ctx, sIx, lIx, task)
		}
	}

	ag = eventloop.New(env, eventloop.Config{
		SessionID:    cfg.SessionID,
		Task:         cfg.Task,
		FirstMessage: cfg.FirstMessage,
		ParentID:     cfg.ParentID,
		Handle:       cfg.Handle,
		AgentType:    AgentType,
		SystemPrompt: systemPrompt +
			eventloop.ExecutionPrinciplesPromptSnippet +
			eventloop.EnvironmentPromptSnippet() +
			eventloop.ToolUsageStrategyPromptSnippet +
			eventloop.SubAgentStrategyPromptSnippet +
			eventloop.CrossRunInspectionPromptSnippet +
			completionSnippet +
			bash.ArtifactDirPromptSnippet,
		Tools:         tools,
		InitialRecall: initialRecall,
		CLITools:      cli.DefaultTools(),
		BlobStore:     blob.NewStore(config.BlobDir(env.RunID)),
	})

	return ag, nil
}

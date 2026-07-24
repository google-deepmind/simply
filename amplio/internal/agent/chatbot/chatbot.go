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

package chatbot

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

//go:embed chatbot_core.md
var corePrompt string

//go:embed chatbot_root.md
var rootPreamble string

//go:embed chatbot_sidecar.md
var sidecarPreamble string

var (
	rootPrompt    = rootPreamble + "\n\n" + corePrompt
	sidecarPrompt = sidecarPreamble + "\n\n" + corePrompt
)

// systemPromptFor picks the role-appropriate prompt. The chatbot is a sidecar
// iff the run already has another top-level (parentless) session that isn't a
// chatbot — i.e. an autonomous main-agent. Otherwise it's the run's root.
func systemPromptFor(env *agent.Env, sessionID string) string {
	sessions, err := env.Store.ListSessions(context.Background(), env.RunID)
	if err != nil {
		return rootPrompt // safe default; root prompt has no shared-workspace claims
	}
	for _, s := range sessions {
		if s.ParentID == "" && s.SessionID != sessionID && s.AgentType != AgentType {
			return sidecarPrompt
		}
	}
	return rootPrompt
}

// AgentType is the registry name (shared via config so the server can request it
// without importing this package).
const AgentType = config.ChatbotAgentType

func init() {
	agent.Register(AgentType, factory)
}

func factory(env *agent.Env, cfg *agent.Config) (agent.Agent, error) {
	cwd := env.Workspace.Root()

	coordDeps := &coordination.Deps{Store: env.Store, RunID: env.RunID, Registry: env.Registry}
	inspectDeps := &inspect.Deps{Store: env.Store, RunID: env.RunID}

	var ag *eventloop.EventLoopAgent

	// Same toolset as the standard agent: a chatbot can work directly or delegate
	// to spawned sub-agents.
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
	// Add the recall tools whenever the index OBJECTS exist — not just when
	// they're currently built. The skill index is built in a background
	// goroutine that may finish AFTER this agent is constructed; the tools
	// gate per-corpus with IsBuilt at call time, so an unbuilt index just
	// skips that corpus until it's ready (no respawn needed). Mirrors standard.go.
	if env.SkillIndex != nil || env.LessonIndex != nil {
		tools = append(tools, recall.Search(env.SkillIndex, env.LessonIndex), recall.Load(env.SkillIndex, env.LessonIndex))
	}

	ag = eventloop.New(env, eventloop.Config{
		SessionID:    cfg.SessionID,
		Task:         cfg.Task,
		FirstMessage: cfg.FirstMessage,
		ParentID:     cfg.ParentID,
		Handle:       cfg.Handle,
		AgentType:    AgentType,
		SystemPrompt: systemPromptFor(env, cfg.SessionID) +
			eventloop.EnvironmentPromptSnippet() +
			eventloop.ToolUsageStrategyPromptSnippet +
			eventloop.SubAgentStrategyPromptSnippet +
			eventloop.CrossRunInspectionPromptSnippet +
			bash.ArtifactDirPromptSnippet,
		Tools:       tools,
		Interactive: true,
		CLITools:    cli.DefaultTools(),
		BlobStore:   blob.NewStore(config.BlobDir(env.RunID)),
	})

	return ag, nil
}

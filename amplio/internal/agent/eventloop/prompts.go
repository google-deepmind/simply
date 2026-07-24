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

package eventloop

import (
	_ "embed"
)

// Shared system-prompt sections composed by the standard and chatbot agents.
// Kept here (the parent package) so the wording stays in lockstep across
// agents. Each file begins with its own heading and leading blank line, so
// callers concatenate them directly.

// ExecutionPrinciplesPromptSnippet is the autonomous-agent mindset (ambition,
// scope, exhaustiveness, sustainability, fast feedback). Intended for autonomous
// agents only — it deliberately contradicts the interactive chatbot's
// small-turns / wait-for-the-operator norms.
//
//go:embed prompts/execution_principles.md
var ExecutionPrinciplesPromptSnippet string

// ToolUsageStrategyPromptSnippet is shared by standard + chatbot.
//
//go:embed prompts/tool_usage_strategy.md
var ToolUsageStrategyPromptSnippet string

// SubAgentStrategyPromptSnippet is shared by standard + chatbot.
//
//go:embed prompts/sub_agent_strategy.md
var SubAgentStrategyPromptSnippet string

// CrossRunInspectionPromptSnippet is shared by standard + chatbot: the
// inspection tools accept an optional run_id to read a prior run's data.
//
//go:embed prompts/cross_run_inspection.md
var CrossRunInspectionPromptSnippet string

// EnvironmentPromptSnippet returns the 1P internal environment guidance, or ""
// in OSS. The OSS build keeps this stub; the internal build overrides it via
// init() in prompts_internal.go and serves the embedded MD.
var EnvironmentPromptSnippet = func() string { return "" }

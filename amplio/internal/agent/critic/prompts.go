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

package critic

// criticCitationConventions is the CITATION CONVENTIONS bullet list substituted
// into keen_critic.md (the __CITATION_CONVENTIONS__ placeholder) when the system
// prompt is assembled. Build-split: this neutral OSS default keeps the run
// report's citations environment-agnostic, while prompts_internal.go swaps in
// the concrete corp identifier shapes so the internal report cites
// experiments/changes in canonical short form.
var criticCitationConventions = "  * Experiment / job run: `<id>`\n" +
	"  * Change / commit: the id or hash your review tool or VCS shows\n" +
	"  * Trajectory step: `step <N>` or `step <N>-<M>` (optionally `session <id> step <N>`\n" +
	"  * File path: verbatim (workspace-relative or absolute)\n" +
	"  * Metric: `<key>=<value>` (e.g. `eval_accuracy=0.6953`)"

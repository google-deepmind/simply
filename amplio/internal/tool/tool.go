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

package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"runtime/debug"

	"amplio/internal/event"
	"amplio/internal/llm"

	"github.com/invopop/jsonschema"
	"golang.org/x/sync/errgroup"
)

// Result is the return value from a tool executor.
type Result struct {
	Content     string
	Attachments []event.Attachment
	IsError     bool
}

// Executor is the function signature for tool execution. The args parameter
// is the raw JSON from the LLM's tool call. The executor is responsible for
// parsing it into the expected struct.
type Executor func(ctx context.Context, args json.RawMessage) (*Result, error)

// Tool defines a single tool available to the agent.
type Tool struct {
	Name            string
	Description     string
	ParamType       any // pointer to zero value of the param struct, for schema generation
	Execute         Executor
	Exclusive       bool // if true, must be the only tool call in the step
	SessionRequired bool // if true, cannot be used in ephemeral loops (needs registry, DB events, etc.)
}

// Def returns the llm.ToolDef for this tool, suitable for sending to the LLM.
func (t *Tool) Def() llm.ToolDef {
	r := &jsonschema.Reflector{DoNotReference: true}
	schema := r.Reflect(t.ParamType)
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		// Reflecting a struct into JSON Schema effectively never fails, but don't
		// ship a garbage/empty schema silently if it ever does.
		slog.Error("tool schema marshal failed", "tool", t.Name, "error", err)
	}
	return llm.ToolDef{
		Name:        t.Name,
		Description: t.Description,
		Schema:      schemaJSON,
	}
}

// ParseAndExecute parses the JSON args into the param struct and calls the
// executor. JSON parse errors are returned as error Results (not Go errors),
// since the LLM needs to see the error message to self-correct. A panic in a
// tool is recovered into an error Result so a buggy tool can never crash the
// process (executors run in goroutines, where an unrecovered panic is fatal).
func (t *Tool) ParseAndExecute(ctx context.Context, argsJSON string) (result *Result) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("tool panicked", "tool", t.Name, "panic", r, "stack", string(debug.Stack()))
			result = &Result{
				Content: fmt.Sprintf("Error: tool %s crashed: %v", t.Name, r),
				IsError: true,
			}
		}
	}()
	res, err := t.Execute(ctx, json.RawMessage(argsJSON))
	if err != nil {
		return &Result{
			Content: fmt.Sprintf("Error executing tool %s: %s", t.Name, err),
			IsError: true,
		}
	}
	// Defend the contract: an executor must return a non-nil Result on success.
	// A (nil, nil) return would otherwise propagate a nil *Result that the
	// caller dereferences in a goroutine where the panic is fatal (outside this
	// recover). Turn it into an error Result instead.
	if res == nil {
		return &Result{
			Content: fmt.Sprintf("Error: tool %s returned no result", t.Name),
			IsError: true,
		}
	}
	return res
}

// Defs converts a slice of Tools to llm.ToolDefs for the LLM request.
func Defs(tools []*Tool) []llm.ToolDef {
	defs := make([]llm.ToolDef, len(tools))
	for i, t := range tools {
		defs[i] = t.Def()
	}
	return defs
}

// ByName returns a map of tool name to Tool for fast lookup during execution.
func ByName(tools []*Tool) map[string]*Tool {
	m := make(map[string]*Tool, len(tools))
	for _, t := range tools {
		m[t.Name] = t
	}
	return m
}

// CallResult pairs a tool call ID with its execution result.
type CallResult struct {
	ToolCallID string
	Result     *Result
}

// maxParallelTools bounds how many tool calls in a single step run concurrently.
// A turn rarely emits more than a few parallel calls, so this is mostly a safety
// ceiling against a pathological burst (and to bound concurrent bash/file IO),
// not a throughput knob.
const maxParallelTools = 8

// ExecuteAll runs tool calls in parallel (capped at maxParallelTools), handling unknown tools
// and exclusive tool constraints. Returns results in the same order as calls.
// This is the shared execution logic used by both EventLoopAgent and EphemeralLoop.
//
// If onResult is non-nil it is invoked with each result the moment that tool
// finishes — i.e. in completion order, and concurrently for parallel tools — so
// a caller can persist results as they land instead of waiting for the slowest
// tool (e.g. a blocking await_event) to release the whole batch. A callback that
// touches shared state must synchronize itself: onResult may run on the worker
// goroutines AND on this goroutine (for unknown/exclusive-tool errors emitted
// inline), so the calls can interleave.
func ExecuteAll(ctx context.Context, calls []llm.ToolCall, toolMap map[string]*Tool, onResult func(CallResult)) []CallResult {
	results := make([]CallResult, len(calls))
	emit := func(i int) {
		if onResult != nil {
			onResult(results[i])
		}
	}

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(maxParallelTools)

	for i, tc := range calls {
		results[i] = CallResult{ToolCallID: tc.ID}

		t, ok := toolMap[tc.Name]
		if !ok {
			results[i].Result = &Result{
				Content: fmt.Sprintf("Error: unknown tool %q", tc.Name),
				IsError: true,
			}
			emit(i)
			continue
		}
		if t.Exclusive && len(calls) > 1 {
			results[i].Result = &Result{
				Content: fmt.Sprintf("Error: tool %q must be the only tool call in the step", tc.Name),
				IsError: true,
			}
			emit(i)
			continue
		}

		i, tc, t := i, tc, t
		g.Go(func() error {
			results[i].Result = t.ParseAndExecute(gCtx, tc.Arguments)
			emit(i)
			return nil
		})
	}

	_ = g.Wait() //nolint:errcheck // goroutines always return nil
	return results
}

// ParseArgs is a generic helper for tool executors to parse JSON args
// into a typed struct. Returns an error Result on parse failure.
func ParseArgs[T any](argsJSON json.RawMessage) (*T, *Result) {
	var params T
	if err := json.Unmarshal(argsJSON, &params); err != nil {
		return nil, &Result{
			Content: fmt.Sprintf("Invalid arguments: %s", err),
			IsError: true,
		}
	}
	return &params, nil
}

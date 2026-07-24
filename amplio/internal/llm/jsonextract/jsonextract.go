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

// Package jsonextract turns LLM free-text output into a typed Go value, with an
// optional bounded "ask the model to fix the formatting" repair pass.
//
// The happy path is free: it runs util.ExtractJSONObject (fence-tolerant) and
// json.Unmarshal with ZERO extra LLM calls. Only on a parse failure, and only
// when a repair provider is supplied, does it make a CONTEXTLESS repair call:
// "here is text that should be JSON; it didn't parse: <err>; reply with ONLY
// corrected JSON." It re-extracts and retries up to a bounded number of passes.
//
// Crucially, repair fixes SYNTAX, not CONTENT. The repair call has no access to
// the original task — it only sees the malformed text and the parse error. So a
// response that's well-formed but semantically wrong (missing fields the model
// never produced) is beyond saving, and that's the correct boundary: it's what
// lets this helper be call-site-agnostic (usable after any single llm.Call or an
// ephemeral loop result, not tied to any particular conversation).
package jsonextract

import (
	"context"
	"encoding/json"
	"fmt"

	"amplio/internal/llm"
	"amplio/internal/util"
)

// defaultRepairPasses bounds the repair attempts when a repair provider is set
// and the first extraction fails. A syntax fix that doesn't land on the first
// repair almost never lands later, so the default is deliberately small; each
// pass is one LLM round-trip.
const defaultRepairPasses = 1

// Options configures Extract. The zero value (Repair nil) is valid: it means
// extract-only — no LLM calls, equivalent to util.ExtractJSONObject + Unmarshal.
type Options struct {
	// Repair, when non-nil, is the provider used for the formatting-repair pass
	// after a parse/validation failure. nil disables repair (extract-only).
	Repair llm.Provider
	// Hint is an optional natural-language description of the expected shape,
	// included in the repair prompt. Ignored when Schema is set (the schema is a
	// stronger, machine-checkable description). Empty is fine.
	Hint string
	// Schema, when non-empty, is a JSON Schema string the extracted value is
	// VALIDATED against (beyond plain Unmarshal): it catches missing required
	// fields, wrong enums, and type mismatches Go would silently zero-fill. The
	// validation error (precise, per-field) is fed into the repair prompt, so the
	// model can fix content errors, not just syntax. Use SchemaFor[T]() to render
	// one from the target type.
	Schema string
	// MaxRepairPasses overrides the number of repair round-trips (<=0 uses the
	// default). Ignored when Repair is nil.
	MaxRepairPasses int
}

// Extract parses raw LLM output into *T, optionally validating against
// opts.Schema. It first tries fence-tolerant extraction (free, no LLM); on a
// parse OR validation failure, if opts.Repair is set, it makes up to
// MaxRepairPasses contextless repair calls. Returns the last error if every
// attempt fails.
func Extract[T any](ctx context.Context, raw string, opts Options) (*T, error) {
	sch, err := compileSchema(opts.Schema)
	if err != nil {
		// A malformed caller-supplied schema is a programming error, not an LLM
		// error; surface it rather than silently skipping validation.
		return nil, err
	}
	// Happy path: extract + unmarshal (+ validate), zero LLM calls.
	if v, perr := tryParse[T](raw, sch); perr == nil {
		return v, nil
	} else if opts.Repair == nil {
		return nil, perr // extract-only: surface the error without a repair attempt
	} else {
		return repair[T](ctx, raw, perr, sch, opts)
	}
}

// tryParse extracts, unmarshals into *T, and (if sch != nil) validates against
// the schema. Shared by the happy path and each repair pass so semantics match.
func tryParse[T any](raw string, sch *compiledSchema) (*T, error) {
	js := util.ExtractJSONObject(raw)
	if js == "" {
		return nil, fmt.Errorf("jsonextract: no JSON object found in output")
	}
	var v T
	if err := json.Unmarshal([]byte(js), &v); err != nil {
		return nil, fmt.Errorf("jsonextract: parse: %w", err)
	}
	if err := sch.validate([]byte(js)); err != nil {
		return nil, fmt.Errorf("jsonextract: schema validation failed: %w", err)
	}
	return &v, nil
}

// repair runs the bounded repair loop. lastErr is the parse/validation error
// that triggered it; the loop keeps the most recent error to return if all
// passes fail.
func repair[T any](ctx context.Context, raw string, lastErr error, sch *compiledSchema, opts Options) (*T, error) {
	passes := opts.MaxRepairPasses
	if passes <= 0 {
		passes = defaultRepairPasses
	}
	// Prefer the schema over the free-text hint in the repair prompt: it's a
	// stronger, machine-checkable description of the target shape.
	desc := opts.Hint
	if opts.Schema != "" {
		desc = "a JSON value matching this schema:\n" + opts.Schema
	}
	cur := raw
	for i := 0; i < passes; i++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		fixed, err := repairCall(ctx, opts.Repair, cur, lastErr, desc)
		if err != nil {
			// A failed repair call (network, etc.) ends the loop; return the
			// underlying parse/validation error, the more actionable one for the
			// caller's fallback path.
			return nil, lastErr
		}
		if v, perr := tryParse[T](fixed, sch); perr == nil {
			return v, nil
		} else {
			lastErr = perr
			cur = fixed // feed the still-broken output into the next pass
		}
	}
	return nil, lastErr
}

const repairSystemPrompt = "You are a JSON fixer. The user gives you text that was supposed to be a single " +
	"JSON value but failed to parse or validate, plus the error and (optionally) the expected shape. " +
	"Reply with ONLY the corrected JSON value — no prose, no markdown fences, no explanation. Preserve " +
	"the original content as much as possible; change only what the error requires."

// repairCall asks the provider to fix broken into valid JSON. It is contextless
// by design (only the broken text + the error + an optional shape description),
// so it can fix syntax and shape errors the description/schema pins down, but
// cannot recover content the original output never contained.
func repairCall(ctx context.Context, p llm.Provider, broken string, fixErr error, desc string) (string, error) {
	var user string
	if desc != "" {
		user = fmt.Sprintf("Expected: %s\n\n", desc)
	}
	user += fmt.Sprintf("Error: %s\n\nText to fix:\n%s", fixErr, broken)
	resp, err := p.Call(ctx, llm.Request{
		SystemPrompt: repairSystemPrompt,
		Messages:     []llm.Message{{Role: llm.RoleUser, Content: user}},
	})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

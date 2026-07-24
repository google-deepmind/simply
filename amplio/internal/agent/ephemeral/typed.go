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

package ephemeral

import (
	"context"
	"fmt"
	"strings"

	"amplio/internal/llm/jsonextract"
)

// defaultOutputRepairs bounds how many times RunTyped re-prompts the model when
// its terminal turn doesn't parse/validate as T. Small: a well-described schema
// almost always lands on the first or second try, and each retry is one more
// LLM round-trip inside the loop (separate from MaxIterations, which still caps
// the whole loop).
const defaultOutputRepairs = 2

// RunTyped runs an ephemeral tool loop whose FINAL output is a JSON value of
// type T. It renders T's JSON schema, appends it to the system prompt (so the
// caller's task prompt need not restate the shape), and validates the terminal
// no-tool turn against that schema. On a parse/validation failure it appends the
// precise error as a user turn and lets the SAME loop retry IN CONTEXT (the
// model still sees its own bad reply and the whole conversation), so it can fix
// content errors, not just syntax — up to outputRepairs attempts.
//
// outputRepairs <= 0 uses the default. Returns the parsed *T, or an error if the
// loop never produced a schema-valid final turn.
func RunTyped[T any](ctx context.Context, cfg Config, task string, outputRepairs int) (*T, error) {
	if outputRepairs <= 0 {
		outputRepairs = defaultOutputRepairs
	}
	schema := jsonextract.SchemaFor[T]()

	// Append the output contract to the system prompt rather than the task, so a
	// caller can reuse one task message across shapes.
	cfg.SystemPrompt = cfg.SystemPrompt + "\n\n" + outputContract(schema)

	var result *T
	var lastErr error
	attempts := 0

	final, err := run(ctx, cfg, task, func(content string) (string, bool) {
		// Validate the terminal turn against the schema (no repair provider here:
		// the ephemeral loop IS the in-context repair mechanism).
		v, perr := jsonextract.Extract[T](ctx, content, jsonextract.Options{Schema: schema})
		if perr == nil {
			result = v
			return "", true // accept; end the loop
		}
		lastErr = perr
		attempts++
		if attempts >= outputRepairs {
			return "", true // give up: end the loop, caller sees lastErr below
		}
		// Reject: nudge with the precise error AND re-state the schema. The model
		// retries with full context, but on a long loop the original schema (in the
		// system prompt) is far back, so repeating it here — right next to the error
		// it must fix — anchors the correction. Cheap relative to the round-trip.
		return fmt.Sprintf("Your final message was not valid output. %s\n\n"+
			"Reply again with ONLY the corrected JSON value (no tool calls, no prose, no fences) matching this schema:\n%s",
			strings.TrimPrefix(perr.Error(), "jsonextract: "), schema), false
	})
	if err != nil {
		return nil, err
	}
	if result == nil {
		if lastErr != nil {
			return nil, fmt.Errorf("ephemeral RunTyped: no schema-valid output after %d attempt(s): %w", attempts, lastErr)
		}
		// onFinal never fired (e.g. loop hit MaxIterations without a no-tool turn).
		return nil, fmt.Errorf("ephemeral RunTyped: loop ended without a final output turn (last content: %.80q)", final)
	}
	return result, nil
}

func outputContract(schema string) string {
	return "## Final output\n\n" +
		"When your work is complete, end your turn with NO tool calls and reply with ONLY a single " +
		"JSON value matching this schema (no prose, no markdown fences):\n" + schema
}

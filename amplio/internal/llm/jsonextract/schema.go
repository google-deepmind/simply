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

package jsonextract

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/invopop/jsonschema"
	validator "github.com/santhosh-tekuri/jsonschema/v6"
)

// SchemaFor renders a JSON Schema (draft 2020-12) string for T, for embedding in
// a prompt. DoNotReference inlines nested types so the model sees one
// self-contained schema (no $defs/$ref indirection). Returns "" on the
// (effectively impossible) reflect/marshal failure.
func SchemaFor[T any]() string {
	r := &jsonschema.Reflector{DoNotReference: true}
	var zero T
	b, err := json.MarshalIndent(r.Reflect(&zero), "", "  ")
	if err != nil {
		return ""
	}
	return string(b)
}

// compiledSchema wraps a santhosh-tekuri validator schema. Compiling is mildly
// expensive, so callers that validate repeatedly should compile once.
type compiledSchema struct{ sch *validator.Schema }

// compileSchema compiles a schema string for validation. Returns nil (no error
// surfaced) for an empty schema, so "no schema" cleanly means "skip validation".
func compileSchema(schemaJSON string) (*compiledSchema, error) {
	if strings.TrimSpace(schemaJSON) == "" {
		return nil, nil
	}
	doc, err := validator.UnmarshalJSON(strings.NewReader(schemaJSON))
	if err != nil {
		return nil, fmt.Errorf("jsonextract: parse schema: %w", err)
	}
	c := validator.NewCompiler()
	const url = "mem://schema.json"
	if err := c.AddResource(url, doc); err != nil {
		return nil, fmt.Errorf("jsonextract: add schema: %w", err)
	}
	sch, err := c.Compile(url)
	if err != nil {
		return nil, fmt.Errorf("jsonextract: compile schema: %w", err)
	}
	return &compiledSchema{sch: sch}, nil
}

// validate checks raw JSON bytes against the compiled schema, returning the
// validator's detailed multi-line error (e.g. "missing property 'summary'",
// "at '/grade': maximum: got 9, want 5") suitable for feeding back to an LLM.
// A nil receiver validates nothing (skip).
func (c *compiledSchema) validate(js []byte) error {
	if c == nil || c.sch == nil {
		return nil
	}
	inst, err := validator.UnmarshalJSON(strings.NewReader(string(js)))
	if err != nil {
		return fmt.Errorf("jsonextract: parse instance for validation: %w", err)
	}
	return c.sch.Validate(inst)
}

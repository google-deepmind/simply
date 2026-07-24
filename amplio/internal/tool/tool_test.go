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
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

type echoParams struct {
	Message string `json:"message" jsonschema:"required,description=Message to echo"`
	Loud    bool   `json:"loud,omitempty" jsonschema:"description=Whether to uppercase"`
}

func echoExecutor(_ context.Context, args json.RawMessage) (*Result, error) {
	params, errResult := ParseArgs[echoParams](args)
	if errResult != nil {
		return errResult, nil
	}
	msg := params.Message
	if params.Loud {
		msg = strings.ToUpper(msg)
	}
	return &Result{Content: msg}, nil
}

func newEchoTool() *Tool {
	return &Tool{
		Name:        "echo",
		Description: "Echo a message back",
		ParamType:   &echoParams{},
		Execute:     echoExecutor,
	}
}

func TestTool_Def(t *testing.T) {
	tool := newEchoTool()
	def := tool.Def()

	if def.Name != "echo" {
		t.Errorf("Name: %q", def.Name)
	}
	if def.Description != "Echo a message back" {
		t.Errorf("Description: %q", def.Description)
	}

	// Schema should contain the parameter fields.
	var schema map[string]any
	if err := json.Unmarshal(def.Schema, &schema); err != nil {
		t.Fatalf("schema unmarshal: %v", err)
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("missing properties in schema: %v", schema)
	}
	if _, ok := props["message"]; !ok {
		t.Error("missing 'message' property")
	}
	if _, ok := props["loud"]; !ok {
		t.Error("missing 'loud' property")
	}
}

func TestTool_ParseAndExecute(t *testing.T) {
	tool := newEchoTool()

	// Valid args.
	result := tool.ParseAndExecute(context.Background(), `{"message":"hello","loud":true}`)
	if result.Content != "HELLO" {
		t.Errorf("content: %q", result.Content)
	}
	if result.IsError {
		t.Error("unexpected error flag")
	}
}

func TestTool_ParseAndExecute_DefaultOptional(t *testing.T) {
	tool := newEchoTool()
	result := tool.ParseAndExecute(context.Background(), `{"message":"hi"}`)
	if result.Content != "hi" {
		t.Errorf("content: %q", result.Content)
	}
}

func TestTool_ParseAndExecute_InvalidJSON(t *testing.T) {
	tool := newEchoTool()
	result := tool.ParseAndExecute(context.Background(), `{not valid json}`)
	if !result.IsError {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseArgs_MissingRequired(t *testing.T) {
	// "message" is required but not provided.
	_, errResult := ParseArgs[echoParams](json.RawMessage(`{"loud":true}`))
	// Go's json.Unmarshal doesn't enforce "required" — it just uses zero value.
	// So this doesn't error. Required enforcement is at the schema level (LLM-side).
	if errResult != nil {
		t.Error("ParseArgs shouldn't fail for missing fields (Go zero-value semantics)")
	}
}

func TestDefs(t *testing.T) {
	tools := []*Tool{newEchoTool()}
	defs := Defs(tools)
	if len(defs) != 1 || defs[0].Name != "echo" {
		t.Errorf("Defs: %v", defs)
	}
}

func TestByName(t *testing.T) {
	tools := []*Tool{newEchoTool()}
	m := ByName(tools)
	if _, ok := m["echo"]; !ok {
		t.Error("missing 'echo' in ByName map")
	}
}

func TestTool_ParseAndExecute_RecoversPanic(t *testing.T) {
	panicky := &Tool{
		Name:      "boom",
		ParamType: &echoParams{},
		Execute: func(context.Context, json.RawMessage) (*Result, error) {
			panic("kaboom")
		},
	}
	// A panic must become an error Result, not crash the process.
	result := panicky.ParseAndExecute(context.Background(), `{"message":"x"}`)
	if !result.IsError {
		t.Fatal("expected error result from panicking tool")
	}
	if !strings.Contains(result.Content, "boom") || !strings.Contains(result.Content, "kaboom") {
		t.Errorf("content: %q", result.Content)
	}
}

func TestTool_ParseAndExecute_NilResult(t *testing.T) {
	// An executor that returns (nil, nil) must be turned into an error Result,
	// not a nil *Result the caller would deref (fatal in a goroutine). See M15.
	bad := &Tool{
		Name:      "nilreturn",
		ParamType: &echoParams{},
		Execute: func(context.Context, json.RawMessage) (*Result, error) {
			return nil, nil
		},
	}
	result := bad.ParseAndExecute(context.Background(), `{"message":"x"}`)
	if result == nil {
		t.Fatal("ParseAndExecute returned nil *Result")
	}
	if !result.IsError || !strings.Contains(result.Content, "no result") {
		t.Errorf("expected 'no result' error, got: %q", result.Content)
	}
}

// TestNoInlineSchemaDescription enforces the repo convention: descriptions on
// jsonschema-tagged struct fields MUST use the dedicated `jsonschema_description`
// tag, never an inline `description=` directive inside `jsonschema:"..."`.
//
// Rationale: the invopop tag parser splits the jsonschema tag on (unescaped)
// commas, so a comma in an inline description silently truncates it in the
// generated schema the LLM sees. The dedicated tag is read whole and is immune.
// This guard makes the safe form the only mergeable form (production code).
//
// It scans non-test .go files under the module root; test fixtures (this file
// included) may still use the inline form to exercise the parser.
func TestNoInlineSchemaDescription(t *testing.T) {
	root := moduleRoot(t)
	// Match a jsonschema:"..." tag whose contents contain a `description=`
	// directive (the comma-fragile form we forbid).
	re := regexp.MustCompile(`jsonschema:"[^"]*description=`)
	var offenders []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip the module cache / vendored deps if present under root.
			if d.Name() == "vendor" || d.Name() == ".git" || d.Name() == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			return rerr
		}
		for i, line := range strings.Split(string(data), "\n") {
			if re.MatchString(line) {
				rel, _ := filepath.Rel(root, path)
				offenders = append(offenders, fmt.Sprintf("%s:%d", rel, i+1))
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(offenders) > 0 {
		t.Errorf("inline jsonschema description= found (use a separate jsonschema_description:\"...\" tag instead):\n  %s",
			strings.Join(offenders, "\n  "))
	}
}

// moduleRoot walks up from the cwd to the directory containing go.mod.
func moduleRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find go.mod above cwd")
		}
		dir = parent
	}
}

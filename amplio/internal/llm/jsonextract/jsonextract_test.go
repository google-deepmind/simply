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
	"context"
	"strings"
	"testing"

	"amplio/internal/llm"
)

type doc struct {
	Title   string `json:"title"`
	Summary string `json:"summary"`
}

func TestExtract_HappyPath_NoLLMCall(t *testing.T) {
	mock := &llm.MockProvider{Model: "test"} // any call would be a bug
	got, err := Extract[doc](context.Background(),
		`{"title":"T","summary":"S"}`,
		Options{Repair: mock})
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "T" || got.Summary != "S" {
		t.Errorf("got %+v", got)
	}
	if mock.CallCount() != 0 {
		t.Errorf("happy path must not call the LLM, got %d calls", mock.CallCount())
	}
}

func TestExtract_FenceTolerant_NoLLMCall(t *testing.T) {
	mock := &llm.MockProvider{Model: "test"}
	raw := "Here you go:\n```json\n{\"title\":\"T\",\"summary\":\"S\"}\n```\nDone."
	got, err := Extract[doc](context.Background(), raw, Options{Repair: mock})
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "T" {
		t.Errorf("got %+v", got)
	}
	if mock.CallCount() != 0 {
		t.Errorf("fenced JSON should parse without repair, got %d calls", mock.CallCount())
	}
}

func TestExtract_NilProvider_ExtractOnly(t *testing.T) {
	// Unparseable + no repair provider => error, no LLM.
	_, err := Extract[doc](context.Background(), "not json at all", Options{})
	if err == nil {
		t.Fatal("expected parse error with no repair provider")
	}
}

func TestExtract_RepairSucceeds(t *testing.T) {
	// First the caller passes broken text; the repair model returns valid JSON.
	mock := &llm.MockProvider{
		Model:     "test",
		Responses: []llm.Response{{Content: `{"title":"T","summary":"S"}`}},
	}
	got, err := Extract[doc](context.Background(),
		`{"title":"T", "summary":}`, // trailing-value syntax error
		Options{Repair: mock, Hint: "object with title, summary"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "T" || got.Summary != "S" {
		t.Errorf("got %+v", got)
	}
	if mock.CallCount() != 1 {
		t.Errorf("expected exactly 1 repair call, got %d", mock.CallCount())
	}
	// The repair prompt should carry the hint and the parse error.
	rc := mock.Recorded()[0]
	if len(rc.Messages) == 0 || !strings.Contains(rc.Messages[0].Content, "title, summary") {
		t.Errorf("repair prompt missing hint: %+v", rc.Messages)
	}
}

func TestExtract_RepairExhausted(t *testing.T) {
	// Repair model keeps returning junk; after MaxRepairPasses we give up with
	// the parse error.
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			{Content: "still not json"},
			{Content: "nope"},
		},
	}
	_, err := Extract[doc](context.Background(), "broken",
		Options{Repair: mock, MaxRepairPasses: 2})
	if err == nil {
		t.Fatal("expected error after exhausting repair passes")
	}
	if mock.CallCount() != 2 {
		t.Errorf("expected 2 repair passes, got %d", mock.CallCount())
	}
}

func TestExtract_RepairPassesDefaultToOne(t *testing.T) {
	mock := &llm.MockProvider{
		Model: "test",
		Responses: []llm.Response{
			{Content: "still broken"},
			{Content: `{"title":"T","summary":"S"}`}, // would succeed on pass 2
		},
	}
	_, err := Extract[doc](context.Background(), "broken", Options{Repair: mock})
	if err == nil {
		t.Fatal("expected failure: default is a single repair pass, which fails here")
	}
	if mock.CallCount() != 1 {
		t.Errorf("default should be exactly 1 repair pass, got %d", mock.CallCount())
	}
}

func TestExtract_CancelledContextDuringRepair(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	mock := &llm.MockProvider{Model: "test", Responses: []llm.Response{{Content: "junk"}}}
	_, err := Extract[doc](ctx, "broken", Options{Repair: mock})
	if err == nil {
		t.Fatal("expected error on cancelled context")
	}
	// The ctx check precedes the repair call, so no LLM call is made.
	if mock.CallCount() != 0 {
		t.Errorf("cancelled ctx should skip the repair call, got %d", mock.CallCount())
	}
}

// --- schema validation ---

type graded struct {
	Summary string `json:"summary"`
	Grade   int    `json:"grade"`
}

func TestExtract_SchemaCatchesMissingRequired(t *testing.T) {
	// Parseable JSON that is missing a required field: plain Unmarshal would
	// accept it (zero-fill), schema validation rejects it.
	schema := `{"type":"object","required":["summary","grade"],"properties":{"summary":{"type":"string"},"grade":{"type":"integer"}}}`
	_, err := Extract[graded](context.Background(), `{"summary":"ok"}`,
		Options{Schema: schema}) // no repair -> surfaces the validation error
	if err == nil {
		t.Fatal("expected schema validation error for missing required 'grade'")
	}
	if !strings.Contains(err.Error(), "grade") {
		t.Errorf("error should mention the missing field: %v", err)
	}
}

func TestExtract_SchemaRepairFixesContent(t *testing.T) {
	// First output omits grade (schema-invalid but parseable); the repair model
	// returns a complete object. This is the case plain-parse repair couldn't
	// catch at all.
	schema := SchemaFor[graded]()
	mock := &llm.MockProvider{
		Model:     "test",
		Responses: []llm.Response{{Content: `{"summary":"ok","grade":4}`}},
	}
	got, err := Extract[graded](context.Background(), `{"summary":"ok"}`,
		Options{Repair: mock, Schema: schema})
	if err != nil {
		t.Fatal(err)
	}
	if got.Grade != 4 {
		t.Errorf("got %+v", got)
	}
	if mock.CallCount() != 1 {
		t.Errorf("expected 1 repair call, got %d", mock.CallCount())
	}
	// The repair prompt should carry the schema (not just a vague hint).
	if !strings.Contains(mock.Recorded()[0].Messages[0].Content, "\"properties\"") {
		t.Errorf("repair prompt should include the schema: %s", mock.Recorded()[0].Messages[0].Content)
	}
}

func TestExtract_SchemaHappyPath_NoLLM(t *testing.T) {
	schema := SchemaFor[graded]()
	mock := &llm.MockProvider{Model: "test"}
	got, err := Extract[graded](context.Background(), `{"summary":"ok","grade":3}`,
		Options{Repair: mock, Schema: schema})
	if err != nil {
		t.Fatal(err)
	}
	if got.Grade != 3 || mock.CallCount() != 0 {
		t.Errorf("valid input must not call repair: grade=%d calls=%d", got.Grade, mock.CallCount())
	}
}

func TestExtract_MalformedSchemaIsCallerError(t *testing.T) {
	_, err := Extract[graded](context.Background(), `{"grade":1,"summary":"x"}`,
		Options{Schema: "{not valid schema"})
	if err == nil {
		t.Fatal("expected error for a malformed caller-supplied schema")
	}
}

func TestSchemaFor_RendersFields(t *testing.T) {
	s := SchemaFor[graded]()
	if !strings.Contains(s, "summary") || !strings.Contains(s, "grade") {
		t.Errorf("schema missing fields: %s", s)
	}
}

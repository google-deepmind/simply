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

package workspace

import (
	"context"
	"encoding/json"
	"testing"
)

// fakeWS is a minimal registered backend for exercising Marshal/Unmarshal.
type fakeWS struct {
	P string `json:"p"`
}

func (f *fakeWS) Root() string                                            { return f.P }
func (f *fakeWS) Name() string                                            { return f.P }
func (f *fakeWS) Kind() string                                            { return "faketest" }
func (f *fakeWS) CreateLinked(context.Context, string) (Workspace, error) { return nil, nil }
func (f *fakeWS) LinkedFrom() string                                      { return "" }
func (f *fakeWS) Describe(context.Context, string) string                 { return "" }
func (f *fakeWS) Validate(context.Context) error                          { return nil }
func (f *fakeWS) ResolveAlias(context.Context) (string, error)            { return "", nil }

func init() {
	RegisterKind("faketest", func(data []byte) (Workspace, error) {
		var w fakeWS
		if err := json.Unmarshal(data, &w); err != nil {
			return nil, err
		}
		return &w, nil
	})
}

func TestMarshalRoundTrip(t *testing.T) {
	blob, err := Marshal(&fakeWS{P: "/some/path"})
	if err != nil {
		t.Fatal(err)
	}
	w, err := Unmarshal(blob)
	if err != nil {
		t.Fatal(err)
	}
	if w.Kind() != "faketest" || w.Root() != "/some/path" {
		t.Errorf("round-trip mismatch: kind=%q root=%q", w.Kind(), w.Root())
	}
}

func TestUnmarshalErrors(t *testing.T) {
	cases := map[string]string{
		"unknown kind": `{"kind":"nope"}`,
		"missing kind": `{"root":"/x"}`,
		"not json":     `}{`,
	}
	for name, in := range cases {
		if _, err := Unmarshal([]byte(in)); err == nil {
			t.Errorf("%s: expected error for %q", name, in)
		}
	}
}

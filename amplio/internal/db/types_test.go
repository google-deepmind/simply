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

package db

import (
	"testing"

	"amplio/internal/event"
)

func TestClassify(t *testing.T) {
	tests := []struct {
		name string
		evt  event.Event
		want EventClass
	}{
		{"user input", &event.UserEvent{Content: "hi"}, ClassInput},
		{"agent message", &event.MessageEvent{Content: "x", SenderType: event.SenderTypeAgent}, ClassInput},
		// The behavior change: an environment notification ($AMPLIO_NOTIFY) now
		// revives a dormant recipient (was ClassNotice).
		{"environment notify", &event.MessageEvent{Content: "done", SenderType: event.SenderTypeEnvironment}, ClassInput},
		{"child concluded", &event.ChildResultEvent{Verdict: SessionConcluded}, ClassInput},
		{"child crashed", &event.ChildResultEvent{Verdict: SessionCrashed}, ClassNotice},
		{"recover", &event.RecoverEvent{}, ClassInput},
		{"self/system write", &event.AssistantEvent{Content: "thinking"}, ClassNotice},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Classify(tt.evt); got != tt.want {
				t.Errorf("Classify(%s) = %v, want %v", tt.name, got, tt.want)
			}
			if got := IsInput(tt.evt); got != (tt.want == ClassInput) {
				t.Errorf("IsInput(%s) = %v, want %v", tt.name, got, tt.want == ClassInput)
			}
		})
	}
}

func TestIsSpine(t *testing.T) {
	tests := []struct {
		name string
		s    SessionRecord
		want bool
	}{
		{"ongoing", SessionRecord{Status: SessionOngoing}, true},
		{"awaiting", SessionRecord{Status: SessionAwaiting}, true},
		{"crashed root", SessionRecord{Status: SessionCrashed}, true},
		{"crashed child", SessionRecord{Status: SessionCrashed, ParentID: "p"}, false},
		{"idle", SessionRecord{Status: SessionIdle}, false},
		{"concluded", SessionRecord{Status: SessionConcluded}, false},
		{"cancelled", SessionRecord{Status: SessionCancelled}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsSpine(tt.s); got != tt.want {
				t.Errorf("IsSpine(%+v) = %v, want %v", tt.s, got, tt.want)
			}
		})
	}
}

func TestIsSummarizationFailure(t *testing.T) {
	tests := []struct {
		name    string
		summary string
		want    bool
	}{
		{"real summary", "explored the codebase", false},
		{"empty", "", false},
		{"failure sentinel", SummarizationFailedPrefix + " LLM call failed", true},
		{"failure sentinel leading space", "  " + SummarizationFailedPrefix + " boom", true},
		{"mentions phrase mid-string", "the [summarization failed] earlier", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsSummarizationFailure(tt.summary); got != tt.want {
				t.Errorf("IsSummarizationFailure(%q) = %v, want %v", tt.summary, got, tt.want)
			}
		})
	}
}

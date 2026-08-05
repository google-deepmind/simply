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

package event

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"amplio/internal/util"
)

// Event is the sealed interface for all event types.
//
// ToText renders the event as a labeled, plain-text section for consumers
// (e.g. summarizer, session inspection tools). The format is a
// banner header — "==== TYPE · k=v · … ====" — followed by the content, with
// no closing marker (the next header or turn boundary delimits it). It
// deliberately avoids XML/JSON/tool-call syntax so the model never mistakes
// presented content for an action it should emit.
//
// ToText is *not* what the LLMs consume as their own conversation history.
// That is separately handled via typed projection to the message formats
// expected by each LLM API Provider.
type Event interface {
	eventTag() // unexported — only types in this package can implement
	ToText() string
	EventType() string
}

// ColumnFields holds DB column values injected after deserialization.
// These are never included in JSON serialization (they come from DB columns).
type ColumnFields struct {
	Step       int       `json:"-"`
	Generation int       `json:"-"`
	CreatedAt  time.Time `json:"-"`
}

// --- Nested types ---

type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// Attachment is a binary artifact produced by a tool (today only image/* from
// view_file). Raw bytes live in Data in-flight (tool → event loop) and are
// written to the run's on-disk blob store before persistence; only the content
// key, mime type and size are persisted in the event log (Data is never
// serialized), keeping SQLite event/FTS rows small. The UI fetches the bytes via
// the blob HTTP route using BlobKey.
type Attachment struct {
	MimeType   string `json:"mime_type"`
	BlobKey    string `json:"blob_key,omitempty"`
	Size       int    `json:"size,omitempty"`
	SourceHint string `json:"source_hint,omitempty"`
	Data       []byte `json:"-"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// Cache token counts (omitempty keeps zero-cache/old rows unchanged).
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

// banner renders content as a labeled section: a "==== LABEL · k=v · … ===="
// header line followed by the content. Empty attrs are dropped. There is no
// closing marker by design (see the Event interface doc).
func banner(label string, attrs []string, content string) string {
	head := label
	for _, a := range attrs {
		if a != "" {
			head += " · " + a
		}
	}
	return fmt.Sprintf("==== %s ====\n%s", head, content)
}

// tsAttr returns a "name=<local-iso>" attribute, or "" if t is zero.
func tsAttr(name string, t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return name + "=" + util.FormatLocalISO(t)
}

// --- Concrete event types ---

// SystemEvent markers: a stable tag identifying which system event this is, so
// readers (tests, UI, inspection) can match on a constant rather than a string
// literal that can silently drift from the producer. Persisted in the event
// JSON, so DO NOT change an existing value (it would orphan stored rows); add a
// new one instead.
const (
	MarkerSystemPrompt  = "system_prompt"  // the agent's system prompt (step-0 bootstrap)
	MarkerCLITools      = "cli_tools"      // available external CLI tools (step-0 bootstrap)
	MarkerInitialRecall = "initial_recall" // task-relevant skills seeded at bootstrap
	MarkerWorkspace     = "workspace"      // the agent's working-tree description (step-0 bootstrap)
	MarkerAgentsMD      = "agents_md"      // operator AGENTS.md instructions (step-0 bootstrap)
	MarkerNewSession    = "new_session"    // the trailing "new session started" bootstrap marker
	MarkerCancelled     = "cancelled"      // session was cancelled (self-marker on terminate)
	MarkerError         = "error"          // a crash/error self-marker
)

type SystemEvent struct {
	ColumnFields
	Content string `json:"content"`
	Marker  string `json:"marker,omitempty"` // a Marker* constant; see above
}

func (*SystemEvent) eventTag()         {}
func (*SystemEvent) EventType() string { return "system" }

func (e *SystemEvent) ToText() string {
	var attrs []string
	if e.Marker != "" {
		attrs = append(attrs, e.Marker)
	}
	return banner("system", attrs, e.Content)
}

type UserEvent struct {
	ColumnFields
	Content string `json:"content"`
}

func (*UserEvent) eventTag()         {}
func (*UserEvent) EventType() string { return "user" }
func (e *UserEvent) ToText() string  { return banner("user", nil, e.Content) }

type AssistantEvent struct {
	ColumnFields
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Thoughts  string     `json:"thoughts,omitempty"`
	// StopReason is the provider's native finish/stop reason for this turn
	// (Anthropic: end_turn/max_tokens/tool_use/…; Gemini: STOP/MAX_TOKENS/…;
	// subprocess: bridge-defined). NOT normalized across providers — persisted
	// verbatim for inspection/debugging (e.g. diagnosing an accidental empty
	// no-tool conclusion). Omitted for events written before this field existed.
	StopReason    string         `json:"stop_reason,omitempty"`
	ProviderExtra map[string]any `json:"provider_extra,omitempty"`
	Usage         *Usage         `json:"usage,omitempty"`
}

func (*AssistantEvent) eventTag()         {}
func (*AssistantEvent) EventType() string { return "assistant" }

func (e *AssistantEvent) ToText() string {
	var body strings.Builder
	if e.Content != "" {
		body.WriteString(e.Content)
	}
	for _, tc := range e.ToolCalls {
		if body.Len() > 0 {
			body.WriteByte('\n')
		}
		// A lighter rule so a nested call reads as part of the assistant block,
		// not a new top-level section.
		fmt.Fprintf(&body, "---- tool_call · name=%s · id=%s ----\n%s", tc.Name, tc.ID, tc.Arguments)
	}
	return banner("assistant", nil, body.String())
}

// IsNoToolAssistant reports whether e is an AssistantEvent with no tool calls —
// the "natural rest" tail predicate used by crash recovery (a stream ending in
// such an event means the agent produced its final reply and is done).
func IsNoToolAssistant(e Event) bool {
	a, ok := e.(*AssistantEvent)
	return ok && len(a.ToolCalls) == 0
}

type ToolResultEvent struct {
	ColumnFields
	Content     string       `json:"content"`
	ToolCallID  string       `json:"tool_call_id"`
	Attachments []Attachment `json:"attachments,omitempty"`
	// IsError is the tool's own judgment that the call failed to do its job (a
	// parse error, target-not-found, unavailable backend, panic, …). It mirrors
	// tool.Result.IsError. omitempty keeps successful results byte-identical to
	// pre-field rows, so this is a backward-compatible additive change (absent key
	// → false → "not an error", the correct default for historical events).
	IsError bool `json:"is_error,omitempty"`
}

func (*ToolResultEvent) eventTag()         {}
func (*ToolResultEvent) EventType() string { return "tool_result" }

func (e *ToolResultEvent) ToText() string {
	body := e.Content
	if len(e.Attachments) > 0 {
		body += fmt.Sprintf("\n[%d attachment(s) omitted]", len(e.Attachments))
	}
	return banner("tool_result", []string{"id=" + e.ToolCallID}, body)
}

type CompactionEvent struct {
	ColumnFields
	Content string `json:"content"`
}

func (*CompactionEvent) eventTag()         {}
func (*CompactionEvent) EventType() string { return "compaction" }
func (e *CompactionEvent) ToText() string {
	return banner("compaction", []string{tsAttr("at", e.CreatedAt)}, e.Content)
}

// MessageEvent sender types. SenderType classifies the sender for rendering/
// attribution. Both agent messages (send_message) and environment notifications
// ($AMPLIO_NOTIFY) are Input-class in db.Classify. SenderType additionally gates
// ONE thing at the wake path (runtime.NewCommitNotifier): an environment
// notification does not revive a deliberately finished session. See
// docs/session_lifecycle.md.
const (
	SenderTypeAgent       = "agent"
	SenderTypeEnvironment = "environment"
)

// EnvironmentSenderID is the default Sender (identity, not type) for a
// MessageEvent from an external background script (e.g. the notify CLI) when no
// --from label is given. Distinct axis from SenderTypeEnvironment above
// (category): the two share the literal "environment" only by coincidence — a
// named notifier sets its own Sender while SenderType stays environment.
const EnvironmentSenderID = "environment"

// MessageEvent is an agent-to-agent message (via send_message) or an environment
// notification ($AMPLIO_NOTIFY). It is NOT used for operator/user input — that is
// a UserEvent (a plain user turn), so a chatbot responds to it naturally instead
// of trying to "reply" to the user via send_message.
//
// SenderType classifies the sender (agent vs environment); Sender identifies the
// specific sender: the sending session's ID for an agent message, or the script's
// name for an environment notification (so multiple notifiers are distinguishable).
type MessageEvent struct {
	ColumnFields
	Content    string `json:"content"`
	Sender     string `json:"sender,omitempty"`
	SenderType string `json:"sender_type,omitempty"`
}

func (*MessageEvent) eventTag()         {}
func (*MessageEvent) EventType() string { return "message" }

func (e *MessageEvent) ToText() string {
	var attrs []string
	if e.Sender != "" {
		attrs = append(attrs, "from="+e.Sender)
	}
	// Render the sender type only for non-agent senders; agent is the common
	// case and is left implicit.
	if e.SenderType != "" && e.SenderType != SenderTypeAgent {
		attrs = append(attrs, "type="+e.SenderType)
	}
	attrs = append(attrs, tsAttr("at", e.CreatedAt))
	return banner("message", attrs, e.Content)
}

type ChildResultEvent struct {
	ColumnFields
	Content        string `json:"content"`
	ChildSessionID string `json:"child_session_id"`
	Verdict        string `json:"verdict"` // concluded, crashed, cancelled
}

func (*ChildResultEvent) eventTag()         {}
func (*ChildResultEvent) EventType() string { return "child_result" }

func (e *ChildResultEvent) ToText() string {
	attrs := []string{"child=" + e.ChildSessionID, "verdict=" + e.Verdict, tsAttr("at", e.CreatedAt)}
	return banner("child_result", attrs, e.Content)
}

// RecoverEvent is the "resumed" marker appended by run-level Recover to a
// crash-recovered session's own stream (see docs/session_lifecycle.md). Recover
// is the only producer (never normal agent flow) and sets Content to the marker
// text the LLM sees. It is a user-role event (providers require the last
// pre-assistant message to be a user/tool-result message) and classifies as Input.
type RecoverEvent struct {
	ColumnFields
	Content string `json:"content,omitempty"`
}

func (*RecoverEvent) eventTag()         {}
func (*RecoverEvent) EventType() string { return "recover" }

func (e *RecoverEvent) ToText() string { return banner("recover", nil, e.Content) }

// --- Serialization ---

// Marshal converts an Event to a JSON blob with a "type" discriminator.
// Column fields (Step, Generation, CreatedAt) are NOT included (json:"-").
func Marshal(e Event) ([]byte, error) {
	data, err := json.Marshal(e)
	if err != nil {
		return nil, err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	typeBytes, _ := json.Marshal(e.EventType())
	m["type"] = typeBytes
	return json.Marshal(m)
}

// Unmarshal parses a JSON blob into a typed Event. The caller must set
// ColumnFields (Step, Generation, CreatedAt) separately from DB columns.
func Unmarshal(data []byte) (Event, error) {
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("event unmarshal probe: %w", err)
	}

	switch probe.Type {
	case "system":
		var e SystemEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "user":
		var e UserEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "assistant":
		var e AssistantEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "tool_result":
		var e ToolResultEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "compaction":
		var e CompactionEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "message":
		var e MessageEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "child_result":
		var e ChildResultEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "recover":
		var e RecoverEvent
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return &e, nil
	case "":
		return nil, fmt.Errorf("event unmarshal: missing 'type' field")
	default:
		return nil, fmt.Errorf("event unmarshal: unknown type %q", probe.Type)
	}
}

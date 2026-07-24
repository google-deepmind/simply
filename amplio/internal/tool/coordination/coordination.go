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

// Package coordination provides tools for multi-agent coordination:
// await_event (park until woken), send_message, and session_cancel.
package coordination

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/session"
	"amplio/internal/tool"
	"amplio/internal/util"
)

// Deps holds the dependencies needed by coordination tools.
type Deps struct {
	Store    db.Store
	RunID    string
	Registry *session.Registry
}

// --- await_event ---

// HandleFunc returns the session handle lazily (the handle is created
// during Run(), not at tool construction time).
type HandleFunc func() *session.Handle

// defaultAwaitTimeoutSeconds bounds a wait so a parked agent always re-evaluates
// rather than blocking forever (and so resume/lost-process detection makes
// progress). 5-minute default.
const defaultAwaitTimeoutSeconds = 300.0

type awaitParams struct {
	TimeoutSeconds float64 `json:"timeout,omitempty" jsonschema_description:"Max seconds to wait before returning (default 300). Returns early as soon as a new event arrives."`
}

func AwaitEvent(deps *Deps, sessionID string, getHandle HandleFunc) *tool.Tool {
	return &tool.Tool{
		Name: "await_event",
		Description: "Park this agent and wait for a new event in your session (e.g. a message from another agent or " +
			"the operator, or a result from a spawned child). Returns early as soon as something arrives, otherwise " +
			"after the timeout. Call it in the SAME step as the action that will produce the event — e.g. alongside " +
			"spawn_agent, send_message, or a background bash job — to act and wait in one turn.",
		ParamType: &awaitParams{},
		// Not exclusive: await_event may share a step with the action that will
		// produce the event (spawn_agent / send_message / a background bash job).
		// Its wake condition is a genuine future event — one at step T+1 or later
		// in the event store — NOT the notification counter. So the per-tool
		// results of its own step (written back at the call step T, and which now
		// commit mid-step as each tool finishes) never wake it, however the
		// concurrent appends interleave. The counter/WaitAfter is only a blocking
		// primitive: a wake not backed by a step-T+1 event just re-checks the
		// store and re-sleeps (mirrors waitForFollowUp's idle park).
		SessionRequired: true,
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[awaitParams](args)
			if errResult != nil {
				return errResult, nil
			}
			handle := getHandle()
			if handle == nil {
				return &tool.Result{Content: "Error: session not initialized.", IsError: true}, nil
			}

			timeoutSeconds := params.TimeoutSeconds
			if timeoutSeconds <= 0 {
				timeoutSeconds = defaultAwaitTimeoutSeconds
			}
			deadline := time.Now().Add(time.Duration(timeoutSeconds * float64(time.Second)))

			sess, err := deps.Store.GetSession(ctx, deps.RunID, sessionID)
			if err != nil {
				return &tool.Result{Content: fmt.Sprintf("Error reading session: %s", err), IsError: true}, nil
			}
			// await is a tool call in step T; CurrentStep is T+1 (the loop
			// advanced the step before running tools). Wake only for events at
			// >= T+1: the agent's own tool results land back at T and must be
			// ignored. The store — not the counter — is the source of truth.
			step := sess.CurrentStep

			// Fast path: peer events already landed at T+1 during the LLM call —
			// return without flipping status.
			if n, _ := deps.Store.GetEventCount(ctx, deps.RunID, sessionID,
				db.EventFilter{StartStep: &step}); n > 0 {
				return &tool.Result{Content: fmt.Sprintf(
					"New event(s) already present; returning immediately at %s without waiting.",
					util.FormatLocalISO(time.Now()))}, nil
			}

			_ = deps.Store.UpdateSessionStatus(ctx, deps.RunID, sessionID, db.SessionAwaiting)

			awoke := false
			for {
				// Snapshot the counter BEFORE the store check so a notify landing
				// between the check and WaitAfter still advances it and unblocks
				// the wait (closes the missed-notify race) — re-armed each loop.
				before := handle.Counter()
				if n, _ := deps.Store.GetEventCount(ctx, deps.RunID, sessionID,
					db.EventFilter{StartStep: &step}); n > 0 {
					awoke = true
					break
				}
				remaining := time.Until(deadline)
				if remaining <= 0 {
					break // timed out
				}
				if _, waitErr := handle.WaitAfter(ctx, before, remaining); waitErr != nil {
					// Interrupted by ctx — a cancel, or process shutdown. A cancel
					// already set the terminal status; shutdown preserves awaiting
					// for recovery. Either way, do NOT restore ongoing.
					return &tool.Result{Content: "Wait interrupted.", IsError: true}, nil
				}
				// Woke on a notify (possibly our own step-T result, or the
				// assistant): loop and re-check the store for a real T+1 event.
			}

			// Normal wake/timeout. A cancel may have raced in via a notify, so
			// re-read the status and skip the restore if it is already terminal —
			// never clobber a terminal status with ongoing.
			if cur, err := deps.Store.GetSession(context.Background(), deps.RunID, sessionID); err == nil &&
				db.SessionTerminalStatuses[cur.Status] {
				return &tool.Result{Content: "Wait ended: session terminated.", IsError: true}, nil
			}
			_ = deps.Store.UpdateSessionStatus(context.Background(), deps.RunID, sessionID, db.SessionOngoing)
			if awoke {
				return &tool.Result{Content: fmt.Sprintf(
					"Awakened at %s: new event(s) arrived.", util.FormatLocalISO(time.Now()))}, nil
			}
			return &tool.Result{Content: fmt.Sprintf(
				"Timed out after %.0fs at %s; no new events arrived.",
				timeoutSeconds, util.FormatLocalISO(time.Now()))}, nil
		},
	}
}

// --- send_message ---

type sendMessageParams struct {
	SessionID string `json:"session_id" jsonschema:"required" jsonschema_description:"Target session ID to send the message to"`
	Content   string `json:"content" jsonschema:"required" jsonschema_description:"Message content"`
}

func SendMessage(deps *Deps, fromSessionID string) *tool.Tool {
	return &tool.Tool{
		Name: "send_message",
		Description: "Send a message to another agent session; it appears in their event stream. " +
			"The recipient can be in ANY state: if it is currently live (running or parked) the message is " +
			"delivered to it; if it has stopped (idle, concluded, crashed, or cancelled) the message REVIVES it — " +
			"the agent restarts with your message appended to its full prior context and keeps working.",
		ParamType:       &sendMessageParams{},
		SessionRequired: true,
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[sendMessageParams](args)
			if errResult != nil {
				return errResult, nil
			}

			// Verify the target exists so a typo'd id gets a clear message rather
			// than a generic store error from the append below.
			if _, err := deps.Store.GetSession(ctx, deps.RunID, params.SessionID); err != nil {
				return &tool.Result{
					Content: fmt.Sprintf("Error: no session %q in this run.", params.SessionID),
					IsError: true,
				}, nil
			}

			_, err := deps.Store.AppendEvent(ctx, deps.RunID, params.SessionID,
				&event.MessageEvent{Content: params.Content, Sender: fromSessionID, SenderType: event.SenderTypeAgent})
			if err != nil {
				return &tool.Result{
					Content: fmt.Sprintf("Error sending message to %s: %s", params.SessionID, err),
					IsError: true,
				}, nil
			}
			// Notify is automatic via the store notifier.
			return &tool.Result{Content: fmt.Sprintf("Message sent to %s at %s",
				params.SessionID, util.FormatLocalISO(time.Now()))}, nil
		},
	}
}

// --- session_cancel ---

type sessionCancelParams struct {
	SessionID string `json:"session_id" jsonschema:"required" jsonschema_description:"Session ID to cancel"`
	Reason    string `json:"reason,omitempty" jsonschema_description:"Why you are cancelling this session"`
}

func SessionCancel(deps *Deps) *tool.Tool {
	return &tool.Tool{
		Name: "session_cancel",
		Description: "Cancel another agent session and its sub-agents. The target is stopped " +
			"immediately — any in-flight work is discarded. The reason is recorded on the " +
			"target's stream and reported to its parent.",
		ParamType:       &sessionCancelParams{},
		SessionRequired: true,
		Execute: func(_ context.Context, args json.RawMessage) (*tool.Result, error) {
			params, errResult := tool.ParseArgs[sessionCancelParams](args)
			if errResult != nil {
				return errResult, nil
			}

			reason := params.Reason
			if reason == "" {
				reason = "cancelled by peer agent"
			}

			// Canceller-driven (not cooperative): set status + write marker +
			// notify parent + interrupt the goroutine, recursively, right here.
			if err := session.CancelSession(deps.Store, deps.Registry, deps.RunID, params.SessionID, reason); err != nil {
				return &tool.Result{
					Content: fmt.Sprintf("Failed to cancel %s: %s", params.SessionID, err),
					IsError: true,
				}, nil
			}
			return &tool.Result{Content: fmt.Sprintf("Cancelled %s at %s",
				params.SessionID, util.FormatLocalISO(time.Now()))}, nil
		},
	}
}

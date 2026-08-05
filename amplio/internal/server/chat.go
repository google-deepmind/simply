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

package server

import (
	"context"
	"net/http"
	"sort"
	"time"

	"amplio/internal/db"
	"amplio/internal/event"
	"amplio/internal/toolsummary"
)

// chatBubble is a projected chat message. Tool-result bodies never cross the
// wire (the chat view doesn't render them); instead each tool call carries a
// server-computed completion flag.
type chatBubble struct {
	EventID  string `json:"event_id"`
	Kind     string `json:"kind"` // "operator" | "chatbot" | "agent" | "environment" | "child_result"
	Content  string `json:"content"`
	Thoughts string `json:"thoughts,omitempty"`
	From     string `json:"from,omitempty"`
	// Verdict is set only for kind=="child_result": the sub-agent's terminal
	// status (concluded | crashed | cancelled), so the client can badge it.
	Verdict   string         `json:"verdict,omitempty"`
	Step      int            `json:"step"`
	CreatedAt time.Time      `json:"created_at"`
	ToolCalls []chatToolCall `json:"tool_calls"`
}

type chatToolCall struct {
	ID        string `json:"id"` // tool-call id; lets the UI fetch the call+result detail on demand
	Name      string `json:"name"`
	Verb      string `json:"verb,omitempty"` // action label for bash (e.g. "search"); empty for other tools
	Completed bool   `json:"completed"`
	Errored   bool   `json:"errored,omitempty"` // tool reported IsError; chip renders red
	Detail    string `json:"detail,omitempty"`  // short arg summary / target for the chip (e.g. a filename)
}

// phaseCard is a closed (summarized) phase, rendered collapsed above the live
// bubbles so long chats stay bounded.
type phaseCard struct {
	StartStep int    `json:"start_step"`
	EndStep   int    `json:"end_step"`
	Title     string `json:"title"`
	Summary   string `json:"summary"`
}

// chatUsage is the latest assistant turn's token usage, for the status bar.
type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type chatFeed struct {
	Messages   []chatBubble `json:"messages"`
	PhaseCards []phaseCard  `json:"phase_cards"`
	Usage      *chatUsage   `json:"usage"`
}

// handleChat projects a session into the chat view: rolled-up phase cards plus
// the live conversation bubbles (operator + chatbot, plus inbound
// agent/environment messages), excluding tool-result bodies and other
// non-conversational events.
//
// Two modes:
//
//   - LIVE (no range params): everything at or below the phase boundary is
//     rolled into cards and omitted from the bubbles, so a long chat stays
//     bounded. This is what the chat page renders.
//   - RANGED (from_step/to_step): the read-only session-log viewer browsing ONE
//     phase. No rollup, no cards (the client already has the phase index from
//     the trajectory endpoint), and no usage — a historical slice has no
//     "latest turn".
//
// The projection itself is identical in both modes and works for any session,
// not just a chatbot: an autonomous agent's turns render as the same bubbles.
// Tool results are pinned to their call's step (see docs/step_model.md), so a
// range never splits a call from its result.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	id, sid := r.PathValue("id"), r.PathValue("sid")

	from, to := stepRange(r)
	ranged := from != nil || to != nil

	boundary := -1 // ranged: nothing is rolled up, so nothing is skipped below
	cards := []phaseCard{}
	if !ranged {
		var c []phaseCard
		boundary, c = s.phaseState(r.Context(), id, sid)
		if c != nil {
			cards = c
		}
	}
	recs, err := s.store.GetEvents(r.Context(), id, sid, db.EventFilter{StartStep: from, EndStep: to})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	// A tool call is "done" once any tool_result for its id exists in the session;
	// errored if that result carried the tool's IsError flag.
	completed := make(map[string]bool)
	errored := make(map[string]bool)
	for _, rec := range recs {
		if tr, ok := rec.Event.(*event.ToolResultEvent); ok {
			completed[tr.ToolCallID] = true
			if tr.IsError {
				errored[tr.ToolCallID] = true
			}
		}
	}

	msgs := make([]chatBubble, 0, len(recs))
	for _, rec := range recs {
		if rec.Step <= boundary {
			continue // rolled up into a phase card
		}
		switch ev := rec.Event.(type) {
		case *event.UserEvent:
			msgs = append(msgs, chatBubble{
				EventID: rec.EventID, Kind: "operator", Content: ev.Content,
				From: "user", Step: rec.Step, CreatedAt: rec.CreatedAt,
				ToolCalls: []chatToolCall{},
			})
		case *event.AssistantEvent:
			if ev.Content == "" && len(ev.ToolCalls) == 0 {
				continue // empty assistant turn; nothing to show
			}
			tcs := make([]chatToolCall, 0, len(ev.ToolCalls))
			for _, tc := range ev.ToolCalls {
				verb, detail := toolsummary.ToolChip(tc.Name, tc.Arguments)
				tcs = append(tcs, chatToolCall{
					ID: tc.ID, Name: tc.Name, Verb: verb, Completed: completed[tc.ID], Errored: errored[tc.ID], Detail: detail,
				})
			}
			msgs = append(msgs, chatBubble{
				EventID: rec.EventID, Kind: "chatbot", Content: ev.Content,
				Thoughts: ev.Thoughts, Step: rec.Step, CreatedAt: rec.CreatedAt,
				ToolCalls: tcs,
			})
		case *event.MessageEvent:
			// Inbound message: another agent (send_message) or the environment
			// (amplio notify). Environment bodies are typically shell output, so
			// the client renders them as <pre> rather than markdown.
			kind := "agent"
			if ev.SenderType == event.SenderTypeEnvironment {
				kind = "environment"
			}
			msgs = append(msgs, chatBubble{
				EventID: rec.EventID, Kind: kind, Content: ev.Content,
				From: ev.Sender, Step: rec.Step, CreatedAt: rec.CreatedAt,
				ToolCalls: []chatToolCall{},
			})
		case *event.ChildResultEvent:
			// A spawned sub-agent finished: its terminal result posted back to this
			// (parent) session. Surface it with the verdict so the client can badge
			// concluded vs crashed/cancelled distinctly.
			msgs = append(msgs, chatBubble{
				EventID: rec.EventID, Kind: "child_result", Content: ev.Content,
				From: ev.ChildSessionID, Verdict: ev.Verdict,
				Step: rec.Step, CreatedAt: rec.CreatedAt,
				ToolCalls: []chatToolCall{},
			})
		case *event.CompactionEvent:
			// Context compaction: the prior turns were rolled into a summary at
			// this boundary. Surface it as a distinct seam (rendered as a divider
			// the user can expand), not a chat bubble — it's meta, not a turn.
			msgs = append(msgs, chatBubble{
				EventID: rec.EventID, Kind: "compaction", Content: ev.Content,
				Step: rec.Step, CreatedAt: rec.CreatedAt,
				ToolCalls: []chatToolCall{},
			})
		}
	}
	// Latest assistant turn's usage drives the LIVE status-bar token count. A
	// ranged (historical) slice has no "latest turn" to report, so it stays nil.
	var usage *chatUsage
	for i := len(recs) - 1; !ranged && i >= 0; i-- {
		a, ok := recs[i].Event.(*event.AssistantEvent)
		if !ok {
			continue
		}
		if a.Usage != nil {
			usage = &chatUsage{a.Usage.PromptTokens, a.Usage.CompletionTokens, a.Usage.TotalTokens}
		}
		break // newest assistant event (with or without usage)
	}

	writeJSON(w, http.StatusOK, chatFeed{Messages: msgs, PhaseCards: cards, Usage: usage})
}

// phaseState returns the phase boundary (events at or before it are rolled into
// cards) and the cards themselves. The newest closed phase is kept inline as
// live bubbles, so the boundary is the second-largest phase end_step; fewer than
// two phases means nothing is rolled up.
func (s *Server) phaseState(ctx context.Context, runID, sid string) (int, []phaseCard) {
	obs, err := s.store.GetObservations(ctx, runID, db.ObsFilter{
		Kind: "phase_summary", SessionID: sid,
	})
	if err != nil || len(obs) < 2 {
		return -1, nil
	}
	sort.Slice(obs, func(i, j int) bool { return obsInt(obs[i], "end_step", 0) < obsInt(obs[j], "end_step", 0) })

	rollable := obs[:len(obs)-1] // drop the newest closed phase (stays inline)
	cards := make([]phaseCard, 0, len(rollable))
	boundary := -1
	for _, o := range rollable {
		end := obsInt(o, "end_step", 0)
		cards = append(cards, phaseCard{
			StartStep: obsInt(o, "start_step", end),
			EndStep:   end,
			Title:     obsStr(o, "title"),
			Summary:   obsStr(o, "summary"),
		})
		if end > boundary {
			boundary = end
		}
	}
	return boundary, cards
}

// obsInt reads an integer from an observation's Data bag, tolerating the
// float64 that JSON round-trips produce.
func obsInt(o db.ObservationRecord, key string, def int) int {
	switch v := o.Data[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return def
	}
}

func obsStr(o db.ObservationRecord, key string) string {
	if s, ok := o.Data[key].(string); ok {
		return s
	}
	return ""
}

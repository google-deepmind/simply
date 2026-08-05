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

package llm

import "testing"

// The specs here are real ones taken from run configs, not invented: the point
// of the labeller is to be right about what people actually type.
func TestShortLabel(t *testing.T) {
	cases := []struct{ spec, want, why string }{
		{
			"vertex-claude:claude-opus-5?cache_ttl=1h&thinking.type=adaptive&output_config.effort=xhigh&thinking.display=summarized",
			"opus-5 \u00b7 xhigh",
			"drops the family prefix and every infra knob, keeps the one facet that distinguishes two Claude entries",
		},
		{"vertex-claude:claude-sonnet-5", "sonnet-5", "no args, nothing to summarise"},
		{
			"subprocess:mybridge?model=llama-3-70b",
			"llama-3-70b",
			"the bridge name is in the model position; without this rule every bridge run looks identical",
		},
		{"subprocess:mybridge", "mybridge", "no model arg: the bridge name is all there is"},
		{
			"openai:qwen3.5:35b?base_url=http://localhost:11434/v1",
			"qwen3.5:35b \u00b7 @localhost:11434",
			"the server is what distinguishes two openai-protocol entries; the port distinguishes two local ones",
		},
		{"openai:gpt-5.4-nano", "gpt-5.4-nano", "hosted API: no base_url, no facet"},
		{
			"openai:some-model?base_url=localhost:4000&reasoning_effort=high",
			"some-model \u00b7 @localhost:4000 \u00b7 high",
			"scheme-less base_url still yields a host; effort is recognised under its openai spelling",
		},
		{"vertex-gemini:gemini-3.5-pro?thinking_budget=32768", "gemini-3.5-pro", "gemini keeps its family prefix, or nothing identifies it"},
		{
			"subprocess:mybridge?model=exp-endpoint-7#candidate-rc2",
			"candidate-rc2",
			"an explicit nickname wins outright \u2014 the case a heuristic cannot serve, e.g. a reused test endpoint",
		},
		{"vertex-claude:claude-opus-5?output_config.effort=xhigh  #  spaced  ", "spaced", "nickname and spec are trimmed"},
		{"plain-nonsense", "plain-nonsense", "not a spec: echo it rather than fail"},
		{"", "", "empty in, empty out"},
		{"openai:m?base_url=%zz", "m", "unparseable args degrade to no facets, never to an error"},
		{
			"vertex-claude:claude-a-very-long-model-name-that-keeps-going-and-going",
			"a-very-long-model-name-that-keeps-going\u2026",
			"truncation is the backstop so a chip can't be blown out",
		},
	}
	for _, c := range cases {
		if got := ShortLabel(c.spec); got != c.want {
			t.Errorf("ShortLabel(%q) = %q, want %q\n  (%s)", c.spec, got, c.want, c.why)
		}
	}
}

// A label is display-only; the spec that reaches a provider must be unchanged
// apart from the nickname being removed.
func TestBaseSpec(t *testing.T) {
	cases := []struct{ spec, want string }{
		{"vertex-claude:claude-opus-5?cache_ttl=1h#nick", "vertex-claude:claude-opus-5?cache_ttl=1h"},
		{"vertex-claude:claude-opus-5?cache_ttl=1h", "vertex-claude:claude-opus-5?cache_ttl=1h"},
		{"  spaced:model  #  nick  ", "spaced:model"},
		{"", ""},
	}
	for _, c := range cases {
		if got := BaseSpec(c.spec); got != c.want {
			t.Errorf("BaseSpec(%q) = %q, want %q", c.spec, got, c.want)
		}
	}
}

// TestShortLabel_SubprocessShapes: a chip must name the MODEL, not the bridge —
// four endpoints behind one bridge rendered identically before this.
func TestShortLabel_SubprocessShapes(t *testing.T) {
	for _, tc := range []struct{ spec, want string }{
		{"subprocess{bin=/opt/bridges/corp}:exp-endpoint-7", "exp-endpoint-7"},
		{"subprocess:/opt/bridges/corp?model=exp-endpoint-7", "exp-endpoint-7"},
		{"subprocess{bin=/opt/bridges/corp}:exp-endpoint-7#rc2", "rc2"},
	} {
		if got := ShortLabel(tc.spec); got != tc.want {
			t.Errorf("ShortLabel(%q) = %q, want %q", tc.spec, got, tc.want)
		}
	}
}

// TestShortLabel_Bridge: a bridged model must be labelled by what it IS, with
// the endpoint as a facet — two bridges serving the same model are otherwise
// indistinguishable in the picker.
func TestShortLabel_Bridge(t *testing.T) {
	for _, tc := range []struct{ spec, want string }{
		{"bridge{url=https://ws.corp:26759/api/llm}:opus-xhigh", "⇄ opus-xhigh @ws.corp:26759"},
		{"bridge{url=https://ws.corp:26759/api/llm}:vertex-claude:claude-opus-5?output_config.effort=xhigh",
			"⇄ opus-5 · xhigh @ws.corp:26759"},
		// A unix socket's file name is the distinguishing part; the path would eat
		// the chip.
		{"bridge{url=unix:///tmp/dev-bridge.sock}:vertex-claude:claude-opus-5", "⇄ opus-5 @dev-bridge.sock"},
		// An explicit nickname still wins outright, as everywhere else.
		{"bridge{url=https://ws:26759}:vertex-claude:claude-opus-5#remote opus", "remote opus"},
	} {
		if got := ShortLabel(tc.spec); got != tc.want {
			t.Errorf("ShortLabel(%q) =\n %q\nwant %q", tc.spec, got, tc.want)
		}
	}
}

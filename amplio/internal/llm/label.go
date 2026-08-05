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

import (
	"net/url"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

// This file derives short, human-facing labels for LLM specs, which have grown
// long enough to be unreadable in a chip or a menu row:
//
//	vertex-claude:claude-opus-5?cache_ttl=1h&thinking.type=adaptive&output_config.effort=xhigh&thinking.display=summarized
//
// Three rules govern everything here:
//
//  1. PRESENTATION ONLY. A label is never persisted, never a map key, never in
//     an error or a log line (those keep the raw spec, so what an operator reads
//     is what they can paste back into config), and never sent to a provider.
//  2. NEVER SHOWN ALONE. Every display site must keep the full spec reachable —
//     a tooltip, or a dimmed second line. The label is a summary, not an
//     identity, and it is lossy on purpose.
//  3. BEST EFFORT. The rules below are heuristics over conventions, not a
//     parser. A new provider, or an unusual spec, falls through to something
//     reasonable rather than failing; getting a label wrong costs a slightly
//     confusing chip, and is fixed by editing this file.

// maxLabelLen caps a label so a chip can't be blown out by a long model name or
// host. Deliberately generous: truncation is the last resort, not the mechanism,
// and it cuts from the tail — where the facets are — so a cap tight enough to
// bite a normal three-part label would eat the very thing that distinguishes it.
const maxLabelLen = 40

// nicknameSep introduces an optional operator-supplied display override:
//
//	subprocess:mybridge?model=exp-endpoint-7#candidate-rc2
//
// '#' is chosen for the analogy: a URL fragment is by definition never sent to
// the server, and neither is this. It sits OUTSIDE the query string on purpose
// — every '?k=v' key belongs to the provider, and a harness-owned key in there
// would break that rule for readers as well as for code.
const nicknameSep = "#"

// SplitNickname separates a spec from its optional display nickname.
//
// A plain cut at the first '#' is correct even with client blocks in play,
// because a literal '#' inside a block must be percent-encoded (see spec.go) —
// so the first raw one always introduces the nickname. Unlike ParseSpec this
// never fails: it runs on operator-typed text that may not be a valid spec at
// all, and labelling and dedup still have to do something reasonable with it.
func SplitNickname(spec string) (base, nickname string) {
	base, nickname, _ = strings.Cut(spec, nicknameSep)
	return strings.TrimSpace(base), strings.TrimSpace(nickname)
}

// BaseSpec is the spec with any display nickname removed: what every consumer
// that actually resolves a provider must use. Callers that merely store or
// compare specs should keep them verbatim, so the operator's label survives.
func BaseSpec(spec string) string {
	base, _ := SplitNickname(spec)
	return base
}

// ShortLabel returns a compact display name for an LLM spec. See the file
// comment for the three rules that constrain it — in particular, callers must
// keep the full spec visible alongside.
func ShortLabel(spec string) string {
	sp, err := ParseSpec(spec)
	if err != nil {
		// Not a spec we can parse — the new-run form accepts arbitrary text.
		// Show it as-is, still honouring a nickname if one is discernible.
		base, nickname := SplitNickname(spec)
		if nickname != "" {
			return truncateLabel(nickname)
		}
		return truncateLabel(base)
	}
	if sp.Nickname != "" {
		return truncateLabel(sp.Nickname) // an explicit override wins outright
	}
	model, args, err := sp.Model()
	if err != nil {
		// Best effort: unparseable args just yield no facets.
		m, _, _ := strings.Cut(sp.Tail, "?")
		return truncateLabel(m)
	}
	// Labelling also reads the query, unlike everything else: a menu row written
	// before the block existed is still displayed, long after it stopped being
	// constructible.
	arg := func(k string) string {
		if v := sp.Client.Get(k); v != "" {
			return v
		}
		return args.Get(k)
	}

	if sp.Provider == "bridge" {
		// A bridged model is labelled by what it IS, not by how it is reached, so
		// recurse on the handle: a spec handle gets the same treatment it would
		// get locally (family prefix dropped, effort surfaced), and a nickname
		// handle is already the label its operator chose. The endpoint follows as
		// a facet, because two bridges serving the same model are otherwise
		// indistinguishable. Returns early: the tail carries its own args, so the
		// generic effort facet below would repeat what the recursion found.
		label := "\u21c4 " + ShortLabel(sp.Tail)
		if h := endpointLabel(sp.Client.Get("url")); h != "" {
			label += " @" + h
		}
		return truncateLabel(label)
	}

	var facets []string
	switch sp.Provider {
	case "vertex-claude", "claude":
		// "claude-opus-5" -> "opus-5": the family name is redundant with the model
		// name, which stays recognisable without it. NOT done for Gemini, whose
		// models are named "gemini-3.5-pro" — strip there and only "3.5-pro" is
		// left, which identifies nothing.
		model = strings.TrimPrefix(model, "claude-")
	case "subprocess":
		// With bin= the model position IS the model. A row stored before that put
		// the binary there instead, with the model in the query; without this,
		// every model behind one bridge renders as the same chip.
		if sp.Client.Get("bin") == "" {
			if m := arg("model"); m != "" {
				model = m
			}
		}
	case "openai":
		// One provider fronts the whole OpenAI-compatible ecosystem, so the server
		// is often the only thing distinguishing two entries (a local ollama from a
		// proxy from the hosted API). Host, including port, since ports are exactly
		// what separates two local servers.
		if h := hostOf(arg("base_url")); h != "" {
			facets = append(facets, "@"+h)
		}
	}
	if e := effortOf(args); e != "" {
		facets = append(facets, e)
	}
	return truncateLabel(strings.Join(append([]string{model}, facets...), " \u00b7 "))
}

// effortLevelKeys are the spelling variants for "how hard should it think",
// across families. Effort is worth surfacing because it is the one argument
// that routinely distinguishes two otherwise identical menu entries.
var effortLevelKeys = []string{"output_config.effort", "reasoning_effort", "effort"}

func effortOf(args url.Values) string {
	for _, k := range effortLevelKeys {
		if v := strings.TrimSpace(args.Get(k)); v != "" {
			return v
		}
	}
	return ""
}

// endpointLabel names a bridge endpoint compactly: the host for a network
// endpoint, the socket's file name for a unix one (where a full path would eat
// the whole chip and the last component is the distinguishing part anyway).
func endpointLabel(raw string) string {
	if socket, ok := strings.CutPrefix(strings.TrimSpace(raw), "unix://"); ok {
		return filepath.Base(socket)
	}
	return hostOf(raw)
}

// hostOf extracts host:port from a base_url, tolerating a bare "host:port" with
// no scheme (url.Parse would read that as scheme:opaque).
func hostOf(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if !strings.Contains(raw, "//") {
		raw = "//" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return u.Host
}

func truncateLabel(s string) string {
	s = strings.TrimSpace(s)
	if utf8.RuneCountInString(s) <= maxLabelLen {
		return s
	}
	return string([]rune(s)[:maxLabelLen-1]) + "\u2026"
}

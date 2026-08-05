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
	"fmt"
	"net/url"
	"sort"
	"strings"
)

// A Menu is the set of llm models a server offers, and — when it is serving
// generations for someone else — the set it will accept. One mechanism does both
// jobs, which is the point: resolution has to consult the menu anyway to
// interpret nicknames, so the restriction costs nothing.
//
// THE RULE: the handle names WHICH model; the menu supplies HOW TO REACH IT.
//
//	handle              what it must match
//	------              ------------------
//	a nickname          exactly one menu entry's #nickname
//	a spec              exactly one menu entry's (provider, model)
//
// and then:
//
//   - the CLIENT BLOCK comes from the menu entry, never from the caller.
//     Endpoints, binaries and credential references are the server's business.
//   - the caller's MODEL ARGS are merged over the entry's, caller winning, so
//     temperature and thinking knobs stay freely variable. They are the model's
//     vocabulary, the model validates them, and nothing dangerous lives there.
//
// Matching on (provider, model) rather than on the whole canonical spec is
// deliberate: an exact-spec rule would refuse a perfectly reasonable
// ?temperature=0.2 on a model the operator has already allowed.
type Menu struct {
	// Specs are the menu's entries, verbatim and in display order.
	Specs []string
}

// Resolve turns a caller's handle into a spec this server will run, or explains
// why it won't. The returned spec is canonical and carries no nickname: it is
// for construction, not display.
func (m Menu) Resolve(handle string) (string, error) {
	handle = strings.TrimSpace(handle)
	if handle == "" {
		return "", fmt.Errorf("no model requested; want a spec or a nickname from this server's menu (%s)", m.summary())
	}
	entries := m.parse()
	if len(entries) == 0 {
		return "", fmt.Errorf("this server offers no models")
	}

	if req, err := ParseSpec(handle); err == nil {
		if spec, err := resolveSpec(entries, req); err == nil {
			return spec, nil
		} else if _, isAlias := matchAlias(entries, handle); !isAlias {
			// Not an alias either, so the spec-shaped error is the useful one.
			return "", err
		}
	}
	entry, ok := matchAlias(entries, handle)
	if !ok {
		return "", fmt.Errorf("%q is not in this server's menu; it offers %s", handle, m.summary())
	}
	return entry.String(), nil
}

// resolveSpec applies the rule above to a handle that parses as a spec.
func resolveSpec(entries []menuEntry, req *Spec) (string, error) {
	if len(req.Client) > 0 {
		return "", fmt.Errorf("client args are not accepted from a caller (%s): this server supplies them",
			strings.Join(sortedKeys(req.Client), ", "))
	}
	reqModel, reqArgs, err := req.Model()
	if err != nil {
		return "", err
	}
	// Rows for this provider and model, and among them the rows whose args the
	// caller reproduced exactly. Being MORE specific must never be worse than
	// being vague: two rows differing only in their args are ambiguous when asked
	// for by model alone, but a caller who names one exactly has chosen it.
	var byModel, exact []*Spec
	for _, e := range entries {
		model, args, err := e.sp.Model()
		if err != nil || e.sp.Provider != req.Provider || model != reqModel {
			continue
		}
		byModel = append(byModel, e.sp)
		if EncodeArgs(args) == EncodeArgs(reqArgs) {
			exact = append(exact, e.sp)
		}
	}
	candidates := byModel
	if len(exact) > 0 {
		candidates = exact
	}
	if len(candidates) == 0 {
		return "", fmt.Errorf("%s:%s is not in this server's menu", req.Provider, reqModel)
	}

	// Several rows can be one model wearing different labels — relabelling an
	// entry is done by adding a #nickname, which the menu keeps as a separate
	// row. Those are not ambiguous: they resolve to the same thing.
	resolved := map[string]*Spec{}
	for _, c := range candidates {
		model, args, err := c.Model()
		if err != nil {
			continue
		}
		resolved[(&Spec{Provider: c.Provider, Client: c.Client, Tail: model + argSuffix(args)}).String()] = c
	}
	if len(resolved) != 1 {
		// Genuinely different models behind one name — different args, or a
		// different endpoint in the client block. Picking one would be a guess
		// with a billing consequence.
		return "", fmt.Errorf("%s:%s matches %d menu entries; ask for one by the name the menu shows, or give its full spec",
			req.Provider, reqModel, len(resolved))
	}

	entry := candidates[0]
	model, args, err := entry.Model()
	if err != nil {
		return "", err
	}
	// Caller's model args win: they are the model's own vocabulary, and the model
	// validates them.
	merged := url.Values{}
	for k, v := range args {
		merged[k] = v
	}
	for k, v := range reqArgs {
		merged[k] = v
	}
	out := &Spec{Provider: entry.Provider, Client: entry.Client, Tail: model + argSuffix(merged)}
	return out.String(), nil
}

// argSuffix renders a query, or nothing when there are no args.
func argSuffix(args url.Values) string {
	if len(args) == 0 {
		return ""
	}
	return "?" + EncodeArgs(args)
}

// matchAlias resolves a handle against what the menu CALLS its entries: the
// operator's #nickname if there is one, otherwise the label ShortLabel derives —
// the string the picker shows. Accepting the derived label means a caller can
// use what they see without the operator having to nickname every row, which
// they would otherwise have to do for a name to exist at all.
//
// Nicknames are tried first, so naming a row always wins over whatever the
// heuristic would have produced for it. Either pass refuses an ambiguous handle
// rather than guessing.
//
// A derived label is a convenience, not a contract: it comes from a heuristic
// that can change between releases (a facet added, the length cap moved), which
// a #nickname or a spec cannot. When that happens the caller gets a refusal
// listing what is on offer, not a wrong model.
func matchAlias(entries []menuEntry, handle string) (*Spec, bool) {
	for _, name := range []func(menuEntry) string{
		func(e menuEntry) string { return e.sp.Nickname },
		func(e menuEntry) string { return ShortLabel(e.spec) },
	} {
		var hit *Spec
		ambiguous := false
		for _, e := range entries {
			if name(e) != handle {
				continue
			}
			if hit != nil {
				ambiguous = true
				break
			}
			hit = e.sp
		}
		if ambiguous {
			return nil, false
		}
		if hit != nil {
			// Drop the nickname: it is a display label, and the resolved spec is
			// for construction.
			return &Spec{Provider: hit.Provider, Client: hit.Client, Tail: hit.Tail}, true
		}
	}
	return nil, false
}

// menuEntry pairs a parsed spec with the verbatim string, which ShortLabel needs
// and which is also what the operator sees in the picker.
type menuEntry struct {
	spec string
	sp   *Spec
}

func (m Menu) parse() []menuEntry {
	out := make([]menuEntry, 0, len(m.Specs))
	for _, s := range m.Specs {
		sp, err := ParseSpec(s)
		if err != nil {
			continue // a menu entry that doesn't parse can't be run locally either
		}
		out = append(out, menuEntry{spec: s, sp: sp})
	}
	return out
}

// summary lists what a caller could have asked for. Nicknames first, since they
// are the stable handles; then bare provider:model pairs. Truncated, because an
// error is not a catalogue.
func (m Menu) summary() string {
	const maxItems = 12
	seen := map[string]bool{}
	var items []string
	add := func(s string) {
		if s == "" || seen[s] {
			return
		}
		seen[s] = true
		items = append(items, s)
	}
	for _, e := range m.parse() {
		add(ShortLabel(e.spec)) // the nickname if there is one, else the label shown
	}
	for _, e := range m.parse() {
		if model, _, err := e.sp.Model(); err == nil {
			add(e.sp.Provider + ":" + model)
		}
	}
	if len(items) == 0 {
		return "nothing"
	}
	if len(items) > maxItems {
		return strings.Join(items[:maxItems], ", ") + fmt.Sprintf(", … (%d more)", len(items)-maxItems)
	}
	return strings.Join(items, ", ")
}

func sortedKeys(v url.Values) []string {
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

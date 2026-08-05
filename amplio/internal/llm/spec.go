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
	"strconv"
	"strings"
)

// This file defines the LLM spec language and its one parser.
//
//	provider block? ':' tail ( '#' nickname )?
//
//	provider   registry key: vertex-claude | openai | subprocess | bridge | …
//	block      '{' k=v ('&' k=v)* '}'   arguments amplio INTERPRETS
//	tail       model ('?' k=v…)?        model plus arguments PASSED THROUGH
//	           | an inner spec          for a provider that forwards (bridge)
//	nickname   display only; stripped before dispatch, never sent anywhere
//
// One rule governs which side an argument belongs on:
//
//	The block holds arguments amplio interprets.
//	The query holds arguments passed through untouched.
//
// Note what that does NOT say: interpreted arguments frequently do reach the
// API. cache_ttl=1h becomes Anthropic's cache_control{type:ephemeral,ttl:"1h"};
// profile=litellm sets several body fields. What separates them is who owns the
// MEANING. The test for a new argument: would it still mean the same thing
// pointed at a different provider? If yes it is amplio's vocabulary and belongs
// in the block; if it is the upstream API's own field name, it belongs in the
// query and we never look at it.
//
// The block exists because nesting a spec inside a spec — a bridge forwarding to
// a provider — otherwise forces every hop's arguments to share one query
// namespace, where '.' already means "nest into the request body" for two
// providers. Separating them structurally beats any reserved-name convention: it
// cannot collide, and it lets a provider validate the half it owns while leaving
// the model's half deliberately unvalidated.
//
// BACKWARD COMPATIBLE by construction: a spec with no block parses to exactly
// what it meant before this file existed.

const (
	blockOpen  = '{'
	blockClose = '}'
	nicknameCh = '#'
)

// Spec is a parsed LLM spec. Build one with ParseSpec; render the canonical form
// with String.
type Spec struct {
	// Provider is the registry key: the spec's first token.
	Provider string
	// Client holds the block's arguments — the ones amplio interprets. Never nil.
	Client url.Values
	// Tail is everything after the ':', verbatim and with the nickname removed.
	// For a terminal provider it is "model[?args]" (see Model); for a forwarding
	// provider it is an inner spec, which we deliberately do not interpret.
	Tail string
	// Nickname is the operator's display override, or "". Presentation only:
	// never persisted as an identity, never sent to a provider (see label.go).
	Nickname string
}

// ParseSpec parses an LLM spec. It is strict about the structure it owns (the
// provider token, the block, the nickname) and indifferent to everything inside
// the tail, which belongs to the provider.
func ParseSpec(spec string) (*Spec, error) {
	s := strings.TrimSpace(spec)
	if s == "" {
		return nil, fmt.Errorf("empty LLM spec")
	}

	base, nickname, err := splitNickname(s)
	if err != nil {
		return nil, fmt.Errorf("LLM spec %q: %w", spec, err)
	}

	provider, block, tail, err := splitHead(base)
	if err != nil {
		return nil, fmt.Errorf("LLM spec %q: %w", spec, err)
	}
	if provider == "" {
		return nil, fmt.Errorf("LLM spec %q: empty provider; want <provider>:<model>[?k=v&…]", spec)
	}
	if tail == "" {
		return nil, fmt.Errorf("LLM spec %q: nothing after %q:; want <provider>:<model>[?k=v&…]", spec, provider)
	}

	client, err := parseArgs(block)
	if err != nil {
		return nil, fmt.Errorf("LLM spec %q: client block: %w", spec, err)
	}
	return &Spec{Provider: provider, Client: client, Tail: tail, Nickname: nickname}, nil
}

// Model splits the tail into the model and its pass-through arguments. Only
// meaningful for a provider that terminates the chain; a forwarding provider
// should use Tail verbatim instead.
//
// The split is on the FIRST '?' only: a model may contain almost anything
// (an ollama tag, a filesystem path), and all of it is the provider's business.
func (sp *Spec) Model() (string, url.Values, error) {
	model, rawArgs, _ := strings.Cut(sp.Tail, "?")
	if model == "" {
		return "", nil, fmt.Errorf("empty model in %q", sp.Tail)
	}
	args, err := parseArgs(rawArgs)
	if err != nil {
		return "", nil, fmt.Errorf("model args in %q: %w", sp.Tail, err)
	}
	return model, args, nil
}

// String renders the canonical form: block and query keys sorted, values escaped
// minimally, an empty block or empty query omitted, the nickname preserved.
//
// Canonical form is for MATCHING (menu lookup, dedup, cache keys), never for
// display: it may reorder what the operator typed, so errors and the UI keep the
// verbatim string. Parsing a canonical spec and re-rendering it yields the same
// string, which is what makes it usable as a key.
func (sp *Spec) String() string {
	var b strings.Builder
	b.WriteString(sp.Provider)
	if len(sp.Client) > 0 {
		b.WriteByte(blockOpen)
		b.WriteString(encodeArgs(sp.Client))
		b.WriteByte(blockClose)
	}
	b.WriteByte(':')
	model, args, err := sp.Model()
	if err != nil {
		b.WriteString(sp.Tail) // unsplittable tail: keep it verbatim rather than lose it
	} else {
		b.WriteString(model)
		if len(args) > 0 {
			b.WriteByte('?')
			b.WriteString(encodeArgs(args))
		}
	}
	if sp.Nickname != "" {
		b.WriteByte(nicknameCh)
		b.WriteString(sp.Nickname)
	}
	return b.String()
}

// splitNickname cuts the trailing "#nickname", which is only recognised OUTSIDE
// a block — a literal '#' inside one must be percent-encoded, so this stays a
// single pass with no lookahead.
func splitNickname(s string) (base, nickname string, err error) {
	depth := 0
	for i, r := range s {
		switch r {
		case blockOpen:
			if depth > 0 {
				return "", "", fmt.Errorf("nested %q in a client block; percent-encode it as %%7B", string(blockOpen))
			}
			depth++
		case blockClose:
			if depth == 0 {
				return "", "", fmt.Errorf("unmatched %q; percent-encode a literal one as %%7D", string(blockClose))
			}
			depth--
		case nicknameCh:
			if depth > 0 {
				return "", "", fmt.Errorf("raw %q in a client block; percent-encode it as %%23", string(nicknameCh))
			}
			return strings.TrimSpace(s[:i]), strings.TrimSpace(s[i+1:]), nil
		}
	}
	if depth > 0 {
		return "", "", fmt.Errorf("unterminated client block: missing %q", string(blockClose))
	}
	return strings.TrimSpace(s), "", nil
}

// splitHead separates the provider token, its optional block, and the tail.
func splitHead(base string) (provider, block, tail string, err error) {
	open := strings.IndexByte(base, blockOpen)
	colon := strings.IndexByte(base, ':')
	switch {
	case open < 0:
		// No block: the classic "provider:tail" form.
		if colon < 0 {
			return "", "", "", fmt.Errorf("missing %q; want <provider>:<model>[?k=v&…]", ":")
		}
		return strings.TrimSpace(base[:colon]), "", base[colon+1:], nil
	case colon >= 0 && colon < open:
		// A '{' after the colon belongs to the tail (a nested spec's own block,
		// or a model name we have no business inspecting).
		return strings.TrimSpace(base[:colon]), "", base[colon+1:], nil
	}
	end := strings.IndexByte(base, blockClose) // balance already checked by splitNickname
	if end < 0 {
		return "", "", "", fmt.Errorf("unterminated client block: missing %q", string(blockClose))
	}
	rest := base[end+1:]
	if !strings.HasPrefix(rest, ":") {
		return "", "", "", fmt.Errorf("expected %q after the client block, got %q", ":", rest)
	}
	return strings.TrimSpace(base[:open]), base[open+1 : end], rest[1:], nil
}

// parseArgs decodes a "k=v&k=v" fragment. Empty yields empty (not nil) Values.
func parseArgs(raw string) (url.Values, error) {
	if strings.TrimSpace(raw) == "" {
		return url.Values{}, nil
	}
	v, err := url.ParseQuery(raw)
	if err != nil {
		return nil, err
	}
	return v, nil
}

// EncodeArgs renders args the way a canonical spec does — sorted, and escaped
// only where a reader would otherwise misread them. Exported for the bridge
// provider, which rebuilds a handle from its parts and needs the result to match
// what the far end computes for the same spec.
func EncodeArgs(v url.Values) string { return encodeArgs(v) }

// encodeArgs is url.Values.Encode with a gentler escape set. Encode percent-
// escapes ':' and '/', which turns a perfectly readable
//
//	{url=https://ws:26759}
//
// into {url=https%3A%2F%2Fws%3A26759} — technically fine, since canonical specs
// round-trip either way, but canonical specs land in menus, DB rows and logs,
// and a URL nobody can read is a poor trade for nothing. Both characters are
// legal in a query per RFC 3986, so we escape only what the READER needs:
// whatever url.ParseQuery treats specially, plus our own delimiters.
func encodeArgs(v url.Values) string {
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		for _, val := range v[k] {
			if b.Len() > 0 {
				b.WriteByte('&')
			}
			b.WriteString(escapeArg(k))
			b.WriteByte('=')
			b.WriteString(escapeArg(val))
		}
	}
	return b.String()
}

// argMustEscape reports whether c would change meaning if left literal:
// '%' (starts an escape), '&' '=' ';' (ParseQuery's separators), '+' (which
// ParseQuery decodes to a space), whitespace, and our own '{' '}' '#'.
func argMustEscape(c byte) bool {
	switch c {
	case '%', '&', '=', ';', '+', ' ', '\t', '\n', '\r', blockOpen, blockClose, nicknameCh:
		return true
	}
	return c < 0x20 || c == 0x7f
}

func escapeArg(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if c := s[i]; argMustEscape(c) {
			fmt.Fprintf(&b, "%%%02X", c)
		} else {
			b.WriteByte(c)
		}
	}
	return b.String()
}

// universalClientArgs are client args every provider accepts. They are handled
// centrally (see createProvider) rather than by each factory, so "cap the output
// at N" means the same thing everywhere instead of being reimplemented — or
// silently forwarded as a body field — once per provider.
var universalClientArgs = map[string]bool{
	"max_tokens": true,
}

// ClientArgs validates a spec's block against what this provider declares, and
// returns it. There is no corresponding check on the query, and that absence is
// the design: the block is amplio's namespace and the query is the model's, so
// the two cannot collide and nothing here needs to guess which is which.
//
//   - An undeclared key in the BLOCK is an error naming what the provider does
//     accept. We own that namespace, so a typo in it is knowable — as opposed to
//     being shipped to the server as a body field, where it either 400s
//     confusingly or is quietly ignored.
//   - Query keys are never inspected, let alone validated. The server is the
//     authority on what it accepts, and an allowlist here would make every new
//     server knob a code change. A key there that happens to share a name with a
//     client arg is simply the model's: it is forwarded, and the model decides.
func ClientArgs(block url.Values, declared map[string]bool) (url.Values, error) {
	client := url.Values{}
	for k, vs := range block {
		if !declared[k] && !universalClientArgs[k] {
			return nil, fmt.Errorf("unknown client arg %q; this provider takes %s", k, declaredList(declared))
		}
		client[k] = vs
	}
	return client, nil
}

// MaxTokensArg reads — and consumes — the one client arg every provider accepts,
// returning def when it is absent. Consuming it keeps the contract simple: by
// the time a provider sees clientArgs, the output cap has already been applied
// to its maxTokens parameter, so there is no second copy to disagree with.
//
// Handled centrally rather than per provider so "cap the output at N" means the
// same thing everywhere. Before this it was honoured only by subprocess and fell
// through to the request body elsewhere — where, on the openai profile, it
// produced a body carrying BOTH max_completion_tokens and max_tokens, the latter
// rejected outright by newer models.
func MaxTokensArg(clientArgs url.Values, def int) (int, error) {
	v := clientArgs.Get("max_tokens")
	if v == "" {
		return def, nil
	}
	clientArgs.Del("max_tokens")
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("max_tokens=%q; want a positive integer", v)
	}
	return n, nil
}

func declaredList(declared map[string]bool) string {
	keys := make([]string, 0, len(declared)+len(universalClientArgs))
	for k := range declared {
		keys = append(keys, k)
	}
	for k := range universalClientArgs {
		keys = append(keys, k)
	}
	if len(keys) == 0 {
		return "no client args"
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

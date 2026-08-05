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
	"strings"
	"testing"
)

func TestParseSpec(t *testing.T) {
	tests := []struct {
		name     string
		spec     string
		provider string
		client   map[string]string
		tail     string
		nickname string
	}{
		{
			name:     "classic spec, no block",
			spec:     "vertex-claude:claude-opus-5?cache_ttl=1h&thinking.type=adaptive",
			provider: "vertex-claude",
			tail:     "claude-opus-5?cache_ttl=1h&thinking.type=adaptive",
		},
		{
			name:     "bare model, no args",
			spec:     "openai:gpt-5",
			provider: "openai",
			tail:     "gpt-5",
		},
		{
			name:     "client block",
			spec:     "openai{base_url=http://localhost:4000/v1&profile=litellm}:claude?reasoning.effort=high",
			provider: "openai",
			client:   map[string]string{"base_url": "http://localhost:4000/v1", "profile": "litellm"},
			tail:     "claude?reasoning.effort=high",
		},
		{
			name:     "empty block is legal and means no client args",
			spec:     "openai{}:gpt-5",
			provider: "openai",
			tail:     "gpt-5",
		},
		{
			name:     "nickname",
			spec:     "vertex-claude:claude-opus-5#opus xhigh",
			provider: "vertex-claude",
			tail:     "claude-opus-5",
			nickname: "opus xhigh",
		},
		{
			name:     "block and nickname",
			spec:     "bridge{url=https://ws:26759}:opus-xhigh#remote",
			provider: "bridge",
			client:   map[string]string{"url": "https://ws:26759"},
			tail:     "opus-xhigh",
			nickname: "remote",
		},
		{
			name:     "nested spec in the tail is not interpreted",
			spec:     "bridge{url=https://ws:26759}:vertex-claude:claude-opus-5?output_config.effort=xhigh",
			provider: "bridge",
			client:   map[string]string{"url": "https://ws:26759"},
			tail:     "vertex-claude:claude-opus-5?output_config.effort=xhigh",
		},
		{
			name:     "two hops: only the outermost is parsed",
			spec:     "bridge{url=A}:bridge{url=B}:openai:gpt-5",
			provider: "bridge",
			client:   map[string]string{"url": "A"},
			tail:     "bridge{url=B}:openai:gpt-5",
		},
		{
			name:     "a path in the model position keeps its slashes and colons",
			spec:     "subprocess:/opt/bridges/corp?model=some-model",
			provider: "subprocess",
			tail:     "/opt/bridges/corp?model=some-model",
		},
		{
			name:     "ollama-style tag: the model may contain a colon",
			spec:     "openai:nomic-embed-text:latest",
			provider: "openai",
			tail:     "nomic-embed-text:latest",
		},
		{
			name:     "percent-encoded delimiters inside a block",
			spec:     "bridge{url=https://ws/%7Bx%7D&note=a%23b}:opus",
			provider: "bridge",
			client:   map[string]string{"url": "https://ws/{x}", "note": "a#b"},
			tail:     "opus",
		},
		{
			name:     "surrounding whitespace is tolerated",
			spec:     "  openai:gpt-5  ",
			provider: "openai",
			tail:     "gpt-5",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sp, err := ParseSpec(tt.spec)
			if err != nil {
				t.Fatalf("ParseSpec(%q) = error %v", tt.spec, err)
			}
			if sp.Provider != tt.provider {
				t.Errorf("provider = %q, want %q", sp.Provider, tt.provider)
			}
			if sp.Tail != tt.tail {
				t.Errorf("tail = %q, want %q", sp.Tail, tt.tail)
			}
			if sp.Nickname != tt.nickname {
				t.Errorf("nickname = %q, want %q", sp.Nickname, tt.nickname)
			}
			if got, want := len(sp.Client), len(tt.client); got != want {
				t.Errorf("client has %d args, want %d (%v)", got, want, sp.Client)
			}
			for k, v := range tt.client {
				if got := sp.Client.Get(k); got != v {
					t.Errorf("client[%q] = %q, want %q", k, got, v)
				}
			}
		})
	}
}

func TestParseSpecErrors(t *testing.T) {
	tests := []struct {
		name string
		spec string
		want string // substring of the error
	}{
		{"empty", "", "empty LLM spec"},
		{"no colon", "gpt-5", "missing"},
		{"empty provider", ":gpt-5", "empty provider"},
		{"nothing after the colon", "openai:", "nothing after"},
		{"unterminated block", "openai{base_url=x:gpt-5", "unterminated"},
		{"unmatched close", "openai}:gpt-5", "unmatched"},
		{"nested open", "openai{a={b}:gpt-5", "nested"},
		{"raw # inside a block", "bridge{url=http://x/#frag}:m", "%23"},
		{"missing colon after the block", "openai{a=b}gpt-5", "expected"},
		{"malformed escape in a block", "openai{base_url=%zz}:gpt-5", "client block"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseSpec(tt.spec)
			if err == nil {
				t.Fatalf("ParseSpec(%q) = nil error, want one", tt.spec)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("ParseSpec(%q) error = %q, want it to contain %q", tt.spec, err, tt.want)
			}
		})
	}
}

func TestSpecCanonicalString(t *testing.T) {
	tests := []struct {
		name string
		spec string
		want string
	}{
		{
			name: "argument order does not matter",
			spec: "vertex-claude:claude-opus-5?thinking.type=adaptive&cache_ttl=1h",
			want: "vertex-claude:claude-opus-5?cache_ttl=1h&thinking.type=adaptive",
		},
		{
			name: "block keys are sorted too",
			spec: "openai{profile=litellm&base_url=http://localhost:4000/v1}:claude",
			want: "openai{base_url=http://localhost:4000/v1&profile=litellm}:claude",
		},
		{
			name: "an empty block is dropped",
			spec: "openai{}:gpt-5",
			want: "openai:gpt-5",
		},
		{
			name: "a trailing ? is dropped",
			spec: "openai:gpt-5?",
			want: "openai:gpt-5",
		},
		{
			name: "URLs stay readable: ':' and '/' are not escaped",
			spec: "bridge{url=https://ws:26759/x}:opus",
			want: "bridge{url=https://ws:26759/x}:opus",
		},
		{
			name: "delimiters are escaped on the way out",
			spec: "bridge{note=a%23b%7Dc}:opus",
			want: "bridge{note=a%23b%7Dc}:opus",
		},
		{
			name: "'+' means space on the way in, and is spelled %20 on the way out",
			spec: "bridge{note=a+b}:opus",
			want: "bridge{note=a%20b}:opus",
		},
		{
			name: "the nickname survives canonicalisation",
			spec: "vertex-claude:claude-opus-5?b=2&a=1#my label",
			want: "vertex-claude:claude-opus-5?a=1&b=2#my label",
		},
		{
			name: "a nested tail is preserved verbatim",
			spec: "bridge{url=https://ws:26759}:vertex-claude:claude-opus-5?b=2&a=1",
			want: "bridge{url=https://ws:26759}:vertex-claude:claude-opus-5?a=1&b=2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sp, err := ParseSpec(tt.spec)
			if err != nil {
				t.Fatalf("ParseSpec(%q) = error %v", tt.spec, err)
			}
			got := sp.String()
			if got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
			// Idempotence is the property that makes the canonical form usable as
			// a map key: re-parsing must be a fixed point.
			sp2, err := ParseSpec(got)
			if err != nil {
				t.Fatalf("ParseSpec(canonical %q) = error %v", got, err)
			}
			if again := sp2.String(); again != got {
				t.Errorf("String() is not idempotent: %q -> %q", got, again)
			}
		})
	}
}

// TestSpecMatchesLegacyParsing is the no-behaviour-change guard. It reimplements
// the parsing that createProvider/createEmbedder/ShortLabel each did by hand
// before ParseSpec existed, and asserts the two agree on every spec shape we
// have actually seen in the wild — config.toml, the CustomModel table, the docs
// and the provider tests. A block cannot appear here by construction: nothing
// written before this file existed had one.
func TestSpecMatchesLegacyParsing(t *testing.T) {
	corpus := []string{
		// ~/.amplio/config.toml [run].llms and system tiers
		"vertex-claude:claude-opus-5?cache_ttl=1h&thinking.type=adaptive&output_config.effort=xhigh&thinking.display=summarized",
		"vertex-claude:claude-sonnet-5?cache_ttl=1h&thinking.type=adaptive&thinking.display=summarized",
		"vertex-claude:claude-opus-5?thinking.type=adaptive&thinking.display=summarized",
		// CustomModel rows
		"vertex-gemini:gemini-3.1-pro-preview?thinking_budget=2048&include_thoughts=true",
		"subprocess:corp_bridge?model=some-thought-summarizer",
		"subprocess:corp_bridge?model=exp-endpoint-7#candidate rc2",
		"openai:claude?base_url=http://localhost:4000/v1&profile=litellm",
		// docs/llm.md and the provider tests
		"openai:gpt-5.4-nano?profile=openai",
		"openai:qwen3.5?base_url=http://localhost:11434/v1&profile=ollama",
		"openai:nomic-embed-text:latest?base_url=http://localhost:11434/v1",
		"gemini:gemini-3.5-flash",
		"claude:claude-opus-5",
	}
	for _, spec := range corpus {
		t.Run(spec, func(t *testing.T) {
			// Legacy: SplitNickname, then Cut(":"), then Cut("?").
			legacyBase, legacyNick, _ := strings.Cut(spec, "#")
			legacyBase, legacyNick = strings.TrimSpace(legacyBase), strings.TrimSpace(legacyNick)
			legacyPrefix, legacyRest, ok := strings.Cut(legacyBase, ":")
			if !ok {
				t.Fatalf("corpus entry %q has no ':'", spec)
			}
			legacyModel, legacyRaw, _ := strings.Cut(legacyRest, "?")
			legacyArgs, err := url.ParseQuery(legacyRaw)
			if err != nil {
				t.Fatalf("legacy parse of %q: %v", spec, err)
			}

			sp, err := ParseSpec(spec)
			if err != nil {
				t.Fatalf("ParseSpec(%q) = error %v", spec, err)
			}
			model, args, err := sp.Model()
			if err != nil {
				t.Fatalf("Model() on %q = error %v", spec, err)
			}
			if sp.Provider != legacyPrefix {
				t.Errorf("provider = %q, legacy %q", sp.Provider, legacyPrefix)
			}
			if model != legacyModel {
				t.Errorf("model = %q, legacy %q", model, legacyModel)
			}
			if sp.Nickname != legacyNick {
				t.Errorf("nickname = %q, legacy %q", sp.Nickname, legacyNick)
			}
			if len(sp.Client) != 0 {
				t.Errorf("client block = %v, want empty for a legacy spec", sp.Client)
			}
			if got, want := args.Encode(), legacyArgs.Encode(); got != want {
				t.Errorf("args = %q, legacy %q", got, want)
			}
		})
	}
}

func TestSplitNicknameIsBlockAware(t *testing.T) {
	// A '#' inside a block is percent-encoded, so the first raw '#' is always the
	// nickname — but BaseSpec must not be fooled by an encoded one either.
	base, nick := SplitNickname("bridge{url=http://x/%23y}:opus#label")
	if want := "bridge{url=http://x/%23y}:opus"; base != want {
		t.Errorf("base = %q, want %q", base, want)
	}
	if nick != "label" {
		t.Errorf("nickname = %q, want %q", nick, "label")
	}
	// Malformed input still degrades to something usable rather than panicking:
	// the new-run form accepts arbitrary text.
	if got := BaseSpec("openai{unterminated:gpt-5#x"); got == "" {
		t.Errorf("BaseSpec on malformed input returned empty")
	}
}

func TestClientArgs(t *testing.T) {
	declared := map[string]bool{"base_url": true, "profile": true}

	t.Run("a declared key is accepted", func(t *testing.T) {
		client, err := ClientArgs(url.Values{"base_url": {"http://x"}}, declared)
		if err != nil {
			t.Fatalf("ClientArgs: %v", err)
		}
		if got := client.Get("base_url"); got != "http://x" {
			t.Errorf("base_url = %q", got)
		}
	})

	t.Run("an undeclared key is an error naming what is accepted", func(t *testing.T) {
		_, err := ClientArgs(url.Values{"bse_url": {"x"}}, declared)
		if err == nil {
			t.Fatal("want an error for an undeclared client arg")
		}
		for _, want := range []string{"bse_url", "base_url", "profile", "max_tokens"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not mention %q", err, want)
			}
		}
	})

	t.Run("max_tokens is universal: accepted with no declaration at all", func(t *testing.T) {
		client, err := ClientArgs(url.Values{"max_tokens": {"128"}}, nil)
		if err != nil {
			t.Fatalf("ClientArgs: %v", err)
		}
		got, err := MaxTokensArg(client, 65536)
		if err != nil {
			t.Fatalf("MaxTokensArg: %v", err)
		}
		if got != 128 {
			t.Errorf("max tokens = %d, want 128", got)
		}
		// Consumed: a provider must not find a second copy to disagree with.
		if _, left := client["max_tokens"]; left {
			t.Error("max_tokens survived in the client args")
		}
	})

	t.Run("a bad max_tokens fails fast", func(t *testing.T) {
		if _, err := MaxTokensArg(url.Values{"max_tokens": {"lots"}}, 10); err == nil {
			t.Error("want an error for a non-numeric max_tokens")
		}
		if _, err := MaxTokensArg(url.Values{"max_tokens": {"0"}}, 10); err == nil {
			t.Error("want an error for max_tokens=0")
		}
	})
}

// TestQueryIsNeverInspected pins the property that makes the two namespaces
// independent: nothing reads the query, so a key there that happens to share a
// name with a client arg is still the MODEL's, and reaches the model. Inspecting
// it — even only to reject it — is what re-creates the collision the block
// exists to remove.
func TestQueryIsNeverInspected(t *testing.T) {
	sp, err := ParseSpec("openai{base_url=http://ours}:gpt-5?base_url=http://theirs&profile=fast")
	if err != nil {
		t.Fatal(err)
	}
	client, err := ClientArgs(sp.Client, map[string]bool{"base_url": true, "profile": true})
	if err != nil {
		t.Fatalf("ClientArgs: %v", err)
	}
	if got := client.Get("base_url"); got != "http://ours" {
		t.Errorf("client base_url = %q, want the block's value", got)
	}
	_, args, err := sp.Model()
	if err != nil {
		t.Fatal(err)
	}
	if got := args.Get("base_url"); got != "http://theirs" {
		t.Errorf("model base_url = %q, want the query's value untouched", got)
	}
	if got := args.Get("profile"); got != "fast" {
		t.Errorf("model profile = %q, want it forwarded rather than claimed", got)
	}
}

func TestCacheProviders(t *testing.T) {
	var built int
	build := CacheProviders(func(spec string) (Provider, error) {
		built++
		if spec == "bad:model" {
			return nil, errTest
		}
		return &MockProvider{Model: spec}, nil
	})
	a, err := build("openai:m")
	if err != nil {
		t.Fatal(err)
	}
	b, err := build("openai:m")
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Error("the same spec should yield the same provider; construction mints credentials and pools connections")
	}
	if built != 1 {
		t.Errorf("built %d times, want 1", built)
	}
	if _, err := build("openai:other"); err != nil || built != 2 {
		t.Errorf("a different spec should be built: built=%d err=%v", built, err)
	}
	// Failures are not cached: a missing credential should be fixable without a
	// restart.
	for range 2 {
		if _, err := build("bad:model"); err == nil {
			t.Fatal("want an error")
		}
	}
	if built != 4 {
		t.Errorf("built %d times, want the failing spec retried", built)
	}
}

var errTest = fmt.Errorf("boom")

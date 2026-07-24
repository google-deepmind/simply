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

// Package cli describes optional external command-line tools the agent runs via
// its bash tool (code search, code outline, web search, …). amplio never links
// these: it probes for them on $PATH at runtime and, when present, injects a
// usage snippet into an agent's bootstrap so the LLM knows the command exists.
// Absent tools are silently omitted (the agent just won't use them), with a
// one-time operator warning at startup. This keeps amplio's core dependency-free
// while letting corp (1p) and OSS environments light up whatever is installed.
package cli

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

// Tool is one external CLI capability. Name is the exact command the agent
// types (a bare command like "ast-bro", or an absolute path) — it is at once
// what we probe, what the snippet shows, and what bash runs, so there is a
// single source of truth and no name/path divergence.
type Tool struct {
	Name        string // exact command, e.g. "web_search", "cs", "ast-bro"
	Snippet     string // usage description; may be multi-line with examples
	InstallHint string // operator-facing: how to install it / where it lives
}

// Available reports whether the command is runnable right now. exec.LookPath is
// the unified check: a Name containing a slash is tried directly, otherwise it
// is searched on $PATH — both verify the executable bit, matching how bash will
// resolve the command in the agent's subprocess.
func (t Tool) Available() bool {
	_, err := exec.LookPath(t.Name)
	return err == nil
}

// availability probes every tool concurrently and returns name → available.
// A LookPath stat over a cloud-backed $PATH entry (e.g. the corp BinFS dir
// amplio prepends) can run into many tens of ms cold; serializing 6 of these
// at startup is the difference between "snappy" and "occasionally hangs for a
// second". Goroutines are cheap and the tool list is small, so an unbounded
// fan-out is fine. Each goroutine writes a disjoint slice index — no mutex
// needed; the map is rebuilt sequentially after the join.
func availability(tools []Tool) map[string]bool {
	avail := make([]bool, len(tools))
	var wg sync.WaitGroup
	for i, t := range tools {
		wg.Add(1)
		go func(i int, t Tool) {
			defer wg.Done()
			avail[i] = t.Available()
		}(i, t)
	}
	wg.Wait()
	out := make(map[string]bool, len(tools))
	for i, t := range tools {
		out[t.Name] = avail[i]
	}
	return out
}

// Known tools. Snippets are intentionally our own (stable) and scoped to the
// flags we rely on, rather than each tool's full surface.
var (
	// RipGrep: fast content search.
	RipGrep = Tool{
		Name: "rg",
		Snippet: "Fast recursive content search, gitignore-aware. Prefer over grep for file contents.\n" +
			"Examples:\n" +
			"  rg -n 'CreateLinked' internal/\n" +
			"  rg -n -t go 'func .*Workspace'\n" +
			"  rg -l 'TODO'",
		InstallHint: "install ripgrep: apt install ripgrep | brew install ripgrep | cargo install ripgrep",
	}
	// Fd: fast, gitignore-aware file finder.
	Fd = Tool{
		Name: "fd",
		Snippet: "Fast file finder, gitignore-aware — friendlier and faster than find.\n" +
			"Examples:\n" +
			"  fd '\\.go$' internal/\n" +
			"  fd -t f -e md\n" +
			"  fd -H -t d node_modules",
		InstallHint: "install fd: brew install fd | cargo install fd-find, see https://github.com/sharkdp/fd",
	}
	// AstGrep: structural code search & rewrite (complements ast-bro's outline).
	AstGrep = Tool{
		Name: "ast-grep",
		Snippet: "Structural code search & rewrite via AST patterns (tree-sitter, 20+ langs) — succeeds where regex is fragile.\n" +
			"Examples:\n" +
			"  ast-grep -p 'console.log($$$A)' -l js\n" +
			"  ast-grep -p '$X && $X()' --rewrite '$X?.()' -l ts\n" +
			"  ast-grep -p 'func $N($$$) error' -l go --json\n" +
			"Prefer over regex for matching/refactoring code constructs; complements ast-bro (outline).",
		InstallHint: "install ast-grep: cargo install ast-grep | brew install ast-grep | npm i -g @ast-grep/cli, see https://github.com/ast-grep/ast-grep",
	}
	// CodeOutline: the OSS ast-bro AST navigator.
	CodeOutline = Tool{
		Name: "ast-bro",
		Snippet: "Fast AST-based code structure (signatures + line ranges, no bodies) — a token-cheap\n" +
			"alternative to reading whole files to learn their shape.\n" +
			"Examples:\n" +
			"  ast-bro map path/to/file.py      # outline a single file\n" +
			"  ast-bro digest path/to/dir       # compact structure of a directory\n" +
			"  ast-bro show file.py SymbolName  # print one symbol's source",
		InstallHint: "Install ast-bro: cargo install ast-bro | brew install aeroxy/tap/ast-bro, npm install broken on Linux, see https://github.com/aeroxy/ast-bro",
	}
)

// internalExtras holds corp-only tool entries appended by an init() in
// cli_internal.go (build tag: internal). Nil in OSS builds, where All()
// returns just the four OSS-friendly entries above.
var internalExtras []Tool

// All returns every known tool (used for the startup availability report).
func All() []Tool {
	return append([]Tool{RipGrep, Fd, AstGrep, CodeOutline}, internalExtras...)
}

// DefaultTools is the set general-purpose worker agents (standard, chatbot)
// declare. Ephemeral helper loops (critic, compaction) declare none, keeping
// their bounded context free of research-tool snippets.
func DefaultTools() []Tool { return All() }

// BindPaths prepends dirs to $PATH (highest priority first) so amplio's shipped
// 1p tools resolve by bare name in both our probes and the agent's bash
// subprocesses (which inherit our environment). Call once at startup, before
// launching agents. Nonexistent dirs are harmless; empty entries are skipped.
func BindPaths(dirs []string) {
	var prepend []string
	for _, d := range dirs {
		if d != "" {
			prepend = append(prepend, d)
		}
	}
	if len(prepend) == 0 {
		return
	}
	sep := string(os.PathListSeparator)
	joined := strings.Join(prepend, sep)
	if cur := os.Getenv("PATH"); cur != "" {
		joined += sep + cur
	}
	_ = os.Setenv("PATH", joined)
}

// ANSI colors for the status block (used only when writing to a TTY).
const (
	cReset  = "\x1b[0m"
	cGreen  = "\x1b[32m"
	cYellow = "\x1b[33m"
)

// PrintStatus writes a one-time, operator-facing block listing each tool's
// availability: a green ✓ for present tools and a yellow ✗ + install hint for
// missing ones. Color is used only when w is a TTY and NO_COLOR is unset.
func PrintStatus(w io.Writer, tools []Tool) {
	avail := availability(tools)
	color := isTTY(w) && os.Getenv("NO_COLOR") == ""
	paint := func(s, code string) string {
		if !color {
			return s
		}
		return code + s + cReset
	}
	width := 0
	for _, t := range tools {
		if n := len(t.Name); n > width {
			width = n
		}
	}
	fmt.Fprintln(w, "CLI tools available to agents (invoked via bash):")
	for _, t := range tools {
		pad := strings.Repeat(" ", width-len(t.Name))
		if avail[t.Name] {
			fmt.Fprintf(w, "  %s %s%s  available\n", paint("✓", cGreen), t.Name, pad)
		} else {
			fmt.Fprintf(w, "  %s %s%s  %s\n", paint("✗", cYellow), t.Name, pad,
				paint("not found — "+t.InstallHint, cYellow))
		}
	}
}

// isTTY reports whether w is a character device (terminal), so we only emit
// ANSI color when it will actually render.
func isTTY(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// BootstrapSnippet renders the system-prompt block for the currently-available
// subset of tools, or "" if none are available. Probed per call (concurrently),
// so a tool installed after server start is picked up by newly-spawned agents.
// Each block leads with the runnable command name (the single source of truth):
//
//	**<Name>**:
//	<Snippet>
func BootstrapSnippet(tools []Tool) string {
	avail := availability(tools)
	var blocks []string
	for _, t := range tools {
		if avail[t.Name] {
			blocks = append(blocks, "**"+t.Name+"**:\n"+t.Snippet)
		}
	}
	if len(blocks) == 0 {
		return ""
	}
	return "Useful CLI utilities (run them with the bash tool):\n\n" +
		strings.Join(blocks, "\n\n")
}

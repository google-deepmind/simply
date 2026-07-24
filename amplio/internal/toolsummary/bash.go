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

package toolsummary

import (
	"path"
	"regexp"
	"strings"
)

// BashSummary extracts a short {verb, target} description from a bash command
// for the chat chip. It is a best-effort, purely procedural heuristic (no LLM).
//
// Bash commands are dominated by scaffolding wrapped around one real action, so
// we: strip leading `cd`, `VAR=val` assignments, comment lines, and `echo "…"`
// status labels; peel `sudo`/`env`/`time`/`timeout <dur>`/`nohup`/`nice`/`xargs`
// wrappers and recurse into `bash -c '…'`; resolve `$VAR` tool paths from the
// assignments we saw; then map the leading program to a verb and pull out its
// most salient argument (search pattern, file, blaze target, query, duration…).
//
// Either return may be empty. verb is "bash" when nothing better is found, so
// the chip degrades to today's behavior. The UI decides how much of target to
// show.
func BashSummary(command string) (verb, target string) {
	env := map[string]string{}
	st := firstRealStatement(command, env)
	if st == "" {
		return "bash", ""
	}

	first := ""
	if fields := strings.Fields(st); len(fields) > 0 {
		first = baseName(fields[0])
	}
	if control[first] || strings.HasPrefix(st, "(") || strings.HasPrefix(st, "{") || funcDef.MatchString(st) {
		return "script", ""
	}

	seg0 := firstPipeSeg(st)
	// File write: leading cat/tee/printf/echo/: with a real stdout redirect.
	if leadingWrite.MatchString(st) && stdoutRedirect.MatchString(seg0) {
		return "write", redirectTarget(seg0, env)
	}

	prog, args := peel(seg0, env)
	if prog == "" {
		return "bash", ""
	}
	// A subshell/brace-group/control keyword surfacing only AFTER wrappers are
	// peeled (e.g. `time ( for … )`) is a script, not a program named "(".
	if prog == "(" || prog == "{" || control[prog] {
		return "script", ""
	}
	if a, ok := alias[prog]; ok {
		return a, extractTarget(prog, args)
	}
	if v, ok := verbMap[prog]; ok {
		return v, extractTarget(prog, args)
	}
	// An unresolved $VAR / ${VAR} program (no matching assignment) has no
	// meaningful name to show — degrade to plain "bash" rather than "run $VAR".
	if strings.HasPrefix(prog, "$") {
		return "bash", ""
	}
	if strings.HasSuffix(prog, ".sh") || strings.HasSuffix(prog, ".py") || strings.HasPrefix(prog, "./") {
		return "run", prog
	}
	return "run " + prog, extractTarget(prog, args)
}

// --- tables -----------------------------------------------------------------

var verbMap = map[string]string{
	"grep": "search", "rg": "search", "egrep": "search", "fgrep": "search", "ag": "search", "ack": "search",
	"find": "find", "fd": "find", "locate": "find",
	"sed": "sed", "awk": "awk",
	"cat": "read", "head": "read", "tail": "read", "bat": "read", "less": "read", "more": "read", "zcat": "read",
	"ls": "list", "tree": "list", "stat": "inspect", "file": "inspect", "du": "disk", "df": "disk",
	"wc": "count", "sort": "sort", "uniq": "dedup", "cut": "cut", "tr": "translate", "column": "format", "jq": "jq",
	"echo": "print", "printf": "print",
	"git": "git", "jj": "jj", "hg": "hg",
	"go": "go", "gofmt": "gofmt", "goimports": "goimports",
	"npm": "npm", "yarn": "yarn", "pnpm": "pnpm", "node": "node", "npx": "npx", "tsc": "tsc", "svelte-check": "svelte-check",
	"bazel":  "bazel",
	"python": "python", "python3": "python", "pip": "pip", "pytest": "pytest", "uv": "uv", "ruff": "ruff",
	"mkdir": "mkdir", "rm": "remove", "rmdir": "remove", "cp": "copy", "mv": "move", "touch": "touch",
	"chmod": "chmod", "chown": "chown", "ln": "symlink",
	"curl": "fetch", "wget": "fetch",
	"diff": "diff", "patch": "patch", "comm": "compare", "cmp": "compare",
	"make": "make", "cargo": "cargo", "docker": "docker", "kubectl": "kubectl", "gcloud": "gcloud",
	"sleep": "wait", "date": "date", "pwd": "pwd", "whoami": "whoami", "which": "which", "seq": "seq",
	"tar": "tar", "unzip": "unzip", "zip": "zip",
	"pgrep": "find proc", "pkill": "kill", "kill": "kill", "ps": "ps", "nproc": "nproc",
	"uptime": "uptime", "free": "mem", "lscpu": "cpu",
	"xargs": "xargs",
	// Environment-specific tool verbs are merged into this map by the internal
	// build's init() (see the internal overlay), so the OSS binary maps them
	// through the generic "run <prog>" fallback instead.
}

// alias maps recurring project CLIs (often invoked via a $VAR path) to a verb.
// Environment-specific CLI names are merged into this map by the internal build's
// init() (see the internal overlay), so they don't ship in the OSS binary.
// Empty by default.
var alias = map[string]string{}

var (
	wrapPlain = map[string]bool{"sudo": true, "env": true, "time": true, "command": true, "stdbuf": true, "exec": true, "\\": true, "builtin": true}
	wrapSkip1 = map[string]bool{"timeout": true, "nice": true, "nohup": true, "ionice": true, "setsid": true}
	control   = map[string]bool{"for": true, "while": true, "if": true, "case": true, "until": true, "function": true, "select": true, "do": true, "then": true, "{": true, "}": true, "((": true, "[[": true, "test": true, "[": true}
	// Public build/VCS/cloud CLIs. Environment-specific subcommand tools are added
	// to this set by the internal build's init() (see the internal overlay).
	subcmdTools = map[string]bool{"git": true, "jj": true, "hg": true, "go": true, "blaze": true, "bazel": true, "npm": true, "yarn": true, "pnpm": true, "cargo": true, "docker": true, "kubectl": true, "gcloud": true, "pip": true, "uv": true}
)

var (
	// Shell builtins that are pure scaffolding when they lead a statement (no
	// real action to summarize): variable declarations / shell options.
	declBuiltin = map[string]bool{"declare": true, "export": true, "local": true, "readonly": true, "typeset": true, "set": true, "shopt": true, "unset": true}
)

var (
	assignDef      = regexp.MustCompile(`^([A-Za-z_][A-Za-z0-9_]*)=(\S+)$`)
	assignLead     = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=`)
	varRef         = regexp.MustCompile(`^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$`)
	funcDef        = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*\s*\(\)`)
	leadingWrite   = regexp.MustCompile(`^\s*(cat|tee|printf|echo|:|>)`)
	stdoutRedirect = regexp.MustCompile(`(^|[^0-9&])>>?\s*[^\s|&;<>]`)
	numStart       = regexp.MustCompile(`^[0-9]`)
)

// --- helpers ----------------------------------------------------------------

// firstRealStatement splits command into top-level statements, records every
// VAR=value assignment into env (for later $VAR resolution), skips scaffolding
// (comments, pure assignments, cd/declare/export…, and leading label echoes),
// and returns the first statement that represents a real action ("" if none).
func firstRealStatement(command string, env map[string]string) string {
	var real []string
	for _, st := range splitStatements(command) {
		st = strings.TrimSpace(st)
		if st == "" || strings.HasPrefix(st, "#") {
			continue
		}
		// Simple VAR=value (single token): record for $VAR resolution below.
		if m := assignDef.FindStringSubmatch(st); m != nil {
			env[m[1]] = m[2]
			continue
		}
		// A PURE assignment statement — incl. VAR=$(cmd), VAR="multi word", or
		// several chained assignments — has nothing left after its leading
		// assignments, so it's setup: skip it. (An env-PREFIXED command like
		// `FOO=bar prog args` keeps a trailing command; peel strips the prefix.)
		if assignLead.MatchString(st) && stripLeadingAssigns(st) == "" {
			continue
		}
		if fields := strings.Fields(st); len(fields) > 0 {
			if b := baseName(fields[0]); b == "cd" || declBuiltin[b] {
				continue // cd / declare / export / local / set … are scaffolding
			}
		}
		real = append(real, st)
	}
	if len(real) == 0 {
		return ""
	}
	// Skip leading echo/printf "label" statements when a real command follows.
	i := 0
	for i < len(real)-1 && isLabelEcho(real[i]) {
		i++
	}
	return real[i]
}

// splitStatements breaks a command line into top-level statements at unquoted,
// unnested `;`, `&&`, `||`, and newlines. It tracks single/double quotes,
// backticks, and (…)/{…} nesting (which also covers $(...) and ${...}) so a
// separator inside a quoted string, subshell, or awk/python body does not split
// the statement.
func splitStatements(s string) []string {
	var out []string
	var b strings.Builder
	flush := func() {
		if st := strings.TrimSpace(b.String()); st != "" {
			out = append(out, st)
		}
		b.Reset()
	}
	var inS, inD, inB bool // single-quote, double-quote, backtick
	inComment := false
	depth := 0 // () and {} nesting
	rs := []rune(s)
	// wordStart reports whether position i begins a new word (so a `#` there
	// starts a comment), i.e. it's at the start or follows whitespace/separator.
	wordStart := func(i int) bool {
		if i == 0 {
			return true
		}
		switch rs[i-1] {
		case ' ', '\t', '\n', ';', '&', '|', '(', '{':
			return true
		}
		return false
	}
	for i := 0; i < len(rs); i++ {
		c := rs[i]
		switch {
		case inComment:
			// A comment runs to end-of-line; the newline is handled normally.
			if c == '\n' {
				inComment = false
				if depth == 0 {
					flush()
				} else {
					b.WriteRune(c)
				}
			}
		case inS:
			b.WriteRune(c)
			if c == '\'' {
				inS = false
			}
		case inD:
			b.WriteRune(c)
			if c == '"' {
				inD = false
			}
		case inB:
			b.WriteRune(c)
			if c == '`' {
				inB = false
			}
		case c == '\'':
			inS = true
			b.WriteRune(c)
		case c == '"':
			inD = true
			b.WriteRune(c)
		case c == '`':
			inB = true
			b.WriteRune(c)
		case c == '#' && wordStart(i):
			inComment = true
		case c == '(' || c == '{':
			depth++
			b.WriteRune(c)
		case c == ')' || c == '}':
			if depth > 0 {
				depth--
			}
			b.WriteRune(c)
		case c == '\n' || c == ';':
			if depth == 0 {
				flush()
			} else {
				b.WriteRune(c)
			}
		case (c == '&' || c == '|') && depth == 0 && i+1 < len(rs) && rs[i+1] == c:
			flush()
			i++ // consume the doubled && / ||
		default:
			b.WriteRune(c)
		}
	}
	flush()
	return out
}

func baseName(tok string) string {
	tok = strings.Trim(tok, `"'`)
	return path.Base(tok)
}

func isLabelEcho(st string) bool {
	if !strings.HasPrefix(strings.TrimSpace(st), "echo") && !strings.HasPrefix(strings.TrimSpace(st), "printf") {
		return false
	}
	return !strings.Contains(st, "$(") && !strings.Contains(st, "`")
}

func resolveTok(tok string, env map[string]string) string {
	if m := varRef.FindStringSubmatch(tok); m != nil {
		if v, ok := env[m[1]]; ok {
			return baseName(v)
		}
	}
	return baseName(tok)
}

// stripLeadingAssigns removes leading `VAR=value` assignments from a statement
// and returns whatever command remains ("" if the statement is pure
// assignments). The value may be a $(...)/`...` command substitution, a quoted
// string with spaces, or a bare token — so we scan character-by-character
// balancing parens/backticks/quotes rather than splitting on spaces.
func stripLeadingAssigns(s string) string {
	s = strings.TrimSpace(s)
	for {
		m := assignLead.FindString(s)
		if m == "" {
			return strings.TrimSpace(s)
		}
		i := len(m) // just past "VAR="
		depth := 0
		var inS, inD bool
		for i < len(s) {
			c := s[i]
			switch {
			case inS:
				if c == '\'' {
					inS = false
				}
			case inD:
				if c == '"' {
					inD = false
				}
			case c == '\'':
				inS = true
			case c == '"':
				inD = true
			case c == '(':
				depth++
			case c == ')':
				if depth > 0 {
					depth--
				}
			case c == ' ' || c == '\t':
				if depth == 0 {
					goto doneVal
				}
			}
			i++
		}
	doneVal:
		s = strings.TrimSpace(s[i:])
		if s == "" {
			return ""
		}
		// If what remains no longer starts with an assignment, it's the command.
	}
}

// dropTok removes the leading run of non-space/non-tab characters (one
// whitespace-delimited token) and returns the rest, preserving newlines.
func dropTok(s string) string {
	if i := strings.IndexAny(s, " \t"); i >= 0 {
		return s[i:]
	}
	return ""
}

// tokenize splits a string into shell-ish tokens, honoring single/double quotes
// and stripping them. Good enough for pulling out argument values.
func tokenize(s string) []string {
	var toks []string
	var b strings.Builder
	inS, inD := false, false
	flush := func() {
		if b.Len() > 0 {
			toks = append(toks, b.String())
			b.Reset()
		}
	}
	for _, r := range s {
		switch {
		case inS:
			if r == '\'' {
				inS = false
			} else {
				b.WriteRune(r)
			}
		case inD:
			if r == '"' {
				inD = false
			} else {
				b.WriteRune(r)
			}
		case r == '\'':
			inS = true
		case r == '"':
			inD = true
		case r == ' ' || r == '\t':
			flush()
		default:
			b.WriteRune(r)
		}
	}
	flush()
	return toks
}

// peel strips wrappers/assignments off a pipe segment and returns the leading
// program (basename, $VAR-resolved) plus its remaining argument tokens.
func peel(seg string, env map[string]string) (string, []string) {
	s := strings.TrimSpace(seg)
	for n := 0; n < 10; n++ {
		if s == "" {
			return "", nil
		}
		fields := strings.Fields(s)
		t0 := fields[0]
		if assignLead.MatchString(t0) {
			s = strings.TrimSpace(s[len(t0):])
			continue
		}
		b := resolveTok(t0, env)
		if wrapPlain[b] {
			s = strings.TrimSpace(s[len(t0):])
			continue
		}
		if wrapSkip1[b] {
			// Slice past the wrapper token, its flags, and one optional numeric
			// arg (e.g. `timeout 300`) using the ORIGINAL string so a wrapped
			// multiline `bash -c '…'` keeps its newlines and $(…) intact.
			rem := strings.TrimLeft(s[len(t0):], " \t")
			for strings.HasPrefix(rem, "-") {
				rem = strings.TrimLeft(dropTok(rem), " \t")
			}
			if numStart.MatchString(rem) {
				rem = strings.TrimLeft(dropTok(rem), " \t")
			}
			s = rem
			continue
		}
		if b == "xargs" {
			rest := fields[1:]
			j := 0
			for j < len(rest) && strings.HasPrefix(rest[j], "-") {
				j++
			}
			s = strings.Join(rest[j:], " ")
			if strings.TrimSpace(s) == "" {
				return "xargs", nil
			}
			continue
		}
		if (b == "bash" || b == "sh") && len(fields) >= 2 {
			if fields[1] == "-c" {
				if idx := strings.Index(s, "-c"); idx >= 0 {
					inner := strings.TrimSpace(s[idx+2:])
					inner = strings.Trim(inner, `'"`)
					// The -c string is a whole command line: run it through the
					// same scaffolding filter (skips its own cd/assignments) and
					// peel the first real segment.
					if real := firstRealStatement(inner, env); real != "" {
						return peel(firstPipeSeg(real), env)
					}
					return "", nil
				}
			}
			// `bash script.sh args` (no -c): the script IS the action. Resolve
			// its basename (via $VAR if needed) as the program to run.
			if !strings.HasPrefix(fields[1], "-") {
				return resolveTok(fields[1], env), stripRedirects(tokenize(strings.Join(fields[2:], " ")))
			}
		}
		return b, stripRedirects(tokenize(strings.TrimSpace(s[len(t0):])))
	}
	return "", nil
}

// stripRedirects drops redirection tokens (>, >>, 2>/dev/null, &>, and the file
// after a bare > / <) so they don't masquerade as the target argument.
func stripRedirects(args []string) []string {
	out := make([]string, 0, len(args))
	skipNext := false
	for _, a := range args {
		if skipNext {
			skipNext = false
			continue
		}
		if a == ">" || a == ">>" || a == "<" || a == "2>" || a == "&>" || a == "1>" {
			skipNext = true
			continue
		}
		if strings.ContainsAny(a, "<>") {
			continue
		}
		out = append(out, a)
	}
	return out
}

func firstNonFlag(args []string, takesVal map[string]bool) string {
	for i := 0; i < len(args); i++ {
		a := args[i]
		if takesVal[a] {
			i++
			continue
		}
		if strings.HasPrefix(a, "-") {
			continue
		}
		return a
	}
	return ""
}

func flagVal(args []string, names ...string) string {
	for i, a := range args {
		for _, n := range names {
			if a == n && i+1 < len(args) {
				return args[i+1]
			}
			if strings.HasPrefix(a, n+"=") {
				return strings.TrimPrefix(a, n+"=")
			}
		}
	}
	return ""
}

func shortenPath(p string) string {
	p = strings.Trim(p, `"'`)
	if p == "." || p == ".." || p == "./" || p == "~" || p == "" {
		return p
	}
	return path.Base(strings.TrimRight(p, "/"))
}

// firstPipeSeg returns s up to the first pipe that is not inside quotes, so a
// `|` inside a quoted grep pattern doesn't truncate the segment.
func firstPipeSeg(s string) string {
	inS, inD := false, false
	for i, r := range s {
		switch r {
		case '\'':
			if !inD {
				inS = !inS
			}
		case '"':
			if !inS {
				inD = !inD
			}
		case '|':
			if !inS && !inD {
				return s[:i]
			}
		}
	}
	return s
}

func redirectTarget(seg string, env map[string]string) string {
	m := stdoutRedirectCap.FindStringSubmatch(seg)
	if m == nil {
		return ""
	}
	tok := strings.Trim(m[1], `"'`)
	if v := varRef.FindStringSubmatch(tok); v != nil {
		if val, ok := env[v[1]]; ok {
			return shortenPath(val)
		}
	}
	return shortenPath(tok)
}

var stdoutRedirectCap = regexp.MustCompile(`(?:^|[^0-9&])>>?\s*([^\s|&;<>]+)`)

// extractTarget pulls the most salient argument for prog.
func extractTarget(prog string, args []string) string {
	switch prog {
	case "grep", "rg", "egrep", "fgrep", "ag", "ack":
		if p := flagVal(args, "-e", "--regexp"); p != "" {
			return p
		}
		return firstNonFlag(args, map[string]bool{"-e": true, "-m": true, "--include": true, "--exclude": true, "-A": true, "-B": true, "-C": true})
	case "find", "fd", "locate":
		if g := flagVal(args, "-name", "-iname"); g != "" {
			return g
		}
		return ""
	case "sed", "awk":
		// the file is the last token that looks like a path/file (script is first)
		for i := len(args) - 1; i >= 0; i-- {
			a := args[i]
			if !strings.HasPrefix(a, "-") && looksLikeFile(a) {
				return shortenPath(a)
			}
		}
		return ""
	case "cat", "head", "tail", "bat", "less", "more", "zcat", "wc", "file", "stat", "ls", "tree":
		f := firstNonFlag(args, map[string]bool{"-n": true, "-c": true, "--lines": true})
		if f != "" {
			return shortenPath(f)
		}
		return ""
	case "python", "python3", "node":
		f := firstNonFlag(args, nil)
		if f == "" || f == "-" {
			return "(inline)"
		}
		return shortenPath(f)
	case "sleep":
		if len(args) > 0 {
			return args[0] + "s"
		}
		return ""
	case "curl", "wget":
		u := firstNonFlag(args, map[string]bool{"-o": true, "-O": true, "-H": true, "-X": true, "-d": true, "--data": true})
		if m := urlHost.FindStringSubmatch(u); m != nil {
			return m[1]
		}
		return shortenPath(u)
	case "mkdir", "rm", "rmdir", "cp", "mv", "touch", "chmod", "chown", "ln":
		if p := firstNonFlag(args, map[string]bool{"-m": true}); p != "" {
			return shortenPath(p)
		}
		return ""
	case "web_search":
		return flagVal(args, "--query")
	}
	if subcmdTools[prog] {
		sub := firstNonFlag(args, nil)
		if prog == "blaze" || prog == "bazel" || prog == "go" {
			for _, a := range args {
				if strings.HasPrefix(a, "//") || strings.HasPrefix(a, "./") || strings.Contains(a, ":") || strings.Contains(a, "/") {
					if a == sub {
						continue
					}
					tgt := a
					if idx := strings.LastIndex(tgt, ":"); idx >= 0 {
						tgt = tgt[idx+1:]
					}
					tgt = path.Base(tgt)
					return strings.TrimSpace(sub + " " + tgt)
				}
			}
		}
		return sub
	}
	// generic / aliased CLI: first non-flag arg, unless it's an unresolved $VAR
	if nf := firstNonFlag(args, nil); nf != "" && !strings.HasPrefix(nf, "$") {
		return shortenPath(nf)
	}
	return ""
}

var urlHost = regexp.MustCompile(`^https?://([^/]+)`)

func looksLikeFile(a string) bool {
	if strings.Contains(a, "/") {
		return true
	}
	for _, ext := range []string{".py", ".go", ".ts", ".tsx", ".md", ".txt", ".json", ".tsv", ".csv", ".svelte", ".css", ".html", ".sh", ".yaml", ".yml", ".proto", ".sql"} {
		if strings.HasSuffix(a, ext) {
			return true
		}
	}
	return false
}

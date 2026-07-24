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

import "testing"

func TestBashSummary(t *testing.T) {
	cases := []struct {
		name       string
		cmd        string
		wantVerb   string
		wantTarget string
	}{
		// --- core verbs + targets ---
		{"grep", `grep -n "weight_dim_annotation" third_party/py/model.py`, "search", "weight_dim_annotation"},
		{"grep -e", `grep -rn -e "foo|bar" .`, "search", "foo|bar"},
		{"find name", `find . -type f -name '*.py'`, "find", "*.py"},
		{"cat read", `cat $AMPLIO_ARTIFACT_DIR/plan.md`, "read", "plan.md"},
		{"head read", `head -50 internal/server/chat.go`, "read", "chat.go"},
		{"sed file", `sed -n '1889,1960p' third_party/py/optax/alias.py`, "sed", "alias.py"},
		{"ls", `ls third_party/py`, "list", "py"},
		{"sleep", `sleep 90`, "wait", "90s"},
		{"python script", `python3 jj_edit_commit_race.py --phase commit`, "python", "jj_edit_commit_race.py"},
		{"jj subcommand", `jj describe -m "simply: WSD schedule"`, "jj", "describe"},
		{"git subcommand", `git status`, "git", "status"},
		{"curl host", `curl -sL "https://arxiv.org/abs/2509.14483"`, "fetch", "arxiv.org"},
		{"mkdir", `mkdir -p /tmp/cadam_research`, "mkdir", "cadam_research"},

		// bazel: subcommand + target basename (public build tool; the corp "blaze"
		// name is exercised in the build-specific test files).
		{"bazel test", `bazel test //foo/bar/baz:widget_test --test_arg=x`, "bazel", "test widget_test"},

		// --- scaffolding peeling ---
		{"cd prefix", `cd /home/me/repo && grep -n foo bar.py`, "search", "foo"},
		{"timeout wrapper", `timeout 290 bazel test //pkg:config_lib_test`, "bazel", "test config_lib_test"},
		{"echo label then grep", `echo "=== search ==="; grep -rl needle /home/me/repo/`, "search", "needle"},
		{"$VAR resolution", "TOOL=/usr/local/bin/searchtool\n$TOOL --query=oauth", "run searchtool", ""},
		{"nohup bash -c sleep", `nohup bash -c 'sleep 90; notify' & echo started`, "wait", "90s"},

		// --- file writes (heredoc / redirect) ---
		{"cat heredoc write", "cat > /tmp/launch_w3.sh << 'EOF'\necho hi\nEOF", "write", "launch_w3.sh"},
		{"cat append write", `cat >> notebook.md <<'EOF'`, "write", "notebook.md"},
		{"colon truncate write", "OUT=/tmp/wave4.txt; : > \"$OUT\"", "write", "wave4.txt"},

		// --- edge fixes ---
		{"stderr redirect not a write", `cat optimizers.py 2>/dev/null | head -120`, "read", "optimizers.py"},
		{"find path exclusion not target", `find . -maxdepth 2 -type f -not -path './.git/*'`, "find", ""},
		{"for loop is script", `for x in 1 2 3; do echo $x; done`, "script", ""},
		{"func def is script", `poll() { echo hi; }`, "script", ""},

		// --- quote/paren/comment-aware statement splitting ---
		// `;` inside an awk body must not split the statement into a fake `run if(...)`.
		{"awk body semicolons", `find . -name '*.py' | awk -F/ '{n=split($0,a,"/"); if(n<=6) print}'`, "find", "*.py"},
		// Apostrophe inside a # comment must not open a phantom quote.
		{"apostrophe in comment", "# it's fine\ngrep -n needle file.go", "search", "needle"},
		// A subshell surfacing after a wrapper is a script, not a program "(".
		{"time subshell is script", `time ( for n in a b; do echo $n; done )`, "script", ""},
		// setsid is a wrapper; the wrapped multiline bash -c keeps its structure.
		{"setsid wrapper", "setsid bash -c '\n  cd /x\n  grep -rn foo bar.py\n' &", "search", "foo"},
		// bash -c with a leading cd in the -c string peels to the real command.
		{"bash -c strips inner cd", `bash -c 'cd /repo && rg -n needle .'`, "search", "needle"},

		// --- fallbacks ---
		{"unknown cli named", `nvidia-smi`, "run nvidia-smi", ""},
		{"empty", ``, "bash", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, tg := BashSummary(tc.cmd)
			if v != tc.wantVerb || tg != tc.wantTarget {
				t.Errorf("BashSummary(%q)\n  got  verb=%q target=%q\n  want verb=%q target=%q",
					tc.cmd, v, tg, tc.wantVerb, tc.wantTarget)
			}
		})
	}
}

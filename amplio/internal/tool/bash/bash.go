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

// Package bash provides the bash command execution tool.
package bash

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"
	"unicode/utf8"

	"amplio/internal/config"
	"amplio/internal/tool"
	"amplio/internal/util"
)

// ArtifactDirPromptSnippet documents the per-run artifact scratch directory.
// Appended to the system prompt of agents whose bash tool is run/session bound.
const ArtifactDirPromptSnippet = "\n\n**ARTIFACT DIRECTORY**\n" +
	"You have a per-run private scratch directory at the absolute path in the " +
	"`$AMPLIO_ARTIFACT_DIR` environment variable — separate from your working " +
	"directory (the user's project). Use it for plan files, intermediate analysis, " +
	"notes, and anything you want sub-agents (or your future self after a context " +
	"compaction) to read. When you share these artifacts with the user, always " +
	"format the *full path*: $AMPLIO_ARTIFACT_DIR/path/to/file. The UI will linkify it " +
	"into a helpful link for the user."

const (
	DefaultTimeout         = 300 * time.Second // 5 min — many tools have long delays
	DefaultMaxOutputLength = 32_000            // ~800 lines of average-width code
)

type Params struct {
	Command         string  `json:"command" jsonschema:"required" jsonschema_description:"The bash command to execute"`
	TimeoutSeconds  float64 `json:"timeout,omitempty" jsonschema_description:"Timeout in seconds (default 300)"`
	MaxOutputLength int     `json:"max_output_length,omitempty" jsonschema_description:"Max characters in output (default 32000)"`
}

// New builds the bash tool for an agent rooted at cwd. runID/sessionID (when
// non-empty) are exported to every subprocess: AMPLIO_RUN_ID, AMPLIO_SESSION_ID,
// AMPLIO_ARTIFACT_DIR (the per-run scratch dir), and AMPLIO_NOTIFY (this binary's
// path, so a background script can `"$AMPLIO_NOTIFY" notify ...` back). Empty ids
// (ephemeral helper loops) skip the injection.
func New(cwd, runID, sessionID string) *tool.Tool {
	return &tool.Tool{
		Name:        "bash",
		Description: fmt.Sprintf("Execute a bash command. CWD=%q.", cwd),
		ParamType:   &Params{},
		Execute:     makeExecutor(cwd, agentEnv(runID, sessionID)),
	}
}

// agentEnv builds the AMPLIO_* additions for subprocesses, or nil when there's
// no run context (so the bash subprocess just inherits the parent env).
func agentEnv(runID, sessionID string) []string {
	if runID == "" {
		return nil
	}
	env := []string{
		config.EnvRunID + "=" + runID,
		config.EnvArtifactDir + "=" + config.ArtifactDir(runID), // also creates the dir
	}
	if sessionID != "" {
		env = append(env, config.EnvSessionID+"="+sessionID)
	}
	// Deprecated, kept because scripts and agents already written against it —
	// including every watcher currently running — must keep working. New agents
	// are taught `amplio-notify` instead, and this can go once runs predating it
	// have ended.
	if exe, err := os.Executable(); err == nil {
		env = append(env, config.EnvNotify+"="+exe)
	}
	// Both entry points arrive on PATH from one directory: `amplio-notify` (what
	// the prompt teaches) and `amplio` (what the task-manager skill teaches). The
	// directory holds only those two symlinks, so prepending cannot shadow
	// anything else on the agent's PATH.
	if dir := config.ShimDir(); dirHasShim(dir) {
		env = append(env, "PATH="+dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	return env
}

func dirHasShim(dir string) bool {
	_, err := os.Lstat(filepath.Join(dir, config.NotifyShimName))
	return err == nil
}

func makeExecutor(cwd string, extraEnv []string) tool.Executor {
	return func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
		params, errResult := tool.ParseArgs[Params](args)
		if errResult != nil {
			return errResult, nil
		}

		timeout := DefaultTimeout
		if params.TimeoutSeconds > 0 {
			timeout = time.Duration(params.TimeoutSeconds * float64(time.Second))
		}
		maxOutput := DefaultMaxOutputLength
		if params.MaxOutputLength > 0 {
			maxOutput = params.MaxOutputLength
		}

		output := execute(ctx, params.Command, cwd, extraEnv, timeout, maxOutput)
		return &tool.Result{Content: output}, nil
	}
}

func execute(ctx context.Context, command, cwd string, extraEnv []string, timeout time.Duration, maxOutput int) string {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Use temp files for stdout/stderr (not pipes) so background
	// subprocesses (trailing &) don't block us on EOF.
	stdoutF, err := os.CreateTemp("", "amplio-bash-stdout-*")
	if err != nil {
		return fmt.Sprintf("Error: could not create temp file for stdout: %s", err)
	}
	stderrF, err := os.CreateTemp("", "amplio-bash-stderr-*")
	if err != nil {
		stdoutF.Close()
		os.Remove(stdoutF.Name())
		return fmt.Sprintf("Error: could not create temp file for stderr: %s", err)
	}
	defer func() {
		stdoutF.Close()
		stderrF.Close()
		os.Remove(stdoutF.Name())
		os.Remove(stderrF.Name())
	}()

	cmd := exec.CommandContext(ctx, "bash", "-c", command)
	cmd.Dir = cwd
	if len(extraEnv) > 0 {
		cmd.Env = append(os.Environ(), extraEnv...)
	}
	cmd.Stdin = nil // EOF on stdin so interactive prompts fail fast
	cmd.Stdout = stdoutF
	cmd.Stderr = stderrF
	// ...and no controlling terminal, so prompts that bypass stdin by opening
	// /dev/tty fail too instead of hanging on (and writing to) the terminal that
	// runs `amplio serve`. See detachTerminal.
	detachTerminal(cmd)
	// On timeout, take the whole process group with it: killing only the shell
	// leaves its children running, which is how a hung command becomes an orphan.
	cmd.Cancel = func() error { return killGroup(cmd) }

	start := time.Now()
	err = cmd.Run()
	elapsed := time.Since(start)

	timedOut := ctx.Err() != nil
	exitCode := 0
	var launchErr error // a failure to RUN bash (not a non-zero exit of the command)
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if timedOut {
			exitCode = -1
		} else {
			// fork/exec failure (bash missing, resource limit, …) — NOT a command
			// exit-1. Keep the error so the status line can distinguish it from a
			// genuine `exit 1`.
			exitCode = -1
			launchErr = err
		}
	}

	// Read output from temp files, capped per stream. The final result is
	// truncated to maxOutput anyway, but reading the WHOLE file first would let a
	// runaway producer (`yes`, `cat huge.bin`) buffer gigabytes into memory
	// before we discard ~all of it. Cap each stream's read at maxOutput so peak
	// memory is bounded (~2*maxOutput worst case), then let formatOutput do the
	// final head+tail truncation.
	stdoutF.Seek(0, 0) //nolint:errcheck
	stderrF.Seek(0, 0) //nolint:errcheck
	stdout := readCapped(stdoutF, maxOutput)
	stderr := readCapped(stderrF, maxOutput)

	status := statusLine(timedOut, timeout, exitCode, elapsed, launchErr)
	return formatOutput(stdout, stderr, status, maxOutput)
}

// readCapped reads up to limit bytes from f. If the file is larger, it appends a
// note with the true size so the agent knows output was dropped at the read
// boundary (not just the later display truncation). A negative/zero limit reads
// nothing meaningful, but maxOutput is always positive here.
func readCapped(f *os.File, limit int) string {
	buf := make([]byte, limit)
	n, _ := io.ReadFull(f, buf)
	out := string(buf[:n])
	// If there's still more to read, the output exceeded the cap.
	if extra, _ := f.Seek(0, io.SeekEnd); extra > int64(n) {
		out += fmt.Sprintf("\n[...output truncated at %d bytes; %d total...]", limit, extra)
	}
	return out
}

// statusLine builds the single trailing summary line that reports how the
// command ended, the wall-clock finish time, and the elapsed duration. It's the
// one place the agent learns the current time from a bash call (the eventloop no
// longer injects an ambient clock), so it always carries an absolute timestamp.
func statusLine(timedOut bool, timeout time.Duration, exitCode int, elapsed time.Duration, launchErr error) string {
	now := util.FormatLocalISO(time.Now())
	secs := elapsed.Seconds()
	switch {
	case timedOut:
		// No elapsed: it's just the timeout plus jitter, carrying no information.
		return fmt.Sprintf("Timed out after %.2fs at %s", timeout.Seconds(), now)
	case launchErr != nil:
		// bash itself failed to run — distinct from a command exit code.
		return fmt.Sprintf("Failed to run command at %s (elapsed=%.2fs): %s", now, secs, launchErr)
	case exitCode == 0:
		return fmt.Sprintf("Finished normally at %s (elapsed=%.2fs)", now, secs)
	default:
		return fmt.Sprintf("Finished with return code %d at %s (elapsed=%.2fs)", exitCode, now, secs)
	}
}

// formatOutput assembles the result and caps the WHOLE thing (status line
// included) at maxOutput. Truncation keeps head+tail, so the trailing status
// line survives even when the body is large.
func formatOutput(stdout, stderr, status string, maxOutput int) string {
	var parts []string
	if stdout != "" {
		parts = append(parts, "STDOUT:\n"+stdout)
	}
	if stderr != "" {
		parts = append(parts, "STDERR:\n"+stderr)
	}
	parts = append(parts, "STATUS:\n"+status)

	text := joinParts(parts)
	if len(text) > maxOutput {
		text = truncateText(text, maxOutput)
	}
	return text
}

func joinParts(parts []string) string {
	result := ""
	for i, p := range parts {
		if i > 0 {
			result += "\n\n"
		}
		result += p
	}
	return result
}

func truncateText(text string, maxLength int) string {
	if len(text) <= maxLength {
		return text
	}
	marker := "\n[...truncated...]\n"
	// If the marker alone meets/exceeds the budget there's no room for content;
	// return just the marker (trimmed) rather than overshooting maxLength.
	if len(marker) >= maxLength {
		return truncRunesEnd(marker, maxLength)
	}
	half := (maxLength - len(marker)) / 2
	// Back each cut off to a UTF-8 boundary so we never emit a split rune. This
	// only ever shrinks the slices, so the total stays within maxLength.
	head := truncRunesEnd(text, half)
	tail := truncRunesStart(text, half)
	return head + marker + tail
}

// truncRunesEnd returns the longest valid-UTF-8 prefix of s that is at most n
// bytes (backing off the tail if a cut would land mid-rune).
func truncRunesEnd(s string, n int) string {
	if n >= len(s) {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// truncRunesStart returns the longest valid-UTF-8 suffix of s that is at most n
// bytes (advancing the head off a mid-rune cut).
func truncRunesStart(s string, n int) string {
	if n >= len(s) {
		return s
	}
	start := len(s) - n
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return s[start:]
}

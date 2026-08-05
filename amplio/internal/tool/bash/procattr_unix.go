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

//go:build unix

package bash

import (
	"os/exec"
	"syscall"
)

// detachTerminal starts the command in its own session, so it has NO controlling
// terminal.
//
// Stdin is already /dev/null, which makes stdin-reading prompts fail fast — but
// that is not how CLIs usually prompt. The conventional way to ask a question
// even when stdin is redirected is to open /dev/tty, and a child inherits the
// controlling terminal of whatever started `amplio serve`. The result: a CLI
// asking for confirmation writes "Continue? [y/n]" into the OPERATOR'S terminal,
// reads nothing, loops on its own re-prompt, and hangs forever —
// observed as multi-day zombie processes. It can also swallow keystrokes the
// operator types into that window.
//
// With no controlling terminal, opening /dev/tty fails with ENXIO and those
// tools take their non-interactive path, which is what Stdin=/dev/null was
// meant to achieve in the first place.
//
// Setsid also makes the child a process-group leader, which is what lets a
// timeout kill the whole tree instead of just the shell (see killGroup).
func detachTerminal(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}

// killGroup kills the command's entire process group. Used only on timeout or
// cancellation: a bare Process.Kill() reaps the shell and leaves whatever it
// spawned running, which is how a hung command turns into an orphan.
//
// NOT used on normal completion — backgrounding a watcher that outlives the
// call is a supported pattern (see the $AMPLIO_NOTIFY workflow).
func killGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	// Negative pid = the whole group, which exists because of Setsid above.
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
		return cmd.Process.Kill() // group gone already; fall back to the leader
	}
	return nil
}

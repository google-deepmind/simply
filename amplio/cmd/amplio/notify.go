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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"amplio/internal/config"

	"github.com/spf13/cobra"
)

// notify exit codes are stable so background scripts can distinguish failure
// modes (usage vs unreachable vs server-refused).
const (
	notifyExitUsage   = 1 // bad args / missing env
	notifyExitUnreach = 2 // no server / dial failed
	notifyExitRefused = 3 // server returned non-2xx
)

// codedError carries a specific process exit code (see main's Execute handler).
type codedError struct {
	code int
	msg  string
}

func (e *codedError) Error() string { return e.msg }

func usageErr(m string) error   { return &codedError{notifyExitUsage, m} }
func unreachErr(m string) error { return &codedError{notifyExitUnreach, m} }
func refusedErr(m string) error { return &codedError{notifyExitRefused, m} }

func notifyCmd() *cobra.Command {
	var from, session string
	cmd := &cobra.Command{
		Use:   "notify [--session=ID] <message | ->",
		Short: "Send a message from a background script to a running agent session",
		Long: "Deliver an environment notification to a running amplio agent session. " +
			"Intended for background scripts an agent spawns via the bash tool. The " +
			"target session defaults to $AMPLIO_SESSION_ID (the spawning agent); use " +
			"--session=ID to target another. Use '-' as the message to read from stdin.\n\n" +
			"Avoid uninformative heartbeats and unbounded notification loops: the " +
			"server caps environment notifications per session step, and beyond it " +
			"refuses with exit code 3 and 'env_notice_capped' on stderr — match that " +
			"to stop a loop.",
		// Exactly one positional: the message. The target used to be an optional
		// FIRST positional, which meant the meaning of an argument depended on how
		// many there were — so `notify done building` (someone forgot the quotes)
		// silently addressed a session named "done". A stray argument is now an
		// error, and the target is named.
		Args: func(_ *cobra.Command, args []string) error {
			switch {
			case len(args) == 1:
				return nil
			case len(args) > 1:
				return fmt.Errorf("notify takes one message (got %d arguments); "+
					"quote the message, and use --session=<id> to target another session", len(args))
			default:
				return errors.New("notify needs a message (or '-' to read it from stdin)")
			}
		},
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return executeNotify(cmd.Context(), args[0], session, from)
		},
	}
	cmd.Flags().StringVar(&from, "from", "", "source label for the message (default: environment)")
	cmd.Flags().StringVar(&session, "session", "",
		"target session id (default: $AMPLIO_SESSION_ID, the spawning agent)")
	return cmd
}

func executeNotify(ctx context.Context, msgArg, session, from string) error {
	runID := os.Getenv(config.EnvRunID)
	if runID == "" {
		return usageErr("$AMPLIO_RUN_ID not set — notify must be run from a script spawned by an agent's bash tool")
	}
	sid := session
	if sid == "" {
		sid = os.Getenv(config.EnvSessionID)
	}
	if sid == "" {
		return usageErr("no target session: pass --session=<id>, or set $AMPLIO_SESSION_ID")
	}
	msg, err := readNotifyMessage(msgArg)
	if err != nil {
		return usageErr(err.Error())
	}
	if msg == "" {
		return usageErr("empty message")
	}

	info, err := readServerInfo(config.DataDir())
	if err != nil {
		return unreachErr(fmt.Sprintf("no running server for %s (is `amplio serve` up?): %v", config.DataDir(), err))
	}
	base := info.Addr
	if base == "" {
		base = info.URL
	}

	// Stamp the caller's parent pid onto the sender so a revived agent can
	// identify (and kill) the background script that notified it. os.Getppid() is
	// the process that invoked `amplio notify` — for any "launched a watcher
	// script/loop" pattern this is the long-running job the agent can kill (it
	// matches the `$!` the launch printed). The exception is an inline
	// fire-and-forget where notify is the LAST command in a backgrounded subshell
	// (`( sleep 10; notify ) &`): bash exec-optimizes it, so notify replaces the
	// subshell and is reparented to init (ppid=1). Skip the annotation there — it
	// would be both wrong and useless (there's no persistent script to kill).
	sender := from
	if sender == "" {
		sender = "environment"
	}
	if ppid := os.Getppid(); ppid > 1 {
		sender = fmt.Sprintf("%s (pid=%d)", sender, ppid)
	}
	body, _ := json.Marshal(map[string]string{"content": msg, "sender": sender})
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	url := fmt.Sprintf("%s/api/runs/%s/sessions/%s/notify", base, runID, sid)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+info.Token)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return unreachErr(fmt.Sprintf("contact server at %s: %v", base, err))
	}
	defer resp.Body.Close() //nolint:errcheck
	if resp.StatusCode != http.StatusAccepted {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
		return refusedErr(fmt.Sprintf("server returned %s: %s", resp.Status, strings.TrimSpace(string(b))))
	}
	fmt.Fprintf(os.Stderr, "notified %s (%d bytes)\n", sid, len(msg))
	return nil
}

// readNotifyMessage returns the message, reading stdin when arg is "-".
func readNotifyMessage(arg string) (string, error) {
	if arg != "-" {
		return arg, nil
	}
	buf, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", fmt.Errorf("read message from stdin: %w", err)
	}
	return strings.TrimRight(string(buf), "\n"), nil
}

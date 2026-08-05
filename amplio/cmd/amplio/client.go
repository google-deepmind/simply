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
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"amplio/internal/config"

	"github.com/spf13/cobra"
)

// clientCmd groups operations that talk to a RUNNING `amplio serve` for the
// current data directory. The server is auto-discovered via the data dir's
// server.json (PID + loopback address + auth token); no flag wrangling is
// needed in the common case. These commands are usable both by humans and by
// agents inside a run (the bash tool inherits $AMPLIO_DATA_DIR so a sub-agent
// shells out and reaches the SAME server its parent runs in).
//
// For headless / no-server execution see `amplio headless` instead.
func clientCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "client",
		Short: "Talk to a running amplio server (auto-discovered via server.json)",
		Long: "Client-side operations against the amplio server running in the current" +
			" data directory. The target server is auto-discovered from server.json;" +
			" all subcommands are scoped to that single instance. Useful both for" +
			" humans (quick status checks, cancel/restart from a terminal) and for" +
			" agents-managing-other-runs from inside a run via the bash tool.",
	}
	cmd.AddCommand(
		clientSubmitCmd(),
		clientCancelCmd(),
		clientRestartCmd(),
		clientStatusCmd(),
		clientListCmd(),
	)
	return cmd
}

// --- Submit ----------------------------------------------------------------

func clientSubmitCmd() *cobra.Command {
	var (
		task      string
		title     string
		workspace string
		agentType string
		llmSpec   string
	)
	cmd := &cobra.Command{
		Use:   "submit [task]",
		Short: "Submit a new run to the server",
		Long: "Hand a task to the server running in this data directory. The run" +
			" executes inside that server; this command returns immediately with" +
			" the run id.",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				task = args[0]
			}
			if task == "" {
				return fmt.Errorf("a task is required (positional arg or --task)")
			}
			body := map[string]string{
				"task":      task,
				"title":     title,
				"workspace": workspace,
				"agent":     agentType,
				"llm":       llmSpec,
			}
			var out struct {
				RunID string `json:"run_id"`
			}
			info, raw, err := clientDo(cmd.Context(), http.MethodPost, "/api/runs", body)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(raw, &out); err != nil {
				return fmt.Errorf("parse server response: %w", err)
			}
			fmt.Printf("run started: %s\n  %s/?token=%s\n", out.RunID, info.URL, info.Token)
			return nil
		},
	}
	cmd.Flags().StringVar(&task, "task", "", "Task description (or pass as a positional arg)")
	cmd.Flags().StringVar(&title, "title", "", "Run title (default: server auto-generates from the task). Handy to tag/compare runs, e.g. a model-name suffix.")
	cmd.Flags().StringVar(&workspace, "workspace", "", "Working directory (default: server-side run.workspace)")
	cmd.Flags().StringVar(&agentType, "agent", "", "Agent type (default: server-side run.agent_type)")
	cmd.Flags().StringVar(&llmSpec, "llm", "", "Agent LLM spec (default: server-side run.llm)")
	return cmd
}

// --- Cancel ----------------------------------------------------------------

func clientCancelCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "cancel <run-id>",
		Short: "Cancel a run (cascades to all sessions)",
		Long: "Request cancellation of a run. The server cascades to every session" +
			" under the run; this command returns once the request is accepted, NOT" +
			" once cancellation has actually propagated.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			runID := args[0]
			if _, _, err := clientDo(cmd.Context(), http.MethodPost, "/api/runs/"+runID+"/cancel", nil); err != nil {
				return err
			}
			fmt.Printf("cancellation requested for %s\n", runID)
			return nil
		},
	}
}

// --- Restart ---------------------------------------------------------------

func clientRestartCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "restart <run-id>",
		Short: "Revive a crashed run (idempotent for already-running runs)",
		Long: "Revive a run's active spine — the same path the server takes at boot" +
			" for crashed runs. Idempotent: a run already at rest returns" +
			" {revived: 0}.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			runID := args[0]
			_, raw, err := clientDo(cmd.Context(), http.MethodPost, "/api/runs/"+runID+"/restart", nil)
			if err != nil {
				return err
			}
			var out struct {
				Revived int `json:"revived"`
			}
			_ = json.Unmarshal(raw, &out)
			fmt.Printf("restart requested for %s: %d session(s) revived\n", runID, out.Revived)
			return nil
		},
	}
}

// --- Status ----------------------------------------------------------------

func clientStatusCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "status <run-id>",
		Short: "Print a run's configuration and current status",
		Long: "Fetch a run's full detail from the server: id, title, task, workspace," +
			" LLM, agent type, system tiers, created/updated timestamps, and all" +
			" sessions with their statuses + current step. --json emits the raw" +
			" RunDetail JSON for scripting / agent consumption.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			runID := args[0]
			_, raw, err := clientDo(cmd.Context(), http.MethodGet, "/api/runs/"+runID, nil)
			if err != nil {
				return err
			}
			if asJSON {
				fmt.Print(string(raw))
				if len(raw) > 0 && raw[len(raw)-1] != '\n' {
					fmt.Println()
				}
				return nil
			}
			return printRunDetail(raw)
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "Emit the raw RunDetail JSON (scripting / agent consumption)")
	return cmd
}

// printRunDetail renders a runDetail JSON blob as an aligned key/value block
// plus a small sessions table. Mirrors the overview page's metadata card.
func printRunDetail(raw []byte) error {
	var d struct {
		RunID         string    `json:"run_id"`
		Title         string    `json:"title"`
		Task          string    `json:"task"`
		Workspace     string    `json:"workspace"`
		LLM           string    `json:"llm"`
		AgentType     string    `json:"agent_type"`
		SystemLLMHQ   string    `json:"system_llm_hq"`
		SystemLLMFast string    `json:"system_llm_fast"`
		Starred       bool      `json:"starred"`
		Grade         *string   `json:"grade"`
		ReportGrade   *string   `json:"report_grade"`
		Archived      bool      `json:"archived"`
		CreatedAt     time.Time `json:"created_at"`
		UpdatedAt     time.Time `json:"updated_at"`
		Sessions      []struct {
			SessionID   string `json:"session_id"`
			ParentID    string `json:"parent_id"`
			AgentType   string `json:"agent_type"`
			Status      string `json:"status"`
			CurrentStep int    `json:"current_step"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return fmt.Errorf("parse run detail: %w", err)
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "run_id\t%s\n", d.RunID)
	if d.Title != "" {
		fmt.Fprintf(w, "title\t%s\n", d.Title)
	}
	if d.Task != "" {
		fmt.Fprintf(w, "task\t%s\n", firstLine(d.Task))
	}
	fmt.Fprintf(w, "workspace\t%s\n", d.Workspace)
	fmt.Fprintf(w, "model\t%s\n", d.LLM)
	fmt.Fprintf(w, "agent type\t%s\n", d.AgentType)
	if d.SystemLLMHQ != "" {
		fmt.Fprintf(w, "system (hq)\t%s\n", d.SystemLLMHQ)
	}
	if d.SystemLLMFast != "" {
		fmt.Fprintf(w, "system (fast)\t%s\n", d.SystemLLMFast)
	}
	fmt.Fprintf(w, "created\t%s\n", d.CreatedAt.Local().Format(time.RFC3339))
	if !d.UpdatedAt.IsZero() && !d.UpdatedAt.Equal(d.CreatedAt) {
		fmt.Fprintf(w, "last edit\t%s\n", d.UpdatedAt.Local().Format(time.RFC3339))
	}
	if d.Starred {
		fmt.Fprintf(w, "starred\ttrue\n")
	}
	// Effective grade: the human grade overrides the cached critic grade.
	if d.Grade != nil {
		fmt.Fprintf(w, "grade\t%s\n", *d.Grade)
	} else if d.ReportGrade != nil {
		fmt.Fprintf(w, "grade\t%s (critic)\n", *d.ReportGrade)
	}
	if d.Archived {
		fmt.Fprintf(w, "archived\ttrue\n")
	}
	if err := w.Flush(); err != nil {
		return err
	}
	if len(d.Sessions) == 0 {
		return nil
	}
	fmt.Println()
	fmt.Println("sessions:")
	sw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(sw, "  SESSION\tAGENT\tSTATUS\tSTEP\tPARENT")
	for _, s := range d.Sessions {
		parent := s.ParentID
		if parent == "" {
			parent = "-"
		}
		fmt.Fprintf(sw, "  %s\t%s\t%s\t%d\t%s\n",
			s.SessionID, s.AgentType, s.Status, s.CurrentStep, parent)
	}
	return sw.Flush()
}

// --- List ------------------------------------------------------------------

func clientListCmd() *cobra.Command {
	var (
		asJSON   bool
		archived bool
		limit    int
	)
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List runs on the server (most-recent first)",
		Long: "Print a table of the most-recent runs known to the local server. Shows" +
			" one page (default 50, --limit to change, server caps at 200); the web UI" +
			" pages through the rest. --archived includes archived runs; --json emits" +
			" the runSummary[] JSON array for scripting.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			q := url.Values{}
			if archived {
				q.Set("archived", "1")
			}
			if limit > 0 {
				q.Set("limit", strconv.Itoa(limit))
			}
			path := "/api/runs"
			if enc := q.Encode(); enc != "" {
				path += "?" + enc
			}
			_, raw, err := clientDo(cmd.Context(), http.MethodGet, path, nil)
			if err != nil {
				return err
			}
			var page struct {
				Runs    json.RawMessage `json:"runs"`
				HasMore bool            `json:"has_more"`
			}
			if err := json.Unmarshal(raw, &page); err != nil {
				return fmt.Errorf("parse runs page: %w", err)
			}
			// --json emits the flat runSummary[] array (the documented contract),
			// not the pagination envelope — scripts wanting cursors hit the API.
			if asJSON {
				out := page.Runs
				if len(out) == 0 {
					out = json.RawMessage("[]")
				}
				fmt.Println(string(out))
				return nil
			}
			return printRunSummaries(page.Runs, page.HasMore)
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "Emit the runSummary[] JSON array")
	cmd.Flags().BoolVar(&archived, "archived", false, "Include archived runs")
	cmd.Flags().IntVar(&limit, "limit", 50, "Max runs to show (server caps at 200); the web UI pages through the rest")
	return cmd
}

// printRunSummaries renders one page of the dashboard's run list as an aligned
// table. hasMore appends a hint that more runs exist beyond this page.
func printRunSummaries(raw []byte, hasMore bool) error {
	var runs []struct {
		RunID       string  `json:"run_id"`
		Title       string  `json:"title"`
		Task        string  `json:"task"`
		LLM         string  `json:"llm"`
		RootStatus  string  `json:"root_status"`
		RootStep    int     `json:"root_step"`
		Starred     bool    `json:"starred"`
		Grade       *string `json:"grade"`
		ReportGrade *string `json:"report_grade"`
		Archived    bool    `json:"archived"`
	}
	if err := json.Unmarshal(raw, &runs); err != nil {
		return fmt.Errorf("parse runs: %w", err)
	}
	if len(runs) == 0 {
		fmt.Println("(no runs)")
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "RUN_ID\tSTATUS\tSTEP\tFLAGS\tTITLE")
	for _, r := range runs {
		title := r.Title
		if title == "" {
			title = firstLine(r.Task)
		}
		flags := ""
		if r.Starred {
			flags += "★"
		}
		// Effective grade abbreviation; lowercase = critic fallback, uppercase =
		// human-set, so the flag column hints at the source at a glance.
		if g := gradeFlag(r.Grade, r.ReportGrade); g != "" {
			flags += g
		}
		if r.Archived {
			flags += "📁"
		}
		if flags == "" {
			flags = "-"
		}
		fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%s\n",
			r.RunID, r.RootStatus, r.RootStep, flags, title)
	}
	if err := w.Flush(); err != nil {
		return err
	}
	if hasMore {
		fmt.Printf("\n… more runs beyond this page (showing %d; use --limit, or the web UI to page).\n", len(runs))
	}
	return nil
}

// --- Shared HTTP helper ----------------------------------------------------

// clientDo wraps the boilerplate of every client request: load server.json,
// build URL + auth, send, check status, return the raw response. Returns the
// serverInfo too so callers can render the user-facing URL in success messages
// (banner URL, not the loopback used for the request itself — the banner host
// may be a non-loopback FQDN that's unreachable from this process).
func clientDo(ctx context.Context, method, path string, body any) (serverInfo, []byte, error) {
	dataDir := config.DataDir()
	info, err := readServerInfo(dataDir)
	if err != nil {
		return serverInfo{}, nil, fmt.Errorf("no running server for %s (start one with `amplio serve`): %w", dataDir, err)
	}
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return info, nil, err
		}
		reader = bytes.NewReader(b)
	}
	// Talk over loopback (Addr); the banner URL may go through a reverse proxy
	// and be unreachable / time out on auth.
	base := info.Addr
	if base == "" {
		base = info.URL
	}
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, base+path, reader)
	if err != nil {
		return info, nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Authorization", "Bearer "+info.Token)
	resp, err := loopbackHTTPClient(base).Do(req)
	if err != nil {
		return info, nil, fmt.Errorf("contact server at %s (is it still running?): %w", base, err)
	}
	defer resp.Body.Close()                                     //nolint:errcheck
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20)) //nolint:errcheck
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return info, respBody, fmt.Errorf("server returned %s: %s", resp.Status, string(respBody))
	}
	return info, respBody, nil
}

// loopbackHTTPClient returns an HTTP client suitable for talking to the local
// amplio server. For HTTPS-to-loopback (mkcert / self-signed dev certs on
// 127.0.0.1 / ::1 / localhost), TLS verification is skipped: the connection
// is local-only, the server is the same user's own process, and demanding
// the client side trust the mkcert CA would require finding and loading it.
// For HTTP, this is just http.DefaultClient. For HTTPS to non-loopback, full
// verification stays on.
func loopbackHTTPClient(serverURL string) *http.Client {
	u, err := url.Parse(serverURL)
	if err != nil || u.Scheme != "https" || !isLoopback(u.Hostname()) {
		return http.DefaultClient
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.TLSClientConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec // loopback-only
	return &http.Client{Transport: tr}
}

// isLoopback reports whether host is 127.0.0.1, ::1, or localhost.
func isLoopback(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// firstLine returns s up to the first newline (or all of s). Used by list/
// status to keep the table compact when a title or task is multi-line.
func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' || s[i] == '\r' {
			return s[:i]
		}
	}
	return s
}

// gradeFlag renders a one-letter flag for a run's effective grade in the run
// list's FLAGS column. The human grade wins; the critic fallback is shown
// lowercase so the source is distinguishable at a glance. "" when ungraded.
func gradeFlag(human, critic *string) string {
	if human != nil {
		return strings.ToUpper(gradeLetter(*human))
	}
	if critic != nil {
		return strings.ToLower(gradeLetter(*critic))
	}
	return ""
}

// gradeLetter maps a grade string to its first letter (E/G/M/B/garbage uses X
// to avoid colliding with Bad). Empty for an unknown string.
func gradeLetter(grade string) string {
	switch grade {
	case "excellent":
		return "E"
	case "good":
		return "G"
	case "meh":
		return "M"
	case "bad":
		return "B"
	case "garbage":
		return "X"
	default:
		return ""
	}
}

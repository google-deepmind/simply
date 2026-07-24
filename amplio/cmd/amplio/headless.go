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

import "github.com/spf13/cobra"

// headlessCmd groups the no-server, foreground-execution operations:
//
//	amplio headless run [task]
//	amplio headless resume <run-id>
//
// Both own the data directory for the lifetime of the process — they cannot
// share a data dir with a running `amplio serve` (use `amplio client submit`
// instead when a server is up). The distinction is explicit so a new user
// doesn't mistake `amplio run` for "start the server".
func headlessCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "headless",
		Short: "Run a task or resume one in this process (no server, CI/eval mode)",
		Long: "Headless mode: this process owns the data directory and runs a single" +
			" agent task in the foreground until it concludes, crashes, or you cancel" +
			" it. Use this for batch evaluation, CI, or any time you don't want a long-" +
			"running server. To submit a task to an already-running `amplio serve`," +
			" use `amplio client submit` instead.",
	}
	cmd.AddCommand(runCmd(), resumeCmd())
	return cmd
}

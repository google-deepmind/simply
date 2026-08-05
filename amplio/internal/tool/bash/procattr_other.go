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

//go:build !unix

package bash

import "os/exec"

// Sessions and process groups are POSIX concepts; on other platforms the
// command keeps the default behaviour.
func detachTerminal(*exec.Cmd) {}

func killGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}

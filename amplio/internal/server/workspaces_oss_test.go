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

//go:build !internal

package server

import "testing"

// This build resolves plain paths only, so the picker must offer nothing else
// — advertising a source the resolver rejects turns a valid-looking choice
// into a failure at run start. Machine-independent by construction: no probe
// can add a mode here.
func TestExtraWorkspaceModes_NoneOffered(t *testing.T) {
	if got := extraWorkspaceModes(); len(got) != 0 {
		t.Errorf("extra workspace modes = %v, want none", got)
	}
}

func TestWorkspaceRecentRoot_Unset(t *testing.T) {
	if workspaceRecentRoot != "" {
		t.Errorf("recent-workspace root = %q, want empty", workspaceRecentRoot)
	}
}

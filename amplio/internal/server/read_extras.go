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

import "amplio/internal/workspace"

// attachCorpWorkspaceMeta is the stub for the OSS build. The internal build
// (see read_extras_internal.go) populates Alias / NumericID / CiderURL for
// backends that provide them; here those fields are always zero and drop out
// of the JSON via the omitempty tags on the DTO.
func attachCorpWorkspaceMeta(*workspaceMeta, workspace.Workspace) {}

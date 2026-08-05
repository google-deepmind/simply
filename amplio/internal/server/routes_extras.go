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

import "net/http"

// registerInternalRoutes is the stub for the OSS build. The internal build
// replaces it (see routes_extras_internal.go) with handlers that need
// optional backends (workspace alias management). Server.go always calls this
// after the generic routes are wired; OSS gets a no-op.
func (s *Server) registerInternalRoutes(*http.ServeMux) {}

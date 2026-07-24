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

// Package config provides process-wide configuration constants and types.
//
//   - Constants: reserved session IDs, env var names, embedder spec.
//   - Config / Load: persistent settings from <data-dir>/config.toml over
//     DefaultConfig (Go is the single source of truth). See load.go.
//   - RunConfig: user-supplied per-run config, persisted to Run.config_json in DB.
//   - Data directories: DataDir (--data-dir > $AMPLIO_DATA_DIR > ~/.amplio), ArtifactDir, etc.
package config

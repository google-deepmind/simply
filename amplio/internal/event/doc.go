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

// Package event defines typed events for the agent loop.
//
// The Event interface is sealed (unexported marker method) so only types in
// this package can implement it. Consumers use type switches for dispatch;
// the exhaustive linter catches missing cases.
//
// Serialization: events round-trip through JSON via Marshal/Unmarshal.
// The "type" discriminator field is written by Marshal and read by Unmarshal.
// Column-derived fields (Step, Generation, CreatedAt) are NOT included in the
// JSON blob — they are set by the caller after deserialization from DB column
// values.
package event

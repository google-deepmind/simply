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

// Package nickname generates short, memorable session IDs in adj-noun format.
//
// Pool: 100 adjectives x 100 nouns = 10,000 grid. The components of the two
// canonical role ids (main-agent, chatty-bot) are excluded from the dynamic
// pool. On exhaustion, falls back to <adj>-<noun>-<3 alphanumeric> suffix.
package nickname

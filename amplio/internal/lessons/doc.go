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

// Package lessons provides an in-memory cosine-similarity index over mined
// lessons, with confidence-weighted ranking. Unlike the skill index it reads
// from the DB (lessons carry their own stored embeddings), so Build does no
// embedding — only Search embeds (the query). Ranking multiplies each lesson's
// raw cosine by a confidence factor derived from its accumulated score and load
// count, so well-attributed lessons surface above marginal ones.
package lessons

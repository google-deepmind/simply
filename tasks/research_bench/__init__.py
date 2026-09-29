# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simply Research Bench tasks (OSS port).

The research-bench tasks, ported from the internal scaffolding to run on
Google Cloud.
`config_lib` registers every task config in its own namespace, `main` is the
training entry point, and `PORTING_NOTES.md` records every place the port
deviates from the internal source.
"""

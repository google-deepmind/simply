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

package llm

import "sync"

// CacheProviders memoises provider construction by spec.
//
// For a run, a provider is built once and used for hours; for a served API it
// would otherwise be built per request, and construction is not cheap — the
// Vertex providers mint ADC credentials, and every provider builds an HTTP
// client whose connection pool is the thing that makes the second request fast.
//
// The cache is unbounded on purpose: it is keyed by spec, and the specs come
// from a menu the operator wrote, so it cannot grow without one. Failures are
// not cached — a provider that failed to build because a credential was missing
// should succeed once the credential is there, without a restart.
func CacheProviders(build func(spec string) (Provider, error)) func(spec string) (Provider, error) {
	var (
		mu    sync.Mutex
		cache = map[string]Provider{}
	)
	return func(spec string) (Provider, error) {
		mu.Lock()
		p, ok := cache[spec]
		mu.Unlock()
		if ok {
			return p, nil
		}
		p, err := build(spec)
		if err != nil {
			return nil, err
		}
		mu.Lock()
		// Another caller may have built the same spec concurrently; either is
		// equivalent, so keep the one already published to avoid two providers
		// for one spec.
		if existing, ok := cache[spec]; ok {
			p = existing
		} else {
			cache[spec] = p
		}
		mu.Unlock()
		return p, nil
	}
}

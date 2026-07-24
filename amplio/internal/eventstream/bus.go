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

package eventstream

import (
	"sync"
	"sync/atomic"
)

// defaultQueueDepth bounds each subscriber's buffer. A busy run emits a handful
// of events per second, so this is many seconds of slack before overflow.
const defaultQueueDepth = 1024

// Bus fans out RunEvents to subscribers. A subscriber filters to one run, or to
// all runs (the dashboard) when subscribed with runID == "".
type Bus struct {
	mu    sync.Mutex
	subs  map[*Subscription]struct{}
	depth int
}

func NewBus() *Bus {
	return &Bus{subs: make(map[*Subscription]struct{}), depth: defaultQueueDepth}
}

// Subscription is one open stream. It is primed with a RefetchAll so the client
// loads initial state, then receives matching events. Always Close it.
type Subscription struct {
	bus      *Bus
	ch       chan RunEvent
	runID    string // "" = all runs (dashboard)
	overflow atomic.Bool
	once     sync.Once
}

// Subscribe opens a subscription for runID ("" = all runs).
func (b *Bus) Subscribe(runID string) *Subscription {
	s := &Subscription{bus: b, ch: make(chan RunEvent, b.depth), runID: runID}
	b.mu.Lock()
	b.subs[s] = struct{}{}
	b.mu.Unlock()
	s.ch <- RunEvent{Kind: KindRefetchAll, RunID: runID, Reason: "subscribe"} // buffered: never blocks
	return s
}

// Publish delivers ev to every matching subscriber (non-blocking; a full
// subscriber is flagged for a RefetchAll instead of stalling the publisher).
func (b *Bus) Publish(ev RunEvent) {
	b.mu.Lock()
	targets := make([]*Subscription, 0, len(b.subs))
	for s := range b.subs {
		// Empty event RunID = global signal (e.g. workspace_alias): every
		// subscriber receives it. Otherwise per-run subscribers must match.
		if s.runID == "" || ev.RunID == "" || s.runID == ev.RunID {
			targets = append(targets, s)
		}
	}
	b.mu.Unlock()
	for _, s := range targets {
		s.deliver(ev)
	}
}

func (b *Bus) remove(s *Subscription) {
	b.mu.Lock()
	delete(b.subs, s)
	b.mu.Unlock()
}

// C is the receive side of the subscription.
func (s *Subscription) C() <-chan RunEvent { return s.ch }

// TakeOverflow reports (and clears) whether events were dropped since the last
// call. The SSE handler checks it on a ticker and emits a RefetchAll when set.
func (s *Subscription) TakeOverflow() bool { return s.overflow.Swap(false) }

// Close detaches the subscription. The channel is never closed (publishers may
// still hold a snapshot reference); detached, it simply stops receiving.
func (s *Subscription) Close() { s.once.Do(func() { s.bus.remove(s) }) }

func (s *Subscription) deliver(ev RunEvent) {
	select {
	case s.ch <- ev:
	default:
		s.overflow.Store(true) // client is behind; it'll get a RefetchAll
	}
}

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

package sysstat

import (
	"sync"
	"testing"
)

func TestUpdate_NotifiesEveryCall(t *testing.T) {
	var mu sync.Mutex
	var got int
	w := New(func(_ Snapshot) {
		mu.Lock()
		got++
		mu.Unlock()
	})
	for range 5 {
		w.update(func(_ *Snapshot) {})
	}
	mu.Lock()
	defer mu.Unlock()
	if got != 5 {
		t.Errorf("notify count = %d, want 5 (always notifies, no equality short-circuit)", got)
	}
}

func TestParseCPULine(t *testing.T) {
	// fields: user nice system idle iowait irq softirq steal guest guest_nice
	ticks, err := parseCPULine("cpu  100 0 50 800 50 0 0 0 0 0")
	if err != nil {
		t.Fatal(err)
	}
	// total = 1000; idle (800) + iowait (50) = 850; active = 150
	if ticks.total != 1000 || ticks.active != 150 {
		t.Errorf("ticks = %+v, want {active:150 total:1000}", ticks)
	}
}

func TestCPUDeltaPct(t *testing.T) {
	prev := cpuTicks{active: 100, total: 1000}
	cur := cpuTicks{active: 250, total: 1500} // +150 active out of +500 total = 30%
	if got := cpuDeltaPct(prev, cur); got != 30 {
		t.Errorf("cpu delta pct = %v, want 30", got)
	}
	// No advance → 0 (avoid NaN).
	if got := cpuDeltaPct(cur, cur); got != 0 {
		t.Errorf("no-advance delta = %v, want 0", got)
	}
}

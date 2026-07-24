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

// Package sysstat is the long-lived watcher that probes server-host system
// signals (cpu/mem/swap/load; credential expiry in the internal build) and
// publishes change events to a notify callback (the server hooks it up to its
// SSE bus). One watcher serves all connected clients; clients seed via
// GET /api/sysstat then keep current via the kind=sysstat SSE event.
//
// Why server-pushed instead of per-client polling: see the design discussion —
// tab-throttling proof, lower aggregate server work (one probe vs N), and a
// single channel that future signals plug into without per-widget plumbing.
package sysstat

import (
	"context"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

const loadCadence = 5 * time.Second

// Snapshot is the latest known value of every signal. Fields are pointers so
// "unknown" (probe not yet run / failed) is distinguishable from a zero value.
// The whole struct rides each SSE event — full state, no diff/reconciliation.
type Snapshot struct {
	CredentialSeconds *int     `json:"credential_seconds,omitempty"` // auth credential remaining seconds; nil = unknown or no probe (OSS)
	LoadAvg1m         *float64 `json:"load_avg_1m,omitempty"`        // 1-minute load average
	CPUPercent        *float64 `json:"cpu_pct,omitempty"`            // aggregate CPU busy %, 0-100
	MemPercent        *float64 `json:"mem_pct,omitempty"`            // (MemTotal - MemAvailable) / MemTotal × 100
	SwapPercent       *float64 `json:"swap_pct,omitempty"`           // omitted when swap is disabled
}

// NotifyFunc is invoked from a probe goroutine when the snapshot's value
// changes. Must be safe for concurrent use.
type NotifyFunc func(Snapshot)

// Watcher owns the periodic probes and the latest snapshot. Construct once at
// server startup, call Run from a goroutine; Latest is safe for concurrent use.
type Watcher struct {
	notify NotifyFunc

	mu     sync.RWMutex
	latest Snapshot
}

// New constructs a watcher. notify may be nil (watcher still updates Latest()).
func New(notify NotifyFunc) *Watcher { return &Watcher{notify: notify} }

// Latest returns the current snapshot. Cheap; used by GET /api/sysstat to seed
// new clients before SSE updates begin.
func (w *Watcher) Latest() Snapshot {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.latest
}

// Run starts the probe goroutines and blocks until ctx is cancelled. Each
// signal has its own cadence (tiered for the credential probe; load probes
// at a fixed 5s). In the OSS build runCredentialProbe is a no-op stub and
// Run blocks on the load probe alone.
func (w *Watcher) Run(ctx context.Context) {
	go w.runLoad(ctx)
	w.runCredentialProbe(ctx)
}

func (w *Watcher) update(apply func(*Snapshot)) {
	w.mu.Lock()
	apply(&w.latest)
	snap := w.latest
	w.mu.Unlock()
	// Always notify: load values (cpu/mem) are intrinsically continuous and we
	// want the UI to reflect every probe; suppressing on equality would only
	// matter when no signal changed at all, which is rare and not worth the
	// per-field plumbing.
	if w.notify != nil {
		w.notify(snap)
	}
}

// --- load / mem / swap / blaze probe ---
//
// Linux-only (/proc/*). On other OSes the file reads fail silently and the
// chips render "—". cadence is fixed 5s — fast
// enough to feel live, slow enough that 1 reader serves all tabs at ~12
// events/min, dominated by the size of the comm scan (sub-ms for typical
// process counts).

func (w *Watcher) runLoad(ctx context.Context) {
	// First probe seeds prevCPU; the first published CPUPercent is therefore
	// from the SECOND tick (need two reads for a delta).
	var prevCPU cpuTicks
	var havePrev bool
	for {
		var cpuPct *float64
		if cur, err := readCPUTicks(); err == nil {
			if havePrev {
				pct := cpuDeltaPct(prevCPU, cur)
				cpuPct = &pct
			}
			prevCPU = cur
			havePrev = true
		}
		mem, swap := readMemSwap()
		load := readLoadAvg1m()

		w.update(func(s *Snapshot) {
			s.CPUPercent = cpuPct
			s.MemPercent = mem
			s.SwapPercent = swap
			s.LoadAvg1m = load
		})
		select {
		case <-ctx.Done():
			return
		case <-time.After(loadCadence):
		}
	}
}

// cpuTicks is the active + total accumulators from one /proc/stat read.
type cpuTicks struct{ active, total uint64 }

// readCPUTicks parses /proc/stat's first line ("cpu  u n s i io irq sirq steal …"),
// returning the cumulative active/total tick counts. CPU% is a delta of these
// between two probes.
func readCPUTicks() (cpuTicks, error) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return cpuTicks{}, err
	}
	line := data
	if i := strings.IndexByte(string(data), '\n'); i >= 0 {
		line = data[:i]
	}
	return parseCPULine(string(line))
}

func parseCPULine(line string) (cpuTicks, error) {
	fields := strings.Fields(line)
	if len(fields) < 5 || fields[0] != "cpu" {
		return cpuTicks{}, errParse
	}
	// fields[1:]: user nice system idle iowait irq softirq steal guest guest_nice
	vals := make([]uint64, 0, len(fields)-1)
	for _, f := range fields[1:] {
		n, err := strconv.ParseUint(f, 10, 64)
		if err != nil {
			return cpuTicks{}, err
		}
		vals = append(vals, n)
	}
	var total uint64
	for _, v := range vals {
		total += v
	}
	// idle = vals[3], iowait = vals[4] (when present). Treat iowait as "not
	// CPU-busy" so the % matches operator intuition ("waiting on disk" ≠ "busy").
	idle := vals[3]
	if len(vals) > 4 {
		idle += vals[4]
	}
	return cpuTicks{active: total - idle, total: total}, nil
}

// cpuDeltaPct returns 0-100 from two ticks. Returns 0 if the total didn't
// advance (clock skew or back-to-back reads).
func cpuDeltaPct(prev, cur cpuTicks) float64 {
	dt := cur.total - prev.total
	if dt == 0 {
		return 0
	}
	da := cur.active - prev.active
	return float64(da) / float64(dt) * 100
}

// readMemSwap returns (memPercent, swapPercent), with swap nil when none
// configured (SwapTotal=0). MemAvailable is the kernel's modern "free for
// reuse" figure (cache pressure aware) — preferred over MemFree.
func readMemSwap() (*float64, *float64) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return nil, nil
	}
	kv := map[string]uint64{}
	for _, line := range strings.Split(string(data), "\n") {
		k, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) == 0 {
			continue
		}
		v, err := strconv.ParseUint(f[0], 10, 64)
		if err != nil {
			continue
		}
		kv[k] = v
	}
	var memPct, swapPct *float64
	if total, avail := kv["MemTotal"], kv["MemAvailable"]; total > 0 && avail <= total {
		p := float64(total-avail) / float64(total) * 100
		memPct = &p
	}
	if total, free := kv["SwapTotal"], kv["SwapFree"]; total > 0 && free <= total {
		p := float64(total-free) / float64(total) * 100
		swapPct = &p
	}
	return memPct, swapPct
}

// readLoadAvg1m returns the 1-minute load average from /proc/loadavg, or nil
// when unreadable.
func readLoadAvg1m() *float64 {
	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return nil
	}
	f := strings.Fields(string(data))
	if len(f) == 0 {
		return nil
	}
	v, err := strconv.ParseFloat(f[0], 64)
	if err != nil {
		return nil
	}
	return &v
}

// errParse is the sentinel returned by parsers when input isn't shaped right.
var errParse = errParseError{}

type errParseError struct{}

func (errParseError) Error() string { return "sysstat: parse error" }

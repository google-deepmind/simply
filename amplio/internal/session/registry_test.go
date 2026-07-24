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

package session

import (
	"context"
	"testing"
	"time"
)

func TestRegistry_RegisterAndUnregister(t *testing.T) {
	r := NewRegistry()
	h := NewHandle(func() {})

	if err := r.Register("s1", h); err != nil {
		t.Fatal(err)
	}
	if !r.IsRegistered("s1") {
		t.Error("should be registered")
	}
	if err := r.Register("s1", h); err == nil {
		t.Error("expected error for duplicate")
	}

	r.Unregister("s1")
	if r.IsRegistered("s1") {
		t.Error("should be unregistered")
	}
}

func TestRegistry_NotifyWakesWaiter(t *testing.T) {
	r := NewRegistry()
	h := NewHandle(func() {})
	_ = r.Register("s1", h)

	before := h.Counter()
	woke := make(chan uint64, 1)
	go func() {
		c, _ := h.WaitAfter(context.Background(), before, time.Second)
		woke <- c
	}()
	// Give the waiter a moment to block, then notify.
	time.Sleep(10 * time.Millisecond)
	if !r.Notify("s1") {
		t.Fatal("Notify should report the session as registered")
	}
	select {
	case c := <-woke:
		if c <= before {
			t.Errorf("counter did not advance: got %d, before %d", c, before)
		}
	case <-time.After(time.Second):
		t.Fatal("notify did not wake the waiter")
	}
}

func TestRegistry_NotifyMissingReturnsFalse(t *testing.T) {
	r := NewRegistry()
	if r.Notify("nonexistent") {
		t.Error("Notify on unregistered session should return false")
	}
}

func TestWaiter_SnapshotBeforeNotifyNoMiss(t *testing.T) {
	h := NewHandle(func() {})
	before := h.Counter()
	h.waiter.notify() // notify lands between snapshot and wait
	c, err := h.WaitAfter(context.Background(), before, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if c <= before {
		t.Errorf("WaitAfter should return immediately with advanced counter: got %d, before %d", c, before)
	}
}

func TestWaiter_Timeout(t *testing.T) {
	h := NewHandle(func() {})
	before := h.Counter()
	c, err := h.WaitAfter(context.Background(), before, 20*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if c != before {
		t.Errorf("expected counter unchanged on timeout: got %d, before %d", c, before)
	}
}

func TestWaiter_CtxCancel(t *testing.T) {
	h := NewHandle(func() {})
	before := h.Counter()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := h.WaitAfter(ctx, before, time.Second)
	if err == nil {
		t.Error("expected context error")
	}
}

func TestRegistry_Cancel(t *testing.T) {
	r := NewRegistry()
	ctx, cancel := context.WithCancel(context.Background())
	_ = r.Register("s1", NewHandle(cancel))

	r.Interrupt("s1")
	if ctx.Err() == nil {
		t.Error("should be cancelled")
	}
}

func TestRegistry_CancelAll(t *testing.T) {
	r := NewRegistry()
	ctx1, cancel1 := context.WithCancel(context.Background())
	ctx2, cancel2 := context.WithCancel(context.Background())
	_ = r.Register("s1", NewHandle(cancel1))
	_ = r.Register("s2", NewHandle(cancel2))

	r.CancelAll()
	if ctx1.Err() == nil || ctx2.Err() == nil {
		t.Error("both should be cancelled")
	}
}

func TestRegistry_IsEmpty(t *testing.T) {
	r := NewRegistry()
	if !r.IsEmpty() {
		t.Error("should be empty")
	}
	_ = r.Register("s1", NewHandle(func() {}))
	if r.IsEmpty() {
		t.Error("should not be empty")
	}
	r.Unregister("s1")
	if !r.IsEmpty() {
		t.Error("should be empty after unregister")
	}
}

func TestRegistry_RegisterAndContext(t *testing.T) {
	r := NewRegistry()

	// First claim wins: registers the slot and returns a live cancelable ctx.
	ctx, h, release, ok := r.RegisterAndContext(context.Background(), "s1")
	if !ok || h == nil || ctx == nil || release == nil {
		t.Fatalf("first claim: ok=%v h!=nil=%v ctx!=nil=%v release!=nil=%v, want a full claim",
			ok, h != nil, ctx != nil, release != nil)
	}
	if !r.IsRegistered("s1") {
		t.Error("session should be registered immediately after RegisterAndContext")
	}

	// A second claim on the same id loses (ok=false, nothing returned).
	ctx2, h2, release2, ok2 := r.RegisterAndContext(context.Background(), "s1")
	if ok2 || h2 != nil || ctx2 != nil || release2 != nil {
		t.Errorf("second claim should lose: ok=%v h!=nil=%v ctx!=nil=%v release!=nil=%v",
			ok2, h2 != nil, ctx2 != nil, release2 != nil)
	}

	// Interrupt fires the slot's cancel (its ctx is done).
	r.Interrupt("s1")
	if ctx.Err() == nil {
		t.Error("Interrupt should cancel the slot's ctx")
	}

	// release cancels (idempotent) and unregisters.
	release()
	if r.IsRegistered("s1") {
		t.Error("release should unregister the slot")
	}
}

// The parent ctx cancels the derived slot ctx (child-of-parent semantics).
func TestRegistry_RegisterAndContext_ParentCancels(t *testing.T) {
	r := NewRegistry()
	parent, cancelParent := context.WithCancel(context.Background())
	ctx, _, release, ok := r.RegisterAndContext(parent, "s1")
	if !ok {
		t.Fatal("claim should win")
	}
	defer release()
	cancelParent()
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Error("cancelling the parent ctx should cancel the slot ctx")
	}
}

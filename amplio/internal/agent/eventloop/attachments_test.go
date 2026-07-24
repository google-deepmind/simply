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

package eventloop

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"testing"

	"amplio/internal/blob"
	"amplio/internal/event"
)

// pngBytes builds a w×h PNG for attachment-clamp tests.
func pngBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y), 100, 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func TestAttachmentBlobRoundTrip(t *testing.T) {
	a := newT(testCfg{BlobStore: blob.NewStore(t.TempDir())})
	data := []byte("\x89PNG\r\n\x1a\nfake image bytes")

	ref, ok := a.storeAttachment(event.Attachment{MimeType: "image/png", Data: data, SourceHint: "x.png"})
	if !ok {
		t.Fatal("storeAttachment returned ok=false")
	}
	if !blob.ValidKey(ref.BlobKey) || ref.Size != len(data) || ref.MimeType != "image/png" {
		t.Fatalf("bad ref: %+v", ref)
	}
	if len(ref.Data) != 0 {
		t.Error("persistable ref should not carry raw Data")
	}

	atts := a.loadAttachments([]event.Attachment{ref})
	if len(atts) != 1 {
		t.Fatalf("loadAttachments len = %d", len(atts))
	}
	if atts[0].MimeType != "image/png" {
		t.Errorf("mime: %q", atts[0].MimeType)
	}
	if want := base64.StdEncoding.EncodeToString(data); atts[0].Base64Data != want {
		t.Error("round-tripped bytes differ from original")
	}
}

func TestAttachmentNilStoreDegrades(t *testing.T) {
	a := newT(testCfg{}) // no BlobStore configured
	if _, ok := a.storeAttachment(event.Attachment{MimeType: "image/png", Data: []byte("x")}); ok {
		t.Error("expected storeAttachment to drop when no blob store")
	}
	if got := a.loadAttachments([]event.Attachment{{BlobKey: "deadbeef", MimeType: "image/png"}}); got != nil {
		t.Errorf("expected nil attachments with no store, got %v", got)
	}
}

// An oversized image is downscaled before it's persisted, so the stored blob
// (and the ref) reflect the clamped dimensions — not the original.
func TestStoreAttachmentClampsOversized(t *testing.T) {
	a := newT(testCfg{BlobStore: blob.NewStore(t.TempDir())})
	orig := pngBytes(t, 4000, 1000) // long edge 4000 > 2000 cap

	ref, ok := a.storeAttachment(event.Attachment{MimeType: "image/png", Data: orig, SourceHint: "big.png"})
	if !ok {
		t.Fatal("storeAttachment returned ok=false")
	}
	if ref.Size >= len(orig) {
		t.Errorf("stored size %d not smaller than original %d; expected downscale", ref.Size, len(orig))
	}
	// Read the stored bytes back and confirm the pixel dimensions were clamped.
	atts := a.loadAttachments([]event.Attachment{ref})
	if len(atts) != 1 {
		t.Fatalf("loadAttachments len = %d", len(atts))
	}
	raw, err := base64.StdEncoding.DecodeString(atts[0].Base64Data)
	if err != nil {
		t.Fatalf("decode base64: %v", err)
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("decode stored image: %v", err)
	}
	if cfg.Width != 2000 || cfg.Height != 500 {
		t.Errorf("stored dims = %dx%d, want 2000x500 (clamped, aspect preserved)", cfg.Width, cfg.Height)
	}
}

// An in-bounds image is stored byte-for-byte (no re-encode).
func TestStoreAttachmentKeepsInBounds(t *testing.T) {
	a := newT(testCfg{BlobStore: blob.NewStore(t.TempDir())})
	orig := pngBytes(t, 800, 600)

	ref, ok := a.storeAttachment(event.Attachment{MimeType: "image/png", Data: orig, SourceHint: "ok.png"})
	if !ok {
		t.Fatal("storeAttachment returned ok=false")
	}
	if ref.Size != len(orig) {
		t.Errorf("stored size %d != original %d; in-bounds image should not be re-encoded", ref.Size, len(orig))
	}
	atts := a.loadAttachments([]event.Attachment{ref})
	if want := base64.StdEncoding.EncodeToString(orig); atts[0].Base64Data != want {
		t.Error("in-bounds image bytes changed; want original preserved")
	}
}

// A non-decodable "image" (or non-image mime) is stored unchanged rather than
// dropped — a clamp failure must never lose the attachment.
func TestStoreAttachmentUndecodableFallsBack(t *testing.T) {
	a := newT(testCfg{BlobStore: blob.NewStore(t.TempDir())})
	data := []byte("\x89PNG\r\n\x1a\nnot a real png")

	ref, ok := a.storeAttachment(event.Attachment{MimeType: "image/png", Data: data, SourceHint: "x.png"})
	if !ok {
		t.Fatal("storeAttachment returned ok=false; undecodable image must fall back, not drop")
	}
	if ref.Size != len(data) || ref.MimeType != "image/png" {
		t.Errorf("fallback ref = %+v, want original size %d / image/png", ref, len(data))
	}
	atts := a.loadAttachments([]event.Attachment{ref})
	if want := base64.StdEncoding.EncodeToString(data); atts[0].Base64Data != want {
		t.Error("undecodable image bytes changed; want original preserved")
	}
}

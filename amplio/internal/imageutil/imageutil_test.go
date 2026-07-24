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

package imageutil

import (
	"bytes"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"testing"
)

// makeImage builds a w×h image with a simple gradient (so resampling has real
// content to work on) and encodes it in the requested format.
func makeImage(t *testing.T, w, h int, format string) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), 128, 255})
		}
	}
	var buf bytes.Buffer
	var err error
	switch format {
	case "png":
		err = png.Encode(&buf, img)
	case "jpeg":
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90})
	case "gif":
		err = gif.Encode(&buf, img, nil)
	default:
		t.Fatalf("unknown format %q", format)
	}
	if err != nil {
		t.Fatalf("encode %s: %v", format, err)
	}
	return buf.Bytes()
}

// dims decodes just the header to read an image's pixel dimensions.
func dims(t *testing.T, data []byte) (int, int) {
	t.Helper()
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decode config: %v", err)
	}
	return cfg.Width, cfg.Height
}

func TestClamp_WithinBounds_ReturnsOriginal(t *testing.T) {
	orig := makeImage(t, 800, 600, "png")
	out, mime, resized, err := Clamp(orig, "image/png", 2000)
	if err != nil {
		t.Fatalf("Clamp: %v", err)
	}
	if resized {
		t.Error("resized = true, want false for an in-bounds image")
	}
	if mime != "image/png" {
		t.Errorf("mime = %q, want image/png", mime)
	}
	// Must be the exact same bytes (no re-encode).
	if !bytes.Equal(out, orig) {
		t.Error("in-bounds image was re-encoded; want original bytes returned untouched")
	}
}

func TestClamp_OversizedWidth_Downscales(t *testing.T) {
	orig := makeImage(t, 4000, 1000, "png")
	out, mime, resized, err := Clamp(orig, "image/png", 2000)
	if err != nil {
		t.Fatalf("Clamp: %v", err)
	}
	if !resized {
		t.Fatal("resized = false, want true for an oversized image")
	}
	if mime != "image/png" {
		t.Errorf("mime = %q, want image/png", mime)
	}
	w, h := dims(t, out)
	if w != 2000 {
		t.Errorf("width = %d, want 2000 (long edge clamped)", w)
	}
	// Aspect ratio 4:1 preserved => height 500.
	if h != 500 {
		t.Errorf("height = %d, want 500 (aspect preserved)", h)
	}
}

func TestClamp_OversizedHeight_Downscales(t *testing.T) {
	// A tall screenshot: small bytes, big pixel height — the exact case that
	// sails past a byte-size cap but is rejected by the provider.
	orig := makeImage(t, 1000, 5000, "png")
	out, _, resized, err := Clamp(orig, "image/png", 2000)
	if err != nil {
		t.Fatalf("Clamp: %v", err)
	}
	if !resized {
		t.Fatal("resized = false, want true")
	}
	w, h := dims(t, out)
	if h != 2000 {
		t.Errorf("height = %d, want 2000 (long edge clamped)", h)
	}
	// 1000:5000 = 1:5 preserved => width 400.
	if w != 400 {
		t.Errorf("width = %d, want 400 (aspect preserved)", w)
	}
}

func TestClamp_JPEGStaysJPEG(t *testing.T) {
	orig := makeImage(t, 3000, 3000, "jpeg")
	out, mime, resized, err := Clamp(orig, "image/jpeg", 2000)
	if err != nil {
		t.Fatalf("Clamp: %v", err)
	}
	if !resized {
		t.Fatal("resized = false, want true")
	}
	if mime != "image/jpeg" {
		t.Errorf("mime = %q, want image/jpeg (JPEG should stay JPEG)", mime)
	}
	w, h := dims(t, out)
	if w != 2000 || h != 2000 {
		t.Errorf("dims = %dx%d, want 2000x2000", w, h)
	}
	// Confirm the output really decodes as JPEG.
	_, format, err := image.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if format != "jpeg" {
		t.Errorf("output format = %q, want jpeg", format)
	}
}

func TestClamp_GIFReencodedToPNG(t *testing.T) {
	// Keep GIF dimensions modest: GIF encoding does palette quantization, which
	// is slow on large gradients. maxDim=100 still exercises the oversize path.
	orig := makeImage(t, 300, 200, "gif")
	out, mime, resized, err := Clamp(orig, "image/gif", 100)
	if err != nil {
		t.Fatalf("Clamp: %v", err)
	}
	if !resized {
		t.Fatal("resized = false, want true")
	}
	if mime != "image/png" {
		t.Errorf("mime = %q, want image/png (GIF re-encoded to PNG)", mime)
	}
	_, format, err := image.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if format != "png" {
		t.Errorf("output format = %q, want png", format)
	}
}

func TestClamp_DefaultMaxDim(t *testing.T) {
	orig := makeImage(t, 2500, 2500, "png")
	out, _, resized, err := Clamp(orig, "image/png", 0) // 0 => DefaultMaxDim
	if err != nil {
		t.Fatalf("Clamp: %v", err)
	}
	if !resized {
		t.Fatal("resized = false, want true (2500 > DefaultMaxDim)")
	}
	w, h := dims(t, out)
	if w != DefaultMaxDim || h != DefaultMaxDim {
		t.Errorf("dims = %dx%d, want %dx%[3]d", w, h, DefaultMaxDim)
	}
}

func TestClamp_ExactlyAtLimit_NotResized(t *testing.T) {
	orig := makeImage(t, 2000, 2000, "png")
	out, _, resized, err := Clamp(orig, "image/png", 2000)
	if err != nil {
		t.Fatalf("Clamp: %v", err)
	}
	if resized {
		t.Error("resized = true, want false for image exactly at the limit")
	}
	if !bytes.Equal(out, orig) {
		t.Error("image at the limit was re-encoded; want original bytes")
	}
}

func TestClamp_InvalidData_ReturnsOriginalAndError(t *testing.T) {
	bad := []byte("this is not an image")
	out, mime, resized, err := Clamp(bad, "image/png", 2000)
	if err == nil {
		t.Fatal("err = nil, want a decode error")
	}
	if resized {
		t.Error("resized = true, want false on decode failure")
	}
	if mime != "image/png" {
		t.Errorf("mime = %q, want original image/png preserved on error", mime)
	}
	if !bytes.Equal(out, bad) {
		t.Error("want original bytes returned on decode failure (never drop the image)")
	}
}

func TestScaledDims(t *testing.T) {
	cases := []struct {
		w, h, max    int
		wantW, wantH int
	}{
		{4000, 1000, 2000, 2000, 500},  // landscape
		{1000, 4000, 2000, 500, 2000},  // portrait
		{3000, 3000, 2000, 2000, 2000}, // square
		{10000, 100, 2000, 2000, 20},   // extreme landscape
		{100, 10000, 2000, 20, 2000},   // extreme portrait
		{0, 0, 2000, 1, 1},             // degenerate
	}
	for _, c := range cases {
		gotW, gotH := scaledDims(c.w, c.h, c.max)
		if gotW != c.wantW || gotH != c.wantH {
			t.Errorf("scaledDims(%d,%d,%d) = %dx%d, want %dx%d",
				c.w, c.h, c.max, gotW, gotH, c.wantW, c.wantH)
		}
	}
}

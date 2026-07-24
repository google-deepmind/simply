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

// Package imageutil provides safeguards for image attachments before they are
// persisted and later sent to an LLM. Its main job is Clamp: aspect-preserving
// downscaling of images whose pixel dimensions exceed a provider's limit.
//
// Model providers reject images by PIXEL DIMENSION, not byte size (e.g. Vertex/
// Claude 400s on an image whose long edge is too large, and Anthropic itself
// downscales anything over ~1568px before processing). A tall-but-small-bytes
// screenshot therefore sails past a byte-size cap yet still fails at the API, so
// we clamp dimensions at the single point every attachment funnels through.
package imageutil

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"

	// Register GIF as a decodable format for image.Decode (decode-only; a
	// resized GIF is re-encoded to PNG).
	_ "image/gif"

	"golang.org/x/image/draw"

	// Register WebP as a decodable format for image.Decode. Decode-only; we
	// re-encode a resized WebP to PNG (x/image has no WebP encoder).
	_ "golang.org/x/image/webp"
)

// DefaultMaxDim is the default long-edge clamp in pixels. 2000px sits safely
// under every provider's hard rejection threshold while leaving headroom above
// Anthropic's ~1568px internal downscale target.
const DefaultMaxDim = 2000

// jpegQuality is the re-encode quality for JPEG output. 90 is visually
// near-lossless while keeping the byte size well below the base64 caps.
const jpegQuality = 90

// Clamp downscales an image so neither dimension exceeds maxDim, preserving
// aspect ratio. It returns the (possibly new) bytes, the output MIME type, and
// whether a resize happened.
//
// Behavior:
//   - If the image already fits (both dimensions <= maxDim), the ORIGINAL bytes
//     and MIME are returned untouched (resized=false), so no re-encode artifacts
//     are introduced on images that don't need it.
//   - JPEG stays JPEG (q90); every other decodable format (PNG, GIF, WebP) is
//     re-encoded to PNG, since x/image ships no encoders for GIF-animation or
//     WebP and PNG is universally accepted and lossless.
//   - On any decode/encode failure the original bytes+MIME are returned with a
//     non-nil error; callers should keep the original and log, never drop the
//     image or fail the turn.
//
// maxDim <= 0 selects DefaultMaxDim.
func Clamp(data []byte, mime string, maxDim int) (out []byte, outMime string, resized bool, err error) {
	if maxDim <= 0 {
		maxDim = DefaultMaxDim
	}

	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return data, mime, false, fmt.Errorf("decode image config: %w", err)
	}
	// Already within bounds: return the original bytes untouched.
	if cfg.Width <= maxDim && cfg.Height <= maxDim {
		return data, mime, false, nil
	}

	src, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return data, mime, false, fmt.Errorf("decode image: %w", err)
	}

	nw, nh := scaledDims(cfg.Width, cfg.Height, maxDim)
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	// CatmullRom is a high-quality resampling kernel — good for downscaling
	// screenshots/diagrams where sharp text should stay legible.
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)

	var buf bytes.Buffer
	// Keep JPEG as JPEG; re-encode everything else to PNG.
	if format == "jpeg" {
		if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: jpegQuality}); err != nil {
			return data, mime, false, fmt.Errorf("encode jpeg: %w", err)
		}
		return buf.Bytes(), "image/jpeg", true, nil
	}
	if err := png.Encode(&buf, dst); err != nil {
		return data, mime, false, fmt.Errorf("encode png: %w", err)
	}
	return buf.Bytes(), "image/png", true, nil
}

// scaledDims returns the largest width/height that fit within a maxDim square
// while preserving the source aspect ratio. Each result is at least 1px.
func scaledDims(w, h, maxDim int) (int, int) {
	if w <= 0 || h <= 0 {
		return 1, 1
	}
	if w >= h {
		nh := h * maxDim / w
		if nh < 1 {
			nh = 1
		}
		return maxDim, nh
	}
	nw := w * maxDim / h
	if nw < 1 {
		nw = 1
	}
	return nw, maxDim
}

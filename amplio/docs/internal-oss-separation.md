# Internal / OSS separation

This file describes the build-and-mirror conventions that keep `amplio`
buildable as both an internal and an open-source tree. Read it before adding
new files.

## Two build configurations

`amplio` is dual-built from one source tree:

- **Internal build** (default in the internal repo): includes all
  environment-specific backends — e.g. workspace integration, credential
  probing, extra UI affordances, and a host-environment system prompt. All
  Makefile targets and the nightly workflow pass `-tags=internal`.
- **OSS build** (produced via Copybara mirror to a public Git repo): stubs
  in place of environment-specific code. `go build ./...` with no tags compiles
  the OSS subset and is the safety fallback for anyone bypassing the Makefile.

The Copybara mirror drops the build-tagged `*_internal*.go` files, overlays OSS
replacements for the internal Svelte components, and rewrites `Makefile` and
`.golangci.yml` to drop `-tags=internal`. It also **excludes** the internal-only
`.github/workflows/nightly.yml` outright, so the OSS mirror ships source only and
carries no binary-release CI. The OSS Makefile target therefore reduces to plain
`go build`.

## Build tag convention

Corp-only Go files carry both the build tag and a filename suffix:

```go
//go:build internal
```

| Kind        | Suffix                  |
| ----------- | ----------------------- |
| Production  | `*_internal.go`         |
| Tests       | `*_internal_test.go`    |

Copybara excludes these via a single glob (`**/*_internal*.go`); no per-file
maintenance.

## Adding new corp-only code

Two equivalent patterns; pick the one that fits the call shape.

### Pattern A — function variable (preferred for package-level APIs)

Stub vars in an untagged file; real impls swap them in via `init()`. Both
files compile in the internal build; only the stub file ships to OSS.

```go
// foo/api.go  (untagged, ships to OSS)
package foo

var DoThing = func(x int) (string, error) { return "", ErrNotAvailable }

// foo/impl_internal.go  (//go:build internal, Copybara excludes)
package foo

func init() { DoThing = realDoThing }
func realDoThing(x int) (string, error) { /* real impl */ }
```

Use for: environment-specific lookups (workspace metadata, host probes) —
where callers want a plain
package-level function reference.

### Pattern B — mutually-exclusive build tags

For methods on a type, or when you want a plain `func` (no var indirection),
use complementary build tags so only ONE file compiles per configuration.

```go
// foo/extras.go  (//go:build !internal)
package foo

func (s *Server) doThing(x int) error { return nil } // OSS no-op

// foo/extras_internal.go  (//go:build internal)
package foo

func (s *Server) doThing(x int) error { /* real impl */ }
```

Use for: methods on shared types (where you can't reassign), or when the OSS
behavior is genuinely "do nothing" with no return value to stub.

### Verification

Verify both configurations build cleanly before submitting:
- `make build` — internal tag set (default workflow)
- `go build ./...` — no tags, OSS path

## Frontend convention

Corp-only Svelte components live under `frontend/src/lib/components/internal/`.
Importers route through `frontend/src/lib/components/internal/index.ts` (the
barrel). Copybara replaces the barrel with a stub that exports no-op
components and excludes the rest of the directory.

For DTO fields that only exist on the internal build (e.g. an internal-only
`foo_url` or `credential_seconds`), use `omitempty` on the Go side. The field
drops out of JSON automatically. Frontend code guards usage with field-presence
checks (`{#if summary.foo_url}`), so the same Svelte source works on both builds.

## Auditing for corp leakage

When adding code that touches corp infrastructure (paths, tool names, service
URLs, prompt text), decide which bucket it belongs to:

- **Build tag**: real internal implementation (environment-specific subprocess
  calls, host filesystems, private RPCs). Use `_internal.go`.
- **Pure prose**: a sentence in a README or comment naming a corp tool.
  Copybara handles trivial scrubs via `core.replace`.
- **Restructure**: corp-specific data in a shared struct or constant. Move
  it behind a build tag or a generic abstraction.

When in doubt, prefer the build tag — Copybara `core.replace` rules grow
brittle as the corpus changes.

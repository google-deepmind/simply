# webdist

The Go server embeds the built SvelteKit SPA from this directory via
`//go:embed all:webdist` (see `static.go`).

The actual build output lives in `client/` and is **generated, not committed**:

```sh
make frontend-build   # cd frontend && npm run build  →  webdist/client/
```

`client/` is git-ignored. This `README.md` is the committed anchor that keeps
`go:embed` compiling on a fresh checkout (an empty embed pattern is a build
error). Until the frontend is built, the server returns `503 frontend not
built` for SPA routes; the JSON API and SSE endpoints work regardless.

Release tarballs may ship a pre-built `client/` so consumers can `go build`
without a Node toolchain.

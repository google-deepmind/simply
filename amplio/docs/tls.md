# TLS and HTTP/2

## The problem

Browsers cap concurrent HTTP/1.1 connections per origin at 6. amplio's UI
holds one long-lived SSE stream open per tab (plus occasional XHRs), so a
multi-tab session — a dashboard plus a few run tabs — exhausts the pool
within seconds. The UI then stalls silently: no error, just no updates.

HTTP/2 fixes it by multiplexing all streams onto a single TCP connection
(server-advertised `SETTINGS_MAX_CONCURRENT_STREAMS`, typically ~100). But
browsers only negotiate HTTP/2 over TLS — there's no plain-HTTP `h2c`
support — so the fix is "serve HTTPS, get HTTP/2 for free" (Go's
`net/http` enables HTTP/2 automatically on `ServeTLS`).

## How amplio picks up TLS

amplio looks for `<data-dir>/cert.pem` + `<data-dir>/key.pem` at startup.
If both exist, it serves HTTPS; otherwise it falls back to plain HTTP. The
sections below cover the three common deployment shapes.

## Local development (server and browser on the same machine)

[mkcert](https://github.com/FiloSottile/mkcert) is the easiest way to get a
browser-trusted localhost cert.

```bash
# One-time per machine (installs a local CA into your browser trust store):
brew install mkcert nss      # macOS; on Debian/Ubuntu: apt install mkcert
mkcert -install

# One-time per data dir:
mkcert -cert-file ~/.amplio/cert.pem -key-file ~/.amplio/key.pem \
  localhost 127.0.0.1 ::1
```

Restart `amplio serve` and the banner URL will print `https://...`.

If `mkcert` is on `$PATH` when amplio starts and no cert exists in the data
dir yet, amplio attempts to generate one automatically. You still need to
have run `mkcert -install` at least once for the browser to trust the local
CA — that step modifies the system trust store and we don't run it on your
behalf.

## SSH-tunnel deployment (server on a remote, browser on a laptop)

```
[laptop browser] ──TLS──> [SSH tunnel: opaque bytes] ──> [remote: amplio HTTPS]
```

The SSH tunnel forwards TCP bytes without terminating TLS — the browser
handshakes directly with the amplio server through the tunnel. For the
browser to trust the cert, it must be signed by a CA in the **laptop's**
trust store, not the remote's. So mkcert needs to be run on the laptop and
the resulting cert files copied to the remote:

```bash
# === On the LAPTOP (one-time install of mkcert + local CA): ===
brew install mkcert nss
mkcert -install              # adds local CA to the laptop's browser trust store

# === On the LAPTOP (per remote): ===
mkcert -cert-file /tmp/cert.pem -key-file /tmp/key.pem \
  localhost 127.0.0.1 ::1

# === Copy to the remote's amplio data dir: ===
scp /tmp/{cert,key}.pem remote:~/.amplio/

# === On the REMOTE: ===
amplio serve                 # picks up the laptop-signed cert

# === Back on the LAPTOP: ===
ssh -L 26759:localhost:26759 remote
# Browser: https://localhost:26759 → trusted, HTTP/2 enabled
```

amplio's auto-mkcert (running on the remote) doesn't help here — it would
sign the cert with the remote's CA, which the laptop's browser doesn't
trust. The flow above puts the trust anchor and the certificate on
opposite sides of the tunnel deliberately.

SSH itself provides wire-level confidentiality. The reason to still want
TLS in this setup is HTTP/2: browsers don't speak HTTP/2 over plain HTTP
(no `h2c` support), only over TLS via ALPN negotiation. If you're only
ever in one or two tabs at a time, plain HTTP through SSH is fine and
this whole document is optional.

## Reverse-proxy deployment

If amplio is behind a reverse proxy that already terminates TLS toward the
browser (corporate proxies, ingress controllers, Cloudflare, etc.), don't
configure TLS at the amplio process. The proxy handles HTTP/2 negotiation
with the browser; the loopback hop between the proxy and amplio is happy
to stay plain HTTP. Adding TLS at the backend in this case adds overhead
and a second cert to maintain for no benefit.

## How amplio resolves the cert files

The startup logic in [`cmd/amplio/tls.go`](../cmd/amplio/tls.go):

1. Look for `<data-dir>/cert.pem` AND `<data-dir>/key.pem`. If both exist,
   use them (no shell-out, no environment assumptions).
2. If either is missing, check for `mkcert` on `$PATH`. If present, run
   `mkcert -cert-file ... -key-file ... localhost 127.0.0.1 ::1` and use
   the generated files.
3. If `mkcert` is absent and no cert is configured, fall back to plain
   HTTP. amplio logs the fallback at debug level with a hint pointing at
   this doc.

The cert auto-gen step has SANs `localhost`, `127.0.0.1`, `::1` so the
local CLI clients (`amplio submit`, `amplio notify`) over loopback also
validate cleanly against the cert. The clients skip cert verification
for loopback HTTPS regardless (the connection is local and we're already
trusting the locally-issued cert), but the SANs avoid logspam.

The auth cookie's `Secure` flag is set when amplio is serving HTTPS and
unset otherwise. Modern browsers refuse to send `Secure` cookies over
plain HTTP, so the flag has to track the actual scheme — see
[`internal/server/auth.go`](../internal/server/auth.go).

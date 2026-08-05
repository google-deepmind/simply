# amplio LLM bridges

A **bridge** is any process that speaks the NDJSON protocol below over HTTP. It
lets amplio reach model backends whose SDKs can't (or shouldn't) be linked into
the main binary — e.g. corp-only APIs — without importing their dependencies.
Because the protocol carries amplio's own request and response types (including
`provider_extra`, which is where thinking signatures live), nothing is lost in
translation the way it would be through a foreign schema.

Transports differ only in **who owns the bridge process**. Today: `subprocess:`,
where amplio spawns it and talks over a private unix socket.

```
amplio (Go)  ──spawn──▶  bridge (any language)  ──▶  model backend
  └──── HTTP/NDJSON over a unix socket ────┘
```

## Spec form

```
--llm 'subprocess{bin=/path/to/bridge}:MODEL[?k=v...]'
```

- `bin=` is the bridge executable (it is `exec`'d with `--socket <path>`), and
  the model position is the model — so a spec says which model it is wherever it
  is displayed.
- The rest of the spec is forwarded to the bridge via the `AMPLIO_BRIDGE_SPEC`
  env var (urlencoded) with `model=` always included, and is part of the **reuse
  key**: identical specs share one long-lived subprocess; distinct specs each get
  their own. `bin=` is amplio's and is never forwarded.
- `max_tokens=` is a *client* arg handled by amplio itself, so it is not
  forwarded in the spec — it arrives on every request as the wire field of the
  same name, and two specs differing only in the cap share one subprocess.

Requests carry `X-Amplio-Bridge-Protocol: 1`. A bridge may ignore it; when the
protocol changes incompatibly the number does too, so a bridge that checks it can
fail loudly instead of misreading a payload.

A bridge must **ignore unknown fields and unknown NDJSON line types** — that is
what lets the protocol grow (a keepalive `{"type":"ping"}`, a new request field)
without a flag day between two separately built programs.

Process lifecycle is managed by amplio: it spawns lazily on first use, reuses the
process across runs/goroutines, restarts it on crash (with a one-shot retry of
the in-flight request), and reaps it on exit (plus `Pdeathsig` on Linux so a
bridge never outlives a crashed amplio).

## Protocol (what a bridge must implement)

HTTP/1.0 or 1.1 over the Unix socket given by `--socket`:

### `GET /health` → `200`
Return 200 once you are ready to serve. amplio polls this on startup (up to 15s).

### `POST /generate` → NDJSON stream
Request body (JSON):

```json
{
  "model": "gemini-...",          // the real model id (the spec's model position)
  "max_tokens": 65536,
  "temperature": null,            // or a float
  "system_prompt": "",            // usually empty; system arrives as role:"system" messages
  "messages": [
    {"role": "system|user|assistant|tool_result",
     "content": "...",
     "tool_calls": [{"id": "...", "name": "...", "arguments": "{json}"}],  // assistant
     "tool_call_id": "...",                                                 // tool_result
     "is_error": false,
     "attachments": [{"mime_type": "image/png", "base64_data": "..."}],
     "provider_extra": {}          // opaque round-tripped cargo (e.g. thought signatures)
    }
  ],
  "tools": [{"name": "...", "description": "...", "schema": {json schema}}]
}
```

Respond `200` with `Content-Type: application/x-ndjson` and write one JSON object
per line:

```jsonc
{"type":"delta","text":"partial answer"}            // any subset of fields:
{"type":"delta","thoughts":"partial reasoning"}
{"type":"delta","tool_call_start":{"id":"c1","name":"foo"}}
{"type":"delta","tool_call_delta":{"id":"c1","arguments_delta":"{\"x\":"}}
// ... then exactly one terminator:
{"type":"final","response":{
   "content":"full answer","thoughts":"full reasoning",
   "tool_calls":[{"id":"c1","name":"foo","arguments":"{\"x\":1}"}],
   "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,
            "cache_read_tokens":0,"cache_write_tokens":0},
   "stop_reason":"end_turn",
   "provider_extra":{}
}}
// or, on failure:
{"type":"error","error":"message"}
```

Notes:
- Deltas are for live UI only; the **`final.response` is authoritative** (amplio's
  blocking `Call` ignores deltas and returns `final.response`; `Stream` yields
  deltas and exposes `final.response` afterward). A bridge that doesn't stream may
  emit just the single `final` line.
- `provider_extra` keys must be bridge-namespaced (e.g. `beyond.fc_sigs_b64`).
  amplio persists it on the assistant turn and replays it on the next request's
  matching assistant message, so a bridge can round-trip opaque per-turn state
  (like thought signatures).
- Stream NDJSON incrementally (flush each line). With HTTP/1.0 the stream ends on
  connection close after the `final` line.

## Reference implementation (this repo)

`bridge.py` is a stdlib-only, dependency-free reference: the protocol server plus
an **`echo`** backend (echoes the last user/tool message). It's used by amplio's
tests (via the equivalent Go `cmd/bridgemock`) and for cross-language smoke
testing — and it's a copy-paste starting point for a real bridge in any language.

```bash
amplio ... --llm 'subprocess{bin='"$PWD"'/bridges/bridge.py}:echo-model?backend=echo'
```

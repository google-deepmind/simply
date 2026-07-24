# amplio LLM subprocess bridges

amplio's `subprocess:` LLM provider lets amplio reach model backends whose SDKs
can't (or shouldn't) be linked into the main binary — e.g. corp-only APIs. 
Instead of importing those dependencies, amplio spawns a small **bridge** 
process and talks to it over a tiny HTTP protocol on a Unix domain socket.

```
amplio (Go)  ──spawn──▶  bridge (any language)  ──▶  model backend
  └──── HTTP/NDJSON over a unix socket ────┘
```

## Spec form

```
--llm 'subprocess:/path/to/bridge?model=NAME[&k=v...]'
```

- The path is the bridge executable (it is `exec`'d with `--socket <path>`).
- The query (`?model=...&...`) is forwarded to the bridge via the
  `AMPLIO_BRIDGE_SPEC` env var (urlencoded), and is part of the **reuse key**:
  identical specs share one long-lived subprocess; distinct specs each get their
  own. `?model=` is required; `?max_tokens=` overrides amplio's default cap.

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
  "model": "gemini-...",          // the real model id (from ?model=)
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
  (like Gemini/Beyond thought signatures).
- Stream NDJSON incrementally (flush each line). With HTTP/1.0 the stream ends on
  connection close after the `final` line.

## Reference implementation (this repo)

`bridge.py` is a stdlib-only, dependency-free reference: the protocol server plus
an **`echo`** backend (echoes the last user/tool message). It's used by amplio's
tests (via the equivalent Go `cmd/bridgemock`) and for cross-language smoke
testing — and it's a copy-paste starting point for a real bridge in any language.

```bash
amplio ... --llm 'subprocess:'"$PWD"'/bridges/bridge.py?model=echo-model&backend=echo'
```

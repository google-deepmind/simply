#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference amplio LLM subprocess bridge (echo backend).

amplio's `subprocess:` provider spawns a bridge process and talks to it over
HTTP on a Unix domain socket (path from `--socket`), configured via the
`AMPLIO_BRIDGE_SPEC` env var (the spec's urlencoded `?k=v` query). Protocol:

  GET  /health    -> 200 once ready (amplio polls on startup).
  POST /generate  -> NDJSON: zero+ {"type":"delta",...} then one
                     {"type":"final","response":{...}} (or {"type":"error",...}).

See bridges/README.md for the full wire contract. This file is a dependency-free
reference: the protocol server plus an "echo" backend (echoes the last user/tool
message), used for smoke testing and as a copy-paste starting point for a real
bridge in any language.
"""

import argparse
import json
import os
import socketserver
import sys
import urllib.parse
from http import server as http_server


def echo_backend(spec):
  """Echoes the last user/tool message. `__CRASH__` hard-exits mid-request.

  Optional `context_window=N` spec arg (chars, a rough token proxy) makes the
  backend a COMPACTION TESTER: when the request's total input size exceeds N, it
  returns a context-window-overflow error instead of echoing. amplio's reactive
  compaction then kicks in (its fast-model judge recognizes the "prompt is too
  long" wording), so you can exercise the whole compaction path on a short
  conversation without a real long-context model. Example spec:

    subprocess{bin=/path/to/bridge.py}:echo-model?backend=echo&context_window=2000
  """
  # context_window: 0/absent = unlimited (plain echo); >0 = overflow past N chars.
  context_window = int(spec.get("context_window", ["0"])[0])

  def generate(req):
    last = ""
    total = 0
    for m in req.get("messages", []):
      total += len(m.get("content", ""))
      if m.get("role") in ("user", "tool_result"):
        last = m.get("content", "")
    if "__CRASH__" in last:
      os._exit(1)
    if context_window and total > context_window:
      # Wording matches what real providers return for an over-long prompt, so
      # amplio's context-window judge classifies it as an overflow.
      yield {"type": "error",
             "error": (f"prompt is too long: {total} tokens > {context_window} "
                       "maximum context length")}
      return
    reply = "echo: " + last
    mid = len(reply) // 2
    yield {"type": "delta", "text": reply[:mid]}
    yield {"type": "delta", "text": reply[mid:]}
    yield {"type": "final", "response": {
        "content": reply,
        "usage": {"prompt_tokens": total, "completion_tokens": len(reply),
                  "total_tokens": total + len(reply)},
        "stop_reason": "end_turn",
    }}

  return generate


BACKENDS = {"echo": echo_backend}


class _UnixServer(socketserver.ThreadingUnixStreamServer):
  daemon_threads = True

  def server_bind(self):
    try:
      os.unlink(self.server_address)
    except FileNotFoundError:
      pass
    super().server_bind()


def _make_handler(generate):

  class Handler(http_server.BaseHTTPRequestHandler):
    # HTTP/1.0 so the unbounded NDJSON stream ends cleanly on connection close.
    protocol_version = "HTTP/1.0"

    def address_string(self):  # client_address is a str for unix sockets
      return "unix"

    def log_message(self, *args):
      sys.stderr.write("bridge: " + (args[0] % args[1:]) + "\n")

    def do_GET(self):  # noqa: N802
      if self.path == "/health":
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()
      else:
        self.send_error(404)

    def do_POST(self):  # noqa: N802
      if self.path != "/generate":
        self.send_error(404)
        return
      length = int(self.headers.get("Content-Length", "0"))
      try:
        req = json.loads(self.rfile.read(length) or b"{}")
      except (ValueError, json.JSONDecodeError) as e:
        self.send_error(400, f"bad request: {e}")
        return
      self.send_response(200)
      self.send_header("Content-Type", "application/x-ndjson")
      self.end_headers()
      try:
        for line in generate(req):
          self.wfile.write((json.dumps(line) + "\n").encode("utf-8"))
          self.wfile.flush()
      except Exception as e:  # pylint: disable=broad-except
        try:
          self.wfile.write((json.dumps({"type": "error", "error": str(e)}) + "\n").encode("utf-8"))
          self.wfile.flush()
        except OSError:
          pass

  return Handler


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--socket", required=True)
  args = parser.parse_args()

  spec = urllib.parse.parse_qs(os.environ.get("AMPLIO_BRIDGE_SPEC", ""))
  backend_name = spec.get("backend", ["echo"])[0]
  factory = BACKENDS.get(backend_name)
  if factory is None:
    sys.stderr.write(f"bridge: unknown backend {backend_name!r}; have {list(BACKENDS)}\n")
    sys.exit(2)
  generate = factory(spec)

  server = _UnixServer(args.socket, _make_handler(generate))
  sys.stderr.write(f"bridge: serving on {args.socket} (backend={backend_name})\n")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()

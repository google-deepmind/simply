# Copyright 2026 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plugin registry for `remote_decode_eval` model backends.

`remote_decode_eval` selects its model_fn implementation from the
`--server_address` flag's URI scheme: a `<scheme>:...` address looks up a
factory in `ModelBackendRegistry` keyed by `<scheme>`, and anything that
doesn't match a registered scheme falls through to the built-in legacy
gRPC stub path. The two intentional design points are:

  * Plugins register at module import time as a side-effect (matching the
    same pattern as Simply's `EvaluationRegistry` / `DataSourceRegistry`).
    Binaries opt in to a backend by importing the backend's module (e.g.
    via a `plugin.py` aggregator).

  * The factory owns its URI grammar -- it receives the raw URI body
    (everything after `<scheme>:`) and parses it however it likes. This
    keeps backend-specific schema knowledge out of `remote_decode_eval`.

Sampling knobs (`temperature`, `top_p`, `top_k`, `max_decode_steps`) are
passed verbatim as keyword arguments; backends are free to honor or ignore
them.

Example (concrete plugin, lives outside this module):

    from simply.eval import model_backends

    def _my_backend_factory(uri_body, *, temperature, top_p, top_k,
                            max_decode_steps):
      ...  # parse uri_body, build an async model_fn, return it.

    model_backends.ModelBackendRegistry.register(
        _my_backend_factory, name='my_backend',
    )

Then `--server_address=my_backend:opaque/body/string` is dispatched to it.
"""

from collections.abc import Coroutine, Mapping, Sequence
from typing import Any, Callable, ClassVar

from simply.utils import registry

# Same shape `evaluation.evaluate_async` consumes; mirrors the one defined
# inside the legacy gRPC path in `remote_decode_eval.py`. Typed as a
# coroutine (not just an Awaitable) so callers can pass the result directly
# to `asyncio.run`, which requires a coroutine.
ModelFn = Callable[..., Coroutine[Any, Any, Mapping[str, Any]]]

# Backend factory signature. Receives the raw URI body (everything after
# the `<scheme>:` prefix) plus the standard sampling knobs (which the
# backend is free to honor or ignore). Sampling knobs are positional-only
# keyword-arg semantics so plugins can name them as they wish; pass `None`
# to defer to the backend's own default.
ModelFnFactory = Callable[..., ModelFn]


class ModelBackendRegistry(registry.RootRegistry):
  """Maps a `--server_address` URI scheme to a `ModelFnFactory`.

  In addition to per-scheme entries, this registry recognizes one special
  sentinel name `'_default'`: a factory registered under it is invoked
  whenever `--server_address` has no scheme, has an unknown scheme, or is
  empty -- i.e. all the cases that historically fell through to the legacy
  gRPC stub path in `remote_decode_eval`. Use `register_default(factory)`
  to install one.
  """

  namespace: ClassVar[str] = 'model_backend'


_DEFAULT_BACKEND_NAME = '_default'


def register_default(factory: 'ModelFnFactory') -> 'ModelFnFactory':
  """Registers `factory` as the fallback backend for schemeless/unknown URIs.

  The default factory receives the FULL `--server_address` value as its
  `uri_body` argument (rather than the stripped body), since there is no
  scheme to strip. This matches the legacy `simply_service_stub` contract
  where the flag is a bare BNS / host:port (or empty, signalling work-unit-0
  fallback).

  Args:
    factory: A `ModelFnFactory` to install as the default backend.

  Returns:
    `factory` unchanged, so this function can be used as a decorator.
  """
  ModelBackendRegistry.register(factory, name=_DEFAULT_BACKEND_NAME)
  return factory


def dispatch(
    server_address: str | None,
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_decode_steps: int | None,
) -> ModelFn | None:
  """Returns a backend-built `model_fn` for `server_address`, or None.

  Resolution order:
    1. If `server_address` matches `<scheme>:<body>` AND `<scheme>` is
       registered, call that factory with `(body, **sampling_knobs)`.
    2. Otherwise, if a default backend is registered, call it with the
       FULL `server_address` (which may be empty / schemeless / or carry
       an unknown scheme) as the URI body.
    3. Otherwise return `None` so the caller can take its own fallback.

  Args:
    server_address: The raw `--server_address` flag value (may be None).
    temperature: Sampling temperature; passed verbatim to the backend.
    top_p: Top-p sampling cutoff; passed verbatim.
    top_k: Top-k sampling cutoff; passed verbatim.
    max_decode_steps: Per-call output token cap; passed verbatim.
  """
  kwargs = dict(
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      max_decode_steps=max_decode_steps,
  )
  if server_address:
    scheme, sep, body = server_address.partition(':')
    if sep:
      factory = ModelBackendRegistry.get(scheme, raise_error=False)
      if factory is not None:
        return factory(body, **kwargs)
  default_factory = ModelBackendRegistry.get(
      _DEFAULT_BACKEND_NAME, raise_error=False,
  )
  if default_factory is None:
    return None
  return default_factory(server_address or '', **kwargs)


# Re-export for any caller that wants the type alias for typing-only purposes.
__all__ = [
    'ModelBackendRegistry',
    'ModelFn',
    'ModelFnFactory',
    'dispatch',
    'register_default',
    # Suppress "unused" warnings for typing imports re-used by plugins.
    'Sequence',
    'Mapping',
    'Any',
]

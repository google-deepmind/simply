#!/usr/bin/env python3
"""Generates the Python gRPC stubs for `simply/serving/*.proto`.

The serving stack (`simply.serving.common` and everything importing it) needs
`simply/serving/{struct,server}_pb2{,_grpc}.py`, which are generated rather
than checked in.

Usage:
  pip install ".[serving]"
  python setup/gen_protos.py
"""

import pathlib
import sys

from grpc_tools import protoc

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PROTO_DIR = _REPO_ROOT / 'simply' / 'serving'


def main() -> int:
  protos = sorted(p.relative_to(_REPO_ROOT).as_posix()
                  for p in _PROTO_DIR.glob('*.proto'))
  if not protos:
    print(f'No .proto files found under {_PROTO_DIR}.', file=sys.stderr)
    return 1
  # `-I` is the repo root so the generated imports match the `simply.serving.*`
  # package layout.
  exit_code = protoc.main([
      'protoc',
      f'-I{_REPO_ROOT}',
      f'--python_out={_REPO_ROOT}',
      f'--grpc_python_out={_REPO_ROOT}',
      *protos,
  ])
  if exit_code:
    return exit_code
  print('Generated: ' + ', '.join(
      p.name for p in sorted(_PROTO_DIR.glob('*_pb2*.py'))))
  return 0


if __name__ == '__main__':
  sys.exit(main())

# Copyright 2024 The Simply Authors
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
"""Registry."""

import importlib
import traceback
from typing import Any, ClassVar


def _being_reloaded() -> bool:
  """Returns whether we are being called from importlib.reload."""
  for s in traceback.extract_stack():
    if s.name == 'reload' and s.filename == importlib.__file__:
      return True
    if s.name.endswith('cell_maybe_reload'):
      # ecolab autoreload
      return True
  return False


class RootRegistry:
  """A root registry for name-to-function/class dictionary.

  All registry classes must inherit from this class so that they can be tracked
  at this root place.
  """

  registry: ClassVar[dict[str, Any]] = {}
  OVERWRITE_DUPLICATE: ClassVar[bool] = False
  namespace: ClassVar[str] = ''

  @classmethod
  def reset(cls) -> None:
    cls.registry = {}

  @classmethod
  def fullname(cls, name: str) -> str:
    return f'{cls.namespace}:{name}' if cls.namespace else name

  @classmethod
  def register(cls, fn_or_cls: Any, name: str = '') -> Any:
    """Registers a function or class."""
    if not name:
      name = getattr(fn_or_cls, '__name__')
    if not name:
      raise ValueError(f'name must be specified for {fn_or_cls}')
    fullname = cls.fullname(name)
    if fullname in cls.registry and not cls.OVERWRITE_DUPLICATE:
      if not _being_reloaded():
        raise ValueError(f'Duplicate name: {fullname}')
    cls.registry[fullname] = fn_or_cls
    setattr(fn_or_cls, '__registered_name__', fullname)
    return fn_or_cls

  @classmethod
  def register_value(cls, value: Any, name: str) -> Any:
    """Register a function that returns the given value."""
    cls.register(lambda: value, name)

  @classmethod
  def unregister(cls, name) -> None:
    fullname = cls.fullname(name)
    if fullname in cls.registry:
      del cls.registry[fullname]

  @classmethod
  def get(cls, name: str, raise_error: bool = True) -> Any:
    fullname = cls.fullname(name)
    result = cls.registry.get(fullname)
    if raise_error and result is None:
      raise ValueError(f'Unknown name: {name}, fullname: {fullname}')
    else:
      return result

  @classmethod
  def get_instance(cls, name: str, raise_error: bool = True) -> Any:
    return cls.get(name, raise_error=raise_error)()


class FunctionRegistry(RootRegistry):
  """A simple registry for name-to-function dictionary."""

  namespace: ClassVar[str] = 'Function'

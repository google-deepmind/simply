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
"""Common utils for Simply serving."""

from typing import Any

from simply.serving import struct_pb2
from simply.utils import common
from simply.utils import pytree


def json_to_struct_pb(jtree: common.PyTree) -> struct_pb2.Value:
  """Converts a json-like tree to a struct_pb2.Value."""
  res = struct_pb2.Value()
  if pytree.tree_is_mapping(jtree):
    for k, v in jtree.items():
      res.struct_value.fields[k].CopyFrom(json_to_struct_pb(v))
  elif pytree.tree_is_sequence(jtree):
    for v in jtree:
      res.list_value.values.append(json_to_struct_pb(v))
  elif jtree is None:
    res.null_value = struct_pb2.NULL_VALUE
  elif isinstance(jtree, float):
    res.number_value = jtree
  elif isinstance(jtree, str):
    res.string_value = jtree
  elif isinstance(jtree, bool):
    res.bool_value = jtree
  elif isinstance(jtree, int):
    res.int64_value = jtree
  else:
    raise ValueError(f'Unsupported type: {type(jtree)}')
  return res


def struct_pb_to_json(struct: struct_pb2.Value) -> common.PyTree:
  """Converts a struct_pb2.Value to a json-like tree."""
  match struct.WhichOneof('kind'):
    case 'struct_value':
      return {
          k: struct_pb_to_json(v) for k, v in struct.struct_value.fields.items()
      }
    case 'list_value':
      return [struct_pb_to_json(v) for v in struct.list_value.values]
    case 'null_value':
      return None
    case 'number_value':
      return struct.number_value
    case 'string_value':
      return struct.string_value
    case 'bool_value':
      return struct.bool_value
    case 'int64_value':
      return struct.int64_value
    case _:
      raise ValueError(f'Unsupported type: {struct}')


def py_to_struct_pb(py: Any) -> struct_pb2.Value:
  """Converts a python object to a struct_pb2.Value."""
  return json_to_struct_pb(pytree.dump(py))


def struct_pb_to_py(struct: struct_pb2.Value) -> Any:
  """Converts a struct_pb2.Value to a python object."""
  return pytree.load(struct_pb_to_json(struct))

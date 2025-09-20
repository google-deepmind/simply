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
"""Checkpoint library for Simply."""

import abc
from collections import OrderedDict  # pylint: disable=g-importing-member
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import inspect
import logging
import math
import os
import pydoc
import re
import time
from typing import Any, cast, ClassVar, final, Tuple, TypeAlias

from etils import epath
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_manager as ocp_constants

from simply.utils import common
from simply.utils import module
from simply.utils import pytree
from simply.utils import registry
from simply.utils import sharding as sharding_lib

PyTree = common.PyTree

ArgsTree: TypeAlias = Mapping[
    str, ocp.args.CheckpointArgs | Mapping[str, 'ArgsTree']
]

CHECKPOINT_FORMAT_KEY = '__checkpoint_format__'
DATA_ITEM_NAME = 'data'


class CheckpointFormat(abc.ABC):
  """Checkpoint formats for Simply."""

  @final
  def __post_init__(self):
    if not dataclasses.is_dataclass(self):
      raise ValueError(
          f'CheckpointFormat must be a dataclass. {self.__class__.__name__} is'
          ' not.'
      )
    if not CheckpointFormatRegistry.get(self.__class__.__name__):
      raise ValueError(
          'CheckpointFormatRegistry'
          f' {CheckpointFormatRegistry.fullname(self.__class__.__name__)} is'
          ' not registered.'
      )

  def transforms(
      self,
      restore_args: ArgsTree,
      original_metadata: ocp.metadata.Metadata | None = None,
  ) -> Mapping[str, ocp.RestoreTransform]:
    """Returns the transforms for the checkpoint format."""
    del restore_args, original_metadata
    return {}


class CheckpointFormatRegistry(registry.FunctionRegistry):
  """Registry for checkpoint formats."""

  namespace: ClassVar[str] = 'CheckpointFormat'


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class LegacyFormat(CheckpointFormat):
  """The legacy checkpoint format this project is using at the beginning."""


def permute_tuple(tup: Tuple[Any, ...], perm: str) -> Tuple[Any, ...]:
  """Permute an integer sequence."""
  left, right = perm.split('->')
  left = [c for c in left if c]
  right = [c for c in right if c]
  if len(tup) != len(left):
    raise ValueError(f'Tuple {tup} does not match permutation {perm}.')
  left_axis_mapping = {c: i for i, c in enumerate(left)}
  new_seq = []
  for c in right:
    if (i := left_axis_mapping.get(c)) is None:
      raise ValueError(f'Permutation {perm} is not valid.')
    new_seq.append(tup[i])
  return tuple(new_seq)


def permute_arg(arg: ocp.ArrayRestoreArgs, perm: str) -> ocp.ArrayRestoreArgs:
  """Permutes an ArrayRestoreArgs."""
  new_global_shape = permute_tuple(arg.global_shape, perm)
  new_shape = permute_tuple(arg.shape, perm)
  replace_kwargs = dict(global_shape=new_global_shape, shape=new_shape)
  if arg.sharding:
    assert isinstance(arg.sharding, jax.sharding.NamedSharding), (
        f'Unrecognized sharding type for ArrayRestoreArgs: {arg}. Check if'
        ' array and mesh shapes are compatible.'
    )
    spec = arg.sharding.spec
    replace_kwargs['sharding'] = sharding_lib.mesh_sharding(
        permute_tuple(spec + (None,) * (len(arg.shape) - len(spec)), perm),
        arg.sharding.mesh,
    )
  return dataclasses.replace(arg, **replace_kwargs)


def stack_arg(
    arg: ocp.ArrayRestoreArgs, stacked_size: int
) -> ocp.ArrayRestoreArgs:
  """Stacks an ArrayRestoreArgs."""
  new_global_shape = (stacked_size, *arg.global_shape)
  new_shape = (stacked_size, *arg.shape)
  replace_kwargs = dict(global_shape=new_global_shape, shape=new_shape)
  if arg.sharding:
    assert isinstance(arg.sharding, jax.sharding.NamedSharding)
    spec = arg.sharding.spec
    replace_kwargs['sharding'] = sharding_lib.mesh_sharding(
        [None, *spec], arg.sharding.mesh
    )
  return dataclasses.replace(arg, **replace_kwargs)


def unstack_arg(arg: ocp.ArrayRestoreArgs) -> ocp.ArrayRestoreArgs:
  """Unstacks an ArrayRestoreArgs."""
  new_global_shape = arg.global_shape[1:]
  new_shape = arg.shape[1:]
  replace_kwargs = dict(global_shape=new_global_shape, shape=new_shape)
  if arg.sharding:
    assert isinstance(arg.sharding, jax.sharding.NamedSharding)
    spec = arg.sharding.spec
    replace_kwargs['sharding'] = sharding_lib.mesh_sharding(
        spec[1:], arg.sharding.mesh
    )
  return dataclasses.replace(arg, **replace_kwargs)


class GenericFormat(CheckpointFormat):
  r"""Generic checkpoint format.

  Users can define how checkpoint is converted by providing the rules. Each rule
  is a tuple of (restored path regex, original path template, arg functions,
  values functions) in strings.

  Take the following rule as an example:
  ```
  (
      r'(.*)/repeated_blocks/block_(\d+)/attn/kv_proj',  # Rkdnh
      r'{prefix}/layer_{index}/attn/kv_einsum/w',  # kndh
      'permute Rkdnh->kndh',
      'permute kdnh->kndh; stack',
  )
  ```

  The restored path regex is used to match paths in the final restored output.
  It may matches multiple paths, like
  ```
  'params/repeated_blocks/block_0/attn/kv_proj'
  'params/repeated_blocks/block_1/attn/kv_proj'
  'm/repeated_blocks/block_0/attn/kv_proj'
  'm/repeated_blocks/block_1/attn/kv_proj'
  'v/repeated_blocks/block_0/attn/kv_proj'
  'v/repeated_blocks/block_1/attn/kv_proj'
  ```

  At the time `params/repeated_blocks/block_0/attn/kv_proj` is matched, the
  original path template may generate
  ```
  'params/transformer/layer_0/attn/kv_einsum/w'
  'params/transformer/layer_2/attn/kv_einsum/w'
  'params/transformer/layer_4/attn/kv_einsum/w'
  ```
  where `{prefix}` indicates `params/transformer`, `{index}` indicates `0,2,4`.
  These indications are defined by `_replace_prefix` and `_replace_index`. The
  replacement functions follows the signature of
  `_replace_<name>(match: re.Match[str], a, b, ...) -> Sequence[str]`, where
  `a`, `b` take in the corresponding output items of `prepare_global_kwargs`.

  The arg functions are used to convert the restored args to the original args,
  so that `restore()` can know how to read the original checkpoint values.
  Functions are separated by `;`, and each function is a space-separated list of
  arguments following a function name. Arg functions follows the signature of
  `_arg_<name>(arg: ocp.ArrayRestoreArgs, a, b, ...) -> ocp.ArrayRestoreArgs`,
  where `arg` is the restored arg for the matched path, and `a`, `b` take in the
  positional corresponding arguements. In this example, `permute Rkdnh->kndh`
  calls `_arg_permute(arg, 'Rkdnh->kndh')`, where `arg` is the restored arg for
  `params/repeated_blocks/block_0/attn/kv_proj`, and the returned arg is used
  for loading `params/transformer/layer_(0/2/4)/attn/kv_einsum/w`.

  The values functions are used to convert the original values to the restored
  values. Functions are separated by `;`, and each function is a space-separated
  list of arguments following a function name. Values functions follow the
  signature of `_values_<name>(values: OrderedDict[str, jax.Array], a, b, ...)
  -> OrderedDict[str, jax.Array]`, where `values` is the orignal values with
  orignal path template parameters as key, and `a`, `b` take in the positional
  corresponding arguments. OrderedDict is used to preserve the order of the
  values so that we can stack the values in the correct order. In this example,
  `permute kdnh->kndh; stack` calls `_values_permute(values, 'kdnh->kndh')`
  first and then calls `_values_stack(values)`, where `values` is:
  ```
  {
      'prefix=params/transformer,index=0': ...  # original checkpoint value at
          # 'params/transformer/layer_0/attn/kv_einsum/w',
      'prefix=params/transformer,index=2':  ...  # original checkpoint value at
          # 'params/transformer/layer_2/attn/kv_einsum/w',
      'prefix=params/transformer,index=4':  ...  # original checkpoint value at
          # 'params/transformer/layer_4/attn/kv_einsum/w',
  }
  ```

  Supported Arg Functions:
  - _arg_stack: Adds a leading dimension of `stacked_size` to the restored arg.
  - _arg_permute: Permutes the dimension order of the restored arg.
  - _arg_unstack: Removes the leading dimension of the restored arg.

  Supported Values Functions:
  - _values_take: Takes the `index`-th value from the values.
  - _values_stack: Stacks the values by path template parameter `name`.
  - _values_permute: Permutes the dimension order of the values.
  """

  REPLACE_FUNC_PREFIX: ClassVar[str] = '_replace_'
  ARG_FUNC_PREFIX: ClassVar[str] = '_arg_'
  VALUES_FUNC_PREFIX: ClassVar[str] = '_values_'

  @property
  @abc.abstractmethod
  def rules(self) -> Sequence[Tuple[str, ...]]:
    """Returns the rules for the checkpoint format."""

  @dataclasses.dataclass
  class CompiledRule:
    """Compiled rule for checkpoint format.

    Attributes:
      rpath_re: The regex pattern for the restored path.
      opath_tmpl: The template for the original path.
      arg_funcs: The functions to process the restored args.
      values_funcs: The functions to process the restored values.
    """

    rpath_re: re.Pattern[str]
    opath_tmpl: str
    arg_funcs: str = ''
    values_funcs: str = ''

  @dataclasses.dataclass(frozen=True)
  class CompiledFunction:
    """Compiled function for checkpoint format.

    Attributes:
      func: The function to be called.
      kwargs: The keyword arguments to be passed to the function.
    """

    func: Callable[..., Any]
    kwargs: Mapping[str, Any]

    def __call__(self, *args, **kwargs):
      params = list(inspect.signature(self.func).parameters.values())
      if len(args) + len(self.kwargs) > len(params):
        raise ValueError(
            f'Function call `{self.func}` has too many arguments for method'
            f' `{self.func}` with parameters `{params}`.'
        )
      final_kwargs = {**self.kwargs}
      for i in range(len(args), len(params)):
        pname = params[i].name
        if pname not in final_kwargs:
          if pname in kwargs:
            final_kwargs[pname] = kwargs[pname]
          elif params[i].default is inspect.Parameter.empty:
            raise ValueError(
                f'Function call `{self.func}` is missing argument `{pname}`.'
            )
      return self.func(*args, **final_kwargs)

  @functools.cached_property
  def compiled_rules(self) -> Sequence[CompiledRule]:
    """Returns the compiled rules for the checkpoint format."""
    compiled = []
    for rule in self.rules:
      if len(rule) not in (2, 4):
        raise ValueError(f'Invalid rule: {rule}')
      rpath_re = re.compile(rule[0])
      opath_tmpl = rule[1]
      compiled_rule = self.CompiledRule(rpath_re, opath_tmpl)
      if len(rule) == 4:
        compiled_rule.arg_funcs = rule[2]
        compiled_rule.values_funcs = rule[3]
      compiled.append(compiled_rule)
    return compiled

  def _compile_functions(
      self, funcs: str, func_prefix: str
  ) -> Sequence[CompiledFunction]:
    compiled = []
    for func in funcs.split(';'):
      if func := func.strip():
        compiled.append(self._compile_function(func, func_prefix))
    return compiled

  def _compile_function(
      self, func: str, func_prefix: str = ''
  ) -> CompiledFunction:
    """Compiles a command for checkpoint format."""
    items = [item.strip() for item in func.split(' ') if item.strip()]
    args = items[1:]
    method = getattr(self, func_prefix + items[0])
    param_map = inspect.signature(method).parameters
    params = list(param_map.values())[1:]
    if len(args) > len(params):
      raise ValueError(
          f'Function call `{func}` has too many arguments for method'
          f' `{method}` with parameters `{param_map}`.'
      )
    kwargs = {}
    for i, arg in enumerate(args):
      param = params[i]
      kwargs[param.name] = (
          arg
          if param.annotation is inspect.Parameter.empty
          else param.annotation(arg)
      )
    return self.CompiledFunction(method, kwargs)

  def _arg_stack(
      self, arg: ocp.ArrayRestoreArgs, stacked_size: int
  ) -> ocp.ArrayRestoreArgs:
    return stack_arg(arg, stacked_size=stacked_size)

  def _arg_permute(
      self, arg: ocp.ArrayRestoreArgs, perm: str
  ) -> ocp.ArrayRestoreArgs:
    return permute_arg(arg, perm)

  def _arg_unstack(self, arg: ocp.ArrayRestoreArgs) -> ocp.ArrayRestoreArgs:
    return unstack_arg(arg)

  def _values_take(
      self, values: OrderedDict[str, jax.Array], index: int
  ) -> OrderedDict[str, jax.Array]:
    return OrderedDict((k, v[index]) for k, v in values.items())

  def _values_stack(
      self, values: OrderedDict[str, jax.Array], name: str = ''
  ) -> OrderedDict[str, jax.Array]:
    """Stacks the values."""
    if not name:
      return OrderedDict({'': jnp.stack(list(values.values()))})
    res = OrderedDict()
    for key, value in values.items():
      # key is in the format of something like 'a=1,b=2', if 'a' is the name to
      # be stacked, then we will remove it from the key and it becomes 'b=2'.
      new_key = ','.join(
          n for n in key.split(',') if not n.startswith(name + '=')
      )
      if list_to_append := res.get(new_key):
        list_to_append.append(value)
      else:
        res[new_key] = [value]
    return OrderedDict((k, jnp.stack(v)) for k, v in res.items())

  def _values_permute(
      self, values: OrderedDict[str, jax.Array], perm: str
  ) -> OrderedDict[str, jax.Array]:
    return OrderedDict((k, jnp.einsum(perm, v)) for k, v in values.items())

  def prepare_global_kwargs(
      self,
      restore_args: ArgsTree,
      original_metadata: ocp.metadata.Metadata | None = None,
  ) -> Mapping[str, Any]:
    """Prepares global kwargs for replace/arg/values functions."""
    del restore_args, original_metadata
    return {}

  def transforms(
      self,
      restore_args: ArgsTree,
      original_metadata: ocp.metadata.Metadata | None = None,
  ) -> Mapping[str, ocp.RestoreTransform]:
    global_kwargs = self.prepare_global_kwargs(restore_args, original_metadata)

    transforms = {}
    flatten_restore_args = ocp.tree.to_flat_dict(restore_args, sep='/')
    for path, arg in flatten_restore_args.items():
      assert isinstance(arg, ocp.ArrayRestoreArgs)
      for rule in self.compiled_rules:
        if match := rule.rpath_re.fullmatch(path):
          # Step 1: Get orignial path template.
          opath_tmpl_str = rule.rpath_re.sub(rule.opath_tmpl, path)
          opath_tmpl_params = {}
          for pname in common.ParameterizedString.parameter_names(
              opath_tmpl_str
          ):
            func = getattr(self, self.REPLACE_FUNC_PREFIX + pname)
            func = self.CompiledFunction(func, {})
            opath_tmpl_params[pname] = func(match, **global_kwargs)
          opath_tmpl = common.ParameterizedString(
              opath_tmpl_str, opath_tmpl_params
          )

          # Step 2: Convert the restored args to the original args.
          for afunc in self._compile_functions(
              rule.rpath_re.sub(rule.arg_funcs, path),
              func_prefix=self.ARG_FUNC_PREFIX,
          ):
            arg = afunc(arg, **global_kwargs)
          input_args = {}
          for kwargs in opath_tmpl:
            opath = opath_tmpl.format(**kwargs)
            input_args[opath] = arg

          # Step 3: Convert the original values to the restored values.
          def _multi_value_fn(
              key,
              tree,
              arg,
              opath_tmpl=opath_tmpl,
              rule=rule,
          ):
            del arg
            flatten_tree = ocp.tree.to_flat_dict(tree, sep='/')
            values = OrderedDict()
            for params in opath_tmpl:
              # Convert parameters to a string key, e.g. 'a=1,b=2'.
              new_key = ','.join(
                  f'{k}={params[k]}' for k in opath_tmpl.available_parameters
              )
              values[new_key] = flatten_tree[opath_tmpl.format(**params)]
            for vfunc in self._compile_functions(
                rule.rpath_re.sub(rule.values_funcs, key),
                func_prefix=self.VALUES_FUNC_PREFIX,
            ):
              values = vfunc(values, **global_kwargs)
            if len(values) != 1:
              raise ValueError(
                  f'{list(values.keys())} is not aggregated for {key}.'
              )
            return next(iter(values.values()))

          transforms[path] = ocp.RestoreTransform(
              multi_value_fn=_multi_value_fn,
              multi_value_fn_input_args=input_args,
          )

    return transforms


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class V1Format(GenericFormat):
  """V1 checkpoint format (DEPRECATED)."""

  def prepare_global_kwargs(
      self,
      restore_args: ArgsTree,
      original_metadata: ocp.metadata.Metadata | None = None,
  ) -> Mapping[str, Any]:
    kwargs = {}
    repeated_regex = re.compile(r'(.*)/repeated_blocks/block_(\d+)/(.*)')
    flatten_original_metadata = ocp.tree.to_flat_dict(
        getattr(original_metadata, 'tree'), sep='/'
    )
    block_set = set()
    for path, metadata in flatten_original_metadata.items():
      if repeated_match := repeated_regex.fullmatch(path):
        assert isinstance(metadata, ocp.metadata.ArrayMetadata)
        block_i = int(repeated_match.group(2))
        block_set.add(block_i)
        kwargs['n_repeats'] = metadata.shape[0]
    kwargs['n_blocks_per_repeat'] = len(block_set)
    return kwargs

  @property
  def rules(self):
    return [
        (
            r'(.*)/block_(\d+)/(.*)',
            r'\1/repeated_blocks/block_{index}/\3',
            'stack_block',
            r'take_block \2',
        ),
    ]

  def _replace_index(
      self, match: re.Match[str], n_blocks_per_repeat: int
  ) -> Sequence[str]:
    i = int(match.group(2))
    return [str(i % n_blocks_per_repeat)]

  def _arg_stack_block(
      self, arg: ocp.ArrayRestoreArgs, n_repeats: int
  ) -> ocp.ArrayRestoreArgs:
    return stack_arg(arg, stacked_size=n_repeats)

  def _values_take_block(
      self,
      values: OrderedDict[str, jax.Array],
      index: int,
      n_blocks_per_repeat: int,
  ) -> OrderedDict[str, jax.Array]:
    return OrderedDict(
        (k, v[index // n_blocks_per_repeat]) for k, v in values.items()
    )


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2Format(GenericFormat):
  """Gemma2 checkpoint format."""

  @property
  def rules(self) -> Sequence[Tuple[str, ...]]:
    return [
        (
            r'(.*)/embed',  # vd
            r'{prefix}/embedder/input_embedding',  # vd
        ),
        (  # For backward compatibility.
            r'(.*)/output_layer/b',  # b
            r'{prefix}/embedder/output_bias',  # b
        ),
        (
            r'(.*)/final_ln/(.*)',  # s
            r'{prefix}/final_norm/\2',  # s
        ),
        (
            r'(.*)/block_(\d+)/attn/qkv_proj',  # kdnh
            r'{prefix}/layer_\2/attn/qkv_einsum/w',  # kndh
            'permute kdnh->kndh',
            'permute kndh->kdnh',
        ),
        (
            r'(.*)/block_(\d+)/attn/kv_proj',  # kdnh
            r'{prefix}/layer_\2/attn/kv_einsum/w',  # kndh
            'permute kdnh->kndh',
            'permute kndh->kdnh',
        ),
        (
            r'(.*)/block_(\d+)/attn/o_proj',  # dnh
            r'{prefix}/layer_\2/attn/attn_vec_einsum/w',  # nhd
            'permute dnh->nhd',
            'permute nhd->dnh',
        ),
        (
            r'(.*)/block_(\d+)/attn/q_proj',  # dnh
            r'{prefix}/layer_\2/attn/q_einsum/w',  # ndh
            'permute dnh->ndh',
            'permute ndh->dnh',
        ),
        (  # For backward compatibility.
            r'(.*)/block_(\d+)/attn/per_dim_scale/scale',  # b
            r'{prefix}/layer_\2/attn/query_per_dim_scale/scale',  # b
        ),
        (
            r'(.*)/block_(\d+)/attn/q_norm/(.*)',  # s
            r'{prefix}/layer_\2/attn/_query_norm/\3',  # s
        ),
        (
            r'(.*)/block_(\d+)/attn/k_norm/(.*)',  # s
            r'{prefix}/layer_\2/attn/_key_norm/\3',  # s
        ),
        (
            r'(.*)/block_(\d+)/ffn_0/(.*)',  # df or b
            r'{prefix}/layer_\2/mlp/gating_einsum/\3',  # 2df or 2b
            'stack 2',
            'take 1',
        ),
        (
            r'(.*)/block_(\d+)/ffn_0_gate/(.*)',  # df or b
            r'{prefix}/layer_\2/mlp/gating_einsum/\3',  # 2df or 2b
            'stack 2',
            'take 0',
        ),
        (
            r'(.*)/block_(\d+)/ffn_1/(.*)',  # fd or b
            r'{prefix}/layer_\2/mlp/linear/\3',  # fd or b
        ),
        (
            r'(.*)/block_(\d+)/post_ln_0/(.*)',  # s
            r'{prefix}/layer_\2/post_attention_norm/\3',  # s
        ),
        (
            r'(.*)/block_(\d+)/post_ln_1/(.*)',  # s
            r'{prefix}/layer_\2/post_ffw_norm/\3',  # s
        ),
        (
            r'(.*)/block_(\d+)/pre_ln_0/(.*)',  # s
            r'{prefix}/layer_\2/pre_attention_norm/\3',  # s
        ),
        (
            r'(.*)/block_(\d+)/pre_ln_1/(.*)',  # s
            r'{prefix}/layer_\2/pre_ffw_norm/\3',  # s
        ),
        (
            'steps',
            'step_on_device',
        ),
    ]

  @property
  def prefix_mapping(self):
    return {
        'params': 'params/transformer',
        'm': 'opt_state/1/0/mu/transformer',
        'v': 'opt_state/1/0/nu/transformer',
    }

  def _replace_prefix(self, match: re.Match[str]) -> Sequence[str]:
    return [self.prefix_mapping[match.group(1)]]


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2TransposeFormat(Gemma2Format):
  """Gemma2 checkpoint format with transposed ffn weights."""

  @property
  def rules(self) -> Sequence[Tuple[str, ...]]:
    new_rules = []
    for rule in super().rules:
      if '/ffn_0' not in rule[0]:
        new_rules.append(rule)
    new_rules.extend([
        (
            r'(.*)/block_(\d+)/ffn_0/w',  # df
            r'{prefix}/layer_\2/mlp/gating_einsum/w',  # 2fd
            'permute df->fd; stack 2',
            'take 1; permute fd->df',
        ),
        (  # For backward compatibility.
            r'(.*)/block_(\d+)/ffn_0/b',  # b
            r'{prefix}/layer_\2/mlp/gating_einsum/b',  # 2b
            'stack 2',
            'take 1',
        ),
        (
            r'(.*)/block_(\d+)/ffn_0_gate/w',  # Rdf
            r'{prefix}/layer_\2/mlp/gating_einsum/w',  # 2fd
            'permute df->fd; stack 2',
            'take 0; permute fd->df',
        ),
        (  # For backward compatibility.
            r'(.*)/block_(\d+)/ffn_0_gate/b',  # b
            r'{prefix}/layer_\2/mlp/gating_einsum/b',  # 2b
            'stack 2',
            'take 0',
        ),
    ])
    return new_rules


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma3pLegacyFormat(Gemma2Format):
  """Gemma third-party checkpoint format without transposed ffn weights."""

  @property
  def prefix_mapping(self):
    return {'params': 'transformer'}


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma3pFormat(Gemma2TransposeFormat):
  """Gemma third-party checkpoint format with transposed ffn weights."""

  @property
  def prefix_mapping(self):
    return {'params': 'transformer'}


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen2Format(GenericFormat):
  """Qwen2 checkpoint format."""

  def prepare_global_kwargs(
      self,
      restore_args: ArgsTree,
      original_metadata: ocp.metadata.Metadata | None = None,
  ) -> Mapping[str, Any]:
    q_proj_arg = restore_args['params']['block_0']['attn']['q_proj']
    q_proj_arg = cast(ocp.ArrayRestoreArgs, q_proj_arg)
    model_dim, n_heads, per_head_dim = q_proj_arg.global_shape
    return dict(
        model_dim=model_dim,
        n_heads=n_heads,
        per_head_dim=per_head_dim,
    )

  def _arg_merge(
      self, arg: ocp.ArrayRestoreArgs, start: int, end: int
  ) -> ocp.ArrayRestoreArgs:
    """Merges dimensions [start, end)."""
    new_global_shape = (
        *arg.global_shape[:start],
        math.prod(arg.global_shape[start:end]),
        *arg.global_shape[end:],
    )
    new_shape = (
        *arg.shape[:start],
        math.prod(arg.shape[start:end]),
        *arg.shape[end:],
    )

    replace_kwargs = dict(global_shape=new_global_shape, shape=new_shape)
    if arg.sharding:
      assert isinstance(arg.sharding, jax.sharding.NamedSharding)
      spec = arg.sharding.spec
      spec = spec + (None,) * (len(arg.shape) - len(spec))
      range_spec = []
      for i in range(start, end):
        if pytree.tree_is_sequence(spec[i]):
          range_spec.extend(spec[i])
        else:
          range_spec.append(spec[i])
      range_spec = list(filter(lambda x: x is not None, range_spec))
      if len(range_spec) == 1 and range_spec[0] is None:
        range_spec = None
      spec = [*spec[:start], range_spec, *spec[end:]]
      replace_kwargs['sharding'] = jax.sharding.NamedSharding(
          arg.sharding.mesh, jax.sharding.PartitionSpec(*spec)
      )
    return dataclasses.replace(arg, **replace_kwargs)

  def _values_splithead(
      self,
      values: OrderedDict[str, Any],
      axis: int,
      per_head_dim: int,
  ) -> OrderedDict[str, Any]:
    new_values = OrderedDict()
    for k, v in values.items():
      new_shape = (*v.shape[:axis], -1, per_head_dim, *v.shape[axis + 1 :])
      new_values[k] = jnp.reshape(v, new_shape)
    return new_values

  @property
  def rules(self) -> Sequence[Tuple[str, ...]]:
    return [
        (
            r'params/embed',  # vd
            r'model.embed_tokens.weight',  # vd
        ),
        (
            r'params/output_layer/w',  # dv
            r'lm_head.weight',  # vd
            'permute dv->vd',
            'permute vd->dv',
        ),
        (
            r'params/final_ln/scale',  # s
            r'model.norm.weight',  # s
        ),
        (
            r'params/block_(\d+)/attn/q_norm/scale',  # s
            r'model.layers.\1.self_attn.q_norm.weight',  # s
        ),
        (
            r'params/block_(\d+)/attn/k_norm/scale',  # s
            r'model.layers.\1.self_attn.k_norm.weight',  # s
        ),
        (
            r'params/block_(\d+)/attn/([qkv])_proj',  # dnh
            r'model.layers.\1.self_attn.\2_proj.weight',  # (nh)d
            'permute dnh->nhd; merge 0 2',
            'splithead 0; permute nhd->dnh',
        ),
        (
            r'params/block_(\d+)/attn/([qkv])_bias',  # nh
            r'model.layers.\1.self_attn.\2_proj.bias',  # (nh)
            'merge 0 2',
            'splithead 0',
        ),
        (
            r'params/block_(\d+)/attn/o_proj',  # dnh
            r'model.layers.\1.self_attn.o_proj.weight',  # d(nh)
            'merge 1 3',
            'splithead 1',
        ),
        (
            r'params/block_(\d+)/ffn_0/w',  # df
            r'model.layers.\1.mlp.up_proj.weight',  # fd
            'permute df->fd',
            'permute fd->df',
        ),
        (
            r'params/block_(\d+)/ffn_0_gate/w',  # df
            r'model.layers.\1.mlp.gate_proj.weight',  # fd
            'permute df->fd',
            'permute fd->df',
        ),
        (
            r'params/block_(\d+)/ffn_1/w',  # fd
            r'model.layers.\1.mlp.down_proj.weight',  # df
            'permute fd->df',
            'permute df->fd',
        ),
        (
            r'params/block_(\d+)/pre_ln_0/scale',  # s
            r'model.layers.\1.input_layernorm.weight',  # s
        ),
        (
            r'params/block_(\d+)/pre_ln_1/scale',  # s
            r'model.layers.\1.post_attention_layernorm.weight',  # s
        ),
    ]


# TODO: Add unit tests when the format is finalized.
@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class GemmaStackedFormat(V1Format, Gemma2TransposeFormat):
  """Gemma stacked checkpoint format."""

  def prepare_global_kwargs(
      self,
      restore_args: ArgsTree,
      original_metadata: ocp.metadata.Metadata | None = None,
  ) -> Mapping[str, Any]:
    kwargs = {}
    repeated_regex = re.compile(
        r'(.*)/stacked_layers/attention_type_(\d+)/(.*)'
    )
    flatten_original_metadata = ocp.tree.to_flat_dict(
        getattr(original_metadata, 'tree'), sep='/'
    )
    block_set = set()
    for path, metadata in flatten_original_metadata.items():
      if repeated_match := repeated_regex.fullmatch(path):
        assert isinstance(metadata, ocp.metadata.ArrayMetadata)
        block_i = int(repeated_match.group(2))
        block_set.add(block_i)
        kwargs['n_repeats'] = metadata.shape[0]
    kwargs['n_blocks_per_repeat'] = len(block_set)
    return kwargs

  @property
  def rules(self) -> Sequence[Tuple[str, ...]]:
    # pylint: disable=line-too-long
    return [
        (
            r'(.*)/embed',  # vd
            r'{prefix}/embedder/input_embedding',  # vd
        ),
        (
            r'(.*)/final_ln/(.*)',  # s
            r'{prefix}/final_norm/\2',  # s
        ),
        (
            r'(.*)/block_(\d+)/attn/qkv_proj',  # kdnh
            r'{prefix}/stacked_layers/attention_type_{index}/attn/qkv_einsum/w',  # Rkndh
            'permute kdnh->kndh; stack_block',
            r'take_block \2; permute kndh->kdnh',
        ),
        (
            r'(.*)/block_(\d+)/attn/kv_proj',  # kdnh
            r'{prefix}/stacked_layers/attention_type_{index}/attn/kv_einsum/w',  # Rkndh
            'permute kdnh->kndh; stack_block',
            r'take_block \2; permute kndh->kdnh',
        ),
        (
            r'(.*)/block_(\d+)/attn/o_proj',  # dnh
            r'{prefix}/stacked_layers/attention_type_{index}/attn/attn_vec_einsum/w',  # Rnhd
            'permute dnh->nhd; stack_block',
            r'take_block \2; permute nhd->dnh',
        ),
        (
            r'(.*)/block_(\d+)/attn/q_proj',  # dnh
            r'{prefix}/stacked_layers/attention_type_{index}/attn/q_einsum/w',  # Rndh
            'permute dnh->ndh; stack_block',
            r'take_block \2; permute ndh->dnh',
        ),
        (
            r'(.*)/block_(\d+)/attn/q_norm/(.*)',  # s
            r'{prefix}/stacked_layers/attention_type_{index}/attn/query_norm/\3',  # Rs
            'stack_block',
            r'take_block \2',
        ),
        (
            r'(.*)/block_(\d+)/attn/k_norm/(.*)',  # s
            r'{prefix}/stacked_layers/attention_type_{index}/attn/key_norm/\3',  # Rs
            'stack_block',
            r'take_block \2',
        ),
        (
            r'(.*)/block_(\d+)/ffn_0/(.*)',  # df
            r'{prefix}/stacked_layers/attention_type_{index}/mlp/gating_einsum/\3',  # R2fd
            'permute df->fd; stack 2; stack_block',
            r'take_block \2; take 1; permute fd->df',
        ),
        (
            r'(.*)/block_(\d+)/ffn_0_gate/(.*)',  # df
            r'{prefix}/stacked_layers/attention_type_{index}/mlp/gating_einsum/\3',  # R2fd
            'permute df->fd; stack 2; stack_block',
            r'take_block \2; take 0; permute fd->df',
        ),
        (
            r'(.*)/block_(\d+)/ffn_1/(.*)',  # fd or b
            r'{prefix}/stacked_layers/attention_type_{index}/mlp/linear/\3',  # Rfd or Rb
            'stack_block',
            r'take_block \2',
        ),
        (
            r'(.*)/block_(\d+)/post_ln_0/(.*)',  # s
            r'{prefix}/stacked_layers/attention_type_{index}/post_attention_norm/\3',  # Rs
            'stack_block',
            r'take_block \2',
        ),
        (
            r'(.*)/block_(\d+)/post_ln_1/(.*)',  # s
            r'{prefix}/stacked_layers/attention_type_{index}/post_ffw_norm/\3',  # Rs
            'stack_block',
            r'take_block \2',
        ),
        (
            r'(.*)/block_(\d+)/pre_ln_0/(.*)',  # s
            r'{prefix}/stacked_layers/attention_type_{index}/pre_attention_norm/\3',  # Rs
            'stack_block',
            r'take_block \2',
        ),
        (
            r'(.*)/block_(\d+)/pre_ln_1/(.*)',  # s
            r'{prefix}/stacked_layers/attention_type_{index}/pre_ffw_norm/\3',  # Rs
            'stack_block',
            r'take_block \2',
        ),
        (
            'steps',
            'step_on_device',
        ),
    ]
    # pylint: enable=line-too-long


def readonly_checkpoint_manager(ckpt_dir: str):
  """Returns a readonly checkpoint manager for the given ckpt_dir."""
  logging.warning('DEPRECATED: Please use Checkpointer to load checkpoint.')
  handler_registry = ocp.DefaultCheckpointHandlerRegistry()
  handler_registry.add(
      'default', ocp.args.PyTreeRestore, ocp.PyTreeCheckpointHandler()
  )
  handler_registry.add(
      'state', ocp.args.PyTreeRestore, ocp.PyTreeCheckpointHandler()
  )
  handler_registry.add(
      'metadata', ocp.args.JsonRestore, ocp.JsonCheckpointHandler()
  )
  return ocp.CheckpointManager(
      ckpt_dir,
      options=ocp.CheckpointManagerOptions(read_only=True),
      handler_registry=handler_registry,
  )


def load_checkpoint_from_manager(
    checkpoint_manager: ocp.CheckpointManager,
    abstract_state: PyTree,
    ckpt_step: int = -1,
    ckpt_format: CheckpointFormat | str = '',
):
  """Loads a checkpoint at ckpt_step in the format of abstract_state."""
  return load_checkpoint_from_dir(
      checkpoint_manager.directory.as_posix(),
      abstract_state,
      ckpt_step,
      ckpt_format=ckpt_format,
  )


def last_checkpoint_step(ckpt_dir: str) -> int:
  last_step = -1
  ckpt_dir_path = epath.Path(ckpt_dir)
  if not ckpt_dir_path.is_dir():
    return -1
  for item in ckpt_dir_path.iterdir():
    step = item.name
    if step.isdigit() and int(step) > last_step:
      last_step = int(step)
  return last_step


def load_checkpoint_from_dir(
    ckpt_dir: str,
    abstract_state: PyTree,
    ckpt_step: int = -1,
    ckpt_format: CheckpointFormat | str = '',
):
  """Loads a checkpoint at ckpt_step in the format of abstract_state."""
  if ckpt_step < 0:
    ckpt_step = last_checkpoint_step(ckpt_dir)
  if ckpt_step < 0:
    raise ValueError(f'No checkpoint found in {ckpt_dir}.')
  return load_checkpoint_from_path(
      os.path.join(ckpt_dir, str(ckpt_step)),
      abstract_state,
      ckpt_format=ckpt_format,
  )


def resolve_checkpoint_handler_from_json(
    handler_in_json: PyTree,
) -> ocp.CheckpointHandler:
  """Resolves a checkpoint handler from a handler represented in json."""
  if isinstance(handler_in_json, str):
    handler_cls = pydoc.locate(handler_in_json)
    if not isinstance(handler_cls, type(ocp.CheckpointHandler)):
      raise ValueError(f'Unsupported checkpoint handler class: {handler_cls}')
    if handler_cls is ocp.StandardCheckpointHandler:
      # Use PyTreeCheckpointHandler as the standard handler.
      return ocp.PyTreeCheckpointHandler()
    return handler_cls()
  if pytree.tree_is_mapping(handler_in_json):
    return ocp.CompositeCheckpointHandler(**{
        k: resolve_checkpoint_handler_from_json(v)
        for k, v in handler_in_json.items()
    })
  raise ValueError(f'Unsupported checkpoint handler: {handler_in_json}')


def resolve_checkpoint_handler_from_path(
    ckpt_path: str,
) -> ocp.CheckpointHandler:
  """Resolves a checkpoint handler from a checkpoint path."""
  checkpoint_metadata = ocp.metadata.get_step_metadata(ckpt_path)
  if checkpoint_metadata.item_handlers is not None:
    return resolve_checkpoint_handler_from_json(
        checkpoint_metadata.item_handlers
    )
  # Some old ORBAX checkpoints do not have handler information written in
  # checkpoint metadata. We need to infer it from the checkpoint structure.
  items = [p.name for p in epath.Path(ckpt_path).iterdir()]
  if '_METADATA' in items:
    return ocp.PyTreeCheckpointHandler()
  handlers = {}
  for item in items:
    if item in ('state', 'default'):
      handlers[item] = ocp.PyTreeCheckpointHandler()
    elif item in ('metadata', 'data'):
      handlers[item] = ocp.JsonCheckpointHandler()
  return ocp.CompositeCheckpointHandler(**handlers)


def load_checkpoint_from_path(
    ckpt_path: str,
    abstract_state: PyTree,
    ckpt_format: CheckpointFormat | str = '',
):
  """Loads a checkpoint in the format of abstract_state using ckpt_format."""
  raw_abstract_state = common.get_raw_arrays(abstract_state)
  restore_args = ocp.checkpoint_utils.construct_restore_args(raw_abstract_state)

  logging.info('Loading checkpoint from %s', ckpt_path)
  handler = resolve_checkpoint_handler_from_path(ckpt_path)
  start_time = time.time()
  logging.info('Loading checkpoint from %s', ckpt_path)
  with ocp.Checkpointer(handler) as checkpointer:
    item_metadata = checkpointer.metadata(ckpt_path).item_metadata

    if isinstance(ckpt_format, str):
      if ckpt_format:
        ckpt_format = CheckpointFormatRegistry.get_instance(ckpt_format)
    if not ckpt_format:
      ckpt_format = LegacyFormat()
      if ocp_constants.METADATA_ITEM_NAME in item_metadata:
        restored = checkpointer.restore(
            ckpt_path,
            args=ocp.args.Composite(
                **{ocp_constants.METADATA_ITEM_NAME: ocp.args.JsonRestore()}
            ),
        )
        metadata = pytree.load_dataclasses(restored.metadata)
        ckpt_format = metadata[CHECKPOINT_FORMAT_KEY]
    assert isinstance(ckpt_format, CheckpointFormat)

    # Guess state key.
    state_key = None
    if 'state' in item_metadata:
      state_key = 'state'
    elif 'default' in item_metadata:
      state_key = 'default'

    original_metadata = item_metadata[state_key] if state_key else item_metadata
    pytree_restore = ocp.args.PyTreeRestore(
        raw_abstract_state,
        restore_args=restore_args,
        transforms=ckpt_format.transforms(
            restore_args, original_metadata=original_metadata
        ),
    )
    if state_key:
      pytree_restore = ocp.args.Composite(**{
          state_key: pytree_restore,
      })

    restored = checkpointer.restore(ckpt_path, pytree_restore)

  logging.info(
      'Checkpoint from %s loaded with %f seconds spent.',
      ckpt_path,
      time.time() - start_time,
  )
  state = restored[state_key] if state_key else restored
  state = common.transfer_metadata(abstract_state, state)
  return state


def load_data_state_from_dir(ckpt_dir: str, ckpt_step: int = -1) -> PyTree:
  """Loads data from a checkpoint at ckpt_step."""
  with ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(
          **{DATA_ITEM_NAME: ocp.JsonCheckpointHandler()}
      )
  ) as checkpointer:
    restored = checkpointer.restore(
        os.path.join(ckpt_dir, str(ckpt_step)),
        args=ocp.args.Composite(**{DATA_ITEM_NAME: ocp.args.JsonRestore()}),
    )
    return restored[DATA_ITEM_NAME]


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    state: PyTree,
    ckpt_step: int,
    ckpt_format: CheckpointFormat = LegacyFormat(),
    data: PyTree | None = None,
    **kwargs: Any,
):
  """Saves a checkpoint at ckpt_step in ckpt_format."""
  extra_args = {}
  if data is not None:
    extra_args['data'] = ocp.args.JsonSave(data)
  return checkpoint_manager.save(
      ckpt_step,
      args=ocp.args.Composite(
          state=ocp.args.PyTreeSave(common.get_raw_arrays(state)),
          metadata=ocp.args.JsonSave(
              pytree.dump_dataclasses({CHECKPOINT_FORMAT_KEY: ckpt_format})
          ),
          **extra_args,
      ),
      **kwargs,
  )


def get_abstract_params(model: module.SimplyModule) -> PyTree:
  return common.eval_shape_with_sharding(model.init, jax.random.PRNGKey(0))

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
from collections.abc import Mapping
import dataclasses
import functools
import logging
import os
import pydoc
import re
import time
from typing import Any, ClassVar, TypeAlias, final

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
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    """Transforms the stored state."""
    del target_abstract_state
    return stored_state


class CheckpointFormatRegistry(registry.FunctionRegistry):
  """Registry for checkpoint formats."""

  namespace: ClassVar[str] = 'CheckpointFormat'


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class LegacyFormat(CheckpointFormat):
  """The legacy checkpoint format this project is using at the beginning."""


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma3pFormat(CheckpointFormat):
  """Gemma third-party checkpoint format with transposed ffn weights."""

  transpose_ffn_weights: bool = True

  @functools.cached_property
  def prefix_mapping(self) -> Mapping[str, str]:
    return {'transformer': 'params'}

  @functools.cached_property
  def ln_mapping(self) -> Mapping[str, str]:
    return {
        'pre_attention_norm': 'pre_ln_0',
        'pre_ffw_norm': 'pre_ln_1',
        'post_attention_norm': 'post_ln_0',
        'post_ffw_norm': 'post_ln_1',
    }

  def transforms(
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    flatten_stored_state = ocp.tree.to_flat_dict(stored_state, sep='/')
    transformed_state = {}
    for k, v in flatten_stored_state.items():
      if m := re.fullmatch(r'(.*)/embedder/input_embedding', k):
        new_k = self.prefix_mapping[m.group(1)] + '/embed'
        transformed_state[new_k] = v
      elif m := re.fullmatch(r'(.*)/final_norm/(.*)', k):
        new_k = self.prefix_mapping[m.group(1)] + f'/final_ln/{m.group(2)}'
        transformed_state[new_k] = v
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/attn/(q?kv)_einsum/w', k):
        new_k = (
            self.prefix_mapping[m.group(1)]
            + f'/block_{m.group(2)}/attn/{m.group(3)}_proj'
        )
        transformed_state[new_k] = jnp.einsum('kndh->kdnh', v)
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/attn/q_einsum/w', k):
        new_k = (
            self.prefix_mapping[m.group(1)] + f'/block_{m.group(2)}/attn/q_proj'
        )
        transformed_state[new_k] = jnp.einsum('ndh->dnh', v)
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/attn/attn_vec_einsum/w', k):
        new_k = (
            self.prefix_mapping[m.group(1)] + f'/block_{m.group(2)}/attn/o_proj'
        )
        transformed_state[new_k] = jnp.einsum('nhd->dnh', v)
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/attn/_query_norm/(.*)', k):
        new_k = (
            self.prefix_mapping[m.group(1)]
            + f'/block_{m.group(2)}/attn/q_norm/{m.group(3)}'
        )
        transformed_state[new_k] = v
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/attn/_key_norm/(.*)', k):
        new_k = (
            self.prefix_mapping[m.group(1)]
            + f'/block_{m.group(2)}/attn/k_norm/{m.group(3)}'
        )
        transformed_state[new_k] = v
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/mlp/gating_einsum/(.*)', k):
        suffix = m.group(3)
        new_k0 = (
            self.prefix_mapping[m.group(1)]
            + f'/block_{m.group(2)}/ffn_0_gate/{suffix}'
        )
        new_k1 = (
            self.prefix_mapping[m.group(1)]
            + f'/block_{m.group(2)}/ffn_0/{suffix}'
        )
        if suffix == 'w' and self.transpose_ffn_weights:
          transformed_state[new_k0] = jnp.transpose(v[0])
          transformed_state[new_k1] = jnp.transpose(v[1])
        else:
          transformed_state[new_k0] = v[0]
          transformed_state[new_k1] = v[1]
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/mlp/linear/(.*)', k):
        new_k = (
            self.prefix_mapping[m.group(1)]
            + f'/block_{m.group(2)}/ffn_1/{m.group(3)}'
        )
        transformed_state[new_k] = v
      elif m := re.fullmatch(r'(.*)/layer_(\d+)/(\w+_norm)/(.*)', k):
        new_k = (
            self.prefix_mapping[m.group(1)]
            + f'/block_{m.group(2)}/{self.ln_mapping[m.group(3)]}/{m.group(4)}'
        )
        transformed_state[new_k] = v
      elif m := re.fullmatch(r'(.*)/block_(\d+)/([a-z]+_ln_\d+)/(.*)', k):
        old_k = (
            self.prefix_mapping[m.group(1)]
            + f'/layer_{m.group(2)}/{self.ln_mapping[m.group(3)]}/{m.group(4)}'
        )
        transformed_state[k] = flatten_stored_state[old_k]
      elif k == 'step_on_device':
        transformed_state['steps'] = v
      else:
        logging.warning('stored_state[%s] is ignored by %s', k, self.__class__)
    return ocp.tree.from_flat_dict(transformed_state, sep='/')


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma3pLegacyFormat(Gemma3pFormat):
  """Gemma third-party checkpoint format without transposed ffn weights."""

  transpose_ffn_weights: bool = False


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2TransposeFormat(Gemma3pFormat):
  """Gemma2 checkpoint format with transposed ffn weights."""

  @functools.cached_property
  def prefix_mapping(self) -> Mapping[str, str]:
    return {
        'params/transformer': 'params',
        'opt_state/1/0/mu/transformer': 'm',
        'opt_state/1/0/nu/transformer': 'v',
    }


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Gemma2Format(Gemma3pLegacyFormat):
  """Gemma2 checkpoint format."""

  @functools.cached_property
  def prefix_mapping(self) -> Mapping[str, str]:
    return {
        'params/transformer': 'params',
        'opt_state/1/0/mu/transformer': 'm',
        'opt_state/1/0/nu/transformer': 'v',
    }


@CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen2Format(CheckpointFormat):
  """Qwen2 checkpoint format."""

  def transforms(
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    flatten_stored_state = ocp.tree.to_flat_dict(stored_state, sep='/')
    per_head_dim = target_abstract_state['params']['block_0']['attn'][
        'q_proj'
    ].shape[-1]
    transformed_state = {}
    for k, v in flatten_stored_state.items():
      if k == 'model.embed_tokens.weight':
        transformed_state['params/embed'] = v
      elif k == 'lm_head.weight':
        transformed_state['params/output_layer/w'] = jnp.transpose(v)
      elif k == 'model.norm.weight':
        transformed_state['params/final_ln/scale'] = v
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.([qk]_norm).weight', k
      ):
        new_k = f'params/block_{m.group(1)}/attn/{m.group(2)}/scale'
        transformed_state[new_k] = v
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.([qkv]_proj).weight', k
      ):
        new_k = f'params/block_{m.group(1)}/attn/{m.group(2)}'
        transformed_state[new_k] = jnp.einsum(
            'nhd->dnh', jnp.reshape(v, (-1, per_head_dim, *v.shape[1:]))
        )
      elif m := re.fullmatch(
          r'model.layers.(\d+).self_attn.([qkv])_proj.bias', k
      ):
        new_k = f'params/block_{m.group(1)}/attn/{m.group(2)}_bias'
        transformed_state[new_k] = jnp.reshape(v, (-1, per_head_dim))
      elif m := re.fullmatch(r'model.layers.(\d+).self_attn.o_proj.weight', k):
        new_k = f'params/block_{m.group(1)}/attn/o_proj'
        transformed_state[new_k] = jnp.reshape(
            v, (*v.shape[:-1], -1, per_head_dim)
        )
      elif m := re.fullmatch(r'model.layers.(\d+).mlp.up_proj.weight', k):
        new_k = f'params/block_{m.group(1)}/ffn_0/w'
        transformed_state[new_k] = jnp.transpose(v)
      elif m := re.fullmatch(r'model.layers.(\d+).mlp.gate_proj.weight', k):
        new_k = f'params/block_{m.group(1)}/ffn_0_gate/w'
        transformed_state[new_k] = jnp.transpose(v)
      elif m := re.fullmatch(r'model.layers.(\d+).mlp.down_proj.weight', k):
        new_k = f'params/block_{m.group(1)}/ffn_1/w'
        transformed_state[new_k] = jnp.transpose(v)
      elif m := re.fullmatch(r'model.layers.(\d+).input_layernorm.weight', k):
        transformed_state[f'params/block_{m.group(1)}/pre_ln_0/scale'] = v
      elif m := re.fullmatch(
          r'model.layers.(\d+).post_attention_layernorm.weight', k
      ):
        transformed_state[f'params/block_{m.group(1)}/pre_ln_1/scale'] = v
      else:
        logging.warning('stored_state[%s] is ignored by %s', k, self.__class__)
    return ocp.tree.from_flat_dict(transformed_state, sep='/')


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


def construct_restore_item(x: PyTree) -> PyTree:
  """Constructs a restore item from a PyTree."""
  leaves, treedef = jax.tree_util.tree_flatten(x)

  mesh = sharding_lib.get_default_mesh()
  partitions = sharding_lib.batch_partition_with_minimum_redundancy(
      [leaf.shape for leaf in leaves], mesh.axis_names, mesh.axis_sizes
  )

  structs = []
  for leaf, partition in zip(leaves, partitions, strict=True):
    if not isinstance(
        leaf, (ocp.metadata.ArrayMetadata, jax.Array, jax.ShapeDtypeStruct)
    ):
      raise ValueError(f'Unsupported leaf type: {type(leaf)}')
    structs.append(
        jax.ShapeDtypeStruct(
            shape=leaf.shape,
            dtype=leaf.dtype,
            sharding=sharding_lib.mesh_sharding(partition),
        )
    )
  return jax.tree_util.tree_unflatten(treedef, structs)


def load_checkpoint_from_path(
    ckpt_path: str,
    abstract_state: PyTree,
    ckpt_format: CheckpointFormat | str = '',
):
  """Loads a checkpoint in the format of abstract_state using ckpt_format."""
  target_abstract_state = common.get_raw_arrays(abstract_state)

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
        metadata = pytree.load(restored.metadata)
        ckpt_format = metadata[CHECKPOINT_FORMAT_KEY]
    assert isinstance(ckpt_format, CheckpointFormat)

    # Guess state key.
    state_key = None
    if 'state' in item_metadata:
      state_key = 'state'
    elif 'default' in item_metadata:
      state_key = 'default'

    original_metadata = item_metadata[state_key] if state_key else item_metadata
    restore_item = construct_restore_item(original_metadata.tree)

    def transform_state_fn(stored_state: PyTree) -> PyTree:  # pylint: disable=function-redefined

      state = ckpt_format.transforms(stored_state, target_abstract_state)

      def _get_regularized_value(
          path: jax.tree_util.KeyPath,
          abstract: jax.ShapeDtypeStruct | jax.Array,
      ):
        try:
          value = pytree.tree_value(state, path)
          if value.shape != abstract.shape:
            raise ValueError(
                f'Shape mismatch for {path}: restored is {value.shape} while '
                f'target is {abstract.shape}'
            )
          return sharding_lib.with_sharding_constraint(
              jnp.astype(value, abstract.dtype), abstract.sharding
          )
        except KeyError as e:
          logging.warning(
              'Value at %s is not loaded from checkpoint: %s', path, e
          )
          return abstract

      state = jax.tree_util.tree_map_with_path(
          _get_regularized_value, target_abstract_state
      )
      return state

    unused_argpaths = common.find_unused_argpaths(
        transform_state_fn, restore_item
    )
    for unused_argpath in unused_argpaths:
      pytree.set_tree_value(restore_item, unused_argpath, ocp.PLACEHOLDER)

    pytree_restore = ocp.args.PyTreeRestore(
        restore_item,
        restore_args=ocp.checkpoint_utils.construct_restore_args(restore_item),
    )

    if state_key:
      pytree_restore = ocp.args.Composite(**{state_key: pytree_restore})
    restored = checkpointer.restore(ckpt_path, pytree_restore)

  logging.info(
      'Checkpoint from %s loaded with %f seconds spent.',
      ckpt_path,
      time.time() - start_time,
  )
  state = restored[state_key] if state_key else restored

  for unused_argpath in unused_argpaths:
    # Turn PLACEHOLDER to None to avoid feeding into jitted function.
    pytree.set_tree_value(state, unused_argpath, None)

  state = jax.jit(transform_state_fn, donate_argnums=0)(state)

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
              pytree.dump({CHECKPOINT_FORMAT_KEY: ckpt_format})
          ),
          **extra_args,
      ),
      **kwargs,
  )


def get_abstract_params(model: module.SimplyModule) -> PyTree:
  return common.eval_shape_with_sharding(model.init, jax.random.PRNGKey(0))

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
"""Restores the released Qwen3.8 checkpoint into this package's parameter tree.

The release converts with the stock `python -m simply.tools.hf_to_orbax`,
which writes the HuggingFace tensor names verbatim (1199 of them:
`model.language_model.*`, `lm_head.weight`, plus a vision tower and a
multi-token-prediction head this text model does not have). The whole
name/shape mapping therefore happens here, at restore, in `transforms` -- the
one hook `utils/checkpoint_lib.CheckpointFormat` offers. (That binary does not
link this package, so it cannot stamp `Qwen38Format` into the checkpoint
metadata; what selects this format is `config.init_ckpt_format`, which
`load_checkpoint_from_dir` prefers over the metadata.)

Three things are not a rename:

  * the attention projections are stored flat, `(heads * head_dim, model_dim)`,
    and Simply keeps the head dim separate, `(model_dim, heads, head_dim)`
    (`q_proj` doubly wide: the released query and its output gate are one
    tensor);
  * every `nn.Linear` is `(out, in)` and every Simply `EinsumLinear` is
    `(in, out)`;
  * the GatedDeltaNet depthwise conv is `(channels, 1, taps)` and Simply's is
    `(channels, taps)`.

`Qwen38Format._rules` is the whole mapping, one row per released tensor
family, and `utils/ckpt_format_test.py` re-derives it independently from the
config. `convert_from_mapping` runs the same mapping offline, described by a
config instead of by the tree being restored into, for callers that convert
HuggingFace weights without building the model.

Under-restore is silent in core (`checkpoint_lib.py` only `logging.warning`s a
key it could not fill, and leaves the uninitialised value in place), so
`transforms` refuses to return a tree that does not cover every parameter:
a released rename would otherwise cost a 27B job's worth of garbage output.
"""

from collections.abc import Callable, Mapping
import dataclasses
import re
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import pytree as pytree_lib

PyTree = Any
# `Qwen38ExperimentConfig`, kept structural so that this module -- which every
# conversion path depends on -- does not depend on the config.
SimplyConfig = Any

# A stored tensor, the Simply path it maps to and the tree being filled -> the
# value to write there.
Transform = Callable[[Any, str, 'Target'], Any]

# Name under which the format is registered; `config.init_ckpt_format`.
FORMAT_NAME = 'Qwen38Format'

# Released tensor families with no counterpart in a text-only dense Qwen3.8,
# dropped without a warning because dropping them is the design:
#   `model.visual.*`  the vision tower (333 tensors), never entered by the
#                     text model;
#   `mtp.*`           the multi-token-prediction head (15 tensors), which
#                     needs `mtp_num_hidden_layers > 0` and a speculative
#                     decoder to run. See README.md.
DROPPED_PREFIXES = ('model.visual.', 'mtp.')

# The prefixes `tools:hf_to_orbax` can leave on a text tensor. Qwen3.8 is a
# multimodal release, so its text weights live under `model.language_model.`.
_TEXT_PREFIXES = ('model.language_model.', 'language_model.', 'model.')

# `config_lib.LINEAR_ATTENTION`, and the `token_mixer` leaves each kind of
# layer has. Named here rather than imported: this module is on the path of
# every conversion and stays independent of the config module. Only
# `target_from_config` reads them, and `ckpt_format_test.py`'s
# `test_matches_the_restore_path` is what keeps them in step with the modules.
_LINEAR_ATTENTION = 'linear_attention'
_GDN_LEAVES = frozenset({
    'in_proj_qkv',
    'in_proj_z',
    'in_proj_b',
    'in_proj_a',
    'conv1d',
    'A_log',
    'dt_bias',
    'norm',
    'out_proj',
})
_ATTENTION_LEAVES = frozenset(
    {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm'}
)


# --- Value transforms -------------------------------------------------------


def _verbatim(v: Any, path: str, target: 'Target') -> Any:
  """Norm scales, the embedding table, the LM head, the GDN per-head scalars."""
  del path, target
  return v


def _transposed(v: Any, path: str, target: 'Target') -> Any:
  """`nn.Linear` `(out, in)` -> `EinsumLinear` `(in, out)`."""
  del path, target
  return jnp.transpose(v)


def _squeezed_conv(v: Any, path: str, target: 'Target') -> Any:
  """Depthwise conv `(channels, 1, taps)` -> Simply's `(channels, taps)`."""
  del path, target
  return jnp.squeeze(v, axis=1)


# --- The tree being filled --------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Target:
  """The two questions the mapping asks about the tree it is filling.

  Attributes:
    wants: Is this Simply parameter path part of the model? A released tensor
      whose target is absent -- a layer a smaller deployment does not have --
      is dropped rather than invented.
    head_dim: The last dimension of an attention leaf, which is what the flat
      released projection has to be split by. One function serves `q_proj`
      (whose head dim is doubled by the fused output gate) and `k`/`v`/`o`
      alike.
  """

  wants: Callable[[str], bool]
  head_dim: Callable[[str], int]


def target_from_abstract_state(state: PyTree) -> Target:
  """At restore: the tree being filled answers both questions itself."""
  if state is None:
    raise ValueError(f'{FORMAT_NAME} needs target_abstract_state.')

  def head_dim(path: str) -> int:
    # The leaf is abstract: read `.shape`, do not touch the value.
    return pytree_lib.tree_value(state, path).shape[-1]  # pyrefly: ignore[missing-attribute]

  return Target(wants=lambda path: _has_path(state, path), head_dim=head_dim)


def target_from_config(config: SimplyConfig) -> Target:
  """Offline: the config answers both, so no model has to be built.

  Args:
    config: A `Qwen38ExperimentConfig`.

  Returns:
    The same description of the target tree that the abstract state gives.

  Raises:
    ValueError: for a variant this port does not build.
  """
  layer_types = config.resolved_layer_types()
  if not config.attn_output_gate:
    # `utils/attn.py` raises on the same condition: its `q_proj` is `[q|gate]`
    # unconditionally, so an ungated head dim would describe no real tree.
    raise ValueError(f'{FORMAT_NAME}: only gated attention is implemented.')

  def wants(path: str) -> bool:
    block = re.match(r'params/block_(\d+)/(.*)', path)
    if block is None:
      return True
    layer = int(block[1])
    if layer >= len(layer_types):
      return False
    mixer = re.match(r'token_mixer/([^/]+)', block[2])
    if mixer is None:
      return True
    # Both mixers live under the `token_mixer` key, so only the layer type
    # says which set of leaves this block actually has.
    wanted = (
        _GDN_LEAVES
        if layer_types[layer] == _LINEAR_ATTENTION
        else _ATTENTION_LEAVES
    )
    return mixer[1] in wanted

  def head_dim(path: str) -> int:
    # `q_proj` holds the query and its output gate, `[q | gate]` per head.
    stacks = 2 if path.endswith('/q_proj/w') else 1
    return stacks * config.per_head_dim

  return Target(wants=wants, head_dim=head_dim)


# --- The mapping ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Rule:
  """One released tensor family: where its value goes, and how.

  Attributes:
    pattern: Matched against the tensor name with the text prefix stripped.
    template: `re.Match.expand` template of the Simply parameter path.
    transform: Applied to the stored value.
  """

  pattern: str
  template: str
  transform: Transform


def strip_text_prefix(name: str) -> str:
  """The released tensor name without the container the release wraps it in."""
  for prefix in _TEXT_PREFIXES:
    if name.startswith(prefix):
      return name.removeprefix(prefix)
  return name


@ckpt_lib.CheckpointFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38Format(ckpt_lib.Qwen2Format):
  """Maps raw HuggingFace Qwen3.8 tensor names onto this package's tree.

  `Qwen2Format` is core's restore-time HuggingFace mapping; its `transforms` is
  replaced wholesale (Qwen3.8 has two kinds of token mixer, which core's key
  layout cannot express) and only its `_split_head` is reused.

  Every field must have a default: `save_checkpoint` bakes an instance built
  with no arguments into the checkpoint metadata, and both ways of getting the
  format back -- that metadata, or `config.init_ckpt_format` through
  `CheckpointFormatRegistry.get_instance` -- construct it the same way.
  """

  def _rules(self) -> tuple[Rule, ...]:
    """The mapping table. Bound late: two rows need `self._split_head`."""
    block = r'params/block_\1'
    mixer = rf'{block}/token_mixer'
    return (
        Rule(r'embed_tokens\.weight', 'params/embed_linear/embed', _verbatim),
        # Untied: `config_from_hf` rejects `tie_word_embeddings`, so the head
        # is always its own tensor.
        Rule(r'lm_head\.weight', 'params/embed_linear/w', _verbatim),
        Rule(r'norm\.weight', 'params/final_ln/scale', _verbatim),
        Rule(
            r'layers\.(\d+)\.(input_layernorm|post_attention_layernorm)'
            r'\.weight',
            rf'{block}/\2/scale',
            _verbatim,
        ),
        Rule(
            r'layers\.(\d+)\.mlp\.gate_proj\.weight',
            rf'{block}/ffn/ffn_0_gate/w',
            _transposed,
        ),
        Rule(
            r'layers\.(\d+)\.mlp\.up_proj\.weight',
            rf'{block}/ffn/ffn_0/w',
            _transposed,
        ),
        Rule(
            r'layers\.(\d+)\.mlp\.down_proj\.weight',
            rf'{block}/ffn/ffn_1/w',
            _transposed,
        ),
        Rule(
            r'layers\.(\d+)\.linear_attn\.'
            r'(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\.weight',
            rf'{mixer}/\2/w',
            _transposed,
        ),
        Rule(
            r'layers\.(\d+)\.linear_attn\.conv1d\.weight',
            rf'{mixer}/conv1d/w',
            _squeezed_conv,
        ),
        Rule(
            r'layers\.(\d+)\.linear_attn\.(A_log|dt_bias)',
            rf'{mixer}/\2',
            _verbatim,
        ),
        Rule(
            r'layers\.(\d+)\.linear_attn\.norm\.weight',
            rf'{mixer}/norm/scale',
            _verbatim,
        ),
        Rule(
            r'layers\.(\d+)\.self_attn\.([qk]_norm)\.weight',
            rf'{mixer}/\2/scale',
            _verbatim,
        ),
        Rule(
            r'layers\.(\d+)\.self_attn\.([qkv]_proj)\.weight',
            rf'{mixer}/\2/w',
            self._split_proj,
        ),
        Rule(
            r'layers\.(\d+)\.self_attn\.o_proj\.weight',
            rf'{mixer}/o_proj/w',
            self._split_out_proj,
        ),
    )

  # --- Transforms that need to know the head dim ---

  def _split_proj(self, v: Any, path: str, target: Target) -> Any:
    """`(heads * head_dim, model_dim)` -> `(model_dim, heads, head_dim)`."""
    return jnp.einsum('nhd->dnh', self._split_head(v, target.head_dim(path)))

  def _split_out_proj(self, v: Any, path: str, target: Target) -> Any:
    """`(model_dim, heads * head_dim)` -> `(model_dim, heads, head_dim)`."""
    return self._split_head(v, target.head_dim(path), axis=1)

  # --- The mapping, and the restore hook that is one call to it ---

  def convert(
      self,
      stored_state: PyTree,
      target: Target,
      dtype: jax.typing.DTypeLike | None = None,
  ) -> PyTree:
    """Rewrites the stored HuggingFace tree into the Simply parameter tree.

    Args:
      stored_state: HuggingFace tensors by name, however deeply nested.
      target: The tree being filled; see `Target`.
      dtype: Cast every mapped value to it. `None` keeps the stored dtype,
        which is what restoring wants -- core casts to the abstract state's
        dtype afterwards.

    Returns:
      The tree in this package's layout. Paths the target does not want are
      absent.
    """
    rules = self._rules()
    transformed = {}
    flat = ocp.tree.to_flat_dict(stored_state, sep='/')
    for raw_name, stored in flat.items():
      if raw_name.startswith(DROPPED_PREFIXES):
        continue
      name = strip_text_prefix(raw_name)
      for rule in rules:
        if m := re.fullmatch(rule.pattern, name):
          path = m.expand(rule.template)
          if target.wants(path):
            # Cast first: every transform is a permutation, so this is the
            # same answer at a fraction of the peak memory.
            value = stored if dtype is None else jnp.asarray(stored, dtype)
            transformed[path] = rule.transform(value, path, target)
          break
      else:
        logging.warning('%s ignores stored tensor %s', FORMAT_NAME, raw_name)
    return ocp.tree.from_flat_dict(transformed, sep='/')

  def transforms(
      self, stored_state: PyTree, target_abstract_state: PyTree = None
  ) -> PyTree:
    """The `CheckpointFormat` hook: convert against the tree being restored.

    Args:
      stored_state: The checkpoint as written by `tools:hf_to_orbax`.
      target_abstract_state: The tree being restored into. Required.

    Returns:
      The tree in this package's layout, covering every parameter of it.

    Raises:
      ValueError: if the checkpoint does not fill every parameter -- core
        would only warn, and run the model on uninitialised weights.
    """
    converted = self.convert(
        stored_state, target_from_abstract_state(target_abstract_state)
    )
    if missing := unfilled_params(converted, target_abstract_state):
      raise ValueError(
          f'{FORMAT_NAME}: {len(missing)} parameter(s) have no tensor in the'
          f' checkpoint, e.g. {missing[:5]}. A released tensor was renamed, or'
          ' this is not a Qwen3.8 checkpoint.'
      )
    return converted


def convert_from_mapping(
    state_dict: Mapping[str, Any],
    config: SimplyConfig,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> PyTree:
  """HuggingFace state dict -> Simply parameter tree, without a model.

  The same mapping `Qwen38Format` restores with, described by the config
  instead of by an abstract parameter tree, so that a fixture loader can
  convert released weights without depending on `model_lib`.

  Args:
    state_dict: HuggingFace tensors by name, with or without the release's
      `model.language_model.` / `model.` prefix.
    config: A `Qwen38ExperimentConfig` matching those tensors.
    dtype: Cast every mapped value to it; `None` keeps the stored dtype.

  Returns:
    The `{'embed_linear': ..., 'block_0': ..., ...}` tree, without the
    `params` level `transforms` produces (that level belongs to the
    checkpoint's state, not to the model).
  """
  converted = Qwen38Format().convert(
      state_dict, target_from_config(config), dtype
  )
  return converted.get('params', {})


def unfilled_params(converted: PyTree, target: PyTree) -> list[str]:
  """Parameter paths of `target` that `converted` has no value for.

  Args:
    converted: The output of `Qwen38Format.convert`.
    target: The tree being filled.

  Returns:
    The missing `params/...` paths, sorted. Anything outside `params/` is
    ignored: core restores optimizer state through the same call, and a
    checkpoint that carries none is a legitimate inference checkpoint.
  """
  got = set(ocp.tree.to_flat_dict(converted, sep='/'))
  # `AnnotatedArray` leaves flatten to `<path>/0`; core hands `transforms` a
  # raw tree, but a caller need not.
  want = {
      path.removesuffix('/0')
      for path in ocp.tree.to_flat_dict(target, sep='/')
  }
  return sorted(p for p in want - got if p.startswith('params/'))


def _has_path(tree: PyTree, path: str) -> bool:
  try:
    pytree_lib.tree_value(tree, path)
  except (KeyError, ValueError, IndexError, TypeError):
    # `TypeError`: the path ran through a leaf rather than a subtree.
    return False
  return True

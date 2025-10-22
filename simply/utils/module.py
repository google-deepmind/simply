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
"""Module."""

import abc
from collections.abc import Sequence
import dataclasses
from typing import Any, cast, ClassVar, final
import einops
import jax
import jax.numpy as jnp
import jax.typing
from simply.utils import common
from simply.utils import initializer
from simply.utils import registry
from simply.utils import sharding as sharding_lib


PyTree = common.PyTree
AnnotatedArray = common.AnnotatedArray
get_raw_arrays = common.get_raw_arrays
Array = common.Array
PRNGKey = jax.typing.ArrayLike


class SimplyModule(abc.ABC):
  """An untra-simplified version of `flax.nn.Module`."""

  @final
  def __post_init__(self):
    if not dataclasses.is_dataclass(self):
      raise ValueError(
          f'SimplyModule must be a dataclass. {self.__class__.__name__} is not.'
      )
    if not ModuleRegistry.get(self.__class__.__name__):
      raise ValueError(
          'SimplyModule'
          f' {ModuleRegistry.fullname(self.__class__.__name__)} is not'
          ' registered.'
      )
    cast(SimplyModule, self).setup()

  def __getattr__(self, name: str) -> Any:
    raise AttributeError(
        f'Attribute {name} not found in {self.__class__.__name__}'
    )

  def setup(self) -> None:
    """Setup any attributes. Typically used for instantiating sub-modules."""

  def init(self, prng_key: jax.Array) -> PyTree:
    """initialize the parameters associated with the module."""

  @abc.abstractmethod
  def apply(self, params: PyTree, x: Any, **kwargs: Any) -> Any:
    """Run forward pass of the module with parameters and inputs."""


class ModuleRegistry(registry.RootRegistry):
  """Registry for modules."""

  namespace: ClassVar[str] = 'Module'


def _reshape_bias(
    bias: Array, *, output_term: str, bias_term: str,
    output_shape: Sequence[int]) -> Array:
  """Reshapes bias tensor to be broadcastable to output tensor.

  When `...` is used in `output_term`, bias dimensions corresponding to `...`
  are filled with 1. Other dimensions in `output_term` that are not in
  `bias_term` are also filled with 1.

  Args:
    bias: The bias tensor, with dimensions specified by `bias_term`.
    output_term: The output term of einsum equation, e.g., '...f' or 'b...hd'.
    bias_term: A string containing characters that represent bias dimensions,
      e.g., 'f' or 'hd'.
    output_shape: The shape of activation tensor this bias will be added to.

  Returns:
    Reshaped bias tensor.
  """
  if '...' in output_term:
    prefix, suffix = output_term.split('...')
  else:
    prefix, suffix = output_term, ''
  def create_rearranged_bias_term(string, bias_term):
    new_string = ''
    for char in string:
      if char in bias_term:
        new_string += char
      else:
        # If a dimension only appears in the output_term, then add 1
        # for broadcasting the bias at that dimension.
        new_string += '1'
    return new_string
  new_bias_term = ''
  new_bias_term += create_rearranged_bias_term(prefix, bias_term)
  # Fill the missing dimensions with 1 for broadcasting.
  new_bias_term += '1' * (len(output_shape) - len(prefix + suffix))
  new_bias_term += create_rearranged_bias_term(suffix, bias_term)
  # Prepare einops string for the rearrage.
  einops_bias_string = ' '.join(bias_term)
  einops_new_bias_string = ' '.join(new_bias_term)
  new_bias = einops.rearrange(
      bias, f'{einops_bias_string} -> {einops_new_bias_string}')
  return new_bias


def _parse_einsum_eqn(eqn: str) -> tuple[str, str, str]:
  """Parses einsum equation into weight, input, and output strings."""
  eqn = eqn.replace(' ', '')
  if '->' not in eqn:
    raise ValueError('`eqn` must be explicit and include "->".')
  if eqn.count(',') != 1:
    raise ValueError('`eqn` must have exactly two operands.')

  operands, output_term = eqn.split('->')
  weight_term, input_term = operands.split(',')

  if '...' in weight_term:
    raise ValueError('Einsum string weight part cannot contain `...`.')
  return weight_term, input_term, output_term


def create_char_dict(term: str, seq: Sequence[Any] | None) -> dict[str, Any]:
  """Creates a dictionary mapping dimension characters to sequence elements.

  This function maps characters in an einsum term (e.g., 'b...hd') to
  elements in a sequence (e.g., shape or partition specs). It correctly
  handles ellipsis (...) but assumes that at most one (...) is present.

  Args:
    term: Einsum term string, e.g., 'b...hd'.
    seq: A sequence (e.g., shape tuple) of values to map to characters in term.
      If None, all characters are mapped to None.

  Returns:
    A dictionary mapping characters in term to elements in seq.
  """
  # If seq is None, then assume all dimensions should corresponds to None.
  if seq is None:
    return {c: None for c in term.replace('...', '')}
  if '...' not in term:
    return {c: seq[i] for i, c in enumerate(term)}
  assert term.count('...') == 1, 'Term must contain zero or exactly one `...`.'
  str_before_ellipsis, str_after_ellipsis = term.split('...')
  assert len(seq) >= len(str_before_ellipsis) + len(str_after_ellipsis), (
      f'Sequence {seq} is too short for term {term}.')
  char_dict = {}
  n_chars_before_ellipsis = len(str_before_ellipsis)
  n_chars_after_ellipsis = len(str_after_ellipsis)
  for i in range(n_chars_before_ellipsis):
    char = str_before_ellipsis[i]
    char_dict[char] = seq[i]
  n = len(seq)
  for i in range(n_chars_after_ellipsis):
    char = str_after_ellipsis[i]
    char_dict[char] = seq[n-n_chars_after_ellipsis+i]
  return char_dict


@ModuleRegistry.register
@dataclasses.dataclass
class EinsumLinear(SimplyModule):
  """An Einsum layer with learnable weights and optional bias.

  This layer performs a linear transformation using jnp.einsum. It supports
  `...` in input and output terms of the equation, but not in weight term.

  If `weight_dim_annotation` is not provided, it will be inferred from `eqn`:
  'i' for contracting dims (in weight and input, not output),
  'o' for output dims (in weight and output, not input),
  '.' for independent dims (in weight, input, and output).

  If `bias_term` is provided, a bias vector is created. `bias_term` must be a
  string containing characters that specify which dimensions of output are
  biased. Each character in `bias_term` must be present in both weight and
  output terms of `eqn`.

  Attributes:
    eqn: Einsum equation string, e.g., 'df,...d->...f'.
    weight_shape: Shape of weight tensor.
    bias_term: If provided, bias is enabled. It's a string specifying bias
      dimensions, e.g., 'f'. All characters in `bias_term` must appear in
      weight and output parts of `eqn`.
    weight_dim_annotation: Optional dimension annotation string for weight
      initialization, e.g., 'io'. If empty, it's inferred from `eqn`.
    weight_init: Initializer for weight tensor.
    bias_init: Initializer for bias tensor.
    weight_dtype: Dtype of weight tensor.
    activation_dtype: Dtype of layer activation.
    weight_partition: Sharding annotation for weight tensor.
    output_partition: Sharding annotation for layer output.
    weight_name: Name of weight parameter in parameter PyTree.
    bias_name: Name of bias parameter in parameter PyTree.
    bias_dim_annotation: Dimension annotation for bias tensor, inferred during
      setup if `bias_term` is provided.
    bias_partition: Sharding annotation for bias tensor, inferred during setup
      if `bias_term` is provided.
    bias_shape: Shape of bias tensor, inferred during setup if `bias_term` is
      provided.
    output_term: Output term of parsed einsum equation, set during setup if
      `bias_term` is provided to help with reshaping the bias in forward pass.
  """
  eqn: str
  weight_shape: Sequence[int]
  bias_term: str = ''
  weight_dim_annotation: str = ''
  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  bias_init: initializer.Initializer = initializer.ZeroInit()
  # Mixed precision related.
  weight_dtype: jax.typing.DTypeLike = 'float32'
  activation_dtype: jax.typing.DTypeLike = 'bfloat16'
  # Sharding related.
  weight_partition: common.PartitionAnnotation = None
  output_partition: common.PartitionAnnotation = (
      sharding_lib.NOT_ANNOTATED
  )
  # Others.
  weight_name: str = 'w'
  bias_name: str = 'b'

  def setup(self):
    (weight_term, input_term, output_term) = _parse_einsum_eqn(self.eqn)
    if len(weight_term) != len(self.weight_shape):
      raise ValueError(
          f'Einsum string weight part "{weight_term}" '
          f'does not match weight_shape {self.weight_shape}'
      )
    if self.weight_dim_annotation:
      if len(weight_term) != len(self.weight_dim_annotation):
        raise ValueError(
            f'Einsum string weight part "{weight_term}" '
            f'does not match weight_dim_annotation {self.weight_dim_annotation}'
        )
    else:
      # Infer the weight_dim_annotation from the einsum terms.
      self.weight_dim_annotation = ''
      for char in weight_term:
        if char in input_term:
          # Dimensions appearing in both input and output are independent.
          # Dimensions appearing in input but not in output are contracted thus
          # are input dimensions.
          self.weight_dim_annotation += ('.' if char in output_term else 'i')
        elif char in output_term:
          # Dimensions appearing in output but not in input are expanded thus
          # are output dimensions.
          self.weight_dim_annotation += 'o'
        else:
          raise ValueError(f"Invalid character '{char}' in einsum string.")
    weight_shape_char_dict = create_char_dict(weight_term, self.weight_shape)
    weight_dim_annotation_char_dict = create_char_dict(
        weight_term, self.weight_dim_annotation)
    output_partition_char_dict = create_char_dict(
        output_term, self.output_partition)
    if self.bias_term:
      assert '...' not in self.bias_term, ('Bias term cannot contain "...".')
      self.output_term = output_term
      self.bias_dim_annotation = ''
      bias_shape = []
      bias_partition = []
      for char in self.bias_term:
        if not (char in weight_term and char in output_term):
          raise ValueError(
              f'Character {char} in bias_term must be in both weight_term and'
              f' output_term.')
        # bias shape and dimension annotation should follow the weight's.
        bias_shape.append(weight_shape_char_dict[char])
        wd = weight_dim_annotation_char_dict[char]
        self.bias_dim_annotation += '.' if wd == '.' else 'h'
        # We assume bias should follow the output's sharding.
        bias_partition.append(output_partition_char_dict[char])
      self.bias_shape = tuple(bias_shape)
      self.bias_partition = tuple(bias_partition)

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    key_w, key_b = jax.random.split(prng_key)

    params[self.weight_name] = self.weight_init(
        key_w,
        shape=self.weight_shape,
        dtype=self.weight_dtype,
        dim_annotation=self.weight_dim_annotation,
    )
    if self.weight_partition is not None:
      params[self.weight_name] = sharding_lib.with_sharding_constraint(
          params[self.weight_name], self.weight_partition
      )
    params[self.weight_name] = AnnotatedArray.create(
        params[self.weight_name], dim_annotation=self.weight_dim_annotation)

    if self.bias_term:
      params[self.bias_name] = self.bias_init(
          key_b,
          shape=self.bias_shape,
          dtype=self.weight_dtype,
          dim_annotation=self.bias_dim_annotation,
      )
      params[self.bias_name] = sharding_lib.with_sharding_constraint(
          params[self.bias_name], self.bias_partition
      )
      params[self.bias_name] = AnnotatedArray.create(
          params[self.bias_name], dim_annotation=self.bias_dim_annotation)
    return params

  def apply(self, params: PyTree, x: Array) -> Array:
    raw_params = get_raw_arrays(params)
    x = jnp.asarray(x, dtype=self.activation_dtype)
    weight = common.convert_or_dequantize(
        raw_params[self.weight_name], dtype=self.activation_dtype)
    output = jnp.einsum(self.eqn, weight, x)

    if self.bias_term:
      bias = common.convert_or_dequantize(
          raw_params[self.bias_name], dtype=self.activation_dtype)
      output += _reshape_bias(
          bias, output_term=self.output_term, bias_term=self.bias_term,
          output_shape=output.shape)
    output = sharding_lib.with_sharding_constraint(
        output, self.output_partition
    )
    return output

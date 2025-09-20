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
import re
from typing import Any, ClassVar, cast, final

import jax
import jax.numpy as jnp
import jax.typing

from simply.utils import common
from simply.utils import initializer
from simply.utils import registry
from simply.utils import sharding

PyTree = common.PyTree


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


def _split_einsum_equation(equation: str) -> tuple[Sequence[str], str]:
  """Splits einsum equation.

  Args:
    equation: Einsum equation.

  Returns:
    A tuple of left equations and right equation.
  """
  eqn_sub = re.sub(r'\.\.\.', '0', equation)
  left_eqn, right_eqn = eqn_sub.split('->')
  if re.fullmatch(r'[0a-zA-Z]+(,[0a-zA-Z]+)*', left_eqn) is None:
    raise ValueError(f'Invalid einsum equation({equation})')
  if re.fullmatch(r'[0a-zA-Z]*', right_eqn) is None:
    raise ValueError(f'Invalid einsum equation({equation})')
  left_eqns = [e.group(0) for e in re.finditer(r'[0a-zA-Z]+', left_eqn)]
  return left_eqns, right_eqn


@ModuleRegistry.register
@dataclasses.dataclass
class EinsumLinear(SimplyModule):
  """Feed-forward Einsum linear layer."""

  eqn: str
  weight_shape: Sequence[int]
  bias_eqn: str = ''

  weight_init: initializer.Initializer = initializer.XavierUniformInit()
  weight_dim_annotation: str = ''

  # Sharding related.
  weight_partition: common.PartitionAnnotation = None
  output_partition: common.PartitionAnnotation = None
  # Mixed precision related.
  weight_dtype: jax.typing.DTypeLike = 'float32'
  activation_dtype: jax.typing.DTypeLike = 'bfloat16'
  # Others.
  weight_name: str = 'w'
  bias_name: str = 'b'

  def setup(self):
    left_eqns, self.output_eqn = _split_einsum_equation(self.eqn)
    if len(left_eqns) != 2:
      raise ValueError(
          f'EinsumLinear only accept 2-oprand equation: ({self.eqn})'
      )
    self.weight_eqn, self.input_eqn = left_eqns
    if '0' in self.weight_eqn:
      raise ValueError(f'Weight equation cannot contain (...): ({self.eqn})')
    if len(self.weight_shape) != len(self.weight_eqn):
      raise ValueError(
          f'Weight shape ({self.weight_shape}) dose not match weight equation'
          f' ({self.weight_eqn})'
      )
    if self.output_eqn.count('0') > 1:
      raise ValueError(
          f'Output equation cannot contain multiple (...): ({self.eqn})'
      )

    if not self.weight_dim_annotation:
      for c in self.weight_eqn:
        if c in self.input_eqn and c not in self.output_eqn:
          self.weight_dim_annotation += 'i'
        elif c in self.output_eqn and c not in self.input_eqn:
          self.weight_dim_annotation += 'o'
        elif c in self.input_eqn and c in self.output_eqn:
          self.weight_dim_annotation += '.'
        else:
          raise ValueError(
              f'Weight equation ({self.weight_eqn}) contains unrecognizable'
              f' position: ({c})'
          )

    if self.bias_eqn:
      if '0' in self.bias_eqn:
        raise ValueError(
            f'Bias equation cannot contain (...): ({self.bias_eqn})'
        )

      weight_index = {c: i for i, c in enumerate(self.weight_eqn)}
      # bias_partition follows weight_partition.
      self.bias_partition = [] if self.weight_partition else None
      self.bias_shape = []
      for c in self.bias_eqn:
        if c in weight_index:
          self.bias_shape.append(self.weight_shape[weight_index[c]])
          if self.bias_partition is not None:
            self.bias_partition.append(self.weight_partition[weight_index[c]])
        elif c == '1':
          self.bias_shape.append(1)
          if self.bias_partition is not None:
            self.bias_partition.append(None)
        else:
          raise ValueError(
              f'bias_eqn ({self.bias_eqn}) contains unrecognizable position:'
              f' ({c})'
          )

  def init(self, prng_key: jax.Array) -> PyTree:
    params = {}
    w = self.weight_init(
        prng_key,
        shape=self.weight_shape,
        dim_annotation=self.weight_dim_annotation,
        dtype=self.weight_dtype,
    )
    params[self.weight_name] = sharding.with_sharding_constraint(
        w, self.weight_partition
    )

    if self.bias_eqn:
      b = jnp.zeros(shape=self.bias_shape, dtype=self.weight_dtype)
      params[self.bias_name] = sharding.with_sharding_constraint(
          b, self.bias_partition
      )
    return params

  def apply(self, params: PyTree, x: Any) -> Any:
    w = common.convert_or_dequantize(
        params[self.weight_name], dtype=self.activation_dtype
    )
    output = jnp.einsum(self.eqn, w, x)
    output = sharding.with_sharding_constraint(output, self.output_partition)
    if self.bias_eqn:
      b = common.convert_or_dequantize(
          params[self.bias_name], dtype=self.activation_dtype
      )
      output += b
      output = sharding.with_sharding_constraint(output, self.output_partition)
    return output

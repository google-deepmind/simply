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
"""Library for controling how modules are executed."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import inspect
from typing import Any, Callable, TypeAlias

import einops  # pylint: disable=unused-import
import jax
import jax.numpy as jnp

from simply.utils import module as module_lib
from simply.utils import pytree
from simply.utils import registry
from simply.utils.common import PyTree  # pylint: disable=g-importing-member


@registry.RootRegistry.register
@dataclasses.dataclass(frozen=True)
class Constant:
  value: PyTree


OutputSpec: TypeAlias = (
    None | str | Sequence['OutputSpec'] | Mapping[str, 'OutputSpec']
)


@functools.partial(registry.RootRegistry.register, name='ControlStep')
@dataclasses.dataclass(kw_only=True)
class ControlStep:
  """Control step.

  It can be either a simply module or a function.

  1. When module is provided, fn is ignored
    module_name must be provided module's params can be stored under the name.
    module.module_fn is called at this step.

  2. When module is not provided,
    a. module_name is provided, fn is ignored.
      module's (stored under module_name) module_fn is called.
    b. fn is provided,
      fn can be a lambda function string or a registred function.
    c. neither module nor fn is provided,
      input is directly passed to output.

  ControlFlow automatically binds input parameters with corresponding state
  values, e.g. a=a, b=b. On top of that, overwrite_input_spec can override input
  parameters to different state values or constants.

  output_spec speficies the structured output of this step.
  """

  module_name: str = ''  # Module name.
  module: module_lib.SimplyModule | None = None
  fn: str = ''
  module_fn: str = 'apply'

  overwrite_input_spec: Mapping[str, str | Constant] = dataclasses.field(
      default_factory=dict
  )
  output_spec: OutputSpec = 'x'


def default_input_spec(fn: Callable[..., Any]) -> dict[str, str]:
  parameters = inspect.signature(fn).parameters
  return {
      k: k
      for k, v in parameters.items()
      if v.kind
      not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
  }


@module_lib.ModuleRegistry.register
@dataclasses.dataclass
class ControlFlow(module_lib.SimplyModule):
  """A sequence of ControlSteps that can be executed in order."""

  steps: Sequence[ControlStep]
  output_spec: OutputSpec = 'x'

  def setup(self) -> None:
    name_count = collections.defaultdict(int)
    self.modules: dict[str, module_lib.SimplyModule] = {}
    for step in self.steps:
      # Only module needs to be setup.
      if step.module:
        name = step.module_name
        count = name_count[step.module_name]
        if count > 0:
          name += f'__{count}'
          step.module_name = name
        name_count[step.module_name] += 1
        self.modules[step.module_name] = step.module

  def init(self, prng_key: jax.Array) -> PyTree:
    params = {}
    for step in self.steps:
      if step.module:
        prng_key, subkey = jax.random.split(prng_key)
        params[step.module_name] = step.module.init(subkey)
    return params

  def apply(self, params: PyTree, x: PyTree, **kwargs) -> Any:
    state = {'params': params, 'x': x, **kwargs}
    for step in self.steps:
      fn: Callable[..., Any] = lambda x: x
      if step.module_name:
        fn = getattr(self.modules[step.module_name], step.module_fn)
      elif step.fn:
        if step.fn.startswith('lambda '):
          try:
            fn = eval(step.fn)  # pylint: disable=eval-used
          except Exception as e:  # pylint: disable=broad-except
            e.add_note(f'Failed to parse lambda expression: {step.fn}')
            raise e
        else:
          fn = registry.FunctionRegistry.get(step.fn)
          if not isinstance(fn, Callable):
            raise ValueError(f'Function {step.fn} is not a callable in {step}.')
      input_spec = default_input_spec(fn)
      if step.module_name:
        input_spec['params'] = f'params/{step.module_name}'
      for k, v in step.overwrite_input_spec.items():
        input_spec[k] = v
      step.input_spec = input_spec

      input_parameters = {}
      for k, v in input_spec.items():
        if isinstance(v, Constant):
          input_parameters[k] = v.value
        else:
          try:
            input_parameters[k] = pytree.tree_value(state, v)
          except KeyError:
            signature = inspect.signature(fn).parameters[v]
            input_parameters[k] = signature.default

      try:
        output = fn(**input_parameters)
      except Exception as e:  # pylint: disable=broad-except
        e.add_note(f'Failed to run step: {step}')
        e.add_note(f'with input parameters: {input_parameters}')
        raise e

      def _set_state_value(v, p, output=output, state=state):
        pytree.set_tree_value(state, v, pytree.tree_value(output, p))

      pytree.traverse_tree_with_path(_set_state_value, step.output_spec)

    output = pytree.traverse_tree_with_path(
        lambda v, _: pytree.tree_value(state, v), self.output_spec
    )
    return output


@module_lib.ModuleRegistry.register
@dataclasses.dataclass
class ScanModule(module_lib.SimplyModule):
  """Scan module."""

  module: module_lib.SimplyModule
  length: int = 1
  unroll: bool | int = 1
  per_step_args: Sequence[str] = dataclasses.field(default_factory=list)
  overwrite_input_spec: Mapping[str, str | Constant] = dataclasses.field(
      default_factory=dict
  )

  def setup(self) -> None:
    self.module.setup()

  def init(self, prng_key: jax.Array) -> PyTree:
    params = []
    for _ in range(self.length):
      prng_key, subkey = jax.random.split(prng_key)
      params.append(self.module.init(subkey))
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *params)

  def apply(
      self, params: PyTree, x: PyTree, **kwargs: Mapping[str, Any]
  ) -> PyTree:
    global_state = {}
    per_step_state = {'params': params}
    for k, v in kwargs.items():
      if k in self.per_step_args:
        per_step_state[k] = v
      else:
        global_state[k] = v

    def _process_module(x, state):
      local_state = {**global_state, **state, 'x': x}
      input_spec = default_input_spec(self.module.apply)
      for k, v in self.overwrite_input_spec.items():
        input_spec[k] = v
      input_parameters = {}
      for k, v in input_spec.items():
        if isinstance(v, Constant):
          input_parameters[k] = v.value
        else:
          try:
            input_parameters[k] = pytree.tree_value(local_state, v)
          except KeyError:
            signature = inspect.signature(self.module.apply).parameters[v]
            input_parameters[k] = signature.default
      result = self.module.apply(**input_parameters)
      return result

    output = jax.lax.scan(
        jax.remat(_process_module),
        init=x,
        xs=per_step_state,
        length=self.length,
        unroll=self.unroll,
    )
    return output

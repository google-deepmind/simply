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
"""LLM bridging layer using litellm."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import os
from typing import Annotated, Any, ClassVar

import litellm

from simply.agent import tools as tools_lib
from simply.utils import registry


class LLMRegistry(registry.RootRegistry):
  """Registry for LLM providers."""
  namespace: ClassVar[str] = 'LLM'

  @classmethod
  def get_llm(
      cls,
      llm_scheme: str
  ) -> 'LLMBase':
    """Gets an LLM instance from a scheme string.

    Args:
      llm_scheme: String in format "provider:spec" where spec can start with
        '$' to reference an environment variable.

    Returns:
      An instance of LLMBase for the specified provider.

    Raises:
      ValueError: If environment variable reference is not found.
    """
    llm_provider, llm_spec = llm_scheme.split(':', 1)
    if llm_spec.startswith('$'):  # Replace with environment variable lookup
      llm_spec = llm_spec.removeprefix('$')
      llm_spec_env = os.environ.get(llm_spec, '')
      if not llm_spec_env:
        raise ValueError(f'Environment variable {llm_spec} not found.')
      llm_spec = llm_spec_env
    return cls.get(llm_provider)(llm_spec)


@dataclasses.dataclass(frozen=True)
class ToolCall:
  """Tool call."""
  name: Annotated[str, 'Name of the tool.']
  arguments: Annotated[str, 'JSON string of arguments.']


@dataclasses.dataclass(frozen=True)
class LLMOutput:
  """LLM output."""
  text: str
  tool_calls: list[ToolCall]


@dataclasses.dataclass(frozen=True)
class LLMBase(abc.ABC):
  """Base class for LLM interfaces."""
  llm_spec: str

  @abc.abstractmethod
  def completion(
      self,
      messages: Sequence[Mapping[str, Any]],
      tools: Sequence[tools_lib.Tool],
      system_prompt: str | None = None,
      num_retries: int | None = None
  ) -> LLMOutput:
    """Query LLM to generate a response."""

  @property
  @abc.abstractmethod
  def max_tokens(self) -> int:
    """Returns the maximum number of tokens supported."""

  @abc.abstractmethod
  def count_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
    """Returns the approxmate number of tokens in the given messages."""


@dataclasses.dataclass(frozen=True)
@LLMRegistry.register
class LiteLLM(LLMBase):
  """LiteLLM interface."""

  def completion(
      self,
      messages: Sequence[Mapping[str, Any]],
      tools: Sequence[tools_lib.Tool],
      system_prompt: str | None = None,
      num_retries: int | None = None,
  ) -> LLMOutput:
    """Query LLM to generate a response.

    Args:
      messages: Conversation history in OpenAI message format.
      tools: List of available tools for function calling.
      system_prompt: Optional system prompt to prepend to messages.
      num_retries: Number of retry attempts for API calls.

    Returns:
      LLMOutput containing text response and any tool calls.
    """
    tools_schema = [tool.schema for tool in tools]
    if system_prompt:
      messages = [{'role': 'system', 'content': system_prompt}] + list(messages)
    if self.llm_spec.startswith('vertex_ai/claude-opus-'):
      reasoning_args = {  # explicit config for Opus 4.7
          'output_config': {'effort': 'high'},
          'thinking': {'type': 'adaptive'},
      }
    else:  # let LiteLLM handle mapping to underlying API parameters
      reasoning_args = {'reasoning_effort': 'high'}
    kwargs = dict(
        model=self.llm_spec,
        messages=messages,
        tools=tools_schema if tools_schema else None,
        num_retries=num_retries,
        **reasoning_args,
    )
    res = litellm.completion(**kwargs)
    llm_message = res.choices[0].message
    llm_content = (llm_message.content or '').strip()
    if hasattr(llm_message, 'tool_calls') and llm_message.tool_calls:
      tool_calls = [
          ToolCall(
              name=tc.function.name,
              arguments=tc.function.arguments,
          )
          for tc in llm_message.tool_calls
      ]
    else:
      tool_calls = []
    return LLMOutput(text=llm_content, tool_calls=tool_calls)

  @property
  def max_tokens(self) -> int:
    """Returns the maximum number of tokens supported."""
    model_info = litellm.get_model_info(self.llm_spec)
    return model_info['max_input_tokens']  # pyrefly: ignore[bad-return]

  def count_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
    """Returns the approxmate number of tokens in the given messages.

    Args:
      messages: Conversation history in OpenAI message format.

    Returns:
      Estimated token count for the messages.
    """
    return litellm.token_counter(model=self.llm_spec, messages=messages)  # pyrefly: ignore[bad-argument-type]

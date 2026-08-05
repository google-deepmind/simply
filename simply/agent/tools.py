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
"""Tools for the agent."""

import dataclasses
import json
from typing import Annotated, Any, Callable, Mapping, Type

import pydantic


class Action(pydantic.BaseModel):
  """An action describe a tool use call.

  We use pydantic for automatic JSON schema generation and input validation. The
  initializer will raise pydantic.ValidationError if the input is invalid.
  """

  def to_llm(self) -> str:
    """Returns the string representation of the action for LLM consumption."""
    return self.model_dump_json(indent=2)


class ToolCallError(Exception):
  """An exception raised by a tool."""
  # If a tool executor want to raise an exception that is automatically caught
  # and reported by the agent loop, it should raise this or a subclass of this.
  # General Python exceptions indicate system bugs or errors, we will let it
  # propagate to the agent loop and crash the program.


@dataclasses.dataclass(frozen=True, kw_only=True)
class Tool[T: Action]:
  """A tool for the agent."""
  name: Annotated[str, 'The name of the tool.']
  description: Annotated[str, 'The description of the tool.']
  action_type: Annotated[Type[T], 'The type of the action.']
  executor: Annotated[Callable[[T], str], 'The executor of the action.']

  @property
  def schema(self) -> Mapping[str, Any]:
    """Returns the JSON schema of the action type."""
    return {
        'type': 'function',
        'function': {
            'name': self.name,
            'description': self.description,
            'parameters': self.action_type.model_json_schema(),
        }
    }

  def execute(self, args_json: str) -> tuple[Action | None, str]:
    """Executes the action.

    Args:
      args_json: The JSON string of the action arguments (from the LLM).

    Returns:
      A tuple of the action and the result of the execution. If the input JSON
      is invalid or the arguments are invalid according to the tool schema,
      returns (None, free-form string error message). Otherwise, returns
      (action, result), where action is the validated action object and result
      is the string result of the execution.
    """
    try:
      kwargs = json.loads(args_json)
      action = self.action_type(**kwargs)
    except json.JSONDecodeError as e:
      return None, 'Error: Invalid JSON input, ' + str(e)
    except pydantic.ValidationError as e:
      return None, f'Error: Invalid arguments for tool {self.name}, ' + str(e)
    try:
      return action, self.executor(action)
    except ToolCallError as e:
      return action, f'Error: Caught {type(e).__name__}, ' + str(e)


# ==============================================================================
# Bash Tool
# ==============================================================================
# Use 5 min as default timeout, as many tools (e.g. build and launch
# commands) could have long delays.
DEFAULT_TIMEOUT = 300.0
# Sufficient for catting ~800 lines of code of average line width 40. The LLM
# can increase this if needed, or use more concise commands.
DEFAULT_MAX_OUTPUT_LENGTH = 32_000


def truncate_text(text: str, max_length: int) -> str:
  """Truncates the text to the maximum number of characters."""
  if len(text) <= max_length:
    return text
  info_tmpl = '\n[...{} chars omitted...]\n'
  # truncated length is smaller than max_length, but may not be exactly equal
  max_digits = str(len(text))  # overestimation to simplify implementation
  max_length -= len(info_tmpl.format(max_digits))
  head, tail = text[: max_length // 2], text[-max_length // 2 :]
  info = info_tmpl.format(len(text) - len(head) - len(tail))
  return head + info + tail


class BashAction(Action):
  """An action to execute a bash command."""
  command: Annotated[
      str, pydantic.Field(description='The bash command to execute.')
  ]
  timeout: Annotated[
      float | None,
      pydantic.Field(
          description=f'The timeout in seconds (default is {DEFAULT_TIMEOUT}).'
      ),
  ] = DEFAULT_TIMEOUT
  max_output_length: Annotated[
      int | None,
      pydantic.Field(
          description=(
              'The maximum number of characters of the output (default is'
              + f' {DEFAULT_MAX_OUTPUT_LENGTH}).'
          )
      ),
  ] = DEFAULT_MAX_OUTPUT_LENGTH

  def to_llm(self) -> str:
    """Returns the string representation of the action for LLM consumption."""
    return f'COMMAND:\n{self.command}\nTIMEOUT: {self.timeout:.2f}'

  @classmethod
  def format_output(
      cls,
      stdout: str,
      stderr: str,
      exit_code: int,
      elapsed: float,
      max_output_length: int | None,
  ) -> str:
    """Formats the output of the bash command."""
    parts = []
    if stdout:
      parts.append(f'STDOUT:\n{stdout}')
    if stderr:
      parts.append(f'STDERR:\n{stderr}')
    parts += [
        f'RETURN CODE: {exit_code}',
        f'TIME ELAPSED: {elapsed:.2f}',
    ]
    text = '\n\n'.join(parts)
    if max_output_length is not None and len(text) > max_output_length:
      truncated_parts = []
      char_budget = max_output_length
      char_budget -= 2 * (len(parts) - 1)  # for the double newlines
      # go in reverse order, prioritize STDERR over STDOUT
      for part in reversed(parts):
        truncated_parts.append(truncate_text(part, char_budget))
        char_budget -= len(truncated_parts[-1])
        if char_budget <= 0:
          break
      text = '\n\n'.join(reversed(truncated_parts))
    return text


@dataclasses.dataclass(frozen=True, kw_only=True)
class BashTool(Tool[BashAction]):
  """Bash tool."""

  executor: Callable[[BashAction], str]

  name: str = 'bash'
  description: str = (
      'Executes a bash command in a shell. This tool is useful for executing' +
      ' shell commands, such as ls, grep, sed, awk, etc.'
  )
  action_type: Type[BashAction] = BashAction

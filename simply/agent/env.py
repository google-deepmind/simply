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
"""Execution environments for the agent."""

import abc
import functools
import getpass
import os
import re
import subprocess
import time
from typing import ClassVar
import uuid

from simply.agent import tools as tools_lib
from simply.utils import registry


class EnvRegistry(registry.RootRegistry):
  """Registry for execution environments."""

  namespace: ClassVar[str] = 'Env'

  @classmethod
  def get_env(cls, env_spec: str) -> 'Environment':
    """Factory method to create an environment from a spec.

    Args:
      env_spec: String in format "provider:param".

    Returns:
      An instance of Environment.
    """
    provider, param = env_spec.split(':', 1)
    return cls.get(provider)(param)


class Environment(abc.ABC):
  """Abstract base class for execution environments.

  An environment provides tools for interacting with the execution context
  (e.g. running bash commands).
  """

  @abc.abstractmethod
  def get_tools(self) -> list[tools_lib.Tool]:
    """Returns tools for interacting with this environment.

    Must include at least a BashTool.
    """


def execute_bash_locally(
    action: tools_lib.BashAction, cwd: str | None = None
) -> str:
  """Executes a bash command locally via subprocess.

  Args:
    action: The bash action to execute.
    cwd: The working directory for the command. If None, uses the current
      process working directory.

  Returns:
    The formatted output of the command.
  """
  try:
    start_time = time.perf_counter()
    result = subprocess.run(
        action.command,
        shell=True,
        check=False,
        text=True,
        encoding='utf-8',
        errors='backslashreplace',
        capture_output=True,
        timeout=action.timeout,
        cwd=cwd,
    )
    stop_time = time.perf_counter()
    return tools_lib.BashAction.format_output(
        result.stdout,
        result.stderr,
        result.returncode,
        stop_time - start_time,
        action.max_output_length,
    )
  except subprocess.TimeoutExpired as e:
    text = tools_lib.BashAction.format_output(
        e.stdout.decode('utf-8', errors='backslashreplace')  # pyrefly: ignore[bad-argument-type]
        if isinstance(e.stdout, bytes)
        else e.stdout,
        e.stderr.decode('utf-8', errors='backslashreplace')  # pyrefly: ignore[bad-argument-type]
        if isinstance(e.stderr, bytes)
        else e.stderr,
        1,
        action.timeout,  # pyrefly: ignore[bad-argument-type]
        action.max_output_length,
    )
    return f'Command timed out after {action.timeout} seconds.\n\n' + text


@EnvRegistry.register
class Local(Environment):
  """An environment that executes commands locally via subprocess."""

  def __init__(self, working_dir: str):
    self.working_dir = os.path.abspath(working_dir)

  def __str__(self) -> str:
    return f'{self.__class__.__name__}:{self.working_dir}'

  def get_tools(self) -> list[tools_lib.Tool]:
    """Returns a BashTool configured for local execution."""
    executor = functools.partial(execute_bash_locally, cwd=self.working_dir)
    return [
        tools_lib.BashTool(
            executor=executor,
            description=tools_lib.BashTool.description
            + f' CWD={self.working_dir!r}.',
        )
    ]

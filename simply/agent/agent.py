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
"""The agent drives the LLM / tool call loop."""

import dataclasses
import pickle
import pprint
from typing import Annotated, Any

from absl import logging
from etils import epath
from simply.agent import env as env_lib
from simply.agent import llm as llm_lib
from simply.agent import memory as mem_lib
from simply.agent import tools as tools_lib
from simply.agent import tui as tui_lib


def save_pickle_atomic(obj: Any, path: epath.Path):
  """Saves an object to a pickle file with atomic write and backup.

  If the file already exists, it is first backed up to `path.bak`.

  Args:
    obj: The object to pickle.
    path: The destination path.
  """
  if path.exists():
    backup_path = path.parent / (path.name + '.bak')
    path.replace(backup_path)
  with path.open('wb') as f:
    pickle.dump(obj, f)


@dataclasses.dataclass
class Agent:
  """Agent with a memory system that drives the LLM / tool call loop.

  The agent uses an `Environment` to obtain execution tools (e.g. BashTool)
  and combines them with memory management tools.
  """

  task: Annotated[str, 'The task to solve.']
  env: Annotated[env_lib.Environment, 'The execution environment.']
  predefined_knowledge: Annotated[
      list[mem_lib.MemoryFile],
      'The existing knowledge useful for the task. Can be empty.',
  ]
  llm_scheme: Annotated[str, 'The LLM to use (see llm.py).']
  session_dir: Annotated[epath.Path, 'The directory to store session files.']
  tui: Annotated[
      tui_lib.DisplayBase, 'The TUI display for the agent.'
  ] = dataclasses.field(default_factory=tui_lib.PrintDisplay)

  llm: Annotated[llm_lib.LLMBase, 'The LLM interface.'] = dataclasses.field(
      init=False
  )
  memory_system: Annotated[mem_lib.MemorySystem, 'The memory system.'] = (
      dataclasses.field(init=False)
  )
  tools: Annotated[
      dict[str, tools_lib.Tool], 'The tools available to the agent.'
  ] = dataclasses.field(init=False)

  def __post_init__(self):
    self.llm = llm_lib.LLMRegistry.get_llm(self.llm_scheme)
    self.memory_system = mem_lib.MemorySystem(
        task=self.task,
        predefined_knowledge=self.predefined_knowledge,
        max_token_budget=self.llm.max_tokens,
    )
    all_tools = self.env.get_tools() + mem_lib.get_memory_tools(
        self.memory_system
    )
    self.tools = {tool.name: tool for tool in all_tools}

  def _token_counter(self, text: str) -> int:
    """Token counter using the agent's LLM."""
    return self.llm.count_tokens([{'role': 'user', 'content': text}])

  def restore_memory_system(self, memory_system: mem_lib.MemorySystem):
    """Restores a previously saved MemorySystem (e.g. for resuming).

    This replaces the agent's memory system and re-binds memory tools to it.

    Args:
      memory_system: A MemorySystem loaded from disk (e.g. via pickle).
    """
    self.memory_system = memory_system
    # Re-bind memory tools to the restored memory system.
    for tool in mem_lib.get_memory_tools(self.memory_system):
      self.tools[tool.name] = tool

  def step(self) -> str | None:
    """Run a single step of the agent.

    Returns:
      The final LLM output if the agent is finished, None otherwise.
    """
    step = self.memory_system.last_snapshot.system_status.status_step + 1
    system_status = self.memory_system.last_snapshot.system_status
    self.tui.update_system_status(system_status)
    self.tui.update_status(f'\u23f3 Calling LLM (step {step})...')

    user_message = 'Start.' if step == 0 else 'Continue.'
    context_messages = [
        {'role': 'system', 'content': self.memory_system.llm_view},
        # some APIs requires at least one user message
        {'role': 'user', 'content': user_message},
    ]

    llm_message = self.llm.completion(
        messages=context_messages,
        tools=list(self.tools.values()),
        num_retries=5,
    )
    with self.memory_system.capture_snapshot(token_counter=self._token_counter):
      if llm_message.text:
        llm_output = llm_message.text.strip()
        self.memory_system.record_llm_output(llm_output)
        self.tui.display_llm_output(llm_output)

      if not llm_message.tool_calls:
        # When LLM does not call any tool, we consider the task finished
        # Log the entire final LLM message for debugging purpose.
        logging.info('Final LLM message: %s', pprint.pformat(llm_message))
        return (llm_message.text or '').strip()

      n_tools = len(llm_message.tool_calls)
      for idx, tool_call in enumerate(llm_message.tool_calls):
        self.tui.update_status(
            f'\U0001f527 Running tool: {tool_call.name}'
            f' [{idx + 1}/{n_tools}]'
        )
        self.make_tool_call(
            tool_call.name,
            tool_call.arguments,
            idx,
        )
    return None

  def make_tool_call(
      self, name: str, args: str, tool_call_idx: int
  ):
    """Makes a tool call and records the result in the memory."""
    if name not in self.tools:
      action, result = None, f'Error: Tool {name} not found.'
    else:
      action, result = self.tools[name].execute(args)
    tool_inputs = action.to_llm() if action is not None else args
    self.memory_system.record_tool_call(
        name, tool_call_idx, tool_inputs, result
    )
    self.tui.display_tool_call(name, tool_inputs, result)

  def save_memory_snapshot(self):
    """Saves the memory snapshot to the session directory."""
    ctx_view_dir = self.session_dir / 'context_view'
    ctx_view_dir.mkdir(parents=True, exist_ok=True)
    curr_step = self.memory_system.last_snapshot.system_status.status_step + 1
    (ctx_view_dir / f'ctx_for_step_{curr_step:06d}.md').write_text(
        self.memory_system.llm_view
    )
    save_pickle_atomic(
        self.memory_system, self.session_dir / 'memory_system.pkl'
    )

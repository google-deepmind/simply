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
"""Memory system for the agent."""

import contextlib
import copy
import dataclasses
import datetime
import enum
import functools
import json
import pathlib
import re
from typing import Annotated, Any, Callable
import xml.sax.saxutils

import pydantic
from simply.agent import tools as tools_lib


SYSTEM_PROMPT = r"""# AI / ML Research Agent

You are an autonomous agent for solving AI / ML research tasks. You are given a
task and a set of tools. Your goal is to solve the task using the tools.

After you finish the task, report your findings in the final message. The agent
loop will stop when you produce a message without any tool calls.
"""

_MODULE_DIR = pathlib.Path(__file__).parent


################################################################################
# Utilities
################################################################################
escape_xml = xml.sax.saxutils.escape
escape_cdata = lambda x: x.replace(']]>', ']]>]]><![CDATA[>')


# Display labels for each URI scheme, used in llm_view section headings.
SCHEME_LABELS: dict[str, str] = {
    'kb://': 'Knowledge Base',
    'pad://': 'Scratchpad',
    'log://': 'Event Log',
}

REGEX_MEMORY_URI = re.compile(r'(log|kb|pad)://([a-zA-Z0-9_\-. ]+)')


def is_valid_memory_uri(uri: str) -> bool:
  """Checks if the memory URI is valid."""
  return re.fullmatch(REGEX_MEMORY_URI, uri) is not None


def approximate_token_count(text: str) -> int:
  """Returns the approximate token count of the text."""
  return len(text) // 4


################################################################################
# Contents in the Memory
################################################################################
class DisplayMode(enum.StrEnum):
  SUMMARY = 'summary'
  FULL = 'full'


@dataclasses.dataclass(frozen=True)
class MemoryFile:
  """A file in the memory."""
  uri: Annotated[str, 'The URI of the file (e.g. kb://notes.md).']
  display: Annotated[DisplayMode, 'The display mode of the file.']
  content: Annotated[str, 'The full content of the file.']
  summary: Annotated[str, 'The summary of the file.']
  update_step: Annotated[int, 'The step when the file was updated or created.']
  metadata: Annotated[dict[str, Any], 'The metadata of the file.'] = (
      dataclasses.field(default_factory=dict)
  )

  def __post_init__(self):
    if not is_valid_memory_uri(self.uri):
      raise ValueError(f'Invalid memory URI: {self.uri}')

  def to_llm(self, force_display: DisplayMode | None = None) -> str:
    """Returns the string representation of the file for LLM consumption."""
    display = force_display or self.display
    lines = [
        f'<memory uri="{self.uri}" display="{display.value}"'
        f' length="{len(self.content)}" update_step="{self.update_step}">'
    ]
    lines.append(f'<summary>{escape_xml(self.summary)}</summary>')
    if display == DisplayMode.FULL:
      if not self.content:
        lines.append('<content></content>')
      else:
        lines.append('<content><![CDATA[')
        lines.append(escape_cdata(self.content))
        lines.append(']]></content>')
    lines.append('</memory>')
    return '\n'.join(lines)


class SystemStatus(pydantic.BaseModel):
  """The status of the agent system."""
  model_config = pydantic.ConfigDict(frozen=True)
  max_token_budget: Annotated[int, 'The max token budget of the agent.']
  approximate_token_usage: Annotated[
      int, 'The approximate current token usage of the agent.'
  ]
  # The status is generated for the context view of LLM prompt at step t,
  # based on previous conversation history up to step t-1. So the approximate
  # token usage only count the token usage up to step t-1. But the status_step
  # is t.
  status_step: Annotated[int, 'The step when the system status is generated.']
  current_time: Annotated[datetime.datetime, 'The current time.']
  elapsed_seconds: Annotated[
      float, 'The elapsed time in seconds since the beginning of the task.'
  ]


class MetricEntry(pydantic.BaseModel):
  """A single metric measurement for a specific variant."""
  model_config = pydantic.ConfigDict(frozen=True)

  variant: Annotated[
      str,
      pydantic.Field(
          description='The experiment variant, e.g. "baseline", "lr1e-7".'
      ),
  ]
  metric: Annotated[
      str,
      pydantic.Field(
          description=(
              'The metric name, e.g. "accuracy", "loss". When tracking the'
              ' same metric, use the same name across iterations for grouping.'
          )
      ),
  ]
  value: Annotated[
      float,
      pydantic.Field(description='The scalar metric value.'),
  ]


class ProgressEntry(pydantic.BaseModel):
  """A progress log entry recorded by the agent."""
  model_config = pydantic.ConfigDict(frozen=True)

  step: Annotated[int, 'The step when this progress was recorded.']
  description: Annotated[str, 'A text description of the progress.']
  metrics: Annotated[list[MetricEntry], 'A list of metric entries.']


################################################################################
# Memory System
################################################################################
CONTEXT_ERROR_MSG = (
    'Memory operation should be executed in a `capture_snapshot` context.'
)


@dataclasses.dataclass(frozen=True)
class Memory:
  """The memory of the agent."""
  files: Annotated[
      dict[str, MemoryFile], 'The files in the memory, keyed by URI.'
  ]
  log_order: Annotated[list[str], 'The ordered URIs of the log files.']


@dataclasses.dataclass(frozen=True)
class MemorySnapshot:
  """The memory snapshot."""
  memory: Annotated[Memory, 'The actual memory.']
  system_status: Annotated[SystemStatus, 'The status of the system.']
  llm_view: Annotated[str, 'The LLM view of the memory.']


@dataclasses.dataclass(kw_only=True)
class MemorySystem:
  """The memory system.

  The lifecycle of the memory system:
  - After initialization:
    - `self.memory_snapshots` has a single snapshot of step 0.
    - `self.memory` is set to None.
  - While in `capture_snapshot` context:
    - `self.memory` is constructed as a deep copy of the last snapshot.
    - events happened in this step are recorded in `self.memory`.
    - memory tool calls will modify `self.memory`.
  - After exiting the `capture_snapshot` context:
    - a new snapshot is created based on `self.memory` from the context,
      and appended to `self.memory_snapshots`.
    - `self.memory` is set to None.
  """

  max_token_budget: Annotated[int, 'The max token budget of the agent.']
  task: Annotated[str, 'The description of the task to solve.']
  system_prompt: Annotated[str, 'The system prompt for the agent.'] = (
      SYSTEM_PROMPT
  )
  predefined_knowledge: Annotated[
      list[MemoryFile], 'Predefined knowledge useful for the task.'
  ] = dataclasses.field(default_factory=list)
  default_summary_length: Annotated[
      int, 'The default length for auto generated summaries.'
  ] = 600
  display_memory_tool_calls: Annotated[
      bool,
      'Whether to display the memory tool calls in the log.',
  ] = False
  n_full_display_recent_steps: Annotated[
      int,
      'The number of recent steps to display in full mode.',
  ] = 50

  # Non-init fields, will be set in __post_init__.
  memory_snapshots: Annotated[list[MemorySnapshot], 'The memory snapshots.'] = (
      dataclasses.field(init=False)
  )
  start_time: Annotated[datetime.datetime, 'The start time of the agent.'] = (
      dataclasses.field(init=False)
  )
  memory: Annotated[
      Memory,
      'The current memory. This object will always be None unless in' +
      ' the `capture_snapshot` context.',
  ] = dataclasses.field(init=False)
  memory_system_description: Annotated[str, 'Memory system description.'] = (
      dataclasses.field(init=False)
  )
  progress_log: Annotated[list[ProgressEntry], 'The progress log entries.'] = (
      dataclasses.field(init=False)
  )

  def __post_init__(self):
    self.start_time = datetime.datetime.now()
    self.progress_log = []

    self.memory_system_description = (
        _MODULE_DIR / 'memory.md'
    ).read_text()

    # initialize the starting memory snapshot
    files = {
        f'pad://{fn}': MemoryFile(
            uri=f'pad://{fn}',
            display=DisplayMode.FULL,
            content='',
            summary='',
            update_step=0,
        )
        for fn in ('plan.md', 'todo.md', 'scratch.md', 'journey.md')
    }
    for knowledge in self.predefined_knowledge:
      if not knowledge.uri.startswith('kb://'):
        raise ValueError(
            f'Predefined knowledge URI {knowledge.uri} does not start with'
            ' kb://.'
        )
      if not knowledge.summary:
        summary = self.generate_summary(knowledge.content)
        knowledge = dataclasses.replace(knowledge, summary=summary)
      files[knowledge.uri] = knowledge

    self.memory = None  # pyrefly: ignore[bad-assignment]
    self.memory_snapshots = []
    self._record_memory_snapshot(
        Memory(files=files, log_order=[]),
        snapshot_step=0,
        token_counter=approximate_token_count,
    )

  @property
  def last_snapshot(self) -> MemorySnapshot:
    """Returns the last memory snapshot."""
    # Note `self.memory_snapshots` is guaranteed to have a least one snapshot
    # after object initialization.
    return self.memory_snapshots[-1]

  @property
  def llm_view(self) -> str:
    """The LLM view of the last memory snapshot."""
    return self.last_snapshot.llm_view

  def get_display_mode(
      self, file: MemoryFile, display_step: int
  ) -> DisplayMode | None:
    """Returns the display mode of the file for a given step.

    In addition to the file's own display setting, there are some heuristics to
    determine the display mode. We put the logic here so it is reusable by
    `visualizer.py`.

    Args:
      file: The memory file.
      display_step: The step of the LLM context view we are generating.

    Returns:
      The actual display mode of the file. If None, the file should not be
      displayed (neither summary nor full).
    """
    # Always display the pad (whiteboard) files in full display mode.
    if file.uri.startswith('pad://'):
      return DisplayMode.FULL

    if file.uri.startswith('log://'):  # logic for event log
      event_step = file.update_step
      if event_step >= display_step - self.n_full_display_recent_steps:
        # Always display the recent events in full display mode.
        return DisplayMode.FULL

      # -- rest are older events --
      if (
          file.metadata['event_type'] == 'tool_call'
          and file.metadata['tool_name'].startswith('mem_')
          and not self.display_memory_tool_calls
      ):  # Skip all older memory tool calls if configured to do so
        return None

    return file.display  # file's own display setting

  def _record_memory_snapshot(
      self,
      memory: Memory,
      snapshot_step: int,
      token_counter: Callable[[str], int] = approximate_token_count,
  ):
    """Makes a memory snapshot and push to `self.memory_snapshots`."""
    llm_view = self._memory_to_llm_view(memory, snapshot_step=snapshot_step)
    approx_token_usage = token_counter(llm_view)
    system_status = self._make_system_status(
        approx_token_usage, status_step=snapshot_step
    )
    system_status_json = system_status.model_dump_json(indent=2)
    if self.progress_log:
      progress_json = json.dumps(
          [e.model_dump() for e in self.progress_log],
          indent=2,
      )
      llm_view += (
          '\n\n## Progress Log\n\n```json\n'
          + progress_json
          + '\n```\n'
      )
    llm_view += (
        '\n\n## System Status\n\n```json\n'
        + system_status_json.strip()
        + '\n```\n'
    )
    self.memory_snapshots.append(MemorySnapshot(
        memory=memory,
        system_status=system_status,
        llm_view=llm_view,
    ))

  def _memory_to_llm_view(self, memory: Memory, snapshot_step: int) -> str:
    """Returns the LLM view of the memory."""
    blocks = [
        self.system_prompt,
        self.task,
        self.memory_system_description,
        '## Memory Contents',
    ]
    for scheme in ['kb://', 'pad://']:
      if view := self._user_files_to_llm_view(
          memory, scheme=scheme, snapshot_step=snapshot_step
      ):
        blocks.append(view)
    blocks.append(self._log_to_llm_view(memory, snapshot_step=snapshot_step))
    return '\n\n'.join(blocks)

  def _log_to_llm_view(self, memory: Memory, snapshot_step: int) -> str:
    """Returns the LLM view of the event log in memory."""
    lines = ['### ' + SCHEME_LABELS['log://']]
    for uri in memory.log_order:
      file = memory.files[uri]
      if display := self.get_display_mode(file, snapshot_step):
        lines.append(memory.files[uri].to_llm(force_display=display))
    return '\n\n'.join(lines)

  def _user_files_to_llm_view(
      self, memory: Memory, scheme: str, snapshot_step: int
  ) -> str:
    """Returns the LLM view of the user files in memory."""
    user_files = [
        file for uri, file in memory.files.items() if uri.startswith(scheme)
    ]
    if not user_files:
      return ''
    lines = [f'### {SCHEME_LABELS[scheme]}']
    for file in user_files:
      if display := self.get_display_mode(file, snapshot_step):
        lines.append(file.to_llm(force_display=display))
    return '\n\n'.join(lines)

  def generate_summary(self, content: str) -> str:
    """Generates the summary for the content."""
    return tools_lib.truncate_text(content, self.default_summary_length)

  def _make_system_status(
      self, approximate_token_usage: int, status_step: int
  ) -> SystemStatus:
    """Gets the current system status.

    A `SystemStatus` object belongs to a `MemorySnapshot`. When LLM sees
    the status in the context window at step t, it actually represents the
    information of the previous snapshot (i.e. step t-1).

    Args:
      approximate_token_usage: The approximate token usage of the memory.
      status_step: The step for which the system status is generated.

    Returns:
      The system status.
    """
    now = datetime.datetime.now()
    elapsed_seconds = (now - self.start_time).total_seconds()
    return SystemStatus(
        max_token_budget=self.max_token_budget,
        approximate_token_usage=approximate_token_usage,
        status_step=status_step,
        current_time=now,
        elapsed_seconds=elapsed_seconds,
    )

  @contextlib.contextmanager
  def capture_snapshot(
      self,
      token_counter: Callable[[str], int] = approximate_token_count,
  ):
    """Enter the context of capturing a memory snapshot.

    All the tool calls that modifies the memory needs to be executed in this
    context. Upon exiting the context, the memory snapshot will be captured and
    stored in `self.memory_snapshots`.

    Args:
      token_counter: A callable that counts tokens in a string.

    Raises:
      RuntimeError: If already in a snapshot context.
    """
    if self.memory is not None:
      raise RuntimeError('Already in a snapshot context.')
    self.memory = copy.deepcopy(self.last_snapshot.memory)
    try:
      yield  # yield control to the `with` block
    finally:
      snapshot_step = self.last_snapshot.system_status.status_step + 1
      self._record_memory_snapshot(
          self.memory,
          snapshot_step=snapshot_step,
          token_counter=token_counter,
      )
      self.memory = None  # pyrefly: ignore[bad-assignment]

  def get_events_for_step(self, step: int) -> list[MemoryFile]:
    """Returns all log events belonging to a particular step.

    Looks up the snapshot at the given step index and filters its log files
    for events whose `event_step` metadata matches the step.

    Args:
      step: The step index to retrieve events for.

    Returns:
      A list of MemoryFiles whose event_step matches the given step.
    """
    snapshot = self.memory_snapshots[step]
    events = []
    for uri in snapshot.memory.log_order:
      file = snapshot.memory.files[uri]
      if file.metadata.get('event_step', -1) == step:
        events.append(file)
    return events

  def record_llm_output(self, content: str):
    """Records the LLM response in the memory."""
    assert self.memory is not None, CONTEXT_ERROR_MSG
    curr_step = self.last_snapshot.system_status.status_step + 1
    uri = f'log://step_{curr_step:06d}_llm_output.md'
    if uri in self.memory.files:
      raise KeyError(
          f'{uri} already exists, record_llm_output should only be called once'
          ' per step.'
      )
    file = MemoryFile(
        uri=uri,
        display=DisplayMode.SUMMARY,
        content=content,
        summary=self.generate_summary(content),
        update_step=curr_step,
        metadata={'event_type': 'llm_output', 'event_step': curr_step},
    )
    self.memory.files[uri] = file
    self.memory.log_order.append(uri)

  def record_tool_call(
      self, tool_name: str, index: int, inputs: str, outputs: str
  ):
    """Records the tool call in the memory.

    Args:
      tool_name: The name of the tool.
      index: The index of the tool call in the step, to make unique filename.
      inputs: The inputs of the tool call.
      outputs: The outputs of the tool call.

    Raises:
      KeyError: If the file already exists.
    """
    assert self.memory is not None, CONTEXT_ERROR_MSG
    curr_step = self.last_snapshot.system_status.status_step + 1
    uri = f'log://step_{curr_step:06d}_tool_call_{index}.md'
    if uri in self.memory.files:
      raise KeyError(f'{uri} already exists.')
    metadata = {
        'event_type': 'tool_call',
        'event_step': curr_step,
        'tool_name': tool_name,
    }
    content = '\n\n'.join([
        f'## Tool Call of {tool_name}',
        f'### Inputs\n\n{inputs}',
        f'### Outputs\n\n{outputs}',
    ])
    summary = '\n'.join([
        tool_name,
        self.generate_summary(inputs),
        self.generate_summary(outputs),
    ])
    file = MemoryFile(
        uri=uri,
        display=DisplayMode.SUMMARY,
        content=content,
        summary=summary,
        update_step=curr_step,
        metadata=metadata,
    )
    self.memory.files[uri] = file
    self.memory.log_order.append(uri)

  # Tool call related methods
  def set_display(self, uri: str, display: DisplayMode) -> str:
    """Sets the display mode of a memory file."""
    assert self.memory is not None, CONTEXT_ERROR_MSG
    if uri.startswith('pad://') and display == DisplayMode.SUMMARY:
      return 'Error: Files in pad:// are always in full display mode.'
    mem_files = self.memory.files
    if uri not in mem_files:
      return 'Error: File not found.'
    mem_files[uri] = dataclasses.replace(mem_files[uri], display=display)
    return 'OK'

  def write(self, uri: str, content: str, summary: str | None = None) -> str:
    """Writes a memory file."""
    assert self.memory is not None, CONTEXT_ERROR_MSG
    ret = re.fullmatch(REGEX_MEMORY_URI, uri)
    if not ret:
      return f'Error: Invalid URI: {uri}, must match {REGEX_MEMORY_URI.pattern}'
    folder = ret.group(1)
    if folder == 'log':
      return (
          'Error: Cannot directly write to log:// files. Use'
          ' `compress_context` if you want to compress some steps into a'
          ' summary step.'
      )
    if folder not in ('kb', 'pad'):
      return 'Error: Can only write to kb:// or pad:// files.'
    curr_step = self.last_snapshot.system_status.status_step + 1
    file = MemoryFile(
        uri=uri,
        display=DisplayMode.SUMMARY,
        content=content,
        update_step=curr_step,
        summary=summary or self.generate_summary(content),
    )
    self.memory.files[uri] = file
    return 'OK'

  def delete(self, uri: str) -> str:
    """Deletes a memory file."""
    assert self.memory is not None, CONTEXT_ERROR_MSG
    mem_files = self.memory.files
    if uri not in mem_files:
      return 'Error: File not found.'
    if not (uri.startswith('kb://') or uri.startswith('pad://')):
      return 'Error: Only files in kb:// and pad:// can be deleted.'
    del mem_files[uri]
    return 'OK'

  def compress_history(
      self,
      start_step: int,
      end_step: int,
      content: str,
      summary: str | None = None,
  ) -> str:
    """Replace the history from start_step to end_step with a summary step."""
    assert self.memory is not None, CONTEXT_ERROR_MSG
    if start_step > end_step:
      return 'Error: start_step must be less than or equal to end_step.'
    mem_files = self.memory.files
    log_order = self.memory.log_order
    def replacable(file: MemoryFile) -> bool:
      return (
          file.metadata['event_step'] >= start_step
          and file.metadata['event_step'] <= end_step
      )
    idx_replaced = [
        idx for idx, uri in enumerate(log_order) if replacable(mem_files[uri])
    ]
    if not idx_replaced:
      return (
          f'Error: No history log found between step {start_step} and'
          f' {end_step}.'
      )
    uri_replaced = [log_order[idx] for idx in idx_replaced]
    summary_step_idx = idx_replaced[-1]
    summary_step = mem_files[log_order[summary_step_idx]].metadata['event_step']
    curr_step = self.last_snapshot.system_status.status_step + 1
    summary_file = MemoryFile(
        uri=f'log://step_{summary_step:06d}_summary.md',
        display=DisplayMode.SUMMARY,
        content=content,
        summary=summary or self.generate_summary(content),
        update_step=curr_step,
        metadata={
            'event_type': 'summary',
            'event_step': summary_step,
            'replaced_uris': uri_replaced,
        },
    )
    log_order[summary_step_idx] = summary_file.uri
    for idx in reversed(idx_replaced[:-1]):
      del log_order[idx]
    for uri in uri_replaced:
      del mem_files[uri]
    # Note: put this *after* del mem_files[uri], when compressing history
    # twice with the same end_step, the URI of the summary file will be
    # the same, this order avoid deleting the newly generated summary file.
    mem_files[summary_file.uri] = summary_file
    return 'OK'

  def record_progress(
      self,
      description: str,
      metrics: list[MetricEntry],
  ) -> None:
    """Records a progress entry.

    Args:
      description: A text description of the progress.
      metrics: A list of MetricEntry, each with variant, metric, and value.
    """
    assert self.memory is not None, CONTEXT_ERROR_MSG
    curr_step = self.last_snapshot.system_status.status_step + 1
    entry = ProgressEntry(
        step=curr_step,
        description=description,
        metrics=metrics,
    )
    self.progress_log.append(entry)


################################################################################
# Memory Tools
################################################################################
class MemoryDisplayAction(tools_lib.Action):
  """An action to fold a memory file."""
  uri: Annotated[str, pydantic.Field(description='The URI of the memory file.')]


def memory_fold(action: MemoryDisplayAction, mem_system: MemorySystem) -> str:
  return mem_system.set_display(action.uri, DisplayMode.SUMMARY)


def memory_unfold(action: MemoryDisplayAction, mem_system: MemorySystem) -> str:
  return mem_system.set_display(action.uri, DisplayMode.FULL)


class MemoryWriteAction(tools_lib.Action):
  """An action to write a memory file."""
  uri: Annotated[
      str,
      pydantic.Field(
          description=(
              'The URI of the memory file (e.g. kb://notes.md or'
              ' pad://TODO.md).'
          )
      ),
  ]
  content: Annotated[
      str, pydantic.Field(description='The content to write.')
  ]
  summary: Annotated[
      str | None,
      pydantic.Field(
          description=(
              'The summary of the content. If not provided, the system will' +
              ' generate one by taking a short prefix of the content.'
          )
      )
  ] = None


def memory_write(action: MemoryWriteAction, mem_system: MemorySystem) -> str:
  return mem_system.write(action.uri, action.content, action.summary)


class MemoryDeleteAction(tools_lib.Action):
  """An action to delete a memory file."""
  uri: Annotated[
      str, pydantic.Field(description='The URI of the memory file to delete.')
  ]


def memory_delete(
    action: MemoryDeleteAction, mem_system: MemorySystem
) -> str:
  return mem_system.delete(action.uri)


class MemoryCompressHistoryAction(tools_lib.Action):
  """An action to compress the history of memory."""
  start_step: Annotated[
      int,
      pydantic.Field(description='The start step of the history to compress.'),
  ]
  end_step: Annotated[
      int,
      pydantic.Field(description='The end step of the history to compress.'),
  ]
  content: Annotated[
      str, pydantic.Field(description='The content of the summary step.')
  ]
  summary: Annotated[
      str | None,
      pydantic.Field(
          description=(
              'The summary of the content. If not provided, the system will' +
              ' generate one by taking a short prefix of the content.'
          )
      )
  ] = None


def memory_compress_history(
    action: MemoryCompressHistoryAction, mem_system: MemorySystem
) -> str:
  return mem_system.compress_history(
      action.start_step, action.end_step, action.content, action.summary
  )


class RecordProgressAction(tools_lib.Action):
  """An action to record a progress entry with metrics for the current step."""

  description: Annotated[
      str,
      pydantic.Field(
          description=(
              'Description of the progress, include important pointers like'
              ' code commit, experiment ID, etc.'
          )
      ),
  ]
  metrics: Annotated[
      list[MetricEntry],
      pydantic.Field(
          description=(
              'A list of metric entries.'
          )
      ),
  ]


def record_progress(
    action: RecordProgressAction, mem_system: MemorySystem
) -> str:
  mem_system.record_progress(action.description, list(action.metrics))
  return 'OK'


def get_memory_tools(memory_system: MemorySystem) -> list[tools_lib.Tool]:
  """Returns the memory tools."""
  fold = tools_lib.Tool(
      name='mem_fold',
      description='Set the display mode of some memory files to `summary`.',
      action_type=MemoryDisplayAction,
      executor=functools.partial(memory_fold, mem_system=memory_system),
  )
  unfold = tools_lib.Tool(
      name='mem_unfold',
      description='Set the display mode of some memory files to `full`.',
      action_type=MemoryDisplayAction,
      executor=functools.partial(memory_unfold, mem_system=memory_system),
  )
  write = tools_lib.Tool(
      name='mem_write',
      description='Write a memory file.',
      action_type=MemoryWriteAction,
      executor=functools.partial(memory_write, mem_system=memory_system),
  )
  delete = tools_lib.Tool(
      name='mem_delete',
      description='Delete a memory file.',
      action_type=MemoryDeleteAction,
      executor=functools.partial(memory_delete, mem_system=memory_system),
  )
  compress_history = tools_lib.Tool(
      name='mem_compress_history',
      description='Compress the history of memory.',
      action_type=MemoryCompressHistoryAction,
      executor=functools.partial(
          memory_compress_history, mem_system=memory_system
      ),
  )
  progress = tools_lib.Tool(
      name='record_progress',
      description=(
          'Record progress for one research iteration in multi-iteration '
          'research runs.'
      ),
      action_type=RecordProgressAction,
      executor=functools.partial(record_progress, mem_system=memory_system),
  )
  return [
      fold,
      unfold,
      write,
      delete,
      compress_history,
      progress,
  ]

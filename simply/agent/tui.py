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
"""Simple TUI for displaying agent status.

This module provides two display implementations:

- `StatusDisplay`: A compact, live-updating panel using `rich.live.Live` that
  redraws in-place (like htop). It shows header info, system status, a
  truncated task view, and a status bar.

- `PrintDisplay`: A minimal fallback that logs events as plain one-liners.
  Used as the default so the agent never needs to guard TUI calls.
"""

import abc
import datetime
import shutil

import rich.console
import rich.layout
import rich.live
import rich.markup
import rich.panel
import rich.rule
import rich.table
import rich.text

from simply.agent import memory as memory_lib


class DisplayBase(abc.ABC):
  """Abstract base class for agent display."""

  @abc.abstractmethod
  def start(self) -> None:
    """Start the display."""

  @abc.abstractmethod
  def stop(self) -> None:
    """Stop the display."""

  @abc.abstractmethod
  def set_header(self, info: dict[str, str]) -> None:
    """Set the static header info (LLM, session dir, run ID, workspace)."""

  @abc.abstractmethod
  def update_system_status(self, status: memory_lib.SystemStatus) -> None:
    """Update the system status row (step, tokens, elapsed)."""

  @abc.abstractmethod
  def set_task(self, task: str) -> None:
    """Set the task content to display."""

  @abc.abstractmethod
  def update_status(self, message: str) -> None:
    """Update the bottom status bar message."""

  @abc.abstractmethod
  def display_llm_output(self, output: str) -> None:
    """Display the LLM output."""

  @abc.abstractmethod
  def display_tool_call(self, name: str, tool_inputs: str, result: str) -> None:
    """Display the tool call inputs and results."""

  def print_header(self) -> None:
    """Print the header info to stdout (for use after the display stops)."""


class PrintDisplay(DisplayBase):
  """Minimal fallback display that outputs plain one-liners."""

  def __init__(self) -> None:
    self._header: dict[str, str] = {}

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def set_header(self, info: dict[str, str]) -> None:
    self._header = dict(info)
    for key, value in info.items():
      print(f'  {key}: {value}')

  def update_system_status(self, status: memory_lib.SystemStatus) -> None:
    elapsed = _format_elapsed(status.elapsed_seconds)
    tokens = _format_tokens(
        status.approximate_token_usage, status.max_token_budget
    )
    print(
        f'  [Step {status.status_step}] Tokens: {tokens} | Elapsed: {elapsed}'
    )

  def set_task(self, task: str) -> None:
    pass  # Task is not printed in the minimal display.

  def update_status(self, message: str) -> None:
    print(f'  {message}')

  def display_llm_output(self, output: str) -> None:
    pass

  def display_tool_call(self, name: str, tool_inputs: str, result: str) -> None:
    pass

  def print_header(self) -> None:
    for key, value in self._header.items():
      print(f'  {key}: {value}')


class FullDisplay(DisplayBase):
  """Display that outputs full agent output including LLM text and tool calls."""

  def __init__(self) -> None:
    self._header: dict[str, str] = {}
    self.console = rich.console.Console()

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def set_header(self, info: dict[str, str]) -> None:
    self._header = dict(info)
    self.console.print('[bold]Session Info:[/bold]')
    for key, value in info.items():
      self.console.print(f'  [bold]{key}:[/bold] {value}')

  def update_system_status(self, status: memory_lib.SystemStatus) -> None:
    elapsed = _format_elapsed(status.elapsed_seconds)
    tokens = _format_tokens(
        status.approximate_token_usage, status.max_token_budget
    )
    self.console.print(
        f'[bold][Step {status.status_step}][/bold] Tokens: {tokens} | Elapsed:'
        f' {elapsed}'
    )

  def set_task(self, task: str) -> None:
    self.console.print(
        rich.panel.Panel(task, title='[bold]Task[/bold]', border_style='green')
    )

  def update_status(self, message: str) -> None:
    self.console.print(f'  [dim]{message}[/dim]')

  def display_llm_output(self, output: str) -> None:
    self.console.print(
        rich.panel.Panel(
            output, title='[bold]LLM Output[/bold]', border_style='yellow'
        )
    )

  def display_tool_call(self, name: str, tool_inputs: str, result: str) -> None:
    renderable = rich.panel.Panel(
        rich.console.Group(
            rich.markup.escape(tool_inputs),
            rich.rule.Rule(style='grey70'),
            rich.markup.escape(result),
        ),
        title=f'Tool Call: {name}',
        border_style='cyan',
    )
    self.console.print(renderable)


class StatusDisplay(DisplayBase):
  """A compact, live-updating status panel using rich.live.Live."""

  def __init__(self) -> None:
    self._header: dict[str, str] = {}
    self._status_step: int = -1
    self._token_usage: str = '—'
    self._elapsed: str = '—'
    self._task: str = ''
    self._status_message: str = ''
    self._live: rich.live.Live | None = None

  def start(self) -> None:
    self._live = rich.live.Live(
        self._render(),
        auto_refresh=False,
        console=rich.console.Console(),
    )
    self._live.start()

  def stop(self) -> None:
    if self._live is not None:
      self._live.stop()
      self._live = None

  def set_header(self, info: dict[str, str]) -> None:
    self._header = dict(info)
    self._refresh()

  def update_system_status(self, status: memory_lib.SystemStatus) -> None:
    self._status_step = status.status_step
    self._token_usage = _format_tokens(
        status.approximate_token_usage, status.max_token_budget
    )
    self._elapsed = _format_elapsed(status.elapsed_seconds)
    self._refresh()

  def set_task(self, task: str) -> None:
    self._task = task
    self._refresh()

  def update_status(self, message: str) -> None:
    self._status_message = message
    self._refresh()

  def display_llm_output(self, output: str) -> None:
    pass

  def display_tool_call(self, name: str, tool_inputs: str, result: str) -> None:
    pass

  def print_header(self) -> None:
    for key, value in self._header.items():
      print(f'  {key}: {value}')

  # --- Internal rendering ---------------------------------------------------

  def _refresh(self) -> None:
    if self._live is not None:
      self._live.update(self._render(), refresh=True)

  def _render(self) -> rich.console.RenderableType:
    """Build the full status panel."""
    # Section 1: Header
    header_lines: list[str] = []
    for key, value in self._header.items():
      header_lines.append(f'[bold]{key}:[/bold] {rich.markup.escape(value)}')
    header_text = '\n'.join(header_lines) if header_lines else '(no info)'

    # Section 2: System status row
    step = '-' if self._status_step < 0 else str(self._status_step)
    sys_row = (
        f'[bold]Step:[/bold] {step}  │  '
        f'[bold]Tokens:[/bold] {self._token_usage}  │  '
        f'[bold]Elapsed:[/bold] {self._elapsed}'
    )

    # Section 3: Task (truncated to fit)
    task_text = self._truncated_task()

    # Section 4: Status bar
    status_bar = (
        rich.markup.escape(self._status_message)
        if self._status_message
        else '…'
    )

    # Assemble

    table = rich.table.Table.grid(padding=(0, 0), expand=True)
    table.add_column()

    table.add_row(rich.text.Text.from_markup(header_text))
    table.add_row(rich.text.Text.from_markup(sys_row))
    table.add_row(rich.rule.Rule())
    if task_text:
      table.add_row(rich.text.Text.from_markup(task_text))
      table.add_row(rich.rule.Rule())
    table.add_row(rich.text.Text.from_markup(status_bar))

    return rich.panel.Panel(
        table,
        title='[bold blue]Simply Agent[/bold blue]',
        border_style='blue',
    )

  def _truncated_task(self) -> str:
    """Return the task text, truncated to fit available terminal height."""
    if not self._task:
      return ''
    # Reserve lines for other sections:
    # panel border (2) + header (~4) + separators (3) + sys row (1)
    # + status bar (1) + some padding (2) = ~13 lines overhead
    overhead = 13 + len(self._header)
    term_height = shutil.get_terminal_size().lines
    max_task_lines = max(term_height - overhead, 2)

    lines = self._task.splitlines()
    if len(lines) <= max_task_lines:
      return f'[bold]Task:[/bold]\n{rich.markup.escape(self._task)}'

    truncated = '\n'.join(lines[:max_task_lines])
    return (
        f'[bold]Task:[/bold]\n{rich.markup.escape(truncated)}\n'
        '[dim](truncated)[/dim]'
    )


# --- Helpers -----------------------------------------------------------------


def _format_elapsed(seconds: float) -> str:
  """Format elapsed seconds as a human-readable string."""
  td = datetime.timedelta(seconds=int(seconds))
  parts = []
  total_seconds = int(td.total_seconds())
  hours, remainder = divmod(total_seconds, 3600)
  minutes, secs = divmod(remainder, 60)
  if hours:
    parts.append(f'{hours}h')
  if minutes or hours:
    parts.append(f'{minutes}m')
  parts.append(f'{secs}s')
  return ' '.join(parts)


def _format_tokens(usage: int, budget: int) -> str:
  """Format token usage as e.g. '12.3k / 200k'."""

  def _fmt(n: int) -> str:
    if n >= 1000:
      return f'{n / 1000:.1f}k'
    return str(n)

  return f'{_fmt(usage)} / {_fmt(budget)}'

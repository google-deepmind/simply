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
"""GLM-5.2 chat/tool-call template (`GlmChat`), kept out of core lm_format.

Registered via `LMFormatRegistry`, so the core format loader picks it up by
name with no GLM reference in core.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import re
from typing import Any

from simply.utils import lm_format

LMFormat = lm_format.LMFormat
LMFormatRegistry = lm_format.LMFormatRegistry


@LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class GlmChat(LMFormat):
  """LM format for GLM-5 / GLM-5.1 / GLM-5.2 (`glm_moe_dsa`).

  Mirrors the GLM-5.2 HuggingFace chat template: a ``[gMASK]<sop>`` document
  prefix, role-tagged turns with ``<|system|>`` / ``<|user|>`` /
  ``<|assistant|>``
  / ``<|observation|>`` (tool results) markers, and an assistant generation
  prompt that opens a ``<think>`` block (reasoning enabled by default). The
  model
  emits ``</think>`` to end reasoning and one of the EOS tokens to end the turn.
  """

  prefix: str = '[gMASK]<sop>'
  system_marker: str = '<|system|>'
  user_marker: str = '<|user|>'
  assistant_marker: str = '<|assistant|>'
  observation_marker: str = '<|observation|>'
  # GLM-5.2 EOS tokens: <|user|>, <|observation|>, <|endoftext|> end a turn.
  extra_eos_tokens: tuple[str, ...] = (
      '<|user|>',
      '<|observation|>',
      '<|endoftext|>',
  )
  add_think_marker: bool = True
  begin_of_thought_marker: str = '<think>'
  end_of_thought_marker: str = '</think>'
  # GLM-5.2 reasoning-effort system directive. The official chat template
  # prepends `<|system|>Reasoning Effort: Max` (capitalized) right after
  # `[gMASK]<sop>` when thinking is enabled; 'max' is the default and Z.ai
  # recommends it for coding/agentic tasks. Empty disables the directive.
  reasoning_effort: str = 'max'

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    output = self.prefix
    # Reasoning-effort system directive (GLM official template line 2-3).
    if self.add_think_marker and self.reasoning_effort:
      output += self.system_marker + (
          f'Reasoning Effort: {self.reasoning_effort.capitalize()}'
      )
    # Index of the last user turn: assistant reasoning at/before it is collapsed
    # to `<think></think>` (history), only later assistant turns keep their full
    # `<think>...</think>` -- mirrors the template's `last_user_index` logic to
    # avoid context bloat/drift over long agentic loops.
    last_user_index = -1
    for i, m in enumerate(messages):
      if m.get('role') == 'user':
        last_user_index = i
    for idx, message in enumerate(messages):
      role = message['role']
      # `developer` is an OpenAI-style high-priority system role used by some
      # agent harnesses (incl. SWE-bench Pro); GLM has no separate developer
      # turn, so fold it into the system turn. `tool`/`function`/`ipython`
      # carry tool results, which GLM renders as `<|observation|>`.
      if role in ('system', 'developer'):
        marker = self.system_marker
      elif role == 'user':
        marker = self.user_marker
      elif role == 'assistant':
        marker = self.assistant_marker
      elif role in ('observation', 'tool', 'function', 'ipython'):
        marker = self.observation_marker
      else:
        raise ValueError(f'Unknown role: {role}')
      # Tool/function declarations render into GLM's native <tools> block.
      tools_block = self._render_tools(message['content'])
      if tools_block:
        output += self.system_marker + tools_block
      # An assistant message may carry both tool calls and their results; tool
      # calls go under `<|assistant|>`, tool results under `<|observation|>`.
      if role == 'assistant':
        # Collapse historical reasoning (turns at/before the last user turn) to
        # `<think></think>`; keep full reasoning only for turns after it.
        collapse_think = idx <= last_user_index
        text = self._render_assistant_text(
            message['content'], collapse_think=collapse_think
        )
        if text:
          output += marker + text
        obs = self._render_function_responses(message['content'])
        if obs:
          output += self.observation_marker + obs
      else:
        text = self._visible_text(message['content'])
        if text:
          output += marker + text
    output += self.assistant_marker
    if self.add_think_marker:
      output += self.begin_of_thought_marker
    return output

  def _render_assistant_text(self, content: Any, collapse_think: bool) -> str:
    """Renders an assistant turn's text + tool calls (no tool responses)."""
    # ``collapse_think`` collapses historical ``<think>...</think>`` reasoning
    # (at or before the last user turn) to ``<think></think>`` per the GLM-5.2
    # template; later turns keep their full reasoning.
    text = self._visible_text(content, include_responses=False)
    if collapse_think and self.begin_of_thought_marker in text:
      bot, eot = self.begin_of_thought_marker, self.end_of_thought_marker
      if eot in text:
        # Drop the reasoning between the first <think> and its </think>.
        before, _, rest = text.partition(bot)
        _, _, after = rest.partition(eot)
        text = before + bot + eot + after
      else:
        # Unterminated <think>: collapse everything from <think> onward.
        before, _, _ = text.partition(bot)
        text = before + bot + eot
    return text

  def _render_tools(self, content: Any) -> str:
    """Renders ``function_declaration`` parts into a GLM ``<tools>`` block."""
    # Returns the body (after a ``<|system|>`` marker), or '' if there are no
    # tool declarations. Each declaration is emitted as JSON inside
    # ``<tools>...</tools>``, followed by the GLM tool-call format instruction.
    if not isinstance(content, Sequence) or isinstance(content, str):
      return ''
    decls = []
    for item in content:
      if isinstance(item, Mapping) and 'function_declaration' in item:
        decls.append(item['function_declaration'])
    if not decls:
      return ''
    lines = [
        '# Tools',
        '',
        'You may call one or more functions to assist with the user query.',
        '',
        (
            'You are provided with function signatures within <tools></tools>'
            ' XML tags:'
        ),
        '<tools>',
    ]
    for decl in decls:
      lines.append(json.dumps(decl, ensure_ascii=False))
    lines.append('</tools>')
    lines.append('')
    lines.append(
        'For each function call, output the function name and arguments within'
        ' the following XML format:'
    )
    lines.append(
        '<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>'
        '{arg-value-1}</arg_value>...</tool_call>'
    )
    return '\n'.join(lines)

  def _visible_text(self, content: Any, include_responses: bool = True) -> str:
    """Renders message content to text (GLM template ``visible_text``)."""
    # Content is a plain string or an OpenAI-style list of parts.
    # ``function_call`` parts are re-rendered to GLM ``<tool_call>`` text (so
    # replayed history matches the live model's output). ``function_response``
    # (tool result) parts belong under a separate ``<|observation|>`` turn, so
    # ``include_responses=False`` skips them when rendering an assistant turn.
    if isinstance(content, str):
      return content
    if isinstance(content, Mapping):
      if 'function_call' in content:
        return self._render_function_call(content['function_call'])
      if 'function_response' in content:
        return (
            self._render_function_response(content['function_response'])
            if include_responses
            else ''
        )
      return str(content.get('text', ''))
    if isinstance(content, Sequence):
      parts = []
      for item in content:
        if isinstance(item, str):
          parts.append(item)
        elif isinstance(item, Mapping):
          if 'function_call' in item:
            parts.append(self._render_function_call(item['function_call']))
          elif 'function_response' in item:
            if include_responses:
              parts.append(
                  self._render_function_response(item['function_response'])
              )
          else:
            parts.append(str(item.get('text', '')))
      return ''.join(parts)
    return str(content)

  def _render_function_responses(self, content: Any) -> str:
    """Renders a message's ``function_response`` parts to observation text."""
    # Concatenates each rendered tool result in ``content`` ('' if none); the
    # caller (``format``) prefixes the GLM ``<|observation|>`` marker.
    if not isinstance(content, Sequence) or isinstance(content, str):
      if isinstance(content, Mapping) and 'function_response' in content:
        return self._render_function_response(content['function_response'])
      return ''
    parts = []
    for item in content:
      if isinstance(item, Mapping) and 'function_response' in item:
        parts.append(self._render_function_response(item['function_response']))
    return ''.join(parts)

  def _render_function_response(self, fn_resp: Any) -> str:
    r"""Renders a ``{'name', 'response'}`` tool result to observation text."""
    # Wraps the response payload (e.g. ``{'exit_code': '0', ...}``) in a
    # ``<tool_response>`` block keyed by the tool name, one field per line.
    if not isinstance(fn_resp, Mapping):
      return str(fn_resp)
    name = str(fn_resp.get('name', '') or '')
    response = fn_resp.get('response', {})
    lines = []
    if isinstance(response, Mapping):
      for key, val in response.items():
        if isinstance(val, str):
          val_str = val
        else:
          try:
            val_str = json.dumps(val, ensure_ascii=False)
          except (TypeError, ValueError):
            val_str = str(val)
        lines.append(f'{key}: {val_str}')
      body = '\n'.join(lines)
    else:
      body = str(response)
    header = f'<tool_response>{name}\n' if name else '<tool_response>\n'
    return f'{header}{body}\n</tool_response>'

  def _render_function_call(self, fn_call: Any) -> str:
    r"""Renders a ``{'name', 'args'}`` function call to ``<tool_call>`` text."""
    # Inverse of ``_parse_tool_call``: emits
    # ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
    # </tool_call>``. Non-string arg values are JSON-serialized.
    if not isinstance(fn_call, Mapping):
      return ''
    name = str(fn_call.get('name', '') or '')
    args = fn_call.get('args', {}) or {}
    body = [name]
    if isinstance(args, Mapping):
      for key, val in args.items():
        if isinstance(val, str):
          val_str = val
        else:
          try:
            val_str = json.dumps(val, ensure_ascii=False)
          except (TypeError, ValueError):
            val_str = str(val)
        body.append(f'<arg_key>{key}</arg_key><arg_value>{val_str}</arg_value>')
    return f'<tool_call>{"".join(body)}</tool_call>'

  def parse(
      self,
      formatted: str,
      function_declarations: Any = None,
  ) -> Sequence[Mapping[str, Any]]:
    r"""Parses a GLM assistant turn into structured ``output_messages``."""
    # The serving stack calls ``lm_format.parse`` on the model's output to build
    # the ``output_messages`` the agentic eval loop consumes. Within the
    # assistant turn GLM emits optional ``<think>...</think>`` and zero
    # or more tool calls
    # ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
    # </tool_call>``. Returns one assistant message whose content is a list of
    # parts: a ``{'text': ...}`` part for the visible text and one
    # ``{'function_call': {'name', 'args'}}`` part per tool call.
    del function_declarations  # GLM args are strings; no schema coercion here.
    text = formatted
    # Drop the assistant marker prefix if present.
    if text.startswith(self.assistant_marker):
      text = text[len(self.assistant_marker) :]
    # Strip reasoning: keep only content after the last </think> if present.
    if '</think>' in text:
      text = text.split('</think>')[-1]
    else:
      # Remove an unterminated leading <think> block.
      text = re.sub(r'^\s*<think>.*', '', text, flags=re.DOTALL)
    # Trim trailing turn markers / EOS.
    for marker in (self.user_marker, self.observation_marker, '<|endoftext|>'):
      idx = text.find(marker)
      if idx != -1:
        text = text[:idx]

    content: list[dict[str, Any]] = []
    tool_call_re = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    last = 0
    for m in tool_call_re.finditer(text):
      pre = text[last : m.start()].strip()
      if pre:
        content.append({'text': pre})
      content.append({'function_call': self._parse_tool_call(m.group(1))})
      last = m.end()
    tail = text[last:].strip()
    if tail:
      content.append({'text': tail})
    if not content:
      content.append({'text': ''})
    return [{'role': 'assistant', 'content': content}]

  def _parse_tool_call(self, body: str) -> Mapping[str, Any]:
    r"""Parses a single GLM ``<tool_call>`` body into ``{'name', 'args'}``."""
    # Body format: ``NAME<arg_key>K1</arg_key><arg_value>V1</arg_value>...``.
    # Arg values are kept as strings (GLM emits them verbatim); JSON-looking
    # values are parsed so structured args round-trip.
    name_m = re.match(r'\s*([^<]+)', body)
    name = name_m.group(1).strip() if name_m else ''
    args: dict[str, Any] = {}
    pair_re = re.compile(
        r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', re.DOTALL
    )
    for km, vm in ((m.group(1), m.group(2)) for m in pair_re.finditer(body)):
      key = km.strip()
      val: Any = vm.strip()
      try:
        val = json.loads(val)
      except (ValueError, TypeError):
        pass
      args[key] = val
    return {'name': name, 'args': args}

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
r"""Qwen3.8's released chat template as a Simply `LMFormat`, and its inverse.

`Qwen38Chat.format()` is a line-by-line port of the released
`chat_template.jinja` (checked in at `../testdata/chat_template.jinja`, which
`lm_format_test.py` renders with jinja2 and diffs against this code character
for character -- that render is the oracle, so the template is the spec and
this module is the implementation). The scope is the template's TEXT subset:
no vision parts, and `add_generation_prompt` is always true, because Simply
formats prompts to continue. Inside that scope every parameter the template
has is a field here (an unsettable parameter would be an unimplemented
branch), and every quirk is reproduced:

  * thinking is ON unless `enable_thinking=False`, and the generation prompt
    ends with `<think>\n` (thinking off => a pre-closed `<think>\n\n</think>`);
  * `reasoning_effort` injects one sentence at the top of the system turn
    (`xhigh` default, `medium` = nothing, `low`), creating a system turn if the
    caller supplied none;
  * historical assistant turns keep their `<think>` block iff
    `preserve_thinking` or the turn is after the last real user query;
  * tool declarations force a `# Tools` system block and tool CALLS are XML
    (`<tool_call><function=..><parameter=..>`), not JSON;
  * tool RESULTS are folded into a `user` turn as `<tool_response>` blocks.

`parse()` is the inverse an agentic harness needs, shaped as core's serving
hook: `serving/page_batcher.py` finds it with `getattr(self.lm_format,
'parse', None)`, calls it with `assistant_marker + output_text` and stores the
result as `output_messages` -- the only thing the DeepSWE in-sandbox agent
loop reads. It turns the continuation back into FC2.0 parts
(`{'function_call': {'name', 'args'}}` / `{'text'}` /
`{'text', 'channel': 'THOUGHT'}`) and tolerates truncation anywhere in the
stream, because an empty turn is what that loop reads as "the model is done".

Seven names are registered, because the launcher selects a format by string
(`--lm_format=`) and none of these knobs is otherwise reachable: `Qwen38Chat`
(thinking, `reasoning_effort=xhigh`), `Qwen38ChatMedium` / `Qwen38ChatLow`
(the other two efforts the template accepts), `Qwen38ChatNoThink`
(`enable_thinking=False`), and `Qwen38ChatFC` / `Qwen38ChatFCMedium` /
`Qwen38ChatFCLow`, which name the FC2.0 agentic use of the same three efforts.

`format()` accepts two message shapes and normalizes them to one:
  * HuggingFace-style: `content` is a string, thinking in `reasoning_content`,
    calls in `tool_calls`, results in `tool`-role messages;
  * FC2.0-style (what the agent loop produces): `content` is a list of parts --
    `{'text'}`, `{'function_declaration'}` (usually on a `developer` message),
    `{'function_call'}`, `{'function_response'}`.
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
import re
from typing import Any, cast

from simply.utils import lm_format as lm_format_lib

_IM_START = '<|im_start|>'
_IM_END = '<|im_end|>'
_THOUGHT_CHANNEL = 'THOUGHT'

# Verbatim from chat_template.jinja.
_REASONING_INSTRUCTIONS = {
    'xhigh': (
        'Reasoning effort is set to xhigh. Please think carefully through the'
        ' task, validate key assumptions, consider plausible alternatives, and'
        ' prioritize correctness, consistency, and clarity in the final answer.'
    ),
    'medium': '',
    'low': (
        'Reasoning effort is set to low. Keep your thinking brief and focused,'
        ' moving directly to the conclusion without unnecessary elaboration.'
    ),
}

_TOOLS_HEADER = (
    '# Tools\n\nYou have access to the following functions:\n\n<tools>'
)

_TOOLS_INSTRUCTIONS = (
    '\n\nIf you choose to call a function ONLY reply in the following format'
    ' with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n'
    '<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
    '<parameter=example_parameter_2>\nThis is the value for the second'
    ' parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n'
    '</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the'
    ' specified format: an inner <function=...></function> block must be nested'
    ' within <tool_call></tool_call> XML tags\n- Required parameters MUST be'
    ' specified\n- You may provide optional reasoning for your function call in'
    ' natural language BEFORE the function call, but NOT after\n- If there is'
    ' no function call available, answer the question like normal with your'
    ' current'
    ' knowledge and do not tell the user about function calls\n</IMPORTANT>'
)

_TOOL_CALL_OPEN = '<tool_call>'
_TOOL_CALL_CLOSE = '</tool_call>'
_FUNCTION_CLOSE = '</function>'
_FUNCTION_RE = re.compile(r'<function=(.*?)>\n?', re.DOTALL)
# What can follow a parameter value: the next parameter, or the end of the
# function / of the call. Matched as one alternation so the scan always takes
# the EARLIEST of them and can never run past the end of its own block.
_MARKUP_RE = re.compile(
    rf'<parameter=(?P<name>.*?)>\n?|(?P<close>{_FUNCTION_CLOSE}|{_TOOL_CALL_CLOSE})'
)
# A parameter value ends at the first `</parameter>` that is FOLLOWED by one of
# those. Values written by a coding agent routinely contain the literal tag
# (`grep -n '</parameter>' x.py`, or a file that documents this very format),
# and a plain non-greedy match would truncate the command mid-flight while
# still leaving it executable. The one case no schema-free parser can resolve
# is a value containing `</parameter>` immediately followed by `<parameter=` or
# `</function>`: that is indistinguishable from the real end of the value, and
# the released reference parser splits it the same way.
_PARAMETER_END_RE = re.compile(
    rf'\n?</parameter>\s*(?=<parameter=|{_FUNCTION_CLOSE}|{_TOOL_CALL_CLOSE}|$)',
    re.DOTALL,
)
# https://json-schema.org/understanding-json-schema/reference/type
_SCHEMA_TYPES = frozenset(
    ['string', 'number', 'integer', 'object', 'array', 'boolean', 'null']
)
_TOOL_RESPONSE_STDERR_MARKER = '[stderr]'
_TOOL_RESPONSE_EMPTY = '(no output)'


def _tojson(value: Any) -> str:
  """HuggingFace's `tojson` jinja filter."""
  return json.dumps(value, ensure_ascii=False)


def _lower_schema_types(schema: Any) -> Any:
  """FC2.0 declares JSON-schema types in upper case; Qwen's tools are lower."""
  if isinstance(schema, Mapping):
    out = {}
    for k, v in schema.items():
      # Only a declared TYPE, never a `default`/`example` that happens to sit
      # under a `type` key: lowercasing those corrupts the value the model is
      # shown.
      if k == 'type' and isinstance(v, str) and v.lower() in _SCHEMA_TYPES:
        out[k] = v.lower()
      else:
        out[k] = _lower_schema_types(v)
    return out
  if isinstance(schema, (list, tuple)):
    return [_lower_schema_types(v) for v in schema]
  return schema


def _tool_json(decl: Mapping[str, Any]) -> str:
  """Serializes one tool declaration the way the chat template's `tojson` does.

  Args:
    decl: either an OpenAI-style `{'type': 'function', 'function': {...}}`,
      which keeps its own envelope, or a bare FunctionDeclaration `{'name',
      'description', 'parameters'}` (FC2.0 or JSON-schema flavored), which is
      wrapped in one. Either way the schema types are lowercased.

  Returns:
    A single-line JSON string.
  """
  function_field = decl.get('function')
  if decl.get('type') == 'function' and isinstance(function_field, Mapping):
    return _tojson(_lower_schema_types(decl))
  function: dict[str, Any] = {'name': decl['name']}
  if decl.get('description'):
    function['description'] = decl['description']
  if decl.get('parameters') is not None:
    function['parameters'] = _lower_schema_types(decl['parameters'])
  # Any other FC2.0 field (notably `response`, which has no place in Qwen's
  # tool JSON) is dropped: the model was trained on name/description/parameters.
  return _tojson({'type': 'function', 'function': function})


@dataclasses.dataclass(frozen=True)
class _Turn:
  """One chat-template message, with content already rendered and trimmed."""

  role: str
  content: str = ''
  reasoning_content: str = ''
  # (name, args) per call, in emission order.
  tool_calls: tuple[tuple[str, Mapping[str, Any]], ...] = ()


def _call_name_and_args(
    call: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
  """Accepts FC2.0 `{'name','args'}` and HF `{'function': {...}}` call dicts."""
  if 'function' in call and isinstance(call['function'], Mapping):
    call = call['function']
  args = call.get('args', call.get('arguments', {})) or {}
  if isinstance(args, str):
    args = json.loads(args) if args else {}
  return call['name'], args


@lm_format_lib.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38Chat(lm_format_lib.LMFormat):
  """Qwen3.8 chat template, thinking mode, `reasoning_effort=xhigh`."""

  system_marker: str = f'{_IM_START}system\n'
  user_marker: str = f'{_IM_START}user\n'
  assistant_marker: str = f'{_IM_START}assistant\n'
  end_of_message_marker: str = f'{_IM_END}\n'
  # `generation_config.json` stops on <|im_end|> and <|endoftext|>.
  extra_eos_tokens: tuple[str, ...] = (_IM_START, _IM_END, '<|endoftext|>')
  begin_of_thought_marker: str = '<think>'
  end_of_thought_marker: str = '</think>'

  enable_thinking: bool = True
  reasoning_effort: str = 'xhigh'
  # A template parameter with no registered variant: reachable by registering
  # a subclass, kept so that both of the template's history branches exist.
  preserve_thinking: bool = True

  def _reasoning_instructions(self) -> str:
    if not self.enable_thinking:
      return ''
    if self.reasoning_effort not in _REASONING_INSTRUCTIONS:
      raise ValueError(
          f'Unexpected reasoning effort {self.reasoning_effort}. Supported'
          ' types are xhigh (default), medium, and low.'
      )
    return _REASONING_INSTRUCTIONS[self.reasoning_effort]

  def _prepare(
      self, messages: Sequence[Mapping[str, Any]]
  ) -> tuple[list[Mapping[str, Any]], list[_Turn], str]:
    """Normalizes both accepted message shapes into tools, turns and system text.

    Args:
      messages: the caller's messages, in either accepted shape.

    Returns:
      `(tools, turns, system_content)`. `turns` has ONE entry per input message
      (system/developer messages become empty `system` placeholders), so turn
      indices match the message indices the chat template's `loop` uses.
    """
    tools: list[Mapping[str, Any]] = []
    turns: list[_Turn] = []
    system_texts: list[str] = []
    system_seen = False

    for message in messages:
      role = message['role']
      content = message.get('content')
      if isinstance(content, Mapping):
        raise ValueError(f'Unexpected content type: {content!r}')
      # `render_content` for the text-only subset of the template: a string is
      # itself, anything else is a part list.
      parts = content if not isinstance(content, str) and content else ()
      thoughts, texts, calls, responses = [], [], [], []
      for part in parts:
        if not isinstance(part, Mapping):
          raise ValueError(f'Unexpected item type in content: {part!r}')
        if 'function_declaration' in part:
          tools.append(part['function_declaration'])
        elif 'function_call' in part:
          calls.append(_call_name_and_args(part['function_call']))
        elif 'function_response' in part:
          responses.append(part['function_response'])
        elif 'text' in part:
          if (part.get('channel') or '').upper() == _THOUGHT_CHANNEL:
            thoughts.append(part['text'])
          else:
            texts.append(part['text'])
        else:
          raise ValueError(f'Unexpected item type in content: {part!r}')
      text = ''.join(texts) if parts else content or ''

      for call in message.get('tool_calls') or ():
        calls.append(_call_name_and_args(call))
      reasoning = message.get('reasoning_content')
      if not isinstance(reasoning, str):
        reasoning = ''.join(thoughts)

      # Qwen has no `developer` role: its tool declarations become the `# Tools`
      # system block and any prose folds into the system turn.
      if role in ('system', 'developer'):
        if role == 'system':
          if system_seen or any(turn.role != 'system' for turn in turns):
            raise ValueError('System message must be at the beginning.')
          system_seen = True
        if text.strip():
          system_texts.append(text.strip())
        # A placeholder keeps turn indices aligned with message indices: the
        # template's `loop.previtem` (which decides whether a `tool` turn opens
        # a `<|im_start|>user`) indexes messages, so dropping one here would
        # emit a `<tool_response>` with no role marker.
        turns.append(_Turn(role='system'))
        continue
      if role not in ('user', 'assistant', 'tool'):
        raise ValueError(f'Unexpected message role: {role}')
      turns.append(
          _Turn(
              role=role,
              content=text.strip(),
              reasoning_content=reasoning.strip(),
              tool_calls=tuple(calls),
          )
      )
      # The agent loop appends tool results to the *assistant* turn that made
      # the calls; the template wants them as following `tool` messages.
      for response in responses:
        turns.append(
            _Turn(
                role='tool',
                content=_response_text(response).strip(),
            )
        )

    return tools, turns, '\n\n'.join(system_texts)

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    if not messages:
      raise ValueError('No messages provided.')
    tools, turns, system_content = self._prepare(messages)
    reasoning_instructions = self._reasoning_instructions()

    out = []
    if tools:
      out.append(self.system_marker)
      if reasoning_instructions:
        out.append(reasoning_instructions + '\n\n')
      out.append(_TOOLS_HEADER)
      for tool in tools:
        out.append('\n' + _tool_json(tool))
      out.append('\n</tools>')
      out.append(_TOOLS_INSTRUCTIONS)
      if system_content:
        out.append('\n\n' + system_content)
      out.append(self.end_of_message_marker)
    elif system_content:
      out.append(self.system_marker)
      if reasoning_instructions:
        out.append(reasoning_instructions + '\n\n')
      out.append(system_content + self.end_of_message_marker)
    elif reasoning_instructions:
      out.append(
          self.system_marker
          + reasoning_instructions
          + self.end_of_message_marker
      )

    last_query_index = _last_query_index(turns)
    for i, turn in enumerate(turns):
      previous = turns[i - 1] if i else None
      following = turns[i + 1] if i + 1 < len(turns) else None
      if turn.role == 'system':
        pass  # Already rendered above, with the reasoning/tool preamble.
      elif turn.role == 'user':
        out.append(self.user_marker + turn.content + self.end_of_message_marker)
      elif turn.role == 'assistant':
        out.append(self.assistant_marker)
        if self.preserve_thinking or i > last_query_index:
          out.append(
              f'{self.begin_of_thought_marker}\n{turn.reasoning_content}\n'
              f'{self.end_of_thought_marker}\n\n'
          )
        out.append(turn.content)
        out.append(self._format_tool_calls(turn))
        out.append(self.end_of_message_marker)
      else:  # tool
        if previous is not None and previous.role != 'tool':
          out.append(f'{_IM_START}user')
        out.append(f'\n<tool_response>\n{turn.content}\n</tool_response>')
        if following is None or following.role != 'tool':
          out.append(self.end_of_message_marker)

    out.append(self.assistant_marker)
    if self.enable_thinking:
      out.append(f'{self.begin_of_thought_marker}\n')
    else:
      out.append(
          f'{self.begin_of_thought_marker}\n\n{self.end_of_thought_marker}\n\n'
      )
    return ''.join(out)

  def format_tokens(
      self,
      messages: Sequence[Mapping[str, Any]],
      tokenizer: Any,
      trainable_roles: tuple[str, ...] | None = None,
  ) -> tuple[list[int], list[float]]:
    """Refuses to build SFT targets: these formats are inference-only.

    The inherited implementation concatenates `f'{role}_marker'` strings, which
    cannot express this format: no reasoning-effort system turn, no `<think>`
    block, tool calls dropped, and a crash on the FC2.0 content list `format()`
    accepts. Training on that skews every thinking marker against what
    `format()` serves, so this fails loudly instead. Implementing it means
    rendering `format()` segment-wise with a trainable flag per segment, the
    way `zoo/kimi_k3/utils/lm_format.py` does.

    Args:
      messages: the conversation.
      tokenizer: the vocabulary to encode with.
      trainable_roles: roles to include in the loss.

    Raises:
      NotImplementedError: always.
    """
    raise NotImplementedError(
        'Qwen3.8 chat formats are inference-only: format_tokens would need a'
        ' segment-wise renderer to carry the loss mask (see'
        ' zoo/kimi_k3/utils/lm_format.py).'
    )

  def _format_tool_calls(self, turn: _Turn) -> str:
    out = []
    for i, (name, args) in enumerate(turn.tool_calls):
      if i == 0:
        out.append('\n\n' if turn.content.strip() else '')
      else:
        out.append('\n')
      out.append(f'<tool_call>\n<function={name}>\n')
      for key, value in args.items():
        rendered = value if isinstance(value, str) else _tojson(value)
        out.append(f'<parameter={key}>\n{rendered}\n</parameter>\n')
      out.append('</function>\n</tool_call>')
    return ''.join(out)

  def parse(
      self,
      formatted: str,
      function_declarations: (
          Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None
      ) = None,
  ) -> Sequence[Mapping[str, Any]]:
    """Inverse of `format()` on the assistant turns, to FC2.0 parts.

    Only the assistant turns round-trip: a `tool` result comes back as the
    `user` turn the template folded it into (which `format()` re-renders
    byte-identically), and a `system` turn comes back with the injected
    reasoning sentence still in its content, which `format()` would inject
    again. Re-formatting a whole parsed transcript is therefore not stable;
    parsing the continuation, which is what serving does, is.

    Args:
      formatted: a rendered conversation, or (what `page_batcher` passes) the
        assistant marker followed by the model's raw continuation. Because
        `format()`'s generation prompt ends *inside* an open `<think>` block, an
        assistant turn with no `<think>` opener is treated as starting in the
        thought channel.
      function_declarations: optional schemas -- a list of declarations or an
        already-keyed `{name: declaration}` -- used to coerce non-string
        arguments (the XML call format carries no types). Same shapes as the
        other implementation of this hook, `GeminiChat.parse` in eva_research.
        `page_batcher` passes none, so every argument stays a string there --
        correct for the agent loop, whose only tool argument is a string.

    Returns:
      One message dict per turn; assistant content is a list of parts.
    """
    declarations = _declarations_by_name(function_declarations)
    messages: list[dict[str, Any]] = []
    for segment in formatted.split(_IM_START):
      if not segment:
        continue
      role, newline, rest = segment.partition('\n')
      if not newline:
        continue  # Trailing generation prompt, e.g. 'assistant'.
      rest, ended = self._strip_end_of_turn(rest)
      if role == 'assistant':
        content: Any = self._parse_assistant(rest, ended, declarations)
      else:
        content = rest
      if content:
        messages.append({'role': role, 'content': content})
    return messages

  def _strip_end_of_turn(self, text: str) -> tuple[str, bool]:
    """Drops the trailing stop tokens (decoding keeps specials).

    Args:
      text: one turn, as decoded.

    Returns:
      The turn without its stop tokens, and whether any was removed -- i.e.
      whether the model ENDED the turn or was cut off mid-stream, which is the
      only evidence `_parse_assistant` has that an unclosed `<think>` block is
      the model's choice rather than a truncation.
    """
    ended = False
    while True:
      candidate = text.rstrip()
      for token in self.extra_eos_tokens:
        if candidate.endswith(token):
          text = candidate[: -len(token)]
          ended = True
          break
      else:
        return text, ended

  def _parse_assistant(
      self,
      text: str,
      ended: bool,
      declarations: Mapping[str, Mapping[str, Any]],
  ) -> list[Mapping[str, Any]]:
    """Splits one assistant turn into thought / text / function_call parts.

    Args:
      text: the turn, with its stop token already removed.
      ended: whether a stop token WAS removed, i.e. the model finished the turn
        (`_strip_end_of_turn`).
      declarations: `{name: declaration}` for argument type coercion.

    Returns:
      The parts, in emission order.
    """
    parts: list[Mapping[str, Any]] = []
    if text.startswith(self.begin_of_thought_marker):
      text = text[len(self.begin_of_thought_marker) :]
    thought, separator, remainder = text.partition(self.end_of_thought_marker)
    if separator:
      text = remainder
    elif self.enable_thinking and not ended:
      # The generation prompt left the thought block OPEN and the model never
      # closed it or stopped, so the turn was cut off mid-thought.
      text = ''
    else:
      # Everything is content: either `enable_thinking=False` closed the block
      # in the PROMPT (`<think>\n\n</think>\n\n`), or the model ENDED a turn
      # that never opened one. Reading it as thinking would drop every tool
      # call and the agent loop would take the turn as 'the model is done'.
      thought, text = '', thought
    # A turn that really was cut off mid-thought therefore yields no tool call.
    # The agent loop tells that from a deliberate stop with `resp['truncated']`
    # (set from `reached_eos`): its `response_truncated` check runs BEFORE the
    # parts are interpreted.
    if thought.strip():
      parts.append({'text': thought.strip(), 'channel': _THOUGHT_CHANNEL})

    position = 0
    while (start := text.find(_TOOL_CALL_OPEN, position)) != -1:
      prefix = text[position:start].strip()
      if prefix:
        parts.append({'text': prefix})
      calls, position = _parse_tool_call(
          text, start + len(_TOOL_CALL_OPEN), declarations
      )
      if calls:
        parts.extend({'function_call': call} for call in calls)
      else:
        # Keep an unparsable block as text rather than dropping it: an empty
        # turn is what the agent loop reads as 'the model is done'.
        parts.append({'text': text[start:position].strip()})
    tail = text[position:].strip()
    if tail:
      parts.append({'text': tail})
    return parts


def _response_text(response: Mapping[str, Any]) -> str:
  """Renders a `function_response` part as the body of a `<tool_response>`.

  The body is the tool's raw text: this XML tool dialect comes from the
  Qwen3-Coder lineage, where `<tool_response>` carries verbatim tool output,
  and JSON-escaping a 16 KB `cat` costs ~6-9% more tokens *every* turn while
  handing the agent source code with escaped newlines to read.

  Args:
    response: a FC2.0 FunctionResponse (`{'name', 'response'}`) or a bare
      payload.

  Returns:
    The `<tool_response>` body.
  """
  payload = response.get('response', response)
  if isinstance(payload, str):
    return payload  # The caller strips every branch.
  if not isinstance(payload, Mapping):
    return _tojson(payload)
  if not set(payload) <= {'stdout', 'stderr'}:
    return _tojson(payload)  # Some other tool's schema; keep it lossless.
  stdout = str(payload.get('stdout') or '').strip()
  stderr = str(payload.get('stderr') or '').strip()
  if not stdout and not stderr:
    return _TOOL_RESPONSE_EMPTY
  if not stderr:
    return stdout
  return f'{stdout}\n{_TOOL_RESPONSE_STDERR_MARKER}\n{stderr}'.lstrip()


def _last_query_index(turns: Sequence[_Turn]) -> int:
  """Index of the last user turn that is not just a tool result."""
  for i in reversed(range(len(turns))):
    turn = turns[i]
    if turn.role != 'user':
      continue
    if not (
        turn.content.startswith('<tool_response>')
        and turn.content.endswith('</tool_response>')
    ):
      return i
  raise ValueError('No user query found in messages.')


@lm_format_lib.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38ChatMedium(Qwen38Chat):
  """`reasoning_effort=medium`: no injected system sentence."""

  reasoning_effort: str = 'medium'


@lm_format_lib.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38ChatLow(Qwen38Chat):
  """`reasoning_effort=low`."""

  reasoning_effort: str = 'low'


@lm_format_lib.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38ChatNoThink(Qwen38Chat):
  """`enable_thinking=False`: the prompt pre-closes an empty think block."""

  enable_thinking: bool = False


@lm_format_lib.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38ChatFC(Qwen38Chat):
  """Alias of `Qwen38Chat` naming the FC2.0 agentic use (DeepSWE, SWE-Bench).

  `parse()` lives on the base class so that pointing an agentic eval at
  `Qwen38Chat` by mistake cannot silently drop `output_messages`.
  """


@lm_format_lib.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38ChatFCMedium(Qwen38ChatFC):
  """`Qwen38ChatFC` with `reasoning_effort=medium`."""

  reasoning_effort: str = 'medium'


@lm_format_lib.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class Qwen38ChatFCLow(Qwen38ChatFC):
  """`Qwen38ChatFC` with `reasoning_effort=low`."""

  reasoning_effort: str = 'low'


def _declarations_by_name(
    function_declarations: (
        Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None
    ),
) -> dict[str, Mapping[str, Any]]:
  """Normalizes the `parse()` schema argument to `{name: declaration}`."""
  result: dict[str, Mapping[str, Any]] = {}
  if function_declarations is None:
    return result
  if isinstance(function_declarations, Mapping):
    mapping = cast(Mapping[str, Any], function_declarations)
    for key, value in mapping.items():
      if isinstance(key, str) and isinstance(value, Mapping):
        result[key] = value
    return result
  for item in cast(Iterable[Mapping[str, Any]], function_declarations):
    declaration = item
    function = item.get('function')
    if item.get('type') == 'function' and isinstance(function, Mapping):
      declaration = cast(Mapping[str, Any], function)
    name = declaration.get('name')
    if isinstance(name, str):
      result[name] = declaration
  return result


def _coerce_arg(value: str, schema: Mapping[str, Any] | None) -> Any:
  """Casts an XML parameter value back to its declared JSON type."""
  declared = (schema or {}).get('type')
  declared = declared.lower() if isinstance(declared, str) else 'string'
  if declared == 'string':
    return value
  try:
    parsed = json.loads(value)
  except json.JSONDecodeError:
    return value
  if declared == 'integer' and isinstance(parsed, float):
    return int(parsed)
  return parsed


def _parse_tool_call(
    text: str,
    position: int,
    declarations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int]:
  """Parses one `<tool_call>` block into FunctionCall dicts.

  The scan is sequential at every level: a `<function=...>` is read together
  with its parameters and the search for the next one resumes *after* those
  values, so tool markup a coding agent wrote INSIDE a parameter (`echo
  '<function=evil>'`, a heredoc containing `</tool_call>`) stays data instead
  of fabricating a call or truncating the real one. A missing `</function>` /
  `</tool_call>` is tolerated and more than one function per block is accepted:
  dropping a malformed or extra call would hand back a turn with fewer calls
  than the model asked for, and the agent loop reads a turn with none as 'the
  model is done'.

  Args:
    text: the whole assistant turn.
    position: the offset just past this block's `<tool_call>` tag.
    declarations: `{name: declaration}` for argument type coercion.

  Returns:
    One `{'name', 'args'}` per function block in emission order (empty if the
    block holds no recognizable call), and the offset just past the block's
    `</tool_call>` (or the end of `text` if the model stopped before it).
  """
  calls = []
  cursor = position
  while (start := _FUNCTION_RE.search(text, cursor)) is not None:
    close = text.find(_TOOL_CALL_CLOSE, cursor)
    if 0 <= close < start.start():
      break  # That function opens a later block.
    name = start.group(1)
    properties = (declarations.get(name, {}).get('parameters') or {}).get(
        'properties'
    ) or {}
    arguments, cursor = _parse_parameters(text, start.end())
    calls.append({
        'name': name,
        'args': {
            key: _coerce_arg(value, properties.get(key))
            for key, value in arguments
        },
    })
  close = text.find(_TOOL_CALL_CLOSE, cursor)
  if close < 0:  # The model stopped before closing the block.
    return calls or _json_tool_call(text[position:]), len(text)
  if not calls:
    calls = _json_tool_call(text[position:close])
  return calls, close + len(_TOOL_CALL_CLOSE)


def _json_tool_call(body: str) -> list[Mapping[str, Any]]:
  """Reads a pre-3.8 Qwen JSON tool call; `[]` if the block holds no call."""
  try:
    call = json.loads(body)
  except json.JSONDecodeError:
    return []
  if not isinstance(call, Mapping) or 'name' not in call:
    return []
  return [{'name': call['name'], 'args': call.get('arguments') or {}}]


def _parse_parameters(
    text: str, position: int
) -> tuple[list[tuple[str, str]], int]:
  r"""Scans one function's `<parameter=k>\nv\n</parameter>` blocks.

  Values are taken verbatim (no strip): leading indentation and trailing
  newlines are part of the command the agent asked for. The scan is sequential
  -- the cursor jumps past each terminator -- so `<parameter=` written inside a
  value, or `</parameter>` that is not where a value can end, is not mistaken
  for markup. `</parameter>` immediately followed by `<parameter=` or
  `</function>` inside a value is the one case this wire format cannot express
  and no parser here can recover.

  Args:
    text: the whole assistant turn.
    position: the offset just past this function's `<function=NAME>` tag.

  Returns:
    The (name, value) pairs in emission order, and the offset just past the
    function's `</function>` (or at the block's `</tool_call>`, or the end of
    `text`, if the model stopped before writing one).
  """
  parameters = []
  cursor = position
  while (start := _MARKUP_RE.search(text, cursor)) is not None:
    if start.group('close'):
      end_of_function = start.group('close') == _FUNCTION_CLOSE
      return parameters, start.end() if end_of_function else start.start()
    end = _PARAMETER_END_RE.search(text, start.end())
    if end is None:
      # Cut off inside the value; that `\n` would have preceded `</parameter>`.
      value = text[start.end() :].rstrip('\n')
      parameters.append((start.group('name'), value))
      return parameters, len(text)
    parameters.append((start.group('name'), text[start.end() : end.start()]))
    cursor = end.end()
  return parameters, len(text)

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
"""Kimi K3's "XTML" chat format as a Simply `LMFormat`, and its inverse.

K3 ships no jinja template: the release renders prompts programmatically in
`encoding_k3.py:build_chat_segments`, which this module reimplements. Both
layers are golden-tested token-for-token: against `build_chat_segments`
(`testdata/kimi_k3_chat_goldens.json`) and against the public entry point
`TikTokenTokenizer.apply_chat_template`, whose defaults -- notably
`thinking_effort='max'` -- are what a served K3 actually sees
(`testdata/kimi_k3_chat_template_goldens.json`).

    <|open|>message role="user"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>

  * system / user / tool messages are one `message` tag; tool messages carry
    `tool` and 1-based `index` attributes.
  * an assistant message nests channels: a structural `think` channel (emitted
    even when empty -- K3 always thinks), a `response` channel, then an
    optional `tools` block of `call` tags with one `argument` tag per argument.
  * tool declarations, `thinking_effort`, `tool_choice` and `response_format`
    are system messages with a `type` attribute.
  * the generation prompt is
    `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`, so a
    completion starts *inside* the think channel; it ends at `<|end_of_msg|>`.
  * attribute values escape `&` and `"`.

Rendering produces `Segment`s rather than one string so that structural
markers can be encoded as tiktoken specials while content is encoded with
specials disabled -- user text can then never fabricate a control token.
`format()` (the `LMFormat` string API) flattens the segments and relies on
`KimiK3Vocab.encode(allow_special=True)`; `format_tokens()` is the exact,
injection-safe path.

`parse_response` is the inverse an agentic harness needs: it turns a decoded
assistant turn back into `{'thinking', 'content', 'tool_calls'}` and tolerates
truncation anywhere in the stream. `parse` is the same thing shaped as core's
serving hook.
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
import re
from typing import Any

from simply.utils import lm_format
from simply.utils import sampling_lib
from simply.zoo.kimi_k3.utils import tokenization as tokenization_lib

OPEN = tokenization_lib.OPEN
CLOSE = tokenization_lib.CLOSE
SEP = tokenization_lib.SEP
END_OF_MSG = tokenization_lib.END_OF_MSG

GENERATION_PROMPT = f'{OPEN}message role="assistant"{SEP}{OPEN}think{SEP}'

VALID_THINKING_EFFORTS = ('low', 'high', 'max')

_TOOL_DECLARE_HEADER = (
    '# Tools\nHere are the available tools, described in JSONSchema.\n\n'
)
_DYNAMIC_TOOL_DECLARE_HEADER = (
    '## New Tools Available\n'
    'The system dynamically extends the toolset via lazy-loading.\n'
    'You have access to all existing and extended tools.\n'
    'Here are the specs for the extended tools.\n\n'
)


@dataclasses.dataclass(frozen=True)
class Segment:
  """A rendered chunk of the prompt.

  Attributes:
    text: The text.
    allow_special: Whether `text` is a structural marker, i.e. must encode to
      special-token ids. Content segments are always False.
    role: The message role this segment belongs to, for loss masking.
    trainable: Whether this segment is part of what the model generates (and so
      may carry loss when its `role` is trainable). The tags that the generation
      prompt supplies -- the assistant `message` tag and the first channel tag
      -- are False; everything the model itself emits, including the closing
      tags and `<|end_of_msg|>`, is True.
  """

  text: str
  allow_special: bool = False
  role: str | None = None
  trainable: bool = False


def escape_attr(value: Any) -> str:
  return str(value).replace('&', '&amp;').replace('"', '&quot;')


def unescape_attr(value: str) -> str:
  """Inverse of `escape_attr` (`&quot;` first, so `&amp;quot;` survives)."""
  return value.replace('&quot;', '"').replace('&amp;', '&')


def _compact_json(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _deep_sort(obj: Any) -> Any:
  """Sorts every dict by key, recursively (the release sorts tool schemas)."""
  if isinstance(obj, Mapping):
    return {k: _deep_sort(obj[k]) for k in sorted(obj)}
  if isinstance(obj, (list, tuple)):
    return [_deep_sort(x) for x in obj]
  return obj


def xtml_type(value: Any) -> str:
  """The `type` attribute the release gives a tool-call argument value."""
  if isinstance(value, bool):
    return 'boolean'
  if value is None:
    return 'null'
  if isinstance(value, (int, float)):
    return 'number'
  if isinstance(value, str):
    return 'string'
  if isinstance(value, Mapping):
    return 'object'
  return 'array'


def xtml_value(value: Any) -> str:
  """Renders a tool-call argument value: strings raw, everything else JSON."""
  if isinstance(value, str):
    return value
  return json.dumps(value, ensure_ascii=False)


def parse_xtml_value(text: str, type_name: str | None) -> Any:
  """Inverse of `xtml_value`; falls back to the raw text if it is not JSON."""
  if type_name is None or type_name == 'string':
    return text
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def normalize_tool_arguments(
    arguments: Any,
) -> tuple[dict[str, Any], str | None]:
  """Returns (arguments dict, raw JSON block when the string is unparseable).

  Tool calls arrive either with a dict of arguments or with the OpenAI-style
  JSON string; a string the model truncated or mangled cannot be split into
  `argument` tags, and the release renders it verbatim inside a `json` tag
  instead.

  Args:
    arguments: A dict, a JSON object string, or None.

  Returns:
    The arguments dict and the raw block (mutually exclusive).
  """
  if arguments is None:
    return {}, None
  if isinstance(arguments, Mapping):
    return dict(arguments), None
  if isinstance(arguments, str):
    if not arguments.strip():
      return {}, None
    try:
      parsed = json.loads(arguments)
    except json.JSONDecodeError:
      return {}, arguments
    if not isinstance(parsed, dict):
      raise ValueError('Kimi K3 tool call arguments must be a JSON object.')
    return parsed, None
  raise TypeError(
      'Kimi K3 tool call arguments must be a dict or a JSON object string.'
  )


_MARKERS = (OPEN, CLOSE, SEP, END_OF_MSG)
_MARKER_RE = re.compile('|'.join(re.escape(marker) for marker in _MARKERS))
_MAX_MARKER_LEN = max(len(marker) for marker in _MARKERS)


def _tool_call_function(tool_call: Mapping[str, Any]) -> Mapping[str, Any]:
  """Accepts both `{'function': {...}}` and flat `{'name', 'arguments'}`."""
  function = tool_call.get('function')
  return function if isinstance(function, Mapping) else tool_call


def sort_tool_results(
    messages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
  """Re-sorts each run of tool messages into the assistant's call order.

  XTML numbers tool results by position, so a harness that returns them out of
  order (OpenAI messages are matched by `tool_call_id`, not by position) would
  otherwise mislabel them. A run in which any message cannot be matched is
  left untouched, and the matched call's name wins over an explicit `tool`.
  Mirrors `encoding_k3.normalize_xtml_tool_result_messages`; side-effect free.

  Args:
    messages: The conversation.

  Returns:
    The conversation with each tool-message run sorted.
  """
  output: list[Mapping[str, Any]] = []
  index: dict[str, tuple[int, Any]] = {}
  i = 0
  while i < len(messages):
    message = messages[i]
    if message.get('role') == 'assistant':
      index = {}
      for position, tool_call in enumerate(message.get('tool_calls') or [], 1):
        call_id = tool_call.get('id')
        if call_id is not None and str(call_id) not in index:
          index[str(call_id)] = (
              position,
              _tool_call_function(tool_call).get('name'),
          )
      output.append(message)
      i += 1
      continue
    if message.get('role') != 'tool':
      output.append(message)
      i += 1
      continue

    run = []
    while i < len(messages) and messages[i].get('role') == 'tool':
      tool_message = messages[i]
      call_id = tool_message.get('tool_call_id', tool_message.get('id'))
      matched = index.get(str(call_id)) if call_id is not None else None
      run.append((matched, len(run), tool_message))
      i += 1
    if any(matched is None for matched, _, _ in run):
      output.extend(tool_message for _, _, tool_message in run)
      continue
    run.sort(key=lambda item: (item[0][0], item[1]))
    for matched, _, tool_message in run:
      if matched[1] is None:  # Matched a call that has no name to impose.
        output.append(tool_message)
        continue
      resolved = dict(tool_message)
      resolved['tool'] = matched[1]
      if 'name' in resolved:
        resolved['name'] = matched[1]
      output.append(resolved)
  return output


class _Renderer:
  """Accumulates `Segment`s for one conversation."""

  def __init__(self):
    self.segments: list[Segment] = []
    self.role: str | None = None
    self.trainable: bool = False

  def control(self, text: str) -> None:
    self.segments.append(
        Segment(text, True, role=self.role, trainable=self.trainable)
    )

  def text(self, text: Any) -> None:
    text = str(text)
    if text:
      self.segments.append(
          Segment(text, False, role=self.role, trainable=self.trainable)
      )

  def open_tag(self, tag: str, attrs: Iterable[tuple[str, Any]] = ()) -> None:
    self.control(OPEN)
    self.text(tag)
    for key, value in attrs:
      self.text(f' {key}')
      self.text('="')
      self.text(escape_attr(value))
      self.text('"')
    self.control(SEP)

  def close_tag(self, tag: str) -> None:
    self.control(CLOSE)
    self.text(tag)
    self.control(SEP)

  def end_of_msg(self) -> None:
    self.control(END_OF_MSG)

  def content(self, content: Any) -> None:
    """Renders message content: a string, or a list of typed parts."""
    if isinstance(content, str):
      self.text(content)
    elif content is not None:
      for part in content:
        if part['type'] in ('image', 'image_url'):
          raise NotImplementedError(
              'Kimi K3 image content is outside the Simply text path.'
          )
        self.text(part['text'])

  def simple_message(
      self, attrs: Sequence[tuple[str, Any]], content: Any
  ) -> None:
    self.open_tag('message', attrs)
    self.trainable = True
    self.content(content)
    self.close_tag('message')
    self.end_of_msg()
    self.trainable = False

  def system_message(self, message_type: str, body: str) -> None:
    """An internal (system-generated) system message; never trainable."""
    self.role = 'system'
    self.open_tag('message', [('role', 'system'), ('type', message_type)])
    self.text(body.strip())
    self.close_tag('message')
    self.end_of_msg()
    self.role = None


def _render_tool_declare(
    renderer: _Renderer, tools: Any, *, dynamic: bool = False
) -> None:
  header = _DYNAMIC_TOOL_DECLARE_HEADER if dynamic else _TOOL_DECLARE_HEADER
  body = f'{header}```json\n{_compact_json(_deep_sort(tools))}\n```'
  renderer.system_message('tool-declare', body)


def _render_assistant(
    renderer: _Renderer, message: Mapping[str, Any], thinking: bool
) -> None:
  """The assistant body: think and response channels, then tool calls."""
  if thinking:
    renderer.open_tag('think')
    renderer.trainable = True
    reasoning = message.get('reasoning_content') or message.get('reasoning')
    if reasoning is not None and str(reasoning).strip():
      renderer.text(reasoning)
    renderer.close_tag('think')
    renderer.open_tag('response')
  else:
    renderer.open_tag('response')
    renderer.trainable = True
  renderer.content(message.get('content'))
  renderer.close_tag('response')

  tool_calls = message.get('tool_calls')
  if not tool_calls:
    return
  renderer.open_tag('tools')
  for index, tool_call in enumerate(tool_calls, start=1):
    function = _tool_call_function(tool_call)
    renderer.open_tag('call', [('tool', function['name']), ('index', index)])
    arguments, json_block = normalize_tool_arguments(function.get('arguments'))
    if json_block is not None:
      renderer.open_tag('json', [('type', 'object')])
      renderer.text(json_block)
      renderer.close_tag('json')
    else:
      for key, value in arguments.items():
        renderer.open_tag(
            'argument', [('key', key), ('type', xtml_type(value))]
        )
        renderer.text(xtml_value(value))
        renderer.close_tag('argument')
    renderer.close_tag('call')
  renderer.close_tag('tools')


def render_segments(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    *,
    add_generation_prompt: bool = True,
    thinking: bool = True,
    thinking_effort: str | None = None,
    tool_choice: str | None = None,
    response_format: Mapping[str, Any] | str | None = None,
    response_schema: Any = None,
) -> list[Segment]:
  """Renders a conversation to XTML segments (see the module docstring).

  Args:
    messages: OpenAI-style messages: `role`, `content`, and optionally `name`,
      `reasoning_content`, `tool_calls`, `tool`/`tool_call_id`, or `tools` (on a
      system message: a dynamic tool declaration).
    tools: Tool declarations (JSONSchema dicts), rendered as the leading
      tool-declare system message.
    add_generation_prompt: Append the assistant message + first channel tag.
    thinking: Render the structural think channel. K3 always thinks; False is
      the release's non-thinking mode, which drops the channel entirely.
    thinking_effort: One of `VALID_THINKING_EFFORTS`, as a system message.
    tool_choice: 'required' or 'none', as a system message.
    response_format: OpenAI response_format (`json_object` / `json_schema`), as
      a system message.
    response_schema: Overrides the schema dug out of `response_format`.

  Returns:
    The segments, in order.

  Raises:
    ValueError: On an unusable `thinking_effort`, an unknown role, or a tool
      message whose tool name cannot be resolved.
    NotImplementedError: On image content, which the Simply text path does not
      carry (the release renders an image placeholder).
  """
  renderer = _Renderer()

  if tools:
    _render_tool_declare(renderer, tools)

  if thinking and thinking_effort is not None:
    if thinking_effort not in VALID_THINKING_EFFORTS:
      raise ValueError(
          f'Unsupported thinking_effort={thinking_effort!r}; supported values'
          f' are {sorted(VALID_THINKING_EFFORTS)}.'
      )
    renderer.system_message(
        'thinking-effort',
        '`thinking_effort` guides on how much to think in your thinking'
        ' channel (not including the response channel), supported values'
        ' include `low`, `medium`, `high`, and `max`.\nNow the system is'
        f' invoked with `thinking_effort={thinking_effort}`.',
    )

  tool_calls: Sequence[Mapping[str, Any]] | None = None
  tool_index = 0
  for message in sort_tool_results(messages):
    role = message['role']
    renderer.role = role
    if role == 'user':
      attrs = [('role', 'user')]
      if message.get('name'):
        attrs.append(('name', message['name']))
      renderer.simple_message(attrs, message.get('content'))
    elif role == 'system' and message.get('tools'):
      _render_tool_declare(renderer, message['tools'], dynamic=True)
    elif role == 'system':
      attrs = [('role', 'system')]
      if message.get('name'):
        attrs.append(('name', message['name']))
      renderer.simple_message(attrs, message.get('content'))
    elif role == 'tool':
      tool_index += 1
      tool_name = message.get('tool', message.get('name'))
      if tool_name is None and tool_calls and tool_index <= len(tool_calls):
        tool_name = _tool_call_function(tool_calls[tool_index - 1])['name']
      if tool_name is None:
        raise ValueError(
            'Kimi K3 tool messages need a resolvable tool name: carry'
            ' `tool`/`name`, or match a preceding assistant tool_call by'
            ' order.'
        )
      renderer.simple_message(
          [('role', 'tool'), ('tool', tool_name), ('index', tool_index)],
          message.get('content'),
      )
    elif role == 'assistant':
      tool_calls = message.get('tool_calls')
      tool_index = 0
      attrs = [('role', 'assistant')]
      if message.get('name'):
        attrs.append(('name', message['name']))
      renderer.open_tag('message', attrs)
      _render_assistant(renderer, message, thinking)
      renderer.close_tag('message')
      renderer.end_of_msg()
      renderer.trainable = False
    else:
      raise ValueError(f'Unknown role: {role!r}')
    renderer.role = None

  if tool_choice == 'required':
    renderer.system_message(
        'tool-choice',
        'The system is invoked with `tool_choice=required`.\n'
        'You MUST call tools in the next message.',
    )
  elif tool_choice == 'none':
    renderer.system_message(
        'tool-choice',
        'The system is invoked with `tool_choice=none`.\n'
        'You MUST NOT call any tools in the next message.',
    )

  _render_response_format(renderer, response_format, response_schema)

  if add_generation_prompt:
    renderer.role = 'assistant'
    renderer.open_tag('message', [('role', 'assistant')])
    renderer.open_tag('think' if thinking else 'response')
    renderer.role = None
  return renderer.segments


def extract_response_schema(response_format: Any) -> Any:
  """Digs the JSON schema out of an OpenAI `response_format`.

  Mirrors `encoding_k3.extract_response_schema`, including its unwrapping of
  a `json_schema` nested inside a `json_schema`.

  Args:
    response_format: The OpenAI `response_format` object.

  Returns:
    The schema, or None.
  """
  if not isinstance(response_format, Mapping):
    return None
  json_schema = response_format.get('json_schema')
  if json_schema is None:
    return None
  if isinstance(json_schema, Mapping):
    return json_schema.get(
        'schema', json_schema.get('json_schema', json_schema)
    )
  return json_schema


def _render_response_format(
    renderer: _Renderer,
    response_format: Mapping[str, Any] | str | None,
    response_schema: Any = None,
) -> None:
  """The `response_format=json_object|json_schema` system messages."""
  if response_format is None:
    return
  if isinstance(response_format, Mapping):
    format_type = response_format.get('type')
  else:
    format_type = response_format
  if format_type == 'json_object':
    renderer.system_message(
        'response-format',
        'The system is invoked with `response_format=json_object`.\n'
        'Your response must be raw JSON data without markdown code blocks'
        ' (```json) or any additional formatting.',
    )
  elif format_type == 'json_schema':
    schema = (
        response_schema
        if response_schema is not None
        else extract_response_schema(response_format)
    )
    renderer.system_message(
        'response-format',
        'The system is invoked with `response_format=json_schema`.\n'
        'Your response must be raw JSON data without markdown code blocks'
        ' (```json) or any additional formatting.\n'
        'The JSON data must match the following schema:\n'
        f'```json\n{_compact_json(_deep_sort(schema))}\n```',
    )


def encode_segments(segments: Sequence[Segment], vocab: Any) -> list[int]:
  """Encodes segments, keeping control tokens and content strictly apart."""
  token_ids: list[int] = []
  for segment in segments:
    token_ids.extend(
        vocab.encode(segment.text, allow_special=segment.allow_special)
    )
  return token_ids


_HEADER_ATTR_RE = re.compile(r' ([^\s=]+)="([^"]*)"')


def _header_segments(header: str) -> list[Segment]:
  """Splits a tag header back into the pieces the renderer emitted."""
  tag = header.split(' ', 1)[0]
  segments = [Segment(tag)] if tag else []
  rebuilt = tag
  for key, value in _HEADER_ATTR_RE.findall(header[len(tag) :]):
    segments.append(Segment(f' {key}'))
    segments.append(Segment('="'))
    if value:
      segments.append(Segment(value))
    segments.append(Segment('"'))
    rebuilt += f' {key}="{value}"'
  if rebuilt != header:  # Not a header we rendered; encode it as one run.
    return [Segment(header)]
  return segments


def resegment(text: str) -> list[Segment]:
  """Recovers the renderer's segmentation from an already rendered prompt.

  BPE merges across a segment boundary, so encoding the flat `format()` string
  in one shot does NOT always reproduce the reference ids: the release encodes
  a tag header piece by piece (` key`, `="`, value, `"`), and for ~1 attribute
  in 6 the merged spelling tokenizes differently. Re-splitting the string on
  the markers and on the header grammar restores the exact segmentation, which
  is what `KimiK3InputProcessor` uses to keep Simply's string-based sampling
  path reference-exact.

  This recovers *tokenization*, not provenance: text that spells a marker is
  indistinguishable from a marker here (use `KimiK3Chat.format_tokens` when
  the content is untrusted -- though this is still the safer of the two string
  paths, mis-reading 4 markers where a plain one-shot encode promotes all 256
  special spellings), and content passed as a list of parts, which the release
  renders one segment per part, cannot be split again.

  Args:
    text: A rendered prompt (or any text; unrecognized runs stay whole).

  Returns:
    The segments.
  """
  segments: list[Segment] = []
  position = 0
  in_header = False
  for match in _MARKER_RE.finditer(text):
    run = text[position : match.start()]
    if run:
      segments.extend(_header_segments(run) if in_header else [Segment(run)])
    segments.append(Segment(match.group(), allow_special=True))
    in_header = match.group() in (OPEN, CLOSE)
    position = match.end()
  tail = text[position:]
  if tail:
    segments.extend(_header_segments(tail) if in_header else [Segment(tail)])
  return segments


# --- Parsing an assistant turn back out of XTML ------------------------------


@dataclasses.dataclass(frozen=True)
class _Tag:
  name: str
  attrs: dict[str, str]
  closing: bool


_ATTR_RE = re.compile(r'([^\s=]+)="([^"]*)"')


# Yielded by `_scan` when the turn ends at `<|end_of_msg|>`.
_END_OF_MESSAGE = object()


def _strip_partial_marker(text: str) -> str:
  """Drops a trailing marker prefix, i.e. a decode that stopped mid-marker.

  Costs a trailing `<`-ish fragment of genuinely truncated content, which is
  the better error: emitting `<|op` as content invites a harness to display or
  re-send half a control token.

  Args:
    text: A content run that reached the end of the output.

  Returns:
    The content without the dangling marker prefix.
  """
  for length in range(min(_MAX_MARKER_LEN - 1, len(text)), 0, -1):
    suffix = text[-length:]
    if any(
        marker.startswith(suffix) and len(marker) > length
        for marker in _MARKERS
    ):
      return text[:-length]
  return text


def _scan(text: str):
  """Yields `str` (content), `_Tag` and `_END_OF_MESSAGE` events.

  A trailing partial marker or a tag with no closing `<|sep|>` (i.e. the
  output was truncated mid-tag) ends the scan and is dropped.

  Args:
    text: The decoded assistant output.

  Yields:
    Content strings, `_Tag`s and at most one `_END_OF_MESSAGE`, in order.
  """
  position = 0
  while position < len(text):
    candidates = [
        (found, marker)
        for marker in (OPEN, CLOSE, END_OF_MSG)
        if (found := text.find(marker, position)) >= 0
    ]
    if not candidates:
      yield _strip_partial_marker(text[position:])
      return
    start, marker = min(candidates)
    if start > position:
      yield text[position:start]
    if marker == END_OF_MSG:
      yield _END_OF_MESSAGE
      return
    header_start = start + len(marker)
    header_end = text.find(SEP, header_start)
    if header_end < 0:
      return
    header = text[header_start:header_end]
    name, _, attr_text = header.partition(' ')
    yield _Tag(
        name=name.strip(),
        attrs={
            key: unescape_attr(value)
            for key, value in _ATTR_RE.findall(attr_text)
        },
        closing=marker == CLOSE,
    )
    position = header_end + len(SEP)


class _ResponseParser:
  """Turns one assistant XTML turn into thinking / content / tool calls."""

  def __init__(self, initial_channel: str | None):
    self._channel = initial_channel
    self._chunks: dict[str, list[str]] = {'think': [], 'response': []}
    self._tool_calls: list[dict[str, Any]] = []
    self._call: dict[str, Any] | None = None
    self._argument: tuple[str, str | None] | None = None
    self._buffer: list[str] = []
    self._in_tools = False
    self._complete = False

  def parse(self, text: str) -> dict[str, Any]:
    """Runs the scanner to the end of the message, or to the end of `text`.

    Args:
      text: one decoded assistant turn.

    Returns:
      The parsed turn; see `KimiK3Chat.parse_response`.
    """
    for event in _scan(text):
      if isinstance(event, str):
        self._on_text(event)
      elif event is _END_OF_MESSAGE:
        self._complete = True
      elif event.closing:
        self._on_close(event)
      else:
        self._on_open(event)
      if self._complete:
        break
    return self._result()

  def _on_text(self, text: str) -> None:
    if self._argument is not None:
      self._buffer.append(text)
    elif self._channel in self._chunks:
      self._chunks[self._channel].append(text)
    elif not self._in_tools:
      # Untagged output (a model that skipped the channel tags) is content.
      self._chunks['response'].append(text)

  def _on_open(self, tag: _Tag) -> None:
    """Enters the channel, tool block, call or argument the tag opens."""
    if tag.name in self._chunks:
      self._channel = tag.name
    elif tag.name == 'tools':
      self._channel = None
      self._in_tools = True
    elif tag.name == 'call':
      self._call = {
          'name': tag.attrs.get('tool', ''),
          'index': _to_int(tag.attrs.get('index'), len(self._tool_calls) + 1),
          'arguments': {},
          'arguments_json': None,
      }
      self._in_tools = True
    elif tag.name == 'argument':
      self._argument = (tag.attrs.get('key', ''), tag.attrs.get('type'))
      self._buffer = []
    elif tag.name == 'json':
      self._argument = ('', 'json')
      self._buffer = []
    elif tag.name == 'message':
      self._channel = None

  def _on_close(self, tag: _Tag) -> None:
    if tag.name in self._chunks:
      self._channel = None
    elif tag.name in ('argument', 'json'):
      self._finish_argument()
    elif tag.name == 'call':
      self._finish_call()
    elif tag.name == 'tools':
      self._in_tools = False
    elif tag.name == 'message':
      self._complete = True

  def _finish_argument(self) -> None:
    if self._argument is None or self._call is None:
      self._argument = None
      return
    key, type_name = self._argument
    text = ''.join(self._buffer)
    if type_name == 'json':
      self._call['arguments_json'] = text
    else:
      self._call['arguments'][key] = parse_xtml_value(text, type_name)
    self._argument = None
    self._buffer = []

  def _finish_call(self) -> None:
    if self._call is None:
      return
    self._tool_calls.append(self._call)
    self._call = None

  def _result(self) -> dict[str, Any]:
    # A call still open at EOF was truncated: its arguments are incomplete, so
    # it is reported separately instead of being handed to a tool executor.
    self._finish_argument()
    partial = self._call
    return {
        'thinking': ''.join(self._chunks['think']),
        'content': ''.join(self._chunks['response']),
        'tool_calls': [
            _as_openai_tool_call(call, position)
            for position, call in enumerate(self._tool_calls, start=1)
        ],
        'partial_tool_call': (
            None
            if partial is None
            else _as_openai_tool_call(partial, len(self._tool_calls) + 1)
        ),
        'complete': self._complete,
    }


def _to_int(value: str | None, default: int) -> int:
  if value is None:
    return default
  try:
    return int(value)
  except ValueError:
    return default


def _as_openai_tool_call(
    call: Mapping[str, Any], position: int
) -> dict[str, Any]:
  """OpenAI shape, so a harness can execute it and feed it back to `format`.

  XTML carries no call id -- it matches results to calls by position -- so one
  is synthesized from the order the calls were parsed in. Deliberately not
  from the `index` attribute: that is model-controlled, and a repeated value
  would give two calls the same id, which `sort_tool_results` then matches to
  the same call. The raw attribute is kept as `index`.

  Args:
    call: The internal `{'name', 'index', 'arguments', 'arguments_json'}` form.
    position: 1-based parse order.

  Returns:
    `{'id', 'index', 'type', 'function': {'name', 'arguments'}}`, with
    `arguments` a JSON string (the raw block verbatim when the model emitted
    a `json` tag) and `index` XTML's 1-based attribute -- not OpenAI's
    0-based streaming index.
  """
  arguments = call['arguments_json']
  if arguments is None:
    arguments = json.dumps(call['arguments'], ensure_ascii=False)
  return {
      'id': f'call_{position}',
      'index': call['index'],
      'type': 'function',
      'function': {'name': call['name'], 'arguments': arguments},
  }


def as_assistant_message(parsed: Mapping[str, Any]) -> dict[str, Any]:
  """Turns a `parse_response` result into the next prompt's assistant message.

  K3 is trained in preserved thinking history mode: the reasoning of every
  earlier assistant turn must be replayed, under the key `reasoning_content`
  that `render_segments` reads. The truncated call and the `complete` flag are
  dropped -- a half-parsed call must not be replayed as if it had happened.
  `KimiK3Chat.parse` hands this shape to core's serving path.

  Args:
    parsed: A `KimiK3Chat.parse_response` result.

  Returns:
    An assistant message.
  """
  return {
      'role': 'assistant',
      'reasoning_content': parsed['thinking'],
      'content': parsed['content'],
      'tool_calls': list(parsed['tool_calls']),
  }


@lm_format.LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class KimiK3Chat(lm_format.LMFormat):
  """Kimi K3's XTML chat format.

  Attributes:
    bos_id: None; K3 prepends no [BOS] (see `KimiK3Vocab.bos_id`).
    pad_id: [PAD].
    extra_eos_tokens: `<|end_of_msg|>` -- the token K3 ends a message with, and
      the only stop condition for a chat turn. [EOS] is added too: it never
      appears in chat, so stopping on it can only save a runaway decode.
    thinking: Render the think channel (K3 always thinks; see
      `render_segments`).
    add_generation_prompt: Append the assistant + first channel tags.
    thinking_effort: 'low' / 'high' / 'max'. Defaults to 'max' because that is
      what the release's public entry point sends
      (`tokenization_kimi.py:apply_chat_template` does
      `kwargs.setdefault('thinking_effort', 'max')`), i.e. what every served K3
      sees; None drops the system message and is NOT the served default.
    tool_choice: Optional 'required' / 'none'.
    response_format_json: An OpenAI `response_format` object, JSON-encoded
      (dataclass fields must be immutable), e.g. '{"type": "json_object"}'.
  """

  bos_id: int | None = None
  pad_id: int | None = tokenization_lib.PAD_ID
  extra_eos_tokens: tuple[str, ...] = (
      tokenization_lib.END_OF_MSG,
      tokenization_lib.EOS,
  )
  begin_of_thought_marker: str | None = f'{OPEN}think{SEP}'
  end_of_thought_marker: str | None = f'{CLOSE}think{SEP}'
  thinking: bool = True
  add_generation_prompt: bool = True
  thinking_effort: str | None = 'max'
  tool_choice: str | None = None
  response_format_json: str | None = None

  def render_segments(
      self, messages: Sequence[Mapping[str, Any]]
  ) -> list[Segment]:
    """Renders `messages` (plus any `tools` they carry) to segments."""
    return render_segments(
        messages,
        _declared_tools(messages),
        add_generation_prompt=self.add_generation_prompt,
        thinking=self.thinking,
        thinking_effort=self.thinking_effort,
        tool_choice=self.tool_choice,
        response_format=(
            None
            if self.response_format_json is None
            else json.loads(self.response_format_json)
        ),
    )

  def format(self, messages: Sequence[Mapping[str, Any]]) -> str:
    """Renders the prompt as a string (encode it with `allow_special=True`)."""
    return ''.join(s.text for s in self.render_segments(messages))

  def format_tokens(
      self,
      messages: Sequence[Mapping[str, Any]],
      tokenizer,
      trainable_roles: tuple[str, ...] | None = None,
  ) -> tuple[list[int], list[float]]:
    """Tokenizes the prompt segment-wise, with a per-token loss mask.

    Only structural markers are encoded as special tokens, so content can
    never fabricate one. The mask covers exactly what the model generates: the
    assistant channels, their closing tags and `<|end_of_msg|>`, but not the
    tags the generation prompt supplies.

    Args:
      messages: The conversation.
      tokenizer: A `KimiK3Vocab` (or any vocab whose `encode` takes
        `allow_special`).
      trainable_roles: Roles to include in the loss; None (the base-class
        default) trains user and tool content too, so SFT wants
        `('assistant',)`.

    Returns:
      The token ids and the loss mask.
    """
    tokens: list[int] = []
    mask: list[float] = []
    for segment in self.render_segments(messages):
      segment_tokens = tokenizer.encode(
          segment.text, allow_special=segment.allow_special
      )
      trainable = segment.trainable and (
          trainable_roles is None or segment.role in trainable_roles
      )
      tokens.extend(segment_tokens)
      mask.extend([1.0 if trainable else 0.0] * len(segment_tokens))
    return tokens, mask

  def parse_response(
      self, response: str | Sequence[int], vocab: Any = None
  ) -> dict[str, Any]:
    """Parses one assistant turn (the inverse of the generation prompt).

    Generation starts inside the think channel, so a completion that stops
    before `<|close|>think<|sep|>` is all thinking and no content. Everything
    is best-effort: a truncated tag is dropped, a truncated tool call is
    reported as `partial_tool_call`, and `complete` says whether the turn
    actually ended.

    Args:
      response: The decoded text, or token ids to decode with `vocab`.
      vocab: A `KimiK3Vocab`, required when `response` is token ids.

    Returns:
      `{'thinking', 'content', 'tool_calls', 'partial_tool_call', 'complete'}`,
      where `tool_calls` are OpenAI-shaped. Build the next turn's assistant
      message with `as_assistant_message`: K3 is trained in *preserved
      thinking history* mode and needs `thinking` fed back as
      `reasoning_content`, which a bare `{'role': 'assistant', **parsed}`
      would silently drop. `complete` keys off the closing `message` tag,
      because Simply's sampler strips the stop token by default
      (`include_eos_in_output_text=False`).

    Raises:
      ValueError: If `response` is token ids and `vocab` is None.
    """
    if not isinstance(response, str):
      if vocab is None:
        raise ValueError('parse_response needs a vocab to decode token ids.')
      response = vocab.decode(list(response))
    parser = _ResponseParser('think' if self.thinking else 'response')
    return parser.parse(response)

  def parse(
      self, response: str | Sequence[int], vocab: Any = None
  ) -> list[dict[str, Any]]:
    """The assistant turn as core's `output_messages` (a serving hook).

    `serving/page_batcher.py` finds this method with
    `getattr(self.lm_format, 'parse', None)`, calls it with the decoded output
    text and stores the result as `output_messages`. That
    field is a *sequence of messages*, not a parsed turn: the hook's other
    implementation, `GeminiChat.parse` in the eva_research fork
    (`learning/gemini/rex/projects/leapfrog/simply/eva_research/utils/`),
    returns `Sequence[Mapping[str, Any]]` and its consumers iterate it for the
    message with `role == 'assistant'`. So this returns the turn as one
    replayable assistant message; `parse_response` stays the full parse, whose
    truncation flags this shape has no room for.

    Core prefixes the text it passes with `assistant_marker`; K3 deliberately
    declares none, because the prefix would have to name the channel the turn
    starts in and that channel depends on `thinking` -- a constant would drop
    a non-thinking turn's content into `thinking`.

    Args:
      response: The decoded text, or token ids to decode with `vocab`.
      vocab: A `KimiK3Vocab`, required when `response` is token ids.

    Returns:
      One assistant message; see `as_assistant_message`.
    """
    return [as_assistant_message(self.parse_response(response, vocab))]


@sampling_lib.InputProcessorRegistry.register
class KimiK3InputProcessor(sampling_lib.BasicTextInputProcessor):
  """Sampling input processor that re-splits the prompt before encoding.

  Simply's sampling path hands the input processor the *string* produced by
  `LMFormat.format`, and the default processor encodes it in one shot, which
  merges BPE tokens across the release's segment boundaries (see `resegment`).
  Set `input_processor_name='KimiK3InputProcessor'` on the experiment config to
  make that path token-identical to the reference encoder.
  """

  def encode(
      self,
      chunks: sampling_lib.ChunkSequence,
      max_input_len: int | None = None,
  ) -> sampling_lib.ProcessedInput:
    tokens = [] if self.bos_id is None else [self.bos_id]
    for chunk in chunks:
      if chunk.type != sampling_lib.Chunk.Type.TEXT:
        raise ValueError(f'Kimi K3 takes text chunks only, got {chunk.type}.')
      tokens.extend(encode_segments(resegment(str(chunk.content)), self.vocab))
    if max_input_len is not None:
      tokens = tokens[-max_input_len:]
    return sampling_lib.ProcessedInput(tokens=tokens)


def _declared_tools(
    messages: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]] | None:
  """Returns the static tool declarations carried by the conversation.

  `LMFormat.format` takes only messages, so a caller that has no access to
  `render_segments`' `tools` argument declares the toolset by hanging it on
  any non-system message (`{'role': 'user', ..., 'tools': [...]}`). `tools` on
  a *system* message means something else -- the release's mid-conversation
  dynamic declaration -- and is rendered in place.

  Args:
    messages: The conversation.

  Returns:
    The tools of the first message that declares them, or None.
  """
  for message in messages:
    tools = message.get('tools')
    if tools and message.get('role') != 'system':
      return tools
  return None

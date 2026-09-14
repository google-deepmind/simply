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
"""Tests for the Qwen3.8 chat formats.

The golden test renders the RELEASED `chat_template.jinja` (checked in under
`testdata/`, byte-identical to `~/hf/Qwen3.8-27B/chat_template.jinja`, to the
copy under `<VOCABS_DIR>/Qwen3.8/`, and to the `chat_template` field of
its `tokenizer_config.json`) with jinja2, configured exactly as HuggingFace
configures it, and requires `Qwen38Chat.format()` to be character-identical.

The template has no notion of FC2.0 parts, so cases that feed `format()` the
agent-loop message shape supply a hand-written `messages_override` for the
reference render: those cases pin the RENDERING, not the FC2.0 -> HF mapping.
The mapping is covered by the explicit-string assertions and `RoundTripTest`.
"""

import json
import os
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
from jinja2 import ext as jinja2_ext
from jinja2 import sandbox as jinja2_sandbox
from simply.utils import lm_format as lm_format_lib
from simply.zoo.qwen3p8.utils import lm_format

# The released template, byte-identical to
# `~/hf/Qwen3.8-27B/chat_template.jinja` and to the copy under
# `<VOCABS_DIR>/Qwen3.8/`. It lives with the model it belongs to, one level up;
# this test reaches it through the `..:chat_template` data dependency.
_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'testdata',
    'chat_template.jinja',
)

_BASH_TOOL_FC = {
    'name': 'google:bash',
    'description': (
        'A stateful non-interactive bash terminal without access to the'
        ' internet.'
    ),
    'parameters': {
        'type': 'OBJECT',
        'properties': {
            'command': {
                'type': 'STRING',
                'description': 'The command to execute.',
            },
        },
        'required': ['command'],
    },
    # FC2.0-only; must not reach the rendered tool JSON.
    'response': {
        'type': 'OBJECT',
        'properties': {
            'stdout': {'type': 'STRING'},
            'stderr': {'type': 'STRING'},
        },
    },
}

_BASH_TOOL_JINJA = {
    'type': 'function',
    'function': {
        'name': 'google:bash',
        'description': _BASH_TOOL_FC['description'],
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The command to execute.',
                },
            },
            'required': ['command'],
        },
    },
}

# The same declaration in the OpenAI envelope a HuggingFace caller copies, with
# the JSON-schema types upper case as FC2.0 writes them: the rendered tool JSON
# must still be `_BASH_TOOL_JINJA`.
_BASH_TOOL_OPENAI_UPPER = {
    'type': 'function',
    'function': {
        'name': 'google:bash',
        'description': _BASH_TOOL_FC['description'],
        'parameters': {
            'type': 'OBJECT',
            'properties': {
                'command': {
                    'type': 'STRING',
                    'description': 'The command to execute.',
                },
            },
            'required': ['command'],
        },
    },
}


# A declaration whose description needs more than ASCII, and the tool JSON it
# must render as.
_SEARCH_TOOL_FC = {
    'name': 'search',
    'description': 'Recherche de donn\u00e9es \u00e9tendues \u2713',
    'parameters': {
        'type': 'OBJECT',
        'properties': {'query': {'type': 'OBJECT'}},
    },
}

_SEARCH_TOOL_JINJA = {
    'type': 'function',
    'function': {
        'name': 'search',
        'description': _SEARCH_TOOL_FC['description'],
        'parameters': {
            'type': 'object',
            'properties': {'query': {'type': 'object'}},
        },
    },
}


def _tool_result_messages(
    stdout: str = 'def f():\n    return "x"\n', stderr: str = 'boom'
) -> list[dict[str, Any]]:
  """An agent-loop turn: one bash call plus its response."""
  return [
      {
          'role': 'developer',
          'content': [{'function_declaration': _BASH_TOOL_FC}],
      },
      {'role': 'user', 'content': 'Show me f.'},
      {
          'role': 'assistant',
          'content': [
              {
                  'function_call': {
                      'name': 'google:bash',
                      'args': {'command': 'cat f.py'},
                  }
              },
              {
                  'function_response': {
                      'name': 'google:bash',
                      'response': {'stdout': stdout, 'stderr': stderr},
                  }
              },
          ],
      },
  ]


def _render_reference(messages, **kwargs) -> str:
  """Renders the released template the way `transformers` renders it."""

  def raise_exception(message):
    raise RuntimeError(message)

  def tojson(
      value, ensure_ascii=False, indent=None, separators=None, sort_keys=False
  ):
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )

  env = jinja2_sandbox.ImmutableSandboxedEnvironment(
      trim_blocks=True,
      lstrip_blocks=True,
      extensions=[jinja2_ext.loopcontrols],
  )
  cast(dict[str, Any], env.filters)['tojson'] = tojson
  cast(dict[str, Any], env.globals)['raise_exception'] = raise_exception
  with open(_TEMPLATE_PATH) as f:
    template = env.from_string(f.read())
  return template.render(
      messages=messages, add_generation_prompt=True, **kwargs
  )


class GoldenPromptTest(parameterized.TestCase):
  """`format()` vs the released jinja template, character for character."""

  @parameterized.named_parameters(
      dict(
          testcase_name='single_turn_xhigh',
          chat=lm_format.Qwen38Chat(),
          messages=[{'role': 'user', 'content': 'What is 2+2?'}],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='single_turn_medium',
          chat=lm_format.Qwen38ChatMedium(),
          messages=[{'role': 'user', 'content': 'What is 2+2?'}],
          jinja_kwargs={'reasoning_effort': 'medium'},
      ),
      dict(
          testcase_name='single_turn_low',
          chat=lm_format.Qwen38ChatLow(),
          messages=[{'role': 'user', 'content': 'What is 2+2?'}],
          jinja_kwargs={'reasoning_effort': 'low'},
      ),
      dict(
          testcase_name='single_turn_no_thinking',
          chat=lm_format.Qwen38ChatNoThink(),
          messages=[{'role': 'user', 'content': 'What is 2+2?'}],
          jinja_kwargs={'enable_thinking': False},
      ),
      dict(
          testcase_name='system_prompt',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'system', 'content': 'You are a helpful assistant.'},
              {'role': 'user', 'content': 'What is 2+2?'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='system_prompt_medium',
          chat=lm_format.Qwen38ChatMedium(),
          messages=[
              {'role': 'system', 'content': 'You are a helpful assistant.'},
              {'role': 'user', 'content': 'What is 2+2?'},
          ],
          jinja_kwargs={'reasoning_effort': 'medium'},
      ),
      dict(
          testcase_name='content_parts',
          chat=lm_format.Qwen38Chat(),
          messages=[{'role': 'user', 'content': [{'text': 'What is 2+2?'}]}],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='multi_turn_with_prior_thinking',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'user', 'content': 'hi'},
              {
                  'role': 'assistant',
                  'content': 'Hello!',
                  'reasoning_content': 'The user greets me.',
              },
              {'role': 'user', 'content': 'What is 2+2?'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='multi_turn_no_preserve_thinking',
          chat=lm_format.Qwen38Chat(preserve_thinking=False),
          messages=[
              {'role': 'user', 'content': 'hi'},
              {
                  'role': 'assistant',
                  'content': 'Hello!',
                  'reasoning_content': 'The user greets me.',
              },
              {'role': 'user', 'content': 'What is 2+2?'},
          ],
          jinja_kwargs={'preserve_thinking': False},
      ),
      dict(
          testcase_name='tool_declaration',
          chat=lm_format.Qwen38ChatFC(),
          messages=[
              {'role': 'system', 'content': 'You are a coding agent.'},
              {
                  'role': 'developer',
                  'content': [{'function_declaration': _BASH_TOOL_FC}],
              },
              {'role': 'user', 'content': 'List the files.'},
          ],
          jinja_kwargs={
              'tools': [_BASH_TOOL_JINJA],
              # The template has no `developer` role: declarations arrive as
              # the `tools` kwarg instead.
              'messages_override': [
                  {'role': 'system', 'content': 'You are a coding agent.'},
                  {'role': 'user', 'content': 'List the files.'},
              ],
          },
      ),
      dict(
          testcase_name='tool_declaration_openai_shape',
          # A declaration already in the OpenAI envelope keeps it (the bare
          # FC2.0 one above is wrapped), and its types are lowercased.
          chat=lm_format.Qwen38ChatFC(),
          messages=[
              {
                  'role': 'developer',
                  'content': [
                      {'function_declaration': _BASH_TOOL_OPENAI_UPPER}
                  ],
              },
              {'role': 'user', 'content': 'List the files.'},
          ],
          jinja_kwargs={
              'tools': [_BASH_TOOL_JINJA],
              'messages_override': [
                  {'role': 'user', 'content': 'List the files.'},
              ],
          },
      ),
      dict(
          testcase_name='tool_call_and_result',
          chat=lm_format.Qwen38ChatFC(),
          messages=[
              {
                  'role': 'developer',
                  'content': [{'function_declaration': _BASH_TOOL_FC}],
              },
              {'role': 'user', 'content': 'List the files.'},
              {
                  'role': 'assistant',
                  'content': [
                      {'text': 'Let me look.', 'channel': 'THOUGHT'},
                      {'text': 'Listing now.'},
                      {
                          'function_call': {
                              'name': 'google:bash',
                              'args': {'command': 'ls -l\ncat README'},
                          }
                      },
                      {
                          'function_response': {
                              'name': 'google:bash',
                              'response': {'stdout': 'a.py\n', 'stderr': ''},
                          }
                      },
                  ],
              },
          ],
          jinja_kwargs={
              'tools': [_BASH_TOOL_JINJA],
              'messages_override': [
                  {'role': 'user', 'content': 'List the files.'},
                  {
                      'role': 'assistant',
                      'content': 'Listing now.',
                      'reasoning_content': 'Let me look.',
                      'tool_calls': [{
                          'type': 'function',
                          'function': {
                              'name': 'google:bash',
                              'arguments': {'command': 'ls -l\ncat README'},
                          },
                      }],
                  },
                  # `function_response` parts become `tool` turns carrying the
                  # tool's raw text (see `_response_text`).
                  {'role': 'tool', 'content': 'a.py'},
              ],
          },
      ),
      dict(
          testcase_name='agentic_two_rounds',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'system', 'content': 'sys'},
              {'role': 'user', 'content': 'go'},
              {
                  'role': 'assistant',
                  'content': '',
                  'reasoning_content': 'r1',
                  'tool_calls': [
                      {'name': 'google:bash', 'arguments': {'command': 'ls'}}
                  ],
              },
              {'role': 'tool', 'content': 'a.py'},
              {
                  'role': 'assistant',
                  'content': 'done',
                  'reasoning_content': 'r2',
                  'tool_calls': [
                      {'name': 'google:bash', 'arguments': {'command': 'pwd'}}
                  ],
              },
              {'role': 'tool', 'content': '/w'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='unicode_and_trimming',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'system', 'content': '  systeme \u2713  '},
              {'role': 'user', 'content': '  quoi ?  \n'},
              {
                  'role': 'assistant',
                  'content': ' reponse ',
                  'reasoning_content': '  pensee  ',
              },
              {'role': 'user', 'content': 'encore'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='non_string_tool_call_args',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'user', 'content': 'hi'},
              {
                  'role': 'assistant',
                  'content': '',
                  'tool_calls': [{
                      'name': 'f',
                      'arguments': {
                          'n': 3,
                          'b': True,
                          'o': {'k': [1, 'x']},
                          'z': None,
                      },
                  }],
              },
              {'role': 'tool', 'content': 'ok'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='consecutive_assistant_turns',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'user', 'content': 'hi'},
              {'role': 'assistant', 'content': 'a1', 'reasoning_content': 'r1'},
              {'role': 'assistant', 'content': 'a2', 'reasoning_content': 'r2'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='empty_leading_system_then_tool',
          # The empty system message renders nothing, but it still occupies a
          # message slot, so the `tool` turn must open a `<|im_start|>user`.
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'system', 'content': ''},
              {'role': 'tool', 'content': 'x'},
              {'role': 'user', 'content': 'q'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='whitespace_leading_system_then_tool',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'system', 'content': '   '},
              {'role': 'tool', 'content': 'x'},
              {'role': 'user', 'content': 'q'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='none_leading_system_then_tool',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'system', 'content': None},
              {'role': 'tool', 'content': 'x'},
              {'role': 'user', 'content': 'q'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='tool_message_first',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'tool', 'content': 'x'},
              {'role': 'user', 'content': 'hi'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='openai_string_arguments',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'user', 'content': 'hi'},
              {
                  'role': 'assistant',
                  'content': '',
                  'tool_calls': [{
                      'type': 'function',
                      'function': {
                          'name': 'f',
                          'arguments': '{"command": "ls"}',
                      },
                  }],
              },
              {'role': 'tool', 'content': 'ok'},
          ],
          jinja_kwargs={
              # The template cannot parse a JSON string, so the reference gets
              # the decoded dict.
              'messages_override': [
                  {'role': 'user', 'content': 'hi'},
                  {
                      'role': 'assistant',
                      'content': '',
                      'tool_calls': [{
                          'type': 'function',
                          'function': {
                              'name': 'f',
                              'arguments': {'command': 'ls'},
                          },
                      }],
                  },
                  {'role': 'tool', 'content': 'ok'},
              ]
          },
      ),
      dict(
          testcase_name='two_tool_calls_no_text',
          chat=lm_format.Qwen38Chat(),
          messages=[
              {'role': 'user', 'content': 'Do two things.'},
              {
                  'role': 'assistant',
                  'content': '',
                  'reasoning_content': 'Two calls.',
                  'tool_calls': [
                      {'name': 'google:bash', 'arguments': {'command': 'ls'}},
                      {
                          'name': 'google:bash',
                          'arguments': {'command': 'pwd', 'timeout': 30},
                      },
                  ],
              },
              {'role': 'tool', 'content': 'a.py'},
              {'role': 'tool', 'content': '/workspace'},
          ],
          jinja_kwargs={},
      ),
      dict(
          testcase_name='tool_response_prefixed_user_query',
          # A user turn that OPENS with `<tool_response>` but goes on to ask
          # something is a real query, so it -- not the earlier one -- is the
          # last query, and the assistant turn before it loses its thinking.
          chat=lm_format.Qwen38Chat(preserve_thinking=False),
          messages=[
              {'role': 'user', 'content': 'u0'},
              {'role': 'assistant', 'content': 'a0', 'reasoning_content': 'r0'},
              {
                  'role': 'user',
                  'content': (
                      '<tool_response>\nout\n</tool_response>\nand also: why?'
                  ),
              },
              {'role': 'assistant', 'content': 'a1', 'reasoning_content': 'r1'},
          ],
          jinja_kwargs={'preserve_thinking': False},
      ),
      dict(
          testcase_name='non_ascii_in_tool_json_and_arguments',
          # `tojson` renders unicode raw (`ensure_ascii=False`), in the tool
          # declarations and in a structured tool-call argument alike.
          chat=lm_format.Qwen38ChatFC(),
          messages=[
              {
                  'role': 'developer',
                  'content': [{'function_declaration': _SEARCH_TOOL_FC}],
              },
              {'role': 'user', 'content': 'cherche'},
              {
                  'role': 'assistant',
                  'content': '',
                  'reasoning_content': 'r',
                  'tool_calls': [{
                      'name': 'search',
                      'arguments': {'query': {'lang': 'fran\u00e7ais \u2713'}},
                  }],
              },
              {'role': 'tool', 'content': 'ok'},
          ],
          jinja_kwargs={
              'tools': [_SEARCH_TOOL_JINJA],
              'messages_override': [
                  {'role': 'user', 'content': 'cherche'},
                  {
                      'role': 'assistant',
                      'content': '',
                      'reasoning_content': 'r',
                      'tool_calls': [{
                          'name': 'search',
                          'arguments': {
                              'query': {'lang': 'fran\u00e7ais \u2713'}
                          },
                      }],
                  },
                  {'role': 'tool', 'content': 'ok'},
              ],
          },
      ),
      dict(
          testcase_name='system_and_developer_prose',
          # Qwen has no `developer` role, so its prose folds into the system
          # turn -- as its own paragraph, not glued to the system prompt.
          chat=lm_format.Qwen38ChatFC(),
          messages=[
              {'role': 'system', 'content': 'You are a coding agent.'},
              {'role': 'developer', 'content': [{'text': 'Work in /tmp.'}]},
              {'role': 'user', 'content': 'go'},
          ],
          jinja_kwargs={
              'messages_override': [
                  {
                      'role': 'system',
                      'content': 'You are a coding agent.\n\nWork in /tmp.',
                  },
                  {'role': 'user', 'content': 'go'},
              ],
          },
      ),
  )
  def test_matches_released_template(self, chat, messages, jinja_kwargs):
    jinja_kwargs = dict(jinja_kwargs)
    reference_messages = jinja_kwargs.pop('messages_override', messages)
    expected = _render_reference(reference_messages, **jinja_kwargs)
    self.assertEqual(chat.format(messages), expected)

  def test_tool_response_body_is_raw_text(self):
    prompt = lm_format.Qwen38ChatFC().format(_tool_result_messages())
    self.assertIn(
        '<|im_start|>user\n<tool_response>\ndef f():\n    return "x"\n'
        '[stderr]\nboom\n</tool_response><|im_end|>\n',
        prompt,
    )

  def test_string_tool_response_is_trimmed(self):
    # A tool that returns bare text (not a `{stdout, stderr}` payload): the
    # template trims every turn's content, so the body must not carry the
    # tool's trailing newline into the prompt.
    prompt = lm_format.Qwen38ChatFC().format([
        {'role': 'user', 'content': 'go'},
        {
            'role': 'assistant',
            'content': [
                {'function_response': {'name': 'b', 'response': '\n a.py \n\n'}}
            ],
        },
    ])
    self.assertIn('<tool_response>\na.py\n</tool_response>', prompt)

  def test_empty_tool_response(self):
    prompt = lm_format.Qwen38ChatFC().format(
        _tool_result_messages(stdout='', stderr='')
    )
    self.assertIn('<tool_response>\n(no output)\n</tool_response>', prompt)

  def test_simple_thinking_turn_is_exactly_the_documented_prompt(self):
    formatted = lm_format.Qwen38Chat().format(
        [{'role': 'user', 'content': 'What is 2+2?'}]
    )
    self.assertEqual(
        formatted,
        '<|im_start|>system\nReasoning effort is set to xhigh. Please think'
        ' carefully through the task, validate key assumptions, consider'
        ' plausible alternatives, and prioritize correctness, consistency, and'
        ' clarity in the final answer.<|im_end|>\n<|im_start|>user\nWhat is'
        ' 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n',
    )


class RegistrationTest(absltest.TestCase):

  def test_registered_names(self):
    for name in (
        'Qwen38Chat',
        'Qwen38ChatMedium',
        'Qwen38ChatLow',
        'Qwen38ChatNoThink',
        'Qwen38ChatFC',
        'Qwen38ChatFCMedium',
        'Qwen38ChatFCLow',
    ):
      self.assertIsInstance(
          lm_format_lib.LMFormatRegistry.get_instance(name),
          lm_format.Qwen38Chat,
      )

  def test_parse_strips_every_eos_token(self):
    # Stops are data, not code: `generation_config.json` stops on `<|im_end|>`
    # and `<|endoftext|>`, and `<|im_start|>` catches a model that runs on into
    # the next turn. Whichever one decoding kept must not reach the agent loop.
    chat = lm_format.Qwen38ChatFC()
    self.assertEqual(
        chat.extra_eos_tokens,
        ('<|im_start|>', '<|im_end|>', '<|endoftext|>'),
    )
    for token in chat.extra_eos_tokens:
      with self.subTest(token):
        [message] = chat.parse(
            f'{chat.assistant_marker}</think>\n\nThe answer is 4.{token}\n'
        )
        self.assertEqual(message['content'], [{'text': 'The answer is 4.'}])

  def test_parse_strips_a_stop_token_run(self):
    # Two stops and trailing whitespace: a leaked `<|im_end|>` would be read
    # back as part of the model's answer.
    chat = lm_format.Qwen38ChatFC()
    [message] = chat.parse(
        f'{chat.assistant_marker}</think>\n\nDone.<|im_end|> <|endoftext|>\n\n'
    )
    self.assertEqual(message['content'], [{'text': 'Done.'}])

  def test_format_tokens_refuses_instead_of_skewing(self):
    # The inherited implementation would emit a template-less conversation (no
    # reasoning system turn, no `<think>`, tool calls dropped).
    with self.assertRaises(NotImplementedError):
      lm_format.Qwen38Chat().format_tokens(
          [{'role': 'user', 'content': 'hi'}], tokenizer=None
      )

  def test_unknown_reasoning_effort_raises(self):
    with self.assertRaises(ValueError):
      lm_format.Qwen38Chat(reasoning_effort='ultra').format(
          [{'role': 'user', 'content': 'hi'}]
      )

  def test_second_system_message_raises(self):
    # The template raises `System message must be at the beginning.` for any
    # system message that is not `loop.first`.
    with self.assertRaises(ValueError):
      lm_format.Qwen38Chat().format([
          {'role': 'system', 'content': 'a'},
          {'role': 'system', 'content': 'b'},
          {'role': 'user', 'content': 'q'},
      ])

  def test_empty_late_system_message_raises(self):
    with self.assertRaises(ValueError):
      lm_format.Qwen38Chat().format([
          {'role': 'user', 'content': 'hi'},
          {'role': 'system', 'content': ''},
      ])

  def test_late_system_message_raises(self):
    with self.assertRaises(ValueError):
      lm_format.Qwen38Chat().format([
          {'role': 'user', 'content': 'hi'},
          {'role': 'system', 'content': 'too late'},
      ])

  def test_no_user_query_raises(self):
    with self.assertRaises(ValueError):
      lm_format.Qwen38Chat().format([{'role': 'system', 'content': 'hi'}])


class ParseTest(absltest.TestCase):
  """`parse()` on what the server actually hands it."""

  def setUp(self):
    super().setUp()
    self.lm_format = lm_format.Qwen38ChatFC()

  def _parse_continuation(self, continuation: str, **kwargs) -> Any:
    """Mirrors serving/page_batcher.py: assistant_marker + output_text."""
    return self.lm_format.parse(
        self.lm_format.assistant_marker + continuation, **kwargs
    )

  def test_parses_a_whole_conversation(self):
    # `parse()` is the inverse of `format()`, not just of the assistant turn;
    # the trailing generation prompt contributes no message.
    chat = lm_format.Qwen38Chat()
    parsed = chat.parse(chat.format([{'role': 'user', 'content': 'hi'}]))
    self.assertEqual(
        [message['role'] for message in parsed], ['system', 'user']
    )
    self.assertStartsWith(parsed[0]['content'], 'Reasoning effort is set to')
    self.assertEqual(parsed[1]['content'], 'hi')

  def test_thought_text_and_tool_call(self):
    continuation = (
        'I should list the files.\n</think>\n\nListing now.\n\n<tool_call>\n'
        '<function=google:bash>\n<parameter=command>\nls -l\ncat README\n'
        '</parameter>\n</function>\n</tool_call><|im_end|>\n'
    )
    [message] = self._parse_continuation(continuation)
    self.assertEqual(message['role'], 'assistant')
    self.assertEqual(
        message['content'],
        [
            {'text': 'I should list the files.', 'channel': 'THOUGHT'},
            {'text': 'Listing now.'},
            {
                'function_call': {
                    'name': 'google:bash',
                    'args': {'command': 'ls -l\ncat README'},
                }
            },
        ],
    )

  def test_no_tool_call_is_text(self):
    [message] = self._parse_continuation(
        'Thinking...\n</think>\n\nThe answer is 4.<|im_end|>'
    )
    self.assertEqual(
        message['content'],
        [
            {'text': 'Thinking...', 'channel': 'THOUGHT'},
            {'text': 'The answer is 4.'},
        ],
    )

  def test_truncated_thought_is_all_thought(self):
    [message] = self._parse_continuation('Still reasoning about it')
    self.assertEqual(
        message['content'],
        [{'text': 'Still reasoning about it', 'channel': 'THOUGHT'}],
    )

  def test_ended_turn_with_no_close_think_keeps_its_tool_call(self):
    # The model ended the turn (`<|im_end|>`) without ever closing `<think>`,
    # so `truncated` is False upstream and nothing can recover a dropped call:
    # everything after an unopened thought block is content.
    [message] = self._parse_continuation(
        'I will run it.\n<tool_call>\n<function=t>\n<parameter=command>\nls\n'
        '</parameter>\n</function>\n</tool_call><|im_end|>\n'
    )
    self.assertEqual(
        message['content'],
        [
            {'text': 'I will run it.'},
            {'function_call': {'name': 't', 'args': {'command': 'ls'}}},
        ],
    )

  def test_unended_turn_with_no_close_think_is_still_all_thought(self):
    # Same text without the stop token: the model was cut off mid-thought, and
    # the half-written call must NOT be replayed as one.
    [message] = self._parse_continuation(
        'I will run it.\n<tool_call>\n<function=t>\n<parameter=command>\nls'
    )
    self.assertEqual(
        [part.get('channel') for part in message['content']], ['THOUGHT']
    )

  def test_two_calls_and_arg_coercion(self):
    continuation = (
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\nls\n'
        '</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=t>\n'
        '<parameter=command>\npwd\n</parameter>\n<parameter=timeout>\n30\n'
        '</parameter>\n</function>\n</tool_call><|im_end|>\n'
    )
    declaration = {
        'name': 't',
        'parameters': {
            'type': 'OBJECT',
            'properties': {
                'command': {'type': 'STRING'},
                'timeout': {'type': 'INTEGER'},
            },
        },
    }
    [message] = self._parse_continuation(
        continuation, function_declarations=[declaration]
    )
    self.assertEqual(
        message['content'],
        [
            {'function_call': {'name': 't', 'args': {'command': 'ls'}}},
            {
                'function_call': {
                    'name': 't',
                    'args': {'command': 'pwd', 'timeout': 30},
                }
            },
        ],
    )

  def test_declarations_as_a_keyed_mapping(self):
    # The shape the BYOM serving path hands the hook (it collects the inbound
    # request's declarations into `{name: declaration}`).
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=timeout>\n30\n'
        '</parameter>\n</function>\n</tool_call>',
        function_declarations={
            't': {
                'name': 't',
                'parameters': {'properties': {'timeout': {'type': 'INTEGER'}}},
            }
        },
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'], {'timeout': 30}
    )

  def test_untyped_args_stay_strings(self):
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=timeout>\n30\n'
        '</parameter>\n</function>\n</tool_call>'
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'], {'timeout': '30'}
    )

  def test_value_containing_the_closing_tag(self):
    # A non-greedy match would hand the loop a truncated-but-runnable command.
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\n'
        'grep -n "</parameter>" x.py\n</parameter>\n</function>\n</tool_call>'
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'],
        {'command': 'grep -n "</parameter>" x.py'},
    )

  def test_value_whitespace_is_preserved(self):
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\n'
        "cat <<'EOF' > a.py\n    indented\n\nEOF\n</parameter>\n</function>\n"
        '</tool_call>'
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'],
        {'command': "cat <<'EOF' > a.py\n    indented\n\nEOF"},
    )

  def test_value_whitespace_is_verbatim_at_both_ends(self):
    # Leading blank lines and trailing spaces are part of the file the agent is
    # writing; only the `\n` before `</parameter>` belongs to the markup.
    value = "cat <<'EOF' > a.py\n\n    def f():\n        return 1\n\nEOF  "
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\n'
        f'{value}\n</parameter>\n</function>\n</tool_call>'
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'], {'command': value}
    )

  def test_function_markup_inside_a_value_is_not_a_call(self):
    # A scan that restarted from the top of the block would truncate the real
    # command at `echo '` and append a call to a tool named `evil`.
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\n'
        "echo '<function=evil>' > x\n</parameter>\n</function>\n</tool_call>"
    )
    self.assertEqual(
        message['content'],
        [{
            'function_call': {
                'name': 't',
                'args': {'command': "echo '<function=evil>' > x"},
            }
        }],
    )

  def test_tool_call_markup_inside_a_value_does_not_end_the_block(self):
    # A heredoc documenting this very format: the block ends at the LAST
    # `</tool_call>`, not at the one inside the value.
    value = (
        "cat <<'EOF' > doc.md\n<tool_call>\n<function=x>\n</function>\n"
        '</tool_call>\nEOF'
    )
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\n'
        f'{value}\n</parameter>\n</function>\n</tool_call>'
    )
    self.assertEqual(
        message['content'],
        [{'function_call': {'name': 't', 'args': {'command': value}}}],
    )

  def test_close_parameter_followed_by_a_parameter_is_unrecoverable(self):
    # The one case the wire format cannot express: inside a value,
    # `</parameter>` immediately followed by `<parameter=` is byte-identical to
    # the end of that value, so the split below is the only reading available
    # (the released reference parser splits it the same way). Pinned so the
    # limitation is visible rather than discovered in a sandbox.
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\n'
        "cat <<'EOF' > doc.md\nUse </parameter>\n<parameter=next> to continue\n"
        'EOF\n</parameter>\n</function>\n</tool_call>'
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'],
        {
            'command': "cat <<'EOF' > doc.md\nUse ",
            'next': ' to continue\nEOF',
        },
    )

  def test_inline_parameter_without_newlines(self):
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call><function=t><parameter=command>ls</parameter>'
        '</function></tool_call>'
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'], {'command': 'ls'}
    )

  def test_unterminated_tool_call_still_parses(self):
    # Dropping it would look like 'no tool calls' and end the episode.
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\nls\n'
        '</parameter>'
    )
    self.assertEqual(
        message['content'][0]['function_call'],
        {'name': 't', 'args': {'command': 'ls'}},
    )

  def test_end_of_text_is_stripped(self):
    [message] = self._parse_continuation(
        '</think>\n\nThe answer is 4.<|endoftext|>'
    )
    self.assertEqual(message['content'], [{'text': 'The answer is 4.'}])

  def test_no_think_format_keeps_tool_calls(self):
    # `Qwen38ChatNoThink`'s prompt closes the think block itself, so the
    # continuation is content, not thinking (contrast
    # `test_truncated_thought_is_all_thought`). Classifying it as THOUGHT would
    # hand the agent loop an empty turn and end the episode.
    chat = lm_format.Qwen38ChatNoThink()
    [message] = chat.parse(
        chat.assistant_marker
        + 'Listing.\n\n<tool_call>\n<function=t>\n'
        '<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>'
        '<|im_end|>\n'
    )
    self.assertEqual(
        message['content'],
        [
            {'text': 'Listing.'},
            {'function_call': {'name': 't', 'args': {'command': 'ls'}}},
        ],
    )

  def test_parameter_markup_inside_a_value(self):
    # A SWE agent writing a file about this very format.
    value = '<parameter=x>\nnot markup\n</parameter>'
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\n'
        f"cat <<'EOF' > doc.md\n{value}\nEOF\n</parameter>\n</function>\n"
        '</tool_call>'
    )
    self.assertEqual(
        message['content'][0]['function_call']['args'],
        {'command': f"cat <<'EOF' > doc.md\n{value}\nEOF"},
    )

  def test_two_functions_in_one_tool_call_block(self):
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n<function=t>\n<parameter=command>\nls\n'
        '</parameter>\n</function>\n<function=t>\n<parameter=command>\npwd\n'
        '</parameter>\n</function>\n</tool_call>'
    )
    self.assertEqual(
        [part['function_call'] for part in message['content']],
        [
            {'name': 't', 'args': {'command': 'ls'}},
            {'name': 't', 'args': {'command': 'pwd'}},
        ],
    )

  def test_unparsable_tool_call_survives_as_text(self):
    # Dropping it would leave `output_messages` empty, which the agent loop
    # reads as 'the model is done'.
    [message] = self._parse_continuation(
        '</think>\n\nHere goes.\n\n<tool_call>\nnot a function at all\n'
        '</tool_call>'
    )
    self.assertEqual(
        message['content'],
        [
            {'text': 'Here goes.'},
            {'text': '<tool_call>\nnot a function at all\n</tool_call>'},
        ],
    )

  def test_json_style_tool_call(self):
    [message] = self._parse_continuation(
        '</think>\n\n<tool_call>\n{"name": "t", "arguments": {"command":'
        ' "ls"}}\n</tool_call>'
    )
    self.assertEqual(
        message['content'],
        [{'function_call': {'name': 't', 'args': {'command': 'ls'}}}],
    )


class RoundTripTest(absltest.TestCase):
  """format(agent messages) -> model text -> parse() -> the same parts."""

  def test_agent_loop_round_trip(self):
    chat = lm_format.Qwen38ChatFC()
    parts = [
        {'text': 'I will list the files.', 'channel': 'THOUGHT'},
        {
            'function_call': {
                'name': 'google:bash',
                'args': {'command': 'ls -l'},
            }
        },
    ]
    messages = [
        {'role': 'system', 'content': 'You are a coding agent.'},
        {
            'role': 'developer',
            'content': [{'function_declaration': _BASH_TOOL_FC}],
        },
        {'role': 'user', 'content': 'List the files.'},
    ]
    prompt = chat.format(messages)
    self.assertTrue(prompt.endswith('<|im_start|>assistant\n<think>\n'))

    # What the model emits after the generation prompt.
    continuation = (
        'I will list the files.\n</think>\n\n<tool_call>\n'
        '<function=google:bash>\n<parameter=command>\nls -l\n</parameter>\n'
        '</function>\n</tool_call><|im_end|>\n'
    )
    [message] = chat.parse(
        chat.assistant_marker + continuation,
        function_declarations=[_BASH_TOOL_FC],
    )
    self.assertEqual(message['content'], parts)

    # Feeding the parsed turn (plus the tool result) back in reproduces the
    # same prefix, i.e. format() and parse() agree on the wire format.
    tool_result = {
        'function_response': {
            'name': 'google:bash',
            'response': {'stdout': 'a.py\n', 'stderr': ''},
        }
    }
    next_prompt = chat.format(
        messages
        + [{
            'role': 'assistant',
            'content': list(message['content']) + [tool_result],
        }]
    )
    self.assertEqual(
        next_prompt[: len(prompt) - len('<|im_start|>assistant\n<think>\n')],
        prompt[: -len('<|im_start|>assistant\n<think>\n')],
    )
    self.assertIn(
        '<|im_start|>assistant\n<think>\nI will list the files.\n</think>\n\n'
        '<tool_call>\n<function=google:bash>\n<parameter=command>\nls -l\n'
        '</parameter>\n</function>\n</tool_call><|im_end|>\n'
        '<|im_start|>user\n<tool_response>\na.py\n</tool_response><|im_end|>\n',
        next_prompt,
    )


if __name__ == '__main__':
  absltest.main()

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
"""Tests for the GLM-5.2 chat/tool-call template (GlmChat)."""

import dataclasses

from absl.testing import absltest
from simply.zoo.glm5.utils import lm_format as glm5_lm_format


class GlmChatTest(absltest.TestCase):

  def test_glm_chat_tool_call_replay_roundtrip(self):
    """parse() -> format() must reproduce the model's GLM <tool_call> text.

    This is the S2-replay path: a recorded assistant turn (with a
    function_call part) is re-rendered into the conversation history. If the
    function_call part rendered empty, the replayed turn would collapse to a
    bare <|assistant|> marker and degenerate the model. The round-trip must
    preserve both visible text and the tool call.
    """
    lm_format = glm5_lm_format.GlmChat()
    # A model assistant turn: reasoning + visible text + a tool call.
    model_turn = (
        '<think>let me list files</think>'
        "I'll inspect the repo."
        '<tool_call>execute_bash<arg_key>command</arg_key>'
        '<arg_value>ls -la</arg_value></tool_call>'
    )
    # 1. parse() builds output_messages (text + function_call parts).
    parsed = lm_format.parse(model_turn)
    self.assertEqual(parsed[0]['role'], 'assistant')
    content = parsed[0]['content']
    self.assertTrue(any('function_call' in p for p in content))
    fc = next(p['function_call'] for p in content if 'function_call' in p)
    self.assertEqual(fc['name'], 'execute_bash')
    self.assertEqual(fc['args'], {'command': 'ls -la'})
    # 2. _visible_text re-renders the function_call part back to GLM text
    #    (NOT empty) -- the core replay-degeneration fix.
    rendered = lm_format._visible_text(content)
    self.assertIn("I'll inspect the repo.", rendered)
    self.assertIn(
        '<tool_call>execute_bash<arg_key>command</arg_key>'
        '<arg_value>ls -la</arg_value></tool_call>',
        rendered,
    )
    # 3. Full format() of the replayed assistant turn carries the tool call
    #    (so the replayed history is faithful, not a bare marker).
    formatted = lm_format.format([dict(role='user', content='go'), parsed[0]])
    self.assertIn('<|assistant|>', formatted)
    self.assertIn('<tool_call>execute_bash', formatted)
    self.assertNotIn('<|assistant|><|assistant|>', formatted)

  def test_glm_chat_render_function_call_structured_args(self):
    """Structured (non-string) args round-trip via JSON in <arg_value>."""
    lm_format = glm5_lm_format.GlmChat()
    fc = {'name': 'finish', 'args': {'ok': True, 'items': [1, 2]}}
    rendered = lm_format._render_function_call(fc)
    self.assertEqual(
        rendered,
        '<tool_call>finish<arg_key>ok</arg_key><arg_value>true</arg_value>'
        '<arg_key>items</arg_key><arg_value>[1, 2]</arg_value></tool_call>',
    )
    # And the parse() of this rendering recovers the structured args.
    reparsed = lm_format._parse_tool_call(
        rendered.removeprefix('<tool_call>').removesuffix('</tool_call>')
    )
    self.assertEqual(reparsed['name'], 'finish')
    self.assertEqual(reparsed['args'], {'ok': True, 'items': [1, 2]})

  def test_glm_chat_function_response_rendered_as_observation(self):
    """Tool RESULTS must reach the model under <|observation|> (the core bug).

    The agentic in-sandbox loop appends the tool result (function_response
    part, carrying stdout/exit_code) to the SAME assistant message that holds
    the model's function_call. If GlmChat.format dropped the function_response
    (no 'text' key), the model would never see its command output and would
    loop reporting "empty output". This test asserts the stdout reaches the
    rendered prompt under the GLM observation marker.
    """
    lm_format = glm5_lm_format.GlmChat()
    # An assistant turn with a tool call AND its result (as the loop builds it).
    assistant_msg = {
        'role': 'assistant',
        'content': [
            {'text': 'Let me list the repo.'},
            {
                'function_call': {
                    'name': 'execute_bash',
                    'args': {'command': 'ls /app'},
                }
            },
            {
                'function_response': {
                    'name': 'execute_bash',
                    'response': {
                        'exit_code': '0',
                        'stdout': 'README.md\nsrc\ntests',
                    },
                }
            },
        ],
    }
    formatted = lm_format.format(
        [dict(role='user', content='solve it'), assistant_msg]
    )
    # Assistant text + tool call under <|assistant|>.
    self.assertIn('<|assistant|>Let me list the repo.', formatted)
    self.assertIn(
        '<tool_call>execute_bash<arg_key>command</arg_key>'
        '<arg_value>ls /app</arg_value></tool_call>',
        formatted,
    )
    # CRUCIAL: the command output reaches the model under <|observation|>.
    self.assertIn('<|observation|>', formatted)
    self.assertIn('README.md\nsrc\ntests', formatted)
    self.assertIn('exit_code: 0', formatted)
    # The observation comes AFTER the assistant tool call.
    self.assertLess(
        formatted.index('<tool_call>execute_bash'),
        formatted.index('<|observation|>'),
    )

  def test_glm_chat_render_function_response_payload(self):
    """_render_function_response renders each field on its own line."""
    lm_format = glm5_lm_format.GlmChat()
    rendered = lm_format._render_function_response({
        'name': 'execute_bash',
        'response': {'exit_code': '0', 'stdout': 'hello'},
    })
    self.assertIn('execute_bash', rendered)
    self.assertIn('exit_code: 0', rendered)
    self.assertIn('stdout: hello', rendered)
    self.assertIn('<tool_response>', rendered)
    self.assertIn('</tool_response>', rendered)

  def test_glm_chat_official_agentic_format(self):
    """Matches the GLM-5.2 official chat template's agentic fidelity points.

    (a) the rendered prompt STARTS with the reasoning-effort system directive
    `[gMASK]<sop><|system|>Reasoning Effort: Max`; (b) historical assistant
    reasoning (turns at/before the last user turn) is collapsed to
    `<think></think>` while later turns keep full reasoning; (c) the generation
    prompt ends with `<|assistant|><think>`.
    """
    lm_format = glm5_lm_format.GlmChat()
    messages = [
        dict(role='user', content='fix the bug'),
        dict(
            role='assistant',
            content='<think>long historical reasoning</think>I will explore.',
        ),
        dict(role='user', content='continue'),
    ]
    formatted = lm_format.format(messages)
    # (a) reasoning-effort directive first.
    self.assertTrue(
        formatted.startswith('[gMASK]<sop><|system|>Reasoning Effort: Max'),
        msg=formatted[:80],
    )
    # (b) historical reasoning collapsed (this assistant turn is before the
    # last user turn), visible text retained.
    self.assertIn('<think></think>I will explore.', formatted)
    self.assertNotIn('long historical reasoning', formatted)
    # (c) generation prompt opens a fresh think block.
    self.assertTrue(formatted.endswith('<|assistant|><think>'))

  def test_glm_chat_reasoning_effort_configurable(self):
    """`reasoning_effort` is configurable; empty disables the directive."""
    high = dataclasses.replace(
        glm5_lm_format.GlmChat(), reasoning_effort='high'
    )
    self.assertTrue(
        high.format([dict(role='user', content='hi')]).startswith(
            '[gMASK]<sop><|system|>Reasoning Effort: High'
        )
    )
    none = dataclasses.replace(glm5_lm_format.GlmChat(), reasoning_effort='')
    self.assertTrue(
        none.format([dict(role='user', content='hi')]).startswith(
            '[gMASK]<sop><|user|>'
        )
    )


if __name__ == '__main__':
  absltest.main()

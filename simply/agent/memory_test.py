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
import dataclasses
import textwrap

from absl.testing import absltest

from simply.agent import memory


class MemoryTest(absltest.TestCase):

  def test_llm_view(self):
    not_rand_string = '0329u342u493u29ujfiejwjf'
    max_token_budget = 128_000
    mem = memory.MemorySystem(
        task=not_rand_string,
        max_token_budget=max_token_budget,
    )
    self.assertIn(mem.memory_system_description, mem.llm_view)
    self.assertIn(not_rand_string, mem.llm_view)
    self.assertIn('<memory uri="pad://plan.md"', mem.llm_view)
    self.assertIn('<memory uri="pad://todo.md"', mem.llm_view)
    self.assertIn('<memory uri="pad://scratch.md"', mem.llm_view)
    self.assertIn('<memory uri="pad://journey.md"', mem.llm_view)
    # system status
    self.assertIn('"status_step": 0', mem.llm_view)
    self.assertIn(f'"max_token_budget": {max_token_budget}', mem.llm_view)

  def test_memory_uri_names(self):
    valid_uri_examples = [
        'kb://test.md',
        'pad://TODO.md',
        'pad://some file.md',
    ]
    for uri in valid_uri_examples:
      self.assertTrue(memory.is_valid_memory_uri(uri))

    invalid_uri_examples = [
        '/other_dir/test.md',
        'no folder file\n',
        'unknown://test.md',
    ]
    for uri in invalid_uri_examples:
      self.assertFalse(memory.is_valid_memory_uri(uri))

  def test_memory_file_to_llm(self):
    update_step = 9999
    memory_file = memory.MemoryFile(
        uri='kb://test',
        display=memory.DisplayMode.FULL,
        content='content',
        summary='summary',
        update_step=update_step,
    )
    self.assertEqual(
        memory_file.to_llm(),
        textwrap.dedent(f"""\
        <memory uri="kb://test" display="full" length="7" update_step="{update_step}">
        <summary>summary</summary>
        <content><![CDATA[
        content
        ]]></content>
        </memory>"""),
    )
    folded_file = dataclasses.replace(
        memory_file, display=memory.DisplayMode.SUMMARY
    )
    self.assertEqual(
        folded_file.to_llm(),
        textwrap.dedent(f"""\
        <memory uri="kb://test" display="summary" length="7" update_step="{update_step}">
        <summary>summary</summary>
        </memory>"""),
    )

  def test_write_knowledge(self):
    content = 'x' * 100
    summary_length = 40
    mem = memory.MemorySystem(
        task='test task',
        max_token_budget=128_000,
        default_summary_length=summary_length,
    )
    with mem.capture_snapshot():
      ret = mem.write(
          uri='kb://test',
          content=content,
      )
    self.assertEqual(ret, 'OK')
    self.assertLen(mem.memory_snapshots, 2)  # initial + the new one
    self.assertEqual(
        mem.memory_snapshots[-1].memory.files['kb://test'].content,
        content,
    )
    self.assertLessEqual(
        len(mem.memory_snapshots[-1].memory.files['kb://test'].summary),
        summary_length,
    )
    self.assertIn('<memory uri="kb://test"', mem.llm_view)
    self.assertLen(
        [
            uri
            for uri in mem.memory_snapshots[-1].memory.files
            if uri.startswith('kb://')
        ],
        1,
    )

    # delete the file
    with mem.capture_snapshot():
      ret = mem.delete(uri='kb://test')
    self.assertEqual(ret, 'OK')
    self.assertLen(mem.memory_snapshots, 3)
    self.assertNotIn('<memory uri="kb://test"', mem.llm_view)
    self.assertEmpty(
        [
            uri
            for uri in mem.memory_snapshots[-1].memory.files
            if uri.startswith('kb://')
        ],
    )

  def test_update_step(self):
    mem = memory.MemorySystem(task='test task', max_token_budget=128_000)
    uri = 'kb://test'
    with mem.capture_snapshot():
      mem.write(uri=uri, content='content')
    self.assertEqual(mem.memory_snapshots[-1].memory.files[uri].update_step, 1)
    with mem.capture_snapshot():
      mem.write(uri=uri, content='content')
    self.assertEqual(mem.memory_snapshots[-1].memory.files[uri].update_step, 2)

  def test_tool_call_error(self):
    mem = memory.MemorySystem(task='test task', max_token_budget=128_000)
    tools = memory.get_memory_tools(mem)
    write_tool = [t for t in tools if t.name == 'mem_write'][0]
    # tool call should raise error if not in a snapshot context
    self.assertRaises(
        AssertionError,
        write_tool.execute,
        args_json='{"uri": "kb://test", "content": "content"}',
    )
    with mem.capture_snapshot():
      action, result = write_tool.execute(
          args_json='{"uri": "kb://test", "content": "content"}'
      )
    self.assertIsNotNone(action)
    self.assertEqual(result, 'OK')
    self.assertEqual(
        mem.memory_snapshots[-1].memory.files['kb://test'].content,
        'content',
    )

  def test_compress_history(self):
    mem = memory.MemorySystem(task='test task', max_token_budget=128_000)
    n_steps = 8
    content_format = 'SomeDummyLLMContent-{}'
    compressed_content = 'DummyCompressedContent'
    for i in range(n_steps):
      with mem.capture_snapshot():
        mem.record_llm_output(content_format.format(i))
    self.assertLen(mem.memory_snapshots, n_steps + 1)
    for i in range(n_steps):
      self.assertIn(content_format.format(i), mem.llm_view)

    with mem.capture_snapshot():
      ret = mem.compress_history(
          start_step=3,
          end_step=n_steps - 1,
          content=compressed_content,
      )
    self.assertEqual(ret, 'OK')
    self.assertLen(mem.memory_snapshots, n_steps + 2)
    for i in range(2, n_steps - 1):
      self.assertNotIn(content_format.format(i), mem.llm_view)
    self.assertIn(compressed_content, mem.llm_view)

    with mem.capture_snapshot():
      ret = mem.compress_history(
          start_step=1,
          end_step=n_steps - 1,
          content=compressed_content[::-1],
      )
    self.assertEqual(ret, 'OK')
    self.assertLen(mem.memory_snapshots, n_steps + 3)
    self.assertNotIn(compressed_content, mem.llm_view)
    self.assertIn(compressed_content[::-1], mem.llm_view)


if __name__ == '__main__':
  absltest.main()

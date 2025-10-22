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

from absl.testing import absltest
from simply.utils import lm_format as lm_format_lib


class LmFormatTest(absltest.TestCase):

  def test_pretrain(self):
    lm_format = lm_format_lib.Pretrain()
    self.assertEqual(
        lm_format.format([dict(role='user', content='foo')]), 'foo'
    )
    with self.assertRaises(ValueError):
      lm_format.format(
          [dict(role='system', content='foo'), dict(role='user', content='bar')]
      )

  def test_simply_v1_chat(self):
    lm_format = lm_format_lib.SimplyV1Chat()
    self.assertEqual(
        lm_format.format([
            dict(role='system', content='foo'),
            dict(role='user', content='bar'),
            dict(role='assistant', content='baz'),
        ]),
        (
            '<reserved_3>foo<reserved_4><reserved_1>bar<reserved_4>'
            '<reserved_2>baz<reserved_4><reserved_2>'
        ),
    )
    self.assertEqual(
        lm_format.format([
            dict(role='user', content='bar'),
        ]),
        '<reserved_1>bar<reserved_4><reserved_2>',
    )

  def test_gemma_v2_chat(self):
    lm_format = lm_format_lib.GemmaV2Chat()
    self.assertEqual(
        lm_format.format([
            dict(role='user', content='foo'),
            dict(role='assistant', content='bar'),
        ]),
        (
            '<start_of_turn>user\nfoo<end_of_turn>\n'
            '<start_of_turn>model\nbar<end_of_turn>\n<start_of_turn>model\n'
        ),
    )
    self.assertEqual(
        lm_format.format([
            dict(role='user', content='bar'),
        ]),
        '<start_of_turn>user\nbar<end_of_turn>\n<start_of_turn>model\n',
    )

  def test_deep_seek_qwen_r1_distill_chat(self):
    lm_format = lm_format_lib.DeepSeekQwenR1DistillChat()
    self.assertEqual(
        lm_format.format([
            dict(role='user', content='foo'),
            dict(role='assistant', content='bar'),
        ]),
        '<｜User｜>foo<｜Assistant｜>bar<｜end▁of▁sentence｜>'
        '<｜Assistant｜><think>\n',
    )
    self.assertEqual(
        lm_format.format([
            dict(role='user', content='bar'),
        ]),
        '<｜User｜>bar<｜Assistant｜><think>\n',
    )

  def test_qwen_v2_chat(self):
    lm_format = lm_format_lib.QwenV2Chat()
    self.assertEqual(
        lm_format.format([
            dict(role='system', content='foo'),
            dict(role='user', content='bar'),
            dict(role='assistant', content='tar'),
        ]),
        (
            '<|im_start|>system\nfoo<|im_end|>\n'
            '<|im_start|>user\nbar<|im_end|>\n'
            '<|im_start|>assistant\ntar<|im_end|>\n'
            '<|im_start|>assistant\n'
        ),
    )
    self.assertEqual(
        lm_format.format([
            dict(role='user', content='bar'),
        ]),
        '<|im_start|>user\nbar<|im_end|>\n<|im_start|>assistant\n',
    )

  def test_qwq_chat(self):
    lm_format = lm_format_lib.QwQChat()
    self.assertEqual(
        lm_format.format([
            dict(role='system', content='foo'),
            dict(role='user', content='bar'),
        ]),
        (
            '<|im_start|>system\nfoo<|im_end|>\n'
            '<|im_start|>user\nbar<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n'
        ),
    )

  def test_gemini_chat(self):
    lm_format = lm_format_lib.GeminiChat()
    self.assertEqual(
        lm_format.format([
            dict(role='system', content='test'),
            dict(role='user', content='foo'),
            dict(role='assistant', content='bar'),
        ]),
        (
            '<ctrl99>system\ntest<ctrl100>\n'
            '<ctrl99>user\nfoo<ctrl100>\n'
            '<ctrl99>model\nbar<ctrl100>\n<ctrl99>model\n'
        ),
    )
    self.assertEqual(
        lm_format.format([
            dict(role='user', content='bar'),
        ]),
        (
            '<ctrl99>user\nbar<ctrl100>\n<ctrl99>model\n'
        ),
    )


if __name__ == '__main__':
  absltest.main()

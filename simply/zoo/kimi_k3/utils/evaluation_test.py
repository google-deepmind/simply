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
r"""Tests for the K3 multiple-choice scorer.

The strings below are verbatim from the GPQA-Diamond run on the released 2.8T
weights, so this test fails if the extractor stops handling
what the model actually writes.
"""

from absl.testing import absltest
from absl.testing import parameterized
from simply.utils import evaluation_lib
from simply.zoo.kimi_k3.utils import evaluation


class ExtractOptionLetterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # What K3 actually writes, 185 times out of 198.
      ('latex_text_with_paren', r'\boxed{\text{(B) }10^{-4}\ \text{eV}}', 'B'),
      ('bare_letter', r'\boxed{C}', 'C'),
      ('parenthesised', r'\boxed{(D)}', 'D'),
      ('letter_then_text', r'\boxed{A) the first option}', 'A'),
      ('textbf', r'\boxed{\textbf{B}}', 'B'),
      ('trailing_period', r'\boxed{B.}', 'B'),
      # The last box wins: the think channel is full of earlier ones.
      ('last_box_wins', r'first \boxed{A} then \boxed{D}', 'D'),
      ('nested_braces', r'\boxed{\text{(C) } \frac{1}{2}}', 'C'),
  )
  def test_reads_the_letter(self, response, expected):
    self.assertEqual(evaluation.extract_option_letter(response), expected)

  @parameterized.named_parameters(
      ('no_box', 'I think it is B, probably.'),
      # A value, not an option: refuse rather than guess.
      ('value_only', r'\boxed{138^\circ}'),
      ('empty', r'\boxed{}'),
  )
  def test_refuses_to_guess(self, response):
    self.assertIsNone(evaluation.extract_option_letter(response))

  def test_falls_back_to_matching_the_option_text(self):
    options = ('10^-11 eV', '10^-4 eV', '10^-8 eV', '10^-9 eV')
    self.assertEqual(
        evaluation.extract_option_letter(r'\boxed{10^-4 eV}', options), 'B'
    )

  def test_ambiguous_option_text_is_refused(self):
    self.assertIsNone(
        evaluation.extract_option_letter(r'\boxed{eV}', ('eV', 'eV', 'x', 'y'))
    )


class KimiK3GPQADiamondTest(absltest.TestCase):

  EXAMPLE = {
      'question': 'Two quantum states ...',
      'correct_answer': '10^-4 eV',
      'incorrect_answer_1': '10^-11 eV',
      'incorrect_answer_2': '10^-8 eV',
      'incorrect_answer_3': '10^-9 eV',
      'example_id': 'rec06pnAkLOr2t2mp',
  }

  def _gold(self) -> str:
    evaluation_ = evaluation.KimiK3GPQADiamond()
    return evaluation_.make_raw_question_and_answer(self.EXAMPLE)[1]

  def test_the_prompt_is_unchanged_from_the_base_evaluation(self):
    """The prompt is half the protocol; only the scoring may differ."""
    base = evaluation_lib.ZeroShotGPQADeepSeekQwenR1CoTBoxed()
    self.assertEqual(
        evaluation.KimiK3GPQADiamond().get_prompt(self.EXAMPLE),
        base.get_prompt(self.EXAMPLE),
    )

  def test_scores_the_option_k3_boxes(self):
    gold = self._gold()
    response = rf'... so \boxed{{\text{{({gold}) }}10^{{-4}}\ \text{{eV}}}}'
    result = evaluation.KimiK3GPQADiamond().evaluate(self.EXAMPLE, response)
    self.assertEqual(result, {'correct': True, 'reward': 1.0, 'answered': True})

  def test_the_base_evaluation_scores_that_same_answer_wrong(self):
    """The bug this class exists for, pinned so it cannot be 'fixed' silently."""
    gold = self._gold()
    response = rf'... so \boxed{{\text{{({gold}) }}10^{{-4}}\ \text{{eV}}}}'
    base = evaluation_lib.ZeroShotGPQADeepSeekQwenR1CoTBoxed()
    self.assertFalse(base.evaluate(self.EXAMPLE, response)['correct'])

  def test_an_unanswered_response_is_wrong_but_flagged(self):
    result = evaluation.KimiK3GPQADiamond().evaluate(
        self.EXAMPLE, 'still thinking when the cap hit'
    )
    self.assertEqual(
        result, {'correct': False, 'reward': 0.0, 'answered': False}
    )


if __name__ == '__main__':
  absltest.main()

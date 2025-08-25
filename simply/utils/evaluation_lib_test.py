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
"""Unit test for evaluation_lib.py.
"""

import json
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from simply.utils import evaluation_lib
from simply.utils import math_eval


TEST_EXAMPLE = {
    'question': (
        'What is 1 + 1?'
    ),
    'short_answer': '2',
    'answer': '1 apple and 1 apple are 2 apples, so the answer is 2.',
    'uid': 'unit_test-0',
    'id': 0,
}

TEST_MMLU_EXAMPLE_1 = {
    'answer': 1,
    'choices': ['0', '1', '2', '3'],
    'question': 'What is 1 + 0?',
    'subject': 'math'}

TEST_MMLU_EXAMPLE_2 = {
    'answer': 2,
    'choices': ['True, True', 'False, False', 'True, False', 'False, True'],
    'question': 'Statement 1: 1 + 1 = 2. Statement 2: 1 + 1 = 3.',
    'subject': 'math'}

TEST_MMLU_EXAMPLE_3 = {
    'question': (
        'Which one is the result of 1 + 1?'
    ),
    'answer': 2,
    'choices': ['0', '1', '2', '3'],
    'subject': 'math',
}

TEST_GPQA_EXAMPLE_1 = {
    'question': 'What is 1 + 0?',
    'correct_answer': '1',
    'incorrect_answer_1': '0',
    'incorrect_answer_2': '2',
    'incorrect_answer_3': '3',
    'example_id': 'abc',
}


GSM8K_8_SHOTS_PROMPT = """As an expert problem solver solve step by step the following mathematical questions.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.

"""

TEST_MMLU_2_SHOTS_PROMPT = """Q: What is 1 + 0?
(A) 0 (B) 1 (C) 2 (D) 3
A: B

Q: Statement 1: 1 + 1 = 2. Statement 2: 1 + 1 = 3.
(A) True, True (B) False, False (C) True, False (D) False, True
A: C

"""


class EvaluationLibTest(absltest.TestCase):

  def test_few_shot_gsm8k_evaluation(self):
    evaluation = evaluation_lib.FewShotGSM8KEvaluation()
    prompt = evaluation.get_messages(TEST_EXAMPLE)
    expected_prompt = (
        GSM8K_8_SHOTS_PROMPT + 'Question: What is 1 + 1?\nAnswer: ')
    self.assertEqual(prompt, [{'role': 'user', 'content': expected_prompt}])
    result = evaluation.evaluate(TEST_EXAMPLE, 'The answer is 2.')
    self.assertDictEqual({'correct': 1, 'reward': 1}, result)
    result = evaluation.evaluate(TEST_EXAMPLE, 'The answer is 3.')
    self.assertDictEqual({'correct': 0, 'reward': 0}, result)

  def test_zero_shot_boxed_in_question_evaluation(self):
    evaluation = evaluation_lib.ZeroShotBoxedInQuestionEvaluation()
    prompt = evaluation.get_messages(TEST_EXAMPLE)
    expected_prompt = (
        r'Question: What is 1 + 1? '
        r'Put your answer number as the format: \boxed{<Your Answer>}.'
        '\nAnswer: '
        )
    self.assertEqual(prompt, [{'role': 'user', 'content': expected_prompt}])
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{2}.')
    self.assertDictEqual({'correct': 1, 'reward': 1}, result)
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{3}.')
    self.assertDictEqual({'correct': 0, 'reward': 0}, result)
    result = evaluation.evaluate(TEST_EXAMPLE, 'The answer is 2.')
    self.assertDictEqual({'correct': 0, 'reward': 0}, result)

  def test_zero_shot_cot_boxed_in_question_evaluation(self):
    evaluation = evaluation_lib.ZeroShotCoTBoxedInQuestionEvaluation()
    prompt = evaluation.get_messages(TEST_EXAMPLE)
    expected_prompt = (
        r'Question: What is 1 + 1? '
        r'Put your answer number as the format: \boxed{<Your Answer>}.'
        "\nAnswer: Let's think step by step."
        )
    self.assertEqual(prompt, [{'role': 'user', 'content': expected_prompt}])
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{2}.')
    self.assertDictEqual({'correct': 1, 'reward': 1}, result)
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{3}.')
    self.assertDictEqual({'correct': 0, 'reward': 0}, result)

  def test_zero_shot_deep_seek_qwen_r1_cot_boxed(self):
    evaluation = evaluation_lib.ZeroShotDeepSeekQwenR1CoTBoxed()
    prompt = evaluation.get_messages(TEST_EXAMPLE)
    expected_prompt = (
        r'What is 1 + 1?'
        r' Please reason step by step, and put your final'
        r' answer within \boxed{}.'
        )
    self.assertEqual(prompt, [{'role': 'user', 'content': expected_prompt}])
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{2}.')
    self.assertDictEqual({'correct': 1, 'reward': 1}, result)
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{3}.')
    self.assertDictEqual({'correct': 0, 'reward': 0}, result)

  def test_zero_shot_system_cot_boxed(self):
    evaluation = evaluation_lib.ZeroShotSystemCoTBoxed()
    prompt = evaluation.get_messages(TEST_EXAMPLE)
    self.assertEqual(
        prompt,
        [
            {
                'role': 'system',
                'content': (
                    r'Please reason step by step, and put your final answer'
                    r' within \boxed{}.'
                ),
            },
            {'role': 'user', 'content': 'What is 1 + 1?'},
        ],
    )
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{2}.')
    self.assertDictEqual({'correct': 1, 'reward': 1}, result)
    result = evaluation.evaluate(TEST_EXAMPLE, r'The answer is \boxed{3}.')
    self.assertDictEqual({'correct': 0, 'reward': 0}, result)


class AnswerNormalizerTest(parameterized.TestCase):

  # Unit tests for the answer normalizer
  # TODO: Add more unit tests when necessary.
  @parameterized.named_parameters(
      ('fraction_equivalence', r'1/3', r'\frac{1}{3}', True),
      ('dfrac_to_frac', r'\dfrac{9\pi}{20}', r'\frac{9\pi}{20}', True),
      (
          'remove_left_and_right',
          r'\left( 2, \frac{8 \pi}{7}, \frac{7 \pi}{9} \right)',
          r'(2,\frac{8\pi}{7},\frac{7\pi}{9})',
          True,
      ),
      (
          'replace_double_backslash_with_single',
          r'\begin{pmatrix} -1 & -5 \\ 1 & 4 \end{pmatrix}',
          r'\begin{pmatrix}-1&-5\1&4\end{pmatrix}',
          True,
      ),
      ('remove_circ', r'30^\circ', '30', True),
      ('remove_text', r'108\text{ degrees}', '108', True),
      ('remove_brackets_for_multiple_choice_answer', r'(A)', r'A', True),
  )
  def test_match(self, answer, ground_truth, expected):
    self.assertEqual(math_eval.match(answer, ground_truth), expected)


if __name__ == '__main__':
  absltest.main()

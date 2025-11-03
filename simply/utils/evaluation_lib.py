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
"""Utils for evaluation."""

import abc
import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import json
import textwrap
import types
from typing import Any, ClassVar

from etils import epath
import numpy as np
from simply.utils import lm_format as lm_format_lib
from simply.utils import math_eval
from simply.utils import registry
from simply.utils import sampling_lib


maybe_remove_comma = math_eval.maybe_remove_comma
find_number = math_eval.find_number
extract_boxed_answer = math_eval.extract_boxed_answer
match = math_eval.match


GSM8K_8_SHOTS_TXT = textwrap.dedent("""
  There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
  We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
  If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
  There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
  Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
  Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
  Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
  Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
  Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
  He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
  There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
  There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
  Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
  Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
  Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
  She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.""")


class EvaluationRegistry(registry.RootRegistry):
  """Evaluation registry."""

  namespace: ClassVar[str] = 'evaluation'


class Evaluation(abc.ABC):
  """Base class for evaluation."""

  @abc.abstractmethod
  def evaluate(
      self, example: Mapping[str, Any], response: Any
  ) -> Mapping[str, Any]:
    """Evaluates the response in for the given example."""

  def get_messages(
      self, example: Mapping[str, Any]
  ) -> Sequence[Mapping[str, Any]]:
    """API that is compatible with HuggingFace apply_chat_template.

    Args:
      example: The example to be evaluated.

    Returns:
      A list of messages. Each message looks like
      {"role": "user", "content": prompt}.
    """
    raise NotImplementedError()

  def get_sampling_input(
      self, example: Mapping[str, Any], lm_format: lm_format_lib.LMFormat
  ) -> sampling_lib.SamplingInput:
    """Converts raw example to SamplingInput.

    Args:
      example: The example to be evaluated.
      lm_format: LMFormat for putting text in the proper text format.

    Returns:
      SamplingInput.

    NOTE: This is just a default implementation for backwards compatibility with
    text-only evaluations based on get_messages(). If you are using input that
    is not pure text (e.g. interleaving text and images), consider overriding
    with your own custom handling.
    """
    chunks = list(
        sampling_lib.input_as_chunks(
            lm_format.format(self.get_messages(example))
        )
    )
    extra_inputs = example.get('extra_inputs', {})
    for extra_input_value in extra_inputs.values():
      chunks.append(
          sampling_lib.Chunk(
              type=sampling_lib.Chunk.Type.ARRAY,
              content=extra_input_value,
          )
      )
    return chunks


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class FewShotGSM8KEvaluation(Evaluation):
  """Few-shot GSM8K evaluation."""

  prompt_template: ClassVar[str] = '{system_marker}{system_message}{turns}'
  partial_turn_template: ClassVar[str] = (
      '{input_marker}{input}{input_end}{output_marker}'
  )
  turn_template: ClassVar[str] = partial_turn_template + '{output}{output_end}'
  question_template: ClassVar[str] = (
      '{question_start}{question}{question_end}{answer_start}'
  )
  default_8_shots_txt: ClassVar[str] = GSM8K_8_SHOTS_TXT
  n_shots: int = 8

  input_marker: str = 'Question: '
  output_marker: str = 'Answer: '
  input_end: str = '\n'
  output_end: str = '\n\n'
  system_marker: str = ''
  system_message: str = (
      'As an expert problem solver solve step by step the following'
      ' mathematical questions.\n\n'
  )

  @functools.cached_property
  def shots(self):
    """Returns the shots."""
    lines = []
    for line in self.default_8_shots_txt.split('\n'):
      if s := line.strip():
        lines.append(s)
    assert len(lines) % 2 == 0
    total_shots = len(lines) // 2
    assert total_shots >= self.n_shots
    shots = []
    for i in range(min(total_shots, self.n_shots)):
      shots.append((lines[i * 2], lines[i * 2 + 1]))
    return shots

  def get_messages(
      self, example: Mapping[str, Any]
  ) -> Sequence[Mapping[str, Any]]:
    """Returns the prompt for the given example."""
    turn_list = []
    for question, answer in self.shots:
      turn_list.append(
          self.turn_template.format(
              input_marker=self.input_marker,
              output_marker=self.output_marker,
              input_end=self.input_end,
              output_end=self.output_end,
              input=question,
              output=answer,
          )
      )
    preamble = self.prompt_template.format(
        system_marker=self.system_marker,
        system_message=self.system_message,
        turns=''.join(turn_list),
    )
    prompt = self.question_template.format(
        question_start=self.input_marker,
        question=example['question'],
        question_end=self.input_end,
        answer_start=self.output_marker,
    )
    return [dict(role='user', content=preamble + prompt)]

  def evaluate(
      self, example: Mapping[str, Any], response: str
  ) -> Mapping[str, Any]:
    """Evaluate the response in for the given example."""
    response_answer = maybe_remove_comma(find_number(response))
    expected_answer = maybe_remove_comma(example['short_answer'])
    res = {}
    try:
      correct = int(float(response_answer) == float(expected_answer))
    except Exception:  # pylint: disable=broad-except
      correct = int(response_answer == expected_answer)
    res['correct'] = correct
    res['reward'] = float(res['correct'])
    return res


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class ZeroShotBoxedInQuestionEvaluation(Evaluation):
  r"""0-shot that asks for \boxed{} in the question part."""

  # TODO: In both eval and training, only keep contents before
  # "Question:" in the response string.
  system_message: str = ''
  question_start: str = 'Question: '
  question_end: str = (
      r' Put your answer number as the format: \boxed{<Your Answer>}.' + '\n'
  )
  answer_start: str = 'Answer: '

  def get_prompt(self, example: Mapping[str, Any]) -> str:
    """Returns the prompt for the given example."""
    return (
        self.question_start
        + example['question']
        + self.question_end
        + self.answer_start
    )

  def get_messages(
      self, example: Mapping[str, Any]
  ) -> Sequence[Mapping[str, Any]]:
    """Returns the prompt for the given example."""
    output = []
    if self.system_message:
      output.append(dict(role='system', content=self.system_message))
    prompt = self.get_prompt(example)
    output.append(dict(role='user', content=prompt))
    return output

  def evaluate(
      self, example: Mapping[str, Any], response: str
  ) -> Mapping[str, Any]:
    """Rates the response in for the given example."""
    # Extract answer from \boxed{}. The answer not in \boxed{} is treated as
    # incorrect.
    response_answer = extract_boxed_answer(response)
    expected_answer = maybe_remove_comma(example['short_answer'])
    res = {}
    if response_answer:
      correct = match(response_answer, expected_answer)
    else:
      correct = False
    res['correct'] = correct
    res['reward'] = float(res['correct'])
    return res


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class ZeroShotCoTBoxedInQuestionEvaluation(ZeroShotBoxedInQuestionEvaluation):
  r"""0-shot that asks for \boxed{} in the question part and have "Let's think step by step" at the beginning of answer."""

  # TODO: In both eval and training, only keep contents before
  # "Question:" in the response string.
  # TODO: Rename to names like CoTV1 because we may want to try
  # different CoT versions.
  answer_start: str = "Answer: Let's think step by step."


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class QAToolUseEvaluation(ZeroShotBoxedInQuestionEvaluation):
  question_start: str = textwrap.dedent("""
    You are a helpful agent with a search tool to answer questions. If you need search, use search tool in this exact format: `<query>your search query</query>`. Stop immediately after one search call and wait for the user to provide the result.
    Example:
    User: Question: What is the capital of France? Put your answer in \\boxed{<Your answer>}.
    Model: I need to use google search to find the France capital. <query>France capital</query>
    User: ```research_result
    # 'Paris'
    Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine.
    ```
    Model: The research result states that Paris is the France's capital. \\boxed{Paris}
    
    Question: """)


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class ZeroShotDeepSeekQwenR1CoTBoxed(ZeroShotBoxedInQuestionEvaluation):
  r"""0-shot that asks to reason step by step and put answer in \boxed{}.

  Following instructions in:
  https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

  Can be used by DeepSeekQwen and QwQ models.
  """

  question_start: str = ''
  question_end: str = (
      r' Please reason step by step, and put your final answer within \boxed{}.'
  )
  answer_start: str = ''


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class ZeroShotGeminiCoTBoxed(ZeroShotDeepSeekQwenR1CoTBoxed):
  """0-shot that uses gemini thinking system instruction."""

  system_message: str = 'SPECIAL INSTRUCTION: think silently.'


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class ZeroShotSystemCoTBoxed(ZeroShotDeepSeekQwenR1CoTBoxed):
  r"""0-shot that asks to reason step by step and put answer in \boxed{}.

  It asks in the system message.
  """
  system_message: str = (
      r'Please reason step by step, and put your final answer within \boxed{}.'
  )
  question_start: str = ''
  question_end: str = ''
  answer_start: str = ''

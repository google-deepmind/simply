# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LM formats + GSM8K few-shot eval classes for the model-PORTING tasks.

Provides the FIXED (participant-immutable) GSM8K eval protocols + few-shot
lm_formats used by `port_falcon_h1_0p5b` (5-shot strict, no-trailing-space,
greedy) and `port_recurrentgemma_2b` (5-shot canonical-CoT strict, temp 0.4).

Everything here registers on the SHARED core registries (LMFormatRegistry /
EvaluationRegistry) so a config can reference them by name; NO core lm_format.py
or evaluation_lib.py change is required. `main.py` imports this module so the
registrations fire.

The eval class shipped/frozen for the task is
`FewShotGSM8K5ShotStrictNoTrailSpaceEvaluation`: 5-shot, lm-eval-harness
strict-match extraction, and the trailing-space-FIXED prompt marker. The
trailing-space FIX matters: the plain `FewShotGSM8K5ShotStrictEvaluation`
(marker 'Answer: ' with a dangling space) scores a correct port several points
LOWER -- a tokenization artifact, since the query then ends in a lone-space
token never seen in the exemplars -- which would false-negative a correct port.
The fixed variant glues that space onto each exemplar output instead (marker
'Answer:').
"""

import dataclasses
import re
from typing import Any, Mapping

from simply.utils import evaluation_lib as _eval
from simply.utils import lm_format as _lmf

LMFormatRegistry = _lmf.LMFormatRegistry
EvaluationRegistry = _eval.EvaluationRegistry

# math_eval helpers used by the strict extractor (reused from core).
_maybe_remove_comma = _eval.maybe_remove_comma


@LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class FalconH1GSM8K(_lmf.LMFormat):
  r"""Falcon-H1 few-shot GSM8K format.

  Plain pretrain text; stop at the few-shot separator ('\n\n', a single token in
  the Falcon-H1 vocab) so the answer extractor reads only the first completed
  answer. Falcon-H1 base few-shot pads with pad_token_id=0 and does NOT prepend
  BOS.
  """

  extra_eos_tokens: tuple[str, ...] = ('\n\n',)
  pad_id: int = 0

  def format(self, messages):
    if len(messages) != 1:
      raise ValueError(
          f'FalconH1GSM8K only supports 1 message, got {len(messages)}.'
      )
    return messages[0]['content']


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class FewShotGSM8K5ShotStrictEvaluation(_eval.FewShotGSM8KEvaluation):
  """5-shot GSM8K with lm-eval-harness 'strict-match' extraction.

  A building block, not a task eval: it supplies the strict extraction reused by
  `FewShotGSM8K5ShotStrictNoTrailSpaceEvaluation` (the eval the Falcon port task
  is scored under).

  strict-match anchors on the exemplar answer format only: the number must
  appear right after 'The answer is' (optionally '#### <num>'). If neither
  anchored pattern is present, the answer is empty (counted wrong) -- this is
  stricter than the flexible extractor (which falls back to the last number).
  """

  n_shots: int = 5

  def evaluate(
      self, example: Mapping[str, Any], response: str
  ) -> Mapping[str, Any]:
    resp = response
    ans = ''
    # Anchored strict extraction: the number must appear right after the
    # 'The answer is' delimiter (or the GSM8K gold '#### <num>' format). Allow a
    # leading currency symbol / whitespace and thousands separators; strip a
    # trailing period. Mirrors lm-eval-harness gsm8k strict-match.
    m = re.findall(r'The answer is\s*\$?\s*(-?[\d,]*\.?\d+)', resp)
    if m:
      ans = m[0]
    else:
      m2 = re.findall(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', resp)
      if m2:
        ans = m2[0]
    response_answer = _maybe_remove_comma(ans)
    expected_answer = _maybe_remove_comma(example['short_answer'])
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
class FewShotGSM8KNoTrailSpaceEvaluation(_eval.FewShotGSM8KEvaluation):
  """GSM8K with NO trailing space after the query 'Answer:' marker.

  A building block, not a task eval: it supplies the prompt construction reused
  by `FewShotGSM8K5ShotStrictNoTrailSpaceEvaluation` (the eval the Falcon port
  task is scored under).

  With output_marker='Answer: ' (trailing space) the exemplars tokenize the
  space as part of the next word, but the QUERY ends with a dangling lone-space
  token that never appears in the exemplars -- a distribution shift that costs a
  few points. Here the marker carries no trailing space and each output starts
  with its own leading space, so exemplars and the query share the identical
  'Answer:'+word tokenization.
  """

  output_marker: str = 'Answer:'

  def get_messages(self, example):
    turn_list = []
    for question, answer in self.shots:
      turn_list.append(
          self.turn_template.format(
              input_marker=self.input_marker,
              output_marker=self.output_marker,
              input_end=self.input_end,
              output_end=self.output_end,
              input=question,
              output=' ' + answer,  # leading space glued to first word
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
        answer_start=self.output_marker,  # no trailing space
    )
    return [dict(role='user', content=preamble + prompt)]


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class FewShotGSM8K5ShotStrictNoTrailSpaceEvaluation(
    FewShotGSM8K5ShotStrictEvaluation
):
  """OFFICIAL frozen protocol: 5-shot, strict-match, no trailing space.

  This is the FIXED eval the task scaffolding ships: strict-match extraction
  from `FewShotGSM8K5ShotStrictEvaluation`, no-trailing-space prompt
  construction from `FewShotGSM8KNoTrailSpaceEvaluation`. Prompt formatting
  moves this score by several points, so the target for a correct port is the
  one stated in the task, not a published figure for the same model.
  """

  output_marker: str = 'Answer:'

  # Reuse the no-trailing-space prompt construction.
  get_messages = FewShotGSM8KNoTrailSpaceEvaluation.get_messages


#################################################################################
## RecurrentGemma-2B port: 5-shot canonical-CoT STRICT GSM8K eval + lm_format.
##
## The RecurrentGemma-2B port is scored under the PAPER-FAITHFUL protocol:
## 5-shot with the CANONICAL 8-shot CoT exemplars (first 5), STRICT anchored
## extraction
## (`find_number_strict`: only a number right after 'The answer is' / gold
## '#### <num>', no lenient last-number fallback), and a SINGLE sample at
## temperature 0.4 over the FULL gsm8k test. This reproduces the report's 13.4
## (measured: temp0.4 mean 13.82%, seeds 13.5-14.3%; greedy temp0 -> 15.92% as a
## low-variance sanity check). Everything registers on the SHARED core
## registries; NO core lm_format.py / evaluation_lib.py change is required.
#################################################################################


def find_number_strict(x: str, answer_delimiter: str = 'The answer is') -> str:
  """Strict extraction: only accept a number anchored to the answer delimiter.

  Matches the report's deterministic '#### <num>' / 'The answer is <num>'
  protocol. Allows an optional leading '$' and takes the first number that
  immediately follows the LAST answer delimiter. NO lenient last-number
  fallback: if the delimiter is absent (or no number follows it) -> '' (wrong).

  Args:
    x: The model response text.
    answer_delimiter: The primary anchor phrase.

  Returns:
    The extracted number string, or '' if no anchored number is present.
  """
  idx = -1
  used = None
  for delim in (answer_delimiter, '####'):
    j = x.rfind(delim)
    if j > idx:
      idx = j
      used = delim
  if idx < 0 or used is None:
    return ''
  tail = x[idx + len(used):]
  m = re.search(r'\$?\s*(-?[\d,]*\.?\d+)', tail)
  if m:
    return m.group(1)
  return ''


@EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class FewShot5StrictGSM8KEvaluation(_eval.FewShotGSM8KEvaluation):
  """OFFICIAL frozen RecurrentGemma protocol: 5-shot canonical CoT, STRICT.

  Uses the canonical 8-shot CoT exemplars (first 5) with the default
  'Question: '/'Answer: ' markers, and STRICT anchored extraction
  (`find_number_strict`): only accepts a number right after 'The answer is'
  (or the gold '#### <num>'), optional leading '$'; NO lenient last-number
  fallback. Scored with a single sample at temperature 0.4 (config-driven) over
  the full gsm8k test -> reproduces the paper's 13.4.
  """

  n_shots: int = 5

  def evaluate(
      self, example: Mapping[str, Any], response: str
  ) -> Mapping[str, Any]:
    response_answer = _maybe_remove_comma(find_number_strict(response))
    expected_answer = _maybe_remove_comma(example['short_answer'])
    res = {}
    try:
      correct = int(float(response_answer) == float(expected_answer))
    except Exception:  # pylint: disable=broad-except
      correct = int(
          bool(response_answer) and response_answer == expected_answer
      )
    res['correct'] = correct
    res['reward'] = float(res['correct'])
    return res


@LMFormatRegistry.register
@dataclasses.dataclass(frozen=True)
class RecurrentGemmaGSM8K(_lmf.LMFormat):
  r"""RecurrentGemma-2B few-shot GSM8K format.

  Base (non-instruct) model. Prepend the Gemma BOS (id 2), and stop at the shot
  separator ('\n\n', a single Gemma vocab token) so the strict extractor reads
  only the model's first completed answer instead of a hallucinated later Q/A
  pair.
  """

  bos_id: int = 2
  extra_eos_tokens: tuple[str, ...] = ('\n\n',)

  def format(self, messages):
    if len(messages) != 1:
      raise ValueError(
          f'RecurrentGemmaGSM8K only supports 1 message, got {len(messages)}.'
      )
    return messages[0]['content']

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
r"""Scoring a multiple-choice answer the way Kimi K3 writes one.

`ZeroShotGPQADeepSeekQwenR1CoTBoxed` asks for the final answer in `\boxed{}`
and then compares the boxed string to the gold *letter*. K3 does not box a bare
letter -- it boxes the option, e.g.

    \boxed{\text{(B) }10^{-4}\ \text{eV}}

and the prompt never asked for anything narrower. Scored literally, a run of
GPQA-Diamond on the released weights returns **14.65%, below the 25% of random
guessing**; the same generations read with the extractor here return 175 of
the 198 questions right -- 88.4% of the set, 93.3% of the 188 answers that
parse at all, against the K3 report's 93.5. A scorer that
turns right answers into wrong ones is worse than no scorer, so this one is
explicit about what it accepts and about refusing to guess.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import re
from typing import Any

from simply.utils import evaluation_lib

_LETTERS = 'ABCD'
# `\text{(B) }`, `\mathrm{B}`, `\textbf{B}` -- LaTeX wrappers that carry no
# meaning for an option letter.
_WRAPPER = re.compile(r'\\(?:text|textbf|textrm|mathrm|mathbf)\s*\{([^{}]*)\}')
_LEADING_LETTER = re.compile(r'^\(?([A-D])\)?(?:[\s.:,)]|$)')
_PARENTHESISED = re.compile(r'\(([A-D])\)')


def last_boxed(response: str) -> str | None:
  r"""The content of the last `\boxed{...}`, with braces balanced.

  The last one, not the first: K3's think channel is full of intermediate
  `\boxed{}`s and the answer is the final one.

  Args:
    response: the model's output.

  Returns:
    The boxed content, or None if there is no `\boxed{`.
  """
  start = response.rfind('\\boxed{')
  if start < 0:
    return None
  i = start + len('\\boxed{')
  depth = 1
  while i < len(response) and depth:
    depth += (response[i] == '{') - (response[i] == '}')
    i += 1
  return response[start + len('\\boxed{') : i - 1]


def _normalize(text: str) -> str:
  text = _WRAPPER.sub(r'\1', text)
  text = text.replace('$', '').replace('\\ ', ' ').replace('\\,', ' ')
  return ' '.join(text.split())


def extract_option_letter(
    response: str, options: Sequence[str] = ()
) -> str | None:
  r"""The option letter the response settles on, or None if it is not clear.

  The rule, in order: the last `\boxed{}`; strip LaTeX text wrappers; a leading
  `A`-`D` (bare or parenthesised); else a `(A)`-`(D)` anywhere in it; else, if
  the boxed text matches exactly one of `options`, that one.

  Returning None rather than guessing is deliberate: a wrong guess on a
  four-way choice is indistinguishable from a wrong answer in the aggregate.

  Args:
    response: the model's output.
    options: the four answer texts, in the order they were presented.

  Returns:
    'A'..'D', or None.
  """
  boxed = last_boxed(response)
  if boxed is None:
    return None
  text = _normalize(boxed)
  if match := _LEADING_LETTER.match(text):
    return match.group(1)
  if match := _PARENTHESISED.search(text):
    return match.group(1)
  hits = [
      _LETTERS[i]
      for i, option in enumerate(options)
      if (normalized := _normalize(option))
      and (normalized in text or text in normalized)
  ]
  return hits[0] if len(hits) == 1 else None


def options_from_question(question: str) -> tuple[str, ...]:
  """The `(A) ...` / `(B) ...` lines of a rendered multiple-choice question."""
  return tuple(re.findall(r'^\([A-D]\)\s*(.*)$', question, flags=re.MULTILINE))


@evaluation_lib.EvaluationRegistry.register
@dataclasses.dataclass(frozen=True)
class KimiK3GPQADiamond(evaluation_lib.ZeroShotGPQADeepSeekQwenR1CoTBoxed):
  r"""GPQA-Diamond, scored on the option K3 boxes rather than on a bare letter.

  Identical prompt to the base class -- which matters, because the prompt is
  half of the protocol a reproduction has to match -- and a different reading
  of the answer.
  """

  def evaluate(
      self, example: Mapping[str, Any], response: str
  ) -> Mapping[str, Any]:
    question, expected = self.make_raw_question_and_answer(example)
    predicted = extract_option_letter(response, options_from_question(question))
    correct = predicted is not None and predicted == expected
    return {
        'correct': correct,
        'reward': float(correct),
        # Kept because the two failure modes need different fixes: no answer
        # usually means the generation was cut off, a wrong answer does not.
        'answered': predicted is not None,
    }

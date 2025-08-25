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
"""Utility functions for math evaluation.
"""

import re


# Below are utility functions to extract answer
def find_numbers(x: str) -> list[str]:
  """Finds all numbers in a string."""
  # Search for number, possibly negative (hyphen), with thousand separators
  # (comma), and with a decimal point (period inbetween digits).
  numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
  ).findall(x)
  return numbers


def find_number(x: str, answer_delimiter: str = 'The answer is') -> str:
  """Finds the most relevant number in a string."""
  # If model uses the answer delimiter, then select the first number following
  # that format.
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]

  # In general, select the last number in the string.
  numbers = find_numbers(x)
  if numbers:
    return numbers[-1]
  return ''


def maybe_remove_comma(x: str) -> str:
  # Example: 5,600 -> 5600
  return x.replace(',', '')


# Utility functions to extract answer from "\boxed{}"
# from https://github.com/agentica-project/deepscaler/blob/95ef74ffee001cce89463f2d61dfdce1d03576c1/deepscaler/rewards/math_utils/utils.py#L387-L428
def last_boxed_only_string(string: str) -> str | None:
  r"""find the last \\boxed{} or \\fbox{} part in a string."""
  idx = string.rfind('\\boxed')
  if idx < 0:
    idx = string.rfind('\\fbox')
    if idx < 0:
      return None

  i = idx
  right_brace_idx = None
  num_left_braces_open = 0
  while i < len(string):
    if string[i] == '{':
      num_left_braces_open += 1
    elif string[i] == '}':
      num_left_braces_open -= 1
      if num_left_braces_open == 0:
        right_brace_idx = i
        break
    i += 1

  if not right_brace_idx:
    retval = None
  else:
    retval = string[idx : right_brace_idx + 1]

  return retval


def remove_boxed(s: str) -> str | None:
  r"""Remove the \\boxed{} command from a string."""
  left = '\\boxed{'
  try:
    assert s[: len(left)] == left
    assert s[-1] == '}'
    return s[len(left) : -1]
  except:  # pylint: disable=bare-except
    return None


def extract_boxed_answer(solution: str) -> str | None:
  r"""Extract the answer from inside a LaTeX \\boxed{} command."""
  solution = last_boxed_only_string(solution)
  solution = remove_boxed(solution)
  return solution


# Utility function to normalize answers like "1/3" -> "\frac{1}{3}"
# From https://github.com/agentica-project/deepscaler/blob/95ef74ffee001cce89463f2d61dfdce1d03576c1/deepscaler/rewards/math_reward.py
# and https://github.com/agentica-project/deepscaler/blob/95ef74ffee001cce89463f2d61dfdce1d03576c1/deepscaler/rewards/math_utils/utils.py
def match(answer: str, ground_truth: str) -> bool:
  """Matches answer against the ground-truth."""
  ground_truths = [ground_truth]

  processed_ground_truths = []
  for truth in ground_truths:
    truth = str(truth)
    if '\\boxed' in truth:
      processed_truth = extract_boxed_answer(truth)
      if processed_truth is not None:
        processed_ground_truths.append(processed_truth)
    else:
      processed_ground_truths.append(truth)

  for ground_truth in processed_ground_truths:
    is_correct = grade_answer_mathd(answer, ground_truth) or grade_answer_sympy(
        answer, ground_truth
    )
    if is_correct:
      return True
  return False


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
  """Grade answer using mathematical correctness."""
  ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
  given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

  # be at least as lenient as mathd
  return ground_truth_normalized_mathd == given_answer_normalized_mathd


def mathd_normalize_answer(answer: str | None) -> str | None:
  if answer is None:
    return None
  answer = answer.strip()
  try:
    # Remove enclosing `\text{}`.
    m = re.search(r'^\\text\{(?P<text>.+?)\}$', answer)
    if m is not None:
      answer = m.group('text').strip()
    return _strip_string(answer)
  except:  # pylint: disable=bare-except
    return answer


def _strip_string(string: str) -> str:
  """Strips string."""

  def _fix_fracs(string: str) -> str:
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
      substrs = substrs[1:]
      for substr in substrs:
        new_str += '\\frac'
        if substr[0] == '{':
          new_str += substr
        else:
          try:
            assert len(substr) >= 2
          except:  # pylint: disable=bare-except
            return string
          a = substr[0]
          b = substr[1]
          if b != '{':
            if len(substr) > 2:
              post_substr = substr[2:]
              new_str += '{' + a + '}{' + b + '}' + post_substr
            else:
              new_str += '{' + a + '}{' + b + '}'
          else:
            if len(substr) > 2:
              post_substr = substr[2:]
              new_str += '{' + a + '}' + b + post_substr
            else:
              new_str += '{' + a + '}' + b
    string = new_str
    return string

  def _fix_a_slash_b(string: str) -> str:
    if len(string.split('/')) != 2:
      return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
      a = int(a)
      b = int(b)
      assert string == '{}/{}'.format(a, b)
      new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
      return new_string
    except:  # pylint: disable=bare-except
      return string

  def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if '\\text{ ' in string:
      splits = string.split('\\text{ ')
      assert len(splits) == 2
      return splits[0]
    else:
      return string

  def _fix_sqrt(string: str) -> str:
    if '\\sqrt' not in string:
      return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
      if split[0] != '{':
        a = split[0]
        new_substr = '\\sqrt{' + a + '}' + split[1:]
      else:
        new_substr = '\\sqrt' + split
      new_string += new_substr
    return new_string

  # remove linebreaks
  string = string.replace('\n', '')

  # remove inverse spaces
  string = string.replace('\\!', '')

  # replace \\ with \
  string = string.replace('\\\\', '\\')

  # replace tfrac and dfrac with frac
  string = string.replace('tfrac', 'frac')
  string = string.replace('dfrac', 'frac')

  # remove \left and \right
  string = string.replace('\\left', '')
  string = string.replace('\\right', '')

  # Remove circ (degrees)
  string = string.replace('^{\\circ}', '')
  string = string.replace('^\\circ', '')

  # remove dollar signs
  string = string.replace('\\$', '')

  # remove units (on the right)
  string = _remove_right_units(string)

  # remove percentage
  string = string.replace('\\%', '')
  string = string.replace('%', '')

  # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
  string = string.replace(' .', ' 0.')
  string = string.replace('{.', '{0.')
  # if empty, return empty string
  if not string:
    return string
  if string[0] == '.':
    string = '0' + string

  # to consider: get rid of e.g. "k = " or "q = " at beginning
  if len(string.split('=')) == 2:
    if len(string.split('=')[0]) <= 2:
      string = string.split('=')[1]

  # fix sqrt3 --> sqrt{3}
  string = _fix_sqrt(string)

  # remove spaces
  string = string.replace(' ', '')

  # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
  string = _fix_fracs(string)

  # manually change 0.5 --> \frac{1}{2}
  if string == '0.5':
    string = '\\frac{1}{2}'

  # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
  string = _fix_a_slash_b(string)

  return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ['^{', '^(']
BAD_REGEXES = [r'\^[0-9]+\^', r'\^[0-9][0-9]+']
TUPLE_CHARS = '()[]'


def _sympy_parse(expr: str):
  """Parses an expression with sympy."""
  from sympy.parsing import sympy_parser  # pylint: disable=g-import-not-at-top
  py_expr = expr.replace('^', '**')
  return sympy_parser.parse_expr(
      py_expr,
      transformations=(
          sympy_parser.standard_transformations
          + (sympy_parser.implicit_multiplication_application,)
      ),
  )


def _parse_latex(expr: str) -> str:
  """Attempts to parse latex to an expression sympy can read."""
  import pylatexenc  # pylint: disable=g-import-not-at-top
  expr = expr.replace('\\tfrac', '\\frac')
  expr = expr.replace('\\dfrac', '\\frac')
  expr = expr.replace('\\frac', ' \\frac')  # Play nice with mixed numbers.
  expr = pylatexenc.latex2text.LatexNodes2Text().latex_to_text(expr)

  # Replace the specific characters that this parser uses.
  expr = expr.replace('√', 'sqrt')
  expr = expr.replace('π', 'pi')
  expr = expr.replace('∞', 'inf')
  expr = expr.replace('∪', 'U')
  expr = expr.replace('·', '*')
  expr = expr.replace('×', '*')

  return expr.strip()


def _is_float(num: str) -> bool:
  try:
    float(num)
    return True
  except ValueError:
    return False


def _is_int(x: float) -> bool:
  try:
    return abs(x - int(round(x))) <= 1e-7
  except:  # pylint: disable=bare-except
    return False


def _is_frac(expr: str) -> bool:
  return bool(re.search(r'^-?[0-9]+.?/0*[1-9][0-9]*.?$', expr))


def _str_is_int(x: str) -> bool:
  try:
    x = _strip_properly_formatted_commas(x)
    x = float(x)
    return abs(x - int(round(x))) <= 1e-7
  except:  # pylint: disable=bare-except
    return False


def _str_to_int(x: str) -> int:
  x = x.replace(',', '')
  # TODO: Handle the case where the value is larger than 2^53.
  x = float(x)
  return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
  """Automatically make a mixed number evalable, like 7 3/4 => 7+3/4."""
  p1 = re.compile('([0-9]) +([0-9])')
  step = p1.sub('\\1+\\2', step)  ## implicit mults
  return step


def _strip_properly_formatted_commas(expr: str):
  # We want to be careful because we don't want to strip tuple commas
  p1 = re.compile(r'(\d)(,)(\d\d\d)($|\D)')
  while True:
    next_expr = p1.sub('\\1\\3\\4', expr)
    if next_expr == expr:
      break
    expr = next_expr
  return next_expr


def _normalize(expr: str) -> str:
  """Normalize answer expressions."""
  if expr is None:
    return None

  # Remove enclosing `\text{}`.
  m = re.search(r'^\\text\{(?P<text>.+?)\}$', expr)
  if m is not None:
    expr = m.group('text')

  expr = expr.replace('\\%', '%')
  expr = expr.replace('\\$', '$')
  expr = expr.replace('$', '')
  expr = expr.replace('%', '')
  expr = expr.replace(' or ', ' , ')
  expr = expr.replace(' and ', ' , ')

  expr = expr.replace('million', '*10^6')
  expr = expr.replace('billion', '*10^9')
  expr = expr.replace('trillion', '*10^12')

  for unit in [
      'degree',
      'cm',
      'centimeter',
      'meter',
      'mile',
      'second',
      'minute',
      'hour',
      'day',
      'week',
      'month',
      'year',
      'foot',
      'feet',
      'inch',
      'yard',
  ]:
    expr = re.sub(f'{unit}(es)?(s)? *(\\^[0-9]+)?', '', expr)
  expr = re.sub(r'\^ *\\circ', '', expr)

  if expr and expr[0] == '{' and expr[-1] == '}':
    expr = expr[1:-1]

  expr = re.sub(',\\\\! *', '', expr)
  if _is_float(expr) and _is_int(float(expr)):
    expr = str(int(round(float(expr))))
  if '\\' in expr:
    try:
      expr = _parse_latex(expr)
    except:  # pylint: disable=bare-except
      pass

  # edge case with mixed numbers and negative signs
  expr = re.sub('- *', '-', expr)

  expr = _inject_implicit_mixed_number(expr)
  expr = expr.replace(' ', '')

  # if we somehow still have latex braces here, just drop them
  expr = expr.replace('{', '')
  expr = expr.replace('}', '')

  # don't be case sensitive for text answers
  expr = expr.lower()

  if _str_is_int(expr):
    expr = str(_str_to_int(expr))

  return expr


def count_unknown_letters_in_expr(expr: str):
  expr = expr.replace('sqrt', '')
  expr = expr.replace('frac', '')
  letters_in_expr = set([x for x in expr if x.isalpha()])
  return len(letters_in_expr)


def should_allow_eval(expr: str) -> bool:
  """Whether we should try to eval in the first place: we don't want to try parsing unknown text or functions of more than two variables."""
  if count_unknown_letters_in_expr(expr) > 2:
    return False

  for bad_string in BAD_SUBSTRINGS:
    if bad_string in expr:
      return False

  for bad_regex in BAD_REGEXES:
    if re.search(bad_regex, expr) is not None:
      return False

  return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
  """Check equality using sympy."""
  import sympy  # pylint: disable=g-import-not-at-top
  are_equal = False
  try:
    expr = f'({ground_truth_normalized})-({given_normalized})'
    if should_allow_eval(expr):
      sympy_diff = _sympy_parse(expr)
      simplified = sympy.simplify(sympy_diff)
      if simplified == 0:
        are_equal = True
  except:  # pylint: disable=bare-except
    pass
  return are_equal


def split_tuple(expr: str):
  """Split the elements in a tuple/interval, while handling well-formatted commas in large numbers."""
  expr = _strip_properly_formatted_commas(expr)
  if not expr:
    return []
  if (
      len(expr) > 2
      and expr[0] in TUPLE_CHARS
      and expr[-1] in TUPLE_CHARS
      and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
  ):
    elems = [elem.strip() for elem in expr[1:-1].split(',')]
  else:
    elems = [expr]
  return elems


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
  """Grade the answer using sympy."""
  is_correct = False
  ground_truth_normalized = _normalize(ground_truth)
  given_normalized = _normalize(given_answer)

  if ground_truth_normalized is None:
    return False

  if ground_truth_normalized == given_normalized:
    return True

  if not given_normalized:
    return False

  ground_truth_elems = split_tuple(ground_truth_normalized)
  given_elems = split_tuple(given_normalized)

  if len(ground_truth_elems) > 1 and (
      ground_truth_normalized[0] != given_normalized[0]
      or ground_truth_normalized[-1] != given_normalized[-1]
  ):
    is_correct = False
  elif len(ground_truth_elems) != len(given_elems):
    is_correct = False
  else:
    for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
      if _is_frac(ground_truth_elem) and _is_frac(given_elem):
        # if fractions aren't reduced, then shouldn't be marked as correct
        # so, we don't want to allow sympy.simplify in this case
        is_correct = ground_truth_elem == given_elem
      elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
        # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
        is_correct = False
      else:
        is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
      if not is_correct:
        break

  return is_correct

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
"""Tests for math_eval."""

from absl.testing import absltest
from absl.testing import parameterized
from simply.utils import math_eval


class FindNumbersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("single", "the value is 42", ["42"]),
      ("multiple", "3 cats and 4 dogs", ["3", "4"]),
      ("negative", "it dropped by -5 today", ["-5"]),
      ("decimal", "pi is about 3.14", ["3.14"]),
      ("thousands_separator", "5,600 items", ["5,600"]),
      ("none", "no digits here", []),
  )
  def test_find_numbers(self, text, expected):
    self.assertEqual(math_eval.find_numbers(text), expected)

  def test_find_number_prefers_answer_delimiter(self):
    text = "First I compute 10, so the answer is 42."
    self.assertEqual(math_eval.find_number(text), "42")

  def test_find_number_falls_back_to_last_number(self):
    text = "We saw 10 then 20 then 30."
    self.assertEqual(math_eval.find_number(text), "30")

  def test_find_number_empty_when_no_numbers(self):
    self.assertEqual(math_eval.find_number("no numbers"), "")

  def test_maybe_remove_comma(self):
    self.assertEqual(math_eval.maybe_remove_comma("5,600"), "5600")
    self.assertEqual(math_eval.maybe_remove_comma("42"), "42")


class BoxedAnswerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("boxed", r"the answer is \boxed{42}", r"\boxed{42}"),
      ("fbox", r"\fbox{7}", r"\fbox{7}"),
      ("nested_braces", r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}"),
      ("last_wins", r"\boxed{1} and \boxed{2}", r"\boxed{2}"),
  )
  def test_last_boxed_only_string(self, text, expected):
    self.assertEqual(math_eval.last_boxed_only_string(text), expected)

  def test_last_boxed_only_string_returns_none_when_absent(self):
    self.assertIsNone(math_eval.last_boxed_only_string("no box here"))

  @parameterized.named_parameters(
      ("simple", r"\boxed{42}", "42"),
      ("fraction", r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
  )
  def test_remove_boxed(self, text, expected):
    self.assertEqual(math_eval.remove_boxed(text), expected)

  def test_remove_boxed_returns_none_on_bad_input(self):
    self.assertIsNone(math_eval.remove_boxed("not boxed"))

  def test_extract_boxed_answer(self):
    self.assertEqual(
        math_eval.extract_boxed_answer(r"work... \boxed{99}"), "99"
    )

  def test_extract_boxed_answer_returns_none_when_absent(self):
    self.assertIsNone(math_eval.extract_boxed_answer("plain text"))


class StrToIntTest(parameterized.TestCase):
  """Tests for `_str_to_int`, including the >2^53 precision regression."""

  @parameterized.named_parameters(
      ("small", "5", 5),
      ("negative", "-42", -42),
      ("thousands_separator", "5,600", 5600),
      ("multi_separator", "1,000,000", 1_000_000),
      ("float_formatted", "1.0", 1),
      ("scientific", "5e3", 5000),
  )
  def test_str_to_int(self, text, expected):
    self.assertEqual(math_eval._str_to_int(text), expected)

  def test_preserves_precision_above_2_pow_53(self):
    # 2^53 + 1 is the smallest positive integer a float64 cannot represent
    # exactly; converting via float() would round it down to 2^53.
    big = "9007199254740993"  # 2**53 + 1
    self.assertEqual(math_eval._str_to_int(big), 9007199254740993)

  def test_preserves_precision_very_large_integer(self):
    huge = "12345678901234567890"
    self.assertEqual(math_eval._str_to_int(huge), 12345678901234567890)


class StrIsIntTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("plain_int", "42", True),
      ("int_with_commas", "5,600", True),
      ("float_valued_int", "7.0", True),
      ("non_int_float", "3.5", False),
      ("text", "abc", False),
  )
  def test_str_is_int(self, text, expected):
    self.assertEqual(math_eval._str_is_int(text), expected)


class IsFracTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("simple", "1/3", True),
      ("negative", "-2/5", True),
      ("not_a_fraction", "12", False),
      ("zero_denominator", "1/0", False),
  )
  def test_is_frac(self, text, expected):
    self.assertEqual(math_eval._is_frac(text), expected)


class StripPropertyFormattedCommasTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("thousands", "5,600", "5600"),
      ("millions", "1,000,000", "1000000"),
      # Tuple commas are preserved (not a thousands separator pattern).
      ("tuple_commas", "(1, 2)", "(1, 2)"),
  )
  def test_strip_properly_formatted_commas(self, text, expected):
    self.assertEqual(
        math_eval._strip_properly_formatted_commas(text), expected
    )


class CountUnknownLettersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("no_letters", "1+2", 0),
      ("one_var", "2x", 1),
      ("two_vars", "x+y", 2),
      # 'sqrt' and 'frac' are stripped before counting.
      ("sqrt_ignored", "sqrt2", 0),
      ("frac_ignored", "frac12", 0),
  )
  def test_count_unknown_letters_in_expr(self, expr, expected):
    self.assertEqual(math_eval.count_unknown_letters_in_expr(expr), expected)


class ShouldAllowEvalTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("simple", "1+2", True),
      ("two_vars_ok", "x+y", True),
      ("too_many_vars", "x+y+z+w", False),
      ("bad_substring_caret_brace", "2^{3}", False),
      ("bad_substring_caret_paren", "2^(3)", False),
  )
  def test_should_allow_eval(self, expr, expected):
    self.assertEqual(math_eval.should_allow_eval(expr), expected)


class SplitTupleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("scalar", "42", ["42"]),
      ("pair", "(1, 2)", ["1", "2"]),
      ("interval", "[3, 4]", ["3", "4"]),
      ("empty", "", []),
  )
  def test_split_tuple(self, expr, expected):
    self.assertEqual(math_eval.split_tuple(expr), expected)


class NormalizeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("strip_text", r"\text{42}", "42"),
      ("remove_percent", "50\\%", "50"),
      ("float_to_int", "7.0", "7"),
      ("none_passthrough", None, None),
  )
  def test_normalize(self, expr, expected):
    self.assertEqual(math_eval._normalize(expr), expected)


class MathdNormalizeTest(parameterized.TestCase):

  def test_none_returns_none(self):
    self.assertIsNone(math_eval.mathd_normalize_answer(None))

  def test_strips_enclosing_text(self):
    self.assertEqual(math_eval.mathd_normalize_answer(r"\text{hello}"), "hello")

  def test_grade_answer_mathd_exact_match(self):
    self.assertTrue(math_eval.grade_answer_mathd("42", "42"))

  def test_grade_answer_mathd_mismatch(self):
    self.assertFalse(math_eval.grade_answer_mathd("42", "43"))


class MatchTest(parameterized.TestCase):
  """End-to-end `match`. These exercise the sympy path; skip if unavailable."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      import sympy  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
    except ImportError:
      raise absltest.SkipTest("sympy not installed")

  @parameterized.named_parameters(
      ("identical_int", "42", "42", True),
      ("boxed_ground_truth", "42", r"\boxed{42}", True),
      ("fraction_equiv", r"\frac{1}{2}", "0.5", True),
      ("mismatch", "42", "43", False),
  )
  def test_match(self, answer, ground_truth, expected):
    self.assertEqual(math_eval.match(answer, ground_truth), expected)


if __name__ == "__main__":
  absltest.main()

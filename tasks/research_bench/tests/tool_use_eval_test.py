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

"""Tests for BFCLFunctionCallEvaluation, incl.

never-crash on adversarial output.
"""

import json

from absl.testing import absltest
from tasks.research_bench import tool_use_eval

_EX = {
    'question': 'Area of a triangle with base 10 height 5.',
    'function': json.dumps([{
        'name': 'calc_area',
        'description': 'area',
        'parameters': {
            'type': 'dict',
            'properties': {
                'base': {'type': 'integer'},
                'height': {'type': 'integer'},
            },
            'required': ['base', 'height'],
        },
    }]),
    'ground_truth': json.dumps([{'calc_area': {'base': [10], 'height': [5]}}]),
}


class BFCLEvalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ev = tool_use_eval.BFCLFunctionCallEvaluation()

  def test_perfect_and_wrong(self):
    self.assertEqual(
        self.ev.evaluate(_EX, '[calc_area(base=10, height=5)]')['correct'], 1
    )
    self.assertEqual(
        self.ev.evaluate(_EX, '[calc_area(base=1, height=5)]')['correct'], 0
    )

  def test_never_crashes_on_adversarial_output(self):
    # Each of these previously could crash the parser (the set-of-dict one
    # raised TypeError: unhashable type: 'dict'). They must score, not raise.
    adversarial = [
        '[calc_area(base={ {"a": 1} })]',  # set literal containing a dict
        '[calc_area(x={1, 2, 3})]',  # set of scalars
        '[calc_area(base=<garbage>)]',  # not literal-eval-able
        'no call here at all',  # no bracket
        '[not_a_call]',  # list element not a Call
        '[calc_area(**kwargs)]',  # ** unpacking
        '[calc_area(base=[[1,2],[3,4]])]',  # nested lists
        '',  # empty
    ]
    for resp in adversarial:
      r = self.ev.evaluate(_EX, resp)  # must not raise
      self.assertIn('correct', r)
      self.assertIn('reward', r)

  def test_json_string_fields_roundtrip(self):
    self.assertIn('calc_area', self.ev.get_prompt(_EX))
    # abstention: empty-string ground_truth
    ex2 = dict(_EX, ground_truth='')
    self.assertEqual(self.ev.evaluate(ex2, '[]')['correct'], 1)
    self.assertEqual(
        self.ev.evaluate(ex2, '[calc_area(base=10)]')['correct'], 0
    )

  def test_type_strict_no_false_positive(self):
    # Upstream is type-strict: wrong-typed values must NOT match (anti-gaming).
    ex = {
        'question': 'q',
        'function': json.dumps([{
            'name': 'f',
            'description': '',
            'parameters': {
                'type': 'dict',
                'properties': {
                    'n': {'type': 'integer'},
                    's': {'type': 'string'},
                    'b': {'type': 'boolean'},
                },
                'required': ['n', 's', 'b'],
            },
        }]),
        'ground_truth': json.dumps(
            [{'f': {'n': [5], 's': ['5'], 'b': [True]}}]
        ),
    }
    # exact types -> correct
    self.assertEqual(
        self.ev.evaluate(ex, '[f(n=5, s="5", b=True)]')['correct'], 1
    )
    # string for int -> wrong (was a false positive before)
    self.assertEqual(
        self.ev.evaluate(ex, '[f(n="5", s="5", b=True)]')['correct'], 0
    )
    # int for bool -> wrong
    self.assertEqual(self.ev.evaluate(ex, '[f(n=5, s="5", b=1)]')['correct'], 0)
    # int for string -> wrong
    self.assertEqual(
        self.ev.evaluate(ex, '[f(n=5, s=5, b=True)]')['correct'], 0
    )

  def test_int_to_float_widening_allowed(self):
    ex = {
        'question': 'q',
        'function': json.dumps([{
            'name': 'f',
            'description': '',
            'parameters': {
                'type': 'dict',
                'properties': {'x': {'type': 'float'}},
                'required': ['x'],
            },
        }]),
        'ground_truth': json.dumps([{'f': {'x': [5.0]}}]),
    }
    # upstream allows python int->float widening
    self.assertEqual(self.ev.evaluate(ex, '[f(x=5)]')['correct'], 1)

  def test_string_standardization(self):
    ex = {
        'question': 'q',
        'function': json.dumps([{
            'name': 'f',
            'description': '',
            'parameters': {
                'type': 'dict',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
        }]),
        'ground_truth': json.dumps([{'f': {'city': ['New York, NY']}}]),
    }
    # punctuation/space-insensitive match (upstream standardize_string)
    self.assertEqual(
        self.ev.evaluate(ex, '[f(city="new york ny")]')['correct'], 1
    )

  def test_dict_arg_value(self):
    ex = {
        'question': 'q',
        'function': json.dumps([{
            'name': 'f',
            'description': '',
            'parameters': {
                'type': 'dict',
                'properties': {'cfg': {'type': 'dict'}},
                'required': ['cfg'],
            },
        }]),
        'ground_truth': json.dumps([{'f': {'cfg': [{'mode': ['fast']}]}}]),
    }
    # dict arg must match via dict_checker (was always-false before)
    self.assertEqual(
        self.ev.evaluate(ex, '[f(cfg={"mode": "fast"})]')['correct'], 1
    )
    self.assertEqual(
        self.ev.evaluate(ex, '[f(cfg={"mode": "slow"})]')['correct'], 0
    )

  def test_nested_dict_arg_with_scalar_leaves(self):
    # ToolACE ground truth nests a plain call dict, whose leaves are scalars
    # rather than acceptable-value lists. Numeric leaves used to raise
    # TypeError ('int' object is not iterable) and string leaves used to be
    # compared character-by-character, so the call could never match.
    ex = {
        'question': 'q',
        'function': json.dumps([{
            'name': 'f',
            'description': '',
            'parameters': {
                'type': 'dict',
                'properties': {'cfg': {'type': 'dict'}},
                'required': ['cfg'],
            },
        }]),
        'ground_truth': json.dumps(
            [{'f': {'cfg': [{'mode': 'fast', 'retries': 3}]}}]
        ),
    }
    self.assertEqual(
        self.ev.evaluate(ex, '[f(cfg={"mode": "fast", "retries": 3})]')[
            'correct'
        ],
        1,
    )
    self.assertEqual(
        self.ev.evaluate(ex, '[f(cfg={"mode": "fast", "retries": 4})]')[
            'correct'
        ],
        0,
    )

  def test_nested_dict_arg_list_leaves_still_work(self):
    # The BFCL-shaped nesting (leaves ARE acceptable-value lists) is unchanged:
    # this is the shape the scored live set uses.
    ex = {
        'question': 'q',
        'function': json.dumps([{
            'name': 'f',
            'description': '',
            'parameters': {
                'type': 'dict',
                'properties': {'cfg': {'type': 'dict'}},
                'required': ['cfg'],
            },
        }]),
        'ground_truth': json.dumps(
            [{'f': {'cfg': [{'mode': ['fast', 'quick']}]}}]
        ),
    }
    self.assertEqual(
        self.ev.evaluate(ex, '[f(cfg={"mode": "quick"})]')['correct'], 1
    )
    self.assertEqual(
        self.ev.evaluate(ex, '[f(cfg={"mode": "slow"})]')['correct'], 0
    )

  def test_abstention_accepts_text_refusal(self):
    ex = dict(_EX, ground_truth='')  # irrelevance
    # plain-text refusal (no parseable call) counts as correct abstention
    self.assertEqual(
        self.ev.evaluate(ex, 'None of the tools apply.')['correct'], 1
    )
    self.assertEqual(self.ev.evaluate(ex, '[]')['correct'], 1)
    # but making a call is wrong
    self.assertEqual(self.ev.evaluate(ex, '[calc_area(base=1)]')['correct'], 0)

  def test_none_optional_arg_variable_path(self):
    # C1 regression: an optional arg whose allowed set contains None, with the
    # model passing None, must MATCH (upstream is_variable path). Previously the
    # strict type gate rejected None for a declared 'string' arg.
    ex = {
        'question': 'q',
        'function': json.dumps([{
            'name': 'f', 'description': '',
            'parameters': {'type': 'dict',
                           'properties': {'x': {'type': 'integer'},
                                          'flt': {'type': 'string'}},
                           'required': ['x']}}]),
        'ground_truth': json.dumps([{'f': {'x': [5], 'flt': ['', None]}}]),
    }
    # model passes None for the optional string arg -> correct (variable path)
    self.assertEqual(self.ev.evaluate(ex, '[f(x=5, flt=None)]')['correct'], 1)
    # omitting it -> also correct ('' in allowed)
    self.assertEqual(self.ev.evaluate(ex, '[f(x=5)]')['correct'], 1)
    # passing a real (non-allowed) string -> wrong
    self.assertEqual(self.ev.evaluate(ex, '[f(x=5, flt="hi")]')['correct'], 0)


if __name__ == '__main__':
  absltest.main()

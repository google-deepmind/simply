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
"""Unit test for tool_lib.py."""

import math

from absl.testing import absltest
from simply import tool_lib


class ToolLibTest(absltest.TestCase):

  def test_calculator_tool_executor(self):
    executor = tool_lib.CalculatorToolExecutor()
    sample_text = "What is 2+2? <calc>2+2</calc> and 3*5? <calc>3*5</calc>"
    tool_inputs = executor.extract_tool_inputs_from_sample(sample_text)
    self.assertLen(tool_inputs, 2)
    self.assertEqual(tool_inputs[0].input_text, "2+2")
    self.assertEqual(tool_inputs[1].input_text, "3*5")

    tool_outputs = executor.make_tool_calls(tool_inputs)
    self.assertLen(tool_outputs, 2)
    self.assertEqual(tool_outputs[0].output_text, "4")
    self.assertEqual(tool_outputs[1].output_text, "15")

    parsed_output = executor.parse_tool_output(tool_outputs[0])
    self.assertEqual(parsed_output, "<result>4</result>")

    execution_result = executor.execute_tool_from_sample(sample_text)
    self.assertEqual(
        execution_result, "<result>4</result>\n<result>15</result>"
    )

  def test_calculator_tool_executor_error(self):
    executor = tool_lib.CalculatorToolExecutor()
    sample_text = "<calc>1/0</calc>"
    execution_result = executor.execute_tool_from_sample(sample_text)
    self.assertIn("Error: division by zero", execution_result)

  def test_calculator_tool_executor_batch(self):
    executor = tool_lib.CalculatorToolExecutor()
    sample_texts = [
        "Add 5 and 3: <calc>5+3</calc>",
        "Multiply 4 and 6: <calc>4*6</calc>",
        "Invalid: <calc>2x2</calc>",
    ]
    results = executor.execute_tool_batch(sample_texts)
    self.assertLen(results, 3)
    self.assertEqual(results[0], "<result>8</result>")
    self.assertEqual(results[1], "<result>24</result>")
    self.assertIn("Error", results[2])

  def test_calculator_tool_executor_safe_eval(self):
    executor = tool_lib.CalculatorToolExecutor()
    # Test allowed math function
    sample_text_sqrt = "<calc>sqrt(16)</calc>"
    execution_result_sqrt = executor.execute_tool_from_sample(sample_text_sqrt)
    self.assertEqual(execution_result_sqrt, "<result>4.0</result>")

    # Test constant
    sample_text_pi = "<calc>pi</calc>"
    execution_result_pi = executor.execute_tool_from_sample(sample_text_pi)
    self.assertEqual(execution_result_pi, f"<result>{math.pi}</result>")

    # Test disallowed function
    sample_text_disallowed = (
        "<calc>__import__('os').system('echo pwned')</calc>"
    )
    execution_result_disallowed = executor.execute_tool_from_sample(
        sample_text_disallowed
    )
    self.assertIn("Error:", execution_result_disallowed)
    self.assertIn("not defined", execution_result_disallowed)


if __name__ == "__main__":
  absltest.main()

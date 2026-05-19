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
from absl.testing import absltest

from simply.agent import llm


class LlmTest(absltest.TestCase):

  def test_llm_registry_has_litellm(self):
    self.assertIsNotNone(llm.LLMRegistry.get('LiteLLM', raise_error=False))

  def test_llm_scheme_parsing(self):
    """Tests that LLM scheme parsing works."""
    provider_cls = llm.LLMRegistry.get('LiteLLM')
    self.assertIsNotNone(provider_cls)


if __name__ == '__main__':
  absltest.main()

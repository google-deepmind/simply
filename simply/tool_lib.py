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
"""Tool library for simply."""

from concurrent import futures
import copy
import dataclasses
import functools
import math
import re
from typing import Any, ClassVar, Optional, Sequence

from absl import logging
import jax
from jax.experimental import multihost_utils
import numpy as np
import pandas as pd
from simply import config_lib
from simply import model_lib
from simply.utils import common
from simply.utils import lm_format as lm_format_lib
from simply.utils import registry
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib

PRNGKey = jax.typing.ArrayLike
SamplingParams = model_lib.SamplingParams
SamplingOutput = model_lib.SamplingOutput
RawT = common.RawT
PyTree = common.PyTree
ChunkSequence = sampling_lib.ChunkSequence
SamplingInput = sampling_lib.SamplingInput


################################################################################
# Tool definitions


class ToolRegistry(registry.RootRegistry):
  """Tool registry."""

  namespace: ClassVar[str] = "tool"


@dataclasses.dataclass(frozen=True, slots=True)
class ToolInput:
  """Tool input."""

  input_text: str
  tool_name: str


@dataclasses.dataclass(frozen=True, slots=True)
class ToolOutput:
  """Tool output."""

  output_text: str
  tool_name: str
  metadata: dict[str, Any] | None = None


@sampling_lib.SamplingRegistry.register
@dataclasses.dataclass(frozen=True)
class SamplingOutputToolUse(SamplingOutput):
  answer_mask: list[bool]
  turns: int

  @property
  def is_throttled(self) -> bool:
    return is_als_throttled(self.output_text)


@dataclasses.dataclass(frozen=True)
class ToolExecutor:
  """Tool executor base class."""

  tool_start_marker: str = ""
  tool_end_marker: str = ""
  result_start_marker: str = ""
  result_end_marker: str = ""

  def extract_tool_inputs_from_sample(
      self, sample_text: str
  ) -> list[ToolInput]:
    """Extract tool inputs from sample_text."""
    raise NotImplementedError()

  def make_tool_calls(self, tool_inputs: list[ToolInput]) -> list[ToolOutput]:
    """Make tool calls."""
    raise NotImplementedError()

  def parse_tool_output(self, tool_output: ToolOutput) -> str:
    """Parse tool output to text."""
    raise NotImplementedError()

  def execute_tool_from_sample(self, sample_text: str) -> str:
    """Execute tool from text."""
    raise NotImplementedError()

  def execute_tool_batch(self, sample_texts: Sequence[str]) -> list[str]:
    """Execute tool batch."""
    raise NotImplementedError()

  def execute_tool_batch_with_sharding(
      self,
      tool_inputs_str: list[str],
      lm_interface: model_lib.LMInterface,
      tool_sync_length: int = 2048,
  ) -> list[str]:
    """Executes tool batch with sharding."""
    process_count = jax.process_count()
    if process_count > 1:
      # Multi-host - We shard the tool inputs and gather the results.
      global_batch_size = len(tool_inputs_str)
      if global_batch_size % process_count != 0:
        raise ValueError(
            "Global batch size must be divisible by process count for"
            f" sharding. Got {global_batch_size} and {process_count}"
        )
      local_tool_inputs_str = sharding_lib.multihost_sharded(tool_inputs_str)
      local_tool_texts = self.execute_tool_batch(local_tool_inputs_str)
      # NOTE: process_allgather does not support text inputs. We use a
      #   workaround here: convert text to tokens and gather tokens, then
      #   convert back to text.tool_sync_length is the max length of each tool
      #   call.
      vocab = lm_interface.input_processor.vocab  # pytype: disable=attribute-error

      local_tool_tokens = [
          vocab.encode(tool_text) for tool_text in local_tool_texts
      ]
      for i in range(len(local_tool_tokens)):
        local_tool_tokens[i] = local_tool_tokens[i][:tool_sync_length]
        local_tool_tokens[i] += [vocab.pad_id] * (
            tool_sync_length - len(local_tool_tokens[i])
        )
      local_tool_tokens = np.array(local_tool_tokens)  # [B, L]
      all_tool_tokens = multihost_utils.process_allgather(
          local_tool_tokens, tiled=True
      )  # [B * num_hosts, L]
      tool_texts = [
          vocab.decode(tool_tokens) for tool_tokens in all_tool_tokens
      ]
    else:
      # Single-host - We execute the tool batch directly.
      tool_texts = self.execute_tool_batch(tool_inputs_str)
    return tool_texts

  def sample_with_tool(
      self,
      lm_interface: model_lib.LMInterface,
      lm_format: lm_format_lib.LMFormat,
      input_text: Sequence[sampling_lib.SamplingInput],
      sampling_params: SamplingParams | None = None,
      prng_key: int | PRNGKey | None = None,
      params: PyTree = None,
      prefill_size: int = -1,
      max_turns: int = 1,
      max_tool_response_len: int = 1024,
  ) -> list[SamplingOutputToolUse] | list[list[SamplingOutputToolUse]]:
    """Sample generations with tool use.

    This function orchestrates a multi-turn conversation between a language
    model and a tool executor. In each turn, the model generates text (which may
    include a tool call), the tool is executed, and its output is appended to
    the conversation history. This process repeats for a maximum of `max_turns`.

    The final sequence length is guaranteed to not exceed the `max_seq_len`
    defined in the `sampling_params`. Length is controlled by several
    parameters: `sampling_params.max_seq_len` sets a hard ceiling for the total
    sequence length. `max_decode_steps` and `max_tool_response_len` are soft
    limits for per-turn model and tool outputs, respectively, dynamically capped
    by the remaining space within `max_seq_len`. `max_turns` limits the
    interaction cycles, and `sampling_params.max_input_len` only applies to the
    initial prompt.

    Args:
      lm_interface: The language model interface.
      lm_format: The format for the language model.
      input_text: A sequence of input texts to start the conversation.
      sampling_params: Parameters for the sampling process. If None, uses
        `lm_interface.default_sampling_params`.
      prng_key: A JAX PRNG key for stochastic operations.
      params: Optional parameters for the language model.
      prefill_size: The size for prefilling the model.
      max_turns: The maximum number of turns in the conversation.
      max_tool_response_len: The maximum length of the tool response. If the
        tool response is longer than this, it will be truncated.

    Returns:
      A list of lists of `SamplingOutputToolUse`. The outer list
      corresponds to the input texts. Each inner list contains
      `num_samples` outputs generated for the respective input text.
    """
    sampling_params = sampling_params or lm_interface.default_sampling_params
    num_samples = sampling_params.num_samples

    # Manually repeat the input `num_samples` times to allow different
    # num_turns for each rollout.
    input_chunks_list: list[ChunkSequence] = [
        sampling_lib.input_as_chunks(x) for x in input_text
    ]
    all_input_chunks_list: list[ChunkSequence] = repeat_elements(
        input_chunks_list, num_samples
    )
    cur_sampling_params = dataclasses.replace(sampling_params, num_samples=1)
    running_input_chunks_list = copy.deepcopy(all_input_chunks_list)
    running_output_chunks_list = [
        [] for _ in range(len(running_input_chunks_list))
    ]
    is_active_list = [True] * len(running_input_chunks_list)
    used_turns_list = [0] * len(running_input_chunks_list)

    # Marker for multi-turn.
    assistant_start_marker = getattr(lm_format, "assistant_marker")
    assistant_end_marker = getattr(lm_format, "end_of_message_marker")
    assistant_end_marker = assistant_end_marker.strip()  # Remove trailing \n.

    def _encode(
        sampling_input: sampling_lib.SamplingInput,
        include_bos_token: bool = False,
    ) -> list[int]:
      cs = sampling_lib.input_as_chunks(sampling_input)
      tokens = lm_interface.input_processor.encode(cs).tokens
      if include_bos_token:
        return tokens
      return tokens[1:]

    result_start_tokens = _encode(self.result_start_marker)
    result_end_tokens = _encode(self.result_end_marker)
    assistant_start_tokens = _encode(assistant_start_marker)
    assistant_end_tokens = _encode(assistant_end_marker)

    # Sampling Loop Start
    for num_turn in range(max_turns):
      if not any(is_active_list):
        break
      prng_key, subkey = jax.random.split(prng_key)
      sampling_outputs = lm_interface.generate(
          input_text=running_input_chunks_list,
          prng_key=subkey,
          sampling_params=cur_sampling_params,
          params=params,
          prefill_size=prefill_size,
          include_eos_in_output_text=True,
          scoring_inputs=False,
      )
      # max_input_len only applies to the initial prompt.
      if num_turn == 0:
        cur_sampling_params = dataclasses.replace(
            cur_sampling_params,
            max_input_len=cur_sampling_params.max_seq_len,
        )

      # Ensure that each sample contains exactly one output.
      assert not sampling_outputs or len(sampling_outputs[0]) == 1
      cur_output_chunks_list = [so[0].output_chunks for so in sampling_outputs]

      # For inactive samples, tool execution is skipped using empty string.
      output_for_tool_manager = [
          sampling_lib.chunks_as_text(cur_output_chunks) if is_active else ""
          for cur_output_chunks, is_active in zip(
              cur_output_chunks_list, is_active_list
          )
      ]

      # Sharding for tool call. Otherwise, all hosts will call the tool
      # repeatedly, causing non-deterministic behavior.
      tool_texts = self.execute_tool_batch_with_sharding(
          output_for_tool_manager,
          lm_interface,
          tool_sync_length=max_tool_response_len + 1,
      )

      assert len(tool_texts) == len(output_for_tool_manager)
      for i in range(len(running_input_chunks_list)):
        if not is_active_list[i]:
          continue

        logging.info(
            "Turn %d/%d Sample %d Replicate %d ID %d",
            num_turn + 1,
            max_turns,
            i // num_samples,
            i % num_samples,
            i,
        )
        running_input_chunks = running_input_chunks_list[i]
        cur_output_chunks = cur_output_chunks_list[i]
        tool_text = tool_texts[i]

        # Calculate the precise remaining space based on max_seq_len.
        input_tokens = _encode(running_input_chunks, include_bos_token=True)
        output_tokens = _encode(cur_output_chunks, include_bos_token=False)
        tool_tokens = _encode(tool_text, include_bos_token=False)
        cur_total_len = len(input_tokens) + len(output_tokens)
        tool_response_budget = min(
            max_tool_response_len,
            cur_sampling_params.max_seq_len - cur_total_len,
        )
        assert tool_response_budget >= 0

        if len(tool_tokens) > tool_response_budget:
          truncate_len = tool_response_budget - len(result_end_tokens)
          if truncate_len > len(result_start_tokens):
            tool_tokens = tool_tokens[:truncate_len] + result_end_tokens
          else:  # Not enough space for the end token, discard tool response.
            tool_tokens = []
          assert len(tool_tokens) <= tool_response_budget
        tool_chunks = lm_interface.input_processor.decode(tool_tokens)

        # If the sequence is full, the tool response is empty, or we have
        # reached the max number of turns, stop the generation.
        cur_total_len += len(tool_tokens)
        assert cur_total_len <= cur_sampling_params.max_seq_len
        sequence_is_full = (
            cur_total_len + len(assistant_start_tokens) + 1
            >= cur_sampling_params.max_seq_len
        )
        if not tool_text or num_turn >= max_turns - 1 or sequence_is_full:
          is_active_list[i] = False

        combined_chunks = cur_output_chunks + tool_chunks
        if is_active_list[i]:
          combined_chunks += sampling_lib.input_as_chunks(
              assistant_start_marker
          )
        logging.info(
            "Input: %s",
            sampling_lib.chunks_as_text(running_input_chunks_list[i]),
        )
        logging.info(
            "Output: %s",
            sampling_lib.chunks_as_text(combined_chunks),
        )
        running_input_chunks_list[i] += combined_chunks
        running_output_chunks_list[i] += combined_chunks

        used_turns_list[i] += 1

    all_sampling_outputs = []
    for i, (input_chunks, output_chunks) in enumerate(
        zip(all_input_chunks_list, running_output_chunks_list)
    ):
      input_token_ids = _encode(input_chunks, include_bos_token=True)
      output_token_ids = _encode(output_chunks, include_bos_token=False)
      output_token_ids.append(lm_interface.input_processor.eos_ids[0])
      # TODO: Add eos token to output_chunks if required.
      assert (
          len(input_token_ids) + len(output_token_ids)
          <= sampling_params.max_seq_len
      )
      answer_mask = get_answer_mask(
          input_token_ids + output_token_ids,
          assistant_start_tokens,
          assistant_end_tokens,
      )
      logging.info("#Answer mask: %s", str(sum(answer_mask)))
      all_sampling_outputs.append(
          SamplingOutputToolUse(
              input_chunks=input_chunks,
              input_token_ids=input_token_ids,
              output_chunks=output_chunks,
              output_token_ids=output_token_ids,
              input_token_scores=np.zeros(len(input_token_ids) - 1).tolist(),
              output_token_scores=np.zeros(len(output_token_ids)).tolist(),
              output_token_logprobs=np.zeros(len(output_token_ids)).tolist(),
              answer_mask=answer_mask,
              turns=used_turns_list[i],
          )
      )
    all_sampling_outputs = [
        all_sampling_outputs[i : i + num_samples]
        for i in range(0, len(all_sampling_outputs), num_samples)
    ]
    return all_sampling_outputs


@ToolRegistry.register
@dataclasses.dataclass(frozen=True)
class CalculatorToolExecutor(ToolExecutor):
  """A simple calculator tool executor for demonstration purposes.

  Tool call format: `<calc>{math_expression}</calc>`
  Tool response format: `<result>{math_result}</result>`
  """

  tool_start_marker: str = "<calc>"
  tool_end_marker: str = "</calc>"
  result_start_marker: str = "<result>"
  result_end_marker: str = "</result>"

  def extract_tool_inputs_from_sample(
      self, sample_text: str
  ) -> list[ToolInput]:
    """Extracts arithmetic expressions from the sample text."""
    code_pattern = re.compile(
        rf"{self.tool_start_marker}(.*?){self.tool_end_marker}", re.DOTALL
    )
    matches = code_pattern.findall(sample_text)
    tool_inputs = []
    for match in matches:
      tool_inputs.append(
          ToolInput(input_text=match.strip(), tool_name="calculator")
      )
    return tool_inputs

  def make_single_tool_call(self, tool_input: ToolInput) -> ToolOutput:
    """Evaluates a single arithmetic expression."""
    allowed_functions = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": math.pow,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "float": float,
        "int": int,
    }
    try:
      # WARNING: Using eval() is not safe with untrusted input, even with
      # a restricted environment. This is for demonstration purposes only.
      # In a real-world scenario, use a safe evaluation library like
      # ast.literal_eval or a dedicated math expression parser.
      result = eval(  # pylint: disable=eval-used
          tool_input.input_text, {"__builtins__": {}}, allowed_functions
      )
      output_text = str(result)
      metadata = {}
    except Exception as e:  # pylint: disable=broad-exception-caught
      output_text = f"Error: {e}"
      metadata = {"error_type": "evaluation", "error_message": str(e)}
    return ToolOutput(
        output_text=output_text,
        tool_name=tool_input.tool_name,
        metadata=metadata,
    )

  def make_tool_calls(self, tool_inputs: list[ToolInput]) -> list[ToolOutput]:
    """Evaluates a batch of arithmetic expressions."""
    return [
        self.make_single_tool_call(tool_input) for tool_input in tool_inputs
    ]

  def parse_tool_output(self, tool_output: ToolOutput) -> str:
    """Formats the tool output."""
    return (
        f"{self.result_start_marker}{tool_output.output_text}"
        f"{self.result_end_marker}"
    )

  def execute_tool_from_sample(self, sample_text: str) -> str:
    """Executes the calculator tool on a sample text."""
    tool_inputs = self.extract_tool_inputs_from_sample(sample_text)
    if not tool_inputs:
      return ""
    tool_outputs = self.make_tool_calls(tool_inputs)
    return "\n".join([self.parse_tool_output(out) for out in tool_outputs])

  def execute_tool_batch(self, sample_texts: Sequence[str]) -> list[str]:
    """Executes tool batch."""
    return [self.execute_tool_from_sample(s) for s in sample_texts]


################################################################################
# Sampling with tool use.


def _find_sublist_index(
    main_list: list[int], sub_list: list[int], start_index: int = 0
) -> int:
  """Finds the starting index of the first sub_list after start_index."""
  n = len(main_list)
  m = len(sub_list)
  if m == 0:
    return start_index
  if start_index < 0:
    start_index = 0
  if m > n - start_index:
    return -1
  for i in range(start_index, n - m + 1):
    if main_list[i : i + m] == sub_list:
      return i
  return -1


def get_answer_mask(
    tokens: list[int],
    start_tokens: list[int],
    end_tokens: list[int],
) -> list[bool]:
  """Generates a token mask for multiple non-overlapping turns."""
  num_tokens = len(tokens)
  mask = [False] * num_tokens

  if not start_tokens or not end_tokens:
    return mask

  current_idx = 0
  while current_idx < num_tokens:
    s_idx = _find_sublist_index(tokens, start_tokens, current_idx)
    if s_idx == -1:
      break  # No more start tags
    mask_start_idx = s_idx + len(start_tokens)
    if mask_start_idx > num_tokens:
      break  # Should not happen if s_idx is valid
    e_idx = _find_sublist_index(tokens, end_tokens, mask_start_idx)
    if e_idx == -1:
      break  # No matching end tag for the last found start tag
    mask_end_idx = e_idx + len(end_tokens)
    for i in range(mask_start_idx, mask_end_idx):
      if 0 <= i < num_tokens:
        mask[i] = True
    current_idx = mask_end_idx

  return mask


def repeat_elements(input_list, n):
  """[1, 2, 3], n = 2 -> [1, 1, 2, 2, 3, 3]."""
  output_list = []
  for item in input_list:
    for _ in range(n):
      output_list.append(copy.deepcopy(item))
  return output_list


def create_tool_executor(
    config: config_lib.ExperimentConfig,
) -> ToolExecutor | None:
  if tool_manager_name := getattr(config, "tool_manager_name", None):
    return ToolRegistry.get(tool_manager_name)()
  return None

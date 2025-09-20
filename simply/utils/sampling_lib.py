"""Sampling helpers for Simply."""

import dataclasses
import sys
from typing import Tuple

import einops
import numpy as np

from simply.utils import common


@dataclasses.dataclass(frozen=True)
class DecodingSchedule:
  """Encapsulates the indices used for decoding in chunks.

  Decoding at position i means predicting the token in position i+1. We will
  decode intervals at a time of the form

  [prefill_size + k * chunk_size, prefill_size + (k+1) * chunk_size),

  truncated at begin_position (inclusive) and end_position (exclusive).

  Attributes:
    prefill_size: The number of tokens to prefill.
    begin_position: The minimum position to decode (inclusive).
    end_position: The maximum position to decode (exclusive).
    chunk_size: The number of tokens to decode in each chunk.
  """

  prefill_size: int
  begin_position: int
  end_position: int
  chunk_size: int

  def get_next_length(self, cur_position: int) -> int:
    if cur_position < self.begin_position:
      return self.begin_position
    step_multiple = ((cur_position - self.prefill_size) // self.chunk_size) + 1
    pos = self.prefill_size + self.chunk_size * step_multiple
    return min(pos, self.end_position)


@dataclasses.dataclass(frozen=True)
class SamplingParams:
  """Sampling parameters."""

  temperature: float = 1.0
  top_k: int = -1
  top_p: float = 1.0
  max_decode_steps: int = 256
  # `max_seq_len` is bos/eos counted, required > 0.
  # When both `max_seq_len` and `max_decode_steps` are set, we will stop
  # decoding at `min(input_len + max_decode_steps, max_seq_len)`. So when you
  # intend to leverage `max_seq_len` behavior, you usually need to make sure
  # `max_decode_steps` is large enough.
  # The typical use case to set `max_seq_len` is to sample sequences for
  # training, because training requires a fixed length.
  max_seq_len: int = sys.maxsize
  # Maximum length of input to accept (bos counted). Longer inputs will be
  # truncated to the trailing `max_input_len` tokens.
  max_input_len: int | None = None
  num_samples: int = 1

  min_prefill_size: int = 256
  # Length to prefill, if specified. Otherwise, a prefill size will be inferred
  # from other settings.
  prefill_size: int | None = None

  intermediate_decode_steps: int | None = None  # Recommended to set.
  sort_by: str | None = None

  def get_decoding_schedule(
      self, min_input_length: int, max_input_length: int
  ) -> DecodingSchedule:
    """Creates DecodingSchedule based on the sampling params."""
    prefill_size = self.prefill_size
    if prefill_size is None:
      prefill_size = max(
          int(np.exp2(np.ceil(np.log2(min_input_length)))),
          self.min_prefill_size,
      )

    # If prefill_size >= min_input_length, we start at min_input_length - 1 so
    # that we can decode the first output token at index min_input_length.
    #
    # If prefill_size < min_input_length, we start at prefill_size, which is the
    # index of the first non-prefilled decode state.
    begin_position = min(prefill_size, min_input_length - 1)
    # For sequences of length up to L, the largest decoding position we need is
    # L - 2, which decodes for index L - 1.
    end_position_exclusive = min(
        self.max_seq_len - 1, max_input_length + self.max_decode_steps - 1
    )
    chunk_size = self.intermediate_decode_steps
    if chunk_size is None:
      chunk_size = self.max_decode_steps
    return DecodingSchedule(
        prefill_size=prefill_size,
        begin_position=begin_position,
        end_position=end_position_exclusive,
        chunk_size=chunk_size,
    )


@dataclasses.dataclass(frozen=True)
class ProcessedInputBatch:
  """Holder for all sampling input after processing.

  Processing includes tokenization, padding, etc. The contents here are ready to
  be fed into the model.
  """

  tokens: common.Array
  lengths: common.Array

  @property
  def batch_size(self):
    return self.tokens.shape[0]

  @property
  def min_length(self):
    return min(self.lengths)

  @property
  def max_length(self):
    return max(self.lengths)

  def token_slice(self, start, end):
    return self.tokens[:, start:end]

  def repeat(self, n):
    return ProcessedInputBatch(
        tokens=einops.repeat(self.tokens, 'b t -> (b n) t', n=n),
        lengths=einops.repeat(self.lengths, 'b -> (b n)', n=n),
    )


def prepare_sampling_input(
    sampling_params: SamplingParams,
    input_tokens: list[list[int]],
    pad_id: int,
) -> Tuple[ProcessedInputBatch, DecodingSchedule]:
  """Prepares the sampling input given raw tokens (e.g. padding/truncation)."""

  all_tokens = input_tokens
  if sampling_params.max_input_len is not None:
    all_tokens = [t[-sampling_params.max_input_len :] for t in input_tokens]
  all_lengths = [len(t) for t in all_tokens]

  max_length = max(all_lengths)
  min_length = min(all_lengths)
  decoding_schedule = sampling_params.get_decoding_schedule(
      min_input_length=min_length, max_input_length=max_length
  )
  initial_length = 1 + max(
      decoding_schedule.get_next_length(max_length - 1),
      decoding_schedule.prefill_size,
  )

  # Pad inputs to the longest length.
  for i, tokens in enumerate(all_tokens):
    all_tokens[i] += [pad_id] * (initial_length - len(tokens))
  processed_input = ProcessedInputBatch(
      tokens=np.array(all_tokens),
      lengths=np.array(all_lengths),
  )
  if sampling_params.num_samples > 1:
    processed_input = processed_input.repeat(sampling_params.num_samples)
  return processed_input, decoding_schedule
